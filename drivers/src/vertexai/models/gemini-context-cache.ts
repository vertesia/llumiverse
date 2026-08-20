import { createHash } from 'node:crypto';
import type {
    CachedContent,
    Content,
    GenerateContentParameters,
    GoogleGenAI,
    Part,
    Tool,
    ToolConfig,
} from '@google/genai';
import type { ExecutionOptions } from '@llumiverse/core';
import type { GenerateContentPrompt, VertexAIDriver } from '../index.js';

/**
 * Explicit Vertex context caching for Gemini.
 *
 * Gemini has two caches. The *implicit* one is free and automatic, but the provider decides whether
 * a prefix qualifies — measured on production traffic, `gemini-3.7-flash` served 0 % of its tokens
 * from it even on prompts that are ~70 % shared prefix. The *explicit* one is a `cachedContents`
 * resource: the caller pays a small storage rent (~$1 per million tokens per hour) and every read of
 * that prefix is billed at the cached-input discount, guaranteed.
 *
 * This module turns `ExecutionOptions.prompt_cache_key` into that resource. The key is the caller's
 * statement that "these executions share a prefix"; without it nothing here runs and the provider
 * payload is byte-identical to what the driver has always sent.
 *
 * ## Prefix policy
 *
 * The Anthropic drivers mark a *breakpoint* inside the request — Claude caches everything before an
 * annotated block, so a mis-placed breakpoint only costs a cache miss. Vertex has no in-request
 * marker: the prefix has to be lifted out into a separate resource, and a prefix that includes a
 * per-call fragment creates a brand-new cache resource on every call. So the policy here is
 * deliberately more conservative than Claude's `content[-2]` breakpoint:
 *
 *   prefix = systemInstruction + the leading `Content` blocks that hold nothing but static text,
 *            stopping before the first block containing a non-text part (image, file, tool call,
 *            signed thought) or before the final block, whichever comes first.
 *
 * The final block is never cached: it is the turn that changes from call to call. Derivation runs on
 * the prompt's `Content` blocks *before* `mergeConsecutiveRole` collapses same-role neighbours, so
 * the boundary the caller expressed by sending separate segments survives. A prompt shaped
 * `[system, user: catalog text, user: task text + image]` therefore caches `system + catalog` and
 * sends `task + image` — which is the shape that matters, since the task text names the photo.
 *
 * ## Identity: the content, not the caller's key
 *
 * A cache is identified by a hash of exactly what goes *into* the resource — model, system
 * instruction, prefix contents, tools, tool config — and by nothing else. In particular
 * `prompt_cache_key` is not part of it: the key is the on/off trigger, not the identity. Callers
 * shard that key (`route:0` … `route:3`) to spread load, and hashing it would mint four resources
 * for four byte-identical prefixes. Content addressing collapses the shards onto one cache for free.
 *
 * ## Registry: Vertex itself, with a local memo in front
 *
 * A registry that lives only in a process costs hit rate linearly in fleet size — every instance
 * pays its own cold create, multiplied by every shard. So the shared registry *is* Vertex: each
 * cache is created with a deterministic `displayName` built from the content hash, and a cold
 * instance lists `cachedContents` and adopts a live match instead of creating its own.
 *
 * The in-memory map on the driver instance is a memo in front of that, not the registry itself:
 * listing happens on a cold key, on expiry, and after a 404 — never per call.
 *
 * Two instances that go cold at the same moment can still both create. That is left alone
 * deliberately: the duplicate expires on its own TTL, costs storage rent for that window rather than
 * a re-billed prefix, and both instances converge on one resource at the next list. Locking would
 * cost more than the duplication does.
 *
 * `ListCachedContents` carries no filter field, so the match is client-side: one control-plane read
 * per cold key, paged at 100 (the API coerces anything above 1000), stopping at the first hit and
 * bounded at 1000 entries so a project full of unrelated caches cannot stall a completion. Neither
 * the API surface nor the SDK documents a rate limit on the call; the bound here is our own.
 *
 * ## Fallback
 *
 * Every failure on the cache path is non-fatal. A create error, a min-token rejection, an expired
 * resource, a permission error: one warning through the driver logger, then the full un-cached
 * request exactly as before. A min-token rejection additionally marks the prefix uncacheable so the
 * next thousand calls do not each pay for a doomed create.
 */

/** Default lifetime of a created `cachedContents` resource. */
export const DEFAULT_GEMINI_CONTEXT_CACHE_TTL_SECONDS = 30 * 60;

/** Refresh a cache whose remaining lifetime has dropped below this, instead of racing its expiry. */
const CACHE_REFRESH_MARGIN_MS = 5 * 60 * 1000;

/** Registry bound. Entries are small; the cap only keeps a pathological key space from growing. */
const REGISTRY_MAX_ENTRIES = 512;

/**
 * Marks a `cachedContents` resource as one of ours and makes it discoverable by content.
 * Short on purpose: it is repeated on every listed resource.
 */
export const GEMINI_CACHE_DISPLAY_NAME_PREFIX = 'llmv-cache';

/**
 * Vertex documents no explicit limit on `CachedContent.display_name`, but every other Vertex
 * resource caps its display name at 128 characters, so the format is built to stay well inside that.
 */
const DISPLAY_NAME_MAX_LENGTH = 128;
const DISPLAY_NAME_MODEL_SEGMENT_MAX_LENGTH = 64;
/** 64 bits of the content hash: ~3e-12 collision probability across 10k distinct prefixes. */
const DISPLAY_NAME_HASH_LENGTH = 16;

/** `ListCachedContentsRequest` has no filter field, so matching is client-side over pages. */
const LIST_PAGE_SIZE = 100;
/** Bound on a single discovery sweep, so a project full of unrelated caches cannot stall a call. */
const LIST_MAX_SCANNED = 1000;

export interface GeminiContextCacheEntry {
    /** Server-generated resource name, e.g. `projects/p/locations/l/cachedContents/123`. */
    name: string;
    expiresAtMs: number;
}

export interface GeminiContextCacheManagerOptions {
    /** TTL used when an execution does not carry `prompt_cache_ttl_seconds`. */
    ttlSeconds?: number;
}

/**
 * Per-driver memo of Vertex `cachedContents` resources, keyed by content hash.
 *
 * Deliberately dumb: a bounded map of live entries, a set of prefixes Vertex has refused to cache,
 * and de-duplication of concurrent lookups for the same content. It holds no client and makes no
 * calls; the caller supplies the loader, so discovery and creation stay testable without GCP.
 *
 * This is a memo, not the registry — the registry is Vertex, reached by listing. Its only job is to
 * keep that listing off the per-call path.
 */
export class GeminiContextCacheManager {
    private readonly entries = new Map<string, GeminiContextCacheEntry>();
    private readonly uncacheable = new Set<string>();
    private readonly pending = new Map<string, Promise<GeminiContextCacheEntry | undefined>>();
    readonly defaultTtlSeconds: number;

    constructor(options: GeminiContextCacheManagerOptions = {}) {
        this.defaultTtlSeconds = options.ttlSeconds ?? DEFAULT_GEMINI_CONTEXT_CACHE_TTL_SECONDS;
    }

    get(key: string): GeminiContextCacheEntry | undefined {
        return this.entries.get(key);
    }

    set(key: string, entry: GeminiContextCacheEntry): void {
        this.entries.delete(key);
        this.entries.set(key, entry);
        if (this.entries.size > REGISTRY_MAX_ENTRIES) {
            const oldest = this.entries.keys().next();
            if (!oldest.done) this.entries.delete(oldest.value);
        }
    }

    /** Forget a resource that is gone or unusable. The next execution rediscovers or recreates it. */
    invalidate(key: string): void {
        this.entries.delete(key);
    }

    isUncacheable(key: string): boolean {
        return this.uncacheable.has(key);
    }

    /** Vertex refused to cache this prefix at all (typically below the model's minimum token count). */
    markUncacheable(key: string): void {
        this.entries.delete(key);
        this.uncacheable.add(key);
        if (this.uncacheable.size > REGISTRY_MAX_ENTRIES) {
            const oldest = this.uncacheable.values().next();
            if (!oldest.done) this.uncacheable.delete(oldest.value);
        }
    }

    /**
     * Return the memoized entry for `key`, or run `load` — discovery then creation — on a miss or
     * near expiry. Concurrent callers for the same content share one lookup.
     */
    async resolve(
        key: string,
        load: () => Promise<GeminiContextCacheEntry | undefined>,
    ): Promise<GeminiContextCacheEntry | undefined> {
        const existing = this.entries.get(key);
        if (existing && existing.expiresAtMs - Date.now() > CACHE_REFRESH_MARGIN_MS) {
            return existing;
        }
        const inflight = this.pending.get(key);
        if (inflight) return inflight;

        const started = load().finally(() => this.pending.delete(key));
        this.pending.set(key, started);
        return started;
    }
}

// ---------------------------------------------------------------------------
// Prefix derivation
// ---------------------------------------------------------------------------

export interface GeminiCachePrefix {
    system?: Content;
    contents: Content[];
    /** Number of leading `Part`s the prefix covers once same-role blocks are merged. */
    partCount: number;
}

/**
 * A `Part` is static text only when `text` is the single field carrying anything. Written as a
 * deny-everything-else scan rather than a list of known media keys so a `Part` kind added by a
 * future SDK is treated as dynamic instead of silently landing in a cached prefix.
 */
function isStaticTextPart(part: Part): boolean {
    if (typeof part.text !== 'string' || part.text.length === 0) return false;
    for (const [key, value] of Object.entries(part)) {
        if (key === 'text') continue;
        // `thought: false` and explicit nulls carry no content; anything else does.
        if (value !== undefined && value !== null && value !== false) return false;
    }
    return true;
}

/**
 * Derive the cacheable prefix of a Gemini prompt. See the module comment for the policy.
 * Returns `undefined` when there is nothing worth caching.
 */
export function deriveGeminiCachePrefix(prompt: GenerateContentPrompt): GeminiCachePrefix | undefined {
    const contents = prompt.contents ?? [];
    const prefix: Content[] = [];
    let partCount = 0;
    // `contents.length - 1`: the last block is this call's dynamic turn and is never cached.
    for (let i = 0; i < contents.length - 1; i++) {
        const parts = contents[i].parts ?? [];
        if (parts.length === 0 || !parts.every(isStaticTextPart)) break;
        prefix.push(contents[i]);
        partCount += parts.length;
    }
    if (!prompt.system && prefix.length === 0) return undefined;
    return { system: prompt.system, contents: prefix, partCount };
}

/**
 * Drop the first `count` `Part`s from a merged `Content[]`, dropping blocks that empty out.
 * Returns `undefined` if the request holds fewer parts than the prefix claims — a mismatch means the
 * payload is not the merge of the prompt we derived from, and caching must not touch it.
 */
export function stripLeadingParts(contents: Content[], count: number): Content[] | undefined {
    if (count <= 0) return contents;
    const out: Content[] = [];
    let remaining = count;
    for (const content of contents) {
        if (remaining <= 0) {
            out.push(content);
            continue;
        }
        const parts = content.parts ?? [];
        if (remaining >= parts.length) {
            remaining -= parts.length;
            continue;
        }
        out.push({ ...content, parts: parts.slice(remaining) });
        remaining = 0;
    }
    return remaining === 0 ? out : undefined;
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

function errorStatus(error: unknown): number | undefined {
    if (error && typeof error === 'object' && 'status' in error) {
        const status = (error as { status?: unknown }).status;
        if (typeof status === 'number') return status;
    }
    return undefined;
}

function errorMessage(error: unknown): string {
    if (error && typeof error === 'object' && 'message' in error) {
        const message = (error as { message?: unknown }).message;
        if (typeof message === 'string') return message;
    }
    return String(error);
}

/**
 * Vertex refuses to cache a prefix below the model's minimum token count (1k–4k depending on the
 * model). That is a property of the prefix, not of the moment, so the key is retired rather than
 * retried on every call.
 */
export function isMinimumTokenCountError(error: unknown): boolean {
    const status = errorStatus(error);
    if (status !== undefined && status !== 400) return false;
    return /minimum|min_?total_?token|too small|too few tokens|at least \d+ tokens/i.test(errorMessage(error));
}

/**
 * The named resource cannot be used: it expired, was deleted, belongs to another project, or the
 * caller lost permission on it. Recoverable by forgetting it and building a new one.
 */
export function isCacheUnusableError(error: unknown): boolean {
    const status = errorStatus(error);
    if (status === 404) return true;
    if (status === 400 || status === 403) {
        return /cached ?content|cache/i.test(errorMessage(error));
    }
    return false;
}

// ---------------------------------------------------------------------------
// Execution path
// ---------------------------------------------------------------------------

interface GeminiContextCachePlan {
    payload: GenerateContentParameters;
    contentHash: string;
    cacheName: string;
    manager: GeminiContextCacheManager;
}

/**
 * Identity of a cache: a hash of everything that goes into the resource, and nothing else.
 * `prompt_cache_key` is deliberately absent — see the module comment.
 */
export function geminiCacheContentHash(
    model: string,
    prefix: GeminiCachePrefix,
    tools: Tool[] | undefined,
    toolConfig: ToolConfig | undefined,
): string {
    return createHash('sha256')
        .update(
            JSON.stringify({
                model,
                system: prefix.system ?? null,
                contents: prefix.contents,
                tools: tools ?? null,
                toolConfig: toolConfig ?? null,
            }),
        )
        .digest('hex');
}

/**
 * The name a cache for this content is *always* given, so any instance can find it by listing.
 * Deterministic in the content hash, which is what makes Vertex usable as the shared registry.
 */
export function geminiCacheDisplayName(model: string, contentHash: string): string {
    const shortModel = (model.split('/').pop() ?? model).slice(0, DISPLAY_NAME_MODEL_SEGMENT_MAX_LENGTH);
    const displayName = `${GEMINI_CACHE_DISPLAY_NAME_PREFIX}:${shortModel}:${contentHash.slice(0, DISPLAY_NAME_HASH_LENGTH)}`;
    return displayName.slice(0, DISPLAY_NAME_MAX_LENGTH);
}

function toCacheEntry(cached: CachedContent, ttlSeconds: number): GeminiContextCacheEntry | undefined {
    if (!cached.name) return undefined;
    const expiresAtMs = cached.expireTime ? Date.parse(cached.expireTime) : Number.NaN;
    return {
        name: cached.name,
        expiresAtMs: Number.isNaN(expiresAtMs) ? Date.now() + ttlSeconds * 1000 : expiresAtMs,
    };
}

function resolveTtlSeconds(options: ExecutionOptions, manager: GeminiContextCacheManager): number {
    const requested = options.prompt_cache_ttl_seconds;
    if (typeof requested === 'number' && Number.isFinite(requested) && requested > 0) {
        return Math.floor(requested);
    }
    return manager.defaultTtlSeconds;
}

/**
 * Build the cached variant of `payload`, creating or reusing the `cachedContents` resource.
 * Returns `undefined` whenever the request must go out unchanged.
 */
async function planGeminiContextCache(
    driver: VertexAIDriver,
    client: GoogleGenAI,
    options: ExecutionOptions,
    prompt: GenerateContentPrompt,
    payload: GenerateContentParameters,
    /** Resource names already proven unusable on this call, so discovery does not re-adopt them. */
    excludeCacheNames?: ReadonlySet<string>,
): Promise<GeminiContextCachePlan | undefined> {
    // Order matters: without a cache key nothing below runs, so a driver that never opted in — or a
    // test double that does not implement the registry — sees exactly today's payload.
    if (!options.prompt_cache_key) return undefined;
    if (options.prompt_cache_mode === 'off') return undefined;
    // Image generation carries its own config shape and no reusable prefix.
    if (options.model.toLowerCase().includes('image')) return undefined;
    const manager = driver.getGeminiContextCacheManager?.();
    if (!manager) return undefined;
    if (!Array.isArray(payload.contents)) return undefined;

    const prefix = deriveGeminiCachePrefix(prompt);
    if (!prefix) return undefined;

    // Vertex rejects a generateContent request that sets systemInstruction, tools or toolConfig
    // alongside cachedContent — those belong to the cache. So they are part of its identity.
    const tools = payload.config?.tools as Tool[] | undefined;
    const toolConfig = payload.config?.toolConfig;
    const contentHash = geminiCacheContentHash(payload.model, prefix, tools, toolConfig);
    if (manager.isUncacheable(contentHash)) return undefined;

    const requestContents = stripLeadingParts(payload.contents as Content[], prefix.partCount);
    if (!requestContents || requestContents.length === 0) {
        driver.logger.warn(
            { model: payload.model },
            '[VertexAI] Gemini context cache: request contents do not match the derived prefix, sending un-cached',
        );
        return undefined;
    }

    const entry = await manager.resolve(contentHash, () =>
        adoptOrCreateCache({
            driver,
            client,
            manager,
            contentHash,
            model: payload.model,
            prefix,
            tools,
            toolConfig,
            ttlSeconds: resolveTtlSeconds(options, manager),
            excludeCacheNames,
        }),
    );
    if (!entry) return undefined;

    return {
        contentHash,
        cacheName: entry.name,
        manager,
        payload: {
            ...payload,
            contents: requestContents,
            config: {
                ...payload.config,
                systemInstruction: undefined,
                tools: undefined,
                toolConfig: undefined,
                cachedContent: entry.name,
            },
        },
    };
}

interface AdoptOrCreateCacheParams {
    driver: VertexAIDriver;
    client: GoogleGenAI;
    manager: GeminiContextCacheManager;
    contentHash: string;
    model: string;
    prefix: GeminiCachePrefix;
    tools: Tool[] | undefined;
    toolConfig: ToolConfig | undefined;
    ttlSeconds: number;
    excludeCacheNames?: ReadonlySet<string>;
}

/**
 * Find the `cachedContents` resource for this content, or make one.
 *
 * Three steps, cheapest first: extend a memoized resource that is nearing expiry, adopt a live one
 * another instance already created (found by listing on the deterministic display name), and only
 * then create. Never throws: the caller treats `undefined` as "send the request un-cached".
 */
async function adoptOrCreateCache({
    driver,
    client,
    manager,
    contentHash,
    model,
    prefix,
    tools,
    toolConfig,
    ttlSeconds,
    excludeCacheNames,
}: AdoptOrCreateCacheParams): Promise<GeminiContextCacheEntry | undefined> {
    const displayName = geminiCacheDisplayName(model, contentHash);

    // 1. Memoized and near expiry — extend rather than lose the prefix mid-flight.
    const existing = manager.get(contentHash);
    if (existing && !excludeCacheNames?.has(existing.name)) {
        const refreshed = await extendCacheTtl(driver, client, existing, model, ttlSeconds);
        if (refreshed) {
            manager.set(contentHash, refreshed);
            return refreshed;
        }
        manager.invalidate(contentHash);
    }

    // 2. Vertex is the shared registry: adopt whatever a peer instance already built.
    const adopted = await findCacheByDisplayName(driver, client, displayName, model, excludeCacheNames);
    if (adopted) {
        const entry =
            adopted.expiresAtMs - Date.now() < CACHE_REFRESH_MARGIN_MS
                ? ((await extendCacheTtl(driver, client, adopted, model, ttlSeconds)) ?? adopted)
                : adopted;
        manager.set(contentHash, entry);
        return entry;
    }

    // 3. Nobody has one. Build it.
    try {
        const created = await client.caches.create({
            model,
            config: {
                contents: prefix.contents,
                systemInstruction: prefix.system,
                tools,
                toolConfig,
                ttl: `${ttlSeconds}s`,
                displayName,
            },
        });
        const entry = toCacheEntry(created, ttlSeconds);
        if (!entry) {
            driver.logger.warn({ model }, '[VertexAI] Gemini context cache: create returned no resource name');
            return undefined;
        }
        manager.set(contentHash, entry);
        return entry;
    } catch (error) {
        if (isMinimumTokenCountError(error)) {
            // Below the model's minimum cacheable size. This will never succeed for this prefix.
            manager.markUncacheable(contentHash);
            driver.logger.warn(
                { error, model },
                '[VertexAI] Gemini context cache: prefix below the model minimum token count, caching disabled ' +
                    'for this prefix',
            );
            return undefined;
        }
        driver.logger.warn({ error, model }, '[VertexAI] Gemini context cache: create failed, sending un-cached');
        return undefined;
    }
}

/** Push a resource's expiry out. Returns `undefined` when Vertex would not extend it. */
async function extendCacheTtl(
    driver: VertexAIDriver,
    client: GoogleGenAI,
    entry: GeminiContextCacheEntry,
    model: string,
    ttlSeconds: number,
): Promise<GeminiContextCacheEntry | undefined> {
    try {
        const updated = await client.caches.update({ name: entry.name, config: { ttl: `${ttlSeconds}s` } });
        return toCacheEntry(updated, ttlSeconds) ?? { name: entry.name, expiresAtMs: Date.now() + ttlSeconds * 1000 };
    } catch (error) {
        driver.logger.warn(
            { error, model },
            '[VertexAI] Gemini context cache: TTL refresh failed, looking for another cache',
        );
        return undefined;
    }
}

/**
 * Scan `cachedContents` for a live resource carrying `displayName`.
 *
 * `ListCachedContentsRequest` has no filter field, so the match is client-side; the sweep is bounded
 * and stops at the first hit. Only entries with a parseable expiry still in the future are adopted —
 * a resource that expires between the list and the request would only produce a 404, and the
 * un-parseable case would mean inventing a lifetime we do not know.
 */
async function findCacheByDisplayName(
    driver: VertexAIDriver,
    client: GoogleGenAI,
    displayName: string,
    model: string,
    excludeCacheNames?: ReadonlySet<string>,
): Promise<GeminiContextCacheEntry | undefined> {
    try {
        const pager = await client.caches.list({ config: { pageSize: LIST_PAGE_SIZE } });
        let scanned = 0;
        for await (const cached of pager) {
            if (++scanned > LIST_MAX_SCANNED) break;
            if (cached.displayName !== displayName) continue;
            if (!cached.name || excludeCacheNames?.has(cached.name)) continue;
            const expiresAtMs = cached.expireTime ? Date.parse(cached.expireTime) : Number.NaN;
            if (Number.isNaN(expiresAtMs) || expiresAtMs <= Date.now()) continue;
            return { name: cached.name, expiresAtMs };
        }
    } catch (error) {
        // Discovery is an optimisation; losing it costs a duplicate cache, not a request.
        driver.logger.warn(
            { error, model },
            '[VertexAI] Gemini context cache: listing cached contents failed, creating a new cache',
        );
    }
    return undefined;
}

/**
 * Send `payload` through `send`, routed via an explicit context cache when this execution asked for
 * one. Any cache trouble degrades to the full un-cached request.
 *
 * Shared by the blocking and streaming paths: `send` is whichever `client.models.*` call the caller
 * would have made, so the response type is unchanged.
 */
export async function generateWithGeminiContextCache<T>(
    driver: VertexAIDriver,
    client: GoogleGenAI,
    options: ExecutionOptions,
    prompt: GenerateContentPrompt,
    payload: GenerateContentParameters,
    send: (payload: GenerateContentParameters) => Promise<T>,
): Promise<T> {
    // `planGeminiContextCache` already swallows provider failures, but nothing on this path is
    // allowed to cost the caller a completion — so an unexpected throw degrades too.
    const plan = await planGeminiContextCache(driver, client, options, prompt, payload).catch((error: unknown) => {
        driver.logger.warn(
            { error, model: payload.model },
            '[VertexAI] Gemini context cache: preparation failed, sending un-cached',
        );
        return undefined;
    });
    if (!plan) return send(payload);

    try {
        return await send(plan.payload);
    } catch (error) {
        if (!isCacheUnusableError(error)) throw error;
        // The resource is gone (expired, deleted, or created by an instance that no longer owns it).
        // Forget it, then re-plan: discovery may find a peer's live cache before falling through to
        // a create. The dead name is excluded so a stale listing cannot hand it back.
        plan.manager.invalidate(plan.contentHash);
        driver.logger.warn(
            { error, model: payload.model },
            '[VertexAI] Gemini context cache: cached content unusable, rediscovering',
        );
        const exclude = new Set([plan.cacheName]);
        const retry = await planGeminiContextCache(driver, client, options, prompt, payload, exclude).catch(
            () => undefined,
        );
        if (retry) {
            try {
                return await send(retry.payload);
            } catch (retryError) {
                if (!isCacheUnusableError(retryError)) throw retryError;
                retry.manager.invalidate(retry.contentHash);
                driver.logger.warn(
                    { error: retryError, model: payload.model },
                    '[VertexAI] Gemini context cache: replacement cache also unusable, sending un-cached',
                );
            }
        }
        return send(payload);
    }
}
