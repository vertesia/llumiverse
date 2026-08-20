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
 * ## Registry
 *
 * The find-or-create registry is in memory on the driver instance, keyed by model + cache key +
 * a hash of the prefix (and of the tools, which Vertex requires to live in the cache rather than the
 * request). Two driver instances — two processes, or a driver cache eviction — will each create
 * their own `cachedContents` resource for the same prefix. That is harmless: the duplicate expires
 * on its own TTL and costs storage rent for that window, not a re-billed prefix.
 *
 * ## Fallback
 *
 * Every failure on the cache path is non-fatal. A create error, a min-token rejection, an expired
 * resource, a permission error: one warning through the driver logger, then the full un-cached
 * request exactly as before. A min-token rejection additionally marks the key uncacheable so the
 * next thousand calls do not each pay for a doomed create.
 */

/** Default lifetime of a created `cachedContents` resource. */
export const DEFAULT_GEMINI_CONTEXT_CACHE_TTL_SECONDS = 30 * 60;

/** Refresh a cache whose remaining lifetime has dropped below this, instead of racing its expiry. */
const CACHE_REFRESH_MARGIN_MS = 5 * 60 * 1000;

/** Registry bound. Entries are small; the cap only keeps a pathological key space from growing. */
const REGISTRY_MAX_ENTRIES = 512;

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
 * Per-driver registry of Vertex `cachedContents` resources.
 *
 * Deliberately dumb: a bounded map of live entries, a set of keys Vertex has refused to cache, and
 * de-duplication of concurrent creates for the same key. It holds no client and makes no calls; the
 * caller supplies the factory so the whole find-or-create path stays testable without GCP.
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

    /** Forget a resource that is gone or unusable. The next execution creates a new one. */
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
     * Return the live entry for `key`, creating one through `create` on a miss. Concurrent callers
     * for the same key share a single create.
     */
    async resolve(
        key: string,
        create: () => Promise<GeminiContextCacheEntry | undefined>,
    ): Promise<GeminiContextCacheEntry | undefined> {
        const existing = this.entries.get(key);
        if (existing && existing.expiresAtMs - Date.now() > CACHE_REFRESH_MARGIN_MS) {
            return existing;
        }
        const inflight = this.pending.get(key);
        if (inflight) return inflight;

        const started = create().finally(() => this.pending.delete(key));
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
    cacheKey: string;
    manager: GeminiContextCacheManager;
}

function cacheKeyFor(
    model: string,
    promptCacheKey: string,
    prefix: GeminiCachePrefix,
    tools: Tool[] | undefined,
    toolConfig: ToolConfig | undefined,
): string {
    return createHash('sha256')
        .update(
            JSON.stringify({
                model,
                promptCacheKey,
                system: prefix.system ?? null,
                contents: prefix.contents,
                tools: tools ?? null,
                toolConfig: toolConfig ?? null,
            }),
        )
        .digest('hex');
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
    const cacheKey = cacheKeyFor(payload.model, options.prompt_cache_key, prefix, tools, toolConfig);
    if (manager.isUncacheable(cacheKey)) return undefined;

    const requestContents = stripLeadingParts(payload.contents as Content[], prefix.partCount);
    if (!requestContents || requestContents.length === 0) {
        driver.logger.warn(
            { model: payload.model },
            '[VertexAI] Gemini context cache: request contents do not match the derived prefix, sending un-cached',
        );
        return undefined;
    }

    const entry = await manager.resolve(cacheKey, () =>
        createOrRefreshCache({
            driver,
            client,
            manager,
            cacheKey,
            model: payload.model,
            prefix,
            tools,
            toolConfig,
            ttlSeconds: resolveTtlSeconds(options, manager),
        }),
    );
    if (!entry) return undefined;

    return {
        cacheKey,
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

interface CreateOrRefreshCacheParams {
    driver: VertexAIDriver;
    client: GoogleGenAI;
    manager: GeminiContextCacheManager;
    cacheKey: string;
    model: string;
    prefix: GeminiCachePrefix;
    tools: Tool[] | undefined;
    toolConfig: ToolConfig | undefined;
    ttlSeconds: number;
}

async function createOrRefreshCache({
    driver,
    client,
    manager,
    cacheKey,
    model,
    prefix,
    tools,
    toolConfig,
    ttlSeconds,
}: CreateOrRefreshCacheParams): Promise<GeminiContextCacheEntry | undefined> {
    const ttl = `${ttlSeconds}s`;
    const existing = manager.get(cacheKey);
    if (existing) {
        // Live but close to expiry: extend it rather than lose the prefix mid-flight.
        try {
            const updated = await client.caches.update({ name: existing.name, config: { ttl } });
            const refreshed = toCacheEntry(updated, ttlSeconds) ?? {
                name: existing.name,
                expiresAtMs: Date.now() + ttlSeconds * 1000,
            };
            manager.set(cacheKey, refreshed);
            return refreshed;
        } catch (error) {
            driver.logger.warn(
                { error, model },
                '[VertexAI] Gemini context cache: TTL refresh failed, creating a new cache',
            );
            manager.invalidate(cacheKey);
        }
    }

    try {
        const created = await client.caches.create({
            model,
            config: {
                contents: prefix.contents,
                systemInstruction: prefix.system,
                tools,
                toolConfig,
                ttl,
                displayName: `llumiverse-${cacheKey.slice(0, 32)}`,
            },
        });
        const entry = toCacheEntry(created, ttlSeconds);
        if (!entry) {
            driver.logger.warn({ model }, '[VertexAI] Gemini context cache: create returned no resource name');
            return undefined;
        }
        manager.set(cacheKey, entry);
        return entry;
    } catch (error) {
        if (isMinimumTokenCountError(error)) {
            // Below the model's minimum cacheable size. This will never succeed for this prefix.
            manager.markUncacheable(cacheKey);
            driver.logger.warn(
                { error, model },
                '[VertexAI] Gemini context cache: prefix below the model minimum token count, caching disabled ' +
                    'for this key',
            );
            return undefined;
        }
        driver.logger.warn({ error, model }, '[VertexAI] Gemini context cache: create failed, sending un-cached');
        return undefined;
    }
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
        // The resource is gone (expired, deleted, or created by a driver instance that no longer
        // owns it). Forget it, build one replacement, and retry once.
        plan.manager.invalidate(plan.cacheKey);
        driver.logger.warn(
            { error, model: payload.model },
            '[VertexAI] Gemini context cache: cached content unusable, recreating',
        );
        const retry = await planGeminiContextCache(driver, client, options, prompt, payload).catch(() => undefined);
        if (retry) {
            try {
                return await send(retry.payload);
            } catch (retryError) {
                if (!isCacheUnusableError(retryError)) throw retryError;
                retry.manager.invalidate(retry.cacheKey);
                driver.logger.warn(
                    { error: retryError, model: payload.model },
                    '[VertexAI] Gemini context cache: recreated cache also unusable, sending un-cached',
                );
            }
        }
        return send(payload);
    }
}
