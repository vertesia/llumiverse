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
import type { ExecutionOptions, PromptCacheDiagnostic, PromptCachePath } from '@llumiverse/core';
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
 * The final block is never cached: it is the turn that changes from call to call. Prompt content
 * blocks are never merged, so the boundary the caller expressed by sending separate segments
 * survives. A prompt shaped
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
 * ## Registry: host coordination, with a local memo and Vertex recovery
 *
 * A registry that lives only in a process costs hit rate linearly in fleet size — every instance
 * pays its own cold create, multiplied by every shard. Hosts can inject a distributed coordinator;
 * Studio supplies Redis. The coordinator stores resource name/expiry, grants one bounded lease per
 * content hash, shares create cooldowns, and limits concurrent creates per Vertex project/location.
 *
 * The in-memory map on the driver instance is a memo in front of that registry. Vertex listing by
 * deterministic `displayName` remains recovery for a missing registry entry or a pre-existing
 * resource; it is not the normal fleet lookup. If the injected coordinator is unavailable, a live
 * local memo may still be used, but a cold instance sends uncached rather than creating in a storm.
 *
 * `ListCachedContents` carries no filter field, so the match is client-side: one control-plane read
 * per cold key, paged at 100 (the API coerces anything above 1000), stopping at the first hit and
 * bounded at 1000 entries so a project full of unrelated caches cannot stall a completion. Neither
 * the API surface nor the SDK documents a rate limit on the call; the bound here is our own.
 *
 * ## Fallback
 *
 * In `auto` mode every failure on the cache path is non-fatal. A create error, a min-token rejection,
 * an expired resource, or a permission error produces a safe diagnostic and then the full uncached
 * request. In `required` mode the same preparation failure is surfaced for validation. A min-token
 * rejection additionally marks the prefix uncacheable so later calls do not repeat a doomed create.
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

const CACHE_LEASE_MS = 60_000;
const CACHE_WAIT_MS = 5_000;
const CREATE_PERMIT_WAIT_MS = 1_500;
const CREATE_PERMIT_LEASE_MS = 60_000;
const MAX_CONCURRENT_CREATES_PER_LOCATION = 2;
const DEFAULT_QUOTA_COOLDOWN_MS = 30_000;

export interface GeminiContextCacheEntry {
    /** Server-generated resource name, e.g. `projects/p/locations/l/cachedContents/123`. */
    name: string;
    expiresAtMs: number;
}

export interface GeminiContextCacheCoordinationKey {
    /** Studio environment ID or another caller-defined isolation scope. */
    scope?: string;
    project: string;
    location: string;
    model: string;
    contentHash: string;
}

/**
 * Optional fleet coordinator supplied by the host application.
 *
 * Llumiverse deliberately owns no Redis dependency. Studio injects these functions when it creates
 * a Vertex driver; another host can implement the same semantics with its own coordination store.
 * A rejected operation means coordination is unavailable and causes a safe uncached fallback.
 */
export interface GeminiContextCacheCoordinator {
    getEntry(key: GeminiContextCacheCoordinationKey): Promise<GeminiContextCacheEntry | undefined>;
    acquireLease(key: GeminiContextCacheCoordinationKey, leaseMs: number): Promise<string | undefined>;
    waitForEntry(
        key: GeminiContextCacheCoordinationKey,
        timeoutMs: number,
    ): Promise<GeminiContextCacheEntry | undefined>;
    publishEntry(
        key: GeminiContextCacheCoordinationKey,
        leaseToken: string,
        entry: GeminiContextCacheEntry,
        ttlMs: number,
    ): Promise<boolean>;
    releaseLease(key: GeminiContextCacheCoordinationKey, leaseToken: string): Promise<void>;
    invalidateEntry(key: GeminiContextCacheCoordinationKey, expectedName: string): Promise<void>;
    getCooldownUntil(key: GeminiContextCacheCoordinationKey): Promise<number | undefined>;
    setCooldownUntil(key: GeminiContextCacheCoordinationKey, untilMs: number): Promise<void>;
    acquireCreatePermit(
        key: GeminiContextCacheCoordinationKey,
        limit: number,
        leaseMs: number,
        waitMs: number,
    ): Promise<string | undefined>;
    releaseCreatePermit(key: GeminiContextCacheCoordinationKey, permitToken: string): Promise<void>;
}

export interface GeminiContextCacheManagerOptions {
    /** TTL used when an execution does not carry `prompt_cache_ttl_seconds`. */
    ttlSeconds?: number;
    coordinator?: GeminiContextCacheCoordinator;
    coordinationScope?: string;
}

interface GeminiContextCacheResolution {
    entry?: GeminiContextCacheEntry;
    path: PromptCachePath;
    waitLatencyMs?: number;
    providerStatus?: number;
}

/**
 * Per-driver memo of Vertex `cachedContents` resources, keyed by content hash.
 *
 * Deliberately dumb: a bounded map of live entries, a set of prefixes Vertex has refused to cache,
 * and de-duplication of concurrent lookups for the same content. It holds no client and makes no
 * calls; the caller supplies the loader, so discovery and creation stay testable without GCP.
 *
 * This is a memo, not the fleet registry. Its job is to keep distributed lookups off the hot path
 * and collapse concurrent calls inside one driver process.
 */
export class GeminiContextCacheManager {
    private readonly entries = new Map<string, GeminiContextCacheEntry>();
    private readonly uncacheable = new Set<string>();
    private readonly pending = new Map<string, Promise<GeminiContextCacheResolution>>();
    private createCooldownUntilMs = 0;
    private activeCreates = 0;
    readonly defaultTtlSeconds: number;
    readonly coordinator: GeminiContextCacheCoordinator | undefined;
    readonly coordinationScope: string | undefined;

    constructor(options: GeminiContextCacheManagerOptions = {}) {
        this.defaultTtlSeconds = options.ttlSeconds ?? DEFAULT_GEMINI_CONTEXT_CACHE_TTL_SECONDS;
        this.coordinator = options.coordinator;
        this.coordinationScope = options.coordinationScope;
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

    getLocalCooldownUntil(): number | undefined {
        return this.createCooldownUntilMs > Date.now() ? this.createCooldownUntilMs : undefined;
    }

    setLocalCooldownUntil(untilMs: number): void {
        this.createCooldownUntilMs = Math.max(this.createCooldownUntilMs, untilMs);
    }

    async acquireLocalCreatePermit(waitMs: number): Promise<boolean> {
        const deadline = Date.now() + waitMs;
        while (this.activeCreates >= MAX_CONCURRENT_CREATES_PER_LOCATION) {
            if (Date.now() >= deadline) return false;
            await delay(Math.min(25, Math.max(1, deadline - Date.now())));
        }
        this.activeCreates++;
        return true;
    }

    releaseLocalCreatePermit(): void {
        this.activeCreates = Math.max(0, this.activeCreates - 1);
    }

    /**
     * Return the memoized entry for `key`, or run `load` — discovery then creation — on a miss or
     * near expiry. Concurrent callers for the same content share one lookup.
     */
    async resolve(
        key: string,
        load: () => Promise<GeminiContextCacheResolution>,
    ): Promise<GeminiContextCacheResolution> {
        const existing = this.entries.get(key);
        if (existing && existing.expiresAtMs - Date.now() > CACHE_REFRESH_MARGIN_MS) {
            return { entry: existing, path: 'local_memo_hit' };
        }
        const inflight = this.pending.get(key);
        if (inflight) {
            const startedAt = Date.now();
            const resolution = await inflight;
            return resolution.entry
                ? { ...resolution, path: 'waited_for_creator', waitLatencyMs: Date.now() - startedAt }
                : resolution;
        }

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
    /** Number of leading `Part`s lifted into the cache resource. */
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
    // Vertex rejects cachedContents.create when its request ends in a model turn. Keep any trailing
    // model blocks in the uncached continuation; the resource must end on a user turn.
    while (prefix.at(-1)?.role === 'model') {
        partCount -= prefix.pop()?.parts?.length ?? 0;
    }
    if (!prompt.system && prefix.length === 0) return undefined;
    return { system: prompt.system, contents: prefix, partCount };
}

/**
 * Drop the first `count` `Part`s from a `Content[]`, dropping blocks that empty out. Returns
 * `undefined` if the request holds fewer parts than the prefix claims; caching must not touch a
 * payload that no longer matches the prompt from which the prefix was derived.
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
    return /minimum token|min_?total_?token|too (?:small|few).*token|at least \d+ tokens/i.test(errorMessage(error));
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
    managerKey: string;
    cacheName: string;
    manager: GeminiContextCacheManager;
    coordinationKey?: GeminiContextCacheCoordinationKey;
    diagnostic: PromptCacheDiagnostic;
}

interface GeminiContextCachePlanResult {
    plan?: GeminiContextCachePlan;
    diagnostic: PromptCacheDiagnostic;
}

export interface GeminiContextCacheExecution<T> {
    value: T;
    diagnostic: PromptCacheDiagnostic;
}

export class GeminiContextCacheRequiredError extends Error {
    constructor(readonly diagnostic: PromptCacheDiagnostic) {
        super(`Gemini explicit context cache was required but unavailable (${diagnostic.path})`);
        this.name = 'GeminiContextCacheRequiredError';
    }
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
    location: string,
    /** Resource names already proven unusable on this call, so discovery does not re-adopt them. */
    excludeCacheNames?: ReadonlySet<string>,
): Promise<GeminiContextCachePlanResult> {
    const startedAt = Date.now();
    const baseDiagnostic = (path: PromptCachePath): PromptCacheDiagnostic => ({
        path,
        explicit_cache_used: false,
        model: payload.model,
        preparation_latency_ms: Date.now() - startedAt,
    });
    // Order matters: without a cache key nothing below runs, so a driver that never opted in — or a
    // test double that does not implement the registry — sees exactly today's payload.
    if (options.prompt_cache_mode === 'off') return { diagnostic: baseDiagnostic('disabled') };
    if (!options.prompt_cache_key) return { diagnostic: baseDiagnostic('no_key') };
    // Image generation carries its own config shape and no reusable prefix.
    if (options.model.toLowerCase().includes('image')) return { diagnostic: baseDiagnostic('disabled') };
    const manager = driver.getGeminiContextCacheManager?.();
    if (!manager) return { diagnostic: baseDiagnostic('disabled') };
    if (!Array.isArray(payload.contents)) return { diagnostic: baseDiagnostic('fallback_provider_error') };

    const prefix = deriveGeminiCachePrefix(prompt);
    if (!prefix) return { diagnostic: baseDiagnostic('fallback_provider_error') };

    // Vertex rejects a generateContent request that sets systemInstruction, tools or toolConfig
    // alongside cachedContent — those belong to the cache. So they are part of its identity.
    const tools = payload.config?.tools as Tool[] | undefined;
    const toolConfig = payload.config?.toolConfig;
    const contentHash = geminiCacheContentHash(payload.model, prefix, tools, toolConfig);
    const managerKey = `${location}:${contentHash}`;
    const coordinationKey = driver.getGeminiContextCacheCoordinationKey?.(location, payload.model, contentHash);
    const diagnostic = (path: PromptCachePath, extra: Partial<PromptCacheDiagnostic> = {}): PromptCacheDiagnostic => ({
        path,
        explicit_cache_used: false,
        content_hash_prefix: contentHash.slice(0, 12),
        model: payload.model,
        scope: coordinationKey
            ? [coordinationKey.scope, coordinationKey.project, coordinationKey.location].filter(Boolean).join(':')
            : manager.coordinationScope,
        cacheable_part_count: prefix.partCount,
        preparation_latency_ms: Date.now() - startedAt,
        ...extra,
    });
    if (manager.isUncacheable(managerKey)) {
        return { diagnostic: diagnostic('minimum_token_rejection') };
    }

    const requestContents = stripLeadingParts(payload.contents as Content[], prefix.partCount);
    if (!requestContents || requestContents.length === 0) {
        driver.logger.warn(
            { model: payload.model },
            '[VertexAI] Gemini context cache: request contents do not match the derived prefix, sending un-cached',
        );
        return { diagnostic: diagnostic('fallback_provider_error') };
    }

    const resolution = await manager.resolve(managerKey, () =>
        adoptOrCreateCache({
            driver,
            client,
            manager,
            contentHash,
            managerKey,
            model: payload.model,
            prefix,
            tools,
            toolConfig,
            ttlSeconds: resolveTtlSeconds(options, manager),
            excludeCacheNames,
            coordinationKey,
        }),
    );
    if (!resolution.entry) {
        return {
            diagnostic: diagnostic(resolution.path, {
                wait_latency_ms: resolution.waitLatencyMs,
                provider_status: resolution.providerStatus,
            }),
        };
    }

    const cacheDiagnostic = diagnostic(resolution.path, {
        explicit_cache_used: true,
        wait_latency_ms: resolution.waitLatencyMs,
        provider_status: resolution.providerStatus,
    });

    return {
        diagnostic: cacheDiagnostic,
        plan: {
            managerKey,
            cacheName: resolution.entry.name,
            manager,
            coordinationKey,
            diagnostic: cacheDiagnostic,
            payload: {
                ...payload,
                contents: requestContents,
                config: {
                    ...payload.config,
                    systemInstruction: undefined,
                    tools: undefined,
                    toolConfig: undefined,
                    cachedContent: resolution.entry.name,
                },
            },
        },
    };
}

interface AdoptOrCreateCacheParams {
    driver: VertexAIDriver;
    client: GoogleGenAI;
    manager: GeminiContextCacheManager;
    contentHash: string;
    managerKey: string;
    model: string;
    prefix: GeminiCachePrefix;
    tools: Tool[] | undefined;
    toolConfig: ToolConfig | undefined;
    ttlSeconds: number;
    excludeCacheNames?: ReadonlySet<string>;
    coordinationKey?: GeminiContextCacheCoordinationKey;
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
    managerKey,
    model,
    prefix,
    tools,
    toolConfig,
    ttlSeconds,
    excludeCacheNames,
    coordinationKey,
}: AdoptOrCreateCacheParams): Promise<GeminiContextCacheResolution> {
    const displayName = geminiCacheDisplayName(model, contentHash);

    if (manager.coordinator && coordinationKey) {
        return coordinateAdoptOrCreateCache({
            driver,
            client,
            manager,
            managerKey,
            model,
            prefix,
            tools,
            toolConfig,
            ttlSeconds,
            excludeCacheNames,
            coordinationKey,
            displayName,
        });
    }

    // Standalone provider-library path: local singleflight plus Vertex recovery. Studio always
    // injects a coordinator, so this path never turns a Redis outage into a fleet-wide create storm.
    const existing = manager.get(managerKey);
    if (existing && existing.expiresAtMs > Date.now() && !excludeCacheNames?.has(existing.name)) {
        const refreshed = await extendCacheTtl(driver, client, existing, model, ttlSeconds);
        if (refreshed) {
            manager.set(managerKey, refreshed);
            return { entry: refreshed, path: 'refreshed' };
        }
        manager.invalidate(managerKey);
    }

    // 2. Vertex is the shared registry: adopt whatever a peer instance already built.
    const adopted = await findCacheByDisplayName(driver, client, displayName, model, excludeCacheNames);
    if (adopted) {
        const entry =
            adopted.expiresAtMs - Date.now() < CACHE_REFRESH_MARGIN_MS
                ? ((await extendCacheTtl(driver, client, adopted, model, ttlSeconds)) ?? adopted)
                : adopted;
        manager.set(managerKey, entry);
        return { entry, path: 'provider_list_recovery' };
    }

    const cooldownUntil = manager.getLocalCooldownUntil();
    if (cooldownUntil) return { path: 'fallback_quota', waitLatencyMs: Math.max(0, cooldownUntil - Date.now()) };
    if (!(await manager.acquireLocalCreatePermit(CREATE_PERMIT_WAIT_MS))) {
        return { path: 'fallback_wait_timeout', waitLatencyMs: CREATE_PERMIT_WAIT_MS };
    }
    try {
        return await createCache({
            driver,
            client,
            manager,
            managerKey,
            model,
            prefix,
            tools,
            toolConfig,
            ttlSeconds,
            displayName,
        });
    } finally {
        manager.releaseLocalCreatePermit();
    }
}

interface CoordinatedAdoptOrCreateCacheParams extends Omit<AdoptOrCreateCacheParams, 'contentHash'> {
    coordinationKey: GeminiContextCacheCoordinationKey;
    displayName: string;
}

async function coordinateAdoptOrCreateCache({
    driver,
    client,
    manager,
    managerKey,
    model,
    prefix,
    tools,
    toolConfig,
    ttlSeconds,
    excludeCacheNames,
    coordinationKey,
    displayName,
}: CoordinatedAdoptOrCreateCacheParams): Promise<GeminiContextCacheResolution> {
    const coordinator = manager.coordinator;
    if (!coordinator) return { path: 'fallback_coordination_unavailable' };

    const localEntry = manager.get(managerKey);
    try {
        const sharedEntry = await coordinator.getEntry(coordinationKey);
        if (
            sharedEntry &&
            sharedEntry.expiresAtMs > Date.now() &&
            !excludeCacheNames?.has(sharedEntry.name) &&
            sharedEntry.expiresAtMs - Date.now() > CACHE_REFRESH_MARGIN_MS
        ) {
            manager.set(managerKey, sharedEntry);
            return { entry: sharedEntry, path: 'distributed_registry_hit' };
        }

        const leaseToken = await coordinator.acquireLease(coordinationKey, CACHE_LEASE_MS);
        if (!leaseToken) {
            const waitStartedAt = Date.now();
            const waitedEntry = await coordinator.waitForEntry(coordinationKey, CACHE_WAIT_MS);
            const waitLatencyMs = Date.now() - waitStartedAt;
            if (waitedEntry && waitedEntry.expiresAtMs > Date.now() && !excludeCacheNames?.has(waitedEntry.name)) {
                manager.set(managerKey, waitedEntry);
                return { entry: waitedEntry, path: 'waited_for_creator', waitLatencyMs };
            }
            const cooldownUntil = await coordinator.getCooldownUntil(coordinationKey);
            return cooldownUntil && cooldownUntil > Date.now()
                ? { path: 'fallback_quota', waitLatencyMs }
                : { path: 'fallback_wait_timeout', waitLatencyMs };
        }

        try {
            // Re-read after acquiring the lease: a previous leader may have published between our
            // initial read and acquisition.
            const current = await coordinator.getEntry(coordinationKey);
            if (current && current.expiresAtMs > Date.now() && !excludeCacheNames?.has(current.name)) {
                if (current.expiresAtMs - Date.now() > CACHE_REFRESH_MARGIN_MS) {
                    manager.set(managerKey, current);
                    return { entry: current, path: 'distributed_registry_hit' };
                }
                const refreshed = await extendCacheTtl(driver, client, current, model, ttlSeconds);
                if (refreshed) {
                    await coordinator.publishEntry(
                        coordinationKey,
                        leaseToken,
                        refreshed,
                        Math.max(1_000, refreshed.expiresAtMs - Date.now()),
                    );
                    manager.set(managerKey, refreshed);
                    return { entry: refreshed, path: 'refreshed' };
                }
                await coordinator.invalidateEntry(coordinationKey, current.name);
            }

            const adopted = await findCacheByDisplayName(driver, client, displayName, model, excludeCacheNames);
            if (adopted) {
                const entry =
                    adopted.expiresAtMs - Date.now() < CACHE_REFRESH_MARGIN_MS
                        ? ((await extendCacheTtl(driver, client, adopted, model, ttlSeconds)) ?? adopted)
                        : adopted;
                await coordinator.publishEntry(
                    coordinationKey,
                    leaseToken,
                    entry,
                    Math.max(1_000, entry.expiresAtMs - Date.now()),
                );
                manager.set(managerKey, entry);
                return { entry, path: 'provider_list_recovery' };
            }

            const cooldownUntil = await coordinator.getCooldownUntil(coordinationKey);
            if (cooldownUntil && cooldownUntil > Date.now()) {
                return { path: 'fallback_quota', waitLatencyMs: Math.max(0, cooldownUntil - Date.now()) };
            }

            const permitToken = await coordinator.acquireCreatePermit(
                coordinationKey,
                MAX_CONCURRENT_CREATES_PER_LOCATION,
                CREATE_PERMIT_LEASE_MS,
                CREATE_PERMIT_WAIT_MS,
            );
            if (!permitToken) return { path: 'fallback_wait_timeout', waitLatencyMs: CREATE_PERMIT_WAIT_MS };
            try {
                const resolution = await createCache({
                    driver,
                    client,
                    manager,
                    managerKey,
                    model,
                    prefix,
                    tools,
                    toolConfig,
                    ttlSeconds,
                    displayName,
                    coordinator,
                    coordinationKey,
                });
                if (resolution.entry) {
                    await coordinator.publishEntry(
                        coordinationKey,
                        leaseToken,
                        resolution.entry,
                        Math.max(1_000, resolution.entry.expiresAtMs - Date.now()),
                    );
                }
                return resolution;
            } finally {
                await coordinator.releaseCreatePermit(coordinationKey, permitToken);
            }
        } finally {
            await coordinator.releaseLease(coordinationKey, leaseToken);
        }
    } catch {
        // A still-live memo is safe to use while the registry is unavailable. On a cold instance we
        // deliberately do not create: uncached inference is preferable to a fleet-wide create storm.
        if (localEntry && localEntry.expiresAtMs > Date.now() && !excludeCacheNames?.has(localEntry.name)) {
            return { entry: localEntry, path: 'local_memo_hit' };
        }
        return { path: 'fallback_coordination_unavailable' };
    }
}

interface CreateCacheParams
    extends Omit<AdoptOrCreateCacheParams, 'contentHash' | 'excludeCacheNames' | 'coordinationKey'> {
    displayName: string;
    coordinator?: GeminiContextCacheCoordinator;
    coordinationKey?: GeminiContextCacheCoordinationKey;
}

async function createCache({
    driver,
    client,
    manager,
    managerKey,
    model,
    prefix,
    tools,
    toolConfig,
    ttlSeconds,
    displayName,
    coordinator,
    coordinationKey,
}: CreateCacheParams): Promise<GeminiContextCacheResolution> {
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
            return { path: 'fallback_provider_error' };
        }
        manager.set(managerKey, entry);
        return { entry, path: 'created' };
    } catch (error) {
        const status = errorStatus(error);
        if (isMinimumTokenCountError(error)) {
            manager.markUncacheable(managerKey);
            return { path: 'minimum_token_rejection', providerStatus: status };
        }

        // This is not a provider retry loop. In auto mode the cache-create 429 is swallowed so the
        // completion can proceed uncached and Temporal cannot observe it. Retain only Vertex's
        // Retry-After as a shared create-suppression window for the remaining fleet traffic.
        if (status === 429) {
            const untilMs = Date.now() + (retryAfterMilliseconds(error) ?? DEFAULT_QUOTA_COOLDOWN_MS);
            if (coordinator && coordinationKey) await coordinator.setCooldownUntil(coordinationKey, untilMs);
            else manager.setLocalCooldownUntil(untilMs);
            return { path: 'fallback_quota', providerStatus: status };
        }

        return { path: 'fallback_provider_error', providerStatus: status };
    }
}

function delay(milliseconds: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function retryAfterMilliseconds(error: unknown): number | undefined {
    if (!error || typeof error !== 'object') return undefined;
    const candidates = [
        (error as { headers?: unknown }).headers,
        (error as { response?: { headers?: unknown } }).response?.headers,
    ];
    for (const headers of candidates) {
        let value: unknown;
        if (headers && typeof headers === 'object' && 'get' in headers && typeof headers.get === 'function') {
            value = headers.get('retry-after');
        } else if (headers && typeof headers === 'object') {
            value =
                (headers as Record<string, unknown>)['retry-after'] ??
                (headers as Record<string, unknown>)['Retry-After'];
        }
        if (typeof value !== 'string' && typeof value !== 'number') continue;
        const seconds = Number(value);
        if (Number.isFinite(seconds) && seconds >= 0) return Math.ceil(seconds * 1000);
        if (typeof value === 'string') {
            const dateMs = Date.parse(value);
            if (!Number.isNaN(dateMs)) return Math.max(0, dateMs - Date.now());
        }
    }
    return undefined;
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
    location: string,
): Promise<GeminiContextCacheExecution<T>> {
    // `planGeminiContextCache` already swallows provider failures, but nothing on this path is
    // allowed to cost the caller a completion — so an unexpected throw degrades too.
    const planned: GeminiContextCachePlanResult = await planGeminiContextCache(
        driver,
        client,
        options,
        prompt,
        payload,
        location,
    ).catch((error: unknown): GeminiContextCachePlanResult => {
        driver.logger.warn(
            { error, model: payload.model },
            '[VertexAI] Gemini context cache: preparation failed, sending un-cached',
        );
        return {
            diagnostic: {
                path: 'fallback_provider_error' as const,
                explicit_cache_used: false,
                model: payload.model,
                preparation_latency_ms: 0,
                provider_status: errorStatus(error),
            },
        };
    });
    logCacheDiagnostic(driver, planned.diagnostic);
    const plan = planned.plan;
    if (!plan) {
        if (options.prompt_cache_mode === 'required') throw new GeminiContextCacheRequiredError(planned.diagnostic);
        return { value: await send(payload), diagnostic: planned.diagnostic };
    }

    try {
        return { value: await send(plan.payload), diagnostic: plan.diagnostic };
    } catch (error) {
        if (!isCacheUnusableError(error)) throw error;
        // The resource is gone (expired, deleted, or created by an instance that no longer owns it).
        // Forget it, then re-plan: discovery may find a peer's live cache before falling through to
        // a create. The dead name is excluded so a stale listing cannot hand it back.
        plan.manager.invalidate(plan.managerKey);
        if (plan.coordinationKey && plan.manager.coordinator) {
            await plan.manager.coordinator.invalidateEntry(plan.coordinationKey, plan.cacheName).catch(() => undefined);
        }
        driver.logger.warn(
            { error, model: payload.model },
            '[VertexAI] Gemini context cache: cached content unusable, rediscovering',
        );
        const exclude = new Set([plan.cacheName]);
        const retryResult = await planGeminiContextCache(
            driver,
            client,
            options,
            prompt,
            payload,
            location,
            exclude,
        ).catch(() => undefined);
        const retry = retryResult?.plan;
        if (retry) {
            try {
                const diagnostic = { ...retry.diagnostic, path: 'unusable_resource_recreated' as const };
                logCacheDiagnostic(driver, diagnostic);
                return { value: await send(retry.payload), diagnostic };
            } catch (retryError) {
                if (!isCacheUnusableError(retryError)) throw retryError;
                retry.manager.invalidate(retry.managerKey);
                if (retry.coordinationKey && retry.manager.coordinator) {
                    await retry.manager.coordinator
                        .invalidateEntry(retry.coordinationKey, retry.cacheName)
                        .catch(() => undefined);
                }
                driver.logger.warn(
                    { error: retryError, model: payload.model },
                    '[VertexAI] Gemini context cache: replacement cache also unusable, sending un-cached',
                );
            }
        }
        const diagnostic: PromptCacheDiagnostic = {
            ...(retryResult?.diagnostic ?? plan.diagnostic),
            path: 'fallback_provider_error',
            explicit_cache_used: false,
            provider_status: errorStatus(error),
        };
        logCacheDiagnostic(driver, diagnostic);
        if (options.prompt_cache_mode === 'required') throw new GeminiContextCacheRequiredError(diagnostic);
        return { value: await send(payload), diagnostic };
    }
}

function logCacheDiagnostic(driver: VertexAIDriver, diagnostic: PromptCacheDiagnostic): void {
    const fields = { prompt_cache: diagnostic };
    if (
        diagnostic.provider_status !== undefined ||
        diagnostic.path === 'fallback_coordination_unavailable' ||
        diagnostic.path === 'fallback_wait_timeout'
    ) {
        driver.logger.warn(fields, '[VertexAI] Gemini explicit context cache degraded');
    } else {
        driver.logger.debug?.(fields, '[VertexAI] Gemini explicit context cache path');
    }
}
