import {
    type CachedContent,
    type CreateCachedContentParameters,
    FinishReason,
    type GenerateContentParameters,
    type GenerateContentResponse,
    type GoogleGenAI,
    type ListCachedContentsParameters,
    type Pager,
    type UpdateCachedContentParameters,
} from '@google/genai';
import { type DataSource, type ExecutionOptions, type Logger, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { type GenerateContentPrompt, VertexAIDriver, type VertexAIDriverOptions } from '../index.js';
import { GeminiModelDefinition, getGeminiPayload } from './gemini.js';
import {
    deriveGeminiCachePrefix,
    GEMINI_CACHE_DISPLAY_NAME_PREFIX,
    type GeminiContextCacheCoordinationKey,
    type GeminiContextCacheCoordinator,
    type GeminiContextCacheEntry,
    GeminiContextCacheRequiredError,
    geminiCacheContentHash,
    geminiCacheDisplayName,
    isMinimumTokenCountError,
    stripLeadingParts,
} from './gemini-context-cache.js';

const MODEL = 'gemini-3.7-flash';
const CACHE_NAME = 'projects/test-project/locations/us-central1/cachedContents/1';
const PEER_CACHE_NAME = 'projects/test-project/locations/us-central1/cachedContents/peer';
const CATALOG_TEXT = 'catalog entry. '.repeat(1200);

/**
 * The shape this feature exists for: a stable system instruction, a large static catalog the model
 * routes against, and a per-photo turn that carries the image. The catalog is the same on every
 * call; the task text names the photo, so it must stay out of the cached prefix.
 */
function baliseRoutePrompt(): GenerateContentPrompt {
    return {
        system: { role: 'user', parts: [{ text: 'Route each photo to a domain.' }] },
        contents: [
            { role: 'user', parts: [{ text: CATALOG_TEXT }] },
            {
                role: 'user',
                parts: [{ text: 'Photo 42: route it.' }, { inlineData: { data: 'aW1hZ2U=', mimeType: 'image/jpeg' } }],
            },
        ],
    };
}

function completionResponse(overrides: Partial<GenerateContentResponse> = {}): GenerateContentResponse {
    return {
        usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 10, totalTokenCount: 110 },
        candidates: [
            {
                finishReason: FinishReason.STOP,
                content: { role: 'model', parts: [{ text: 'places' }] },
                safetyRatings: [],
            },
        ],
        ...overrides,
    } as GenerateContentResponse;
}

function cachedContent(name = CACHE_NAME, ttlSeconds = 1800): CachedContent {
    return { name, expireTime: new Date(Date.now() + ttlSeconds * 1000).toISOString() };
}

function apiError(status: number, message: string): Error {
    return Object.assign(new Error(message), { status });
}

class FakeDistributedCoordinator implements GeminiContextCacheCoordinator {
    readonly entries = new Map<string, GeminiContextCacheEntry>();
    readonly leases = new Map<string, string>();
    readonly cooldowns = new Map<string, number>();
    readonly permits = new Set<string>();
    invalidations = 0;
    unavailable = false;

    private cacheKey(key: GeminiContextCacheCoordinationKey): string {
        return JSON.stringify([key.scope, key.project, key.location, key.model, key.contentHash]);
    }

    private scopeKey(key: GeminiContextCacheCoordinationKey): string {
        return JSON.stringify([key.scope, key.project, key.location]);
    }

    private assertAvailable(): void {
        if (this.unavailable) throw new Error('fake Redis unavailable');
    }

    async getEntry(key: GeminiContextCacheCoordinationKey): Promise<GeminiContextCacheEntry | undefined> {
        this.assertAvailable();
        return this.entries.get(this.cacheKey(key));
    }

    async acquireLease(key: GeminiContextCacheCoordinationKey): Promise<string | undefined> {
        this.assertAvailable();
        const cacheKey = this.cacheKey(key);
        if (this.leases.has(cacheKey)) return undefined;
        const token = `lease-${this.leases.size}-${Math.random()}`;
        this.leases.set(cacheKey, token);
        return token;
    }

    async waitForEntry(
        key: GeminiContextCacheCoordinationKey,
        timeoutMs: number,
    ): Promise<GeminiContextCacheEntry | undefined> {
        this.assertAvailable();
        const cacheKey = this.cacheKey(key);
        const deadline = Date.now() + timeoutMs;
        while (Date.now() < deadline) {
            const entry = this.entries.get(cacheKey);
            if (entry) return entry;
            if (!this.leases.has(cacheKey) || this.cooldowns.has(this.scopeKey(key))) return undefined;
            await new Promise((resolve) => setTimeout(resolve, 1));
        }
        return undefined;
    }

    async publishEntry(
        key: GeminiContextCacheCoordinationKey,
        leaseToken: string,
        entry: GeminiContextCacheEntry,
    ): Promise<boolean> {
        this.assertAvailable();
        const cacheKey = this.cacheKey(key);
        if (this.leases.get(cacheKey) !== leaseToken) return false;
        this.entries.set(cacheKey, entry);
        return true;
    }

    async releaseLease(key: GeminiContextCacheCoordinationKey, leaseToken: string): Promise<void> {
        this.assertAvailable();
        const cacheKey = this.cacheKey(key);
        if (this.leases.get(cacheKey) === leaseToken) this.leases.delete(cacheKey);
    }

    async invalidateEntry(key: GeminiContextCacheCoordinationKey, expectedName: string): Promise<void> {
        this.assertAvailable();
        const cacheKey = this.cacheKey(key);
        if (this.entries.get(cacheKey)?.name === expectedName) {
            this.entries.delete(cacheKey);
            this.invalidations++;
        }
    }

    async getCooldownUntil(key: GeminiContextCacheCoordinationKey): Promise<number | undefined> {
        this.assertAvailable();
        const until = this.cooldowns.get(this.scopeKey(key));
        return until && until > Date.now() ? until : undefined;
    }

    async setCooldownUntil(key: GeminiContextCacheCoordinationKey, untilMs: number): Promise<void> {
        this.assertAvailable();
        const scopeKey = this.scopeKey(key);
        this.cooldowns.set(scopeKey, Math.max(this.cooldowns.get(scopeKey) ?? 0, untilMs));
    }

    async acquireCreatePermit(key: GeminiContextCacheCoordinationKey, limit: number): Promise<string | undefined> {
        this.assertAvailable();
        const prefix = `${this.scopeKey(key)}:`;
        if ([...this.permits].filter((permit) => permit.startsWith(prefix)).length >= limit) return undefined;
        const token = `${prefix}${Math.random()}`;
        this.permits.add(token);
        return token;
    }

    async releaseCreatePermit(_key: GeminiContextCacheCoordinationKey, permitToken: string): Promise<void> {
        this.assertAvailable();
        this.permits.delete(permitToken);
    }
}

/** The display name the driver must give — and look for — for the Balise route prefix. */
function routePrefixDisplayName(model = MODEL): string {
    const prefix = deriveGeminiCachePrefix(baliseRoutePrompt());
    if (!prefix) throw new Error('the route prompt must have a cacheable prefix');
    return geminiCacheDisplayName(model, geminiCacheContentHash(model, prefix, undefined, undefined));
}

function makeDriver(options: Partial<VertexAIDriverOptions> = {}) {
    const warn = vi.fn();
    const logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() } as unknown as Logger;
    const requests: GenerateContentParameters[] = [];
    const generateContent = vi.fn(async (request: GenerateContentParameters) => {
        requests.push(request);
        return completionResponse();
    });
    const generateContentStream = vi.fn(async (request: GenerateContentParameters) => {
        requests.push(request);
        return (async function* () {
            yield completionResponse();
        })();
    });
    const create = vi.fn(async (_params: CreateCachedContentParameters) => cachedContent());
    const update = vi.fn(async (_params: UpdateCachedContentParameters) => cachedContent());
    // What `caches.list` reports — i.e. what caches other instances of the fleet have already built.
    const listed: CachedContent[] = [];
    const list = vi.fn(async (_params?: ListCachedContentsParameters) => {
        const page = [...listed];
        return {
            async *[Symbol.asyncIterator]() {
                yield* page;
            },
        } as unknown as Pager<CachedContent>;
    });

    class TestVertexAIDriver extends VertexAIDriver {
        override getGoogleGenAIClient(): GoogleGenAI {
            return {
                models: { generateContent, generateContentStream },
                caches: { create, update, list },
            } as unknown as GoogleGenAI;
        }
    }

    const driver = new TestVertexAIDriver({
        project: 'test-project',
        region: 'us-central1',
        logger,
        ...options,
    });
    return { driver, warn, requests, generateContent, generateContentStream, create, update, list, listed };
}

function cachedOptions(overrides: Partial<ExecutionOptions> = {}): ExecutionOptions {
    return { model: MODEL, prompt_cache_key: 'route-catalog-v3', ...overrides };
}

describe('deriveGeminiCachePrefix', () => {
    it('caches the leading static text and never the final turn', () => {
        const prefix = deriveGeminiCachePrefix(baliseRoutePrompt());

        expect(prefix?.system).toEqual({ role: 'user', parts: [{ text: 'Route each photo to a domain.' }] });
        expect(prefix?.contents).toEqual([{ role: 'user', parts: [{ text: CATALOG_TEXT }] }]);
        expect(prefix?.partCount).toBe(1);
    });

    it('stops at the first block holding a non-text part', () => {
        const prefix = deriveGeminiCachePrefix({
            contents: [
                { role: 'user', parts: [{ text: 'instructions' }] },
                { role: 'user', parts: [{ text: 'exhibit' }, { fileData: { fileUri: 'gs://b/o' } }] },
                { role: 'model', parts: [{ text: 'noted' }] },
                { role: 'user', parts: [{ text: 'question' }] },
            ],
        });

        expect(prefix?.contents).toEqual([{ role: 'user', parts: [{ text: 'instructions' }] }]);
    });

    it('does not cache signed thoughts or tool traffic', () => {
        const prefix = deriveGeminiCachePrefix({
            contents: [
                { role: 'model', parts: [{ text: 'plan', thought: true, thoughtSignature: 'sig' }] },
                { role: 'user', parts: [{ text: 'go' }] },
            ],
        });

        expect(prefix).toBeUndefined();
    });

    it('has nothing to cache for a single-turn prompt without a system instruction', () => {
        expect(deriveGeminiCachePrefix({ contents: [{ role: 'user', parts: [{ text: 'hello' }] }] })).toBeUndefined();
    });
});

describe('cache identity', () => {
    const prefix = { contents: [{ role: 'user', parts: [{ text: CATALOG_TEXT }] }], partCount: 1 };
    const otherPrefix = { contents: [{ role: 'user', parts: [{ text: 'other catalog' }] }], partCount: 1 };

    it('names a cache deterministically from its content', () => {
        const first = geminiCacheDisplayName(MODEL, geminiCacheContentHash(MODEL, prefix, undefined, undefined));
        const second = geminiCacheDisplayName(MODEL, geminiCacheContentHash(MODEL, prefix, undefined, undefined));

        expect(first).toBe(second);
        expect(first).toMatch(new RegExp(`^${GEMINI_CACHE_DISPLAY_NAME_PREFIX}:gemini-3\\.7-flash:[0-9a-f]{16}$`));
    });

    it('gives different content different names', () => {
        expect(geminiCacheContentHash(MODEL, prefix, undefined, undefined)).not.toBe(
            geminiCacheContentHash(MODEL, otherPrefix, undefined, undefined),
        );
        expect(geminiCacheContentHash(MODEL, prefix, undefined, undefined)).not.toBe(
            geminiCacheContentHash('gemini-3.7-pro', prefix, undefined, undefined),
        );
        expect(geminiCacheContentHash(MODEL, prefix, undefined, undefined)).not.toBe(
            geminiCacheContentHash(MODEL, prefix, [{ functionDeclarations: [{ name: 'lookup' }] }], undefined),
        );
        expect(geminiCacheContentHash(MODEL, prefix, undefined, undefined)).not.toBe(
            geminiCacheContentHash(
                MODEL,
                { ...prefix, system: { role: 'user', parts: [{ text: 's' }] } },
                undefined,
                undefined,
            ),
        );
    });

    it('stays inside the Vertex display-name budget for a long model id', () => {
        const longModel = 'publishers/google/models/gemini-9.9-flash-lite-preview-experimental-09-2031';
        const name = geminiCacheDisplayName(longModel, 'a'.repeat(64));

        expect(name.length).toBeLessThanOrEqual(128);
        // The publisher path is not part of the identity — only the model id the request carries.
        expect(name).toBe(
            `${GEMINI_CACHE_DISPLAY_NAME_PREFIX}:gemini-9.9-flash-lite-preview-experimental-09-2031:${'a'.repeat(16)}`,
        );
    });
});

describe('stripLeadingParts', () => {
    it('removes the prefix parts and drops the blocks that empty out', () => {
        const contents = [{ role: 'user', parts: [{ text: 'a' }, { text: 'b' }, { text: 'c' }] }];

        expect(stripLeadingParts(contents, 2)).toEqual([{ role: 'user', parts: [{ text: 'c' }] }]);
    });

    it('refuses to strip more parts than the request holds', () => {
        expect(stripLeadingParts([{ role: 'user', parts: [{ text: 'a' }] }], 3)).toBeUndefined();
    });
});

describe('Gemini explicit context caching', () => {
    it('caches the system instruction and catalog, and sends only the dynamic turn', async () => {
        const { driver, create, requests, warn } = makeDriver();

        await new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());

        expect(create).toHaveBeenCalledTimes(1);
        expect(create.mock.calls[0][0]).toMatchObject({
            model: MODEL,
            config: {
                systemInstruction: { role: 'user', parts: [{ text: 'Route each photo to a domain.' }] },
                contents: [{ role: 'user', parts: [{ text: CATALOG_TEXT }] }],
                ttl: '1800s',
                // Deterministic in the content, so a peer instance can find this resource by listing.
                displayName: routePrefixDisplayName(),
            },
        });
        expect(requests[0].contents).toEqual([
            {
                role: 'user',
                parts: [{ text: 'Photo 42: route it.' }, { inlineData: { data: 'aW1hZ2U=', mimeType: 'image/jpeg' } }],
            },
        ]);
        expect(requests[0].config?.cachedContent).toBe(CACHE_NAME);
        // Vertex rejects a request that repeats the cached system instruction.
        expect(requests[0].config?.systemInstruction).toBeUndefined();
        expect(warn).not.toHaveBeenCalled();
    });

    it('reuses the memoized cache without listing or creating again', async () => {
        const { driver, create, list, requests } = makeDriver();
        const model = new GeminiModelDefinition(MODEL);

        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());
        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());

        expect(create).toHaveBeenCalledTimes(1);
        // Discovery is a cold-start cost, not a per-call one.
        expect(list).toHaveBeenCalledTimes(1);
        expect(requests).toHaveLength(2);
        expect(requests[1].config?.cachedContent).toBe(CACHE_NAME);
    });

    it('collapses sharded cache keys onto one cache', async () => {
        const { driver, create, requests } = makeDriver();
        const model = new GeminiModelDefinition(MODEL);

        // Callers shard prompt_cache_key to spread load. The prefix is byte-identical, so the
        // shards must share one resource rather than mint one each.
        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions({ prompt_cache_key: 'route:0' }));
        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions({ prompt_cache_key: 'route:3' }));

        expect(create).toHaveBeenCalledTimes(1);
        expect(requests[0].config?.cachedContent).toBe(CACHE_NAME);
        expect(requests[1].config?.cachedContent).toBe(CACHE_NAME);
    });

    it('adopts a live cache a peer instance already created', async () => {
        const { driver, create, list, listed, requests, warn } = makeDriver();
        listed.push({
            name: PEER_CACHE_NAME,
            displayName: routePrefixDisplayName(),
            expireTime: new Date(Date.now() + 20 * 60 * 1000).toISOString(),
        });

        await new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());

        expect(list).toHaveBeenCalledTimes(1);
        expect(create).not.toHaveBeenCalled();
        expect(requests[0].config?.cachedContent).toBe(PEER_CACHE_NAME);
        expect(warn).not.toHaveBeenCalled();
    });

    it('ignores listed caches that are expired, undated, or hold other content', async () => {
        const { driver, create, listed, requests } = makeDriver();
        const displayName = routePrefixDisplayName();
        listed.push(
            // Right content, already dead: adopting it would only buy a 404.
            {
                name: `${PEER_CACHE_NAME}-expired`,
                displayName,
                expireTime: new Date(Date.now() - 60_000).toISOString(),
            },
            // Right content, no expiry reported: its lifetime is unknowable, so it is not adopted.
            { name: `${PEER_CACHE_NAME}-undated`, displayName },
            // Live, but it caches a different prefix.
            {
                name: `${PEER_CACHE_NAME}-other`,
                displayName: geminiCacheDisplayName(
                    MODEL,
                    geminiCacheContentHash(
                        MODEL,
                        { contents: [{ role: 'user', parts: [{ text: 'another catalog' }] }], partCount: 1 },
                        undefined,
                        undefined,
                    ),
                ),
                expireTime: new Date(Date.now() + 20 * 60 * 1000).toISOString(),
            },
        );

        await new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());

        expect(create).toHaveBeenCalledTimes(1);
        expect(requests[0].config?.cachedContent).toBe(CACHE_NAME);
    });

    it('moves declared tools into the cache and out of the request', async () => {
        const { driver, create, requests } = makeDriver();
        const tools = [{ name: 'lookup', input_schema: { type: 'object' } }];

        await new GeminiModelDefinition(MODEL).requestTextCompletion(
            driver,
            baliseRoutePrompt(),
            cachedOptions({ tools }),
        );

        expect(create.mock.calls[0][0].config?.tools).toEqual([
            {
                functionDeclarations: [
                    { name: 'lookup', description: undefined, parametersJsonSchema: tools[0].input_schema },
                ],
            },
        ]);
        expect(requests[0].config?.tools).toBeUndefined();
        expect(requests[0].config?.toolConfig).toBeUndefined();
    });

    it('honours the per-execution TTL', async () => {
        const { driver, create } = makeDriver({ geminiContextCacheTtlSeconds: 600 });

        await new GeminiModelDefinition(MODEL).requestTextCompletion(
            driver,
            baliseRoutePrompt(),
            cachedOptions({ prompt_cache_ttl_seconds: 120 }),
        );

        expect(create.mock.calls[0][0].config?.ttl).toBe('120s');
    });

    it('falls back to the driver TTL when the execution does not set one', async () => {
        const { driver, create } = makeDriver({ geminiContextCacheTtlSeconds: 600 });

        await new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());

        expect(create.mock.calls[0][0].config?.ttl).toBe('600s');
    });

    it('sends the full un-cached request when cache creation fails', async () => {
        const { driver, create, requests, warn } = makeDriver();
        create.mockRejectedValue(apiError(400, 'cache creation rejected'));

        const completion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );

        expect(warn).toHaveBeenCalledTimes(1);
        expect(requests[0].config?.cachedContent).toBeUndefined();
        expect(requests[0].config?.systemInstruction).toEqual({
            role: 'user',
            parts: [{ text: 'Route each photo to a domain.' }],
        });
        expect(requests[0].contents).toEqual(baliseRoutePrompt().contents);
        expect(completion.result).toEqual([{ type: 'text', value: 'places' }]);
    });

    it('stops retrying a prefix Vertex refuses to cache', async () => {
        const { driver, create, warn } = makeDriver();
        create.mockRejectedValue(apiError(400, 'Cached content is too small. The minimum token count is 2048.'));
        const model = new GeminiModelDefinition(MODEL);

        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());
        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());

        expect(create).toHaveBeenCalledTimes(1);
        expect(warn).toHaveBeenCalledTimes(1);
    });

    it('does not classify a minimum TTL error as a minimum token error', () => {
        expect(isMinimumTokenCountError(apiError(400, 'TTL must be at least the minimum of 60 seconds.'))).toBe(false);
    });

    it('rediscovers, then recreates once, when the resource has gone', async () => {
        const { driver, create, list, listed, generateContent, warn } = makeDriver();
        const staleName = `${CACHE_NAME}-stale`;
        create
            .mockImplementationOnce(async () => {
                const entry = cachedContent(staleName);
                // Vertex keeps advertising a resource after it stops being usable, so the listing
                // must not hand it back to the very call that just proved it dead.
                listed.push({ ...entry, displayName: routePrefixDisplayName() });
                return entry;
            })
            .mockImplementationOnce(async () => cachedContent());
        generateContent.mockRejectedValueOnce(apiError(404, 'CachedContent not found.'));

        const completion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );

        // Listed twice: the cold-start sweep, then the sweep that precedes the recreate.
        expect(list).toHaveBeenCalledTimes(2);
        expect(create).toHaveBeenCalledTimes(2);
        expect(generateContent).toHaveBeenCalledTimes(2);
        expect(generateContent.mock.calls[0][0].config?.cachedContent).toBe(staleName);
        expect(generateContent.mock.calls[1][0].config?.cachedContent).toBe(CACHE_NAME);
        expect(warn).toHaveBeenCalledTimes(1);
        expect(completion.result).toEqual([{ type: 'text', value: 'places' }]);
    });

    it('does not swallow errors that have nothing to do with the cache', async () => {
        const { driver, generateContent } = makeDriver();
        generateContent.mockRejectedValueOnce(apiError(429, 'RESOURCE_EXHAUSTED'));

        await expect(
            new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions()),
        ).rejects.toThrow('RESOURCE_EXHAUSTED');
    });

    it('routes the streaming path through the same cache', async () => {
        const { driver, create, requests } = makeDriver();

        const stream = await new GeminiModelDefinition(MODEL).requestTextCompletionStream(
            driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );
        for await (const _chunk of stream) {
            /* drain */
        }

        expect(create).toHaveBeenCalledTimes(1);
        expect(requests[0].config?.cachedContent).toBe(CACHE_NAME);
    });

    it('reports explicit cache reads as prompt_cached', async () => {
        const { driver, generateContent } = makeDriver();
        generateContent.mockResolvedValueOnce(
            completionResponse({
                usageMetadata: {
                    promptTokenCount: 4200,
                    cachedContentTokenCount: 4000,
                    candidatesTokenCount: 30,
                    totalTokenCount: 4230,
                },
            }),
        );

        const completion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );

        expect(completion.token_usage).toEqual({
            prompt: 4200,
            prompt_new: 200,
            prompt_cached: 4000,
            result: 30,
            total: 4230,
        });
    });

    it('deduplicates 20 concurrent creates inside one driver process', async () => {
        const { driver, create } = makeDriver();
        create.mockImplementation(async () => {
            await new Promise((resolve) => setTimeout(resolve, 10));
            return cachedContent();
        });
        const model = new GeminiModelDefinition(MODEL);

        await Promise.all(
            Array.from({ length: 20 }, () => model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions())),
        );

        expect(create).toHaveBeenCalledTimes(1);
    });

    it('leaves transient create retry policy to the host workflow', async () => {
        const { driver, create } = makeDriver();
        create.mockRejectedValueOnce(apiError(503, 'UNAVAILABLE'));

        const completion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );

        expect(create).toHaveBeenCalledTimes(1);
        expect(completion.prompt_cache_diagnostic).toMatchObject({
            path: 'fallback_provider_error',
            provider_status: 503,
        });
    });
});

describe('Gemini distributed context cache coordination', () => {
    const coordinatedOptions = (coordinator: FakeDistributedCoordinator): Partial<VertexAIDriverOptions> => ({
        geminiContextCacheCoordinator: coordinator,
        geminiContextCacheScope: 'environment-1',
    });

    it('produces one create across 20 driver instances sharing a coordinator', async () => {
        const coordinator = new FakeDistributedCoordinator();
        const harnesses = Array.from({ length: 20 }, () => makeDriver(coordinatedOptions(coordinator)));

        const completions = await Promise.all(
            harnesses.map(({ driver }) =>
                new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions()),
            ),
        );

        expect(harnesses.reduce((sum, harness) => sum + harness.create.mock.calls.length, 0)).toBe(1);
        expect(
            completions.filter((completion) => completion.prompt_cache_diagnostic?.explicit_cache_used),
        ).toHaveLength(20);
    });

    it('converges route and extraction calls with identical cached contents', async () => {
        const coordinator = new FakeDistributedCoordinator();
        const route = makeDriver(coordinatedOptions(coordinator));
        const extraction = makeDriver(coordinatedOptions(coordinator));
        const extractionPrompt = baliseRoutePrompt();
        extractionPrompt.contents[1] = {
            role: 'user',
            parts: [
                { text: 'Photo 99: extract fields.' },
                { inlineData: { data: 'cGhvdG8=', mimeType: 'image/jpeg' } },
            ],
        };

        await Promise.all([
            new GeminiModelDefinition(MODEL).requestTextCompletion(route.driver, baliseRoutePrompt(), cachedOptions()),
            new GeminiModelDefinition(MODEL).requestTextCompletion(
                extraction.driver,
                extractionPrompt,
                cachedOptions(),
            ),
        ]);

        expect(route.create.mock.calls.length + extraction.create.mock.calls.length).toBe(1);
        expect(route.requests[0].config?.cachedContent).toBe(extraction.requests[0].config?.cachedContent);
    });

    it('never shares a resource for different cache contents even when keys match', async () => {
        const coordinator = new FakeDistributedCoordinator();
        const first = makeDriver(coordinatedOptions(coordinator));
        const second = makeDriver(coordinatedOptions(coordinator));
        const otherPrompt = baliseRoutePrompt();
        otherPrompt.contents[0] = { role: 'user', parts: [{ text: `${CATALOG_TEXT}different` }] };

        await Promise.all([
            new GeminiModelDefinition(MODEL).requestTextCompletion(first.driver, baliseRoutePrompt(), cachedOptions()),
            new GeminiModelDefinition(MODEL).requestTextCompletion(second.driver, otherPrompt, cachedOptions()),
        ]);

        expect(first.create.mock.calls.length + second.create.mock.calls.length).toBe(2);
        expect(coordinator.entries.size).toBe(2);
    });

    it('shares a 429 cooldown and allows exactly one retry after it clears', async () => {
        const coordinator = new FakeDistributedCoordinator();
        const first = makeDriver(coordinatedOptions(coordinator));
        const second = makeDriver(coordinatedOptions(coordinator));
        first.create.mockRejectedValue(
            Object.assign(apiError(429, 'RESOURCE_EXHAUSTED'), { headers: { 'retry-after': '2' } }),
        );

        const firstCompletion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            first.driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );
        const secondCompletion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            second.driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );

        expect(first.create).toHaveBeenCalledTimes(1);
        expect(second.create).not.toHaveBeenCalled();
        expect(firstCompletion.prompt_cache_diagnostic?.path).toBe('fallback_quota');
        expect(secondCompletion.prompt_cache_diagnostic?.path).toBe('fallback_quota');
        expect([...coordinator.cooldowns.values()][0] - Date.now()).toBeGreaterThan(1_500);

        coordinator.cooldowns.clear();
        const retries = [makeDriver(coordinatedOptions(coordinator)), makeDriver(coordinatedOptions(coordinator))];
        await Promise.all(
            retries.map(({ driver }) =>
                new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions()),
            ),
        );
        expect(retries.reduce((sum, harness) => sum + harness.create.mock.calls.length, 0)).toBe(1);
    });

    it('coordinates a near-expiry TTL refresh once', async () => {
        const coordinator = new FakeDistributedCoordinator();
        const seed = makeDriver(coordinatedOptions(coordinator));
        const prefix = deriveGeminiCachePrefix(baliseRoutePrompt());
        if (!prefix) throw new Error('expected cacheable prefix');
        const contentHash = geminiCacheContentHash(MODEL, prefix, undefined, undefined);
        const key = seed.driver.getGeminiContextCacheCoordinationKey('us-central1', MODEL, contentHash);
        coordinator.entries.set(JSON.stringify([key.scope, key.project, key.location, key.model, key.contentHash]), {
            name: CACHE_NAME,
            expiresAtMs: Date.now() + 60_000,
        });
        const harnesses = Array.from({ length: 20 }, () => makeDriver(coordinatedOptions(coordinator)));

        await Promise.all(
            harnesses.map(({ driver }) =>
                new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions()),
            ),
        );

        expect(harnesses.reduce((sum, harness) => sum + harness.update.mock.calls.length, 0)).toBe(1);
        expect(harnesses.reduce((sum, harness) => sum + harness.create.mock.calls.length, 0)).toBe(0);
    });

    it('invalidates a 404 by expected name and coordinates one recreation', async () => {
        const coordinator = new FakeDistributedCoordinator();
        const harness = makeDriver(coordinatedOptions(coordinator));
        const prefix = deriveGeminiCachePrefix(baliseRoutePrompt());
        if (!prefix) throw new Error('expected cacheable prefix');
        const contentHash = geminiCacheContentHash(MODEL, prefix, undefined, undefined);
        const key = harness.driver.getGeminiContextCacheCoordinationKey('us-central1', MODEL, contentHash);
        coordinator.entries.set(JSON.stringify([key.scope, key.project, key.location, key.model, key.contentHash]), {
            name: PEER_CACHE_NAME,
            expiresAtMs: Date.now() + 20 * 60_000,
        });
        harness.generateContent.mockRejectedValueOnce(apiError(404, 'CachedContent not found.'));

        const completion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            harness.driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );

        expect(coordinator.invalidations).toBe(1);
        expect(harness.create).toHaveBeenCalledTimes(1);
        expect(completion.prompt_cache_diagnostic?.path).toBe('unusable_resource_recreated');
    });

    it('preserves inference without any create when coordination is unavailable', async () => {
        const coordinator = new FakeDistributedCoordinator();
        coordinator.unavailable = true;
        const harnesses = Array.from({ length: 20 }, () => makeDriver(coordinatedOptions(coordinator)));

        const completions = await Promise.all(
            harnesses.map(({ driver }) =>
                new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions()),
            ),
        );

        expect(harnesses.reduce((sum, harness) => sum + harness.create.mock.calls.length, 0)).toBe(0);
        expect(completions.every((completion) => completion.result[0]?.value === 'places')).toBe(true);
        expect(
            completions.every(
                (completion) => completion.prompt_cache_diagnostic?.path === 'fallback_coordination_unavailable',
            ),
        ).toBe(true);
    });
});

describe('Gemini PromptSegment cache breakpoint', () => {
    function image(): DataSource {
        return {
            name: 'photo.jpg',
            mime_type: 'image/jpeg',
            getURI: vi.fn().mockResolvedValue('gs://balise-dev/photo.jpg'),
            getURL: vi.fn(),
            getStream: vi.fn(),
        } as unknown as DataSource;
    }

    const segments = (): PromptSegment[] => [
        { role: PromptRole.system, content: 'Route each photo to a domain.' },
        { role: PromptRole.user, content: CATALOG_TEXT },
        { role: PromptRole.user, content: 'Route this photo.', files: [image()] },
    ];

    it('keeps stable user text in cachedContents and the image task in both provider paths', async () => {
        const blocking = makeDriver();
        const streaming = makeDriver();
        const model = new GeminiModelDefinition(MODEL);
        const blockingPrompt = await model.createPrompt(blocking.driver, segments(), cachedOptions());
        const streamingPrompt = await model.createPrompt(streaming.driver, segments(), cachedOptions());

        expect(blockingPrompt.contents).toHaveLength(2);
        await model.requestTextCompletion(blocking.driver, blockingPrompt, cachedOptions());
        const stream = await model.requestTextCompletionStream(streaming.driver, streamingPrompt, cachedOptions());
        for await (const _chunk of stream) {
            // Drain the native stream so its final diagnostic and conversation are available.
        }

        for (const harness of [blocking, streaming]) {
            expect(harness.create.mock.calls[0][0].config?.contents).toEqual([
                { role: 'user', parts: [{ text: CATALOG_TEXT }] },
            ]);
            expect(harness.requests[0].contents).toEqual([
                {
                    role: 'user',
                    parts: [
                        { text: 'Route this photo.' },
                        { fileData: { fileUri: 'gs://balise-dev/photo.jpg', mimeType: 'image/jpeg' } },
                    ],
                },
            ]);
        }
    });

    it('keeps a trailing model segment out of cachedContents.create', async () => {
        const harness = makeDriver();
        const prompt = baliseRoutePrompt();
        prompt.contents.splice(1, 0, { role: 'model', parts: [{ text: 'Ready to route a photo.' }] });

        await new GeminiModelDefinition(MODEL).requestTextCompletion(harness.driver, prompt, cachedOptions());

        expect(harness.create.mock.calls[0][0].config?.contents).toEqual([
            { role: 'user', parts: [{ text: CATALOG_TEXT }] },
        ]);
        expect(harness.requests[0].contents).toEqual([
            { role: 'model', parts: [{ text: 'Ready to route a photo.' }] },
            prompt.contents[2],
        ]);
    });
});

describe('Gemini explicit context caching opt-out', () => {
    async function capturePayload(options: ExecutionOptions, driverOptions: Partial<VertexAIDriverOptions> = {}) {
        const { driver, create, requests, warn } = makeDriver(driverOptions);
        await new GeminiModelDefinition(MODEL).requestTextCompletion(driver, baliseRoutePrompt(), options);
        return { request: requests[0], create, warn };
    }

    it('leaves the provider payload unchanged when prompt_cache_mode is off', async () => {
        const options = cachedOptions({ prompt_cache_mode: 'off' });
        const { request, create, warn } = await capturePayload(options);

        expect(request).toEqual(getGeminiPayload(options, baliseRoutePrompt()));
        expect(request.config).not.toHaveProperty('cachedContent');
        expect(create).not.toHaveBeenCalled();
        expect(warn).not.toHaveBeenCalled();
    });

    it('leaves the provider payload unchanged when the driver kill switch is off', async () => {
        const options = cachedOptions();
        const { request, create } = await capturePayload(options, { geminiContextCache: false });

        expect(request).toEqual(getGeminiPayload(options, baliseRoutePrompt()));
        expect(request.config).not.toHaveProperty('cachedContent');
        expect(create).not.toHaveBeenCalled();
    });

    it('leaves the provider payload unchanged when no cache key is supplied', async () => {
        const options: ExecutionOptions = { model: MODEL };
        const { request, create } = await capturePayload(options);

        expect(request).toEqual(getGeminiPayload(options, baliseRoutePrompt()));
        expect(create).not.toHaveBeenCalled();
    });

    it('surfaces preparation failure when prompt_cache_mode is required', async () => {
        const { driver, create, generateContent } = makeDriver();
        create.mockRejectedValue(apiError(400, 'cache creation rejected'));

        await expect(
            new GeminiModelDefinition(MODEL).requestTextCompletion(
                driver,
                baliseRoutePrompt(),
                cachedOptions({ prompt_cache_mode: 'required' }),
            ),
        ).rejects.toBeInstanceOf(GeminiContextCacheRequiredError);
        expect(generateContent).not.toHaveBeenCalled();
    });

    it('exposes no registry when the driver kill switch is off', () => {
        const { driver } = makeDriver({ geminiContextCache: false });

        expect(driver.getGeminiContextCacheManager()).toBeUndefined();
    });
});
