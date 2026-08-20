import {
    type CachedContent,
    type CreateCachedContentParameters,
    FinishReason,
    type GenerateContentParameters,
    type GenerateContentResponse,
    type GoogleGenAI,
    type UpdateCachedContentParameters,
} from '@google/genai';
import type { ExecutionOptions, Logger } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { type GenerateContentPrompt, VertexAIDriver, type VertexAIDriverOptions } from '../index.js';
import { GeminiModelDefinition, getGeminiPayload } from './gemini.js';
import { deriveGeminiCachePrefix, stripLeadingParts } from './gemini-context-cache.js';

const MODEL = 'gemini-3.7-flash';
const CACHE_NAME = 'projects/test-project/locations/us-central1/cachedContents/1';
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

    class TestVertexAIDriver extends VertexAIDriver {
        override getGoogleGenAIClient(): GoogleGenAI {
            return {
                models: { generateContent, generateContentStream },
                caches: { create, update },
            } as unknown as GoogleGenAI;
        }
    }

    const driver = new TestVertexAIDriver({
        project: 'test-project',
        region: 'us-central1',
        logger,
        ...options,
    });
    return { driver, warn, requests, generateContent, generateContentStream, create, update };
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

    it('reuses the registered cache instead of creating a second one', async () => {
        const { driver, create, requests } = makeDriver();
        const model = new GeminiModelDefinition(MODEL);

        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());
        await model.requestTextCompletion(driver, baliseRoutePrompt(), cachedOptions());

        expect(create).toHaveBeenCalledTimes(1);
        expect(requests).toHaveLength(2);
        expect(requests[1].config?.cachedContent).toBe(CACHE_NAME);
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
        create.mockRejectedValueOnce(apiError(503, 'UNAVAILABLE'));

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
        expect(requests[0].contents).toHaveLength(1);
        expect((requests[0].contents as { parts: unknown[] }[])[0].parts).toHaveLength(3);
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

    it('recreates the cache once when the resource has expired', async () => {
        const { driver, create, generateContent, warn } = makeDriver();
        create.mockResolvedValueOnce(cachedContent(`${CACHE_NAME}-stale`)).mockResolvedValueOnce(cachedContent());
        generateContent.mockRejectedValueOnce(apiError(404, 'CachedContent not found.'));

        const completion = await new GeminiModelDefinition(MODEL).requestTextCompletion(
            driver,
            baliseRoutePrompt(),
            cachedOptions(),
        );

        expect(create).toHaveBeenCalledTimes(2);
        expect(generateContent).toHaveBeenCalledTimes(2);
        expect(generateContent.mock.calls[0][0].config?.cachedContent).toBe(`${CACHE_NAME}-stale`);
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

    it('exposes no registry when the driver kill switch is off', () => {
        const { driver } = makeDriver({ geminiContextCache: false });

        expect(driver.getGeminiContextCacheManager()).toBeUndefined();
    });
});
