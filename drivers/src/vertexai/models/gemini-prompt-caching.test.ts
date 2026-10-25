import { type GenerateContentResponseUsageMetadata, MediaModality } from '@google/genai';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { GenerateContentPrompt, VertexAIDriver } from '../index.js';
import { GeminiModelDefinition, getGeminiPayload } from './gemini.js';

describe('Gemini implicit prompt caching', () => {
    const prompt: GenerateContentPrompt = {
        system: { role: 'user', parts: [{ text: 'stable system' }] },
        contents: [
            { role: 'user', parts: [{ text: 'stable document source' }] },
            { role: 'user', parts: [{ text: 'dynamic extraction task' }] },
        ],
    };
    const options: ExecutionOptions = { model: 'gemini-2.5-flash' };

    // `getGeminiPayload` is pure and stays that way: explicit context caching needs the driver's
    // client, so it rewrites the payload in the execution path instead. See
    // gemini-context-cache.test.ts for the payload a cached execution actually sends.
    it('keeps the provider payload identical when a routing identity is supplied', () => {
        const baseline = getGeminiPayload(options, prompt);
        const routed = getGeminiPayload({ ...options, prompt_cache_key: 'document-prefix' }, prompt);

        expect(routed).toEqual(baseline);
    });

    it('reports tokens served by the implicit cache', () => {
        const model = new GeminiModelDefinition('gemini-2.5-flash');
        const driver = { logger: { warn: vi.fn() } } as unknown as VertexAIDriver;
        const usage = {
            promptTokenCount: 125,
            cachedContentTokenCount: 100,
            candidatesTokenCount: 10,
            totalTokenCount: 135,
        } satisfies GenerateContentResponseUsageMetadata;

        expect(model.usageMetadataToTokenUsage(driver, usage)).toEqual({
            prompt: 125,
            prompt_new: 25,
            prompt_cached: 100,
            result: 10,
            total: 135,
        });
        expect(driver.logger.warn).not.toHaveBeenCalled();
    });

    it('reports the image tokens of a generated image apart from the text', () => {
        const model = new GeminiModelDefinition('gemini-2.5-flash-image');
        const driver = { logger: { warn: vi.fn() } } as unknown as VertexAIDriver;
        const usage = {
            promptTokenCount: 12,
            candidatesTokenCount: 1300,
            candidatesTokensDetails: [
                { modality: MediaModality.TEXT, tokenCount: 10 },
                { modality: MediaModality.IMAGE, tokenCount: 1290 },
            ],
            totalTokenCount: 1312,
        } satisfies GenerateContentResponseUsageMetadata;

        expect(model.usageMetadataToTokenUsage(driver, usage)).toEqual({
            prompt: 12,
            prompt_new: 12,
            result: 1300,
            result_image: 1290,
            total: 1312,
        });
    });
});
