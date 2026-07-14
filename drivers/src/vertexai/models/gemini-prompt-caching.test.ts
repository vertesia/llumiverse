import type { GenerateContentResponseUsageMetadata } from '@google/genai';
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
});
