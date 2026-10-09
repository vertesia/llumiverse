import { PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { VertexAIDriver } from './index.js';
import { getListedVertexMistralModels, VERTEX_MISTRAL_CHAT_MODELS } from './mistral-models.js';
import { getModelDefinition } from './models.js';

function createDriverStub() {
    const post = vi.fn(async (_endpoint: string, options: { reader?: string }) => {
        if (options.reader === 'sse')
            return new ReadableStream({
                start(controller) {
                    controller.close();
                },
            });
        return {
            choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        };
    });
    const getFetchClientForRegion = vi.fn(() => ({ post }));
    return { driver: { getFetchClientForRegion } as unknown as VertexAIDriver, post, getFetchClientForRegion };
}

describe('Vertex regional Mistral chat models', () => {
    it('lists only supported chat models in both documented regions', () => {
        const models = getListedVertexMistralModels();
        expect(models.map((model) => model.id).sort()).toEqual(
            VERTEX_MISTRAL_CHAT_MODELS.flatMap((model) =>
                ['us-central1', 'europe-west4'].map(
                    (region) => `locations/${region}/publishers/mistralai/models/${model}`,
                ),
            ).sort(),
        );
        expect(models.every((model) => model.owner === 'mistralai' && model.can_stream)).toBe(true);
        expect(models.some((model) => model.id.includes('ocr'))).toBe(false);
    });

    it.each(VERTEX_MISTRAL_CHAT_MODELS)('routes unary and streaming %s through publisher endpoints', async (model) => {
        const modelId = `locations/europe-west4/publishers/mistralai/models/${model}`;
        const definition = getModelDefinition(modelId);
        const stub = createDriverStub();
        const options = { model: modelId, model_options: { _option_id: 'text-fallback' as const, temperature: 0 } };
        const prompt = await definition.createPrompt(
            stub.driver,
            [{ role: PromptRole.user, content: 'hello' }],
            options,
        );
        const completion = await definition.requestTextCompletion(stub.driver, prompt, options);
        expect(completion.result).toEqual([{ type: 'text', value: 'ok' }]);
        await definition.requestTextCompletionStream(stub.driver, prompt, options);
        expect(stub.getFetchClientForRegion).toHaveBeenCalledWith('europe-west4', undefined);
        expect(stub.post).toHaveBeenNthCalledWith(1, `publishers/mistralai/models/${model}:rawPredict`, {
            payload: expect.objectContaining({ model, stream: false, temperature: 0 }),
        });
        expect(stub.post).toHaveBeenNthCalledWith(2, `publishers/mistralai/models/${model}:streamRawPredict`, {
            payload: expect.objectContaining({ model, stream: true, temperature: 0 }),
            reader: 'sse',
        });
    });
});
