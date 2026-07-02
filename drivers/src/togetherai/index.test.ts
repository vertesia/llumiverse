import { ModelType, Providers } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { TogetherAIDriver } from './index.js';

describe('TogetherAIDriver', () => {
    it('maps Together array-shaped model catalog responses', async () => {
        const driver = new TogetherAIDriver({ apiKey: 'test-key' });
        const get = vi.fn(async () => [
            {
                id: 'Qwen/Qwen3.5-9B',
                display_name: 'Qwen3.5 9B',
                organization: 'Qwen',
                type: 'chat',
            },
            {
                id: 'togethercomputer/m2-bert-80M-8k-retrieval',
                display_name: 'M2 BERT Retrieval',
                organization: 'Together',
                type: 'embedding',
            },
        ]);
        driver.service = { get } as unknown as TogetherAIDriver['service'];

        const models = await driver.listModels();

        expect(get).toHaveBeenCalledWith('/models');
        expect(models).toEqual([
            expect.objectContaining({
                id: 'Qwen/Qwen3.5-9B',
                name: 'Qwen3.5 9B',
                owner: 'Qwen',
                provider: Providers.togetherai,
                type: ModelType.Chat,
                tool_support: true,
            }),
        ]);
    });

    it('keeps direct embedding generation support without listing embedding models', async () => {
        const driver = new TogetherAIDriver({ apiKey: 'test-key' });
        const create = vi.fn(async () => ({
            data: [
                { index: 1, embedding: [0.3, 0.4] },
                { index: 0, embedding: [0.1, 0.2] },
            ],
            usage: { prompt_tokens: 3, total_tokens: 3 },
        }));
        driver.service = {
            embeddings: { create },
        } as unknown as TogetherAIDriver['service'];

        const result = await driver.generateEmbeddings({
            model: 'togethercomputer/m2-bert-80M-8k-retrieval',
            inputs: [
                { type: 'text', text: 'first' },
                { type: 'text', text: 'second' },
            ],
        });

        expect(create).toHaveBeenCalledWith({
            input: ['first', 'second'],
            model: 'togethercomputer/m2-bert-80M-8k-retrieval',
            encoding_format: 'float',
        });
        expect(result.results).toEqual([
            { outputs: [{ values: [0.1, 0.2], modality: 'text' }] },
            { outputs: [{ values: [0.3, 0.4], modality: 'text' }] },
        ]);
        expect(result.usage).toEqual({ input_tokens: 3, input_text_tokens: 3 });
    });
});
