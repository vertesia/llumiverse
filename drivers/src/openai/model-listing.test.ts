import { Providers } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIDriver } from './openai.js';

describe('OpenAI model listing', () => {
    it('keeps executable Codex and pro reasoning models while excluding embeddings', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test-key' });
        const list = vi.fn(async () => ({
            data: [
                { id: 'gpt-5.6-codex', object: 'model', created: 1, owned_by: 'system' },
                { id: 'o1-pro', object: 'model', created: 1, owned_by: 'system' },
                { id: 'gpt-5-audiovisual', object: 'model', created: 1, owned_by: 'system' },
                { id: 'text-embedding-3-small', object: 'model', created: 1, owned_by: 'system' },
            ],
        }));
        driver.service = { models: { list } } as unknown as OpenAIDriver['service'];

        const models = await driver.listModels();
        expect(models).toHaveLength(3);
        expect(models).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ id: 'gpt-5.6-codex', provider: Providers.openai }),
                expect.objectContaining({ id: 'o1-pro', provider: Providers.openai }),
                expect.objectContaining({ id: 'gpt-5-audiovisual', provider: Providers.openai }),
            ]),
        );
    });
});
