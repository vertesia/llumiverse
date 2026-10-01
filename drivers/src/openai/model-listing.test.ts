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
                { id: 'gpt-4o-mini-tts', object: 'model', created: 1, owned_by: 'system' },
                { id: 'gpt-4o-transcribe', object: 'model', created: 1, owned_by: 'system' },
                { id: 'gpt-audio', object: 'model', created: 1, owned_by: 'system' },
                { id: 'gpt-image-1', object: 'model', created: 1, owned_by: 'system' },
                { id: 'omni-moderation-latest', object: 'model', created: 1, owned_by: 'system' },
                { id: 'sora-3', object: 'model', created: 1, owned_by: 'system' },
            ],
        }));
        driver.service = { models: { list } } as unknown as OpenAIDriver['service'];

        const models = await driver.listModels();
<<<<<<< HEAD
        expect(models.map((model) => model.id)).not.toContain('sora-3');
=======
        expect(models).toHaveLength(4);
>>>>>>> e6d93ac (feat: support OpenAI image generation and editing (#722))
        expect(models).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ id: 'gpt-5.6-codex', provider: Providers.openai }),
                expect.objectContaining({ id: 'gpt-4o-mini-tts', provider: Providers.openai }),
                expect.objectContaining({ id: 'gpt-4o-transcribe', provider: Providers.openai }),
                expect.objectContaining({ id: 'gpt-audio', provider: Providers.openai, type: 'audio' }),
                expect.objectContaining({ id: 'o1-pro', provider: Providers.openai }),
                expect.objectContaining({ id: 'gpt-5-audiovisual', provider: Providers.openai }),
            ]),
        );
    });
});

it('discovers aliases, snapshots, qualified image IDs and future generations without advertising DALL-E', async () => {
    const driver = new OpenAIDriver({ apiKey: 'test-key' });
    const ids = [
        'gpt-image-2.5-sunburst',
        'gpt-image-2.5-flare-2026-09-08',
        'openai/GPT-IMAGE-3',
        'chatgpt-image-latest',
        'dall-e-2',
        'dall-e-3',
    ];
    driver.service.models.list = vi.fn().mockResolvedValue({ data: ids.map((id) => ({ id, owned_by: 'openai' })) });
    const models = await driver.listModels();
    expect(models.map((model) => model.id).sort()).toEqual(ids.slice(0, 4).sort());
    for (const model of models) {
        expect(model.type).toBe('image');
        expect(model.output_modalities).toEqual(['image']);
    }
});
