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
});
