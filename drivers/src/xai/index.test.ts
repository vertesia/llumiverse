import { Providers } from '@llumiverse/core';
import type { FetchClient } from '@vertesia/api-fetch-client';
import { describe, expect, it, vi } from 'vitest';
import { xAIDriver } from './index.js';

describe('xAI model listing', () => {
    it('uses verified catalog capabilities instead of runtime modality claims', async () => {
        const driver = new xAIDriver({ apiKey: 'test-key' });
        const get = vi.fn(async () => ({
            models: [
                {
                    id: 'grok-4.3',
                    owned_by: 'xAI',
                    input_modalities: ['audio'],
                    output_modalities: ['video'],
                },
                {
                    id: 'text-embedding-future',
                    owned_by: 'xAI',
                    input_modalities: ['text'],
                    output_modalities: ['embedding'],
                },
            ],
        }));
        driver.xai_service = { get } as unknown as FetchClient;

        expect(await driver.listModels()).toEqual([
            expect.objectContaining({
                id: 'grok-4.3',
                provider: Providers.xai,
                input_modalities: ['text', 'image'],
                output_modalities: ['text'],
                tool_support: true,
            }),
        ]);
    });
});
