import type Anthropic from '@anthropic-ai/sdk';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from './index.js';

describe('Anthropic model listing', () => {
    it('reports the deprecation and retirement dates Anthropic lists', async () => {
        const driver = new AnthropicDriver({ apiKey: 'test-key' });
        const list = vi.fn(async () => ({
            data: [
                {
                    id: 'claude-sonnet-4-5-20250929',
                    display_name: 'Claude Sonnet 4.5',
                    lifecycle: 'deprecated',
                    deprecated_at: '2026-09-30T00:00:00Z',
                    retires_at: '2026-11-30T00:00:00Z',
                },
                {
                    id: 'claude-opus-5-5',
                    display_name: 'Claude Opus 5.5',
                    lifecycle: 'active',
                    deprecated_at: null,
                    retires_at: null,
                },
            ],
        }));
        driver.client = { models: { list } } as unknown as Anthropic;

        const models = await driver.listModels();
        expect(models.find((model) => model.id === 'claude-sonnet-4-5-20250929')).toMatchObject({
            deprecation_date: '2026-09-30',
            retirement_date: '2026-11-30',
        });
        const active = models.find((model) => model.id === 'claude-opus-5-5');
        expect(active).not.toHaveProperty('deprecation_date');
        expect(active).not.toHaveProperty('retirement_date');
    });
});
