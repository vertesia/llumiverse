import { describe, expect, it } from 'vitest';
import { getVertexAIModelLimits } from './vertexai-limits.js';

describe('Vertex AI model limits', () => {
    it.each([
        ['glm-5.2-maas', 1_000_000, 64_000],
        ['deepseek-v3.1-maas', 163_840, 32_768],
    ] as const)('uses documented context and output limits for %s', (model, context_window, max_output_tokens) => {
        expect(getVertexAIModelLimits(model)).toEqual({ context_window, max_output_tokens });
    });

    it.each([
        ['grok-4.7', 524_288],
        ['grok-4.6', 524_288],
        ['grok-4.3', 200_000],
        ['grok-4.20-reasoning', 2_000_000],
        ['grok-4.20-non-reasoning', 2_000_000],
        ['grok-4.1-fast-reasoning', 128_000],
        ['grok-4.1-fast-non-reasoning', 128_000],
        ['mistral-small-2503', 128_000],
        ['mistral-medium-3', 128_000],
        ['codestral-2', 128_000],
    ] as const)('overrides context without guessing output limits for %s', (model, context_window) => {
        expect(getVertexAIModelLimits(model)).toEqual({ context_window });
    });

    it('normalizes a complete Vertex resource name', () => {
        expect(
            getVertexAIModelLimits('projects/example/locations/global/publishers/zai-org/models/GLM-5.2-MAAS'),
        ).toEqual({ context_window: 1_000_000, max_output_tokens: 64_000 });
    });

    it.each(['grok-4.8', 'glm-5.3-maas', 'deepseek-v3.1', 'mistral-medium-3.6', ''])(
        'leaves unknown model %s to canonical family limits',
        (model) => {
            expect(getVertexAIModelLimits(model)).toEqual({});
        },
    );
});
