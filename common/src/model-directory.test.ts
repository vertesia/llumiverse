import { describe, expect, it } from 'vitest';
import { getModelCapabilities } from './capability.js';
import { resolveModelProfile } from './model-directory.js';
import { getOptions } from './options.js';
import { Providers } from './types.js';

describe('central model directory', () => {
    it('resolves provider-qualified Gemini models through OpenAI-compatible transport', () => {
        const profile = resolveModelProfile('google/gemini-3.5-flash', Providers.openai_compatible);
        const capabilities = getModelCapabilities('google/gemini-3.5-flash', Providers.openai_compatible);
        const options = getOptions('google/gemini-3.5-flash', Providers.openai_compatible);

        expect(profile.family).toBe('gemini');
        expect(profile.source_provider).toBe('google');
        expect(capabilities.input.image).toBe(true);
        expect(capabilities.tool_support).toBe(true);
        expect(options._option_id).toBe('openai-text');
        expect(options.options.map((option) => option.name)).not.toContain('flex');
        expect(options.options.map((option) => option.name)).not.toContain('thinking_level');
        expect(options.options.find((option) => option.name === 'max_tokens')).toMatchObject({ max: 65_535 });
        expect(options.options.find((option) => option.name === 'effort')).toMatchObject({
            enum: { minimal: 'minimal', low: 'low', medium: 'medium', high: 'high' },
        });
    });

    it('keeps the same canonical Gemini family on Vertex transport', () => {
        const profile = resolveModelProfile('gemini-3.5-flash', Providers.vertexai);
        const options = getOptions('gemini-3.5-flash', Providers.vertexai);

        expect(profile.family).toBe('gemini');
        expect(options._option_id).toBe('vertexai-gemini');
        expect(options.options.map((option) => option.name)).toContain('flex');
    });

    it('applies Bedrock Mantle overrides separately from direct OpenAI', () => {
        const direct = resolveModelProfile('gpt-5.5', Providers.openai);
        const mantle = resolveModelProfile('openai.gpt-5.5', Providers.bedrock_mantle);

        expect(direct.context_window).toBe(1_050_000);
        expect(mantle.context_window).toBe(272_000);
        expect(mantle.max_output_tokens).toBe(128_000);
    });

    it('carries version rules into future model releases', () => {
        expect(resolveModelProfile('gpt-6.1', Providers.openai).reasoning_effort_levels).toEqual([
            'none',
            'low',
            'medium',
            'high',
            'xhigh',
            'max',
        ]);
        expect(resolveModelProfile('gemini-4.0-flash', Providers.openai_compatible).family).toBe('gemini');
        expect(resolveModelProfile('anthropic/claude-opus-6', Providers.openai_compatible).family).toBe('claude');
    });

    it('allows trusted listing metadata to override inferred endpoint behavior', () => {
        const profile = resolveModelProfile('google/gemini-3.5-flash', Providers.openai_compatible, {
            input_modalities: ['text'],
            output_modalities: ['text'],
            tool_support: false,
            context_window: 128_000,
            max_output_tokens: 8_192,
        });

        expect(profile.capabilities.input.image).toBeUndefined();
        expect(profile.capabilities.tool_support).toBe(false);
        expect(profile.context_window).toBe(128_000);
        expect(profile.max_output_tokens).toBe(8_192);
    });

    it('classifies embeddings as non-inference models', () => {
        const profile = resolveModelProfile('google/text-embedding-005', Providers.openai_compatible);
        expect(profile.family).toBe('embedding');
        expect(profile.capabilities.output.embed).toBe(true);
        expect(profile.capabilities.tool_support).toBe(false);
    });
});
