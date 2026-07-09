import { describe, expect, it } from 'vitest';
import { getModelCapabilities } from '../capability.js';
import { getOptions } from '../options.js';
import { Providers } from '../types.js';
import { getBedrockOptions } from './bedrock.js';
import { getContextWindowSize } from './context-windows.js';

describe('Bedrock Mantle metadata', () => {
    it('uses Responses options for GPT-5.5 under the Bedrock Mantle provider', () => {
        const options = getOptions('openai.gpt-5.5', Providers.bedrock_mantle);
        const optionNames = options.options.map((option) => option.name);

        expect(options._option_id).toBe('bedrock-mantle-responses');
        expect(optionNames).toContain('max_tokens');
        expect(optionNames).toContain('effort');
        expect(optionNames).toContain('reasoning_effort');
        expect(optionNames).toContain('verbosity');
        expect(optionNames).toContain('image_detail');
        expect(optionNames).not.toContain('stop_sequence');
    });

    it('sets GPT-5.5 Bedrock Mantle capabilities and context window', () => {
        const capabilities = getModelCapabilities('openai.gpt-5.5', Providers.bedrock_mantle);

        expect(capabilities.input.text).toBe(true);
        expect(capabilities.input.image).toBe(true);
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(true);
        expect(capabilities.tool_support_streaming).toBe(true);
        expect(getContextWindowSize('openai.gpt-5.5')).toBe(272_000);
    });

    it('uses Responses options for Grok 4.3 under the Bedrock Mantle provider', () => {
        const options = getOptions('xai.grok-4.3', Providers.bedrock_mantle);
        const optionNames = options.options.map((option) => option.name);
        const effort = options.options.find((option) => option.name === 'effort');

        expect(options._option_id).toBe('bedrock-mantle-responses');
        expect(optionNames).toContain('max_tokens');
        expect(optionNames).toContain('temperature');
        expect(optionNames).toContain('top_p');
        expect(optionNames).toContain('effort');
        expect(optionNames).toContain('reasoning_effort');
        expect(optionNames).toContain('image_detail');
        expect(optionNames).not.toContain('verbosity');
        expect(optionNames).not.toContain('stop_sequence');
        expect(effort).toMatchObject({
            enum: { none: 'none', low: 'low', medium: 'medium', high: 'high' },
            default: 'low',
        });
    });

    it('sets Grok 4.3 Bedrock Mantle capabilities and context window', () => {
        const capabilities = getModelCapabilities('xai.grok-4.3', Providers.bedrock_mantle);

        expect(capabilities.input.text).toBe(true);
        expect(capabilities.input.image).toBe(true);
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(true);
        expect(capabilities.tool_support_streaming).toBe(true);
        expect(getContextWindowSize('xai.grok-4.3')).toBe(1_000_000);
    });

    it('does not expose Mantle options or capabilities from the Bedrock provider', () => {
        const options = getBedrockOptions('openai.gpt-5.5');
        const capabilities = getModelCapabilities('openai.gpt-5.5', Providers.bedrock);

        expect(options._option_id).toBe('bedrock-converse');
        expect(capabilities.input.image).not.toBe(true);
        expect(capabilities.tool_support).not.toBe(true);
    });
});
