import { describe, expect, it } from 'vitest';
import { getModelCapabilities } from '../capability.js';
import { Providers } from '../types.js';
import { getBedrockOptions } from './bedrock.js';
import { getContextWindowSize } from './context-windows.js';

describe('Bedrock OpenAI Mantle metadata', () => {
    it('uses Responses options for GPT-5.5', () => {
        const options = getBedrockOptions('openai.gpt-5.5');
        const optionNames = options.options.map((option) => option.name);

        expect(options._option_id).toBe('bedrock-openai-responses');
        expect(optionNames).toContain('max_tokens');
        expect(optionNames).toContain('effort');
        expect(optionNames).toContain('reasoning_effort');
        expect(optionNames).toContain('verbosity');
        expect(optionNames).toContain('image_detail');
        expect(optionNames).not.toContain('stop_sequence');
    });

    it('sets GPT-5.5 Bedrock capabilities and context window', () => {
        const capabilities = getModelCapabilities('openai.gpt-5.5', Providers.bedrock);

        expect(capabilities.input.text).toBe(true);
        expect(capabilities.input.image).toBe(true);
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(true);
        expect(capabilities.tool_support_streaming).toBe(true);
        expect(getContextWindowSize('openai.gpt-5.5')).toBe(272_000);
    });
});
