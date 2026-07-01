import { describe, expect, it } from 'vitest';
import { getModelCapabilities } from '../capability.js';
import { getMaxTokensLimitVertexAi, getVertexAiOptions } from './vertexai.js';

describe('Vertex AI MaaS metadata', () => {
    it('uses family capability prefixes for future open MaaS models', () => {
        const capabilities = getModelCapabilities(
            'locations/global/publishers/qwen/models/qwen4-new-instruct-maas',
            'vertexai',
        );

        expect(capabilities.input.text).toBe(true);
        expect(capabilities.input.image).toBe(false);
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(true);
    });

    it('keeps model-specific MaaS capability exceptions', () => {
        const capabilities = getModelCapabilities(
            'locations/global/publishers/deepseek-ai/models/deepseek-ocr-maas',
            'vertexai',
        );

        expect(capabilities.input.text).toBe(true);
        expect(capabilities.input.image).toBe(true);
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(false);
    });

    it('uses OpenAI-compatible options for open MaaS chat families', () => {
        const optionNames = getVertexAiOptions(
            'locations/global/publishers/zai-org/models/glm-6-future-maas',
        ).options.map((option) => option.name);

        expect(optionNames).toContain('max_tokens');
        expect(optionNames).toContain('temperature');
        expect(optionNames).toContain('top_p');
        expect(optionNames).not.toContain('top_k');
        expect(optionNames).not.toContain('presence_penalty');
        expect(optionNames).not.toContain('frequency_penalty');
    });

    it('uses model-specific MaaS output token limits where known', () => {
        expect(getMaxTokensLimitVertexAi('deepseek-ocr-maas')).toBe(8192);
        expect(getMaxTokensLimitVertexAi('qwen3-next-80b-a3b-thinking-maas')).toBe(262144);
    });
});
