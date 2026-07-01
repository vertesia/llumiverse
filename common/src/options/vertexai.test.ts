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
        const deepSeekOcr = getModelCapabilities(
            'locations/global/publishers/deepseek-ai/models/deepseek-ocr-maas',
            'vertexai',
        );

        expect(deepSeekOcr.input.text).toBe(true);
        expect(deepSeekOcr.input.image).toBe(true);
        expect(deepSeekOcr.output.text).toBe(true);
        expect(deepSeekOcr.tool_support).toBe(false);

        const gemma = getModelCapabilities(
            'locations/global/publishers/google/models/gemma-4-26b-a4b-it-maas',
            'vertexai',
        );
        expect(gemma.input.text).toBe(true);
        expect(gemma.input.image).toBe(false);
        expect(gemma.output.text).toBe(true);
        expect(gemma.tool_support).toBe(false);
    });

    it('uses MaaS modality and tool-support metadata for key model families', () => {
        const llama4 = getModelCapabilities(
            'locations/us-east5/publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas',
            'vertexai',
        );
        expect(llama4.input.image).toBe(true);
        expect(llama4.tool_support).toBe(true);

        const llama33 = getModelCapabilities(
            'locations/us-central1/publishers/meta/models/llama-3.3-70b-instruct-maas',
            'vertexai',
        );
        expect(llama33.input.text).toBe(true);
        expect(llama33.input.image).toBe(false);
        expect(llama33.tool_support).toBe(true);

        expect(
            getModelCapabilities('locations/global/publishers/openai/models/gpt-oss-120b-maas', 'vertexai')
                .tool_support,
        ).toBe(true);
        expect(
            getModelCapabilities('locations/global/publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas', 'vertexai')
                .tool_support,
        ).toBe(true);
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
