import { describe, expect, it } from 'vitest';
import { getModelCapabilities } from '../capability.js';
import { Providers } from '../types.js';
import { getMaxTokensLimitVertexAi, getVertexAiOptions, isFlexSupportedGeminiModel } from './vertexai.js';

describe('Vertex AI MaaS metadata', () => {
    it('exposes model-specific Gemini Omni tasks and resolutions', () => {
        const omni10 = getVertexAiOptions('gemini-omni-flash-preview');
        const omni11 = getVertexAiOptions('locations/global/publishers/google/models/gemini-omni-1.1-flash-preview');

        expect(omni10.options.find((option) => option.name === 'task')).toMatchObject({
            enum: {
                'Text to video': 'text_to_video',
                'Image to video': 'image_to_video',
                'References to video': 'reference_to_video',
                'Edit video': 'edit',
            },
        });
        expect(omni10.options.find((option) => option.name === 'resolution')).toMatchObject({
            enum: { '720p': '720p' },
        });
        expect(omni11.options.find((option) => option.name === 'task')).toMatchObject({
            enum: { 'Extend video': 'extend' },
        });
        expect(omni11.options.find((option) => option.name === 'resolution')).toMatchObject({
            enum: {
                '360p': '360p',
                '720p': '720p',
                '1080p': '1080p',
                '4K': '4k',
            },
        });
    });

    it.each(['gemini-3.5-flash', 'gemini-3.5-flash-lite', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-4.0-flash'])(
        'supports current Gemini Flash Flex inference for %s',
        (model) => {
            expect(isFlexSupportedGeminiModel(model)).toBe(true);
            const options = getVertexAiOptions(model).options;
            expect(options.map((option) => option.name)).toEqual(
                expect.arrayContaining(['effort', 'include_thoughts', 'max_tokens', 'service_tier']),
            );
            expect(options.find((option) => option.name === 'service_tier')).toMatchObject({
                default: 'default',
                enum: { Default: 'default', Flex: 'flex' },
            });
        },
    );

    it('uses family capability prefixes for future open MaaS models', () => {
        const capabilities = getModelCapabilities(
            'locations/global/publishers/qwen/models/qwen4-new-instruct-maas',
            Providers.vertexai,
        );

        expect(capabilities.input.text).toBe(true);
        expect(capabilities.input.image).toBe(false);
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(true);
    });

    it('exposes Gemma 4 MaaS tool support', () => {
        const gemma = getModelCapabilities(
            'locations/global/publishers/google/models/gemma-4-26b-a4b-it-maas',
            Providers.vertexai,
        );
        expect(gemma.input.text).toBe(true);
        expect(gemma.input.image).toBe(true);
        expect(gemma.output.text).toBe(true);
        expect(gemma.tool_support).toBe(true);
    });

    it('uses MaaS modality and tool-support metadata for key model families', () => {
        const llama4 = getModelCapabilities(
            'locations/us-east5/publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas',
            Providers.vertexai,
        );
        expect(llama4.input.image).toBe(true);
        expect(llama4.tool_support).toBe(true);

        const llama33 = getModelCapabilities(
            'locations/us-central1/publishers/meta/models/llama-3.3-70b-instruct-maas',
            Providers.vertexai,
        );
        expect(llama33.input.text).toBe(true);
        expect(llama33.input.image).toBe(false);
        expect(llama33.tool_support).toBe(true);

        expect(
            getModelCapabilities('locations/global/publishers/openai/models/gpt-oss-120b-maas', Providers.vertexai)
                .tool_support,
        ).toBe(true);
        expect(
            getModelCapabilities(
                'locations/global/publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas',
                Providers.vertexai,
            ).tool_support,
        ).toBe(true);
    });

    it('uses OpenAI-compatible options for open MaaS chat families', () => {
        const options = getVertexAiOptions('locations/global/publishers/zai-org/models/glm-6-future-maas');
        const optionNames = options.options.map((option) => option.name);

        expect(options._option_id).toBe('openai-text');
        expect(optionNames).toContain('max_tokens');
        expect(optionNames).toContain('temperature');
        expect(optionNames).toContain('top_p');
        expect(optionNames).not.toContain('top_k');
        expect(optionNames).not.toContain('presence_penalty');
        expect(optionNames).not.toContain('frequency_penalty');
    });

    it('inherits verified GPT-OSS reasoning options on future Vertex MaaS versions', () => {
        const options = getVertexAiOptions('locations/global/publishers/openai/models/gpt-oss-200b-maas');

        expect(options._option_id).toBe('openai-text');
        expect(options.options.find((option) => option.name === 'effort')).toMatchObject({
            enum: { low: 'low', medium: 'medium', high: 'high' },
        });
    });

    it('uses model-specific MaaS output token limits where known', () => {
        expect(getMaxTokensLimitVertexAi('qwen3-next-80b-a3b-thinking-maas')).toBe(262144);
    });

    it('uses Claude Sonnet 4.6 128K output limit on Vertex AI', () => {
        const options = getVertexAiOptions('claude-sonnet-4-6');

        expect(getMaxTokensLimitVertexAi('claude-sonnet-4-6')).toBe(128_000);
        expect(options.options.find((option) => option.name === 'max_tokens')).toMatchObject({ max: 128_000 });
    });
});

describe('Vertex Gemini reasoning metadata', () => {
    it.each([
        ['gemini-3.6-flash', ['minimal', 'low', 'medium', 'high']],
        ['gemini-3.7-flash', ['low', 'medium', 'high']],
        ['gemini-3.8-flash-cyber', ['low', 'medium', 'high']],
        ['gemini-4.0-flash', ['low', 'medium', 'high']],
        ['gemini-3-pro-preview', ['low', 'high']],
        ['gemini-3.1-pro-preview', ['low', 'medium', 'high']],
        ['gemini-3.1-flash-lite-image', ['minimal', 'high']],
        ['gemini-nano-banana-2.1', ['minimal', 'medium', 'high']],
        ['gemini-nano-banana-3.0', ['minimal', 'medium', 'high']],
    ])('offers supported effort levels for %s', (model, values) => {
        const effort = getVertexAiOptions(`publishers/google/models/${model}`).options.find(
            (item) => item.name === 'effort',
        );
        expect(effort?.type).toBe('enum');
        if (effort?.type !== 'enum') throw new Error('Missing effort option');
        expect(Object.values(effort.enum)).toEqual(values);
    });

    it.each(['gemini-3.7-flash', 'gemini-3.8-flash', 'gemini-4.0-flash'])(
        'hides unsupported sampling options for %s',
        (model) => {
            const names = getVertexAiOptions(model).options.map((item) => item.name);
            for (const name of ['temperature', 'top_p', 'top_k', 'presence_penalty', 'frequency_penalty']) {
                expect(names).not.toContain(name);
            }
        },
    );
});

describe('Gemini 2.5 thinking defaults', () => {
    it.each([
        ['gemini-2.5-flash-lite', false],
        ['gemini-2.5-flash', true],
        ['gemini-2.5-pro', true],
    ])('uses a compatible summary default for %s', (model, includeThoughts) => {
        const options = getVertexAiOptions(model).options;
        expect(options.find((option) => option.name === 'include_thoughts')?.default).toBe(includeThoughts);
        expect(options.find((option) => option.name === 'thinking_budget_tokens')?.default).toBeUndefined();
        expect(options.find((option) => option.name === 'effort')?.default).toBeUndefined();
    });
});

describe('Vertex regional Mistral chat metadata', () => {
    it.each([
        'mistral-small-2503',
        'mistral-medium-3',
        'codestral-2',
        'mistral-small-2603',
        'mistral-medium-4',
        'codestral-3',
    ])('uses compatible chat options without reasoning defaults for %s', (model) => {
        const options = getVertexAiOptions(`locations/europe-west4/publishers/mistralai/models/${model}`);
        expect(options._option_id).toBe('openai-text');
        expect(options.options.map((option) => option.name)).toEqual(
            expect.arrayContaining(['temperature', 'top_p', 'max_tokens', 'stop_sequence', 'extra_body']),
        );
        expect(options.options.map((option) => option.name)).not.toContain('top_k');
        expect(options.options.map((option) => option.name)).not.toContain('presence_penalty');
        expect(options.options.map((option) => option.name)).not.toContain('frequency_penalty');
        expect(options.options.map((option) => option.name)).not.toContain('effort');
    });
    it.each(['mistral-small-2503', 'mistral-medium-3', 'codestral-2'])(
        'preserves documented capabilities for %s',
        (model) => {
            const capabilities = getModelCapabilities(model, Providers.vertexai);
            expect(capabilities.input).toMatchObject({ text: true, image: model !== 'codestral-2' });
            expect(capabilities.tool_support).toBe(model !== 'codestral-2');
        },
    );
    it('excludes OCR from the compatible chat option surface', () => {
        expect(getVertexAiOptions('publishers/mistralai/models/mistral-ocr-2505')._option_id).toBe('text-fallback');
    });
});
