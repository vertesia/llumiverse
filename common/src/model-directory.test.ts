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
        expect(options.options.map((option) => option.name)).not.toContain('service_tier');
        expect(options.options.find((option) => option.name === 'extra_body')).toMatchObject({
            type: 'json_object',
        });
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
        expect(options.options.find((option) => option.name === 'service_tier')).toMatchObject({
            enum: { Default: 'default', Flex: 'flex' },
        });
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

    it('keeps documented OpenAI variant limits instead of applying the base-model limit', () => {
        expect(resolveModelProfile('gpt-5.4-mini', Providers.openai)).toMatchObject({
            context_window: 400_000,
            max_output_tokens: 128_000,
        });
        expect(resolveModelProfile('gpt-5.4-nano', Providers.openai)).toMatchObject({
            context_window: 400_000,
            max_output_tokens: 128_000,
        });
        expect(resolveModelProfile('gpt-5-chat-latest', Providers.openai)).toMatchObject({
            family: 'gpt',
            context_window: 128_000,
            max_output_tokens: 16_384,
        });
        expect(resolveModelProfile('chat-latest', Providers.openai)).toMatchObject({
            family: 'gpt',
            source_provider: 'openai',
            context_window: 400_000,
            max_output_tokens: 128_000,
            capabilities: { input: { text: true, image: true }, output: { text: true }, tool_support: true },
        });
    });

    it('classifies embeddings as non-inference models', () => {
        const profile = resolveModelProfile('google/text-embedding-005', Providers.openai_compatible);
        expect(profile.family).toBe('embedding');
        expect(profile.capabilities.output.embed).toBe(true);
        expect(profile.capabilities.tool_support).toBe(false);
    });

    it('classifies listing aliases and provider-qualified IDs without losing family semantics', () => {
        expect(resolveModelProfile('~openai/gpt-latest', Providers.openai_compatible).capabilities).toMatchObject({
            input: { text: true, image: true },
            output: { text: true },
            tool_support: true,
        });
        expect(resolveModelProfile('~google/gemini-flash-latest', Providers.openai_compatible).family).toBe('gemini');
        expect(resolveModelProfile('google/gemma-4-E4B-it', Providers.openai_compatible).family).toBe('gemma');
        expect(
            resolveModelProfile(
                'arn:aws:bedrock:us-east-1:381492301031:inference-profile/global.anthropic.claude-sonnet-4-6',
                Providers.bedrock,
            ).family,
        ).toBe('claude');
        expect(resolveModelProfile('OPENAI/GPT-5.6-SOL', Providers.openai_compatible).canonical_id).toBe('gpt-5.6-sol');
        expect(resolveModelProfile('openai/gpt-oss-120b', Providers.openai_compatible).capabilities.input.image).toBe(
            false,
        );
        expect(resolveModelProfile('o5-preview', Providers.openai).family).toBe('gpt');
        expect(resolveModelProfile('openai/gpt-5.6-sol-batch', Providers.openai_compatible).family).toBe('gpt');
    });

    it('uses open-model family semantics for Together and OpenAI-compatible catalogs', () => {
        expect(resolveModelProfile('google/gemma-4-31B-it', Providers.togetherai).capabilities).toMatchObject({
            input: { text: true, image: true },
            output: { text: true },
            tool_support: true,
        });
        expect(resolveModelProfile('google/gemma-3-1B-it', Providers.togetherai).capabilities.input.image).toBe(false);
        expect(resolveModelProfile('llama-4-scout', Providers.openai_compatible).capabilities.input.image).toBe(true);
        expect(resolveModelProfile('qwen3-vl-32b', Providers.openai_compatible).capabilities.input.image).toBe(true);
    });

    it('keeps source identity separate from transport-specific limits', () => {
        expect(resolveModelProfile('openai.gpt-5.6-sol', Providers.openai_compatible).context_window).toBe(1_050_000);
        expect(resolveModelProfile('openai.gpt-5.6-sol', Providers.bedrock_mantle).context_window).toBe(272_000);
    });

    it('uses conservative capabilities and options for genuinely unknown models', () => {
        const profile = resolveModelProfile('future-provider/new-capability-v1', Providers.openai_compatible);
        expect(profile.capabilities).toMatchObject({ input: { text: true }, output: { text: true } });
        expect(profile.capabilities.tool_support).toBeUndefined();
        expect(profile.context_window).toBeUndefined();
        expect(profile.max_output_tokens).toBeUndefined();
        expect(
            getOptions(profile.model_id, Providers.openai_compatible).options.map((option) => option.name),
        ).not.toContain('effort');
        expect(
            getOptions(profile.model_id, Providers.openai_compatible).options.find(
                (option) => option.name === 'max_tokens',
            ),
        ).toMatchObject({ max: 4_096 });
        expect(resolveModelProfile('future-provider/new-capability-v1', Providers.vertexai).capabilities).toMatchObject(
            {
                input: { text: true },
                output: { text: true },
            },
        );
        expect(resolveModelProfile('vectorized-chat-v1', Providers.openai_compatible).family).toBe('generic');
    });

    it('inherits the newest known Llama family behavior across transports', () => {
        expect(resolveModelProfile('meta-llama/llama-5-scout', Providers.openai_compatible).capabilities).toMatchObject(
            {
                input: { text: true, image: true },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            },
        );
        expect(resolveModelProfile('meta.llama5-scout', Providers.bedrock).capabilities).toMatchObject({
            input: { text: true, image: true },
            output: { text: true },
            tool_support: true,
            tool_support_streaming: false,
        });
        expect(resolveModelProfile('meta.llama5-scout', Providers.bedrock).context_window).toBe(10_000_000);
        expect(resolveModelProfile('llama-5-scout-maas', Providers.vertexai).capabilities).toMatchObject({
            input: { text: true, image: true },
            output: { text: true },
            tool_support: true,
        });
        expect(resolveModelProfile('llama-5-scout', Providers.azure_foundry).capabilities.input.image).toBe(true);
        expect(resolveModelProfile('llama-5-scout', Providers.azure_foundry).context_window).toBe(10_000_000);
        const azureOptions = getOptions('deployment::LLAMA-5-SCOUT', Providers.azure_foundry);
        expect(azureOptions.options.find((option) => option.name === 'max_tokens')).toMatchObject({ max: 8_192 });
        expect(azureOptions.options.map((option) => option.name)).toContain('image_detail');
    });

    it('uses current Mistral family rules without claiming an output limit', () => {
        const small = resolveModelProfile('MISTRAL-SMALL-2603', Providers.mistralai);
        expect(small).toMatchObject({
            family: 'mistral',
            source_provider: 'mistralai',
            context_window: 256_000,
            capabilities: { input: { text: true, image: true }, output: { text: true }, tool_support: true },
            reasoning_effort_levels: ['none', 'high'],
        });
        expect(small.max_output_tokens).toBeUndefined();
        expect(resolveModelProfile('mistral-medium-3.6', Providers.mistralai).reasoning_effort_levels).toEqual([
            'none',
            'high',
        ]);
        expect(resolveModelProfile('voxtral-small-latest', Providers.mistralai).capabilities.input.audio).toBe(true);
        expect(resolveModelProfile('voxtral-mini-2507', Providers.mistralai).capabilities).toMatchObject({
            input: { text: true, audio: true },
            output: { text: true },
            tool_support: false,
        });
        expect(resolveModelProfile('voxtral-mini-transcribe-2602', Providers.mistralai).capabilities).toMatchObject({
            input: { audio: true },
            output: { text: true },
            tool_support: false,
        });
        expect(resolveModelProfile('mistral-tts-latest', Providers.mistralai).capabilities).toMatchObject({
            input: { text: true },
            output: { audio: true },
            tool_support: false,
        });
        const maxTokens = getOptions('mistral-small-2603', Providers.mistralai).options.find(
            (option) => option.name === 'max_tokens',
        );
        expect(maxTokens).not.toHaveProperty('max');
        expect(resolveModelProfile('labs-leanstral-1-5', Providers.mistralai)).toMatchObject({
            family: 'mistral',
            context_window: 256_000,
            capabilities: { tool_support: true },
        });
        const compatibleOptions = getOptions('mistralai/mistral-small-2603', Providers.openai_compatible);
        expect(compatibleOptions._option_id).toBe('openai-text');
        expect(compatibleOptions.options.map((option) => option.name)).not.toContain('safe_prompt');
    });

    it('reapplies exact source semantics after provider overlays', () => {
        for (const provider of [Providers.openai, Providers.azure_foundry, Providers.bedrock]) {
            expect(resolveModelProfile('gpt-image-1', provider).capabilities).toMatchObject({
                input: { text: true, image: true },
                output: { text: false, image: true },
                tool_support: false,
                tool_support_streaming: false,
            });
        }
        expect(resolveModelProfile('image-deployment::gpt-image-1', Providers.azure_foundry).family).toBe('image');
        expect(resolveModelProfile('speech-deployment::gpt-4o-mini-tts', Providers.azure_foundry).family).toBe(
            'speech',
        );
    });

    it('classifies current OpenAI audio models independently from text GPT models', () => {
        expect(resolveModelProfile('gpt-audio-1.5', Providers.openai)).toMatchObject({
            family: 'audio',
            context_window: 128_000,
            max_output_tokens: 16_384,
            capabilities: {
                input: { text: true, audio: true },
                output: { text: true, audio: true },
                tool_support: true,
            },
        });
        expect(resolveModelProfile('gpt-realtime-3', Providers.openai)).toMatchObject({
            family: 'realtime',
            context_window: 128_000,
            max_output_tokens: 32_000,
        });
        expect(resolveModelProfile('gpt-realtime-translate', Providers.openai)).toMatchObject({
            family: 'realtime',
            context_window: 16_000,
            max_output_tokens: 2_000,
            capabilities: {
                input: { text: false, image: false, audio: true },
                output: { text: true, audio: true },
                tool_support: false,
            },
        });
    });

    it('uses image-generation metadata for Gemini 3.1 Flash Image', () => {
        expect(resolveModelProfile('gemini-3.1-flash-image-preview', Providers.vertexai)).toMatchObject({
            context_window: 131_072,
            max_output_tokens: 32_768,
            capabilities: {
                input: { text: true, image: true, video: true, audio: false },
                output: { text: true, image: true },
                tool_support: false,
            },
        });
    });

    it('retains Nemotron family identity and newest family limits', () => {
        expect(resolveModelProfile('nvidia.nemotron-nano-12b-v2-vl-bf16', Providers.bedrock)).toMatchObject({
            family: 'nemotron',
            source_provider: 'nvidia',
            context_window: 128_000,
            max_output_tokens: 8_192,
            capabilities: { input: { text: true, image: true }, output: { text: true } },
        });
        expect(resolveModelProfile('nvidia.nemotron-super-4-180b', Providers.bedrock)).toMatchObject({
            family: 'nemotron',
            context_window: 256_000,
            max_output_tokens: 32_768,
        });
    });

    it('keeps moderation models executable without advertising tool use', () => {
        const profile = resolveModelProfile('meta-llama/llama-prompt-guard-2-86m', Providers.groq);
        expect(profile.family).toBe('moderation');
        expect(profile.capabilities).toMatchObject({
            input: { text: true },
            output: { text: true },
            tool_support: false,
            tool_support_streaming: false,
        });
    });

    it('exposes reasoning effort only when it is compatible with the transport', () => {
        expect(resolveModelProfile('openai/gpt-oss-120b', Providers.togetherai).reasoning_effort_levels).toEqual([
            'low',
            'medium',
            'high',
        ]);
        expect(resolveModelProfile('gpt-5.6-sol', Providers.togetherai).reasoning_effort_levels).toBeUndefined();
        expect(resolveModelProfile('mistral-small-latest', Providers.mistralai).reasoning_effort_levels).toEqual([
            'none',
            'high',
        ]);
        expect(resolveModelProfile('grok-4.3', Providers.xai).reasoning_effort_levels).toEqual([
            'none',
            'low',
            'medium',
            'high',
        ]);
        expect(resolveModelProfile('grok-4.5', Providers.xai).reasoning_effort_levels).toEqual([
            'low',
            'medium',
            'high',
        ]);
        expect(resolveModelProfile('grok-4.20-multi-agent', Providers.xai).reasoning_effort_levels).toEqual([
            'low',
            'medium',
            'high',
            'xhigh',
        ]);
    });
});
