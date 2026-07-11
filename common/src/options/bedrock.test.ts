import { describe, expect, it } from 'vitest';
import { getModelCapabilities } from '../capability.js';
import { getOptions } from '../options.js';
import { Providers } from '../types.js';
import { getBedrockOptions } from './bedrock.js';
import { getBedrockMantleProtocol } from './bedrock_mantle.js';
import { getContextWindowSize, getMaxOutputTokens } from './context-windows.js';
import { getClaudeMaxTokensLimit } from './shared-parsing.js';
import { hasSamplingParameterRestriction, parseClaudeVersion, supportsAdaptiveThinking } from './version-parsing.js';

const MANTLE_PROTOCOL_CASES = [
    ['openai.gpt-5.5', 'responses'],
    ['xai.grok-4.3', 'responses'],
    ['openai.gpt-oss-20b', 'chat_completions'],
    ['deepseek.v3.2', 'chat_completions'],
    ['google.gemma-3-4b-it', 'chat_completions'],
    ['google.gemma-4-31b', 'responses'],
    ['minimax.minimax-m2.5', 'chat_completions'],
    ['mistral.mistral-large-3-675b-instruct', 'chat_completions'],
    ['moonshotai.kimi-k2.5', 'chat_completions'],
    ['nvidia.nemotron-super-3-120b', 'chat_completions'],
    ['qwen.qwen3-coder-next', 'chat_completions'],
    ['writer.palmyra-vision-7b', 'chat_completions'],
    ['zai.glm-5', 'chat_completions'],
    ['anthropic.claude-opus-4-8', 'messages'],
] as const;

describe('Bedrock Mantle metadata', () => {
    it.each(MANTLE_PROTOCOL_CASES)('classifies %s as %s', (model, protocol) => {
        expect(getBedrockMantleProtocol(model)).toBe(protocol);
    });

    it('assumes future models follow the latest protocol for their publisher family', () => {
        expect(getBedrockMantleProtocol('openai.gpt-6.1')).toBe('responses');
        expect(getBedrockMantleProtocol('openai.gpt-oss-200b')).toBe('chat_completions');
        expect(getBedrockMantleProtocol('anthropic.claude-opus-6')).toBe('messages');
        expect(getBedrockMantleProtocol('qwen.qwen4-coder')).toBe('chat_completions');
        expect(getBedrockMantleProtocol('mistral.mistral-large-4')).toBe('chat_completions');
        expect(getBedrockMantleProtocol('google.gemma-5-70b')).toBe('responses');
        expect(getBedrockMantleProtocol('unverified.model-1')).toBeUndefined();
    });

    it('uses Responses options and capabilities for listed OpenAI models', () => {
        const options = getOptions('openai.gpt-5.5', Providers.bedrock_mantle);
        const capabilities = getModelCapabilities('openai.gpt-5.5', Providers.bedrock_mantle);

        expect(options._option_id).toBe('bedrock-mantle-responses');
        expect(options.options.map((option) => option.name)).toContain('verbosity');
        expect(capabilities.input).toMatchObject({ text: true, image: true });
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(true);
        expect(capabilities.tool_support_streaming).toBe(true);
    });

    it('uses Chat Completions for GPT-OSS on Bedrock Mantle', () => {
        const options = getOptions('openai.gpt-oss-120b', Providers.bedrock_mantle);
        const capabilities = getModelCapabilities('openai.gpt-oss-120b', Providers.bedrock_mantle);

        expect(options._option_id).toBe('bedrock-mantle-chat-completions');
        expect(options.options.map((option) => option.name)).not.toContain('verbosity');
        expect(capabilities.input).toMatchObject({ text: true, image: false });
        expect(capabilities.output.text).toBe(true);
    });

    it('uses Chat Completions options and model-card modalities', () => {
        const options = getOptions('google.gemma-3-4b-it', Providers.bedrock_mantle);
        const capabilities = getModelCapabilities('google.gemma-3-4b-it', Providers.bedrock_mantle);

        expect(options._option_id).toBe('bedrock-mantle-chat-completions');
        expect(options.options.map((option) => option.name)).toEqual([
            'max_tokens',
            'temperature',
            'top_p',
            'stop_sequence',
        ]);
        expect(capabilities.input).toMatchObject({ text: true, image: true });
        expect(capabilities.output.text).toBe(true);
    });

    it('uses Responses options for Gemma 4 and later generations', () => {
        for (const model of ['google.gemma-4-31b', 'google.gemma-5-70b']) {
            const options = getOptions(model, Providers.bedrock_mantle);
            const capabilities = getModelCapabilities(model, Providers.bedrock_mantle);

            expect(options._option_id).toBe('bedrock-mantle-responses');
            expect(capabilities.input).toMatchObject({ text: true, image: true });
            expect(capabilities.output.text).toBe(true);
        }
    });

    it('uses Claude options and 1M context metadata for Mantle Messages models', () => {
        const options = getOptions('anthropic.claude-fable-5', Providers.bedrock_mantle);
        const capabilities = getModelCapabilities('anthropic.claude-fable-5', Providers.bedrock_mantle);

        expect(options._option_id).toBe('bedrock-mantle-claude');
        expect(options.options.map((option) => option.name)).toEqual(
            expect.arrayContaining(['max_tokens', 'effort', 'include_thoughts', 'cache_enabled']),
        );
        expect(capabilities.input).toMatchObject({ text: true, image: true });
        expect(getContextWindowSize('anthropic.claude-fable-5')).toBe(1_000_000);
        expect(getContextWindowSize('anthropic.claude-mythos-preview')).toBe(1_000_000);
        expect(getContextWindowSize('anthropic.claude-opus-4-8')).toBe(1_000_000);
        expect(getClaudeMaxTokensLimit('anthropic.claude-fable-5')).toBe(128_000);
        expect(getClaudeMaxTokensLimit('anthropic.claude-opus-4-8')).toBe(128_000);
        expect(parseClaudeVersion('anthropic.claude-mythos-preview')).toEqual({
            major: 5,
            minor: 0,
            variant: 'mythos',
        });
        expect(supportsAdaptiveThinking('anthropic.claude-fable-5')).toBe(true);
        expect(hasSamplingParameterRestriction('anthropic.claude-mythos-preview')).toBe(true);
    });

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
        expect(options.options.find((option) => option.name === 'effort')).toMatchObject({
            enum: {
                low: 'low',
                medium: 'medium',
                high: 'high',
                xhigh: 'xhigh',
            },
        });
    });

    it('sets GPT-5.5 Bedrock Mantle capabilities and context window', () => {
        const capabilities = getModelCapabilities('openai.gpt-5.5', Providers.bedrock_mantle);

        expect(capabilities.input.text).toBe(true);
        expect(capabilities.input.image).toBe(true);
        expect(capabilities.output.text).toBe(true);
        expect(capabilities.tool_support).toBe(true);
        expect(capabilities.tool_support_streaming).toBe(true);
        expect(getContextWindowSize('openai.gpt-5.5')).toBe(272_000);
        expect(getContextWindowSize('openai.gpt-6.1')).toBe(272_000);
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
        expect(getContextWindowSize('xai.grok-5.0')).toBe(1_000_000);
    });

    it('inherits the latest DeepSeek limits for later versions and publisher-style IDs', () => {
        expect(getMaxOutputTokens('deepseek.v3.2')).toBe(65_536);
        expect(getMaxOutputTokens('deepseek.v3.3')).toBe(65_536);
        expect(getContextWindowSize('deepseek.v3.3')).toBe(163_840);
    });

    it('does not expose Mantle options or capabilities from the Bedrock provider', () => {
        const options = getBedrockOptions('openai.gpt-5.5');
        const capabilities = getModelCapabilities('openai.gpt-5.5', Providers.bedrock);

        expect(options._option_id).toBe('bedrock-converse');
        expect(capabilities.input.image).not.toBe(true);
        expect(capabilities.tool_support).not.toBe(true);
    });

    it('inherits Bedrock Runtime capabilities across newly discovered model families', () => {
        expect(getModelCapabilities('google.gemma-3-30b-it', Providers.bedrock).input.image).toBe(true);
        expect(getModelCapabilities('minimax.minimax-m3', Providers.bedrock).input.text).toBe(true);
        expect(getModelCapabilities('moonshotai.kimi-k3', Providers.bedrock).input.image).toBe(true);
        expect(getModelCapabilities('nvidia.nemotron-nano-12b-v3', Providers.bedrock).input.image).toBe(true);
        expect(getModelCapabilities('nvidia.nemotron-super-4-180b', Providers.bedrock).input.image).toBe(false);
        expect(getModelCapabilities('zai.glm-6', Providers.bedrock).output.text).toBe(true);
    });
});
