import { describe, expect, it } from 'vitest';
import { resolveModelProfile } from '../model-directory.js';
import { getOptions } from '../options.js';
import { OptionType, Providers, SharedOptions } from '../types.js';

function speedOption(model: string, provider: Providers) {
    return getOptions(model, provider).options.find((option) => option.name === 'speed');
}

function effortValues(model: string, provider: Providers): string[] {
    const effort = getOptions(model, provider).options.find((option) => option.name === SharedOptions.effort);
    expect(effort?.type).toBe(OptionType.enum);
    return effort?.type === OptionType.enum ? Object.values(effort.enum) : [];
}

describe('current reasoning model options', () => {
    it.each(['openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'qwen/qwen3-32b', 'qwen/qwen3.8-27b'])(
        'retains the thought-visibility control alongside Groq effort for %s',
        (model) => {
            const options = getOptions(model, Providers.groq).options;
            expect(options.find((item) => item.name === 'include_thoughts')).toMatchObject({
                type: OptionType.boolean,
                default: true,
            });
            expect(options.filter((item) => item.name === 'include_thoughts')).toHaveLength(1);
            expect(options.find((item) => item.name === 'effort')).toBeDefined();
        },
    );
    it.each(['grok-4.6', 'grok-4.7'])('exposes xAI xhigh effort for %s', (model) => {
        expect(effortValues(model, Providers.xai)).toEqual(['low', 'medium', 'high', 'xhigh']);
    });

    it('uses xAI Grok 4.7 limits without a guessed output cap', () => {
        expect(resolveModelProfile('grok-4.7', Providers.xai)).toMatchObject({ context_window: 500_000 });
        const options = getOptions('grok-4.7', Providers.xai).options;
        expect(options.find((option) => option.name === 'max_tokens')).not.toHaveProperty('max');
        for (const name of ['stop_sequence', 'presence_penalty', 'frequency_penalty'])
            expect(options.map((option) => option.name)).not.toContain(name);
    });

    it('keeps Chat-only controls out of direct Responses metadata', () => {
        for (const provider of [Providers.openai, Providers.azure_openai]) {
            const names = getOptions('gpt-4.1', provider).options.map((option) => option.name);
            for (const name of ['stop_sequence', 'presence_penalty', 'frequency_penalty'])
                expect(names).not.toContain(name);
        }
        expect(getOptions('gpt-4.1', Providers.openai_compatible).options.map((option) => option.name)).toEqual(
            expect.arrayContaining(['stop_sequence', 'presence_penalty', 'frequency_penalty']),
        );
    });

    it.each(['gpt-6.1-sol', 'deployment::gpt-6.1-sol'])(
        'keeps direct OpenAI controls out of Azure Foundry metadata for %s',
        (model) => {
            const options = getOptions(model, Providers.azure_foundry).options;
            expect(options.find((option) => option.name === 'service_tier')).toMatchObject({
                enum: { Auto: 'auto', Default: 'default', Priority: 'priority' },
                default: 'auto',
            });
            const tier = options.find((option) => option.name === 'service_tier');
            expect(tier?.type === OptionType.enum && tier.enum).toEqual({
                Auto: 'auto',
                Default: 'default',
                Priority: 'priority',
            });
            // Azure documents reasoning.context, unlike the Fast and Ultrafast tiers.
            // https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning
            expect(options.map((option) => option.name)).toContain('reasoning_context');
        },
    );

    it.each([Providers.azure_openai, Providers.azure_foundry])(
        'offers Azure Flex only for documented model versions through %s',
        (provider) => {
            for (const model of [
                'gpt-5.6-sol',
                'gpt-5.6-luna',
                'gpt-5.6-terra',
                'gpt-6-astra',
                'gpt-6-astra-2026-09-03',
                'deployment::gpt-6-astra',
            ]) {
                const id =
                    provider === Providers.azure_foundry && !model.includes('::') ? `deployment::${model}` : model;
                expect(getOptions(id, provider).options.find((option) => option.name === 'service_tier')).toMatchObject(
                    { enum: { Flex: 'flex' }, default: 'auto' },
                );
            }
            const tier = getOptions('gpt-6.1-sol', provider).options.find((option) => option.name === 'service_tier');
            expect(tier?.type === OptionType.enum && tier.enum).toEqual({
                Auto: 'auto',
                Default: 'default',
                Priority: 'priority',
            });
        },
    );

    it.each(['openai/gpt-oss-20b', 'openai/gpt-oss-120b'])(
        'exposes Groq effort for %s without injecting a default',
        (model) => {
            expect(effortValues(model, Providers.groq)).toEqual(['low', 'medium', 'high']);
            expect(resolveModelProfile(model, Providers.groq)).toMatchObject({
                context_window: 131_072,
                max_output_tokens: 65_536,
            });
            expect(
                getOptions(model, Providers.groq).options.find((option) => option.name === 'max_tokens'),
            ).toMatchObject({ max: 65_536 });
            expect(
                getOptions(model, Providers.groq).options.find((option) => option.name === 'effort'),
            ).not.toHaveProperty('default');
        },
    );

    it('exposes current Groq Qwen effort and Together DeepSeek effort', () => {
        expect(resolveModelProfile('qwen/qwen3.8-27b', Providers.groq)).toMatchObject({
            context_window: 131_072,
            max_output_tokens: 16_384,
        });
        expect(effortValues('qwen/qwen3.8-27b', Providers.groq)).toEqual(['none', 'low', 'medium', 'high']);
        expect(effortValues('deepseek-ai/DeepSeek-V4-Pro-0813', Providers.togetherai)).toEqual(['high', 'max']);
    });

    it.each(['gpt-6-astra', 'gpt-6.1-sol'])('advertises current OpenAI processing tiers for %s', (model) => {
        expect(
            getOptions(model, Providers.openai).options.find((option) => option.name === 'service_tier'),
        ).toMatchObject({
            default: 'auto',
            enum: { Fast: 'fast', Ultrafast: 'ultrafast' },
        });
    });

    it.each([
        [Providers.anthropic, 'claude-haiku-5-5'],
        [Providers.bedrock, 'anthropic.claude-haiku-5-5'],
        [Providers.bedrock_mantle, 'anthropic.claude-haiku-5-5'],
        [Providers.vertexai, 'publishers/anthropic/models/claude-haiku-5-5'],
        [Providers.azure_foundry, 'deployment::claude-haiku-5-5'],
    ])('exposes Haiku 5.5 limits and supported controls through %s', (provider, model) => {
        const options = getOptions(model, provider).options;
        expect(resolveModelProfile(model, provider).context_window).toBe(1_000_000);
        expect(options.find((option) => option.name === SharedOptions.max_tokens)).toMatchObject({
            max: provider === Providers.bedrock ? 127_999 : 128_000,
        });
        expect(options.find((option) => option.name === SharedOptions.effort)).toMatchObject({
            enum: { Low: 'low', 'Medium (default)': 'medium', High: 'high', 'Extra High': 'xhigh', Max: 'max' },
        });
        for (const name of ['temperature', 'top_p', 'top_k', 'thinking_budget_tokens', 'thinking_mode']) {
            expect(options.map((option) => option.name)).not.toContain(name);
        }
    });

    it.each([Providers.azure_openai, Providers.mistralai, Providers.togetherai, Providers.xai])(
        'provides inference options for %s',
        (provider) => {
            expect(getOptions('gpt-5.6-sol', provider).options.length).toBeGreaterThan(0);
        },
    );

    it.each([
        [Providers.anthropic, 'claude-sonnet-5-5'],
        [Providers.bedrock, 'us.anthropic.claude-sonnet-5-5'],
        [Providers.bedrock_mantle, 'anthropic.claude-sonnet-5-5'],
        [Providers.vertexai, 'claude-sonnet-5-5'],
        [Providers.anthropic, 'claude-sonnet-5-6'],
        [Providers.anthropic, 'claude-sonnet-6'],
    ])('offers Sonnet thinking mode through %s for %s', (provider, model) => {
        const options = getOptions(model, provider).options;
        expect(options.find((option) => option.name === 'thinking_mode')).toMatchObject({
            type: OptionType.enum,
            enum: { 'Adaptive (default)': 'adaptive', 'Between tools': 'between_tools' },
        });
        expect(effortValues(model, provider)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(options.map((option) => option.name)).not.toContain('thinking_budget_tokens');
    });

    it.each(['claude-sonnet-5', 'claude-sonnet-5-4', 'claude-sonnet-4-6', 'claude-opus-5-5', 'claude-haiku-5-5'])(
        'does not advertise between-tools thinking for %s',
        (model) => {
            expect(getOptions(model, Providers.anthropic).options.map((option) => option.name)).not.toContain(
                'thinking_mode',
            );
        },
    );

    it('advertises forward-compatible Claude effort for all current families', () => {
        expect(effortValues('claude-fable-5', Providers.anthropic)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('claude-mythos-5', Providers.anthropic)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('claude-sonnet-5', Providers.anthropic)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
    });

    it('advertises GPT effort levels and labels documented defaults', () => {
        const options = getOptions('gpt-5.6-sol', Providers.openai);
        expect(effortValues('gpt-5.6-sol', Providers.openai)).toEqual([
            'none',
            'low',
            'medium',
            'high',
            'xhigh',
            'max',
        ]);
        expect(options.options.find((option) => option.name === SharedOptions.effort)).not.toHaveProperty('default');
        expect(effortValues('gpt-5.5', Providers.openai)).toEqual(['none', 'low', 'medium', 'high', 'xhigh']);
        expect(
            getOptions('gpt-5.5', Providers.openai).options.find((option) => option.name === SharedOptions.effort),
        ).toMatchObject({ enum: { 'Medium (default)': 'medium' } });
        expect(
            getOptions('gpt-5.6-sol', Providers.openai).options.find((option) => option.name === SharedOptions.effort),
        ).toMatchObject({ enum: { 'Medium (default)': 'medium' } });
        expect(effortValues('gpt-6', Providers.openai)).toEqual(['none', 'low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('gpt-6-astra', Providers.openai)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('gpt-6-sol', Providers.openai)).toEqual(['none', 'low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('gpt-6-luna', Providers.openai)).toEqual(['none', 'low', 'medium', 'high', 'xhigh', 'max']);
    });

    it('advertises reasoning context only for supported OpenAI Responses models', () => {
        const reasoningContext = (model: string, provider: Providers) =>
            getOptions(model, provider).options.find((option) => option.name === SharedOptions.reasoning_context);

        expect(reasoningContext('gpt-5.6-sol', Providers.openai)).toMatchObject({
            type: OptionType.enum,
            enum: { Auto: 'auto', 'Current turn': 'current_turn', 'All turns': 'all_turns' },
        });
        expect(reasoningContext('gpt-6-astra', Providers.openai)).toBeDefined();
        expect(reasoningContext('gpt-5.5', Providers.openai)).toBeUndefined();
        expect(reasoningContext('gpt-5.6-sol', Providers.azure_openai)).toBeDefined();
        expect(reasoningContext('deployment::gpt-6-luna', Providers.azure_foundry)).toBeDefined();
        expect(reasoningContext('gpt-5.5', Providers.azure_openai)).toBeUndefined();
        expect(reasoningContext('gpt-5.6-sol', Providers.openai_compatible)).toBeUndefined();
    });

    it('advertises the prompt-cache lifetimes each OpenAI model accepts', () => {
        const cacheTtls = (model: string, provider: Providers = Providers.openai) => {
            const option = getOptions(model, provider).options.find((item) => item.name === 'cache_ttl');
            return option?.type === OptionType.enum ? Object.values(option.enum) : undefined;
        };

        for (const model of ['gpt-5.6-sol', 'gpt-5.6-luna', 'gpt-6-astra', 'gpt-6.1-terra']) {
            expect(cacheTtls(model)).toEqual(['30m', '12h']);
        }
        for (const model of ['gpt-5.4', 'gpt-5.1-codex-mini', 'gpt-5', 'gpt-4.1', 'gpt-4.1-2025-04-14']) {
            expect(cacheTtls(model)).toEqual(['in_memory', '24h']);
        }
        // GPT-5.5 accepts 24h only, and models outside OpenAI's extended-retention list have no choice.
        for (const model of ['gpt-5.5', 'gpt-5.5-pro', 'gpt-5-mini', 'gpt-4.1-mini', 'gpt-4o']) {
            expect(cacheTtls(model)).toBeUndefined();
        }
        // Other OpenAI-compatible providers document neither lifetime.
        expect(cacheTtls('gpt-6-astra', Providers.openai_compatible)).toBeUndefined();
    });

    it.each([Providers.azure_openai, Providers.azure_foundry])(
        'advertises the prompt-cache lifetimes Azure documents through %s',
        (provider) => {
            const cacheTtls = (model: string) => {
                const id = provider === Providers.azure_foundry ? `deployment::${model}` : model;
                const option = getOptions(id, provider).options.find((item) => item.name === 'cache_ttl');
                return option?.type === OptionType.enum ? Object.values(option.enum) : undefined;
            };
            for (const model of ['gpt-5.6-sol', 'gpt-6-astra', 'gpt-6.1-terra', 'gpt-7']) {
                expect(cacheTtls(model)).toEqual(['30m', '12h']);
            }
            // Azure's extended-retention list includes gpt-5.3-codex and gpt-5.1-chat, unlike OpenAI's.
            for (const model of ['gpt-5.4', 'gpt-5.3-codex', 'gpt-5.1-chat', 'gpt-4.1']) {
                expect(cacheTtls(model)).toEqual(['in_memory', '24h']);
            }
            for (const model of ['gpt-5.5', 'gpt-5-mini', 'gpt-4o', 'gpt-5.1-chat-latest']) {
                expect(cacheTtls(model)).toBeUndefined();
            }
        },
    );

    it('does not advertise unverified effort for unknown OpenAI-compatible models', () => {
        const options = getOptions('custom-reasoning-model', Providers.openai_compatible).options;
        expect(options.map((option) => option.name)).not.toContain(SharedOptions.effort);
        expect(options.find((option) => option.name === 'extra_body')).toMatchObject({
            type: OptionType.json_object,
        });
    });

    it('advertises the native Mistral options that its transport serializes', () => {
        const options = getOptions('mistral-small-latest', Providers.mistralai);
        const optionNames = options.options.map((option) => option.name);
        expect(options._option_id).toBe('mistral-text');
        expect(optionNames).toEqual(
            expect.arrayContaining([
                'max_tokens',
                'temperature',
                'top_p',
                'presence_penalty',
                'frequency_penalty',
                'stop_sequence',
                'effort',
                'random_seed',
                'safe_prompt',
                'parallel_tool_calls',
                'tool_choice',
                'prompt_mode',
                'include_thoughts',
            ]),
        );
        expect(optionNames).not.toContain('image_detail');
        expect(optionNames).not.toContain('extra_body');
        expect(effortValues('mistral-small-latest', Providers.mistralai)).toEqual(['none', 'high']);
    });

    it('advertises provider-specific service tiers', () => {
        const openAiTier = getOptions('gpt-5.6-sol', Providers.openai).options.find(
            (option) => option.name === 'service_tier',
        );
        expect(openAiTier).toMatchObject({
            type: OptionType.enum,
            default: 'auto',
            enum: { Auto: 'auto', Default: 'default', Flex: 'flex', Priority: 'priority' },
        });

        const unsupportedFlexTier = getOptions('gpt-4.1', Providers.openai).options.find(
            (option) => option.name === 'service_tier',
        );
        expect(unsupportedFlexTier).toMatchObject({
            enum: { Auto: 'auto', Default: 'default', Priority: 'priority' },
        });
        expect((unsupportedFlexTier as { enum?: Record<string, string> }).enum?.Flex).toBeUndefined();

        const azureOpenAiTier = getOptions('gpt-5.6-sol', Providers.azure_openai).options.find(
            (option) => option.name === 'service_tier',
        );
        expect(azureOpenAiTier).toMatchObject({
            type: OptionType.enum,
            default: 'auto',
            enum: { Auto: 'auto', Default: 'default', Priority: 'priority' },
        });
        expect((azureOpenAiTier as { enum?: Record<string, string> }).enum?.Flex).toBe('flex');

        const bedrockTier = getOptions('anthropic.claude-sonnet-4-6-v1:0', Providers.bedrock).options.find(
            (option) => option.name === 'service_tier',
        );
        expect(bedrockTier).toMatchObject({
            type: OptionType.enum,
            default: 'default',
            enum: { Default: 'default', Flex: 'flex', Priority: 'priority', Reserved: 'reserved' },
        });

        const vertexTier = getOptions('gemini-3.1-pro-preview', Providers.vertexai).options.find(
            (option) => option.name === 'service_tier',
        );
        expect(vertexTier).toMatchObject({
            type: OptionType.enum,
            default: 'default',
            enum: { Default: 'default', Flex: 'flex' },
        });

        expect(
            getOptions('gpt-5.6-sol', Providers.openai_compatible).options.some(
                (option) => option.name === 'service_tier',
            ),
        ).toBe(false);

        const xaiTier = getOptions('grok-4.5', Providers.xai).options.find((option) => option.name === 'service_tier');
        expect(xaiTier).toMatchObject({
            type: OptionType.enum,
            default: 'default',
            enum: { Default: 'default', Priority: 'priority' },
        });
    });

    it('offers fast mode on the Claude API only, for the models that support it', () => {
        for (const model of [
            'claude-opus-4-8',
            'claude-opus-5',
            'claude-opus-5-5',
            'claude-fable-5-2',
            'claude-mythos-5-2',
        ]) {
            expect(speedOption(model, Providers.anthropic)).toMatchObject({
                type: OptionType.enum,
                default: 'standard',
                enum: { Standard: 'standard', Fast: 'fast' },
            });
        }
        expect(speedOption('claude-opus-4-7', Providers.anthropic)).toBeUndefined();
        expect(speedOption('claude-sonnet-5', Providers.anthropic)).toBeUndefined();
        expect(speedOption('claude-fable-5-1', Providers.anthropic)).toBeUndefined();
        // Vertex AI does not serve fast mode for Opus.
        expect(speedOption('claude-opus-5-5', Providers.vertexai)).toBeUndefined();
        // Microsoft Foundry (a separate `-fast` model) and Amazon Bedrock don't take the `speed` field.
        for (const model of ['claude-opus-5-5', 'claude-fable-5-2']) {
            expect(speedOption(`deployment::${model}`, Providers.azure_foundry)).toBeUndefined();
            expect(speedOption(`deployment::${model}-fast`, Providers.azure_foundry)).toBeUndefined();
            expect(speedOption(`anthropic.${model}`, Providers.bedrock_mantle)).toBeUndefined();
            expect(speedOption(`global.anthropic.${model}`, Providers.bedrock)).toBeUndefined();
        }
    });

    it('describes the fast mode research preview only on the models still in it', () => {
        const description = (model: string) => speedOption(model, Providers.anthropic)?.description;
        expect(description('claude-opus-5')).toContain('research preview');
        expect(description('claude-opus-5-5')).not.toContain('research preview');
        expect(description('claude-fable-5-2')).not.toContain('research preview');
    });

    it.each([
        [Providers.anthropic, 'claude-fable-5-2'],
        [Providers.anthropic, 'claude-mythos-5-2'],
        [Providers.vertexai, 'locations/global/publishers/anthropic/models/claude-fable-5-2'],
        [Providers.bedrock, 'global.anthropic.claude-fable-5-2'],
        [Providers.bedrock_mantle, 'anthropic.claude-fable-5-2'],
        [Providers.azure_foundry, 'deployment::claude-fable-5-2'],
    ])('exposes Fable and Mythos 5.2 limits and supported controls through %s for %s', (provider, model) => {
        const options = getOptions(model, provider).options;
        const names = options.map((option) => option.name);
        expect(resolveModelProfile(model, provider).context_window).toBe(1_000_000);
        // Amazon Bedrock Runtime enforces an exclusive 128K bound.
        expect(options.find((option) => option.name === SharedOptions.max_tokens)).toMatchObject({
            max: provider === Providers.bedrock ? 127_999 : 128_000,
        });
        expect(options.find((option) => option.name === SharedOptions.effort)).toMatchObject({
            enum: { Low: 'low', Medium: 'medium', 'High (default)': 'high', 'Extra High': 'xhigh', Max: 'max' },
        });
        expect(effortValues(model, provider)).toHaveLength(5);
        expect(names).toContain('include_thoughts');
        for (const removed of ['temperature', 'top_p', 'top_k', 'thinking_budget_tokens', 'thinking_mode']) {
            expect(names).not.toContain(removed);
        }
    });

    it('advertises current Gemini thinking levels without adding a default', () => {
        const flash = getOptions('gemini-3.5-flash', Providers.vertexai);
        const pro = getOptions('gemini-3.1-pro-preview', Providers.vertexai);
        expect(effortValues('gemini-3.5-flash', Providers.vertexai)).toEqual(['minimal', 'low', 'medium', 'high']);
        expect(effortValues('gemini-3.1-pro-preview', Providers.vertexai)).toEqual(['low', 'medium', 'high']);
        expect(flash.options.find((option) => option.name === SharedOptions.effort)).not.toHaveProperty('default');
        expect(pro.options.find((option) => option.name === SharedOptions.effort)).not.toHaveProperty('default');
    });
});
