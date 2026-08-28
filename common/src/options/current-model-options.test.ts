import { describe, expect, it } from 'vitest';
import { getOptions } from '../options.js';
import { OptionType, Providers, SharedOptions } from '../types.js';

function effortValues(model: string, provider: Providers): string[] {
    const effort = getOptions(model, provider).options.find((option) => option.name === SharedOptions.effort);
    expect(effort?.type).toBe(OptionType.enum);
    return effort?.type === OptionType.enum ? Object.values(effort.enum) : [];
}

describe('current reasoning model options', () => {
    it.each([Providers.azure_openai, Providers.mistralai, Providers.togetherai, Providers.xai])(
        'provides inference options for %s',
        (provider) => {
            expect(getOptions('gpt-5.6-sol', provider).options.length).toBeGreaterThan(0);
        },
    );

    it('advertises forward-compatible Claude effort for all current families', () => {
        expect(effortValues('claude-fable-5', Providers.anthropic)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('claude-mythos-5', Providers.anthropic)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('claude-sonnet-5', Providers.anthropic)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
    });

    it('advertises version-specific GPT-5 effort without adding a default', () => {
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
        expect(effortValues('gpt-6', Providers.openai)).toEqual(['none', 'low', 'medium', 'high', 'xhigh', 'max']);
    });

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
        expect((azureOpenAiTier as { enum?: Record<string, string> }).enum?.Flex).toBeUndefined();

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
