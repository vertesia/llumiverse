import { describe, expect, it } from 'vitest';
import { getOptions } from '../options.js';
import { OptionType, SharedOptions } from '../types.js';

function effortValues(model: string, provider: string): string[] {
    const effort = getOptions(model, provider).options.find((option) => option.name === SharedOptions.effort);
    expect(effort?.type).toBe(OptionType.enum);
    return effort?.type === OptionType.enum ? Object.values(effort.enum) : [];
}

describe('current reasoning model options', () => {
    it('advertises forward-compatible Claude effort for all current families', () => {
        expect(effortValues('claude-fable-5', 'anthropic')).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('claude-mythos-5', 'anthropic')).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
        expect(effortValues('claude-sonnet-5', 'anthropic')).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
    });

    it('advertises version-specific GPT-5 effort without adding a default', () => {
        const options = getOptions('gpt-5.6-sol', 'openai');
        expect(effortValues('gpt-5.6-sol', 'openai')).toEqual(['none', 'low', 'medium', 'high', 'xhigh', 'max']);
        expect(options.options.find((option) => option.name === SharedOptions.effort)).not.toHaveProperty('default');
        expect(effortValues('gpt-6', 'openai')).toEqual(['none', 'low', 'medium', 'high', 'xhigh', 'max']);
    });

    it('advertises effort for models served through an OpenAI-compatible endpoint', () => {
        expect(effortValues('custom-reasoning-model', 'openai_compatible')).toEqual([
            'none',
            'minimal',
            'low',
            'medium',
            'high',
            'xhigh',
            'max',
        ]);
    });

    it('advertises provider-specific service tiers', () => {
        const openAiTier = getOptions('gpt-5.6-sol', 'openai').options.find((option) => option.name === 'service_tier');
        expect(openAiTier).toMatchObject({
            type: OptionType.enum,
            default: 'auto',
            enum: { Auto: 'auto', Default: 'default', Flex: 'flex', Priority: 'priority' },
        });

        const unsupportedFlexTier = getOptions('gpt-4.1', 'openai').options.find(
            (option) => option.name === 'service_tier',
        );
        expect(unsupportedFlexTier).toMatchObject({
            enum: { Auto: 'auto', Default: 'default', Priority: 'priority' },
        });
        expect((unsupportedFlexTier as { enum?: Record<string, string> }).enum?.Flex).toBeUndefined();

        const azureOpenAiTier = getOptions('gpt-5.6-sol', 'azure_openai').options.find(
            (option) => option.name === 'service_tier',
        );
        expect(azureOpenAiTier).toMatchObject({
            type: OptionType.enum,
            default: 'auto',
            enum: { Auto: 'auto', Default: 'default', Priority: 'priority' },
        });
        expect((azureOpenAiTier as { enum?: Record<string, string> }).enum?.Flex).toBeUndefined();

        const bedrockTier = getOptions('anthropic.claude-sonnet-4-6-v1:0', 'bedrock').options.find(
            (option) => option.name === 'service_tier',
        );
        expect(bedrockTier).toMatchObject({
            type: OptionType.enum,
            default: 'default',
            enum: { Default: 'default', Flex: 'flex', Priority: 'priority', Reserved: 'reserved' },
        });

        const vertexTier = getOptions('gemini-3.1-pro-preview', 'vertexai').options.find(
            (option) => option.name === 'service_tier',
        );
        expect(vertexTier).toMatchObject({
            type: OptionType.enum,
            default: 'default',
            enum: { Default: 'default', Flex: 'flex' },
        });

        expect(
            getOptions('gpt-5.6-sol', 'openai_compatible').options.some((option) => option.name === 'service_tier'),
        ).toBe(false);
    });

    it('advertises current Gemini thinking levels without adding a default', () => {
        const flash = getOptions('gemini-3.5-flash', 'vertexai');
        const pro = getOptions('gemini-3.1-pro-preview', 'vertexai');
        expect(effortValues('gemini-3.5-flash', 'vertexai')).toEqual(['minimal', 'low', 'medium', 'high']);
        expect(effortValues('gemini-3.1-pro-preview', 'vertexai')).toEqual(['low', 'medium', 'high']);
        expect(flash.options.find((option) => option.name === SharedOptions.effort)).not.toHaveProperty('default');
        expect(pro.options.find((option) => option.name === SharedOptions.effort)).not.toHaveProperty('default');
    });
});
