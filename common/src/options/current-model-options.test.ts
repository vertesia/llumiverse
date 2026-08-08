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
        expect(
            getOptions('custom-reasoning-model', Providers.openai_compatible).options.map((option) => option.name),
        ).not.toContain(SharedOptions.effort);
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
