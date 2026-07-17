import { describe, expect, it } from 'vitest';
import {
    getAvailableEffortLevels,
    getGeminiModelVersion,
    getOpenAIReasoningEffortLevels,
    isClaudeVersionGTE,
    isGeminiModelVersionGte,
    isOpenAIGptVersionGTE,
    parseClaudeVersion,
    parseOpenAIGptVersion,
    supportsAdaptiveThinking,
    supportsEffort,
} from './version-parsing.js';

describe('Claude model version parsing', () => {
    it.each([
        ['claude-sonnet-5', { major: 5, minor: 0, variant: 'sonnet' }],
        ['claude-fable-5-20260701', { major: 5, minor: 0, variant: 'fable' }],
        ['publishers/anthropic/models/claude-mythos-5@20260701', { major: 5, minor: 0, variant: 'mythos' }],
        ['claude-mythos-preview', { major: 5, minor: 0, variant: 'mythos' }],
        ['us.anthropic.claude-opus-4-8-v1:0', { major: 4, minor: 8, variant: 'opus' }],
        ['claude-3-7-sonnet-20250219', { major: 3, minor: 7, variant: 'sonnet' }],
    ] as const)('parses %s', (model, expected) => {
        expect(parseClaudeVersion(model)).toEqual(expected);
    });

    it('uses numeric GTE comparisons for current and future families', () => {
        expect(isClaudeVersionGTE('claude-sonnet-5', 4, 7)).toBe(true);
        expect(isClaudeVersionGTE('claude-opus-4-10', 4, 8)).toBe(true);
        expect(isClaudeVersionGTE('claude-sonnet-4-6', 4, 7)).toBe(false);
    });

    it('does not infer behavior for an unknown Claude family', () => {
        expect(parseClaudeVersion('claude-unknown-5')).toBeNull();
    });

    it.each([
        'claude-fable-5',
        'claude-mythos-5',
        'claude-sonnet-5',
        'claude-opus-4-8',
    ])('advertises adaptive thinking and current effort levels for %s', (model) => {
        expect(supportsAdaptiveThinking(model)).toBe(true);
        expect(supportsEffort(model)).toBe(true);
        expect(getAvailableEffortLevels(model)).toEqual({
            Low: 'low',
            Medium: 'medium',
            'High (default)': 'high',
            'Extra High': 'xhigh',
            Max: 'max',
        });
    });
});

describe('OpenAI GPT model version parsing', () => {
    it.each([
        ['gpt-5', { major: 5, minor: 0 }],
        ['gpt-5.6-sol', { major: 5, minor: 6 }],
        ['models/gpt-5.4-2026-03-05', { major: 5, minor: 4 }],
        ['gpt-5.7-pro-20270101', { major: 5, minor: 7 }],
        ['us.openai.gpt-5.6-sol-v1:0', { major: 5, minor: 6 }],
        ['gpt-deployment::gpt-5', { major: 5, minor: 0 }],
        ['gpt-6', { major: 6, minor: 0 }],
    ] as const)('parses %s', (model, expected) => {
        expect(parseOpenAIGptVersion(model)).toEqual(expected);
    });

    it('uses numeric GTE comparisons rather than string ordering', () => {
        expect(isOpenAIGptVersionGTE('gpt-5.10', 5, 6)).toBe(true);
        expect(isOpenAIGptVersionGTE('gpt-5.2', 5, 6)).toBe(false);
        expect(parseOpenAIGptVersion('notgpt-6')).toBeNull();
    });

    it.each([
        ['gpt-5', { Minimal: 'minimal', Low: 'low', Medium: 'medium', High: 'high' }],
        ['gpt-5.1', { None: 'none', Low: 'low', Medium: 'medium', High: 'high' }],
        ['gpt-5.4', { None: 'none', Low: 'low', Medium: 'medium', High: 'high', 'Extra High': 'xhigh' }],
        [
            'gpt-5.6-sol',
            {
                None: 'none',
                Low: 'low',
                Medium: 'medium',
                High: 'high',
                'Extra High': 'xhigh',
                Max: 'max',
            },
        ],
        ['gpt-5-pro', { 'High (only)': 'high' }],
        ['gpt-5.4-pro', { Medium: 'medium', 'High (default)': 'high', 'Extra High': 'xhigh' }],
    ] as const)('advertises documented effort levels for %s', (model, expected) => {
        expect(getOpenAIReasoningEffortLevels(model)).toEqual(expected);
    });
});

describe('Gemini model version parsing', () => {
    it.each([
        ['gemini-3.5-flash', '3.5'],
        ['publishers/google/models/gemini-3.1-pro-preview', '3.1'],
        ['gemini-4-flash', '4'],
    ])('parses %s', (model, expected) => {
        expect(getGeminiModelVersion(model)).toBe(expected);
    });

    it('uses numeric GTE comparisons for future models', () => {
        expect(isGeminiModelVersionGte('gemini-3.10-flash', '3.5')).toBe(true);
        expect(isGeminiModelVersionGte('gemini-3.1-pro', '3.5')).toBe(false);
        expect(isGeminiModelVersionGte('gemini-4-pro', '3.5')).toBe(true);
    });
});
