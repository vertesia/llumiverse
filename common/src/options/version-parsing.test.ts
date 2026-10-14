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
    supportsClaudeForcedToolChoice,
    supportsEffort,
    supportsMidConversationToolChanges,
} from './version-parsing.js';

describe('Claude model version parsing', () => {
    it.each([
        ['claude-sonnet-5', { major: 5, minor: 0, variant: 'sonnet' }],
        ['claude-opus-5-5', { major: 5, minor: 5, variant: 'opus' }],
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

    it.each(['claude-fable-5', 'claude-mythos-5', 'claude-sonnet-5', 'claude-sonnet-5-5', 'claude-opus-4-8'])(
        'advertises adaptive thinking and current effort levels for %s',
        (model) => {
            expect(supportsAdaptiveThinking(model)).toBe(true);
            expect(supportsEffort(model)).toBe(true);
            expect(getAvailableEffortLevels(model)).toEqual({
                Low: 'low',
                Medium: 'medium',
                'High (default)': 'high',
                'Extra High': 'xhigh',
                Max: 'max',
            });
        },
    );

    it.each(['claude-haiku-5', 'claude-haiku-5-4', 'claude-opus-5', 'claude-opus-5-4', 'claude-opus-6'])(
        'preserves high default effort below the Haiku boundary and for other generations: %s',
        (model) => {
            expect(getAvailableEffortLevels(model)).toHaveProperty('High (default)', 'high');
            expect(getAvailableEffortLevels(model)).not.toHaveProperty('Medium (default)');
        },
    );

    it.each(['claude-opus-5-5', 'claude-haiku-5-5', 'us.anthropic.claude-haiku-5-5', 'claude-haiku-6'])(
        'marks medium as the default effort for %s',
        (model) => {
            expect(getAvailableEffortLevels(model)).toEqual({
                Low: 'low',
                'Medium (default)': 'medium',
                High: 'high',
                'Extra High': 'xhigh',
                Max: 'max',
            });
        },
    );
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
        ['gpt-6-astra', { major: 6, minor: 0 }],
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
                'Medium (default)': 'medium',
                High: 'high',
                'Extra High': 'xhigh',
                Max: 'max',
            },
        ],
        ['gpt-5-pro', { 'High (only)': 'high' }],
        ['gpt-5.4-pro', { Medium: 'medium', 'High (default)': 'high', 'Extra High': 'xhigh' }],
        ['gpt-6-astra', { Low: 'low', 'Medium (default)': 'medium', High: 'high', 'Extra High': 'xhigh', Max: 'max' }],
        [
            'gpt-6-sol',
            {
                None: 'none',
                Low: 'low',
                'Medium (default)': 'medium',
                High: 'high',
                'Extra High': 'xhigh',
                Max: 'max',
            },
        ],
        [
            'gpt-6-luna',
            {
                None: 'none',
                Low: 'low',
                'Medium (default)': 'medium',
                High: 'high',
                'Extra High': 'xhigh',
                Max: 'max',
            },
        ],
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

describe('supportsMidConversationToolChanges', () => {
    it.each([
        'claude-opus-4-8',
        'claude-opus-5',
        'claude-opus-5-5',
        'claude-sonnet-5-5',
        'claude-haiku-5-5',
        'claude-fable-5',
        'claude-fable-5-1',
        'claude-mythos-5-1',
        'us.anthropic.claude-sonnet-5-5',
        'arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-opus-5-5',
        'publishers/anthropic/models/claude-sonnet-5-5@20260901',
        'claude-sonnet-6',
        'claude-haiku-6',
        'claude-opus-6',
        'anthropic.claude-opus-4-8-v1:0',
        'claude-opus-4-8@20260901',
        'eu.anthropic.claude-haiku-5-5-20261001-v1:0',
        'claude-opus-4-10',
        'claude-sonnet-5-6',
        'claude-mythos-5',
        'claude-fable-6',
    ])('accepts %s', (model) => {
        expect(supportsMidConversationToolChanges(model)).toBe(true);
    });

    it.each([
        'claude-sonnet-5',
        'claude-haiku-5',
        'claude-opus-4-7',
        'claude-sonnet-4-6',
        'claude-haiku-4-5',
        'claude-3-7-sonnet-20250219',
        'claude-mythos-preview',
        'gpt-5',
        'claude-sonnet-5@20260901',
        'claude-sonnet-5-20260901',
        'claude-sonnet-5-v1:0',
        'claude-opus-4',
        'claude-opus-4-1',
        'us.anthropic.claude-mythos-preview-v1:0',
        'claude-mythos-preview-1',
    ])('rejects %s', (model) => {
        expect(supportsMidConversationToolChanges(model)).toBe(false);
    });
});

describe('supportsClaudeForcedToolChoice', () => {
    it.each([
        'claude-opus-5-5',
        'claude-sonnet-5-5',
        'claude-sonnet-5-6',
        'claude-opus-6',
        'claude-sonnet-6',
        'us.anthropic.claude-opus-5-5-v1:0',
        'global.anthropic.claude-sonnet-5-5',
        'arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-opus-5-5',
        'publishers/anthropic/models/claude-sonnet-5-5@20260901',
        'claude-opus-5-5@20260901',
    ])('rejects a forced tool choice on %s', (model) => {
        expect(supportsClaudeForcedToolChoice(model)).toBe(false);
    });

    it.each([
        'claude-haiku-5-5',
        'claude-haiku-6',
        'claude-opus-5',
        'claude-sonnet-5',
        'claude-sonnet-5-20260901',
        'claude-opus-4-8',
        'claude-sonnet-4-6',
        'claude-fable-5',
        'claude-fable-6',
        'claude-mythos-5',
        'eu.anthropic.claude-haiku-5-5-20261001-v1:0',
        'anthropic.claude-opus-4-8-v1:0',
        'claude-3-7-sonnet-20250219',
        'gpt-5',
    ])('accepts a forced tool choice on %s', (model) => {
        expect(supportsClaudeForcedToolChoice(model)).toBe(true);
    });
});
