import { getOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { resolveClaudeThinking } from './claude-thinking.js';

describe('resolveClaudeThinking', () => {
    it('selects adaptive thinking with medium effort for Sonnet 4.6', () => {
        expect(resolveClaudeThinking('claude-sonnet-4-6', { effort: 'medium' })).toMatchObject({
            thinking: { type: 'adaptive', display: 'omitted' },
            outputConfig: { effort: 'medium' },
        });
    });

    it('prefers adaptive effort over a stale legacy budget', () => {
        expect(
            resolveClaudeThinking('claude-sonnet-4-6', {
                effort: 'medium',
                thinking_budget_tokens: 12_000,
            }),
        ).toMatchObject({
            thinking: { type: 'adaptive', display: 'omitted' },
            outputConfig: { effort: 'medium' },
        });
    });

    it('preserves explicitly budgeted extended thinking on dual-mode models', () => {
        expect(resolveClaudeThinking('claude-sonnet-4-6', { thinking_budget_tokens: 12_000 })).toMatchObject({
            thinking: { type: 'enabled', budget_tokens: 12_000 },
            outputConfig: undefined,
        });
        expect(getOptions('claude-sonnet-4-6', 'anthropic').options.map((option) => option.name)).toContain(
            'thinking_budget_tokens',
        );
        expect(getOptions('claude-sonnet-5', 'anthropic').options.map((option) => option.name)).not.toContain(
            'thinking_budget_tokens',
        );
    });

    it('keeps extended-only models budget-driven', () => {
        expect(resolveClaudeThinking('claude-3-7-sonnet', { thinking_budget_tokens: 8000 })).toMatchObject({
            thinking: { type: 'enabled', budget_tokens: 8000 },
            outputConfig: undefined,
        });
        expect(resolveClaudeThinking('claude-3-7-sonnet', { effort: 'medium' })).toMatchObject({
            thinking: { type: 'disabled' },
            outputConfig: { effort: 'medium' },
        });
    });

    it('supports adaptive thinking on future Claude models', () => {
        expect(resolveClaudeThinking('claude-sonnet-5', { effort: 'medium', include_thoughts: true })).toMatchObject({
            thinking: { type: 'adaptive', display: 'summarized' },
            outputConfig: { effort: 'medium' },
        });
    });
});
