import { describe, expect, it } from 'vitest';
import { getContextWindowSize, getMaxOutputTokens } from './context-windows.js';

describe('Claude context window limits', () => {
    it.each([
        'claude-sonnet-5',
        'claude-sonnet-5-20260701',
        'publishers/anthropic/models/claude-sonnet-5@20260701',
        'us.anthropic.claude-sonnet-5-v1:0',
        'arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-5-v1:0',
        'claude-sonnet-5-1',
    ])('uses future-compatible limits for %s', (model) => {
        expect(getMaxOutputTokens(model)).toBe(128_000);
        expect(getContextWindowSize(model)).toBe(1_000_000);
    });

    it('preserves release limits for older Claude models', () => {
        expect(getMaxOutputTokens('claude-sonnet-4-6')).toBe(64_000);
        expect(getContextWindowSize('claude-sonnet-4-6')).toBe(200_000);
        expect(getMaxOutputTokens('claude-3-5-sonnet-20241022')).toBe(8192);
        expect(getContextWindowSize('claude-3-5-sonnet-20241022')).toBe(200_000);
    });
});
