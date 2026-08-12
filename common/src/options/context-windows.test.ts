import { describe, expect, it } from 'vitest';
import { getContextWindowSize, getMaxOutputTokens } from './context-windows.js';

describe('Claude context window limits', () => {
    it.each([
        'claude-sonnet-5',
        'claude-fable-5',
        'claude-mythos-5',
        'claude-mythos-preview',
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

describe('OpenAI context window limits', () => {
    it.each(['gpt-5', 'gpt-5.2', 'models/gpt-5.3-20260101'])('preserves GPT-5 through 5.3 limits for %s', (model) => {
        expect(getMaxOutputTokens(model)).toBe(128_000);
        expect(getContextWindowSize(model)).toBe(400_000);
    });

    it.each(['gpt-5.4', 'gpt-5.6-sol', 'models/gpt-5.7-20270101'])(
        'uses current and future GPT-5 context limits for %s',
        (model) => {
            expect(getMaxOutputTokens(model)).toBe(128_000);
            expect(getContextWindowSize(model)).toBe(1_050_000);
        },
    );

    it('normalizes uppercase qualified IDs', () => {
        expect(getContextWindowSize('~OPENAI/GPT-5.6-SOL')).toBe(1_050_000);
        expect(getMaxOutputTokens('~OPENAI/GPT-5.6-SOL')).toBe(128_000);
    });

    it.each(['openai.gpt-5.4', 'us.openai.gpt-5.6-sol-v1:0'])(
        'does not infer transport limits from a source-qualified model ID for %s',
        (model) => {
            expect(getMaxOutputTokens(model)).toBe(128_000);
            expect(getContextWindowSize(model)).toBe(1_050_000);
        },
    );

    it('preserves the larger GPT-5 Pro output limit and o-series limits', () => {
        expect(getMaxOutputTokens('gpt-5-pro')).toBe(272_000);
        expect(getMaxOutputTokens('gpt-5.4-pro')).toBe(128_000);
        expect(getMaxOutputTokens('gpt-6-pro')).toBe(128_000);
        expect(getMaxOutputTokens('o4-mini')).toBe(100_000);
        expect(getMaxOutputTokens('~O5-PREVIEW')).toBe(100_000);
        expect(getContextWindowSize('~O5-PREVIEW')).toBe(200_000);
    });
});

describe('open-model context window limits', () => {
    it('inherits the newest known Llama context rule for future versions', () => {
        expect(getContextWindowSize('meta-llama/LLAMA-5-SCOUT')).toBe(10_000_000);
        expect(getContextWindowSize('meta.llama6-maverick-v1:0')).toBe(1_000_000);
    });

    it('uses current Mistral limits and leaves unknown limits unset', () => {
        expect(getContextWindowSize('mistral-small-2603')).toBe(256_000);
        expect(getContextWindowSize('mistral-medium-3.6')).toBe(256_000);
        expect(getContextWindowSize('magistral-medium-2509')).toBe(128_000);
        expect(getContextWindowSize('voxtral-small-2507')).toBe(32_000);
        expect(getContextWindowSize('labs-leanstral-1-5')).toBe(256_000);
        expect(getContextWindowSize('future-provider/new-capability-v1')).toBeUndefined();
    });
});
