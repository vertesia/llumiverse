import { describe, expect, it } from 'vitest';
import { openAIReasoningEffort } from './index.js';

describe('OpenAI reasoning effort', () => {
    it.each([
        'none',
        'minimal',
        'low',
        'medium',
        'high',
        'xhigh',
        'max',
    ])('preserves caller-supplied %s effort for reasoning models', (effort) => {
        expect(openAIReasoningEffort('gpt-5.6-sol', effort)).toBe(effort);
    });

    it('does not send effort to a non-reasoning model', () => {
        expect(openAIReasoningEffort('gpt-4o', 'medium')).toBeUndefined();
    });

    it('preserves effort for Bedrock Mantle Grok reasoning models', () => {
        expect(openAIReasoningEffort('xai.grok-4.3', 'none')).toBe('none');
    });
});
