import { describe, expect, it } from 'vitest';
import { openAIReasoningEffort } from './index.js';

describe('OpenAI reasoning effort', () => {
    it.each(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])(
        'preserves caller-supplied %s effort for reasoning models',
        (effort) => {
            expect(openAIReasoningEffort('gpt-5.6-sol', effort)).toBe(effort);
        },
    );

    it('does not send effort to a non-reasoning model', () => {
        expect(openAIReasoningEffort('gpt-4o', 'medium')).toBeUndefined();
    });

    it('preserves effort for namespaced Bedrock Mantle Grok reasoning models', () => {
        expect(openAIReasoningEffort('xai.grok-4.3', 'none')).toBe('none');
    });

    it.each(['grok-4.5', 'grok-5-fast'])('preserves effort for direct xAI model id %s', (model) => {
        expect(openAIReasoningEffort(model, 'high')).toBe('high');
    });
});
