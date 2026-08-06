import { describe, expect, it } from 'vitest';
import { openAIReasoningEffort } from './index.js';

describe('OpenAI reasoning effort', () => {
    it.each(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])(
        'preserves caller-supplied %s effort for reasoning models',
        (effort) => {
            expect(openAIReasoningEffort(effort)).toBe(effort);
        },
    );

    it('omits effort when the caller does not supply it', () => {
        expect(openAIReasoningEffort(undefined)).toBeUndefined();
    });
});
