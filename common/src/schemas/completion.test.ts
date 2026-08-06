import { describe, expect, it } from 'vitest';
import { CompletionResultSchema } from './completion.js';

describe('CompletionResultSchema', () => {
    it('accepts thoughts as a separate completion result type', () => {
        expect(CompletionResultSchema.parse({ type: 'thoughts', value: 'Reasoning summary' })).toEqual({
            type: 'thoughts',
            value: 'Reasoning summary',
        });
    });
});
