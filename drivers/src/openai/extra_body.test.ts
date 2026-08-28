import { describe, expect, it } from 'vitest';
import { getOpenAIExtraBody, mergeOpenAIExtraBody } from './extra_body.js';

describe('OpenAI-compatible extra body', () => {
    it('extracts only object-shaped extension fields', () => {
        expect(getOpenAIExtraBody({ extra_body: { provider: { sort: 'price' } } })).toEqual({
            provider: { sort: 'price' },
        });
        expect(getOpenAIExtraBody({ extra_body: ['invalid'] })).toBeUndefined();
        expect(getOpenAIExtraBody(undefined)).toBeUndefined();
    });

    it('merges extensions at the top level while preserving core request fields', () => {
        expect(
            mergeOpenAIExtraBody(
                { model: 'actual-model', stream: false },
                { provider: { sort: 'throughput' }, model: 'override', stream: true },
            ),
        ).toEqual({
            provider: { sort: 'throughput' },
            model: 'actual-model',
            stream: false,
        });
    });
});
