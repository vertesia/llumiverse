import { describe, expect, it } from 'vitest';
import { resolveGeminiRegion } from './gemini.js';

describe('Gemini region resolution', () => {
    it.each([
        ['gemini-4-argon', undefined, 'global'],
        ['gemini-4-argon', 'us', 'us'],
        ['gemini-4-argon', 'eu', 'eu'],
        ['gemini-3.8-flash', undefined, undefined],
        ['gemini-3.8-flash', 'us-central1', 'us-central1'],
        ['gemini-nano-banana-2.1', 'us-central1', 'global'],
        ['gemini-2.5-flash-image', undefined, undefined],
        ['gemini-2.5-flash-image', 'us-central1', 'us-central1'],
    ])('resolves %s with location %s to %s', (model, explicitRegion, expected) => {
        expect(resolveGeminiRegion(model, explicitRegion)).toBe(expected);
    });
});
