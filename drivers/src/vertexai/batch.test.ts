import { describe, expect, it } from 'vitest';
import { parseVertexModelTarget, parseVertexResourceLocation } from './batch.js';

describe('Vertex batch resource normalization', () => {
    it('moves a catalog model location onto the client target', () => {
        expect(parseVertexModelTarget('locations/global/publishers/google/models/gemini-3.5-flash-lite')).toEqual({
            model: 'gemini-3.5-flash-lite',
            location: 'global',
        });
    });

    it('accepts partial and bare Gemini model ids', () => {
        expect(parseVertexModelTarget('publishers/google/models/gemini-2.5-flash')).toEqual({
            model: 'gemini-2.5-flash',
            location: undefined,
        });
        expect(parseVertexModelTarget('gemini-2.5-flash')).toEqual({
            model: 'gemini-2.5-flash',
            location: undefined,
        });
    });

    it('recovers the location from a batch job resource name', () => {
        expect(parseVertexResourceLocation('projects/test/locations/global/batchPredictionJobs/123')).toBe('global');
    });
});
