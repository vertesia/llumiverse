import { describe, expect, it } from 'vitest';
import { parseVertexEmbeddingBatchResult } from './batch.js';

describe('embedding batch result parsing', () => {
    it('parses the alternate embeddings array and ignores non-string keys', () => {
        expect(
            parseVertexEmbeddingBatchResult({
                key: 42,
                response: { embeddings: [{ values: [1, 2] }] },
            }),
        ).toEqual({ key: undefined, vector: [1, 2], providerError: false });
    });

    it.each([JSON.stringify({ code: 7, message: 'private details' }), { code: 7 }])(
        'recognizes Gemini status permission failures without exposing provider messages',
        (status) => {
            expect(parseVertexEmbeddingBatchResult({ key: 'row', status, response: {} })).toEqual({
                key: 'row',
                vector: undefined,
                providerError: true,
                failureCategory: 'provider_permission_denied',
            });
        },
    );

    it.each(['', { code: 0 }, JSON.stringify({ code: 0 })])('accepts successful status %j', (status) => {
        expect(parseVertexEmbeddingBatchResult({ status }).providerError).toBe(false);
    });

    it('treats an unparseable status as a provider error', () => {
        expect(parseVertexEmbeddingBatchResult({ status: 'unexpected failure' }).providerError).toBe(true);
    });
    it('parses Gemini output by top-level key', () => {
        expect(parseVertexEmbeddingBatchResult({ key: 'gemini', response: { embedding: { values: [1, 2] } } })).toEqual(
            {
                key: 'gemini',
                vector: [1, 2],
                providerError: false,
            },
        );
    });

    it('parses legacy output by the echoed instance key', () => {
        expect(
            parseVertexEmbeddingBatchResult({
                instance: { key: 'legacy', content: 'omitted' },
                predictions: [{ embeddings: { values: [3, 4] } }],
            }),
        ).toEqual({ key: 'legacy', vector: [3, 4], providerError: false });
    });

    it('classifies provider row errors without retaining their message', () => {
        expect(
            parseVertexEmbeddingBatchResult({ key: 'failed', error: { message: 'sensitive provider detail' } }),
        ).toEqual({
            key: 'failed',
            vector: undefined,
            providerError: true,
        });
    });
});
