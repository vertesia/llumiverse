import { describe, expect, it } from 'vitest';
import {
    formatVertexEmbeddingBatchRow,
    parseVertexEmbeddingBatchResult,
    vertexEmbeddingBatchCorrelationKey,
} from './batch.js';

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

    it('parses legacy output by echoed input fields, not discarded custom keys', () => {
        expect(
            parseVertexEmbeddingBatchResult({
                instance: { key: 'legacy', content: 'omitted' },
                predictions: [{ embeddings: { values: [3, 4] } }],
            }),
        ).toEqual({
            key: vertexEmbeddingBatchCorrelationKey({ content: 'omitted' }),
            vector: [3, 4],
            providerError: false,
        });
    });

    it('correlates actual legacy output independently of field order and preserves all input semantics', async () => {
        const row = await formatVertexEmbeddingBatchRow({
            key: 'object-snapshot',
            model: 'text-embedding-004',
            dimensions: 256,
            input: { type: 'text', text: 'Café\nsecond line 🐈' },
        });
        const key = vertexEmbeddingBatchCorrelationKey(row);
        expect(key).toMatch(/^input:[a-f0-9]{64}$/);
        expect(
            parseVertexEmbeddingBatchResult({
                instance: { content: 'Café\nsecond line 🐈' },
                status: '',
                predictions: [{ embeddings: { values: [1, 2] } }],
            }),
        ).toEqual({ key, providerError: false, vector: [1, 2] });
        expect(vertexEmbeddingBatchCorrelationKey({ key: 'another-object', content: 'Café\nsecond line 🐈' })).toBe(
            key,
        );
        for (const changed of [
            { content: 'different' },
            { content: 'Café\nsecond line 🐈', task_type: 'RETRIEVAL_QUERY' },
            { content: 'Café\nsecond line 🐈', title: 'title' },
        ]) {
            expect(vertexEmbeddingBatchCorrelationKey(changed)).not.toBe(key);
        }
        expect(vertexEmbeddingBatchCorrelationKey({ content: 'x', task_type: 42 })).toBeUndefined();
        expect(vertexEmbeddingBatchCorrelationKey({ content: 42 })).toBeUndefined();
        expect(
            parseVertexEmbeddingBatchResult({ instance: { content: 'Café\nsecond line 🐈' }, status: 'row failed' }),
        ).toMatchObject({ key, providerError: true });
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
