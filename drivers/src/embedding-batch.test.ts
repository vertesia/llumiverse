import type { Driver, EmbeddingBatchArtifactStore } from '@llumiverse/core';
import { URLDataSource } from '@llumiverse/core';
import { assert, describe, expect, it, vi } from 'vitest';
import { getEmbeddingBatchAdapter } from './embedding-batch.js';

describe('embedding batch adapter', () => {
    it('rejects unsupported batch media locations before downloading or copying', async () => {
        const adapter = getEmbeddingBatchAdapter('vertexai');
        assert(adapter);
        const source = new URLDataSource('image.jpg', 'image/jpeg', 'https://example.com/image.jpg');
        const read = vi.spyOn(source, 'getStream');
        await expect(
            adapter.formatRow({
                key: 'row',
                model: 'gemini-embedding-2',
                dimensions: 2,
                inputFormat: 'gemini',
                input: { type: 'image', source },
            }),
        ).rejects.toThrow('gs://');
        expect(read).not.toHaveBeenCalled();
    });
    it('opts in explicitly and resolves model-specific location and limits', () => {
        expect(getEmbeddingBatchAdapter('openai')).toBeUndefined();
        const adapter = getEmbeddingBatchAdapter('vertexai');
        assert(adapter);
        expect(adapter.capability('gemini-embedding-2', 'image', 'us-central1')).toMatchObject({
            inputFormat: 'gemini',
            location: 'global',
            maxRows: 1_000_000,
        });
        expect(adapter.capability('text-embedding-004', 'text', 'us-central1')).toMatchObject({
            inputFormat: 'legacy',
            location: 'us-central1',
            maxRows: 30_000,
        });
        expect(adapter.capability('multimodalembedding@001', 'image')).toBeUndefined();
    });

    it.each(['gemini', 'legacy'])('owns input/output correlation for %s rows', async (inputFormat) => {
        const adapter = getEmbeddingBatchAdapter('vertexai');
        assert(adapter);
        const { row, resultKey } = await adapter.formatRow({
            key: 'source-key',
            model: inputFormat === 'legacy' ? 'text-embedding-004' : 'gemini-embedding-2',
            input: { type: 'text', text: 'hello' },
            dimensions: 2,
            inputFormat,
        });
        const record =
            inputFormat === 'legacy'
                ? { instance: row, predictions: [{ embeddings: { values: [1, 2] } }] }
                : { key: row.key, response: { embedding: { values: [1, 2] } } };
        expect(adapter.parseResult(record)).toMatchObject({ key: resultKey, vector: [1, 2], providerError: false });
    });

    it.each(['succeeded', 'failed', 'cancelled'] as const)('discovers all %s shards without copying', async (state) => {
        const artifacts: EmbeddingBatchArtifactStore = {
            list: vi
                .fn()
                .mockResolvedValue(['gs://bucket/output/a/predictions.jsonl', 'gs://bucket/output/b/errors.jsonl']),
            read: vi.fn(),
            write: vi.fn(),
        };
        const adapter = getEmbeddingBatchAdapter('vertexai');
        assert(adapter);
        await expect(
            adapter.outputArtifacts(
                {} as Driver,
                {
                    name: 'jobs/1',
                    state,
                    outputUri: 'gs://bucket/output/',
                },
                artifacts,
            ),
        ).resolves.toHaveLength(2);
        expect(artifacts.list).toHaveBeenCalledWith('gs://bucket/output/');
        expect(artifacts.read).not.toHaveBeenCalled();
        expect(artifacts.write).not.toHaveBeenCalled();
    });
});
