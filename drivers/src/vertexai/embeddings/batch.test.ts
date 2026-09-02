import { URLDataSource } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { VertexAIDriver } from '../index.js';
import {
    cancelVertexEmbeddingBatch,
    createVertexEmbeddingBatch,
    deleteVertexEmbeddingBatch,
    formatVertexEmbeddingBatchRow,
    getVertexEmbeddingBatch,
    getVertexEmbeddingBatchCapability,
    normalizeVertexEmbeddingModelId,
    vertexBatchTextForParity,
} from './batch.js';

describe('Vertex embedding batch capabilities', () => {
    it('uses an explicit model and modality matrix', () => {
        expect(getVertexEmbeddingBatchCapability('gemini-embedding-2', 'image')).toMatchObject({
            schema: 'gemini',
            maxRows: 1_000_000,
            location: 'global',
            taskEncoding: 'prefix',
            maxSynchronousInputs: 1,
        });
        expect(getVertexEmbeddingBatchCapability('gemini-embedding-001', 'image')).toBeUndefined();
        expect(getVertexEmbeddingBatchCapability('text-embedding-005', 'text')).toMatchObject({
            schema: 'legacy',
            maxRows: 30_000,
        });
        expect(getVertexEmbeddingBatchCapability('multimodalembedding@001', 'image')).toBeUndefined();
        expect(getVertexEmbeddingBatchCapability('future-gemini-embedding-2', 'text')).toBeUndefined();
    });

    it('normalizes a Vertex resource name without fuzzy matching', () => {
        expect(normalizeVertexEmbeddingModelId('publishers/google/models/gemini-embedding-2')).toBe(
            'gemini-embedding-2',
        );
    });
});

describe('Vertex embedding batch lifecycle', () => {
    it('adopts an exact matching deterministic display name before creating', async () => {
        const create = vi.fn();
        const existing = {
            name: 'projects/p/locations/global/batchPredictionJobs/1',
            displayName: 'stable-name',
            state: 'JOB_STATE_RUNNING',
            model: 'gemini-embedding-2',
            src: { gcsUri: ['gs://bucket/input.jsonl'] },
            dest: { gcsUri: 'gs://bucket/output/' },
        };
        const client = {
            batches: {
                list: vi.fn().mockResolvedValue({
                    async *[Symbol.asyncIterator]() {
                        yield existing;
                    },
                }),
                create,
            },
        };
        const driver = { getGoogleGenAIClient: vi.fn(() => client) } as unknown as VertexAIDriver;
        await expect(
            createVertexEmbeddingBatch(driver, {
                model: 'gemini-embedding-2',
                modality: 'text',
                displayName: 'stable-name',
                inputUri: 'gs://bucket/input.jsonl',
                outputUri: 'gs://bucket/output/',
            }),
        ).resolves.toMatchObject({ name: existing.name, state: 'running' });
        expect(create).not.toHaveBeenCalled();
    });

    it('creates, cancels, polls and deletes through the generic batches API', async () => {
        const job = { name: 'jobs/1', state: 'JOB_STATE_PENDING', model: 'gemini-embedding-001' };
        const batches = {
            list: vi.fn().mockResolvedValue({ async *[Symbol.asyncIterator]() {} }),
            create: vi.fn().mockResolvedValue(job),
            get: vi.fn().mockResolvedValue(job),
            cancel: vi.fn().mockResolvedValue(undefined),
            delete: vi.fn().mockResolvedValue(undefined),
        };
        const driver = { getGoogleGenAIClient: vi.fn(() => ({ batches })) } as unknown as VertexAIDriver;
        await createVertexEmbeddingBatch(driver, {
            model: 'gemini-embedding-001',
            modality: 'text',
            displayName: 'stable-name',
            inputUri: 'gs://bucket/input.jsonl',
            outputUri: 'gs://bucket/output/',
        });
        await cancelVertexEmbeddingBatch(driver, 'gemini-embedding-001', 'text', 'jobs/1');
        await deleteVertexEmbeddingBatch(driver, 'gemini-embedding-001', 'text', 'jobs/1');
        expect(batches.create).toHaveBeenCalledWith(expect.objectContaining({ model: 'gemini-embedding-001' }));
        expect(batches.cancel).toHaveBeenCalledWith({ name: 'jobs/1' });
        expect(batches.get).toHaveBeenCalledWith({ name: 'jobs/1' });
        expect(batches.delete).toHaveBeenCalledWith({ name: 'jobs/1' });
    });

    it('rejects a provider job that belongs to a different model', async () => {
        const batches = {
            get: vi.fn().mockResolvedValue({
                name: 'jobs/1',
                state: 'JOB_STATE_RUNNING',
                model: 'publishers/google/models/text-embedding-005',
            }),
        };
        const driver = { getGoogleGenAIClient: vi.fn(() => ({ batches })) } as unknown as VertexAIDriver;

        await expect(getVertexEmbeddingBatch(driver, 'gemini-embedding-001', 'text', 'jobs/1')).rejects.toThrow(
            'belongs to model',
        );
    });
});

describe('Vertex embedding batch rows', () => {
    it('formats Gemini 001 text with API task configuration', async () => {
        await expect(
            formatVertexEmbeddingBatchRow({
                key: 'text:1:etag',
                model: 'gemini-embedding-001',
                dimensions: 768,
                input: { type: 'text', text: 'hello', task_type: 'document', title: 'Greeting' },
            }),
        ).resolves.toEqual({
            key: 'text:1:etag',
            request: {
                content: { parts: [{ text: 'hello' }] },
                embed_content_config: {
                    output_dimensionality: 768,
                    task_type: 'RETRIEVAL_DOCUMENT',
                    title: 'Greeting',
                },
            },
        });
    });

    it('formats Gemini 2 text with the same prompt prefix as synchronous embedding', async () => {
        const input = { type: 'text', text: 'hello', task_type: 'document' } as const;
        const row = await formatVertexEmbeddingBatchRow({
            key: 'text:1:etag',
            model: 'gemini-embedding-2',
            dimensions: 1024,
            input,
        });
        expect(row).toEqual({
            key: 'text:1:etag',
            request: {
                content: { parts: [{ text: 'title: none | text: hello' }] },
                embed_content_config: { output_dimensionality: 1024 },
            },
        });
        expect(vertexBatchTextForParity(input, 'gemini-embedding-2')).toBe('title: none | text: hello');
    });

    it('formats Gemini 2 images as GCS file data', async () => {
        await expect(
            formatVertexEmbeddingBatchRow({
                key: 'image:1:etag',
                model: 'gemini-embedding-2',
                dimensions: 768,
                input: {
                    type: 'image',
                    source: new URLDataSource('rendition.jpg', 'image/jpeg', 'gs://bucket/rendition.jpg'),
                },
            }),
        ).resolves.toEqual({
            key: 'image:1:etag',
            request: {
                content: {
                    parts: [
                        {
                            fileData: { fileUri: 'gs://bucket/rendition.jpg', mimeType: 'image/jpeg' },
                        },
                    ],
                },
                embed_content_config: { output_dimensionality: 768 },
            },
        });
    });

    it('formats stable legacy text rows', async () => {
        await expect(
            formatVertexEmbeddingBatchRow({
                key: 'properties:1:etag',
                model: 'text-embedding-005',
                dimensions: 768,
                input: { type: 'text', text: '{"name":"Ada"}' },
            }),
        ).resolves.toEqual({
            key: 'properties:1:etag',
            content: '{"name":"Ada"}',
            outputDimensionality: 768,
        });
    });

    it('rejects a row format that does not match the snapshotted model profile', async () => {
        await expect(
            formatVertexEmbeddingBatchRow({
                key: 'text:1:etag',
                model: 'gemini-embedding-001',
                dimensions: 768,
                schema: 'legacy',
                input: { type: 'text', text: 'hello' },
            }),
        ).rejects.toThrow('uses gemini batch rows');
    });
});
