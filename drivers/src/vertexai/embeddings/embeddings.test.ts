import { Base64DataSource, URLDataSource } from '@llumiverse/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { generateVertexAiEmbeddings } from './embed.js';
import { generateLegacyMultimodalEmbeddings } from './embed-legacy-multimodal.js';

const FAKE_VECTOR = [0.1, 0.2, 0.3];

function makeDriver(responses: unknown[]) {
    const driver = new VertexAIDriver({ project: 'test-project', region: 'us-central1' });
    const mockPost = vi.spyOn(driver, 'postVertexModel');
    for (const response of responses) {
        mockPost.mockResolvedValueOnce(response as never);
    }
    return { driver, mockPost };
}

function textPredictResponse(vectors: number[][], tokenCounts: number[] = []) {
    return {
        predictions: vectors.map((values, index) => ({
            embeddings: {
                values,
                statistics: tokenCounts[index] === undefined ? undefined : { token_count: tokenCounts[index] },
            },
        })),
    };
}

function embedContentResponse(values: number[], promptTokenCount?: number) {
    return {
        embedding: { values },
        usageMetadata: promptTokenCount === undefined ? undefined : { promptTokenCount },
    };
}

describe('generateVertexAiEmbeddings - text predict models', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it('embeds a single text through predict and maps token usage', async () => {
        const { driver, mockPost } = makeDriver([textPredictResponse([FAKE_VECTOR], [5])]);

        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: 'text', text: 'hello world' }],
            model: 'text-embedding-005',
        });

        expect(mockPost).toHaveBeenCalledWith('text-embedding-005', 'predict', {
            instances: [{ content: 'hello world', task_type: undefined, title: undefined }],
            parameters: undefined,
        });
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[0].outputs[0].modality).toBe('text');
        expect(result.usage?.input_tokens).toBe(5);
        expect(result.usage?.input_text_tokens).toBe(5);
    });

    it('maps query/document task types into predict instances', async () => {
        const { driver, mockPost } = makeDriver([textPredictResponse([FAKE_VECTOR, [0.4, 0.5]])]);

        await generateVertexAiEmbeddings(driver, {
            inputs: [
                { type: 'text', text: 'search query', task_type: 'query' },
                { type: 'text', text: 'a document', task_type: 'document', title: 'Doc' },
            ],
            model: 'text-embedding-005',
            dimensions: 128,
        });

        expect(mockPost).toHaveBeenCalledWith('text-embedding-005', 'predict', {
            instances: [
                { content: 'search query', task_type: 'RETRIEVAL_QUERY', title: undefined },
                { content: 'a document', task_type: 'RETRIEVAL_DOCUMENT', title: 'Doc' },
            ],
            parameters: { outputDimensionality: 128 },
        });
    });

    it('keeps gemini-embedding-001 to one input per predict request', async () => {
        const { driver, mockPost } = makeDriver([
            textPredictResponse([FAKE_VECTOR]),
            textPredictResponse([[0.4, 0.5]]),
        ]);

        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [
                { type: 'text', text: 'first' },
                { type: 'text', text: 'second' },
            ],
            model: 'gemini-embedding-001',
        });

        expect(mockPost).toHaveBeenCalledTimes(2);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[1].outputs[0].values).toEqual([0.4, 0.5]);
    });
});

describe('generateVertexAiEmbeddings - embedContent models', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it('uses global embedContent and task prefix for gemini-embedding-2 query text', async () => {
        const { driver, mockPost } = makeDriver([embedContentResponse(FAKE_VECTOR, 7)]);

        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: 'text', text: 'find something', task_type: 'query' }],
            model: 'gemini-embedding-2',
        });

        expect(mockPost).toHaveBeenCalledWith(
            'gemini-embedding-2',
            'embedContent',
            {
                content: { role: 'user', parts: [{ text: 'task: search result | query: find something' }] },
                title: undefined,
            },
            { region: 'global' },
        );
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.usage?.input_tokens).toBe(7);
    });

    it("uses 'title: none' document prefix when title is absent", async () => {
        const { driver, mockPost } = makeDriver([embedContentResponse(FAKE_VECTOR)]);

        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: 'text', text: 'content', task_type: 'document' }],
            model: 'gemini-embedding-2',
        });

        const payload = mockPost.mock.calls[0][2] as { content: { parts: Array<{ text: string }> } };
        expect(payload.content.parts[0].text).toBe('title: none | text: content');
    });

    it('uses inlineData or fileData for media inputs', async () => {
        const b64 = Buffer.from('img-bytes').toString('base64');
        const inline = new Base64DataSource('img.jpg', 'image/jpeg', b64);
        const gcs = new URLDataSource('img.jpg', 'image/jpeg', 'gs://my-bucket/img.jpg');
        const { driver, mockPost } = makeDriver([embedContentResponse(FAKE_VECTOR), embedContentResponse([0.4])]);

        await generateVertexAiEmbeddings(driver, {
            inputs: [
                { type: 'image', source: inline },
                { type: 'image', source: gcs },
            ],
            model: 'gemini-embedding-2',
        });

        expect(mockPost.mock.calls[0][2]).toMatchObject({
            content: { role: 'user', parts: [{ inlineData: { data: b64, mimeType: 'image/jpeg' } }] },
        });
        expect(mockPost.mock.calls[1][2]).toMatchObject({
            content: { role: 'user', parts: [{ fileData: { fileUri: 'gs://my-bucket/img.jpg' } }] },
        });
    });
});

describe('generateLegacyMultimodalEmbeddings - multimodalembedding@001', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it('embeds a text input through predict', async () => {
        const { driver, mockPost } = makeDriver([{ predictions: [{ textEmbedding: FAKE_VECTOR }] }]);
        const result = await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: 'text', text: 'hello' }],
        });

        expect(mockPost).toHaveBeenCalledWith('multimodalembedding@001', 'predict', {
            instances: [{ text: 'hello' }],
            parameters: undefined,
        });
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[0].outputs[0].modality).toBe('text');
    });

    it('embeds inline images and passes dimensions', async () => {
        const b64 = Buffer.from('img').toString('base64');
        const ds = new Base64DataSource('img.jpg', 'image/jpeg', b64);
        const { driver, mockPost } = makeDriver([{ predictions: [{ imageEmbedding: FAKE_VECTOR }] }]);

        const result = await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: 'image', source: ds }],
            dimensions: 256,
        });

        expect(mockPost.mock.calls[0][2]).toMatchObject({
            instances: [{ image: { bytesBase64Encoded: b64, mimeType: 'image/jpeg' } }],
            parameters: { dimension: 256 },
        });
        expect(result.results[0].outputs[0].modality).toBe('image');
    });

    it('routes audio through the video payload path and preserves audio modality', async () => {
        const ds = new URLDataSource('a.mp3', 'audio/mpeg', 'gs://my-bucket/a.mp3');
        const { driver } = makeDriver([{ predictions: [{ videoEmbeddings: [{ embedding: FAKE_VECTOR }] }] }]);

        const result = await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: 'audio', source: ds }],
        });

        expect(result.results[0].outputs[0].modality).toBe('audio');
    });

    it('throws when prediction count mismatches input count', async () => {
        const { driver } = makeDriver([
            { predictions: [{ textEmbedding: FAKE_VECTOR }, { textEmbedding: [0.9, 0.8] }] },
        ]);

        await expect(
            generateLegacyMultimodalEmbeddings(driver, {
                inputs: [{ type: 'text', text: 'one' }],
            }),
        ).rejects.toThrow(/predictions for .* instances/);
    });

    it('routes generateVertexAiEmbeddings to the legacy predict path', async () => {
        const { driver, mockPost } = makeDriver([{ predictions: [{ textEmbedding: FAKE_VECTOR }] }]);
        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: 'text', text: 'hello' }],
            model: 'multimodalembedding@001',
        });

        expect(mockPost).toHaveBeenCalledTimes(1);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
    });
});
