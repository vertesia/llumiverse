import type { S3Client } from '@aws-sdk/client-s3';
import { BatchInferenceJobStatus } from '@llumiverse/common';
import {
    type BatchBlobStore,
    type BatchInferenceRequestItem,
    type DataSource,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import type { AuthClient } from 'google-auth-library';
import { describe, expect, it, vi } from 'vitest';
import {
    isBedrockBatchOutputKey,
    mapModelInvocationJobStatus,
    parseBedrockBatchOutputLine,
    parseNativeModelOutput,
    parseS3Bucket,
    s3List,
} from '../src/bedrock/batch.js';
import { BedrockDriver } from '../src/bedrock/index.js';
import { OpenAIDriver } from '../src/openai/openai.js';
import { TestDriver } from '../src/test-driver/index.js';
import {
    gcsList,
    mapBatchJobState,
    parseBatchOutputLine,
    parseGcsBucket,
    toRestGenerateContentRequest,
} from '../src/vertexai/batch.js';
import { VertexAIDriver } from '../src/vertexai/index.js';

const manyTrailingSlashes = '/'.repeat(100_000);

describe('vertex batch helpers', () => {
    it('parseGcsBucket handles gs://, bucket/prefix and bucket-only', () => {
        expect(parseGcsBucket('gs://my-bucket/pre/fix/')).toEqual({ bucket: 'my-bucket', prefix: 'pre/fix' });
        expect(parseGcsBucket('my-bucket/pre')).toEqual({ bucket: 'my-bucket', prefix: 'pre' });
        expect(parseGcsBucket('my-bucket')).toEqual({ bucket: 'my-bucket', prefix: '' });
        expect(parseGcsBucket(`gs://my-bucket/prefix${manyTrailingSlashes}`)).toEqual({
            bucket: 'my-bucket',
            prefix: 'prefix',
        });
    });

    it('mapBatchJobState maps Vertex JobState to provider-agnostic status', () => {
        expect(mapBatchJobState('JOB_STATE_SUCCEEDED')).toBe(BatchInferenceJobStatus.succeeded);
        expect(mapBatchJobState('JOB_STATE_FAILED')).toBe(BatchInferenceJobStatus.failed);
        expect(mapBatchJobState('JOB_STATE_EXPIRED')).toBe(BatchInferenceJobStatus.failed);
        expect(mapBatchJobState('JOB_STATE_CANCELLED')).toBe(BatchInferenceJobStatus.cancelled);
        expect(mapBatchJobState('JOB_STATE_PENDING')).toBe(BatchInferenceJobStatus.queued);
        expect(mapBatchJobState('JOB_STATE_RUNNING')).toBe(BatchInferenceJobStatus.running);
        expect(mapBatchJobState(undefined)).toBe(BatchInferenceJobStatus.running);
    });

    it('mapBatchJobState treats JOB_STATE_PARTIALLY_SUCCEEDED as terminal (succeeded)', () => {
        // Partial results are retrievable; item-level errors surface per-record.
        // Mapping to 'running' would make pollers loop forever.
        expect(mapBatchJobState('JOB_STATE_PARTIALLY_SUCCEEDED')).toBe(BatchInferenceJobStatus.succeeded);
    });

    it('gcsList follows nextPageToken pagination', async () => {
        const request = vi
            .fn()
            .mockResolvedValueOnce({ data: { items: [{ name: 'a' }, { name: 'b' }], nextPageToken: 'page-2' } })
            .mockResolvedValueOnce({ data: { items: [{ name: 'c' }] } });
        const auth = { request } as unknown as AuthClient;
        const names = await gcsList(auth, 'bucket', 'prefix');
        expect(names).toEqual(['a', 'b', 'c']);
        expect(request).toHaveBeenCalledTimes(2);
        expect(vi.mocked(request).mock.calls[1][0].url).toContain('pageToken=page-2');
    });

    it('parseBatchOutputLine falls back to request.labels.custom_id when not echoed top-level', () => {
        const line = JSON.stringify({
            request: { labels: { custom_id: 'page-42' } },
            response: { candidates: [{ content: { parts: [{ text: 'ok' }] } }] },
        });
        const item = parseBatchOutputLine(line, 0);
        expect(item?.custom_id).toBe('page-42');
    });

    it('startBatchInference rejects requests targeting different models', async () => {
        const driver = new VertexAIDriver({ project: 'test-project', region: 'us-central1' });
        const mkItem = (custom_id: string, model: string): BatchInferenceRequestItem => ({
            custom_id,
            segments: [{ role: PromptRole.user, content: 'hi' }],
            options: { model } as ExecutionOptions,
        });
        await expect(
            driver.startBatchInference([
                mkItem('a', 'publishers/google/models/gemini-2.5-flash'),
                mkItem('b', 'publishers/google/models/gemini-2.5-pro'),
            ]),
        ).rejects.toThrow(/same model/);
    });

    it('startBatchInference mirrors custom_id into per-request labels', async () => {
        const request: BatchInferenceRequestItem = {
            custom_id: 'page-7',
            segments: [{ role: PromptRole.user, content: 'hello' }],
            options: { model: 'publishers/google/models/gemini-2.5-flash' } as ExecutionOptions,
        };
        const putText = vi.fn().mockResolvedValue('gs://tenant-bucket/batch/input.jsonl');
        const blobStore: BatchBlobStore = { putText, readOutput: vi.fn().mockResolvedValue([]) };
        const driver = new VertexAIDriver({ project: 'test-project', region: 'us-central1' });
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue({
            batches: {
                create: vi.fn().mockResolvedValue({ name: 'batch-1', state: 'JOB_STATE_PENDING' }),
            },
        } as unknown as ReturnType<VertexAIDriver['getGoogleGenAIClient']>);

        await driver.startBatchInference([request], { name: 'labels', blobStore });

        const input = JSON.parse(vi.mocked(putText).mock.calls[0][1]) as {
            custom_id: string;
            request: { labels?: Record<string, string> };
        };
        expect(input.custom_id).toBe('page-7');
        expect(input.request.labels).toEqual({ custom_id: 'page-7' });
    });

    it('toRestGenerateContentRequest nests generation params under generationConfig and prunes undefined', () => {
        const params = {
            model: 'gemini-3.1-flash-lite',
            contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
            config: {
                systemInstruction: { parts: [{ text: 'sys' }] },
                temperature: 0.2,
                maxOutputTokens: 1024,
                responseMimeType: 'application/json',
                safetySettings: [{ category: 'X', threshold: 'Y' }],
            },
        } as never;
        const req = toRestGenerateContentRequest(params);
        expect(req.contents).toEqual([{ role: 'user', parts: [{ text: 'hi' }] }]);
        expect(req.systemInstruction).toEqual({ parts: [{ text: 'sys' }] });
        expect(req.safetySettings).toEqual([{ category: 'X', threshold: 'Y' }]);
        expect(req.generationConfig).toEqual({
            temperature: 0.2,
            maxOutputTokens: 1024,
            responseMimeType: 'application/json',
        });
        expect('topK' in (req.generationConfig as object)).toBe(false);
    });

    it('parseBatchOutputLine extracts custom_id, text and token usage', () => {
        const line = JSON.stringify({
            custom_id: 'page-3',
            request: {},
            status: '',
            response: {
                candidates: [{ content: { parts: [{ text: '# Heading\ntext' }] }, finishReason: 'STOP' }],
                usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 40, totalTokenCount: 140 },
            },
        });
        const item = parseBatchOutputLine(line, 0);
        expect(item?.custom_id).toBe('page-3');
        expect(item?.result).toEqual([{ type: 'text', value: '# Heading\ntext' }]);
        expect(item?.token_usage).toEqual({ total: 140, prompt: 100, result: 40 });
        expect(item?.finish_reason).toBe('STOP');
    });

    it('parseBatchOutputLine falls back to index custom_id and reports errors', () => {
        const item = parseBatchOutputLine(JSON.stringify({ request: {}, status: 'RESOURCE_EXHAUSTED' }), 7);
        expect(item?.custom_id).toBe('7');
        expect(item?.error).toBe('RESOURCE_EXHAUSTED');
    });

    it('parseBatchOutputLine reports a structured item error even when Vertex emits an empty response', () => {
        const item = parseBatchOutputLine(
            JSON.stringify({
                custom_id: 'image-1',
                response: {},
                status: JSON.stringify({ code: 7, message: 'storage.objects.get denied' }),
            }),
            0,
        );
        expect(item).toEqual({ custom_id: 'image-1', error: 'storage.objects.get denied' });
    });

    it('parseBatchOutputLine ignores blank/invalid lines', () => {
        expect(parseBatchOutputLine('   ', 0)).toBeUndefined();
        expect(parseBatchOutputLine('not json', 0)).toBeUndefined();
    });

    it('startBatchInference serializes signed media URLs without reading their bytes', async () => {
        const getStream = vi.fn().mockResolvedValue(new ReadableStream());
        const getURL = vi.fn().mockResolvedValue('https://signed.example/image.jpg?signature=redacted');
        const source: DataSource = {
            name: 'image.jpg',
            mime_type: 'image/jpeg',
            getStream,
            getURL,
            getURI: vi.fn().mockResolvedValue('gs://tenant-bucket/image.jpg'),
        };
        const request: BatchInferenceRequestItem = {
            custom_id: 'image-1',
            segments: [{ role: PromptRole.user, content: 'Describe it.', files: [source] }],
            options: {
                model: 'publishers/google/models/gemini-2.5-flash',
            } as ExecutionOptions,
        };
        const putText = vi.fn().mockResolvedValue('gs://tenant-bucket/batch/input.jsonl');
        const blobStore: BatchBlobStore = {
            putText,
            readOutput: vi.fn().mockResolvedValue([]),
        };
        const driver = new VertexAIDriver({ project: 'test-project', region: 'us-central1' });
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue({
            batches: {
                create: vi.fn().mockResolvedValue({ name: 'batch-1', state: 'JOB_STATE_PENDING' }),
            },
        } as unknown as ReturnType<VertexAIDriver['getGoogleGenAIClient']>);

        await driver.startBatchInference([request], { name: 'signed-media', blobStore });

        expect(getURL).toHaveBeenCalledOnce();
        expect(getStream).not.toHaveBeenCalled();
        const input = JSON.parse(vi.mocked(putText).mock.calls[0][1]) as {
            request: { contents: Array<{ parts: Array<{ fileData?: { fileUri: string } }> }> };
        };
        expect(input.request.contents[0].parts).toContainEqual({
            fileData: {
                fileUri: 'https://signed.example/image.jpg?signature=redacted',
                mimeType: 'image/jpeg',
            },
        });
    });
});

describe('bedrock batch helpers', () => {
    it('parseS3Bucket handles s3://, bucket/prefix and bucket-only', () => {
        expect(parseS3Bucket('s3://b/p/q/')).toEqual({ bucket: 'b', prefix: 'p/q' });
        expect(parseS3Bucket('b/p')).toEqual({ bucket: 'b', prefix: 'p' });
        expect(parseS3Bucket('b')).toEqual({ bucket: 'b', prefix: '' });
        expect(parseS3Bucket(`s3://b/prefix${manyTrailingSlashes}`)).toEqual({
            bucket: 'b',
            prefix: 'prefix',
        });
    });

    it('mapModelInvocationJobStatus maps Bedrock statuses', () => {
        expect(mapModelInvocationJobStatus('Completed')).toBe(BatchInferenceJobStatus.succeeded);
        expect(mapModelInvocationJobStatus('PartiallyCompleted')).toBe(BatchInferenceJobStatus.succeeded);
        expect(mapModelInvocationJobStatus('Failed')).toBe(BatchInferenceJobStatus.failed);
        expect(mapModelInvocationJobStatus('Stopped')).toBe(BatchInferenceJobStatus.cancelled);
        expect(mapModelInvocationJobStatus('Submitted')).toBe(BatchInferenceJobStatus.queued);
        expect(mapModelInvocationJobStatus('InProgress')).toBe(BatchInferenceJobStatus.running);
    });

    it('parseNativeModelOutput handles the Anthropic Messages shape', () => {
        const out = parseNativeModelOutput({
            content: [{ type: 'text', text: 'hello' }],
            usage: { input_tokens: 12, output_tokens: 5 },
            stop_reason: 'end_turn',
        });
        expect(out.result).toEqual([{ type: 'text', value: 'hello' }]);
        expect(out.token_usage).toEqual({ prompt: 12, result: 5, total: 17 });
        expect(out.finish_reason).toBe('end_turn');
    });

    it('parseNativeModelOutput handles the Amazon Nova shape', () => {
        const out = parseNativeModelOutput({
            output: { message: { content: [{ text: 'nova out' }] } },
            usage: { inputTokens: 20, outputTokens: 8, totalTokens: 28 },
            stopReason: 'end_turn',
        });
        expect(out.result).toEqual([{ type: 'text', value: 'nova out' }]);
        expect(out.token_usage).toEqual({ prompt: 20, result: 8, total: 28 });
    });

    it('parseBedrockBatchOutputLine maps recordId to custom_id', () => {
        const line = JSON.stringify({
            recordId: 'p1',
            modelOutput: { content: [{ text: 'x' }], usage: { input_tokens: 1, output_tokens: 1 } },
        });
        const item = parseBedrockBatchOutputLine(line, 0);
        expect(item?.custom_id).toBe('p1');
        expect(item?.result).toEqual([{ type: 'text', value: 'x' }]);
    });

    it('isBedrockBatchOutputKey excludes the manifest job-summary file', () => {
        expect(isBedrockBatchOutputKey('run/output/input.jsonl.out')).toBe(true);
        expect(isBedrockBatchOutputKey('run/output/records.jsonl')).toBe(true);
        expect(isBedrockBatchOutputKey('run/output/manifest.json.out')).toBe(false);
        expect(isBedrockBatchOutputKey('run/output/results.csv')).toBe(false);
    });

    it('s3List follows ContinuationToken pagination', async () => {
        const send = vi
            .fn()
            .mockResolvedValueOnce({
                Contents: [{ Key: 'a' }, { Key: 'b' }],
                IsTruncated: true,
                NextContinuationToken: 'tok-2',
            })
            .mockResolvedValueOnce({ Contents: [{ Key: 'c' }], IsTruncated: false });
        const s3 = { send } as unknown as S3Client;
        const keys = await s3List(s3, 'bucket', 'prefix');
        expect(keys).toEqual(['a', 'b', 'c']);
        expect(send).toHaveBeenCalledTimes(2);
        expect(vi.mocked(send).mock.calls[1][0].input).toMatchObject({ ContinuationToken: 'tok-2' });
    });

    it('supportsBatchInference is gated off until native modelInput mapping lands', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        expect(driver.supportsBatchInference()).toBe(false);
    });

    it('startBatchInference rejects an injected blobStore', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const blobStore: BatchBlobStore = {
            putText: vi.fn().mockResolvedValue('s3://b/k'),
            readOutput: vi.fn().mockResolvedValue([]),
        };
        const item: BatchInferenceRequestItem = {
            custom_id: 'a',
            segments: [{ role: PromptRole.user, content: 'hi' }],
            options: { model: 'anthropic.claude-3-haiku' } as ExecutionOptions,
        };
        await expect(driver.startBatchInference([item], { blobStore })).rejects.toThrow(/blobStore/);
    });
});

describe('batch inference limits', () => {
    it('drivers report provider-documented job limits', () => {
        const vertex = new VertexAIDriver({ project: 'p', region: 'us-central1' });
        expect(vertex.getBatchInferenceLimits()).toEqual({ max_requests_per_job: 200_000, max_concurrent_jobs: 75 });

        const openai = new OpenAIDriver({ apiKey: 'test-key' });
        expect(openai.getBatchInferenceLimits()).toEqual({
            max_requests_per_job: 50_000,
            max_input_bytes: 200 * 1024 * 1024,
        });

        const bedrock = new BedrockDriver({ region: 'us-east-1' });
        expect(bedrock.getBatchInferenceLimits()).toEqual({ max_requests_per_job: 50_000, min_requests_per_job: 100 });
    });

    it('drivers without a batch implementation report the conservative default', () => {
        expect(new TestDriver().getBatchInferenceLimits()).toEqual({ max_requests_per_job: 10_000 });
    });
});
