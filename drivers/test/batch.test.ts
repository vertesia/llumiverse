import { BatchInferenceJobStatus } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import {
    mapModelInvocationJobStatus,
    parseBedrockBatchOutputLine,
    parseNativeModelOutput,
    parseS3Bucket,
} from '../src/bedrock/batch.js';
import {
    mapBatchJobState,
    parseBatchOutputLine,
    parseGcsBucket,
    toRestGenerateContentRequest,
} from '../src/vertexai/batch.js';

describe('vertex batch helpers', () => {
    it('parseGcsBucket handles gs://, bucket/prefix and bucket-only', () => {
        expect(parseGcsBucket('gs://my-bucket/pre/fix/')).toEqual({ bucket: 'my-bucket', prefix: 'pre/fix' });
        expect(parseGcsBucket('my-bucket/pre')).toEqual({ bucket: 'my-bucket', prefix: 'pre' });
        expect(parseGcsBucket('my-bucket')).toEqual({ bucket: 'my-bucket', prefix: '' });
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

    it('parseBatchOutputLine ignores blank/invalid lines', () => {
        expect(parseBatchOutputLine('   ', 0)).toBeUndefined();
        expect(parseBatchOutputLine('not json', 0)).toBeUndefined();
    });
});

describe('bedrock batch helpers', () => {
    it('parseS3Bucket handles s3://, bucket/prefix and bucket-only', () => {
        expect(parseS3Bucket('s3://b/p/q/')).toEqual({ bucket: 'b', prefix: 'p/q' });
        expect(parseS3Bucket('b/p')).toEqual({ bucket: 'b', prefix: 'p' });
        expect(parseS3Bucket('b')).toEqual({ bucket: 'b', prefix: '' });
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
});
