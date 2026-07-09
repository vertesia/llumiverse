/**
 * Batch-inference helpers for the Bedrock driver.
 *
 * Uses Bedrock's asynchronous batch inference (`CreateModelInvocationJob`) with
 * S3-staged JSONL input/output. Each line is `{ recordId, modelInput }`; the output
 * is `{ recordId, modelOutput }`, so `recordId` carries the caller's `custom_id`.
 *
 * NOTE: Bedrock batch uses the model-native **InvokeModel** body for `modelInput`
 * (NOT the Converse API shape the driver uses for synchronous calls). This module
 * parses the common native response shapes (Anthropic Messages, Amazon Nova). Full
 * parity for every model family (Converse → native request mapping) is a follow-up.
 * Bedrock batch also requires a minimum number of records per job and an IAM role
 * with S3 access — see the driver methods.
 */

import { GetObjectCommand, ListObjectsV2Command, PutObjectCommand, type S3Client } from '@aws-sdk/client-s3';
import {
    BatchInferenceJobStatus,
    type BatchInferenceResultItem,
    type CompletionResult,
    type ExecutionTokenUsage,
} from '@llumiverse/common';

// ---------------------------------------------------------------------------
// S3 text I/O
// ---------------------------------------------------------------------------

export interface S3Location {
    bucket: string;
    prefix: string;
}

/** Parse "s3://bucket/prefix", "bucket/prefix" or "bucket" into {bucket, prefix}. */
export function parseS3Bucket(spec: string): S3Location {
    let s = spec.trim();
    if (s.startsWith('s3://')) {
        s = s.slice('s3://'.length);
    }
    const slash = s.indexOf('/');
    if (slash === -1) {
        return { bucket: s, prefix: '' };
    }
    return { bucket: s.slice(0, slash), prefix: s.slice(slash + 1).replace(/\/+$/, '') };
}

export async function s3UploadText(s3: S3Client, bucket: string, key: string, text: string): Promise<string> {
    await s3.send(new PutObjectCommand({ Bucket: bucket, Key: key, Body: text, ContentType: 'application/jsonl' }));
    return `s3://${bucket}/${key}`;
}

export async function s3DownloadText(s3: S3Client, bucket: string, key: string): Promise<string> {
    const res = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    return (await res.Body?.transformToString()) ?? '';
}

export async function s3List(s3: S3Client, bucket: string, prefix: string): Promise<string[]> {
    const res = await s3.send(new ListObjectsV2Command({ Bucket: bucket, Prefix: prefix }));
    return (res.Contents ?? []).map((o) => o.Key ?? '').filter(Boolean);
}

// ---------------------------------------------------------------------------
// Status + response mapping
// ---------------------------------------------------------------------------

/** Map a Bedrock ModelInvocationJobStatus to our provider-agnostic batch status. */
export function mapModelInvocationJobStatus(status: string | undefined): BatchInferenceJobStatus {
    switch (status) {
        case 'Completed':
            return BatchInferenceJobStatus.succeeded;
        case 'PartiallyCompleted':
            return BatchInferenceJobStatus.succeeded;
        case 'Failed':
        case 'Expired':
            return BatchInferenceJobStatus.failed;
        case 'Stopped':
        case 'Stopping':
            return BatchInferenceJobStatus.cancelled;
        case 'Submitted':
        case 'Validating':
        case 'Scheduled':
            return BatchInferenceJobStatus.queued;
        default:
            // InProgress or anything new
            return BatchInferenceJobStatus.running;
    }
}

interface NativeModelOutput {
    // Anthropic Messages
    content?: Array<{ type?: string; text?: string }>;
    usage?: {
        input_tokens?: number;
        output_tokens?: number;
        inputTokens?: number;
        outputTokens?: number;
        totalTokens?: number;
    };
    stop_reason?: string;
    // Amazon Nova / Converse-style
    output?: { message?: { content?: Array<{ text?: string }> } };
    stopReason?: string;
    // Amazon Titan
    results?: Array<{ outputText?: string; completionReason?: string }>;
}

/** Extract text + token usage from a native Bedrock modelOutput (Anthropic / Nova / Titan shapes). */
export function parseNativeModelOutput(modelOutput: NativeModelOutput | undefined): {
    result: CompletionResult[];
    token_usage?: ExecutionTokenUsage;
    finish_reason?: string;
} {
    const result: CompletionResult[] = [];
    let finish_reason: string | undefined;

    if (Array.isArray(modelOutput?.content)) {
        // Anthropic Messages
        for (const c of modelOutput.content) {
            if (typeof c.text === 'string' && c.text.length > 0) {
                result.push({ type: 'text', value: c.text });
            }
        }
        finish_reason = modelOutput.stop_reason;
    } else if (modelOutput?.output?.message?.content) {
        // Amazon Nova / Converse-style native output
        for (const c of modelOutput.output.message.content) {
            if (typeof c.text === 'string' && c.text.length > 0) {
                result.push({ type: 'text', value: c.text });
            }
        }
        finish_reason = modelOutput.stopReason;
    } else if (Array.isArray(modelOutput?.results)) {
        // Amazon Titan
        for (const r of modelOutput.results) {
            if (typeof r.outputText === 'string' && r.outputText.length > 0) {
                result.push({ type: 'text', value: r.outputText });
            }
        }
        finish_reason = modelOutput.results[0]?.completionReason;
    }

    const u = modelOutput?.usage;
    const prompt = u?.input_tokens ?? u?.inputTokens;
    const output = u?.output_tokens ?? u?.outputTokens;
    const token_usage: ExecutionTokenUsage | undefined =
        prompt == null && output == null
            ? undefined
            : { prompt, result: output, total: u?.totalTokens ?? (prompt ?? 0) + (output ?? 0) };

    return { result, token_usage, finish_reason };
}

/** Parse a single Bedrock batch output JSONL line (`{ recordId, modelOutput }`). */
export function parseBedrockBatchOutputLine(line: string, index: number): BatchInferenceResultItem | undefined {
    const trimmed = line.trim();
    if (!trimmed) {
        return undefined;
    }
    let obj: { recordId?: string; modelOutput?: NativeModelOutput; error?: unknown };
    try {
        obj = JSON.parse(trimmed);
    } catch {
        return undefined;
    }
    const custom_id = obj.recordId ?? String(index);
    if (obj.modelOutput) {
        return { custom_id, ...parseNativeModelOutput(obj.modelOutput) };
    }
    return { custom_id, error: obj.error ? JSON.stringify(obj.error) : 'no modelOutput in batch record' };
}
