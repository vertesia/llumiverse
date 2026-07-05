/**
 * Batch-inference helpers for the Vertex AI driver.
 *
 * Uses Vertex's asynchronous batch prediction for Gemini (via the `@google/genai`
 * `batches` API) with GCS-staged input/output. Each request is formatted with the
 * driver's normal `createPrompt` + `getGeminiPayload`, so a batch line is identical
 * to the synchronous request for the same model — batch output matches interactive
 * output. GCS I/O goes through the driver's existing GoogleAuth client, so no extra
 * dependency (`@google-cloud/storage`) is required.
 */

import type { GenerateContentParameters, GenerateContentResponse } from '@google/genai';
import {
    BatchInferenceJobStatus,
    type BatchInferenceResultItem,
    type CompletionResult,
    type ExecutionTokenUsage,
} from '@llumiverse/common';
import type { AuthClient } from 'google-auth-library';

// ---------------------------------------------------------------------------
// GCS location parsing + I/O via the auth client (no @google-cloud/storage dep)
// ---------------------------------------------------------------------------

export interface GcsLocation {
    bucket: string;
    prefix: string;
}

/** Parse "gs://bucket/prefix", "bucket/prefix" or "bucket" into {bucket, prefix}. */
export function parseGcsBucket(spec: string): GcsLocation {
    let s = spec.trim();
    if (s.startsWith('gs://')) {
        s = s.slice('gs://'.length);
    }
    const slash = s.indexOf('/');
    if (slash === -1) {
        return { bucket: s, prefix: '' };
    }
    return { bucket: s.slice(0, slash), prefix: s.slice(slash + 1).replace(/\/+$/, '') };
}

export async function gcsUploadText(
    auth: AuthClient,
    bucket: string,
    name: string,
    text: string,
    contentType = 'application/x-ndjson',
): Promise<string> {
    const url = `https://storage.googleapis.com/upload/storage/v1/b/${encodeURIComponent(
        bucket,
    )}/o?uploadType=media&name=${encodeURIComponent(name)}`;
    await auth.request({ url, method: 'POST', headers: { 'Content-Type': contentType }, data: text });
    return `gs://${bucket}/${name}`;
}

export async function gcsDownloadText(auth: AuthClient, bucket: string, name: string): Promise<string> {
    const url = `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(bucket)}/o/${encodeURIComponent(
        name,
    )}?alt=media`;
    const res = await auth.request({ url, method: 'GET', responseType: 'text' });
    return typeof res.data === 'string' ? res.data : JSON.stringify(res.data);
}

export async function gcsList(auth: AuthClient, bucket: string, prefix: string): Promise<string[]> {
    const url = `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(
        bucket,
    )}/o?prefix=${encodeURIComponent(prefix)}`;
    const res = await auth.request({ url, method: 'GET' });
    const data = res.data as { items?: Array<{ name: string }> };
    return (data.items ?? []).map((i) => i.name);
}

// ---------------------------------------------------------------------------
// Request / response mapping
// ---------------------------------------------------------------------------

/** Map a Vertex JobState string to our provider-agnostic batch status. */
export function mapBatchJobState(state: string | undefined): BatchInferenceJobStatus {
    switch (state) {
        case 'JOB_STATE_SUCCEEDED':
            return BatchInferenceJobStatus.succeeded;
        case 'JOB_STATE_FAILED':
        case 'JOB_STATE_EXPIRED':
            return BatchInferenceJobStatus.failed;
        case 'JOB_STATE_CANCELLED':
        case 'JOB_STATE_CANCELLING':
            return BatchInferenceJobStatus.cancelled;
        case 'JOB_STATE_PENDING':
        case 'JOB_STATE_QUEUED':
            return BatchInferenceJobStatus.queued;
        default:
            // RUNNING, PAUSED, UPDATING, UNSPECIFIED, or anything new
            return BatchInferenceJobStatus.running;
    }
}

function pruneUndefined<T extends Record<string, unknown>>(obj: T): T {
    for (const k of Object.keys(obj)) {
        if (obj[k] === undefined) {
            delete obj[k];
        }
    }
    return obj;
}

/**
 * Convert the SDK `GenerateContentParameters` ({ model, contents, config }) into the
 * Vertex REST `GenerateContentRequest` used for GCS batch JSONL input — generation
 * params are nested under `generationConfig`, everything else stays at the top level.
 */
export function toRestGenerateContentRequest(params: GenerateContentParameters): Record<string, unknown> {
    const config = (params.config ?? {}) as Record<string, unknown>;
    const generationConfig = pruneUndefined({
        temperature: config.temperature,
        topP: config.topP,
        topK: config.topK,
        candidateCount: config.candidateCount,
        maxOutputTokens: config.maxOutputTokens,
        stopSequences: config.stopSequences,
        presencePenalty: config.presencePenalty,
        frequencyPenalty: config.frequencyPenalty,
        seed: config.seed,
        responseMimeType: config.responseMimeType,
        responseJsonSchema: config.responseJsonSchema,
        responseModalities: config.responseModalities,
        thinkingConfig: config.thinkingConfig,
    });
    return pruneUndefined({
        contents: params.contents,
        systemInstruction: config.systemInstruction,
        safetySettings: config.safetySettings,
        tools: config.tools,
        toolConfig: config.toolConfig,
        labels: config.labels,
        generationConfig,
    });
}

/** Parse a Gemini batch output line's `response` into llumiverse result + token usage. */
export function parseGeminiBatchResponse(response: GenerateContentResponse | undefined): {
    result: CompletionResult[];
    token_usage?: ExecutionTokenUsage;
    finish_reason?: string;
} {
    const candidate = response?.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];
    const result: CompletionResult[] = [];
    for (const p of parts) {
        if (typeof p.text === 'string' && p.text.length > 0) {
            result.push({ type: 'text', value: p.text });
        }
    }
    const um = response?.usageMetadata;
    const token_usage: ExecutionTokenUsage | undefined =
        um?.totalTokenCount == null
            ? undefined
            : {
                  total: um.totalTokenCount,
                  prompt: um.promptTokenCount,
                  result: (um.candidatesTokenCount ?? 0) + (um.thoughtsTokenCount ?? 0),
              };
    return { result, token_usage, finish_reason: candidate?.finishReason };
}

/**
 * Parse a single output JSONL line from a Vertex Gemini batch prediction file.
 * Handles the `{ custom_id?, request, response|status }` shape; `custom_id` falls back
 * to the provided index when not echoed by the backend.
 */
export function parseBatchOutputLine(line: string, index: number): BatchInferenceResultItem | undefined {
    const trimmed = line.trim();
    if (!trimmed) {
        return undefined;
    }
    let obj: Record<string, unknown>;
    try {
        obj = JSON.parse(trimmed);
    } catch {
        return undefined;
    }
    const request = obj.request as { labels?: Record<string, string> } | undefined;
    const custom_id = (obj.custom_id as string | undefined) ?? request?.labels?.custom_id ?? String(index);
    const response = obj.response as GenerateContentResponse | undefined;
    if (response) {
        return { custom_id, ...parseGeminiBatchResponse(response) };
    }
    const status = obj.status ?? (obj.error as { message?: string } | undefined)?.message;
    return { custom_id, error: typeof status === 'string' ? status : 'no response in batch output' };
}
