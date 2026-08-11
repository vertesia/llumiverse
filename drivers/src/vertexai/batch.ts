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
import { parseBatchBucketLocation } from '../batch-location.js';

// ---------------------------------------------------------------------------
// GCS location parsing + I/O via the auth client (no @google-cloud/storage dep)
// ---------------------------------------------------------------------------

export interface GcsLocation {
    bucket: string;
    prefix: string;
}

export interface VertexModelTarget {
    model: string;
    location?: string;
}

/**
 * Convert a catalog/resource model id into the form expected by the Google Gen AI
 * batch client. Unlike generateContent, batches.create does not reliably accept the
 * Vertesia catalog form (`locations/<region>/publishers/google/models/<model>`).
 * The location belongs on the client and the request receives the bare model id.
 */
export function parseVertexModelTarget(resourceName: string): VertexModelTarget {
    const parts = resourceName.split('/').filter(Boolean);
    const modelIndex = parts.lastIndexOf('models');
    const locationIndex = parts.lastIndexOf('locations');
    return {
        model:
            modelIndex >= 0 && modelIndex + 1 < parts.length ? parts[modelIndex + 1] : (parts.at(-1) ?? resourceName),
        location: locationIndex >= 0 && locationIndex + 1 < parts.length ? parts[locationIndex + 1] : undefined,
    };
}

/** Extract the location embedded in a Vertex resource name such as a batch job id. */
export function parseVertexResourceLocation(resourceName: string): string | undefined {
    const parts = resourceName.split('/').filter(Boolean);
    const locationIndex = parts.lastIndexOf('locations');
    return locationIndex >= 0 && locationIndex + 1 < parts.length ? parts[locationIndex + 1] : undefined;
}

/** Parse "gs://bucket/prefix", "bucket/prefix" or "bucket" into {bucket, prefix}. */
export function parseGcsBucket(spec: string): GcsLocation {
    return parseBatchBucketLocation(spec, 'gs');
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
    const status = obj.status ?? (obj.error as { message?: string } | undefined)?.message;
    const hasErrorStatus = typeof status === 'string' ? status.trim().length > 0 : status != null;
    if (hasErrorStatus) {
        let message = typeof status === 'string' ? status : JSON.stringify(status);
        try {
            const parsedStatus = typeof status === 'string' ? (JSON.parse(status) as { message?: unknown }) : status;
            if (
                parsedStatus &&
                typeof parsedStatus === 'object' &&
                'message' in parsedStatus &&
                typeof parsedStatus.message === 'string'
            ) {
                message = parsedStatus.message;
            }
        } catch {
            // Plain-text provider statuses are already useful as-is.
        }
        return { custom_id, error: message };
    }
    const response = obj.response as GenerateContentResponse | undefined;
    if (response) {
        return { custom_id, ...parseGeminiBatchResponse(response) };
    }
    return { custom_id, error: 'no response in batch output' };
}
