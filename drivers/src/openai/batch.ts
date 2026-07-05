/**
 * Batch-inference helpers for the OpenAI (Responses API) driver.
 *
 * Uses the OpenAI Batch API: a JSONL of `{ custom_id, method, url, body }` lines is
 * uploaded via the Files API, submitted with `batches.create`, polled, and the output
 * file is downloaded and parsed. Each `body` is built by the driver's shared
 * `buildResponsesBody`, so a batch request is identical to the synchronous request.
 */

import {
    BatchInferenceJobStatus,
    type BatchInferenceResultItem,
    type CompletionResult,
    type ExecutionTokenUsage,
} from '@llumiverse/common';

/** Map an OpenAI batch status to our provider-agnostic batch status. */
export function mapOpenAIBatchStatus(status: string | undefined): BatchInferenceJobStatus {
    switch (status) {
        case 'completed':
            return BatchInferenceJobStatus.succeeded;
        case 'failed':
        case 'expired':
            return BatchInferenceJobStatus.failed;
        case 'cancelled':
        case 'cancelling':
            return BatchInferenceJobStatus.cancelled;
        case 'validating':
            return BatchInferenceJobStatus.queued;
        default:
            // in_progress, finalizing, or anything new
            return BatchInferenceJobStatus.running;
    }
}

interface ResponsesBody {
    output?: Array<{ type?: string; content?: Array<{ type?: string; text?: string }> }>;
    output_text?: string;
    usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
}

/** Extract text + token usage from a Responses API output body. */
export function parseResponsesBody(body: ResponsesBody | undefined): {
    result: CompletionResult[];
    token_usage?: ExecutionTokenUsage;
} {
    const result: CompletionResult[] = [];
    if (typeof body?.output_text === 'string' && body.output_text.length > 0) {
        result.push({ type: 'text', value: body.output_text });
    } else if (Array.isArray(body?.output)) {
        for (const item of body.output) {
            if (item.type === 'message' && Array.isArray(item.content)) {
                for (const c of item.content) {
                    if (c.type === 'output_text' && typeof c.text === 'string' && c.text.length > 0) {
                        result.push({ type: 'text', value: c.text });
                    }
                }
            }
        }
    }
    const u = body?.usage;
    const token_usage: ExecutionTokenUsage | undefined = u
        ? { prompt: u.input_tokens, result: u.output_tokens, total: u.total_tokens }
        : undefined;
    return { result, token_usage };
}

/** Parse a single OpenAI batch output JSONL line (`{ custom_id, response: { body }, error }`). */
export function parseOpenAIBatchOutputLine(line: string, index: number): BatchInferenceResultItem | undefined {
    const trimmed = line.trim();
    if (!trimmed) {
        return undefined;
    }
    let obj: { custom_id?: string; response?: { status_code?: number; body?: ResponsesBody }; error?: unknown };
    try {
        obj = JSON.parse(trimmed);
    } catch {
        return undefined;
    }
    const custom_id = obj.custom_id ?? String(index);
    if (obj.error) {
        return { custom_id, error: typeof obj.error === 'string' ? obj.error : JSON.stringify(obj.error) };
    }
    if (obj.response?.body) {
        return { custom_id, ...parseResponsesBody(obj.response.body) };
    }
    return { custom_id, error: 'no response body in batch output' };
}
