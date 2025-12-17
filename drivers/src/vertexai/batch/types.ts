/**
 * VertexAI-specific batch types.
 * These types are used internally to interface with the Google Cloud APIs.
 */

import { BatchJobStatus } from "@llumiverse/common";

// ============== Claude Batch REST API Types ===============

/**
 * Request body for creating a Claude batch prediction job.
 * Used with the aiplatform REST API: POST /batchPredictionJobs
 */
export interface ClaudeBatchJobRequest {
    /** Display name for the job */
    displayName: string;
    /** Model resource path (e.g., "publishers/anthropic/models/claude-3-5-haiku") */
    model: string;
    /** Input configuration */
    inputConfig: {
        /** Instances format - always "jsonl" for Claude */
        instancesFormat?: string;
        /** GCS source configuration */
        gcsSource?: {
            /** List of GCS URIs to input JSONL files */
            uris: string[];
        };
        /** BigQuery source configuration */
        bigquerySource?: {
            /** BigQuery table URI */
            inputUri: string;
        };
    };
    /** Output configuration */
    outputConfig: {
        /** Predictions format - always "jsonl" for Claude */
        predictionsFormat?: string;
        /** GCS destination configuration */
        gcsDestination?: {
            /** GCS URI prefix for output */
            outputUriPrefix: string;
        };
        /** BigQuery destination configuration */
        bigqueryDestination?: {
            /** BigQuery table URI */
            outputUri: string;
        };
    };
    /** Model parameters (optional) */
    modelParameters?: Record<string, unknown>;
}

/**
 * Response from Claude batch prediction job creation/get.
 */
export interface ClaudeBatchJobResponse {
    /** Resource name (e.g., "projects/.../locations/.../batchPredictionJobs/123") */
    name: string;
    /** Display name */
    displayName: string;
    /** Job state */
    state: string;
    /** Model resource path */
    model?: string;
    /** Input configuration */
    inputConfig?: {
        instancesFormat?: string;
        gcsSource?: { uris: string[] };
        bigquerySource?: { inputUri: string };
    };
    /** Output configuration */
    outputConfig?: {
        predictionsFormat?: string;
        gcsDestination?: { outputUriPrefix: string };
        bigqueryDestination?: { outputUri: string };
    };
    /** Creation timestamp (ISO 8601) */
    createTime: string;
    /** Start timestamp (ISO 8601) */
    startTime?: string;
    /** End timestamp (ISO 8601) */
    endTime?: string;
    /** Update timestamp (ISO 8601) */
    updateTime?: string;
    /** Error information if failed */
    error?: {
        code: number;
        message: string;
        details?: unknown[];
    };
    /** Completion statistics */
    completionStats?: {
        successfulCount: string;
        failedCount: string;
        incompleteCount?: string;
    };
}

// ============== Claude Input Record Format ===============

/**
 * Input record format for Claude batch JSONL files.
 * Each line in the input file should follow this format.
 */
export interface ClaudeBatchInputRecord {
    /** Unique identifier for matching input to output */
    custom_id: string;
    /** The Claude API request */
    request: {
        /** Messages for the conversation */
        messages: Array<{
            role: "user" | "assistant";
            content: string | Array<{ type: string; text?: string; source?: unknown }>;
        }>;
        /** Anthropic API version (e.g., "vertex-2023-10-16") */
        anthropic_version: string;
        /** Maximum tokens to generate */
        max_tokens: number;
        /** Model name (optional, overrides job-level model) */
        model?: string;
        /** System prompt (optional) */
        system?: string;
        /** Temperature (optional) */
        temperature?: number;
        /** Top-p (optional) */
        top_p?: number;
        /** Stop sequences (optional) */
        stop_sequences?: string[];
    };
}

// ============== Job State Mapping ===============

/**
 * Claude/aiplatform job states
 */
export type ClaudeJobState =
    | "JOB_STATE_UNSPECIFIED"
    | "JOB_STATE_QUEUED"
    | "JOB_STATE_PENDING"
    | "JOB_STATE_RUNNING"
    | "JOB_STATE_SUCCEEDED"
    | "JOB_STATE_FAILED"
    | "JOB_STATE_CANCELLING"
    | "JOB_STATE_CANCELLED"
    | "JOB_STATE_PAUSED"
    | "JOB_STATE_EXPIRED"
    | "JOB_STATE_UPDATING"
    | "JOB_STATE_PARTIALLY_SUCCEEDED";

/**
 * Maps Claude/aiplatform job state to unified BatchJobStatus.
 */
export function mapClaudeJobState(state: string): BatchJobStatus {
    switch (state) {
        case "JOB_STATE_QUEUED":
        case "JOB_STATE_PENDING":
            return BatchJobStatus.pending;
        case "JOB_STATE_RUNNING":
        case "JOB_STATE_UPDATING":
            return BatchJobStatus.running;
        case "JOB_STATE_SUCCEEDED":
            return BatchJobStatus.succeeded;
        case "JOB_STATE_FAILED":
        case "JOB_STATE_EXPIRED":
            return BatchJobStatus.failed;
        case "JOB_STATE_CANCELLING":
        case "JOB_STATE_CANCELLED":
            return BatchJobStatus.cancelled;
        case "JOB_STATE_PARTIALLY_SUCCEEDED":
            return BatchJobStatus.partial;
        default:
            return BatchJobStatus.pending;
    }
}

/**
 * Maps Google GenAI SDK JobState to unified BatchJobStatus.
 * The SDK uses the same state names as the REST API.
 */
export function mapGeminiJobState(state: string | undefined): BatchJobStatus {
    if (!state) return BatchJobStatus.pending;
    return mapClaudeJobState(state);
}

// ============== Batch Job ID Encoding ===============

/**
 * Batch provider type for routing.
 */
export type BatchProvider = "gemini" | "claude" | "embeddings";

/**
 * Encodes a batch job ID with provider prefix for routing.
 * Format: "provider:providerJobId"
 */
export function encodeBatchJobId(provider: BatchProvider, providerJobId: string): string {
    return `${provider}:${providerJobId}`;
}

/**
 * Decodes a batch job ID to extract provider and original ID.
 */
export function decodeBatchJobId(jobId: string): { provider: BatchProvider; providerJobId: string } {
    const colonIndex = jobId.indexOf(":");
    if (colonIndex === -1) {
        // Assume gemini if no prefix (backwards compatibility)
        return { provider: "gemini", providerJobId: jobId };
    }
    const provider = jobId.substring(0, colonIndex) as BatchProvider;
    const providerJobId = jobId.substring(colonIndex + 1);
    return { provider, providerJobId };
}
