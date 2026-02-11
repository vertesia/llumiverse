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

// ============== Gemini API HTTP Types ===============

/**
 * Gemini API batch state enum values.
 * Used by the Generative Language API (not Vertex AI).
 */
export type GeminiBatchState =
    | "BATCH_STATE_UNSPECIFIED"
    | "BATCH_STATE_PENDING"
    | "BATCH_STATE_RUNNING"
    | "BATCH_STATE_SUCCEEDED"
    | "BATCH_STATE_FAILED"
    | "BATCH_STATE_CANCELLED"
    | "BATCH_STATE_EXPIRED";

/**
 * Maps Gemini API batch state to unified BatchJobStatus.
 */
export function mapGeminiBatchState(state: string | undefined): BatchJobStatus {
    if (!state) return BatchJobStatus.pending;
    switch (state) {
        case "BATCH_STATE_PENDING":
            return BatchJobStatus.pending;
        case "BATCH_STATE_RUNNING":
            return BatchJobStatus.running;
        case "BATCH_STATE_SUCCEEDED":
            return BatchJobStatus.succeeded;
        case "BATCH_STATE_FAILED":
        case "BATCH_STATE_EXPIRED":
            return BatchJobStatus.failed;
        case "BATCH_STATE_CANCELLED":
            return BatchJobStatus.cancelled;
        default:
            return BatchJobStatus.pending;
    }
}

/**
 * Batch stats from Gemini API response.
 */
export interface GeminiBatchStats {
    /** Total number of requests in the batch */
    requestCount?: number;
    /** Number of successfully completed requests */
    successfulRequestCount?: number;
    /** Number of failed requests */
    failedRequestCount?: number;
    /** Number of pending requests */
    pendingRequestCount?: number;
}

/**
 * Input configuration for Gemini API batch.
 * Either fileName or requests must be provided.
 */
export interface GeminiBatchInputConfig {
    /** Reference to a File containing requests (format: files/{fileId}) */
    fileName?: string;
    /** Inline requests container for batch processing */
    requests?: GeminiInlinedEmbedRequestsContainer;
}

/**
 * Container for inline embed content requests.
 * Wraps the array of individual requests.
 */
export interface GeminiInlinedEmbedRequestsContainer {
    /** Array of inline embed content requests */
    requests: GeminiInlinedEmbedRequestWrapper[];
}

/**
 * Wrapper for a single inline embed content request.
 */
export interface GeminiInlinedEmbedRequestWrapper {
    /** The actual embed content request */
    request: GeminiEmbedContentRequest;
    /** Optional metadata associated with this request */
    metadata?: Record<string, unknown>;
}

/**
 * Embed content request for batch embeddings.
 */
export interface GeminiEmbedContentRequest {
    /** Model to use (format: models/{model}) */
    model?: string;
    /** Content to embed */
    content: {
        parts: Array<{ text: string }>;
    };
    /** Task type for embeddings */
    taskType?: string;
    /** Title for document embeddings */
    title?: string;
    /** Output dimensionality */
    outputDimensionality?: number;
}

/**
 * Output configuration for Gemini API batch.
 */
export interface GeminiBatchOutputConfig {
    /** File ID containing JSONL responses */
    responsesFile?: string;
    /** Inline responses (when input was inlined) */
    inlinedResponses?: Array<{
        embedding?: {
            values: number[];
        };
        error?: {
            code: number;
            message: string;
        };
    }>;
}

/**
 * Gemini API batch resource returned by get/list operations.
 */
export interface GeminiBatchResource {
    /** Resource name (format: batches/{batchId}) */
    name?: string;
    /** Model used for the batch (format: models/{model}) */
    model?: string;
    /** User-defined display name */
    displayName?: string;
    /** Current batch state */
    state?: GeminiBatchState;
    /** Input configuration */
    inputConfig?: GeminiBatchInputConfig;
    /** Output configuration (populated after completion) */
    output?: GeminiBatchOutputConfig;
    /** Creation timestamp (RFC 3339) */
    createTime?: string;
    /** Completion timestamp (RFC 3339) */
    endTime?: string;
    /** Last update timestamp (RFC 3339) */
    updateTime?: string;
    /** Batch statistics */
    batchStats?: GeminiBatchStats;
    /** Priority value (higher = earlier processing) */
    priority?: number;
    /** Error information if failed */
    error?: {
        code: number;
        message: string;
        details?: unknown[];
    };
}

/**
 * Request body for asyncBatchEmbedContent endpoint.
 */
export interface GeminiAsyncBatchEmbedRequest {
    batch: {
        /** Model to use (format: models/{model}) */
        model: string;
        /** User-defined display name */
        displayName: string;
        /** Input configuration */
        inputConfig: GeminiBatchInputConfig;
        /** Priority value (optional, default 0) */
        priority?: number;
    };
}

/**
 * Operation response from async Gemini API methods.
 */
export interface GeminiOperationResponse {
    /** Server-assigned operation name */
    name?: string;
    /** Service-specific metadata */
    metadata?: Record<string, unknown>;
    /** Whether the operation is complete */
    done?: boolean;
    /** Error if operation failed */
    error?: {
        code: number;
        message: string;
        details?: unknown[];
    };
    /** Successful response (the batch resource) */
    response?: GeminiBatchResource;
}

/**
 * List batches response from Gemini API.
 */
export interface GeminiListBatchesResponse {
    /** List of batch resources */
    batches?: GeminiBatchResource[];
    /** Token for next page */
    nextPageToken?: string;
}

// ============== Gemini File API Types ===============

/**
 * State of a file in the Gemini File API.
 */
export type GeminiFileState = "STATE_UNSPECIFIED" | "PROCESSING" | "ACTIVE" | "FAILED";

/**
 * File resource from the Gemini File API.
 * Represents an uploaded file that can be used in batch operations.
 */
export interface GeminiFileResource {
    /** Resource name (format: files/{fileId}) */
    name: string;
    /** User-defined display name */
    displayName?: string;
    /** MIME type of the file */
    mimeType: string;
    /** Pre-signed download URL for the file contents */
    downloadUri?: string;
    /** Size of the file in bytes (as string) */
    sizeBytes: string;
    /** Creation timestamp (RFC 3339) */
    createTime: string;
    /** Last update timestamp (RFC 3339) */
    updateTime: string;
    /** Expiration timestamp (RFC 3339) - files are automatically deleted after this time */
    expirationTime?: string;
    /** SHA-256 hash of the file content */
    sha256Hash?: string;
    /** Full URI for accessing the file */
    uri: string;
    /** Current file state */
    state: GeminiFileState;
    /** Source of the file resource (e.g., FILE_SOURCE_STORAGE) */
    source?: string;
    /** Error information if processing failed */
    error?: {
        code: number;
        message: string;
    };
}

/**
 * Response from uploading a file to the Gemini File API.
 */
export interface GeminiUploadFileResponse {
    /** The uploaded file resource */
    file: GeminiFileResource;
}

/**
 * Response from listing files in the Gemini File API.
 */
export interface GeminiListFilesResponse {
    /** List of file resources */
    files?: GeminiFileResource[];
    /** Token for next page */
    nextPageToken?: string;
}
