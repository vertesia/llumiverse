/**
 * Provider-agnostic batch job types for async batch processing.
 * Supports batch inference and embeddings across multiple providers.
 */

import { ModelOptions, Providers } from "./types.js";

// ============== Batch Job Status ===============

/**
 * Unified batch job status (provider-agnostic).
 * Maps to provider-specific states.
 */
export enum BatchJobStatus {
    /** Job is queued or preparing to run */
    pending = "pending",
    /** Job is currently running */
    running = "running",
    /** Job completed successfully */
    succeeded = "succeeded",
    /** Job failed */
    failed = "failed",
    /** Job was cancelled */
    cancelled = "cancelled",
    /** Job partially succeeded (some requests failed) */
    partial = "partial"
}

/**
 * Type of batch job operation.
 */
export enum BatchJobType {
    /** Text/multimodal inference batch */
    inference = "inference",
    /** Embeddings generation batch */
    embeddings = "embeddings"
}

// ============== Batch Job Source/Destination ===============

/**
 * Base batch source - flexible for all providers.
 * Use this at API boundaries when provider is unknown.
 */
export interface BatchSource {
    [key: string]: any;
}

/**
 * Source configuration for batch input data.
 * Supports GCS, BigQuery, and inline sources.
 */
export interface GCSBatchSource extends BatchSource {
    /** GCS URIs for input JSONL files */
    gcsUris?: string[];
    /** BigQuery table URI for input data */
    bigqueryUri?: string;
    /** Inline requests (for testing or small batches) */
    inlinedRequests?: any[];
}

/**
 * Base batch destination - flexible for all providers.
 * Use this at API boundaries when provider is unknown.
 */
export interface BatchDestination {
    [key: string]: any;
}

/**
 * Destination configuration for batch output data.
 * Supports GCS, BigQuery, and inline destinations.
 */
export interface GCSBatchDestination extends BatchDestination {
    /** GCS URI prefix for output files */
    gcsUri?: string;
    /** BigQuery table URI for output data */
    bigqueryUri?: string;
    /** Inline destination (results returned directly, for testing) */
    inline?: boolean;
}
// ============== Batch Job Error & Stats ===============

/**
 * Error information for failed batch jobs.
 */
export interface BatchJobError {
    /** Error code */
    code: string;
    /** Human-readable error message */
    message: string;
    /** Additional error details */
    details?: string;
}

/**
 * Statistics for batch job progress and completion.
 */
export interface BatchJobStats {
    /** Total number of requests in the batch */
    totalRequests?: number;
    /** Number of successfully completed requests */
    completedRequests?: number;
    /** Number of failed requests */
    failedRequests?: number;
}

// ============== Batch Job Interface ===============

/**
 * Main batch job interface representing a batch processing job.
 * This is the unified representation returned by all batch operations.
 */
export interface BatchJob<T, U> extends CreateBatchJobOptions<T, U> {
    /** Unique job identifier (provider-generated) */
    id: string;
    /** Human-readable display name */
    displayName?: string;
    /** Current job status */
    status: BatchJobStatus;
    /** Job creation timestamp */
    createdAt?: Date;
    /** Job completion timestamp */
    completedAt?: Date;
    /** Error information if job failed */
    error?: BatchJobError;
    /** Job statistics */
    stats?: BatchJobStats;
    /** Provider identifier (e.g., "vertexai", "openai") */
    provider: Providers;
    /** Original provider-specific job ID/name */
    providerJobId?: string;
}

// ============== Batch Job Options ===============

/**
 * Options for creating a new batch job.
 */
export interface CreateBatchJobOptions<T, U> {
    /** Model to use for the batch job */
    model: string;
    /** Type of batch operation */
    type: BatchJobType;
    /** Optional display name for the job */
    displayName?: string;
    /** Model-specific options/parameters */
    modelOptions?: ModelOptions;
    /** Source configuration for input data */
    source: T;
    /** Destination configuration for output data, must be specified if request is not inline */
    destination?: U;
}

/**
 * Options for listing batch jobs.
 */
export interface ListBatchJobsOptions {
    /** Maximum number of jobs to return per page */
    pageSize?: number;
    /** Token for pagination */
    pageToken?: string;
    /** Filter expression (provider-specific) */
    filter?: string;
}

/**
 * Result from listing batch jobs.
 */
export interface ListBatchJobsResult<T, U> {
    /** List of batch jobs */
    jobs: BatchJob<T, U>[];
    /** Token for next page (if more results exist) */
    nextPageToken?: string;
}
