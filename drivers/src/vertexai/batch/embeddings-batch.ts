/**
 * Embeddings batch operations using the @google/genai SDK.
 * Supports batch text and multimodal embeddings.
 */

import { BatchJob as SDKBatchJob } from "@google/genai";
import {
    BatchJob,
    BatchJobType,
    CreateBatchJobOptions,
    GCSBatchDestination,
    GCSBatchSource,
    ListBatchJobsOptions,
    ListBatchJobsResult,
    Providers,
} from "@llumiverse/common";
import { VertexAIDriver } from "../index.js";
import {
    cancelGeminiBatchJob,
    deleteGeminiBatchJob,
    getGeminiBatchJob,
    listGeminiBatchJobs,
} from "./gemini-batch.js";
import { encodeBatchJobId, mapGeminiJobState } from "./types.js";

/**
 * Default embedding model for text embeddings.
 */
export const DEFAULT_TEXT_EMBEDDING_MODEL = "gemini-embedding-001";

/**
 * Default model for multimodal embeddings.
 */
export const DEFAULT_MULTIMODAL_EMBEDDING_MODEL = "multimodalembedding@001";

/**
 * Maps a Google GenAI SDK BatchJob to our unified BatchJob type for embeddings.
 */
function mapEmbeddingsBatchJob(sdkJob: SDKBatchJob): BatchJob<GCSBatchSource, GCSBatchDestination> {
    const providerJobId = sdkJob.name || "";

    return {
        id: encodeBatchJobId("embeddings", providerJobId),
        displayName: sdkJob.displayName,
        status: mapGeminiJobState(sdkJob.state),
        type: BatchJobType.embeddings,
        model: sdkJob.model || "",
        source: {
            gcsUris: sdkJob.src?.gcsUri,
            bigqueryUri: sdkJob.src?.bigqueryUri,
        },
        destination: sdkJob.dest ? {
            gcsUri: sdkJob.dest.gcsUri,
            bigqueryUri: sdkJob.dest.bigqueryUri,
        } : undefined,
        createdAt: sdkJob.createTime ? new Date(sdkJob.createTime) : undefined,
        completedAt: sdkJob.endTime ? new Date(sdkJob.endTime) : undefined,
        error: sdkJob.error ? {
            code: String(sdkJob.error.code || "UNKNOWN"),
            message: sdkJob.error.message || "Unknown error",
            details: sdkJob.error.details?.join("; "),
        } : undefined,
        stats: sdkJob.completionStats ? {
            completedRequests: typeof sdkJob.completionStats.successfulCount === 'number'
                ? sdkJob.completionStats.successfulCount
                : parseInt(String(sdkJob.completionStats.successfulCount || 0), 10),
            failedRequests: typeof sdkJob.completionStats.failedCount === 'number'
                ? sdkJob.completionStats.failedCount
                : parseInt(String(sdkJob.completionStats.failedCount || 0), 10),
            totalRequests:
                (typeof sdkJob.completionStats.successfulCount === 'number'
                    ? sdkJob.completionStats.successfulCount
                    : parseInt(String(sdkJob.completionStats.successfulCount || 0), 10)) +
                (typeof sdkJob.completionStats.failedCount === 'number'
                    ? sdkJob.completionStats.failedCount
                    : parseInt(String(sdkJob.completionStats.failedCount || 0), 10)),
        } : undefined,
        provider: Providers.vertexai,
        providerJobId,
    };
}

/**
 * Creates an embeddings batch job.
 *
 * Note: The SDK's batches.createEmbeddings() is marked as **Experimental**.
 *
 * @param driver - The VertexAI driver instance
 * @param options - Batch job options
 * @returns The created batch job
 */
export async function createEmbeddingsBatchJob(
    driver: VertexAIDriver,
    options: CreateBatchJobOptions<GCSBatchSource, GCSBatchDestination>
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    const client = await driver.getGoogleGenAIClient(undefined, "GEMINI");

    // Validate required fields
    if (!options.source) {
        throw new Error("Batch job requires source configuration");
    }
    if (!options.source.gcsUris?.length && !options.source.bigqueryUri) {
        throw new Error("Batch job requires either gcsUris or bigqueryUri in source");
    }
    // Note: destination validation is less strict for embeddings as SDK may auto-generate

    // Default to gemini-embedding-001 if no model specified
    const model = options.model || DEFAULT_TEXT_EMBEDDING_MODEL;

    // Note: createEmbeddings API is experimental and has limited source options
    // It supports fileName (for file uploads) or inlinedRequests
    // For GCS-based batch embeddings, we may need to use the standard batches.create
    // For now, use the file-based approach which is more common for large batches
    let batchJob: SDKBatchJob;
    try {
        batchJob = await client.batches.createEmbeddings({
            model,
            src: {
                // The SDK expects fileName for file-based input
                // If using GCS, upload the file first and pass the file name
                fileName: options.source.gcsUris?.[0],
            },
            config: {
                displayName: options.displayName,
            },
        });
    } catch (error) {
        console.log("Error creating embeddings batch job:", error);
        throw new Error(`Failed to create embeddings batch job: ${error}`);
    }

    return mapEmbeddingsBatchJob(batchJob);
}

/**
 * Gets an embeddings batch job by ID.
 * Reuses the Gemini batch get function with embeddings type.
 */
export async function getEmbeddingsBatchJob(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    return getGeminiBatchJob(driver, providerJobId, BatchJobType.embeddings);
}

/**
 * Lists embeddings batch jobs.
 * Note: The SDK doesn't distinguish between inference and embeddings jobs,
 * so this returns all batch jobs. Filter by type on the client side if needed.
 */
export async function listEmbeddingsBatchJobs(
    driver: VertexAIDriver,
    options?: ListBatchJobsOptions
): Promise<ListBatchJobsResult<GCSBatchSource, GCSBatchDestination>> {
    return listGeminiBatchJobs(driver, options, BatchJobType.embeddings);
}

/**
 * Cancels an embeddings batch job.
 */
export async function cancelEmbeddingsBatchJob(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    return cancelGeminiBatchJob(driver, providerJobId, BatchJobType.embeddings);
}

/**
 * Deletes an embeddings batch job.
 */
export async function deleteEmbeddingsBatchJob(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<void> {
    return deleteGeminiBatchJob(driver, providerJobId);
}

/**
 * Checks if a model is a multimodal embedding model.
 */
export function isMultimodalEmbeddingModel(model: string): boolean {
    return model.toLowerCase().includes("multimodal");
}

/**
 * Checks if a model is a text embedding model.
 */
export function isTextEmbeddingModel(model: string): boolean {
    return model.toLowerCase().includes("embedding") &&
        !isMultimodalEmbeddingModel(model);
}
