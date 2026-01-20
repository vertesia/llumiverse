/**
 * Gemini batch operations using the @google/genai SDK.
 * Supports batch inference for Gemini models.
 */

import { BatchJob as SDKBatchJob } from "@google/genai";
import {
    BatchJob,
    BatchJobStatus,
    BatchJobType,
    CreateBatchJobOptions,
    GCSBatchDestination,
    GCSBatchSource,
    ListBatchJobsOptions,
    ListBatchJobsResult,
    Providers,
} from "@llumiverse/common";
import { VertexAIDriver } from "../index.js";
import { encodeBatchJobId, mapGeminiJobState } from "./types.js";

/**
 * Maps a Google GenAI SDK BatchJob to our unified BatchJob type.
 */
function mapGeminiBatchJob(sdkJob: SDKBatchJob, type: BatchJobType = BatchJobType.inference): BatchJob<GCSBatchSource, GCSBatchDestination> {
    const providerJobId = sdkJob.name || "";

    return {
        id: encodeBatchJobId(type === BatchJobType.embeddings ? "embeddings" : "gemini", providerJobId),
        displayName: sdkJob.displayName,
        status: mapGeminiJobState(sdkJob.state),
        type,
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
 * Creates a Gemini batch inference job.
 */
export async function createGeminiBatchJob(
    driver: VertexAIDriver,
    options: CreateBatchJobOptions<GCSBatchSource, GCSBatchDestination>
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    const client = await driver.getGoogleGenAIClient();

    // Validate required fields
    if (!options.source) {
        throw new Error("Batch job requires source configuration");
    }
    if (!options.source.gcsUris?.length && !options.source.bigqueryUri) {
        throw new Error("Batch job requires either gcsUris or bigqueryUri in source");
    }
    if (!options.destination) {
        throw new Error("Batch job requires destination configuration");
    }
    if (!options.destination.gcsUri && !options.destination.bigqueryUri) {
        throw new Error("Batch job requires either gcsUri or bigqueryUri in destination");
    }

    const batchJob = await client.batches.create({
        model: options.model,
        src: options.source.gcsUris
            ? { gcsUri: options.source.gcsUris }
            : { bigqueryUri: options.source.bigqueryUri },
        config: {
            dest: options.destination.gcsUri
                ? { gcsUri: options.destination.gcsUri }
                : { bigqueryUri: options.destination.bigqueryUri },
            displayName: options.displayName,
        },
    });

    return mapGeminiBatchJob(batchJob, BatchJobType.inference);
}

/**
 * Gets a Gemini batch job by ID.
 */
export async function getGeminiBatchJob(
    driver: VertexAIDriver,
    providerJobId: string,
    type: BatchJobType = BatchJobType.inference
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>>{
    const client = await driver.getGoogleGenAIClient();

    const batchJob = await client.batches.get({ name: providerJobId });

    return mapGeminiBatchJob(batchJob, type);
}

/**
 * Lists Gemini batch jobs.
 */
export async function listGeminiBatchJobs(
    driver: VertexAIDriver,
    options?: ListBatchJobsOptions,
    type: BatchJobType = BatchJobType.inference
): Promise<ListBatchJobsResult<GCSBatchSource, GCSBatchDestination>> {
    const client = await driver.getGoogleGenAIClient();

    const pager = await client.batches.list({
        config: {
            pageSize: options?.pageSize,
            pageToken: options?.pageToken,
        },
    });

    const jobs: BatchJob<GCSBatchSource, GCSBatchDestination>[] = [];
    for await (const job of pager) {
        jobs.push(mapGeminiBatchJob(job, type));
        // If we've reached the page size, stop iterating
        if (options?.pageSize && jobs.length >= options.pageSize) {
            break;
        }
    }

    return {
        jobs,
        // Note: The SDK pager doesn't expose nextPageToken directly
        // We would need to handle this differently for proper pagination
        nextPageToken: undefined,
    };
}

/**
 * Cancels a Gemini batch job.
 */
export async function cancelGeminiBatchJob(
    driver: VertexAIDriver,
    providerJobId: string,
    type: BatchJobType = BatchJobType.inference
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    const client = await driver.getGoogleGenAIClient();

    await client.batches.cancel({ name: providerJobId });

    // Get the updated job status
    return getGeminiBatchJob(driver, providerJobId, type);
}

/**
 * Deletes a Gemini batch job.
 */
export async function deleteGeminiBatchJob(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<void> {
    const client = await driver.getGoogleGenAIClient();

    await client.batches.delete({ name: providerJobId });
}

/**
 * Checks if a batch job is in a terminal state.
 */
export function isTerminalState(status: BatchJobStatus): boolean {
    return status === BatchJobStatus.succeeded ||
           status === BatchJobStatus.failed ||
           status === BatchJobStatus.cancelled ||
           status === BatchJobStatus.partial;
}
