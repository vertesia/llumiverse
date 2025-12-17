/**
 * Claude batch operations using the aiplatform REST API.
 * Supports batch inference for Claude models on VertexAI.
 */

import {
    BatchJob,
    BatchJobType,
    CreateBatchJobOptions,
    ListBatchJobsOptions,
    ListBatchJobsResult,
} from "@llumiverse/common";
import { VertexAIDriver } from "../index.js";
import {
    ClaudeBatchJobRequest,
    ClaudeBatchJobResponse,
    encodeBatchJobId,
    mapClaudeJobState,
} from "./types.js";

/**
 * Extracts the model name from various model path formats.
 * Handles formats like:
 * - "claude-3-5-haiku"
 * - "publishers/anthropic/models/claude-3-5-haiku"
 * - "locations/us/publishers/anthropic/models/claude-3-5-haiku"
 */
function extractClaudeModelName(model: string): string {
    // If it contains /models/, extract the part after it
    const modelsIndex = model.indexOf("/models/");
    if (modelsIndex !== -1) {
        return model.substring(modelsIndex + 8); // 8 = length of "/models/"
    }
    // Otherwise return as-is (simple model name)
    return model;
}

/**
 * Maps a Claude batch job response to our unified BatchJob type.
 */
function mapClaudeBatchJob(response: ClaudeBatchJobResponse): BatchJob {
    const providerJobId = response.name;

    return {
        id: encodeBatchJobId("claude", providerJobId),
        displayName: response.displayName,
        status: mapClaudeJobState(response.state),
        type: BatchJobType.inference,
        model: response.model || "",
        source: {
            gcsUris: response.inputConfig?.gcsSource?.uris,
            bigqueryUri: response.inputConfig?.bigquerySource?.inputUri,
        },
        destination: {
            gcsUri: response.outputConfig?.gcsDestination?.outputUriPrefix,
            bigqueryUri: response.outputConfig?.bigqueryDestination?.outputUri,
        },
        createdAt: response.createTime ? new Date(response.createTime) : undefined,
        completedAt: response.endTime ? new Date(response.endTime) : undefined,
        error: response.error ? {
            code: String(response.error.code),
            message: response.error.message,
            details: response.error.details ? JSON.stringify(response.error.details) : undefined,
        } : undefined,
        stats: response.completionStats ? {
            completedRequests: parseInt(response.completionStats.successfulCount, 10) || 0,
            failedRequests: parseInt(response.completionStats.failedCount, 10) || 0,
            totalRequests:
                (parseInt(response.completionStats.successfulCount, 10) || 0) +
                (parseInt(response.completionStats.failedCount, 10) || 0) +
                (parseInt(response.completionStats.incompleteCount || "0", 10) || 0),
        } : undefined,
        provider: "vertexai",
        providerJobId,
    };
}

/**
 * Creates a Claude batch inference job.
 */
export async function createClaudeBatchJob(
    driver: VertexAIDriver,
    options: CreateBatchJobOptions
): Promise<BatchJob> {
    const client = driver.getFetchClient();

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

    const modelName = extractClaudeModelName(options.model);

    const requestBody: ClaudeBatchJobRequest = {
        displayName: options.displayName || `claude-batch-${Date.now()}`,
        model: `publishers/anthropic/models/${modelName}`,
        inputConfig: {
            instancesFormat: "jsonl",
            gcsSource: options.source.gcsUris
                ? { uris: options.source.gcsUris }
                : undefined,
            bigquerySource: options.source.bigqueryUri
                ? { inputUri: options.source.bigqueryUri }
                : undefined,
        },
        outputConfig: {
            predictionsFormat: "jsonl",
            gcsDestination: options.destination.gcsUri
                ? { outputUriPrefix: options.destination.gcsUri }
                : undefined,
            bigqueryDestination: options.destination.bigqueryUri
                ? { outputUri: options.destination.bigqueryUri }
                : undefined,
        },
        modelParameters: options.modelOptions,
    };

    const response = await client.post("/batchPredictionJobs", {
        payload: requestBody,
    }) as ClaudeBatchJobResponse;

    return mapClaudeBatchJob(response);
}

/**
 * Gets a Claude batch job by ID.
 * The providerJobId should be the full resource path.
 */
export async function getClaudeBatchJob(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<BatchJob> {
    const client = driver.getFetchClient();

    // The providerJobId is expected to be the full path like:
    // "projects/.../locations/.../batchPredictionJobs/123"
    // But we need to make a relative request to our base URL
    // Extract just the batchPredictionJobs/ID part
    const jobPath = extractJobPath(providerJobId);

    const response = await client.get(jobPath) as ClaudeBatchJobResponse;

    return mapClaudeBatchJob(response);
}

/**
 * Extracts the relative job path from a full resource name.
 */
function extractJobPath(providerJobId: string): string {
    // If it's already just "batchPredictionJobs/xxx", return as-is
    if (providerJobId.startsWith("batchPredictionJobs/")) {
        return `/${providerJobId}`;
    }

    // Extract from full path: "projects/.../locations/.../batchPredictionJobs/xxx"
    const batchIndex = providerJobId.indexOf("batchPredictionJobs/");
    if (batchIndex !== -1) {
        return `/${providerJobId.substring(batchIndex)}`;
    }

    // Assume it's just the job ID
    return `/batchPredictionJobs/${providerJobId}`;
}

/**
 * Lists Claude batch jobs.
 */
export async function listClaudeBatchJobs(
    driver: VertexAIDriver,
    options?: ListBatchJobsOptions
): Promise<ListBatchJobsResult> {
    const client = driver.getFetchClient();

    // Build query parameters
    const params = new URLSearchParams();
    if (options?.pageSize) {
        params.set("pageSize", String(options.pageSize));
    }
    if (options?.pageToken) {
        params.set("pageToken", options.pageToken);
    }
    if (options?.filter) {
        params.set("filter", options.filter);
    }

    const queryString = params.toString();
    const url = `/batchPredictionJobs${queryString ? `?${queryString}` : ""}`;

    const response = await client.get(url) as {
        batchPredictionJobs?: ClaudeBatchJobResponse[];
        nextPageToken?: string;
    };

    const jobs = (response.batchPredictionJobs || []).map(mapClaudeBatchJob);

    return {
        jobs,
        nextPageToken: response.nextPageToken,
    };
}

/**
 * Cancels a Claude batch job.
 */
export async function cancelClaudeBatchJob(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<BatchJob> {
    const client = driver.getFetchClient();

    const jobPath = extractJobPath(providerJobId);

    // POST to cancel endpoint
    await client.post(`${jobPath}:cancel`, { payload: {} });

    // Get the updated job status
    return getClaudeBatchJob(driver, providerJobId);
}

/**
 * Deletes a Claude batch job.
 * Note: aiplatform uses DELETE method for job deletion.
 */
export async function deleteClaudeBatchJob(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<void> {
    const client = driver.getFetchClient();

    const jobPath = extractJobPath(providerJobId);

    // DELETE request
    await client.delete(jobPath);
}
