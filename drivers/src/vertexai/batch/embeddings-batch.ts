/**
 * Embeddings batch operations using the @google/genai SDK and HTTP API.
 * Supports batch text and multimodal embeddings.
 *
 * Two implementations are available:
 * - SDK-based: Uses @google/genai SDK (experimental batches.createEmbeddings)
 * - HTTP-based: Uses direct HTTP calls to the Generative Language API
 */

import { EmbeddingsBatchJobSource, BatchJob as SDKBatchJob } from "@google/genai";
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
import { registerGeminiFiles } from "./gemini-files.js";
import {
    encodeBatchJobId,
    GeminiAsyncBatchEmbedRequest,
    GeminiBatchResource,
    GeminiListBatchesResponse,
    GeminiOperationResponse,
    mapGeminiBatchState,
    mapGeminiJobState,
} from "./types.js";

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
export async function createEmbeddingsBatchJobSDK(
    driver: VertexAIDriver,
    options: CreateBatchJobOptions<GCSBatchSource, GCSBatchDestination>
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    const client = await driver.getGoogleGenAIClient(undefined, "GEMINI");

    // Validate required fields
    if (!options.source) {
        throw new Error("Batch job requires source configuration");
    }
    if (!options.source.gcsUris?.length && !options.source.bigqueryUri && !options.source.inlinedRequests?.length) {
        throw new Error("Batch job requires either gcsUris, bigqueryUri, or inlinedRequests in source");
    }
    // Note: destination validation is less strict for embeddings as SDK may auto-generate

    // Default to gemini-embedding-001 if no model specified
    const model = options.model || DEFAULT_TEXT_EMBEDDING_MODEL;

    // Note: createEmbeddings API is experimental
    // It supports fileName (for file uploads), inlinedRequests, or GCS URIs
    let batchJob: SDKBatchJob;
    try {
        // Temporary hard-coded registration for manual testing until dynamic wiring lands
        const gcsUri = options.source.gcsUris?.[0];
        if (!gcsUri) {
            throw new Error("Currently only GCS URI source is supported for embeddings batch job creation");
        }
        const registration = await registerGeminiFiles(driver, { uris: [gcsUri], quotaProject: driver.options.project });
        const registeredFileName = registration.files[0]?.name;

        if (!registeredFileName) {
            throw new Error("Gemini file registration did not return a file reference");
        }

        // Build source configuration based on what's provided
        const src: EmbeddingsBatchJobSource = { fileName: registeredFileName };

        // if (options.source.inlinedRequests?.length) {
        //     console.log("Creating embeddings batch job with inlined requests:", JSON.stringify(options.source.inlinedRequests, null, 2));
        //     // Use inline requests for testing or small batches
        //     // Note: The SDK expects 'contents' to be Content[], but our options usually come
        //     // wrapped as { content: Content, ... }. We need to extract the content.
        //     const contents = options.source.inlinedRequests.map((req: any) => {
        //         const content = req.content || req;
        //         // Ensure parts is present
        //         if (!content.parts && typeof content.text === 'string') {
        //             return { parts: [{ text: content.text }] };
        //         }
        //         return content;
        //     });
        //     console.log("Mapped contents:", JSON.stringify(contents, null, 2));
        //     src.inlinedRequests = { contents };
        // } else if (options.source.gcsUris?.length) {
        //     // Use fileName for GCS-based input
        //     src.fileName = options.source.gcsUris[0];
        // }

        batchJob = await client.batches.createEmbeddings({
            model,
            src,
            config: {
                displayName: options.displayName,
            },
        });
        console.log("Embeddings batch job created with ID:", JSON.stringify(batchJob));
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

// ============== HTTP-based Implementations ===============

/**
 * Maps a Gemini API batch resource to our unified BatchJob type.
 */
function mapHTTPBatchJob(batch: GeminiBatchResource): BatchJob<GCSBatchSource, GCSBatchDestination> {
    const providerJobId = batch.name || "";

    // Extract inline requests from nested structure if present
    // API returns: inputConfig.requests.requests[].request
    const inlinedRequests = batch.inputConfig?.requests?.requests?.map(wrapper => wrapper.request);

    return {
        id: encodeBatchJobId("embeddings", providerJobId),
        displayName: batch.displayName,
        status: mapGeminiBatchState(batch.state),
        type: BatchJobType.embeddings,
        model: batch.model || "",
        source: {
            // Gemini API uses fileName for file-based input
            gcsUris: batch.inputConfig?.fileName ? [batch.inputConfig.fileName] : undefined,
            inlinedRequests: inlinedRequests,
        },
        destination: batch.output?.responsesFile ? {
            gcsUri: batch.output.responsesFile,
        } : undefined,
        createdAt: batch.createTime ? new Date(batch.createTime) : undefined,
        completedAt: batch.endTime ? new Date(batch.endTime) : undefined,
        error: batch.error ? {
            code: String(batch.error.code || "UNKNOWN"),
            message: batch.error.message || "Unknown error",
            details: batch.error.details?.map(d => JSON.stringify(d)).join("; "),
        } : undefined,
        stats: batch.batchStats ? {
            totalRequests: batch.batchStats.requestCount,
            completedRequests: batch.batchStats.successfulRequestCount,
            failedRequests: batch.batchStats.failedRequestCount,
        } : undefined,
        provider: Providers.vertexai,
        providerJobId,
    };
}

/**
 * Normalizes model name to the format expected by the Gemini API.
 * Ensures model is in format "models/{model}" without version suffix.
 */
function normalizeModelForGeminiApi(model: string): string {
    // Remove any version suffix (e.g., @001)
    const baseModel = model.includes("@") ? model.split("@")[0] : model;
    // Ensure it has the models/ prefix
    return baseModel.startsWith("models/") ? baseModel : `models/${baseModel}`;
}

/**
 * Creates an embeddings batch job using HTTP API.
 * Uses the Generative Language API's asyncBatchEmbedContent endpoint.
 *
 * @param driver - The VertexAI driver instance
 * @param options - Batch job options
 * @returns The created batch job
 */
export async function createEmbeddingsBatchJobHTTP(
    driver: VertexAIDriver,
    options: CreateBatchJobOptions<GCSBatchSource, GCSBatchDestination>
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    driver.logger.debug("Creating embeddings batch job via HTTP API");
    const client = driver.getGeminiApiFetchClient();

    // Validate required fields
    if (!options.source) {
        throw new Error("Batch job requires source configuration");
    }
    if (!options.source.gcsUris?.length && !options.source.inlinedRequests?.length) {
        throw new Error("Batch job requires either gcsUris (file reference) or inlinedRequests in source");
    }

    // Default to gemini-embedding-001 if no model specified
    const model = normalizeModelForGeminiApi(options.model || DEFAULT_TEXT_EMBEDDING_MODEL);

    // Build the request body
    const requestBody: GeminiAsyncBatchEmbedRequest = {
        batch: {
            model,
            displayName: options.displayName || `embeddings-batch-${Date.now()}`,
            inputConfig: {},
        },
    };

    // Set input configuration
    if (options.source.inlinedRequests?.length) {
        // Transform inlined requests to the nested structure expected by the API
        // API expects: inputConfig.requests.requests[].request.content
        // Input should have: req.content as { parts: [{ text: string }] }
        requestBody.batch.inputConfig.requests = {
            requests: options.source.inlinedRequests.map((req: any) => ({
                request: {
                    model: req.model || model,
                    content: req.content,
                    taskType: req.taskType,
                    title: req.title,
                    outputDimensionality: req.outputDimensionality,
                },
                metadata: req.metadata,
            })),
        };
    } else if (options.source.gcsUris?.length) {
        // For Gemini API, fileName should be a File resource reference
        requestBody.batch.inputConfig.fileName = options.source.gcsUris[0];
    }

    // Make the HTTP request
    // Endpoint: POST /models/{model}:asyncBatchEmbedContent
    const endpoint = `/${model}:asyncBatchEmbedContent`;

    let response: GeminiOperationResponse;
    try {
        response = await client.post(endpoint, {
            payload: requestBody,
        }) as GeminiOperationResponse;
    } catch (error) {
        console.log("Error creating embeddings batch job via HTTP:", JSON.stringify(error));
        throw new Error(`Failed to create embeddings batch job: ${error}`);
    }

    // The response is an Operation. If done, response contains the batch.
    // Otherwise, we need to extract the batch info from metadata or poll.
    if (response.done && response.response) {
        return mapHTTPBatchJob(response.response);
    }

    // If not done yet, create a minimal batch job from operation info
    // The operation name typically contains the batch ID
    const operationName = response.name || "";
    return {
        id: encodeBatchJobId("embeddings", operationName),
        displayName: options.displayName,
        status: mapGeminiBatchState("BATCH_STATE_PENDING"),
        type: BatchJobType.embeddings,
        model: options.model || DEFAULT_TEXT_EMBEDDING_MODEL,
        source: options.source,
        destination: options.destination,
        provider: Providers.vertexai,
        providerJobId: operationName,
    };
}

/**
 * Gets an embeddings batch job by ID using HTTP API.
 *
 * @param driver - The VertexAI driver instance
 * @param providerJobId - The provider job ID (format: batches/{batchId})
 * @returns The batch job
 */
export async function getEmbeddingsBatchJobHTTP(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    const client = driver.getGeminiApiFetchClient();

    // Endpoint: GET /batches/{batchId}
    const endpoint = `/${providerJobId}`;

    let response: GeminiBatchResource;
    try {
        response = await client.get(endpoint) as GeminiBatchResource;
    } catch (error) {
        throw new Error(`Failed to get embeddings batch job: ${error}`);
    }

    return mapHTTPBatchJob(response);
}

/**
 * Lists embeddings batch jobs using HTTP API.
 *
 * @param driver - The VertexAI driver instance
 * @param options - List options
 * @returns List of batch jobs
 */
export async function listEmbeddingsBatchJobsHTTP(
    driver: VertexAIDriver,
    options?: ListBatchJobsOptions
): Promise<ListBatchJobsResult<GCSBatchSource, GCSBatchDestination>> {
    const client = driver.getGeminiApiFetchClient();

    // Build query parameters
    const params: Record<string, string> = {};
    if (options?.pageSize) {
        params.pageSize = String(options.pageSize);
    }
    if (options?.pageToken) {
        params.pageToken = options.pageToken;
    }

    // Endpoint: GET /batches
    const queryString = Object.keys(params).length > 0
        ? "?" + new URLSearchParams(params).toString()
        : "";

    let response: GeminiListBatchesResponse;
    try {
        response = await client.get(`/batches${queryString}`) as GeminiListBatchesResponse;
    } catch (error) {
        throw new Error(`Failed to list embeddings batch jobs: ${error}`);
    }

    return {
        jobs: (response.batches || []).map(mapHTTPBatchJob),
        nextPageToken: response.nextPageToken,
    };
}

/**
 * Cancels an embeddings batch job using HTTP API.
 *
 * @param driver - The VertexAI driver instance
 * @param providerJobId - The provider job ID (format: batches/{batchId})
 * @returns The cancelled batch job
 */
export async function cancelEmbeddingsBatchJobHTTP(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
    const client = driver.getGeminiApiFetchClient();

    // Endpoint: POST /batches/{batchId}:cancel
    const endpoint = `/${providerJobId}:cancel`;

    try {
        await client.post(endpoint, { payload: {} });
    } catch (error) {
        throw new Error(`Failed to cancel embeddings batch job: ${error}`);
    }

    // Get the updated job status
    return getEmbeddingsBatchJobHTTP(driver, providerJobId);
}

/**
 * Deletes an embeddings batch job using HTTP API.
 *
 * @param driver - The VertexAI driver instance
 * @param providerJobId - The provider job ID (format: batches/{batchId})
 */
export async function deleteEmbeddingsBatchJobHTTP(
    driver: VertexAIDriver,
    providerJobId: string
): Promise<void> {
    const client = driver.getGeminiApiFetchClient();

    // Endpoint: DELETE /batches/{batchId}
    const endpoint = `/${providerJobId}`;

    try {
        await client.delete(endpoint);
    } catch (error) {
        throw new Error(`Failed to delete embeddings batch job: ${error}`);
    }
}
