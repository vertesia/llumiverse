/**
 * VertexAI Batch API support.
 *
 * Provides unified batch operations for:
 * - Gemini inference (via @google/genai SDK)
 * - Claude inference (via aiplatform REST API)
 * - Text and multimodal embeddings (via @google/genai SDK)
 *
 * @example
 * ```typescript
 * const driver = new VertexAIDriver({ project: 'my-project', region: 'us-central1' });
 * const batchClient = driver.getBatchClient();
 *
 * // Create a batch job
 * const job = await batchClient.createBatchJob({
 *     model: 'gemini-2.0-flash',
 *     type: BatchJobType.inference,
 *     source: { gcsUris: ['gs://bucket/input.jsonl'] },
 *     destination: { gcsUri: 'gs://bucket/output/' },
 * });
 *
 * // Check status
 * const status = await batchClient.getBatchJob(job.id);
 * ```
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
    cancelClaudeBatchJob,
    createClaudeBatchJob,
    deleteClaudeBatchJob,
    getClaudeBatchJob,
    listClaudeBatchJobs,
} from "./claude-batch.js";
import {
    cancelEmbeddingsBatchJob,
    createEmbeddingsBatchJob,
    deleteEmbeddingsBatchJob,
    getEmbeddingsBatchJob,
} from "./embeddings-batch.js";
import {
    cancelGeminiBatchJob,
    createGeminiBatchJob,
    deleteGeminiBatchJob,
    getGeminiBatchJob,
    isTerminalState,
    listGeminiBatchJobs,
} from "./gemini-batch.js";
import { BatchProvider, decodeBatchJobId } from "./types.js";

// Re-export types for convenience
export * from "./types.js";
export { isTerminalState } from "./gemini-batch.js";
export {
    DEFAULT_TEXT_EMBEDDING_MODEL,
    DEFAULT_MULTIMODAL_EMBEDDING_MODEL,
    isMultimodalEmbeddingModel,
    isTextEmbeddingModel,
} from "./embeddings-batch.js";

/**
 * VertexAI Batch Client.
 *
 * Provides a unified interface for batch operations across different model types.
 * Automatically routes to the appropriate implementation based on the model.
 */
export class VertexAIBatchClient {
    constructor(private driver: VertexAIDriver) {}

    /**
     * Creates a new batch job.
     *
     * @param options - Options for creating the batch job
     * @returns The created batch job
     * @throws Error if required fields are missing or the model is not supported
     *
     * @example
     * ```typescript
     * // Gemini inference batch
     * const job = await batchClient.createBatchJob({
     *     model: 'gemini-2.0-flash',
     *     type: BatchJobType.inference,
     *     source: { gcsUris: ['gs://bucket/input.jsonl'] },
     *     destination: { gcsUri: 'gs://bucket/output/' },
     * });
     *
     * // Claude inference batch
     * const job = await batchClient.createBatchJob({
     *     model: 'claude-3-5-haiku',
     *     type: BatchJobType.inference,
     *     source: { gcsUris: ['gs://bucket/claude-input.jsonl'] },
     *     destination: { gcsUri: 'gs://bucket/output/' },
     * });
     *
     * // Embeddings batch
     * const job = await batchClient.createBatchJob({
     *     model: 'gemini-embedding-001',
     *     type: BatchJobType.embeddings,
     *     source: { gcsUris: ['gs://bucket/texts.jsonl'] },
     *     destination: { gcsUri: 'gs://bucket/embeddings/' },
     * });
     * ```
     */
    async createBatchJob(options: CreateBatchJobOptions): Promise<BatchJob> {
        const provider = this.getModelProvider(options.model, options.type);

        switch (provider) {
            case "claude":
                return createClaudeBatchJob(this.driver, options);
            case "embeddings":
                return createEmbeddingsBatchJob(this.driver, options);
            case "gemini":
            default:
                return createGeminiBatchJob(this.driver, options);
        }
    }

    /**
     * Gets a batch job by ID.
     *
     * @param jobId - The batch job ID (includes provider prefix)
     * @returns The batch job with current status
     *
     * @example
     * ```typescript
     * const job = await batchClient.getBatchJob('gemini:batches/123');
     * console.log(job.status); // 'running', 'succeeded', etc.
     * ```
     */
    async getBatchJob(jobId: string): Promise<BatchJob> {
        const { provider, providerJobId } = decodeBatchJobId(jobId);

        switch (provider) {
            case "claude":
                return getClaudeBatchJob(this.driver, providerJobId);
            case "embeddings":
                return getEmbeddingsBatchJob(this.driver, providerJobId);
            case "gemini":
            default:
                return getGeminiBatchJob(this.driver, providerJobId);
        }
    }

    /**
     * Lists batch jobs.
     *
     * Note: This returns jobs from all providers. The SDK doesn't support
     * filtering by provider, so filtering should be done client-side if needed.
     *
     * @param options - Options for listing (pagination, filter)
     * @returns List of batch jobs and pagination token
     *
     * @example
     * ```typescript
     * const { jobs, nextPageToken } = await batchClient.listBatchJobs({
     *     pageSize: 10,
     * });
     * ```
     */
    async listBatchJobs(options?: ListBatchJobsOptions): Promise<ListBatchJobsResult> {
        // List from all providers and combine
        // For now, we primarily use the Gemini SDK listing which includes most jobs
        // Claude jobs are listed separately via REST API
        const [geminiResult, claudeResult] = await Promise.all([
            listGeminiBatchJobs(this.driver, options),
            listClaudeBatchJobs(this.driver, options).catch(() => ({ jobs: [], nextPageToken: undefined })),
        ]);

        return {
            jobs: [...geminiResult.jobs, ...claudeResult.jobs],
            nextPageToken: geminiResult.nextPageToken || claudeResult.nextPageToken,
        };
    }

    /**
     * Cancels a running batch job.
     *
     * @param jobId - The batch job ID to cancel
     * @returns The batch job with updated status
     *
     * @example
     * ```typescript
     * const job = await batchClient.cancelBatchJob('gemini:batches/123');
     * console.log(job.status); // 'cancelled'
     * ```
     */
    async cancelBatchJob(jobId: string): Promise<BatchJob> {
        const { provider, providerJobId } = decodeBatchJobId(jobId);

        switch (provider) {
            case "claude":
                return cancelClaudeBatchJob(this.driver, providerJobId);
            case "embeddings":
                return cancelEmbeddingsBatchJob(this.driver, providerJobId);
            case "gemini":
            default:
                return cancelGeminiBatchJob(this.driver, providerJobId);
        }
    }

    /**
     * Deletes a batch job.
     *
     * @param jobId - The batch job ID to delete
     *
     * @example
     * ```typescript
     * await batchClient.deleteBatchJob('gemini:batches/123');
     * ```
     */
    async deleteBatchJob(jobId: string): Promise<void> {
        const { provider, providerJobId } = decodeBatchJobId(jobId);

        switch (provider) {
            case "claude":
                return deleteClaudeBatchJob(this.driver, providerJobId);
            case "embeddings":
                return deleteEmbeddingsBatchJob(this.driver, providerJobId);
            case "gemini":
            default:
                return deleteGeminiBatchJob(this.driver, providerJobId);
        }
    }

    /**
     * Waits for a batch job to complete (reach a terminal state).
     *
     * @param jobId - The batch job ID to wait for
     * @param pollIntervalMs - Polling interval in milliseconds (default: 30000)
     * @param maxWaitMs - Maximum wait time in milliseconds (default: 24 hours)
     * @returns The completed batch job
     * @throws Error if timeout is reached
     *
     * @example
     * ```typescript
     * const completedJob = await batchClient.waitForCompletion(job.id, {
     *     pollIntervalMs: 60000, // Check every minute
     * });
     * if (completedJob.status === BatchJobStatus.succeeded) {
     *     // Process results
     * }
     * ```
     */
    async waitForCompletion(
        jobId: string,
        pollIntervalMs: number = 30000,
        maxWaitMs: number = 24 * 60 * 60 * 1000
    ): Promise<BatchJob> {
        const startTime = Date.now();

        while (true) {
            const job = await this.getBatchJob(jobId);

            if (isTerminalState(job.status)) {
                return job;
            }

            if (Date.now() - startTime > maxWaitMs) {
                throw new Error(`Batch job ${jobId} did not complete within ${maxWaitMs}ms`);
            }

            await sleep(pollIntervalMs);
        }
    }

    /**
     * Determines the provider to use based on model name and job type.
     */
    private getModelProvider(model: string, type: BatchJobType): BatchProvider {
        const modelLower = model.toLowerCase();

        // Embeddings type always goes to embeddings provider
        if (type === BatchJobType.embeddings) {
            return "embeddings";
        }

        // Claude models use the REST API
        if (modelLower.includes("claude") || modelLower.includes("anthropic")) {
            return "claude";
        }

        // Embedding models use the SDK embeddings API
        if (modelLower.includes("embedding") || modelLower.includes("multimodal")) {
            return "embeddings";
        }

        // Default to Gemini SDK
        return "gemini";
    }
}

/**
 * Helper function to sleep for a given duration.
 */
function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}
