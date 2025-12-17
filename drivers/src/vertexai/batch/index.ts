/**
 * VertexAI Batch API support.
 *
 * Provides batch operations for:
 * - Gemini inference (via @google/genai SDK)
 * - Claude inference (via aiplatform REST API)
 * - Text and multimodal embeddings (via @google/genai SDK)
 *
 * Batch operations are exposed directly on the VertexAIDriver class:
 * - driver.createBatchJob(options)
 * - driver.getBatchJob(jobId)
 * - driver.listBatchJobs(options)
 * - driver.cancelBatchJob(jobId)
 * - driver.deleteBatchJob(jobId)
 * - driver.waitForBatchJobCompletion(jobId, pollIntervalMs, maxWaitMs)
 *
 * @example
 * ```typescript
 * const driver = new VertexAIDriver({ project: 'my-project', region: 'us-central1' });
 *
 * // Create a batch job
 * const job = await driver.createBatchJob({
 *     model: 'gemini-2.0-flash',
 *     type: BatchJobType.inference,
 *     source: { gcsUris: ['gs://bucket/input.jsonl'] },
 *     destination: { gcsUri: 'gs://bucket/output/' },
 * });
 *
 * // Check status
 * const status = await driver.getBatchJob(job.id);
 *
 * // Wait for completion
 * const completedJob = await driver.waitForBatchJobCompletion(job.id);
 * ```
 */

// Re-export types for convenience
export * from "./types.js";

// Re-export GCS helpers and formatters
export * from "./gcs-helpers.js";
export * from "./formatters.js";

// Re-export utility functions for advanced usage
export {
    createGeminiBatchJob,
    getGeminiBatchJob,
    listGeminiBatchJobs,
    cancelGeminiBatchJob,
    deleteGeminiBatchJob,
    isTerminalState,
} from "./gemini-batch.js";

export {
    createClaudeBatchJob,
    getClaudeBatchJob,
    listClaudeBatchJobs,
    cancelClaudeBatchJob,
    deleteClaudeBatchJob,
} from "./claude-batch.js";

export {
    createEmbeddingsBatchJob,
    getEmbeddingsBatchJob,
    cancelEmbeddingsBatchJob,
    deleteEmbeddingsBatchJob,
    DEFAULT_TEXT_EMBEDDING_MODEL,
    DEFAULT_MULTIMODAL_EMBEDDING_MODEL,
    isMultimodalEmbeddingModel,
    isTextEmbeddingModel,
} from "./embeddings-batch.js";
