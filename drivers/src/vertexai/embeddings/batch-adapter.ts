import type { EmbeddingBatchAdapter } from '@llumiverse/core';
import type { VertexAIDriver } from '../index.js';
import {
    cancelVertexEmbeddingBatch,
    createVertexEmbeddingBatch,
    deleteVertexEmbeddingBatch,
    formatVertexEmbeddingBatchRow,
    getVertexEmbeddingBatch,
    getVertexEmbeddingBatchCapability,
    parseVertexEmbeddingBatchResult,
    vertexEmbeddingBatchCorrelationKey,
} from './batch.js';

export const vertexEmbeddingBatchAdapter: EmbeddingBatchAdapter = {
    capability(model, modality, location) {
        const profile = getVertexEmbeddingBatchCapability(model, modality);
        return profile
            ? {
                  model: profile.model,
                  inputFormat: profile.schema,
                  maxRows: profile.maxRows,
                  location: profile.location === 'global' ? 'global' : location,
              }
            : undefined;
    },
    async formatRow({ inputFormat, ...options }) {
        if (inputFormat !== 'gemini' && inputFormat !== 'legacy') {
            throw new Error(`Unsupported Vertex embedding batch input format ${inputFormat}`);
        }
        const row = await formatVertexEmbeddingBatchRow({ ...options, schema: inputFormat });
        const resultKey = vertexEmbeddingBatchCorrelationKey(row);
        if (!resultKey) throw new Error('Embedding batch input cannot be correlated with its result');
        return { row, resultKey };
    },
    parseResult: parseVertexEmbeddingBatchResult,
    create: (driver, options) => createVertexEmbeddingBatch(driver as VertexAIDriver, options),
    get: (driver, options) =>
        getVertexEmbeddingBatch(driver as VertexAIDriver, options.model, options.modality, options.name),
    cancel: (driver, options) =>
        cancelVertexEmbeddingBatch(driver as VertexAIDriver, options.model, options.modality, options.name),
    delete: (driver, options) =>
        deleteVertexEmbeddingBatch(driver as VertexAIDriver, options.model, options.modality, options.name),
    async outputArtifacts(_driver, job, artifacts) {
        if (!job.outputUri) throw new Error('Vertex embedding batch omitted its output prefix');
        // Vertex writes shards directly to application storage; never copy the output or source media.
        return artifacts.list(job.outputUri);
    },
};
