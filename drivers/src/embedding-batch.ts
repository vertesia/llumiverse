import type { EmbeddingBatchAdapter } from '@llumiverse/core';
import { vertexEmbeddingBatchAdapter } from './vertexai/embeddings/batch-adapter.js';

/** Providers opt in explicitly; unsupported providers keep their synchronous embedding path. */
export function getEmbeddingBatchAdapter(provider: string): EmbeddingBatchAdapter | undefined {
    return provider === 'vertexai' ? vertexEmbeddingBatchAdapter : undefined;
}
