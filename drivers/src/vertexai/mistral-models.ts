import { type AIModel, ModelType, Providers } from '@llumiverse/core';
import { resolveModelListingMetadata } from '../shared/model-listing.js';

/** Mistral's regional publisher APIs use rawPredict, rather than the Open MaaS chat endpoint. */
export const VERTEX_MISTRAL_CHAT_MODELS = ['mistral-small-2503', 'mistral-medium-3', 'codestral-2'] as const;
const MISTRAL_REGIONS = ['us-central1', 'europe-west4'] as const;

export function isVertexMistralChatModel(publisher: string | undefined, model: string): boolean {
    return publisher === 'mistralai' && VERTEX_MISTRAL_CHAT_MODELS.some((entry) => entry === model);
}

export function getListedVertexMistralModels(): AIModel[] {
    return VERTEX_MISTRAL_CHAT_MODELS.flatMap((model) =>
        MISTRAL_REGIONS.map((region) => ({
            id: `locations/${region}/publishers/mistralai/models/${model}`,
            name: model,
            provider: Providers.vertexai,
            owner: 'mistralai',
            type: ModelType.Text,
            can_stream: true,
            ...resolveModelListingMetadata(model, Providers.vertexai),
        })),
    );
}
