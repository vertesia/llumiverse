import type {
    AIModel,
    Completion,
    CompletionChunkObject,
    ExecutionOptions,
    LlumiverseError,
    LlumiverseErrorContext,
    PromptSegment,
} from '@llumiverse/core';
import type { VertexAIDriver, VertexAIPrompt } from './index.js';
import { ClaudeModelDefinition } from './models/claude.js';
import { GeminiModelDefinition } from './models/gemini.js';
import { OpenAICompatibleModelDefinition } from './models/openai_compatible.js';
import { getVertexOpenMaaSRequestModel } from './open-maas-models.js';

export function trimModelName(model: string): string {
    const i = model.lastIndexOf('@');
    return i > -1 ? model.substring(0, i) : model;
}

export interface ModelDefinition<PromptT = VertexAIPrompt> {
    model: AIModel;
    versions?: string[]; // the versions of the model that are available. ex: ['001', '002']
    createPrompt(driver: VertexAIDriver, segments: PromptSegment[], options: ExecutionOptions): Promise<PromptT>;
    requestTextCompletion(driver: VertexAIDriver, prompt: PromptT, options: ExecutionOptions): Promise<Completion>;
    requestTextCompletionStream(
        driver: VertexAIDriver,
        prompt: PromptT,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>>;
    preValidationProcessing?(
        result: Completion,
        options: ExecutionOptions,
    ): { result: Completion; options: ExecutionOptions };
    /**
     * Format provider-specific errors into standardized LlumiverseError.
     * Optional - if not provided, VertexAIDriver will use default error handling.
     */
    formatLlumiverseError?(driver: VertexAIDriver, error: unknown, context: LlumiverseErrorContext): LlumiverseError;
}

export function getModelDefinition(model: string): ModelDefinition {
    const splits = model.split('/');

    // Handle both formats: "publishers/anthropic/models/..." and "locations/.../publishers/anthropic/models/..."
    let publisher: string | undefined;
    let modelName: string;
    let region: string | undefined;

    const publisherIndex = splits.indexOf('publishers');
    const locationIndex = splits.indexOf('locations');
    if (publisherIndex !== -1 && publisherIndex + 1 < splits.length) {
        publisher = splits[publisherIndex + 1];
        modelName = trimModelName(splits[splits.length - 1]);
        if (locationIndex !== -1 && locationIndex + 1 < splits.length) {
            region = splits[locationIndex + 1];
        }
    } else {
        // Fallback to old logic for backward compatibility
        publisher = splits[1];
        modelName = trimModelName(splits[splits.length - 1]);
    }

    if (publisher?.includes('anthropic')) {
        return new ClaudeModelDefinition(modelName);
    } else {
        const openMaaSModel = getVertexOpenMaaSRequestModel(publisher, modelName);
        if (openMaaSModel) {
            return new OpenAICompatibleModelDefinition({
                modelName: openMaaSModel.modelName,
                region: region ?? openMaaSModel.region,
                apiVersion: openMaaSModel.apiVersion,
                endpointRegion: openMaaSModel.endpointRegion,
                defaultMaxTokens: openMaaSModel.defaultMaxTokens,
                extraBody: openMaaSModel.extraBody,
            });
        }
    }

    if (publisher === 'xai') {
        // Use OpenAI-compatible endpoint for xAI Grok models via Vertex AI's openapi endpoint
        // xAI/Grok models only exist in the "global" region, not regional endpoints
        return new OpenAICompatibleModelDefinition({ modelName: `xai/${modelName}`, region: 'global' });
    }

    if (publisher?.includes('google') && modelName.includes('gemini')) {
        return new GeminiModelDefinition(modelName);
    } else if (publisher?.includes('google')) {
        return new GeminiModelDefinition(modelName);
    }

    //Fallback, assume it is Gemini.
    return new GeminiModelDefinition(modelName);
}
