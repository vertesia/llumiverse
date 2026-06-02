import type { Message } from '@anthropic-ai/sdk/resources/index.js';
import type { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type ExecutionOptions,
    type LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptSegment,
    type VertexAIClaudeOptions,
} from '@llumiverse/core';
import { transformSSEStream } from '@llumiverse/core/async';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import {
    type ClaudePrompt,
    completionFromClaudeMessage,
    createClaudeStreamEventMapper,
    formatAnthropicLlumiverseError,
    formatClaudePrompt,
    getClaudePayload,
    isClaudeErrorRetryable,
    updateClaudeConversation,
} from '../../shared/claude-messages.js';

import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

const VERTEX_ANTHROPIC_VERSION = 'vertex-2023-10-16';

export const ANTHROPIC_REGIONS: Record<string, string> = {
    us: 'us-east5',
    europe: 'europe-west1',
    global: 'global',
};

export const NON_GLOBAL_ANTHROPIC_MODELS = ['claude-3-5', 'claude-3'];

/**
 * Parse a VertexAI model path (e.g. "locations/us-east5/claude-3-5-sonnet") into
 * its region and model name components.
 */
function resolveVertexAIModelPath(options: ExecutionOptions): {
    modelName: string;
    region: string | undefined;
    options: ExecutionOptions;
} {
    const splits = options.model.split('/');
    let region: string | undefined;
    if (splits[0] === 'locations' && splits.length >= 2) {
        region = splits[1];
    }
    const modelName = splits[splits.length - 1];
    return { modelName, region, options: { ...options, model: modelName } };
}

function mapAnthropicRegion(region: string): string {
    const regionPrefix = region.split('-')[0];
    return ANTHROPIC_REGIONS[regionPrefix] || region;
}

function vertexClaudePayload(prompt: ClaudePrompt, options: ExecutionOptions, stream: boolean) {
    const conversation = updateClaudeConversation(options.conversation as ClaudePrompt | undefined, prompt);
    const { payload, requestOptions } = getClaudePayload(options, conversation, stream);
    const vertexPayload = {
        ...payload,
        anthropic_version: VERTEX_ANTHROPIC_VERSION,
    } as Record<string, unknown>;
    delete vertexPayload.model;
    return { conversation, payload: vertexPayload, headers: requestOptions?.headers };
}

export class ClaudeModelDefinition implements ModelDefinition<ClaudePrompt> {
    model: AIModel;

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } satisfies AIModel;
    }

    async createPrompt(
        _driver: VertexAIDriver,
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<ClaudePrompt> {
        return formatClaudePrompt(segments, options);
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
    ): Promise<Completion> {
        const { region, options: resolvedOptions } = resolveVertexAIModelPath(options);
        const model_options = resolvedOptions.model_options as VertexAIClaudeOptions | undefined;
        if (
            model_options?._option_id !== undefined &&
            model_options?._option_id !== 'vertexai-claude' &&
            model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: resolvedOptions.model_options }, 'Unexpected option id');
        }
        const { conversation, payload, headers } = vertexClaudePayload(prompt, resolvedOptions, false);
        const result = await driver.postVertexModel<Message>(options.model, 'rawPredict', payload, {
            region: mapAnthropicRegion(region ?? driver.options.region),
            headers,
        });
        return completionFromClaudeMessage(result, conversation, resolvedOptions);
    }

    async requestTextCompletionStream(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        const { region, options: resolvedOptions } = resolveVertexAIModelPath(options);
        const model_options = resolvedOptions.model_options as VertexAIClaudeOptions | undefined;
        if (
            model_options?._option_id !== undefined &&
            model_options?._option_id !== 'vertexai-claude' &&
            model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: resolvedOptions.model_options }, 'Unexpected option id');
        }
        const { payload, headers } = vertexClaudePayload(prompt, resolvedOptions, true);
        const stream = await driver.streamVertexModel(options.model, 'rawPredict', payload, {
            region: mapAnthropicRegion(region ?? driver.options.region),
            headers,
            query: { alt: 'sse' },
        });
        const mapper = createClaudeStreamEventMapper(resolvedOptions);
        return transformSSEStream(stream as ReadableStream<ServerSentEvent>, (data) => {
            return mapper(JSON.parse(data) as RawMessageStreamEvent);
        });
    }

    isClaudeErrorRetryable(
        error: unknown,
        httpStatusCode: number | undefined,
        errorType: string | undefined,
    ): boolean | undefined {
        return isClaudeErrorRetryable(error, httpStatusCode, errorType);
    }

    formatLlumiverseError(_driver: VertexAIDriver, error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        return formatAnthropicLlumiverseError(error, context);
    }
}
