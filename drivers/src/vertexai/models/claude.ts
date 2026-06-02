import type { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    type LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptSegment,
    type ToolUse,
    type VertexAIClaudeOptions,
} from '@llumiverse/core';
import { transformSSEStream } from '@llumiverse/core/async';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import {
    buildClaudeStreamingConversation,
    type ClaudePrompt,
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

type StreamingToolUse = ToolUse<unknown> & { _actual_id?: string };

function accumulateResults(target: CompletionResult[], results: CompletionResult[] | undefined) {
    if (!results || results.length === 0) {
        return;
    }
    for (const result of results) {
        const lastResult = target[target.length - 1];
        if (
            lastResult &&
            ((lastResult.type === 'text' && result.type === 'text') ||
                (lastResult.type === 'json' && result.type === 'json'))
        ) {
            if (result.type === 'text') {
                lastResult.value += result.value;
            } else if (result.type === 'json') {
                if (
                    lastResult.value !== null &&
                    typeof lastResult.value === 'object' &&
                    result.value !== null &&
                    typeof result.value === 'object'
                ) {
                    lastResult.value = { ...lastResult.value, ...result.value };
                } else {
                    lastResult.value = String(lastResult.value) + String(result.value);
                }
            }
        } else {
            target.push(result);
        }
    }
}

function accumulateToolUse(target: Map<string, StreamingToolUse>, tools: ToolUse<unknown>[] | undefined) {
    if (!tools || tools.length === 0) {
        return;
    }
    for (const tool of tools) {
        const existing = target.get(tool.id);
        if (!existing) {
            target.set(tool.id, { ...tool });
            continue;
        }
        if (tool.tool_input !== null && tool.tool_input !== undefined) {
            const existingInput = existing.tool_input as unknown;
            const newInput = tool.tool_input as unknown;
            if (typeof existingInput === 'string' && typeof newInput === 'string') {
                existing.tool_input = existingInput + newInput;
            } else if (existingInput && typeof existingInput === 'object' && newInput && typeof newInput === 'object') {
                existing.tool_input = {
                    ...(existingInput as Record<string, unknown>),
                    ...(newInput as Record<string, unknown>),
                };
            } else {
                existing.tool_input = tool.tool_input;
            }
        }
        if (tool.tool_name) {
            existing.tool_name = tool.tool_name;
        }
        const streamingTool = tool as StreamingToolUse;
        if (streamingTool._actual_id) {
            existing._actual_id = streamingTool._actual_id;
        }
    }
}

async function completionFromClaudeStream(
    stream: AsyncIterable<CompletionChunkObject>,
    prompt: ClaudePrompt,
    options: ExecutionOptions,
): Promise<Completion> {
    const accumulatedResults: CompletionResult[] = [];
    const accumulatedToolUse = new Map<string, StreamingToolUse>();
    let finishReason: string | undefined;
    let promptTokens = 0;
    let resultTokens: number | undefined;
    let promptCachedTokens: number | undefined;
    let promptCacheWriteTokens: number | undefined;
    let promptNewTokens: number | undefined;

    for await (const chunk of stream) {
        if (chunk.finish_reason) {
            finishReason = chunk.finish_reason;
        }
        if (chunk.token_usage) {
            promptTokens = Math.max(promptTokens, chunk.token_usage.prompt ?? 0);
            resultTokens = Math.max(resultTokens ?? 0, chunk.token_usage.result ?? 0);
            if (chunk.token_usage.prompt_cached != null) {
                promptCachedTokens = chunk.token_usage.prompt_cached;
            }
            if (chunk.token_usage.prompt_cache_write != null) {
                promptCacheWriteTokens = chunk.token_usage.prompt_cache_write;
            }
            if (chunk.token_usage.prompt_new != null) {
                promptNewTokens = chunk.token_usage.prompt_new;
            }
        }
        accumulateToolUse(accumulatedToolUse, chunk.tool_use);
        accumulateResults(accumulatedResults, chunk.result);
    }

    const tokenUsage: ExecutionTokenUsage | undefined =
        resultTokens !== undefined
            ? {
                  prompt: promptTokens,
                  result: resultTokens,
                  total: promptTokens + resultTokens,
                  ...(promptCachedTokens != null && { prompt_cached: promptCachedTokens }),
                  ...(promptCacheWriteTokens != null && { prompt_cache_write: promptCacheWriteTokens }),
                  ...(promptNewTokens != null && { prompt_new: promptNewTokens }),
              }
            : undefined;

    let toolUseArray = accumulatedToolUse.size > 0 ? Array.from(accumulatedToolUse.values()) : undefined;
    if (toolUseArray) {
        const truncatedToolIds = new Set<string>();
        for (const tool of toolUseArray) {
            if (tool._actual_id) {
                tool.id = tool._actual_id;
                delete tool._actual_id;
            }
            if (typeof tool.tool_input === 'string') {
                try {
                    tool.tool_input = JSON.parse(tool.tool_input);
                } catch {
                    tool.tool_input = {};
                    truncatedToolIds.add(tool.id);
                }
            }
        }
        if (finishReason === 'length' && truncatedToolIds.size > 0) {
            toolUseArray = toolUseArray.filter((tool) => !truncatedToolIds.has(tool.id));
            if (toolUseArray.length === 0) {
                toolUseArray = undefined;
            }
        }
    }

    const result = accumulatedResults.length > 0 ? accumulatedResults : [{ type: 'text' as const, value: '' }];
    return {
        result,
        token_usage: tokenUsage,
        finish_reason: finishReason,
        tool_use: toolUseArray as Completion['tool_use'],
        conversation: buildClaudeStreamingConversation(prompt, result, toolUseArray, options),
    };
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
        const { payload, headers } = vertexClaudePayload(prompt, resolvedOptions, true);
        const stream = await driver.streamVertexModel(options.model, 'streamRawPredict', payload, {
            region: mapAnthropicRegion(region ?? driver.options.region),
            headers,
        });
        const mapper = createClaudeStreamEventMapper(resolvedOptions);
        const chunks = transformSSEStream(stream as ReadableStream<ServerSentEvent>, (data) => {
            return mapper(JSON.parse(data) as RawMessageStreamEvent);
        });
        return completionFromClaudeStream(chunks, prompt, resolvedOptions);
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
        const stream = await driver.streamVertexModel(options.model, 'streamRawPredict', payload, {
            region: mapAnthropicRegion(region ?? driver.options.region),
            headers,
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
