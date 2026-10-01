import {
    type AIModel,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    type Completion,
    type DriverCompletionStream,
    type ExecutionOptions,
    type LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptSegment,
    type VertexAIClaudeOptions,
} from '@llumiverse/core';
import type { ClaudePrompt } from '../../shared/claude-messages.js';
import {
    executeCanonicalClaudeCompletion,
    executeCanonicalClaudeContext,
    executeClaudeCompletion,
    formatAnthropicLlumiverseError,
    formatClaudePrompt,
    isClaudeErrorRetryable,
    streamCanonicalClaudeContextEvents,
    streamCanonicalClaudeEvents,
    streamClaudeCompletion,
} from '../../shared/claude-messages.js';

import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

export const ANTHROPIC_REGIONS: Record<string, string> = {
    us: 'us-east5',
    europe: 'europe-west1',
    global: 'global',
};

export function resolveVertexAIAnthropicRegion(region: string): string {
    return ANTHROPIC_REGIONS[region.split('-')[0]] ?? region;
}

export const NON_GLOBAL_ANTHROPIC_MODELS = ['claude-3-5', 'claude-3'];

/**
 * Parse a VertexAI model path (e.g. "locations/us-east5/claude-3-5-sonnet") into
 * its region and model name components.
 */
function resolveVertexAIModelPath(options: ExecutionOptions): {
    modelName: string;
    region: string | undefined;
} {
    const splits = options.model.split('/');
    let region: string | undefined;
    if (splits[0] === 'locations' && splits.length >= 2) {
        region = splits[1];
    } else if (splits[0] === 'global') {
        region = 'global';
    }
    const modelName = splits[splits.length - 1];
    return { modelName, region };
}

function vertexClaudeTransport(driver: VertexAIDriver, options: ExecutionOptions) {
    const resolved = resolveVertexAIModelPath(options);
    const region = resolveVertexAIAnthropicRegion(resolved.region ?? driver.getVertexRegion());
    return {
        ...resolved,
        region,
        identity: { model: resolved.modelName, target_options: { region } },
    };
}

export class ClaudeModelDefinition implements ModelDefinition<ClaudePrompt> {
    model: AIModel;
    readonly canonical_conversation_supported = true;

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
        return formatClaudePrompt(segments, options, _driver.logger);
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const transport = vertexClaudeTransport(driver, options);
        const client = await driver.getAnthropicClient(transport.region, options.httpTimeout);
        const model_options = options.model_options as VertexAIClaudeOptions | undefined;
        if (
            model_options?._option_id !== undefined &&
            model_options?._option_id !== 'vertexai-claude' &&
            model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        return executeClaudeCompletion(
            client,
            prompt,
            options,
            driver.logger,
            driver.provider,
            signal ? { signal } : undefined,
            transport.identity,
        );
    }

    async requestCanonicalTextCompletion(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const transport = vertexClaudeTransport(driver, options);
        const client = await driver.getAnthropicClient(transport.region, options.httpTimeout);
        return executeCanonicalClaudeCompletion(
            client,
            prompt,
            options,
            driver.logger,
            driver.provider,
            signal ? { signal } : undefined,
            transport.identity,
        );
    }

    async requestCanonicalContextCompletion(
        driver: VertexAIDriver,
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const transport = vertexClaudeTransport(driver, options);
        const client = await driver.getAnthropicClient(transport.region, options.httpTimeout);
        return executeCanonicalClaudeContext(
            client,
            options,
            driver.logger,
            driver.provider,
            signal ? { signal } : undefined,
            transport.identity,
        );
    }

    async requestTextCompletionStream(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        const transport = vertexClaudeTransport(driver, options);
        const client = await driver.getAnthropicClient(transport.region, options.httpTimeout);
        const model_options = options.model_options as VertexAIClaudeOptions | undefined;
        if (
            model_options?._option_id !== undefined &&
            model_options?._option_id !== 'vertexai-claude' &&
            model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        return streamClaudeCompletion(
            client,
            prompt,
            options,
            driver.logger,
            driver.provider,
            signal ? { signal } : undefined,
            transport.identity,
        );
    }

    async requestCanonicalTextCompletionEventStream(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const transport = vertexClaudeTransport(driver, options);
        const client = await driver.getAnthropicClient(transport.region, options.httpTimeout);
        return streamCanonicalClaudeEvents(
            client,
            prompt,
            options,
            open,
            driver.logger,
            driver.provider,
            signal ? { signal } : undefined,
            transport.identity,
        );
    }

    async requestCanonicalContextCompletionEventStream(
        driver: VertexAIDriver,
        options: CanonicalExecutionContextOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const transport = vertexClaudeTransport(driver, options);
        const client = await driver.getAnthropicClient(transport.region, options.httpTimeout);
        return streamCanonicalClaudeContextEvents(
            client,
            options,
            open,
            driver.logger,
            driver.provider,
            signal ? { signal } : undefined,
            transport.identity,
        );
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
