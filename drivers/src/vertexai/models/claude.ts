import { isClaudeVersionGTE } from '@llumiverse/common';
import {
    type AIModel,
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
    executeClaudeCompletion,
    formatAnthropicLlumiverseError,
    formatClaudePrompt,
    isClaudeErrorRetryable,
    streamClaudeCompletion,
} from '../../shared/claude-messages.js';
import {
    ClaudeStructuredOutputDenials,
    isClaudeStructuredOutputPolicyDenial,
    usesClaudeNativeResultSchema,
    withClaudeStructuredOutputFallback,
} from '../../shared/claude-structured-output.js';

import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

export const ANTHROPIC_REGIONS: Record<string, string> = {
    us: 'us-east5',
    europe: 'europe-west1',
    global: 'global',
};

export const NON_GLOBAL_ANTHROPIC_MODELS = ['claude-3-5', 'claude-3'];

/** Multi-region locations that serve Claude generations without single-region endpoints. */
const ANTHROPIC_MULTI_REGIONS: Record<string, string> = {
    us: 'us',
    europe: 'eu',
};

/**
 * Resolve the Vertex location for a Claude request. Single-region endpoints serve Claude up to the Sonnet 4.6
 * generation; newer models are served only from `global` and the `us`/`eu` multi-regions, so the driver's default
 * region maps to those for 4.7+ models. A location the caller names in the model path is kept as-is.
 * https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
 */
export function resolveAnthropicVertexRegion(
    explicitRegion: string | undefined,
    driverRegion: string,
    model?: string,
): string {
    const region = explicitRegion ?? driverRegion;
    if (region === 'global' || region === 'us' || region === 'eu') return region;
    const prefix = region.split('-')[0];
    if (explicitRegion === undefined && model && isClaudeVersionGTE(model, 4, 7)) {
        return ANTHROPIC_MULTI_REGIONS[prefix] ?? 'global';
    }
    return ANTHROPIC_REGIONS[prefix] || region;
}

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

/**
 * Project, region and model combinations where an organization policy refused Claude structured outputs. Shared by
 * every model definition in the process, so each combination fails once and then uses the prompt path.
 */
export const claudeStructuredOutputDenials = new ClaudeStructuredOutputDenials();

function structuredOutputDenialKey(driver: VertexAIDriver, region: string | undefined, model: string): string {
    return ClaudeStructuredOutputDenials.key(
        driver.options.project,
        resolveAnthropicVertexRegion(region, driver.options.region, model),
        model,
    );
}

function logStructuredOutputDenial(driver: VertexAIDriver, model: string, error: unknown): void {
    driver.logger.warn(
        { model, error: error instanceof Error ? error.message : String(error) },
        '[VertexAI] Claude structured outputs are not allowed for this model; using the prompt-based result schema',
    );
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
        // Same decision as getClaudePayload, which sees the model name without its location path.
        const { region, options: resolvedOptions } = resolveVertexAIModelPath(options);
        return formatClaudePrompt(segments, options, _driver.logger, {
            nativeResultSchema:
                usesClaudeNativeResultSchema(resolvedOptions) &&
                !claudeStructuredOutputDenials.has(structuredOutputDenialKey(_driver, region, resolvedOptions.model)),
        });
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const { region, options: resolvedOptions } = resolveVertexAIModelPath(options);
        const client = await driver.getAnthropicClient(region, resolvedOptions.httpTimeout, resolvedOptions.model);
        const model_options = resolvedOptions.model_options as VertexAIClaudeOptions | undefined;
        if (
            model_options?._option_id !== undefined &&
            model_options?._option_id !== 'vertexai-claude' &&
            model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: resolvedOptions.model_options }, 'Unexpected option id');
        }
        const execute = (allowNativeResultSchema: boolean) =>
            executeClaudeCompletion(
                client,
                prompt,
                resolvedOptions,
                driver.logger,
                driver.provider,
                signal ? { signal } : undefined,
                undefined,
                { allowNativeResultSchema },
            );
        if (!usesClaudeNativeResultSchema(resolvedOptions)) return execute(false);
        const denialKey = structuredOutputDenialKey(driver, region, resolvedOptions.model);
        if (claudeStructuredOutputDenials.has(denialKey)) return execute(false);
        try {
            return await execute(true);
        } catch (error) {
            if (!isClaudeStructuredOutputPolicyDenial(error)) throw error;
            claudeStructuredOutputDenials.add(denialKey);
            logStructuredOutputDenial(driver, resolvedOptions.model, error);
            return execute(false);
        }
    }

    async requestTextCompletionStream(
        driver: VertexAIDriver,
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        const { region, options: resolvedOptions } = resolveVertexAIModelPath(options);
        const client = await driver.getAnthropicClient(region, resolvedOptions.httpTimeout, resolvedOptions.model);
        const model_options = resolvedOptions.model_options as VertexAIClaudeOptions | undefined;
        if (
            model_options?._option_id !== undefined &&
            model_options?._option_id !== 'vertexai-claude' &&
            model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: resolvedOptions.model_options }, 'Unexpected option id');
        }
        const stream = (allowNativeResultSchema: boolean) =>
            streamClaudeCompletion(
                client,
                prompt,
                resolvedOptions,
                driver.logger,
                driver.provider,
                signal ? { signal } : undefined,
                undefined,
                { allowNativeResultSchema },
            );
        if (!usesClaudeNativeResultSchema(resolvedOptions)) return stream(false);
        const denialKey = structuredOutputDenialKey(driver, region, resolvedOptions.model);
        if (claudeStructuredOutputDenials.has(denialKey)) return stream(false);
        return withClaudeStructuredOutputFallback(
            await stream(true),
            () => stream(false),
            (error) => {
                claudeStructuredOutputDenials.add(denialKey);
                logStructuredOutputDenial(driver, resolvedOptions.model, error);
            },
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
