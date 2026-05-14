import {
    type AIModel, type Completion, type CompletionChunkObject, type ExecutionOptions, type LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptSegment,
    type VertexAIClaudeOptions,
} from "@llumiverse/core";
import type { ClaudePrompt } from "../../shared/claude-messages.js";
import {
    executeClaudeCompletion,
    formatAnthropicLlumiverseError,
    formatClaudePrompt,
    streamClaudeCompletion,
} from "../../shared/claude-messages.js";
import type { VertexAIDriver } from "../index.js";
import type { ModelDefinition } from "../models.js";

export const ANTHROPIC_REGIONS: Record<string, string> = {
    us: "us-east5",
    europe: "europe-west1",
    global: "global",
}

export const NON_GLOBAL_ANTHROPIC_MODELS = [
    "claude-3-5",
    "claude-3",
];

/**
 * Parse a VertexAI model path (e.g. "locations/us-east5/claude-3-5-sonnet") into
 * its region and model name components.
 */
function resolveVertexAIModelPath(options: ExecutionOptions): { modelName: string; region: string | undefined; options: ExecutionOptions } {
    const splits = options.model.split("/");
    let region: string | undefined;
    if (splits[0] === "locations" && splits.length >= 2) {
        region = splits[1];
    }
    const modelName = splits[splits.length - 1];
    return { modelName, region, options: { ...options, model: modelName } };
}

export class ClaudeModelDefinition implements ModelDefinition<ClaudePrompt> {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } satisfies AIModel;
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: ExecutionOptions): Promise<ClaudePrompt> {
        return formatClaudePrompt(segments, options);
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<Completion> {
        const { region, options: resolvedOptions } = resolveVertexAIModelPath(options);
        const client = await driver.getAnthropicClient(region);
        const model_options = resolvedOptions.model_options as VertexAIClaudeOptions | undefined;
        if (model_options?._option_id !== undefined &&
            model_options?._option_id !== "vertexai-claude" &&
            model_options?._option_id !== "text-fallback") {
            driver.logger.debug({ options: resolvedOptions.model_options }, "Unexpected option id");
        }
        return executeClaudeCompletion(client, prompt, resolvedOptions);
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const { region, options: resolvedOptions } = resolveVertexAIModelPath(options);
        const client = await driver.getAnthropicClient(region);
        const model_options = resolvedOptions.model_options as VertexAIClaudeOptions | undefined;
        if (model_options?._option_id !== undefined &&
            model_options?._option_id !== "vertexai-claude" &&
            model_options?._option_id !== "text-fallback") {
            driver.logger.debug({ options: resolvedOptions.model_options }, "Unexpected option id");
        }
        return streamClaudeCompletion(client, prompt, resolvedOptions);
    }

    formatLlumiverseError(
        _driver: VertexAIDriver,
        error: unknown,
        context: LlumiverseErrorContext
    ): LlumiverseError {
        return formatAnthropicLlumiverseError(error, context);
    }
}
