import Anthropic from '@anthropic-ai/sdk';
import type { AnthropicClaudeOptions } from "@llumiverse/common";
import {
    AbstractDriver,
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    LlumiverseError,
    type LlumiverseErrorContext,
    type ModelSearchPayload,
    ModelType,
    type PromptSegment,
    Providers,
} from "@llumiverse/core";
import {
    buildClaudeStreamingConversation,
    type ClaudePrompt,
    executeClaudeCompletion,
    formatAnthropicLlumiverseError,
    formatClaudePrompt,
    streamClaudeCompletion,
} from "../shared/claude-messages.js";

export interface AnthropicDriverOptions extends DriverOptions {
    apiKey?: string;
    baseURL?: string;
}

export class AnthropicDriver extends AbstractDriver<AnthropicDriverOptions, ClaudePrompt> {

    provider = Providers.anthropic;
    client: Anthropic;

    constructor(opts: AnthropicDriverOptions) {
        super(opts);
        this.client = new Anthropic({ apiKey: opts.apiKey, ...(opts.baseURL ? { baseURL: opts.baseURL } : {}) });
    }

    protected formatPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<ClaudePrompt> {
        return formatClaudePrompt(segments, opts);
    }

    async requestTextCompletion(prompt: ClaudePrompt, options: ExecutionOptions): Promise<Completion> {
        const model_options = options.model_options as AnthropicClaudeOptions | undefined;
        if (model_options?._option_id !== undefined && model_options?._option_id !== "anthropic-claude") {
            this.logger.debug({ options: options.model_options }, "Unexpected option id");
        }
        return executeClaudeCompletion(this.client, prompt, options);
    }

    async requestTextCompletionStream(prompt: ClaudePrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const model_options = options.model_options as AnthropicClaudeOptions | undefined;
        if (model_options?._option_id !== undefined && model_options?._option_id !== "anthropic-claude") {
            this.logger.debug({ options: options.model_options }, "Unexpected option id");
        }
        return streamClaudeCompletion(this.client, prompt, options);
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        const page = await this.client.models.list({ limit: 1000 });
        return page.data.map((m) => ({
            id: m.id,
            name: m.display_name ?? m.id,
            provider: Providers.anthropic,
            type: ModelType.Text,
            can_stream: true,
        } satisfies AIModel));
    }

    async listEmbeddingModels(): Promise<AIModel[]> {
        return [];
    }

    async validateConnection(): Promise<boolean> {
        try {
            await this.client.models.list({ limit: 1 });
            return true;
        } catch {
            return false;
        }
    }

    async generateEmbeddings(_opts: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new LlumiverseError(
            '[anthropic] Anthropic does not support embeddings',
            false,
            { provider: Providers.anthropic, model: _opts.model ?? 'unknown', operation: 'execute' },
            undefined,
        );
    }

    buildStreamingConversation(
        prompt: ClaudePrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions
    ): ClaudePrompt {
        return buildClaudeStreamingConversation(prompt, result, toolUse, options);
    }

    formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        return formatAnthropicLlumiverseError(error, context);
    }
}
