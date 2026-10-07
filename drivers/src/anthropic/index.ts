import Anthropic from '@anthropic-ai/sdk';
import type { AnthropicClaudeOptions } from '@llumiverse/common';
import type { ConversationDocument, ConversationModelSwitchProjection, ModelTarget } from '@llumiverse/conversation';
import { parseConversationDocument } from '@llumiverse/conversation';
import { ModelTargetSchema } from '@llumiverse/conversation/schemas';
import {
    type AIModel,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalHostCapabilities,
    type CanonicalModelSwitchCountResult,
    type CanonicalModelSwitchProjectionControls,
    type CanonicalStreamOpenOptions,
    type Completion,
    type DriverCompletionStream,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    getModelCapabilities,
    LlumiverseError,
    type LlumiverseErrorContext,
    type ModelSearchPayload,
    ModelType,
    modelModalitiesToArray,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import type { AnthropicDriverOptions } from '../driver-options.js';
import {
    buildClaudeStreamingConversation,
    type ClaudePrompt,
    executeCanonicalClaudeCompletion,
    executeCanonicalClaudeContext,
    executeClaudeCompletion,
    formatAnthropicLlumiverseError,
    formatClaudeDebugPrompt,
    formatClaudePrompt,
    streamCanonicalClaudeContextEvents,
    streamCanonicalClaudeEvents,
    streamClaudeCompletion,
} from '../shared/claude-messages.js';
import {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
} from '../shared/claude-messages-conversation-adapter.js';
import { countAnthropicModelSwitchRequest } from './canonical-model-switch-count.js';
import { countAnthropicIndexedRequest } from './indexed-count.js';
import {
    executeAnthropicIndexedRequest,
    prepareAnthropicIndexedRequest,
    streamAnthropicIndexedRequest,
} from './indexed-request.js';

export type { AnthropicDriverOptions } from '../driver-options.js';

export class AnthropicDriver extends AbstractDriver<AnthropicDriverOptions, ClaudePrompt> {
    provider = Providers.anthropic;
    client: Anthropic;

    /** @internal Native indexed compiler and transport retain the normal configured client. */
    prepareIndexedTextRequest(...args: Parameters<typeof prepareAnthropicIndexedRequest>) {
        return prepareAnthropicIndexedRequest(...args);
    }

    executeCommittedIndexedTextRequest(
        ...args: [input: Parameters<typeof executeAnthropicIndexedRequest>[1], host?: CanonicalHostCapabilities]
    ) {
        return executeAnthropicIndexedRequest(
            this.client,
            args[0],
            args[1],
            this.getDriverRequestOptions(args[0].options, args[0].signal),
        );
    }

    streamCommittedIndexedTextRequest(
        input: Parameters<typeof streamAnthropicIndexedRequest>[1],
        host?: CanonicalHostCapabilities,
    ) {
        return streamAnthropicIndexedRequest(
            this.client,
            input,
            host,
            this.getDriverRequestOptions(input.options, input.signal),
        );
    }

    countIndexedNativeRequest(nativeRequest: unknown, target: ModelTarget, signal?: AbortSignal) {
        return countAnthropicIndexedRequest(this.client, nativeRequest, target, signal);
    }

    override async resolveCanonicalModelSwitchTarget(
        model: string,
        options?: ModelTarget['options'],
    ): Promise<ModelTarget | undefined> {
        return ModelTargetSchema.parse({
            provider: this.provider,
            protocol: CLAUDE_MESSAGES_PROTOCOL,
            model,
            adapter_version: CLAUDE_MESSAGES_ADAPTER_VERSION,
            ...(options === undefined ? {} : { options: structuredClone(options) }),
        });
    }

    override async countCanonicalModelSwitchNativeRequest(
        nativeRequest: unknown,
        target: ModelTarget,
        signal?: AbortSignal,
    ): Promise<CanonicalModelSwitchCountResult> {
        return countAnthropicModelSwitchRequest(this.client, nativeRequest, target, signal);
    }

    override async projectCanonicalModelSwitchRequest(
        document: ConversationDocument,
        target: ModelTarget,
        operation: 'execute' | 'stream',
        controls?: CanonicalModelSwitchProjectionControls,
    ): Promise<ConversationModelSwitchProjection> {
        if (controls && Object.values(controls).some((value) => value !== undefined)) {
            return { status: 'unsupported', reason: 'Claude switch requires an explicit request-control policy' };
        }
        const ownedDocument = parseConversationDocument(document);
        const ownedTarget = ModelTargetSchema.parse(structuredClone(target));
        const ownedOperation = operation;
        if (ownedTarget.provider !== this.provider) {
            return { status: 'unsupported', reason: 'Model switch target provider differs from configured driver' };
        }
        const { compileClaudeModelSwitchRequest, ClaudeModelSwitchUnsupportedError } = await import(
            '../shared/claude-model-switch.js'
        );
        try {
            return {
                status: 'compiled',
                native_request: await compileClaudeModelSwitchRequest({
                    document: ownedDocument,
                    target: ownedTarget,
                    operation: ownedOperation,
                }),
            };
        } catch (error: unknown) {
            if (error instanceof ClaudeModelSwitchUnsupportedError) {
                return { status: 'unsupported', reason: error.message };
            }
            throw error;
        }
    }

    protected supportsCanonicalConversation(_options: ExecutionOptions): boolean {
        return true;
    }

    protected supportsCanonicalContextConversation(_options: CanonicalExecutionContextOptions): boolean {
        return true;
    }

    constructor(opts: AnthropicDriverOptions) {
        super(opts);
        this.client = new Anthropic({
            apiKey: opts.apiKey,
            ...(opts.baseURL ? { baseURL: opts.baseURL } : {}),
            fetch: this.getDriverFetch(),
            timeout: this.getDriverRequestTimeoutMs(),
        });
    }

    protected formatPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<ClaudePrompt> {
        return formatClaudePrompt(segments, opts, this.logger);
    }

    public formatDebugPrompt(prompt: ClaudePrompt): ClaudePrompt {
        return formatClaudeDebugPrompt(prompt);
    }

    async requestTextCompletion(
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const model_options = options.model_options as AnthropicClaudeOptions | undefined;
        if (model_options?._option_id !== undefined && model_options?._option_id !== 'anthropic-claude') {
            this.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        return executeClaudeCompletion(
            this.client,
            prompt,
            options,
            this.logger,
            this.provider,
            this.getDriverRequestOptions(options, signal),
        );
    }

    async requestCanonicalTextCompletion(
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionResponse> {
        return executeCanonicalClaudeCompletion(
            this.client,
            prompt,
            options,
            this.logger,
            this.provider,
            this.getDriverRequestOptions(options, signal),
            undefined,
            hostCapabilities,
        );
    }

    async requestCanonicalContextCompletion(
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionResponse> {
        return executeCanonicalClaudeContext(
            this.client,
            options,
            this.logger,
            this.provider,
            this.getDriverRequestOptions(options, signal),
            undefined,
            hostCapabilities,
        );
    }

    async requestTextCompletionStream(
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        const model_options = options.model_options as AnthropicClaudeOptions | undefined;
        if (model_options?._option_id !== undefined && model_options?._option_id !== 'anthropic-claude') {
            this.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        return streamClaudeCompletion(
            this.client,
            prompt,
            options,
            this.logger,
            this.provider,
            this.getDriverRequestOptions(options, signal),
        );
    }

    async requestCanonicalTextCompletionEventStream(
        prompt: ClaudePrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionEventStream> {
        return streamCanonicalClaudeEvents(
            this.client,
            prompt,
            options,
            open,
            this.logger,
            this.provider,
            this.getDriverRequestOptions(options, signal),
            undefined,
            hostCapabilities,
        );
    }

    async requestCanonicalContextCompletionEventStream(
        options: CanonicalExecutionContextOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionEventStream> {
        return streamCanonicalClaudeContextEvents(
            this.client,
            options,
            open,
            this.logger,
            this.provider,
            this.getDriverRequestOptions(options, signal),
            undefined,
            hostCapabilities,
        );
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        const page = await this.client.models.list({ limit: 1000 });
        return page.data.map((m) => {
            const capabilities = getModelCapabilities(m.id, this.provider);
            return {
                id: m.id,
                name: m.display_name ?? m.id,
                provider: Providers.anthropic,
                type: ModelType.Text,
                can_stream: true,
                is_multimodal: capabilities.input.image === true,
                input_modalities: modelModalitiesToArray(capabilities.input),
                output_modalities: modelModalitiesToArray(capabilities.output),
                tool_support: capabilities.tool_support,
            } satisfies AIModel;
        });
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
        options: ExecutionOptions,
    ): ClaudePrompt {
        return buildClaudeStreamingConversation(prompt, result, toolUse, options);
    }

    formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        return formatAnthropicLlumiverseError(error, context);
    }
}
