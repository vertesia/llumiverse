import { AnthropicBedrockMantle, type BedrockMantleClientOptions } from '@anthropic-ai/bedrock-sdk';
import { getTokenProvider } from '@aws/bedrock-token-generator';
import {
    type AIModel,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    type Completion,
    type DriverCompletionStream,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    getBedrockMantleModelInfo,
    getBedrockMantleProtocol,
    isEmbeddingModel,
    type LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import type OpenAI from 'openai';
import { BedrockOpenAI } from 'openai';
import type { BedrockMantleDriverOptions } from '../driver-options.js';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import {
    buildOpenAIChatCompletionsStreamingConversation,
    type OpenAIChatCompletionsPrompt,
    OpenAISDKChatCompletionsProtocol,
} from '../openai/openai_chat_completions.js';
import { formatOpenAIDebugPrompt } from '../openai/openai_format.js';
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
import { resolveModelListingMetadata } from '../shared/model-listing.js';

export type { BedrockMantleDriverOptions } from '../driver-options.js';

type BedrockMantleResponsesPrompt = OpenAI.Responses.ResponseInputItem[];
export type BedrockMantlePrompt = BedrockMantleResponsesPrompt | OpenAIChatCompletionsPrompt | ClaudePrompt;

class BedrockMantleResponsesDelegate extends OpenAIResponsesDriverBase {
    readonly provider = Providers.bedrock_mantle;
    service: OpenAI;

    constructor(opts: BedrockMantleDriverOptions, service: OpenAI) {
        super(opts);
        this.service = service;
    }
}

function modelDisplayName(model: string): string {
    const modelName = model.slice(model.indexOf('.') + 1);
    if (model.startsWith('openai.')) return modelName.replace(/^gpt-/i, 'GPT-');
    if (model.startsWith('xai.')) return modelName.replace(/^grok-/i, 'Grok ');
    return modelName;
}

function isResponsesPrompt(prompt: BedrockMantlePrompt): prompt is BedrockMantleResponsesPrompt {
    return Array.isArray(prompt);
}

function isChatCompletionsPrompt(prompt: BedrockMantlePrompt): prompt is OpenAIChatCompletionsPrompt {
    return (
        !Array.isArray(prompt) && '_is_openai_chat_completions' in prompt && prompt._is_openai_chat_completions === true
    );
}

function requireResponsesPrompt(prompt: BedrockMantlePrompt): BedrockMantleResponsesPrompt {
    if (!isResponsesPrompt(prompt)) {
        throw new TypeError('Bedrock Mantle Responses models require an OpenAI Responses prompt');
    }
    return prompt;
}

function requireChatCompletionsPrompt(prompt: BedrockMantlePrompt): OpenAIChatCompletionsPrompt {
    if (!isChatCompletionsPrompt(prompt)) {
        throw new TypeError('Bedrock Mantle Chat Completions models require a Chat Completions prompt');
    }
    return prompt;
}

function requireClaudePrompt(prompt: BedrockMantlePrompt): ClaudePrompt {
    if (isResponsesPrompt(prompt) || isChatCompletionsPrompt(prompt)) {
        throw new TypeError('Bedrock Mantle Claude models require an Anthropic Messages prompt');
    }
    return prompt;
}

export function isBedrockMantleModel(model: string): boolean {
    return getBedrockMantleProtocol(model) !== undefined;
}

export class BedrockMantleDriver extends AbstractDriver<BedrockMantleDriverOptions, BedrockMantlePrompt> {
    service: BedrockOpenAI;
    private readonly responsesDelegate: BedrockMantleResponsesDelegate;
    private readonly chatCompletionsProtocol: OpenAISDKChatCompletionsProtocol;
    private readonly alignedChatCompletionsProtocol: OpenAISDKChatCompletionsProtocol;
    private readonly anthropicService: AnthropicBedrockMantle;
    readonly provider = Providers.bedrock_mantle;

    protected supportsCanonicalConversation(options: ExecutionOptions): boolean {
        const protocol = getBedrockMantleProtocol(options.model);
        return protocol === 'responses' || protocol === 'chat_completions' || protocol === 'messages';
    }

    protected supportsCanonicalContextConversation(options: CanonicalExecutionContextOptions): boolean {
        const protocol = getBedrockMantleProtocol(options.model);
        return protocol === 'responses' || protocol === 'chat_completions' || protocol === 'messages';
    }

    constructor(opts: BedrockMantleDriverOptions) {
        super(opts);

        const bedrockTokenProvider = opts.credentials
            ? getTokenProvider({ region: opts.region, credentials: opts.credentials })
            : getTokenProvider({ region: opts.region });
        const driverFetch = this.getDriverFetch();
        const timeout = this.getDriverRequestTimeoutMs();
        const v1BaseURL = `https://bedrock-mantle.${opts.region}.api.aws/v1`;

        this.service = new BedrockOpenAI({
            baseURL: v1BaseURL,
            awsRegion: opts.region,
            bedrockTokenProvider,
            fetch: driverFetch,
            timeout,
        });
        const responsesService = new BedrockOpenAI({
            baseURL: `https://bedrock-mantle.${opts.region}.api.aws/openai/v1`,
            awsRegion: opts.region,
            bedrockTokenProvider,
            fetch: driverFetch,
            timeout,
        });
        this.responsesDelegate = new BedrockMantleResponsesDelegate(opts, responsesService);
        this.chatCompletionsProtocol = new OpenAISDKChatCompletionsProtocol({
            resultSchemaMode: 'response_format',
            toolSchemaMode: 'compatible',
            resolveRequestOptions: (options, signal) => this.getDriverRequestOptions(options, signal),
        });
        this.alignedChatCompletionsProtocol = new OpenAISDKChatCompletionsProtocol({
            resultSchemaMode: 'response_format',
            includeResultSchemaInPrompt: true,
            toolSchemaMode: 'compatible',
            resolveRequestOptions: (options, signal) => this.getDriverRequestOptions(options, signal),
        });

        const credentials = opts.credentials;
        const credentialOptions: BedrockMantleClientOptions =
            typeof credentials === 'function'
                ? { providerChainResolver: async () => credentials }
                : credentials
                  ? {
                        awsAccessKey: credentials.accessKeyId,
                        awsSecretAccessKey: credentials.secretAccessKey,
                        awsSessionToken: credentials.sessionToken,
                    }
                  : {};
        this.anthropicService = new AnthropicBedrockMantle({
            awsRegion: opts.region,
            fetch: driverFetch,
            timeout,
            ...credentialOptions,
        });
    }

    private getChatCompletionsProtocol(model: string): OpenAISDKChatCompletionsProtocol {
        return model.toLowerCase().includes('mistral.magistral-')
            ? this.alignedChatCompletionsProtocol
            : this.chatCompletionsProtocol;
    }

    protected async formatPrompt(segments: PromptSegment[], options: ExecutionOptions): Promise<BedrockMantlePrompt> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.createPrompt(segments, options);
            case 'chat_completions':
                return this.getChatCompletionsProtocol(options.model).createPrompt(this, segments, options);
            case 'messages':
                return formatClaudePrompt(segments, options, this.logger);
            default:
                throw new Error(`Unsupported Bedrock Mantle model: ${options.model}`);
        }
    }

    public formatDebugPrompt(prompt: BedrockMantlePrompt): BedrockMantlePrompt {
        if (isResponsesPrompt(prompt)) return formatOpenAIDebugPrompt(prompt);
        if (isChatCompletionsPrompt(prompt)) return prompt;
        return formatClaudeDebugPrompt(prompt);
    }

    requestTextCompletion(
        prompt: BedrockMantlePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.requestTextCompletion(requireResponsesPrompt(prompt), options, signal);
            case 'chat_completions':
                return this.getChatCompletionsProtocol(options.model).requestTextCompletion(
                    this,
                    requireChatCompletionsPrompt(prompt),
                    options,
                    signal,
                );
            case 'messages':
                return executeClaudeCompletion(
                    this.anthropicService,
                    requireClaudePrompt(prompt),
                    options,
                    undefined,
                    'anthropic',
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error(`Unsupported Bedrock Mantle model: ${options.model}`);
        }
    }

    requestCanonicalTextCompletion(
        prompt: BedrockMantlePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.requestCanonicalTextCompletion(
                    requireResponsesPrompt(prompt),
                    options,
                    signal,
                );
            case 'chat_completions':
                return this.getChatCompletionsProtocol(options.model).requestCanonicalTextCompletion(
                    this,
                    requireChatCompletionsPrompt(prompt),
                    options,
                    signal,
                );
            case 'messages':
                return executeCanonicalClaudeCompletion(
                    this.anthropicService,
                    requireClaudePrompt(prompt),
                    options,
                    undefined,
                    this.provider,
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error(`Unsupported Bedrock Mantle model: ${options.model}`);
        }
    }

    requestCanonicalContextCompletion(
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.requestCanonicalContextCompletion(options, signal);
            case 'chat_completions':
                return this.getChatCompletionsProtocol(options.model).requestCanonicalContextCompletion(
                    this,
                    options,
                    signal,
                );
            case 'messages':
                return executeCanonicalClaudeContext(
                    this.anthropicService,
                    options,
                    undefined,
                    this.provider,
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error('Unsupported Bedrock Mantle model for canonical context execution');
        }
    }

    requestTextCompletionStream(
        prompt: BedrockMantlePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.requestTextCompletionStream(
                    requireResponsesPrompt(prompt),
                    options,
                    signal,
                );
            case 'chat_completions':
                return this.getChatCompletionsProtocol(options.model).requestTextCompletionStream(
                    this,
                    requireChatCompletionsPrompt(prompt),
                    options,
                    signal,
                );
            case 'messages':
                return streamClaudeCompletion(
                    this.anthropicService,
                    requireClaudePrompt(prompt),
                    options,
                    undefined,
                    'bedrock-mantle',
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error(`Unsupported Bedrock Mantle model: ${options.model}`);
        }
    }

    requestCanonicalTextCompletionEventStream(
        prompt: BedrockMantlePrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.requestCanonicalTextCompletionEventStream(
                    requireResponsesPrompt(prompt),
                    options,
                    signal,
                    open,
                );
            case 'chat_completions':
                return this.getChatCompletionsProtocol(options.model).requestCanonicalTextCompletionEventStream(
                    this,
                    requireChatCompletionsPrompt(prompt),
                    options,
                    signal,
                    open,
                );
            case 'messages':
                return streamCanonicalClaudeEvents(
                    this.anthropicService,
                    requireClaudePrompt(prompt),
                    options,
                    open,
                    undefined,
                    this.provider,
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error(`Unsupported Bedrock Mantle model: ${options.model}`);
        }
    }

    requestCanonicalContextCompletionEventStream(
        options: CanonicalExecutionContextOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.requestCanonicalContextCompletionEventStream(options, signal, open);
            case 'chat_completions':
                return this.getChatCompletionsProtocol(options.model).requestCanonicalContextCompletionEventStream(
                    this,
                    options,
                    signal,
                    open,
                );
            case 'messages':
                return streamCanonicalClaudeContextEvents(
                    this.anthropicService,
                    options,
                    open,
                    undefined,
                    this.provider,
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error('Unsupported Bedrock Mantle model for canonical context typed streaming');
        }
    }

    buildStreamingConversation(
        prompt: BedrockMantlePrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): BedrockMantlePrompt | undefined {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                return this.responsesDelegate.buildStreamingConversation(
                    requireResponsesPrompt(prompt),
                    result,
                    toolUse,
                    options,
                );
            case 'chat_completions':
                return buildOpenAIChatCompletionsStreamingConversation(
                    requireChatCompletionsPrompt(prompt),
                    result,
                    toolUse,
                    options,
                );
            case 'messages':
                return buildClaudeStreamingConversation(requireClaudePrompt(prompt), result, toolUse, options);
            default:
                return undefined;
        }
    }

    formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        if (getBedrockMantleProtocol(context.model) === 'messages') {
            return formatAnthropicLlumiverseError(error, context);
        }
        return this.responsesDelegate.formatLlumiverseError(error, context);
    }

    supportsStreaming(options: ExecutionOptions): Promise<boolean> {
        return this.canStream(options);
    }

    async listModels(): Promise<AIModel[]> {
        const models = (await this.service.models.list()).data;
        return models
            .flatMap((model) => {
                if (isEmbeddingModel(model, this.provider)) return [];
                const info = getBedrockMantleModelInfo(model.id);
                if (!info) return [];
                const modelMetadata = resolveModelListingMetadata(model.id, this.provider);
                return [
                    {
                        id: model.id,
                        name: modelDisplayName(model.id),
                        provider: this.provider,
                        owner: info.owner,
                        type: ModelType.Text,
                        can_stream: true,
                        ...modelMetadata,
                    } satisfies AIModel,
                ];
            })
            .sort((a, b) => a.id.localeCompare(b.id));
    }

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.models.list();
            return true;
        } catch {
            return false;
        }
    }

    generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return this.responsesDelegate.generateEmbeddings(options);
    }
}
