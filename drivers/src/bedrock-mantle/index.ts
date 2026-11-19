import { AnthropicBedrockMantle, type BedrockMantleClientOptions } from '@anthropic-ai/bedrock-sdk';
import { getTokenProvider } from '@aws/bedrock-token-generator';
import {
    type AIModel,
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
    resolveModelProfile,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import type OpenAI from 'openai';
import { BedrockOpenAI } from 'openai';
import type { BedrockMantleDriverOptions } from '../driver-options.js';
import {
    buildOpenAIResponsesStreamingConversation,
    formatOpenAIResponsesPrompt,
    generateOpenAIEmbeddings,
    type OpenAIImageContext,
    type OpenAIResponsesContext,
    OpenAIResponsesProtocol,
    requestOpenAIImageGeneration,
    requestOpenAIImageStream,
} from '../openai/index.js';
import {
    buildOpenAIChatCompletionsStreamingConversation,
    type OpenAIChatCompletionsPrompt,
    OpenAISDKChatCompletionsProtocol,
} from '../openai/openai_chat_completions.js';
import { formatOpenAICompatibleLlumiverseError } from '../openai/openai_compatible.js';
import { formatOpenAIDebugPrompt, formatOpenAILikeMultimodalPrompt } from '../openai/openai_format.js';
import { claudeApiVersionHeaders } from '../shared/claude-api-version.js';
import {
    buildClaudeStreamingConversation,
    type ClaudePrompt,
    executeClaudeCompletion,
    formatAnthropicLlumiverseError,
    formatClaudeDebugPrompt,
    formatClaudePrompt,
    streamClaudeCompletion,
} from '../shared/claude-messages.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';

export type { BedrockMantleDriverOptions } from '../driver-options.js';

type BedrockMantleResponsesPrompt = OpenAI.Responses.ResponseInputItem[];
export type BedrockMantlePrompt = BedrockMantleResponsesPrompt | OpenAIChatCompletionsPrompt | ClaudePrompt;

function modelDisplayName(model: string, owner: string): string {
    const modelName = model.slice(model.indexOf('.') + 1);
    if (model.startsWith('openai.')) return modelName.replace(/^gpt-/i, 'GPT-');
    if (model.startsWith('xai.')) return modelName.replace(/^grok-/i, 'Grok ');
    // IDs such as `deepseek.v3.1` hold only a version after the publisher; keep the publisher so the name is usable.
    if (/^[a-z]?\d/i.test(modelName)) return `${owner} ${modelName.charAt(0).toUpperCase()}${modelName.slice(1)}`;
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
    /** The OpenAI-compatible endpoint (`/openai/v1`): Responses, Images and Embeddings. */
    private readonly responsesService: BedrockOpenAI;
    private readonly responsesContext: OpenAIResponsesContext;
    private readonly imageContext: OpenAIImageContext;
    private readonly responsesProtocol: OpenAIResponsesProtocol;
    private readonly chatCompletionsProtocol: OpenAISDKChatCompletionsProtocol;
    private readonly alignedChatCompletionsProtocol: OpenAISDKChatCompletionsProtocol;
    private readonly anthropicService: AnthropicBedrockMantle;
    readonly provider = Providers.bedrock_mantle;

    constructor(opts: BedrockMantleDriverOptions) {
        super(opts);

        const bedrockTokenProvider = opts.credentials
            ? getTokenProvider({ region: opts.region, credentials: opts.credentials })
            : getTokenProvider({ region: opts.region });
        const driverFetch = this.getDriverFetch();
        const timeout = this.getDriverRequestTimeoutMs();
        const v1BaseURL = `https://bedrock-mantle.${opts.region}.api.aws/v1`;

        // Callers own retries, as for every other OpenAI SDK client in Llumiverse: the SDK's own retries would repeat a
        // failed request with backoff before the caller's retry policy sees it.
        this.service = new BedrockOpenAI({
            baseURL: v1BaseURL,
            awsRegion: opts.region,
            bedrockTokenProvider,
            fetch: driverFetch,
            maxRetries: 0,
            timeout,
        });
        this.responsesService = new BedrockOpenAI({
            baseURL: `https://bedrock-mantle.${opts.region}.api.aws/openai/v1`,
            awsRegion: opts.region,
            bedrockTokenProvider,
            fetch: driverFetch,
            maxRetries: 0,
            timeout,
        });
        this.responsesContext = {
            provider: this.provider,
            logger: this.logger,
            getResponsesService: () => this.responsesService,
            getResponsesRequestModel: (model) => model,
            getPromptCacheSourceModel: (model) => model.split('::').pop() ?? model,
        };
        this.imageContext = {
            getImageService: () => this.responsesService,
            getResponsesRequestModel: (model) => model,
            getImageSourceModel: (model) => model,
            requestOptions: (options, signal) => this.getDriverRequestOptions(options, signal),
            requestTimeoutMs: (httpTimeout) => this.getDriverRequestTimeoutMs(httpTimeout),
            fetch: driverFetch,
        };
        this.responsesProtocol = new OpenAIResponsesProtocol((options, signal) =>
            this.getDriverRequestOptions(options, signal),
        );
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
            defaultHeaders: claudeApiVersionHeaders(),
            ...credentialOptions,
        });
    }

    private getChatCompletionsProtocol(model: string): OpenAISDKChatCompletionsProtocol {
        return model.toLowerCase().includes('mistral.magistral-')
            ? this.alignedChatCompletionsProtocol
            : this.chatCompletionsProtocol;
    }

    /** Mantle serves OpenAI's image models (`openai.gpt-image-*`) through its OpenAI Images endpoint. */
    private isOpenAIImageModel(model: string): boolean {
        return resolveModelProfile(model, this.provider).family === 'image';
    }

    protected async formatPrompt(segments: PromptSegment[], options: ExecutionOptions): Promise<BedrockMantlePrompt> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                if (this.isOpenAIImageModel(options.model)) {
                    return formatOpenAILikeMultimodalPrompt(segments, {
                        ...options,
                        imageGeneration: true,
                        result_schema: undefined,
                    });
                }
                return formatOpenAIResponsesPrompt(this.responsesContext, segments, options);
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
                if (this.isOpenAIImageModel(options.model)) {
                    return requestOpenAIImageGeneration(
                        this.imageContext,
                        requireResponsesPrompt(prompt),
                        options,
                        signal,
                    );
                }
                return this.responsesProtocol.requestTextCompletion(
                    this.responsesContext,
                    requireResponsesPrompt(prompt),
                    options,
                    signal,
                );
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
                    this.logger,
                    this.provider,
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error(`Unsupported Bedrock Mantle model: ${options.model}`);
        }
    }

    requestTextCompletionStream(
        prompt: BedrockMantlePrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        switch (getBedrockMantleProtocol(options.model)) {
            case 'responses':
                if (this.isOpenAIImageModel(options.model)) {
                    return requestOpenAIImageStream(this.imageContext, requireResponsesPrompt(prompt), options, signal);
                }
                return this.responsesProtocol.requestTextCompletionStream(
                    this.responsesContext,
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
                    this.logger,
                    this.provider,
                    this.getDriverRequestOptions(options, signal),
                );
            default:
                throw new Error(`Unsupported Bedrock Mantle model: ${options.model}`);
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
                return buildOpenAIResponsesStreamingConversation(
                    this.provider,
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
                return buildClaudeStreamingConversation(
                    requireClaudePrompt(prompt),
                    result,
                    toolUse,
                    options,
                    this.provider,
                );
            default:
                return undefined;
        }
    }

    formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        if (getBedrockMantleProtocol(context.model) === 'messages') {
            return formatAnthropicLlumiverseError(error, context);
        }
        return formatOpenAICompatibleLlumiverseError(error, context) ?? super.formatLlumiverseError(error, context);
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
                        name: modelDisplayName(model.id, info.owner),
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
        return generateOpenAIEmbeddings(this.responsesService, options, this.provider, (error, context) =>
            this.formatLlumiverseError(error, context),
        );
    }
}
