import {
    type AIModel,
    type CompletionStream,
    type DecisionsOptions,
    type DecisionsResult,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionResponse,
    isXAIGrokImageModel,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import OpenAI from 'openai';
import type { OpenAIChatCompletionsDriverOptions, xAiDriverOptions } from '../driver-options.js';
import {
    type OpenAIChatCompletionsPrompt,
    OpenAISDKChatCompletionsDriverBase,
} from '../openai/openai_chat_completions.js';
import { NON_OPENAI_ACCOUNT_OPTIONS } from '../openai/openai_compatible.js';
import { xAIDriver } from './index.js';

/**
 * xAI on its Chat Completions API (`openai_api: 'chat_completions'`), which xAI still serves next to Responses.
 * Text models run through the OpenAI SDK at `/chat/completions`. Image models, model listing, embeddings and decisions
 * stay with the Responses driver, so they are the same whichever API text inference uses.
 */
export class xAIChatCompletionsDriver extends OpenAISDKChatCompletionsDriverBase<
    xAiDriverOptions & OpenAIChatCompletionsDriverOptions
> {
    readonly provider = Providers.xai;
    service: OpenAI;
    private readonly responsesDriver: xAIDriver;

    constructor(options: xAiDriverOptions) {
        super({ ...options, resultSchemaMode: 'response_format' });
        this.responsesDriver = new xAIDriver(options);
        this.service = new OpenAI({
            apiKey: options.apiKey,
            ...NON_OPENAI_ACCOUNT_OPTIONS,
            // The Responses driver's client already holds the endpoint override or xAI's default.
            baseURL: this.responsesDriver.service.baseURL,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
    }

    override async execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<OpenAIChatCompletionsPrompt>> {
        if (!isXAIGrokImageModel(options.model)) return super.execute(segments, options, signal);
        // Image generation keeps no Chat Completions conversation; its result is the Responses driver's.
        return (await this.responsesDriver.execute(
            segments,
            options,
            signal,
        )) as unknown as ExecutionResponse<OpenAIChatCompletionsPrompt>;
    }

    override async stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<OpenAIChatCompletionsPrompt>> {
        if (!isXAIGrokImageModel(options.model)) return super.stream(segments, options, signal);
        return (await this.responsesDriver.stream(
            segments,
            options,
            signal,
        )) as unknown as CompletionStream<OpenAIChatCompletionsPrompt>;
    }

    listModels(): Promise<AIModel[]> {
        return this.responsesDriver.listModels();
    }

    validateConnection(): Promise<boolean> {
        return this.responsesDriver.validateConnection();
    }

    generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return this.responsesDriver.generateEmbeddings(options);
    }

    override decide(options: DecisionsOptions, signal?: AbortSignal): Promise<DecisionsResult> {
        return this.responsesDriver.decide(options, signal);
    }

    protected override destroyProviderResources(): void {
        this.responsesDriver.destroy();
    }
}
