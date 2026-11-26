import {
    type AIModel,
    type CompletionStream,
    type DecisionsOptions,
    type DecisionsResult,
    type Driver,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionResponse,
    type ModelSearchPayload,
    type PromptSegment,
    type Providers,
    resolveModelProfile,
} from '@llumiverse/core';
import OpenAI from 'openai';
import { OpenAIResponsesDriverBase } from './index.js';
import { NON_OPENAI_ACCOUNT_OPTIONS } from './openai_compatible.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

/** Where a provider serves the OpenAI Responses API. */
export interface ProviderResponsesEndpoint {
    apiKey: string;
    /** The base URL the OpenAI SDK appends `/responses` to. */
    baseURL: string;
    defaultHeaders?: Record<string, string>;
}

/**
 * A provider's driver configured for the Responses API (`openai_api: 'responses'`), for a provider whose own driver
 * calls Chat Completions. Text inference runs through the OpenAI SDK at the provider's Responses endpoint. Everything
 * else (image models, model listing, connection checks, embeddings, decisions) stays with the provider's own driver,
 * so it is the same whichever API text inference uses.
 */
export abstract class ProviderResponsesDriver extends OpenAIResponsesDriverBase {
    abstract readonly provider: Providers.openrouter | Providers.groq | Providers.togetherai;
    service: OpenAI;

    protected constructor(
        options: DriverOptions,
        /** The provider's own driver, which serves everything but inference. */
        protected readonly providerDriver: Driver,
        endpoint: ProviderResponsesEndpoint,
    ) {
        super(options);
        // The OpenAI SDK would otherwise read OPENAI_API_KEY and send the server's OpenAI key to this provider.
        if (!endpoint.apiKey) throw new Error('apiKey is required');
        this.service = new OpenAI({
            apiKey: endpoint.apiKey,
            ...NON_OPENAI_ACCOUNT_OPTIONS,
            baseURL: endpoint.baseURL,
            defaultHeaders: endpoint.defaultHeaders,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
    }

    /** Image models run through the provider's own driver (see {@link execute}), so none reaches this one. */
    override isImageModel(_model: string): boolean {
        return false;
    }

    private isProviderImageModel(model: string): boolean {
        return resolveModelProfile(model, this.provider).family === 'image';
    }

    override async execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<ResponseInputItem[]>> {
        if (!this.isProviderImageModel(options.model)) return super.execute(segments, options, signal);
        // An image result keeps the provider driver's conversation, which no Responses turn continues.
        return (await this.providerDriver.execute(segments, options, signal)) as unknown as ExecutionResponse<
            ResponseInputItem[]
        >;
    }

    override async stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<ResponseInputItem[]>> {
        if (!this.isProviderImageModel(options.model)) return super.stream(segments, options, signal);
        return (await this.providerDriver.stream(segments, options, signal)) as unknown as CompletionStream<
            ResponseInputItem[]
        >;
    }

    override listModels(params?: ModelSearchPayload): Promise<AIModel[]> {
        return this.providerDriver.listModels(params) as Promise<AIModel[]>;
    }

    override validateConnection(): Promise<boolean> {
        return this.providerDriver.validateConnection();
    }

    override generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return this.providerDriver.generateEmbeddings(options);
    }

    override decide(options: DecisionsOptions, signal?: AbortSignal): Promise<DecisionsResult> {
        return this.providerDriver.decide(options, signal);
    }

    protected override destroyProviderResources(): void {
        this.providerDriver.destroy();
    }
}
