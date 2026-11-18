import { type AIModel, type ExecutionOptions, ModelType } from '@llumiverse/core';
import type OpenAI from 'openai';
import type { Stream } from 'openai/streaming';
import { mergeOpenAIExtraBody } from '../../openai/extra_body.js';
import {
    normalizeOpenAIChatCompletionsResponse,
    normalizeOpenAIChatCompletionsStream,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsProtocolOptions,
    type OpenAIChatCompletionsResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
    toOpenAINonStreamingPayload,
    toOpenAIStreamingPayload,
} from '../../openai/openai_chat_completions.js';
import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

/**
 * Options for configuring the Vertex OpenAI-compatible model.
 */
export interface VertexOpenAIChatCompletionsOptions extends OpenAIChatCompletionsProtocolOptions {
    modelName: string;
    /** Custom endpoint path override (defaults to "endpoints/openapi/chat/completions") */
    endpointPath?: string;
    /** Streaming endpoint path when it differs from the unary endpoint. */
    streamingEndpointPath?: string;
    /** Regional publisher APIs such as Mistral take extension fields at the body root; MaaS retains the wrapper. */
    flattenExtraBody?: boolean;
    /** Region override for the Vertex AI endpoint. Useful when a model only exists in a specific region. */
    region?: string;
    /** Vertex API version for this OpenAI-compatible endpoint. */
    apiVersion?: string;
    /** Host region override; keeps the request location unchanged while selecting a different API hostname. */
    endpointRegion?: string;
}

/**
 * Vertex AI model definition for OpenAI-compatible chat completions endpoints.
 *
 * Generic OpenAI Chat Completions prompt conversion, payload construction, and
 * response parsing live in the shared OpenAI driver module. This adapter keeps
 * only Vertex-specific routing: region selection, endpoint path, and request
 * model name.
 */
export class OpenAIChatCompletionsModelDefinition
    extends OpenAIChatCompletionsProtocol<VertexAIDriver>
    implements ModelDefinition<OpenAIChatCompletionsPrompt>
{
    model: AIModel;
    private readonly vertexOptions: VertexOpenAIChatCompletionsOptions;

    constructor(options: VertexOpenAIChatCompletionsOptions) {
        super({ ...options, toolSchemaMode: options.toolSchemaMode ?? 'compatible' });
        this.vertexOptions = options;
        const modelName = options.modelName.split('/').pop() || options.modelName;
        this.model = {
            id: options.modelName,
            name: modelName,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } as AIModel;
    }

    protected async postChatCompletion(
        driver: VertexAIDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse> {
        const requestOptions = this.requestOptions(driver, options, signal);
        if (this.vertexOptions.endpointPath) {
            // Publisher model paths such as Mistral's `:rawPredict` are not OpenAI routes, so they use the SDK's
            // generic request with the same authentication.
            const client = this.getClient(driver, '');
            return (await client.post(`/${this.vertexOptions.endpointPath}`, {
                body: this.preparePayload(driver, { ...payload, stream: false }),
                ...requestOptions,
            })) as OpenAIChatCompletionsResponse;
        }
        const response = await this.getClient(driver).chat.completions.create(
            this.withMaaSExtraBody(
                toOpenAINonStreamingPayload({ ...payload, extra_body: undefined }, driver.logger),
                payload,
            ),
            requestOptions,
        );
        return preserveOpenAIChatCompletionsOriginalResponse(
            normalizeOpenAIChatCompletionsResponse(response),
            response,
        );
    }

    protected async postChatCompletionStream(
        driver: VertexAIDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream> {
        const requestOptions = this.requestOptions(driver, options, signal);
        if (this.vertexOptions.endpointPath) {
            const client = this.getClient(driver, '');
            const stream = (await client.post(
                `/${this.vertexOptions.streamingEndpointPath ?? this.vertexOptions.endpointPath}`,
                {
                    body: this.preparePayload(driver, { ...payload, stream: true }),
                    stream: true,
                    ...requestOptions,
                },
            )) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>;
            return openAIChatCompletionsStreamToSSE(normalizeOpenAIChatCompletionsStream(stream), () =>
                stream.controller.abort(),
            );
        }
        const stream = await this.getClient(driver).chat.completions.create(
            this.withMaaSExtraBody(
                toOpenAIStreamingPayload({ ...payload, extra_body: undefined }, driver.logger),
                payload,
            ),
            requestOptions,
        );
        return openAIChatCompletionsStreamToSSE(normalizeOpenAIChatCompletionsStream(stream), () =>
            stream.controller.abort(),
        );
    }

    /** Regional publisher APIs such as Mistral take extension fields at the body root. */
    private preparePayload(
        driver: VertexAIDriver,
        payload: OpenAIChatCompletionsPayload,
    ): OpenAIChatCompletionsPayload {
        if (!this.vertexOptions.flattenExtraBody) return payload;
        const { extra_body, ...request } = payload;
        return mergeOpenAIExtraBody(request, extra_body, driver.logger, payload.model);
    }

    /**
     * The OpenAI-compatible MaaS endpoint reads Google's extensions from a literal `extra_body` field (for example
     * Llama's `google.model_safety_settings`), so it is sent as a field rather than merged into the body root.
     */
    private withMaaSExtraBody<RequestT extends object>(
        request: RequestT,
        payload: OpenAIChatCompletionsPayload,
    ): RequestT {
        return payload.extra_body === undefined ? request : { ...request, extra_body: payload.extra_body };
    }

    private requestOptions(
        driver: VertexAIDriver,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): { timeout: number; signal?: AbortSignal } {
        return { timeout: driver.getRequestTimeoutMs(options.httpTimeout), ...(signal && { signal }) };
    }

    /** Without a model-path location, the driver's configured region is used. */
    private getClient(driver: VertexAIDriver, basePath?: string): OpenAI {
        return driver.getOpenAIClientForRegion(
            this.vertexOptions.region ?? driver.options.region,
            this.vertexOptions.apiVersion,
            this.vertexOptions.endpointRegion,
            basePath,
        );
    }
}
