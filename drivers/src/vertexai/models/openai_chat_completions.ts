import { type AIModel, ModelType } from '@llumiverse/core';
import {
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsProtocolOptions,
    type OpenAIChatCompletionsResponse,
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
    ): Promise<OpenAIChatCompletionsResponse> {
        const client = this.getClient(driver);
        return (await client.post(this.endpoint, {
            payload,
        })) as OpenAIChatCompletionsResponse;
    }

    protected async postChatCompletionStream(
        driver: VertexAIDriver,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<ReadableStream> {
        const client = this.getClient(driver);
        return (await client.post(this.endpoint, {
            payload,
            reader: 'sse',
        })) as ReadableStream;
    }

    private getClient(driver: VertexAIDriver) {
        if (!this.vertexOptions.region) {
            return driver.getFetchClient();
        }
        if (this.vertexOptions.endpointRegion) {
            return driver.getFetchClientForRegion(
                this.vertexOptions.region,
                this.vertexOptions.apiVersion,
                this.vertexOptions.endpointRegion,
            );
        }
        return driver.getFetchClientForRegion(this.vertexOptions.region, this.vertexOptions.apiVersion);
    }

    private get endpoint(): string {
        return this.vertexOptions.endpointPath || 'endpoints/openapi/chat/completions';
    }
}
