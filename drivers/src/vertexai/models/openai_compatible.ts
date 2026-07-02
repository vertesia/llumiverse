import { type AIModel, type Completion, type ExecutionOptions, ModelType } from '@llumiverse/core';
import {
    OpenAICompletionsModelDefinitionBase,
    type OpenAICompletionsModelOptions,
    type OpenAICompletionsPayload,
    type OpenAICompletionsPrompt,
    type OpenAICompletionsResponse,
    stripOpenAICompletionsThinkBlocksFromCompletion,
} from '../../openai/openai_comp_completions.js';
import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

export type {
    OpenAICompletionsMessage as OpenAIMessage,
    OpenAICompletionsPrompt as OpenAIPrompt,
    OpenAICompletionsResponse as OpenAIResponse,
    OpenAICompletionsStreamResponse as OpenAIStreamResponse,
} from '../../openai/openai_comp_completions.js';

/**
 * Options for configuring the Vertex OpenAI-compatible model.
 */
export interface OpenAICompatibleOptions extends OpenAICompletionsModelOptions {
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
export class OpenAICompatibleModelDefinition
    extends OpenAICompletionsModelDefinitionBase<VertexAIDriver>
    implements ModelDefinition<OpenAICompletionsPrompt>
{
    model: AIModel;
    private readonly vertexOptions: OpenAICompatibleOptions;

    constructor(options: OpenAICompatibleOptions) {
        super(options);
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
        payload: OpenAICompletionsPayload,
    ): Promise<OpenAICompletionsResponse> {
        const client = this.getClient(driver);
        return (await client.post(this.endpoint, {
            payload,
        })) as OpenAICompletionsResponse;
    }

    protected async postChatCompletionStream(
        driver: VertexAIDriver,
        payload: OpenAICompletionsPayload,
    ): Promise<ReadableStream> {
        const client = this.getClient(driver);
        return (await client.post(this.endpoint, {
            payload,
            reader: 'sse',
        })) as ReadableStream;
    }

    preValidationProcessing(
        result: Completion,
        options: ExecutionOptions,
    ): { result: Completion; options: ExecutionOptions } {
        return { result: stripOpenAICompletionsThinkBlocksFromCompletion(result), options };
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
