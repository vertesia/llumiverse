import { isVertexOpenAIResponsesModel } from '@llumiverse/common';
import {
    type AIModel,
    type Completion,
    type DriverCompletionStream,
    type ExecutionOptions,
    type LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import {
    formatOpenAIResponsesPrompt,
    type OpenAIResponsesContext,
    type OpenAIResponsesPolicy,
    OpenAIResponsesProtocol,
} from '../../openai/index.js';
import { formatOpenAICompatibleLlumiverseError } from '../../openai/openai_compatible.js';
import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

export type VertexAIResponsesPrompt = OpenAI.Responses.ResponseInputItem[];

/**
 * Where Vertex's publisher Responses endpoints differ from OpenAI's.
 * https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/meta
 * https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/openai
 */
const VERTEX_RESPONSES_POLICY: OpenAIResponsesPolicy = {
    // Vertex constrains output to the schema without showing it to the model, and emits the properties
    // alphabetically; without the schema in the prompt, Muse fills the constrained fields with wrong or garbled
    // values. OpenAI's GPT models are the exception: OpenAI's Responses API enforces the schema natively, so they stay
    // native-only until measured on Vertex.
    supplementsSchemaInPrompt: (model) => !isVertexOpenAIResponsesModel(model),
    // Vertex lists reasoning summaries as unsupported for OpenAI's models.
    requestsReasoningSummary: (model) => !isVertexOpenAIResponsesModel(model),
};

/** The endpoint takes the bare model ID (`muse-wonder-1.0`), not the Vertex resource path. */
function publisherRequestModel(model: string): string {
    const name = model.split('/').pop() ?? model;
    const version = name.lastIndexOf('@');
    return version > -1 ? name.slice(0, version) : name;
}

/**
 * Vertex models served by a publisher's OpenAI-compatible Responses API
 * (`.../locations/{location}/publishers/{publisher}/v1/responses`), such as Meta Muse and OpenAI GPT.
 */
export class VertexAIResponsesModelDefinition implements ModelDefinition<VertexAIResponsesPrompt> {
    model: AIModel;

    constructor(
        modelName: string,
        private readonly publisher: string,
        private readonly region: string,
    ) {
        this.model = {
            id: modelName,
            name: modelName,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } as AIModel;
    }

    private context(driver: VertexAIDriver): OpenAIResponsesContext {
        return {
            provider: Providers.vertexai,
            logger: driver.logger,
            responsesPolicy: VERTEX_RESPONSES_POLICY,
            getResponsesService: () => driver.getResponsesClient(this.publisher, this.region),
            getResponsesRequestModel: publisherRequestModel,
            getPromptCacheSourceModel: (model) => model.split('::').pop() ?? model,
        };
    }

    private protocol(driver: VertexAIDriver): OpenAIResponsesProtocol {
        return new OpenAIResponsesProtocol((options, signal) => driver.getRequestOptions(options, signal));
    }

    createPrompt(
        driver: VertexAIDriver,
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<VertexAIResponsesPrompt> {
        if (options.format) {
            return Promise.resolve(options.format(segments, options.result_schema) as VertexAIResponsesPrompt);
        }
        return formatOpenAIResponsesPrompt(this.context(driver), segments, options);
    }

    requestTextCompletion(
        driver: VertexAIDriver,
        prompt: VertexAIResponsesPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        return this.protocol(driver).requestTextCompletion(this.context(driver), prompt, options, signal);
    }

    requestTextCompletionStream(
        driver: VertexAIDriver,
        prompt: VertexAIResponsesPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        return this.protocol(driver).requestTextCompletionStream(this.context(driver), prompt, options, signal);
    }

    /** OpenAI SDK errors keep the OpenAI-compatible format; other errors take the Vertex driver's default. */
    formatLlumiverseError(
        _driver: VertexAIDriver,
        error: unknown,
        context: LlumiverseErrorContext,
    ): LlumiverseError | undefined {
        return formatOpenAICompatibleLlumiverseError(error, context);
    }
}
