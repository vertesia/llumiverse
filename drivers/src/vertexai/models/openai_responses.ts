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
import type { VertexAIDriverOptions } from '../../driver-options.js';
import { OpenAIResponsesDriverBase } from '../../openai/index.js';
import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

export type VertexAIResponsesPrompt = OpenAI.Responses.ResponseInputItem[];

/**
 * Runs the shared OpenAI Responses protocol against a Vertex publisher's Responses endpoint
 * (`.../locations/{location}/publishers/{publisher}/v1/responses`), with Vertex capability metadata.
 */
export class VertexAIResponsesDelegate extends OpenAIResponsesDriverBase {
    readonly provider = Providers.vertexai;
    service: OpenAI;

    constructor(opts: VertexAIDriverOptions, service: OpenAI) {
        super(opts);
        this.service = service;
    }

    /** The endpoint takes the bare model ID (`muse-wonder-1.0`), not the Vertex resource path. */
    override getResponsesRequestModel(model: string): string {
        const name = model.split('/').pop() ?? model;
        const version = name.lastIndexOf('@');
        return version > -1 ? name.slice(0, version) : name;
    }
}

/** Vertex models served by a publisher's OpenAI-compatible Responses API, such as Meta Muse. */
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

    private delegate(driver: VertexAIDriver): VertexAIResponsesDelegate {
        return driver.getResponsesDelegate(this.publisher, this.region);
    }

    createPrompt(
        driver: VertexAIDriver,
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<VertexAIResponsesPrompt> {
        return this.delegate(driver).createPrompt(segments, options);
    }

    requestTextCompletion(
        driver: VertexAIDriver,
        prompt: VertexAIResponsesPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        return this.delegate(driver).requestTextCompletion(prompt, options, signal);
    }

    requestTextCompletionStream(
        driver: VertexAIDriver,
        prompt: VertexAIResponsesPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        return this.delegate(driver).requestTextCompletionStream(prompt, options, signal);
    }

    formatLlumiverseError(driver: VertexAIDriver, error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        return this.delegate(driver).formatLlumiverseError(error, context);
    }
}
