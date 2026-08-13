import {
    type AIModel,
    type DriverOptions,
    getModelCapabilities,
    isEmbeddingModel,
    modelModalitiesToArray,
    type PromptOptions,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import { FetchClient } from '@vertesia/api-fetch-client';
import OpenAI from 'openai';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import { formatOpenAILikeMultimodalPrompt, type OpenAIPromptFormatterOptions } from '../openai/openai_format.js';

export interface xAiDriverOptions extends DriverOptions {
    apiKey: string;

    endpoint?: string;
}

export class xAIDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.xai;
    xai_service: FetchClient;
    DEFAULT_ENDPOINT = 'https://api.x.ai/v1';

    constructor(opts: xAiDriverOptions) {
        super(opts);

        if (!opts.apiKey) {
            throw new Error('apiKey is required');
        }

        this.service = new OpenAI({
            apiKey: opts.apiKey,
            baseURL: opts.endpoint ?? this.DEFAULT_ENDPOINT,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
        this.xai_service = new FetchClient(
            opts.endpoint ?? this.DEFAULT_ENDPOINT,
            this.getDriverFetch(),
        ).withAuthCallback(async () => `Bearer ${opts.apiKey}`);
        //this.formatPrompt = this._formatPrompt; //TODO: fix xai prompt formatting
    }

    async _formatPrompt(
        segments: PromptSegment[],
        opts: PromptOptions,
    ): Promise<OpenAI.Chat.Completions.ChatCompletionMessageParam[]> {
        const options: OpenAIPromptFormatterOptions = {
            multimodal: opts.model.includes('vision'),
            schema: opts.result_schema,
            useToolForFormatting: false,
        };

        const p = (await formatOpenAILikeMultimodalPrompt(segments, {
            ...options,
            ...opts,
        })) as OpenAI.Chat.Completions.ChatCompletionMessageParam[];

        return p;
    }

    // Note: We intentionally do NOT override extractDataFromResponse here.
    // The base class implementation properly handles tool_calls extraction.
    // xAI's API is OpenAI-compatible and returns tool_calls in the same format.

    async listModels(): Promise<AIModel[]> {
        const lm = (await this.xai_service.get('/language-models')) as xAIModelResponse;

        // xAI listing modalities have been incomplete and occasionally describe endpoint artifacts rather than the
        // language-model execution path. Prefer the curated family directory and use runtime data for availability.
        const models = lm.models
            .filter((model) => !isEmbeddingModel(model, this.provider))
            .map((model) => {
                const capabilities = getModelCapabilities(model.id, this.provider);
                const inputModalities = modelModalitiesToArray(capabilities.input);
                const outputModalities = modelModalitiesToArray(capabilities.output);
                return {
                    id: model.id,
                    provider: this.provider,
                    name: model.id,
                    description: `${model.id} by ${model.owned_by}`,
                    is_multimodal: capabilities.input.image === true,
                    input_modalities: inputModalities,
                    output_modalities: outputModalities,
                    tool_support: capabilities.tool_support,
                    tags: [
                        ...inputModalities.map((modality) => `i:${modality}`),
                        ...outputModalities.map((modality) => `o:${modality}`),
                    ],
                } satisfies AIModel;
            });

        return models;
    }
}

interface xAIModelResponse {
    models: xAIModel[];
}

interface xAIModel {
    completion_text_token_price: number;
    created: number;
    id: string;
    input_modalities: string[];
    object: string;
    output_modalities: string[];
    owned_by: string;
    prompt_image_token_price: number;
    prompt_text_token_price: number;
}
