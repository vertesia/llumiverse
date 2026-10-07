import {
    type AIModel,
    type CanonicalExecutionResponse,
    type Completion,
    type CompletionResult,
    type ExecutionOptions,
    getModelCapabilities,
    isEmbeddingModel,
    isXAIGrokImageModel,
    ModelType,
    modelModalitiesToArray,
    type PromptOptions,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import { FetchClient } from '@vertesia/api-fetch-client';
import OpenAI from 'openai';
import type { xAiDriverOptions } from '../driver-options.js';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import { formatOpenAILikeMultimodalPrompt, type OpenAIPromptFormatterOptions } from '../openai/openai_format.js';
import {
    executeXAIImageCanonical,
    validateXAICanonicalImageInput,
    type XAIImageRequest,
    type XAIImageResponse,
    xAIImageEndpoint,
    xAIImageRequest,
} from './image-canonical.js';

export type { xAiDriverOptions } from '../driver-options.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

export class xAIDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.xai;
    xai_service: FetchClient;
    DEFAULT_ENDPOINT = 'https://api.x.ai/v1';
    private readonly imageEndpoint: string;

    constructor(opts: xAiDriverOptions) {
        super(opts);

        if (!opts.apiKey) {
            throw new Error('apiKey is required');
        }
        const endpoint = opts.endpoint ?? this.DEFAULT_ENDPOINT;
        let endpointEnd = endpoint.length;
        while (endpointEnd > 0 && endpoint[endpointEnd - 1] === '/') endpointEnd--;
        this.imageEndpoint = endpoint.slice(0, endpointEnd);

        this.service = new OpenAI({
            apiKey: opts.apiKey,
            baseURL: this.imageEndpoint,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
        this.xai_service = new FetchClient(this.imageEndpoint, this.getDriverFetch()).withAuthCallback(
            async () => `Bearer ${opts.apiKey}`,
        );
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

    override isImageModel(model: string): boolean {
        return isXAIGrokImageModel(model);
    }

    protected override supportsCanonicalImageGeneration(_options: ExecutionOptions): boolean {
        return true;
    }

    protected override validateCanonicalImageInput(segments: PromptSegment[], options: ExecutionOptions): void {
        validateXAICanonicalImageInput(segments, options);
    }

    private invokeImage(
        request: XAIImageRequest,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<XAIImageResponse> {
        return this.xai_service.post<XAIImageResponse>(xAIImageEndpoint(request), {
            payload: request,
            ...(signal === undefined ? {} : { signal }),
            ...(options.httpTimeout === undefined
                ? {}
                : { timeoutMs: this.getDriverRequestTimeoutMs(options.httpTimeout) }),
        });
    }

    override requestCanonicalImageGeneration(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        return executeXAIImageCanonical({
            endpoint: this.imageEndpoint,
            fetch_image: this.getDriverFetch(),
            invoke: (request, invokeSignal) => this.invokeImage(request, options, invokeSignal),
            options,
            prompt,
            provider: this.provider,
            signal,
        });
    }

    async requestImageGeneration(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        this.logger.debug(`[${this.provider}] Generating image with model ${options.model}`);
        const payload = xAIImageRequest(prompt, options);

        try {
            const response = await this.invokeImage(payload, options, signal);
            const results: CompletionResult[] = [];

            for (const image of response.data ?? []) {
                if (!image) continue;
                if (typeof image.b64_json === 'string' && image.b64_json.trim()) {
                    results.push({
                        type: 'image',
                        value: `data:${image.mime_type ?? 'image/jpeg'};base64,${image.b64_json}`,
                    });
                } else if (typeof image.url === 'string' && image.url.trim()) {
                    results.push({ type: 'image', value: image.url });
                }
            }

            const costTicks = response.usage?.cost_in_usd_ticks;
            return {
                result: results,
                ...(results.length === 0 && {
                    error: { code: 'validation_error', message: 'Image generation returned no usable images' },
                }),
                ...(typeof costTicks === 'number' && {
                    token_usage: { provider_cost_usd: costTicks / XAI_USD_TICKS },
                }),
            };
        } catch (error: unknown) {
            this.logger.error({ error }, `[${this.provider}] Image generation failed`);
            const generationError = error instanceof Error ? error : new Error(String(error));
            const errorCode =
                (error as { code?: unknown })?.code === 'content_policy_violation'
                    ? 'content_policy_violation'
                    : 'validation_error';
            return {
                result: [],
                error: {
                    message: generationError.message,
                    code: errorCode,
                },
            };
        }
    }

    async listModels(): Promise<AIModel[]> {
        const [languageResult, imageResult] = await Promise.allSettled([
            this.xai_service.get<xAILanguageModelResponse>('/language-models'),
            this.xai_service.get<xAIImageModelResponse>('/image-generation-models'),
        ]);
        if (languageResult.status === 'rejected' && imageResult.status === 'rejected') {
            throw languageResult.reason;
        }
        if (languageResult.status === 'rejected') {
            this.logger.warn({ error: languageResult.reason }, '[xai] Failed to list language models');
        }
        if (imageResult.status === 'rejected') {
            this.logger.warn({ error: imageResult.reason }, '[xai] Failed to list image generation models');
        }
        const languageModels = languageResult.status === 'fulfilled' ? languageResult.value.models : [];
        const imageModels = imageResult.status === 'fulfilled' ? imageResult.value.models : [];

        // xAI listing modalities have been incomplete and occasionally describe endpoint artifacts rather than the
        // language-model execution path. Prefer the curated family directory and use runtime data for availability.
        const models = languageModels
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

        const images = imageModels.map((model) => {
            const capabilities = getModelCapabilities(model.id, this.provider);
            const inputModalities = modelModalitiesToArray(capabilities.input);
            const outputModalities = modelModalitiesToArray(capabilities.output);
            return {
                id: model.id,
                provider: this.provider,
                name: model.id,
                description: `${model.id} by ${model.owned_by}`,
                version: model.version,
                owner: model.owned_by,
                type: ModelType.Image,
                can_stream: false,
                is_multimodal: inputModalities.length > 1,
                input_modalities: inputModalities,
                output_modalities: outputModalities,
                tool_support: false,
                tags: [
                    ...inputModalities.map((modality) => `i:${modality}`),
                    ...outputModalities.map((modality) => `o:${modality}`),
                ],
            } satisfies AIModel;
        });

        return [...models, ...images].sort((a, b) => a.id.localeCompare(b.id));
    }
}

interface xAILanguageModelResponse {
    models: xAILanguageModel[];
}

interface xAILanguageModel {
    id: string;
    owned_by: string;
}

interface xAIImageModelResponse {
    models: xAIImageModel[];
}

interface xAIImageModel {
    id: string;
    owned_by: string;
    version: string;
}

/** xAI reports what a request cost in ticks of 10^-10 USD. */
const XAI_USD_TICKS = 1e10;
