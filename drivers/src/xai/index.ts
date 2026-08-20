import {
    type AIModel,
    type Completion,
    type CompletionResult,
    type DriverOptions,
    type ExecutionOptions,
    getModelCapabilities,
    isEmbeddingModel,
    isXAIGrokImageModel,
    ModelType,
    modelModalitiesToArray,
    type PromptOptions,
    type PromptSegment,
    Providers,
    type XAIGrokImageOptions,
} from '@llumiverse/core';
import { FetchClient } from '@vertesia/api-fetch-client';
import OpenAI from 'openai';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import { formatOpenAILikeMultimodalPrompt, type OpenAIPromptFormatterOptions } from '../openai/openai_format.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

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

    override isImageModel(model: string): boolean {
        return isXAIGrokImageModel(model);
    }

    async requestImageGeneration(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<Completion> {
        this.logger.debug(`[${this.provider}] Generating image with model ${options.model}`);

        const { promptText, images } = extractImageRequest(prompt);
        const modelOptions = options.model_options as XAIGrokImageOptions | undefined;
        const payload: XAIImageRequest = {
            model: options.model,
            prompt: promptText,
            ...(modelOptions?.aspect_ratio && { aspect_ratio: modelOptions.aspect_ratio }),
            ...(modelOptions?.resolution && { resolution: modelOptions.resolution }),
            ...(modelOptions?.quality && { quality: modelOptions.quality }),
            ...(modelOptions?.response_format && { response_format: modelOptions.response_format }),
            ...(modelOptions?.n && { n: modelOptions.n }),
        };

        if (images.length === 1) {
            payload.image = images[0];
        } else if (images.length > 1) {
            payload.images = images;
        }

        try {
            const endpoint = images.length > 0 ? '/images/edits' : '/images/generations';
            const response = await this.xai_service.post<XAIImageResponse>(endpoint, { payload });
            const results: CompletionResult[] = [];

            for (const image of response.data ?? []) {
                if (image.b64_json) {
                    results.push({
                        type: 'image',
                        value: `data:${image.mime_type ?? 'image/jpeg'};base64,${image.b64_json}`,
                    });
                } else if (image.url) {
                    results.push({ type: 'image', value: image.url });
                }
            }

            return { result: results };
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

function extractImageRequest(prompt: ResponseInputItem[]): { promptText: string; images: XAIImageInput[] } {
    const text: string[] = [];
    const images: XAIImageInput[] = [];

    for (const item of prompt) {
        if (!('content' in item)) continue;
        if (typeof item.content === 'string') {
            text.push(item.content);
            continue;
        }
        if (!Array.isArray(item.content)) continue;

        for (const part of item.content) {
            if (part.type === 'input_text') {
                text.push(part.text);
            } else if (part.type === 'input_image') {
                if (part.image_url) {
                    images.push({ type: 'image_url', url: part.image_url });
                } else if (part.file_id) {
                    images.push({ file_id: part.file_id });
                }
            }
        }
    }

    return { promptText: text.join('\n').trim(), images };
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

type XAIImageInput = { file_id: string } | { type: 'image_url'; url: string };

interface XAIImageRequest {
    aspect_ratio?: XAIGrokImageOptions['aspect_ratio'];
    image?: XAIImageInput;
    images?: XAIImageInput[];
    model: string;
    n?: number;
    prompt: string;
    quality?: XAIGrokImageOptions['quality'];
    resolution?: XAIGrokImageOptions['resolution'];
    response_format?: XAIGrokImageOptions['response_format'];
}

interface XAIImageResponse {
    data?: Array<{
        b64_json?: string;
        mime_type?: string;
        url?: string;
    }>;
}
