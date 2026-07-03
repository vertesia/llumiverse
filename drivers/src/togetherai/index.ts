import {
    type AIModel,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    getModelCapabilities,
    ModelType,
    modelModalitiesToArray,
    normalizeEmbeddingsOptions,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
    Providers,
} from '@llumiverse/core';
import OpenAI from 'openai';
import {
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
} from '../openai/openai_chat_completions.js';

export interface TogetherAIDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint?: string;
}

interface TogetherModel {
    id: string;
    display_name?: string;
    organization?: string;
    type?: string;
}

export class TogetherAIDriver extends OpenAIChatCompletionsDriverBase<TogetherAIDriverOptions> {
    static readonly PROVIDER = Providers.togetherai;
    readonly provider = Providers.togetherai;
    service: OpenAI;

    constructor(opts: TogetherAIDriverOptions) {
        super({ ...opts, resultSchemaMode: 'prompt' });

        if (!opts.apiKey) {
            throw new Error('apiKey is required');
        }

        this.service = new OpenAI({
            apiKey: opts.apiKey,
            baseURL: opts.endpoint ?? 'https://api.together.ai/v1',
            fetch: this.getDriverFetch(),
        });
    }

    /**
     * TogetherAI is treated as Chat Completions only. Its Responses API surface is either unsupported
     * or image-broken for hosted open models, while Chat Completions correctly accepts multimodal
     * content as OpenAI-style `image_url` parts.
     */
    async validateConnection(): Promise<boolean> {
        try {
            await this.service.models.list();
            return true;
        } catch {
            return false;
        }
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        const model = normalized.model ?? OPENAI_DEFAULT_EMBEDDING_MODEL;
        const texts = normalized.inputs.map((input) => {
            if (input.type !== 'text') {
                throw new Error(
                    `Provider 'togetherai' does not support '${input.type}' embeddings; only 'text' is supported.`,
                );
            }
            return input.text;
        });

        const response = await this.service.embeddings.create({
            input: texts,
            model,
            ...(normalized.dimensions ? { dimensions: normalized.dimensions } : {}),
            encoding_format: 'float',
        });
        const ordered = [...response.data].sort((a, b) => a.index - b.index);
        const items = ordered.map((entry): EmbeddingResultItem => {
            if (!entry.embedding || entry.embedding.length === 0) {
                throw new Error(`TogetherAI embedding empty for input index ${entry.index}`);
            }
            return { outputs: [{ values: entry.embedding, modality: 'text' }] };
        });
        const usage = response.usage
            ? { input_tokens: response.usage.prompt_tokens, input_text_tokens: response.usage.prompt_tokens }
            : undefined;

        return { model, results: items, usage };
    }

    async listModels(): Promise<AIModel[]> {
        const result = await this.service.get<TogetherModel[]>('/models');
        return result
            .flatMap((model) => {
                const type = togetherModelType(model.type);
                if (type === ModelType.Embedding) {
                    return [];
                }
                const modelCapability = getModelCapabilities(model.id, this.provider);
                return [
                    {
                        id: model.id,
                        name: model.display_name ?? model.id,
                        provider: this.provider,
                        owner: model.organization,
                        type,
                        can_stream: true,
                        is_multimodal: modelCapability.input.image === true,
                        input_modalities: modelModalitiesToArray(modelCapability.input),
                        output_modalities: modelModalitiesToArray(modelCapability.output),
                        tool_support: modelCapability.tool_support,
                    } satisfies AIModel<string>,
                ];
            })
            .sort((a, b) => a.id.localeCompare(b.id));
    }
}

function togetherModelType(type?: string): ModelType {
    switch (type) {
        case 'chat':
            return ModelType.Chat;
        case 'code':
            return ModelType.Code;
        case 'image':
            return ModelType.Image;
        case 'embedding':
            return ModelType.Embedding;
        default:
            return ModelType.Text;
    }
}
