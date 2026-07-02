import {
    type AIModel,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    getModelCapabilities,
    ModelType,
    modelModalitiesToArray,
    Providers,
} from '@llumiverse/core';
import OpenAI from 'openai';
import {
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
} from '../openai/openai_comp_completions.js';

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

    generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Embeddings not implemented for TogetherAI.');
    }

    async listModels(): Promise<AIModel[]> {
        const result = await this.service.get<TogetherModel[]>('/models');
        return result
            .map((model) => {
                const modelCapability = getModelCapabilities(model.id, this.provider);
                return {
                    id: model.id,
                    name: model.display_name ?? model.id,
                    provider: this.provider,
                    owner: model.organization,
                    type: togetherModelType(model.type),
                    can_stream: true,
                    is_multimodal: modelCapability.input.image === true,
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                } satisfies AIModel<string>;
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
