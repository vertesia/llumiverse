import {
    type AIModel,
    type DriverOptions,
    getModelCapabilities,
    ModelType,
    modelModalitiesToArray,
    Providers,
} from '@llumiverse/core';
import OpenAI from 'openai';
import { BaseOpenAIDriver } from '../openai/index.js';

export interface TogetherAIDriverOptions extends DriverOptions {
    apiKey: string;
    endpoint?: string;
}

interface TogetherModel {
    id: string;
    display_name?: string;
    organization?: string;
    type?: string;
}

export class TogetherAIDriver extends BaseOpenAIDriver {
    static readonly PROVIDER = Providers.togetherai;
    readonly provider = Providers.togetherai;
    service: OpenAI;

    constructor(opts: TogetherAIDriverOptions) {
        super(opts);

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
     * TogetherAI's OpenAI-compatible surface only implements Chat Completions (`/v1/chat/completions`)
     * and does NOT support the Responses API (`/v1/responses`). Requests must therefore go through
     * Chat Completions, where images are sent as `image_url` parts. Sending them via the Responses API
     * makes vision requests fail: `openai/gpt-oss-120b` returns HTTP 400 "does not support the Responses
     * api", while MiniMax/Kimi return 200 but never receive the image ("no content").
     */
    protected override useChatCompletionsApi(): boolean {
        return true;
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
