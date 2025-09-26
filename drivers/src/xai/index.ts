import { AIModel, Completion, DriverOptions, ExecutionOptions, PromptOptions, PromptSegment } from "@llumiverse/core";
import { formatOpenAILikeMultimodalPrompt, OpenAIPromptFormatterOptions } from "../openai/openai_format.js";
import { FetchClient } from "@vertesia/api-fetch-client";
import OpenAI from "openai";
import { BaseOpenAIDriver } from "../openai/index.js";

export interface xAiDriverOptions extends DriverOptions {

    apiKey: string;

    endpoint?: string;

}

export class xAIDriver extends BaseOpenAIDriver {

    service: OpenAI;
    provider: "xai";
    xai_service: FetchClient;
    DEFAULT_ENDPOINT = "https://api.x.ai/v1";

    constructor(opts: xAiDriverOptions) {
        super(opts);

        if (!opts.apiKey) {
            throw new Error("apiKey is required");
        }

        this.service = new OpenAI({
            apiKey: opts.apiKey,
            baseURL: opts.endpoint ?? this.DEFAULT_ENDPOINT,
        });
        this.xai_service = new FetchClient(opts.endpoint ?? this.DEFAULT_ENDPOINT).withAuthCallback(async () => `Bearer ${opts.apiKey}`);
        this.provider = "xai";
        //this.formatPrompt = this._formatPrompt; //TODO: fix xai prompt formatting
    }

    async _formatPrompt(segments: PromptSegment[], opts: PromptOptions): Promise<OpenAI.Chat.Completions.ChatCompletionMessageParam[]> {

        const options: OpenAIPromptFormatterOptions = {
            multimodal: opts.model.includes("vision"),
            schema: opts.result_schema,
            useToolForFormatting: false,
        }

        const p = await formatOpenAILikeMultimodalPrompt(segments, { ...options, ...opts }) as OpenAI.Chat.Completions.ChatCompletionMessageParam[];

        return p;

    }

    extractDataFromResponse(_options: ExecutionOptions, result: OpenAI.Chat.Completions.ChatCompletion): Completion {
        return {
            result: result.choices[0].message.content ? [{ type: "text", value: result.choices[0].message.content }] : [],
            finish_reason: result.choices[0].finish_reason,
            token_usage: {
                prompt: result.usage?.prompt_tokens,
                result: result.usage?.completion_tokens,
                total: result.usage?.total_tokens,
            }
        }
    }

    async listModels(): Promise<AIModel[]> {
        const [lm, em] = await Promise.all([
            this.xai_service.get("/language-models"),
            this.xai_service.get("/embedding-models")
        ]) as [xAIModelResponse, xAIModelResponse];


        em.models.forEach(m => {
            m.output_modalities.push("vectors");
        });

        const models = [...lm.models, ...em.models].map(model => {
            return {
                id: model.id,
                provider: this.provider,
                name: model.object,
                description: model.object,
                is_multimodal: model.input_modalities.length > 1,
                tags: [...model.input_modalities.map(m => `ì:${m}`), ...model.output_modalities.map(m => `ì:${m}`)],
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
