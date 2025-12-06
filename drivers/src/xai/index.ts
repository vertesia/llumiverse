import { AIModel, DriverOptions, PromptOptions, PromptSegment, Providers } from "@llumiverse/core";
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
    readonly provider = Providers.xai;
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

    // Note: We intentionally do NOT override extractDataFromResponse here.
    // The base class implementation properly handles tool_calls extraction.
    // xAI's API is OpenAI-compatible and returns tool_calls in the same format.

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
                name: model.id,
                description: `${model.id} by ${model.owned_by}`,
                is_multimodal: model.input_modalities.length > 1,
                input_modalities: model.input_modalities,
                output_modalities: model.output_modalities,
                tags: [...model.input_modalities.map(m => `i:${m}`), ...model.output_modalities.map(m => `o:${m}`)],
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
