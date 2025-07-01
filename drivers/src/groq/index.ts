import { AIModel, AbstractDriver, Completion, CompletionChunkObject, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, PromptSegment, TextFallbackOptions } from "@llumiverse/core";
import { transformAsyncIterator } from "@llumiverse/core/async";
import { getJSONSafetyNotice } from "@llumiverse/core/formatters";
import { formatOpenAILikeTextPrompt, OpenAITextMessage } from "../openai/openai_format.js";

import Groq from "groq-sdk";

interface GroqDriverOptions extends DriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export class GroqDriver extends AbstractDriver<GroqDriverOptions, OpenAITextMessage[]> {
    static PROVIDER = "groq";
    provider = GroqDriver.PROVIDER;
    apiKey: string;
    client: Groq;
    endpointUrl?: string;

    constructor(options: GroqDriverOptions) {
        super(options);
        this.apiKey = options.apiKey;
        this.client = new Groq({
            apiKey: options.apiKey,
            baseURL: options.endpoint_url
        });
    }

    // protected canStream(options: ExecutionOptions): Promise<boolean> {
    //     if (options.result_schema) {
    //         // not yet streaming json responses
    //         return Promise.resolve(false);
    //     } else {
    //         return Promise.resolve(true);
    //     }
    // }

    getResponseFormat(_options: ExecutionOptions): undefined {
        //TODO: when forcing json_object type the streaming is not supported.
        // either implement canStream as above or comment the code below:
        // const responseFormatJson: Groq.Chat.Completions.CompletionCreateParams.ResponseFormat = {
        //     type: "json_object",
        // }

        // return _options.result_schema ? responseFormatJson : undefined;
        return undefined;
    }

    protected async formatPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<OpenAITextMessage[]> {
        const messages = formatOpenAILikeTextPrompt(segments);
        //Add JSON instruction is schema is provided
        if (opts.result_schema) {
            messages.push({
                role: "user",
                content: "IMPORTANT: " + getJSONSafetyNotice(opts.result_schema)
            });
        }
        return messages;
    }

    async requestTextCompletion(messages: OpenAITextMessage[], options: ExecutionOptions): Promise<Completion<any>> {
        if (options.model_options?._option_id !== "text-fallback" && options.model_options?._option_id !== "groq-deepseek-thinking") {
            this.logger.warn("Invalid model options", {options: options.model_options });
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const res = await this.client.chat.completions.create({
            model: options.model,
            messages: messages,
            max_completion_tokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            response_format: this.getResponseFormat(options),
        });


        const choice = res.choices[0];
        const result = choice.message.content;

        return {
            result: result,
            token_usage: {
                prompt: res.usage?.prompt_tokens,
                result: res.usage?.completion_tokens,
                total: res.usage?.total_tokens,
            },
            finish_reason: choice.finish_reason,
            original_response: options.include_original_response ? res : undefined,
        };
    }

    async requestTextCompletionStream(messages: OpenAITextMessage[], options: ExecutionOptions): Promise < AsyncIterable < CompletionChunkObject >> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn("Invalid model options", {options: options.model_options });
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const res = await this.client.chat.completions.create({
            model: options.model,
            messages: messages,
            max_completion_tokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            stream: true,
        });

        return transformAsyncIterator(res, (res) => ({
            result: res.choices[0].delta.content ?? '',
            finish_reason: res.choices[0].finish_reason,
            token_usage: {
                prompt: res.x_groq?.usage?.prompt_tokens,
                result: res.x_groq?.usage?.completion_tokens,
                total: res.x_groq?.usage?.total_tokens,
            },
            } as CompletionChunkObject));
    }

    async listModels(): Promise<AIModel<string>[]> {
        const models = await this.client.models.list();

        if (!models.data) {
            throw new Error("No models found");
        }

        const aiModels = models.data?.map(m => {
            if (!m.id) {
                throw new Error("Model id is missing");
            }
            return {
                id: m.id,
                name: m.id,
                description: undefined,
                provider: this.provider,
                owner: m.owned_by || '',
            }
        });

        return aiModels;
    }

    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }

    async generateEmbeddings({ }: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error("Method not implemented.");
    }

}