import { AIModel, AbstractDriver, Completion, CompletionChunkObject, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, PromptSegment, TextFallbackOptions } from "@llumiverse/core";
import { transformSSEStream } from "@llumiverse/core/async";
import { getJSONSafetyNotice } from "@llumiverse/core/formatters";
import { formatOpenAILikeTextPrompt, OpenAITextMessage } from "../openai/openai_format.js";
import { FetchClient } from "@vertesia/api-fetch-client";
import { ChatCompletionResponse, CompletionRequestParams, ListModelsResponse, ResponseFormat } from "./types.js";

//TODO retry on 429
//const RETRY_STATUS_CODES = [429, 500, 502, 503, 504];

const ENDPOINT = 'https://api.mistral.ai';

interface MistralAIDriverOptions extends DriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export class MistralAIDriver extends AbstractDriver<MistralAIDriverOptions, OpenAITextMessage[]> {
    static PROVIDER = "mistralai";

    provider = MistralAIDriver.PROVIDER;
    apiKey: string;
    client: FetchClient;
    endpointUrl?: string;

    constructor(options: MistralAIDriverOptions) {
        super(options);
        this.apiKey = options.apiKey;
        //this.client = new MistralClient(options.apiKey, options.endpointUrl);
        this.client = new FetchClient(options.endpoint_url || ENDPOINT).withHeaders({
            authorization: `Bearer ${this.apiKey}`
        });
    }

    getResponseFormat = (_options: ExecutionOptions): ResponseFormat | undefined => {

        // const responseFormatJson: ResponseFormat = {
        //     type: "json_object",
        // } as ResponseFormat

        // const responseFormatText: ResponseFormat = {
        //     type: "text",
        // } as ResponseFormat;


        // return _options.result_schema ? responseFormatJson : responseFormatText;

        //TODO remove this when Mistral properly supports the parameters - it makes an error for now
        // some models like mixtral mistral tiny or medium are throwing an error when using the response_format parameter
        return undefined
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

    async requestTextCompletion(messages: OpenAITextMessage[], options: ExecutionOptions): Promise<Completion> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn({ options: options.model_options }, "Invalid model options");
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const requestPayload = _makeChatCompletionRequest({
            model: options.model,
            messages: messages,
            maxTokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            responseFormat: this.getResponseFormat(options),
        });
        this.logger.debug({ payload: JSON.stringify(requestPayload) }, "Mistral request payload");

        const res = await this.client.post('/v1/chat/completions', {
            payload: requestPayload,
        }) as ChatCompletionResponse;

        const choice = res.choices[0];
        const result = choice.message.content;

        return {
            result: result ? [{ type: "text", value: result }] : [],
            token_usage: {
                prompt: res.usage.prompt_tokens,
                result: res.usage.completion_tokens,
                total: res.usage.total_tokens,
            },
            finish_reason: choice.finish_reason,        //Uses expected "stop" , "length" format
            original_response: options.include_original_response ? res : undefined,
        };
    }

    async requestTextCompletionStream(messages: OpenAITextMessage[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn({ options: options.model_options }, "Invalid model options");
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const streamPayload = _makeChatCompletionRequest({
            model: options.model,
            messages: messages,
            maxTokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            topP: options.model_options?.top_p,
            responseFormat: this.getResponseFormat(options),
            stream: true,
            stopSequences: options.model_options?.stop_sequence,
        });
        this.logger.debug({ payload: JSON.stringify(streamPayload) }, "Mistral stream request payload");

        const stream = await this.client.post('/v1/chat/completions', {
            payload: streamPayload,
            reader: 'sse'
        });

        return transformSSEStream(stream, (data: string) => {
            const json = JSON.parse(data);
            const content = json.choices[0]?.delta.content;
            return {
                result: content ? [{ type: "text", value: content }] : [],
                finish_reason: json.choices[0]?.finish_reason,      //Uses expected "stop" , "length" format
                token_usage: {
                    prompt: json.usage?.prompt_tokens,
                    result: json.usage?.completion_tokens,
                    total: json.usage?.total_tokens,
                },
            };
        });

    }

    async listModels(): Promise<AIModel<string>[]> {
        const models: ListModelsResponse = await this.client.get('v1/models');

        const aiModels = models.data.map(m => {
            return {
                id: m.id,
                name: m.id,
                description: undefined,
                provider: this.provider,
                owner: m.owned_by,
            }
        });

        return aiModels;
    }

    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }

    async generateEmbeddings({ text, model = "mistral-embed" }: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const r = await this.client.post('/v1/embeddings', {
            payload: {
                model,
                input: [text],
                encoding_format: "float"
            },
        });
        return {
            values: r.data[0].embedding,
            model,
            token_count: r.usage.total_tokens || r.usage.prompt_tokens + r.usage.completion_tokens,
        }
    }

}

/**
 * Creates a chat completion request
 * @param {*} model
 * @param {*} messages
 * @param {*} tools
 * @param {*} temperature
 * @param {*} maxTokens
 * @param {*} topP
 * @param {*} randomSeed
 * @param {*} stream
 * @param {*} safeMode deprecated use safePrompt instead
 * @param {*} safePrompt
 * @param {*} toolChoice
 * @param {*} responseFormat
 * @return {Promise<Object>}
 */
function _makeChatCompletionRequest({
    model,
    messages,
    tools,
    temperature,
    maxTokens,
    topP,
    randomSeed,
    stream,
    toolChoice,
    responseFormat,
    stopSequences,
}: CompletionRequestParams) {
    return {
        model,
        messages,
        tools: tools ?? undefined,
        temperature: temperature ?? undefined,
        max_tokens: maxTokens ?? undefined,
        top_p: topP ?? undefined,
        random_seed: randomSeed ?? undefined,
        stream: stream ?? undefined,
        tool_choice: toolChoice ?? undefined,
        response_format: responseFormat ?? undefined,
        stop: stopSequences ?? undefined,
    };
};
