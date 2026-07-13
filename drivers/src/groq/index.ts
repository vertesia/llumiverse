import type { GroqDeepseekThinkingOptions } from '@llumiverse/common';
import type { AIModel, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions } from '@llumiverse/core';
import { Providers } from '@llumiverse/core';
import Groq from 'groq-sdk';
import type {
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletionCreateParamsStreaming,
    ChatCompletionMessageParam,
    ChatCompletionTool,
} from 'groq-sdk/resources/chat/completions';
import {
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    type OpenAIChatCompletionsResponse,
    type OpenAIChatCompletionsStreamResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
} from '../openai/openai_chat_completions.js';
import { truncateDataUrlForDebug } from '../shared/debug-prompt.js';

export interface GroqDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export class GroqDriver extends OpenAIChatCompletionsDriverBase<GroqDriverOptions> {
    static readonly PROVIDER = Providers.groq;
    readonly provider = Providers.groq;
    readonly apiKey: string;
    readonly client: Groq;
    readonly endpointUrl?: string;

    constructor(options: GroqDriverOptions) {
        super({ ...options, resultSchemaMode: 'prompt', toolSchemaMode: 'compatible' });
        this.apiKey = options.apiKey;
        this.endpointUrl = options.endpoint_url;
        this.client = new Groq({
            apiKey: options.apiKey,
            baseURL: options.endpoint_url,
            fetch: this.getDriverFetch(),
        });
    }

    public formatDebugPrompt(prompt: OpenAIChatCompletionsPrompt): OpenAIChatCompletionsPrompt {
        return {
            ...prompt,
            messages: prompt.messages.map((message) => ({
                ...message,
                content: Array.isArray(message.content)
                    ? message.content.map((part) =>
                          part.type === 'image_url'
                              ? {
                                    ...part,
                                    image_url: {
                                        ...part.image_url,
                                        url: truncateDataUrlForDebug(part.image_url.url),
                                    },
                                }
                              : part,
                      )
                    : message.content,
            })),
        };
    }

    async _postChatCompletion(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
    ): Promise<OpenAIChatCompletionsResponse> {
        const request = toGroqRequest(payload, options, false);
        const response = await this.client.chat.completions.create(request);
        return preserveOpenAIChatCompletionsOriginalResponse(normalizeGroqResponse(response), response);
    }

    async _postChatCompletionStream(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
    ): Promise<ReadableStream> {
        const stream = await this.client.chat.completions.create(toGroqRequest(payload, options, true));
        return openAIChatCompletionsStreamToSSE(normalizeGroqStream(stream));
    }

    async listModels(): Promise<AIModel<string>[]> {
        const models = await this.client.models.list();
        return models.data.map((model) => ({
            id: model.id,
            name: model.id,
            provider: this.provider,
            owner: model.owned_by || '',
        }));
    }

    async validateConnection(): Promise<boolean> {
        try {
            await this.client.models.list();
            return true;
        } catch {
            return false;
        }
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Groq does not expose an embeddings transport.');
    }
}

function toGroqRequest(
    payload: OpenAIChatCompletionsPayload,
    options: ExecutionOptions,
    stream: false,
): ChatCompletionCreateParamsNonStreaming;
function toGroqRequest(
    payload: OpenAIChatCompletionsPayload,
    options: ExecutionOptions,
    stream: true,
): ChatCompletionCreateParamsStreaming;
function toGroqRequest(
    payload: OpenAIChatCompletionsPayload,
    options: ExecutionOptions,
    stream: boolean,
): ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming {
    const modelOptions = options.model_options;
    const reasoningFormat =
        modelOptions?._option_id === 'groq-deepseek-thinking'
            ? (modelOptions as GroqDeepseekThinkingOptions).reasoning_format
            : undefined;
    const request = {
        model: payload.model,
        messages: payload.messages.map(toGroqMessage),
        max_completion_tokens: payload.max_tokens ?? undefined,
        temperature: payload.temperature ?? undefined,
        top_p: payload.top_p ?? undefined,
        presence_penalty: payload.presence_penalty ?? undefined,
        frequency_penalty: payload.frequency_penalty ?? undefined,
        stop: payload.stop ?? undefined,
        n: payload.n ?? undefined,
        tools: payload.tools?.flatMap(toGroqTool),
        reasoning_format: reasoningFormat,
        extra_body: payload.extra_body,
        stream,
    } satisfies (ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming) & {
        extra_body?: Record<string, unknown>;
    };
    return request;
}

function toGroqMessage(message: OpenAIChatCompletionsPayload['messages'][number]): ChatCompletionMessageParam {
    const textContent = typeof message.content === 'string' || message.content === null ? message.content : undefined;
    switch (message.role) {
        case 'system':
        case 'developer':
            return { role: 'system', content: textContent ?? '' };
        case 'assistant':
            return {
                role: 'assistant',
                content: textContent,
                tool_calls: message.tool_calls?.map((toolCall) => ({
                    id: toolCall.id,
                    type: 'function',
                    function: {
                        name: toolCall.function.name,
                        arguments: toolCall.function.arguments,
                    },
                })),
            };
        case 'tool':
            return { role: 'tool', content: textContent ?? '', tool_call_id: message.tool_call_id ?? '' };
        default:
            return {
                role: 'user',
                content:
                    typeof message.content === 'string'
                        ? message.content
                        : (message.content?.map((part) =>
                              part.type === 'text'
                                  ? { type: 'text' as const, text: part.text }
                                  : { type: 'image_url' as const, image_url: part.image_url },
                          ) ?? ''),
            };
    }
}

function toGroqTool(tool: NonNullable<OpenAIChatCompletionsPayload['tools']>[number]): ChatCompletionTool[] {
    if (tool.type !== 'function') {
        return [];
    }
    return [
        {
            type: 'function',
            function: {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
            },
        },
    ];
}

function normalizeGroqResponse(response: ChatCompletion): OpenAIChatCompletionsResponse {
    const usage = response.usage;
    return {
        id: response.id,
        object: 'chat.completion',
        created: response.created,
        model: response.model,
        choices: response.choices.map((choice) => ({
            index: choice.index,
            finish_reason: choice.finish_reason,
            logprobs: choice.logprobs,
            message: {
                role: choice.message.role,
                content: choice.message.content,
                reasoning: choice.message.reasoning,
                tool_calls: choice.message.tool_calls?.map((toolCall) => ({
                    id: toolCall.id,
                    type: 'function',
                    function: {
                        name: toolCall.function.name,
                        arguments: toolCall.function.arguments,
                    },
                })),
            },
        })),
        usage: usage
            ? {
                  prompt_tokens: usage.prompt_tokens,
                  completion_tokens: usage.completion_tokens,
                  total_tokens: usage.total_tokens,
              }
            : undefined,
    };
}

async function* normalizeGroqStream(
    stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    for await (const chunk of stream) {
        const usage = chunk.x_groq?.usage;
        yield {
            id: chunk.id,
            object: 'chat.completion.chunk',
            created: chunk.created,
            model: chunk.model,
            choices: chunk.choices.map((choice) => ({
                index: choice.index,
                finish_reason: choice.finish_reason,
                logprobs: choice.logprobs,
                delta: {
                    role: choice.delta.role,
                    content: choice.delta.content,
                    reasoning: choice.delta.reasoning,
                    tool_calls: choice.delta.tool_calls?.map((toolCall) => ({
                        index: toolCall.index,
                        id: toolCall.id,
                        type: toolCall.type,
                        function: toolCall.function,
                    })),
                },
            })),
            usage: usage
                ? {
                      prompt_tokens: usage.prompt_tokens,
                      completion_tokens: usage.completion_tokens,
                      total_tokens: usage.total_tokens,
                  }
                : undefined,
        };
    }
}
