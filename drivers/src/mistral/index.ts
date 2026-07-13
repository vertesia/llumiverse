import {
    type AIModel,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    MISTRAL_DEFAULT_EMBEDDING_MODEL,
    normalizeEmbeddingsOptions,
    Providers,
} from '@llumiverse/core';
import { HTTPClient, Mistral } from '@mistralai/mistralai';
import type {
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
    ChatCompletionRequestTool,
    ChatCompletionResponse,
    CompletionChunk,
    ContentChunk,
} from '@mistralai/mistralai/models/components';
import {
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsResponse,
    type OpenAIChatCompletionsStreamResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
} from '../openai/openai_chat_completions.js';

const ENDPOINT = 'https://api.mistral.ai';

export interface MistralAIDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export class MistralAIDriver extends OpenAIChatCompletionsDriverBase<MistralAIDriverOptions> {
    static readonly PROVIDER = Providers.mistralai;
    readonly provider = Providers.mistralai;
    readonly apiKey: string;
    readonly client: Mistral;
    readonly endpointUrl?: string;

    constructor(options: MistralAIDriverOptions) {
        super({ ...options, resultSchemaMode: 'prompt', toolSchemaMode: 'compatible' });
        this.apiKey = options.apiKey;
        this.endpointUrl = options.endpoint_url;
        this.client = new Mistral({
            apiKey: options.apiKey,
            serverURL: options.endpoint_url ?? ENDPOINT,
            httpClient: new HTTPClient({ fetcher: this.getDriverFetch() }),
        });
    }

    async _postChatCompletion(payload: OpenAIChatCompletionsPayload): Promise<OpenAIChatCompletionsResponse> {
        const request = toMistralRequest(payload, false);
        const response = await this.client.chat.complete(request);
        return preserveOpenAIChatCompletionsOriginalResponse(normalizeMistralResponse(response), response);
    }

    async _postChatCompletionStream(payload: OpenAIChatCompletionsPayload): Promise<ReadableStream> {
        const stream = await this.client.chat.stream(toMistralRequest(payload, true));
        return openAIChatCompletionsStreamToSSE(normalizeMistralStream(stream));
    }

    async listModels(): Promise<AIModel<string>[]> {
        const models = await this.client.models.list();
        return (models.data ?? []).flatMap((model) =>
            'id' in model
                ? [
                      {
                          id: model.id,
                          name: ('name' in model && model.name) || model.id,
                          description: ('description' in model && model.description) || undefined,
                          provider: this.provider,
                          owner: 'ownedBy' in model ? model.ownedBy : '',
                      } satisfies AIModel<string>,
                  ]
                : [],
        );
    }

    async validateConnection(): Promise<boolean> {
        try {
            await this.client.models.list();
            return true;
        } catch {
            return false;
        }
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        const model = normalized.model ?? MISTRAL_DEFAULT_EMBEDDING_MODEL;
        const texts = normalized.inputs.map((input) => {
            if (input.type !== 'text') {
                throw new Error(
                    `Provider 'mistralai' does not support '${input.type}' embeddings; only 'text' is supported.`,
                );
            }
            return input.text;
        });
        const response = await this.client.embeddings.create({ model, inputs: texts, encodingFormat: 'float' });
        const ordered = [...response.data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
        const results = ordered.map((entry, index): EmbeddingResultItem => {
            if (!entry.embedding?.length) {
                throw new Error(`Mistral embedding empty for input index ${entry.index ?? index}`);
            }
            return { outputs: [{ values: entry.embedding, modality: 'text' }] };
        });
        const promptTokens = response.usage.promptTokens ?? response.usage.totalTokens;
        return {
            model,
            results,
            ...(promptTokens === undefined
                ? {}
                : { usage: { input_tokens: promptTokens, input_text_tokens: promptTokens } }),
        };
    }
}

function toMistralRequest(payload: OpenAIChatCompletionsPayload, stream: boolean): ChatCompletionRequest {
    return {
        model: payload.model,
        messages: payload.messages.map(toMistralMessage),
        maxTokens: payload.max_tokens ?? undefined,
        temperature: payload.temperature ?? undefined,
        topP: payload.top_p ?? undefined,
        presencePenalty: payload.presence_penalty ?? undefined,
        frequencyPenalty: payload.frequency_penalty ?? undefined,
        stop: payload.stop ?? undefined,
        n: payload.n ?? undefined,
        tools: payload.tools?.flatMap(toMistralTool),
        stream,
    } satisfies ChatCompletionRequest;
}

function toMistralMessage(message: OpenAIChatCompletionsPayload['messages'][number]): ChatCompletionRequestMessage {
    const textContent = typeof message.content === 'string' || message.content === null ? message.content : undefined;
    switch (message.role) {
        case 'system':
        case 'developer':
            return { role: 'system', content: textContent ?? '' };
        case 'assistant':
            return {
                role: 'assistant',
                content: textContent,
                toolCalls: message.tool_calls?.map((toolCall, index) => ({
                    id: toolCall.id,
                    index,
                    type: 'function',
                    function: {
                        name: toolCall.function.name,
                        arguments: toolCall.function.arguments,
                    },
                })),
            };
        case 'tool':
            return { role: 'tool', content: textContent ?? '', toolCallId: message.tool_call_id };
        default:
            return {
                role: 'user',
                content:
                    typeof message.content === 'string'
                        ? message.content
                        : (message.content?.map(
                              (part): ContentChunk =>
                                  part.type === 'text'
                                      ? { type: 'text', text: part.text }
                                      : { type: 'image_url', imageUrl: part.image_url.url },
                          ) ?? ''),
            };
    }
}

function toMistralTool(tool: NonNullable<OpenAIChatCompletionsPayload['tools']>[number]): ChatCompletionRequestTool[] {
    if (tool.type !== 'function') {
        return [];
    }
    return [
        {
            type: 'function',
            function: {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters ?? {},
            },
        },
    ];
}

function normalizeMistralContent(content: string | ContentChunk[] | null | undefined): {
    content: string | null;
    reasoning?: string;
} {
    if (typeof content === 'string' || content == null) {
        return { content: content ?? null };
    }
    const text: string[] = [];
    const reasoning: string[] = [];
    for (const part of content) {
        if (part.type === 'text') {
            text.push(part.text);
        } else if (part.type === 'thinking' && 'thinking' in part && Array.isArray(part.thinking)) {
            reasoning.push(
                ...part.thinking.flatMap((entry) =>
                    typeof entry === 'object' && entry && 'text' in entry && typeof entry.text === 'string'
                        ? [entry.text]
                        : [],
                ),
            );
        }
    }
    return { content: text.join(''), reasoning: reasoning.length ? reasoning.join('') : undefined };
}

function normalizeMistralResponse(response: ChatCompletionResponse): OpenAIChatCompletionsResponse {
    return {
        id: response.id,
        object: 'chat.completion',
        created: response.created,
        model: response.model,
        choices: response.choices.map((choice) => {
            const normalizedContent = normalizeMistralContent(choice.message?.content);
            return {
                index: choice.index,
                finish_reason: choice.finishReason,
                message: {
                    role: choice.message?.role,
                    ...normalizedContent,
                    tool_calls: choice.message?.toolCalls?.map((toolCall) => ({
                        id: toolCall.id ?? '',
                        type: 'function',
                        function: {
                            name: toolCall.function.name,
                            arguments:
                                typeof toolCall.function.arguments === 'string'
                                    ? toolCall.function.arguments
                                    : JSON.stringify(toolCall.function.arguments),
                        },
                    })),
                },
            };
        }),
        usage: {
            prompt_tokens: response.usage.promptTokens ?? 0,
            completion_tokens: response.usage.completionTokens ?? 0,
            total_tokens: response.usage.totalTokens ?? 0,
        },
    };
}

async function* normalizeMistralStream(
    stream: AsyncIterable<{ data: CompletionChunk }>,
): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    for await (const event of stream) {
        const chunk = event.data;
        yield {
            id: chunk.id,
            object: 'chat.completion.chunk',
            created: chunk.created ?? 0,
            model: chunk.model,
            choices: chunk.choices.map((choice) => {
                const normalizedContent = normalizeMistralContent(choice.delta.content);
                return {
                    index: choice.index,
                    finish_reason: choice.finishReason,
                    delta: {
                        role: choice.delta.role ?? undefined,
                        ...normalizedContent,
                        tool_calls: choice.delta.toolCalls?.map((toolCall) => ({
                            index: toolCall.index,
                            id: toolCall.id,
                            type: 'function',
                            function: {
                                name: toolCall.function.name,
                                arguments:
                                    typeof toolCall.function.arguments === 'string'
                                        ? toolCall.function.arguments
                                        : JSON.stringify(toolCall.function.arguments),
                            },
                        })),
                    },
                };
            }),
            usage: chunk.usage
                ? {
                      prompt_tokens: chunk.usage.promptTokens ?? 0,
                      completion_tokens: chunk.usage.completionTokens ?? 0,
                      total_tokens: chunk.usage.totalTokens ?? 0,
                  }
                : undefined,
        };
    }
}
