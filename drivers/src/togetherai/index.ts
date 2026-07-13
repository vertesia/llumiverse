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
import Together from 'together-ai';
import type {
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionTool,
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
} from 'together-ai/resources/chat/completions';
import type { Embedding, EmbeddingCreateParams } from 'together-ai/resources/embeddings';
import {
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsResponse,
    type OpenAIChatCompletionsStreamResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
} from '../openai/openai_chat_completions.js';

export interface TogetherAIDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint?: string;
}

export class TogetherAIDriver extends OpenAIChatCompletionsDriverBase<TogetherAIDriverOptions> {
    static readonly PROVIDER = Providers.togetherai;
    readonly provider = Providers.togetherai;
    service: Together;

    constructor(opts: TogetherAIDriverOptions) {
        super({ ...opts, resultSchemaMode: 'prompt' });

        if (!opts.apiKey) {
            throw new Error('apiKey is required');
        }

        this.service = new Together({
            apiKey: opts.apiKey,
            baseURL: opts.endpoint ?? 'https://api.together.ai/v1',
            fetch: this.getDriverFetch(),
        });
    }

    async _postChatCompletion(payload: OpenAIChatCompletionsPayload): Promise<OpenAIChatCompletionsResponse> {
        const request = toTogetherRequest(payload, false);
        const response = await this.service.chat.completions.create(request);
        return preserveOpenAIChatCompletionsOriginalResponse(normalizeTogetherResponse(response), response);
    }

    async _postChatCompletionStream(payload: OpenAIChatCompletionsPayload): Promise<ReadableStream> {
        const stream = await this.service.chat.completions.create(toTogetherRequest(payload, true));
        return openAIChatCompletionsStreamToSSE(normalizeTogetherStream(stream));
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

        const request = {
            input: texts,
            model,
            ...(normalized.dimensions !== undefined ? { dimensions: normalized.dimensions } : {}),
            encoding_format: 'float',
        } satisfies EmbeddingCreateParams & { dimensions?: number; encoding_format: 'float' };
        const response: Embedding & { usage?: { prompt_tokens?: number; total_tokens?: number } } =
            await this.service.embeddings.create(request);
        const ordered = [...response.data].sort((a, b) => a.index - b.index);
        const items = ordered.map((entry): EmbeddingResultItem => {
            if (!entry.embedding || entry.embedding.length === 0) {
                throw new Error(`TogetherAI embedding empty for input index ${entry.index}`);
            }
            return { outputs: [{ values: entry.embedding, modality: 'text' }] };
        });
        const inputTokens = response.usage?.prompt_tokens ?? response.usage?.total_tokens;
        return {
            model,
            results: items,
            ...(inputTokens === undefined
                ? {}
                : { usage: { input_tokens: inputTokens, input_text_tokens: inputTokens } }),
        };
    }

    async listModels(): Promise<AIModel[]> {
        const result = await this.service.models.list();
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

function toTogetherRequest(payload: OpenAIChatCompletionsPayload, stream: false): CompletionCreateParamsNonStreaming;
function toTogetherRequest(payload: OpenAIChatCompletionsPayload, stream: true): CompletionCreateParamsStreaming;
function toTogetherRequest(
    payload: OpenAIChatCompletionsPayload,
    stream: boolean,
): CompletionCreateParamsNonStreaming | CompletionCreateParamsStreaming {
    const request = {
        model: payload.model,
        messages: payload.messages.map(toTogetherMessage),
        max_tokens: payload.max_tokens ?? undefined,
        temperature: payload.temperature ?? undefined,
        top_p: payload.top_p ?? undefined,
        presence_penalty: payload.presence_penalty ?? undefined,
        frequency_penalty: payload.frequency_penalty ?? undefined,
        stop: Array.isArray(payload.stop) ? payload.stop : payload.stop ? [payload.stop] : undefined,
        n: payload.n ?? undefined,
        tools: payload.tools?.flatMap(toTogetherTool),
        extra_body: payload.extra_body,
        stream,
    } satisfies (CompletionCreateParamsNonStreaming | CompletionCreateParamsStreaming) & {
        extra_body?: Record<string, unknown>;
    };
    return request;
}

function toTogetherMessage(message: OpenAIChatCompletionsPayload['messages'][number]): ChatCompletionMessageParam {
    const textContent = typeof message.content === 'string' || message.content === null ? message.content : undefined;
    switch (message.role) {
        case 'system':
        case 'developer':
            return { role: 'system', content: textContent ?? '' };
        case 'assistant':
            return {
                role: 'assistant',
                content: textContent,
                tool_calls: message.tool_calls?.map((toolCall, index) => ({
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
                                  : { type: 'image_url' as const, image_url: { ...part.image_url } },
                          ) ?? ''),
            };
    }
}

type TogetherCompatibleTool = ChatCompletionTool & {
    function: ChatCompletionTool['function'] & { strict: false };
};

function toTogetherTool(tool: NonNullable<OpenAIChatCompletionsPayload['tools']>[number]): TogetherCompatibleTool[] {
    if (tool.type !== 'function') {
        return [];
    }
    return [
        {
            type: 'function',
            function: {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters ?? undefined,
                strict: false,
            },
        },
    ];
}

function normalizeTogetherResponse(response: ChatCompletion): OpenAIChatCompletionsResponse {
    return {
        id: response.id,
        object: 'chat.completion',
        created: response.created,
        model: response.model,
        choices: response.choices.map((choice, index) => ({
            index: choice.index ?? index,
            message: {
                role: choice.message?.role,
                content: choice.message?.content,
                reasoning: choice.message?.reasoning,
                tool_calls: choice.message?.tool_calls?.map((tool) => ({
                    id: tool.id,
                    type: 'function',
                    function: {
                        name: tool.function.name,
                        arguments: tool.function.arguments,
                    },
                })),
            },
            finish_reason: normalizeTogetherFinishReason(choice.finish_reason),
            logprobs: choice.logprobs ?? null,
        })),
        usage: response.usage ?? undefined,
    };
}

async function* normalizeTogetherStream(
    stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    for await (const chunk of stream) {
        yield {
            id: chunk.id,
            object: 'chat.completion.chunk',
            created: chunk.created,
            model: chunk.model,
            choices: chunk.choices.map((choice) => ({
                index: choice.index,
                delta: {
                    role: choice.delta.role,
                    content: choice.delta.content,
                    reasoning: choice.delta.reasoning,
                    tool_calls: choice.delta.tool_calls?.map((tool) => ({
                        index: tool.index,
                        id: tool.id,
                        type: 'function',
                        function: {
                            name: tool.function.name,
                            arguments: tool.function.arguments,
                        },
                    })),
                },
                finish_reason: normalizeTogetherFinishReason(choice.finish_reason),
                logprobs: choice.logprobs ?? null,
            })),
            usage: chunk.usage ?? undefined,
        };
    }
}

function normalizeTogetherFinishReason(reason: string | null | undefined): string | null {
    if (reason === 'eos') {
        return 'stop';
    }
    return reason ?? null;
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
