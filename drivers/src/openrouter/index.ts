import {
    type AIModel,
    dataSourceToBase64,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    isDedicatedInferenceModel,
    isEmbeddingModel,
    ModelType,
    normalizeEmbeddingsOptions,
    Providers,
} from '@llumiverse/core';
import { HTTPClient, OpenRouter } from '@openrouter/sdk';
import type {
    ChatContentItems,
    ChatFunctionTool,
    ChatMessages,
    ChatRequest,
    ChatResult,
    ChatStreamChunk,
    ChatUsage,
    Model as OpenRouterModel,
    ProviderPreferences,
} from '@openrouter/sdk/models';
import type { Input as OpenRouterEmbeddingInput } from '@openrouter/sdk/models/operations';
import {
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsRequestMessage,
    type OpenAIChatCompletionsResponse,
    type OpenAIChatCompletionsStreamResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
} from '../openai/openai_chat_completions.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';

export interface OpenRouterDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint?: string;
    httpReferer?: string;
    appTitle?: string;
    appCategories?: string;
}

/** OpenRouter transport backed by the provider's native TypeScript SDK. */
export class OpenRouterDriver extends OpenAIChatCompletionsDriverBase<OpenRouterDriverOptions> {
    static readonly PROVIDER = Providers.openrouter;
    readonly provider = Providers.openrouter;
    service: OpenRouter;

    constructor(options: OpenRouterDriverOptions) {
        super({
            ...options,
            resultSchemaMode: 'response_format',
            toolSchemaMode: 'compatible',
        });
        if (!options.apiKey) {
            throw new Error('apiKey is required');
        }

        this.service = new OpenRouter({
            apiKey: options.apiKey,
            ...(options.endpoint ? { serverURL: options.endpoint } : {}),
            ...(options.httpReferer ? { httpReferer: options.httpReferer } : {}),
            ...(options.appTitle ? { appTitle: options.appTitle } : {}),
            ...(options.appCategories ? { appCategories: options.appCategories } : {}),
            httpClient: new HTTPClient({ fetcher: this.getDriverFetch() }),
            retryConfig: { strategy: 'none' },
            timeoutMs: this.getDriverRequestTimeoutMs(),
        });
    }

    async _postChatCompletion(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse> {
        const response = await this.service.chat.send(
            { chatRequest: toOpenRouterRequest(payload, false, options.model_options) },
            this.getOpenRouterRequestOptions(options, signal),
        );
        const result = asOpenRouterChatResult(response);
        return preserveOpenAIChatCompletionsOriginalResponse(normalizeOpenRouterResponse(result), result);
    }

    async _postChatCompletionStream(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream> {
        const response = await this.service.chat.send(
            { chatRequest: toOpenRouterRequest(payload, true, options.model_options) },
            this.getOpenRouterRequestOptions(options, signal),
        );
        const stream = asOpenRouterChatStream(response);
        return openAIChatCompletionsStreamToSSE(normalizeOpenRouterStream(stream), () => void stream.cancel());
    }

    async listModels(): Promise<AIModel[]> {
        const page = await this.service.models.list({ limit: 1_000 });
        return page.result.data
            .filter((model) => isOpenRouterChatModel(model, this.provider))
            .map((model) => {
                const metadata = resolveModelListingMetadata(model.id, this.provider, {
                    input_modalities: model.architecture.inputModalities,
                    output_modalities: model.architecture.outputModalities,
                });
                return {
                    id: model.id,
                    name: model.name,
                    description: model.description,
                    owner: model.id.split('/')[0],
                    provider: this.provider,
                    type: ModelType.Text,
                    can_stream: true,
                    is_multimodal: metadata.input_modalities.length > 1,
                    ...metadata,
                    tool_support: metadata.tool_support ?? model.supportedParameters.includes('tools'),
                } satisfies AIModel;
            })
            .sort((left, right) => left.id.localeCompare(right.id));
    }

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.models.list({ limit: 1 });
            return true;
        } catch {
            return false;
        }
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        if (!normalized.model) {
            throw new Error("Provider 'openrouter' requires an explicit embedding model.");
        }
        const inputs = await Promise.all(normalized.inputs.map(toOpenRouterEmbeddingInput));
        const response = await this.service.embeddings.generate({
            requestBody: {
                model: normalized.model,
                input: inputs,
                encodingFormat: 'float',
                ...(normalized.dimensions !== undefined ? { dimensions: normalized.dimensions } : {}),
                ...(normalized.task_type ? { inputType: normalized.task_type } : {}),
            },
        });
        if (typeof response === 'string') {
            throw new Error(`OpenRouter embeddings returned an unexpected text response: ${response}`);
        }

        const results = [...response.data]
            .sort((left, right) => (left.index ?? 0) - (right.index ?? 0))
            .map((entry, index): EmbeddingResultItem => {
                if (!Array.isArray(entry.embedding) || entry.embedding.length === 0) {
                    throw new Error(`OpenRouter embedding empty or non-float for input index ${entry.index ?? index}`);
                }
                const inputIndex = entry.index ?? index;
                const sourceInput = normalized.inputs[inputIndex];
                if (!sourceInput) {
                    throw new Error(`OpenRouter embedding references unknown input index ${inputIndex}`);
                }
                return { outputs: [{ values: entry.embedding, modality: sourceInput.type }] };
            });
        const promptTokenDetails = response.usage?.promptTokensDetails;
        const onlyTextInputs = normalized.inputs.every((input) => input.type === 'text');
        return {
            model: response.model,
            results,
            ...(response.usage
                ? {
                      usage: {
                          input_tokens: response.usage.promptTokens,
                          ...(promptTokenDetails?.textTokens !== undefined || onlyTextInputs
                              ? { input_text_tokens: promptTokenDetails?.textTokens ?? response.usage.promptTokens }
                              : {}),
                          ...(promptTokenDetails?.imageTokens !== undefined
                              ? { input_image_tokens: promptTokenDetails.imageTokens }
                              : {}),
                      },
                  }
                : {}),
        };
    }

    private getOpenRouterRequestOptions(
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): { signal?: AbortSignal; timeoutMs?: number } | undefined {
        const timeoutMs = options.httpTimeout ? this.getDriverRequestTimeoutMs(options.httpTimeout) : undefined;
        if (signal && timeoutMs !== undefined) return { signal, timeoutMs };
        if (signal) return { signal };
        if (timeoutMs !== undefined) return { timeoutMs };
        return undefined;
    }
}

async function toOpenRouterEmbeddingInput(
    input: EmbeddingsOptions['inputs'][number],
): Promise<OpenRouterEmbeddingInput> {
    switch (input.type) {
        case 'text':
            return { content: [{ type: 'text', text: input.text }] };
        case 'image':
            return { content: [{ type: 'image_url', imageUrl: { url: await input.source.getURL() } }] };
        case 'audio':
            return {
                content: [
                    {
                        type: 'input_audio',
                        inputAudio: {
                            data: await dataSourceToBase64(input.source),
                            format: mediaFormat(input.source.mime_type),
                        },
                    },
                ],
            };
        case 'video':
            return {
                content: [
                    {
                        type: 'input_video',
                        inputVideo: {
                            data: await dataSourceToBase64(input.source),
                            format: mediaFormat(input.source.mime_type),
                        },
                    },
                ],
            };
    }
}

function mediaFormat(mimeType: string): string {
    return mimeType.split('/')[1] ?? mimeType;
}

function toOpenRouterRequest(
    payload: OpenAIChatCompletionsPayload,
    stream: false,
    modelOptions: ExecutionOptions['model_options'],
): ChatRequest & { stream: false };
function toOpenRouterRequest(
    payload: OpenAIChatCompletionsPayload,
    stream: true,
    modelOptions: ExecutionOptions['model_options'],
): ChatRequest & { stream: true };
function toOpenRouterRequest(
    payload: OpenAIChatCompletionsPayload,
    stream: boolean,
    modelOptions: ExecutionOptions['model_options'],
): ChatRequest {
    return {
        model: payload.model,
        messages: payload.messages.map(toOpenRouterMessage),
        maxTokens: payload.max_tokens ?? undefined,
        temperature: payload.temperature ?? undefined,
        topP: payload.top_p ?? undefined,
        presencePenalty: payload.presence_penalty ?? undefined,
        frequencyPenalty: payload.frequency_penalty ?? undefined,
        stop: payload.stop ?? undefined,
        seed: payload.seed ?? undefined,
        reasoningEffort: payload.reasoning_effort,
        serviceTier: payload.service_tier,
        tools: payload.tools?.flatMap(toOpenRouterTool),
        toolChoice: payload.tool_choice as ChatRequest['toolChoice'],
        parallelToolCalls: payload.parallel_tool_calls,
        responseFormat: toOpenRouterResponseFormat(payload.response_format),
        provider: toOpenRouterProviderPreferences(modelOptions),
        stream,
        ...(stream ? { streamOptions: { includeUsage: true } } : {}),
    } as ChatRequest;
}

function toOpenRouterProviderPreferences(options: ExecutionOptions['model_options']): ProviderPreferences | undefined {
    if (options?._option_id !== 'openrouter-text') return undefined;
    const preferences: ProviderPreferences = {
        sort: options.provider_sort,
        order: options.provider_order,
        only: options.provider_only,
        ignore: options.provider_ignore,
        allowFallbacks: options.provider_allow_fallbacks,
        requireParameters: options.provider_require_parameters,
        dataCollection: options.provider_data_collection,
        zdr: options.provider_zdr,
        quantizations: options.provider_quantizations,
    };
    return Object.values(preferences).some((value) => value !== undefined) ? preferences : undefined;
}

function toOpenRouterMessage(message: OpenAIChatCompletionsRequestMessage): ChatMessages {
    const content = toOpenRouterContent(message.content);
    switch (message.role) {
        case 'assistant':
            return {
                role: 'assistant',
                content,
                reasoning: message.reasoning ?? message.reasoning_content,
                toolCalls: message.tool_calls?.map((toolCall) => ({
                    id: toolCall.id,
                    type: 'function',
                    function: toolCall.function,
                })),
            };
        case 'developer':
        case 'system':
            return { role: message.role, content: toOpenRouterTextContent(message.content) };
        case 'tool':
            if (!message.tool_call_id) {
                throw new TypeError('OpenRouter tool messages require tool_call_id');
            }
            return { role: 'tool', content: content ?? '', toolCallId: message.tool_call_id };
        case 'user':
            if (content === null || content === undefined) {
                throw new TypeError('OpenRouter user messages require content');
            }
            return { role: 'user', content };
        default:
            throw new TypeError(`Unsupported OpenRouter message role: ${message.role}`);
    }
}

function toOpenRouterContent(
    content: OpenAIChatCompletionsRequestMessage['content'],
): string | ChatContentItems[] | null | undefined {
    if (!Array.isArray(content)) return content;
    return content.map((part) =>
        part.type === 'text'
            ? { type: 'text' as const, text: part.text }
            : { type: 'image_url' as const, imageUrl: { ...part.image_url } },
    );
}

function toOpenRouterTextContent(
    content: OpenAIChatCompletionsRequestMessage['content'],
): string | { type: 'text'; text: string }[] {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content.flatMap((part) => (part.type === 'text' ? [{ type: 'text' as const, text: part.text }] : []));
}

function toOpenRouterTool(tool: NonNullable<OpenAIChatCompletionsPayload['tools']>[number]): ChatFunctionTool[] {
    if (tool.type !== 'function') return [];
    return [
        {
            type: 'function',
            function: {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters ?? undefined,
                strict: tool.function.strict,
            },
        },
    ];
}

function toOpenRouterResponseFormat(
    responseFormat: OpenAIChatCompletionsPayload['response_format'],
): ChatRequest['responseFormat'] {
    if (!responseFormat) return undefined;
    if (responseFormat.type === 'json_schema') {
        return {
            type: 'json_schema',
            jsonSchema: {
                name: responseFormat.json_schema.name,
                description: responseFormat.json_schema.description,
                schema: responseFormat.json_schema.schema,
                strict: responseFormat.json_schema.strict,
            },
        };
    }
    if (responseFormat.type === 'json_object') return { type: 'json_object' };
    return { type: 'text' };
}

function normalizeOpenRouterResponse(response: ChatResult): OpenAIChatCompletionsResponse {
    return {
        id: response.id,
        object: response.object,
        created: response.created,
        model: response.model,
        service_tier: response.serviceTier as OpenAIChatCompletionsResponse['service_tier'],
        system_fingerprint: response.systemFingerprint ?? undefined,
        choices: response.choices.map((choice) => ({
            index: choice.index,
            finish_reason: choice.finishReason,
            logprobs: choice.logprobs,
            message: {
                role: choice.message.role,
                content: fromOpenRouterContent(choice.message.content),
                reasoning: choice.message.reasoning,
                tool_calls: choice.message.toolCalls?.map((toolCall) => ({
                    id: toolCall.id,
                    type: 'function',
                    function: toolCall.function,
                })),
            },
        })),
        usage: normalizeOpenRouterChatUsage(response.usage),
    };
}

async function* normalizeOpenRouterStream(
    stream: AsyncIterable<ChatStreamChunk>,
): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    for await (const chunk of stream) {
        if (chunk.error) {
            throw new OpenRouterStreamError(chunk.error.message, chunk.error.code);
        }
        yield {
            id: chunk.id,
            object: chunk.object,
            created: chunk.created,
            model: chunk.model,
            service_tier: chunk.serviceTier as OpenAIChatCompletionsStreamResponse['service_tier'],
            system_fingerprint: chunk.systemFingerprint,
            choices: chunk.choices.map((choice) => ({
                index: choice.index,
                finish_reason: choice.finishReason,
                logprobs: choice.logprobs,
                delta: {
                    role: choice.delta.role,
                    content: choice.delta.content,
                    reasoning: choice.delta.reasoning,
                    tool_calls: choice.delta.toolCalls?.map((toolCall) => ({
                        index: toolCall.index,
                        id: toolCall.id,
                        type: toolCall.type,
                        function: toolCall.function,
                    })),
                },
            })),
            usage: normalizeOpenRouterChatUsage(chunk.usage),
        };
    }
}

function normalizeOpenRouterChatUsage(usage: ChatUsage | undefined): OpenAIChatCompletionsResponse['usage'] {
    return usage
        ? {
              prompt_tokens: usage.promptTokens,
              completion_tokens: usage.completionTokens,
              total_tokens: usage.totalTokens,
          }
        : undefined;
}

function fromOpenRouterContent(
    content: ChatResult['choices'][number]['message']['content'],
): OpenAIChatCompletionsResponse['choices'][number]['message']['content'] {
    if (!Array.isArray(content)) return content;
    const result: NonNullable<OpenAIChatCompletionsResponse['choices'][number]['message']['content']> = [];
    for (const part of content) {
        if (part.type === 'text' && 'text' in part) result.push({ type: 'text', text: part.text });
        if (part.type === 'image_url' && 'imageUrl' in part) {
            const nativeDetail = part.imageUrl.detail as string | undefined;
            const detail: 'auto' | 'low' | 'high' | undefined =
                nativeDetail === 'auto' || nativeDetail === 'low' || nativeDetail === 'high'
                    ? nativeDetail
                    : nativeDetail
                      ? 'high'
                      : undefined;
            result.push({
                type: 'image_url',
                image_url: { url: part.imageUrl.url, ...(detail ? { detail } : {}) },
            });
        }
    }
    return result;
}

type OpenRouterChatStream = AsyncIterable<ChatStreamChunk> & { cancel(): Promise<void> };

function asOpenRouterChatResult(response: ChatResult | OpenRouterChatStream): ChatResult {
    if ('choices' in response) return response;
    throw new TypeError('OpenRouter returned a stream for a non-streaming chat request');
}

function asOpenRouterChatStream(response: ChatResult | OpenRouterChatStream): OpenRouterChatStream {
    if (Symbol.asyncIterator in response) return response;
    throw new TypeError('OpenRouter returned a non-streaming result for a streaming chat request');
}

function isOpenRouterChatModel(model: OpenRouterModel, provider: Providers): boolean {
    return (
        model.architecture.outputModalities.includes('text') &&
        !isEmbeddingModel(
            {
                id: model.id,
                input_modalities: model.architecture.inputModalities,
                output_modalities: model.architecture.outputModalities,
            },
            provider,
        ) &&
        !isDedicatedInferenceModel(model.id, provider)
    );
}

class OpenRouterStreamError extends Error {
    readonly statusCode: number;

    constructor(message: string, statusCode: number) {
        super(message);
        this.name = 'OpenRouterStreamError';
        this.statusCode = statusCode;
    }
}
