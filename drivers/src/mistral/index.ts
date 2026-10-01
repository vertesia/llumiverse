import {
    type AIModel,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    type Completion,
    type CompletionResult,
    type DriverCompletionStream,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    getConversationMeta,
    getModelCapabilities,
    incrementConversationTurn,
    isEmbeddingModel,
    type JSONObject,
    LlumiverseError,
    MISTRAL_DEFAULT_EMBEDDING_MODEL,
    type MistralTextOptions,
    ModelType,
    modelModalitiesToArray,
    normalizeEmbeddingsOptions,
    type PromptOptions,
    PromptRole,
    type PromptSegment,
    Providers,
    readStreamAsBase64,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    type TextFallbackOptions,
    type ToolDefinition,
    type ToolUse,
    truncateLargeTextInConversation,
} from '@llumiverse/core';
import { HTTPClient, Mistral } from '@mistralai/mistralai';
import type {
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
    ChatCompletionRequestTool,
    ChatCompletionResponse,
    CompletionEvent,
    ContentChunk,
    ToolCall,
} from '@mistralai/mistralai/models/components';
import {
    HTTPClientError,
    InvalidRequestError,
    MistralError,
    RequestAbortedError,
} from '@mistralai/mistralai/models/errors';
import { providerJsonValue } from '../conversation/canonical-runtime.js';
import type { MistralAIDriverOptions } from '../driver-options.js';
import {
    type OpenAIChatCompletionsMessage,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsResponse,
    type OpenAIChatCompletionsStreamResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
} from '../openai/openai_chat_completions.js';
import { type CompatibleAPIError, OpenAICompatibleDriverBase } from '../openai/openai_compatible.js';

export type { MistralAIDriverOptions } from '../driver-options.js';

const ENDPOINT = 'https://api.mistral.ai';
const MISTRAL_CHAT_PROTOCOL = 'mistral.chat.completions';
const MISTRAL_CHAT_ADAPTER_VERSION = '2026-09-30.canonical.1';

export interface MistralPrompt {
    messages: ChatCompletionRequestMessage[];
}

export class MistralAIDriver extends OpenAICompatibleDriverBase<MistralAIDriverOptions, MistralPrompt> {
    static readonly PROVIDER = Providers.mistralai;
    readonly provider = Providers.mistralai;
    readonly apiKey: string;
    readonly client: Mistral;
    readonly endpointUrl?: string;
    private readonly canonicalProtocol: MistralCanonicalChatProtocol;

    constructor(options: MistralAIDriverOptions) {
        super({ ...options, resultSchemaMode: 'prompt', toolSchemaMode: 'compatible' });
        this.apiKey = options.apiKey;
        this.endpointUrl = options.endpoint_url;
        this.client = new Mistral({
            apiKey: options.apiKey,
            serverURL: options.endpoint_url ?? ENDPOINT,
            httpClient: new HTTPClient({ fetcher: this.getDriverFetch() }),
            timeoutMs: this.getDriverRequestTimeoutMs(),
        });
        this.canonicalProtocol = new MistralCanonicalChatProtocol(options.defaultMaxTokens);
    }

    protected async formatPrompt(segments: PromptSegment[], options: PromptOptions): Promise<MistralPrompt> {
        return { messages: await formatMistralMessages(segments, options) };
    }

    protected supportsCanonicalConversation(_options: ExecutionOptions): boolean {
        return true;
    }

    async requestCanonicalTextCompletion(
        prompt: MistralPrompt | OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        return this.canonicalProtocol.requestCanonicalTextCompletion(
            this,
            mistralPromptToOpenAI(prompt),
            canonicalMistralOptions(options),
            signal,
        );
    }

    async requestCanonicalTextCompletionEventStream(
        prompt: MistralPrompt | OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        return this.canonicalProtocol.requestCanonicalTextCompletionEventStream(
            this,
            mistralPromptToOpenAI(prompt),
            canonicalMistralOptions(options),
            signal,
            open,
        );
    }

    async requestTextCompletion(
        prompt: MistralPrompt | OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const conversation = prepareMistralConversation(options.conversation, prompt);
        const request = buildMistralRequest(conversation, options, false, this.options.defaultMaxTokens);
        const driverRequestOptions = this.getDriverRequestOptions(options, signal);
        const requestOptions = driverRequestOptions
            ? { signal: driverRequestOptions.signal, timeoutMs: driverRequestOptions.timeout }
            : undefined;
        const response = requestOptions
            ? await this.client.chat.complete(request, requestOptions)
            : await this.client.chat.complete(request);
        const choice = response.choices[0];
        const message = choice?.message;
        if (!message) throw new Error('Mistral response is not valid: no assistant message');

        const includeThoughts =
            (options.model_options as TextFallbackOptions | MistralTextOptions | undefined)?.include_thoughts !== false;
        const result = projectMistralContent(message.content, includeThoughts);
        const tool_use = collectMistralTools(message.toolCalls);
        const completed = finalizeMistralConversation(conversation, { ...message, role: 'assistant' }, options);

        return {
            result,
            tool_use,
            token_usage: mapMistralUsage(response),
            finish_reason: tool_use?.length ? 'tool_use' : (choice.finishReason ?? undefined),
            original_response: options.include_original_response ? response : undefined,
            conversation: completed,
        };
    }

    async requestTextCompletionStream(
        prompt: MistralPrompt | OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        const conversation = prepareMistralConversation(options.conversation, prompt);
        const request = buildMistralRequest(conversation, options, true, this.options.defaultMaxTokens);
        const driverRequestOptions = this.getDriverRequestOptions(options, signal);
        const requestOptions =
            driverRequestOptions?.timeout !== undefined
                ? { signal: driverRequestOptions.signal, timeoutMs: driverRequestOptions.timeout }
                : { signal };
        const response = await this.client.chat.stream(request, requestOptions);
        const includeThoughts =
            (options.model_options as TextFallbackOptions | MistralTextOptions | undefined)?.include_thoughts !== false;
        const nativeContent: ContentChunk[] = [];
        const nativeToolCalls = new Map<number, ToolCall>();

        const stream: DriverCompletionStream = {
            async *[Symbol.asyncIterator]() {
                for await (const event of response) {
                    const chunk = event.data;
                    const choice = chunk.choices[0];
                    const delta = choice?.delta;
                    if (!delta) continue;

                    const content = normalizeMistralDeltaContent(delta.content);
                    appendMistralContent(nativeContent, content);
                    const tool_use = appendMistralToolDeltas(nativeToolCalls, delta.toolCalls);
                    const projected = projectMistralContent(content, includeThoughts);
                    yield {
                        result: projected,
                        tool_use,
                        finish_reason: tool_use?.length ? 'tool_use' : (choice.finishReason ?? undefined),
                        token_usage: chunk.usage
                            ? {
                                  prompt: chunk.usage.promptTokens ?? 0,
                                  result: chunk.usage.completionTokens ?? 0,
                                  total: chunk.usage.totalTokens ?? 0,
                              }
                            : undefined,
                    };
                }
            },
            finalizeConversation: () =>
                finalizeMistralConversation(
                    conversation,
                    {
                        role: 'assistant',
                        content: nativeContent,
                        toolCalls: [...nativeToolCalls.entries()]
                            .sort(([left], [right]) => left - right)
                            .map(([, toolCall]) => toolCall),
                    },
                    options,
                ),
        };
        return stream;
    }

    async listModels(): Promise<AIModel[]> {
        const models = await this.client.models.list();
        return (models.data ?? []).flatMap((model) => {
            if (!('id' in model) || isEmbeddingModel({ id: model.id }, this.provider)) return [];
            // The Models API explicitly identifies artifacts that cannot use Chat Completions. Keep entries with
            // absent capability metadata visible because runtime metadata is incomplete for some valid chat models.
            if ('capabilities' in model && model.capabilities?.completionChat === false) return [];
            const capabilities = getModelCapabilities(model.id, this.provider);
            return [
                {
                    id: model.id,
                    name: ('name' in model && model.name) || model.id,
                    description: ('description' in model && model.description) || undefined,
                    provider: this.provider,
                    owner: 'ownedBy' in model ? model.ownedBy : '',
                    type: ModelType.Text,
                    can_stream: true,
                    is_multimodal: capabilities.input.image === true,
                    input_modalities: modelModalitiesToArray(capabilities.input),
                    output_modalities: modelModalitiesToArray(capabilities.output),
                    tool_support: capabilities.tool_support,
                } satisfies AIModel,
            ];
        });
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
        try {
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
        } catch (error: unknown) {
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model,
                operation: 'execute',
            });
        }
    }

    /** @internal Resolve request cancellation/timeout for the canonical protocol adapter. */
    getMistralRequestOptions(options: ExecutionOptions, signal?: AbortSignal) {
        return this.getDriverRequestOptions(options, signal);
    }

    protected isCompatibleAPIError(error: unknown): error is CompatibleAPIError {
        return error instanceof MistralError || error instanceof HTTPClientError || super.isCompatibleAPIError(error);
    }

    protected isOpenAIErrorRetryable(
        error: unknown,
        httpStatusCode: number | undefined,
        errorCode: string | null | undefined,
        errorType: string | undefined,
    ): boolean | undefined {
        if (error instanceof RequestAbortedError) return true;
        if (error instanceof InvalidRequestError) return false;
        return super.isOpenAIErrorRetryable(error, httpStatusCode, errorCode, errorType);
    }
}

type MistralReplayPayload = {
    type: 'mistral_assistant_content';
    content: ContentChunk[];
};

function isJsonRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isMistralThinkingPart(value: unknown): boolean {
    if (!isJsonRecord(value)) return false;
    if (value.type === 'text') return typeof value.text === 'string';
    if (value.type === 'reference') {
        return (
            Array.isArray(value.referenceIds) &&
            value.referenceIds.every((id) => typeof id === 'string' || typeof id === 'number')
        );
    }
    if (value.type === 'tool_reference') {
        return (
            typeof value.tool === 'string' &&
            typeof value.title === 'string' &&
            (value.url === undefined || value.url === null || typeof value.url === 'string') &&
            (value.favicon === undefined || value.favicon === null || typeof value.favicon === 'string') &&
            (value.description === undefined || value.description === null || typeof value.description === 'string')
        );
    }
    return false;
}

function isMistralAssistantReplayChunk(value: unknown): value is ContentChunk {
    if (!isJsonRecord(value)) return false;
    if (value.type === 'text') return typeof value.text === 'string';
    if (value.type === 'thinking') {
        return (
            Array.isArray(value.thinking) &&
            value.thinking.every(isMistralThinkingPart) &&
            (value.signature === undefined || value.signature === null || typeof value.signature === 'string') &&
            (value.closed === undefined || typeof value.closed === 'boolean')
        );
    }
    if (value.type === 'reference') {
        return (
            Array.isArray(value.referenceIds) &&
            value.referenceIds.every((id) => typeof id === 'string' || typeof id === 'number')
        );
    }
    return false;
}

function mistralReplayPayload(value: unknown): MistralReplayPayload {
    if (
        !isJsonRecord(value) ||
        value.type !== 'mistral_assistant_content' ||
        !Array.isArray(value.content) ||
        !value.content.every(isMistralAssistantReplayChunk)
    ) {
        throw new TypeError('Mistral signed-thinking replay has an unsupported payload');
    }
    return structuredClone(value) as MistralReplayPayload;
}

function mistralContentSemantics(content: string | ContentChunk[] | null | undefined): {
    text: string;
    reasoning: string;
} {
    if (typeof content === 'string') return { text: content, reasoning: '' };
    let text = '';
    let reasoning = '';
    for (const part of content ?? []) {
        if (part.type === 'text') text += part.text;
        if (part.type === 'thinking') {
            for (const thought of part.thinking) {
                if (thought.type === 'text') reasoning += thought.text;
            }
        }
    }
    return { text, reasoning };
}

function openAIMessageText(message: OpenAIChatCompletionsMessage): string {
    if (typeof message.content === 'string') return message.content;
    return (message.content ?? [])
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('');
}

function mistralContentToOpenAI(
    content: string | ContentChunk[] | null | undefined,
    includeReplay = true,
): Pick<OpenAIChatCompletionsMessage, 'content' | 'reasoning_content' | 'provider_replay'> {
    const semantics = mistralContentSemantics(content);
    const replayContent = typeof content === 'string' || content == null ? undefined : structuredClone(content);
    const replayPayload =
        replayContent === undefined || !includeReplay
            ? undefined
            : mistralReplayPayload({
                  type: 'mistral_assistant_content',
                  content: replayContent,
              });
    return {
        content: semantics.text || null,
        ...(semantics.reasoning ? { reasoning_content: semantics.reasoning } : {}),
        ...(replayPayload === undefined
            ? {}
            : {
                  provider_replay: {
                      provider: Providers.mistralai,
                      protocol: MISTRAL_CHAT_PROTOCOL,
                      adapter_version: MISTRAL_CHAT_ADAPTER_VERSION,
                      payload: providerJsonValue(replayPayload),
                  },
              }),
    };
}

function mistralToolCallsToOpenAI(toolCalls: ToolCall[] | null | undefined) {
    return toolCalls?.map((toolCall) => ({
        id: toolCall.id ?? '',
        type: 'function' as const,
        function: {
            name: toolCall.function.name,
            arguments:
                typeof toolCall.function.arguments === 'string'
                    ? toolCall.function.arguments
                    : toolCall.function.arguments == null
                      ? ''
                      : JSON.stringify(toolCall.function.arguments),
        },
    }));
}

function mistralMessageToOpenAI(message: ChatCompletionRequestMessage): OpenAIChatCompletionsMessage {
    const role = message.role === 'system' ? 'system' : message.role;
    if (role === 'assistant') {
        const assistant = message as Extract<ChatCompletionRequestMessage, { role: 'assistant' }>;
        return {
            role,
            ...mistralContentToOpenAI(assistant.content),
            ...(assistant.toolCalls?.length ? { tool_calls: mistralToolCallsToOpenAI(assistant.toolCalls) } : {}),
        };
    }
    const content = message.content;
    if (typeof content === 'string' || content == null) {
        return {
            role,
            content: content ?? '',
            ...(role === 'tool' && 'toolCallId' in message && typeof message.toolCallId === 'string'
                ? { tool_call_id: message.toolCallId }
                : {}),
        };
    }
    const parts = content.map((part) => {
        if (part.type === 'text') return { type: 'text' as const, text: part.text };
        if (part.type === 'image_url') {
            return {
                type: 'image_url' as const,
                image_url: {
                    url: typeof part.imageUrl === 'string' ? part.imageUrl : part.imageUrl.url,
                    detail: 'auto' as const,
                },
            };
        }
        if (part.type === 'input_audio') {
            throw new TypeError(
                'Mistral canonical execution cannot infer the MIME type of retained native audio input',
            );
        }
        throw new TypeError(`Mistral canonical execution cannot import native ${part.type} content`);
    });
    return {
        role,
        content: parts,
        ...(role === 'tool' && 'toolCallId' in message && typeof message.toolCallId === 'string'
            ? { tool_call_id: message.toolCallId }
            : {}),
    };
}

function mistralPromptToOpenAI(prompt: MistralPrompt | OpenAIChatCompletionsPrompt): OpenAIChatCompletionsPrompt {
    if ('_is_openai_chat_completions' in prompt && prompt._is_openai_chat_completions === true) return prompt;
    const native = prompt as MistralPrompt;
    return {
        _is_openai_chat_completions: true,
        messages: native.messages.map(mistralMessageToOpenAI),
    };
}

function canonicalMistralOptions(options: ExecutionOptions): ExecutionOptions {
    const conversation = options.conversation;
    if (
        !isJsonRecord(conversation) ||
        !Array.isArray(conversation.messages) ||
        conversation._is_openai_chat_completions === true
    ) {
        return options;
    }
    return {
        ...options,
        conversation: mistralPromptToOpenAI({ messages: conversation.messages as ChatCompletionRequestMessage[] }),
    };
}

function openAIContentToMistral(message: OpenAIChatCompletionsMessage): string | ContentChunk[] | null | undefined {
    const replay = message.provider_replay;
    if (replay !== undefined) {
        if (
            replay.provider !== Providers.mistralai ||
            replay.protocol !== MISTRAL_CHAT_PROTOCOL ||
            replay.adapter_version !== MISTRAL_CHAT_ADAPTER_VERSION
        ) {
            throw new TypeError('Mistral cannot restore provider replay outside its compatibility scope');
        }
        const payload = mistralReplayPayload(replay.payload);
        const semantics = mistralContentSemantics(payload.content);
        const visibleReasoning = message.reasoning_content ?? message.reasoning ?? '';
        if (semantics.text !== openAIMessageText(message) || semantics.reasoning !== visibleReasoning) {
            throw new TypeError('Mistral signed-thinking replay no longer matches canonical semantic content');
        }
        return payload.content;
    }
    const content = message.content;
    const parts: ContentChunk[] =
        typeof content === 'string' || content == null
            ? content
                ? [{ type: 'text', text: content }]
                : []
            : content.map((part): ContentChunk => {
                  if (part.type === 'text') return { type: 'text', text: part.text };
                  if (part.type === 'image_url') return { type: 'image_url', imageUrl: part.image_url.url };
                  return { type: 'input_audio', inputAudio: part.input_audio.data };
              });
    const reasoning = message.reasoning_content ?? message.reasoning;
    if (reasoning) {
        parts.unshift({
            type: 'thinking',
            thinking: [{ type: 'text', text: reasoning }],
            closed: true,
        });
    }
    if (parts.length === 0) return null;
    if (parts.length === 1 && parts[0]?.type === 'text') return parts[0].text;
    return parts;
}

function openAIMessageToMistral(message: OpenAIChatCompletionsMessage): ChatCompletionRequestMessage {
    switch (message.role) {
        case 'system':
        case 'developer':
            return { role: 'system', content: openAIMessageText(message) };
        case 'assistant':
            return {
                role: 'assistant',
                content: openAIContentToMistral(message),
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
            if (!message.tool_call_id) throw new TypeError('Mistral tool messages require tool_call_id');
            return {
                role: 'tool',
                toolCallId: message.tool_call_id,
                content: openAIContentToMistral(message) ?? '',
            };
        case 'user':
            return { role: 'user', content: openAIContentToMistral(message) ?? '' };
        default:
            throw new TypeError(`Mistral does not support canonical role ${message.role}`);
    }
}

/** @internal Translate the canonical compatibility projection to the exact Mistral SDK request. */
export function mistralRequestFromOpenAI(
    payload: OpenAIChatCompletionsPayload,
    options: ExecutionOptions,
    defaultMaxTokens?: number,
): ChatCompletionRequest {
    return buildMistralRequest(
        { messages: payload.messages.map(openAIMessageToMistral) },
        options,
        payload.stream,
        defaultMaxTokens,
    );
}

function normalizedMistralUsage(usage: ChatCompletionResponse['usage'] | undefined) {
    if (usage === undefined) return undefined;
    return {
        // The installed Mistral SDK's inbound UsageInfo schema defaults omitted count fields to zero.
        prompt_tokens: usage.promptTokens ?? 0,
        completion_tokens: usage.completionTokens ?? 0,
        total_tokens: usage.totalTokens ?? 0,
        provider_usage: providerJsonValue({
            source: 'mistral_sdk_usage_info',
            omitted_token_count_semantics: 'sdk_default_zero',
            payload: usage,
        }),
    };
}

function normalizeMistralFinishReason(reason: string | null | undefined): string | null {
    return reason === 'model_length' ? 'length' : (reason ?? null);
}

function normalizeMistralResponse(response: ChatCompletionResponse): OpenAIChatCompletionsResponse {
    return preserveOpenAIChatCompletionsOriginalResponse(
        {
            id: response.id,
            object: 'chat.completion',
            created: response.created,
            model: response.model,
            choices: response.choices.flatMap((choice) =>
                choice.message === undefined
                    ? []
                    : [
                          {
                              index: choice.index,
                              finish_reason: normalizeMistralFinishReason(choice.finishReason),
                              message: {
                                  role: 'assistant',
                                  ...mistralContentToOpenAI(choice.message.content),
                                  tool_calls: mistralToolCallsToOpenAI(choice.message.toolCalls),
                              },
                          },
                      ],
            ),
            usage: normalizedMistralUsage(response.usage),
        },
        response,
    );
}

/** @internal Normalize native Mistral chunks without repeatedly serializing cumulative protected replay. */
export async function* normalizeMistralStream(
    stream: AsyncIterable<CompletionEvent>,
): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    const replayContent: ContentChunk[] = [];
    let lastChunk: CompletionEvent['data'] | undefined;
    let lastChoiceIndex = 0;
    for await (const event of stream) {
        const chunk = event.data;
        lastChunk = chunk;
        const choice = chunk.choices[0];
        if (choice !== undefined) lastChoiceIndex = choice.index;
        const deltaContent = normalizeMistralDeltaContent(choice?.delta.content);
        appendMistralContent(replayContent, deltaContent);
        const normalizedContent = mistralContentToOpenAI(deltaContent, false);
        yield {
            id: chunk.id,
            object: 'chat.completion.chunk',
            created: chunk.created as number,
            model: chunk.model,
            choices:
                choice === undefined
                    ? []
                    : [
                          {
                              index: choice.index,
                              finish_reason: normalizeMistralFinishReason(choice.finishReason),
                              delta: {
                                  role: choice.delta.role ?? undefined,
                                  ...normalizedContent,
                                  tool_calls: choice.delta.toolCalls?.map((toolCall, offset) => ({
                                      index: toolCall.index ?? offset,
                                      id: toolCall.id ?? undefined,
                                      type: toolCall.type,
                                      function: {
                                          name: toolCall.function.name,
                                          arguments:
                                              typeof toolCall.function.arguments === 'string'
                                                  ? toolCall.function.arguments
                                                  : toolCall.function.arguments == null
                                                    ? ''
                                                    : JSON.stringify(toolCall.function.arguments),
                                      },
                                  })),
                              },
                          },
                      ],
            usage: normalizedMistralUsage(chunk.usage),
        };
    }
    const providerReplay = mistralContentToOpenAI(replayContent).provider_replay;
    if (lastChunk !== undefined && providerReplay !== undefined) {
        yield {
            id: lastChunk.id,
            object: 'chat.completion.chunk',
            created: lastChunk.created as number,
            model: lastChunk.model,
            choices: [
                {
                    index: lastChoiceIndex,
                    finish_reason: null,
                    delta: { provider_replay: providerReplay },
                },
            ],
        };
    }
}

class MistralCanonicalChatProtocol extends OpenAIChatCompletionsProtocol<MistralAIDriver> {
    constructor(private readonly defaultMaxTokens?: number) {
        super({ resultSchemaMode: 'prompt', toolSchemaMode: 'compatible', defaultMaxTokens });
    }

    protected override requestBinding(payload: OpenAIChatCompletionsPayload, options: ExecutionOptions) {
        const request = mistralRequestFromOpenAI(payload, options, this.defaultMaxTokens);
        const { messages: _messages, tools: _tools, metadata: _metadata, ...effectiveOptions } = request;
        return {
            payload: providerJsonValue(request),
            target_options: providerJsonValue({
                transport: 'mistral_sdk',
                ...effectiveOptions,
            }) as JSONObject,
        };
    }

    protected async postChatCompletion(
        driver: MistralAIDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse> {
        const request = mistralRequestFromOpenAI(payload, options, this.defaultMaxTokens);
        const driverRequestOptions = driver.getMistralRequestOptions(options, signal);
        const requestOptions = driverRequestOptions
            ? { signal: driverRequestOptions.signal, timeoutMs: driverRequestOptions.timeout }
            : undefined;
        const response = requestOptions
            ? await driver.client.chat.complete(request, requestOptions)
            : await driver.client.chat.complete(request);
        return normalizeMistralResponse(response);
    }

    protected async postChatCompletionStream(
        driver: MistralAIDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream> {
        const request = mistralRequestFromOpenAI(payload, options, this.defaultMaxTokens);
        const driverRequestOptions = driver.getMistralRequestOptions(options, signal);
        const requestOptions =
            driverRequestOptions?.timeout !== undefined
                ? { signal: driverRequestOptions.signal, timeoutMs: driverRequestOptions.timeout }
                : { signal };
        const stream = await driver.client.chat.stream(request, requestOptions);
        return openAIChatCompletionsStreamToSSE(normalizeMistralStream(stream));
    }
}

function legacyOpenAIMessageToMistral(
    message: OpenAIChatCompletionsPrompt['messages'][number],
): ChatCompletionRequestMessage {
    const textContent = typeof message.content === 'string' || message.content === null ? message.content : undefined;
    const contentParts = Array.isArray(message.content)
        ? message.content.map(
              (part): ContentChunk =>
                  part.type === 'input_audio'
                      ? unsupportedAudioPart()
                      : part.type === 'text'
                        ? { type: 'text', text: part.text }
                        : { type: 'image_url', imageUrl: part.image_url.url },
          )
        : undefined;
    switch (message.role) {
        case 'system':
        case 'developer':
            return { role: 'system', content: textContent ?? '' };
        case 'assistant':
            return {
                role: 'assistant',
                content: contentParts ?? textContent,
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
            return { role: 'tool', content: contentParts ?? textContent ?? '', toolCallId: message.tool_call_id };
        default:
            return {
                role: 'user',
                content: typeof message.content === 'string' ? message.content : (contentParts ?? ''),
            };
    }
}

async function formatMistralMessages(
    segments: PromptSegment[],
    options: PromptOptions,
): Promise<ChatCompletionRequestMessage[]> {
    const messages: ChatCompletionRequestMessage[] = [];
    const system = segments
        .filter((segment) => segment.role === PromptRole.system && segment.content)
        .map((segment) => segment.content)
        .join('\n');
    if (system) messages.push({ role: 'system', content: system });
    if (options.result_schema) {
        messages.push({
            role: 'system',
            content: `Only answer with JSON matching this schema: ${JSON.stringify(options.result_schema)}`,
        });
    }

    for (const segment of segments) {
        if (segment.role === PromptRole.system) continue;
        const parts: ContentChunk[] = [];
        if (segment.content) parts.push({ type: 'text', text: segment.content });
        for (const file of segment.files ?? []) {
            if (file.mime_type?.startsWith('image/')) {
                parts.push({
                    type: 'image_url',
                    imageUrl: `data:${file.mime_type};base64,${await readStreamAsBase64(await file.getStream())}`,
                });
            } else if (file.mime_type?.startsWith('audio/')) {
                parts.push({
                    type: 'input_audio',
                    inputAudio: await readStreamAsBase64(await file.getStream()),
                });
            } else if (file.mime_type?.startsWith('text/')) {
                const chunks: Buffer[] = [];
                for await (const chunk of await file.getStream()) chunks.push(Buffer.from(chunk));
                parts.push({ type: 'text', text: Buffer.concat(chunks).toString('utf8') });
            }
        }
        const content = parts.length === 1 && parts[0]?.type === 'text' ? parts[0].text : parts;
        if (segment.role === PromptRole.tool) {
            if (!segment.tool_use_id) throw new Error('Mistral tool response requires tool_use_id');
            messages.push({ role: 'tool', toolCallId: segment.tool_use_id, content });
        } else {
            messages.push({ role: segment.role === PromptRole.assistant ? 'assistant' : 'user', content });
        }
    }
    return messages;
}

function prepareMistralConversation(
    conversation: unknown,
    prompt: MistralPrompt | OpenAIChatCompletionsPrompt,
): MistralPrompt {
    let existing: ChatCompletionRequestMessage[] = [];
    if (conversation && typeof conversation === 'object' && 'messages' in conversation) {
        const stored = conversation as { messages?: unknown[]; _is_openai_chat_completions?: boolean };
        if (Array.isArray(stored.messages)) {
            // TODO: Remove after 2026-08-14 once migration telemetry reports zero
            // `_is_openai_chat_completions` Mistral records for one full release cycle.
            existing = stored._is_openai_chat_completions
                ? (stored.messages as OpenAIChatCompletionsPrompt['messages']).map(legacyOpenAIMessageToMistral)
                : (stored.messages as ChatCompletionRequestMessage[]);
        }
    }
    const isLegacy = (prompt as OpenAIChatCompletionsPrompt)._is_openai_chat_completions === true;
    const promptMessages = isLegacy
        ? (prompt as OpenAIChatCompletionsPrompt).messages.map(legacyOpenAIMessageToMistral)
        : (prompt as MistralPrompt).messages;
    return { messages: [...existing, ...promptMessages] };
}

function buildMistralRequest(
    conversation: MistralPrompt,
    options: ExecutionOptions,
    stream: boolean,
    defaultMaxTokens?: number,
): ChatCompletionRequest {
    // Continue reading effort from the previously advertised OpenAI-compatible shape so persisted configurations
    // remain usable; Mistral-native fields are only accepted through the transport-specific discriminator.
    const modelOptions = options.model_options as
        | ((MistralTextOptions | (TextFallbackOptions & { effort?: unknown })) & { required_tool_name?: string })
        | undefined;
    const mistralOptions = modelOptions?._option_id === 'mistral-text' ? modelOptions : undefined;
    const toolChoice = modelOptions?.required_tool_name
        ? ({ type: 'function', function: { name: modelOptions.required_tool_name } } as const)
        : mistralOptions?.tool_choice;
    return {
        model: options.model,
        messages: conversation.messages,
        maxTokens: modelOptions?.max_tokens ?? defaultMaxTokens,
        temperature: modelOptions?.temperature,
        topP: modelOptions?.top_p,
        presencePenalty: modelOptions?.presence_penalty,
        frequencyPenalty: modelOptions?.frequency_penalty,
        stop: modelOptions?.stop_sequence,
        randomSeed: mistralOptions?.random_seed,
        safePrompt: mistralOptions?.safe_prompt,
        parallelToolCalls: mistralOptions?.parallel_tool_calls,
        toolChoice,
        promptMode: mistralOptions?.prompt_mode,
        promptCacheKey: options.prompt_cache_key,
        n: 1,
        tools: options.tools?.map(toMistralTool),
        reasoningEffort:
            modelOptions?.effort === 'none' || modelOptions?.effort === 'high' ? modelOptions.effort : undefined,
        stream,
    } satisfies ChatCompletionRequest & { reasoningEffort?: 'none' | 'high' };
}

function toMistralTool(tool: ToolDefinition): ChatCompletionRequestTool {
    return {
        type: 'function',
        function: {
            name: tool.name,
            description: tool.description,
            parameters: (tool.input_schema as JSONObject | undefined) ?? {},
        },
    };
}

function projectMistralContent(
    content: string | ContentChunk[] | null | undefined,
    includeThoughts: boolean,
): CompletionResult[] {
    if (typeof content === 'string') return content ? [{ type: 'text', value: content }] : [];
    const result: CompletionResult[] = [];
    for (const part of content ?? []) {
        if (part.type === 'text' && part.text) {
            result.push({ type: 'text', value: part.text });
        } else if (part.type === 'thinking' && includeThoughts) {
            for (const thought of part.thinking) {
                if (thought.type === 'text' && thought.text) result.push({ type: 'thoughts', value: thought.text });
            }
        }
    }
    return result;
}

function normalizeMistralDeltaContent(content: string | ContentChunk[] | null | undefined): ContentChunk[] {
    if (typeof content === 'string') return content ? [{ type: 'text', text: content }] : [];
    return content ?? [];
}

function appendMistralContent(target: ContentChunk[], incoming: ContentChunk[]): void {
    for (const part of incoming) {
        const previous = target[target.length - 1];
        if (part.type === 'text' && previous?.type === 'text') {
            previous.text += part.text;
        } else if (part.type === 'thinking' && previous?.type === 'thinking' && !previous.closed) {
            previous.thinking.push(...structuredClone(part.thinking));
            if (part.signature !== undefined) previous.signature = part.signature;
            if (part.closed !== undefined) previous.closed = part.closed;
        } else {
            target.push(structuredClone(part));
        }
    }
}

function appendMistralToolDeltas(
    target: Map<number, ToolCall>,
    deltas: ToolCall[] | null | undefined,
): ToolUse<unknown>[] | undefined {
    if (!deltas?.length) return undefined;
    return deltas.map((delta, offset) => {
        const index = delta.index ?? offset;
        const current = target.get(index) ?? {
            id: '',
            index,
            type: 'function' as const,
            function: { name: '', arguments: '' },
        };
        if (delta.id) current.id = delta.id;
        if (delta.function.name) current.function.name += delta.function.name;
        const args = delta.function.arguments;
        if (typeof args === 'string') {
            current.function.arguments = `${current.function.arguments ?? ''}${args}`;
        } else if (args) {
            current.function.arguments = args;
        }
        target.set(index, current);
        return {
            id: `tool_${index}`,
            tool_name: delta.function.name ?? '',
            tool_input: typeof args === 'string' ? args : ((args as JSONObject | undefined) ?? {}),
            ...(delta.id && { _actual_id: delta.id }),
        };
    });
}

function collectMistralTools(toolCalls: ToolCall[] | null | undefined): ToolUse[] | undefined {
    const tools = toolCalls?.map((toolCall) => ({
        id: toolCall.id ?? '',
        tool_name: toolCall.function.name,
        tool_input:
            typeof toolCall.function.arguments === 'string'
                ? safeJsonParse(toolCall.function.arguments)
                : (toolCall.function.arguments as JSONObject),
    }));
    return tools?.length ? tools : undefined;
}

function safeJsonParse(value: string): JSONObject {
    try {
        const parsed = JSON.parse(value) as unknown;
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as JSONObject) : {};
    } catch {
        return {};
    }
}

function mapMistralUsage(response: ChatCompletionResponse): ExecutionTokenUsage {
    return {
        prompt: response.usage.promptTokens ?? 0,
        result: response.usage.completionTokens ?? 0,
        total: response.usage.totalTokens ?? 0,
    };
}

function finalizeMistralConversation(
    conversation: MistralPrompt,
    message: ChatCompletionRequestMessage,
    options: ExecutionOptions,
): MistralPrompt {
    let completed = incrementConversationTurn({ messages: [...conversation.messages, message] }) as MistralPrompt;
    const currentTurn = getConversationMeta(completed).turnNumber;
    const preserveSubtree = (value: unknown): boolean => {
        if (!value || typeof value !== 'object') return false;
        const content = (value as { content?: unknown }).content;
        return (
            Array.isArray(content) &&
            content.some(
                (part) =>
                    !!part &&
                    typeof part === 'object' &&
                    (part as { type?: unknown }).type === 'thinking' &&
                    !!(part as { signature?: unknown }).signature,
            )
        );
    };
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: options.stripTextMaxTokens,
        preserveSubtree,
    };
    completed = stripBase64ImagesFromConversation(completed, stripOptions) as MistralPrompt;
    completed = truncateLargeTextInConversation(completed, stripOptions) as MistralPrompt;
    completed = stripHeartbeatsFromConversation(completed, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
        preserveSubtree,
    }) as MistralPrompt;
    return completed;
}

function unsupportedAudioPart(): never {
    throw new Error('This inference endpoint does not support audio input');
}
