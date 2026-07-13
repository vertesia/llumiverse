import {
    type AIModel,
    type Completion,
    type CompletionResult,
    type DriverCompletionStream,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    getConversationMeta,
    incrementConversationTurn,
    type JSONObject,
    LlumiverseError,
    MISTRAL_DEFAULT_EMBEDDING_MODEL,
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
    ContentChunk,
    ToolCall,
} from '@mistralai/mistralai/models/components';
import {
    HTTPClientError,
    InvalidRequestError,
    MistralError,
    RequestAbortedError,
} from '@mistralai/mistralai/models/errors';
import type {
    OpenAIChatCompletionsDriverOptions,
    OpenAIChatCompletionsPrompt,
} from '../openai/openai_chat_completions.js';
import { type CompatibleAPIError, OpenAICompatibleDriverBase } from '../openai/openai_compatible.js';

const ENDPOINT = 'https://api.mistral.ai';

export interface MistralAIDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export interface MistralPrompt {
    messages: ChatCompletionRequestMessage[];
}

export class MistralAIDriver extends OpenAICompatibleDriverBase<MistralAIDriverOptions, MistralPrompt> {
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

    protected async formatPrompt(segments: PromptSegment[], options: PromptOptions): Promise<MistralPrompt> {
        return { messages: await formatMistralMessages(segments, options) };
    }

    async requestTextCompletion(
        prompt: MistralPrompt | OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
    ): Promise<Completion> {
        const conversation = prepareMistralConversation(options.conversation, prompt);
        const response = await this.client.chat.complete(buildMistralRequest(conversation, options, false));
        const choice = response.choices[0];
        const message = choice?.message;
        if (!message) throw new Error('Mistral response is not valid: no assistant message');

        const includeThoughts =
            (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })?.include_thoughts !== false;
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
    ): Promise<DriverCompletionStream> {
        const conversation = prepareMistralConversation(options.conversation, prompt);
        const response = await this.client.chat.stream(buildMistralRequest(conversation, options, true));
        const includeThoughts =
            (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })?.include_thoughts !== false;
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

function legacyOpenAIMessageToMistral(
    message: OpenAIChatCompletionsPrompt['messages'][number],
): ChatCompletionRequestMessage {
    const textContent = typeof message.content === 'string' || message.content === null ? message.content : undefined;
    const contentParts = Array.isArray(message.content)
        ? message.content.map(
              (part): ContentChunk =>
                  part.type === 'text'
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
            // TODO: Remove after the persisted-conversation migration reports zero
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
): ChatCompletionRequest {
    const modelOptions = options.model_options as TextFallbackOptions;
    return {
        model: options.model,
        messages: conversation.messages,
        maxTokens: modelOptions?.max_tokens,
        temperature: modelOptions?.temperature,
        topP: modelOptions?.top_p,
        presencePenalty: modelOptions?.presence_penalty,
        frequencyPenalty: modelOptions?.frequency_penalty,
        stop: modelOptions?.stop_sequence,
        n: 1,
        tools: options.tools?.map(toMistralTool),
        stream,
    };
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
    const hasSignedThinking = completed.messages.some(
        (storedMessage) =>
            Array.isArray(storedMessage.content) &&
            storedMessage.content.some((part) => part.type === 'thinking' && !!part.signature),
    );
    if (hasSignedThinking) {
        // Mistral does not document a safe way to rewrite history covered by a
        // signed thinking chunk. Retain the complete chain conservatively.
        return completed;
    }
    const currentTurn = getConversationMeta(completed).turnNumber;
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: options.stripTextMaxTokens,
    };
    completed = stripBase64ImagesFromConversation(completed, stripOptions) as MistralPrompt;
    completed = truncateLargeTextInConversation(completed, stripOptions) as MistralPrompt;
    completed = stripHeartbeatsFromConversation(completed, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
    }) as MistralPrompt;
    return completed;
}
