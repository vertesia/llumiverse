import {
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    type DriverOptions,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    getConversationMeta,
    incrementConversationTurn,
    type JSONObject,
    type JSONSchema,
    type PromptOptions,
    PromptRole,
    type PromptSegment,
    readStreamAsBase64,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    type TextFallbackOptions,
    type ToolDefinition,
    type ToolUse,
    truncateLargeTextInConversation,
} from '@llumiverse/core';
import { transformSSEStream } from '@llumiverse/core/async';
import { AbstractDriver } from '@llumiverse/core/driver';
import type OpenAI from 'openai';
import { formatOpenAISchema } from './schema.js';

export type OpenAIChatCompletionsTextPart = OpenAI.Chat.ChatCompletionContentPartText;
export type OpenAIChatCompletionsImageUrlPart = OpenAI.Chat.ChatCompletionContentPartImage;
export type OpenAIChatCompletionsContentPart = OpenAIChatCompletionsTextPart | OpenAIChatCompletionsImageUrlPart;
export type OpenAIChatCompletionsToolCall = OpenAI.Chat.ChatCompletionMessageFunctionToolCall;
export type OpenAIChatCompletionsToolDefinition = OpenAI.Chat.ChatCompletionTool;

export type OpenAIChatCompletionsMessage = {
    role: string;
    content?: string | null | OpenAIChatCompletionsContentPart[];
    /**
     * Required for tool role messages - references the tool_call.id from the assistant's message.
     * Per OpenAI API spec: https://platform.openai.com/docs/api-reference/chat/messages#message-role
     */
    tool_call_id?: string;
    /**
     * Tool calls from assistant messages - stored and sent back with tool results.
     */
    tool_calls?: OpenAIChatCompletionsToolCall[];
};

export type OpenAIChatCompletionsRequestMessage = {
    role: string;
    content?: string | null | OpenAIChatCompletionsContentPart[];
    tool_call_id?: string;
    tool_calls?: OpenAIChatCompletionsToolCall[];
};

export type OpenAIChatCompletionsPayload = Omit<
    OpenAI.Chat.ChatCompletionCreateParams,
    'model' | 'messages' | 'stream' | 'tools'
> & {
    model: string;
    messages: OpenAIChatCompletionsRequestMessage[];
    stream: boolean;
    tools?: OpenAIChatCompletionsToolDefinition[];
    extra_body?: Record<string, unknown>;
};

type OpenAIChatCompletionsUsage = NonNullable<OpenAI.Chat.ChatCompletion['usage']>;
type OpenAIChatCompletionsResponseMessage = Omit<
    Partial<OpenAI.Chat.ChatCompletionMessage>,
    'content' | 'tool_calls'
> & {
    role?: string;
    content?: string | null | OpenAIChatCompletionsContentPart[];
    reasoning_content?: string | null;
    reasoning?: string | null;
    tool_calls?: OpenAIChatCompletionsToolCall[];
};
type OpenAIChatCompletionsResponseChoice = Omit<OpenAI.Chat.ChatCompletion.Choice, 'message'> & {
    message: OpenAIChatCompletionsResponseMessage;
};

export type OpenAIChatCompletionsResponse = Omit<OpenAI.Chat.ChatCompletion, 'choices' | 'usage'> & {
    choices: OpenAIChatCompletionsResponseChoice[];
    usage?: OpenAIChatCompletionsUsage;
};

type OpenAIChatCompletionsStreamChoiceDelta = {
    role?: string;
    content?: string | null | OpenAIChatCompletionsContentPart[];
    reasoning_content?: string | null;
    reasoning?: string | null;
    tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: string;
        function?: {
            name?: string;
            arguments?: string;
        };
    }>;
};

type OpenAIChatCompletionsStreamChoice = Omit<OpenAI.Chat.ChatCompletionChunk.Choice, 'delta'> & {
    delta: OpenAIChatCompletionsStreamChoiceDelta;
};

export type OpenAIChatCompletionsStreamResponse = Omit<OpenAI.Chat.ChatCompletionChunk, 'choices' | 'usage'> & {
    choices: OpenAIChatCompletionsStreamChoice[];
    usage?: OpenAIChatCompletionsUsage;
};

export interface OpenAIChatCompletionsPrompt {
    messages: OpenAIChatCompletionsMessage[];
    /** Discriminator for drivers that share a `messages` array with other provider prompts. */
    _is_openai_chat_completions?: true;
}

export interface OpenAIChatCompletionsProtocolOptions {
    /** The model identifier to send in the request body (for example, "zai-org/glm-5-maas"). */
    modelName?: string;
    /** Model API contract default used only when callers do not provide max_tokens. */
    defaultMaxTokens?: number;
    /** Extra OpenAI-compatible request body fields for model-family-specific options. */
    extraBody?: Record<string, unknown>;
    /**
     * How result_schema should be requested. Vertex MaaS supports response_format, while
     * TogetherAI stays prompt-instruction based because its OpenAI-compatible surface is
     * Chat Completions only and response_format support is not reliable across hosted models.
     */
    resultSchemaMode?: 'response_format' | 'prompt';
}

type StreamChunk = string | Uint8Array;
type ReadableAsyncStream = AsyncIterable<StreamChunk>;
type StreamingOpenAIToolUse = ToolUse<unknown> & { _actual_id?: string };

async function streamToString(stream: ReadableAsyncStream): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks).toString('utf-8');
}

export function updateOpenAIChatCompletionsConversation(
    conversation: OpenAIChatCompletionsPrompt | undefined | null,
    prompt: OpenAIChatCompletionsPrompt,
): OpenAIChatCompletionsPrompt {
    const baseMessages = conversation ? conversation.messages : [];
    return {
        _is_openai_chat_completions: true,
        messages: [...baseMessages, ...(prompt.messages || [])],
    };
}

export function prepareOpenAIChatCompletionsConversation(
    conversation: OpenAIChatCompletionsPrompt,
    options: Pick<ExecutionOptions, 'tools'>,
): OpenAIChatCompletionsPrompt {
    let messages = fixOrphanedOpenAIChatCompletionsToolResults(
        fixOrphanedOpenAIChatCompletionsToolUse(conversation.messages),
    );
    if (!options.tools || options.tools.length === 0) {
        messages = convertOpenAIChatCompletionsToolMessagesToText(messages);
    }
    return { ...conversation, messages };
}

export function parseOpenAIChatCompletionsToolCalls(
    toolCalls: OpenAIChatCompletionsResponseChoice['message']['tool_calls'],
): ToolUse[] | undefined {
    if (!toolCalls || toolCalls.length === 0) {
        return undefined;
    }
    return toolCalls.map((tc) => ({
        id: tc.id ?? '',
        tool_name: tc.function?.name ?? '',
        tool_input: safeJsonParse(tc.function?.arguments),
    }));
}

function safeJsonParse(value: string | undefined): JSONObject {
    if (typeof value !== 'string') {
        return {};
    }
    try {
        const parsed = JSON.parse(value) as unknown;
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as JSONObject) : {};
    } catch {
        return {};
    }
}

function mapOpenAIChatCompletionsUsage(usage?: OpenAIChatCompletionsUsage | null): ExecutionTokenUsage | undefined {
    if (!usage) {
        return undefined;
    }
    return {
        prompt: usage.prompt_tokens,
        result: usage.completion_tokens,
        total: usage.total_tokens,
    };
}

function normalizeOpenAIChatCompletionsFinishReason(
    reason: string | null | undefined,
    hasToolUse: boolean = false,
): string | undefined {
    if (hasToolUse || reason === 'tool_calls' || reason === 'function_call') {
        return 'tool_use';
    }
    return reason || undefined;
}

function toolCallInterruptedMessage(id: string, toolName: string): OpenAIChatCompletionsMessage {
    return {
        role: 'tool',
        tool_call_id: id,
        content: `[Tool interrupted: The user stopped the operation before "${toolName}" could execute.]`,
    };
}

/**
 * Chat Completions requires every assistant tool call to be followed by a tool
 * response. If a run is stopped mid-tool-execution, synthesize a result so the
 * next request can continue instead of failing provider-side validation.
 */
export function fixOrphanedOpenAIChatCompletionsToolUse(
    messages: OpenAIChatCompletionsMessage[],
): OpenAIChatCompletionsMessage[] {
    if (messages.length < 2) return messages;

    const toolResultIds = new Set<string>();
    for (const message of messages) {
        if (message.role === 'tool' && message.tool_call_id) {
            toolResultIds.add(message.tool_call_id);
        }
    }

    const result: OpenAIChatCompletionsMessage[] = [];
    const pendingCalls = new Map<string, string>();

    for (const message of messages) {
        if (message.tool_calls && message.tool_calls.length > 0) {
            for (const toolCall of message.tool_calls) {
                if (toolCall.id && !toolResultIds.has(toolCall.id)) {
                    pendingCalls.set(toolCall.id, toolCall.function?.name ?? 'unknown');
                }
            }
            result.push(message);
            continue;
        }

        if (message.role === 'tool') {
            result.push(message);
            continue;
        }

        if (pendingCalls.size > 0) {
            for (const [callId, toolName] of pendingCalls) {
                result.push(toolCallInterruptedMessage(callId, toolName));
            }
            pendingCalls.clear();
        }
        result.push(message);
    }

    if (pendingCalls.size > 0) {
        for (const [callId, toolName] of pendingCalls) {
            result.push(toolCallInterruptedMessage(callId, toolName));
        }
    }

    return result;
}

/**
 * Drop tool result messages whose matching assistant tool call is no longer in
 * the conversation, usually after compaction/trimming removed the tool-call turn.
 */
export function fixOrphanedOpenAIChatCompletionsToolResults(
    messages: OpenAIChatCompletionsMessage[],
): OpenAIChatCompletionsMessage[] {
    if (messages.length === 0) return messages;

    const toolCallIds = new Set<string>();
    for (const message of messages) {
        for (const toolCall of message.tool_calls ?? []) {
            if (toolCall.id) {
                toolCallIds.add(toolCall.id);
            }
        }
    }

    return messages.filter((message) => {
        if (message.role !== 'tool') return true;
        return !!message.tool_call_id && toolCallIds.has(message.tool_call_id);
    });
}

export function convertOpenAIChatCompletionsToolMessagesToText(
    messages: OpenAIChatCompletionsMessage[],
): OpenAIChatCompletionsMessage[] {
    const hasToolMessages = messages.some((message) => message.role === 'tool' || !!message.tool_calls?.length);
    if (!hasToolMessages) return messages;

    return messages.map((message) => {
        if (message.tool_calls && message.tool_calls.length > 0) {
            const textParts: string[] = [];
            const contentText = extractOpenAIChatCompletionsContentText(message.content);
            if (contentText.trim()) {
                textParts.push(contentText);
            }
            for (const toolCall of message.tool_calls) {
                const args = toolCall.function?.arguments ?? '';
                textParts.push(`[Tool call: ${toolCall.function?.name ?? 'unknown'}(${truncateToolText(args)})]`);
            }
            return { role: message.role, content: textParts.join('\n') || null };
        }

        if (message.role === 'tool') {
            const output = extractOpenAIChatCompletionsContentText(message.content) || 'No output';
            return { role: 'user', content: `[Tool result: ${truncateToolText(output)}]` };
        }

        return message;
    });
}

function truncateToolText(value: string): string {
    return value.length > 500 ? `${value.substring(0, 500)}...` : value;
}

export function stripOpenAIChatCompletionsThinkBlocks(value: string): string {
    return value.replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '').trim();
}

export function stripOpenAIChatCompletionsThinkBlocksFromCompletion(completion: Completion): Completion {
    completion.result = completion.result.map((result): CompletionResult => {
        if (result.type === 'text') {
            return { ...result, value: stripOpenAIChatCompletionsThinkBlocks(result.value) };
        }
        if (result.type === 'json' && typeof result.value === 'string') {
            return { ...result, value: stripOpenAIChatCompletionsThinkBlocks(result.value) };
        }
        return result;
    });

    const conversation = completion.conversation as OpenAIChatCompletionsPrompt | undefined;
    if (conversation?._is_openai_chat_completions && Array.isArray(conversation.messages)) {
        completion.conversation = {
            ...conversation,
            messages: conversation.messages.map((message) => {
                if (typeof message.content === 'string') {
                    return { ...message, content: stripOpenAIChatCompletionsThinkBlocks(message.content) };
                }
                if (Array.isArray(message.content)) {
                    return {
                        ...message,
                        content: message.content.map((part) =>
                            part.type === 'text'
                                ? { ...part, text: stripOpenAIChatCompletionsThinkBlocks(part.text) }
                                : part,
                        ),
                    };
                }
                return message;
            }),
        };
    }

    return completion;
}

export function extractOpenAIChatCompletionsText(
    source:
        | Pick<OpenAIChatCompletionsResponseChoice['message'], 'content' | 'reasoning_content' | 'reasoning'>
        | Pick<OpenAIChatCompletionsStreamChoiceDelta, 'content' | 'reasoning_content' | 'reasoning'>
        | undefined,
    includeThoughts: boolean,
): string {
    if (!source) return '';

    const content = stripOpenAIChatCompletionsThinkBlocks(extractOpenAIChatCompletionsContentText(source.content));
    const reasoning = extractOpenAIChatCompletionsReasoningText(source);

    if (includeThoughts && reasoning) {
        return content ? `${reasoning}\n\n${content}` : reasoning;
    }

    // Some OpenAI-compatible model servers can emit thinking-only chunks or messages
    // using reasoning_content/reasoning. Use those fields only as a fallback so normal
    // assistant content remains the visible output whenever it exists after provider
    // <think>...</think> blocks and whitespace-only output are removed.
    return content || stripOpenAIChatCompletionsThinkBlocks(reasoning);
}

export function extractOpenAIChatCompletionsReasoningText(
    source:
        | Pick<OpenAIChatCompletionsResponseChoice['message'], 'reasoning_content' | 'reasoning'>
        | Pick<OpenAIChatCompletionsStreamChoiceDelta, 'reasoning_content' | 'reasoning'>
        | undefined,
): string {
    if (!source) return '';
    return [source.reasoning_content, source.reasoning]
        .filter((value): value is string => typeof value === 'string' && value.length > 0)
        .join('');
}

export function extractOpenAIChatCompletionsContentText(
    content: string | null | OpenAIChatCompletionsContentPart[] | undefined,
): string {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';

    return content
        .filter((part): part is OpenAIChatCompletionsTextPart => part.type === 'text' && typeof part.text === 'string')
        .map((part) => part.text)
        .join('\n');
}

export function convertToolsToOpenAIChatCompletionsFormat(
    tools: ToolDefinition[] | undefined,
): OpenAIChatCompletionsToolDefinition[] | undefined {
    if (!tools || tools.length === 0) {
        return undefined;
    }

    return tools.map((tool) => {
        let parameters: JSONSchema | undefined;
        let strict: boolean | undefined;
        if (tool.input_schema) {
            // Reuse the same schema normalization as the OpenAI Responses path:
            // strict mode when possible, limited non-strict schema otherwise.
            const formattedSchema = formatOpenAISchema(tool.input_schema as JSONSchema);
            parameters = formattedSchema.schema;
            strict = formattedSchema.strict;
        }

        return {
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description,
                parameters: parameters ?? {},
                ...(strict !== undefined ? { strict } : {}),
            },
        };
    });
}

export function convertToOpenAIChatCompletionsMessages(
    messages: OpenAIChatCompletionsMessage[],
): OpenAIChatCompletionsRequestMessage[] {
    return messages.map((msg) => {
        const result: OpenAIChatCompletionsRequestMessage = {
            role: msg.role,
        };

        if (msg.tool_calls && msg.tool_calls.length > 0) {
            result.tool_calls = msg.tool_calls;
        }

        if (msg.content === null) {
            result.content = null;
        } else if (typeof msg.content === 'string') {
            // Empty string is rejected by several OpenAI-compatible servers; normalize to null.
            result.content = msg.content || null;
        } else if (msg.content === undefined) {
            result.content = null;
        }

        if (Array.isArray(msg.content)) {
            const textParts: string[] = [];
            const imageUrls: OpenAIChatCompletionsImageUrlPart['image_url'][] = [];

            for (const part of msg.content) {
                if (part.type === 'text') {
                    textParts.push(part.text);
                } else if (part.type === 'image_url') {
                    imageUrls.push(part.image_url);
                }
            }

            const content: OpenAIChatCompletionsContentPart[] = [];
            if (textParts.length > 0) {
                content.push({ type: 'text', text: textParts.join('\n') });
            }
            for (const img of imageUrls) {
                content.push({ type: 'image_url', image_url: img });
            }

            if (content.length === 1 && content[0].type === 'text') {
                result.content = content[0].text;
            } else if (content.length > 0) {
                result.content = content;
            }
        }

        if (msg.tool_call_id) {
            result.tool_call_id = msg.tool_call_id;
        }

        return result;
    });
}

export function buildOpenAIChatCompletionsStreamingConversation(
    prompt: OpenAIChatCompletionsPrompt,
    result: unknown[],
    toolUse: unknown[] | undefined,
    options: ExecutionOptions,
): OpenAIChatCompletionsPrompt {
    const completionResults = result as CompletionResult[];
    const textContent = completionResults
        .map((r) => {
            switch (r.type) {
                case 'text':
                    return r.value;
                case 'json':
                    return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                default:
                    return '';
            }
        })
        .join('');

    const assistantMessage: OpenAIChatCompletionsMessage = { role: 'assistant' };
    if (textContent) {
        assistantMessage.content = stripOpenAIChatCompletionsThinkBlocks(textContent);
    } else if (toolUse && toolUse.length > 0) {
        assistantMessage.content = null;
    } else {
        assistantMessage.content = '';
    }

    if (toolUse && toolUse.length > 0) {
        assistantMessage.tool_calls = (toolUse as ToolUse[]).map((t) => ({
            id: t.id,
            type: 'function',
            function: {
                name: t.tool_name,
                arguments: typeof t.tool_input === 'string' ? t.tool_input : JSON.stringify(t.tool_input ?? {}),
            },
        }));
    }

    const conversation: OpenAIChatCompletionsPrompt = {
        _is_openai_chat_completions: true,
        messages: [
            ...prepareOpenAIChatCompletionsConversation(
                updateOpenAIChatCompletionsConversation(
                    options.conversation as OpenAIChatCompletionsPrompt | undefined,
                    prompt,
                ),
                options,
            ).messages,
            assistantMessage,
        ],
    };

    return finalizeOpenAIChatCompletionsConversation(conversation, options);
}

function finalizeOpenAIChatCompletionsConversation(
    conversation: OpenAIChatCompletionsPrompt,
    options: ExecutionOptions,
): OpenAIChatCompletionsPrompt {
    conversation = incrementConversationTurn(conversation) as OpenAIChatCompletionsPrompt;
    const currentTurn = getConversationMeta(conversation).turnNumber;
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: options.stripTextMaxTokens,
    };
    let processedConversation = stripBase64ImagesFromConversation(
        conversation,
        stripOptions,
    ) as OpenAIChatCompletionsPrompt;
    processedConversation = truncateLargeTextInConversation(
        processedConversation,
        stripOptions,
    ) as OpenAIChatCompletionsPrompt;
    processedConversation = stripHeartbeatsFromConversation(processedConversation, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
    }) as OpenAIChatCompletionsPrompt;

    return processedConversation;
}

export abstract class OpenAIChatCompletionsProtocol<DriverT> {
    protected readonly options: OpenAIChatCompletionsProtocolOptions;

    constructor(options: OpenAIChatCompletionsProtocolOptions) {
        this.options = options;
    }

    async createPrompt(
        _driver: DriverT,
        segments: PromptSegment[],
        _options: PromptOptions,
    ): Promise<OpenAIChatCompletionsPrompt> {
        const messages: OpenAIChatCompletionsMessage[] = [];

        let systemContent = '';
        for (const segment of segments) {
            if (segment.role === PromptRole.system && segment.content) {
                systemContent += `${segment.content}\n`;
            }
        }

        if (systemContent.trim()) {
            messages.push({ role: 'system', content: systemContent.trim() });
        }

        if (this.options.resultSchemaMode === 'prompt' && _options.result_schema) {
            messages.push({
                role: 'system',
                content: `IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. <response_schema>${JSON.stringify(_options.result_schema)}</response_schema>`,
            });
        }

        for (const segment of segments) {
            if (segment.role === PromptRole.system) {
                continue;
            }

            if (segment.role === PromptRole.tool) {
                if (!segment.tool_use_id) {
                    throw new Error('Tool prompt segment must have a tool_use_id to reference the original tool call');
                }

                const content: OpenAIChatCompletionsContentPart[] = [];

                if (segment.content) {
                    content.push({ type: 'text', text: segment.content });
                } else {
                    content.push({ type: 'text', text: '' });
                }

                if (segment.files && segment.files.length > 0) {
                    for (const file of segment.files) {
                        if (file.mime_type?.startsWith('image/')) {
                            const stream = await file.getStream();
                            const data = await readStreamAsBase64(stream);
                            content.push({
                                type: 'image_url',
                                image_url: {
                                    url: `data:${file.mime_type};base64,${data}`,
                                    detail: 'auto',
                                },
                            });
                        } else if (file.mime_type?.startsWith('text/')) {
                            const fileStream = await file.getStream();
                            const fileContent = await streamToString(fileStream);
                            content.push({ type: 'text', text: `\n\nFile content:\n${fileContent}` });
                        }
                    }
                }

                const toolMessage: OpenAIChatCompletionsMessage = {
                    role: 'tool',
                    tool_call_id: segment.tool_use_id,
                    content: content.length === 1 && content[0]?.type === 'text' ? content[0].text : content,
                };
                messages.push(toolMessage);
            } else {
                let content: string | OpenAIChatCompletionsContentPart[] = segment.content || '';

                if (segment.files && segment.files.length > 0) {
                    const parts: OpenAIChatCompletionsContentPart[] = [];

                    if (content && typeof content === 'string' && content.trim()) {
                        parts.push({ type: 'text', text: content });
                    }

                    for (const file of segment.files) {
                        if (file.mime_type?.startsWith('image/')) {
                            const stream = await file.getStream();
                            const data = await readStreamAsBase64(stream);
                            parts.push({
                                type: 'image_url',
                                image_url: {
                                    url: `data:${file.mime_type};base64,${data}`,
                                    detail: 'auto',
                                },
                            });
                        } else if (file.mime_type?.startsWith('text/')) {
                            const fileStream = await file.getStream();
                            const fileContent = await streamToString(fileStream);
                            parts.push({ type: 'text', text: `\n\nFile content:\n${fileContent}` });
                        }
                    }

                    if (parts.length > 0) {
                        content = parts;
                    }
                }

                const role = segment.role === PromptRole.assistant ? 'assistant' : 'user';
                messages.push({
                    role,
                    content,
                });
            }
        }

        return {
            _is_openai_chat_completions: true,
            messages,
        };
    }

    async requestTextCompletion(
        driver: DriverT,
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
    ): Promise<Completion> {
        let conversation = updateOpenAIChatCompletionsConversation(
            options.conversation as OpenAIChatCompletionsPrompt,
            prompt,
        );
        conversation = prepareOpenAIChatCompletionsConversation(conversation, options);
        const includeThoughts = (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })
            ?.include_thoughts;
        const payload = this.buildPayload(conversation, options, false);
        const result = await this.postChatCompletion(driver, payload);

        const choice = result?.choices?.[0];
        const message = choice?.message;
        const text = stripOpenAIChatCompletionsThinkBlocks(
            extractOpenAIChatCompletionsText(message, includeThoughts ?? false),
        );
        const tool_use = parseOpenAIChatCompletionsToolCalls(message?.tool_calls);
        if (!text && (!tool_use || tool_use.length === 0)) {
            throw new Error('Chat Completions response is not valid: no data');
        }

        const assistantMessage: OpenAIChatCompletionsMessage = {
            role: 'assistant',
            content: text || null,
        };

        if (tool_use && tool_use.length > 0 && message?.tool_calls) {
            assistantMessage.tool_calls = message.tool_calls.map((tc) => ({
                id: tc.id,
                type: 'function',
                function: {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            }));
        }

        conversation = updateOpenAIChatCompletionsConversation(conversation, {
            messages: [assistantMessage],
        });
        conversation = finalizeOpenAIChatCompletionsConversation(conversation, options);

        return {
            result: text ? [{ type: 'text', value: text }] : [],
            tool_use,
            token_usage: mapOpenAIChatCompletionsUsage(result.usage),
            finish_reason: normalizeOpenAIChatCompletionsFinishReason(choice?.finish_reason, !!tool_use?.length),
            original_response: options.include_original_response ? result : undefined,
            conversation,
        };
    }

    async requestTextCompletionStream(
        driver: DriverT,
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        let conversation = updateOpenAIChatCompletionsConversation(
            options.conversation as OpenAIChatCompletionsPrompt,
            prompt,
        );
        conversation = prepareOpenAIChatCompletionsConversation(conversation, options);
        const includeThoughts = (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })
            ?.include_thoughts;
        const payload = this.buildPayload(conversation, options, true);
        const responseStream = await this.postChatCompletionStream(driver, payload);

        let emittedContent = false;
        let reasoningFallback = '';

        return transformSSEStream(responseStream, (data: string) => {
            const json = JSON.parse(data) as OpenAIChatCompletionsStreamResponse;
            const choice = json.choices?.[0];
            const delta = choice?.delta;

            if (delta?.tool_calls && delta.tool_calls.length > 0) {
                const toolUseChunks: StreamingOpenAIToolUse[] = delta.tool_calls.map((tc) => {
                    const toolUse: StreamingOpenAIToolUse = {
                        id: `tool_${tc.index ?? 0}`,
                        tool_name: tc.function?.name ?? '',
                        tool_input:
                            typeof tc.function?.arguments === 'string' && tc.function.arguments.length > 0
                                ? tc.function.arguments
                                : {},
                    };
                    if (tc.id) {
                        toolUse._actual_id = tc.id;
                    }
                    return toolUse;
                });

                return {
                    result: [],
                    tool_use: toolUseChunks,
                    finish_reason: choice?.finish_reason
                        ? normalizeOpenAIChatCompletionsFinishReason(choice.finish_reason, toolUseChunks.length > 0)
                        : undefined,
                    token_usage: mapOpenAIChatCompletionsUsage(json.usage),
                } satisfies CompletionChunkObject;
            }

            const content = extractOpenAIChatCompletionsContentText(delta?.content);
            const reasoning = extractOpenAIChatCompletionsReasoningText(delta);
            const visibleContent = stripOpenAIChatCompletionsThinkBlocks(content);
            if (visibleContent) {
                emittedContent = true;
            } else if (reasoning) {
                reasoningFallback += reasoning;
            }

            const text = includeThoughts ? reasoning + content : content;
            // Intentionally stream provider <think> blocks as they arrive. The final
            // accumulated Completion/conversation strips them in validateResult();
            // for fallback purposes they do not count as normal content.
            const fallbackText =
                !includeThoughts && choice?.finish_reason && !emittedContent
                    ? stripOpenAIChatCompletionsThinkBlocks(reasoningFallback)
                    : '';
            return {
                result: text || fallbackText ? [{ type: 'text', value: fallbackText || text }] : [],
                finish_reason: normalizeOpenAIChatCompletionsFinishReason(choice?.finish_reason),
                token_usage: mapOpenAIChatCompletionsUsage(json.usage),
            } satisfies CompletionChunkObject;
        });
    }

    protected buildPayload(
        conversation: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        stream: boolean,
    ): OpenAIChatCompletionsPayload {
        const modelOptions = options.model_options as TextFallbackOptions;
        const payload: OpenAIChatCompletionsPayload = {
            model: this.getModelName(options),
            messages: convertToOpenAIChatCompletionsMessages(conversation.messages),
            // Some OpenAI-compatible providers return empty/truncated completions unless a
            // documented or runtime-validated token budget is supplied. Caller options still win.
            max_tokens: modelOptions?.max_tokens ?? this.options.defaultMaxTokens,
            temperature: modelOptions?.temperature,
            top_p: modelOptions?.top_p,
            n: 1,
            stop: modelOptions?.stop_sequence,
            stream,
        };

        if (this.options.extraBody) {
            payload.extra_body = this.options.extraBody;
        }

        const toolsPayload = convertToolsToOpenAIChatCompletionsFormat(options.tools);
        if (toolsPayload && toolsPayload.length > 0) {
            payload.tools = toolsPayload;
        }

        if (options.result_schema && this.options.resultSchemaMode !== 'prompt') {
            const formattedSchema = formatOpenAISchema(options.result_schema as JSONSchema);
            payload.response_format = {
                type: 'json_schema',
                json_schema: { name: 'output', strict: formattedSchema.strict, schema: formattedSchema.schema },
            };
        }

        return payload;
    }

    protected getModelName(options: ExecutionOptions): string {
        return this.options.modelName ?? options.model;
    }

    protected abstract postChatCompletion(
        driver: DriverT,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<OpenAIChatCompletionsResponse>;

    protected abstract postChatCompletionStream(
        driver: DriverT,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<ReadableStream>;
}

export interface OpenAIChatCompletionsDriverOptions extends DriverOptions {
    defaultMaxTokens?: number;
    extraBody?: Record<string, unknown>;
    resultSchemaMode?: OpenAIChatCompletionsProtocolOptions['resultSchemaMode'];
}

type OpenAIChatCompletionsDriver = AbstractDriver<OpenAIChatCompletionsDriverOptions, OpenAIChatCompletionsPrompt> & {
    service: OpenAI;
};

function sdkStreamToSSEStream(stream: AsyncIterable<OpenAIChatCompletionsStreamResponse>): ReadableStream {
    return new ReadableStream({
        async start(controller) {
            try {
                for await (const chunk of stream) {
                    controller.enqueue({ type: 'event', data: JSON.stringify(chunk) });
                }
                controller.close();
            } catch (error) {
                controller.error(error);
            }
        },
    });
}

class OpenAISDKChatCompletionsProtocol extends OpenAIChatCompletionsProtocol<OpenAIChatCompletionsDriver> {
    protected async postChatCompletion(
        driver: OpenAIChatCompletionsDriver,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<OpenAIChatCompletionsResponse> {
        const { extra_body: _extraBody, ...body } = payload;
        return (await driver.service.chat.completions.create(
            body as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
        )) as OpenAIChatCompletionsResponse;
    }

    protected async postChatCompletionStream(
        driver: OpenAIChatCompletionsDriver,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<ReadableStream> {
        const { extra_body: _extraBody, ...body } = payload;
        const stream = (await driver.service.chat.completions.create({
            ...body,
            stream: true,
            stream_options: { include_usage: true },
        } as OpenAI.Chat.ChatCompletionCreateParamsStreaming)) as unknown as AsyncIterable<OpenAIChatCompletionsStreamResponse>;
        return sdkStreamToSSEStream(stream);
    }
}

export abstract class OpenAIChatCompletionsDriverBase<
    OptionsT extends OpenAIChatCompletionsDriverOptions = OpenAIChatCompletionsDriverOptions,
> extends AbstractDriver<OptionsT, OpenAIChatCompletionsPrompt> {
    abstract service: OpenAI;

    private readonly chatCompletionsProtocol: OpenAISDKChatCompletionsProtocol;

    constructor(options: OptionsT) {
        super(options);
        this.chatCompletionsProtocol = new OpenAISDKChatCompletionsProtocol({
            defaultMaxTokens: options.defaultMaxTokens,
            extraBody: options.extraBody,
            resultSchemaMode: options.resultSchemaMode,
        });
    }

    protected async formatPrompt(
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<OpenAIChatCompletionsPrompt> {
        return this.chatCompletionsProtocol.createPrompt(this, segments, options);
    }

    requestTextCompletion(prompt: OpenAIChatCompletionsPrompt, options: ExecutionOptions): Promise<Completion> {
        return this.chatCompletionsProtocol.requestTextCompletion(this, prompt, options);
    }

    requestTextCompletionStream(
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        return this.chatCompletionsProtocol.requestTextCompletionStream(this, prompt, options);
    }

    buildStreamingConversation(
        prompt: OpenAIChatCompletionsPrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): OpenAIChatCompletionsPrompt {
        return buildOpenAIChatCompletionsStreamingConversation(prompt, result, toolUse, options);
    }

    validateResult(result: Completion, options: ExecutionOptions) {
        super.validateResult(stripOpenAIChatCompletionsThinkBlocksFromCompletion(result), options);
    }
}
