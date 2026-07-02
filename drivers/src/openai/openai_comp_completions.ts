import {
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    type DriverOptions,
    type ExecutionOptions,
    getConversationMeta,
    incrementConversationTurn,
    type JSONObject,
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

export type OpenAICompletionsTextPart = OpenAI.Chat.ChatCompletionContentPartText;
export type OpenAICompletionsImageUrlPart = OpenAI.Chat.ChatCompletionContentPartImage;
export type OpenAICompletionsContentPart = OpenAICompletionsTextPart | OpenAICompletionsImageUrlPart;
export type OpenAICompletionsToolCall = OpenAI.Chat.ChatCompletionMessageFunctionToolCall;
export type OpenAICompletionsToolDefinition = OpenAI.Chat.ChatCompletionTool;

export type OpenAICompletionsMessage = {
    role: string;
    content?: string | null | OpenAICompletionsContentPart[];
    /**
     * Required for tool role messages - references the tool_call.id from the assistant's message.
     * Per OpenAI API spec: https://platform.openai.com/docs/api-reference/chat/messages#message-role
     */
    tool_call_id?: string;
    /**
     * Tool calls from assistant messages - stored and sent back with tool results.
     */
    tool_calls?: OpenAICompletionsToolCall[];
};

export type OpenAICompletionsRequestMessage = {
    role: string;
    content?: string | null | OpenAICompletionsContentPart[];
    tool_call_id?: string;
    tool_calls?: OpenAICompletionsToolCall[];
};

export type OpenAICompletionsPayload = Omit<
    OpenAI.Chat.ChatCompletionCreateParams,
    'model' | 'messages' | 'stream' | 'tools'
> & {
    model: string;
    messages: OpenAICompletionsRequestMessage[];
    stream: boolean;
    tools?: OpenAICompletionsToolDefinition[];
    extra_body?: Record<string, unknown>;
};

type OpenAICompletionsUsage = NonNullable<OpenAI.Chat.ChatCompletion['usage']>;
type OpenAICompletionsResponseMessage = Omit<Partial<OpenAI.Chat.ChatCompletionMessage>, 'content' | 'tool_calls'> & {
    role?: string;
    content?: string | null | OpenAICompletionsContentPart[];
    reasoning_content?: string | null;
    reasoning?: string | null;
    tool_calls?: OpenAICompletionsToolCall[];
};
type OpenAICompletionsResponseChoice = Omit<OpenAI.Chat.ChatCompletion.Choice, 'message'> & {
    message: OpenAICompletionsResponseMessage;
};

export type OpenAICompletionsResponse = Omit<OpenAI.Chat.ChatCompletion, 'choices' | 'usage'> & {
    choices: OpenAICompletionsResponseChoice[];
    usage?: OpenAICompletionsUsage;
};

type OpenAICompletionsStreamChoiceDelta = {
    role?: string;
    content?: string | null | OpenAICompletionsContentPart[];
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

type OpenAICompletionsStreamChoice = Omit<OpenAI.Chat.ChatCompletionChunk.Choice, 'delta'> & {
    delta: OpenAICompletionsStreamChoiceDelta;
};

export type OpenAICompletionsStreamResponse = Omit<OpenAI.Chat.ChatCompletionChunk, 'choices' | 'usage'> & {
    choices: OpenAICompletionsStreamChoice[];
    usage?: OpenAICompletionsUsage;
};

export interface OpenAICompletionsPrompt {
    messages: OpenAICompletionsMessage[];
    /** Discriminator for drivers that share a `messages` array with other provider prompts. */
    _is_openai_compat?: true;
}

export interface OpenAICompletionsModelOptions {
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

export function updateOpenAICompletionsConversation(
    conversation: OpenAICompletionsPrompt | undefined | null,
    prompt: OpenAICompletionsPrompt,
): OpenAICompletionsPrompt {
    const baseMessages = conversation ? conversation.messages : [];
    return {
        _is_openai_compat: true,
        messages: [...baseMessages, ...(prompt.messages || [])],
    };
}

export function parseOpenAICompletionsToolCalls(
    toolCalls: OpenAICompletionsResponseChoice['message']['tool_calls'],
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

export function stripOpenAICompletionsThinkBlocks(value: string): string {
    return value.replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '').trim();
}

export function stripOpenAICompletionsThinkBlocksFromCompletion(completion: Completion): Completion {
    completion.result = completion.result.map((result): CompletionResult => {
        if (result.type === 'text') {
            return { ...result, value: stripOpenAICompletionsThinkBlocks(result.value) };
        }
        if (result.type === 'json' && typeof result.value === 'string') {
            return { ...result, value: stripOpenAICompletionsThinkBlocks(result.value) };
        }
        return result;
    });

    const conversation = completion.conversation as OpenAICompletionsPrompt | undefined;
    if (conversation?._is_openai_compat && Array.isArray(conversation.messages)) {
        completion.conversation = {
            ...conversation,
            messages: conversation.messages.map((message) => {
                if (typeof message.content === 'string') {
                    return { ...message, content: stripOpenAICompletionsThinkBlocks(message.content) };
                }
                if (Array.isArray(message.content)) {
                    return {
                        ...message,
                        content: message.content.map((part) =>
                            part.type === 'text'
                                ? { ...part, text: stripOpenAICompletionsThinkBlocks(part.text) }
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

export function extractOpenAICompletionsText(
    source:
        | Pick<OpenAICompletionsResponseChoice['message'], 'content' | 'reasoning_content' | 'reasoning'>
        | Pick<OpenAICompletionsStreamChoiceDelta, 'content' | 'reasoning_content' | 'reasoning'>
        | undefined,
    includeThoughts: boolean,
): string {
    if (!source) return '';

    const content = extractOpenAICompletionsContentText(source.content);
    const reasoning = extractOpenAICompletionsReasoningText(source);

    if (includeThoughts && reasoning) {
        return content ? `${reasoning}\n\n${content}` : reasoning;
    }

    // Some OpenAI-compatible model servers can emit thinking-only chunks or messages
    // using reasoning_content/reasoning. Use those fields only as a fallback so normal
    // assistant content remains the visible output whenever it exists.
    return stripOpenAICompletionsThinkBlocks(content || reasoning);
}

export function extractOpenAICompletionsReasoningText(
    source:
        | Pick<OpenAICompletionsResponseChoice['message'], 'reasoning_content' | 'reasoning'>
        | Pick<OpenAICompletionsStreamChoiceDelta, 'reasoning_content' | 'reasoning'>
        | undefined,
): string {
    if (!source) return '';
    return [source.reasoning_content, source.reasoning]
        .filter((value): value is string => typeof value === 'string' && value.length > 0)
        .join('');
}

export function extractOpenAICompletionsContentText(
    content: string | null | OpenAICompletionsContentPart[] | undefined,
): string {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';

    return content
        .filter((part): part is OpenAICompletionsTextPart => part.type === 'text' && typeof part.text === 'string')
        .map((part) => part.text)
        .join('\n');
}

export function convertToolsToOpenAICompletionsFormat(
    tools: ToolDefinition[] | undefined,
): OpenAICompletionsToolDefinition[] | undefined {
    if (!tools || tools.length === 0) {
        return undefined;
    }

    return tools.map((tool) => ({
        type: 'function',
        function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.input_schema ?? {},
        },
    }));
}

export function convertToOpenAICompletionsMessages(
    messages: OpenAICompletionsMessage[],
): OpenAICompletionsRequestMessage[] {
    return messages.map((msg) => {
        const result: OpenAICompletionsRequestMessage = {
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
            const imageUrls: OpenAICompletionsImageUrlPart['image_url'][] = [];

            for (const part of msg.content) {
                if (part.type === 'text') {
                    textParts.push(part.text);
                } else if (part.type === 'image_url') {
                    imageUrls.push(part.image_url);
                }
            }

            const content: OpenAICompletionsContentPart[] = [];
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

export function buildOpenAICompletionsStreamingConversation(
    prompt: OpenAICompletionsPrompt,
    result: unknown[],
    toolUse: unknown[] | undefined,
    options: ExecutionOptions,
): OpenAICompletionsPrompt {
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

    const assistantMessage: OpenAICompletionsMessage = { role: 'assistant' };
    if (textContent) {
        assistantMessage.content = stripOpenAICompletionsThinkBlocks(textContent);
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

    const existingMessages = (options.conversation as OpenAICompletionsPrompt | undefined)?.messages;
    const priorMessages = Array.isArray(existingMessages) ? existingMessages : [];
    let conversation: OpenAICompletionsPrompt = {
        _is_openai_compat: true,
        messages: [...priorMessages, ...prompt.messages, assistantMessage],
    };

    conversation = incrementConversationTurn(conversation) as OpenAICompletionsPrompt;
    const currentTurn = getConversationMeta(conversation).turnNumber;
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: options.stripTextMaxTokens,
    };
    let processedConversation = stripBase64ImagesFromConversation(
        conversation,
        stripOptions,
    ) as OpenAICompletionsPrompt;
    processedConversation = truncateLargeTextInConversation(
        processedConversation,
        stripOptions,
    ) as OpenAICompletionsPrompt;
    processedConversation = stripHeartbeatsFromConversation(processedConversation, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
    }) as OpenAICompletionsPrompt;

    return processedConversation;
}

export abstract class OpenAICompletionsModelDefinitionBase<DriverT> {
    protected readonly options: OpenAICompletionsModelOptions;

    constructor(options: OpenAICompletionsModelOptions) {
        this.options = options;
    }

    async createPrompt(
        _driver: DriverT,
        segments: PromptSegment[],
        _options: PromptOptions,
    ): Promise<OpenAICompletionsPrompt> {
        const messages: OpenAICompletionsMessage[] = [];

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

                const content: OpenAICompletionsContentPart[] = [];

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

                const toolMessage: OpenAICompletionsMessage = {
                    role: 'tool',
                    tool_call_id: segment.tool_use_id,
                    content: content.length === 1 && content[0]?.type === 'text' ? content[0].text : content,
                };
                messages.push(toolMessage);
            } else {
                let content: string | OpenAICompletionsContentPart[] = segment.content || '';

                if (segment.files && segment.files.length > 0) {
                    const parts: OpenAICompletionsContentPart[] = [];

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
            _is_openai_compat: true,
            messages,
        };
    }

    async requestTextCompletion(
        driver: DriverT,
        prompt: OpenAICompletionsPrompt,
        options: ExecutionOptions,
    ): Promise<Completion> {
        let conversation = updateOpenAICompletionsConversation(options.conversation as OpenAICompletionsPrompt, prompt);
        const includeThoughts = (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })
            ?.include_thoughts;
        const payload = this.buildPayload(conversation, options, false);
        const result = await this.postChatCompletion(driver, payload);

        const choice = result?.choices?.[0];
        const message = choice?.message;
        const text = stripOpenAICompletionsThinkBlocks(extractOpenAICompletionsText(message, includeThoughts ?? false));
        const tool_use = parseOpenAICompletionsToolCalls(message?.tool_calls);

        const assistantMessage: OpenAICompletionsMessage = {
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

        conversation = updateOpenAICompletionsConversation(conversation, {
            messages: [assistantMessage],
        });

        return {
            result: text ? [{ type: 'text', value: text }] : [{ type: 'text', value: '' }],
            tool_use,
            token_usage: result.usage
                ? {
                      prompt: result.usage.prompt_tokens,
                      result: result.usage.completion_tokens,
                      total: result.usage.total_tokens,
                  }
                : undefined,
            finish_reason: choice?.finish_reason || undefined,
            original_response: options.include_original_response ? result : undefined,
            conversation,
        };
    }

    async requestTextCompletionStream(
        driver: DriverT,
        prompt: OpenAICompletionsPrompt,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        const conversation = updateOpenAICompletionsConversation(
            options.conversation as OpenAICompletionsPrompt,
            prompt,
        );
        const includeThoughts = (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })
            ?.include_thoughts;
        const payload = this.buildPayload(conversation, options, true);
        const responseStream = await this.postChatCompletionStream(driver, payload);

        let emittedContent = false;
        let reasoningFallback = '';

        return transformSSEStream(responseStream, (data: string) => {
            const json = JSON.parse(data) as OpenAICompletionsStreamResponse;
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
                } satisfies CompletionChunkObject;
            }

            const content = extractOpenAICompletionsContentText(delta?.content);
            const reasoning = extractOpenAICompletionsReasoningText(delta);
            if (content) {
                emittedContent = true;
            } else if (reasoning) {
                reasoningFallback += reasoning;
            }

            const text = includeThoughts ? reasoning + content : content;
            const fallbackText = !includeThoughts && choice?.finish_reason && !emittedContent ? reasoningFallback : '';
            return {
                result: text || fallbackText ? [{ type: 'text', value: text || fallbackText }] : [],
                finish_reason: choice?.finish_reason || undefined,
                token_usage: json.usage
                    ? {
                          prompt: json.usage.prompt_tokens,
                          result: json.usage.completion_tokens,
                          total: json.usage.total_tokens,
                      }
                    : undefined,
            } satisfies CompletionChunkObject;
        });
    }

    protected buildPayload(
        conversation: OpenAICompletionsPrompt,
        options: ExecutionOptions,
        stream: boolean,
    ): OpenAICompletionsPayload {
        const modelOptions = options.model_options as TextFallbackOptions;
        const payload: OpenAICompletionsPayload = {
            model: this.getModelName(options),
            messages: convertToOpenAICompletionsMessages(conversation.messages),
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

        const toolsPayload = convertToolsToOpenAICompletionsFormat(options.tools);
        if (toolsPayload && toolsPayload.length > 0) {
            payload.tools = toolsPayload;
        }

        if (options.result_schema && this.options.resultSchemaMode !== 'prompt') {
            payload.response_format = {
                type: 'json_schema',
                json_schema: { name: 'output', strict: false, schema: options.result_schema },
            };
        }

        return payload;
    }

    protected getModelName(options: ExecutionOptions): string {
        return this.options.modelName ?? options.model;
    }

    protected abstract postChatCompletion(
        driver: DriverT,
        payload: OpenAICompletionsPayload,
    ): Promise<OpenAICompletionsResponse>;

    protected abstract postChatCompletionStream(
        driver: DriverT,
        payload: OpenAICompletionsPayload,
    ): Promise<ReadableStream>;
}

export interface OpenAIChatCompletionsDriverOptions extends DriverOptions {
    defaultMaxTokens?: number;
    extraBody?: Record<string, unknown>;
    resultSchemaMode?: OpenAICompletionsModelOptions['resultSchemaMode'];
}

type OpenAIChatCompletionsDriver = AbstractDriver<OpenAIChatCompletionsDriverOptions, OpenAICompletionsPrompt> & {
    service: OpenAI;
};

function sdkStreamToSSEStream(stream: AsyncIterable<OpenAICompletionsStreamResponse>): ReadableStream {
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

class OpenAISDKChatCompletionsModelDefinition extends OpenAICompletionsModelDefinitionBase<OpenAIChatCompletionsDriver> {
    protected async postChatCompletion(
        driver: OpenAIChatCompletionsDriver,
        payload: OpenAICompletionsPayload,
    ): Promise<OpenAICompletionsResponse> {
        const { extra_body: _extraBody, ...body } = payload;
        return (await driver.service.chat.completions.create(
            body as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
        )) as OpenAICompletionsResponse;
    }

    protected async postChatCompletionStream(
        driver: OpenAIChatCompletionsDriver,
        payload: OpenAICompletionsPayload,
    ): Promise<ReadableStream> {
        const { extra_body: _extraBody, ...body } = payload;
        const stream = (await driver.service.chat.completions.create({
            ...body,
            stream: true,
            stream_options: { include_usage: true },
        } as OpenAI.Chat.ChatCompletionCreateParamsStreaming)) as unknown as AsyncIterable<OpenAICompletionsStreamResponse>;
        return sdkStreamToSSEStream(stream);
    }
}

export abstract class OpenAIChatCompletionsDriverBase<
    OptionsT extends OpenAIChatCompletionsDriverOptions = OpenAIChatCompletionsDriverOptions,
> extends AbstractDriver<OptionsT, OpenAICompletionsPrompt> {
    abstract service: OpenAI;

    private readonly completionsModel: OpenAISDKChatCompletionsModelDefinition;

    constructor(options: OptionsT) {
        super(options);
        this.completionsModel = new OpenAISDKChatCompletionsModelDefinition({
            defaultMaxTokens: options.defaultMaxTokens,
            extraBody: options.extraBody,
            resultSchemaMode: options.resultSchemaMode,
        });
    }

    protected async formatPrompt(
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<OpenAICompletionsPrompt> {
        return this.completionsModel.createPrompt(this, segments, options);
    }

    requestTextCompletion(prompt: OpenAICompletionsPrompt, options: ExecutionOptions): Promise<Completion> {
        return this.completionsModel.requestTextCompletion(this, prompt, options);
    }

    requestTextCompletionStream(
        prompt: OpenAICompletionsPrompt,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        return this.completionsModel.requestTextCompletionStream(this, prompt, options);
    }

    buildStreamingConversation(
        prompt: OpenAICompletionsPrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): OpenAICompletionsPrompt {
        return buildOpenAICompletionsStreamingConversation(prompt, result, toolUse, options);
    }

    validateResult(result: Completion, options: ExecutionOptions) {
        super.validateResult(stripOpenAICompletionsThinkBlocksFromCompletion(result), options);
    }
}
