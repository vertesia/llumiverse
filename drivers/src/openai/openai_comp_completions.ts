import {
    type Completion,
    type CompletionChunkObject,
    type ExecutionOptions,
    type JSONObject,
    type PromptOptions,
    PromptRole,
    type PromptSegment,
    readStreamAsBase64,
    type TextFallbackOptions,
    type ToolDefinition,
    type ToolUse,
} from '@llumiverse/core';
import { transformSSEStream } from '@llumiverse/core/async';
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
    modelName: string;
    /** Extra OpenAI-compatible request body fields for model-family-specific options. */
    extraBody?: Record<string, unknown>;
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
    return content || reasoning;
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
        const text = extractOpenAICompletionsText(message, includeThoughts ?? false);
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
            model: this.options.modelName,
            messages: convertToOpenAICompletionsMessages(conversation.messages),
            max_tokens: modelOptions?.max_tokens,
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

        if (options.result_schema) {
            payload.response_format = {
                type: 'json_schema',
                json_schema: { name: 'output', strict: false, schema: options.result_schema },
            };
        }

        return payload;
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
