import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type ExecutionOptions,
    type JSONObject,
    ModelType,
    type PromptOptions,
    PromptRole,
    type PromptSegment,
    readStreamAsBase64,
    type TextFallbackOptions,
    type ToolDefinition,
    type ToolUse,
} from '@llumiverse/core';
import { transformSSEStream } from '@llumiverse/core/async';
import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

/**
 * Message format for OpenAI-compatible chat completions API.
 */
export interface OpenAIMessage {
    role: string;
    content?: string | null | Array<TextPart | ImageUrlPart>;
    /**
     * Required for tool role messages - references the tool_call.id from the assistant's message.
     * Per OpenAI API spec: https://platform.openai.com/docs/api-reference/chat/messages#message-role
     */
    tool_call_id?: string;
    /**
     * Tool calls from assistant messages - stored and sent back with tool results.
     */
    tool_calls?: Array<{
        id: string;
        type: string;
        function: {
            name: string;
            arguments: string;
        };
    }>;
}

interface TextPart {
    type: 'text';
    text: string;
}

interface ImageUrlPart {
    type: 'image_url';
    image_url: { url: string; detail?: 'low' | 'high' | 'auto' };
}

type StreamChunk = string | Uint8Array;
type ReadableAsyncStream = AsyncIterable<StreamChunk>;
type OpenAIToolCall = NonNullable<OpenAIResponseChoice['message']['tool_calls']>[number];
type OpenAIContentPart = TextPart | ImageUrlPart;
type OpenAIRequestMessage = {
    role: string;
    content?: string | null | OpenAIContentPart[];
    tool_call_id?: string;
    tool_calls?: OpenAIToolCall[];
};
type OpenAIToolDefinition = {
    type: 'function';
    function: {
        name: string;
        description?: string;
        parameters: ToolDefinition['input_schema'];
    };
};
type OpenAIChatCompletionPayload = {
    model: string;
    messages: OpenAIRequestMessage[];
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    n: number;
    stop?: string | string[];
    stream: boolean;
    tools?: OpenAIToolDefinition[];
    response_format?: {
        type: 'json_schema';
        json_schema: {
            name: string;
            strict: boolean;
            schema: unknown;
        };
    };
    extra_body?: Record<string, unknown>;
};
type StreamingOpenAIToolUse = ToolUse<unknown> & { _actual_id?: string };

/**
 * Prompt structure for OpenAI-compatible chat completions.
 */
export interface OpenAIPrompt {
    messages: OpenAIMessage[];
    /** Discriminator used by VertexAIDriver.buildStreamingConversation to distinguish from ClaudePrompt */
    _is_openai_compat?: true;
}

/**
 * Response structure for non-streaming chat completions.
 */
interface OpenAIResponseChoice {
    index: number;
    message: {
        role: string;
        content?: string | null | OpenAIContentPart[];
        reasoning_content?: string | null;
        reasoning?: string | null;
        tool_calls?: Array<{
            id: string;
            type: string;
            function: {
                name: string;
                arguments: string;
            };
        }>;
    };
    finish_reason: string | null;
}

interface OpenAIResponseUsage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
}

/**
 * Response structure for non-streaming chat completions.
 */
export interface OpenAIResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: OpenAIResponseChoice[];
    usage?: OpenAIResponseUsage;
}

/**
 * Streaming response delta structure.
 */
interface OpenAIStreamChoiceDelta {
    role?: string;
    content?: string | null | OpenAIContentPart[];
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
}

interface OpenAIStreamChoice {
    index: number;
    delta: OpenAIStreamChoiceDelta;
    finish_reason?: string | null;
}

/**
 * Streaming response structure.
 */
export interface OpenAIStreamResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: OpenAIStreamChoice[];
    usage?: OpenAIResponseUsage;
}

/**
 * Options for configuring the OpenAI-compatible model.
 */
export interface OpenAICompatibleOptions {
    /** The model identifier to send in the request body (e.g., "xai/grok-4.20-reasoning" or "meta/llama-4-scout") */
    modelName: string;
    /** Custom endpoint path override (defaults to "endpoints/openapi/chat/completions") */
    endpointPath?: string;
    /** Region override for the Vertex AI endpoint. Useful when a model only exists in a specific region (e.g., "global" for xAI models). */
    region?: string;
    /** Extra Vertex OpenAI-compatible request body fields for model-family-specific options. */
    extraBody?: Record<string, unknown>;
}

/**
 * Convert a stream to a string.
 */
async function streamToString(stream: ReadableAsyncStream): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks).toString('utf-8');
}

/**
 * Update the conversation messages with new prompt messages.
 */
function updateConversation(conversation: OpenAIPrompt | undefined | null, prompt: OpenAIPrompt): OpenAIPrompt {
    const baseMessages = conversation ? conversation.messages : [];
    return {
        _is_openai_compat: true,
        messages: [...baseMessages, ...(prompt.messages || [])],
    };
}

/**
 * Parse tool calls from OpenAI-style function call format.
 */
function parseToolCalls(toolCalls: OpenAIResponseChoice['message']['tool_calls']): ToolUse[] | undefined {
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

function extractOpenAIText(
    source:
        | Pick<OpenAIResponseChoice['message'], 'content' | 'reasoning_content' | 'reasoning'>
        | Pick<OpenAIStreamChoiceDelta, 'content' | 'reasoning_content' | 'reasoning'>
        | undefined,
    includeThoughts: boolean,
): string {
    if (!source) return '';

    const content = extractOpenAIContentText(source.content);
    const reasoning = extractOpenAIReasoningText(source);

    if (includeThoughts && reasoning) {
        return content ? `${reasoning}\n\n${content}` : reasoning;
    }

    // Some MaaS/vLLM-backed models can emit thinking-only chunks or messages using
    // reasoning_content/reasoning. Use those fields as a fallback so callers do not
    // receive a silent blank response when no normal content is present.
    return content || reasoning;
}

function extractOpenAIReasoningText(
    source:
        | Pick<OpenAIResponseChoice['message'], 'reasoning_content' | 'reasoning'>
        | Pick<OpenAIStreamChoiceDelta, 'reasoning_content' | 'reasoning'>
        | undefined,
): string {
    if (!source) return '';
    return [source.reasoning_content, source.reasoning]
        .filter((value): value is string => typeof value === 'string' && value.length > 0)
        .join('');
}

function extractOpenAIContentText(content: string | null | OpenAIContentPart[] | undefined): string {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';

    return content
        .filter((part): part is TextPart => part.type === 'text' && typeof part.text === 'string')
        .map((part) => part.text)
        .join('\n');
}

/**
 * Convert tool definitions from ExecutionOptions to OpenAI tools format.
 */
function convertToolsToOpenAIFormat(tools: ToolDefinition[] | undefined): OpenAIToolDefinition[] | undefined {
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

/**
 * Convert messages to OpenAI chat completions format.
 * Handles content arrays with text and image parts.
 * Preserves tool_call_id for tool role messages.
 */
function convertToOpenAIMessages(messages: OpenAIMessage[]): Array<{
    role: string;
    content?: string | null | OpenAIContentPart[];
    tool_call_id?: string;
    tool_calls?: OpenAIToolCall[];
}> {
    return messages.map((msg) => {
        // Preserve tool_call_id for tool role messages - this is required by OpenAI API
        const result: {
            role: string;
            content?: string | null | OpenAIContentPart[];
            tool_call_id?: string;
            tool_calls?: OpenAIToolCall[];
        } = {
            role: msg.role,
        };

        // Handle tool_calls for assistant messages
        if (msg.tool_calls && msg.tool_calls.length > 0) {
            result.tool_calls = msg.tool_calls;
        }

        if (msg.content === null) {
            result.content = null;
        } else if (typeof msg.content === 'string') {
            // Empty string is rejected by Grok/OpenAI — normalize to null
            result.content = msg.content || null;
        } else if (msg.content === undefined) {
            result.content = null;
        }

        // Handle content arrays with text and image parts - convert to OpenAI format
        if (Array.isArray(msg.content)) {
            const textParts: string[] = [];
            const imageUrls: Array<{ url: string }> = [];

            for (const part of msg.content) {
                if (part.type === 'text') {
                    textParts.push(part.text);
                } else if (part.type === 'image_url') {
                    imageUrls.push(part.image_url);
                }
            }

            const content: OpenAIContentPart[] = [];
            if (textParts.length > 0) {
                content.push({ type: 'text', text: textParts.join('\n') });
            }
            for (const img of imageUrls) {
                content.push({ type: 'image_url', image_url: img });
            }

            // Return simplified format if only text, otherwise return array
            if (content.length === 1 && content[0].type === 'text') {
                result.content = content[0].text;
            } else if (content.length > 0) {
                result.content = content;
            }
            // If content array is empty, don't set result.content (tool messages may have tool_calls instead)
        }

        // Preserve tool_call_id for tool role messages
        if (msg.tool_call_id) {
            result.tool_call_id = msg.tool_call_id;
        }

        return result;
    });
}

/**
 * Generic OpenAI-compatible model definition for Vertex AI's OpenAPI endpoint.
 *
 * This implementation supports any model that exposes an OpenAI-compatible API
 * through Vertex AI's `/endpoints/openapi/chat/completions` endpoint.
 *
 * The correct endpoint format is documented at:
 * https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/call-open-model-apis
 *
 * URL pattern: `https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions`
 *
 * @example
 * // For xAI Grok models
 * new OpenAICompatibleModelDefinition({ modelName: "xai/grok-4.20-reasoning" })
 *
 * @example
 * // For Meta Llama models (via Vertex AI MaaS)
 * new OpenAICompatibleModelDefinition({ modelName: "meta/llama-4-scout" })
 */
export class OpenAICompatibleModelDefinition implements ModelDefinition<OpenAIPrompt> {
    model: AIModel;
    private options: OpenAICompatibleOptions;

    constructor(options: OpenAICompatibleOptions) {
        this.options = options;
        const modelName = options.modelName.split('/').pop() || options.modelName;
        this.model = {
            id: options.modelName,
            name: modelName,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } as AIModel;
    }

    async createPrompt(
        _driver: VertexAIDriver,
        segments: PromptSegment[],
        _options: PromptOptions,
    ): Promise<OpenAIPrompt> {
        const messages: OpenAIMessage[] = [];

        // Extract system message if present
        let systemContent = '';
        for (const segment of segments) {
            if (segment.role === PromptRole.system && segment.content) {
                systemContent += `${segment.content}\n`;
            }
        }

        // Add system message as a system role message
        if (systemContent.trim()) {
            messages.push({ role: 'system', content: systemContent.trim() });
        }

        // Process remaining segments
        for (const segment of segments) {
            if (segment.role === PromptRole.system) {
                continue; // Already handled above
            }

            if (segment.role === PromptRole.tool) {
                // Validate that tool_use_id is present - this is required by OpenAI API
                if (!segment.tool_use_id) {
                    throw new Error('Tool prompt segment must have a tool_use_id to reference the original tool call');
                }

                // Build content array for tool response
                const content: Array<TextPart | ImageUrlPart> = [];

                if (segment.content) {
                    content.push({ type: 'text', text: segment.content });
                } else {
                    content.push({ type: 'text', text: '' });
                }

                // Handle file attachments in tool responses
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
                            } as ImageUrlPart);
                        } else if (file.mime_type?.startsWith('text/')) {
                            const fileStream = await file.getStream();
                            const fileContent = await streamToString(fileStream);
                            content.push({ type: 'text', text: `\n\nFile content:\n${fileContent}` });
                        }
                    }
                }

                // Build the tool message with tool_call_id (required by OpenAI API)
                const toolMessage: OpenAIMessage = {
                    role: 'tool',
                    tool_call_id: segment.tool_use_id,
                    content:
                        content.length === 1 && content[0]?.type === 'text'
                            ? content[0].text
                            : (content as Array<TextPart | ImageUrlPart>),
                };
                messages.push(toolMessage);
            } else {
                // Build content for user/assistant/safety segments
                let content: string | Array<TextPart | ImageUrlPart> = segment.content || '';

                // Handle file attachments (images)
                if (segment.files && segment.files.length > 0) {
                    const parts: Array<TextPart | ImageUrlPart> = [];

                    // Add text content as a part
                    if (content && typeof content === 'string' && content.trim()) {
                        parts.push({ type: 'text', text: content });
                    }

                    // Process image files
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
        driver: VertexAIDriver,
        prompt: OpenAIPrompt,
        options: ExecutionOptions,
    ): Promise<Completion> {
        let conversation = updateConversation(options.conversation as OpenAIPrompt, prompt);

        const modelOptions = options.model_options as TextFallbackOptions;
        const includeThoughts = (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })
            ?.include_thoughts;

        // Build the request payload following OpenAI chat completions format
        const payload: OpenAIChatCompletionPayload = {
            model: this.options.modelName,
            messages: convertToOpenAIMessages(conversation.messages),
            max_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
            top_p: modelOptions?.top_p,
            n: 1,
            stop: modelOptions?.stop_sequence,
            stream: false,
        };
        if (this.options.extraBody) {
            payload.extra_body = this.options.extraBody;
        }

        // Vertex AI OpenAI-compatible endpoint requires tools to be included in the request body
        const toolsPayload = convertToolsToOpenAIFormat(options.tools);
        if (toolsPayload && toolsPayload.length > 0) {
            payload.tools = toolsPayload;
        }

        if (options.result_schema) {
            payload.response_format = {
                type: 'json_schema',
                json_schema: { name: 'output', strict: false, schema: options.result_schema },
            };
        }

        // Make POST request to the OpenAI-compatible endpoint via Vertex AI
        // Use region override if specified (e.g., "global" for xAI models)
        const effectiveRegion = this.options.region || undefined;
        const client = effectiveRegion ? driver.getFetchClientForRegion(effectiveRegion) : driver.getFetchClient();
        const endpoint = this.options.endpointPath || 'endpoints/openapi/chat/completions';
        const result = (await client.post(endpoint, {
            payload,
        })) as OpenAIResponse;

        // Extract response data
        const choice = result?.choices?.[0];
        const message = choice?.message;
        const text = extractOpenAIText(message, includeThoughts ?? false);
        const tool_use = parseToolCalls(message?.tool_calls);

        // Update conversation with the response
        // Format assistant message with tool calls for conversation history
        const assistantMessage: OpenAIMessage = {
            role: 'assistant',
            content: text || null,
        };

        // If there are tool calls, store them in the assistant message with proper OpenAI format
        if (tool_use && tool_use.length > 0 && message?.tool_calls) {
            assistantMessage.tool_calls = message.tool_calls.map((tc) => ({
                id: tc.id,
                type: tc.type,
                function: {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            }));
        }

        conversation = updateConversation(conversation, {
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
        driver: VertexAIDriver,
        prompt: OpenAIPrompt,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        const conversation = updateConversation(options.conversation as OpenAIPrompt, prompt);

        const modelOptions = options.model_options as TextFallbackOptions;
        const includeThoughts = (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })
            ?.include_thoughts;

        // Build the request payload following OpenAI chat completions format
        const payload: OpenAIChatCompletionPayload = {
            model: this.options.modelName,
            messages: convertToOpenAIMessages(conversation.messages),
            max_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
            top_p: modelOptions?.top_p,
            n: 1,
            stop: modelOptions?.stop_sequence,
            stream: true,
        };
        if (this.options.extraBody) {
            payload.extra_body = this.options.extraBody;
        }

        // Vertex AI OpenAI-compatible endpoint requires tools to be included in the request body
        const toolsPayload = convertToolsToOpenAIFormat(options.tools);
        if (toolsPayload && toolsPayload.length > 0) {
            payload.tools = toolsPayload;
        }

        if (options.result_schema) {
            payload.response_format = {
                type: 'json_schema',
                json_schema: { name: 'output', strict: false, schema: options.result_schema },
            };
        }

        // Make POST request to the OpenAI-compatible endpoint via Vertex AI
        // Use region override if specified (e.g., "global" for xAI models)
        const effectiveRegion = this.options.region || undefined;
        const client = effectiveRegion ? driver.getFetchClientForRegion(effectiveRegion) : driver.getFetchClient();
        const endpoint = this.options.endpointPath || 'endpoints/openapi/chat/completions';
        const responseStream = (await client.post(endpoint, {
            payload,
            reader: 'sse',
        })) as ReadableStream;

        let emittedContent = false;
        let reasoningFallback = '';

        return transformSSEStream(responseStream, (data: string) => {
            const json = JSON.parse(data) as OpenAIStreamResponse;
            const choice = json.choices?.[0];
            const delta = choice?.delta;

            // Handle tool calls in streaming mode with proper accumulation
            if (delta?.tool_calls && delta.tool_calls.length > 0) {
                // In streaming mode, tool call arguments come as incremental JSON fragments.
                // OpenAI sends the real id only on the first chunk per tool call, so use the
                // stable index as the accumulation key and let CompletionStream restore the
                // real id later. Loop over all items — each chunk typically carries one but
                // the spec allows multiple.
                const toolUseChunks: StreamingOpenAIToolUse[] = delta.tool_calls.map((tc) => {
                    const toolUse: StreamingOpenAIToolUse = {
                        id: `tool_${tc.index ?? 0}`,
                        tool_name: tc.function?.name ?? '',
                        // Pass raw string — CompletionStream will accumulate and parse
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

            const content = extractOpenAIContentText(delta?.content);
            const reasoning = extractOpenAIReasoningText(delta);
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
}
