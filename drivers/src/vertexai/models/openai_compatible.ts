import {
    AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType,
    PromptOptions, PromptRole, PromptSegment,
    readStreamAsBase64,
    TextFallbackOptions,
    ToolUse
} from "@llumiverse/core";
import { transformSSEStream } from "@llumiverse/core/async";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";

/**
 * Message format for OpenAI-compatible chat completions API.
 */
interface OpenAIMessage {
    role: string;
    content: string | Array<TextPart | ImageUrlPart>;
}

interface TextPart {
    type: "text";
    text: string;
}

interface ImageUrlPart {
    type: "image_url";
    image_url: { url: string; detail?: "low" | "high" | "auto" };
}

/**
 * Prompt structure for OpenAI-compatible chat completions.
 */
export interface OpenAIPrompt {
    messages: OpenAIMessage[];
}

/**
 * Response structure for non-streaming chat completions.
 */
interface OpenAIResponseChoice {
    index: number;
    message: {
        role: string;
        content?: string;
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
    content?: string;
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
}

/**
 * Convert a stream to a string.
 */
async function streamToString(stream: any): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks).toString('utf-8');
}

/**
 * Update the conversation messages with new prompt messages.
 */
function updateConversation(
    conversation: OpenAIPrompt | undefined | null,
    prompt: OpenAIPrompt
): OpenAIPrompt {
    const baseMessages = conversation ? conversation.messages : [];
    return {
        messages: [...baseMessages, ...(prompt.messages || [])],
    };
}

/**
 * Parse tool calls from OpenAI-style function call format.
 */
function parseToolCalls(
    toolCalls: OpenAIResponseChoice['message']['tool_calls']
): ToolUse[] | undefined {
    if (!toolCalls || toolCalls.length === 0) {
        return undefined;
    }
    return toolCalls.map(tc => ({
        id: tc.id ?? '',
        tool_name: tc.function?.name ?? '',
        tool_input: safeJsonParse(tc.function?.arguments),
    }));
}

function safeJsonParse(value: string | undefined): any {
    if (typeof value !== 'string') {
        return {};
    }
    try {
        return JSON.parse(value);
    } catch {
        return {};
    }
}


/**
 * Convert messages to OpenAI chat completions format.
 * Handles content arrays with text and image parts.
 */
function convertToOpenAIMessages(
    messages: OpenAIMessage[]
): Array<{ role: string; content?: string | any[] }> {
    return messages.map(msg => {
        if (typeof msg.content === 'string') {
            return {
                role: msg.role,
                content: msg.content
            };
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

            const content: Array<{ type: string; text?: string; image_url?: { url: string } }> = [];
            if (textParts.length > 0) {
                content.push({ type: 'text', text: textParts.join('\n') });
            }
            for (const img of imageUrls) {
                content.push({ type: 'image_url', image_url: img });
            }

            // Return simplified format if only text, otherwise return array
            if (content.length === 1 && content[0].type === 'text') {
                return {
                    role: msg.role,
                    content: content[0].text
                };
            }

            return {
                role: msg.role,
                content
            };
        }

        return { role: msg.role, content: '' };
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

    model: AIModel
    private options: OpenAICompatibleOptions

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

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], _options: PromptOptions): Promise<OpenAIPrompt> {
        const messages: OpenAIMessage[] = [];

        // Extract system message if present
        let systemContent = '';
        for (const segment of segments) {
            if (segment.role === PromptRole.system && segment.content) {
                systemContent += segment.content + '\n';
            }
        }

        // Add system message as a system role message
        if (systemContent.trim()) {
            messages.push({
                role: 'system',
                content: systemContent.trim()
            });
        }

        // Process remaining segments
        for (const segment of segments) {
            if (segment.role === PromptRole.system) {
                continue; // Already handled above
            }

            if (segment.role === PromptRole.tool && segment.content) {
                // Tool results are sent as tool role messages
                const content: string | Array<TextPart> = [];
                
                if (segment.content) {
                    content.push({ type: 'text', text: segment.content });
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
                                    detail: 'auto'
                                }
                            } as any);
                        } else if (file.mime_type?.startsWith('text/')) {
                            const fileStream = await file.getStream();
                            const fileContent = await streamToString(fileStream);
                            content.push({ type: 'text', text: `\n\nFile content:\n${fileContent}` });
                        }
                    }
                }

                messages.push({
                    role: 'tool',
                    content: content.length === 1 && typeof content === 'string' ? content : (content as any)
                } as any);
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
                                    detail: 'auto'
                                }
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
                    content
                });
            }
        }

        // Add JSON schema instruction if needed
        if (_options.result_schema) {
            messages.push({
                role: 'user',
                content: "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(_options.result_schema)
            });
        }

        return {
            messages,
        };
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: OpenAIPrompt, options: ExecutionOptions): Promise<Completion> {
        let conversation = updateConversation(options.conversation as OpenAIPrompt, prompt);

        const modelOptions = options.model_options as TextFallbackOptions;

        // Build the request payload following OpenAI chat completions format
        const payload: Record<string, any> = {
            model: this.options.modelName,
            messages: convertToOpenAIMessages(conversation.messages),
            max_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
            top_p: modelOptions?.top_p,
            n: 1,
            stop: modelOptions?.stop_sequence,
            stream: false,
        };

        // Make POST request to the OpenAI-compatible endpoint via Vertex AI
        // Use region override if specified (e.g., "global" for xAI models)
        const effectiveRegion = this.options.region || undefined;
        const client = effectiveRegion
            ? driver.getFetchClientForRegion(effectiveRegion)
            : driver.getFetchClient();
        const endpoint = this.options.endpointPath || 'endpoints/openapi/chat/completions';
        const result = await client.post(endpoint, {
            payload
        }) as OpenAIResponse;

        // Extract response data
        const choice = result?.choices?.[0];
        const message = choice?.message;
        const text = message?.content ?? '';
        const tool_use = parseToolCalls(message?.tool_calls);

        // Update conversation with the response
        conversation = updateConversation(conversation, {
            messages: [{
                role: message?.role || 'assistant',
                content: text
            }]
        });

        return {
            result: text ? [{ type: "text", value: text }] : [{ type: "text", value: '' }],
            tool_use,
            token_usage: result.usage ? {
                prompt: result.usage.prompt_tokens,
                result: result.usage.completion_tokens,
                total: result.usage.total_tokens
            } : undefined,
            finish_reason: choice?.finish_reason || undefined,
            conversation
        };
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: OpenAIPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const conversation = updateConversation(options.conversation as OpenAIPrompt, prompt);

        const modelOptions = options.model_options as TextFallbackOptions;

        // Build the request payload following OpenAI chat completions format
        const payload: Record<string, any> = {
            model: this.options.modelName,
            messages: convertToOpenAIMessages(conversation.messages),
            max_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
            top_p: modelOptions?.top_p,
            n: 1,
            stop: modelOptions?.stop_sequence,
            stream: true,
        };

        // Make POST request to the OpenAI-compatible endpoint via Vertex AI
        // Use region override if specified (e.g., "global" for xAI models)
        const effectiveRegion = this.options.region || undefined;
        const client = effectiveRegion
            ? driver.getFetchClientForRegion(effectiveRegion)
            : driver.getFetchClient();
        const endpoint = this.options.endpointPath || 'endpoints/openapi/chat/completions';
        const responseStream = await client.post(endpoint, {
            payload,
            reader: 'sse'
        });

        return transformSSEStream(responseStream, (data: string) => {
            const json = JSON.parse(data) as OpenAIStreamResponse;
            const choice = json.choices?.[0];
            const delta = choice?.delta;

            // Handle tool calls in streaming mode
            if (delta?.tool_calls && delta.tool_calls.length > 0) {
                const tc = delta.tool_calls[0];
                return {
                    result: [],
                    tool_use: [{
                        id: tc.id ?? `tool_${tc.index ?? 0}`,
                        tool_name: tc.function?.name ?? '',
                        tool_input: (tc.function?.arguments as any) ?? ''
                    }]
                } satisfies CompletionChunkObject;
            }

            // Handle text content
            const content = delta?.content ?? '';
            return {
                result: content ? [{ type: "text", value: content }] : [],
                finish_reason: choice?.finish_reason || undefined,
                token_usage: json.usage ? {
                    prompt: json.usage.prompt_tokens,
                    result: json.usage.completion_tokens,
                    total: json.usage.total_tokens,
                } : undefined
            } satisfies CompletionChunkObject;
        });
    }
}
