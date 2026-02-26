import {
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
} from '@anthropic-ai/sdk/error';
import { ContentBlock, ContentBlockParam, DocumentBlockParam, ImageBlockParam, Message, MessageParam, TextBlockParam, ToolResultBlockParam } from "@anthropic-ai/sdk/resources/index.js";
import { MessageStreamParams } from "@anthropic-ai/sdk/resources/index.mjs";
import { MessageCreateParamsBase, MessageCreateParamsNonStreaming, RawMessageStreamEvent } from "@anthropic-ai/sdk/resources/messages.js";
import {
    AIModel, Completion, CompletionChunkObject, ExecutionOptions,
    getConversationMeta,
    getMaxTokensLimitVertexAi,
    incrementConversationTurn,
    JSONObject,
    LlumiverseError, LlumiverseErrorContext,
    ModelType,
    PromptRole, PromptSegment, readStreamAsBase64, readStreamAsString, StatelessExecutionOptions,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    ToolUse,
    truncateLargeTextInConversation,
    VertexAIClaudeOptions
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";

export const ANTHROPIC_REGIONS: Record<string, string> = {
    us: "us-east5",
    europe: "europe-west1",
    global: "global",
}

export const NON_GLOBAL_ANTHROPIC_MODELS = [
    "claude-3-5",
    "claude-3",
];

interface ClaudePrompt {
    messages: MessageParam[];
    system?: TextBlockParam[];
}

function claudeFinishReason(reason: string | undefined) {
    if (!reason) return undefined;
    switch (reason) {
        case 'end_turn': return "stop";
        case 'max_tokens': return "length";
        default: return reason; //stop_sequence
    }
}

export function collectTools(content: ContentBlock[]): ToolUse[] | undefined {
    const out: ToolUse[] = [];

    for (const block of content) {
        if (block.type === "tool_use") {
            out.push({
                id: block.id,
                tool_name: block.name,
                tool_input: block.input as JSONObject,
            });
        }
    }

    return out.length > 0 ? out : undefined;
}

function collectAllTextContent(content: ContentBlock[], includeThoughts: boolean = false) {
    const textParts: string[] = [];

    // First pass: collect thinking blocks
    if (includeThoughts) {
        for (const block of content) {
            if (block.type === 'thinking' && block.thinking) {
                textParts.push(block.thinking);
            } else if (block.type === 'redacted_thinking' && block.data) {
                textParts.push(`[Redacted thinking: ${block.data}]`);
            }
        }
        if (textParts.length > 0) {
            textParts.push(''); // Create a new line after thinking blocks
        }
    }

    // Second pass: collect text blocks
    for (const block of content) {
        if (block.type === 'text' && block.text) {
            textParts.push(block.text);
        }
    }

    return textParts.join('\n');
}

//Used to get a max_token value when not specified in the model options. Claude requires it to be set.
function maxToken(option: StatelessExecutionOptions): number {
    const modelOptions = option.model_options as VertexAIClaudeOptions | undefined;
    if (modelOptions && typeof modelOptions.max_tokens === "number") {
        return modelOptions.max_tokens;
    } else {
        let maxSupportedTokens = getMaxTokensLimitVertexAi(option.model);
        // Fallback to the default max tokens limit for the model
        if (option.model.includes('claude-3-7-sonnet') && (modelOptions?.thinking_budget_tokens ?? 0) < 48000) {
            maxSupportedTokens = 64000; // Claude 3.7 can go up to 128k with a beta header, but when no max tokens is specified, we default to 64k.
        }
        return maxSupportedTokens;
    }
}

// Type-safe overloads for collectFileBlocks
async function collectFileBlocks(segment: PromptSegment, restrictedTypes: true): Promise<Array<TextBlockParam | ImageBlockParam>>;
async function collectFileBlocks(segment: PromptSegment, restrictedTypes?: false): Promise<ContentBlockParam[]>;
async function collectFileBlocks(segment: PromptSegment, restrictedTypes: boolean = false): Promise<ContentBlockParam[]> {
    const contentBlocks: ContentBlockParam[] = [];

    for (const file of segment.files || []) {
        if (file.mime_type?.startsWith("image/")) {
            const allowedTypes = ["image/png", "image/jpeg", "image/gif", "image/webp"];
            if (!allowedTypes.includes(file.mime_type)) {
                throw new Error(`Unsupported image type: ${file.mime_type}`);
            }
            const mimeType = String(file.mime_type) as "image/png" | "image/jpeg" | "image/gif" | "image/webp";

            contentBlocks.push({
                type: 'image',
                source: {
                    type: 'base64',
                    data: await readStreamAsBase64(await file.getStream()),
                    media_type: mimeType
                }
            } satisfies ImageBlockParam);
        } else if (!restrictedTypes) {
            if (file.mime_type === "application/pdf") {
                contentBlocks.push({
                    title: file.name,
                    type: 'document',
                    source: {
                        type: 'base64',
                        data: await readStreamAsBase64(await file.getStream()),
                        media_type: 'application/pdf'
                    }
                } satisfies DocumentBlockParam);
            } else if (file.mime_type?.startsWith("text/")) {
                contentBlocks.push({
                    title: file.name,
                    type: 'document',
                    source: {
                        type: 'text',
                        data: await readStreamAsString(await file.getStream()),
                        media_type: 'text/plain'
                    }
                } satisfies DocumentBlockParam);
            }
        }
    }

    return contentBlocks;
}

export class ClaudeModelDefinition implements ModelDefinition<ClaudePrompt> {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } satisfies AIModel;
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: ExecutionOptions): Promise<ClaudePrompt> {
        // Convert the prompt to the format expected by the Claude API
        let system: TextBlockParam[] | undefined = segments
            .filter(segment => segment.role === PromptRole.system)
            .map(segment => ({
                text: segment.content,
                type: 'text'
            }));

        if (options.result_schema) {
            let schemaText: string = '';
            if (options.tools && options.tools.length > 0) {
                schemaText = "When not calling tools, the answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema);
            } else {
                schemaText = "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema);
            }

            const schemaSegments: TextBlockParam = {
                text: schemaText,
                type: 'text'
            }
            system.push(schemaSegments);
        }

        let messages: MessageParam[] = [];
        const safetyMessages: MessageParam[] = [];
        for (const segment of segments) {
            if (segment.role === PromptRole.system) {
                continue;
            }

            if (segment.role === PromptRole.tool) {
                if (!segment.tool_use_id) {
                    throw new Error("Tool prompt segment must have a tool use ID");
                }

                // Build content blocks for tool results (restricted types)
                const contentBlocks: Array<TextBlockParam | ImageBlockParam> = [];

                if (segment.content) {
                    contentBlocks.push({
                        type: 'text',
                        text: segment.content
                    } satisfies TextBlockParam);
                }

                // Collect file blocks with type safety
                const fileBlocks = await collectFileBlocks(segment, true);
                contentBlocks.push(...fileBlocks);

                messages.push({
                    role: 'user',
                    content: [{
                        type: 'tool_result',
                        tool_use_id: segment.tool_use_id,
                        content: contentBlocks,
                    } satisfies ToolResultBlockParam]
                });

            } else {
                // Build content blocks for regular messages (all types allowed)
                const contentBlocks: ContentBlockParam[] = [];

                if (segment.content) {
                    contentBlocks.push({
                        type: 'text',
                        text: segment.content
                    } satisfies TextBlockParam);
                }

                // Collect file blocks without restrictions
                const fileBlocks = await collectFileBlocks(segment, false);
                contentBlocks.push(...fileBlocks);

                if (contentBlocks.length === 0) {
                    continue; // skip empty segments
                }

                const messageParam: MessageParam = {
                    role: segment.role === PromptRole.assistant ? 'assistant' : 'user',
                    content: contentBlocks
                };

                if (segment.role === PromptRole.safety) {
                    safetyMessages.push(messageParam);
                } else {
                    messages.push(messageParam);
                }
            }
        }

        messages = messages.concat(safetyMessages);

        if (system && system.length === 0) {
            system = undefined; // If system is empty, set to undefined
        }

        return {
            messages: messages,
            system: system
        }
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<Completion> {
        const splits = options.model.split("/");
        let region: string | undefined = undefined;
        if (splits[0] === "locations" && splits.length >= 2) {
            region = splits[1];
        }
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const client = await driver.getAnthropicClient(region);
        options.model_options = options.model_options as VertexAIClaudeOptions;

        if (options.model_options?._option_id !== "vertexai-claude") {
            driver.logger.warn({ options: options.model_options }, "Invalid model options");
        }

        let conversation = updateConversation(options.conversation as ClaudePrompt, prompt);

        const { payload, requestOptions } = getClaudePayload(options, conversation);
        // disable streaming, the create function is overloaded so payload type matters.
        const nonStreamingPayload: MessageCreateParamsNonStreaming = { ...payload, stream: false };

        const result = await client.messages.create(nonStreamingPayload, requestOptions) satisfies Message;

        // Use the new function to collect text content, including thinking if enabled
        const includeThoughts = options.model_options?.include_thoughts ?? false;
        const text = collectAllTextContent(result.content, includeThoughts);
        const tool_use = collectTools(result.content);

        conversation = updateConversation(conversation, createPromptFromResponse(result));

        // Increment turn counter and apply stripping (same pattern as other drivers)
        conversation = incrementConversationTurn(conversation) as ClaudePrompt;
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens,
        };
        let processedConversation = stripBase64ImagesFromConversation(conversation, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        return {
            result: text ? [{ type: "text", value: text }] : [{ type: "text", value: '' }],
            tool_use,
            token_usage: {
                prompt: result.usage.input_tokens,
                result: result.usage.output_tokens,
                total: result.usage.input_tokens + result.usage.output_tokens
            },
            // make sure we set finish_reason to the correct value (claude is normally setting this by itself)
            finish_reason: tool_use ? "tool_use" : claudeFinishReason(result?.stop_reason ?? ''),
            conversation: processedConversation
        } satisfies Completion;
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const splits = options.model.split("/");
        let region: string | undefined = undefined;
        if (splits[0] === "locations" && splits.length >= 2) {
            region = splits[1];
        }
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const client = await driver.getAnthropicClient(region);
        const model_options = options.model_options as VertexAIClaudeOptions | undefined;

        if (model_options?._option_id !== "vertexai-claude") {
            driver.logger.warn({ options: options.model_options }, "Invalid model options");
        }

        // Include conversation history (same as non-streaming)
        const conversation = updateConversation(options.conversation as ClaudePrompt, prompt);

        const { payload, requestOptions } = getClaudePayload(options, conversation);
        const streamingPayload: MessageStreamParams = { ...payload, stream: true };

        const response_stream = await client.messages.stream(streamingPayload, requestOptions);

        // Track current tool use being built from streaming
        let currentToolUse: { id: string; name: string; inputJson: string } | null = null;

        const stream = asyncMap(response_stream, async (streamEvent: RawMessageStreamEvent) => {
            switch (streamEvent.type) {
                case "message_start":
                    return {
                        result: [{ type: "text", value: '' }],
                        token_usage: {
                            prompt: streamEvent.message.usage.input_tokens,
                            result: streamEvent.message.usage.output_tokens
                        }
                    } satisfies CompletionChunkObject;
                case "message_delta":
                    return {
                        result: [{ type: "text", value: '' }],
                        token_usage: {
                            result: streamEvent.usage.output_tokens
                        },
                        finish_reason: claudeFinishReason(streamEvent.delta.stop_reason ?? undefined),
                    } satisfies CompletionChunkObject;
                case "content_block_start":
                    // Handle tool_use blocks
                    if (streamEvent.content_block.type === "tool_use") {
                        currentToolUse = {
                            id: streamEvent.content_block.id,
                            name: streamEvent.content_block.name,
                            inputJson: ''
                        };
                        return {
                            result: [],
                            tool_use: [{
                                id: streamEvent.content_block.id,
                                tool_name: streamEvent.content_block.name,
                                tool_input: '' as any // Will be accumulated via input_json_delta
                            }]
                        } satisfies CompletionChunkObject;
                    }
                    // Handle redacted thinking blocks
                    if (streamEvent.content_block.type === "redacted_thinking" && model_options?.include_thoughts) {
                        return {
                            result: [{ type: "text", value: `[Redacted thinking: ${streamEvent.content_block.data}]` }]
                        } satisfies CompletionChunkObject;
                    }
                    break;
                case "content_block_delta":
                    // Handle different delta types
                    switch (streamEvent.delta.type) {
                        case "text_delta":
                            return {
                                result: streamEvent.delta.text ? [{ type: "text", value: streamEvent.delta.text }] : []
                            } satisfies CompletionChunkObject;
                        case "input_json_delta":
                            // Accumulate tool input JSON
                            if (currentToolUse && streamEvent.delta.partial_json) {
                                return {
                                    result: [],
                                    tool_use: [{
                                        id: currentToolUse.id,
                                        tool_name: '', // Name already sent in content_block_start
                                        tool_input: streamEvent.delta.partial_json as any
                                    }]
                                } satisfies CompletionChunkObject;
                            }
                            break;
                        case "thinking_delta":
                            if (model_options?.include_thoughts) {
                                return {
                                    result: streamEvent.delta.thinking ? [{ type: "text", value: streamEvent.delta.thinking }] : [],
                                } satisfies CompletionChunkObject;
                            }
                            break;
                        case "signature_delta":
                            // Signature deltas, signify the end of the thoughts.
                            if (model_options?.include_thoughts) {
                                return {
                                    result: [{ type: "text", value: '\n\n' }], // Double newline for more spacing
                                } satisfies CompletionChunkObject;
                            }
                            break;
                    }
                    break;
                case "content_block_stop":
                    // Reset current tool use tracking when block ends
                    if (currentToolUse) {
                        currentToolUse = null;
                    }
                    // Handle the end of content blocks, for redacted thinking blocks
                    if (model_options?.include_thoughts) {
                        return {
                            result: [{ type: "text", value: '\n\n' }] // Add double newline for spacing
                        } satisfies CompletionChunkObject;
                    }
                    break;
            }

            // Default case for all other event types
            return {
                result: []
            } satisfies CompletionChunkObject;
        });

        return stream;
    }

    /**
     * Format Anthropic API errors into LlumiverseError with proper status codes and retryability.
     * 
     * Anthropic API errors have a specific structure:
     * - APIError.status: HTTP status code (400, 401, 403, 404, 409, 422, 429, 500+)
     * - APIError.error: Nested error object with type and message
     * - APIError.requestID: Request ID for support (can be null)
     * 
     * Common error types:
     * - BadRequestError (400): Invalid request parameters
     * - AuthenticationError (401): Authentication required
     * - PermissionDeniedError (403): Insufficient permissions
     * - NotFoundError (404): Resource not found
     * - ConflictError (409): Resource conflict
     * - UnprocessableEntityError (422): Validation error
     * - RateLimitError (429): Rate limit exceeded
     * - InternalServerError (500+): Server-side errors
     * - APIConnectionError: Connection issues (no status code)
     * - APIConnectionTimeoutError: Request timeout (no status code)
     * 
     * @see https://docs.anthropic.com/en/api/errors
     */
    formatLlumiverseError(
        _driver: VertexAIDriver,
        error: unknown,
        context: LlumiverseErrorContext
    ): LlumiverseError {
        // Check if it's an Anthropic API error
        const isAnthropicError = this.isAnthropicApiError(error);

        if (!isAnthropicError) {
            // Not an Anthropic API error, use default handling
            throw error;
        }

        const apiError = error as APIError;
        const httpStatusCode = apiError.status;

        // Extract error message and nested error details
        let message = apiError.message || String(error);

        // Extract error type from nested error object if available
        let errorType: string | undefined;
        if (apiError.error && typeof apiError.error === 'object') {
            const nestedError = apiError.error as any;
            if (nestedError.error && typeof nestedError.error === 'object') {
                errorType = nestedError.error.type;
                // Use the nested error message if it's more specific
                if (nestedError.error.message) {
                    message = nestedError.error.message;
                }
            }
        }

        // Build user-facing message with status code
        let userMessage = message;

        // Include status code in message (for end-user visibility)
        if (httpStatusCode) {
            userMessage = `[${httpStatusCode}] ${userMessage}`;
        }

        // Include error type if available
        if (errorType && errorType !== 'error') {
            userMessage = `${errorType}: ${userMessage}`;
        }

        // Add request ID if available (useful for Anthropic support)
        if (apiError.requestID) {
            userMessage += ` (Request ID: ${apiError.requestID})`;
        }

        // Determine retryability based on Anthropic error types
        const retryable = this.isClaudeErrorRetryable(error, httpStatusCode, errorType);

        // Use the error constructor name as the error name
        const errorName = error.constructor?.name || 'AnthropicError';

        return new LlumiverseError(
            `[${context.provider}] ${userMessage}`,
            retryable,
            context,
            error,
            httpStatusCode,
            errorName
        );
    }

    /**
     * Type guard to check if error is an Anthropic API error.
     */
    private isAnthropicApiError(error: unknown): error is APIError {
        return (
            error !== null &&
            typeof error === 'object' &&
            error instanceof APIError
        );
    }

    /**
     * Determine if an Anthropic API error is retryable.
     * 
     * Retryable errors:
     * - RateLimitError (429): Rate limit exceeded, retry with backoff
     * - InternalServerError (500+): Server-side errors
     * - APIConnectionTimeoutError: Request timeout
     * - 408 (Request Timeout): Request timeout
     * - 529 (Overloaded): Service overloaded
     * 
     * Non-retryable errors:
     * - BadRequestError (400): Invalid request parameters
     * - AuthenticationError (401): Authentication failure
     * - PermissionDeniedError (403): Insufficient permissions
     * - NotFoundError (404): Resource not found
     * - ConflictError (409): Resource conflict
     * - UnprocessableEntityError (422): Validation error
     * - Other 4xx client errors
     * - invalid_request_error: Invalid request structure
     * 
     * @param error - The error object
     * @param httpStatusCode - The HTTP status code if available
     * @param errorType - The nested error type if available
     * @returns True if retryable, false if not retryable, undefined if unknown
     */
    private isClaudeErrorRetryable(
        error: unknown,
        httpStatusCode: number | undefined,
        errorType: string | undefined
    ): boolean | undefined {
        // Check specific Anthropic error types by class
        if (error instanceof RateLimitError) return true;
        if (error instanceof InternalServerError) return true;
        if (error instanceof APIConnectionTimeoutError) return true;

        // Non-retryable by error type
        if (error instanceof BadRequestError) return false;
        if (error instanceof AuthenticationError) return false;
        if (error instanceof PermissionDeniedError) return false;
        if (error instanceof NotFoundError) return false;
        if (error instanceof ConflictError) return false;
        if (error instanceof UnprocessableEntityError) return false;

        // Check nested error type
        if (errorType === 'invalid_request_error') return false;

        // Use HTTP status code
        if (httpStatusCode !== undefined) {
            if (httpStatusCode === 429) return true; // Rate limit
            if (httpStatusCode === 408) return true; // Request timeout
            if (httpStatusCode === 529) return true; // Overloaded
            if (httpStatusCode >= 500 && httpStatusCode < 600) return true; // Server errors
            if (httpStatusCode >= 400 && httpStatusCode < 500) return false; // Client errors
        }

        // Connection errors without status codes
        if (error instanceof APIConnectionError && !(error instanceof APIConnectionTimeoutError)) {
            // Generic connection errors might be retryable (network issues)
            return true;
        }

        // Unknown error type - let consumer decide retry strategy
        return undefined;
    }
}

function createPromptFromResponse(response: Message): ClaudePrompt {
    return {
        messages: [{
            role: response.role,
            content: response.content,
        }],
        system: undefined
    }
}

/**
 * Merge consecutive user messages in the conversation.
 * This is required because Anthropic's API expects all tool_result blocks
 * from a single assistant turn to be in one user message.
 * When multiple tool results are added as separate user messages,
 * we need to merge them before sending to the API.
 */
export function mergeConsecutiveUserMessages(messages: MessageParam[]): MessageParam[] {
    if (messages.length === 0) return [];

    // Check if any merging is needed
    const needsMerging = messages.some((msg, i) =>
        i < messages.length - 1 &&
        msg.role === 'user' &&
        messages[i + 1].role === 'user'
    );

    if (!needsMerging) {
        return messages;
    }

    const result: MessageParam[] = [];
    let i = 0;

    while (i < messages.length) {
        const current = messages[i];

        if (current.role === 'user') {
            // Collect all consecutive user messages
            const mergedContent: MessageParam['content'] = [];

            while (i < messages.length && messages[i].role === 'user') {
                const userMsg = messages[i];
                if (Array.isArray(userMsg.content)) {
                    mergedContent.push(...userMsg.content);
                } else if (typeof userMsg.content === 'string') {
                    mergedContent.push({ type: 'text', text: userMsg.content });
                }
                i++;
            }

            result.push({
                role: 'user',
                content: mergedContent
            });
        } else {
            result.push(current);
            i++;
        }
    }

    return result;
}

/**
 * Update the conversation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: ClaudePrompt | undefined | null, prompt: ClaudePrompt): ClaudePrompt {
    const baseSystemMessages = conversation?.system || [];
    const baseMessages = conversation?.messages || [];
    const system = baseSystemMessages.concat(prompt.system || []);
    // Merge consecutive user messages to ensure tool_result blocks are properly grouped
    const mergedMessages = mergeConsecutiveUserMessages(baseMessages.concat(prompt.messages || []));
    return {
        messages: mergedMessages,
        system: system.length > 0 ? system : undefined // If system is empty, set to undefined
    };
}

/**
 * Sanitize messages by removing empty text blocks.
 * Claude API rejects messages with empty text content blocks ("text content blocks must be non-empty").
 * This handles cases where streaming was interrupted and left empty text blocks.
 *
 * - Filters out empty text blocks from each message's content
 * - Removes messages entirely if they have no content after filtering
 */
function sanitizeMessages(messages: MessageParam[]): MessageParam[] {
    const result: MessageParam[] = [];

    for (const message of messages) {
        if (typeof message.content === 'string') {
            // String content - keep only if non-empty
            if (message.content.trim()) {
                result.push(message);
            }
            continue;
        }

        // Array content - filter out empty text blocks
        const filteredContent = message.content.filter(block => {
            if (block.type === 'text') {
                return block.text && block.text.trim().length > 0;
            }
            // Keep all non-text blocks (tool_use, tool_result, image, etc.)
            return true;
        });

        // Only include message if it has content after filtering
        if (filteredContent.length > 0) {
            result.push({
                ...message,
                content: filteredContent
            });
        }
    }

    return result;
}

/**
 * Fix orphaned tool_use blocks in the conversation.
 * @exported for testing
 *
 * When an agent is stopped mid-tool-execution, the assistant message contains tool_use blocks
 * but no corresponding tool_result was added. The Anthropic API requires that every tool_use
 * must be followed by a tool_result in the next user message.
 *
 * This function detects such cases and injects synthetic tool_result blocks indicating
 * the tools were interrupted, allowing the conversation to continue.
 */
export function fixOrphanedToolUse(messages: MessageParam[]): MessageParam[] {
    if (messages.length < 2) return messages;

    const result: MessageParam[] = [];

    for (let i = 0; i < messages.length; i++) {
        const current = messages[i];
        result.push(current);

        // Check if this is an assistant message with tool_use blocks
        if (current.role === 'assistant' && Array.isArray(current.content)) {
            const toolUseBlocks = current.content.filter(
                (block): block is ContentBlockParam & { type: 'tool_use'; id: string; name: string } =>
                    block.type === 'tool_use'
            );

            if (toolUseBlocks.length > 0) {
                // Check if the next message is a user message with matching tool_results
                const nextMessage = messages[i + 1];

                if (nextMessage && nextMessage.role === 'user' && Array.isArray(nextMessage.content)) {
                    // Get tool_result IDs from the next message
                    const toolResultIds = new Set(
                        nextMessage.content
                            .filter((block): block is ToolResultBlockParam => block.type === 'tool_result')
                            .map(block => block.tool_use_id)
                    );

                    // Find orphaned tool_use blocks (no matching tool_result)
                    const orphanedToolUse = toolUseBlocks.filter(block => !toolResultIds.has(block.id));

                    if (orphanedToolUse.length > 0) {
                        // Inject synthetic tool_results for orphaned tool_use
                        const syntheticResults: ToolResultBlockParam[] = orphanedToolUse.map(block => ({
                            type: 'tool_result',
                            tool_use_id: block.id,
                            content: `[Tool interrupted: The user stopped the operation before "${block.name}" could execute.]`
                        }));

                        // Prepend synthetic results to the next user message
                        const updatedNextMessage: MessageParam = {
                            ...nextMessage,
                            content: [...syntheticResults, ...nextMessage.content]
                        };

                        // Replace the next message in our iteration
                        messages[i + 1] = updatedNextMessage;
                    }
                } else if (nextMessage && nextMessage.role === 'user') {
                    // Next message is a user message but not array content (plain text)
                    // We need to convert it and add tool_results
                    const syntheticResults: ToolResultBlockParam[] = toolUseBlocks.map(block => ({
                        type: 'tool_result',
                        tool_use_id: block.id,
                        content: `[Tool interrupted: The user stopped the operation before "${block.name}" could execute.]`
                    }));

                    const textContent: TextBlockParam = typeof nextMessage.content === 'string'
                        ? { type: 'text', text: nextMessage.content }
                        : { type: 'text', text: '' };

                    const updatedNextMessage: MessageParam = {
                        role: 'user',
                        content: [...syntheticResults, textContent]
                    };

                    messages[i + 1] = updatedNextMessage;
                }
                // Note: If there's no nextMessage, we leave the conversation as-is.
                // The tool_use blocks are expected to be there - the next turn will provide tool_results.
            }
        }
    }

    return result;
}

interface RequestOptions {
    headers?: Record<string, string>;
}

function getClaudePayload(options: ExecutionOptions, prompt: ClaudePrompt): { payload: MessageCreateParamsBase, requestOptions: RequestOptions | undefined } {
    const modelName = options.model; // Model name is already extracted in the calling methods
    const model_options = options.model_options as VertexAIClaudeOptions;

    // Add beta header for Claude 3.7 models to enable 128k output tokens
    let requestOptions: RequestOptions | undefined = undefined;
    if (modelName.includes('claude-3-7-sonnet') &&
        ((model_options?.max_tokens ?? 0) > 64000 || (model_options?.thinking_budget_tokens ?? 0) > 64000)) {
        requestOptions = {
            headers: {
                'anthropic-beta': 'output-128k-2025-02-19'
            }
        };
    }

    // Fix orphaned tool_use blocks (can occur when agent is stopped mid-tool-execution)
    const fixedMessages = fixOrphanedToolUse(prompt.messages);
    // Sanitize messages to remove empty text blocks (can occur from interrupted streaming)
    let sanitizedMessages = sanitizeMessages(fixedMessages);

    // Validate tools have input_schema.type set to 'object' as required by the Anthropic SDK
    if (options.tools) {
        for (const tool of options.tools) {
            if (tool.input_schema.type !== 'object') {
                throw new Error(`Tool "${tool.name}" has invalid input_schema.type: expected "object", got "${tool.input_schema.type}"`);
            }
        }
    }

    // When no tools are provided but conversation contains tool_use/tool_result blocks
    // (e.g. checkpoint summary calls), convert tool blocks to text to avoid API errors
    const hasTools = options.tools && options.tools.length > 0;
    if (!hasTools && claudeMessagesContainToolBlocks(sanitizedMessages)) {
        sanitizedMessages = convertClaudeToolBlocksToText(sanitizedMessages);
    }

    const payload = {
        messages: sanitizedMessages,
        system: prompt.system,
        tools: hasTools ? options.tools as MessageCreateParamsBase['tools'] : undefined,
        temperature: model_options?.temperature,
        model: modelName,
        max_tokens: maxToken(options),
        top_p: model_options?.temperature != null ? undefined : model_options?.top_p,
        top_k: model_options?.top_k,
        stop_sequences: model_options?.stop_sequence,
        thinking: model_options?.thinking_mode ?
            {
                budget_tokens: model_options?.thinking_budget_tokens ?? 1024,
                type: "enabled" as const
            } : {
                type: "disabled" as const
            }
    };

    return { payload, requestOptions };
}

/**
 * Checks whether any Claude message contains tool_use or tool_result content blocks.
 */
function claudeMessagesContainToolBlocks(messages: MessageParam[]): boolean {
    for (const msg of messages) {
        if (!Array.isArray(msg.content)) continue;
        for (const block of msg.content) {
            if (typeof block === 'object' && block !== null && 'type' in block) {
                if (block.type === 'tool_use' || block.type === 'tool_result') return true;
            }
        }
    }
    return false;
}

/**
 * Converts tool_use and tool_result blocks to text in Claude messages.
 * Preserves tool call information while removing structured blocks that
 * require tools to be defined in the API request.
 */
function convertClaudeToolBlocksToText(messages: MessageParam[]): MessageParam[] {
    return messages.map(msg => {
        if (!Array.isArray(msg.content)) return msg;
        let hasToolBlocks = false;
        for (const block of msg.content) {
            if (typeof block === 'object' && block !== null && 'type' in block &&
                (block.type === 'tool_use' || block.type === 'tool_result')) {
                hasToolBlocks = true;
                break;
            }
        }
        if (!hasToolBlocks) return msg;

        const newContent: MessageParam['content'] = [];
        for (const block of msg.content) {
            if (typeof block === 'string') {
                newContent.push(block);
                continue;
            }
            if (block.type === 'tool_use') {
                const inputStr = block.input ? JSON.stringify(block.input) : '';
                const truncated = inputStr.length > 500 ? inputStr.substring(0, 500) + '...' : inputStr;
                (newContent as Array<{ type: 'text'; text: string }>).push({
                    type: 'text',
                    text: `[Tool call: ${block.name}(${truncated})]`,
                });
            } else if (block.type === 'tool_result') {
                let resultStr = 'No content';
                if (typeof block.content === 'string') {
                    resultStr = block.content.length > 500 ? block.content.substring(0, 500) + '...' : block.content;
                } else if (Array.isArray(block.content)) {
                    const texts = block.content
                        .filter((c): c is { type: 'text'; text: string } => c.type === 'text')
                        .map(c => c.text.length > 500 ? c.text.substring(0, 500) + '...' : c.text);
                    resultStr = texts.join('\n') || 'No text content';
                }
                (newContent as Array<{ type: 'text'; text: string }>).push({
                    type: 'text',
                    text: `[Tool result: ${resultStr}]`,
                });
            } else {
                newContent.push(block as any);
            }
        }
        return { ...msg, content: newContent };
    });
}
