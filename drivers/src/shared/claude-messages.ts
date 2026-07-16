/**
 * Shared utilities for Anthropic SDK-based drivers.
 *
 * Used by both the native AnthropicDriver (drivers/src/anthropic/) and the
 * VertexAI Claude pathway (drivers/src/vertexai/models/claude.ts).  Both use
 * the same Anthropic Messages API surface — the only difference is the client
 * (Anthropic vs AnthropicVertex) and how auth is wired up.
 */

import type Anthropic from '@anthropic-ai/sdk';
import {
    AnthropicError,
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    APIUserAbortError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
} from '@anthropic-ai/sdk/error';
import type {
    ContentBlock,
    ContentBlockParam,
    DocumentBlockParam,
    ImageBlockParam,
    Message,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
} from '@anthropic-ai/sdk/resources/index.js';
import type { MessageStreamParams } from '@anthropic-ai/sdk/resources/index.mjs';
import type { MessageCreateParamsBase, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import type AnthropicVertex from '@anthropic-ai/vertex-sdk';
import { getClaudeMaxTokensLimit } from '@llumiverse/common';
import {
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    type DriverCompletionStream,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    getConversationMeta,
    incrementConversationTurn,
    type JSONObject,
    LlumiverseError,
    type LlumiverseErrorContext,
    type Logger,
    PromptRole,
    type PromptSegment,
    readStreamAsBase64,
    readStreamAsString,
    type StatelessExecutionOptions,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    type ToolUse,
    truncateLargeTextInConversation,
} from '@llumiverse/core';
import { asyncMap } from '@llumiverse/core/async';
import { resolveClaudeThinking } from './claude-thinking.js';
import { truncateBinaryForDebug } from './debug-prompt.js';

// ============================================================================
// Types
// ============================================================================

// Conversation text-trim policy (applied only when the caller requests text
// trimming via stripTextMaxTokens). Keep the last N messages fully intact (the
// active working set) and cap large text in older messages at the token ceiling
// below — so long agent conversations don't balloon context.
const KEEP_RECENT_MESSAGES = 12;
const OLD_MESSAGE_TEXT_MAX_TOKENS = 2000;

interface WarnLogger {
    warn: (data: Record<string, unknown>, message: string) => void;
}

export interface ClaudePrompt {
    messages: MessageParam[];
    system?: TextBlockParam[];
}

function formatClaudeContentBlockForDebug(block: ContentBlockParam): ContentBlockParam {
    if (block.type === 'image' && block.source.type === 'base64') {
        return {
            ...block,
            source: {
                ...block.source,
                data: truncateBinaryForDebug(block.source.data),
            },
        };
    }
    if (block.type === 'document' && block.source.type === 'base64') {
        return {
            ...block,
            source: {
                ...block.source,
                data: truncateBinaryForDebug(block.source.data),
            },
        };
    }
    return block;
}

export function formatClaudeDebugPrompt(prompt: ClaudePrompt): ClaudePrompt {
    return {
        ...prompt,
        messages: prompt.messages.map((message) => ({
            ...message,
            content: Array.isArray(message.content)
                ? message.content.map(formatClaudeContentBlockForDebug)
                : message.content,
        })),
    };
}

export interface AnthropicUsageLike {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens?: number | null;
    cache_creation_input_tokens?: number | null;
}

/**
 * Duck-typed options interface accepted by the shared Claude utilities.
 * Both `AnthropicClaudeOptions` and `VertexAIClaudeOptions` satisfy this structurally.
 */
export interface ClaudeBaseOptions {
    _option_id?: string;
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    stop_sequence?: string[];
    effort?: string;
    thinking_budget_tokens?: number;
    include_thoughts?: boolean;
    cache_enabled?: boolean;
    cache_ttl?: string;
}

interface RequestOptions {
    headers?: Record<string, string>;
}

type ClaudeTool = NonNullable<MessageCreateParamsBase['tools']>[number];
type ClaudeMessageStream = AsyncIterable<RawMessageStreamEvent> & {
    finalMessage(): Promise<Message>;
};
type ClaudeMessagesStreamClient = {
    messages: {
        stream(body: MessageStreamParams, options?: RequestOptions): ClaudeMessageStream;
    };
};

function streamClaudeMessages(
    client: Anthropic | AnthropicVertex,
    payload: MessageStreamParams,
    requestOptions: RequestOptions | undefined,
): Promise<ClaudeMessageStream> {
    // AnthropicVertex intentionally wraps the Anthropic Messages API, but it depends on its
    // own @anthropic-ai/sdk copy. Cast at the boundary so the implementation can call the
    // shared runtime-compatible stream API without TS trying to call a union of SDK versions.
    return Promise.resolve((client as unknown as ClaudeMessagesStreamClient).messages.stream(payload, requestOptions));
}

// ============================================================================
// Token usage
// ============================================================================

export function anthropicUsageToTokenUsage(usage: AnthropicUsageLike): ExecutionTokenUsage {
    const cacheRead = usage.cache_read_input_tokens ?? 0;
    const cacheWrite = usage.cache_creation_input_tokens ?? 0;
    return {
        prompt_new: usage.input_tokens,
        prompt: usage.input_tokens + cacheRead + cacheWrite,
        result: usage.output_tokens,
        total: usage.input_tokens + usage.output_tokens + cacheRead + cacheWrite,
        prompt_cached: usage.cache_read_input_tokens ?? undefined,
        prompt_cache_write: usage.cache_creation_input_tokens ?? undefined,
    };
}

// ============================================================================
// Finish reason
// ============================================================================

export function claudeFinishReason(reason: string | undefined): string | undefined {
    if (!reason) return undefined;
    switch (reason) {
        case 'end_turn':
            return 'stop';
        case 'max_tokens':
        case 'model_context_window_exceeded':
            return 'length';
        default:
            return reason; // stop_sequence, tool_use
    }
}

/**
 * Keep Claude's provider-native truncation reason visible after both variants
 * are normalized to `length` for cross-provider control flow.
 */
export function logClaudeTruncation(
    logger: Logger | undefined,
    reason: string | null | undefined,
    context: { provider: string; model: string },
): void {
    if (!logger) return;

    const details = {
        ...context,
        finish_reason: 'length',
        provider_finish_reason: reason,
    };
    if (reason === 'max_tokens') {
        logger.warn(details, '[Claude] Completion stopped at the output token limit');
    } else if (reason === 'model_context_window_exceeded') {
        logger.warn(details, '[Claude] Completion exceeded the model context window');
    }
}

// ============================================================================
// Content extraction
// ============================================================================

export function collectClaudeTools(content: ContentBlock[]): ToolUse[] | undefined {
    const out: ToolUse[] = [];
    for (const block of content) {
        if (block.type === 'tool_use') {
            out.push({
                id: block.id,
                tool_name: block.name,
                tool_input: block.input as JSONObject,
            });
        }
    }
    return out.length > 0 ? out : undefined;
}

export function collectAllTextContent(content: ContentBlock[], includeThoughts = false): string {
    const textParts: string[] = [];

    if (includeThoughts) {
        for (const block of content) {
            if (block.type === 'thinking' && block.thinking) {
                textParts.push(block.thinking);
            } else if (block.type === 'redacted_thinking' && block.data) {
                textParts.push(`[Redacted thinking: ${block.data}]`);
            }
        }
        if (textParts.length > 0) {
            textParts.push('');
        }
    }

    for (const block of content) {
        if (block.type === 'text' && block.text) {
            textParts.push(block.text);
        }
    }

    return textParts.join('\n');
}

// ============================================================================
// Max tokens
// ============================================================================

export function claudeMaxTokens(option: StatelessExecutionOptions): number {
    const modelOptions = option.model_options as ClaudeBaseOptions | undefined;
    if (modelOptions && typeof modelOptions.max_tokens === 'number') {
        return modelOptions.max_tokens;
    }
    let maxSupportedTokens = getClaudeMaxTokensLimit(option.model);
    // Claude 3.7 supports up to 128k with a beta header; default to 64k when no budget is set.
    if (option.model.includes('claude-3-7-sonnet') && (modelOptions?.thinking_budget_tokens ?? 0) < 48000) {
        maxSupportedTokens = 64000;
    }
    return maxSupportedTokens;
}

// ============================================================================
// File / multimodal block helpers
// ============================================================================

async function collectFileBlocks(
    segment: PromptSegment,
    logger?: WarnLogger,
): Promise<Array<TextBlockParam | ImageBlockParam | DocumentBlockParam>> {
    const contentBlocks: Array<TextBlockParam | ImageBlockParam | DocumentBlockParam> = [];

    for (const file of segment.files || []) {
        if (file.mime_type?.startsWith('image/')) {
            const allowedTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
            if (!allowedTypes.includes(file.mime_type)) {
                throw new Error(`Unsupported image type: ${file.mime_type}`);
            }
            const mimeType = String(file.mime_type) as 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';
            contentBlocks.push({
                type: 'image',
                source: {
                    type: 'base64',
                    data: await readStreamAsBase64(await file.getStream()),
                    media_type: mimeType,
                },
            } satisfies ImageBlockParam);
        } else if (file.mime_type?.startsWith('video/')) {
            logger?.warn(
                {
                    file_name: file.name,
                    mime_type: file.mime_type,
                },
                '[Claude] Skipping unsupported video attachment',
            );
        } else if (file.mime_type === 'application/pdf') {
            contentBlocks.push({
                title: file.name,
                type: 'document',
                source: {
                    type: 'base64',
                    data: await readStreamAsBase64(await file.getStream()),
                    media_type: 'application/pdf',
                },
            } satisfies DocumentBlockParam);
        } else if (file.mime_type?.startsWith('text/')) {
            contentBlocks.push({
                title: file.name,
                type: 'document',
                source: {
                    type: 'text',
                    data: await readStreamAsString(await file.getStream()),
                    media_type: 'text/plain',
                },
            } satisfies DocumentBlockParam);
        }
    }

    return contentBlocks;
}

// ============================================================================
// Prompt formatting (PromptSegment[] → ClaudePrompt)
// ============================================================================

export async function formatClaudePrompt(
    segments: PromptSegment[],
    options: ExecutionOptions,
    logger?: WarnLogger,
): Promise<ClaudePrompt> {
    let system: TextBlockParam[] | undefined = segments
        .filter((s) => s.role === PromptRole.system)
        .map((s) => ({ text: s.content, type: 'text' as const }));

    if (options.result_schema) {
        const schemaText =
            options.tools && options.tools.length > 0
                ? `When not calling tools, the answer must be a JSON object using the following JSON Schema:\n${JSON.stringify(options.result_schema)}`
                : `The answer must be a JSON object using the following JSON Schema:\n${JSON.stringify(options.result_schema)}`;
        system.push({ text: schemaText, type: 'text' as const });
    }

    let messages: MessageParam[] = [];
    const safetyMessages: MessageParam[] = [];

    for (const segment of segments) {
        if (segment.role === PromptRole.system) continue;

        if (segment.role === PromptRole.tool) {
            if (!segment.tool_use_id) {
                throw new Error('Tool prompt segment must have a tool use ID');
            }
            const contentBlocks: Array<TextBlockParam | ImageBlockParam | DocumentBlockParam> = [];
            if (segment.content) {
                contentBlocks.push({ type: 'text', text: segment.content } satisfies TextBlockParam);
            }
            contentBlocks.push(...(await collectFileBlocks(segment, logger)));
            messages.push({
                role: 'user',
                content: [
                    {
                        type: 'tool_result',
                        tool_use_id: segment.tool_use_id,
                        content: contentBlocks,
                    } satisfies ToolResultBlockParam,
                ],
            });
        } else {
            const contentBlocks: ContentBlockParam[] = [];
            if (segment.content) {
                contentBlocks.push({ type: 'text', text: segment.content } satisfies TextBlockParam);
            }
            contentBlocks.push(...(await collectFileBlocks(segment, logger)));
            if (contentBlocks.length === 0) continue;

            const messageParam: MessageParam = {
                role: segment.role === PromptRole.assistant ? 'assistant' : 'user',
                content: contentBlocks,
            };

            if (segment.role === PromptRole.safety) {
                safetyMessages.push(messageParam);
            } else {
                messages.push(messageParam);
            }
        }
    }

    messages = messages.concat(safetyMessages);
    if (system && system.length === 0) system = undefined;

    return { messages, system };
}

// ============================================================================
// Conversation management
// ============================================================================

export function createPromptFromResponse(response: Message): ClaudePrompt {
    return {
        messages: [{ role: response.role, content: response.content }],
        system: undefined,
    };
}

export function mergeConsecutiveUserMessages(messages: MessageParam[]): MessageParam[] {
    if (messages.length === 0) return [];

    const needsMerging = messages.some(
        (msg, i) => i < messages.length - 1 && msg.role === 'user' && messages[i + 1].role === 'user',
    );
    if (!needsMerging) return messages;

    const result: MessageParam[] = [];
    let i = 0;
    while (i < messages.length) {
        const current = messages[i];
        if (current.role === 'user') {
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
            result.push({ role: 'user', content: mergedContent });
        } else {
            result.push(current);
            i++;
        }
    }
    return result;
}

export function sanitizeMessages(messages: MessageParam[]): MessageParam[] {
    const result: MessageParam[] = [];
    for (const message of messages) {
        if (typeof message.content === 'string') {
            if (message.content.trim()) result.push(message);
            continue;
        }
        const filteredContent = message.content.filter((block) => {
            if (block.type === 'text') return block.text && block.text.trim().length > 0;
            return true;
        });
        if (filteredContent.length > 0) {
            result.push({ ...message, content: filteredContent });
        }
    }
    return result;
}

export function fixOrphanedToolUse(messages: MessageParam[]): MessageParam[] {
    if (messages.length < 2) return messages;
    const result: MessageParam[] = [];
    for (let i = 0; i < messages.length; i++) {
        const current = messages[i];
        result.push(current);

        if (current.role === 'assistant' && Array.isArray(current.content)) {
            const toolUseBlocks = current.content.filter(
                (block): block is ContentBlockParam & { type: 'tool_use'; id: string; name: string } =>
                    block.type === 'tool_use',
            );

            if (toolUseBlocks.length > 0) {
                const nextMessage = messages[i + 1];

                if (nextMessage && nextMessage.role === 'user' && Array.isArray(nextMessage.content)) {
                    const toolResultIds = new Set(
                        nextMessage.content
                            .filter((block): block is ToolResultBlockParam => block.type === 'tool_result')
                            .map((block) => block.tool_use_id),
                    );
                    const orphaned = toolUseBlocks.filter((block) => !toolResultIds.has(block.id));
                    if (orphaned.length > 0) {
                        const syntheticResults: ToolResultBlockParam[] = orphaned.map((block) => ({
                            type: 'tool_result',
                            tool_use_id: block.id,
                            content: `[Tool interrupted: The user stopped the operation before "${block.name}" could execute.]`,
                        }));
                        messages[i + 1] = { ...nextMessage, content: [...syntheticResults, ...nextMessage.content] };
                    }
                } else if (nextMessage && nextMessage.role === 'user') {
                    const syntheticResults: ToolResultBlockParam[] = toolUseBlocks.map((block) => ({
                        type: 'tool_result',
                        tool_use_id: block.id,
                        content: `[Tool interrupted: The user stopped the operation before "${block.name}" could execute.]`,
                    }));
                    const textContent: TextBlockParam =
                        typeof nextMessage.content === 'string'
                            ? { type: 'text', text: nextMessage.content }
                            : { type: 'text', text: '' };
                    messages[i + 1] = { role: 'user', content: [...syntheticResults, textContent] };
                }
            }
        }
    }
    return result;
}

/**
 * Drop tool_result blocks whose tool_use_id has no matching tool_use in the
 * immediately preceding assistant message. This is the mirror of
 * {@link fixOrphanedToolUse}: that function synthesizes results for tool_uses
 * left unanswered (e.g. a cancelled run); this one removes results left dangling
 * after their tool_use was dropped (e.g. by conversation compaction/trimming, or
 * a parallel tool batch whose results were split across messages that cannot be
 * re-paired).
 *
 * Without this, the Anthropic / Vertex-Anthropic API rejects the request with a
 * non-retryable 400: "unexpected `tool_use_id` found in `tool_result` blocks ...
 * Each `tool_result` block must have a corresponding `tool_use` block in the
 * previous message." — which terminates the conversation.
 *
 * Must run AFTER mergeConsecutiveUserMessages so that parallel tool results which
 * were split across separate user messages are first combined into the single
 * user turn that follows the assistant tool_use message (otherwise valid results
 * whose "previous message" is another user message would be wrongly dropped).
 */
export function fixOrphanedToolResults(messages: MessageParam[]): MessageParam[] {
    if (messages.length === 0) return messages;
    const result: MessageParam[] = [];
    for (let i = 0; i < messages.length; i++) {
        const message = messages[i];
        if (message.role !== 'user' || !Array.isArray(message.content)) {
            result.push(message);
            continue;
        }
        const hasToolResult = message.content.some((block) => block.type === 'tool_result');
        if (!hasToolResult) {
            result.push(message);
            continue;
        }
        // Collect the tool_use ids declared by the immediately preceding assistant message.
        const prev = messages[i - 1];
        const allowedIds = new Set<string>();
        if (prev && prev.role === 'assistant' && Array.isArray(prev.content)) {
            for (const block of prev.content) {
                if (block.type === 'tool_use') allowedIds.add(block.id);
            }
        }
        const filtered = message.content.filter((block) =>
            block.type === 'tool_result' ? allowedIds.has(block.tool_use_id) : true,
        );
        // If every block was an orphaned tool_result, drop the message rather than
        // emit an empty (and invalid) content array.
        if (filtered.length === 0) continue;
        result.push(filtered.length === message.content.length ? message : { ...message, content: filtered });
    }
    return result;
}

export function updateClaudeConversation(
    conversation: ClaudePrompt | undefined | null,
    prompt: ClaudePrompt,
): ClaudePrompt {
    const baseSystemMessages = conversation?.system || [];
    const baseMessages = conversation?.messages || [];
    const system = baseSystemMessages.concat(prompt.system || []);
    const combined = sanitizeMessages(baseMessages.concat(prompt.messages || []));
    const mergedMessages = mergeConsecutiveUserMessages(combined);
    return {
        messages: mergedMessages,
        system: system.length > 0 ? system : undefined,
    };
}

export function claudeMessagesContainToolBlocks(messages: MessageParam[]): boolean {
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

export function convertClaudeToolBlocksToText(messages: MessageParam[]): MessageParam[] {
    return messages.map((msg) => {
        if (!Array.isArray(msg.content)) return msg;
        let hasToolBlocks = false;
        for (const block of msg.content) {
            if (
                typeof block === 'object' &&
                block !== null &&
                'type' in block &&
                (block.type === 'tool_use' || block.type === 'tool_result')
            ) {
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
                const truncated = inputStr.length > 500 ? `${inputStr.substring(0, 500)}...` : inputStr;
                (newContent as Array<{ type: 'text'; text: string }>).push({
                    type: 'text',
                    text: `[Tool call: ${block.name}(${truncated})]`,
                });
            } else if (block.type === 'tool_result') {
                let resultStr = 'No content';
                if (typeof block.content === 'string') {
                    resultStr = block.content.length > 500 ? `${block.content.substring(0, 500)}...` : block.content;
                } else if (Array.isArray(block.content)) {
                    const texts = block.content
                        .filter((c): c is { type: 'text'; text: string } => c.type === 'text')
                        .map((c) => (c.text.length > 500 ? `${c.text.substring(0, 500)}...` : c.text));
                    resultStr = texts.join('\n') || 'No text content';
                }
                (newContent as Array<{ type: 'text'; text: string }>).push({
                    type: 'text',
                    text: `[Tool result: ${resultStr}]`,
                });
            } else {
                newContent.push(block as ContentBlockParam);
            }
        }
        return { ...msg, content: newContent };
    });
}

// ============================================================================
// Cache control stripping
// ============================================================================

function stripClaudeCacheControlFromBlock<T extends ContentBlockParam>(block: T): T {
    if (typeof block === 'object' && block !== null && 'cache_control' in block) {
        const { cache_control: _cc, ...rest } = block as T & { cache_control: unknown };
        return rest as T;
    }
    return block;
}

function stripClaudeCacheControlFromMessages(messages: MessageParam[]): MessageParam[] {
    return messages.map((msg) => {
        if (!Array.isArray(msg.content)) return msg;
        return { ...msg, content: msg.content.map(stripClaudeCacheControlFromBlock) };
    });
}

function stripClaudeCacheControlFromSystem(system?: TextBlockParam[]): TextBlockParam[] | undefined {
    if (!system) return undefined;
    return system.map(stripClaudeCacheControlFromBlock);
}

function stripClaudeCacheControlFromTools(
    tools?: MessageCreateParamsBase['tools'],
): MessageCreateParamsBase['tools'] | undefined {
    if (!tools) return undefined;
    return tools.map((tool) => {
        if ('cache_control' in tool) {
            const { cache_control: _cc, ...rest } = tool as ClaudeTool & { cache_control: unknown };
            return rest as ClaudeTool;
        }
        return tool;
    });
}

// ============================================================================
// Payload builder
// ============================================================================

export function getClaudePayload(
    options: ExecutionOptions,
    prompt: ClaudePrompt,
): { payload: MessageCreateParamsBase; requestOptions: RequestOptions | undefined } {
    const modelName = options.model;
    const model_options = options.model_options as ClaudeBaseOptions | undefined;

    let requestOptions: RequestOptions | undefined;
    if (
        modelName.includes('claude-3-7-sonnet') &&
        ((model_options?.max_tokens ?? 0) > 64000 || (model_options?.thinking_budget_tokens ?? 0) > 64000)
    ) {
        requestOptions = { headers: { 'anthropic-beta': 'output-128k-2025-02-19' } };
    }

    // Merge first so parallel tool results split across user messages are recombined
    // into the single user turn after their assistant tool_use message; then fix both
    // orphan directions (tool_use without result, and result without tool_use) so the
    // request can never trip Anthropic/Vertex's tool_use/tool_result pairing validation.
    const mergedMessages = mergeConsecutiveUserMessages(prompt.messages);
    const fixedMessages = fixOrphanedToolResults(fixOrphanedToolUse(mergedMessages));
    let sanitizedMessages = sanitizeMessages(fixedMessages);

    if (options.tools) {
        for (const tool of options.tools) {
            if (tool.input_schema.type !== 'object') {
                throw new Error(
                    `Tool "${tool.name}" has invalid input_schema.type: expected "object", got "${tool.input_schema.type}"`,
                );
            }
        }
    }

    const hasTools = options.tools && options.tools.length > 0;
    if (!hasTools && claudeMessagesContainToolBlocks(sanitizedMessages)) {
        sanitizedMessages = convertClaudeToolBlocksToText(sanitizedMessages);
    }

    sanitizedMessages = stripClaudeCacheControlFromMessages(sanitizedMessages);
    const sanitizedSystem = stripClaudeCacheControlFromSystem(prompt.system);
    const sanitizedTools = hasTools
        ? stripClaudeCacheControlFromTools(options.tools as MessageCreateParamsBase['tools'])
        : undefined;

    const cacheEnabled = model_options?.cache_enabled === true;
    if (cacheEnabled) {
        const cacheTtl = model_options?.cache_ttl as '5m' | '1h' | undefined;
        const cacheControl = { type: 'ephemeral' as const, ...(cacheTtl && { ttl: cacheTtl }) };

        if (sanitizedSystem && sanitizedSystem.length > 0) {
            const lastBlock = sanitizedSystem[sanitizedSystem.length - 1] as TextBlockParam & {
                cache_control?: unknown;
            };
            lastBlock.cache_control = cacheControl;
        }
        if (sanitizedTools && sanitizedTools.length > 0) {
            const lastTool = sanitizedTools[sanitizedTools.length - 1] as ClaudeTool & { cache_control?: unknown };
            lastTool.cache_control = cacheControl;
        }
        if (sanitizedMessages.length >= 4) {
            const pivotMsg = sanitizedMessages[sanitizedMessages.length - 2];
            if (Array.isArray(pivotMsg.content) && pivotMsg.content.length > 0) {
                const lastBlock = pivotMsg.content[pivotMsg.content.length - 1];
                if (
                    typeof lastBlock === 'object' &&
                    lastBlock !== null &&
                    'type' in lastBlock &&
                    lastBlock.type !== 'thinking' &&
                    lastBlock.type !== 'redacted_thinking'
                ) {
                    (lastBlock as TextBlockParam).cache_control = cacheControl;
                }
            }
        }
    }

    const { thinking, outputConfig, hasSamplingRestriction } = resolveClaudeThinking(
        modelName,
        model_options as Parameters<typeof resolveClaudeThinking>[1],
    );

    const payload: MessageCreateParamsBase = {
        messages: sanitizedMessages,
        system: sanitizedSystem,
        tools: sanitizedTools,
        temperature: hasSamplingRestriction ? undefined : model_options?.temperature,
        model: modelName,
        max_tokens: claudeMaxTokens(options),
        top_p: hasSamplingRestriction
            ? undefined
            : model_options?.temperature != null
              ? undefined
              : model_options?.top_p,
        top_k: hasSamplingRestriction ? undefined : model_options?.top_k,
        stop_sequences: model_options?.stop_sequence,
        thinking,
        stream: true,
        ...(outputConfig && { output_config: outputConfig }),
    };

    return { payload, requestOptions };
}

// ============================================================================
// Streaming conversation builder (called after stream completes)
// ============================================================================

export function buildClaudeStreamingConversation(
    prompt: ClaudePrompt,
    result: unknown[],
    toolUse: unknown[] | undefined,
    options: ExecutionOptions,
): ClaudePrompt {
    const completionResults = result as CompletionResult[];
    const text = completionResults
        .filter((r) => r.type === 'text')
        .map((r) => r.value as string)
        .join('');

    let conversation = updateClaudeConversation(options.conversation as ClaudePrompt | undefined, prompt);

    if (text) {
        const assistantMsg: MessageParam = { role: 'assistant', content: text };
        conversation = updateClaudeConversation(conversation, { messages: [assistantMsg] });
    }

    if (toolUse && toolUse.length > 0) {
        const toolBlocks: ContentBlockParam[] = (toolUse as ToolUse[]).map((t) => ({
            type: 'tool_use' as const,
            id: t.id,
            name: t.tool_name,
            input: t.tool_input ?? {},
        }));
        const assistantToolMsg: MessageParam = { role: 'assistant', content: toolBlocks };
        conversation = updateClaudeConversation(conversation, { messages: [assistantToolMsg] });
    }

    conversation = incrementConversationTurn(conversation) as ClaudePrompt;
    const currentTurn = getConversationMeta(conversation).turnNumber;
    // When text trimming is requested, keep the agent's active working set (the
    // most recent messages — the file it just read/wrote, latest diagnostics)
    // fully intact, and aggressively shrink large text only in OLDER messages.
    // Clamp the effective cap so a lax caller value (e.g. 10000) still bites on
    // stale blocks; long agent conversations otherwise balloon context and
    // trigger frequent expensive checkpoints.
    const requestedTextMax = options.stripTextMaxTokens;
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: requestedTextMax ? Math.min(requestedTextMax, OLD_MESSAGE_TEXT_MAX_TOKENS) : undefined,
        keepRecentMessages: requestedTextMax ? KEEP_RECENT_MESSAGES : undefined,
    };
    let processed = stripBase64ImagesFromConversation(conversation, stripOptions);
    processed = truncateLargeTextInConversation(processed, stripOptions);
    processed = stripHeartbeatsFromConversation(processed, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
    });
    return processed as ClaudePrompt;
}

function finalizeClaudeConversation(
    conversation: ClaudePrompt,
    result: Message,
    options: ExecutionOptions,
): ClaudePrompt {
    let completedConversation = updateClaudeConversation(conversation, createPromptFromResponse(result));
    completedConversation = incrementConversationTurn(completedConversation) as ClaudePrompt;
    completedConversation = pruneClaudeThinking(completedConversation);
    const currentTurn = getConversationMeta(completedConversation).turnNumber;
    const activeTurnStart = findClaudeActiveTurnStart(completedConversation);
    const protectedMessages = new Set(
        activeTurnStart >= 0 ? completedConversation.messages.slice(activeTurnStart) : [],
    );
    const preserveSubtree = (value: unknown): boolean => {
        if (!value || typeof value !== 'object') return false;
        if (protectedMessages.has(value as MessageParam)) return true;
        const type = (value as { type?: unknown }).type;
        return type === 'thinking' || type === 'redacted_thinking';
    };
    const stripOpts = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: options.stripTextMaxTokens,
        preserveSubtree,
    };
    let processedConversation = stripBase64ImagesFromConversation(completedConversation, stripOpts);
    processedConversation = truncateLargeTextInConversation(processedConversation, stripOpts);
    processedConversation = stripHeartbeatsFromConversation(processedConversation, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
        preserveSubtree,
    });
    return processedConversation as ClaudePrompt;
}

export function pruneClaudeThinking(conversation: ClaudePrompt): ClaudePrompt {
    const activeTurnStart = findClaudeActiveTurnStart(conversation);
    let latestAssistantIndex = -1;
    for (let index = conversation.messages.length - 1; index >= 0; index--) {
        const message = conversation.messages[index];
        if (message.role === 'assistant') {
            latestAssistantIndex = index;
            break;
        }
    }
    const preserveFrom = activeTurnStart >= 0 ? activeTurnStart : latestAssistantIndex;

    const messages = conversation.messages.map((message, index): MessageParam => {
        if (
            (latestAssistantIndex >= 0 && index >= preserveFrom && index <= latestAssistantIndex) ||
            message.role !== 'assistant' ||
            !Array.isArray(message.content)
        ) {
            return message;
        }
        const content = message.content.filter(
            (block) => block.type !== 'thinking' && block.type !== 'redacted_thinking',
        );
        return content.length === message.content.length ? message : { ...message, content };
    });
    return { ...conversation, messages };
}

function findClaudeActiveTurnStart(conversation: ClaudePrompt): number {
    let activeAssistantIndex = -1;
    for (let index = conversation.messages.length - 1; index >= 0; index--) {
        const message = conversation.messages[index];
        if (message.role !== 'assistant') continue;
        if (Array.isArray(message.content) && message.content.some((block) => block.type === 'tool_use')) {
            activeAssistantIndex = index;
        }
        break;
    }
    if (activeAssistantIndex < 0) return -1;

    for (let index = activeAssistantIndex - 1; index >= 0; index--) {
        const message = conversation.messages[index];
        if (message.role !== 'user') continue;
        const isToolResult =
            Array.isArray(message.content) && message.content.some((block) => block.type === 'tool_result');
        if (!isToolResult) return index + 1;
    }
    return 0;
}

// ============================================================================
// Execution helpers (standalone, take a client parameter)
// ============================================================================

/**
 * Execute a non-streaming Claude completion.
 * Works with any Anthropic-compatible client (Anthropic or AnthropicVertex).
 */
export async function executeClaudeCompletion(
    client: Anthropic | AnthropicVertex,
    prompt: ClaudePrompt,
    options: ExecutionOptions,
    logger?: Logger,
    provider = 'anthropic',
): Promise<Completion> {
    const model_options = options.model_options as ClaudeBaseOptions | undefined;

    const conversation = updateClaudeConversation(options.conversation as ClaudePrompt | undefined, prompt);

    const { payload, requestOptions } = getClaudePayload(options, conversation);

    const responseStream = await streamClaudeMessages(client, payload as MessageStreamParams, requestOptions);
    const result = await responseStream.finalMessage();
    logClaudeTruncation(logger, result.stop_reason, { provider, model: options.model });

    const includeThoughts = model_options?.include_thoughts ?? false;
    const text = collectAllTextContent(result.content, includeThoughts);
    const tool_use = collectClaudeTools(result.content);

    const processedConversation = finalizeClaudeConversation(conversation, result, options);

    return {
        result: text ? [{ type: 'text', value: text }] : [{ type: 'text', value: '' }],
        tool_use,
        token_usage: anthropicUsageToTokenUsage(result.usage),
        finish_reason: tool_use ? 'tool_use' : claudeFinishReason(result?.stop_reason ?? ''),
        conversation: processedConversation,
    };
}

/**
 * Execute a streaming Claude completion.
 * Works with any Anthropic-compatible client (Anthropic or AnthropicVertex).
 */
export async function streamClaudeCompletion(
    client: Anthropic | AnthropicVertex,
    prompt: ClaudePrompt,
    options: ExecutionOptions,
    logger?: Logger,
    provider = 'anthropic',
): Promise<DriverCompletionStream> {
    const model_options = options.model_options as ClaudeBaseOptions | undefined;
    const conversation = updateClaudeConversation(options.conversation as ClaudePrompt | undefined, prompt);

    const { payload, requestOptions } = getClaudePayload(options, conversation);
    const streamingPayload: MessageStreamParams = { ...payload, stream: true };

    const response_stream = await streamClaudeMessages(client, streamingPayload, requestOptions);

    let currentToolUse: { id: string; name: string; inputJson: string } | null = null;
    let pendingSpacing = false;

    const stream = asyncMap(response_stream, async (streamEvent: RawMessageStreamEvent) => {
        switch (streamEvent.type) {
            case 'message_start':
                return {
                    result: [{ type: 'text', value: '' }],
                    token_usage: anthropicUsageToTokenUsage(streamEvent.message.usage as AnthropicUsageLike),
                } satisfies CompletionChunkObject;
            case 'message_delta':
                logClaudeTruncation(logger, streamEvent.delta.stop_reason, { provider, model: options.model });
                return {
                    result: [{ type: 'text', value: '' }],
                    token_usage: { result: streamEvent.usage.output_tokens },
                    finish_reason: claudeFinishReason(streamEvent.delta.stop_reason ?? undefined),
                } satisfies CompletionChunkObject;
            case 'content_block_start':
                if (streamEvent.content_block.type === 'tool_use') {
                    currentToolUse = {
                        id: streamEvent.content_block.id,
                        name: streamEvent.content_block.name,
                        inputJson: '',
                    };
                    return {
                        result: [],
                        tool_use: [
                            {
                                id: streamEvent.content_block.id,
                                tool_name: streamEvent.content_block.name,
                                tool_input: '' as unknown as JSONObject,
                            },
                        ],
                    } satisfies CompletionChunkObject;
                }
                if (streamEvent.content_block.type === 'redacted_thinking' && model_options?.include_thoughts) {
                    return {
                        result: [{ type: 'text', value: `[Redacted thinking: ${streamEvent.content_block.data}]` }],
                    } satisfies CompletionChunkObject;
                }
                break;
            case 'content_block_delta':
                switch (streamEvent.delta.type) {
                    case 'text_delta': {
                        const prefix = pendingSpacing ? '\n\n' : '';
                        pendingSpacing = false;
                        return {
                            result: streamEvent.delta.text
                                ? [{ type: 'text', value: prefix + streamEvent.delta.text }]
                                : [],
                        } satisfies CompletionChunkObject;
                    }
                    case 'input_json_delta':
                        if (currentToolUse && streamEvent.delta.partial_json) {
                            return {
                                result: [],
                                tool_use: [
                                    {
                                        id: currentToolUse.id,
                                        tool_name: '',
                                        tool_input: streamEvent.delta.partial_json as unknown as JSONObject,
                                    },
                                ],
                            } satisfies CompletionChunkObject;
                        }
                        break;
                    case 'thinking_delta':
                        if (model_options?.include_thoughts) {
                            return {
                                result: streamEvent.delta.thinking
                                    ? [{ type: 'text', value: streamEvent.delta.thinking }]
                                    : [],
                            } satisfies CompletionChunkObject;
                        }
                        break;
                    case 'signature_delta':
                        if (model_options?.include_thoughts) {
                            pendingSpacing = true;
                        }
                        break;
                }
                break;
            case 'content_block_stop':
                if (currentToolUse) {
                    currentToolUse = null;
                    pendingSpacing = false;
                }
                break;
        }

        return { result: [] } satisfies CompletionChunkObject;
    });

    return {
        [Symbol.asyncIterator]: () => stream[Symbol.asyncIterator](),
        finalizeConversation: async () => {
            const finalMessage = await response_stream.finalMessage();
            return finalizeClaudeConversation(conversation, finalMessage, options);
        },
    };
}

// ============================================================================
// Error handling
// ============================================================================

export function formatAnthropicLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
    if (error instanceof AnthropicError && !(error instanceof APIError)) {
        // Client-side SDK error (e.g. "Streaming is required for operations that may take longer than 10 minutes").
        // These are structural/configuration errors — retrying will never succeed.
        const errorName = error.constructor?.name || 'AnthropicError';
        return new LlumiverseError(
            `[${context.provider}] ${error.message}`,
            false,
            context,
            error,
            undefined,
            errorName,
        );
    }
    if (!(error instanceof APIError)) {
        // Not an Anthropic error — rethrow for default handling
        throw error;
    }

    const apiError = error as APIError;
    const httpStatusCode = apiError.status;
    let message = apiError.message || String(error);
    let errorType: string | undefined;

    if (apiError.error && typeof apiError.error === 'object') {
        const nested = apiError.error as Record<string, unknown>;
        if (nested.error && typeof nested.error === 'object') {
            const innerError = nested.error as Record<string, unknown>;
            errorType = innerError.type as string | undefined;
            if (typeof innerError.message === 'string') {
                message = innerError.message;
            }
        }
    }

    let userMessage = message;
    if (httpStatusCode) userMessage = `[${httpStatusCode}] ${userMessage}`;
    if (errorType && errorType !== 'error') userMessage = `${errorType}: ${userMessage}`;
    if (apiError.requestID) userMessage += ` (Request ID: ${apiError.requestID})`;

    const retryable = isClaudeErrorRetryable(error, httpStatusCode, errorType, apiError.headers ?? undefined);
    const errorName = error.constructor?.name || 'AnthropicError';

    return new LlumiverseError(
        `[${context.provider}] ${userMessage}`,
        retryable,
        context,
        error,
        httpStatusCode,
        errorName,
    );
}

export function isClaudeErrorRetryable(
    error: unknown,
    httpStatusCode: number | undefined,
    errorType: string | undefined,
    headers?: Headers | undefined,
): boolean | undefined {
    // Honour the server's explicit retry directive first (mirrors SDK shouldRetry logic).
    const shouldRetryHeader = headers?.get('x-should-retry');
    if (shouldRetryHeader === 'true') return true;
    if (shouldRetryHeader === 'false') return false;

    if (error instanceof APIUserAbortError) return false;
    if (error instanceof RateLimitError) return true;
    if (error instanceof InternalServerError) return true;
    if (error instanceof APIConnectionTimeoutError) return true;
    if (error instanceof BadRequestError) return false;
    if (error instanceof AuthenticationError) return false;
    if (error instanceof PermissionDeniedError) return false;
    if (error instanceof NotFoundError) return false;
    if (error instanceof ConflictError) return true; // SDK retries 409 (lock timeouts)
    if (error instanceof UnprocessableEntityError) return false;
    if (errorType === 'invalid_request_error') return false;
    if (httpStatusCode !== undefined) {
        if (httpStatusCode === 429 || httpStatusCode === 408 || httpStatusCode === 529) return true;
        if (httpStatusCode >= 500 && httpStatusCode < 600) return true;
        if (httpStatusCode >= 400 && httpStatusCode < 500) return false;
    }
    if (error instanceof APIConnectionError && !(error instanceof APIConnectionTimeoutError)) return true;
    return undefined;
}
