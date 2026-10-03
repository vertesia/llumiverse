/**
 * Shared utilities for Anthropic SDK-based drivers.
 *
 * Used by the native Anthropic driver, Vertex AI Claude, and Bedrock Mantle
 * Claude pathways. All use the same Anthropic Messages API surface; only the
 * client and authentication wiring differ.
 */

import type { AnthropicBedrockMantle } from '@anthropic-ai/bedrock-sdk';
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
import {
    AGENT_PROMPT_CACHE_KEY_PREFIX,
    getClaudeMaxTokensLimit,
    JSON_SCHEMA_INSTRUCTION_PREFIX,
    TOOL_AWARE_JSON_SCHEMA_INSTRUCTION_PREFIX,
} from '@llumiverse/common';
import {
    type ToolDefinition as CanonicalToolDefinition,
    canonicalJsonContentString,
    createStructuredOutputTransformationProof,
    type DecodedConversationResponse,
    type JsonObject,
    type JsonValue,
    type NativeStreamPosition,
    toolArgumentsForModel,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalHostCapabilities,
    type CanonicalStreamOpenOptions,
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    createCanonicalExecutionResponse,
    type DriverCompletionStream,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    FallbackCanonicalExecutionEventStream,
    getConversationMeta,
    incrementConversationTurn,
    isClaudeVersionGTE,
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
    type ToolDefinition,
    type ToolUse,
    truncateLargeTextInConversation,
} from '@llumiverse/core';
import { asyncMap } from '@llumiverse/core/async';
import { canonicalNativeExecutionEventStream } from '../conversation/canonical-execution-event-stream.js';
import {
    assertAcceptedCanonicalRequest,
    CANONICAL_TOOL_SELECTION_TARGET_OPTION,
    canonicalConversationTurnNumber,
    canonicalToolSelectionTargetOptions,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
} from '../conversation/canonical-runtime.js';
import {
    normalizeDecodedStructuredOutputForSchema,
    rejectDecodedStructuredOutput,
} from '../conversation/structured-output.js';
import {
    appendClaudeCanonicalResponseWithProcessing,
    type CanonicalClaudeToolResultBlockParam,
    CLAUDE_MESSAGES_PROTOCOL,
    decodeClaudeCanonicalResponse,
    finalizeClaudePreparedRequest,
    type PreparedClaudeConversation,
    prepareClaudeCanonicalContext,
    prepareClaudeCanonicalState,
} from './claude-messages-conversation-adapter.js';
import { claudeFinishReason, logClaudeTruncation } from './claude-stop-reason.js';
import { type ClaudeThinkingInput, resolveClaudeThinking } from './claude-thinking.js';
import { truncateBinaryForDebug } from './debug-prompt.js';
import { createToolChoiceConfigurationError } from './tool-choice-error.js';

// ============================================================================
// Types
// ============================================================================

// Conversation text-trim policy (applied only when the caller requests text
// trimming via stripTextMaxTokens). Keep the last N messages fully intact (the
// active working set) and cap large text in older messages at the token ceiling
// below — so long agent conversations don't balloon context.
const KEEP_RECENT_MESSAGES = 12;
const OLD_MESSAGE_TEXT_MAX_TOKENS = 2000;
const AGENT_MESSAGE_CACHE_BLOCK_INTERVAL = 12;
const MAX_CLAUDE_CACHE_BREAKPOINTS = 4;
const CLAUDE_FAST_MODE_BETA = 'fast-mode-2026-02-01';
const RESULT_SCHEMA_INSTRUCTION_PREFIXES = [
    TOOL_AWARE_JSON_SCHEMA_INSTRUCTION_PREFIX,
    JSON_SCHEMA_INSTRUCTION_PREFIX,
] as const;

export interface ClaudeTransportIdentity {
    model: string;
    target_options?: JsonObject;
}

type ClaudeToolDefinition = Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'> | ToolDefinition;

export function isClaudePromptCacheEnabled(options: ExecutionOptions): boolean {
    const modelOptions = options.model_options as ClaudeBaseOptions | undefined;
    return options.prompt_cache_key !== undefined || modelOptions?.cache_enabled === true;
}

function isAgentPromptCacheKey(promptCacheKey: string | undefined): boolean {
    return promptCacheKey?.startsWith(AGENT_PROMPT_CACHE_KEY_PREFIX) === true;
}

function isResultSchemaSystemBlock(block: TextBlockParam): boolean {
    return block.type === 'text' && RESULT_SCHEMA_INSTRUCTION_PREFIXES.some((prefix) => block.text.startsWith(prefix));
}

function claudeResultSchemaInstruction(options: ExecutionOptions, hasTools: boolean): string | undefined {
    if (options.result_schema === undefined) return undefined;
    return hasTools
        ? `${TOOL_AWARE_JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(options.result_schema)}`
        : `${JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(options.result_schema)}`;
}

/** Add request-local schema guidance without changing the retained canonical document. */
export function projectClaudeContextResultSchema(
    conversation: ClaudePrompt,
    options: ExecutionOptions,
    hasTools: boolean,
): ClaudePrompt {
    const schemaText = claudeResultSchemaInstruction(options, hasTools);
    if (schemaText === undefined) return conversation;
    return {
        ...conversation,
        system: [...(conversation.system ?? []), { text: schemaText, type: 'text' }],
    };
}

function mergeClaudeSystemBlocks(base: TextBlockParam[], additions: TextBlockParam[]): TextBlockParam[] {
    if (!additions.some(isResultSchemaSystemBlock)) return base.concat(additions);

    // Agent tool continuations carry result_schema on every request. Replacing
    // the prior generated schema block keeps the system prefix byte-stable and
    // also repairs conversations persisted by older workers, which accumulated
    // one identical ~3.5 KB schema block per tool iteration.
    const combined = base.concat(additions);
    const latestSchemaIndex = combined.findLastIndex(isResultSchemaSystemBlock);
    return combined.filter((block, index) => !isResultSchemaSystemBlock(block) || index === latestSchemaIndex);
}

function addAgentMessageCacheBreakpoints(
    messages: MessageParam[],
    cacheControl: { type: 'ephemeral'; ttl?: '5m' | '1h' },
): boolean {
    let cacheableBlockIndex = 0;
    let breakpointCount = 0;

    for (const message of messages) {
        if (!Array.isArray(message.content)) continue;

        for (const block of message.content) {
            if (block.type === 'thinking' || block.type === 'redacted_thinking') continue;

            if (cacheableBlockIndex === breakpointCount * AGENT_MESSAGE_CACHE_BLOCK_INTERVAL) {
                block.cache_control = cacheControl;
                breakpointCount++;
                if (breakpointCount === MAX_CLAUDE_CACHE_BREAKPOINTS) return true;
            }
            cacheableBlockIndex++;
        }
    }

    return breakpointCount > 0;
}

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
    cache_creation?: { ephemeral_1h_input_tokens?: number | null } | null;
    service_tier?: string | null;
    /** `fast` when the request ran in fast mode. */
    speed?: string | null;
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
    thinking_mode?: ClaudeThinkingInput['thinking_mode'];
    include_thoughts?: boolean;
    cache_enabled?: boolean;
    cache_ttl?: string;
    /** `fast` runs the request in fast mode (Claude API only). */
    speed?: 'standard' | 'fast';
    tool_choice?: 'auto' | 'none' | 'any' | 'required';
    /** Internal execution hint supplied after public model-option validation. */
    required_tool_name?: string;
    /** Internal execution hint used with a required named tool. */
    parallel_tool_calls?: boolean;
}

interface RequestOptions {
    headers?: Record<string, string>;
    signal?: AbortSignal;
    timeout?: number;
}

type ClaudeTool = NonNullable<MessageCreateParamsBase['tools']>[number];
type ClaudeMessageStream = AsyncIterable<RawMessageStreamEvent> & {
    abort(): void;
    finalMessage(): Promise<Message>;
};
type ClaudeMessagesClient = Anthropic | AnthropicVertex | AnthropicBedrockMantle;

function streamClaudeMessages(
    client: ClaudeMessagesClient,
    payload: MessageStreamParams,
    requestOptions: RequestOptions | undefined,
): Promise<ClaudeMessageStream> {
    return Promise.resolve(client.messages.stream(payload, requestOptions));
}

// ============================================================================
// Token usage
// ============================================================================

/**
 * The processing tier a response reports: `fast` for fast mode, else its service tier. Fast mode is priced on
 * its own and cannot run under a Priority Tier commitment, so one value describes both.
 */
export function claudeServiceTier(usage: Pick<AnthropicUsageLike, 'service_tier' | 'speed'>): string | undefined {
    if (usage.speed === 'fast') return 'fast';
    return usage.service_tier ?? undefined;
}

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
        prompt_cache_write_1h: usage.cache_creation?.ephemeral_1h_input_tokens || undefined,
    };
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

export function collectClaudeResults(content: ContentBlock[], includeThoughts = false): CompletionResult[] {
    const results: CompletionResult[] = [];
    for (const block of content) {
        if (block.type === 'thinking' && block.thinking && includeThoughts) {
            results.push({ type: 'thoughts', value: block.thinking });
        } else if (block.type === 'text' && block.text) {
            results.push({ type: 'text', value: block.text });
        }
    }
    return results;
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
        } else if (file.mime_type?.startsWith('audio/')) {
            throw new Error('Claude does not support audio input; supply a transcript instead');
        } else if (file.mime_type?.startsWith('video/')) {
            throw new Error(`Claude does not support video input: ${file.name}`);
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
    _logger?: WarnLogger,
): Promise<ClaudePrompt> {
    let system: TextBlockParam[] | undefined = segments
        .filter((s) => s.role === PromptRole.system)
        .map((s) => ({ text: s.content, type: 'text' as const }));

    let schemaText: string | undefined;
    if (options.result_schema) {
        schemaText =
            options.tools && options.tools.length > 0
                ? `${TOOL_AWARE_JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(options.result_schema)}`
                : `${JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(options.result_schema)}`;
        if (options.prompt_cache_key === undefined) {
            system.push({ text: schemaText, type: 'text' as const });
        }
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
            contentBlocks.push(...(await collectFileBlocks(segment)));
            messages.push({
                role: 'user',
                content: [
                    {
                        type: 'tool_result',
                        tool_use_id: segment.tool_use_id,
                        content: contentBlocks,
                        ...(segment.tool_result_status === undefined
                            ? {}
                            : { _llumiverse_tool_result_status: segment.tool_result_status }),
                    } satisfies CanonicalClaudeToolResultBlockParam,
                ],
            });
        } else {
            const contentBlocks: ContentBlockParam[] = [];
            if (segment.content) {
                contentBlocks.push({ type: 'text', text: segment.content } satisfies TextBlockParam);
            }
            contentBlocks.push(...(await collectFileBlocks(segment)));
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

    if (schemaText && options.prompt_cache_key !== undefined) {
        if (isAgentPromptCacheKey(options.prompt_cache_key)) {
            // Agent result schemas are stable for the whole tool loop. Keep one
            // generated system block rather than appending the same schema to
            // every persisted continuation prompt. Routed one-shot prompts keep
            // the schema on their dynamic task block below.
            system.push({ text: schemaText, type: 'text' as const });
        } else {
            const taskMessage = messages[messages.length - 1];
            const taskBlock = Array.isArray(taskMessage?.content)
                ? taskMessage.content[taskMessage.content.length - 1]
                : undefined;
            if (taskBlock?.type === 'text') {
                taskBlock.text = `${taskBlock.text}\n\n${schemaText}`;
            } else {
                system.push({ text: schemaText, type: 'text' as const });
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
    const system = mergeClaudeSystemBlocks(baseSystemMessages, prompt.system || []);
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
    if (
        typeof block === 'object' &&
        block !== null &&
        ('cache_control' in block || '_llumiverse_tool_result_status' in block)
    ) {
        const {
            cache_control: _cc,
            _llumiverse_tool_result_status: _status,
            ...rest
        } = block as T & { cache_control?: unknown; _llumiverse_tool_result_status?: unknown };
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
    provider = 'anthropic',
    operation: 'execute' | 'stream' = 'stream',
    transport?: ClaudeTransportIdentity,
    tools: readonly ClaudeToolDefinition[] | undefined = options.tools,
): { payload: MessageCreateParamsBase; requestOptions: RequestOptions | undefined } {
    const modelName = transport?.model ?? options.model;
    const model_options = options.model_options as ClaudeBaseOptions | undefined;

    let requestOptions: RequestOptions | undefined;
    if (
        modelName.includes('claude-3-7-sonnet') &&
        ((model_options?.max_tokens ?? 0) > 64000 || (model_options?.thinking_budget_tokens ?? 0) > 64000)
    ) {
        requestOptions = { headers: { 'anthropic-beta': 'output-128k-2025-02-19' } };
    }
    const fastMode = model_options?.speed === 'fast';
    if (fastMode) {
        const betas = [requestOptions?.headers?.['anthropic-beta'], CLAUDE_FAST_MODE_BETA].filter(Boolean);
        requestOptions = {
            ...requestOptions,
            headers: { ...requestOptions?.headers, 'anthropic-beta': betas.join(',') },
        };
    }

    // Merge first so parallel tool results split across user messages are recombined
    // into the single user turn after their assistant tool_use message; then fix both
    // orphan directions (tool_use without result, and result without tool_use) so the
    // request can never trip Anthropic/Vertex's tool_use/tool_result pairing validation.
    const mergedMessages = mergeConsecutiveUserMessages(prompt.messages);
    const fixedMessages = fixOrphanedToolResults(fixOrphanedToolUse(mergedMessages));
    let sanitizedMessages = sanitizeMessages(fixedMessages);

    if (tools) {
        for (const tool of tools) {
            if (
                typeof tool.input_schema !== 'object' ||
                tool.input_schema === null ||
                Array.isArray(tool.input_schema) ||
                tool.input_schema.type !== 'object'
            ) {
                const actualType =
                    typeof tool.input_schema === 'object' && tool.input_schema !== null
                        ? tool.input_schema.type
                        : typeof tool.input_schema;
                throw new Error(
                    'Tool "' +
                        tool.name +
                        '" has invalid input_schema.type: expected "object", got "' +
                        String(actualType) +
                        '"',
                );
            }
        }
    }

    const hasTools = tools !== undefined && tools.length > 0;
    if (!hasTools && claudeMessagesContainToolBlocks(sanitizedMessages)) {
        sanitizedMessages = convertClaudeToolBlocksToText(sanitizedMessages);
    }

    // Claude 4.6+ rejects requests whose conversation ends with an assistant
    // message ("does not support assistant message prefill"). A trailing
    // assistant turn can emerge from upstream retry edges — e.g. a re-sent
    // resume whose tool results were stripped as orphans above, leaving the
    // previous attempt's reply last. Appending a minimal user turn converts a
    // guaranteed 400 into a graceful continue — the same behavior pre-4.6
    // models applied implicitly by treating the trailing turn as prefill.
    // Older models are left untouched: prefill there can be intentional.
    if (isClaudeVersionGTE(modelName, 4, 6)) {
        const lastMessage = sanitizedMessages[sanitizedMessages.length - 1];
        if (lastMessage?.role === 'assistant') {
            sanitizedMessages = [
                ...sanitizedMessages,
                { role: 'user', content: [{ type: 'text', text: 'Continue.' }] },
            ];
        }
    }

    sanitizedMessages = stripClaudeCacheControlFromMessages(sanitizedMessages);
    const sanitizedSystem = stripClaudeCacheControlFromSystem(prompt.system);
    const sanitizedTools = hasTools
        ? stripClaudeCacheControlFromTools(
              tools.map(({ name, description, input_schema }) => ({
                  name,
                  ...(description === undefined ? {} : { description }),
                  input_schema,
              })) as MessageCreateParamsBase['tools'],
          )
        : undefined;

    const cacheEnabled = isClaudePromptCacheEnabled(options);
    if (cacheEnabled) {
        const cacheTtl = model_options?.cache_ttl as '5m' | '1h' | undefined;
        const cacheControl = { type: 'ephemeral' as const, ...(cacheTtl && { ttl: cacheTtl }) };

        // Vertex requires cache_control to remain on the same blocks across
        // requests. Agent conversations therefore use fixed block ordinals:
        // each newly reached breakpoint extends the cached prefix without
        // relocating or deleting an earlier breakpoint. Four message markers
        // also cache the preceding system and tool definitions, while covering
        // the growing tool loop until the next semantic checkpoint.
        const hasAgentMessageBreakpoints =
            isAgentPromptCacheKey(options.prompt_cache_key) &&
            addAgentMessageCacheBreakpoints(sanitizedMessages, cacheControl);

        if (!hasAgentMessageBreakpoints && sanitizedSystem && sanitizedSystem.length > 0) {
            const lastBlock = sanitizedSystem[sanitizedSystem.length - 1] as TextBlockParam & {
                cache_control?: unknown;
            };
            lastBlock.cache_control = cacheControl;
        }
        if (!hasAgentMessageBreakpoints && sanitizedTools && sanitizedTools.length > 0) {
            const lastTool = sanitizedTools[sanitizedTools.length - 1] as ClaudeTool & { cache_control?: unknown };
            lastTool.cache_control = cacheControl;
        }
        if (!hasAgentMessageBreakpoints && options.prompt_cache_key !== undefined) {
            const lastMessage = sanitizedMessages[sanitizedMessages.length - 1];
            if (lastMessage && Array.isArray(lastMessage.content) && lastMessage.content.length >= 2) {
                const stablePrefixBlock = lastMessage.content[lastMessage.content.length - 2];
                if (
                    typeof stablePrefixBlock === 'object' &&
                    stablePrefixBlock !== null &&
                    'type' in stablePrefixBlock &&
                    stablePrefixBlock.type !== 'thinking' &&
                    stablePrefixBlock.type !== 'redacted_thinking'
                ) {
                    stablePrefixBlock.cache_control = cacheControl;
                }
            }
        } else if (!hasAgentMessageBreakpoints && sanitizedMessages.length >= 4) {
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
    const forcedToolChoice = model_options?.required_tool_name
        ? ({ type: 'tool', name: model_options.required_tool_name } as const)
        : model_options?.tool_choice === 'required' || model_options?.tool_choice === 'any'
          ? ({ type: 'any' } as const)
          : undefined;
    if (forcedToolChoice && !hasTools) {
        throw createToolChoiceConfigurationError('A forced Claude tool turn requires at least one tool definition.', {
            provider,
            model: modelName,
            operation,
        });
    }
    if (forcedToolChoice && modelName.toLowerCase().includes('claude-mythos-preview')) {
        throw createToolChoiceConfigurationError(
            `Claude preview model ${modelName} does not support forced tool choice.`,
            { provider, model: modelName, operation },
        );
    }
    const disableManualThinkingForForcedTool = forcedToolChoice !== undefined && thinking?.type === 'enabled';
    const toolChoice = !hasTools
        ? undefined
        : forcedToolChoice
          ? { ...forcedToolChoice, disable_parallel_tool_use: model_options?.parallel_tool_calls === false }
          : model_options?.tool_choice === 'none'
            ? ({ type: 'none' } as const)
            : model_options?.tool_choice === 'auto'
              ? ({ type: 'auto' } as const)
              : undefined;

    const payload: MessageCreateParamsBase = {
        messages: sanitizedMessages,
        system: sanitizedSystem,
        tools: sanitizedTools,
        tool_choice: toolChoice,
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
        thinking: disableManualThinkingForForcedTool ? { type: 'disabled' } : thinking,
        stream: true,
        ...(!disableManualThinkingForForcedTool && outputConfig && { output_config: outputConfig }),
        // `speed` is a beta request field the base type doesn't declare yet.
        ...(fastMode && ({ speed: 'fast' } as Partial<MessageCreateParamsBase>)),
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
    // Sliding-window truncation rewrites one previously sent message whenever it
    // falls out of the recent-message window. That defeats Anthropic's exact
    // prefix cache and turns the growing conversation into a cache write on every
    // agent iteration. Cached Claude conversations are append-only between
    // semantic checkpoints; their model-facing tool results are already bounded
    // and artifact-backed at the Studio execution boundary.
    const preserveCachedTextPrefix = isClaudePromptCacheEnabled(options);
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens:
            requestedTextMax && !preserveCachedTextPrefix
                ? Math.min(requestedTextMax, OLD_MESSAGE_TEXT_MAX_TOKENS)
                : undefined,
        keepRecentMessages: requestedTextMax && !preserveCachedTextPrefix ? KEEP_RECENT_MESSAGES : undefined,
    };
    let processed = stripBase64ImagesFromConversation(conversation, stripOptions);
    processed = truncateLargeTextInConversation(processed, stripOptions);
    processed = stripHeartbeatsFromConversation(processed, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
    });
    return processed as ClaudePrompt;
}

export function projectClaudeConversation(
    conversation: ClaudePrompt,
    options: ExecutionOptions,
    currentTurn: number,
): ClaudePrompt {
    const prunedConversation = pruneClaudeThinking(conversation);
    const activeTurnStart = findClaudeActiveTurnStart(prunedConversation);
    const protectedMessages = new Set(activeTurnStart >= 0 ? prunedConversation.messages.slice(activeTurnStart) : []);
    const preserveSubtree = (value: unknown): boolean => {
        if (!value || typeof value !== 'object') return false;
        if (protectedMessages.has(value as MessageParam)) return true;
        const type = (value as { type?: unknown }).type;
        return type === 'thinking' || type === 'redacted_thinking';
    };
    const stripOpts = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        // See buildClaudeStreamingConversation: cached histories must remain
        // byte-stable between checkpoints, so do not age-rewrite text blocks.
        textMaxTokens: isClaudePromptCacheEnabled(options) ? undefined : options.stripTextMaxTokens,
        preserveSubtree,
    };
    let processedConversation = stripBase64ImagesFromConversation(prunedConversation, stripOpts);
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

function prepareCanonicalClaudeProjection(
    prepared: Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>,
    options: ExecutionOptions,
    contextOnly = false,
): ClaudePrompt {
    const projected = projectClaudeConversation(
        prepared.native_conversation,
        options,
        canonicalConversationTurnNumber(prepared.document),
    );
    return contextOnly
        ? projectClaudeContextResultSchema(projected, options, prepared.tool_definitions.length > 0)
        : projected;
}

function canonicalClaudeUsage(
    prepared: Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>,
): ExecutionTokenUsage | undefined {
    const usage = prepared.accepted_response?.generation.usage;
    if (usage === undefined) return undefined;
    return {
        ...(usage.input_tokens === undefined ? {} : { prompt: usage.input_tokens }),
        ...(usage.output_tokens === undefined ? {} : { result: usage.output_tokens }),
        ...(usage.total_tokens === undefined ? {} : { total: usage.total_tokens }),
        ...(usage.cache_read_tokens === undefined ? {} : { prompt_cached: usage.cache_read_tokens }),
        ...(usage.cache_write_tokens === undefined ? {} : { prompt_cache_write: usage.cache_write_tokens }),
        ...(usage.input_new_tokens === undefined ? {} : { prompt_new: usage.input_new_tokens }),
    };
}

function canonicalClaudeServiceTier(
    prepared: Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>,
): string | undefined {
    const reported = prepared.accepted_response?.generation.usage?.reported_usage?.find(
        (candidate) => candidate.source === 'provider' && candidate.protocol === CLAUDE_MESSAGES_PROTOCOL,
    );
    const payload =
        reported?.payload !== null && typeof reported?.payload === 'object' && !Array.isArray(reported.payload)
            ? (reported.payload as Record<string, unknown>)
            : undefined;
    return claudeServiceTier({
        service_tier: typeof payload?.service_tier === 'string' ? payload.service_tier : undefined,
        speed: typeof payload?.speed === 'string' ? payload.speed : undefined,
    });
}

function recoverClaudeCompletion(
    prepared: Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>,
    options: ExecutionOptions,
    includeThoughts: boolean,
): Completion {
    const accepted = prepared.accepted_response;
    if (accepted === undefined) throw new Error('No accepted Claude Messages response is available');
    if (options.include_original_response) {
        throw new Error('An idempotently recovered Claude Messages response cannot reconstruct original_response');
    }
    const result = accepted.turn.blocks.flatMap((block): CompletionResult[] => {
        if (block.type === 'text') return [{ type: 'text', value: block.text }];
        if (block.type === 'json') return [{ type: 'json', value: block.value }];
        if (block.type === 'reasoning' && includeThoughts) return [{ type: 'thoughts', value: block.text }];
        return [];
    });
    const toolUse = accepted.turn.blocks.flatMap((block): ToolUse[] =>
        block.type === 'tool_call'
            ? [
                  {
                      id: block.call_id,
                      tool_name: block.tool_name,
                      tool_input:
                          block.arguments.type === 'invalid'
                              ? {}
                              : (toolArgumentsForModel(block.arguments) as JSONObject),
                  },
              ]
            : [],
    );
    return {
        result: result.length > 0 ? result : [{ type: 'text', value: '' }],
        ...(toolUse.length === 0 ? {} : { tool_use: toolUse }),
        token_usage: canonicalClaudeUsage(prepared),
        service_tier: canonicalClaudeServiceTier(prepared),
        finish_reason: toolUse.length > 0 ? 'tool_use' : claudeFinishReason(accepted.generation.finish_reason),
        conversation: prepared.document,
    };
}

function recoveredClaudeStream(completion: Completion): DriverCompletionStream {
    const stream = (async function* (): AsyncIterable<CompletionChunkObject> {
        yield {
            result: completion.result,
            tool_use: completion.tool_use,
            token_usage: completion.token_usage,
            finish_reason: completion.finish_reason,
        };
    })();
    return Object.assign(stream, { finalizeConversation: () => completion.conversation });
}

function assertClaudeAcceptedTargetOptions(
    state: Pick<
        PreparedClaudeConversation,
        'accepted_response' | 'response_selection_policy' | 'runtime' | 'target_options'
    >,
): void {
    const accepted = state.accepted_response;
    if (accepted === undefined) return;
    const retainedOptions = accepted.generation.request_receipt.target.options;
    const expectedOptions =
        retainedOptions !== undefined && Object.hasOwn(retainedOptions, CANONICAL_TOOL_SELECTION_TARGET_OPTION)
            ? canonicalToolSelectionTargetOptions(state.target_options, state.response_selection_policy)
            : state.target_options;
    if (canonicalJsonContentString(retainedOptions ?? null) !== canonicalJsonContentString(expectedOptions ?? null)) {
        throw new Error(
            `Accepted response operation ${state.runtime.response_operation_id} has incompatible request routing`,
        );
    }
}

/** Execute Claude Messages directly into the canonical conversation response contract. */
export async function executeCanonicalClaudeCompletion(
    client: ClaudeMessagesClient,
    prompt: ClaudePrompt,
    options: ExecutionOptions,
    logger?: Logger,
    provider = 'anthropic',
    transportOptions?: Pick<RequestOptions, 'signal' | 'timeout'>,
    transport?: ClaudeTransportIdentity,
    hostCapabilities?: CanonicalHostCapabilities,
): Promise<CanonicalExecutionResponse> {
    const canonicalState = await prepareClaudeCanonicalState({
        conversation: options.conversation,
        prompt,
        options,
        provider,
        resolve_asset: hostCapabilities?.resolve_canonical_asset,
        signal: transportOptions?.signal ?? undefined,
        ...(transport?.target_options === undefined ? {} : { target_options: transport.target_options }),
    });
    return executePreparedCanonicalClaudeCompletion(
        client,
        canonicalState,
        options,
        logger,
        provider,
        transportOptions,
        transport,
    );
}

export async function executeCanonicalClaudeContext(
    client: ClaudeMessagesClient,
    options: CanonicalExecutionContextOptions,
    logger?: Logger,
    provider = 'anthropic',
    transportOptions?: Pick<RequestOptions, 'signal' | 'timeout'>,
    transport?: ClaudeTransportIdentity,
    hostCapabilities?: CanonicalHostCapabilities,
): Promise<CanonicalExecutionResponse> {
    const canonicalState = await prepareClaudeCanonicalContext({
        options,
        provider,
        resolve_asset: hostCapabilities?.resolve_canonical_asset,
        signal: transportOptions?.signal ?? undefined,
        ...(transport?.target_options === undefined ? {} : { target_options: transport.target_options }),
    });
    return executePreparedCanonicalClaudeCompletion(
        client,
        canonicalState,
        options,
        logger,
        provider,
        transportOptions,
        transport,
        true,
    );
}

async function executePreparedCanonicalClaudeCompletion(
    client: ClaudeMessagesClient,
    canonicalState: Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>,
    options: ExecutionOptions,
    logger: Logger | undefined,
    provider: string,
    transportOptions: Pick<RequestOptions, 'signal' | 'timeout'> | undefined,
    transport: ClaudeTransportIdentity | undefined,
    contextOnly = false,
): Promise<CanonicalExecutionResponse> {
    const conversation = prepareCanonicalClaudeProjection(canonicalState, options, contextOnly);
    const { payload, requestOptions } = getClaudePayload(
        options,
        conversation,
        provider,
        'execute',
        transport,
        canonicalState.tool_definitions,
    );
    await assertAcceptedCanonicalRequest(
        canonicalState,
        { provider, protocol: CLAUDE_MESSAGES_PROTOCOL, model: options.model },
        providerJsonValue(payload),
    );
    assertClaudeAcceptedTargetOptions(canonicalState);
    if (canonicalState.accepted_response !== undefined) {
        if (options.include_original_response) {
            throw new Error('An idempotently recovered Claude Messages response cannot reconstruct original_response');
        }
        return recoverCanonicalExecutionResponse(canonicalState, options, {
            service_tier: canonicalClaudeServiceTier(canonicalState),
        });
    }
    const prepared = await finalizeClaudePreparedRequest(
        { ...canonicalState, native_conversation: conversation },
        payload,
    );
    await publishCanonicalPreparedRequest(prepared, options);
    const responseStream = await streamClaudeMessages(
        client,
        payload as MessageStreamParams,
        transportOptions ? { ...requestOptions, ...transportOptions } : requestOptions,
    );
    const result = await responseStream.finalMessage();
    logClaudeTruncation(logger, result.stop_reason, { provider, model: options.model });

    const toolUse = collectClaudeTools(result.content);
    const rawDecoded = await decodeClaudeCanonicalResponse(result, prepared);
    const normalized =
        !toolUse?.length && options.result_schema
            ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
            : undefined;
    let decoded =
        normalized?.status === 'valid'
            ? await decodeClaudeCanonicalResponse(result, prepared, normalized.structured_output)
            : rawDecoded;
    if (normalized?.status === 'invalid') decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
    const document = await appendClaudeCanonicalResponseWithProcessing(prepared, decoded);
    return createCanonicalExecutionResponse(document, prepared.runtime.response_operation_id, {
        service_tier: claudeServiceTier(result.usage as AnthropicUsageLike),
        ...(options.include_original_response ? { original_response: result } : {}),
    });
}

/**
 * Execute a non-streaming Claude completion.
 * Works with the Anthropic, Vertex AI, and Bedrock Mantle SDK clients.
 */
export async function executeClaudeCompletion(
    client: ClaudeMessagesClient,
    prompt: ClaudePrompt,
    options: ExecutionOptions,
    logger?: Logger,
    provider = 'anthropic',
    transportOptions?: Pick<RequestOptions, 'signal' | 'timeout'>,
    transport?: ClaudeTransportIdentity,
): Promise<Completion> {
    const model_options = options.model_options as ClaudeBaseOptions | undefined;
    const canonicalState = await prepareClaudeCanonicalState({
        conversation: options.conversation,
        prompt,
        options,
        provider,
        ...(transport?.target_options === undefined ? {} : { target_options: transport.target_options }),
    });
    const includeThoughts = model_options?.include_thoughts ?? false;
    const conversation = prepareCanonicalClaudeProjection(canonicalState, options);
    const { payload, requestOptions } = getClaudePayload(
        options,
        conversation,
        provider,
        'execute',
        transport,
        canonicalState.tool_definitions,
    );
    await assertAcceptedCanonicalRequest(
        canonicalState,
        { provider, protocol: CLAUDE_MESSAGES_PROTOCOL, model: options.model },
        providerJsonValue(payload),
    );
    assertClaudeAcceptedTargetOptions(canonicalState);
    if (canonicalState.accepted_response !== undefined) {
        return recoverClaudeCompletion(canonicalState, options, includeThoughts);
    }
    const prepared = await finalizeClaudePreparedRequest(
        { ...canonicalState, native_conversation: conversation },
        payload,
    );
    await publishCanonicalPreparedRequest(prepared, options);

    const responseStream = await streamClaudeMessages(
        client,
        payload as MessageStreamParams,
        transportOptions ? { ...requestOptions, ...transportOptions } : requestOptions,
    );
    const result = await responseStream.finalMessage();
    logClaudeTruncation(logger, result.stop_reason, { provider, model: options.model });

    const completionResults = collectClaudeResults(result.content, includeThoughts);
    const tool_use = collectClaudeTools(result.content);
    const rawDecoded = await decodeClaudeCanonicalResponse(result, prepared);
    const normalized =
        !tool_use?.length && options.result_schema
            ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
            : undefined;
    const decoded =
        normalized?.status === 'valid'
            ? await decodeClaudeCanonicalResponse(result, prepared, normalized.structured_output)
            : rawDecoded;
    const processedConversation = await appendClaudeCanonicalResponseWithProcessing(prepared, decoded);

    return {
        result: completionResults.length > 0 ? completionResults : [{ type: 'text', value: '' }],
        tool_use,
        token_usage: anthropicUsageToTokenUsage(result.usage),
        service_tier: claudeServiceTier(result.usage as AnthropicUsageLike),
        finish_reason: tool_use ? 'tool_use' : claudeFinishReason(result?.stop_reason ?? ''),
        conversation: processedConversation,
    };
}

/**
 * Execute a streaming Claude completion.
 * Works with the Anthropic, Vertex AI, and Bedrock Mantle SDK clients.
 */
export async function streamClaudeCompletion(
    client: ClaudeMessagesClient,
    prompt: ClaudePrompt,
    options: ExecutionOptions,
    logger?: Logger,
    provider = 'anthropic',
    transportOptions?: Pick<RequestOptions, 'signal' | 'timeout'>,
    transport?: ClaudeTransportIdentity,
): Promise<DriverCompletionStream> {
    const model_options = options.model_options as ClaudeBaseOptions | undefined;
    const canonicalState = await prepareClaudeCanonicalState({
        conversation: options.conversation,
        prompt,
        options,
        provider,
        ...(transport?.target_options === undefined ? {} : { target_options: transport.target_options }),
    });
    const includeThoughts = model_options?.include_thoughts ?? false;
    const conversation = prepareCanonicalClaudeProjection(canonicalState, options);
    const { payload, requestOptions } = getClaudePayload(
        options,
        conversation,
        provider,
        'stream',
        transport,
        canonicalState.tool_definitions,
    );
    const streamingPayload: MessageStreamParams = { ...payload, stream: true };
    await assertAcceptedCanonicalRequest(
        canonicalState,
        { provider, protocol: CLAUDE_MESSAGES_PROTOCOL, model: options.model },
        providerJsonValue(streamingPayload),
    );
    assertClaudeAcceptedTargetOptions(canonicalState);
    if (canonicalState.accepted_response !== undefined) {
        return recoveredClaudeStream(recoverClaudeCompletion(canonicalState, options, includeThoughts));
    }
    const prepared = await finalizeClaudePreparedRequest(
        { ...canonicalState, native_conversation: conversation },
        streamingPayload as unknown as MessageCreateParamsBase,
    );
    await publishCanonicalPreparedRequest(prepared, options);

    const response_stream = await streamClaudeMessages(
        client,
        streamingPayload,
        transportOptions ? { ...requestOptions, ...transportOptions } : requestOptions,
    );

    let currentToolUse: { id: string; name: string; inputJson: string } | null = null;
    let pendingSpacing = false;

    const stream = asyncMap(response_stream, async (streamEvent: RawMessageStreamEvent) => {
        switch (streamEvent.type) {
            case 'message_start':
                return {
                    result: [{ type: 'text', value: '' }],
                    token_usage: anthropicUsageToTokenUsage(streamEvent.message.usage as AnthropicUsageLike),
                    service_tier: claudeServiceTier(streamEvent.message.usage as AnthropicUsageLike),
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
                                tool_input: '',
                            },
                        ],
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
                                        tool_input: streamEvent.delta.partial_json,
                                    },
                                ],
                            } satisfies CompletionChunkObject;
                        }
                        break;
                    case 'thinking_delta':
                        if (includeThoughts) {
                            return {
                                result: streamEvent.delta.thinking
                                    ? [{ type: 'thoughts', value: streamEvent.delta.thinking }]
                                    : [],
                            } satisfies CompletionChunkObject;
                        }
                        break;
                    case 'signature_delta':
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

    async function computeDecodedFinalResponse() {
        const finalMessage = await response_stream.finalMessage();
        const finalTools = collectClaudeTools(finalMessage.content);
        const rawDecoded = await decodeClaudeCanonicalResponse(finalMessage, prepared);
        const normalized =
            !finalTools?.length && options.result_schema
                ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
                : undefined;
        const decoded =
            normalized?.status === 'valid'
                ? await decodeClaudeCanonicalResponse(finalMessage, prepared, normalized.structured_output)
                : rawDecoded;
        return { decoded, finalMessage, normalized };
    }
    let decodedFinalResponse: ReturnType<typeof computeDecodedFinalResponse> | undefined;
    const decodeFinalResponse = () => {
        decodedFinalResponse ??= computeDecodedFinalResponse();
        return decodedFinalResponse;
    };
    const driverStream: DriverCompletionStream = {
        [Symbol.asyncIterator]: () => stream[Symbol.asyncIterator](),
        finalizeConversation: async () => {
            const { decoded } = await decodeFinalResponse();
            return await appendClaudeCanonicalResponseWithProcessing(prepared, decoded);
        },
    };
    return driverStream;
}

interface ClaudeCanonicalDraft {
    draft_block_id: string;
    native_position: NativeStreamPosition;
    kind: 'text' | 'reasoning' | 'tool_call';
    text: string;
    tool_argument_fragments: string[];
    tool_argument_snapshot?: JsonValue;
}

function claudeStreamPosition(index: number, nativeItemId?: string): NativeStreamPosition {
    return {
        protocol: CLAUDE_MESSAGES_PROTOCOL,
        path: ['content', index],
        ...(nativeItemId === undefined ? {} : { native_item_id: nativeItemId }),
    };
}

function claudeSemanticBlocks(decoded: DecodedConversationResponse, turnId: string) {
    const turn = decoded.turns.find((candidate) => candidate.id === turnId);
    if (turn?.kind !== 'agent') throw new Error('Claude stream decode has no generated agent turn');
    return turn.blocks.filter((block) => block.type !== 'native_replay');
}

function claudeSemanticPositions(message: Message): NativeStreamPosition[] {
    return message.content.flatMap((block, index) => {
        if (block.type === 'text' || block.type === 'thinking') return [claudeStreamPosition(index)];
        if (block.type === 'tool_use') return [claudeStreamPosition(index, block.id)];
        return [];
    });
}

function isEmptyJsonObject(value: JsonValue): boolean {
    return typeof value === 'object' && value !== null && !Array.isArray(value) && Object.keys(value).length === 0;
}

function assertClaudeToolDraftArguments(
    draft: ClaudeCanonicalDraft,
    block: ReturnType<typeof claudeSemanticBlocks>[number],
) {
    if (draft.kind !== 'tool_call' || block.type !== 'tool_call' || block.arguments.type === 'invalid') return;
    const expected = canonicalJsonContentString(toolArgumentsForModel(block.arguments));
    if (draft.tool_argument_snapshot !== undefined) {
        if (canonicalJsonContentString(draft.tool_argument_snapshot) !== expected) {
            throw new Error('Claude tool argument snapshot differs from its terminal tool input');
        }
        if (draft.tool_argument_fragments.length > 0) {
            throw new Error('Claude tool stream mixes an initial argument snapshot with JSON fragments');
        }
        return;
    }
    if (draft.tool_argument_fragments.length === 0) return;
    let streamed: JsonValue;
    try {
        streamed = providerJsonValue(JSON.parse(draft.tool_argument_fragments.join('')));
    } catch {
        throw new Error('Claude tool argument fragments do not form valid JSON');
    }
    if (canonicalJsonContentString(streamed) !== expected) {
        throw new Error('Claude tool argument fragments differ from the terminal tool input');
    }
}

/** Execute a Claude Messages stream as request-scoped canonical draft events. */
export async function streamCanonicalClaudeEvents(
    client: ClaudeMessagesClient,
    prompt: ClaudePrompt,
    options: ExecutionOptions,
    open: CanonicalStreamOpenOptions,
    logger?: Logger,
    provider = 'anthropic',
    transportOptions?: Pick<RequestOptions, 'signal' | 'timeout'>,
    transport?: ClaudeTransportIdentity,
    hostCapabilities?: CanonicalHostCapabilities,
): Promise<CanonicalExecutionEventStream> {
    const canonicalState = await prepareClaudeCanonicalState({
        conversation: options.conversation,
        prompt,
        options,
        provider,
        resolve_asset: hostCapabilities?.resolve_canonical_asset,
        signal: transportOptions?.signal ?? undefined,
        ...(transport?.target_options === undefined ? {} : { target_options: transport.target_options }),
    });
    return streamPreparedCanonicalClaudeEvents(
        client,
        canonicalState,
        options,
        open,
        logger,
        provider,
        transportOptions,
        transport,
    );
}

export async function streamCanonicalClaudeContextEvents(
    client: ClaudeMessagesClient,
    options: CanonicalExecutionContextOptions,
    open: CanonicalStreamOpenOptions,
    logger?: Logger,
    provider = 'anthropic',
    transportOptions?: Pick<RequestOptions, 'signal' | 'timeout'>,
    transport?: ClaudeTransportIdentity,
    hostCapabilities?: CanonicalHostCapabilities,
): Promise<CanonicalExecutionEventStream> {
    const canonicalState = await prepareClaudeCanonicalContext({
        options,
        provider,
        resolve_asset: hostCapabilities?.resolve_canonical_asset,
        signal: transportOptions?.signal ?? undefined,
        ...(transport?.target_options === undefined ? {} : { target_options: transport.target_options }),
    });
    return streamPreparedCanonicalClaudeEvents(
        client,
        canonicalState,
        options,
        open,
        logger,
        provider,
        transportOptions,
        transport,
        true,
    );
}

async function streamPreparedCanonicalClaudeEvents(
    client: ClaudeMessagesClient,
    canonicalState: Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>,
    options: ExecutionOptions,
    open: CanonicalStreamOpenOptions,
    logger: Logger | undefined,
    provider: string,
    transportOptions: Pick<RequestOptions, 'signal' | 'timeout'> | undefined,
    transport: ClaudeTransportIdentity | undefined,
    contextOnly = false,
): Promise<CanonicalExecutionEventStream> {
    const conversation = prepareCanonicalClaudeProjection(canonicalState, options, contextOnly);
    const { payload, requestOptions } = getClaudePayload(
        options,
        conversation,
        provider,
        'stream',
        transport,
        canonicalState.tool_definitions,
    );
    const streamingPayload: MessageStreamParams = { ...payload, stream: true };
    await assertAcceptedCanonicalRequest(
        canonicalState,
        { provider, protocol: CLAUDE_MESSAGES_PROTOCOL, model: options.model },
        providerJsonValue(streamingPayload),
    );
    assertClaudeAcceptedTargetOptions(canonicalState);
    const acceptedResponse = canonicalState.accepted_response;
    const identity = {
        request_id: acceptedResponse?.generation.request_id ?? canonicalState.runtime.request_id,
        attempt_id: acceptedResponse?.generation.attempt_id ?? canonicalState.runtime.attempt_id,
        response_operation_id: canonicalState.runtime.response_operation_id,
        generation_id: acceptedResponse?.generation.id ?? canonicalState.generation_id,
        draft_turn_id: acceptedResponse?.turn.id ?? canonicalState.response_turn_id,
    };
    if (acceptedResponse !== undefined) {
        if (options.include_original_response) {
            throw new Error('An idempotently recovered Claude Messages response cannot reconstruct original_response');
        }
        return new FallbackCanonicalExecutionEventStream(
            identity,
            () =>
                recoverCanonicalExecutionResponse(canonicalState, options, {
                    service_tier: canonicalClaudeServiceTier(canonicalState),
                }),
            { ...open, origin: 'accepted_recovery' },
        );
    }

    const prepared = await finalizeClaudePreparedRequest(
        { ...canonicalState, native_conversation: conversation },
        streamingPayload as unknown as MessageCreateParamsBase,
    );
    const abortController = new AbortController();
    const forwardAbort = () => abortController.abort(transportOptions?.signal?.reason);
    let responseStream: ClaudeMessageStream | undefined;
    const drafts = new Map<number, ClaudeCanonicalDraft>();

    const eventStream = canonicalNativeExecutionEventStream({
        identity,
        open,
        openSource: async () => {
            responseStream = await streamClaudeMessages(
                client,
                streamingPayload,
                transportOptions
                    ? { ...requestOptions, ...transportOptions, signal: abortController.signal }
                    : { ...requestOptions, signal: abortController.signal },
            );
            return responseStream;
        },
        map: async (event, writer) => {
            if (event.type === 'content_block_start') {
                const index = event.index;
                const block = event.content_block;
                if (block.type === 'text' || block.type === 'thinking' || block.type === 'tool_use') {
                    const position = claudeStreamPosition(index, block.type === 'tool_use' ? block.id : undefined);
                    const draft: ClaudeCanonicalDraft = {
                        draft_block_id: `${prepared.response_turn_id}:claude:${index}`,
                        native_position: position,
                        kind:
                            block.type === 'thinking' ? 'reasoning' : block.type === 'tool_use' ? 'tool_call' : 'text',
                        text: '',
                        tool_argument_fragments: [],
                    };
                    drafts.set(index, draft);
                    await writer.startBlock({
                        draft_block_id: draft.draft_block_id,
                        native_position: position,
                        block:
                            block.type === 'thinking'
                                ? { type: 'reasoning', visibility: 'display' }
                                : block.type === 'tool_use'
                                  ? {
                                        type: 'tool_call',
                                        executor: 'application',
                                        call_id: block.id,
                                        tool_name: block.name,
                                    }
                                  : { type: 'text' },
                    });
                    if (block.type === 'text' && block.text.length > 0) {
                        draft.text = block.text;
                        await writer.text({
                            draft_block_id: draft.draft_block_id,
                            native_position: position,
                            text: block.text,
                        });
                    } else if (block.type === 'thinking' && block.thinking.length > 0) {
                        draft.text = block.thinking;
                        await writer.reasoning({
                            draft_block_id: draft.draft_block_id,
                            native_position: position,
                            text: block.thinking,
                        });
                    } else if (block.type === 'tool_use') {
                        const initialInput = providerJsonValue(block.input);
                        if (!isEmptyJsonObject(initialInput)) {
                            draft.tool_argument_snapshot = initialInput;
                            await writer.toolArgumentsSnapshot({
                                draft_block_id: draft.draft_block_id,
                                native_position: position,
                                value: initialInput,
                            });
                        }
                    }
                }
            } else if (event.type === 'content_block_delta') {
                const draft = drafts.get(event.index);
                if (draft === undefined) {
                    if (event.delta.type !== 'signature_delta') {
                        throw new Error(`Claude stream delta has no draft at content index ${event.index}`);
                    }
                    return;
                }
                if (event.delta.type === 'text_delta') {
                    draft.text += event.delta.text;
                    await writer.text({
                        draft_block_id: draft.draft_block_id,
                        native_position: draft.native_position,
                        text: event.delta.text,
                    });
                } else if (event.delta.type === 'thinking_delta') {
                    draft.text += event.delta.thinking;
                    await writer.reasoning({
                        draft_block_id: draft.draft_block_id,
                        native_position: draft.native_position,
                        text: event.delta.thinking,
                    });
                } else if (event.delta.type === 'input_json_delta') {
                    draft.text += event.delta.partial_json;
                    draft.tool_argument_fragments.push(event.delta.partial_json);
                    await writer.toolArgumentsFragment({
                        draft_block_id: draft.draft_block_id,
                        native_position: draft.native_position,
                        fragment: event.delta.partial_json,
                    });
                }
            } else if (event.type === 'message_delta') {
                logClaudeTruncation(logger, event.delta.stop_reason, { provider, model: options.model });
            }
        },
        finalize: async () => {
            if (responseStream === undefined) throw new Error('Claude stream ended before transport initialization');
            const finalMessage = await responseStream.finalMessage();
            const finalTools = collectClaudeTools(finalMessage.content);
            const rawDecoded = await decodeClaudeCanonicalResponse(finalMessage, prepared);
            const normalized =
                !finalTools?.length && options.result_schema
                    ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
                    : undefined;
            let decoded =
                normalized?.status === 'valid'
                    ? await decodeClaudeCanonicalResponse(finalMessage, prepared, normalized.structured_output)
                    : rawDecoded;
            if (normalized?.status === 'invalid') {
                decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
            }
            const document = await appendClaudeCanonicalResponseWithProcessing(prepared, decoded);
            const response = createCanonicalExecutionResponse(document, prepared.runtime.response_operation_id, {
                service_tier: claudeServiceTier(finalMessage.usage as AnthropicUsageLike),
                ...(options.include_original_response ? { original_response: finalMessage } : {}),
            });
            return {
                decoded,
                response,
                prepare_reconciliation: async () => {
                    const positions = claudeSemanticPositions(finalMessage);
                    const rawBlocks = claudeSemanticBlocks(rawDecoded, prepared.response_turn_id);
                    if (rawBlocks.length !== positions.length) {
                        throw new Error('Claude stream decode does not match terminal native content positions');
                    }
                    const orderedDrafts = positions.map((position) =>
                        [...drafts.values()].find(
                            (draft) => JSON.stringify(draft.native_position) === JSON.stringify(position),
                        ),
                    );
                    if (orderedDrafts.some((draft) => draft === undefined)) {
                        throw new Error('Claude terminal response has no matching native stream draft');
                    }
                    const completeDrafts = orderedDrafts as ClaudeCanonicalDraft[];
                    for (const [index, block] of rawBlocks.entries()) {
                        const draft = completeDrafts[index];
                        if (draft === undefined)
                            throw new Error('Claude terminal response has no matching stream draft');
                        assertClaudeToolDraftArguments(draft, block);
                    }
                    const itemMappings = rawBlocks.flatMap((block, index) => {
                        const position = positions[index];
                        if (position === undefined) return [];
                        return [
                            { canonical_id: block.id, native_position: position, kind: 'block' as const },
                            ...(block.type === 'tool_call'
                                ? [{ canonical_id: block.call_id, native_position: position, kind: 'call' as const }]
                                : []),
                        ];
                    });
                    const transformations = [];
                    const reconciliations = [];
                    if (normalized?.status === 'valid') {
                        const sources = rawBlocks.filter((block) => block.type === 'text');
                        const result = claudeSemanticBlocks(decoded, prepared.response_turn_id).find(
                            (block) => block.type === 'json',
                        );
                        if (sources.length === 0 || result?.type !== 'json') {
                            throw new Error('Claude structured stream is missing source or result blocks');
                        }
                        const proof = await createStructuredOutputTransformationProof({
                            id: `${prepared.generation_id}:structured-output`,
                            source_blocks: sources,
                            result_block: result,
                        });
                        transformations.push(proof);
                        const sourceDrafts = rawBlocks.flatMap((block, index) =>
                            block.type === 'text' && completeDrafts[index] !== undefined ? [completeDrafts[index]] : [],
                        );
                        reconciliations.push({
                            draft_block_ids: sourceDrafts.map((draft) => draft.draft_block_id),
                            native_positions: sourceDrafts.map((draft) => draft.native_position),
                            committed_block_ids: [result.id],
                            disposition: 'structured_output' as const,
                            transformation_id: proof.id,
                        });
                    }
                    for (const [index, block] of rawBlocks.entries()) {
                        if (normalized?.status === 'valid' && block.type === 'text') continue;
                        const draft = completeDrafts[index];
                        if (draft === undefined) throw new Error('Claude direct reconciliation has no draft');
                        reconciliations.push({
                            draft_block_ids: [draft.draft_block_id],
                            native_positions: [draft.native_position],
                            committed_block_ids: [block.id],
                            disposition: 'direct' as const,
                        });
                    }
                    const decodedWithEvidence = {
                        ...decoded,
                        stream_evidence: { item_mappings: itemMappings, transformations },
                    };
                    return {
                        decoded: decodedWithEvidence,
                        reconciliations,
                        deliver_final_events: async (writer) => {
                            if (decodedWithEvidence.generation.usage !== undefined) {
                                await writer.usage(decodedWithEvidence.generation.usage);
                            }
                            for (const [index, block] of rawBlocks.entries()) {
                                const draft = completeDrafts[index];
                                if (draft === undefined) continue;
                                if (
                                    block.type === 'tool_call' &&
                                    block.arguments.type !== 'invalid' &&
                                    draft.tool_argument_fragments.length === 0 &&
                                    draft.tool_argument_snapshot === undefined
                                ) {
                                    await writer.toolArgumentsSnapshot({
                                        draft_block_id: draft.draft_block_id,
                                        native_position: draft.native_position,
                                        value: toolArgumentsForModel(block.arguments),
                                    });
                                }
                                await writer.finishBlock({
                                    draft_block_id: draft.draft_block_id,
                                    native_position: draft.native_position,
                                    outcome:
                                        block.type === 'tool_call' && block.arguments.type === 'invalid'
                                            ? 'malformed'
                                            : decodedWithEvidence.generation.status === 'cancelled'
                                              ? 'interrupted'
                                              : decodedWithEvidence.generation.status === 'failed'
                                                ? 'failed'
                                                : 'native_complete',
                                });
                            }
                            await writer.finish({
                                outcome:
                                    decodedWithEvidence.generation.status === 'cancelled'
                                        ? 'interrupted'
                                        : decodedWithEvidence.generation.status === 'failed'
                                          ? 'failed'
                                          : 'completed',
                                finish_reason: decodedWithEvidence.generation.finish_reason,
                                ...(claudeServiceTier(finalMessage.usage as AnthropicUsageLike) === undefined
                                    ? {}
                                    : {
                                          service_tier: claudeServiceTier(
                                              finalMessage.usage as AnthropicUsageLike,
                                          ) as string,
                                      }),
                            });
                        },
                    };
                },
                ...(normalized?.status === 'valid' && options.result_schema !== undefined
                    ? { result_schema: options.result_schema }
                    : {}),
            };
        },
        abort: () => {
            abortController.abort();
            responseStream?.abort();
        },
        close: () => transportOptions?.signal?.removeEventListener('abort', forwardAbort),
    });
    await publishCanonicalPreparedRequest(prepared, options);
    if (transportOptions?.signal?.aborted) forwardAbort();
    else transportOptions?.signal?.addEventListener('abort', forwardAbort, { once: true });
    return eventStream;
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
