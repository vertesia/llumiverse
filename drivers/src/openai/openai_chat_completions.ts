import type {
    CanonicalProjectedRequestMeasurement,
    CanonicalRequestProjectionCompiler,
    ExecutionOptions,
} from '@llumiverse/common';
import { ModelOptionsSchema } from '@llumiverse/common/schemas';
import {
    type ToolDefinition as CanonicalToolDefinition,
    type ConversationDocument,
    type ConversationModelSwitchProjection,
    type ConversationPreparedRequestRecord,
    createStructuredOutputTransformationProof,
    type DecodedConversationResponse,
    fingerprintJson,
    type IndexedConversationSelectedContext,
    IndexedConversationSelectedContextSchema,
    type JsonObject,
    type JsonValue,
    type ModelTarget,
    type NativeStreamPosition,
    parseConversationDocument,
    parseConversationPreparedRequestRecord,
    preflightJsonInput,
    type RequestSourceWorkingSet,
    RequestSourceWorkingSetSchema,
    type ResolvedConversationRuntimeContext,
    ResolvedConversationRuntimeContextSchema,
} from '@llumiverse/conversation';
import { ModelTargetSchema } from '@llumiverse/conversation/schemas';
import {
    type AIModel,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type CanonicalExecutionResponse,
    type CanonicalHostCapabilities,
    type CanonicalModelSwitchProjectionControls,
    type CanonicalStreamOpenOptions,
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    type CompletionStream,
    canonicalToolSelectionPolicy,
    createCanonicalExecutionResponse,
    type DriverCompletionStream,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionResponse,
    type ExecutionTokenUsage,
    FallbackCanonicalExecutionEventStream,
    getConversationMeta,
    incrementConversationTurn,
    isDedicatedInferenceModel,
    isEmbeddingModel,
    type JSONObject,
    type JSONSchema,
    legacyCompletionFromCanonicalExecution,
    ModelType,
    markCanonicalAcceptedRecovery,
    normalizeEmbeddingsOptions,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
    ownCanonicalHostCapabilities,
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
import { transformSSEStream } from '@llumiverse/core/async';
import { FallbackCompletionStream } from '@llumiverse/core/driver';
import OpenAI from 'openai';
import { canonicalNativeExecutionEventStream } from '../conversation/canonical-execution-event-stream.js';
import {
    assertAcceptedCanonicalRequest,
    canonicalConversationTurnNumber,
    canonicalToolSelectionTargetOptions,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
    selectedCanonicalTurns,
} from '../conversation/canonical-runtime.js';
import {
    normalizeDecodedStructuredOutputForSchema,
    rejectDecodedStructuredOutput,
} from '../conversation/structured-output.js';
import type { OpenAIChatCompletionsDriverOptions, OpenAIChatCompletionsProtocolOptions } from '../driver-options.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';
import { createToolChoiceConfigurationError } from '../shared/tool-choice-error.js';
import {
    executeOpenAIAudioCanonical,
    executeOpenAIAudioRequest,
    openAIAudioTask,
    openAIInputAudioPart,
    streamOpenAIAudioCanonicalEvents,
} from './audio.js';
import {
    boundedCanonicalProjectionOperation,
    projectCanonicalRequestMeasurement,
} from './canonical-request-measurement.js';
import { getOpenAIExtraBody, mergeOpenAIExtraBody } from './extra_body.js';
import { OpenAICompatibleDriverBase } from './openai_compatible.js';
import {
    appendOpenAIChatCanonicalResponseWithProcessing,
    assertOpenAIChatSelectedPreparedRequestEvidence,
    compileOpenAIChatCompletionsConversation,
    compileOpenAIChatIndexedSelectedText,
    compileOpenAIChatProspectiveJsonMinification,
    compileOpenAIChatSelectedWorkingSet,
    createOpenAIChatIndexedTextRequestReceipt,
    decodeOpenAIChatCanonicalResponse,
    finalizeOpenAIChatPreparedRequest,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    type PreparedOpenAIChatConversation,
    prepareOpenAIChatCanonicalContext,
    prepareOpenAIChatCanonicalState,
} from './openai-chat-conversation-adapter.js';
import { formatOpenAISchema, limitedSchemaFormat } from './schema.js';
import { type ChatCompletionsUsage, mapOpenAIChatCompletionsUsage } from './usage.js';

export type { OpenAIChatCompletionsDriverOptions, OpenAIChatCompletionsProtocolOptions } from '../driver-options.js';

import { hydrateCanonicalSelectedImageAssets } from '../conversation/canonical-host-images.js';

export { compileOpenAIChatIndexedSelectedText } from './openai-chat-conversation-adapter.js';

type OpenAIChatServiceTier = OpenAI.Chat.ChatCompletionCreateParams['service_tier'];

function asOpenAIChatServiceTier(serviceTier?: string): OpenAIChatServiceTier {
    // The public option deliberately accepts future provider values that may predate the installed SDK union.
    return serviceTier as OpenAIChatServiceTier;
}

export type OpenAIChatCompletionsTextPart = OpenAI.Chat.ChatCompletionContentPartText;
export type OpenAIChatCompletionsImageUrlPart = OpenAI.Chat.ChatCompletionContentPartImage;
export type OpenAIChatCompletionsContentPart =
    | OpenAIChatCompletionsTextPart
    | OpenAIChatCompletionsImageUrlPart
    | OpenAI.Chat.ChatCompletionContentPartInputAudio;
export type OpenAIChatCompletionsToolCall = OpenAI.Chat.ChatCompletionMessageFunctionToolCall;
export type OpenAIChatCompletionsToolDefinition = OpenAI.Chat.ChatCompletionTool;

export type OpenAIChatProviderReplay = JsonObject & {
    provider: string;
    protocol: string;
    adapter_version: string;
    payload: JsonValue;
};

export type OpenAIChatCompletionsMessage = {
    role: string;
    content?: string | null | OpenAIChatCompletionsContentPart[];
    /**
     * Required for tool role messages - references the tool_call.id from the assistant's message.
     * Per OpenAI API spec: https://platform.openai.com/docs/api-reference/chat/messages#message-role
     */
    tool_call_id?: string;
    /** Internal canonical-ingestion status; removed when building the provider request. */
    tool_result_status?: 'success' | 'error' | 'cancelled' | 'denied';
    /**
     * Tool calls from assistant messages - stored and sent back with tool results.
     */
    tool_calls?: OpenAIChatCompletionsToolCall[];
    /** Provider-native reasoning fields used by OpenAI-compatible APIs for replay. */
    reasoning_content?: string | null;
    reasoning?: string | null;
    /** Provider-scoped opaque replay. Never serialized by the OpenAI transport. */
    provider_replay?: OpenAIChatProviderReplay;
};

export type OpenAIChatCompletionsRequestMessage = {
    role: string;
    content?: string | null | OpenAIChatCompletionsContentPart[];
    tool_call_id?: string;
    tool_calls?: OpenAIChatCompletionsToolCall[];
    reasoning_content?: string | null;
    reasoning?: string | null;
    provider_replay?: OpenAIChatProviderReplay;
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

type OpenAIChatCompletionsUsage = ChatCompletionsUsage & {
    /** Provider-native usage retained when this compatibility protocol normalizes field names. */
    provider_usage?: JsonValue;
};
type OpenAIChatCompletionsResponseMessage = Omit<
    Partial<OpenAI.Chat.ChatCompletionMessage>,
    'content' | 'tool_calls'
> & {
    role?: string;
    content?: string | null | OpenAIChatCompletionsContentPart[];
    reasoning_content?: string | null;
    reasoning?: string | null;
    provider_replay?: OpenAIChatProviderReplay;
    tool_calls?: OpenAI.Chat.ChatCompletionMessageToolCall[];
};
type OpenAIChatCompletionsResponseChoice = Omit<
    OpenAI.Chat.ChatCompletion.Choice,
    'message' | 'finish_reason' | 'logprobs'
> & {
    message: OpenAIChatCompletionsResponseMessage;
    finish_reason?: string | null;
    logprobs?: unknown;
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
    provider_replay?: OpenAIChatProviderReplay;
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

type OpenAIChatCompletionsStreamChoice = Omit<
    OpenAI.Chat.ChatCompletionChunk.Choice,
    'delta' | 'finish_reason' | 'logprobs'
> & {
    delta: OpenAIChatCompletionsStreamChoiceDelta;
    finish_reason?: string | null;
    logprobs?: unknown;
};

export type OpenAIChatCompletionsStreamResponse = Omit<OpenAI.Chat.ChatCompletionChunk, 'choices' | 'usage'> & {
    choices: OpenAIChatCompletionsStreamChoice[];
    usage?: OpenAIChatCompletionsUsage | null;
};

export interface OpenAIChatCompletionsPrompt {
    messages: OpenAIChatCompletionsMessage[];
    /** Discriminator for drivers that share a `messages` array with other provider prompts. */
    _is_openai_chat_completions?: true;
}

const originalResponseSymbol = Symbol('openai-compatible-original-response');

type OpenAIChatCompletionsResponseWithOriginal = OpenAIChatCompletionsResponse & {
    [originalResponseSymbol]?: unknown;
};

export function preserveOpenAIChatCompletionsOriginalResponse(
    response: OpenAIChatCompletionsResponse,
    originalResponse: unknown,
): OpenAIChatCompletionsResponse {
    Object.defineProperty(response, originalResponseSymbol, { value: originalResponse });
    return response;
}

export function normalizeOpenAIChatCompletionsResponse(
    response: OpenAI.Chat.ChatCompletion,
): OpenAIChatCompletionsResponse {
    return {
        id: response.id,
        object: response.object,
        created: response.created,
        model: response.model,
        service_tier: response.service_tier,
        system_fingerprint: response.system_fingerprint,
        choices: response.choices.map((choice) => ({
            index: choice.index,
            finish_reason: choice.finish_reason,
            logprobs: choice.logprobs,
            message: {
                role: choice.message.role,
                content: choice.message.content,
                reasoning_content: getOptionalString(choice.message, 'reasoning_content'),
                reasoning: getOptionalString(choice.message, 'reasoning'),
                tool_calls: choice.message.tool_calls?.flatMap((toolCall) =>
                    toolCall.type === 'function' ? [toolCall] : [],
                ),
            },
        })),
        usage: response.usage ?? undefined,
    };
}

export async function* normalizeOpenAIChatCompletionsStream(
    stream: AsyncIterable<OpenAI.Chat.ChatCompletionChunk>,
): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    for await (const chunk of stream) {
        yield {
            id: chunk.id,
            object: chunk.object,
            created: chunk.created,
            model: chunk.model,
            service_tier: chunk.service_tier,
            system_fingerprint: chunk.system_fingerprint,
            choices: chunk.choices.map((choice) => ({
                index: choice.index,
                finish_reason: choice.finish_reason,
                logprobs: choice.logprobs,
                delta: {
                    role: choice.delta.role,
                    content: choice.delta.content,
                    reasoning_content: getOptionalString(choice.delta, 'reasoning_content'),
                    reasoning: getOptionalString(choice.delta, 'reasoning'),
                    tool_calls: choice.delta.tool_calls?.map((toolCall) => ({
                        index: toolCall.index,
                        id: toolCall.id,
                        type: toolCall.type,
                        function: toolCall.function,
                    })),
                },
            })),
            usage: chunk.usage ?? undefined,
        };
    }
}

function getOptionalString(value: object, key: string): string | null | undefined {
    if (!(key in value)) return undefined;
    const candidate = (value as Record<string, unknown>)[key];
    return typeof candidate === 'string' || candidate === null ? candidate : undefined;
}

type StreamChunk = string | Uint8Array;
type ReadableAsyncStream = AsyncIterable<StreamChunk>;
type StreamingOpenAIToolUse = ToolUse<unknown> & { _actual_id?: string };

interface OpenAIChatCanonicalDraft {
    draft_block_id: string;
    native_position: NativeStreamPosition;
    kind: 'text' | 'reasoning' | 'tool_call';
    tool_index?: number;
}

function openAIChatStreamPosition(path: Array<string | number>): NativeStreamPosition {
    return { protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL, path };
}

async function* openAIChatNativeSSE(stream: ReadableStream): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    for await (const event of stream as unknown as AsyncIterable<{ type: string; data?: string }>) {
        if (event.type !== 'event' || event.data === undefined || event.data === '[DONE]') continue;
        yield JSON.parse(event.data) as OpenAIChatCompletionsStreamResponse;
    }
}

function openAIChatSemanticBlocks(decoded: DecodedConversationResponse, turnId: string) {
    const turn = decoded.turns.find((candidate) => candidate.id === turnId);
    if (turn?.kind !== 'agent') throw new Error('OpenAI Chat stream decode has no generated agent turn');
    return turn.blocks.filter((block) => block.type !== 'native_replay');
}

async function finalizeOpenAIChatStreamResponse(input: {
    response: OpenAIChatCompletionsResponse;
    prepared: PreparedOpenAIChatConversation;
    finish_reason: string;
    options: ExecutionOptions;
}) {
    const message = input.response.choices[0]?.message;
    if (message === undefined) throw new Error('Chat Completions stream has no terminal assistant message');
    const hasTools = (message.tool_calls?.length ?? 0) > 0;
    const rawDecoded = await decodeOpenAIChatCanonicalResponse(input.response, input.prepared, input.finish_reason);
    const normalized =
        !hasTools && input.options.result_schema
            ? normalizeDecodedStructuredOutputForSchema(rawDecoded, input.options.result_schema)
            : undefined;
    let decoded =
        normalized?.status === 'valid'
            ? await decodeOpenAIChatCanonicalResponse(
                  input.response,
                  input.prepared,
                  input.finish_reason,
                  normalized.structured_output,
              )
            : rawDecoded;
    if (normalized?.status === 'invalid') decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
    const document = await appendOpenAIChatCanonicalResponseWithProcessing(input.prepared, decoded);
    return {
        raw_decoded: rawDecoded,
        decoded,
        normalized,
        response: createCanonicalExecutionResponse(document, input.prepared.runtime.response_operation_id, {
            ...(input.response.service_tier == null ? {} : { service_tier: input.response.service_tier }),
        }),
    };
}

async function streamToString(stream: ReadableAsyncStream): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks).toString('utf-8');
}

export function updateOpenAIChatCompletionsConversation(
    conversation: OpenAIChatCompletionsPrompt | OpenAIChatCompletionsMessage[] | undefined | null,
    prompt: OpenAIChatCompletionsPrompt,
): OpenAIChatCompletionsPrompt {
    // TODO: Remove legacy array-shaped conversation compatibility after 2026-08-14,
    // once all resumable conversations written before the native container migration have expired.
    const baseMessages = Array.isArray(conversation) ? conversation : (conversation?.messages ?? []);
    return {
        _is_openai_chat_completions: true,
        messages: [...baseMessages, ...(prompt.messages || [])],
    };
}

export function prepareOpenAIChatCompletionsConversation(
    conversation: OpenAIChatCompletionsPrompt,
    options: {
        tools?: readonly (Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'> | ToolDefinition)[];
        model?: string;
    },
): OpenAIChatCompletionsPrompt {
    let messages = fixOrphanedOpenAIChatCompletionsToolResults(
        fixOrphanedOpenAIChatCompletionsToolUse(conversation.messages),
    );
    messages = prepareOpenAIChatCompletionsReasoning(messages, options.model);
    if (!options.tools || options.tools.length === 0) {
        messages = convertOpenAIChatCompletionsToolMessagesToText(messages);
    }
    return { ...conversation, messages };
}

type OpenAIChatReasoningReplayPolicy = 'none' | 'omit' | 'active_tool_turn' | 'full_tool_history';

function getOpenAIChatReasoningReplayPolicy(model: string | undefined): OpenAIChatReasoningReplayPolicy {
    const modelId = model?.toLowerCase() ?? '';
    if (/deepseek-(?:r1|reasoner)|deepseek\.r1/.test(modelId)) return 'omit';
    if (/deepseek-v4(?:-|$)/.test(modelId)) return 'full_tool_history';
    if (/deepseek-v3\.2(?:-|$)/.test(modelId)) return 'active_tool_turn';
    return 'none';
}

function withoutOpenAIChatReasoning(message: OpenAIChatCompletionsMessage): OpenAIChatCompletionsMessage {
    if (message.reasoning_content === undefined && message.reasoning === undefined) return message;
    const { reasoning_content: _reasoningContent, reasoning: _reasoning, ...rest } = message;
    return rest;
}

function prepareOpenAIChatCompletionsReasoning(
    messages: OpenAIChatCompletionsMessage[],
    model: string | undefined,
): OpenAIChatCompletionsMessage[] {
    const policy = getOpenAIChatReasoningReplayPolicy(model);
    if (policy === 'omit') return messages.map(withoutOpenAIChatReasoning);
    if (policy !== 'active_tool_turn') return messages;

    let currentTurnStart = -1;
    for (let index = messages.length - 1; index >= 0; index--) {
        if (messages[index].role === 'user') {
            currentTurnStart = index;
            break;
        }
    }
    if (currentTurnStart <= 0) return messages;
    return messages.map((message, index) => (index < currentTurnStart ? withoutOpenAIChatReasoning(message) : message));
}

export function parseOpenAIChatCompletionsToolCalls(
    toolCalls: OpenAIChatCompletionsResponseChoice['message']['tool_calls'],
): ToolUse[] | undefined {
    if (!toolCalls || toolCalls.length === 0) {
        return undefined;
    }
    const functionCalls = toolCalls.filter(
        (toolCall): toolCall is OpenAI.Chat.ChatCompletionMessageFunctionToolCall => toolCall.type === 'function',
    );
    if (functionCalls.length === 0) {
        return undefined;
    }
    return functionCalls.map((tc) => ({
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

function normalizeOpenAIChatCompletionsFinishReason(
    reason: string | null | undefined,
    hasToolUse: boolean = false,
): string | undefined {
    // A provider may include a partial tool-call delta on the token-limit chunk. Preserve
    // truncation so core drops the incomplete call instead of classifying it as malformed JSON.
    if (reason === 'length') return 'length';
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
            if (Array.isArray(message.content) && message.content.some((part) => part.type === 'image_url')) {
                return {
                    role: 'user',
                    content: [{ type: 'text', text: `Tool result ${message.tool_call_id}:` }, ...message.content],
                };
            }
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

export function extractOpenAIChatCompletionsResults(
    source:
        | Pick<OpenAIChatCompletionsResponseChoice['message'], 'content' | 'reasoning_content' | 'reasoning'>
        | Pick<OpenAIChatCompletionsStreamChoiceDelta, 'content' | 'reasoning_content' | 'reasoning'>
        | undefined,
    includeThoughts = true,
): CompletionResult[] {
    if (!source) return [];

    const results = splitOpenAIChatCompletionsThinkBlocks(
        extractOpenAIChatCompletionsContentText(source.content),
        includeThoughts,
    );
    const reasoning = extractOpenAIChatCompletionsReasoningText(source);
    if (includeThoughts && reasoning) {
        results.unshift({ type: 'thoughts', value: reasoning });
    }
    return results;
}

export function splitOpenAIChatCompletionsThinkBlocks(value: string, includeThoughts = true): CompletionResult[] {
    const results: CompletionResult[] = [];
    const pattern = /<think\b[^>]*>([\s\S]*?)(?:<\/think>|$)/gi;
    let offset = 0;
    for (const match of value.matchAll(pattern)) {
        const index = match.index ?? 0;
        if (index > offset) {
            const text = value.slice(offset, index).trim();
            if (text) results.push({ type: 'text', value: text });
        }
        if (includeThoughts && match[1]) {
            results.push({ type: 'thoughts', value: match[1] });
        }
        offset = index + match[0].length;
    }
    if (offset < value.length) {
        const text = value.slice(offset).trim();
        if (text) results.push({ type: 'text', value: text });
    }
    return results;
}

class OpenAIThinkStreamProjector {
    private buffer = '';
    private inThoughts = false;
    private trimNextTextStart = false;

    push(value: string, final = false): CompletionResult[] {
        this.buffer += value;
        const results: CompletionResult[] = [];

        while (this.buffer) {
            if (this.inThoughts) {
                const closeIndex = this.buffer.toLowerCase().indexOf('</think>');
                if (closeIndex >= 0) {
                    this.append(results, 'thoughts', this.buffer.slice(0, closeIndex));
                    this.buffer = this.buffer.slice(closeIndex + '</think>'.length);
                    this.inThoughts = false;
                    this.trimNextTextStart = true;
                    continue;
                }
                if (final) {
                    this.append(results, 'thoughts', this.buffer);
                    this.buffer = '';
                    break;
                }
                const retained = this.partialTagSuffixLength('</think>');
                this.append(results, 'thoughts', this.buffer.slice(0, this.buffer.length - retained));
                this.buffer = this.buffer.slice(this.buffer.length - retained);
                break;
            }

            const open = /<think\b[^>]*>/i.exec(this.buffer);
            if (open?.index !== undefined) {
                this.append(results, 'text', this.buffer.slice(0, open.index).trimEnd());
                this.buffer = this.buffer.slice(open.index + open[0].length);
                this.inThoughts = true;
                continue;
            }
            if (final) {
                this.append(results, 'text', this.buffer);
                this.buffer = '';
                break;
            }

            const lower = this.buffer.toLowerCase();
            const lastOpen = lower.lastIndexOf('<');
            const candidate = lastOpen >= 0 ? lower.slice(lastOpen) : '';
            const retainCandidate =
                candidate.length > 0 &&
                ('<think'.startsWith(candidate) || (candidate.startsWith('<think') && !candidate.includes('>')));
            const retained = retainCandidate ? candidate.length : 0;
            this.append(results, 'text', this.buffer.slice(0, this.buffer.length - retained));
            this.buffer = this.buffer.slice(this.buffer.length - retained);
            break;
        }

        return results;
    }

    private partialTagSuffixLength(tag: string): number {
        const lower = this.buffer.toLowerCase();
        for (let length = Math.min(lower.length, tag.length - 1); length > 0; length--) {
            if (tag.startsWith(lower.slice(-length))) return length;
        }
        return 0;
    }

    private append(results: CompletionResult[], type: 'text' | 'thoughts', value: string): void {
        if (type === 'text' && this.trimNextTextStart) {
            value = value.trimStart();
            if (value) this.trimNextTextStart = false;
        }
        if (type === 'text' && !value.trim()) return;
        if (value) results.push({ type, value });
    }
}

export function extractOpenAIChatCompletionsReasoningText(
    source:
        | Pick<OpenAIChatCompletionsResponseChoice['message'], 'reasoning_content' | 'reasoning'>
        | Pick<OpenAIChatCompletionsStreamChoiceDelta, 'reasoning_content' | 'reasoning'>
        | undefined,
): string {
    if (!source) return '';
    const value = source.reasoning_content ?? source.reasoning;
    return typeof value === 'string' ? value : '';
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
    tools:
        | readonly Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'>[]
        | ToolDefinition[]
        | undefined,
    mode: OpenAIChatCompletionsProtocolOptions['toolSchemaMode'] = 'openai_strict',
): OpenAIChatCompletionsToolDefinition[] | undefined {
    if (!tools || tools.length === 0) {
        return undefined;
    }

    return tools.map((tool) => {
        let parameters: JSONSchema | undefined;
        let strict: boolean | undefined;
        if (tool.input_schema) {
            if (mode === 'compatible') {
                parameters = limitedSchemaFormat(tool.input_schema as JSONSchema);
            } else {
                // Reuse the same schema normalization as the OpenAI Responses path:
                // strict mode when possible, limited non-strict schema otherwise.
                const formattedSchema = formatOpenAISchema(tool.input_schema as JSONSchema);
                parameters = formattedSchema.schema;
                strict = formattedSchema.strict;
            }
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
    const converted: OpenAIChatCompletionsRequestMessage[] = [];
    let attachments: OpenAIChatCompletionsContentPart[] = [];
    const flushAttachments = () => {
        if (attachments.length === 0) return;
        converted.push({ role: 'user', content: attachments });
        attachments = [];
    };
    for (const msg of messages) {
        // Complete all parallel tool results before adding ordinary user content.
        if (msg.role !== 'tool') flushAttachments();
        const result: OpenAIChatCompletionsRequestMessage = {
            role: msg.role,
        };

        if (msg.tool_calls && msg.tool_calls.length > 0) {
            result.tool_calls = msg.tool_calls;
        }
        if (msg.reasoning_content !== undefined) {
            result.reasoning_content = msg.reasoning_content;
        }
        if (msg.reasoning !== undefined) {
            result.reasoning = msg.reasoning;
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
            // Preserve interleaved captions and images, including on replay. Tool messages
            // only support text in the SDK/API, so retain an indexed reference at each image's
            // original position and carry its bytes in a following user message.
            let imageIndex = 0;
            const content = msg.content.map((part): OpenAIChatCompletionsContentPart => {
                if (msg.role !== 'tool' || part.type !== 'image_url') return part;
                imageIndex++;
                const label = `Image ${imageIndex} from tool result ${msg.tool_call_id}:`;
                attachments.push({ type: 'text', text: label }, part);
                return { type: 'text', text: `[Image ${imageIndex} attached below]` };
            });

            if (content.length > 0 && content.every((part) => part.type === 'text')) {
                result.content = content.map((part) => part.text).join('\n');
            } else if (content.length > 0) {
                result.content = content;
            }
        }

        if (msg.tool_call_id) {
            result.tool_call_id = msg.tool_call_id;
        }

        converted.push(result);
    }
    flushAttachments();
    return converted;
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
    return projectOpenAIChatCompletionsHistory(conversation, options, getConversationMeta(conversation).turnNumber);
}

export function projectOpenAIChatCompletionsHistory(
    conversation: OpenAIChatCompletionsPrompt,
    options: ExecutionOptions,
    currentTurn: number,
): OpenAIChatCompletionsPrompt {
    const reasoningPolicy = getOpenAIChatReasoningReplayPolicy(options.model);
    if (reasoningPolicy === 'omit') {
        conversation = { ...conversation, messages: conversation.messages.map(withoutOpenAIChatReasoning) };
    }

    const latestAssistant = conversation.messages.findLast((message) => message.role === 'assistant');
    const hasActiveReasoningToolTurn =
        reasoningPolicy === 'active_tool_turn' &&
        !!latestAssistant?.tool_calls?.length &&
        (latestAssistant.reasoning_content != null || latestAssistant.reasoning != null);
    const hasReasoningToolHistory =
        reasoningPolicy === 'full_tool_history' &&
        conversation.messages.some(
            (message) =>
                !!message.tool_calls?.length && (message.reasoning_content != null || message.reasoning != null),
        );
    if (hasActiveReasoningToolTurn || hasReasoningToolHistory) {
        return conversation;
    }

    if (reasoningPolicy === 'active_tool_turn') {
        conversation = { ...conversation, messages: conversation.messages.map(withoutOpenAIChatReasoning) };
    }
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

function getOpenAIChatDriverProvider(driver: unknown): string {
    if (typeof driver !== 'object' || driver === null || !('provider' in driver)) {
        return Providers.openai_compatible;
    }
    return typeof driver.provider === 'string' ? driver.provider : Providers.openai_compatible;
}

function prepareCanonicalOpenAIProjectionFromNative(
    native: OpenAIChatCompletionsPrompt,
    priorNativeMessageCount: number,
    options: ExecutionOptions,
    currentTurn: number,
    tools: readonly (Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'> | ToolDefinition)[],
): OpenAIChatCompletionsPrompt {
    const prior: OpenAIChatCompletionsPrompt = {
        _is_openai_chat_completions: true,
        messages: native.messages.slice(0, priorNativeMessageCount),
    };
    const projectedPrior = projectOpenAIChatCompletionsHistory(prior, options, currentTurn);
    return prepareOpenAIChatCompletionsConversation(
        {
            _is_openai_chat_completions: true,
            messages: [...projectedPrior.messages, ...native.messages.slice(priorNativeMessageCount)],
        },
        { model: options.model, tools },
    );
}

function prepareCanonicalOpenAIProjection(
    prepared: Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>,
    options: ExecutionOptions,
): OpenAIChatCompletionsPrompt {
    return prepareCanonicalOpenAIProjectionFromNative(
        prepared.native_conversation,
        prepared.prior_native_message_count,
        options,
        canonicalConversationTurnNumber(prepared.document),
        prepared.tool_definitions,
    );
}

function canonicalOpenAIUsage(
    prepared: Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>,
): ExecutionTokenUsage | undefined {
    const usage = prepared.accepted_response?.generation.usage;
    if (usage === undefined) return undefined;
    const reported = usage.reported_usage?.find(
        (candidate) => candidate.source === 'provider' && candidate.protocol === OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    );
    const reportedProjection = mapOpenAIChatCompletionsUsage(reportedChatCompletionsUsage(reported?.payload));
    if (reportedProjection !== undefined) return reportedProjection;
    return {
        ...(usage.input_tokens === undefined ? {} : { prompt: usage.input_tokens }),
        ...(usage.output_tokens === undefined ? {} : { result: usage.output_tokens }),
        ...(usage.total_tokens === undefined ? {} : { total: usage.total_tokens }),
        ...(usage.cache_read_tokens === undefined ? {} : { prompt_cached: usage.cache_read_tokens }),
        ...(usage.cache_write_tokens === undefined ? {} : { prompt_cache_write: usage.cache_write_tokens }),
        ...(usage.input_new_tokens === undefined ? {} : { prompt_new: usage.input_new_tokens }),
    };
}

function reportedChatCompletionsUsage(value: unknown): ChatCompletionsUsage | undefined {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return undefined;
    const payload = value as Record<string, unknown>;
    const promptTokens = payload.prompt_tokens;
    const completionTokens = payload.completion_tokens;
    const totalTokens = payload.total_tokens;
    if (typeof promptTokens !== 'number' || typeof completionTokens !== 'number' || typeof totalTokens !== 'number') {
        return undefined;
    }
    const details = payload.prompt_tokens_details;
    if (details !== undefined && details !== null && (typeof details !== 'object' || Array.isArray(details))) {
        return undefined;
    }
    const detailRecord = details as Record<string, unknown> | null | undefined;
    const cachedTokens = detailRecord?.cached_tokens;
    const cacheWriteTokens = detailRecord?.cache_write_tokens;
    if (
        (cachedTokens !== undefined && cachedTokens !== null && typeof cachedTokens !== 'number') ||
        (cacheWriteTokens !== undefined && cacheWriteTokens !== null && typeof cacheWriteTokens !== 'number')
    ) {
        return undefined;
    }
    const cost = payload.cost;
    const isByok = payload.is_byok;
    if (
        (cost !== undefined && cost !== null && typeof cost !== 'number') ||
        (isByok !== undefined && isByok !== null && typeof isByok !== 'boolean')
    ) {
        return undefined;
    }
    return {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: totalTokens,
        ...(details === undefined
            ? {}
            : {
                  prompt_tokens_details:
                      details === null
                          ? null
                          : {
                                ...(cachedTokens == null ? {} : { cached_tokens: cachedTokens }),
                                ...(cacheWriteTokens == null ? {} : { cache_write_tokens: cacheWriteTokens }),
                            },
              }),
        ...(cost === undefined ? {} : { cost }),
        ...(isByok === undefined ? {} : { is_byok: isByok }),
    };
}

function recoverOpenAICompletion(
    prepared: Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>,
    options: ExecutionOptions,
    includeThoughts: boolean,
): Completion {
    const accepted = prepared.accepted_response;
    if (accepted === undefined) throw new Error('No accepted Chat Completions response is available');
    if (options.include_original_response) {
        throw new Error('An idempotently recovered Chat Completions response cannot reconstruct original_response');
    }
    const projection = compileOpenAIChatCompletionsConversation(prepared.document, {
        provider: prepared.provider,
        model: prepared.requested_model,
    });
    const mapping = projection.mappings.find(
        (candidate) => candidate.kind === 'turn' && candidate.canonical_id === accepted.turn.id,
    );
    const match = mapping === undefined ? undefined : /^messages\/(\d+)$/.exec(mapping.native_id);
    const message = match == null ? undefined : projection.conversation.messages[Number(match[1])];
    if (message?.role !== 'assistant') {
        throw new Error(`Accepted Chat Completions turn ${accepted.turn.id} has no native projection`);
    }
    const toolUse = parseOpenAIChatCompletionsToolCalls(message.tool_calls);
    return {
        result: extractOpenAIChatCompletionsResults(message, includeThoughts),
        tool_use: toolUse,
        token_usage: canonicalOpenAIUsage(prepared),
        finish_reason: normalizeOpenAIChatCompletionsFinishReason(accepted.generation.finish_reason, !!toolUse?.length),
        conversation: prepared.document,
    };
}

function recoveredOpenAIStream(completion: Completion): DriverCompletionStream {
    const stream = (async function* (): AsyncIterable<CompletionChunkObject> {
        yield {
            result: completion.result,
            tool_use: completion.tool_use,
            token_usage: completion.token_usage,
            finish_reason: completion.finish_reason,
            service_tier: completion.service_tier,
        };
    })();
    return markCanonicalAcceptedRecovery(
        Object.assign(stream, {
            finalizeConversation: () => completion.conversation,
        }),
    );
}

interface OpenAIChatRequestBinding {
    payload: JsonValue;
    target_options?: JsonObject;
}

/** Shared native request builder: transport, dry measurement and accepted-view attestation use identical fields. */
export function buildOpenAIChatCompletionsPayload(
    conversation: OpenAIChatCompletionsPrompt,
    options: ExecutionOptions,
    protocolOptions: OpenAIChatCompletionsProtocolOptions,
    modelName: string,
    stream: boolean,
    provider: string = Providers.openai_compatible,
    tools:
        | readonly Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'>[]
        | ToolDefinition[]
        | undefined = options.tools,
): OpenAIChatCompletionsPayload {
    const modelOptions = options.model_options as TextFallbackOptions & {
        effort?: 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';
        reasoning_effort?: 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';
        seed?: number;
        service_tier?: string;
        tool_choice?: 'auto' | 'none' | 'any' | 'required';
        /** Internal execution hint supplied after public model-option validation. */
        required_tool_name?: string;
        /** Internal execution hint used with a required named tool. */
        parallel_tool_calls?: boolean;
    };
    const payload: OpenAIChatCompletionsPayload = {
        model: modelName,
        messages: convertToOpenAIChatCompletionsMessages(conversation.messages),
        // Some OpenAI-compatible providers return empty/truncated completions unless a
        // documented or runtime-validated token budget is supplied. Caller options still win.
        max_tokens: modelOptions?.max_tokens ?? protocolOptions.defaultMaxTokens,
        temperature: modelOptions?.temperature,
        top_p: modelOptions?.top_p,
        presence_penalty: modelOptions?.presence_penalty,
        frequency_penalty: modelOptions?.frequency_penalty,
        n: 1,
        stop: modelOptions?.stop_sequence,
        seed: modelOptions?.seed,
        reasoning_effort: modelOptions?.effort ?? modelOptions?.reasoning_effort,
        service_tier: asOpenAIChatServiceTier(modelOptions?.service_tier),
        stream,
    };

    const executionExtraBody = getOpenAIExtraBody(modelOptions);
    if (protocolOptions.extraBody || executionExtraBody) {
        payload.extra_body = { ...protocolOptions.extraBody, ...executionExtraBody };
    }

    const toolsPayload = convertToolsToOpenAIChatCompletionsFormat(tools, protocolOptions.toolSchemaMode);
    const forcedToolChoice =
        typeof modelOptions?.required_tool_name === 'string' ||
        modelOptions?.tool_choice === 'required' ||
        modelOptions?.tool_choice === 'any';
    if (forcedToolChoice && (!toolsPayload || toolsPayload.length === 0)) {
        throw createToolChoiceConfigurationError(
            '[OpenAI Chat Completions API] A required tool choice was requested, but no tools are available.',
            { provider, model: options.model, operation: stream ? 'stream' : 'execute' },
        );
    }
    if (toolsPayload && toolsPayload.length > 0) {
        payload.tools = toolsPayload;
        payload.tool_choice = modelOptions?.required_tool_name
            ? { type: 'function', function: { name: modelOptions.required_tool_name } }
            : modelOptions?.tool_choice === 'any'
              ? 'required'
              : modelOptions?.tool_choice;
        payload.parallel_tool_calls = modelOptions?.parallel_tool_calls;
    }

    if (options.result_schema && protocolOptions.resultSchemaMode !== 'prompt') {
        const formattedSchema = formatOpenAISchema(options.result_schema as JSONSchema);
        payload.response_format = {
            type: 'json_schema',
            json_schema: { name: 'output', strict: formattedSchema.strict, schema: formattedSchema.schema },
        };
    }

    return payload;
}

export abstract class OpenAIChatCompletionsProtocol<DriverT> {
    protected readonly options: OpenAIChatCompletionsProtocolOptions;

    constructor(options: OpenAIChatCompletionsProtocolOptions) {
        this.options = options;
    }

    /** Prepare exact native payload and receipt from a bounded indexed selection; host durability still gates dispatch. */
    async prepareIndexedTextRequest(
        input: {
            selection: IndexedConversationSelectedContext;
            runtime: ResolvedConversationRuntimeContext;
            options: ExecutionOptions;
            provider: string;
            stream: boolean;
            signal?: AbortSignal;
        },
        hostCapabilities?: CanonicalHostCapabilities,
    ) {
        const ownedHostCapabilities = ownCanonicalHostCapabilities(hostCapabilities);
        if ('resolve_canonical_asset' in input.options)
            throw new TypeError('Indexed asset resolvers must use per-call host capabilities');
        // Own every JSON input used after receipt hashing yields. ExecutionOptions may also contain
        // callbacks, so copy only the request controls consumed by this indexed compiler.
        if (!preflightJsonInput(input.selection).success || !preflightJsonInput(input.runtime).success) {
            throw new TypeError('Indexed OpenAI source or runtime is not bounded JSON');
        }
        const selection = IndexedConversationSelectedContextSchema.parse(structuredClone(input.selection));
        const runtime = ResolvedConversationRuntimeContextSchema.parse(structuredClone(input.runtime));
        const ownField = <Key extends keyof ExecutionOptions>(key: Key): ExecutionOptions[Key] => {
            const descriptor = Object.getOwnPropertyDescriptor(input.options, key);
            if (descriptor !== undefined && !Object.hasOwn(descriptor, 'value')) {
                throw new TypeError(`Indexed OpenAI option ${key} must be an owned value`);
            }
            return descriptor?.value;
        };
        const model = ownField('model');
        const modelOptions = ownField('model_options');
        const resultSchema = ownField('result_schema');
        const stripImagesAfterTurns = ownField('stripImagesAfterTurns');
        const stripHeartbeatsAfterTurns = ownField('stripHeartbeatsAfterTurns');
        const stripTextMaxTokens = ownField('stripTextMaxTokens');
        const controls = {
            model,
            ...(modelOptions === undefined ? {} : { model_options: modelOptions }),
            ...(resultSchema === undefined ? {} : { result_schema: resultSchema }),
            ...(stripImagesAfterTurns === undefined ? {} : { stripImagesAfterTurns }),
            ...(stripHeartbeatsAfterTurns === undefined ? {} : { stripHeartbeatsAfterTurns }),
            ...(stripTextMaxTokens === undefined ? {} : { stripTextMaxTokens }),
        };
        if (!preflightJsonInput(controls).success) throw new TypeError('Indexed OpenAI controls are not bounded JSON');
        const options: ExecutionOptions = structuredClone(controls);
        // Hydration is the first await. Capture constructor-owned native controls before it,
        // including nested extra-body values and the configured model alias.
        const ownProtocolField = <Key extends keyof OpenAIChatCompletionsProtocolOptions>(
            key: Key,
        ): OpenAIChatCompletionsProtocolOptions[Key] => {
            const descriptor = Object.getOwnPropertyDescriptor(this.options, key);
            if (descriptor !== undefined && !Object.hasOwn(descriptor, 'value'))
                throw new TypeError(`Indexed OpenAI protocol option ${key} must be an owned value`);
            return descriptor?.value;
        };
        const defaultMaxTokens = ownProtocolField('defaultMaxTokens');
        const extraBody = ownProtocolField('extraBody');
        const resultSchemaMode = ownProtocolField('resultSchemaMode');
        const toolSchemaMode = ownProtocolField('toolSchemaMode');
        const protocolControls = {
            ...(defaultMaxTokens === undefined ? {} : { defaultMaxTokens }),
            ...(extraBody === undefined ? {} : { extraBody }),
            ...(resultSchemaMode === undefined ? {} : { resultSchemaMode }),
            ...(toolSchemaMode === undefined ? {} : { toolSchemaMode }),
        };
        if (!preflightJsonInput(protocolControls).success)
            throw new TypeError('Indexed OpenAI protocol controls are not bounded JSON');
        const ownedProtocolControls = structuredClone(protocolControls);
        const actualModel = this.getModelName(options);
        const provider = input.provider;
        const stream = input.stream;
        const resolver = ownedHostCapabilities?.resolve_canonical_asset;
        const signal = input.signal;
        signal?.throwIfAborted();
        const imageIds = new Set<string>();
        for (const projection of [
            ...selection.turns,
            ...(selection.replacement_turns ?? []).map((value) => value.projection),
        ]) {
            for (const block of projection.selected_blocks) {
                for (const item of block.type === 'tool_result' ? block.content : [block])
                    if (item.type === 'image') imageIds.add(item.asset_id);
            }
        }
        const nativeAssets = await hydrateCanonicalSelectedImageAssets({
            label: 'Indexed native preparation',
            assets: selection.assets,
            selected_ids: imageIds,
            resolve_asset: resolver,
            signal,
            hydrated: new Map(),
            native_external: () => false,
        });
        signal?.throwIfAborted();
        const compiled = compileOpenAIChatIndexedSelectedText(
            { ...selection, assets: nativeAssets },
            {
                provider,
                model: options.model,
            },
        );
        const tools = selection.context.active_tool_definition_ids.map((id) => {
            const tool = Object.hasOwn(selection.tool_definitions, id) ? selection.tool_definitions[id] : undefined;
            if (!tool) throw new Error(`Indexed OpenAI Chat tool ${id} is unavailable`);
            return tool;
        });
        const nativeConversation = prepareCanonicalOpenAIProjectionFromNative(
            compiled.conversation,
            compiled.conversation.messages.length,
            options,
            selection.source_turn_count,
            tools,
        );
        const builtPayload = buildOpenAIChatCompletionsPayload(
            nativeConversation,
            options,
            ownedProtocolControls,
            actualModel,
            stream,
            provider,
            tools,
        );
        if (builtPayload.extra_body !== undefined && !preflightJsonInput(builtPayload.extra_body).success) {
            throw new TypeError('Indexed OpenAI extra body is not bounded JSON');
        }
        // Protocol configuration is constructor-owned and can contain nested mutable extra-body values.
        // Keep the returned payload and hashed receipt bound to one owned value across the await below.
        const payload = structuredClone(builtPayload);
        const binding = this.requestBinding(payload, options, provider);
        const targetOptions = canonicalToolSelectionTargetOptions(
            binding.target_options,
            canonicalToolSelectionPolicy(options),
        );
        const receipt = await createOpenAIChatIndexedTextRequestReceipt({
            selection,
            runtime,
            target: {
                provider,
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                model: options.model,
                adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                ...(targetOptions === undefined ? {} : { options: targetOptions }),
            },
            native_payload: binding.payload,
            mappings: compiled.mappings,
        });
        return {
            status: 'awaiting_durable_prepared_record' as const,
            source: selection.source,
            native_conversation: nativeConversation,
            payload,
            receipt,
        };
    }

    /** Dispatch only an exact, already committed indexed text request through this configured transport. */
    async executeCommittedIndexedTextRequest(
        driver: DriverT,
        input: {
            selection: IndexedConversationSelectedContext;
            record: ConversationPreparedRequestRecord;
            options: ExecutionOptions;
            provider: string;
            assert_committed: () => Promise<void>;
            signal?: AbortSignal;
        },
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<DecodedConversationResponse> {
        const ownedHostCapabilities = ownCanonicalHostCapabilities(hostCapabilities);
        if ('resolve_canonical_asset' in input.options)
            throw new TypeError('Indexed asset resolvers must use per-call host capabilities');
        const selection = IndexedConversationSelectedContextSchema.parse(structuredClone(input.selection));
        const record = parseConversationPreparedRequestRecord(structuredClone(input.record));
        const ownOption = <Key extends keyof ExecutionOptions>(key: Key): ExecutionOptions[Key] => {
            const descriptor = Object.getOwnPropertyDescriptor(input.options, key);
            if (descriptor !== undefined && !Object.hasOwn(descriptor, 'value')) {
                throw new TypeError(`Indexed OpenAI option ${key} must be an owned value`);
            }
            return descriptor?.value;
        };
        const model = ownOption('model');
        if (typeof model !== 'string') throw new TypeError('Indexed OpenAI model is unavailable');
        const modelOptions = ownOption('model_options');
        const resultSchema = ownOption('result_schema');
        const stripImagesAfterTurns = ownOption('stripImagesAfterTurns');
        const stripHeartbeatsAfterTurns = ownOption('stripHeartbeatsAfterTurns');
        const stripTextMaxTokens = ownOption('stripTextMaxTokens');
        const controls = {
            model,
            ...(modelOptions === undefined ? {} : { model_options: modelOptions }),
            ...(resultSchema === undefined ? {} : { result_schema: resultSchema }),
            ...(stripImagesAfterTurns === undefined ? {} : { stripImagesAfterTurns }),
            ...(stripHeartbeatsAfterTurns === undefined ? {} : { stripHeartbeatsAfterTurns }),
            ...(stripTextMaxTokens === undefined ? {} : { stripTextMaxTokens }),
        };
        if (!preflightJsonInput(controls).success) throw new TypeError('Indexed OpenAI controls are not bounded JSON');
        const options: ExecutionOptions = structuredClone(controls);
        const provider = input.provider;
        const assertCommitted = input.assert_committed;
        const signal = input.signal;
        if (options.result_schema !== undefined) {
            throw new Error('Indexed OpenAI Chat text transport does not support structured output');
        }
        const prepared = await this.prepareIndexedTextRequest(
            {
                selection,
                runtime: record.runtime,
                options,
                provider,
                stream: false,
                signal,
            },
            ownedHostCapabilities,
        );
        if (
            record.source.conversation_id !== selection.source.conversation_id ||
            record.source.revision !== selection.source.revision ||
            (await fingerprintJson(record.request_receipt)) !== (await fingerprintJson(prepared.receipt))
        ) {
            throw new Error('Indexed native request differs from its durably prepared receipt');
        }
        signal?.throwIfAborted();
        await assertCommitted();
        signal?.throwIfAborted();
        const response = await this.postChatCompletion(driver, prepared.payload, options, signal);
        const toolDefinitions = selection.context.active_tool_definition_ids.map((id) => {
            const tool = Object.hasOwn(selection.tool_definitions, id) ? selection.tool_definitions[id] : undefined;
            if (!tool) throw new Error(`Indexed OpenAI Chat tool ${id} is unavailable`);
            return tool;
        });
        return decodeOpenAIChatCanonicalResponse(
            response,
            {
                runtime: record.runtime,
                receipt: record.request_receipt,
                provider,
                requested_model: record.request_receipt.target.model,
                tool_definitions: toolDefinitions,
                generation_id: record.generation_id,
                response_turn_id: record.response_turn_id,
            },
            typeof response.choices[0]?.finish_reason === 'string' ? response.choices[0].finish_reason : undefined,
        );
    }

    /**
     * Compare an authenticated selected-content archive with the existing requestBinding JSON receipt.
     * SDK request conversion happens later and has no separate retained wire digest. This parity check
     * neither makes the archive execution-ready nor authorizes provider transport.
     */
    async attestSelectedPreparedRequest(input: {
        record: ConversationPreparedRequestRecord;
        working_set: RequestSourceWorkingSet;
        runtime: ResolvedConversationRuntimeContext;
        options: ExecutionOptions;
        provider: string;
        stream: boolean;
        current_turn: number;
        prior_native_message_count?: number;
    }): Promise<void> {
        const record = parseConversationPreparedRequestRecord(input.record);
        if (!preflightJsonInput(input.working_set).success) {
            throw new TypeError('Selected OpenAI Chat working set is not bounded JSON');
        }
        const workingSet = RequestSourceWorkingSetSchema.parse(structuredClone(input.working_set));
        const selected = compileOpenAIChatSelectedWorkingSet(workingSet, {
            provider: input.provider,
            model: input.options.model,
        });
        const priorCount = input.prior_native_message_count ?? selected.conversation.messages.length;
        if (
            !Number.isSafeInteger(input.current_turn) ||
            input.current_turn < 0 ||
            !Number.isSafeInteger(priorCount) ||
            priorCount < 0 ||
            priorCount > selected.conversation.messages.length
        ) {
            throw new RangeError('Selected OpenAI Chat projection counters are invalid');
        }
        const tools = workingSet.context.active_tool_definition_ids.map((id) => {
            const tool = Object.hasOwn(workingSet.tool_definitions, id) ? workingSet.tool_definitions[id] : undefined;
            if (tool === undefined) throw new Error(`Selected OpenAI Chat tool ${id} is unavailable`);
            return tool;
        });
        const conversation = prepareCanonicalOpenAIProjectionFromNative(
            selected.conversation,
            priorCount,
            input.options,
            input.current_turn,
            tools,
        );
        const payload = this.buildPayload(conversation, input.options, input.stream, input.provider, tools);
        const binding = this.requestBinding(payload, input.options, input.provider);
        const targetOptions = canonicalToolSelectionTargetOptions(
            binding.target_options,
            canonicalToolSelectionPolicy(input.options),
        );
        await assertOpenAIChatSelectedPreparedRequestEvidence({
            record,
            working_set: workingSet,
            runtime: input.runtime,
            target: {
                provider: input.provider,
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                model: input.options.model,
                adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                ...(targetOptions === undefined ? {} : { options: targetOptions }),
            },
            native_payload: binding.payload,
        });
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

        const resultSchemaInstruction = _options.result_schema
            ? `IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. <response_schema>${JSON.stringify(_options.result_schema)}</response_schema>`
            : undefined;
        if (
            (this.options.resultSchemaMode === 'prompt' ||
                this.options.includeResultSchemaInPrompt ||
                this.options.includeResultSchemaInPromptForModel?.(_options.model)) &&
            resultSchemaInstruction
        ) {
            messages.push({
                role: 'system',
                content: resultSchemaInstruction,
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
                    ...(segment.tool_result_status === undefined
                        ? {}
                        : { tool_result_status: segment.tool_result_status }),
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
                        } else if (file.mime_type?.startsWith('audio/')) {
                            parts.push(await openAIInputAudioPart(file));
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

    async requestCanonicalTextCompletion(
        driver: DriverT,
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const provider = getOpenAIChatDriverProvider(driver);
        const canonicalState = await prepareOpenAIChatCanonicalState({
            conversation: options.conversation,
            prompt,
            options,
            provider,
        });
        return this.requestPreparedCanonicalTextCompletion(driver, canonicalState, options, provider, signal);
    }

    async requestCanonicalContextCompletion(
        driver: DriverT,
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const provider = getOpenAIChatDriverProvider(driver);
        const canonicalState = await prepareOpenAIChatCanonicalContext({ options, provider });
        return this.requestPreparedCanonicalTextCompletion(driver, canonicalState, options, provider, signal);
    }

    private async requestPreparedCanonicalTextCompletion(
        driver: DriverT,
        canonicalState: Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>,
        options: ExecutionOptions,
        provider: string,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const includeThoughts =
            (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })?.include_thoughts !== false;
        const conversation = prepareCanonicalOpenAIProjection(canonicalState, options);
        const payload = this.buildPayload(conversation, options, false, provider, canonicalState.tool_definitions);
        const requestBinding = this.requestBinding(payload, options, provider);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            { provider, protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL, model: options.model },
            requestBinding.payload,
        );
        if (canonicalState.accepted_response !== undefined) {
            if (options.include_original_response) {
                throw new Error(
                    'An idempotently recovered Chat Completions response cannot reconstruct original_response',
                );
            }
            return recoverCanonicalExecutionResponse(canonicalState, options);
        }
        const measured = await this.projectCanonicalMeasurement(
            canonicalState,
            payload,
            requestBinding,
            options,
            provider,
            signal,
        );
        const finalized = await finalizeOpenAIChatPreparedRequest(
            { ...canonicalState, native_conversation: conversation },
            payload,
            requestBinding,
        );
        const prepared =
            measured === undefined
                ? finalized
                : {
                      ...finalized,
                      receipt: { ...finalized.receipt, measurement: measured.measurement },
                  };
        await publishCanonicalPreparedRequest(prepared, options, measured);
        const result = await this.postChatCompletion(driver, payload, options, signal);

        const choice = result?.choices?.[0];
        const message = choice?.message;
        const completionResults = extractOpenAIChatCompletionsResults(message, includeThoughts);
        const tool_use = parseOpenAIChatCompletionsToolCalls(message?.tool_calls);
        if (
            !message ||
            (completionResults.length === 0 && !extractOpenAIChatCompletionsReasoningText(message) && !tool_use?.length)
        ) {
            throw new Error('Chat Completions response is not valid: no data');
        }

        const rawDecoded = await decodeOpenAIChatCanonicalResponse(
            result,
            prepared,
            typeof choice?.finish_reason === 'string' ? choice.finish_reason : undefined,
        );
        const normalized =
            !tool_use?.length && options.result_schema
                ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
                : undefined;
        let decoded =
            normalized?.status === 'valid'
                ? await decodeOpenAIChatCanonicalResponse(
                      result,
                      prepared,
                      typeof choice?.finish_reason === 'string' ? choice.finish_reason : undefined,
                      normalized.structured_output,
                  )
                : rawDecoded;
        if (normalized?.status === 'invalid') decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
        const canonicalConversation = await appendOpenAIChatCanonicalResponseWithProcessing(prepared, decoded);

        return createCanonicalExecutionResponse(canonicalConversation, prepared.runtime.response_operation_id, {
            ...(result.service_tier == null ? {} : { service_tier: result.service_tier }),
            ...(options.include_original_response
                ? {
                      original_response:
                          (result as OpenAIChatCompletionsResponseWithOriginal)[originalResponseSymbol] ?? result,
                  }
                : {}),
        });
    }

    async requestTextCompletion(
        driver: DriverT,
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const provider = getOpenAIChatDriverProvider(driver);
        const canonicalState = await prepareOpenAIChatCanonicalState({
            conversation: options.conversation,
            prompt,
            options,
            provider,
        });
        const includeThoughts =
            (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })?.include_thoughts !== false;
        const conversation = prepareCanonicalOpenAIProjection(canonicalState, options);
        const payload = this.buildPayload(conversation, options, false, provider);
        const requestBinding = this.requestBinding(payload, options, provider);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            { provider, protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL, model: options.model },
            requestBinding.payload,
        );
        if (canonicalState.accepted_response !== undefined) {
            return recoverOpenAICompletion(canonicalState, options, includeThoughts);
        }
        const measured = await this.projectCanonicalMeasurement(
            canonicalState,
            payload,
            requestBinding,
            options,
            provider,
            signal,
        );
        const finalized = await finalizeOpenAIChatPreparedRequest(
            { ...canonicalState, native_conversation: conversation },
            payload,
            requestBinding,
        );
        const prepared =
            measured === undefined
                ? finalized
                : {
                      ...finalized,
                      receipt: { ...finalized.receipt, measurement: measured.measurement },
                  };
        await publishCanonicalPreparedRequest(prepared, options, measured);
        const result = await this.postChatCompletion(driver, payload, options, signal);
        const choice = result?.choices?.[0];
        const message = choice?.message;
        const completionResults = extractOpenAIChatCompletionsResults(message, includeThoughts);
        const toolUse = parseOpenAIChatCompletionsToolCalls(message?.tool_calls);
        if (
            !message ||
            (completionResults.length === 0 && !extractOpenAIChatCompletionsReasoningText(message) && !toolUse?.length)
        ) {
            throw new Error('Chat Completions response is not valid: no data');
        }
        const rawDecoded = await decodeOpenAIChatCanonicalResponse(
            result,
            prepared,
            typeof choice?.finish_reason === 'string' ? choice.finish_reason : undefined,
        );
        const normalized =
            !toolUse?.length && options.result_schema
                ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
                : undefined;
        const decoded =
            normalized?.status === 'valid'
                ? await decodeOpenAIChatCanonicalResponse(
                      result,
                      prepared,
                      typeof choice?.finish_reason === 'string' ? choice.finish_reason : undefined,
                      normalized.structured_output,
                  )
                : rawDecoded;
        const canonicalConversation = await appendOpenAIChatCanonicalResponseWithProcessing(prepared, decoded);
        return {
            result: completionResults,
            tool_use: toolUse,
            token_usage: mapOpenAIChatCompletionsUsage(result.usage),
            service_tier: result.service_tier ?? undefined,
            finish_reason: normalizeOpenAIChatCompletionsFinishReason(choice?.finish_reason, !!toolUse?.length),
            original_response: options.include_original_response
                ? ((result as OpenAIChatCompletionsResponseWithOriginal)[originalResponseSymbol] ?? result)
                : undefined,
            conversation: canonicalConversation,
        };
    }

    async requestTextCompletionStream(
        driver: DriverT,
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        const provider = getOpenAIChatDriverProvider(driver);
        const canonicalState = await prepareOpenAIChatCanonicalState({
            conversation: options.conversation,
            prompt,
            options,
            provider,
        });
        const includeThoughts =
            (options.model_options as TextFallbackOptions & { include_thoughts?: boolean })?.include_thoughts !== false;
        const conversation = prepareCanonicalOpenAIProjection(canonicalState, options);
        const payload = this.buildPayload(conversation, options, true, provider);
        const requestBinding = this.requestBinding(payload, options, provider);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            { provider, protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL, model: options.model },
            requestBinding.payload,
        );
        if (canonicalState.accepted_response !== undefined) {
            const canonical = await recoverCanonicalExecutionResponse(canonicalState, options);
            return recoveredOpenAIStream(
                legacyCompletionFromCanonicalExecution(canonical, { include_reasoning: includeThoughts }),
            );
        }
        const measured = await this.projectCanonicalMeasurement(
            canonicalState,
            payload,
            requestBinding,
            options,
            provider,
            signal,
        );
        const finalized = await finalizeOpenAIChatPreparedRequest(
            { ...canonicalState, native_conversation: conversation },
            payload,
            requestBinding,
        );
        const prepared =
            measured === undefined
                ? finalized
                : {
                      ...finalized,
                      receipt: { ...finalized.receipt, measurement: measured.measurement },
                  };
        await publishCanonicalPreparedRequest(prepared, options, measured);
        const responseStream = await this.postChatCompletionStream(driver, payload, options, signal);

        const projector = new OpenAIThinkStreamProjector();
        let nativeContent = '';
        let nativeReasoningContent: string | undefined;
        let nativeReasoning: string | undefined;
        let nativeProviderReplay: OpenAIChatProviderReplay | undefined;
        let responseId: string | undefined;
        let responseObject: string | undefined;
        let responseCreated: number | undefined;
        let responseModel: string | undefined;
        let responseServiceTier: OpenAIChatServiceTier | undefined;
        let responseSystemFingerprint: string | undefined;
        let responseUsage: OpenAIChatCompletionsUsage | undefined;
        let responseFinishReason: string | undefined;
        const nativeToolCalls = new Map<
            number,
            { id: string; type: 'function'; function: { name: string; arguments: string } }
        >();

        const stream = transformSSEStream(responseStream, (data: string) => {
            const json = JSON.parse(data) as OpenAIChatCompletionsStreamResponse;
            const choice = json.choices?.[0];
            const delta = choice?.delta;
            responseId = json.id;
            responseObject = json.object;
            responseCreated = json.created;
            responseModel = json.model;
            responseServiceTier = json.service_tier;
            if (typeof json.system_fingerprint === 'string') responseSystemFingerprint = json.system_fingerprint;
            if (json.usage != null) responseUsage = json.usage;
            if (typeof choice?.finish_reason === 'string') responseFinishReason = choice.finish_reason;
            const chunkResults: CompletionResult[] = [];
            const content = extractOpenAIChatCompletionsContentText(delta?.content);
            if (content) {
                nativeContent += content;
                chunkResults.push(...projector.push(content, !!choice?.finish_reason));
            } else if (choice?.finish_reason) {
                chunkResults.push(...projector.push('', true));
            }

            if (delta?.provider_replay !== undefined) nativeProviderReplay = structuredClone(delta.provider_replay);

            if (typeof delta?.reasoning_content === 'string') {
                nativeReasoningContent = (nativeReasoningContent ?? '') + delta.reasoning_content;
                if (includeThoughts && delta.reasoning_content) {
                    chunkResults.unshift({ type: 'thoughts', value: delta.reasoning_content });
                }
            } else if (typeof delta?.reasoning === 'string') {
                nativeReasoning = (nativeReasoning ?? '') + delta.reasoning;
                if (includeThoughts && delta.reasoning) {
                    chunkResults.unshift({ type: 'thoughts', value: delta.reasoning });
                }
            }

            let toolUseChunks: StreamingOpenAIToolUse[] | undefined;
            if (delta?.tool_calls && delta.tool_calls.length > 0) {
                toolUseChunks = delta.tool_calls.map((tc) => {
                    const index = tc.index ?? 0;
                    const native = nativeToolCalls.get(index) ?? {
                        id: '',
                        type: 'function' as const,
                        function: { name: '', arguments: '' },
                    };
                    if (tc.id) native.id = tc.id;
                    if (tc.function?.name) native.function.name += tc.function.name;
                    if (tc.function?.arguments) native.function.arguments += tc.function.arguments;
                    nativeToolCalls.set(index, native);
                    const toolUse: StreamingOpenAIToolUse = {
                        id: `tool_${index}`,
                        tool_name: tc.function?.name ?? '',
                        // Empty deltas are zero-byte string fragments, not parsed empty objects.
                        // Keeping one representation prevents a placeholder from replacing prior JSON bytes.
                        tool_input: typeof tc.function?.arguments === 'string' ? tc.function.arguments : '',
                    };
                    if (tc.id) {
                        toolUse._actual_id = tc.id;
                    }
                    return toolUse;
                });
            }
            return {
                result: includeThoughts ? chunkResults : chunkResults.filter((result) => result.type !== 'thoughts'),
                tool_use: toolUseChunks,
                finish_reason: normalizeOpenAIChatCompletionsFinishReason(
                    choice?.finish_reason,
                    !!toolUseChunks?.length,
                ),
                token_usage: mapOpenAIChatCompletionsUsage(json.usage),
                service_tier: json.service_tier ?? undefined,
            } satisfies CompletionChunkObject;
        });

        let finalizedConversation: Promise<unknown> | undefined;
        const finalizeConversation = () => {
            finalizedConversation ??= (async () => {
                const assistantMessage = {
                    role: 'assistant' as const,
                    content: nativeContent || null,
                    ...(nativeReasoningContent !== undefined && { reasoning_content: nativeReasoningContent }),
                    ...(nativeReasoning !== undefined && { reasoning: nativeReasoning }),
                    ...(nativeProviderReplay !== undefined && { provider_replay: nativeProviderReplay }),
                    ...(nativeToolCalls.size > 0 && {
                        tool_calls: [...nativeToolCalls.entries()]
                            .sort(([left], [right]) => left - right)
                            .map(([, toolCall]) => toolCall),
                    }),
                };
                if (
                    responseId === undefined ||
                    responseObject === undefined ||
                    responseCreated === undefined ||
                    responseModel === undefined
                ) {
                    throw new Error('Chat Completions stream ended without a complete native response identity');
                }
                if (responseFinishReason === undefined) {
                    throw new Error('Chat Completions stream ended without a terminal finish reason');
                }
                const response: OpenAIChatCompletionsResponse = {
                    id: responseId,
                    object: responseObject as OpenAI.Chat.ChatCompletion['object'],
                    created: responseCreated,
                    model: responseModel,
                    choices: [
                        {
                            index: 0,
                            message: assistantMessage,
                            finish_reason: responseFinishReason,
                        },
                    ],
                    ...(responseServiceTier === undefined ? {} : { service_tier: responseServiceTier }),
                    ...(responseSystemFingerprint === undefined
                        ? {}
                        : { system_fingerprint: responseSystemFingerprint }),
                    ...(responseUsage === undefined ? {} : { usage: responseUsage }),
                };
                return (
                    await finalizeOpenAIChatStreamResponse({
                        response,
                        prepared,
                        finish_reason: responseFinishReason,
                        options,
                    })
                ).response.conversation;
            })();
            return finalizedConversation;
        };
        return Object.assign(stream, { finalizeConversation });
    }

    async requestCanonicalTextCompletionEventStream(
        driver: DriverT,
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const provider = getOpenAIChatDriverProvider(driver);
        const canonicalState = await prepareOpenAIChatCanonicalState({
            conversation: options.conversation,
            prompt,
            options,
            provider,
        });
        return this.requestPreparedCanonicalTextCompletionEventStream(
            driver,
            canonicalState,
            options,
            provider,
            signal,
            open,
        );
    }

    async requestCanonicalContextCompletionEventStream(
        driver: DriverT,
        options: CanonicalExecutionContextOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const provider = getOpenAIChatDriverProvider(driver);
        const canonicalState = await prepareOpenAIChatCanonicalContext({ options, provider });
        return this.requestPreparedCanonicalTextCompletionEventStream(
            driver,
            canonicalState,
            options,
            provider,
            signal,
            open,
        );
    }

    private async requestPreparedCanonicalTextCompletionEventStream(
        driver: DriverT,
        canonicalState: Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>,
        options: ExecutionOptions,
        provider: string,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const conversation = prepareCanonicalOpenAIProjection(canonicalState, options);
        const payload = this.buildPayload(conversation, options, true, provider, canonicalState.tool_definitions);
        const requestBinding = this.requestBinding(payload, options, provider);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            { provider, protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL, model: options.model },
            requestBinding.payload,
        );
        const acceptedResponse = canonicalState.accepted_response;
        const identity = {
            request_id: acceptedResponse?.generation.request_id ?? canonicalState.runtime.request_id,
            attempt_id: acceptedResponse?.generation.attempt_id ?? canonicalState.runtime.attempt_id,
            response_operation_id: canonicalState.runtime.response_operation_id,
            generation_id: acceptedResponse?.generation.id ?? canonicalState.generation_id,
            draft_turn_id: acceptedResponse?.turn.id ?? canonicalState.response_turn_id,
        };
        if (canonicalState.accepted_response !== undefined) {
            return new FallbackCanonicalExecutionEventStream(
                identity,
                () => recoverCanonicalExecutionResponse(canonicalState, options),
                { ...open, origin: 'accepted_recovery' },
            );
        }
        const measured = await this.projectCanonicalMeasurement(
            canonicalState,
            payload,
            requestBinding,
            options,
            provider,
            signal,
        );
        const finalized = await finalizeOpenAIChatPreparedRequest(
            { ...canonicalState, native_conversation: conversation },
            payload,
            requestBinding,
        );
        const prepared =
            measured === undefined
                ? finalized
                : {
                      ...finalized,
                      receipt: { ...finalized.receipt, measurement: measured.measurement },
                  };

        const abortController = new AbortController();
        const forwardAbort = () => abortController.abort(signal?.reason);
        let nativeContent = '';
        let contentDraft: OpenAIChatCanonicalDraft | undefined;
        let nativeReasoningContent: string | undefined;
        let reasoningContentDraft: OpenAIChatCanonicalDraft | undefined;
        let nativeReasoning: string | undefined;
        let nativeProviderReplay: OpenAIChatProviderReplay | undefined;
        let reasoningDraft: OpenAIChatCanonicalDraft | undefined;
        let responseId: string | undefined;
        let responseObject: string | undefined;
        let responseCreated: number | undefined;
        let responseModel: string | undefined;
        let responseServiceTier: OpenAIChatServiceTier | undefined;
        let responseSystemFingerprint: string | undefined;
        let responseUsage: OpenAIChatCompletionsUsage | undefined;
        let responseFinishReason: string | undefined;
        const nativeToolCalls = new Map<
            number,
            {
                id: string;
                type: 'function';
                function: { name: string; arguments: string };
                draft: OpenAIChatCanonicalDraft;
            }
        >();

        const eventStream = canonicalNativeExecutionEventStream({
            identity,
            open,
            openSource: async () => {
                const stream = await this.postChatCompletionStream(driver, payload, options, abortController.signal);
                return openAIChatNativeSSE(stream);
            },
            map: async (json, writer) => {
                const choice = json.choices?.[0];
                const delta = choice?.delta;
                responseId = json.id;
                responseObject = json.object;
                responseCreated = json.created;
                responseModel = json.model;
                responseServiceTier = json.service_tier;
                if (typeof json.system_fingerprint === 'string') responseSystemFingerprint = json.system_fingerprint;
                if (json.usage != null) responseUsage = json.usage;
                if (typeof choice?.finish_reason === 'string') responseFinishReason = choice.finish_reason;

                const content = extractOpenAIChatCompletionsContentText(delta?.content);
                if (content) {
                    contentDraft ??= {
                        draft_block_id: `${prepared.response_turn_id}:chat:text`,
                        native_position: openAIChatStreamPosition(['choices', 0, 'message', 'content']),
                        kind: 'text',
                    };
                    if (nativeContent.length === 0) {
                        await writer.startBlock({
                            draft_block_id: contentDraft.draft_block_id,
                            native_position: contentDraft.native_position,
                            block: { type: 'text' },
                        });
                    }
                    nativeContent += content;
                    await writer.text({
                        draft_block_id: contentDraft.draft_block_id,
                        native_position: contentDraft.native_position,
                        text: content,
                    });
                }

                if (delta?.provider_replay !== undefined) {
                    nativeProviderReplay = structuredClone(delta.provider_replay);
                }

                if (typeof delta?.reasoning_content === 'string' && delta.reasoning_content.length > 0) {
                    reasoningContentDraft ??= {
                        draft_block_id: `${prepared.response_turn_id}:chat:reasoning-content`,
                        native_position: openAIChatStreamPosition(['choices', 0, 'message', 'reasoning_content']),
                        kind: 'reasoning',
                    };
                    if (nativeReasoningContent === undefined) {
                        await writer.startBlock({
                            draft_block_id: reasoningContentDraft.draft_block_id,
                            native_position: reasoningContentDraft.native_position,
                            block: { type: 'reasoning', visibility: 'display' },
                        });
                    }
                    nativeReasoningContent = (nativeReasoningContent ?? '') + delta.reasoning_content;
                    await writer.reasoning({
                        draft_block_id: reasoningContentDraft.draft_block_id,
                        native_position: reasoningContentDraft.native_position,
                        text: delta.reasoning_content,
                    });
                }
                if (typeof delta?.reasoning === 'string' && delta.reasoning.length > 0) {
                    reasoningDraft ??= {
                        draft_block_id: `${prepared.response_turn_id}:chat:reasoning`,
                        native_position: openAIChatStreamPosition(['choices', 0, 'message', 'reasoning']),
                        kind: 'reasoning',
                    };
                    if (nativeReasoning === undefined) {
                        await writer.startBlock({
                            draft_block_id: reasoningDraft.draft_block_id,
                            native_position: reasoningDraft.native_position,
                            block: { type: 'reasoning', visibility: 'display' },
                        });
                    }
                    nativeReasoning = (nativeReasoning ?? '') + delta.reasoning;
                    await writer.reasoning({
                        draft_block_id: reasoningDraft.draft_block_id,
                        native_position: reasoningDraft.native_position,
                        text: delta.reasoning,
                    });
                }

                for (const toolCall of delta?.tool_calls ?? []) {
                    const index = toolCall.index ?? 0;
                    let native = nativeToolCalls.get(index);
                    if (native === undefined) {
                        const draft: OpenAIChatCanonicalDraft = {
                            draft_block_id: `${prepared.response_turn_id}:chat:tool:${index}`,
                            native_position: openAIChatStreamPosition(['choices', 0, 'message', 'tool_calls', index]),
                            kind: 'tool_call',
                            tool_index: index,
                        };
                        native = {
                            id: toolCall.id ?? '',
                            type: 'function',
                            function: {
                                name: toolCall.function?.name ?? '',
                                arguments: '',
                            },
                            draft,
                        };
                        nativeToolCalls.set(index, native);
                        await writer.startBlock({
                            draft_block_id: draft.draft_block_id,
                            native_position: draft.native_position,
                            block: {
                                type: 'tool_call',
                                executor: 'application',
                                ...(native.id.length === 0 ? {} : { call_id: native.id }),
                                ...(native.function.name.length === 0 ? {} : { tool_name: native.function.name }),
                            },
                        });
                    } else {
                        const priorId = native.id;
                        const priorName = native.function.name;
                        if (toolCall.id) native.id = toolCall.id;
                        if (toolCall.function?.name) native.function.name += toolCall.function.name;
                        if (native.id !== priorId || native.function.name !== priorName) {
                            await writer.toolIdentity({
                                draft_block_id: native.draft.draft_block_id,
                                native_position: native.draft.native_position,
                                ...(native.id === priorId ? {} : { call_id: native.id }),
                                ...(native.function.name === priorName ? {} : { tool_name: native.function.name }),
                            });
                        }
                    }
                    const argumentsFragment = toolCall.function?.arguments;
                    if (typeof argumentsFragment === 'string' && argumentsFragment.length > 0) {
                        native.function.arguments += argumentsFragment;
                        await writer.toolArgumentsFragment({
                            draft_block_id: native.draft.draft_block_id,
                            native_position: native.draft.native_position,
                            fragment: argumentsFragment,
                        });
                    }
                }
            },
            finalize: async () => {
                if (
                    responseId === undefined ||
                    responseObject === undefined ||
                    responseCreated === undefined ||
                    responseModel === undefined
                ) {
                    throw new Error('Chat Completions stream ended without a complete native response identity');
                }
                if (responseFinishReason === undefined) {
                    throw new Error('Chat Completions stream ended without a terminal finish reason');
                }
                const assistantMessage = {
                    role: 'assistant' as const,
                    content: nativeContent || null,
                    ...(nativeReasoningContent === undefined ? {} : { reasoning_content: nativeReasoningContent }),
                    ...(nativeReasoning === undefined ? {} : { reasoning: nativeReasoning }),
                    ...(nativeProviderReplay === undefined ? {} : { provider_replay: nativeProviderReplay }),
                    ...(nativeToolCalls.size === 0
                        ? {}
                        : {
                              tool_calls: [...nativeToolCalls.entries()]
                                  .sort(([left], [right]) => left - right)
                                  .map(([, toolCall]) => ({
                                      id: toolCall.id,
                                      type: toolCall.type,
                                      function: toolCall.function,
                                  })),
                          }),
                };
                const response: OpenAIChatCompletionsResponse = {
                    id: responseId,
                    object: responseObject as OpenAI.Chat.ChatCompletion['object'],
                    created: responseCreated,
                    model: responseModel,
                    choices: [{ index: 0, message: assistantMessage, finish_reason: responseFinishReason }],
                    ...(responseServiceTier === undefined ? {} : { service_tier: responseServiceTier }),
                    ...(responseSystemFingerprint === undefined
                        ? {}
                        : { system_fingerprint: responseSystemFingerprint }),
                    ...(responseUsage === undefined ? {} : { usage: responseUsage }),
                };
                const finalized = await finalizeOpenAIChatStreamResponse({
                    response,
                    prepared,
                    finish_reason: responseFinishReason,
                    options,
                });
                return {
                    decoded: finalized.decoded,
                    response: finalized.response,
                    prepare_reconciliation: async () => {
                        const { raw_decoded: rawDecoded, normalized } = finalized;
                        let { decoded } = finalized;
                        const drafts = [
                            ...(contentDraft === undefined ? [] : [contentDraft]),
                            ...(reasoningContentDraft === undefined ? [] : [reasoningContentDraft]),
                            ...(reasoningDraft === undefined ? [] : [reasoningDraft]),
                            ...[...nativeToolCalls.values()]
                                .sort((left, right) => (left.draft.tool_index ?? 0) - (right.draft.tool_index ?? 0))
                                .map((toolCall) => toolCall.draft),
                        ];
                        const rawBlocks = openAIChatSemanticBlocks(rawDecoded, prepared.response_turn_id);
                        if (rawBlocks.length !== drafts.length) {
                            throw new Error('OpenAI Chat stream drafts do not match its terminal native response');
                        }
                        const itemMappings = rawBlocks.flatMap((block, index) => {
                            const draft = drafts[index];
                            if (draft === undefined) return [];
                            return [
                                {
                                    canonical_id: block.id,
                                    native_position: draft.native_position,
                                    kind: 'block' as const,
                                },
                                ...(block.type === 'tool_call'
                                    ? [
                                          {
                                              canonical_id: block.call_id,
                                              native_position: draft.native_position,
                                              kind: 'call' as const,
                                          },
                                      ]
                                    : []),
                            ];
                        });
                        const transformations = [];
                        const reconciliations = [];
                        if (normalized?.status === 'valid') {
                            const sources = rawBlocks.filter((block) => block.type === 'text');
                            const result = openAIChatSemanticBlocks(decoded, prepared.response_turn_id).find(
                                (block) => block.type === 'json',
                            );
                            if (sources.length === 0 || result?.type !== 'json') {
                                throw new Error(
                                    'OpenAI Chat structured stream decode is missing source or result blocks',
                                );
                            }
                            const proof = await createStructuredOutputTransformationProof({
                                id: `${prepared.generation_id}:structured-output`,
                                source_blocks: sources,
                                result_block: result,
                            });
                            transformations.push(proof);
                            const sourceDrafts = rawBlocks.flatMap((block, index) =>
                                block.type === 'text' && drafts[index] !== undefined ? [drafts[index]] : [],
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
                            const draft = drafts[index];
                            if (draft === undefined || (normalized?.status === 'valid' && block.type === 'text')) {
                                continue;
                            }
                            reconciliations.push({
                                draft_block_ids: [draft.draft_block_id],
                                native_positions: [draft.native_position],
                                committed_block_ids: [block.id],
                                disposition: 'direct' as const,
                            });
                        }
                        decoded = { ...decoded, stream_evidence: { item_mappings: itemMappings, transformations } };
                        return {
                            decoded,
                            reconciliations,
                            deliver_final_events: async (finalWriter) => {
                                if (decoded.generation.usage !== undefined) {
                                    await finalWriter.usage(decoded.generation.usage);
                                }
                                for (const [index, block] of rawBlocks.entries()) {
                                    const draft = drafts[index];
                                    if (draft === undefined) continue;
                                    await finalWriter.finishBlock({
                                        draft_block_id: draft.draft_block_id,
                                        native_position: draft.native_position,
                                        outcome:
                                            block.type === 'tool_call' && block.arguments.type === 'invalid'
                                                ? 'malformed'
                                                : responseFinishReason === 'length'
                                                  ? 'interrupted'
                                                  : 'native_complete',
                                    });
                                }
                                await finalWriter.finish({
                                    outcome: responseFinishReason === 'length' ? 'interrupted' : 'completed',
                                    finish_reason: responseFinishReason,
                                    ...(typeof responseServiceTier === 'string'
                                        ? { service_tier: responseServiceTier }
                                        : {}),
                                });
                            },
                        };
                    },
                    ...(finalized.normalized?.status === 'valid' && options.result_schema !== undefined
                        ? { result_schema: options.result_schema }
                        : {}),
                };
            },
            abort: () => abortController.abort(),
            close: () => signal?.removeEventListener('abort', forwardAbort),
        });
        await publishCanonicalPreparedRequest(prepared, options, measured);
        if (signal?.aborted) forwardAbort();
        else signal?.addEventListener('abort', forwardAbort, { once: true });
        return eventStream;
    }

    protected buildPayload(
        conversation: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        stream: boolean,
        provider: string = Providers.openai_compatible,
        tools:
            | readonly Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'>[]
            | ToolDefinition[]
            | undefined = options.tools,
    ): OpenAIChatCompletionsPayload {
        return buildOpenAIChatCompletionsPayload(
            conversation,
            options,
            this.options,
            this.getModelName(options),
            stream,
            provider,
            tools,
        );
    }

    protected getModelName(options: ExecutionOptions): string {
        return this.options.modelName ?? options.model;
    }

    /** Concrete SDK protocols must expose their actual serialized body, not intermediate receipt binding. */
    protected canonicalTransportBody(_payload: OpenAIChatCompletionsPayload): JsonValue {
        throw new Error('Canonical actual transport projection is unavailable for this protocol');
    }

    private async projectCanonicalMeasurement(
        state: Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>,
        payload: OpenAIChatCompletionsPayload,
        binding: OpenAIChatRequestBinding,
        options: ExecutionOptions,
        provider: string,
        signal?: AbortSignal,
    ): Promise<CanonicalProjectedRequestMeasurement | undefined> {
        const callback = options.on_canonical_request_projected;
        if (!callback) {
            if (state.document.processing.enabled && state.document.processing.budget)
                throw new Error('Required processing measurement is unavailable');
            return undefined;
        }
        const capturedOptions: ExecutionOptions = {
            model: options.model,
            ...(options.model_options === undefined ? {} : { model_options: structuredClone(options.model_options) }),
            ...(options.result_schema === undefined ? {} : { result_schema: structuredClone(options.result_schema) }),
        };
        const actualModel = this.getModelName(options);
        const protocolOptions: OpenAIChatCompletionsProtocolOptions = {
            modelName: actualModel,
            defaultMaxTokens: this.options.defaultMaxTokens,
            extraBody: structuredClone(this.options.extraBody),
            resultSchemaMode: this.options.resultSchemaMode,
            toolSchemaMode: this.options.toolSchemaMode,
        };
        const capturedDocument = parseConversationDocument(state.document);
        const runtime = structuredClone(state.runtime);
        const tools = structuredClone(state.tool_definitions);
        const targetOptions = canonicalToolSelectionTargetOptions(
            binding.target_options,
            state.response_selection_policy,
        );
        const target = {
            provider,
            protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            model: state.requested_model,
            adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
            ...(targetOptions === undefined ? {} : { options: structuredClone(targetOptions) }),
        };
        const nativeRequest = structuredClone(this.canonicalTransportBody(payload));
        const stream = payload.stream === true;
        const build = (conversation: OpenAIChatCompletionsPrompt): JsonValue => {
            const projected = projectOpenAIChatCompletionsHistory(
                conversation,
                capturedOptions,
                canonicalConversationTurnNumber(capturedDocument),
            );
            const prepared = prepareOpenAIChatCompletionsConversation(projected, {
                model: capturedOptions.model,
                tools,
            });
            return this.canonicalTransportBody(
                buildOpenAIChatCompletionsPayload(
                    prepared,
                    capturedOptions,
                    protocolOptions,
                    actualModel,
                    stream,
                    provider,
                    tools,
                ),
            );
        };
        const compiler: CanonicalRequestProjectionCompiler = {
            compileDocument: (document: ConversationDocument, compileSignal?: AbortSignal) => {
                const owned = parseConversationDocument(document);
                return boundedCanonicalProjectionOperation(async (boundedSignal) => {
                    boundedSignal.throwIfAborted();
                    const compiled = compileOpenAIChatCompletionsConversation(owned, target);
                    const request = build(compiled.conversation);
                    boundedSignal.throwIfAborted();
                    return request;
                }, compileSignal);
            },
            compileProspective: (input, compileSignal) => {
                const owned = structuredClone(input);
                return boundedCanonicalProjectionOperation(async (boundedSignal) => {
                    const compiled = await compileOpenAIChatProspectiveJsonMinification(owned, target, boundedSignal);
                    return {
                        status: 'compiled',
                        source_request: build(compiled.original.conversation),
                        replacement_request: build(compiled.replacement.conversation),
                    };
                }, compileSignal);
            },
        };
        // Unsupported historical projections must not be silently replaced by a dry compiler.
        const replayed = await compiler.compileDocument(capturedDocument, signal);
        if ((await fingerprintJson(replayed)) !== (await fingerprintJson(nativeRequest)))
            throw new Error('Canonical dry compiler cannot reproduce the actual native request');
        const measured = await projectCanonicalRequestMeasurement(
            {
                document: capturedDocument,
                runtime,
                target,
                native_request: nativeRequest,
            },
            compiler,
            callback,
            signal,
        );
        return measured;
    }

    protected requestBinding(
        payload: OpenAIChatCompletionsPayload,
        _options: ExecutionOptions,
        _provider: string,
    ): OpenAIChatRequestBinding {
        return { payload: providerJsonValue(payload) };
    }

    protected abstract postChatCompletion(
        driver: DriverT,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse>;

    protected abstract postChatCompletionStream(
        driver: DriverT,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream>;
}

interface OpenAIChatCompletionsTransportDriver {
    _postChatCompletion(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse>;
    _postChatCompletionStream(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream>;
}

export interface OpenAISDKChatCompletionsDriver {
    service: OpenAI;
}

export function openAIChatCompletionsStreamToSSE(
    stream: AsyncIterable<OpenAIChatCompletionsStreamResponse>,
    abortSource?: () => void,
): ReadableStream {
    const iterator = stream[Symbol.asyncIterator]();
    let cancelled = false;

    return new ReadableStream({
        async start(controller) {
            try {
                while (!cancelled) {
                    const { done, value: chunk } = await iterator.next();
                    if (done || cancelled) {
                        break;
                    }
                    controller.enqueue({ type: 'event', data: JSON.stringify(chunk) });
                }
                if (!cancelled) {
                    controller.close();
                }
            } catch (error) {
                if (!cancelled) {
                    controller.error(error);
                }
            }
        },
        async cancel() {
            cancelled = true;
            abortSource?.();
            await iterator.return?.();
        },
    });
}

class DriverChatCompletionsProtocol extends OpenAIChatCompletionsProtocol<OpenAIChatCompletionsTransportDriver> {
    constructor(
        options: OpenAIChatCompletionsProtocolOptions,
        private readonly resolveModelName: (options: ExecutionOptions) => string,
        private readonly projectTransportBody: (payload: OpenAIChatCompletionsPayload) => JsonValue,
    ) {
        super(options);
    }

    protected override getModelName(options: ExecutionOptions): string {
        return this.resolveModelName(options);
    }

    protected override canonicalTransportBody(payload: OpenAIChatCompletionsPayload): JsonValue {
        return this.projectTransportBody(payload);
    }

    protected async postChatCompletion(
        driver: OpenAIChatCompletionsTransportDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse> {
        return driver._postChatCompletion(payload, options, signal);
    }

    protected async postChatCompletionStream(
        driver: OpenAIChatCompletionsTransportDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream> {
        return driver._postChatCompletionStream(payload, options, signal);
    }
}

/** @internal Convert only portable OpenAI fields; provider-scoped replay is intentionally omitted. */
export function toOpenAISDKMessage(
    message: OpenAIChatCompletionsRequestMessage,
): OpenAI.Chat.ChatCompletionMessageParam {
    const textParts = Array.isArray(message.content)
        ? message.content.filter((part): part is OpenAIChatCompletionsTextPart => part.type === 'text')
        : undefined;

    switch (message.role) {
        case 'assistant':
            return {
                role: 'assistant',
                content: typeof message.content === 'string' || message.content === null ? message.content : textParts,
                tool_calls: message.tool_calls,
                reasoning_content: message.reasoning_content,
                reasoning: message.reasoning,
            } as OpenAI.Chat.ChatCompletionAssistantMessageParam;
        case 'developer':
        case 'system':
            return {
                role: message.role,
                content: typeof message.content === 'string' ? message.content : (textParts ?? []),
            };
        case 'tool':
            if (!message.tool_call_id) {
                throw new TypeError('OpenAI tool messages require tool_call_id');
            }
            return {
                role: 'tool',
                content: typeof message.content === 'string' ? message.content : (textParts ?? []),
                tool_call_id: message.tool_call_id,
            };
        case 'user':
            if (message.content === null || message.content === undefined) {
                throw new TypeError('OpenAI user messages require content');
            }
            return { role: 'user', content: message.content };
        default:
            throw new TypeError(`Unsupported OpenAI message role: ${message.role}`);
    }
}

export function toOpenAINonStreamingPayload(
    payload: OpenAIChatCompletionsPayload,
): OpenAI.Chat.ChatCompletionCreateParamsNonStreaming {
    const { messages, stream: _stream, extra_body, ...body } = payload;
    const request = mergeOpenAIExtraBody(
        {
            ...body,
            messages: messages.map(toOpenAISDKMessage),
            stream: false,
        } satisfies OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
        extra_body,
    );
    return request;
}

export function toOpenAIStreamingPayload(
    payload: OpenAIChatCompletionsPayload,
): OpenAI.Chat.ChatCompletionCreateParamsStreaming {
    const { messages, stream: _stream, extra_body, ...body } = payload;
    const request = mergeOpenAIExtraBody(
        {
            ...body,
            messages: messages.map(toOpenAISDKMessage),
            stream: true,
            stream_options: { include_usage: true },
        } satisfies OpenAI.Chat.ChatCompletionCreateParamsStreaming,
        extra_body,
    );
    return request;
}

export class OpenAISDKChatCompletionsProtocol extends OpenAIChatCompletionsProtocol<OpenAISDKChatCompletionsDriver> {
    protected canonicalTransportBody(payload: OpenAIChatCompletionsPayload): JsonValue {
        return providerJsonValue(
            payload.stream ? toOpenAIStreamingPayload(payload) : toOpenAINonStreamingPayload(payload),
        );
    }

    protected async postChatCompletion(
        driver: OpenAISDKChatCompletionsDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
    ): Promise<OpenAIChatCompletionsResponse> {
        const request = toOpenAINonStreamingPayload(payload);
        const requestOptions = this.options.resolveRequestOptions?.(options);
        const response = requestOptions
            ? await driver.service.chat.completions.create(request, requestOptions)
            : await driver.service.chat.completions.create(request);
        return preserveOpenAIChatCompletionsOriginalResponse(
            normalizeOpenAIChatCompletionsResponse(response),
            response,
        );
    }

    protected async postChatCompletionStream(
        driver: OpenAISDKChatCompletionsDriver,
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream> {
        const request = toOpenAIStreamingPayload(payload);
        const requestOptions =
            this.options.resolveRequestOptions?.(options, signal) ?? (signal ? { signal } : undefined);
        const stream = requestOptions
            ? await driver.service.chat.completions.create(request, requestOptions)
            : await driver.service.chat.completions.create(request);
        return openAIChatCompletionsStreamToSSE(normalizeOpenAIChatCompletionsStream(stream), () =>
            stream.controller.abort(),
        );
    }
}

export abstract class OpenAIChatCompletionsDriverBase<
    OptionsT extends OpenAIChatCompletionsDriverOptions = OpenAIChatCompletionsDriverOptions,
> extends OpenAICompatibleDriverBase<OptionsT, OpenAIChatCompletionsPrompt> {
    private readonly chatCompletionsProtocol: DriverChatCompletionsProtocol;

    constructor(options: OptionsT) {
        super(options);
        this.chatCompletionsProtocol = new DriverChatCompletionsProtocol(
            {
                defaultMaxTokens: options.defaultMaxTokens,
                extraBody: options.extraBody,
                resultSchemaMode: options.resultSchemaMode,
                includeResultSchemaInPrompt: options.includeResultSchemaInPrompt,
                includeResultSchemaInPromptForModel: options.includeResultSchemaInPromptForModel,
                toolSchemaMode: options.toolSchemaMode,
            },
            (executionOptions) => this.resolveChatCompletionsRequestModel(executionOptions),
            (payload) => this.projectChatCompletionsTransportBody(payload),
        );
    }

    protected projectChatCompletionsTransportBody(_payload: OpenAIChatCompletionsPayload): JsonValue {
        throw new Error('Actual native body projection is unavailable for this provider transport');
    }

    /** Resolve the transport body model without changing canonical requested-model identity. */
    protected resolveChatCompletionsRequestModel(options: ExecutionOptions): string {
        return options.model;
    }

    protected supportsCanonicalConversation(_options: ExecutionOptions): boolean {
        return true;
    }

    protected supportsCanonicalContextConversation(options: CanonicalExecutionContextOptions): boolean {
        return !openAIAudioTask(options.model);
    }

    /** @internal Provider SDK transport boundary. */
    abstract _postChatCompletion(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
    ): Promise<OpenAIChatCompletionsResponse>;

    /** @internal Provider SDK streaming transport boundary. */
    abstract _postChatCompletionStream(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
    ): Promise<ReadableStream>;

    protected async formatPrompt(
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<OpenAIChatCompletionsPrompt> {
        return this.chatCompletionsProtocol.createPrompt(this, segments, options);
    }

    requestTextCompletion(
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        return this.chatCompletionsProtocol.requestTextCompletion(this, prompt, options, signal);
    }

    requestCanonicalTextCompletion(
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        return this.chatCompletionsProtocol.requestCanonicalTextCompletion(this, prompt, options, signal);
    }

    /** @internal Compile through this configured driver before the host stages indexed evidence. */
    prepareIndexedTextRequest(
        input: {
            selection: IndexedConversationSelectedContext;
            runtime: ResolvedConversationRuntimeContext;
            options: ExecutionOptions;
            stream: boolean;
            signal?: AbortSignal;
        },
        hostCapabilities?: CanonicalHostCapabilities,
    ) {
        return this.chatCompletionsProtocol.prepareIndexedTextRequest(
            { ...input, provider: this.provider },
            hostCapabilities,
        );
    }

    /** @internal The host must supply a durable current-owner assertion before native dispatch. */
    executeCommittedIndexedTextRequest(
        input: {
            selection: IndexedConversationSelectedContext;
            record: ConversationPreparedRequestRecord;
            options: ExecutionOptions;
            assert_committed: () => Promise<void>;
            signal?: AbortSignal;
        },
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<DecodedConversationResponse> {
        return this.chatCompletionsProtocol.executeCommittedIndexedTextRequest(
            this,
            {
                ...input,
                provider: this.provider,
            },
            hostCapabilities,
        );
    }

    requestCanonicalContextCompletion(
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        return this.chatCompletionsProtocol.requestCanonicalContextCompletion(this, options, signal);
    }

    requestTextCompletionStream(
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        return this.chatCompletionsProtocol.requestTextCompletionStream(this, prompt, options, signal);
    }

    requestCanonicalTextCompletionEventStream(
        prompt: OpenAIChatCompletionsPrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        return this.chatCompletionsProtocol.requestCanonicalTextCompletionEventStream(
            this,
            prompt,
            options,
            signal,
            open,
        );
    }

    requestCanonicalContextCompletionEventStream(
        options: CanonicalExecutionContextOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        return this.chatCompletionsProtocol.requestCanonicalContextCompletionEventStream(this, options, signal, open);
    }

    buildStreamingConversation(
        prompt: OpenAIChatCompletionsPrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): OpenAIChatCompletionsPrompt {
        return buildOpenAIChatCompletionsStreamingConversation(prompt, result, toolUse, options);
    }
}

export interface OpenAIChatCompletionsDriverConfig extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint: string;
    default_headers?: Record<string, string>;
}

/** Generic driver for services implementing the OpenAI Chat Completions protocol. */
export class OpenAIChatCompletionsDriver extends OpenAIChatCompletionsDriverBase<OpenAIChatCompletionsDriverConfig> {
    override async resolveCanonicalModelSwitchTarget(
        model: string,
        options?: ModelTarget['options'],
    ): Promise<ModelTarget | undefined> {
        return ModelTargetSchema.parse({
            provider: this.provider,
            protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            model,
            adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
            ...(options === undefined ? {} : { options: structuredClone(options) }),
        });
    }

    override async projectCanonicalModelSwitchRequest(
        document: ConversationDocument,
        target: ModelTarget,
        operation: 'execute' | 'stream',
        controls?: CanonicalModelSwitchProjectionControls,
    ): Promise<ConversationModelSwitchProjection> {
        if (controls && Object.values(controls).some((value) => value !== undefined)) {
            return { status: 'unsupported', reason: 'Chat switch requires an explicit request-control policy' };
        }
        if (target.provider !== this.provider) {
            return { status: 'unsupported', reason: 'Model switch target provider differs from configured driver' };
        }
        const protocolOptions: OpenAIChatCompletionsProtocolOptions = {
            modelName: this.resolveChatCompletionsRequestModel({ model: target.model }),
            defaultMaxTokens: this.options.defaultMaxTokens,
            extraBody: this.options.extraBody,
            resultSchemaMode: this.options.resultSchemaMode,
            includeResultSchemaInPrompt: this.options.includeResultSchemaInPrompt,
            includeResultSchemaInPromptForModel: this.options.includeResultSchemaInPromptForModel,
            toolSchemaMode: this.options.toolSchemaMode,
        };
        return {
            status: 'compiled',
            native_request: await compileOpenAIChatModelSwitchRequest({
                document,
                target,
                protocol_options: protocolOptions,
                stream: operation === 'stream',
            }),
        };
    }

    protected override projectChatCompletionsTransportBody(payload: OpenAIChatCompletionsPayload): JsonValue {
        return providerJsonValue(
            payload.stream ? toOpenAIStreamingPayload(payload) : toOpenAINonStreamingPayload(payload),
        );
    }

    readonly provider = Providers.openai_compatible;
    service: OpenAI;

    override async execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<OpenAIChatCompletionsPrompt>> {
        if (!openAIAudioTask(options.model)) return super.execute(segments, options, signal);
        return executeOpenAIAudioRequest(
            this,
            this.service,
            segments,
            options,
            { _is_openai_chat_completions: true, messages: [] },
            signal,
            options.model,
            this.getDriverRequestOptions(options, signal),
        );
    }

    override async executeCanonical(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        if (!openAIAudioTask(options.model)) return super.executeCanonical(segments, options, signal);
        return executeOpenAIAudioCanonical({
            service: this.service,
            segments,
            options,
            provider: this.provider,
            request_options: this.getDriverRequestOptions(options, signal),
        });
    }

    override async stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<OpenAIChatCompletionsPrompt>> {
        if (!openAIAudioTask(options.model)) return super.stream(segments, options, signal);
        return new FallbackCompletionStream(
            this,
            { _is_openai_chat_completions: true, messages: [] },
            options,
            (streamSignal) =>
                this.execute(segments, options, signal ? AbortSignal.any([signal, streamSignal]) : streamSignal),
        );
    }

    override async streamCanonicalEvents(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        if (!openAIAudioTask(options.model)) {
            return super.streamCanonicalEvents(segments, options, signal, open);
        }
        return streamOpenAIAudioCanonicalEvents({
            segments,
            options,
            signal,
            open,
            execute: (streamSignal) => this.executeCanonical(segments, options, streamSignal),
        });
    }

    constructor(options: OpenAIChatCompletionsDriverConfig) {
        super(options);
        this.service = new OpenAI({
            apiKey: options.apiKey,
            baseURL: options.endpoint,
            defaultHeaders: options.default_headers,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
    }

    async _postChatCompletion(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse> {
        const request = toOpenAINonStreamingPayload(payload);
        const requestOptions = this.getDriverRequestOptions(options, signal);
        const response = requestOptions
            ? await this.service.chat.completions.create(request, requestOptions)
            : await this.service.chat.completions.create(request);
        return preserveOpenAIChatCompletionsOriginalResponse(
            normalizeOpenAIChatCompletionsResponse(response),
            response,
        );
    }

    async _postChatCompletionStream(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream> {
        const request = toOpenAIStreamingPayload(payload);
        const requestOptions = this.getDriverRequestOptions(options, signal);
        const stream = requestOptions
            ? await this.service.chat.completions.create(request, requestOptions)
            : await this.service.chat.completions.create(request);
        return openAIChatCompletionsStreamToSSE(normalizeOpenAIChatCompletionsStream(stream), () =>
            stream.controller.abort(),
        );
    }

    async listModels(): Promise<AIModel[]> {
        return (await this.service.models.list()).data
            .filter(
                (model) =>
                    !isEmbeddingModel({ id: model.id }, this.provider) &&
                    (!isDedicatedInferenceModel(model.id, this.provider) || !!openAIAudioTask(model.id)),
            )
            .map((model) => {
                const modelMetadata = resolveModelListingMetadata(model.id, this.provider);
                return {
                    id: model.id,
                    name: model.id,
                    owner: model.owned_by,
                    provider: this.provider,
                    type: openAIAudioTask(model.id) ? ModelType.Audio : ModelType.Text,
                    ...modelMetadata,
                } satisfies AIModel;
            });
    }

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.models.list();
            return true;
        } catch {
            return false;
        }
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        const model = normalized.model ?? OPENAI_DEFAULT_EMBEDDING_MODEL;
        const input = normalized.inputs.map((item) => {
            if (item.type !== 'text') {
                throw new Error(`Provider '${this.provider}' supports only text embeddings.`);
            }
            return item.text;
        });
        const response = await this.service.embeddings.create({
            input,
            model,
            ...(normalized.dimensions ? { dimensions: normalized.dimensions } : {}),
            encoding_format: 'float',
        });
        const results = [...response.data]
            .sort((a, b) => a.index - b.index)
            .map((entry): EmbeddingResultItem => ({ outputs: [{ values: entry.embedding, modality: 'text' }] }));
        return {
            model,
            results,
            usage: response.usage
                ? {
                      input_tokens: response.usage.prompt_tokens,
                      input_text_tokens: response.usage.prompt_tokens,
                  }
                : undefined,
        };
    }
}

/**
 * Compile a new target through the same adapter, history policy, payload builder and SDK body
 * converter used by transport. This compatible-only path rejects implicit history rewriting.
 */
export async function compileOpenAIChatModelSwitchRequest(input: {
    document: ConversationDocument;
    target: ModelTarget;
    protocol_options: OpenAIChatCompletionsProtocolOptions;
    stream: boolean;
}) {
    const document = parseConversationDocument(input.document);
    const target = ModelTargetSchema.parse(structuredClone(input.target));
    const { includeResultSchemaInPromptForModel, ...serializableProtocolOptions } = input.protocol_options;
    // A configured model policy callback is intentionally shared by identity. All serializable
    // options are owned before the first await, including nested extra-body values.
    const protocolOptions: OpenAIChatCompletionsProtocolOptions = {
        ...structuredClone(serializableProtocolOptions),
        ...(includeResultSchemaInPromptForModel ? { includeResultSchemaInPromptForModel } : {}),
    };
    const stream = input.stream;
    // Deployment aliases belong to the trusted host protocol configuration. Extra body may
    // extend provider-specific parameters, but may not replace the request we attest below.
    const protectedBodyFields = new Set([
        'model',
        'messages',
        'tools',
        'tool_choice',
        'stream',
        'stream_options',
        'response_format',
    ]);
    for (const field of Object.keys(protocolOptions.extraBody ?? {})) {
        if (protectedBodyFields.has(field))
            throw new Error(`OpenAI Chat model switch extraBody cannot override ${field}`);
    }
    if (
        target.protocol !== OPENAI_CHAT_COMPLETIONS_PROTOCOL ||
        target.adapter_version !== OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION
    ) {
        throw new Error('OpenAI Chat model switch target has an unsupported protocol or adapter version');
    }
    const tools = document.context.active_tool_definition_ids.map((id) => {
        const definition = document.tool_definitions[id];
        if (!definition) throw new Error(`OpenAI Chat model switch tool ${id} is unavailable`);
        return definition;
    });
    const selected = selectedCanonicalTurns(document, { allow_interrupted_with_complete_tool_calls: true });
    const unsupportedMedia = new Set(['image', 'document', 'audio', 'video', 'extension']);
    for (const turn of selected) {
        for (const block of turn.blocks) {
            if (unsupportedMedia.has(block.type))
                throw new Error(`OpenAI Chat model switch requires an explicit ${block.type} media policy`);
            if (block.type === 'tool_result' && block.content.some((item) => unsupportedMedia.has(item.type)))
                throw new Error('OpenAI Chat model switch requires an explicit nested tool-result media policy');
        }
    }
    const modelOptions = target.options === undefined ? undefined : ModelOptionsSchema.parse(target.options);
    if (
        target.options !== undefined &&
        (await fingerprintJson(modelOptions)) !== (await fingerprintJson(target.options))
    ) {
        throw new Error('OpenAI Chat model switch target contains unsupported model options');
    }
    const options: ExecutionOptions = {
        model: target.model,
        ...(modelOptions === undefined ? {} : { model_options: modelOptions }),
    };
    const compiled = compileOpenAIChatCompletionsConversation(document, target);
    const projected = projectOpenAIChatCompletionsHistory(
        compiled.conversation,
        options,
        canonicalConversationTurnNumber(document),
    );
    if ((await fingerprintJson(projected.messages)) !== (await fingerprintJson(compiled.conversation.messages))) {
        throw new Error('OpenAI Chat model switch requires an explicit history or media transformation');
    }
    const prepared = prepareOpenAIChatCompletionsConversation(projected, { model: target.model, tools });
    if ((await fingerprintJson(prepared.messages)) !== (await fingerprintJson(projected.messages))) {
        throw new Error('OpenAI Chat model switch requires an explicit tool or replay transformation');
    }
    const payload = buildOpenAIChatCompletionsPayload(
        prepared,
        options,
        protocolOptions,
        protocolOptions.modelName ?? target.model,
        stream,
        target.provider,
        tools,
    );
    return providerJsonValue(stream ? toOpenAIStreamingPayload(payload) : toOpenAINonStreamingPayload(payload));
}
