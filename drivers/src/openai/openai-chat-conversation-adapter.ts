import type { ExecutionOptions } from '@llumiverse/common';
import {
    type AgentContentBlock,
    type Asset,
    appendConversationRecords,
    type ConversationDocument,
    type ConversationTurn,
    type DecodedConversationResponse,
    deriveConversationId,
    type ExecutedGeneration,
    type ExecutionReceipt,
    fingerprintJson,
    type GenerationUsage,
    type ImportedTurnProvenance,
    inlineAssetContentIntegrity,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    type NativeReplayBlock,
    type NestedToolResultContentBlock,
    type PreparedConversationRequest,
    type ProgramContentBlock,
    parseConversationDocument,
    preflightJsonInput,
    type ToolDefinition,
    type ToolResultBlock,
    toolArgumentsForModel,
    type UserContentBlock,
} from '@llumiverse/conversation';
import { type CanonicalStructuredOutput, canonicalToolSelectionPolicy } from '@llumiverse/core';
import {
    acceptedCanonicalRequestDocument,
    acceptedCanonicalResponse,
    appendCanonicalDecodedResponse,
    appendCanonicalPrompt,
    assertProtectedReplayCompatibility,
    type CanonicalPreparedState,
    canonicalResponseIdentities,
    canonicalToolSelectionTargetOptions,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    parseCanonicalConversation,
    providerJsonValue,
    resolveCanonicalToolDefinitions,
    resolveConversationRuntime,
    selectedCanonicalTurns,
} from '../conversation/canonical-runtime.js';
import {
    fingerprintNativeConversationImport,
    guardNativeConversationImport,
    type NativeConversationImportContext,
    type NativeConversationImportOptions,
    type NativeConversationImportResult,
    nativeConversationImportContext,
    nativeConversationImportResult,
    newNativeImportDocument,
    snapshotNativeConversationImportOptions,
} from '../conversation/native-import.js';
import {
    assertStructuredOutputEvidence,
    normalizeDecodedStructuredOutput,
    parseStructuredOutputEvidence,
    remapStructuredOutputReplayDependencies,
    structuredOutputEvidence,
} from '../conversation/structured-output.js';
import type {
    OpenAIChatCompletionsContentPart,
    OpenAIChatCompletionsImageUrlPart,
    OpenAIChatCompletionsMessage,
    OpenAIChatCompletionsPayload,
    OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsResponse,
    OpenAIChatProviderReplay,
} from './openai_chat_completions.js';

export const OPENAI_CHAT_COMPLETIONS_PROTOCOL = 'openai.chat.completions' as const;
export const OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION = '2026-09-11.canonical.1' as const;

type SourceKind = 'imported' | 'received';

type OpenAIReplayPayload = JsonObject & {
    type: 'openai_chat_assistant_fields';
    reasoning_content?: string | null;
    reasoning?: string | null;
    provider_replay?: OpenAIChatProviderReplay;
    tool_arguments?: Array<{ call_id: string; raw: string }>;
    structured_output?: JsonObject & {
        evidence: JsonObject;
        content: JsonValue;
    };
};

export interface PreparedOpenAIChatConversation
    extends CanonicalPreparedState<OpenAIChatCompletionsPrompt>,
        PreparedConversationRequest<OpenAIChatCompletionsPayload> {
    provider: string;
    requested_model: string;
    prior_native_message_count: number;
}

function ownValue(value: object, key: string): unknown {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && 'value' in descriptor ? descriptor.value : undefined;
}

function isOpenAIContentPart(value: unknown): value is OpenAIChatCompletionsContentPart {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const type = ownValue(value, 'type');
    if (type === 'text') return typeof ownValue(value, 'text') === 'string';
    if (type === 'input_audio') {
        const inputAudio = ownValue(value, 'input_audio');
        if (typeof inputAudio !== 'object' || inputAudio === null || Array.isArray(inputAudio)) return false;
        const format = ownValue(inputAudio, 'format');
        return typeof ownValue(inputAudio, 'data') === 'string' && (format === 'mp3' || format === 'wav');
    }
    if (type !== 'image_url') return false;
    const imageUrl = ownValue(value, 'image_url');
    if (typeof imageUrl !== 'object' || imageUrl === null || Array.isArray(imageUrl)) return false;
    const url = ownValue(imageUrl, 'url');
    const detail = ownValue(imageUrl, 'detail');
    return (
        typeof url === 'string' && (detail === undefined || detail === 'auto' || detail === 'low' || detail === 'high')
    );
}

function isOpenAIProviderReplay(value: unknown): value is OpenAIChatProviderReplay {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    return (
        typeof ownValue(value, 'provider') === 'string' &&
        typeof ownValue(value, 'protocol') === 'string' &&
        typeof ownValue(value, 'adapter_version') === 'string' &&
        preflightJsonInput(ownValue(value, 'payload')).success
    );
}

function isOpenAIToolCall(value: unknown): boolean {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const fn = ownValue(value, 'function');
    return (
        typeof ownValue(value, 'id') === 'string' &&
        ownValue(value, 'type') === 'function' &&
        typeof fn === 'object' &&
        fn !== null &&
        !Array.isArray(fn) &&
        typeof ownValue(fn, 'name') === 'string' &&
        typeof ownValue(fn, 'arguments') === 'string'
    );
}

function isOpenAIMessage(value: unknown): value is OpenAIChatCompletionsMessage {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const role = ownValue(value, 'role');
    if (role !== 'system' && role !== 'developer' && role !== 'user' && role !== 'assistant' && role !== 'tool') {
        return false;
    }
    const content = ownValue(value, 'content');
    if (
        content !== undefined &&
        content !== null &&
        typeof content !== 'string' &&
        !(Array.isArray(content) && content.every(isOpenAIContentPart))
    ) {
        return false;
    }
    const toolCallId = ownValue(value, 'tool_call_id');
    if (toolCallId !== undefined && typeof toolCallId !== 'string') return false;
    const toolResultStatus = ownValue(value, 'tool_result_status');
    if (
        toolResultStatus !== undefined &&
        toolResultStatus !== 'success' &&
        toolResultStatus !== 'error' &&
        toolResultStatus !== 'cancelled' &&
        toolResultStatus !== 'denied'
    ) {
        return false;
    }
    const toolCalls = ownValue(value, 'tool_calls');
    if (toolCalls !== undefined && !(Array.isArray(toolCalls) && toolCalls.every(isOpenAIToolCall))) return false;
    const providerReplay = ownValue(value, 'provider_replay');
    if (providerReplay !== undefined && !isOpenAIProviderReplay(providerReplay)) return false;
    return role !== 'tool' || typeof toolCallId === 'string';
}

export function isOpenAIChatCompletionsHistory(
    value: unknown,
    explicitProtocol?: typeof OPENAI_CHAT_COMPLETIONS_PROTOCOL,
): value is OpenAIChatCompletionsPrompt | OpenAIChatCompletionsMessage[] {
    const preflight = preflightJsonInput(value);
    if (!preflight.success) return false;
    if (Array.isArray(value)) {
        return explicitProtocol === OPENAI_CHAT_COMPLETIONS_PROTOCOL && value.every(isOpenAIMessage);
    }
    if (typeof value !== 'object' || value === null) return false;
    const messages = ownValue(value, 'messages');
    const discriminator = ownValue(value, '_is_openai_chat_completions');
    return discriminator === true && Array.isArray(messages) && messages.every(isOpenAIMessage);
}

function historyMessages(
    history: OpenAIChatCompletionsPrompt | OpenAIChatCompletionsMessage[],
): OpenAIChatCompletionsMessage[] {
    return Array.isArray(history) ? history : history.messages;
}

function sourceHistoryTurnNumber(
    history: OpenAIChatCompletionsPrompt | OpenAIChatCompletionsMessage[],
): number | undefined {
    if (Array.isArray(history)) return undefined;
    const metadata = ownValue(history, '_llumiverse_meta');
    if (typeof metadata !== 'object' || metadata === null || Array.isArray(metadata)) return undefined;
    const turnNumber = ownValue(metadata, 'turnNumber');
    return typeof turnNumber === 'number' && Number.isSafeInteger(turnNumber) && turnNumber >= 0
        ? turnNumber
        : undefined;
}

async function entityId(kind: string, scope: string, ...position: Array<string | number>): Promise<string> {
    return deriveConversationId(kind, scope, ...position.map(String));
}

function importedProvenance(index: number, sourceHistoryTurnNumber?: number): ImportedTurnProvenance {
    return {
        type: 'imported' as const,
        source: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        native_id: {
            protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            scope: 'history',
            value: `messages/${index}`,
        },
        ...(sourceHistoryTurnNumber === undefined ? {} : { source_history_turn_number: sourceHistoryTurnNumber }),
        missing_metadata: ['actor_id', 'timestamps', 'exchange'],
    };
}

function receivedProvenance() {
    return { type: 'received' as const };
}

function isProgramContentBlock(block: AgentContentBlock): block is ProgramContentBlock {
    return block.type !== 'tool_call';
}

function isUserContentBlock(block: AgentContentBlock): block is UserContentBlock {
    return block.type !== 'tool_call' && block.type !== 'reasoning' && block.type !== 'native_replay';
}

function isNestedToolResultContentBlock(block: AgentContentBlock): block is NestedToolResultContentBlock {
    return block.type !== 'tool_call';
}

function reasoningReplayPayload(message: OpenAIChatCompletionsMessage): OpenAIReplayPayload | undefined {
    if (
        message.reasoning_content === undefined &&
        message.reasoning === undefined &&
        message.provider_replay === undefined
    ) {
        return undefined;
    }
    return {
        type: 'openai_chat_assistant_fields',
        ...(message.reasoning_content === undefined ? {} : { reasoning_content: message.reasoning_content }),
        ...(message.reasoning === undefined ? {} : { reasoning: message.reasoning }),
        ...(message.provider_replay === undefined ? {} : { provider_replay: message.provider_replay }),
    };
}

function parseToolArguments(
    raw: string,
): { type: 'json'; value: JsonValue } | { type: 'invalid'; raw: string; error: string } {
    try {
        return { type: 'json', value: JSON.parse(raw) as JsonValue };
    } catch (error: unknown) {
        return { type: 'invalid', raw, error: error instanceof Error ? error.message : String(error) };
    }
}

function dataImage(url: string): { mime_type: string; data: string } | undefined {
    const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    return match ? { mime_type: match[1], data: match[2] } : undefined;
}

async function imageRecord(input: {
    part: OpenAIChatCompletionsImageUrlPart;
    turn_id: string;
    scope: string;
    message_index: number;
    block_index: number;
    source: SourceKind;
    recorded_at: string;
}): Promise<{ block: UserContentBlock; asset: Asset }> {
    const blockId = await entityId('block', input.scope, input.message_index, input.block_index);
    const assetId = await entityId('asset', input.scope, input.message_index, input.block_index);
    const parsed = dataImage(input.part.image_url.url);
    const storage = parsed
        ? { type: 'inline_base64' as const, data: parsed.data }
        : {
              type: 'external' as const,
              resolver: 'url',
              locator: {
                  url: input.part.image_url.url,
                  ...(input.part.image_url.detail === undefined ? {} : { detail: input.part.image_url.detail }),
              },
          };
    const integrity = await inlineAssetContentIntegrity(storage);
    const asset: Asset = {
        id: assetId,
        kind: 'image',
        mime_type: parsed?.mime_type ?? 'application/octet-stream',
        storage,
        provenance:
            input.source === 'imported'
                ? { type: 'imported', source: OPENAI_CHAT_COMPLETIONS_PROTOCOL }
                : { type: 'received', source_turn_id: input.turn_id },
        ...(integrity ?? {}),
        created_at: input.recorded_at,
        ...(input.part.image_url.detail === undefined
            ? {}
            : {
                  metadata: {
                      openai_chat_completions: { image_detail: input.part.image_url.detail },
                  },
              }),
    };
    return { block: { id: blockId, type: 'image', asset_id: assetId }, asset };
}

async function audioRecord(input: {
    part: Extract<OpenAIChatCompletionsContentPart, { type: 'input_audio' }>;
    turn_id: string;
    scope: string;
    message_index: number;
    block_index: number;
    source: SourceKind;
    recorded_at: string;
}): Promise<{ block: UserContentBlock; asset: Asset }> {
    const blockId = await entityId('block', input.scope, input.message_index, input.block_index);
    const assetId = await entityId('asset', input.scope, input.message_index, input.block_index);
    const mimeType = input.part.input_audio.format === 'wav' ? 'audio/wav' : 'audio/mpeg';
    const storage = { type: 'inline_base64' as const, data: input.part.input_audio.data };
    const integrity = await inlineAssetContentIntegrity(storage);
    const asset: Asset = {
        id: assetId,
        kind: 'audio',
        mime_type: mimeType,
        storage,
        provenance:
            input.source === 'imported'
                ? { type: 'imported', source: OPENAI_CHAT_COMPLETIONS_PROTOCOL }
                : { type: 'received', source_turn_id: input.turn_id },
        ...(integrity ?? {}),
        created_at: input.recorded_at,
        metadata: { openai_chat_completions: { audio_format: input.part.input_audio.format } },
    };
    return { block: { id: blockId, type: 'audio', asset_id: assetId }, asset };
}

async function messageRecords(input: {
    message: OpenAIChatCompletionsMessage;
    message_index: number;
    scope: string;
    source: SourceKind;
    runtime: NativeConversationImportContext;
    provider: string;
    model?: string;
    tool_definitions: readonly ToolDefinition[];
    source_history_turn_number?: number;
}): Promise<{
    turns: ConversationTurn[];
    assets: Asset[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}> {
    const { message, message_index: messageIndex, scope, source, runtime } = input;
    const turnId = await entityId('turn', scope, messageIndex);
    const blocks: AgentContentBlock[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [
        { canonical_id: turnId, native_id: `messages/${messageIndex}`, kind: 'turn' },
    ];
    const executionReceipts: ExecutionReceipt[] = [];

    const content = message.content;
    const parts: OpenAIChatCompletionsContentPart[] =
        typeof content === 'string'
            ? [{ type: 'text', text: content }]
            : Array.isArray(content)
              ? content
              : content === null
                ? []
                : [];
    for (let blockIndex = 0; blockIndex < parts.length; blockIndex += 1) {
        const part = parts[blockIndex];
        if (part.type === 'text') {
            const blockId = await entityId('block', scope, messageIndex, blockIndex);
            blocks.push({ id: blockId, type: 'text', text: part.text, format: 'plain' });
            mappings.push({
                canonical_id: blockId,
                native_id: `messages/${messageIndex}/content/${blockIndex}`,
                kind: 'block',
            });
        } else if (part.type === 'image_url') {
            const converted = await imageRecord({
                part,
                turn_id: turnId,
                scope,
                message_index: messageIndex,
                block_index: blockIndex,
                source,
                recorded_at: runtime.recorded_at,
            });
            blocks.push(converted.block);
            assets.push(converted.asset);
            mappings.push({
                canonical_id: converted.block.id,
                native_id: `messages/${messageIndex}/content/${blockIndex}`,
                kind: 'block',
            });
        } else if (part.type === 'input_audio') {
            const converted = await audioRecord({
                part,
                turn_id: turnId,
                scope,
                message_index: messageIndex,
                block_index: blockIndex,
                source,
                recorded_at: runtime.recorded_at,
            });
            blocks.push(converted.block);
            assets.push(converted.asset);
            mappings.push({
                canonical_id: converted.block.id,
                native_id: `messages/${messageIndex}/content/${blockIndex}`,
                kind: 'block',
            });
        }
    }

    if (message.role === 'assistant') {
        const reasoningValues = [message.reasoning_content, message.reasoning].filter(
            (value): value is string => typeof value === 'string',
        );
        for (let reasoningIndex = 0; reasoningIndex < reasoningValues.length; reasoningIndex += 1) {
            const blockId = await entityId('reasoning', scope, messageIndex, reasoningIndex);
            blocks.push({
                id: blockId,
                type: 'reasoning',
                text: reasoningValues[reasoningIndex],
                representation: 'text',
            });
            mappings.push({
                canonical_id: blockId,
                native_id: `messages/${messageIndex}/reasoning/${reasoningIndex}`,
                kind: 'block',
            });
        }
        for (let callIndex = 0; callIndex < (message.tool_calls?.length ?? 0); callIndex += 1) {
            const call = message.tool_calls?.[callIndex];
            if (call === undefined) continue;
            const blockId = await entityId('block', scope, messageIndex, 'tool_call', callIndex);
            const definition = input.tool_definitions.find((candidate) => candidate.name === call.function.name);
            blocks.push({
                id: blockId,
                type: 'tool_call',
                call_id: call.id,
                tool_name: call.function.name,
                ...(definition === undefined ? {} : { definition_id: definition.id }),
                executor: 'application',
                arguments: parseToolArguments(call.function.arguments),
                native_id: {
                    protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                    scope: runtime.request_id,
                    value: call.id,
                },
            });
            mappings.push(
                { canonical_id: blockId, native_id: `messages/${messageIndex}/tool_calls/${callIndex}`, kind: 'block' },
                { canonical_id: call.id, native_id: call.id, kind: 'call' },
            );
        }
        const replay = reasoningReplayPayload(message);
        if (replay !== undefined) {
            const blockId = await entityId('replay', scope, messageIndex, 'reasoning');
            const dependencyBlocks = blocks
                .filter(
                    (block) =>
                        block.type === 'reasoning' ||
                        (replay.provider_replay !== undefined && block.type !== 'native_replay'),
                )
                .map((block) => block.id);
            blocks.push({
                id: blockId,
                type: 'native_replay',
                adapter: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                compatibility_scope: {
                    provider: input.provider,
                    ...(input.model === undefined ? {} : { model: input.model }),
                    protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                    adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                },
                payload: replay,
                dependencies: {
                    turn_ids: [turnId],
                    block_ids: dependencyBlocks,
                    call_ids: [],
                    request_ids: [],
                },
            });
            mappings.push({
                canonical_id: blockId,
                native_id: `messages/${messageIndex}/assistant_fields`,
                kind: 'block',
            });
        }
        for (let callIndex = 0; callIndex < (message.tool_calls?.length ?? 0); callIndex += 1) {
            const call = message.tool_calls?.[callIndex];
            if (call === undefined) continue;
            const callBlock = blocks.find((block) => block.type === 'tool_call' && block.call_id === call.id);
            if (callBlock === undefined) throw new TypeError(`OpenAI Chat tool call ${call.id} has no canonical block`);
            const blockId = await entityId('replay', scope, messageIndex, 'tool_arguments', callIndex);
            blocks.push({
                id: blockId,
                type: 'native_replay',
                adapter: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                compatibility_scope: {
                    provider: input.provider,
                    protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                    adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                },
                payload: {
                    type: 'openai_chat_assistant_fields',
                    tool_arguments: [{ call_id: call.id, raw: call.function.arguments }],
                },
                dependencies: {
                    turn_ids: [],
                    block_ids: [callBlock.id],
                    call_ids: [call.id],
                    request_ids: [],
                },
                dependency_policy: 'discard_on_dependency_change',
            });
            mappings.push({
                canonical_id: blockId,
                native_id: `messages/${messageIndex}/tool_calls/${callIndex}/arguments`,
                kind: 'block',
            });
        }
    }

    const common = {
        id: turnId,
        status: 'completed' as const,
        timestamps: { recorded_at: runtime.recorded_at },
        model_visibility: 'include' as const,
        provenance:
            source === 'imported'
                ? importedProvenance(messageIndex, input.source_history_turn_number)
                : receivedProvenance(),
    };
    if (message.role === 'system' || message.role === 'developer') {
        return {
            turns: [
                {
                    ...common,
                    kind: 'program',
                    authority: message.role === 'system' ? 'system' : 'developer',
                    blocks: blocks.filter(isProgramContentBlock),
                },
            ],
            assets,
            mappings,
            execution_receipts: executionReceipts,
        };
    }
    if (message.role === 'assistant') {
        const turn: ConversationTurn =
            source === 'imported'
                ? {
                      ...common,
                      kind: 'agent',
                      authority: 'ordinary',
                      blocks,
                      provenance: importedProvenance(messageIndex, input.source_history_turn_number),
                  }
                : {
                      ...common,
                      kind: 'agent',
                      authority: 'ordinary',
                      blocks,
                      provenance: receivedProvenance(),
                  };
        return { turns: [turn], assets, mappings, execution_receipts: executionReceipts };
    }
    if (message.role === 'tool') {
        if (!message.tool_call_id) throw new Error('OpenAI Chat tool message is missing tool_call_id');
        const resultBlockId = await entityId('block', scope, messageIndex, 'tool_result');
        const resultBlock: ToolResultBlock = {
            id: resultBlockId,
            type: 'tool_result',
            call_id: message.tool_call_id,
            status: message.tool_result_status ?? 'unknown',
            content: blocks.filter(isNestedToolResultContentBlock),
            native_id: {
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                scope: runtime.request_id,
                value: message.tool_call_id,
            },
        };
        if (message.tool_result_status !== undefined) {
            executionReceipts.push({
                id: await entityId('execution_receipt', scope, message.tool_call_id),
                call_id: message.tool_call_id,
                executor: 'application',
                status: message.tool_result_status,
                result_turn_id: turnId,
                result_fingerprint: await fingerprintJson(resultBlock),
                recorded_at: runtime.recorded_at,
            });
        }
        mappings.push(
            { canonical_id: resultBlockId, native_id: `messages/${messageIndex}`, kind: 'block' },
            { canonical_id: message.tool_call_id, native_id: message.tool_call_id, kind: 'call' },
        );
        return {
            turns: [{ ...common, kind: 'tool', authority: 'ordinary', blocks: [resultBlock] }],
            assets,
            mappings,
            execution_receipts: executionReceipts,
        };
    }
    return {
        turns: [{ ...common, kind: 'user', authority: 'ordinary', blocks: blocks.filter(isUserContentBlock) }],
        assets,
        mappings,
        execution_receipts: executionReceipts,
    };
}

async function messagesToRecords(input: {
    messages: readonly OpenAIChatCompletionsMessage[];
    scope: string;
    source: SourceKind;
    runtime: NativeConversationImportContext;
    provider: string;
    model?: string;
    tool_definitions: readonly ToolDefinition[];
    source_history_turn_number?: number;
}): Promise<{
    turns: ConversationTurn[];
    assets: Asset[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}> {
    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    const executionReceipts: ExecutionReceipt[] = [];
    for (let index = 0; index < input.messages.length; index += 1) {
        const converted = await messageRecords({
            message: input.messages[index],
            message_index: index,
            scope: input.scope,
            source: input.source,
            runtime: input.runtime,
            provider: input.provider,
            ...(input.model === undefined ? {} : { model: input.model }),
            tool_definitions: input.tool_definitions,
            ...(input.source_history_turn_number === undefined
                ? {}
                : { source_history_turn_number: input.source_history_turn_number }),
        });
        turns.push(...converted.turns);
        assets.push(...converted.assets);
        mappings.push(...converted.mappings);
        executionReceipts.push(...converted.execution_receipts);
    }
    return { turns, assets, mappings, execution_receipts: executionReceipts };
}

function blockReplay(
    turn: ConversationTurn,
    target?: { provider?: string; model?: string },
): OpenAIReplayPayload | undefined {
    const replayBlocks = turn.blocks.filter((block) => block.type === 'native_replay');
    const matching = replayBlocks.filter((block) => block.protocol === OPENAI_CHAT_COMPLETIONS_PROTOCOL);
    const foreign = replayBlocks.find(
        (block) =>
            block.protocol !== OPENAI_CHAT_COMPLETIONS_PROTOCOL &&
            block.dependency_policy !== 'discard_on_dependency_change',
    );
    if (foreign !== undefined) {
        throw new TypeError(`OpenAI Chat cannot discard protected ${foreign.protocol} replay block ${foreign.id}`);
    }
    if (matching.length === 0) return undefined;
    let reasoningContent: OpenAIReplayPayload['reasoning_content'];
    let reasoning: OpenAIReplayPayload['reasoning'];
    let providerReplay: OpenAIReplayPayload['provider_replay'];
    let structuredOutput: OpenAIReplayPayload['structured_output'];
    const toolArguments = new Map<string, string>();
    for (const replay of matching) {
        if (typeof replay.payload !== 'object' || replay.payload === null || Array.isArray(replay.payload)) {
            throw new TypeError(`OpenAI Chat replay block ${replay.id} has an unsupported payload`);
        }
        if (
            replay.adapter !== OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION ||
            replay.compatibility_scope.protocol !== OPENAI_CHAT_COMPLETIONS_PROTOCOL ||
            replay.compatibility_scope.adapter_version !== OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION ||
            (target?.provider !== undefined && replay.compatibility_scope.provider !== target.provider) ||
            (target?.model !== undefined &&
                replay.compatibility_scope.model !== undefined &&
                replay.compatibility_scope.model !== target.model)
        ) {
            throw new TypeError(`OpenAI Chat replay block ${replay.id} is outside its compatibility scope`);
        }
        if (replay.payload.type !== 'openai_chat_assistant_fields') {
            throw new TypeError(`OpenAI Chat replay block ${replay.id} has an unsupported payload`);
        }
        const payload = replay.payload as OpenAIReplayPayload;
        if (payload.reasoning_content !== undefined) {
            if (reasoningContent !== undefined) throw new TypeError(`OpenAI Chat reasoning replay is duplicated`);
            reasoningContent = payload.reasoning_content;
        }
        if (payload.reasoning !== undefined) {
            if (reasoning !== undefined) throw new TypeError(`OpenAI Chat reasoning replay is duplicated`);
            reasoning = payload.reasoning;
        }
        if (payload.provider_replay !== undefined) {
            if (providerReplay !== undefined) throw new TypeError(`OpenAI Chat provider replay is duplicated`);
            if (target?.provider !== undefined && payload.provider_replay.provider !== target.provider) {
                throw new TypeError(
                    `OpenAI Chat provider replay for ${payload.provider_replay.provider} cannot be projected to ${target.provider}`,
                );
            }
            providerReplay = structuredClone(payload.provider_replay);
        }
        if (payload.structured_output !== undefined) {
            if (structuredOutput !== undefined) throw new TypeError(`OpenAI Chat structured replay is duplicated`);
            structuredOutput = payload.structured_output;
        }
        for (const value of payload.tool_arguments ?? []) {
            if (toolArguments.has(value.call_id)) {
                throw new TypeError(`OpenAI Chat raw arguments for call ${value.call_id} are duplicated`);
            }
            const call = turn.blocks.find((block) => block.type === 'tool_call' && block.call_id === value.call_id);
            let matchesCanonicalArguments = false;
            if (call?.type === 'tool_call') {
                if (call.arguments.type === 'invalid') {
                    matchesCanonicalArguments = call.arguments.raw === value.raw;
                } else if (call.arguments.type === 'json') {
                    try {
                        matchesCanonicalArguments =
                            JSON.stringify(JSON.parse(value.raw)) === JSON.stringify(call.arguments.value);
                    } catch {
                        matchesCanonicalArguments = false;
                    }
                }
            }
            if (!matchesCanonicalArguments) {
                if (replay.dependency_policy === 'discard_on_dependency_change') continue;
                throw new TypeError(
                    `OpenAI Chat raw arguments replay ${replay.id} no longer matches canonical call ${value.call_id}`,
                );
            }
            toolArguments.set(value.call_id, value.raw);
        }
    }
    return {
        type: 'openai_chat_assistant_fields',
        ...(reasoningContent === undefined ? {} : { reasoning_content: reasoningContent }),
        ...(reasoning === undefined ? {} : { reasoning }),
        ...(providerReplay === undefined ? {} : { provider_replay: providerReplay }),
        ...(structuredOutput === undefined ? {} : { structured_output: structuredOutput }),
        ...(toolArguments.size === 0
            ? {}
            : { tool_arguments: [...toolArguments].map(([call_id, raw]) => ({ call_id, raw })) }),
    };
}

function assetPart(asset: Asset): OpenAIChatCompletionsImageUrlPart {
    const metadata = asset.metadata?.openai_chat_completions;
    const detail =
        typeof metadata === 'object' && metadata !== null && !Array.isArray(metadata)
            ? metadata.image_detail
            : undefined;
    const imageDetail = detail === 'auto' || detail === 'low' || detail === 'high' ? detail : undefined;
    if (asset.storage.type === 'inline_base64') {
        return {
            type: 'image_url',
            image_url: {
                url: `data:${asset.mime_type};base64,${asset.storage.data}`,
                ...(imageDetail === undefined ? {} : { detail: imageDetail }),
            },
        };
    }
    if (asset.storage.type === 'external' && typeof asset.storage.locator.url === 'string') {
        const locatorDetail = asset.storage.locator.detail;
        return {
            type: 'image_url',
            image_url: {
                url: asset.storage.locator.url,
                ...(locatorDetail === 'auto' || locatorDetail === 'low' || locatorDetail === 'high'
                    ? { detail: locatorDetail }
                    : imageDetail === undefined
                      ? {}
                      : { detail: imageDetail }),
            },
        };
    }
    throw new Error(`OpenAI Chat cannot resolve image asset ${asset.id}`);
}

function audioAssetPart(asset: Asset): Extract<OpenAIChatCompletionsContentPart, { type: 'input_audio' }> {
    if (asset.kind !== 'audio' || asset.storage.type !== 'inline_base64') {
        throw new Error(`OpenAI Chat cannot resolve audio asset ${asset.id}`);
    }
    const format = asset.mime_type === 'audio/wav' || asset.mime_type === 'audio/x-wav' ? 'wav' : 'mp3';
    if (format === 'mp3' && asset.mime_type !== 'audio/mpeg' && asset.mime_type !== 'audio/mp3') {
        throw new Error(`OpenAI Chat cannot project audio MIME type ${asset.mime_type}`);
    }
    return { type: 'input_audio', input_audio: { data: asset.storage.data, format } };
}

function contentParts(turn: ConversationTurn, document: ConversationDocument): OpenAIChatCompletionsContentPart[] {
    const parts: OpenAIChatCompletionsContentPart[] = [];
    for (const block of turn.blocks) {
        if (block.type === 'text') parts.push({ type: 'text', text: block.text });
        else if (block.type === 'json') parts.push({ type: 'text', text: JSON.stringify(block.value) });
        else if (block.type === 'image') {
            const asset = document.assets[block.asset_id];
            if (asset === undefined) throw new Error(`OpenAI Chat content references missing asset ${block.asset_id}`);
            parts.push(assetPart(asset));
        } else if (block.type === 'audio') {
            const asset = document.assets[block.asset_id];
            if (asset === undefined) throw new Error(`OpenAI Chat content references missing asset ${block.asset_id}`);
            parts.push(audioAssetPart(asset));
        } else if (
            block.type !== 'tool_call' &&
            block.type !== 'reasoning' &&
            block.type !== 'native_replay' &&
            (block.type !== 'extension' || block.model_projection !== 'excluded')
        ) {
            throw new TypeError(`OpenAI Chat cannot project canonical ${block.type} block ${block.id}`);
        }
    }
    return parts;
}

function messageContent(parts: OpenAIChatCompletionsContentPart[]): string | OpenAIChatCompletionsContentPart[] | null {
    if (parts.length === 0) return null;
    if (parts.length === 1 && parts[0].type === 'text') return parts[0].text;
    return parts;
}

function chatContentTexts(content: JsonValue): string[] {
    if (typeof content === 'string') return [content];
    if (!Array.isArray(content)) return [];
    return content.flatMap((part) => {
        if (typeof part !== 'object' || part === null || Array.isArray(part)) return [];
        return part.type === 'text' && typeof part.text === 'string' ? [part.text] : [];
    });
}

function structuredChatContent(
    turn: ConversationTurn,
    replay: OpenAIReplayPayload | undefined,
): OpenAIChatCompletionsMessage['content'] | undefined {
    const structured = replay?.structured_output;
    if (structured === undefined) return undefined;
    const evidence = parseStructuredOutputEvidence(structured.evidence);
    assertStructuredOutputEvidence(turn, evidence, chatContentTexts(structured.content), `turn:${turn.id}`);
    return structuredClone(structured.content) as OpenAIChatCompletionsMessage['content'];
}

function compileTurn(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): OpenAIChatCompletionsMessage[] {
    if (turn.kind === 'tool') {
        const result = turn.blocks[0];
        const nested = result.content.flatMap((block): OpenAIChatCompletionsContentPart[] => {
            if (block.type === 'text') return [{ type: 'text', text: block.text }];
            if (block.type === 'json') return [{ type: 'text', text: JSON.stringify(block.value) }];
            if (block.type === 'image') {
                const asset = document.assets[block.asset_id];
                if (asset === undefined)
                    throw new Error(`OpenAI Chat tool result references missing asset ${block.asset_id}`);
                return [assetPart(asset)];
            }
            if (block.type === 'extension' && block.model_projection === 'excluded') return [];
            throw new TypeError(`OpenAI Chat cannot project canonical ${block.type} inside a tool result`);
        });
        return [{ role: 'tool', tool_call_id: result.call_id, content: messageContent(nested) }];
    }

    const replay = blockReplay(turn, target);
    const parts = contentParts(turn, document);
    const message: OpenAIChatCompletionsMessage = {
        role:
            turn.kind === 'agent'
                ? 'assistant'
                : turn.kind === 'program' && turn.authority === 'system'
                  ? 'system'
                  : turn.kind === 'program' && turn.authority === 'developer'
                    ? 'developer'
                    : 'user',
        content: messageContent(parts),
    };
    if (turn.kind === 'agent') {
        const structuredContent = structuredChatContent(turn, replay);
        if (structuredContent !== undefined) message.content = structuredContent;
        const rawArguments = new Map(replay?.tool_arguments?.map((value) => [value.call_id, value.raw]) ?? []);
        const calls = turn.blocks.flatMap((block) =>
            block.type === 'tool_call'
                ? [
                      {
                          id: block.call_id,
                          type: 'function' as const,
                          function: {
                              name: block.tool_name,
                              arguments:
                                  rawArguments.get(block.call_id) ??
                                  (block.arguments.type === 'invalid'
                                      ? block.arguments.raw
                                      : JSON.stringify(toolArgumentsForModel(block.arguments))),
                          },
                      },
                  ]
                : [],
        );
        if (calls.length > 0) message.tool_calls = calls;
        if (replay?.reasoning_content !== undefined) message.reasoning_content = replay.reasoning_content;
        if (replay?.reasoning !== undefined) message.reasoning = replay.reasoning;
        if (replay?.provider_replay !== undefined) message.provider_replay = replay.provider_replay;
        if (replay === undefined) {
            const reasoning = turn.blocks
                .filter((block) => block.type === 'reasoning')
                .map((block) => block.text)
                .join('');
            if (reasoning) message.reasoning_content = reasoning;
        }
    }
    return [message];
}

export function compileOpenAIChatCompletionsConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): ReturnType<typeof projectOpenAIChatCompletionsConversation> {
    return projectOpenAIChatCompletionsConversation(document, target);
}

function projectOpenAIChatCompletionsConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
    readOnlyCompatibilityProjection = false,
): {
    conversation: OpenAIChatCompletionsPrompt;
    mappings: NativeItemMapping[];
} {
    const messages: OpenAIChatCompletionsMessage[] = [];
    const mappings: NativeItemMapping[] = [];
    const selectedTurns = selectedCanonicalTurns(document, { allow_interrupted_with_complete_tool_calls: true });
    for (const turn of selectedTurns) {
        if (!readOnlyCompatibilityProjection)
            assertProtectedReplayCompatibility(document, turn, OPENAI_CHAT_COMPLETIONS_PROTOCOL, target);
    }
    for (const turn of selectedTurns) {
        const compiled = compileTurn(turn, document, target);
        const messageIndex = messages.length;
        messages.push(...compiled);
        mappings.push({ canonical_id: turn.id, native_id: `messages/${messageIndex}`, kind: 'turn' });
        for (const block of turn.blocks) {
            mappings.push({
                canonical_id: block.id,
                native_id: `messages/${messageIndex}/blocks/${block.id}`,
                kind: 'block',
            });
            if (block.type === 'tool_call') {
                mappings.push({ canonical_id: block.call_id, native_id: block.call_id, kind: 'call' });
            }
        }
    }
    return { conversation: { _is_openai_chat_completions: true, messages }, mappings };
}

/** Pure registered-protocol import. The report explicitly leaves continuation readiness unvalidated. */
export async function importOpenAIChatCompletionsHistory(
    historyInput: unknown,
    options: NativeConversationImportOptions,
): Promise<NativeConversationImportResult> {
    return guardNativeConversationImport(async () => {
        options = snapshotNativeConversationImportOptions(options);
        const runtime = nativeConversationImportContext(options);
        const toolDefinitions = [...(options.tool_definitions ?? [])];
        let document = newNativeImportDocument(options);
        if (!isOpenAIChatCompletionsHistory(historyInput, OPENAI_CHAT_COMPLETIONS_PROTOCOL)) {
            throw new TypeError('Conversation is neither canonical nor registered OpenAI Chat Completions history');
        }
        const historySnapshot = structuredClone(historyInput);
        const imported = await messagesToRecords({
            messages: historyMessages(historySnapshot),
            scope: `${runtime.conversation_id}:legacy`,
            source: 'imported',
            runtime,
            provider: options.provider,
            ...(options.model === undefined ? {} : { model: options.model }),
            tool_definitions: toolDefinitions,
            ...(sourceHistoryTurnNumber(historySnapshot) === undefined
                ? {}
                : { source_history_turn_number: sourceHistoryTurnNumber(historySnapshot) }),
        });
        const importedEntries = await Promise.all(
            imported.turns.map(async (turn, index) => ({
                id: await entityId('context', `${runtime.conversation_id}:legacy`, index),
                type: 'source_turn' as const,
                turn_id: turn.id,
            })),
        );
        const importedFingerprint = await fingerprintNativeConversationImport(
            providerJsonValue(historySnapshot),
            options,
            OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
        );
        document = appendConversationRecords(
            document,
            {
                turns: imported.turns,
                assets: imported.assets,
                tool_definitions: toolDefinitions,
                context_entries: importedEntries,
            },
            {
                expected_revision: document.revision,
                operation_id: await entityId('import', runtime.conversation_id, OPENAI_CHAT_COMPLETIONS_PROTOCOL),
                payload_fingerprint: importedFingerprint,
                recorded_at: runtime.recorded_at,
            },
        ).document;
        return nativeConversationImportResult(
            document,
            options,
            OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
        );
    });
}

export async function prepareOpenAIChatCanonicalState(input: {
    conversation: unknown;
    prompt: OpenAIChatCompletionsPrompt;
    options: ExecutionOptions;
    provider: string;
}): Promise<Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>> {
    const runtime = resolveConversationRuntime(input.options);
    let document = parseCanonicalConversation(input.conversation);
    const toolDefinitions = await resolveCanonicalToolDefinitions(document, input.options.tools);
    if (document === undefined) {
        document = newCanonicalConversation(runtime);
        if (input.conversation !== undefined && input.conversation !== null) {
            document = (
                await importOpenAIChatCompletionsHistory(input.conversation, {
                    conversation_id: runtime.conversation_id,
                    recorded_at: runtime.recorded_at,
                    source_request_id: runtime.request_id,
                    provider: input.provider,
                    tool_definitions: toolDefinitions,
                })
            ).document;
        }
    } else if (
        runtime.conversation_id !== document.id &&
        input.options.conversation_runtime?.conversation_id !== undefined
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }

    const target = { provider: input.provider, model: input.options.model };
    const priorCompiled = compileOpenAIChatCompletionsConversation(document, target).conversation;
    const priorNativeMessageCount = priorCompiled.messages.length;
    const promptRecords = await messagesToRecords({
        messages: input.prompt.messages,
        scope: runtime.input_operation_id,
        source: 'received',
        runtime: { ...runtime, conversation_id: document.id },
        provider: input.provider,
        tool_definitions: toolDefinitions,
    });
    const contextEntries = await Promise.all(
        promptRecords.turns.map(async (turn, index) => ({
            id: await entityId('context', runtime.input_operation_id, index),
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
    );
    const appended = await appendCanonicalPrompt(
        document,
        { ...promptRecords, context_entries: contextEntries, item_mappings: promptRecords.mappings },
        { ...runtime, conversation_id: document.id },
        input.options.tools,
        providerJsonValue(input.prompt),
    );
    const acceptedResponse = acceptedCanonicalResponse(appended.document, runtime.response_operation_id);
    const requestDocument =
        acceptedResponse === undefined
            ? appended.document
            : await acceptedCanonicalRequestDocument(appended.document, acceptedResponse);
    const compiled = compileOpenAIChatCompletionsConversation(requestDocument, target);
    const identities =
        acceptedResponse === undefined
            ? await canonicalResponseIdentities(runtime)
            : { generation_id: acceptedResponse.generation.id, response_turn_id: acceptedResponse.turn.id };
    const responseSelectionPolicy = canonicalToolSelectionPolicy(input.options);
    return {
        document: appended.document,
        native_conversation: compiled.conversation,
        runtime: { ...runtime, conversation_id: document.id },
        generation_id: identities.generation_id,
        response_turn_id: identities.response_turn_id,
        tool_definitions: appended.tool_definitions,
        ...(responseSelectionPolicy === undefined ? {} : { response_selection_policy: responseSelectionPolicy }),
        provider: input.provider,
        requested_model: input.options.model,
        prior_native_message_count: priorNativeMessageCount,
        ...(acceptedResponse === undefined ? {} : { accepted_response: acceptedResponse }),
    };
}

export async function finalizeOpenAIChatPreparedRequest(
    state: Omit<PreparedOpenAIChatConversation, 'payload' | 'receipt' | 'diagnostics'>,
    payload: OpenAIChatCompletionsPayload,
    binding: { payload: JsonValue; target_options?: JsonObject } = { payload: providerJsonValue(payload) },
): Promise<PreparedOpenAIChatConversation> {
    const compiled = compileOpenAIChatCompletionsConversation(state.document, {
        provider: state.provider,
        model: state.requested_model,
    });
    const targetOptions = canonicalToolSelectionTargetOptions(binding.target_options, state.response_selection_policy);
    const receipt = await createRequestReceipt(
        state.document,
        state.runtime,
        {
            provider: state.provider,
            protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            model: state.requested_model,
            adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
            ...(targetOptions === undefined ? {} : { options: targetOptions }),
        },
        binding.payload,
        compiled.mappings,
        state.tool_definitions,
    );
    return { ...state, payload, receipt, diagnostics: [] };
}

function safeUsageNumber(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function openAIUsage(response: OpenAIChatCompletionsResponse): GenerationUsage | undefined {
    const native = response.usage;
    if (native === undefined) return undefined;
    const input = safeUsageNumber(native.prompt_tokens);
    const output = safeUsageNumber(native.completion_tokens);
    const reasoningCandidate = safeUsageNumber(native.completion_tokens_details?.reasoning_tokens);
    const reasoning =
        reasoningCandidate !== undefined && (output === undefined || reasoningCandidate <= output)
            ? reasoningCandidate
            : undefined;
    const cachedCandidate = safeUsageNumber(native.prompt_tokens_details?.cached_tokens);
    const cached =
        input !== undefined && cachedCandidate !== undefined && cachedCandidate <= input ? cachedCandidate : undefined;
    const cacheWriteCandidate = safeUsageNumber(native.prompt_tokens_details?.cache_write_tokens);
    const cacheWrite =
        input !== undefined && cacheWriteCandidate !== undefined && cacheWriteCandidate <= input
            ? cacheWriteCandidate
            : undefined;
    const basis = 'openai_chat_tokens';
    const total =
        input !== undefined && output !== undefined && Number.isSafeInteger(input + output)
            ? input + output
            : undefined;
    const cacheTotal = cached === undefined ? undefined : cached + (cacheWrite ?? 0);
    const newInput =
        input !== undefined &&
        cacheTotal !== undefined &&
        Number.isSafeInteger(cacheTotal) &&
        cacheTotal <= input &&
        (cacheWriteCandidate === undefined || cacheWrite !== undefined)
            ? input - cacheTotal
            : undefined;
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(reasoning === undefined ? {} : { reasoning_tokens: reasoning }),
        ...(total === undefined ? {} : { total_tokens: total }),
        ...(cached === undefined ? {} : { cache_read_tokens: cached }),
        ...(cacheWrite === undefined ? {} : { cache_write_tokens: cacheWrite }),
        ...(newInput === undefined || newInput < 0 ? {} : { input_new_tokens: newInput }),
        accounting_provenance: {
            ...(input === undefined ? {} : { input_tokens: { method: 'reported', accounting_basis: basis } }),
            ...(output === undefined ? {} : { output_tokens: { method: 'reported', accounting_basis: basis } }),
            ...(reasoning === undefined ? {} : { reasoning_tokens: { method: 'reported', accounting_basis: basis } }),
            ...(total === undefined ? {} : { total_tokens: { method: 'derived', accounting_basis: basis } }),
            ...(cached === undefined ? {} : { cache_read_tokens: { method: 'reported', accounting_basis: basis } }),
            ...(cacheWrite === undefined
                ? {}
                : { cache_write_tokens: { method: 'reported', accounting_basis: basis } }),
            ...(newInput === undefined || newInput < 0
                ? {}
                : { input_new_tokens: { method: 'derived', accounting_basis: basis } }),
        },
        ...(newInput === undefined || newInput < 0
            ? {}
            : {
                  input_partition: {
                      type: 'complete_disjoint',
                      cache_write_bucket: cacheWriteCandidate === undefined ? 'inapplicable' : 'included',
                  },
              }),
        reported_usage: [
            {
                source: 'provider',
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                accounting_basis: basis,
                payload: providerJsonValue(native),
            },
        ],
    };
}

export async function decodeOpenAIChatCanonicalResponse(
    response: OpenAIChatCompletionsResponse,
    prepared: PreparedOpenAIChatConversation,
    finishReason: string | undefined,
    structuredOutput?: CanonicalStructuredOutput,
): Promise<DecodedConversationResponse> {
    const choice = response.choices[0];
    if (choice?.message === undefined) throw new Error('OpenAI Chat response has no first message');
    const runtime = { ...prepared.runtime, recorded_at: prepared.runtime.completed_at ?? new Date().toISOString() };
    const records = await messageRecords({
        message: {
            role: 'assistant',
            content: choice.message.content,
            reasoning_content: choice.message.reasoning_content,
            reasoning: choice.message.reasoning,
            provider_replay: choice.message.provider_replay,
            tool_calls: choice.message.tool_calls?.flatMap((call) => (call.type === 'function' ? [call] : [])),
        },
        message_index: 0,
        scope: prepared.runtime.response_operation_id,
        source: 'received',
        runtime,
        provider: prepared.provider,
        model: prepared.requested_model,
        tool_definitions: prepared.tool_definitions,
    });
    const received = records.turns[0];
    if (received?.kind !== 'agent') throw new Error('OpenAI Chat response did not decode to an agent turn');
    const turn: ConversationTurn = {
        ...received,
        id: prepared.response_turn_id,
        status: finishReason === 'length' ? 'interrupted' : 'completed',
        timestamps: {
            recorded_at: runtime.recorded_at,
            ...(prepared.runtime.started_at === undefined ? {} : { started_at: prepared.runtime.started_at }),
            completed_at: runtime.recorded_at,
        },
        provenance: { type: 'generated' },
        generation_id: prepared.generation_id,
    };
    const remappedBlocks = turn.blocks.map((block) => {
        if (block.type !== 'native_replay') return block;
        return {
            ...block,
            dependencies: {
                ...block.dependencies,
                turn_ids: [turn.id],
                request_ids: [prepared.receipt.request_id],
            },
        };
    });
    const finalTurn = { ...turn, blocks: remappedBlocks } as ConversationTurn;
    const generation: ExecutedGeneration = {
        ...(await createExecutedGeneration({
            id: prepared.generation_id,
            runtime: prepared.runtime,
            receipt: prepared.receipt,
            provider: prepared.provider,
            protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
            requested_model: prepared.receipt.target.model,
            resolved_model: response.model,
            provider_response_id: response.id,
            finish_reason: finishReason,
            usage: openAIUsage(response),
        })),
        status: finishReason === 'length' ? 'cancelled' : 'completed',
    };
    const decoded: DecodedConversationResponse = {
        turns: [finalTurn],
        generation,
        assets: records.assets.map((asset) => ({
            ...asset,
            provenance: { type: 'generated', generation_id: generation.id, source_turn_id: finalTurn.id },
        })),
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(providerJsonValue(response)),
    };
    if (structuredOutput === undefined) return decoded;
    const rawContent = providerJsonValue(choice.message.content ?? null);
    return normalizeDecodedStructuredOutput(decoded, structuredOutput, async ({ replay_blocks, binding }) => {
        if (replay_blocks.length > 1) {
            throw new TypeError(`OpenAI Chat turn ${finalTurn.id} has multiple replay blocks`);
        }
        const current = replay_blocks[0];
        const base: NativeReplayBlock =
            current ??
            ({
                id: await entityId('replay', prepared.runtime.response_operation_id, 0),
                type: 'native_replay',
                adapter: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                compatibility_scope: {
                    provider: prepared.provider,
                    model: prepared.requested_model,
                    protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                    adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                },
                payload: { type: 'openai_chat_assistant_fields' },
                dependencies: {
                    turn_ids: [finalTurn.id],
                    block_ids: binding.source_block_ids,
                    call_ids: [],
                    request_ids: [prepared.receipt.request_id],
                },
            } satisfies NativeReplayBlock);
        const replay = remapStructuredOutputReplayDependencies(base, binding);
        const payload = replay.payload as OpenAIReplayPayload;
        return [
            {
                ...replay,
                payload: {
                    ...payload,
                    structured_output: {
                        evidence: structuredOutputEvidence(binding),
                        content: rawContent,
                    },
                },
            },
        ];
    });
}

export function appendOpenAIChatCanonicalResponse(
    prepared: PreparedOpenAIChatConversation,
    decoded: DecodedConversationResponse,
): ConversationDocument {
    return appendCanonicalDecodedResponse(prepared, decoded, {
        operation_id: prepared.runtime.response_operation_id,
        recorded_at: decoded.generation.timestamps.recorded_at,
    }).document;
}

/** Read-only compatibility projection for versioned legacy API responses. */
export function exportLegacyOpenAIChatCompletionsConversation(
    document: ConversationDocument,
): OpenAIChatCompletionsPrompt {
    return projectOpenAIChatCompletionsConversation(parseConversationDocument(document), undefined, true).conversation;
}

export function assertCanonicalOpenAIConversation(value: unknown): ConversationDocument {
    return parseConversationDocument(value);
}
