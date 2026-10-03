import { Buffer } from 'node:buffer';
import type { ExecutionOptions } from '@llumiverse/common';
import {
    type AgentContentBlock,
    type Asset,
    appendConversationRecords,
    type ContentBlock,
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
    type NestedToolResultContentBlock,
    type PreparedConversationRequest,
    type ProgramContentBlock,
    parseConversationDocument,
    preflightJsonInput,
    type ResolveConversationAsset,
    readBoundedConversationAsset,
    type ToolDefinition,
    type ToolResultBlock,
    toolArgumentsForModel,
    type UserContentBlock,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionContextOptions,
    type CanonicalStructuredOutput,
    canonicalToolSelectionPolicy,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import {
    acceptedCanonicalRequestDocument,
    acceptedCanonicalResponse,
    appendCanonicalDecodedResponse,
    appendCanonicalDecodedResponseWithProcessing,
    appendCanonicalPrompt,
    assertCanonicalContextProjection,
    assertProtectedReplayCompatibility,
    type CanonicalPreparedState,
    canonicalResponseIdentities,
    canonicalToolSelectionTargetOptions,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    parseCanonicalConversation,
    prepareCanonicalContext,
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

export const OPENAI_RESPONSES_PROTOCOL = 'openai.responses' as const;
export const OPENAI_RESPONSES_ADAPTER_VERSION = '2026-09-12.canonical.1' as const;

export type OpenAIResponsesInputItem = OpenAI.Responses.ResponseInputItem;
export type OpenAIResponsesPayload =
    | OpenAI.Responses.ResponseCreateParamsNonStreaming
    | OpenAI.Responses.ResponseCreateParamsStreaming;

type SourceKind = 'imported' | 'received';
type CanonicalToolResultStatus = 'success' | 'error' | 'cancelled' | 'denied';

export type OpenAIResponsesMediaCaptionProjection = { type: 'provenance' } | { type: 'semantic_text'; text: string };

export interface OpenAIResponsesProjectionOptions {
    project_media_caption?: (
        block: Extract<ContentBlock, { type: 'image' | 'document' | 'audio' | 'video' }>,
        asset: Asset,
        owner: ConversationTurn,
    ) => OpenAIResponsesMediaCaptionProjection | undefined;
}

export type CanonicalOpenAIResponsesFunctionCallOutput = Omit<
    OpenAI.Responses.ResponseInputItem.FunctionCallOutput,
    'call_id'
> & {
    call_id: string;
    /** Internal ingestion evidence. Removed before provider transport. */
    _llumiverse_tool_result_status?: CanonicalToolResultStatus;
};

function requireFunctionCallOutput(
    item: Record<string, unknown>,
    itemIndex: number,
): CanonicalOpenAIResponsesFunctionCallOutput {
    const callId = ownValue(item, 'call_id');
    if (typeof callId !== 'string' || callId.length === 0) {
        throw new Error(`OpenAI Responses function_call_output at items/${itemIndex} has no call_id`);
    }
    return item as unknown as CanonicalOpenAIResponsesFunctionCallOutput;
}

interface ReplayTextEntry extends JsonObject {
    kind: 'text' | 'reasoning' | 'structured_json';
    block_id: string;
    item_index: number;
    content_index: number;
}

interface ReplayToolCallEntry extends JsonObject {
    kind: 'tool_call';
    block_id: string;
    item_index: number;
}

interface ReplayAssetEntry extends JsonObject {
    kind: 'asset';
    block_id: string;
    asset_id: string;
    item_index: number;
    content_index: number;
}

type ReplaySemanticEntry = ReplayTextEntry | ReplayToolCallEntry | ReplayAssetEntry;

type OpenAIResponsesReplayPayload = JsonObject & {
    type: 'openai_responses_items';
    items: JsonValue[];
    semantic_entries: ReplaySemanticEntry[];
    block_offset?: number;
    item_order?: number;
    structured_output?: JsonObject;
};

interface ConvertedRecords {
    turns: ConversationTurn[];
    assets: Asset[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}

function semanticDependencyIds(turns: readonly ConversationTurn[]): {
    turn_ids: string[];
    block_ids: string[];
    call_ids: string[];
} {
    return {
        turn_ids: [...new Set(turns.map((turn) => turn.id))],
        block_ids: [
            ...new Set(
                turns.flatMap((turn) => {
                    const blocks = turn.kind === 'tool' ? turn.blocks[0].content : turn.blocks;
                    return blocks.filter((block) => block.type !== 'native_replay').map((block) => block.id);
                }),
            ),
        ],
        call_ids: [
            ...new Set(
                turns.flatMap((turn) => {
                    if (turn.kind === 'tool') return [turn.blocks[0].call_id];
                    return turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : []));
                }),
            ),
        ],
    };
}

function bindProtectedResponsesExchange(turns: readonly ConversationTurn[]): ConversationTurn[] {
    const bound: ConversationTurn[] = [];
    let activeExchange: ConversationTurn[] = [];
    for (const sourceTurn of turns) {
        if (sourceTurn.kind === 'user') activeExchange = [];
        const dependencies = semanticDependencyIds([...activeExchange, sourceTurn]);
        const rewrite = (block: ContentBlock): ContentBlock =>
            block.type === 'native_replay' &&
            block.protocol === OPENAI_RESPONSES_PROTOCOL &&
            block.dependency_policy !== 'discard_on_dependency_change'
                ? { ...block, dependencies: { ...block.dependencies, ...dependencies } }
                : block;
        const turn: ConversationTurn =
            sourceTurn.kind === 'agent'
                ? { ...sourceTurn, blocks: sourceTurn.blocks.map(rewrite) as AgentContentBlock[] }
                : sourceTurn;
        bound.push(turn);
        activeExchange.push(turn);
    }
    return bound;
}

export interface PreparedOpenAIResponsesConversation
    extends CanonicalPreparedState<OpenAIResponsesInputItem[]>,
        PreparedConversationRequest<OpenAIResponsesPayload> {
    provider: string;
    requested_model: string;
    prior_native_item_count: number;
    /** Ephemeral native projection proof; never replaces the retained canonical source. */
    native_projection?: { source_fingerprint: string; mappings: NativeItemMapping[] };
}

function ownValue(value: object, key: string): unknown {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && 'value' in descriptor ? descriptor.value : undefined;
}

function isResponseHistoryItem(value: unknown): value is OpenAIResponsesInputItem {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const role = ownValue(value, 'role');
    if (role === 'user' || role === 'system' || role === 'developer' || role === 'assistant') {
        return typeof ownValue(value, 'content') === 'string' || Array.isArray(ownValue(value, 'content'));
    }
    return typeof ownValue(value, 'type') === 'string';
}

function wrappedHistoryItems(value: unknown): OpenAIResponsesInputItem[] | undefined {
    if (Array.isArray(value)) return value.every(isResponseHistoryItem) ? value : undefined;
    if (typeof value !== 'object' || value === null) return undefined;
    const items = ownValue(value, '_arrayConversation');
    return Array.isArray(items) && items.every(isResponseHistoryItem) ? items : undefined;
}

export function isOpenAIResponsesHistory(
    value: unknown,
    explicitProtocol?: typeof OPENAI_RESPONSES_PROTOCOL,
): value is OpenAIResponsesInputItem[] | { _arrayConversation: OpenAIResponsesInputItem[] } {
    if (explicitProtocol !== OPENAI_RESPONSES_PROTOCOL || !preflightJsonInput(value).success) return false;
    return wrappedHistoryItems(value) !== undefined;
}

function sourceHistoryTurnNumber(value: unknown): number | undefined {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined;
    const metadata = ownValue(value, '_llumiverse_meta');
    if (typeof metadata !== 'object' || metadata === null || Array.isArray(metadata)) return undefined;
    const turnNumber = ownValue(metadata, 'turnNumber');
    return typeof turnNumber === 'number' && Number.isSafeInteger(turnNumber) && turnNumber >= 0
        ? turnNumber
        : undefined;
}

async function entityId(kind: string, scope: string, ...position: Array<string | number>): Promise<string> {
    return deriveConversationId(kind, scope, ...position.map(String));
}

function importedProvenance(nativePath: string, sourceTurnNumber?: number): ImportedTurnProvenance {
    return {
        type: 'imported',
        source: OPENAI_RESPONSES_PROTOCOL,
        native_id: { protocol: OPENAI_RESPONSES_PROTOCOL, scope: 'history', value: nativePath },
        ...(sourceTurnNumber === undefined ? {} : { source_history_turn_number: sourceTurnNumber }),
        missing_metadata: ['actor_id', 'timestamps', 'exchange'],
    };
}

function turnProvenance(source: SourceKind, nativePath: string, sourceTurnNumber?: number) {
    return source === 'imported' ? importedProvenance(nativePath, sourceTurnNumber) : ({ type: 'received' } as const);
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

function dataUrl(value: string): { mime_type: string; data: string } | undefined {
    const match = /^data:([^;,]+);base64,(.*)$/s.exec(value);
    return match ? { mime_type: match[1], data: match[2] } : undefined;
}

function contentKind(part: Record<string, unknown>): 'image' | 'document' {
    return part.type === 'input_image' ? 'image' : 'document';
}

async function mediaBlock(input: {
    part: Record<string, unknown>;
    turn_id: string;
    scope: string;
    native_path: string;
    source: SourceKind;
    recorded_at: string;
    provider: string;
}): Promise<{ block: UserContentBlock; asset: Asset }> {
    const { part } = input;
    const kind = contentKind(part);
    const blockId = await entityId('block', input.scope, input.native_path);
    const assetId = await entityId('asset', input.scope, input.native_path);
    const imageUrl = typeof part.image_url === 'string' ? part.image_url : undefined;
    const fileUrl = typeof part.file_url === 'string' ? part.file_url : undefined;
    const fileId = typeof part.file_id === 'string' ? part.file_id : undefined;
    const fileData = typeof part.file_data === 'string' ? part.file_data : undefined;
    const inline = dataUrl(imageUrl ?? fileData ?? '');
    let storage: Asset['storage'];
    let mimeType: string;
    if (inline !== undefined) {
        storage = { type: 'inline_base64', data: inline.data };
        mimeType = inline.mime_type;
    } else if (imageUrl !== undefined || fileUrl !== undefined) {
        const url = imageUrl ?? fileUrl;
        if (url === undefined) throw new TypeError('OpenAI Responses media URL is missing');
        storage = { type: 'external', resolver: 'url', locator: { url } };
        mimeType = kind === 'image' ? 'application/octet-stream' : 'application/pdf';
    } else if (fileId !== undefined) {
        storage = { type: 'external', resolver: 'openai_file', locator: { file_id: fileId } };
        mimeType = 'application/octet-stream';
    } else {
        throw new TypeError(`OpenAI Responses ${String(part.type)} content has no supported source`);
    }
    const metadata: JsonObject = {
        source_field:
            imageUrl !== undefined
                ? 'image_url'
                : fileData !== undefined
                  ? 'file_data'
                  : fileUrl !== undefined
                    ? 'file_url'
                    : 'file_id',
        ...(fileId === undefined ? {} : { provider: input.provider }),
        ...(typeof part.detail === 'string' ? { detail: part.detail } : {}),
        ...(typeof part.filename === 'string' ? { filename: part.filename } : {}),
    };
    const integrity = await inlineAssetContentIntegrity(storage);
    const asset: Asset = {
        id: assetId,
        kind,
        mime_type: mimeType,
        storage,
        provenance:
            input.source === 'imported'
                ? { type: 'imported', source: OPENAI_RESPONSES_PROTOCOL }
                : { type: 'received', source_turn_id: input.turn_id },
        ...(integrity ?? {}),
        created_at: input.recorded_at,
        metadata: { openai_responses: metadata },
    };
    return {
        block: { id: blockId, type: kind, asset_id: assetId } as UserContentBlock,
        asset,
    };
}

async function contentBlocks(input: {
    content: unknown;
    turn_id: string;
    scope: string;
    native_path: string;
    source: SourceKind;
    recorded_at: string;
    provider: string;
}): Promise<{ blocks: UserContentBlock[]; assets: Asset[] }> {
    const blocks: UserContentBlock[] = [];
    const assets: Asset[] = [];
    const parts: unknown[] =
        typeof input.content === 'string'
            ? [{ type: 'input_text', text: input.content }]
            : Array.isArray(input.content)
              ? input.content
              : [];
    for (let index = 0; index < parts.length; index += 1) {
        const part = parts[index];
        if (typeof part !== 'object' || part === null || Array.isArray(part)) {
            throw new TypeError(`OpenAI Responses content ${input.native_path}/${index} is not an object`);
        }
        const record = part as Record<string, unknown>;
        if ((record.type === 'input_text' || record.type === 'output_text') && typeof record.text === 'string') {
            blocks.push({
                id: await entityId('block', input.scope, input.native_path, index),
                type: 'text',
                text: record.text,
                format: 'plain',
            });
            continue;
        }
        if (record.type === 'input_image' || record.type === 'input_file') {
            const converted = await mediaBlock({
                part: record,
                turn_id: input.turn_id,
                scope: input.scope,
                native_path: `${input.native_path}/${index}`,
                source: input.source,
                recorded_at: input.recorded_at,
                provider: input.provider,
            });
            blocks.push(converted.block);
            assets.push(converted.asset);
            continue;
        }
        throw new TypeError(`OpenAI Responses content type ${String(record.type)} is not supported`);
    }
    return { blocks, assets };
}

async function ordinaryMessageRecords(input: {
    item: Record<string, unknown>;
    item_index: number;
    scope: string;
    source: SourceKind;
    runtime: NativeConversationImportContext;
    provider: string;
    source_history_turn_number?: number;
}): Promise<ConvertedRecords> {
    const nativePath = `items/${input.item_index}`;
    const turnId = await entityId('turn', input.scope, input.item_index);
    const converted = await contentBlocks({
        content: input.item.content,
        turn_id: turnId,
        scope: input.scope,
        native_path: `${nativePath}/content`,
        source: input.source,
        recorded_at: input.runtime.recorded_at,
        provider: input.provider,
    });
    const role = input.item.role;
    const common = {
        id: turnId,
        status: input.item.status === 'incomplete' ? ('interrupted' as const) : ('completed' as const),
        timestamps: { recorded_at: input.runtime.recorded_at },
        model_visibility: 'include' as const,
        provenance: turnProvenance(input.source, nativePath, input.source_history_turn_number),
    };
    let turn: ConversationTurn;
    if (role === 'system' || role === 'developer') {
        turn = { ...common, kind: 'program', authority: role, blocks: converted.blocks as ProgramContentBlock[] };
    } else {
        turn = { ...common, kind: 'user', authority: 'ordinary', blocks: converted.blocks };
    }
    return {
        turns: [turn],
        assets: converted.assets,
        mappings: [
            { canonical_id: turnId, native_id: nativePath, kind: 'turn' },
            ...converted.blocks.map((block, index) => ({
                canonical_id: block.id,
                native_id: `${nativePath}/content/${index}`,
                kind: 'block' as const,
            })),
        ],
        execution_receipts: [],
    };
}

function findToolDefinition(definitions: readonly ToolDefinition[], name: string): ToolDefinition | undefined {
    return definitions.find((candidate) => candidate.name === name);
}

async function assistantItemsRecords(input: {
    items: readonly Record<string, unknown>[];
    first_item_index: number;
    scope: string;
    source: SourceKind;
    runtime: NativeConversationImportContext;
    provider: string;
    model?: string;
    tool_definitions: readonly ToolDefinition[];
    source_history_turn_number?: number;
}): Promise<ConvertedRecords> {
    const nativePath = `items/${input.first_item_index}`;
    const turnId = await entityId('turn', input.scope, input.first_item_index);
    const blocks: AgentContentBlock[] = [];
    const assets: Asset[] = [];
    const semanticEntries: ReplaySemanticEntry[] = [];
    const itemBlockOffsets: number[] = [];
    const mappings: NativeItemMapping[] = [{ canonical_id: turnId, native_id: nativePath, kind: 'turn' }];
    for (let localIndex = 0; localIndex < input.items.length; localIndex += 1) {
        const item = input.items[localIndex];
        const itemIndex = input.first_item_index + localIndex;
        itemBlockOffsets.push(blocks.length);
        const type = item.type;
        if (type === 'message' || item.role === 'assistant') {
            const content = Array.isArray(item.content)
                ? item.content
                : typeof item.content === 'string'
                  ? [{ type: 'output_text', text: item.content }]
                  : [];
            for (let contentIndex = 0; contentIndex < content.length; contentIndex += 1) {
                const part = content[contentIndex];
                if (typeof part !== 'object' || part === null || Array.isArray(part)) continue;
                const record = part as Record<string, unknown>;
                if (record.type === 'output_text' && typeof record.text === 'string') {
                    const blockId = await entityId('block', input.scope, itemIndex, 'content', contentIndex);
                    blocks.push({ id: blockId, type: 'text', text: record.text, format: 'plain' });
                    semanticEntries.push({
                        kind: 'text',
                        block_id: blockId,
                        item_index: localIndex,
                        content_index: contentIndex,
                    });
                    mappings.push({
                        canonical_id: blockId,
                        native_id: `items/${itemIndex}/content/${contentIndex}`,
                        kind: 'block',
                    });
                }
            }
        } else if (type === 'reasoning') {
            const values: Array<{ text: string; representation: 'summary' | 'text'; content_index: number }> = [];
            if (Array.isArray(item.summary)) {
                for (let index = 0; index < item.summary.length; index += 1) {
                    const summary = item.summary[index];
                    if (typeof summary === 'object' && summary !== null && !Array.isArray(summary)) {
                        const text = ownValue(summary, 'text');
                        if (typeof text === 'string')
                            values.push({ text, representation: 'summary', content_index: index });
                    }
                }
            }
            if (Array.isArray(item.content)) {
                for (let index = 0; index < item.content.length; index += 1) {
                    const content = item.content[index];
                    if (typeof content === 'object' && content !== null && !Array.isArray(content)) {
                        const text = ownValue(content, 'text');
                        if (typeof text === 'string')
                            values.push({ text, representation: 'text', content_index: index });
                    }
                }
            }
            for (let index = 0; index < values.length; index += 1) {
                const value = values[index];
                const blockId = await entityId('reasoning', input.scope, itemIndex, index);
                blocks.push({ id: blockId, type: 'reasoning', text: value.text, representation: value.representation });
                semanticEntries.push({
                    kind: 'reasoning',
                    block_id: blockId,
                    item_index: localIndex,
                    content_index: value.content_index,
                });
                mappings.push({
                    canonical_id: blockId,
                    native_id: `items/${itemIndex}/reasoning/${index}`,
                    kind: 'block',
                });
            }
        } else if (type === 'function_call') {
            const callId = item.call_id;
            const name = item.name;
            const raw = item.arguments;
            if (typeof callId !== 'string' || typeof name !== 'string' || typeof raw !== 'string') {
                throw new TypeError(`OpenAI Responses function call at ${nativePath} is incomplete`);
            }
            const blockId = await entityId('block', input.scope, itemIndex, 'tool_call');
            const definition = findToolDefinition(input.tool_definitions, name);
            blocks.push({
                id: blockId,
                type: 'tool_call',
                call_id: callId,
                tool_name: name,
                ...(definition === undefined ? {} : { definition_id: definition.id }),
                executor: 'application',
                arguments: parseToolArguments(raw),
                native_id: {
                    protocol: OPENAI_RESPONSES_PROTOCOL,
                    scope: input.runtime.request_id,
                    value: typeof item.id === 'string' ? item.id : callId,
                },
            });
            semanticEntries.push({ kind: 'tool_call', block_id: blockId, item_index: localIndex });
            mappings.push(
                { canonical_id: blockId, native_id: `items/${itemIndex}`, kind: 'block' },
                { canonical_id: callId, native_id: callId, kind: 'call' },
            );
        } else if (type === 'image_generation_call' && typeof item.result === 'string') {
            const blockId = await entityId('block', input.scope, itemIndex, 'image');
            const assetId = await entityId('asset', input.scope, itemIndex, 'image');
            const parsed = dataUrl(item.result);
            const storage: Asset['storage'] = {
                type: 'inline_base64',
                data: parsed?.data ?? item.result,
            };
            const integrity = await inlineAssetContentIntegrity(storage);
            const outputFormat = typeof item.output_format === 'string' ? item.output_format : undefined;
            const asset: Asset = {
                id: assetId,
                kind: 'image',
                mime_type: parsed?.mime_type ?? (outputFormat === undefined ? 'image/png' : `image/${outputFormat}`),
                storage,
                provenance:
                    input.source === 'imported'
                        ? { type: 'imported', source: OPENAI_RESPONSES_PROTOCOL }
                        : { type: 'received', source_turn_id: turnId },
                ...(integrity ?? {}),
                created_at: input.runtime.recorded_at,
            };
            blocks.push({ id: blockId, type: 'image', asset_id: assetId });
            assets.push(asset);
            semanticEntries.push({
                kind: 'asset',
                block_id: blockId,
                asset_id: assetId,
                item_index: localIndex,
                content_index: 0,
            });
            mappings.push({ canonical_id: blockId, native_id: `items/${itemIndex}`, kind: 'block' });
        }
    }
    const semanticBlocks = [...blocks];
    const protectedBatch = input.items.some((item) => {
        const type = item.type;
        return (
            type === 'reasoning' || (type !== 'message' && type !== 'function_call' && type !== 'image_generation_call')
        );
    });
    const batchCallIds = semanticBlocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : []));
    for (let localIndex = 0; localIndex < input.items.length; localIndex += 1) {
        const item = input.items[localIndex];
        const itemIndex = input.first_item_index + localIndex;
        const itemEntries = semanticEntries
            .filter((entry) => entry.item_index === localIndex)
            .map((entry) => ({ ...entry, item_index: 0 }));
        const itemBlockIds = itemEntries.map((entry) => entry.block_id);
        const itemBlocks = semanticBlocks.filter((block) => itemBlockIds.includes(block.id));
        const protectedReplay =
            item.type === 'reasoning' ||
            (item.type !== 'message' && item.type !== 'function_call' && item.type !== 'image_generation_call');
        const replayId = await entityId('replay', input.scope, itemIndex);
        const replay: AgentContentBlock = {
            id: replayId,
            type: 'native_replay',
            adapter: OPENAI_RESPONSES_ADAPTER_VERSION,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            compatibility_scope: {
                provider: input.provider,
                protocol: OPENAI_RESPONSES_PROTOCOL,
                ...(protectedReplay && input.model !== undefined ? { model: input.model } : {}),
                adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
            },
            payload: {
                type: 'openai_responses_items',
                items: [providerJsonValue(item)],
                semantic_entries: itemEntries,
                block_offset: itemBlockOffsets[localIndex],
                item_order: localIndex,
            },
            dependencies: {
                turn_ids: protectedReplay && protectedBatch ? [turnId] : [],
                block_ids: protectedReplay && protectedBatch ? semanticBlocks.map((block) => block.id) : itemBlockIds,
                call_ids:
                    protectedReplay && protectedBatch
                        ? batchCallIds
                        : itemBlocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : [])),
                request_ids: [],
            },
            ...(protectedReplay ? {} : { dependency_policy: 'discard_on_dependency_change' as const }),
        };
        blocks.push(replay);
        mappings.push({ canonical_id: replayId, native_id: `${nativePath}/replay/${localIndex}`, kind: 'block' });
    }
    const common = {
        id: turnId,
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks,
        status: input.items.some((item) => item.status === 'incomplete')
            ? ('interrupted' as const)
            : ('completed' as const),
        timestamps: { recorded_at: input.runtime.recorded_at },
        model_visibility: 'include' as const,
    };
    const turn: ConversationTurn =
        input.source === 'imported'
            ? {
                  ...common,
                  provenance: importedProvenance(nativePath, input.source_history_turn_number),
              }
            : { ...common, provenance: { type: 'received' } };
    return { turns: [turn], assets, mappings, execution_receipts: [] };
}

async function toolResultRecords(input: {
    item: CanonicalOpenAIResponsesFunctionCallOutput;
    item_index: number;
    scope: string;
    source: SourceKind;
    runtime: NativeConversationImportContext;
    provider: string;
    source_history_turn_number?: number;
}): Promise<ConvertedRecords> {
    const nativePath = `items/${input.item_index}`;
    const turnId = await entityId('turn', input.scope, input.item_index);
    const converted = await contentBlocks({
        content: input.item.output,
        turn_id: turnId,
        scope: input.scope,
        native_path: `${nativePath}/output`,
        source: input.source,
        recorded_at: input.runtime.recorded_at,
        provider: input.provider,
    });
    const resultId = await entityId('block', input.scope, input.item_index, 'tool_result');
    const replayId = await entityId('replay', input.scope, input.item_index);
    const nativeItem = providerJsonValue(input.item) as JsonObject;
    delete nativeItem._llumiverse_tool_result_status;
    const replay: NestedToolResultContentBlock = {
        id: replayId,
        type: 'native_replay',
        adapter: OPENAI_RESPONSES_ADAPTER_VERSION,
        protocol: OPENAI_RESPONSES_PROTOCOL,
        compatibility_scope: {
            provider: input.provider,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
        },
        payload: {
            type: 'openai_responses_items',
            items: [nativeItem],
            semantic_entries: converted.blocks.map((block, index) => ({
                kind: block.type === 'text' ? 'text' : 'asset',
                block_id: block.id,
                ...(block.type === 'image' || block.type === 'document' ? { asset_id: block.asset_id } : {}),
                item_index: 0,
                content_index: index,
            })) as ReplaySemanticEntry[],
        },
        dependencies: {
            turn_ids: [turnId],
            block_ids: converted.blocks.map((block) => block.id),
            call_ids: [input.item.call_id],
            request_ids: [],
        },
        dependency_policy: 'discard_on_dependency_change',
    };
    const status = input.item._llumiverse_tool_result_status ?? 'unknown';
    const resultBlock: ToolResultBlock = {
        id: resultId,
        type: 'tool_result',
        call_id: input.item.call_id,
        status,
        content: [...(converted.blocks as NestedToolResultContentBlock[]), replay],
        native_id: {
            protocol: OPENAI_RESPONSES_PROTOCOL,
            scope: input.runtime.request_id,
            value: typeof input.item.id === 'string' ? input.item.id : input.item.call_id,
        },
    };
    const executionReceipts: ExecutionReceipt[] = [];
    if (status !== 'unknown') {
        executionReceipts.push({
            id: await entityId('execution_receipt', input.scope, input.item.call_id),
            call_id: input.item.call_id,
            executor: 'application',
            status,
            result_turn_id: turnId,
            result_fingerprint: await fingerprintJson(resultBlock),
            recorded_at: input.runtime.recorded_at,
        });
    }
    const turn: ConversationTurn = {
        id: turnId,
        kind: 'tool',
        authority: 'ordinary',
        blocks: [resultBlock],
        status: 'completed',
        timestamps: { recorded_at: input.runtime.recorded_at },
        model_visibility: 'include',
        provenance: turnProvenance(input.source, nativePath, input.source_history_turn_number),
    };
    return {
        turns: [turn],
        assets: converted.assets,
        mappings: [
            { canonical_id: turnId, native_id: nativePath, kind: 'turn' },
            { canonical_id: resultId, native_id: nativePath, kind: 'block' },
            { canonical_id: input.item.call_id, native_id: input.item.call_id, kind: 'call' },
        ],
        execution_receipts: executionReceipts,
    };
}

function isOrdinaryMessage(item: Record<string, unknown>): boolean {
    return item.role === 'user' || item.role === 'system' || item.role === 'developer';
}

async function itemsToRecords(input: {
    items: readonly OpenAIResponsesInputItem[];
    scope: string;
    source: SourceKind;
    runtime: NativeConversationImportContext;
    provider: string;
    model?: string;
    tool_definitions: readonly ToolDefinition[];
    source_history_turn_number?: number;
}): Promise<ConvertedRecords> {
    const result: ConvertedRecords = { turns: [], assets: [], mappings: [], execution_receipts: [] };
    const append = (records: ConvertedRecords) => {
        result.turns.push(...records.turns);
        result.assets.push(...records.assets);
        result.mappings.push(...records.mappings);
        result.execution_receipts.push(...records.execution_receipts);
    };
    let assistantItems: Record<string, unknown>[] = [];
    let firstAssistantIndex = 0;
    const flushAssistant = async () => {
        if (assistantItems.length === 0) return;
        append(
            await assistantItemsRecords({
                items: assistantItems,
                first_item_index: firstAssistantIndex,
                scope: input.scope,
                source: input.source,
                runtime: input.runtime,
                provider: input.provider,
                model: input.model,
                tool_definitions: input.tool_definitions,
                ...(input.source_history_turn_number === undefined
                    ? {}
                    : { source_history_turn_number: input.source_history_turn_number }),
            }),
        );
        assistantItems = [];
    };
    for (let index = 0; index < input.items.length; index += 1) {
        const item = input.items[index] as unknown as Record<string, unknown>;
        if (item.type === 'function_call_output') {
            await flushAssistant();
            append(
                await toolResultRecords({
                    item: requireFunctionCallOutput(item, index),
                    item_index: index,
                    scope: input.scope,
                    source: input.source,
                    runtime: input.runtime,
                    provider: input.provider,
                    ...(input.source_history_turn_number === undefined
                        ? {}
                        : { source_history_turn_number: input.source_history_turn_number }),
                }),
            );
        } else if (isOrdinaryMessage(item)) {
            await flushAssistant();
            append(
                await ordinaryMessageRecords({
                    item,
                    item_index: index,
                    scope: input.scope,
                    source: input.source,
                    runtime: input.runtime,
                    provider: input.provider,
                    ...(input.source_history_turn_number === undefined
                        ? {}
                        : { source_history_turn_number: input.source_history_turn_number }),
                }),
            );
        } else {
            if (assistantItems.length === 0) firstAssistantIndex = index;
            assistantItems.push(item);
        }
    }
    await flushAssistant();
    result.turns = bindProtectedResponsesExchange(result.turns);
    return result;
}

function openAIResponsesMetadata(asset: Asset): JsonObject {
    const value = asset.metadata?.openai_responses;
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? value : {};
}

function assetToInputPart(asset: Asset, target?: { provider?: string }): OpenAI.Responses.ResponseInputContent {
    const metadata = openAIResponsesMetadata(asset);
    const detail = metadata.detail;
    const owningProvider = metadata.provider;
    if (
        asset.storage.type === 'external' &&
        asset.storage.resolver === 'openai_file' &&
        target?.provider !== undefined &&
        owningProvider !== target.provider
    ) {
        throw new TypeError(`OpenAI Responses file asset ${asset.id} belongs to provider ${String(owningProvider)}`);
    }
    if (asset.kind === 'image') {
        const value =
            asset.storage.type === 'inline_base64'
                ? `data:${asset.mime_type};base64,${asset.storage.data}`
                : asset.storage.type === 'external' && asset.storage.resolver === 'url'
                  ? asset.storage.locator.url
                  : undefined;
        const fileId =
            asset.storage.type === 'external' && asset.storage.resolver === 'openai_file'
                ? asset.storage.locator.file_id
                : undefined;
        if (typeof value !== 'string' && typeof fileId !== 'string') {
            throw new TypeError(`OpenAI Responses cannot resolve image asset ${asset.id}`);
        }
        return {
            type: 'input_image',
            detail: detail === 'low' || detail === 'high' || detail === 'original' ? detail : 'auto',
            ...(typeof value === 'string' ? { image_url: value } : { file_id: fileId as string }),
        };
    }
    if (asset.kind !== 'document') {
        throw new TypeError(`OpenAI Responses cannot project ${asset.kind} asset ${asset.id}`);
    }
    const filename = typeof metadata.filename === 'string' ? metadata.filename : undefined;
    if (asset.storage.type === 'inline_base64') {
        return {
            type: 'input_file',
            file_data: `data:${asset.mime_type};base64,${asset.storage.data}`,
            ...(detail === 'low' || detail === 'high' || detail === 'auto' ? { detail } : {}),
            ...(filename === undefined ? {} : { filename }),
        };
    }
    if (asset.storage.type === 'external' && asset.storage.resolver === 'url') {
        const url = asset.storage.locator.url;
        if (typeof url !== 'string') throw new TypeError(`OpenAI Responses document asset ${asset.id} has no URL`);
        return {
            type: 'input_file',
            file_url: url,
            ...(detail === 'low' || detail === 'high' || detail === 'auto' ? { detail } : {}),
            ...(filename === undefined ? {} : { filename }),
        };
    }
    if (asset.storage.type === 'external' && asset.storage.resolver === 'openai_file') {
        const fileId = asset.storage.locator.file_id;
        if (typeof fileId !== 'string')
            throw new TypeError(`OpenAI Responses document asset ${asset.id} has no file ID`);
        return {
            type: 'input_file',
            file_id: fileId,
            ...(detail === 'low' || detail === 'high' || detail === 'auto' ? { detail } : {}),
            ...(filename === undefined ? {} : { filename }),
        };
    }
    throw new TypeError(`OpenAI Responses cannot resolve document asset ${asset.id}`);
}

/** Project an image asset with the same ownership and storage checks as retained Responses context. */
export function openAIResponsesImageInput(
    asset: Asset,
    target?: { provider?: string },
): OpenAI.Responses.ResponseInputImage {
    const part = assetToInputPart(asset, target);
    if (part.type !== 'input_image') {
        throw new TypeError(`OpenAI Responses asset ${asset.id} is not an image`);
    }
    return part;
}

function ordinaryBlockToParts(
    block: ContentBlock,
    document: ConversationDocument,
    target?: { provider?: string },
    projection?: OpenAIResponsesProjectionOptions,
    owner?: ConversationTurn,
): OpenAI.Responses.ResponseInputContent[] {
    if (block.type === 'text') return [{ type: 'input_text', text: block.text }];
    if (block.type === 'json') return [{ type: 'input_text', text: JSON.stringify(block.value) }];
    if (block.type === 'image' || block.type === 'document') {
        const asset = document.assets[block.asset_id];
        if (asset === undefined) throw new Error(`OpenAI Responses content references missing asset ${block.asset_id}`);
        const media = assetToInputPart(asset, target);
        if (block.caption === undefined) return [media];
        const caption = owner === undefined ? undefined : projection?.project_media_caption?.(block, asset, owner);
        if (caption === undefined) {
            throw new TypeError(`OpenAI Responses cannot preserve ${block.type} block ${block.id} caption`);
        }
        return caption.type === 'provenance' ? [media] : [{ type: 'input_text', text: caption.text }, media];
    }
    throw new TypeError(`OpenAI Responses cannot project canonical ${block.type} block ${block.id}`);
}

function rawReplayPayload(block: Extract<ContentBlock, { type: 'native_replay' }>): OpenAIResponsesReplayPayload {
    const payload = block.payload;
    if (typeof payload !== 'object' || payload === null || Array.isArray(payload)) {
        throw new TypeError(`OpenAI Responses replay block ${block.id} has an unsupported payload`);
    }
    if (
        payload.type !== 'openai_responses_items' ||
        !Array.isArray(payload.items) ||
        !Array.isArray(payload.semantic_entries)
    ) {
        throw new TypeError(`OpenAI Responses replay block ${block.id} has an unsupported payload`);
    }
    return payload as unknown as OpenAIResponsesReplayPayload;
}

function assertReplayScope(
    block: Extract<ContentBlock, { type: 'native_replay' }>,
    target?: { provider?: string; model?: string },
): void {
    if (
        block.adapter !== OPENAI_RESPONSES_ADAPTER_VERSION ||
        block.protocol !== OPENAI_RESPONSES_PROTOCOL ||
        block.compatibility_scope.protocol !== OPENAI_RESPONSES_PROTOCOL ||
        block.compatibility_scope.adapter_version !== OPENAI_RESPONSES_ADAPTER_VERSION ||
        (target?.provider !== undefined && block.compatibility_scope.provider !== target.provider) ||
        (target?.model !== undefined &&
            block.compatibility_scope.model !== undefined &&
            block.compatibility_scope.model !== target.model)
    ) {
        throw new TypeError(`OpenAI Responses replay block ${block.id} is outside its compatibility scope`);
    }
}

function rawAt(payload: OpenAIResponsesReplayPayload, entry: ReplaySemanticEntry): Record<string, unknown> {
    const item = payload.items[entry.item_index];
    if (typeof item !== 'object' || item === null || Array.isArray(item)) {
        throw new TypeError('OpenAI Responses replay semantic entry references a non-object item');
    }
    return item as Record<string, unknown>;
}

function replayText(raw: Record<string, unknown>, entry: ReplayTextEntry, semantic?: ContentBlock): unknown {
    if (raw.type === 'function_call_output') {
        const part = Array.isArray(raw.output) ? raw.output[entry.content_index] : undefined;
        return typeof part === 'object' && part !== null && !Array.isArray(part)
            ? ownValue(part, 'text')
            : entry.content_index === 0 && typeof raw.output === 'string'
              ? raw.output
              : undefined;
    }
    if (entry.kind === 'reasoning') {
        const collection =
            semantic?.type === 'reasoning' && semantic.representation === 'summary' ? raw.summary : raw.content;
        const part = Array.isArray(collection) ? collection[entry.content_index] : undefined;
        return typeof part === 'object' && part !== null && !Array.isArray(part) ? ownValue(part, 'text') : undefined;
    }
    if (typeof raw.content === 'string' && entry.content_index === 0) return raw.content;
    const part = Array.isArray(raw.content) ? raw.content[entry.content_index] : undefined;
    return typeof part === 'object' && part !== null && !Array.isArray(part) ? ownValue(part, 'text') : undefined;
}

function stableJson(value: unknown): string {
    if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
    if (typeof value === 'object' && value !== null) {
        const record = value as Record<string, unknown>;
        return `{${Object.keys(record)
            .sort()
            .map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`)
            .join(',')}}`;
    }
    return JSON.stringify(value) ?? 'undefined';
}

function assertReplaySemantics(
    turn: ConversationTurn,
    document: ConversationDocument,
    block: Extract<ContentBlock, { type: 'native_replay' }>,
    payload: OpenAIResponsesReplayPayload,
    target?: { provider?: string },
): void {
    if (turn.kind === 'tool') {
        const raw = payload.items[0];
        const rawCallId =
            typeof raw === 'object' && raw !== null && !Array.isArray(raw) ? ownValue(raw, 'call_id') : undefined;
        if (rawCallId !== turn.blocks[0].call_id || block.dependencies.call_ids[0] !== turn.blocks[0].call_id) {
            throw new TypeError(`OpenAI Responses replay block ${block.id} no longer matches its tool result call`);
        }
    }
    const blocks = new Map<string, ContentBlock>();
    for (const candidate of turn.blocks) {
        if (candidate.type !== 'native_replay' && candidate.type !== 'tool_result') {
            blocks.set(candidate.id, candidate);
        }
        if (candidate.type === 'tool_result') {
            for (const nested of candidate.content) if (nested.type !== 'native_replay') blocks.set(nested.id, nested);
        }
    }
    const dependencyIds = [...block.dependencies.block_ids].sort();
    const semanticIds = [...new Set(payload.semantic_entries.map((entry) => entry.block_id))].sort();
    const documentBlockIds = new Set(
        document.turns.flatMap((candidateTurn) => {
            const candidateBlocks =
                candidateTurn.kind === 'tool' ? candidateTurn.blocks[0].content : candidateTurn.blocks;
            return candidateBlocks.map((candidate) => candidate.id);
        }),
    );
    if (
        semanticIds.some((blockId) => !block.dependencies.block_ids.includes(blockId)) ||
        (block.dependency_policy === 'discard_on_dependency_change' &&
            JSON.stringify(dependencyIds) !== JSON.stringify(semanticIds)) ||
        semanticIds.some((blockId) => !blocks.has(blockId)) ||
        dependencyIds.some((blockId) => !documentBlockIds.has(blockId))
    ) {
        throw new TypeError(`OpenAI Responses replay block ${block.id} no longer matches its semantic blocks`);
    }
    for (const entry of payload.semantic_entries) {
        const semantic = blocks.get(entry.block_id);
        if (semantic === undefined)
            throw new TypeError(`OpenAI Responses replay references missing block ${entry.block_id}`);
        const raw = rawAt(payload, entry);
        if (entry.kind === 'tool_call') {
            if (
                semantic.type !== 'tool_call' ||
                raw.call_id !== semantic.call_id ||
                raw.name !== semantic.tool_name ||
                typeof raw.arguments !== 'string'
            ) {
                throw new TypeError(
                    `OpenAI Responses replay tool call ${entry.block_id} no longer matches canonical data`,
                );
            }
            let parsedArguments: unknown;
            try {
                parsedArguments = JSON.parse(raw.arguments);
            } catch {
                parsedArguments = undefined;
            }
            const argumentsMatch =
                semantic.arguments.type === 'invalid'
                    ? semantic.arguments.raw === raw.arguments
                    : stableJson(toolArgumentsForModel(semantic.arguments)) === stableJson(parsedArguments);
            if (!argumentsMatch) {
                throw new TypeError(`OpenAI Responses replay tool call ${entry.block_id} has changed arguments`);
            }
        } else if (entry.kind === 'asset') {
            if ((semantic.type !== 'image' && semantic.type !== 'document') || semantic.asset_id !== entry.asset_id) {
                throw new TypeError(`OpenAI Responses replay asset ${entry.block_id} no longer matches canonical data`);
            }
            const asset = document.assets[entry.asset_id];
            if (asset === undefined)
                throw new Error(`OpenAI Responses replay references missing asset ${entry.asset_id}`);
            if (raw.type === 'image_generation_call') {
                const expected = asset.storage.type === 'inline_base64' ? asset.storage.data : undefined;
                const parsed = typeof raw.result === 'string' ? dataUrl(raw.result) : undefined;
                const actual = typeof raw.result === 'string' ? (parsed?.data ?? raw.result) : undefined;
                const outputFormat = typeof raw.output_format === 'string' ? raw.output_format : undefined;
                const actualMimeType =
                    parsed?.mime_type ?? (outputFormat === undefined ? 'image/png' : `image/${outputFormat}`);
                if (expected === undefined || actual !== expected || asset.mime_type !== actualMimeType) {
                    throw new TypeError(
                        `OpenAI Responses replay image ${entry.asset_id} no longer matches canonical data`,
                    );
                }
            } else if (raw.type === 'function_call_output') {
                const part = Array.isArray(raw.output) ? raw.output[entry.content_index] : undefined;
                if (stableJson(providerJsonValue(assetToInputPart(asset, target))) !== stableJson(part)) {
                    throw new TypeError(
                        `OpenAI Responses replay asset ${entry.asset_id} no longer matches canonical data`,
                    );
                }
            }
        } else if (entry.kind === 'structured_json') {
            if (semantic.type !== 'json' || typeof replayText(raw, entry) !== 'string') {
                throw new TypeError(
                    `OpenAI Responses structured replay ${entry.block_id} has incompatible canonical data`,
                );
            }
        } else {
            let semanticText: string;
            if (entry.kind === 'reasoning') {
                if (semantic.type !== 'reasoning') {
                    throw new TypeError(
                        `OpenAI Responses replay text ${entry.block_id} has incompatible canonical data`,
                    );
                }
                semanticText = semantic.text;
            } else {
                if (semantic.type !== 'text') {
                    throw new TypeError(
                        `OpenAI Responses replay text ${entry.block_id} has incompatible canonical data`,
                    );
                }
                semanticText = semantic.text;
            }
            const text = replayText(raw, entry, semantic);
            if (text !== semanticText) {
                throw new TypeError(`OpenAI Responses replay text ${entry.block_id} no longer matches canonical data`);
            }
        }
    }
}

function canonicalToolCallItem(block: Extract<ContentBlock, { type: 'tool_call' }>): OpenAIResponsesInputItem {
    if (block.executor !== 'application') {
        throw new TypeError(`OpenAI Responses requires protected replay for provider tool call ${block.call_id}`);
    }
    if (block.arguments.type === 'invalid') {
        throw new TypeError(`OpenAI Responses cannot project invalid tool arguments for call ${block.call_id}`);
    }
    return {
        type: 'function_call',
        id: block.native_id?.protocol === OPENAI_RESPONSES_PROTOCOL ? block.native_id.value : block.id,
        call_id: block.call_id,
        name: block.tool_name,
        arguments: JSON.stringify(toolArgumentsForModel(block.arguments)),
        status: 'completed',
    };
}

function replayItems(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
    projection?: OpenAIResponsesProjectionOptions,
): OpenAIResponsesInputItem[] | undefined {
    const replayBlocks: Array<Extract<ContentBlock, { type: 'native_replay' }>> = [];
    for (const block of turn.blocks) {
        if (block.type === 'native_replay') replayBlocks.push(block);
        if (block.type === 'tool_result') {
            for (const nested of block.content) if (nested.type === 'native_replay') replayBlocks.push(nested);
        }
    }
    const foreign = replayBlocks.find(
        (block) =>
            block.protocol !== OPENAI_RESPONSES_PROTOCOL && block.dependency_policy !== 'discard_on_dependency_change',
    );
    if (foreign !== undefined) {
        throw new TypeError(`OpenAI Responses cannot discard protected ${foreign.protocol} replay block ${foreign.id}`);
    }
    const matching = replayBlocks.filter((block) => block.protocol === OPENAI_RESPONSES_PROTOCOL);
    if (matching.length === 0) return undefined;
    const payloads = matching.map((replay) => {
        assertReplayScope(replay, target);
        const payload = rawReplayPayload(replay);
        assertReplaySemantics(turn, document, replay, payload, target);
        return { replay, payload };
    });
    const structuredPayloads = payloads
        .filter(({ payload }) => payload.structured_output !== undefined)
        .sort((left, right) => (left.payload.item_order ?? 0) - (right.payload.item_order ?? 0));
    if (structuredPayloads.length > 0) {
        const evidence = parseStructuredOutputEvidence(structuredPayloads[0].payload.structured_output);
        const evidenceFingerprint = stableJson(evidence);
        const sourceTexts = structuredPayloads.flatMap(({ replay, payload }) => {
            if (stableJson(parseStructuredOutputEvidence(payload.structured_output)) !== evidenceFingerprint) {
                throw new TypeError(`OpenAI Responses structured replay ${replay.id} has inconsistent evidence`);
            }
            return payload.semantic_entries.flatMap((entry) => {
                if (entry.kind !== 'structured_json') return [];
                const text = replayText(rawAt(payload, entry), entry);
                return typeof text === 'string' ? [text] : [];
            });
        });
        assertStructuredOutputEvidence(
            turn,
            evidence,
            sourceTexts,
            structuredPayloads.map(({ replay }) => replay.id).join(','),
        );
    }
    if (payloads.length === 1 && payloads[0].payload.block_offset === undefined) {
        return providerJsonValue(payloads[0].payload.items) as unknown as OpenAIResponsesInputItem[];
    }
    const units: Array<{ block_offset: number; item_order: number; raw: boolean; items: OpenAIResponsesInputItem[] }> =
        [];
    const coveredBlockIds = new Set<string>();
    for (const { replay, payload } of payloads) {
        if (
            typeof payload.block_offset !== 'number' ||
            !Number.isSafeInteger(payload.block_offset) ||
            payload.block_offset < 0 ||
            typeof payload.item_order !== 'number' ||
            !Number.isSafeInteger(payload.item_order) ||
            payload.item_order < 0
        ) {
            throw new TypeError(`OpenAI Responses replay block ${replay.id} has invalid item ordering`);
        }
        for (const blockId of replay.dependencies.block_ids) coveredBlockIds.add(blockId);
        units.push({
            block_offset: payload.block_offset,
            item_order: payload.item_order,
            raw: true,
            items: providerJsonValue(payload.items) as unknown as OpenAIResponsesInputItem[],
        });
    }
    const semanticBlocks = turn.kind === 'tool' ? turn.blocks[0].content : turn.blocks;
    for (let blockIndex = 0; blockIndex < semanticBlocks.length; blockIndex += 1) {
        const block = semanticBlocks[blockIndex];
        if (block.type === 'native_replay' || coveredBlockIds.has(block.id)) continue;
        if (block.type === 'extension' && block.model_projection === 'excluded') continue;
        if (turn.kind === 'tool') return undefined;
        if (block.type === 'tool_call') {
            units.push({
                block_offset: blockIndex,
                item_order: Number.MAX_SAFE_INTEGER,
                raw: false,
                items: [canonicalToolCallItem(block)],
            });
            continue;
        }
        if (block.type === 'reasoning') {
            throw new TypeError(`OpenAI Responses requires protected replay for reasoning block ${block.id}`);
        }
        const parts = ordinaryBlockToParts(block, document, target, projection, turn);
        units.push({
            block_offset: blockIndex,
            item_order: Number.MAX_SAFE_INTEGER,
            raw: false,
            items: [{ role: 'assistant', content: parts } as OpenAIResponsesInputItem],
        });
    }
    units.sort(
        (left, right) =>
            left.block_offset - right.block_offset ||
            Number(right.raw) - Number(left.raw) ||
            left.item_order - right.item_order,
    );
    return units.flatMap((unit) => unit.items);
}

function compileOrdinaryTurn(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string },
    projection?: OpenAIResponsesProjectionOptions,
): OpenAIResponsesInputItem[] {
    if (turn.kind === 'tool') {
        const result = turn.blocks[0];
        const parts = result.content.flatMap((block): OpenAI.Responses.ResponseInputContent[] => {
            if (block.type === 'native_replay') return [];
            if (block.type === 'extension' && block.model_projection === 'excluded') return [];
            return ordinaryBlockToParts(block, document, target, projection, turn);
        });
        const output: string | OpenAI.Responses.ResponseInputContent[] =
            parts.length === 1 && parts[0].type === 'input_text' ? parts[0].text : parts;
        return [{ type: 'function_call_output', call_id: result.call_id, output }];
    }
    const parts = turn.blocks.flatMap((block): OpenAI.Responses.ResponseInputContent[] => {
        if (block.type === 'extension' && block.model_projection === 'excluded') return [];
        if (block.type === 'native_replay') return [];
        if (block.type === 'tool_call') return [];
        if (block.type === 'reasoning') {
            throw new TypeError(
                `OpenAI Responses requires protected replay for canonical ${block.type} block ${block.id}`,
            );
        }
        return ordinaryBlockToParts(block, document, target, projection, turn);
    });
    const content: string | OpenAI.Responses.ResponseInputContent[] =
        parts.length === 1 && parts[0].type === 'input_text' ? parts[0].text : parts;
    const role =
        turn.kind === 'program' && turn.authority === 'system'
            ? 'system'
            : turn.kind === 'program' && turn.authority === 'developer'
              ? 'developer'
              : turn.kind === 'agent'
                ? 'assistant'
                : 'user';
    const messages: OpenAIResponsesInputItem[] =
        parts.length === 0 && turn.kind === 'agent' ? [] : [{ role, content } as OpenAIResponsesInputItem];
    if (turn.kind !== 'agent') return messages;
    const calls = turn.blocks.flatMap((block): OpenAIResponsesInputItem[] =>
        block.type === 'tool_call' ? [canonicalToolCallItem(block)] : [],
    );
    return [...messages, ...calls];
}

export function compileOpenAIResponsesConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
    projection?: OpenAIResponsesProjectionOptions,
): ReturnType<typeof projectOpenAIResponsesConversation> {
    return projectOpenAIResponsesConversation(document, target, false, projection);
}

/** Resolve selected, integrity-declared external images that Responses cannot represent natively. */
async function compileOpenAIResponsesContextWithHostAssets(
    document: ConversationDocument,
    target: { provider: string; model: string },
    resolveAsset: ResolveConversationAsset | undefined,
    signal: AbortSignal | undefined,
    hydrated: Map<string, { fingerprint: string; data: string }>,
): Promise<ReturnType<typeof compileOpenAIResponsesConversation>> {
    const selected = selectedCanonicalTurns(document, {
        allow_interrupted_with_replay_protocol: OPENAI_RESPONSES_PROTOCOL,
        allow_interrupted_with_complete_tool_calls: true,
    });
    const imageAssetIds = new Set<string>();
    for (const turn of selected) {
        for (const block of turn.blocks) {
            if (block.type === 'image') imageAssetIds.add(block.asset_id);
        }
    }
    const externalImageIds = [...imageAssetIds].filter((id) => {
        const asset = document.assets[id];
        if (!asset) throw new TypeError(`OpenAI Responses selected image asset ${id} is missing`);
        return (
            asset.storage.type === 'external' &&
            asset.storage.resolver !== 'url' &&
            asset.storage.resolver !== 'openai_file'
        );
    });
    if (externalImageIds.length === 0) return compileOpenAIResponsesConversation(document, target);
    const nativeDocument = { ...document, assets: { ...document.assets } };
    let aggregateBytes = 0;
    for (const id of externalImageIds) {
        const asset = document.assets[id];
        if (!asset) throw new TypeError(`OpenAI Responses selected image asset ${id} is missing`);
        if (asset.kind !== 'image' || !resolveAsset)
            throw new TypeError(`OpenAI Responses external image asset ${id} has no host resolver`);
        if (asset.mime_type !== 'image/png' && asset.mime_type !== 'image/jpeg')
            throw new TypeError(`OpenAI Responses external image asset ${id} has unsupported MIME`);
        if (asset.byte_length === undefined || asset.byte_length > 32 * 1024 * 1024 - aggregateBytes)
            throw new RangeError('OpenAI Responses external images exceed the aggregate native projection budget');
        const fingerprint = await fingerprintJson(asset);
        signal?.throwIfAborted();
        let data = hydrated.get(id)?.data;
        if (data !== undefined && hydrated.get(id)?.fingerprint !== fingerprint)
            throw new TypeError(`OpenAI Responses external image asset ${id} changed during native preparation`);
        if (data === undefined) {
            const bytes = await readBoundedConversationAsset(asset, resolveAsset, {
                max_bytes: 32 * 1024 * 1024,
                max_chunks: 65_536,
                signal,
                require_integrity: true,
                label: 'OpenAI Responses external image',
            });
            signal?.throwIfAborted();
            const png = Buffer.from(bytes.subarray(0, 8)).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
            const jpeg = bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
            if ((!png || asset.mime_type !== 'image/png') && (!jpeg || asset.mime_type !== 'image/jpeg'))
                throw new TypeError(`OpenAI Responses external image asset ${id} has invalid media bytes`);
            data = Buffer.from(bytes).toString('base64');
            hydrated.set(id, { fingerprint, data });
        }
        aggregateBytes += asset.byte_length;
        nativeDocument.assets[id] = { ...asset, storage: { type: 'inline_base64', data } };
    }
    signal?.throwIfAborted();
    return compileOpenAIResponsesConversation(nativeDocument, target);
}

function projectOpenAIResponsesConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
    readOnlyCompatibilityProjection = false,
    projection?: OpenAIResponsesProjectionOptions,
): { conversation: OpenAIResponsesInputItem[]; mappings: NativeItemMapping[] } {
    const conversation: OpenAIResponsesInputItem[] = [];
    const mappings: NativeItemMapping[] = [];
    const selectedTurns = selectedCanonicalTurns(document, {
        allow_interrupted_with_replay_protocol: OPENAI_RESPONSES_PROTOCOL,
        allow_interrupted_with_complete_tool_calls: true,
    });
    if (!readOnlyCompatibilityProjection) {
        assertCanonicalContextProjection(document, selectedTurns, {
            label: 'OpenAI Responses',
            program_authorities: ['system', 'developer', 'ordinary'],
            ...(projection?.project_media_caption === undefined
                ? {}
                : {
                      preserve_media_caption: (block, owner) => {
                          const asset = document.assets[block.asset_id];
                          return (
                              asset !== undefined &&
                              projection.project_media_caption?.(block, asset, owner) !== undefined
                          );
                      },
                  }),
        });
    }
    for (const turn of selectedTurns) {
        if (!readOnlyCompatibilityProjection)
            assertProtectedReplayCompatibility(document, turn, OPENAI_RESPONSES_PROTOCOL, target);
    }
    for (const turn of selectedTurns) {
        const items =
            replayItems(turn, document, target, projection) ?? compileOrdinaryTurn(turn, document, target, projection);
        const itemIndex = conversation.length;
        conversation.push(...items);
        mappings.push({ canonical_id: turn.id, native_id: `items/${itemIndex}`, kind: 'turn' });
        for (const block of turn.blocks) {
            mappings.push({
                canonical_id: block.id,
                native_id: `items/${itemIndex}/blocks/${block.id}`,
                kind: 'block',
            });
            if (block.type === 'tool_call')
                mappings.push({ canonical_id: block.call_id, native_id: block.call_id, kind: 'call' });
            if (block.type === 'tool_result')
                mappings.push({ canonical_id: block.call_id, native_id: block.call_id, kind: 'call' });
        }
    }
    return { conversation, mappings };
}

/** Pure registered-protocol import. The report explicitly leaves continuation readiness unvalidated. */
export async function importOpenAIResponsesHistory(
    historyInput: unknown,
    options: NativeConversationImportOptions,
): Promise<NativeConversationImportResult> {
    return guardNativeConversationImport(async () => {
        options = snapshotNativeConversationImportOptions(options);
        const runtime = nativeConversationImportContext(options);
        const toolDefinitions = [...(options.tool_definitions ?? [])];
        let document = newNativeImportDocument(options);
        if (!isOpenAIResponsesHistory(historyInput, OPENAI_RESPONSES_PROTOCOL)) {
            throw new TypeError('Conversation is neither canonical nor registered OpenAI Responses history');
        }
        const historySnapshot = structuredClone(historyInput);
        const imported = await itemsToRecords({
            items: wrappedHistoryItems(historySnapshot) ?? [],
            scope: `${runtime.conversation_id}:legacy`,
            source: 'imported',
            runtime,
            provider: options.provider,
            model: options.model,
            tool_definitions: toolDefinitions,
            ...(sourceHistoryTurnNumber(historySnapshot) === undefined
                ? {}
                : { source_history_turn_number: sourceHistoryTurnNumber(historySnapshot) }),
        });
        const contextEntries = await Promise.all(
            imported.turns.map(async (turn, index) => ({
                id: await entityId('context', `${runtime.conversation_id}:legacy`, index),
                type: 'source_turn' as const,
                turn_id: turn.id,
            })),
        );
        document = appendConversationRecords(
            document,
            {
                turns: imported.turns,
                assets: imported.assets,
                tool_definitions: toolDefinitions,
                context_entries: contextEntries,
                execution_receipts: imported.execution_receipts,
            },
            {
                expected_revision: document.revision,
                operation_id: await entityId('import', runtime.conversation_id, OPENAI_RESPONSES_PROTOCOL),
                payload_fingerprint: await fingerprintNativeConversationImport(
                    providerJsonValue(historySnapshot),
                    options,
                    OPENAI_RESPONSES_PROTOCOL,
                    OPENAI_RESPONSES_ADAPTER_VERSION,
                ),
                recorded_at: runtime.recorded_at,
            },
        ).document;
        return nativeConversationImportResult(
            document,
            options,
            OPENAI_RESPONSES_PROTOCOL,
            OPENAI_RESPONSES_ADAPTER_VERSION,
        );
    });
}

export async function prepareOpenAIResponsesCanonicalState(input: {
    conversation: unknown;
    prompt: OpenAIResponsesInputItem[];
    options: ExecutionOptions;
    provider: string;
}): Promise<Omit<PreparedOpenAIResponsesConversation, 'payload' | 'receipt' | 'diagnostics'>> {
    const runtime = resolveConversationRuntime(input.options);
    let document = parseCanonicalConversation(input.conversation);
    const toolDefinitions = await resolveCanonicalToolDefinitions(document, input.options.tools);
    if (document === undefined) {
        document = newCanonicalConversation(runtime);
        if (input.conversation !== undefined && input.conversation !== null) {
            document = (
                await importOpenAIResponsesHistory(input.conversation, {
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
    const priorNativeItemCount = compileOpenAIResponsesConversation(document, target).conversation.length;
    const received = await itemsToRecords({
        items: input.prompt,
        scope: runtime.input_operation_id,
        source: 'received',
        runtime: { ...runtime, conversation_id: document.id },
        provider: input.provider,
        model: input.options.model,
        tool_definitions: toolDefinitions,
    });
    const contextEntries = await Promise.all(
        received.turns.map(async (turn, index) => ({
            id: await entityId('context', runtime.input_operation_id, index),
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
    );
    const appended = await appendCanonicalPrompt(
        document,
        { ...received, context_entries: contextEntries, item_mappings: received.mappings },
        { ...runtime, conversation_id: document.id },
        input.options.tools,
        providerJsonValue(input.prompt),
    );
    const acceptedResponse = acceptedCanonicalResponse(appended.document, runtime.response_operation_id);
    if (
        acceptedResponse !== undefined &&
        (acceptedResponse.generation.request_id !== runtime.request_id ||
            acceptedResponse.generation.provider !== input.provider ||
            acceptedResponse.generation.protocol !== OPENAI_RESPONSES_PROTOCOL ||
            acceptedResponse.generation.requested_model !== input.options.model)
    ) {
        throw new Error(
            `Accepted response operation ${runtime.response_operation_id} has incompatible request identity`,
        );
    }
    const requestDocument =
        acceptedResponse === undefined
            ? appended.document
            : await acceptedCanonicalRequestDocument(appended.document, acceptedResponse);
    const compiled = compileOpenAIResponsesConversation(requestDocument, target);
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
        prior_native_item_count: priorNativeItemCount,
        ...(acceptedResponse === undefined ? {} : { accepted_response: acceptedResponse }),
    };
}

/** Prepare a retained canonical document directly, without importing native response input items. */
export async function prepareOpenAIResponsesCanonicalContext(input: {
    options: CanonicalExecutionContextOptions;
    provider: string;
    signal?: AbortSignal;
}): Promise<Omit<PreparedOpenAIResponsesConversation, 'payload' | 'receipt' | 'diagnostics'>> {
    const resolveAsset = input.options.resolve_canonical_asset;
    const signal = input.signal;
    const prepared = await prepareCanonicalContext({
        options: input.options,
        provider: input.provider,
        protocol: OPENAI_RESPONSES_PROTOCOL,
        adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
    });
    const target = { provider: input.provider, model: input.options.model };
    const hydrated = new Map<string, { fingerprint: string; data: string }>();
    const compiled = await compileOpenAIResponsesContextWithHostAssets(
        prepared.request_document,
        target,
        resolveAsset,
        signal,
        hydrated,
    );
    const priorCompiled =
        prepared.request_document === prepared.document
            ? compiled
            : await compileOpenAIResponsesContextWithHostAssets(
                  prepared.document,
                  target,
                  resolveAsset,
                  signal,
                  hydrated,
              );
    const priorNativeItemCount = priorCompiled.conversation.length;
    const { request_document: _requestDocument, ...base } = prepared;
    return {
        ...base,
        native_conversation: compiled.conversation,
        native_projection: {
            source_fingerprint: await fingerprintJson(prepared.document),
            mappings: compiled.mappings,
        },
        provider: input.provider,
        requested_model: input.options.model,
        prior_native_item_count: priorNativeItemCount,
    };
}

export async function finalizeOpenAIResponsesPreparedRequest(
    state: Omit<PreparedOpenAIResponsesConversation, 'payload' | 'receipt' | 'diagnostics'>,
    payload: OpenAIResponsesPayload,
    requestFingerprintPayload: JsonValue = providerJsonValue(payload),
): Promise<PreparedOpenAIResponsesConversation> {
    const model = state.requested_model;
    if (
        state.native_projection !== undefined &&
        state.native_projection.source_fingerprint !== (await fingerprintJson(state.document))
    )
        throw new TypeError('OpenAI Responses canonical source changed after native image projection');
    const mappings =
        state.native_projection?.mappings ??
        compileOpenAIResponsesConversation(state.document, { provider: state.provider, model }).mappings;
    const targetOptions = canonicalToolSelectionTargetOptions(undefined, state.response_selection_policy);
    const receipt = await createRequestReceipt(
        state.document,
        state.runtime,
        {
            provider: state.provider,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            model,
            adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
            ...(targetOptions === undefined ? {} : { options: targetOptions }),
        },
        requestFingerprintPayload,
        mappings,
        state.tool_definitions,
    );
    return { ...state, payload, receipt, diagnostics: [] };
}

function safeUsageNumber(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

type OpenAIResponsesUsageLike = OpenAI.Responses.ResponseUsage & {
    cached_tokens?: number | null;
    cache_write_tokens?: number | null;
    prompt_tokens_details?: { cached_tokens?: number | null; cache_write_tokens?: number | null } | null;
    input_tokens_details?: OpenAI.Responses.ResponseUsage['input_tokens_details'] & {
        cache_write_tokens?: number | null;
    };
};

function openAIResponsesUsage(native: OpenAI.Responses.ResponseUsage | null | undefined): GenerationUsage | undefined {
    if (native == null) return undefined;
    const details = native as OpenAIResponsesUsageLike;
    const input = safeUsageNumber(native.input_tokens);
    const output = safeUsageNumber(native.output_tokens);
    const reasoningCandidate = safeUsageNumber(native.output_tokens_details?.reasoning_tokens);
    const reasoning =
        reasoningCandidate !== undefined && (output === undefined || reasoningCandidate <= output)
            ? reasoningCandidate
            : undefined;
    const cacheReadCandidate = safeUsageNumber(
        details.input_tokens_details?.cached_tokens ??
            details.prompt_tokens_details?.cached_tokens ??
            details.cached_tokens,
    );
    const cacheRead =
        input !== undefined && cacheReadCandidate !== undefined && cacheReadCandidate <= input
            ? cacheReadCandidate
            : undefined;
    const cacheWrite = safeUsageNumber(
        details.input_tokens_details?.cache_write_tokens ??
            details.prompt_tokens_details?.cache_write_tokens ??
            details.cache_write_tokens,
    );
    const total =
        input !== undefined && output !== undefined && Number.isSafeInteger(input + output)
            ? input + output
            : undefined;
    const inputNewCandidate =
        input !== undefined && cacheRead !== undefined ? input - cacheRead - (cacheWrite ?? 0) : undefined;
    const inputNew = inputNewCandidate !== undefined && inputNewCandidate >= 0 ? inputNewCandidate : undefined;
    const basis = 'openai_responses_tokens';
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(reasoning === undefined ? {} : { reasoning_tokens: reasoning }),
        ...(total === undefined ? {} : { total_tokens: total }),
        ...(cacheRead === undefined ? {} : { cache_read_tokens: cacheRead }),
        ...(cacheWrite === undefined ? {} : { cache_write_tokens: cacheWrite }),
        ...(inputNew === undefined ? {} : { input_new_tokens: inputNew }),
        accounting_provenance: {
            ...(input === undefined ? {} : { input_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(output === undefined
                ? {}
                : { output_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(reasoning === undefined
                ? {}
                : { reasoning_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(total === undefined ? {} : { total_tokens: { method: 'derived' as const, accounting_basis: basis } }),
            ...(cacheRead === undefined
                ? {}
                : { cache_read_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(cacheWrite === undefined
                ? {}
                : { cache_write_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(inputNew === undefined
                ? {}
                : { input_new_tokens: { method: 'derived' as const, accounting_basis: basis } }),
        },
        ...(inputNew === undefined
            ? {}
            : {
                  input_partition: {
                      type: 'complete_disjoint',
                      cache_write_bucket: cacheWrite === undefined ? 'inapplicable' : 'included',
                  },
              }),
        reported_usage: [
            {
                source: 'provider',
                protocol: OPENAI_RESPONSES_PROTOCOL,
                accounting_basis: basis,
                payload: providerJsonValue(native),
            },
        ],
    };
}

export async function decodeOpenAIResponsesCanonicalResponse(input: {
    response: OpenAI.Responses.Response;
    prepared: PreparedOpenAIResponsesConversation;
    fallback_items?: OpenAIResponsesInputItem[];
    structured_output?: CanonicalStructuredOutput;
}): Promise<DecodedConversationResponse> {
    const { response, prepared } = input;
    if (response.status !== 'completed' && response.status !== 'incomplete') {
        throw new Error(`OpenAI Responses response ${response.id} is not a completed terminal response`);
    }
    const items = response.output.length > 0 ? response.output : (input.fallback_items ?? []);
    const completedAt = prepared.runtime.completed_at ?? new Date().toISOString();
    const records = await assistantItemsRecords({
        items: providerJsonValue(items) as unknown as Record<string, unknown>[],
        first_item_index: 0,
        scope: prepared.runtime.response_operation_id,
        source: 'received',
        runtime: { ...prepared.runtime, recorded_at: completedAt },
        provider: prepared.provider,
        model: prepared.requested_model,
        tool_definitions: prepared.tool_definitions,
    });
    const received = records.turns[0];
    if (received?.kind !== 'agent') throw new Error('OpenAI Responses response did not decode to an agent turn');
    const boundReceived = bindProtectedResponsesExchange([
        ...selectedCanonicalTurns(prepared.document, {
            allow_interrupted_with_replay_protocol: OPENAI_RESPONSES_PROTOCOL,
            allow_interrupted_with_complete_tool_calls: true,
        }),
        received,
    ]).at(-1);
    if (boundReceived?.kind !== 'agent') throw new Error('OpenAI Responses response dependency binding failed');
    const hasTools = boundReceived.blocks.some((block) => block.type === 'tool_call');
    const finishReason =
        response.status === 'incomplete'
            ? response.incomplete_details?.reason === 'max_output_tokens'
                ? 'length'
                : (response.incomplete_details?.reason ?? 'incomplete')
            : hasTools
              ? 'tool_use'
              : 'stop';
    const turn: ConversationTurn = {
        ...received,
        id: prepared.response_turn_id,
        status: response.status === 'incomplete' ? 'interrupted' : 'completed',
        timestamps: {
            recorded_at: completedAt,
            ...(prepared.runtime.started_at === undefined ? {} : { started_at: prepared.runtime.started_at }),
            completed_at: completedAt,
        },
        provenance: { type: 'generated' },
        generation_id: prepared.generation_id,
        blocks: boundReceived.blocks.map((block) =>
            block.type !== 'native_replay'
                ? block
                : {
                      ...block,
                      dependencies: {
                          ...block.dependencies,
                          turn_ids: block.dependencies.turn_ids.map((turnId) =>
                              turnId === boundReceived.id ? prepared.response_turn_id : turnId,
                          ),
                          request_ids: [prepared.receipt.request_id],
                      },
                  },
        ),
    };
    const baseGeneration = await createExecutedGeneration({
        id: prepared.generation_id,
        runtime: prepared.runtime,
        receipt: prepared.receipt,
        provider: prepared.provider,
        protocol: OPENAI_RESPONSES_PROTOCOL,
        adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
        requested_model: prepared.requested_model,
        resolved_model: response.model,
        provider_response_id: response.id,
        finish_reason: finishReason,
        usage: openAIResponsesUsage(response.usage),
    });
    const generation: ExecutedGeneration = {
        ...baseGeneration,
        status: response.status === 'incomplete' ? 'cancelled' : 'completed',
        ...(typeof response.service_tier === 'string'
            ? { metadata: { openai_responses: { service_tier: response.service_tier } } }
            : {}),
    };
    const decoded: DecodedConversationResponse = {
        turns: [turn],
        generation,
        assets: records.assets.map((asset) => ({
            ...asset,
            provenance: { type: 'generated', generation_id: generation.id, source_turn_id: turn.id },
        })),
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(providerJsonValue(response)),
    };
    if (input.structured_output === undefined) return decoded;
    return normalizeDecodedStructuredOutput(decoded, input.structured_output, ({ replay_blocks, binding }) => {
        const sources = new Set(binding.source_block_ids);
        let sourceCount = 0;
        const rewritten = replay_blocks.map((candidate) => {
            const payload = rawReplayPayload(candidate);
            let replaySourceCount = 0;
            const semanticEntries = payload.semantic_entries.map((entry): ReplaySemanticEntry => {
                if (entry.kind !== 'text' || !sources.has(entry.block_id)) return entry;
                sourceCount += 1;
                replaySourceCount += 1;
                return { ...entry, kind: 'structured_json', block_id: binding.block_id };
            });
            const dependsOnSource = candidate.dependencies.block_ids.some((blockId) => sources.has(blockId));
            if (replaySourceCount === 0 && !dependsOnSource) return candidate;
            const replay = remapStructuredOutputReplayDependencies(candidate, binding);
            if (replaySourceCount === 0) return replay;
            return {
                ...replay,
                payload: {
                    ...payload,
                    semantic_entries: semanticEntries,
                    structured_output: structuredOutputEvidence(binding),
                },
            };
        });
        if (sourceCount !== binding.source_texts.length) {
            throw new TypeError('OpenAI Responses structured output replay is missing source text partitions');
        }
        return rewritten;
    });
}

export function appendOpenAIResponsesCanonicalResponse(
    prepared: PreparedOpenAIResponsesConversation,
    decoded: DecodedConversationResponse,
): ConversationDocument {
    return appendCanonicalDecodedResponse(prepared, decoded, {
        operation_id: prepared.runtime.response_operation_id,
        recorded_at: decoded.generation.timestamps.recorded_at,
    }).document;
}

export async function appendOpenAIResponsesCanonicalResponseWithProcessing(
    prepared: PreparedOpenAIResponsesConversation,
    decoded: DecodedConversationResponse,
): Promise<ConversationDocument> {
    return (
        await appendCanonicalDecodedResponseWithProcessing(prepared, decoded, {
            operation_id: prepared.runtime.response_operation_id,
            recorded_at: decoded.generation.timestamps.recorded_at,
        })
    ).document;
}

/** Read-only compatibility projection for versioned legacy API responses. */
export function exportLegacyOpenAIResponsesConversation(document: ConversationDocument): OpenAIResponsesInputItem[] {
    return projectOpenAIResponsesConversation(parseConversationDocument(document), undefined, true).conversation;
}
