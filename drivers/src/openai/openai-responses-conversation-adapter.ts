import type { ExecutionOptions } from '@llumiverse/common';
import {
    type AgentContentBlock,
    type Asset,
    appendConversationRecords,
    appendDecodedConversationResponse,
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
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    type NestedToolResultContentBlock,
    type PreparedConversationRequest,
    type ProgramContentBlock,
    parseConversationDocument,
    preflightJsonInput,
    type ToolDefinition,
    type ToolResultBlock,
    type UserContentBlock,
} from '@llumiverse/conversation';
import type OpenAI from 'openai';
import {
    acceptedCanonicalResponse,
    appendCanonicalPrompt,
    type CanonicalPreparedState,
    canonicalResponseIdentities,
    canonicalToolDefinitions,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    parseCanonicalConversation,
    providerJsonValue,
    type ResolvedConversationRuntimeContext,
    resolveConversationRuntime,
    selectedCanonicalTurns,
} from '../conversation/canonical-runtime.js';

export const OPENAI_RESPONSES_PROTOCOL = 'openai.responses' as const;
export const OPENAI_RESPONSES_ADAPTER_VERSION = '2026-09-12.canonical.1' as const;

export type OpenAIResponsesInputItem = OpenAI.Responses.ResponseInputItem;
export type OpenAIResponsesPayload =
    | OpenAI.Responses.ResponseCreateParamsNonStreaming
    | OpenAI.Responses.ResponseCreateParamsStreaming;

type SourceKind = 'imported' | 'received';
type CanonicalToolResultStatus = 'success' | 'error' | 'cancelled' | 'denied';

export type CanonicalOpenAIResponsesFunctionCallOutput = OpenAI.Responses.ResponseInputItem.FunctionCallOutput & {
    /** Internal ingestion evidence. Removed before provider transport. */
    _llumiverse_tool_result_status?: CanonicalToolResultStatus;
};

interface ReplayTextEntry extends JsonObject {
    kind: 'text' | 'reasoning';
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

interface OpenAIResponsesReplayPayload extends JsonObject {
    type: 'openai_responses_items';
    items: JsonValue[];
    semantic_entries: ReplaySemanticEntry[];
}

interface ConvertedRecords {
    turns: ConversationTurn[];
    assets: Asset[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}

export interface PreparedOpenAIResponsesConversation
    extends CanonicalPreparedState<OpenAIResponsesInputItem[]>,
        PreparedConversationRequest<OpenAIResponsesPayload> {
    provider: string;
    requested_model: string;
    prior_native_item_count: number;
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
    const asset: Asset = {
        id: assetId,
        kind,
        mime_type: mimeType,
        storage,
        provenance:
            input.source === 'imported'
                ? { type: 'imported', source: OPENAI_RESPONSES_PROTOCOL }
                : { type: 'received', source_turn_id: input.turn_id },
        content_hash: await fingerprintJson({ mime_type: mimeType, storage }),
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
    runtime: ResolvedConversationRuntimeContext;
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
    runtime: ResolvedConversationRuntimeContext;
    provider: string;
    model: string;
    tool_definitions: readonly ToolDefinition[];
    source_history_turn_number?: number;
}): Promise<ConvertedRecords> {
    const nativePath = `items/${input.first_item_index}`;
    const turnId = await entityId('turn', input.scope, input.first_item_index);
    const blocks: AgentContentBlock[] = [];
    const assets: Asset[] = [];
    const semanticEntries: ReplaySemanticEntry[] = [];
    const mappings: NativeItemMapping[] = [{ canonical_id: turnId, native_id: nativePath, kind: 'turn' }];
    for (let localIndex = 0; localIndex < input.items.length; localIndex += 1) {
        const item = input.items[localIndex];
        const itemIndex = input.first_item_index + localIndex;
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
            const asset: Asset = {
                id: assetId,
                kind: 'image',
                mime_type: parsed?.mime_type ?? 'image/png',
                storage,
                provenance:
                    input.source === 'imported'
                        ? { type: 'imported', source: OPENAI_RESPONSES_PROTOCOL }
                        : { type: 'received', source_turn_id: turnId },
                content_hash: await fingerprintJson({ mime_type: parsed?.mime_type ?? 'image/png', storage }),
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
    const replayId = await entityId('replay', input.scope, input.first_item_index);
    const requiresModelScope = input.items.some((item) => {
        const type = item.type;
        return (
            type === 'reasoning' || (type !== 'message' && type !== 'function_call' && type !== 'image_generation_call')
        );
    });
    const replay: AgentContentBlock = {
        id: replayId,
        type: 'native_replay',
        adapter: OPENAI_RESPONSES_ADAPTER_VERSION,
        protocol: OPENAI_RESPONSES_PROTOCOL,
        compatibility_scope: {
            provider: input.provider,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            ...(requiresModelScope ? { model: input.model } : {}),
            adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
        },
        payload: {
            type: 'openai_responses_items',
            items: providerJsonValue(input.items) as JsonValue[],
            semantic_entries: semanticEntries,
        },
        dependencies: {
            turn_ids: [turnId],
            block_ids: blocks.map((block) => block.id),
            call_ids: blocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : [])),
            request_ids: [],
        },
    };
    blocks.push(replay);
    mappings.push({ canonical_id: replayId, native_id: `${nativePath}/replay`, kind: 'block' });
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
    runtime: ResolvedConversationRuntimeContext;
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
    runtime: ResolvedConversationRuntimeContext;
    provider: string;
    model: string;
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
                    item: item as unknown as CanonicalOpenAIResponsesFunctionCallOutput,
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

function ordinaryBlockToPart(
    block: ContentBlock,
    document: ConversationDocument,
    target?: { provider?: string },
): OpenAI.Responses.ResponseInputContent {
    if (block.type === 'text') return { type: 'input_text', text: block.text };
    if (block.type === 'json') return { type: 'input_text', text: JSON.stringify(block.value) };
    if (block.type === 'image' || block.type === 'document') {
        const asset = document.assets[block.asset_id];
        if (asset === undefined) throw new Error(`OpenAI Responses content references missing asset ${block.asset_id}`);
        return assetToInputPart(asset, target);
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
    const actualIds = [...blocks.keys()].sort();
    const expectedIds = [...block.dependencies.block_ids].sort();
    if (JSON.stringify(actualIds) !== JSON.stringify(expectedIds)) {
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
                    : stableJson(semantic.arguments.value) === stableJson(parsedArguments);
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
                const actualMimeType = parsed?.mime_type ?? 'image/png';
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
            let text: unknown;
            if (raw.type === 'function_call_output') {
                const part = Array.isArray(raw.output) ? raw.output[entry.content_index] : undefined;
                text =
                    typeof part === 'object' && part !== null && !Array.isArray(part)
                        ? ownValue(part, 'text')
                        : entry.content_index === 0 && typeof raw.output === 'string'
                          ? raw.output
                          : undefined;
            } else if (entry.kind === 'reasoning') {
                const collection =
                    semantic.type === 'reasoning' && semantic.representation === 'summary' ? raw.summary : raw.content;
                const part = Array.isArray(collection) ? collection[entry.content_index] : undefined;
                text =
                    typeof part === 'object' && part !== null && !Array.isArray(part)
                        ? ownValue(part, 'text')
                        : undefined;
            } else if (typeof raw.content === 'string' && entry.content_index === 0) {
                text = raw.content;
            } else {
                const part = Array.isArray(raw.content) ? raw.content[entry.content_index] : undefined;
                text =
                    typeof part === 'object' && part !== null && !Array.isArray(part)
                        ? ownValue(part, 'text')
                        : undefined;
            }
            if (text !== semanticText) {
                throw new TypeError(`OpenAI Responses replay text ${entry.block_id} no longer matches canonical data`);
            }
        }
    }
}

function replayItems(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): OpenAIResponsesInputItem[] | undefined {
    const replayBlocks: Array<Extract<ContentBlock, { type: 'native_replay' }>> = [];
    for (const block of turn.blocks) {
        if (block.type === 'native_replay') replayBlocks.push(block);
        if (block.type === 'tool_result') {
            for (const nested of block.content) if (nested.type === 'native_replay') replayBlocks.push(nested);
        }
    }
    const foreign = replayBlocks.find((block) => block.protocol !== OPENAI_RESPONSES_PROTOCOL);
    if (foreign !== undefined) {
        throw new TypeError(`OpenAI Responses cannot discard protected ${foreign.protocol} replay block ${foreign.id}`);
    }
    const matching = replayBlocks.filter((block) => block.protocol === OPENAI_RESPONSES_PROTOCOL);
    if (matching.length > 1) throw new TypeError(`OpenAI Responses turn ${turn.id} has multiple replay blocks`);
    const replay = matching[0];
    if (replay === undefined) return undefined;
    assertReplayScope(replay, target);
    const payload = rawReplayPayload(replay);
    assertReplaySemantics(turn, document, replay, payload, target);
    return providerJsonValue(payload.items) as unknown as OpenAIResponsesInputItem[];
}

function compileOrdinaryTurn(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string },
): OpenAIResponsesInputItem[] {
    if (turn.kind === 'tool') {
        const result = turn.blocks[0];
        const parts = result.content.flatMap((block): OpenAI.Responses.ResponseInputContent[] => {
            if (block.type === 'native_replay') return [];
            if (block.type === 'extension' && block.model_projection === 'excluded') return [];
            return [ordinaryBlockToPart(block, document, target)];
        });
        const output: string | OpenAI.Responses.ResponseInputContent[] =
            parts.length === 1 && parts[0].type === 'input_text' ? parts[0].text : parts;
        return [{ type: 'function_call_output', call_id: result.call_id, output }];
    }
    const parts = turn.blocks.flatMap((block): OpenAI.Responses.ResponseInputContent[] => {
        if (block.type === 'extension' && block.model_projection === 'excluded') return [];
        if (block.type === 'native_replay') return [];
        if (block.type === 'tool_call' || block.type === 'reasoning') {
            throw new TypeError(
                `OpenAI Responses requires protected replay for canonical ${block.type} block ${block.id}`,
            );
        }
        return [ordinaryBlockToPart(block, document, target)];
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
    return [{ role, content } as OpenAIResponsesInputItem];
}

export function compileOpenAIResponsesConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): { conversation: OpenAIResponsesInputItem[]; mappings: NativeItemMapping[] } {
    const conversation: OpenAIResponsesInputItem[] = [];
    const mappings: NativeItemMapping[] = [];
    for (const turn of selectedCanonicalTurns(document, {
        allow_interrupted_with_replay_protocol: OPENAI_RESPONSES_PROTOCOL,
    })) {
        const items = replayItems(turn, document, target) ?? compileOrdinaryTurn(turn, document, target);
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

export async function prepareOpenAIResponsesCanonicalState(input: {
    conversation: unknown;
    prompt: OpenAIResponsesInputItem[];
    options: ExecutionOptions;
    provider: string;
}): Promise<Omit<PreparedOpenAIResponsesConversation, 'payload' | 'receipt' | 'diagnostics'>> {
    const runtime = resolveConversationRuntime(input.options);
    const toolDefinitions = await canonicalToolDefinitions(input.options.tools);
    let document = parseCanonicalConversation(input.conversation);
    if (document === undefined) {
        document = newCanonicalConversation(runtime);
        if (input.conversation !== undefined && input.conversation !== null) {
            if (!isOpenAIResponsesHistory(input.conversation, OPENAI_RESPONSES_PROTOCOL)) {
                throw new TypeError('Conversation is neither canonical nor registered OpenAI Responses history');
            }
            const imported = await itemsToRecords({
                items: wrappedHistoryItems(input.conversation) ?? [],
                scope: `${runtime.conversation_id}:legacy`,
                source: 'imported',
                runtime,
                provider: input.provider,
                model: input.options.model,
                tool_definitions: toolDefinitions,
                ...(sourceHistoryTurnNumber(input.conversation) === undefined
                    ? {}
                    : { source_history_turn_number: sourceHistoryTurnNumber(input.conversation) }),
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
                    payload_fingerprint: await fingerprintJson(providerJsonValue(input.conversation)),
                    recorded_at: runtime.recorded_at,
                },
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
    const compiled = compileOpenAIResponsesConversation(appended.document, target);
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
    const identities =
        acceptedResponse === undefined
            ? await canonicalResponseIdentities(runtime)
            : { generation_id: acceptedResponse.generation.id, response_turn_id: acceptedResponse.turn.id };
    return {
        document: appended.document,
        native_conversation: compiled.conversation,
        runtime: { ...runtime, conversation_id: document.id },
        generation_id: identities.generation_id,
        response_turn_id: identities.response_turn_id,
        tool_definitions: appended.tool_definitions,
        provider: input.provider,
        requested_model: input.options.model,
        prior_native_item_count: priorNativeItemCount,
        ...(acceptedResponse === undefined ? {} : { accepted_response: acceptedResponse }),
    };
}

export async function finalizeOpenAIResponsesPreparedRequest(
    state: Omit<PreparedOpenAIResponsesConversation, 'payload' | 'receipt' | 'diagnostics'>,
    payload: OpenAIResponsesPayload,
): Promise<PreparedOpenAIResponsesConversation> {
    const model = state.requested_model;
    const compiled = compileOpenAIResponsesConversation(state.document, { provider: state.provider, model });
    const receipt = await createRequestReceipt(
        state.document,
        state.runtime,
        {
            provider: state.provider,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            model,
            adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
        },
        providerJsonValue(payload),
        compiled.mappings,
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
    const hasTools = received.blocks.some((block) => block.type === 'tool_call');
    const finishReason = hasTools
        ? 'tool_use'
        : response.status === 'incomplete'
          ? response.incomplete_details?.reason === 'max_output_tokens'
              ? 'length'
              : (response.incomplete_details?.reason ?? 'incomplete')
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
        blocks: received.blocks.map((block) =>
            block.type !== 'native_replay'
                ? block
                : {
                      ...block,
                      dependencies: {
                          ...block.dependencies,
                          turn_ids: [prepared.response_turn_id],
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
        ...(typeof response.service_tier === 'string'
            ? { metadata: { openai_responses: { service_tier: response.service_tier } } }
            : {}),
    };
    return {
        turns: [turn],
        generation,
        assets: records.assets.map((asset) => ({
            ...asset,
            provenance: { type: 'generated', generation_id: generation.id, source_turn_id: turn.id },
        })),
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(providerJsonValue(response)),
    };
}

export function appendOpenAIResponsesCanonicalResponse(
    prepared: PreparedOpenAIResponsesConversation,
    decoded: DecodedConversationResponse,
): ConversationDocument {
    return appendDecodedConversationResponse(prepared, decoded, {
        operation_id: prepared.runtime.response_operation_id,
        recorded_at: decoded.generation.timestamps.recorded_at,
    }).document;
}

/** Read-only compatibility projection for versioned legacy API responses. */
export function exportLegacyOpenAIResponsesConversation(document: ConversationDocument): OpenAIResponsesInputItem[] {
    return compileOpenAIResponsesConversation(parseConversationDocument(document)).conversation;
}
