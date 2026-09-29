import type {
    Content,
    FunctionResponsePart,
    GenerateContentParameters,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
} from '@google/genai';
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
    type NativeItemMapping,
    type NativeReplayBlock,
    type NestedToolResultContentBlock,
    type PreparedConversationRequest,
    type ProgramContentBlock,
    parseConversationDocument,
    preflightJsonInput,
    type ToolDefinition,
    type ToolResultBlock,
    type UserContentBlock,
} from '@llumiverse/conversation';
import type { CanonicalStructuredOutput, ExecutionOptions, JSONObject, ToolUse } from '@llumiverse/core';
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
} from '../../conversation/canonical-runtime.js';
import {
    assertStructuredOutputEvidence,
    normalizeDecodedStructuredOutput,
    parseStructuredOutputEvidence,
    remapStructuredOutputReplayDependencies,
    structuredOutputEvidence,
} from '../../conversation/structured-output.js';
import type { GenerateContentPrompt } from '../index.js';

export const GEMINI_GENERATE_CONTENT_PROTOCOL = 'google.generate_content' as const;
export const GEMINI_GENERATE_CONTENT_ADAPTER_VERSION = '2026-09-30.canonical.1' as const;

type SourceKind = 'imported' | 'received';
type ToolResultStatus = 'success' | 'error' | 'cancelled' | 'denied' | 'unknown';

type GeminiPromptPart = Part & {
    _llumiverse_tool_result_status?: Exclude<ToolResultStatus, 'unknown'>;
};

interface GeminiReplaySemanticEntry {
    block_id: string;
    part_index: number;
    kind: 'text' | 'reasoning' | 'structured_json' | 'asset' | 'tool_call' | 'tool_result_json' | 'extension';
    asset_id?: string;
    response_part_index?: number;
}

interface GeminiReplayPayload {
    type: 'gemini_content';
    content: JsonObject;
    semantic_entries: GeminiReplaySemanticEntry[];
    structured_output?: JsonObject;
}

interface ConvertedRecords {
    turns: ConversationTurn[];
    assets: Asset[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}

interface PendingCall {
    call_id: string;
    tool_name: string;
    native_id?: string;
}

export interface PreparedGeminiConversation
    extends CanonicalPreparedState<GenerateContentPrompt>,
        PreparedConversationRequest<GenerateContentParameters> {
    provider: string;
    requested_model: string;
    prior_native_content_count: number;
    current_native_content_indexes: number[];
}

export interface LegacyGeminiConversation {
    _arrayConversation: Content[];
    _llumiverse_meta: {
        turnNumber: number;
    };
    _llumiverse_system?: Content;
}

function ownValue(value: object, key: string): unknown {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && 'value' in descriptor ? descriptor.value : undefined;
}

async function entityId(kind: string, scope: string, ...position: Array<string | number>): Promise<string> {
    return deriveConversationId(kind, scope, ...position.map(String));
}

function importedProvenance(nativePath: string, sourceHistoryTurnNumber?: number): ImportedTurnProvenance {
    return {
        type: 'imported',
        source: GEMINI_GENERATE_CONTENT_PROTOCOL,
        native_id: { protocol: GEMINI_GENERATE_CONTENT_PROTOCOL, scope: 'history', value: nativePath },
        ...(sourceHistoryTurnNumber === undefined ? {} : { source_history_turn_number: sourceHistoryTurnNumber }),
        missing_metadata: ['actor_id', 'timestamps', 'exchange'],
    };
}

function turnProvenance(source: SourceKind, nativePath: string, sourceHistoryTurnNumber?: number) {
    return source === 'imported'
        ? importedProvenance(nativePath, sourceHistoryTurnNumber)
        : ({ type: 'received' } as const);
}

function sourceHistoryTurnNumber(history: unknown): number | undefined {
    if (typeof history !== 'object' || history === null || Array.isArray(history)) return undefined;
    const metadata = ownValue(history, '_llumiverse_meta');
    if (typeof metadata !== 'object' || metadata === null || Array.isArray(metadata)) return undefined;
    const turnNumber = ownValue(metadata, 'turnNumber');
    return typeof turnNumber === 'number' && Number.isSafeInteger(turnNumber) && turnNumber >= 0
        ? turnNumber
        : undefined;
}

function isGeminiContent(value: unknown): value is Content {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const role = ownValue(value, 'role');
    const parts = ownValue(value, 'parts');
    return (
        (role === undefined || role === 'user' || role === 'model') &&
        (parts === undefined ||
            (Array.isArray(parts) && parts.every((part) => typeof part === 'object' && part !== null)))
    );
}

function historyContents(history: unknown): Content[] | undefined {
    if (Array.isArray(history)) return history.every(isGeminiContent) ? history : undefined;
    if (typeof history !== 'object' || history === null) return undefined;
    const wrapped = ownValue(history, '_arrayConversation');
    return Array.isArray(wrapped) && wrapped.every(isGeminiContent) ? wrapped : undefined;
}

function historySystem(history: unknown): Content | undefined {
    if (typeof history !== 'object' || history === null || Array.isArray(history)) return undefined;
    const system = ownValue(history, '_llumiverse_system');
    return isGeminiContent(system) ? system : undefined;
}

export function isGeminiGenerateContentHistory(
    value: unknown,
    explicitProtocol?: typeof GEMINI_GENERATE_CONTENT_PROTOCOL,
): boolean {
    if (explicitProtocol !== GEMINI_GENERATE_CONTENT_PROTOCOL || !preflightJsonInput(value).success) return false;
    return historyContents(value) !== undefined;
}

function cleanPart(part: GeminiPromptPart): Part {
    const { _llumiverse_tool_result_status: _status, ...cleaned } = part;
    return cleaned;
}

function mediaKind(mimeType: string): Asset['kind'] {
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('audio/')) return 'audio';
    if (mimeType.startsWith('video/')) return 'video';
    return 'document';
}

function assetBlockType(kind: Asset['kind']): 'image' | 'document' | 'audio' | 'video' {
    if (kind === 'image' || kind === 'audio' || kind === 'video') return kind;
    return 'document';
}

async function mediaRecord(input: {
    part: Part | FunctionResponsePart;
    turn_id: string;
    scope: string;
    native_path: string;
    source: SourceKind;
    recorded_at: string;
    provider: string;
}): Promise<{ block: UserContentBlock; asset: Asset }> {
    const inlineData = input.part.inlineData;
    const fileData = input.part.fileData;
    if ((inlineData === undefined) === (fileData === undefined)) {
        throw new TypeError(`Gemini media ${input.native_path} must have exactly one source`);
    }
    const mimeType = inlineData?.mimeType ?? fileData?.mimeType;
    if (typeof mimeType !== 'string' || mimeType.length === 0) {
        throw new TypeError(`Gemini media ${input.native_path} has no MIME type`);
    }
    let storage: Asset['storage'];
    if (inlineData !== undefined) {
        if (typeof inlineData.data !== 'string') {
            throw new TypeError(`Gemini inline media ${input.native_path} has no base64 data`);
        }
        storage = { type: 'inline_base64', data: inlineData.data };
    } else {
        if (typeof fileData?.fileUri !== 'string' || fileData.fileUri.length === 0) {
            throw new TypeError(`Gemini file media ${input.native_path} has no URI`);
        }
        storage = { type: 'external', resolver: 'google_uri', locator: { uri: fileData.fileUri } };
    }
    const kind = mediaKind(mimeType);
    const assetId = await entityId('asset', input.scope, input.native_path);
    const blockId = await entityId('block', input.scope, input.native_path);
    const rawPart = providerJsonValue(input.part) as JsonObject;
    const asset: Asset = {
        id: assetId,
        kind,
        mime_type: mimeType,
        storage,
        provenance:
            input.source === 'imported'
                ? { type: 'imported', source: GEMINI_GENERATE_CONTENT_PROTOCOL }
                : { type: 'received', source_turn_id: input.turn_id },
        content_hash: await fingerprintJson({ mime_type: mimeType, storage }),
        created_at: input.recorded_at,
        metadata: { gemini_generate_content: { provider: input.provider, raw_part: rawPart } },
    };
    return {
        block: { id: blockId, type: assetBlockType(kind), asset_id: assetId } as UserContentBlock,
        asset,
    };
}

function geminiAssetMetadata(asset: Asset): JsonObject {
    const metadata = asset.metadata?.gemini_generate_content;
    return typeof metadata === 'object' && metadata !== null && !Array.isArray(metadata) ? metadata : {};
}

function assetToPart(asset: Asset, target?: { provider?: string }): Part | FunctionResponsePart {
    const metadata = geminiAssetMetadata(asset);
    const owningProvider = metadata.provider;
    if (
        asset.storage.type === 'external' &&
        asset.storage.resolver === 'google_uri' &&
        target?.provider !== undefined &&
        owningProvider !== target.provider
    ) {
        throw new TypeError(`Gemini file asset ${asset.id} belongs to provider ${String(owningProvider)}`);
    }
    const rawPart = metadata.raw_part;
    const portablePart =
        typeof rawPart === 'object' && rawPart !== null && !Array.isArray(rawPart)
            ? (rawPart as Record<string, unknown>)
            : undefined;
    const portableExtras = {
        ...(portablePart?.mediaResolution === undefined ? {} : { mediaResolution: portablePart.mediaResolution }),
        ...(portablePart?.videoMetadata === undefined ? {} : { videoMetadata: portablePart.videoMetadata }),
        ...(portablePart?.mediaProcessing === undefined ? {} : { mediaProcessing: portablePart.mediaProcessing }),
    } as Pick<Part, 'mediaResolution' | 'videoMetadata' | 'mediaProcessing'>;
    if (asset.storage.type === 'inline_base64') {
        const inline =
            portablePart?.inlineData !== undefined &&
            typeof portablePart.inlineData === 'object' &&
            portablePart.inlineData !== null &&
            !Array.isArray(portablePart.inlineData)
                ? portablePart.inlineData
                : {};
        return {
            ...portableExtras,
            inlineData: { ...inline, data: asset.storage.data, mimeType: asset.mime_type },
        } as Part;
    }
    if (
        asset.storage.type === 'external' &&
        asset.storage.resolver === 'google_uri' &&
        typeof asset.storage.locator.uri === 'string'
    ) {
        const file =
            portablePart?.fileData !== undefined &&
            typeof portablePart.fileData === 'object' &&
            portablePart.fileData !== null &&
            !Array.isArray(portablePart.fileData)
                ? portablePart.fileData
                : {};
        return {
            ...portableExtras,
            fileData: { ...file, fileUri: asset.storage.locator.uri, mimeType: asset.mime_type },
        } as Part;
    }
    throw new TypeError(`Gemini cannot resolve asset ${asset.id}`);
}

function isMediaPart(part: Part | FunctionResponsePart): boolean {
    return part.inlineData !== undefined || part.fileData !== undefined;
}

function assertPortableInputMediaPart(part: Part, nativePath: string): void {
    const allowed = new Set(['inlineData', 'fileData', 'mediaResolution', 'videoMetadata', 'mediaProcessing']);
    const unsupported = Object.keys(part).find((key) => !allowed.has(key));
    if (unsupported !== undefined) {
        throw new TypeError(`Gemini user media ${nativePath} has unsupported protected field ${unsupported}`);
    }
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

function contentHasProtectedReplay(content: Content): boolean {
    return (content.parts ?? []).some(
        (part) =>
            (typeof part.thoughtSignature === 'string' && part.thoughtSignature.length > 0) ||
            part.executableCode !== undefined ||
            part.codeExecutionResult !== undefined ||
            part.toolCall !== undefined ||
            part.toolResponse !== undefined,
    );
}

async function replayBlock(input: {
    content: Content;
    semantic_entries: GeminiReplaySemanticEntry[];
    semantic_block_ids: string[];
    call_ids: string[];
    turn_id: string;
    scope: string;
    native_path: string;
    provider: string;
    model: string;
}): Promise<NativeReplayBlock> {
    const payload: GeminiReplayPayload = {
        type: 'gemini_content',
        content: providerJsonValue(input.content) as JsonObject,
        semantic_entries: providerJsonValue(input.semantic_entries) as unknown as GeminiReplaySemanticEntry[],
    };
    return {
        id: await entityId('replay', input.scope, input.native_path),
        type: 'native_replay',
        adapter: GEMINI_GENERATE_CONTENT_ADAPTER_VERSION,
        protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
        compatibility_scope: {
            provider: input.provider,
            protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
            ...(contentHasProtectedReplay(input.content) ? { model: input.model } : {}),
            adapter_version: GEMINI_GENERATE_CONTENT_ADAPTER_VERSION,
        },
        payload: providerJsonValue(payload),
        dependencies: {
            turn_ids: [input.turn_id],
            block_ids: input.semantic_block_ids,
            call_ids: input.call_ids,
            request_ids: [],
        },
        content_hash: await fingerprintJson(providerJsonValue(payload)),
    };
}

function extensionBlock(input: { id: string; part: Part }): AgentContentBlock {
    return {
        id: input.id,
        type: 'extension',
        namespace: 'google.generate_content.native_part',
        version: GEMINI_GENERATE_CONTENT_ADAPTER_VERSION,
        payload: providerJsonValue(input.part),
        model_projection: 'excluded',
    };
}

function callRegistry(document: ConversationDocument): Map<string, PendingCall> {
    const calls = new Map<string, PendingCall>();
    const answered = new Set(document.turns.flatMap((turn) => (turn.kind === 'tool' ? [turn.blocks[0].call_id] : [])));
    for (const turn of document.turns) {
        if (turn.kind !== 'agent') continue;
        for (const block of turn.blocks) {
            if (block.type !== 'tool_call') continue;
            if (answered.has(block.call_id)) continue;
            calls.set(block.call_id, {
                call_id: block.call_id,
                tool_name: block.tool_name,
                ...(block.native_id?.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL
                    ? { native_id: block.native_id.value }
                    : {}),
            });
        }
    }
    return calls;
}

function resolveToolResultCall(
    response: NonNullable<Part['functionResponse']>,
    calls: Map<string, PendingCall>,
    unmatchedByName: Map<string, PendingCall[]>,
): PendingCall {
    const consume = (matched: PendingCall): PendingCall => {
        calls.delete(matched.call_id);
        const queue = unmatchedByName.get(matched.tool_name);
        if (queue !== undefined) {
            const index = queue.findIndex((candidate) => candidate.call_id === matched.call_id);
            if (index >= 0) queue.splice(index, 1);
        }
        return matched;
    };
    if (typeof response.id === 'string' && response.id.length > 0) {
        const matched = calls.get(response.id);
        if (matched === undefined) {
            throw new TypeError(`Gemini function response ${response.id} has no matching function call`);
        }
        if (response.name !== undefined && response.name !== matched.tool_name) {
            throw new TypeError(
                `Gemini function response ${response.id} names ${response.name}, expected ${matched.tool_name}`,
            );
        }
        return consume(matched);
    }
    if (typeof response.name !== 'string' || response.name.length === 0) {
        throw new TypeError('Gemini function response has no name or resolvable call id');
    }
    const candidates = unmatchedByName.get(response.name) ?? [];
    const matched = candidates.shift();
    if (matched !== undefined) {
        calls.delete(matched.call_id);
        return matched;
    }
    throw new TypeError(`Gemini function response for ${response.name} has no matching function call`);
}

function toolResultStatus(part: GeminiPromptPart): ToolResultStatus {
    const status = part._llumiverse_tool_result_status;
    return status === 'success' || status === 'error' || status === 'cancelled' || status === 'denied'
        ? status
        : 'unknown';
}

async function contentRecords(input: {
    content: Content;
    content_index: number;
    scope: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    provider: string;
    model: string;
    tool_definitions: readonly ToolDefinition[];
    calls: Map<string, PendingCall>;
    unmatched_by_name: Map<string, PendingCall[]>;
    call_ordinal: { value: number };
    source_history_turn_number?: number;
    authority?: 'system' | 'developer';
}): Promise<ConvertedRecords> {
    const nativePath = input.authority ? 'system' : `contents/${input.content_index}`;
    const cleanContent: Content = {
        ...input.content,
        ...(input.content.parts === undefined ? {} : { parts: input.content.parts.map((part) => cleanPart(part)) }),
    };
    const functionResponses = (input.content.parts ?? []).flatMap((part, partIndex) =>
        part.functionResponse === undefined ? [] : [{ part: part as GeminiPromptPart, partIndex }],
    );
    if (functionResponses.length > 0) {
        if (
            input.authority !== undefined ||
            (input.content.parts ?? []).some((part) => part.functionResponse === undefined)
        ) {
            throw new TypeError(`Gemini ${nativePath} mixes function responses with incompatible content`);
        }
        const turns: ConversationTurn[] = [];
        const assets: Asset[] = [];
        const mappings: NativeItemMapping[] = [];
        const executionReceipts: ExecutionReceipt[] = [];
        for (const { part, partIndex } of functionResponses) {
            const response = part.functionResponse;
            if (response === undefined) continue;
            const matched = resolveToolResultCall(response, input.calls, input.unmatched_by_name);
            const { id: _responseId, ...responseWithoutId } = response;
            const normalizedPart = {
                ...cleanPart(part),
                functionResponse: {
                    ...responseWithoutId,
                    ...(matched.native_id === undefined ? {} : { id: matched.native_id }),
                    name: matched.tool_name,
                },
            } satisfies Part;
            const turnId = await entityId('turn', input.scope, input.content_index, 'tool_result', partIndex);
            const resultBlockId = await entityId('block', input.scope, input.content_index, 'tool_result', partIndex);
            const nested: NestedToolResultContentBlock[] = [];
            const semanticEntries: GeminiReplaySemanticEntry[] = [];
            if (response.response !== undefined) {
                const blockId = await entityId('block', input.scope, input.content_index, partIndex, 'response');
                nested.push({ id: blockId, type: 'json', value: providerJsonValue(response.response) });
                semanticEntries.push({ block_id: blockId, part_index: partIndex, kind: 'tool_result_json' });
                mappings.push({
                    canonical_id: blockId,
                    native_id: `${nativePath}/parts/${partIndex}/functionResponse/response`,
                    kind: 'block',
                });
            }
            for (let responsePartIndex = 0; responsePartIndex < (response.parts?.length ?? 0); responsePartIndex += 1) {
                const responsePart = response.parts?.[responsePartIndex];
                if (responsePart === undefined || !isMediaPart(responsePart)) {
                    throw new TypeError(`Gemini ${nativePath} has an unsupported function response part`);
                }
                const converted = await mediaRecord({
                    part: responsePart,
                    turn_id: turnId,
                    scope: input.scope,
                    native_path: `${nativePath}/parts/${partIndex}/functionResponse/parts/${responsePartIndex}`,
                    source: input.source,
                    recorded_at: input.runtime.recorded_at,
                    provider: input.provider,
                });
                nested.push(converted.block as NestedToolResultContentBlock);
                assets.push(converted.asset);
                semanticEntries.push({
                    block_id: converted.block.id,
                    part_index: partIndex,
                    response_part_index: responsePartIndex,
                    kind: 'asset',
                    asset_id: converted.asset.id,
                });
                mappings.push({
                    canonical_id: converted.block.id,
                    native_id: `${nativePath}/parts/${partIndex}/functionResponse/parts/${responsePartIndex}`,
                    kind: 'block',
                });
            }
            const replay = await replayBlock({
                content: { ...cleanContent, parts: [normalizedPart] },
                semantic_entries: semanticEntries.map((entry) => ({ ...entry, part_index: 0 })),
                semantic_block_ids: semanticEntries.map((entry) => entry.block_id),
                call_ids: [matched.call_id],
                turn_id: turnId,
                scope: input.scope,
                native_path: `${nativePath}/parts/${partIndex}`,
                provider: input.provider,
                model: input.model,
            });
            nested.push(replay);
            const status = toolResultStatus(part);
            const resultBlock: ToolResultBlock = {
                id: resultBlockId,
                type: 'tool_result',
                call_id: matched.call_id,
                status,
                content: nested,
                ...(typeof response.id === 'string'
                    ? {
                          native_id: {
                              protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                              scope: input.runtime.request_id,
                              value: response.id,
                          },
                      }
                    : {}),
            };
            const common = {
                id: turnId,
                status: 'completed' as const,
                timestamps: { recorded_at: input.runtime.recorded_at },
                model_visibility: 'include' as const,
                provenance: turnProvenance(
                    input.source,
                    `${nativePath}/parts/${partIndex}`,
                    input.source_history_turn_number,
                ),
            };
            turns.push({ ...common, kind: 'tool', authority: 'ordinary', blocks: [resultBlock] });
            mappings.push(
                { canonical_id: turnId, native_id: `${nativePath}/parts/${partIndex}`, kind: 'turn' },
                { canonical_id: resultBlockId, native_id: `${nativePath}/parts/${partIndex}`, kind: 'block' },
                { canonical_id: replay.id, native_id: `${nativePath}/parts/${partIndex}/replay`, kind: 'block' },
                {
                    canonical_id: matched.call_id,
                    native_id: response.id ?? response.name ?? matched.call_id,
                    kind: 'call',
                },
            );
            if (status !== 'unknown') {
                executionReceipts.push({
                    id: await entityId('execution_receipt', input.scope, matched.call_id),
                    call_id: matched.call_id,
                    executor: 'application',
                    status,
                    result_turn_id: turnId,
                    result_fingerprint: await fingerprintJson(resultBlock),
                    recorded_at: input.runtime.recorded_at,
                });
            }
        }
        return { turns, assets, mappings, execution_receipts: executionReceipts };
    }

    const turnId = await entityId('turn', input.scope, input.authority ?? input.content_index);
    const blocks: AgentContentBlock[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [{ canonical_id: turnId, native_id: nativePath, kind: 'turn' }];
    const semanticEntries: GeminiReplaySemanticEntry[] = [];
    const callIds: string[] = [];
    for (let partIndex = 0; partIndex < (input.content.parts?.length ?? 0); partIndex += 1) {
        const rawPart = input.content.parts?.[partIndex] as GeminiPromptPart | undefined;
        if (rawPart === undefined) continue;
        const part = cleanPart(rawPart);
        const blockId = await entityId('block', input.scope, nativePath, partIndex);
        let mappedBlockId = blockId;
        if (typeof part.text === 'string') {
            const block: AgentContentBlock = part.thought
                ? { id: blockId, type: 'reasoning', text: part.text, representation: 'text' }
                : { id: blockId, type: 'text', text: part.text, format: 'plain' };
            blocks.push(block);
            semanticEntries.push({
                block_id: blockId,
                part_index: partIndex,
                kind: part.thought ? 'reasoning' : 'text',
            });
        } else if (isMediaPart(part)) {
            if (input.authority === undefined && input.content.role !== 'model') {
                assertPortableInputMediaPart(part, `${nativePath}/parts/${partIndex}`);
            }
            const converted = await mediaRecord({
                part,
                turn_id: turnId,
                scope: input.scope,
                native_path: `${nativePath}/parts/${partIndex}`,
                source: input.source,
                recorded_at: input.runtime.recorded_at,
                provider: input.provider,
            });
            blocks.push(converted.block as AgentContentBlock);
            assets.push(converted.asset);
            mappedBlockId = converted.block.id;
            semanticEntries.push({
                block_id: converted.block.id,
                part_index: partIndex,
                kind: 'asset',
                asset_id: converted.asset.id,
            });
        } else if (part.functionCall !== undefined) {
            const name = part.functionCall.name;
            if (typeof name !== 'string' || name.length === 0) {
                throw new TypeError(`Gemini function call ${nativePath}/parts/${partIndex} has no name`);
            }
            const nativeId = part.functionCall.id;
            const callOrdinal = input.call_ordinal.value;
            input.call_ordinal.value += 1;
            const callId =
                typeof nativeId === 'string' && nativeId.length > 0
                    ? nativeId
                    : await entityId('call', input.scope, callOrdinal);
            const definition = input.tool_definitions.find((candidate) => candidate.name === name);
            blocks.push({
                id: blockId,
                type: 'tool_call',
                call_id: callId,
                tool_name: name,
                ...(definition === undefined ? {} : { definition_id: definition.id }),
                executor: 'application',
                arguments: { type: 'json', value: providerJsonValue(part.functionCall.args ?? {}) },
                ...(typeof nativeId === 'string' && nativeId.length > 0
                    ? {
                          native_id: {
                              protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                              scope: input.runtime.request_id,
                              value: nativeId,
                          },
                      }
                    : {}),
            });
            const pending = {
                call_id: callId,
                tool_name: name,
                ...(typeof nativeId === 'string' && nativeId.length > 0 ? { native_id: nativeId } : {}),
            };
            input.calls.set(callId, pending);
            const queue = input.unmatched_by_name.get(name) ?? [];
            queue.push(pending);
            input.unmatched_by_name.set(name, queue);
            callIds.push(callId);
            semanticEntries.push({ block_id: blockId, part_index: partIndex, kind: 'tool_call' });
            mappings.push({
                canonical_id: callId,
                native_id: nativeId ?? `${nativePath}/calls/${callOrdinal}`,
                kind: 'call',
            });
        } else {
            const extension = extensionBlock({ id: blockId, part });
            blocks.push(extension);
            semanticEntries.push({ block_id: blockId, part_index: partIndex, kind: 'extension' });
        }
        mappings.push({ canonical_id: mappedBlockId, native_id: `${nativePath}/parts/${partIndex}`, kind: 'block' });
    }

    const common = {
        id: turnId,
        status: 'completed' as const,
        timestamps: { recorded_at: input.runtime.recorded_at },
        model_visibility: 'include' as const,
        provenance: turnProvenance(input.source, nativePath, input.source_history_turn_number),
    };
    if (input.authority !== undefined) {
        const replay = await replayBlock({
            content: cleanContent,
            semantic_entries: semanticEntries,
            semantic_block_ids: blocks.map((block) => block.id),
            call_ids: callIds,
            turn_id: turnId,
            scope: input.scope,
            native_path: nativePath,
            provider: input.provider,
            model: input.model,
        });
        const programBlocks = blocks.filter((block): block is ProgramContentBlock => block.type !== 'tool_call');
        programBlocks.push(replay);
        mappings.push({ canonical_id: replay.id, native_id: `${nativePath}/replay`, kind: 'block' });
        return {
            turns: [{ ...common, kind: 'program', authority: input.authority, blocks: programBlocks }],
            assets,
            mappings,
            execution_receipts: [],
        };
    }
    if (input.content.role === 'model') {
        const replay = await replayBlock({
            content: cleanContent,
            semantic_entries: semanticEntries,
            semantic_block_ids: blocks.map((block) => block.id),
            call_ids: callIds,
            turn_id: turnId,
            scope: input.scope,
            native_path: nativePath,
            provider: input.provider,
            model: input.model,
        });
        blocks.push(replay);
        mappings.push({ canonical_id: replay.id, native_id: `${nativePath}/replay`, kind: 'block' });
        const agentTurn: ConversationTurn =
            input.source === 'imported'
                ? {
                      ...common,
                      kind: 'agent',
                      authority: 'ordinary',
                      blocks,
                      provenance: importedProvenance(nativePath, input.source_history_turn_number),
                  }
                : {
                      ...common,
                      kind: 'agent',
                      authority: 'ordinary',
                      blocks,
                      provenance: { type: 'received' },
                  };
        return {
            turns: [agentTurn],
            assets,
            mappings,
            execution_receipts: [],
        };
    }
    const userBlocks = blocks.filter(
        (block): block is UserContentBlock =>
            block.type !== 'tool_call' && block.type !== 'reasoning' && block.type !== 'native_replay',
    );
    if (userBlocks.length !== blocks.length) {
        throw new TypeError(`Gemini user content ${nativePath} contains unsupported reasoning or tool calls`);
    }
    return {
        turns: [{ ...common, kind: 'user', authority: 'ordinary', blocks: userBlocks }],
        assets,
        mappings,
        execution_receipts: [],
    };
}

async function contentsToRecords(input: {
    contents: readonly Content[];
    system?: Content;
    scope: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    provider: string;
    model: string;
    tool_definitions: readonly ToolDefinition[];
    existing_calls?: Map<string, PendingCall>;
    source_history_turn_number?: number;
}): Promise<ConvertedRecords> {
    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    const executionReceipts: ExecutionReceipt[] = [];
    const calls = input.existing_calls ?? new Map<string, PendingCall>();
    const unmatchedByName = new Map<string, PendingCall[]>();
    for (const call of calls.values()) {
        const queue = unmatchedByName.get(call.tool_name) ?? [];
        queue.push(call);
        unmatchedByName.set(call.tool_name, queue);
    }
    const callOrdinal = { value: 0 };
    const append = (converted: ConvertedRecords) => {
        turns.push(...converted.turns);
        assets.push(...converted.assets);
        mappings.push(...converted.mappings);
        executionReceipts.push(...converted.execution_receipts);
    };
    if (input.system !== undefined) {
        append(
            await contentRecords({
                content: input.system,
                content_index: -1,
                scope: input.scope,
                source: input.source,
                runtime: input.runtime,
                provider: input.provider,
                model: input.model,
                tool_definitions: input.tool_definitions,
                calls,
                unmatched_by_name: unmatchedByName,
                call_ordinal: callOrdinal,
                authority: 'system',
                ...(input.source_history_turn_number === undefined
                    ? {}
                    : { source_history_turn_number: input.source_history_turn_number }),
            }),
        );
    }
    for (let index = 0; index < input.contents.length; index += 1) {
        append(
            await contentRecords({
                content: input.contents[index],
                content_index: index,
                scope: input.scope,
                source: input.source,
                runtime: input.runtime,
                provider: input.provider,
                model: input.model,
                tool_definitions: input.tool_definitions,
                calls,
                unmatched_by_name: unmatchedByName,
                call_ordinal: callOrdinal,
                ...(input.source_history_turn_number === undefined
                    ? {}
                    : { source_history_turn_number: input.source_history_turn_number }),
            }),
        );
    }
    return { turns, assets, mappings, execution_receipts: executionReceipts };
}

function rawReplayPayload(block: NativeReplayBlock): GeminiReplayPayload {
    const payload = block.payload;
    if (
        typeof payload !== 'object' ||
        payload === null ||
        Array.isArray(payload) ||
        payload.type !== 'gemini_content' ||
        typeof payload.content !== 'object' ||
        payload.content === null ||
        Array.isArray(payload.content) ||
        !Array.isArray(payload.semantic_entries)
    ) {
        throw new TypeError(`Gemini replay block ${block.id} has an unsupported payload`);
    }
    return payload as unknown as GeminiReplayPayload;
}

function assertReplayScope(block: NativeReplayBlock, target?: { provider?: string; model?: string }): void {
    if (
        block.adapter !== GEMINI_GENERATE_CONTENT_ADAPTER_VERSION ||
        block.protocol !== GEMINI_GENERATE_CONTENT_PROTOCOL ||
        block.compatibility_scope.protocol !== GEMINI_GENERATE_CONTENT_PROTOCOL ||
        block.compatibility_scope.adapter_version !== GEMINI_GENERATE_CONTENT_ADAPTER_VERSION ||
        (target?.provider !== undefined && block.compatibility_scope.provider !== target.provider) ||
        (target?.model !== undefined &&
            block.compatibility_scope.model !== undefined &&
            block.compatibility_scope.model !== target.model)
    ) {
        throw new TypeError(`Gemini replay block ${block.id} is outside its compatibility scope`);
    }
}

function semanticBlocks(turn: ConversationTurn): Map<string, ContentBlock> {
    const blocks = new Map<string, ContentBlock>();
    for (const block of turn.blocks) {
        if (block.type === 'native_replay') continue;
        if (block.type === 'tool_result') {
            for (const nested of block.content) if (nested.type !== 'native_replay') blocks.set(nested.id, nested);
        } else {
            blocks.set(block.id, block);
        }
    }
    return blocks;
}

function mediaPartMatchesAsset(
    part: Part | FunctionResponsePart,
    asset: Asset,
    target?: { provider?: string },
): boolean {
    const raw = geminiAssetMetadata(asset).raw_part;
    if (stableJson(raw) !== stableJson(providerJsonValue(part))) return false;
    if (part.inlineData !== undefined && asset.storage.type === 'inline_base64') {
        return part.inlineData.data === asset.storage.data && part.inlineData.mimeType === asset.mime_type;
    }
    if (part.fileData !== undefined && asset.storage.type === 'external' && asset.storage.resolver === 'google_uri') {
        if (target?.provider !== undefined && geminiAssetMetadata(asset).provider !== target.provider) return false;
        return part.fileData.fileUri === asset.storage.locator.uri && part.fileData.mimeType === asset.mime_type;
    }
    return false;
}

function assertReplaySemantics(
    turn: ConversationTurn,
    document: ConversationDocument,
    replay: NativeReplayBlock,
    payload: GeminiReplayPayload,
    target?: { provider?: string },
): void {
    const blocks = semanticBlocks(turn);
    const actualIds = [...blocks.keys()].sort();
    const expectedIds = [...replay.dependencies.block_ids].sort();
    if (stableJson(actualIds) !== stableJson(expectedIds)) {
        throw new TypeError(`Gemini replay block ${replay.id} no longer matches its semantic blocks`);
    }
    const content = payload.content as unknown as Content;
    const parts = content.parts ?? [];
    if (payload.structured_output !== undefined) {
        const evidence = parseStructuredOutputEvidence(payload.structured_output);
        const sourceTexts = payload.semantic_entries.flatMap((entry) => {
            if (entry.kind !== 'structured_json') return [];
            const text = parts[entry.part_index]?.text;
            return typeof text === 'string' ? [text] : [];
        });
        assertStructuredOutputEvidence(turn, evidence, sourceTexts, replay.id);
    }
    for (const entry of payload.semantic_entries) {
        const block = blocks.get(entry.block_id);
        const part = parts[entry.part_index];
        if (block === undefined || part === undefined) {
            throw new TypeError(`Gemini replay entry ${entry.block_id} is missing canonical or native content`);
        }
        if (entry.kind === 'structured_json') {
            if (block.type !== 'json' || typeof part.text !== 'string' || part.thought) {
                throw new TypeError(`Gemini structured replay ${entry.block_id} no longer matches canonical data`);
            }
        } else if (entry.kind === 'text' || entry.kind === 'reasoning') {
            const expectedType = entry.kind === 'text' ? 'text' : 'reasoning';
            if (
                block.type !== expectedType ||
                part.text !== block.text ||
                !!part.thought !== (entry.kind === 'reasoning')
            ) {
                throw new TypeError(`Gemini replay text ${entry.block_id} no longer matches canonical data`);
            }
        } else if (entry.kind === 'tool_call') {
            if (
                block.type !== 'tool_call' ||
                part.functionCall === undefined ||
                part.functionCall.name !== block.tool_name ||
                stableJson(part.functionCall.args ?? {}) !==
                    stableJson(block.arguments.type === 'json' ? block.arguments.value : undefined) ||
                (typeof part.functionCall.id === 'string' && part.functionCall.id !== block.call_id)
            ) {
                throw new TypeError(`Gemini replay tool call ${entry.block_id} no longer matches canonical data`);
            }
        } else if (entry.kind === 'asset') {
            if (
                (block.type !== 'image' &&
                    block.type !== 'document' &&
                    block.type !== 'audio' &&
                    block.type !== 'video') ||
                block.asset_id !== entry.asset_id
            ) {
                throw new TypeError(`Gemini replay asset ${entry.block_id} no longer matches canonical data`);
            }
            const asset = entry.asset_id === undefined ? undefined : document.assets[entry.asset_id];
            const mediaPart =
                entry.response_part_index === undefined
                    ? part
                    : part.functionResponse?.parts?.[entry.response_part_index];
            if (asset === undefined || mediaPart === undefined || !mediaPartMatchesAsset(mediaPart, asset, target)) {
                throw new TypeError(`Gemini replay asset ${String(entry.asset_id)} no longer matches canonical data`);
            }
        } else if (entry.kind === 'tool_result_json') {
            if (
                block.type !== 'json' ||
                part.functionResponse === undefined ||
                stableJson(part.functionResponse.response) !== stableJson(block.value)
            ) {
                throw new TypeError(`Gemini replay tool result ${entry.block_id} no longer matches canonical data`);
            }
        } else if (block.type !== 'extension' || stableJson(block.payload) !== stableJson(providerJsonValue(part))) {
            throw new TypeError(`Gemini replay extension ${entry.block_id} no longer matches canonical data`);
        }
    }
    if (turn.kind === 'tool') {
        const result = turn.blocks[0];
        const response = parts[0]?.functionResponse;
        const call = findToolCall(document, result.call_id);
        const expectedNativeId =
            call.native_id?.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL ? call.native_id.value : undefined;
        if (
            response === undefined ||
            response.name !== call.tool_name ||
            (response.id !== undefined && response.id !== expectedNativeId) ||
            !replay.dependencies.call_ids.includes(result.call_id)
        ) {
            throw new TypeError(`Gemini replay block ${replay.id} no longer matches its tool result call`);
        }
    }
}

function replayContent(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): Content | undefined {
    const replayBlocks: NativeReplayBlock[] = [];
    for (const block of turn.blocks) {
        if (block.type === 'native_replay') replayBlocks.push(block);
        if (block.type === 'tool_result') {
            for (const nested of block.content) if (nested.type === 'native_replay') replayBlocks.push(nested);
        }
    }
    const foreign = replayBlocks.find((block) => block.protocol !== GEMINI_GENERATE_CONTENT_PROTOCOL);
    if (foreign !== undefined) {
        throw new TypeError(`Gemini cannot discard protected ${foreign.protocol} replay block ${foreign.id}`);
    }
    const matching = replayBlocks.filter((block) => block.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL);
    if (matching.length > 1) throw new TypeError(`Gemini turn ${turn.id} has multiple replay blocks`);
    const replay = matching[0];
    if (replay === undefined) return undefined;
    assertReplayScope(replay, target);
    const payload = rawReplayPayload(replay);
    assertReplaySemantics(turn, document, replay, payload, target);
    return structuredClone(payload.content) as unknown as Content;
}

function ordinaryBlockToPart(
    block: ContentBlock,
    document: ConversationDocument,
    target?: { provider?: string },
): Part {
    if (block.type === 'text') return { text: block.text };
    if (block.type === 'json') return { text: JSON.stringify(block.value) };
    if (block.type === 'image' || block.type === 'document' || block.type === 'audio' || block.type === 'video') {
        const asset = document.assets[block.asset_id];
        if (asset === undefined) throw new Error(`Gemini content references missing asset ${block.asset_id}`);
        return assetToPart(asset, target) as Part;
    }
    if (block.type === 'tool_call') {
        if (block.arguments.type !== 'json') {
            throw new TypeError(`Gemini cannot project unresolved tool arguments for call ${block.call_id}`);
        }
        if (
            typeof block.arguments.value !== 'object' ||
            block.arguments.value === null ||
            Array.isArray(block.arguments.value)
        ) {
            throw new TypeError(`Gemini tool call ${block.call_id} requires JSON object arguments`);
        }
        return {
            functionCall: {
                id: block.native_id?.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL ? block.native_id.value : undefined,
                name: block.tool_name,
                args: block.arguments.value as Record<string, unknown>,
            },
        };
    }
    if (block.type === 'reasoning') {
        throw new TypeError(`Gemini requires protected replay for reasoning block ${block.id}`);
    }
    if (
        block.type === 'extension' &&
        block.namespace === 'google.generate_content.native_part' &&
        typeof block.payload === 'object' &&
        block.payload !== null &&
        !Array.isArray(block.payload)
    ) {
        return structuredClone(block.payload) as Part;
    }
    throw new TypeError(`Gemini cannot project canonical ${block.type} block ${block.id}`);
}

function findToolCall(document: ConversationDocument, callId: string): Extract<ContentBlock, { type: 'tool_call' }> {
    for (const turn of document.turns) {
        if (turn.kind !== 'agent') continue;
        const call = turn.blocks.find((block) => block.type === 'tool_call' && block.call_id === callId);
        if (call?.type === 'tool_call') return call;
    }
    throw new TypeError(`Gemini tool result references unknown call ${callId}`);
}

function compileOrdinaryTurn(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string },
): { content?: Content; systemParts?: Part[] } {
    if (turn.kind === 'tool') {
        const result = turn.blocks[0];
        const call = findToolCall(document, result.call_id);
        const semanticContent = result.content.filter((block) => block.type !== 'native_replay');
        const primary = semanticContent.filter((block) => block.type === 'json' || block.type === 'text');
        if (primary.length > 1) {
            throw new TypeError(`Gemini cannot project multiple text/JSON values for tool result ${result.id}`);
        }
        const primaryIndex = primary.length === 0 ? -1 : semanticContent.indexOf(primary[0]);
        if (primaryIndex > 0) {
            throw new TypeError(`Gemini cannot reorder tool result content for ${result.id}`);
        }
        const responseParts = semanticContent.flatMap((block): FunctionResponsePart[] => {
            if (block.type === 'json' || block.type === 'text') return [];
            if (block.type === 'extension' && block.model_projection === 'excluded') return [];
            if (
                block.type === 'image' ||
                block.type === 'document' ||
                block.type === 'audio' ||
                block.type === 'video'
            ) {
                const asset = document.assets[block.asset_id];
                if (asset === undefined)
                    throw new Error(`Gemini tool result references missing asset ${block.asset_id}`);
                return [assetToPart(asset, target) as FunctionResponsePart];
            }
            throw new TypeError(`Gemini cannot project tool-result ${block.type} block ${block.id}`);
        });
        const primaryValue = primary[0];
        const response =
            primaryValue?.type === 'json'
                ? typeof primaryValue.value === 'object' &&
                  primaryValue.value !== null &&
                  !Array.isArray(primaryValue.value)
                    ? (primaryValue.value as Record<string, unknown>)
                    : { output: primaryValue.value }
                : primaryValue?.type === 'text'
                  ? { output: primaryValue.text }
                  : undefined;
        return {
            content: {
                role: 'user',
                parts: [
                    {
                        functionResponse: {
                            ...(call.native_id?.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL
                                ? { id: call.native_id.value }
                                : {}),
                            name: call.tool_name,
                            ...(response === undefined ? {} : { response }),
                            ...(responseParts.length === 0 ? {} : { parts: responseParts }),
                        },
                    },
                ],
            },
        };
    }
    const parts = turn.blocks.flatMap((block): Part[] => {
        if (block.type === 'native_replay') return [];
        if (block.type === 'extension' && block.model_projection === 'excluded') return [];
        return [ordinaryBlockToPart(block, document, target)];
    });
    if (turn.kind === 'program') {
        if (turn.authority === 'system') return { systemParts: parts };
        if (turn.authority === 'ordinary') return { content: { role: 'user', parts } };
        throw new TypeError(`Gemini cannot project program authority ${turn.authority}`);
    }
    return { content: { role: turn.kind === 'agent' ? 'model' : 'user', parts } };
}

function geminiReplayBlock(turn: ConversationTurn): NativeReplayBlock | undefined {
    const blocks: readonly ContentBlock[] = turn.kind === 'tool' ? turn.blocks[0].content : turn.blocks;
    for (const block of blocks) {
        if (block.type === 'native_replay' && block.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL) return block;
    }
    return undefined;
}

function compiledBlockMappings(turn: ConversationTurn, nativeBase: string, partOffset = 0): NativeItemMapping[] {
    const mappings: NativeItemMapping[] = [];
    const replay = geminiReplayBlock(turn);
    if (turn.kind === 'tool') {
        const result = turn.blocks[0];
        mappings.push({ canonical_id: result.id, native_id: `${nativeBase}/parts/0/functionResponse`, kind: 'block' });
        if (replay !== undefined) {
            const payload = rawReplayPayload(replay);
            for (const entry of payload.semantic_entries) {
                const nativeId =
                    entry.kind === 'tool_result_json'
                        ? `${nativeBase}/parts/0/functionResponse/response`
                        : entry.kind === 'asset' && entry.response_part_index !== undefined
                          ? `${nativeBase}/parts/0/functionResponse/parts/${entry.response_part_index}`
                          : `${nativeBase}/parts/0`;
                mappings.push({ canonical_id: entry.block_id, native_id: nativeId, kind: 'block' });
            }
            mappings.push({ canonical_id: replay.id, native_id: `${nativeBase}/parts/0`, kind: 'block' });
        } else {
            let responsePartIndex = 0;
            for (const block of result.content) {
                if (block.type === 'json' || block.type === 'text') {
                    mappings.push({
                        canonical_id: block.id,
                        native_id: `${nativeBase}/parts/0/functionResponse/response`,
                        kind: 'block',
                    });
                } else if (
                    block.type === 'image' ||
                    block.type === 'document' ||
                    block.type === 'audio' ||
                    block.type === 'video'
                ) {
                    mappings.push({
                        canonical_id: block.id,
                        native_id: `${nativeBase}/parts/0/functionResponse/parts/${responsePartIndex}`,
                        kind: 'block',
                    });
                    responsePartIndex += 1;
                }
            }
        }
        return mappings;
    }
    if (replay !== undefined) {
        const payload = rawReplayPayload(replay);
        for (const entry of payload.semantic_entries) {
            mappings.push({
                canonical_id: entry.block_id,
                native_id: `${nativeBase}/parts/${partOffset + entry.part_index}`,
                kind: 'block',
            });
        }
        mappings.push({ canonical_id: replay.id, native_id: nativeBase, kind: 'block' });
        return mappings;
    }
    let partIndex = partOffset;
    for (const block of turn.blocks) {
        if (block.type === 'native_replay') continue;
        if (block.type === 'extension' && block.model_projection === 'excluded') continue;
        mappings.push({
            canonical_id: block.id,
            native_id: `${nativeBase}/parts/${partIndex}`,
            kind: 'block',
        });
        partIndex += 1;
    }
    return mappings;
}

export function compileGeminiConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): { conversation: GenerateContentPrompt; mappings: NativeItemMapping[] } {
    const contents: Content[] = [];
    const systemParts: Part[] = [];
    const mappings: NativeItemMapping[] = [];
    for (const turn of selectedCanonicalTurns(document)) {
        if (turn.kind === 'program' && turn.authority === 'developer') {
            throw new TypeError('Gemini cannot project developer program authority');
        }
        const replay = replayContent(turn, document, target);
        const compiled =
            replay !== undefined
                ? turn.kind === 'program'
                    ? turn.authority === 'system'
                        ? { systemParts: replay.parts ?? [] }
                        : { content: { ...replay, role: 'user' as const } }
                    : { content: replay }
                : compileOrdinaryTurn(turn, document, target);
        if (compiled.systemParts !== undefined) {
            const firstPart = systemParts.length;
            systemParts.push(...compiled.systemParts);
            mappings.push({ canonical_id: turn.id, native_id: `system/parts/${firstPart}`, kind: 'turn' });
            mappings.push(...compiledBlockMappings(turn, 'system', firstPart));
        } else if (compiled.content !== undefined) {
            const contentIndex = contents.length;
            contents.push(compiled.content);
            mappings.push({ canonical_id: turn.id, native_id: `contents/${contentIndex}`, kind: 'turn' });
            mappings.push(...compiledBlockMappings(turn, `contents/${contentIndex}`));
        }
        for (const block of turn.blocks) {
            if (block.type === 'tool_call') {
                mappings.push({
                    canonical_id: block.call_id,
                    native_id: block.native_id?.value ?? block.call_id,
                    kind: 'call',
                });
            }
            if (block.type === 'tool_result') {
                mappings.push({
                    canonical_id: block.call_id,
                    native_id: block.native_id?.value ?? block.call_id,
                    kind: 'call',
                });
            }
        }
    }
    return {
        conversation: {
            contents,
            ...(systemParts.length === 0 ? {} : { system: { role: 'user', parts: systemParts } }),
        },
        mappings,
    };
}

export async function prepareGeminiCanonicalState(input: {
    conversation: unknown;
    prompt: GenerateContentPrompt;
    options: ExecutionOptions;
    provider: string;
}): Promise<Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>> {
    const runtime = resolveConversationRuntime(input.options);
    let document = parseCanonicalConversation(input.conversation);
    const toolDefinitions = await canonicalToolDefinitions(input.options.tools);
    if (document === undefined) {
        document = newCanonicalConversation(runtime);
        if (input.conversation !== undefined && input.conversation !== null) {
            if (!isGeminiGenerateContentHistory(input.conversation, GEMINI_GENERATE_CONTENT_PROTOCOL)) {
                throw new TypeError('Conversation is neither canonical nor registered Gemini GenerateContent history');
            }
            const legacyContents = historyContents(input.conversation);
            if (legacyContents === undefined) throw new TypeError('Gemini history has no content array');
            const imported = await contentsToRecords({
                contents: legacyContents,
                ...(historySystem(input.conversation) === undefined
                    ? {}
                    : { system: historySystem(input.conversation) }),
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
                    operation_id: await entityId('import', runtime.conversation_id, GEMINI_GENERATE_CONTENT_PROTOCOL),
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
    const priorCompiled = compileGeminiConversation(document, target).conversation;
    const priorNativeContentCount = priorCompiled.contents.length;
    const cleanPrompt = providerJsonValue(input.prompt) as unknown as GenerateContentPrompt;
    const inputWasAccepted = Object.hasOwn(document.operation_receipts, runtime.input_operation_id);
    const promptSystem = inputWasAccepted || priorCompiled.system === undefined ? cleanPrompt.system : undefined;
    const promptRecords = await contentsToRecords({
        contents: cleanPrompt.contents,
        ...(promptSystem === undefined ? {} : { system: promptSystem }),
        scope: runtime.input_operation_id,
        source: 'received',
        runtime: { ...runtime, conversation_id: document.id },
        provider: input.provider,
        model: input.options.model,
        tool_definitions: toolDefinitions,
        existing_calls: callRegistry(document),
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
        {
            ...promptRecords,
            context_entries: contextEntries,
            item_mappings: promptRecords.mappings,
            execution_receipts: promptRecords.execution_receipts,
        },
        { ...runtime, conversation_id: document.id },
        input.options.tools,
        providerJsonValue(cleanPrompt),
    );
    const compiled = compileGeminiConversation(appended.document, target);
    const acceptedInputTurnIds = new Set(
        appended.document.operation_receipts[runtime.input_operation_id]?.accepted_turn_ids ?? [],
    );
    const currentNativeContentIndexes = compiled.mappings.flatMap((mapping) => {
        if (mapping.kind !== 'turn' || !acceptedInputTurnIds.has(mapping.canonical_id)) return [];
        const match = /^contents\/(\d+)$/.exec(mapping.native_id);
        return match === null ? [] : [Number(match[1])];
    });
    const acceptedResponse = acceptedCanonicalResponse(appended.document, runtime.response_operation_id);
    if (
        acceptedResponse !== undefined &&
        (acceptedResponse.generation.request_id !== runtime.request_id ||
            acceptedResponse.generation.provider !== input.provider ||
            acceptedResponse.generation.protocol !== GEMINI_GENERATE_CONTENT_PROTOCOL ||
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
        prior_native_content_count: priorNativeContentCount,
        current_native_content_indexes: currentNativeContentIndexes,
        ...(acceptedResponse === undefined ? {} : { accepted_response: acceptedResponse }),
    };
}

export async function finalizeGeminiPreparedRequest(
    state: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
    payload: GenerateContentParameters,
): Promise<PreparedGeminiConversation> {
    const compiled = compileGeminiConversation(state.document, {
        provider: state.provider,
        model: state.requested_model,
    });
    const receipt = await createRequestReceipt(
        state.document,
        state.runtime,
        {
            provider: state.provider,
            protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
            model: state.requested_model,
            adapter_version: GEMINI_GENERATE_CONTENT_ADAPTER_VERSION,
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

function safeSum(...values: Array<number | undefined>): number | undefined {
    if (values.some((value) => value === undefined)) return undefined;
    const total = (values as number[]).reduce((sum, value) => sum + value, 0);
    return Number.isSafeInteger(total) ? total : undefined;
}

export function geminiGenerationUsage(
    usageMetadata: GenerateContentResponseUsageMetadata | undefined,
): GenerationUsage | undefined {
    if (usageMetadata === undefined) return undefined;
    const input = safeUsageNumber(usageMetadata.promptTokenCount);
    const candidates = safeUsageNumber(usageMetadata.candidatesTokenCount);
    const thoughtsCandidate = safeUsageNumber(usageMetadata.thoughtsTokenCount);
    const toolUse = safeUsageNumber(usageMetadata.toolUsePromptTokenCount);
    const reportedTotal = safeUsageNumber(usageMetadata.totalTokenCount);
    const bucketOutput = safeSum(candidates, thoughtsCandidate, toolUse);
    const output =
        bucketOutput ??
        (input !== undefined && reportedTotal !== undefined && reportedTotal >= input
            ? reportedTotal - input
            : undefined);
    const thoughts =
        thoughtsCandidate !== undefined && (output === undefined || thoughtsCandidate <= output)
            ? thoughtsCandidate
            : undefined;
    const total =
        input !== undefined && output !== undefined && safeSum(input, output) === reportedTotal
            ? reportedTotal
            : undefined;
    const cacheReadCandidate = safeUsageNumber(usageMetadata.cachedContentTokenCount);
    const cacheRead =
        input !== undefined && cacheReadCandidate !== undefined && cacheReadCandidate <= input
            ? cacheReadCandidate
            : undefined;
    const inputNew = input !== undefined && cacheRead !== undefined ? input - cacheRead : undefined;
    const basis = 'gemini_generate_content_tokens';
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(thoughts === undefined ? {} : { reasoning_tokens: thoughts }),
        ...(total === undefined ? {} : { total_tokens: total }),
        ...(cacheRead === undefined ? {} : { cache_read_tokens: cacheRead }),
        ...(inputNew === undefined ? {} : { input_new_tokens: inputNew }),
        accounting_provenance: {
            ...(input === undefined ? {} : { input_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(output === undefined ? {} : { output_tokens: { method: 'derived' as const, accounting_basis: basis } }),
            ...(thoughts === undefined
                ? {}
                : { reasoning_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(total === undefined ? {} : { total_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(cacheRead === undefined
                ? {}
                : { cache_read_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(inputNew === undefined
                ? {}
                : { input_new_tokens: { method: 'derived' as const, accounting_basis: basis } }),
        },
        ...(inputNew === undefined
            ? {}
            : { input_partition: { type: 'complete_disjoint' as const, cache_write_bucket: 'inapplicable' as const } }),
        reported_usage: [
            {
                source: 'provider',
                protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                accounting_basis: basis,
                payload: providerJsonValue(usageMetadata),
            },
        ],
    };
}

export async function geminiToolUsesFromContent(
    content: Content | undefined,
    responseOperationId: string,
    startOrdinal = 0,
): Promise<ToolUse[] | undefined> {
    const out: ToolUse[] = [];
    let callOrdinal = 0;
    for (const part of content?.parts ?? []) {
        const call = part.functionCall;
        if (call === undefined) continue;
        const name = call.name ?? '';
        const id =
            typeof call.id === 'string' && call.id.length > 0
                ? call.id
                : await entityId('call', responseOperationId, startOrdinal + callOrdinal);
        callOrdinal += 1;
        out.push({
            id,
            tool_name: name,
            tool_input: providerJsonValue(call.args ?? {}) as JSONObject,
            ...(typeof part.thoughtSignature === 'string' ? { thought_signature: part.thoughtSignature } : {}),
        });
    }
    return out.length === 0 ? undefined : out;
}

export async function decodeGeminiCanonicalResponse(input: {
    response: GenerateContentResponse;
    content: Content;
    prepared: PreparedGeminiConversation;
    finish_reason?: string;
    structured_output?: CanonicalStructuredOutput;
}): Promise<DecodedConversationResponse> {
    const runtime = {
        ...input.prepared.runtime,
        recorded_at: input.prepared.runtime.completed_at ?? new Date().toISOString(),
    };
    const records = await contentsToRecords({
        contents: [{ ...input.content, role: 'model' }],
        scope: input.prepared.runtime.response_operation_id,
        source: 'received',
        runtime,
        provider: input.prepared.provider,
        model: input.prepared.requested_model,
        tool_definitions: input.prepared.tool_definitions,
        existing_calls: callRegistry(input.prepared.document),
    });
    const received = records.turns[0];
    if (received?.kind !== 'agent') throw new Error('Gemini response did not decode to an agent turn');
    const turn: ConversationTurn = {
        ...received,
        id: input.prepared.response_turn_id,
        status: input.finish_reason === 'length' ? 'interrupted' : 'completed',
        timestamps: {
            recorded_at: runtime.recorded_at,
            ...(input.prepared.runtime.started_at === undefined
                ? {}
                : { started_at: input.prepared.runtime.started_at }),
            completed_at: runtime.recorded_at,
        },
        provenance: { type: 'generated' },
        generation_id: input.prepared.generation_id,
    };
    const finalTurn = {
        ...turn,
        blocks: turn.blocks.map((block) =>
            block.type !== 'native_replay'
                ? block
                : {
                      ...block,
                      dependencies: {
                          ...block.dependencies,
                          turn_ids: [turn.id],
                          request_ids: [input.prepared.receipt.request_id],
                      },
                  },
        ),
    } as ConversationTurn;
    const generation: ExecutedGeneration = await createExecutedGeneration({
        id: input.prepared.generation_id,
        runtime: input.prepared.runtime,
        receipt: input.prepared.receipt,
        provider: input.prepared.provider,
        protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
        adapter_version: GEMINI_GENERATE_CONTENT_ADAPTER_VERSION,
        requested_model: input.prepared.requested_model,
        resolved_model: input.response.modelVersion ?? input.prepared.payload.model,
        provider_response_id: input.response.responseId,
        finish_reason: input.finish_reason,
        usage: geminiGenerationUsage(input.response.usageMetadata),
    });
    const decoded: DecodedConversationResponse = {
        turns: [finalTurn],
        generation,
        assets: records.assets.map((asset) => ({
            ...asset,
            provenance: { type: 'generated', generation_id: generation.id, source_turn_id: finalTurn.id },
        })),
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(providerJsonValue(input.response)),
    };
    if (input.structured_output === undefined) return decoded;
    return normalizeDecodedStructuredOutput(decoded, input.structured_output, async ({ replay_blocks, binding }) => {
        if (replay_blocks.length !== 1) {
            throw new TypeError(`Gemini structured output turn ${finalTurn.id} requires one replay block`);
        }
        const replay = remapStructuredOutputReplayDependencies(replay_blocks[0], binding);
        const payload = rawReplayPayload(replay);
        const sources = new Set(binding.source_block_ids);
        let sourceCount = 0;
        const semanticEntries = payload.semantic_entries.map((entry): GeminiReplaySemanticEntry => {
            if (entry.kind !== 'text' || !sources.has(entry.block_id)) return entry;
            sourceCount += 1;
            return { ...entry, kind: 'structured_json', block_id: binding.block_id };
        });
        if (sourceCount !== binding.source_texts.length) {
            throw new TypeError('Gemini structured output replay is missing source text partitions');
        }
        const nextPayload: GeminiReplayPayload = {
            ...payload,
            semantic_entries: semanticEntries,
            structured_output: structuredOutputEvidence(binding),
        };
        return [
            {
                ...replay,
                payload: providerJsonValue(nextPayload),
                content_hash: await fingerprintJson(providerJsonValue(nextPayload)),
            },
        ];
    });
}

export function appendGeminiCanonicalResponse(
    prepared: PreparedGeminiConversation,
    decoded: DecodedConversationResponse,
): ConversationDocument {
    return appendDecodedConversationResponse(prepared, decoded, {
        operation_id: prepared.runtime.response_operation_id,
        recorded_at: decoded.generation.timestamps.recorded_at,
    }).document;
}

export function exportLegacyGeminiConversation(document: ConversationDocument): LegacyGeminiConversation {
    const parsed = parseConversationDocument(document);
    const compiled = compileGeminiConversation(parsed).conversation;
    return {
        _arrayConversation: compiled.contents,
        _llumiverse_meta: {
            turnNumber: Object.values(parsed.generations).filter(
                (generation) => generation.record_source === 'executed',
            ).length,
        },
        ...(compiled.system === undefined ? {} : { _llumiverse_system: compiled.system }),
    };
}
