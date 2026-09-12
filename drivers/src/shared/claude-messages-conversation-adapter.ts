import type {
    ContentBlockParam,
    DocumentBlockParam,
    ImageBlockParam,
    Message,
    MessageCreateParamsBase,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
} from '@anthropic-ai/sdk/resources/messages.js';
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
import type { AnthropicUsageLike, ClaudePrompt } from './claude-messages.js';

export const CLAUDE_MESSAGES_PROTOCOL = 'anthropic.messages' as const;
export const CLAUDE_MESSAGES_ADAPTER_VERSION = '2026-09-11.canonical.1' as const;

export type CanonicalToolResultStatus = 'success' | 'error' | 'cancelled' | 'denied';
export type CanonicalClaudeToolResultBlockParam = ToolResultBlockParam & {
    /** Internal ingestion evidence. This property is removed before provider transport. */
    _llumiverse_tool_result_status?: CanonicalToolResultStatus;
};

interface ClaudeReplayCanonicalEntry extends JsonObject {
    kind: 'canonical';
    block_id: string;
}
interface ClaudeReplayThinkingEntry extends JsonObject {
    kind: 'thinking';
    block_id: string;
    signature: string;
}
interface ClaudeReplayRedactedEntry extends JsonObject {
    kind: 'redacted_thinking';
    data: string;
}
type ClaudeReplayEntry = ClaudeReplayCanonicalEntry | ClaudeReplayThinkingEntry | ClaudeReplayRedactedEntry;
type ClaudeReplayPayload = JsonObject & {
    type: 'claude_messages_content_order';
    entries: ClaudeReplayEntry[];
};
type SourceKind = 'imported' | 'received';

export interface PreparedClaudeConversation
    extends CanonicalPreparedState<ClaudePrompt>,
        PreparedConversationRequest<MessageCreateParamsBase> {
    provider: string;
    prior_native_message_count: number;
}

function ownValue(value: object, key: string): unknown {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && 'value' in descriptor ? descriptor.value : undefined;
}

function sourceHistoryTurnNumber(history: ClaudePrompt): number | undefined {
    const metadata = ownValue(history, '_llumiverse_meta');
    if (typeof metadata !== 'object' || metadata === null || Array.isArray(metadata)) return undefined;
    const turnNumber = ownValue(metadata, 'turnNumber');
    return typeof turnNumber === 'number' && Number.isSafeInteger(turnNumber) && turnNumber >= 0
        ? turnNumber
        : undefined;
}

function isMessageParam(value: unknown): value is MessageParam {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const role = ownValue(value, 'role');
    const content = ownValue(value, 'content');
    return (
        (role === 'user' || role === 'assistant' || role === 'system') &&
        (typeof content === 'string' || Array.isArray(content))
    );
}

export function isClaudeMessagesHistory(
    value: unknown,
    explicitProtocol?: typeof CLAUDE_MESSAGES_PROTOCOL,
): value is ClaudePrompt | MessageParam[] {
    if (!preflightJsonInput(value).success) return false;
    if (Array.isArray(value)) {
        return explicitProtocol === CLAUDE_MESSAGES_PROTOCOL && value.every(isMessageParam);
    }
    if (typeof value !== 'object' || value === null || ownValue(value, '_is_openai_chat_completions') === true) {
        return false;
    }
    const messages = ownValue(value, 'messages');
    const system = ownValue(value, 'system');
    return (
        Array.isArray(messages) &&
        messages.every(isMessageParam) &&
        (system === undefined || (Array.isArray(system) && system.every((block) => isSupportedTextBlock(block))))
    );
}

function isSupportedTextBlock(value: unknown): value is TextBlockParam {
    return (
        typeof value === 'object' &&
        value !== null &&
        !Array.isArray(value) &&
        ownValue(value, 'type') === 'text' &&
        typeof ownValue(value, 'text') === 'string'
    );
}

async function entityId(kind: string, scope: string, ...position: Array<string | number>): Promise<string> {
    return deriveConversationId(kind, scope, ...position.map(String));
}

function importedProvenance(nativeId: string, sourceHistoryNumber?: number): ImportedTurnProvenance {
    return {
        type: 'imported',
        source: CLAUDE_MESSAGES_PROTOCOL,
        native_id: { protocol: CLAUDE_MESSAGES_PROTOCOL, scope: 'history', value: nativeId },
        ...(sourceHistoryNumber === undefined ? {} : { source_history_turn_number: sourceHistoryNumber }),
        missing_metadata: ['actor_id', 'timestamps', 'exchange'],
    };
}

function turnProvenance(source: SourceKind, nativeId: string, sourceHistoryNumber?: number) {
    return source === 'imported' ? importedProvenance(nativeId, sourceHistoryNumber) : ({ type: 'received' } as const);
}

function assertAllowedKeys(value: object, allowed: readonly string[], label: string): void {
    const extras = Object.keys(value).filter((key) => !allowed.includes(key));
    if (extras.length > 0) throw new TypeError(`${label} contains unsupported field ${extras[0]}`);
}

function assertTextBlock(block: TextBlockParam): void {
    assertAllowedKeys(block, ['type', 'text', 'cache_control', 'citations'], 'Claude text block');
    if (block.citations !== undefined && block.citations !== null && block.citations.length > 0) {
        throw new TypeError('Claude text citations are not supported by the canonical adapter');
    }
}

function assetMetadata(block: ImageBlockParam | DocumentBlockParam): JsonObject | undefined {
    const metadata: JsonObject = {};
    if ('title' in block && block.title !== undefined && block.title !== null) metadata.title = block.title;
    if ('context' in block && block.context !== undefined && block.context !== null) metadata.context = block.context;
    if ('transformations' in block && block.transformations !== undefined && block.transformations !== null) {
        metadata.transformations = providerJsonValue(block.transformations);
    }
    return Object.keys(metadata).length === 0 ? undefined : { claude_messages: metadata };
}

async function assetBlock(input: {
    block: ImageBlockParam | DocumentBlockParam;
    turn_id: string;
    scope: string;
    native_path: string;
    source: SourceKind;
    recorded_at: string;
}): Promise<{ block: UserContentBlock; asset: Asset }> {
    const { block } = input;
    if (block.type === 'image') {
        assertAllowedKeys(block, ['type', 'source', 'cache_control', 'transformations'], 'Claude image block');
    } else {
        assertAllowedKeys(
            block,
            ['type', 'source', 'cache_control', 'citations', 'context', 'title'],
            'Claude document block',
        );
        if (block.citations !== undefined && block.citations !== null) {
            throw new TypeError('Claude document citations are not supported by the canonical adapter');
        }
    }
    const blockId = await entityId('block', input.scope, input.native_path);
    const assetId = await entityId('asset', input.scope, input.native_path);
    const source = block.source;
    let storage: Asset['storage'];
    let mimeType: string;
    if (source.type === 'base64') {
        storage = { type: 'inline_base64', data: source.data };
        mimeType = source.media_type;
    } else if (source.type === 'text') {
        storage = { type: 'inline_text', text: source.data };
        mimeType = source.media_type;
    } else if (source.type === 'url') {
        storage = { type: 'external', resolver: 'url', locator: { url: source.url } };
        mimeType = block.type === 'image' ? 'application/octet-stream' : 'application/pdf';
    } else if (source.type === 'file') {
        storage = { type: 'external', resolver: 'anthropic_file', locator: { file_id: source.file_id } };
        mimeType = 'application/octet-stream';
    } else {
        throw new TypeError(
            `Claude ${block.type} source type ${source.type} is not supported by the canonical adapter`,
        );
    }
    const asset: Asset = {
        id: assetId,
        kind: block.type,
        mime_type: mimeType,
        storage,
        provenance:
            input.source === 'imported'
                ? { type: 'imported', source: CLAUDE_MESSAGES_PROTOCOL }
                : { type: 'received', source_turn_id: input.turn_id },
        content_hash: await fingerprintJson({ mime_type: mimeType, storage }),
        created_at: input.recorded_at,
        ...(assetMetadata(block) === undefined ? {} : { metadata: assetMetadata(block) }),
    };
    return {
        block: {
            id: blockId,
            type: block.type,
            asset_id: assetId,
            ...('title' in block && typeof block.title === 'string' ? { caption: block.title } : {}),
        } as UserContentBlock,
        asset,
    };
}

interface MessageRecords {
    turns: ConversationTurn[];
    assets: Asset[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}

function isProgramBlock(block: AgentContentBlock): block is ProgramContentBlock {
    return block.type !== 'tool_call';
}

function isUserBlock(block: AgentContentBlock): block is UserContentBlock {
    return block.type !== 'tool_call' && block.type !== 'reasoning' && block.type !== 'native_replay';
}

function isToolResultContent(block: AgentContentBlock): block is NestedToolResultContentBlock {
    return block.type !== 'tool_call';
}

async function canonicalBlock(input: {
    native: ContentBlockParam;
    turn_id: string;
    scope: string;
    native_path: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    tool_definitions: readonly ToolDefinition[];
    provider: string;
}): Promise<{ block?: AgentContentBlock; asset?: Asset; replay_entry?: ClaudeReplayEntry }> {
    const { native } = input;
    if (native.type === 'text') {
        assertTextBlock(native);
        const block: AgentContentBlock = {
            id: await entityId('block', input.scope, input.native_path),
            type: 'text',
            text: native.text,
            format: 'plain',
        };
        return { block, replay_entry: { kind: 'canonical', block_id: block.id } };
    }
    if (native.type === 'image' || native.type === 'document') {
        const converted = await assetBlock({
            block: native,
            turn_id: input.turn_id,
            scope: input.scope,
            native_path: input.native_path,
            source: input.source,
            recorded_at: input.runtime.recorded_at,
        });
        return {
            block: converted.block,
            asset: converted.asset,
            replay_entry: { kind: 'canonical', block_id: converted.block.id },
        };
    }
    if (native.type === 'thinking') {
        assertAllowedKeys(native, ['type', 'thinking', 'signature'], 'Claude thinking block');
        const block: AgentContentBlock = {
            id: await entityId('reasoning', input.scope, input.native_path),
            type: 'reasoning',
            text: native.thinking,
            representation: 'text',
        };
        return { block, replay_entry: { kind: 'thinking', block_id: block.id, signature: native.signature } };
    }
    if (native.type === 'redacted_thinking') {
        assertAllowedKeys(native, ['type', 'data'], 'Claude redacted thinking block');
        return { replay_entry: { kind: 'redacted_thinking', data: native.data } };
    }
    if (native.type === 'tool_use') {
        assertAllowedKeys(native, ['type', 'id', 'name', 'input', 'cache_control'], 'Claude tool_use block');
        const definition = input.tool_definitions.find((candidate) => candidate.name === native.name);
        const block: AgentContentBlock = {
            id: await entityId('block', input.scope, input.native_path),
            type: 'tool_call',
            call_id: native.id,
            tool_name: native.name,
            ...(definition === undefined ? {} : { definition_id: definition.id }),
            executor: 'application',
            arguments: { type: 'json', value: providerJsonValue(native.input) },
            native_id: { protocol: CLAUDE_MESSAGES_PROTOCOL, scope: input.runtime.request_id, value: native.id },
        };
        return { block, replay_entry: { kind: 'canonical', block_id: block.id } };
    }
    throw new TypeError(`Claude content block type ${native.type} is not supported by the canonical adapter`);
}

function normalizedContent(content: MessageParam['content']): ContentBlockParam[] {
    return typeof content === 'string' ? [{ type: 'text', text: content }] : content;
}

async function toolResultRecord(input: {
    native: CanonicalClaudeToolResultBlockParam;
    turn_id: string;
    scope: string;
    native_path: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    tool_definitions: readonly ToolDefinition[];
    provider: string;
    provenance: ConversationTurn['provenance'];
}): Promise<{ turn: ConversationTurn; assets: Asset[]; mappings: NativeItemMapping[]; receipt?: ExecutionReceipt }> {
    assertAllowedKeys(
        input.native,
        ['type', 'tool_use_id', 'content', 'is_error', 'cache_control', '_llumiverse_tool_result_status'],
        'Claude tool_result block',
    );
    const explicitStatus = input.native._llumiverse_tool_result_status;
    if (explicitStatus === 'success' && input.native.is_error === true) {
        throw new Error(
            `Claude tool result ${input.native.tool_use_id} has contradictory success and is_error evidence`,
        );
    }
    if (explicitStatus === 'error' && input.native.is_error === false) {
        throw new Error(`Claude tool result ${input.native.tool_use_id} has contradictory error and is_error evidence`);
    }
    const status: ToolResultBlock['status'] =
        explicitStatus ??
        (input.native.is_error === true ? 'error' : input.source === 'imported' ? 'unknown' : 'success');
    const nestedNative =
        typeof input.native.content === 'string'
            ? ([{ type: 'text', text: input.native.content }] satisfies TextBlockParam[])
            : (input.native.content ?? []);
    const content: NestedToolResultContentBlock[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    for (let index = 0; index < nestedNative.length; index += 1) {
        const native = nestedNative[index];
        if (native.type !== 'text' && native.type !== 'image' && native.type !== 'document') {
            throw new TypeError(
                `Claude tool_result content type ${native.type} is not supported by the canonical adapter`,
            );
        }
        const converted = await canonicalBlock({
            native,
            turn_id: input.turn_id,
            scope: input.scope,
            native_path: `${input.native_path}/content/${index}`,
            source: input.source,
            runtime: input.runtime,
            tool_definitions: input.tool_definitions,
            provider: input.provider,
        });
        if (converted.block !== undefined && isToolResultContent(converted.block)) {
            content.push(converted.block);
            mappings.push({
                canonical_id: converted.block.id,
                native_id: `${input.native_path}/content/${index}`,
                kind: 'block',
            });
        }
        if (converted.asset !== undefined) assets.push(converted.asset);
    }
    const resultBlock: ToolResultBlock = {
        id: await entityId('block', input.scope, input.native_path),
        type: 'tool_result',
        call_id: input.native.tool_use_id,
        status,
        content,
        native_id: {
            protocol: CLAUDE_MESSAGES_PROTOCOL,
            scope: input.runtime.request_id,
            value: input.native.tool_use_id,
        },
    };
    mappings.unshift(
        { canonical_id: resultBlock.id, native_id: input.native_path, kind: 'block' },
        { canonical_id: input.native.tool_use_id, native_id: input.native.tool_use_id, kind: 'call' },
    );
    const turn: ConversationTurn = {
        id: input.turn_id,
        kind: 'tool',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: input.runtime.recorded_at },
        model_visibility: 'include',
        provenance: input.provenance as Exclude<ConversationTurn, { kind: 'agent' }>['provenance'],
        blocks: [resultBlock],
    };
    const receipt =
        input.source === 'received' && status !== 'unknown'
            ? {
                  id: await entityId('execution_receipt', input.scope, input.native.tool_use_id),
                  call_id: input.native.tool_use_id,
                  executor: 'application' as const,
                  status,
                  result_turn_id: turn.id,
                  result_fingerprint: await fingerprintJson(resultBlock),
                  recorded_at: input.runtime.recorded_at,
              }
            : undefined;
    return { turn, assets, mappings, ...(receipt === undefined ? {} : { receipt }) };
}

async function assistantMessageRecords(input: {
    message: MessageParam;
    message_index: number;
    scope: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    tool_definitions: readonly ToolDefinition[];
    provider: string;
    source_history_turn_number?: number;
}): Promise<MessageRecords> {
    const nativePath = `messages/${input.message_index}`;
    const turnId = await entityId('turn', input.scope, nativePath);
    const blocks: AgentContentBlock[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [{ canonical_id: turnId, native_id: nativePath, kind: 'turn' }];
    const entries: ClaudeReplayEntry[] = [];
    const content = normalizedContent(input.message.content);
    for (let index = 0; index < content.length; index += 1) {
        const native = content[index];
        if (native.type === 'tool_result') {
            throw new TypeError('Claude assistant messages cannot contain tool_result blocks');
        }
        const converted = await canonicalBlock({
            native,
            turn_id: turnId,
            scope: input.scope,
            native_path: `${nativePath}/content/${index}`,
            source: input.source,
            runtime: input.runtime,
            tool_definitions: input.tool_definitions,
            provider: input.provider,
        });
        if (converted.block !== undefined) {
            blocks.push(converted.block);
            mappings.push({
                canonical_id: converted.block.id,
                native_id: `${nativePath}/content/${index}`,
                kind: 'block',
            });
            if (converted.block.type === 'tool_call') {
                mappings.push({
                    canonical_id: converted.block.call_id,
                    native_id: converted.block.call_id,
                    kind: 'call',
                });
            }
        }
        if (converted.asset !== undefined) assets.push(converted.asset);
        if (converted.replay_entry !== undefined) entries.push(converted.replay_entry);
    }
    if (entries.some((entry) => entry.kind !== 'canonical')) {
        const replayId = await entityId('replay', input.scope, nativePath);
        blocks.push({
            id: replayId,
            type: 'native_replay',
            adapter: CLAUDE_MESSAGES_ADAPTER_VERSION,
            protocol: CLAUDE_MESSAGES_PROTOCOL,
            compatibility_scope: {
                provider: input.provider,
                protocol: CLAUDE_MESSAGES_PROTOCOL,
                adapter_version: CLAUDE_MESSAGES_ADAPTER_VERSION,
            },
            payload: { type: 'claude_messages_content_order', entries },
            dependencies: {
                turn_ids: [turnId],
                block_ids: blocks.map((block) => block.id),
                call_ids: blocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : [])),
                request_ids: [],
            },
        });
        mappings.push({ canonical_id: replayId, native_id: `${nativePath}/content_order`, kind: 'block' });
    }
    const provenance = turnProvenance(input.source, nativePath, input.source_history_turn_number);
    const turn: ConversationTurn =
        input.source === 'imported'
            ? {
                  id: turnId,
                  kind: 'agent',
                  authority: 'ordinary',
                  status: 'completed',
                  timestamps: { recorded_at: input.runtime.recorded_at },
                  model_visibility: 'include',
                  provenance: provenance as ImportedTurnProvenance,
                  blocks,
              }
            : {
                  id: turnId,
                  kind: 'agent',
                  authority: 'ordinary',
                  status: 'completed',
                  timestamps: { recorded_at: input.runtime.recorded_at },
                  model_visibility: 'include',
                  provenance: { type: 'received' },
                  blocks,
              };
    return { turns: [turn], assets, mappings, execution_receipts: [] };
}

async function nonAssistantMessageRecords(input: {
    message: MessageParam;
    message_index: number;
    native_path?: string;
    scope: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    tool_definitions: readonly ToolDefinition[];
    provider: string;
    source_history_turn_number?: number;
}): Promise<MessageRecords> {
    const basePath = input.native_path ?? `messages/${input.message_index}`;
    const content = normalizedContent(input.message.content);
    const records: MessageRecords = { turns: [], assets: [], mappings: [], execution_receipts: [] };
    let index = 0;
    while (index < content.length) {
        const native = content[index];
        const nativePath = `${basePath}/content/${index}`;
        const provenance = turnProvenance(input.source, basePath, input.source_history_turn_number);
        if (native.type === 'tool_result') {
            if (input.message.role !== 'user') {
                throw new TypeError('Claude tool_result blocks require a user message');
            }
            const turnId = await entityId('turn', input.scope, nativePath);
            const converted = await toolResultRecord({
                native: native as CanonicalClaudeToolResultBlockParam,
                turn_id: turnId,
                scope: input.scope,
                native_path: nativePath,
                source: input.source,
                runtime: input.runtime,
                tool_definitions: input.tool_definitions,
                provider: input.provider,
                provenance,
            });
            records.turns.push(converted.turn);
            records.assets.push(...converted.assets);
            records.mappings.push({ canonical_id: turnId, native_id: basePath, kind: 'turn' }, ...converted.mappings);
            if (converted.receipt !== undefined) records.execution_receipts.push(converted.receipt);
            index += 1;
            continue;
        }

        const segmentStart = index;
        while (index < content.length && content[index].type !== 'tool_result') index += 1;
        const turnId = await entityId(
            'turn',
            input.scope,
            content.length === index - segmentStart ? basePath : `${basePath}/segment/${segmentStart}`,
        );
        const blocks: Array<ProgramContentBlock | UserContentBlock> = [];
        for (let blockIndex = segmentStart; blockIndex < index; blockIndex += 1) {
            const ordinary = content[blockIndex];
            if (ordinary.type === 'tool_use' || ordinary.type === 'thinking' || ordinary.type === 'redacted_thinking') {
                throw new TypeError(`Claude ${ordinary.type} blocks require an assistant message`);
            }
            const ordinaryPath = `${basePath}/content/${blockIndex}`;
            const converted = await canonicalBlock({
                native: ordinary,
                turn_id: turnId,
                scope: input.scope,
                native_path: ordinaryPath,
                source: input.source,
                runtime: input.runtime,
                tool_definitions: input.tool_definitions,
                provider: input.provider,
            });
            if (converted.block === undefined) continue;
            if (input.message.role === 'system') {
                if (!isProgramBlock(converted.block) || converted.block.type !== 'text') {
                    throw new TypeError('Claude system content only supports text blocks');
                }
                blocks.push(converted.block);
            } else if (isUserBlock(converted.block)) {
                blocks.push(converted.block);
            }
            if (converted.asset !== undefined) records.assets.push(converted.asset);
            records.mappings.push({
                canonical_id: converted.block.id,
                native_id: ordinaryPath,
                kind: 'block',
            });
        }
        const common = {
            id: turnId,
            status: 'completed' as const,
            timestamps: { recorded_at: input.runtime.recorded_at },
            model_visibility: 'include' as const,
            provenance: provenance as Exclude<ConversationTurn, { kind: 'agent' }>['provenance'],
        };
        const turn: ConversationTurn =
            input.message.role === 'system'
                ? {
                      ...common,
                      kind: 'program',
                      authority: 'system',
                      blocks: blocks as ProgramContentBlock[],
                  }
                : {
                      ...common,
                      kind: 'user',
                      authority: 'ordinary',
                      blocks: blocks as UserContentBlock[],
                  };
        records.turns.push(turn);
        records.mappings.push({ canonical_id: turnId, native_id: basePath, kind: 'turn' });
    }
    return records;
}

async function messageRecords(input: {
    message: MessageParam;
    message_index: number;
    native_path?: string;
    scope: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    tool_definitions: readonly ToolDefinition[];
    provider: string;
    source_history_turn_number?: number;
}): Promise<MessageRecords> {
    return input.message.role === 'assistant' ? assistantMessageRecords(input) : nonAssistantMessageRecords(input);
}

async function promptRecords(input: {
    prompt: ClaudePrompt;
    scope: string;
    source: SourceKind;
    runtime: ResolvedConversationRuntimeContext;
    tool_definitions: readonly ToolDefinition[];
    provider: string;
    source_history_turn_number?: number;
}): Promise<MessageRecords> {
    const records: MessageRecords = { turns: [], assets: [], mappings: [], execution_receipts: [] };
    const append = (converted: MessageRecords) => {
        records.turns.push(...converted.turns);
        records.assets.push(...converted.assets);
        records.mappings.push(...converted.mappings);
        records.execution_receipts.push(...converted.execution_receipts);
    };
    for (let index = 0; index < (input.prompt.system?.length ?? 0); index += 1) {
        const block = input.prompt.system?.[index];
        if (block === undefined) continue;
        append(
            await messageRecords({
                message: { role: 'system', content: [block] },
                message_index: index,
                native_path: `system/${index}`,
                scope: input.scope,
                source: input.source,
                runtime: input.runtime,
                tool_definitions: input.tool_definitions,
                provider: input.provider,
                ...(input.source_history_turn_number === undefined
                    ? {}
                    : { source_history_turn_number: input.source_history_turn_number }),
            }),
        );
    }
    for (let index = 0; index < input.prompt.messages.length; index += 1) {
        append(
            await messageRecords({
                message: input.prompt.messages[index],
                message_index: index,
                scope: input.scope,
                source: input.source,
                runtime: input.runtime,
                tool_definitions: input.tool_definitions,
                provider: input.provider,
                ...(input.source_history_turn_number === undefined
                    ? {}
                    : { source_history_turn_number: input.source_history_turn_number }),
            }),
        );
    }
    return records;
}

function claudeAssetMetadata(asset: Asset): JsonObject {
    const value = asset.metadata?.claude_messages;
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? value : {};
}

function assetToClaudeBlock(asset: Asset): ImageBlockParam | DocumentBlockParam {
    const metadata = claudeAssetMetadata(asset);
    if (asset.kind === 'image') {
        let source: ImageBlockParam['source'];
        if (asset.storage.type === 'inline_base64') {
            if (!['image/png', 'image/jpeg', 'image/gif', 'image/webp'].includes(asset.mime_type)) {
                throw new TypeError(`Claude cannot project image MIME type ${asset.mime_type}`);
            }
            source = {
                type: 'base64',
                media_type: asset.mime_type as 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp',
                data: asset.storage.data,
            };
        } else if (asset.storage.type === 'external' && asset.storage.resolver === 'url') {
            const url = asset.storage.locator.url;
            if (typeof url !== 'string') throw new TypeError(`Claude image asset ${asset.id} has no URL`);
            source = { type: 'url', url };
        } else if (asset.storage.type === 'external' && asset.storage.resolver === 'anthropic_file') {
            const fileId = asset.storage.locator.file_id;
            if (typeof fileId !== 'string') throw new TypeError(`Claude image asset ${asset.id} has no file ID`);
            source = { type: 'file', file_id: fileId };
        } else {
            throw new TypeError(`Claude cannot project image asset ${asset.id}`);
        }
        return {
            type: 'image',
            source,
            ...(typeof metadata.transformations === 'object' && metadata.transformations !== null
                ? { transformations: metadata.transformations as ImageBlockParam['transformations'] }
                : {}),
        };
    }
    if (asset.kind !== 'document') throw new TypeError(`Claude cannot project ${asset.kind} asset ${asset.id}`);
    let source: DocumentBlockParam['source'];
    if (asset.storage.type === 'inline_base64') {
        if (asset.mime_type !== 'application/pdf') {
            throw new TypeError(`Claude cannot project base64 document MIME type ${asset.mime_type}`);
        }
        source = { type: 'base64', media_type: 'application/pdf', data: asset.storage.data };
    } else if (asset.storage.type === 'inline_text') {
        source = { type: 'text', media_type: 'text/plain', data: asset.storage.text };
    } else if (asset.storage.type === 'external' && asset.storage.resolver === 'url') {
        const url = asset.storage.locator.url;
        if (typeof url !== 'string') throw new TypeError(`Claude document asset ${asset.id} has no URL`);
        source = { type: 'url', url };
    } else if (asset.storage.type === 'external' && asset.storage.resolver === 'anthropic_file') {
        const fileId = asset.storage.locator.file_id;
        if (typeof fileId !== 'string') throw new TypeError(`Claude document asset ${asset.id} has no file ID`);
        source = { type: 'file', file_id: fileId };
    } else {
        throw new TypeError(`Claude cannot project document asset ${asset.id}`);
    }
    return {
        type: 'document',
        source,
        ...(typeof metadata.title === 'string' ? { title: metadata.title } : {}),
        ...(typeof metadata.context === 'string' ? { context: metadata.context } : {}),
    };
}

function canonicalBlockToClaude(block: ContentBlock, document: ConversationDocument): ContentBlockParam {
    if (block.type === 'text') return { type: 'text', text: block.text };
    if (block.type === 'json') return { type: 'text', text: JSON.stringify(block.value) };
    if (block.type === 'image' || block.type === 'document') {
        const asset = document.assets[block.asset_id];
        if (asset === undefined) throw new Error(`Claude content references missing asset ${block.asset_id}`);
        return assetToClaudeBlock(asset);
    }
    if (block.type === 'tool_call') {
        if (block.arguments.type === 'invalid') {
            throw new TypeError(`Claude cannot replay invalid JSON arguments for call ${block.call_id}`);
        }
        return { type: 'tool_use', id: block.call_id, name: block.tool_name, input: block.arguments.value };
    }
    throw new TypeError(`Claude cannot project canonical ${block.type} block`);
}

type ClaudeToolResultContentBlock = Extract<NonNullable<ToolResultBlockParam['content']>, unknown[]>[number];

function canonicalToolResultContentToClaude(
    block: NestedToolResultContentBlock,
    document: ConversationDocument,
): ClaudeToolResultContentBlock {
    if (block.type !== 'text' && block.type !== 'image' && block.type !== 'document') {
        throw new TypeError(`Claude cannot project canonical ${block.type} inside a tool result`);
    }
    return canonicalBlockToClaude(block, document) as ClaudeToolResultContentBlock;
}

function claudeReplay(
    turn: ConversationTurn,
    target?: { provider?: string; model?: string },
): ClaudeReplayPayload | undefined {
    const replayBlocks = turn.blocks.filter((block) => block.type === 'native_replay');
    const replay = replayBlocks.find((block) => block.protocol === CLAUDE_MESSAGES_PROTOCOL);
    const foreign = replayBlocks.find((block) => block.protocol !== CLAUDE_MESSAGES_PROTOCOL);
    if (foreign !== undefined) {
        throw new TypeError(`Claude cannot discard protected ${foreign.protocol} replay block ${foreign.id}`);
    }
    if (replay === undefined) return undefined;
    if (
        typeof replay.payload !== 'object' ||
        replay.payload === null ||
        Array.isArray(replay.payload) ||
        replay.payload.type !== 'claude_messages_content_order' ||
        !Array.isArray(replay.payload.entries)
    ) {
        throw new TypeError(`Claude replay block ${replay.id} has an unsupported payload`);
    }
    if (
        replay.adapter !== CLAUDE_MESSAGES_ADAPTER_VERSION ||
        replay.compatibility_scope.protocol !== CLAUDE_MESSAGES_PROTOCOL ||
        replay.compatibility_scope.adapter_version !== CLAUDE_MESSAGES_ADAPTER_VERSION ||
        (target?.provider !== undefined && replay.compatibility_scope.provider !== target.provider) ||
        (target?.model !== undefined &&
            replay.compatibility_scope.model !== undefined &&
            replay.compatibility_scope.model !== target.model)
    ) {
        throw new TypeError(`Claude replay block ${replay.id} is outside its compatibility scope`);
    }
    return replay.payload as ClaudeReplayPayload;
}

function agentContent(
    turn: ConversationTurn,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): ContentBlockParam[] {
    const replay = claudeReplay(turn, target);
    const blocks = new Map(turn.blocks.map((block) => [block.id, block]));
    if (replay !== undefined) {
        return replay.entries.map((entry): ContentBlockParam => {
            if (entry.kind === 'redacted_thinking') return { type: 'redacted_thinking', data: entry.data };
            const block = blocks.get(entry.block_id);
            if (block === undefined) throw new Error(`Claude replay references missing block ${entry.block_id}`);
            if (entry.kind === 'thinking') {
                if (block.type !== 'reasoning') {
                    throw new Error(`Claude thinking replay ${entry.block_id} does not reference reasoning`);
                }
                return { type: 'thinking', thinking: block.text, signature: entry.signature };
            }
            if (block.type === 'native_replay' || block.type === 'reasoning') {
                throw new Error(`Claude canonical replay entry ${entry.block_id} has incompatible block type`);
            }
            return canonicalBlockToClaude(block, document);
        });
    }
    return turn.blocks.flatMap((block): ContentBlockParam[] => {
        if (block.type === 'native_replay') return [];
        if (block.type === 'reasoning') {
            throw new TypeError(`Claude reasoning block ${block.id} has no signed native replay evidence`);
        }
        return [canonicalBlockToClaude(block, document)];
    });
}

function ordinaryContent(turn: ConversationTurn, document: ConversationDocument): ContentBlockParam[] {
    if (turn.kind === 'tool') {
        const result = turn.blocks[0];
        const content = result.content.map((block) => canonicalToolResultContentToClaude(block, document));
        return [
            {
                type: 'tool_result',
                tool_use_id: result.call_id,
                content,
                ...(result.status === 'success' || result.status === 'unknown' ? {} : { is_error: true }),
            },
        ];
    }
    return turn.blocks.flatMap((block): ContentBlockParam[] => {
        if (block.type === 'extension' && block.model_projection === 'excluded') return [];
        if (block.type === 'extension') {
            throw new TypeError(`Claude has no registered projection for extension block ${block.id}`);
        }
        if (block.type === 'external_reference' || block.type === 'audio' || block.type === 'video') {
            throw new TypeError(`Claude cannot project canonical ${block.type} block ${block.id}`);
        }
        return [canonicalBlockToClaude(block, document)];
    });
}

function appendClaudeMessage(messages: MessageParam[], message: MessageParam): void {
    const prior = messages.at(-1);
    if (prior?.role === 'user' && message.role === 'user') {
        const priorContent = normalizedContent(prior.content);
        prior.content = [...priorContent, ...normalizedContent(message.content)];
        return;
    }
    messages.push(message);
}

export function compileClaudeMessagesConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): {
    conversation: ClaudePrompt;
    mappings: NativeItemMapping[];
} {
    const system: TextBlockParam[] = [];
    const messages: MessageParam[] = [];
    const mappings: NativeItemMapping[] = [];
    for (const turn of selectedCanonicalTurns(document)) {
        if (turn.kind === 'program' && (turn.authority === 'system' || turn.authority === 'developer')) {
            const nativeBlocks = ordinaryContent(turn, document);
            for (const block of nativeBlocks) {
                if (block.type !== 'text') throw new TypeError('Claude system context only supports text blocks');
                const index = system.length;
                system.push(block);
                mappings.push({ canonical_id: turn.id, native_id: `system/${index}`, kind: 'turn' });
            }
            continue;
        }
        const content = turn.kind === 'agent' ? agentContent(turn, document, target) : ordinaryContent(turn, document);
        const role = turn.kind === 'agent' ? 'assistant' : 'user';
        const messageIndex = messages.length;
        appendClaudeMessage(messages, { role, content });
        const mappedIndex = messages.length === messageIndex ? Math.max(0, messageIndex - 1) : messageIndex;
        mappings.push({ canonical_id: turn.id, native_id: `messages/${mappedIndex}`, kind: 'turn' });
        for (const block of turn.blocks) {
            mappings.push({
                canonical_id: block.id,
                native_id: `messages/${mappedIndex}/blocks/${block.id}`,
                kind: 'block',
            });
            if (block.type === 'tool_call') {
                mappings.push({ canonical_id: block.call_id, native_id: block.call_id, kind: 'call' });
            }
        }
    }
    return {
        conversation: { messages, ...(system.length === 0 ? {} : { system }) },
        mappings,
    };
}

export async function prepareClaudeCanonicalState(input: {
    conversation: unknown;
    prompt: ClaudePrompt;
    options: ExecutionOptions;
    provider: string;
}): Promise<Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>> {
    const runtime = resolveConversationRuntime(input.options);
    const toolDefinitions = await canonicalToolDefinitions(input.options.tools);
    let document = parseCanonicalConversation(input.conversation);
    if (document === undefined) {
        document = newCanonicalConversation(runtime);
        if (input.conversation !== undefined && input.conversation !== null) {
            if (!isClaudeMessagesHistory(input.conversation)) {
                throw new TypeError('Conversation is neither canonical nor registered Claude Messages history');
            }
            const history = input.conversation as ClaudePrompt;
            const imported = await promptRecords({
                prompt: history,
                scope: `${runtime.conversation_id}:legacy`,
                source: 'imported',
                runtime,
                tool_definitions: toolDefinitions,
                provider: input.provider,
                ...(sourceHistoryTurnNumber(history) === undefined
                    ? {}
                    : { source_history_turn_number: sourceHistoryTurnNumber(history) }),
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
                },
                {
                    expected_revision: document.revision,
                    operation_id: await entityId('import', runtime.conversation_id, CLAUDE_MESSAGES_PROTOCOL),
                    payload_fingerprint: await fingerprintJson(providerJsonValue(history)),
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
    const priorNativeMessageCount = compileClaudeMessagesConversation(document, target).conversation.messages.length;
    const received = await promptRecords({
        prompt: input.prompt,
        scope: runtime.input_operation_id,
        source: 'received',
        runtime: { ...runtime, conversation_id: document.id },
        tool_definitions: toolDefinitions,
        provider: input.provider,
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
        {
            ...received,
            context_entries: contextEntries,
            item_mappings: received.mappings,
        },
        { ...runtime, conversation_id: document.id },
        input.options.tools,
        providerJsonValue(input.prompt),
    );
    const compiled = compileClaudeMessagesConversation(appended.document, target);
    const acceptedResponse = acceptedCanonicalResponse(appended.document, runtime.response_operation_id);
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
        prior_native_message_count: priorNativeMessageCount,
        ...(acceptedResponse === undefined ? {} : { accepted_response: acceptedResponse }),
    };
}

export async function finalizeClaudePreparedRequest(
    state: Omit<PreparedClaudeConversation, 'payload' | 'receipt' | 'diagnostics'>,
    payload: MessageCreateParamsBase,
): Promise<PreparedClaudeConversation> {
    const compiled = compileClaudeMessagesConversation(state.document, {
        provider: state.provider,
        model: payload.model,
    });
    const receipt = await createRequestReceipt(
        state.document,
        state.runtime,
        {
            provider: state.provider,
            protocol: CLAUDE_MESSAGES_PROTOCOL,
            model: payload.model,
            adapter_version: CLAUDE_MESSAGES_ADAPTER_VERSION,
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

function safeSum(...values: number[]): number | undefined {
    const total = values.reduce((sum, value) => sum + value, 0);
    return Number.isSafeInteger(total) ? total : undefined;
}

function claudeUsage(native: AnthropicUsageLike | undefined): GenerationUsage | undefined {
    if (native === undefined) return undefined;
    const inputNew = safeUsageNumber(native.input_tokens);
    const output = safeUsageNumber(native.output_tokens);
    const cacheRead = safeUsageNumber(native.cache_read_input_tokens);
    const cacheWrite = safeUsageNumber(native.cache_creation_input_tokens);
    const input =
        inputNew !== undefined && cacheRead !== undefined && cacheWrite !== undefined
            ? safeSum(inputNew, cacheRead, cacheWrite)
            : undefined;
    const total = input !== undefined && output !== undefined ? safeSum(input, output) : undefined;
    const basis = 'anthropic_messages_tokens';
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(total === undefined ? {} : { total_tokens: total }),
        ...(cacheRead === undefined ? {} : { cache_read_tokens: cacheRead }),
        ...(cacheWrite === undefined ? {} : { cache_write_tokens: cacheWrite }),
        ...(inputNew === undefined ? {} : { input_new_tokens: inputNew }),
        accounting_provenance: {
            ...(input === undefined ? {} : { input_tokens: { method: 'derived' as const, accounting_basis: basis } }),
            ...(output === undefined
                ? {}
                : { output_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(total === undefined ? {} : { total_tokens: { method: 'derived' as const, accounting_basis: basis } }),
            ...(cacheRead === undefined
                ? {}
                : { cache_read_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(cacheWrite === undefined
                ? {}
                : { cache_write_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(inputNew === undefined
                ? {}
                : { input_new_tokens: { method: 'reported' as const, accounting_basis: basis } }),
        },
        ...(input === undefined
            ? {}
            : { input_partition: { type: 'complete_disjoint', cache_write_bucket: 'included' } }),
        reported_usage: [
            {
                source: 'provider',
                protocol: CLAUDE_MESSAGES_PROTOCOL,
                accounting_basis: basis,
                payload: providerJsonValue(native),
            },
        ],
    };
}

export async function decodeClaudeCanonicalResponse(
    response: Message,
    prepared: PreparedClaudeConversation,
): Promise<DecodedConversationResponse> {
    if (response.stop_reason === null) {
        throw new Error('Claude Messages response ended without a terminal stop reason');
    }
    const completedAt = prepared.runtime.completed_at ?? new Date().toISOString();
    const runtime = { ...prepared.runtime, recorded_at: completedAt };
    const records = await assistantMessageRecords({
        message: { role: 'assistant', content: providerJsonValue(response.content) as unknown as ContentBlockParam[] },
        message_index: 0,
        scope: prepared.runtime.response_operation_id,
        source: 'received',
        runtime,
        tool_definitions: prepared.tool_definitions,
        provider: prepared.provider,
    });
    const received = records.turns[0];
    if (received?.kind !== 'agent') throw new Error('Claude Messages response did not decode to an agent turn');
    const status =
        response.stop_reason === 'max_tokens' || response.stop_reason === 'model_context_window_exceeded'
            ? 'interrupted'
            : 'completed';
    const turn: ConversationTurn = {
        ...received,
        id: prepared.response_turn_id,
        status,
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
    const generation: ExecutedGeneration = await createExecutedGeneration({
        id: prepared.generation_id,
        runtime: prepared.runtime,
        receipt: prepared.receipt,
        provider: prepared.provider,
        protocol: CLAUDE_MESSAGES_PROTOCOL,
        adapter_version: CLAUDE_MESSAGES_ADAPTER_VERSION,
        requested_model: prepared.payload.model,
        resolved_model: response.model,
        provider_response_id: response.id,
        finish_reason: response.stop_reason,
        usage: claudeUsage(response.usage),
    });
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

export function appendClaudeCanonicalResponse(
    prepared: PreparedClaudeConversation,
    decoded: DecodedConversationResponse,
): ConversationDocument {
    return appendDecodedConversationResponse(prepared, decoded, {
        operation_id: prepared.runtime.response_operation_id,
        recorded_at: decoded.generation.timestamps.recorded_at,
    }).document;
}

/** Read-only compatibility projection for versioned legacy API responses. */
export function exportLegacyClaudeMessagesConversation(document: ConversationDocument): ClaudePrompt {
    return compileClaudeMessagesConversation(parseConversationDocument(document)).conversation;
}
