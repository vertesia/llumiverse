/** Direct canonical adapter for AWS Bedrock Converse history, media, reasoning, and application tools. */
import type {
    AudioBlock,
    ContentBlock,
    ConverseRequest,
    ConverseResponse,
    DocumentBlock,
    ImageBlock,
    Message,
    SystemContentBlock,
    TokenUsage,
    ToolResultContentBlock,
    VideoBlock,
} from '@aws-sdk/client-bedrock-runtime';
import type { ExecutionOptions } from '@llumiverse/common';
import {
    type AgentContentBlock,
    type Asset,
    appendConversationRecords,
    type ContextEntry,
    type ConversationDocument,
    type ConversationTurn,
    copyNativeConversationByteView,
    createConversationDocument,
    type DecodedConversationResponse,
    deriveConversationId,
    type ExecutedGeneration,
    type ExecutionReceipt,
    fingerprintJson,
    type GenerationUsage,
    type ImportedTurnProvenance,
    inlineAssetContentIntegrity,
    type JsonValue,
    type NativeItemMapping,
    type NativeReplayBlock,
    type NestedToolResultContentBlock,
    type PreparedConversationRequest,
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
    resolveCanonicalToolDefinitions,
    resolveConversationRuntime,
    selectedCanonicalTurns,
} from '../conversation/canonical-runtime.js';
import {
    assertNativeImportInputBounds,
    fingerprintNativeConversationImport,
    guardNativeConversationImport,
    type NativeConversationImportOptions,
    type NativeConversationImportResult,
    nativeConversationImportResult,
    newNativeImportDocument,
    snapshotNativeConversationImportOptions,
} from '../conversation/native-import.js';
import {
    assertStructuredOutputEvidence,
    type CanonicalStructuredOutputEvidence,
    normalizeDecodedStructuredOutput,
    parseStructuredOutputEvidence,
    remapStructuredOutputReplayDependencies,
    structuredOutputEvidence,
} from '../conversation/structured-output.js';

export const BEDROCK_CONVERSE_PROTOCOL = 'aws.bedrock.converse' as const;
export const BEDROCK_CONVERSE_ADAPTER_VERSION = '2026-09-30.adoption.1' as const;

export type BedrockConverseConversation = Pick<ConverseRequest, 'messages' | 'system'>;

export interface CompiledBedrockConversation {
    conversation: BedrockConverseConversation;
    mappings: NativeItemMapping[];
}

export interface ImportBedrockConverseConversationOptions {
    conversation_id: string;
    recorded_at: string;
    source_history_turn_number?: number;
    tool_definitions?: readonly ToolDefinition[];
    provider?: string;
    model?: string;
}

interface ImportRecords {
    turns: ConversationTurn[];
    assets: Asset[];
    context_entries: ContextEntry[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}

type SourceKind = 'imported' | 'received';
type BedrockNativeToolResult = NonNullable<ContentBlock.ToolResultMember['toolResult']> & {
    _llumiverse_tool_result_status?: ToolResultBlock['status'];
};

export interface PreparedBedrockConverseConversation
    extends CanonicalPreparedState<BedrockConverseConversation>,
        PreparedConversationRequest<ConverseRequest> {
    provider: string;
    requested_model: string;
    prior_native_message_count: number;
}

interface BedrockReplayCanonicalEntry {
    kind: 'canonical';
    block_id: string;
    native: JsonValue;
}

interface BedrockReplayReasoningEntry {
    kind: 'reasoning';
    block_id: string;
    signature?: string;
    native: JsonValue;
}

interface BedrockReplayRedactedEntry {
    kind: 'redacted_reasoning';
    data: string;
    native: JsonValue;
}

interface BedrockReplayStructuredJsonEntry {
    kind: 'structured_json_fragment';
    block_id: string;
    native: JsonValue;
}

type BedrockReplayEntry =
    | BedrockReplayCanonicalEntry
    | BedrockReplayReasoningEntry
    | BedrockReplayRedactedEntry
    | BedrockReplayStructuredJsonEntry;
type BedrockReplayPayload = {
    type: 'bedrock_converse_content_order';
    prefix?: JsonValue;
    entries: BedrockReplayEntry[];
    structured_output?: CanonicalStructuredOutputEvidence;
};

interface BedrockReplayDependencies {
    turn_ids: string[];
    block_ids: string[];
    call_ids: string[];
}

function replayDependenciesForTurns(turns: readonly ConversationTurn[]): BedrockReplayDependencies {
    const blockIds: string[] = [];
    const callIds: string[] = [];
    for (const turn of turns) {
        for (const block of turn.blocks) {
            blockIds.push(block.id);
            if (block.type === 'tool_call') callIds.push(block.call_id);
            if (block.type === 'tool_result') blockIds.push(...block.content.map((nested) => nested.id));
        }
    }
    return { turn_ids: turns.map((turn) => turn.id), block_ids: blockIds, call_ids: callIds };
}

function replayDependenciesForMappings(mappings: readonly NativeItemMapping[]): BedrockReplayDependencies {
    return {
        turn_ids: mappings.filter((mapping) => mapping.kind === 'turn').map((mapping) => mapping.canonical_id),
        block_ids: mappings.filter((mapping) => mapping.kind === 'block').map((mapping) => mapping.canonical_id),
        call_ids: mappings.filter((mapping) => mapping.kind === 'call').map((mapping) => mapping.canonical_id),
    };
}

function uniqueIds(...groups: readonly (readonly string[])[]): string[] {
    return [...new Set(groups.flat())];
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

function assertExactReplayEvidence(actual: unknown, expected: JsonValue, replayId: string, label: string): void {
    if (stableJson(bedrockConverseJsonValue(actual)) !== stableJson(expected)) {
        throw new TypeError(`Bedrock replay block ${replayId} no longer matches protected ${label}`);
    }
}

function ownValue(value: object, key: string): unknown {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && 'value' in descriptor ? descriptor.value : undefined;
}

function assertRecord(value: unknown, label: string): asserts value is Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new TypeError(`${label} must be an object`);
    }
}

function assertAllowedKeys(value: object, allowed: readonly string[], label: string): void {
    const extra = Object.keys(value).find((key) => !allowed.includes(key));
    if (extra !== undefined) throw new TypeError(`${label} contains unsupported field ${extra}`);
}

function assertNonemptyString(value: unknown, label: string): asserts value is string {
    if (typeof value !== 'string' || value.length === 0) throw new TypeError(`${label} must be a nonempty string`);
}

function bedrockToolUseType(value: object, path: string): 'tool_use' | 'server_tool_use' | undefined {
    const type = ownValue(value, 'type');
    if (type === undefined) return undefined;
    if (type !== 'tool_use' && type !== 'server_tool_use') {
        const diagnostic =
            typeof type === 'string' ? JSON.stringify(type.length > 80 ? `${type.slice(0, 80)}…` : type) : typeof type;
        throw new TypeError(`Bedrock tool use type ${diagnostic} at ${path} is unsupported`);
    }
    return type;
}

/** Clone an AWS document value without interpreting any user-owned key names. */
export function bedrockConverseJsonValue(value: unknown): JsonValue {
    if (value instanceof Uint8Array) return { _llumiverse_bedrock_bytes: bytesToBase64(value) };
    if (Array.isArray(value)) return value.map(bedrockConverseJsonValue);
    if (typeof value === 'object' && value !== null) {
        const converted = Object.fromEntries(
            Object.keys(value).flatMap((key): Array<[string, JsonValue]> => {
                const child = ownValue(value, key);
                return child === undefined ? [] : [[key, bedrockConverseJsonValue(child)]];
            }),
        );
        if (!preflightJsonInput(converted).success)
            throw new TypeError('Bedrock document value must be exact JSON data');
        return converted;
    }
    if (!preflightJsonInput(value).success) throw new TypeError('Bedrock document value must be exact JSON data');
    return value as JsonValue;
}

/** Validate application-owned JSON without applying Bedrock's transport byte encoding. */
function exactJsonValue(value: unknown, label: string): JsonValue {
    const checked = preflightJsonInput(value);
    if (!checked.success) throw new TypeError(`${label} must be exact JSON data`);
    return structuredClone(value) as JsonValue;
}

function bytesToBase64(value: Uint8Array): string {
    assertNativeImportInputBounds(value, [Uint8Array.prototype, Buffer.prototype]);
    return Buffer.from(copyNativeConversationByteView(value)).toString('base64');
}

function base64ToBytes(value: string): Uint8Array {
    return new Uint8Array(Buffer.from(value, 'base64'));
}

export interface BedrockConverseFamilyCapabilities {
    family: 'anthropic.claude' | 'amazon.nova' | 'openai.gpt' | 'mistral.pixtral' | 'deepseek.r1' | 'other';
    media: ReadonlySet<'image' | 'document' | 'audio' | 'video'>;
    signed_reasoning_replay: boolean;
}

export function bedrockConverseFamilyCapabilities(model: string): BedrockConverseFamilyCapabilities {
    const normalized = model.toLowerCase().split('/').pop() ?? '';
    if (normalized.includes('anthropic.claude') || normalized.includes('claude')) {
        return { family: 'anthropic.claude', media: new Set(['image', 'document']), signed_reasoning_replay: true };
    }
    if (normalized.includes('amazon.nova') || normalized.includes('nova-')) {
        return {
            family: 'amazon.nova',
            media: new Set(['image', 'document', 'video']),
            signed_reasoning_replay: false,
        };
    }
    if (normalized.includes('openai.gpt')) {
        return { family: 'openai.gpt', media: new Set(['image']), signed_reasoning_replay: false };
    }
    if (normalized.includes('mistral.pixtral')) {
        return { family: 'mistral.pixtral', media: new Set(['image']), signed_reasoning_replay: false };
    }
    if (normalized.includes('deepseek.r1')) {
        return { family: 'deepseek.r1', media: new Set(), signed_reasoning_replay: false };
    }
    return { family: 'other', media: new Set(), signed_reasoning_replay: false };
}

function assertMediaCapability(type: 'image' | 'document' | 'audio' | 'video', model?: string): void {
    if (model === undefined) return;
    const capabilities = bedrockConverseFamilyCapabilities(model);
    if (!capabilities.media.has(type)) {
        throw new TypeError(
            `Bedrock Converse ${capabilities.family} model ${model} does not support canonical ${type} input`,
        );
    }
}

function assertBinaryOrS3Source(value: unknown, path: string): void {
    assertRecord(value, `Bedrock media source ${path}`);
    const bytes = ownValue(value, 'bytes');
    const s3Location = ownValue(value, 's3Location');
    if ((bytes instanceof Uint8Array ? 1 : 0) + (s3Location === undefined ? 0 : 1) !== 1) {
        throw new TypeError(`Bedrock media source ${path} must contain bytes or s3Location`);
    }
    if (s3Location !== undefined) {
        assertRecord(s3Location, `Bedrock S3 source ${path}`);
        assertNonemptyString(ownValue(s3Location, 'uri'), `Bedrock S3 URI ${path}`);
        assertAllowedKeys(s3Location, ['uri', 'bucketOwner'], `Bedrock S3 source ${path}`);
    }
}

function assertMediaMember(member: string, value: unknown, path: string): void {
    assertRecord(value, `Bedrock ${member} block ${path}`);
    if (member === 'image' || member === 'audio' || member === 'video') {
        assertAllowedKeys(value, ['format', 'source'], `Bedrock ${member} block ${path}`);
        assertNonemptyString(ownValue(value, 'format'), `Bedrock ${member} format ${path}`);
        assertBinaryOrS3Source(ownValue(value, 'source'), path);
        return;
    }
    assertAllowedKeys(value, ['format', 'name', 'source', 'context', 'citations'], `Bedrock document block ${path}`);
    assertNonemptyString(ownValue(value, 'name'), `Bedrock document name ${path}`);
    const source = ownValue(value, 'source');
    assertRecord(source, `Bedrock document source ${path}`);
    const members = ['bytes', 's3Location', 'text', 'content'].filter((key) => ownValue(source, key) !== undefined);
    if (members.length !== 1) throw new TypeError(`Bedrock document source ${path} has an unsupported shape`);
    if (members[0] === 'bytes' && !(ownValue(source, 'bytes') instanceof Uint8Array)) {
        throw new TypeError(`Bedrock document bytes ${path} must be Uint8Array`);
    }
    if (members[0] === 'text' && typeof ownValue(source, 'text') !== 'string') {
        throw new TypeError(`Bedrock document text ${path} must be a string`);
    }
    if (members[0] === 'content') bedrockConverseJsonValue(ownValue(source, 'content'));
    if (members[0] === 's3Location') assertBinaryOrS3Source(source, path);
}

function assertReasoningMember(value: unknown, path: string): void {
    assertRecord(value, `Bedrock reasoning block ${path}`);
    const reasoningText = ownValue(value, 'reasoningText');
    const redactedContent = ownValue(value, 'redactedContent');
    if ((reasoningText === undefined ? 0 : 1) + (redactedContent === undefined ? 0 : 1) !== 1) {
        throw new TypeError(`Bedrock reasoning block ${path} has an unsupported shape`);
    }
    if (reasoningText !== undefined) {
        assertRecord(reasoningText, `Bedrock reasoning text ${path}`);
        assertAllowedKeys(reasoningText, ['text', 'signature'], `Bedrock reasoning text ${path}`);
        if (typeof ownValue(reasoningText, 'text') !== 'string') {
            throw new TypeError(`Bedrock reasoning text ${path} must contain text`);
        }
        const signature = ownValue(reasoningText, 'signature');
        if (signature !== undefined) assertNonemptyString(signature, `Bedrock reasoning signature ${path}`);
    } else if (!(redactedContent instanceof Uint8Array)) {
        throw new TypeError(`Bedrock redacted reasoning ${path} must contain bytes`);
    }
}

function assertToolResultContent(block: unknown, path: string): void {
    assertRecord(block, `Bedrock tool result content ${path}`);
    const members = ['text', 'json', 'image', 'document', 'video'].filter((key) => ownValue(block, key) !== undefined);
    if (members.length !== 1) {
        throw new TypeError(`Bedrock tool result content ${path} must contain exactly one supported member`);
    }
    assertAllowedKeys(block, members, `Bedrock tool result content ${path}`);
    if (members[0] === 'text' && typeof ownValue(block, 'text') !== 'string') {
        throw new TypeError(`Bedrock tool result text ${path} must be a string`);
    }
    if (members[0] === 'json') exactJsonValue(ownValue(block, 'json'), `Bedrock tool result JSON ${path}`);
    if (members[0] === 'image' || members[0] === 'document' || members[0] === 'video') {
        assertMediaMember(members[0], ownValue(block, members[0]), path);
    }
}

function assertContentBlock(block: unknown, role: Message['role'], path: string): void {
    assertRecord(block, `Bedrock content block ${path}`);
    const members = ['text', 'image', 'document', 'audio', 'video', 'reasoningContent', 'toolUse', 'toolResult'].filter(
        (key) => ownValue(block, key) !== undefined,
    );
    if (members.length !== 1) {
        throw new TypeError(`Bedrock content block ${path} must contain exactly one supported member`);
    }
    const member = members[0];
    assertAllowedKeys(block, [member], `Bedrock content block ${path}`);
    if (member === 'text') {
        if (typeof ownValue(block, 'text') !== 'string') throw new TypeError(`Bedrock text ${path} must be a string`);
        return;
    }
    if (member === 'image' || member === 'document' || member === 'audio' || member === 'video') {
        if (role !== 'user') throw new TypeError(`Bedrock ${member} ${path} requires a user message`);
        assertMediaMember(member, ownValue(block, member), path);
        return;
    }
    if (member === 'reasoningContent') {
        if (role !== 'assistant') throw new TypeError(`Bedrock reasoning ${path} requires an assistant message`);
        assertReasoningMember(ownValue(block, member), path);
        return;
    }
    if (member === 'toolUse') {
        if (role !== 'assistant') throw new TypeError(`Bedrock tool use ${path} requires an assistant message`);
        const toolUse = ownValue(block, 'toolUse');
        assertRecord(toolUse, `Bedrock tool use ${path}`);
        assertAllowedKeys(toolUse, ['toolUseId', 'name', 'input', 'type'], `Bedrock tool use ${path}`);
        assertNonemptyString(ownValue(toolUse, 'toolUseId'), `Bedrock tool use ID ${path}`);
        assertNonemptyString(ownValue(toolUse, 'name'), `Bedrock tool name ${path}`);
        exactJsonValue(ownValue(toolUse, 'input'), `Bedrock tool input ${path}`);
        bedrockToolUseType(toolUse, path);
        return;
    }
    if (role !== 'user') throw new TypeError(`Bedrock tool result ${path} requires a user message`);
    const result = ownValue(block, 'toolResult');
    assertRecord(result, `Bedrock tool result ${path}`);
    assertAllowedKeys(
        result,
        ['toolUseId', 'content', 'status', '_llumiverse_tool_result_status'],
        `Bedrock tool result ${path}`,
    );
    assertNonemptyString(ownValue(result, 'toolUseId'), `Bedrock tool result ID ${path}`);
    const status = ownValue(result, 'status');
    if (status !== undefined && status !== 'success' && status !== 'error') {
        throw new TypeError(`Bedrock tool result status ${path} is unsupported`);
    }
    const canonicalStatus = ownValue(result, '_llumiverse_tool_result_status');
    if (
        canonicalStatus !== undefined &&
        canonicalStatus !== 'success' &&
        canonicalStatus !== 'error' &&
        canonicalStatus !== 'cancelled' &&
        canonicalStatus !== 'denied' &&
        canonicalStatus !== 'unknown'
    ) {
        throw new TypeError(`Bedrock canonical tool result status ${path} is unsupported`);
    }
    const content = ownValue(result, 'content');
    if (!Array.isArray(content) || content.length === 0) {
        throw new TypeError(`Bedrock tool result ${path} must contain content`);
    }
    content.forEach((child, index) => {
        assertToolResultContent(child, `${path}/content/${index}`);
    });
}

function assertBedrockHistory(value: unknown): asserts value is BedrockConverseConversation {
    assertRecord(value, 'Bedrock Converse history');
    assertAllowedKeys(value, ['messages', 'system'], 'Bedrock Converse history');
    const messages = ownValue(value, 'messages');
    const system = ownValue(value, 'system');
    if (messages !== undefined && !Array.isArray(messages)) throw new TypeError('Bedrock messages must be an array');
    if (system !== undefined && !Array.isArray(system)) throw new TypeError('Bedrock system must be an array');
    (system ?? []).forEach((block, index) => {
        assertRecord(block, `Bedrock system block system/${index}`);
        assertAllowedKeys(block, ['text'], `Bedrock system block system/${index}`);
        if (typeof ownValue(block, 'text') !== 'string') {
            throw new TypeError(`Bedrock system block system/${index} must contain text`);
        }
    });
    (messages ?? []).forEach((message, index) => {
        const path = `messages/${index}`;
        assertRecord(message, `Bedrock message ${path}`);
        assertAllowedKeys(message, ['role', 'content'], `Bedrock message ${path}`);
        const role = ownValue(message, 'role');
        if (role !== 'user' && role !== 'assistant') {
            throw new TypeError(`Bedrock message ${path} has unsupported role ${String(role)}`);
        }
        const content = ownValue(message, 'content');
        if (!Array.isArray(content) || content.length === 0) {
            throw new TypeError(`Bedrock message ${path} must contain content`);
        }
        content.forEach((block, contentIndex) => {
            assertContentBlock(block, role, `${path}/content/${contentIndex}`);
        });
    });
}

export function isBedrockConverseHistory(
    value: unknown,
    explicitProtocol?: typeof BEDROCK_CONVERSE_PROTOCOL,
): value is BedrockConverseConversation {
    try {
        assertBedrockHistory(value);
        if (typeof value !== 'object' || value === null) return false;
        const messages = ownValue(value, 'messages');
        const system = ownValue(value, 'system');
        const hasContent =
            (Array.isArray(messages) && messages.length > 0) || (Array.isArray(system) && system.length > 0);
        return hasContent || explicitProtocol === BEDROCK_CONVERSE_PROTOCOL;
    } catch {
        return false;
    }
}

async function entityId(kind: string, scope: string, path: string): Promise<string> {
    return deriveConversationId(kind, scope, path);
}

function provenance(path: string, turnNumber?: number): ImportedTurnProvenance {
    return {
        type: 'imported',
        source: BEDROCK_CONVERSE_PROTOCOL,
        native_id: { protocol: BEDROCK_CONVERSE_PROTOCOL, scope: 'history', value: path },
        ...(turnNumber === undefined ? {} : { source_history_turn_number: turnNumber }),
        missing_metadata: ['actor_id', 'timestamps', 'exchange'],
    };
}

function turnProvenance(source: SourceKind, path: string, turnNumber?: number): ConversationTurn['provenance'] {
    return source === 'imported' ? provenance(path, turnNumber) : { type: 'received' };
}

function turnFields(recordedAt: string) {
    return {
        status: 'completed' as const,
        timestamps: { recorded_at: recordedAt },
        model_visibility: 'include' as const,
    };
}

function messageGroupMetadata(path: string): ConversationTurn['metadata'] {
    const messageIndex = /^messages\/(\d+)/.exec(path)?.[1];
    return messageIndex === undefined ? undefined : { bedrock_converse: { message_index: messageIndex } };
}

async function textBlock(scope: string, path: string, text: string): Promise<UserContentBlock> {
    return { id: await entityId('block', scope, path), type: 'text', text, format: 'plain' };
}

const MEDIA_MIME_TYPES = {
    image: { png: 'image/png', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp' },
    audio: {
        aac: 'audio/aac',
        flac: 'audio/flac',
        m4a: 'audio/mp4',
        mp3: 'audio/mpeg',
        mp4: 'audio/mp4',
        ogg: 'audio/ogg',
        opus: 'audio/opus',
        wav: 'audio/wav',
        webm: 'audio/webm',
    },
    video: { mov: 'video/quicktime', mkv: 'video/x-matroska', mp4: 'video/mp4', webm: 'video/webm' },
} as const;

function mediaMimeType(type: 'image' | 'audio' | 'video', format: string): string {
    const values = MEDIA_MIME_TYPES[type] as Record<string, string>;
    return values[format] ?? `${type}/${format}`;
}

function documentMimeType(format: string | undefined): string {
    const values: Record<string, string> = {
        pdf: 'application/pdf',
        csv: 'text/csv',
        doc: 'application/msword',
        docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        xls: 'application/vnd.ms-excel',
        xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        html: 'text/html',
        txt: 'text/plain',
        md: 'text/markdown',
    };
    return (format && values[format]) ?? 'application/octet-stream';
}

async function mediaAssetBlock(input: {
    type: 'image' | 'document' | 'audio' | 'video';
    native: ImageBlock | DocumentBlock | AudioBlock | VideoBlock;
    scope: string;
    path: string;
    recorded_at: string;
    source: SourceKind;
    turn_id: string;
}): Promise<{ block: UserContentBlock; asset: Asset }> {
    const native = input.native;
    const source = native.source;
    if (source === undefined) throw new TypeError(`Bedrock ${input.type} ${input.path} has no source`);
    const assetId = await entityId('asset', input.scope, input.path);
    const blockId = await entityId('block', input.scope, input.path);
    let storage: Asset['storage'];
    let sourceType: string;
    if ('bytes' in source && source.bytes instanceof Uint8Array) {
        storage = { type: 'inline_base64', data: bytesToBase64(source.bytes) };
        sourceType = 'bytes';
    } else if ('s3Location' in source && source.s3Location !== undefined) {
        storage = {
            type: 'external',
            resolver: 'aws.s3',
            locator: bedrockConverseJsonValue(source.s3Location) as Record<string, JsonValue>,
        };
        sourceType = 's3Location';
    } else if ('text' in source && typeof source.text === 'string') {
        storage = { type: 'inline_text', text: source.text };
        sourceType = 'text';
    } else if ('content' in source && source.content !== undefined) {
        storage = { type: 'inline_json', value: bedrockConverseJsonValue(source.content) };
        sourceType = 'content';
    } else {
        throw new TypeError(`Bedrock ${input.type} source ${input.path} is unsupported`);
    }
    const format = 'format' in native ? native.format : undefined;
    const mimeType =
        input.type === 'document'
            ? documentMimeType(format)
            : mediaMimeType(input.type, typeof format === 'string' ? format : 'octet-stream');
    const nativeMetadata: Record<string, JsonValue> = {
        source_type: sourceType,
        ...(format === undefined ? {} : { format }),
        ...('name' in native && native.name !== undefined ? { name: native.name } : {}),
        ...('context' in native && native.context !== undefined ? { context: native.context } : {}),
        ...('citations' in native && native.citations !== undefined
            ? { citations: bedrockConverseJsonValue(native.citations) }
            : {}),
    };
    const integrity = await inlineAssetContentIntegrity(storage);
    const asset: Asset = {
        id: assetId,
        kind: input.type,
        mime_type: mimeType,
        storage,
        provenance:
            input.source === 'imported'
                ? { type: 'imported', source: BEDROCK_CONVERSE_PROTOCOL }
                : { type: 'received', source_turn_id: input.turn_id },
        ...(integrity ?? {}),
        created_at: input.recorded_at,
        metadata: { bedrock_converse: nativeMetadata },
    };
    const common = { id: blockId, asset_id: assetId };
    const block: UserContentBlock =
        input.type === 'image'
            ? { ...common, type: 'image' }
            : input.type === 'document'
              ? { ...common, type: 'document', ...('name' in native ? { caption: native.name } : {}) }
              : input.type === 'audio'
                ? { ...common, type: 'audio' }
                : { ...common, type: 'video' };
    return { block, asset };
}

function mediaMetadata(asset: Asset): Record<string, JsonValue> {
    const metadata = asset.metadata?.bedrock_converse;
    return typeof metadata === 'object' && metadata !== null && !Array.isArray(metadata) ? metadata : {};
}

function assetSource(asset: Asset): Record<string, unknown> {
    const sourceType = mediaMetadata(asset).source_type;
    if (sourceType === 'bytes' && asset.storage.type === 'inline_base64') {
        return { bytes: base64ToBytes(asset.storage.data) };
    }
    if (sourceType === 's3Location' && asset.storage.type === 'external' && asset.storage.resolver === 'aws.s3') {
        return { s3Location: structuredClone(asset.storage.locator) };
    }
    if (sourceType === 'text' && asset.storage.type === 'inline_text') return { text: asset.storage.text };
    if (sourceType === 'content' && asset.storage.type === 'inline_json') {
        return { content: structuredClone(asset.storage.value) };
    }
    throw new TypeError(`Bedrock Converse cannot reconstruct ${asset.kind} asset ${asset.id}`);
}

function assetToBedrockBlock(asset: Asset): ContentBlock {
    const metadata = mediaMetadata(asset);
    const format = typeof metadata.format === 'string' ? metadata.format : undefined;
    const source = assetSource(asset);
    if (asset.kind === 'image')
        return { image: { format: format as ImageBlock['format'], source } as unknown as ImageBlock };
    if (asset.kind === 'audio')
        return { audio: { format: format as AudioBlock['format'], source } as unknown as AudioBlock };
    if (asset.kind === 'video')
        return { video: { format: format as VideoBlock['format'], source } as unknown as VideoBlock };
    if (asset.kind === 'document') {
        return {
            document: {
                ...(format === undefined ? {} : { format: format as DocumentBlock['format'] }),
                name: typeof metadata.name === 'string' ? metadata.name : 'document',
                source,
                ...(typeof metadata.context === 'string' ? { context: metadata.context } : {}),
                ...(metadata.citations === undefined ? {} : { citations: structuredClone(metadata.citations) }),
            } as unknown as DocumentBlock,
        };
    }
    throw new TypeError(`Bedrock Converse cannot project ${asset.kind} asset ${asset.id}`);
}

function nativeMedia(
    block: ContentBlock | ToolResultContentBlock,
):
    | { type: 'image' | 'document' | 'audio' | 'video'; native: ImageBlock | DocumentBlock | AudioBlock | VideoBlock }
    | undefined {
    if ('image' in block && block.image !== undefined) return { type: 'image', native: block.image };
    if ('document' in block && block.document !== undefined) return { type: 'document', native: block.document };
    if ('audio' in block && block.audio !== undefined) return { type: 'audio', native: block.audio };
    if ('video' in block && block.video !== undefined) return { type: 'video', native: block.video };
    return undefined;
}

async function toolCallBlock(
    scope: string,
    path: string,
    native: NonNullable<ContentBlock.ToolUseMember['toolUse']>,
    definitions: readonly ToolDefinition[],
): Promise<AgentContentBlock> {
    assertNonemptyString(native.toolUseId, `Bedrock tool use ID ${path}`);
    assertNonemptyString(native.name, `Bedrock tool name ${path}`);
    const definition = definitions.find((candidate) => candidate.name === native.name);
    const nativeType = bedrockToolUseType(native, path);
    return {
        id: await entityId('block', scope, path),
        type: 'tool_call',
        call_id: native.toolUseId,
        tool_name: native.name,
        ...(definition === undefined ? {} : { definition_id: definition.id }),
        executor: nativeType === 'server_tool_use' ? 'provider' : 'application',
        arguments: { type: 'json', value: exactJsonValue(native.input, `Bedrock tool input ${path}`) },
        native_id: { protocol: BEDROCK_CONVERSE_PROTOCOL, scope: 'history', value: native.toolUseId },
    };
}

async function resultContent(input: {
    scope: string;
    path: string;
    native: ToolResultContentBlock;
    recorded_at: string;
    source: SourceKind;
    turn_id: string;
}): Promise<{ block: NestedToolResultContentBlock; asset?: Asset }> {
    const { native } = input;
    if ('text' in native && typeof native.text === 'string') {
        return { block: await textBlock(input.scope, input.path, native.text) };
    }
    if ('json' in native) {
        return {
            block: {
                id: await entityId('block', input.scope, input.path),
                type: 'json',
                value: exactJsonValue(native.json, `Bedrock tool result JSON ${input.path}`),
            },
        };
    }
    const media = nativeMedia(native);
    if (media !== undefined) {
        const converted = await mediaAssetBlock({
            ...media,
            scope: input.scope,
            path: input.path,
            recorded_at: input.recorded_at,
            source: input.source,
            turn_id: input.turn_id,
        });
        return { block: converted.block, asset: converted.asset };
    }
    throw new TypeError(`Bedrock tool result content ${input.path} is unsupported`);
}

async function toolResultTurn(input: {
    scope: string;
    path: string;
    native: BedrockNativeToolResult;
    recorded_at: string;
    turn_number?: number;
    source: SourceKind;
}): Promise<{ turn: ConversationTurn; assets: Asset[]; mappings: NativeItemMapping[]; receipt?: ExecutionReceipt }> {
    assertNonemptyString(input.native.toolUseId, `Bedrock tool result ID ${input.path}`);
    const content: NestedToolResultContentBlock[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    const turnId = await entityId('turn', input.scope, input.path);
    for (let index = 0; index < (input.native.content?.length ?? 0); index += 1) {
        const native = input.native.content?.[index];
        if (native === undefined) continue;
        const path = `${input.path}/toolResult/content/${index}`;
        const converted = await resultContent({
            scope: input.scope,
            path,
            native,
            recorded_at: input.recorded_at,
            source: input.source,
            turn_id: turnId,
        });
        content.push(converted.block);
        mappings.push({ canonical_id: converted.block.id, native_id: path, kind: 'block' });
        if (converted.asset !== undefined) assets.push(converted.asset);
    }
    const canonicalStatus = input.native._llumiverse_tool_result_status;
    const block: ToolResultBlock = {
        id: await entityId('block', input.scope, `${input.path}/toolResult`),
        type: 'tool_result',
        call_id: input.native.toolUseId,
        status: canonicalStatus ?? input.native.status ?? (input.source === 'received' ? 'success' : 'unknown'),
        content,
        native_id: { protocol: BEDROCK_CONVERSE_PROTOCOL, scope: 'history', value: input.native.toolUseId },
    };
    mappings.unshift(
        { canonical_id: turnId, native_id: input.path, kind: 'turn' },
        { canonical_id: block.id, native_id: `${input.path}/toolResult`, kind: 'block' },
        { canonical_id: block.call_id, native_id: block.call_id, kind: 'call' },
    );
    return {
        turn: {
            id: turnId,
            kind: 'tool',
            authority: 'ordinary',
            ...turnFields(input.recorded_at),
            provenance: turnProvenance(input.source, input.path, input.turn_number) as Exclude<
                ConversationTurn,
                { kind: 'agent' }
            >['provenance'],
            metadata: {
                ...messageGroupMetadata(input.path),
                ...(input.native.status === undefined && canonicalStatus === undefined && input.source === 'received'
                    ? { bedrock_converse_tool_result_status_omitted: true }
                    : {}),
            },
            blocks: [block],
        },
        assets,
        mappings,
        ...(input.source === 'received'
            ? {
                  receipt: {
                      id: await entityId('execution_receipt', input.scope, block.call_id),
                      call_id: block.call_id,
                      executor: 'application' as const,
                      status: block.status === 'unknown' ? ('success' as const) : block.status,
                      result_turn_id: turnId,
                      result_fingerprint: await fingerprintJson(block),
                      recorded_at: input.recorded_at,
                  },
              }
            : {}),
    };
}

async function messageRecords(input: {
    message: Message;
    index: number;
    scope: string;
    options: ImportBedrockConverseConversationOptions;
    source: SourceKind;
    prefix: BedrockConverseConversation;
    prefix_dependencies: BedrockReplayDependencies;
}): Promise<{
    turns: ConversationTurn[];
    assets: Asset[];
    mappings: NativeItemMapping[];
    execution_receipts: ExecutionReceipt[];
}> {
    const path = `messages/${input.index}`;
    const nativeBlocks = input.message.content ?? [];
    if (input.message.role === 'assistant') {
        const turnId = await entityId('turn', input.scope, path);
        const blocks: AgentContentBlock[] = [];
        const assets: Asset[] = [];
        const mappings: NativeItemMapping[] = [{ canonical_id: turnId, native_id: path, kind: 'turn' }];
        const replayEntries: BedrockReplayEntry[] = [];
        let hasProtectedReasoning = false;
        let hasTypedToolUse = false;
        let hasServerToolUse = false;
        for (let index = 0; index < nativeBlocks.length; index += 1) {
            const native = nativeBlocks[index];
            const blockPath = `${path}/content/${index}`;
            let block: AgentContentBlock | undefined;
            if ('text' in native && typeof native.text === 'string') {
                block = await textBlock(input.scope, blockPath, native.text);
                replayEntries.push({ kind: 'canonical', block_id: block.id, native: bedrockConverseJsonValue(native) });
            } else if ('toolUse' in native && native.toolUse !== undefined) {
                const toolUseType = bedrockToolUseType(native.toolUse, blockPath);
                hasTypedToolUse ||= toolUseType !== undefined;
                hasServerToolUse ||= toolUseType === 'server_tool_use';
                block = await toolCallBlock(
                    input.scope,
                    blockPath,
                    native.toolUse,
                    input.options.tool_definitions ?? [],
                );
                replayEntries.push({ kind: 'canonical', block_id: block.id, native: bedrockConverseJsonValue(native) });
            } else if ('reasoningContent' in native && native.reasoningContent !== undefined) {
                const reasoning = native.reasoningContent;
                if ('reasoningText' in reasoning && reasoning.reasoningText !== undefined) {
                    const text = reasoning.reasoningText.text;
                    if (text === undefined) throw new TypeError(`Bedrock reasoning text ${blockPath} is missing`);
                    const reasoningBlock: AgentContentBlock = {
                        id: await entityId('reasoning', input.scope, blockPath),
                        type: 'reasoning',
                        text,
                        representation: 'text',
                    };
                    block = reasoningBlock;
                    replayEntries.push({
                        kind: 'reasoning',
                        block_id: reasoningBlock.id,
                        ...(reasoning.reasoningText.signature === undefined
                            ? {}
                            : { signature: reasoning.reasoningText.signature }),
                        native: bedrockConverseJsonValue(native),
                    });
                    hasProtectedReasoning ||= reasoning.reasoningText.signature !== undefined;
                } else if ('redactedContent' in reasoning && reasoning.redactedContent instanceof Uint8Array) {
                    hasProtectedReasoning = true;
                    replayEntries.push({
                        kind: 'redacted_reasoning',
                        data: bytesToBase64(reasoning.redactedContent),
                        native: bedrockConverseJsonValue(native),
                    });
                }
            } else {
                const media = nativeMedia(native);
                if (media !== undefined) {
                    const converted = await mediaAssetBlock({
                        ...media,
                        scope: input.scope,
                        path: blockPath,
                        recorded_at: input.options.recorded_at,
                        source: input.source,
                        turn_id: turnId,
                    });
                    block = converted.block as AgentContentBlock;
                    assets.push(converted.asset);
                    replayEntries.push({
                        kind: 'canonical',
                        block_id: block.id,
                        native: bedrockConverseJsonValue(native),
                    });
                }
            }
            if (block !== undefined) {
                blocks.push(block);
                mappings.push({ canonical_id: block.id, native_id: blockPath, kind: 'block' });
                if (block.type === 'tool_call') {
                    mappings.push({ canonical_id: block.call_id, native_id: block.call_id, kind: 'call' });
                }
            }
        }
        if (hasProtectedReasoning || hasTypedToolUse) {
            const hasProtectedReplay = hasProtectedReasoning || hasServerToolUse;
            const replayId = await entityId('replay', input.scope, path);
            blocks.push({
                id: replayId,
                type: 'native_replay',
                adapter: BEDROCK_CONVERSE_ADAPTER_VERSION,
                protocol: BEDROCK_CONVERSE_PROTOCOL,
                compatibility_scope: {
                    provider: requireBedrockReplayProvider(input.options.provider),
                    protocol: BEDROCK_CONVERSE_PROTOCOL,
                    ...(input.options.model === undefined ? {} : { model: input.options.model }),
                    adapter_version: BEDROCK_CONVERSE_ADAPTER_VERSION,
                },
                payload: {
                    type: 'bedrock_converse_content_order',
                    ...(hasProtectedReplay ? { prefix: bedrockConverseJsonValue(input.prefix) } : {}),
                    entries: replayEntries,
                } as unknown as JsonValue,
                dependencies: {
                    turn_ids: uniqueIds(hasProtectedReplay ? input.prefix_dependencies.turn_ids : [], [turnId]),
                    block_ids: uniqueIds(
                        hasProtectedReplay ? input.prefix_dependencies.block_ids : [],
                        blocks.map((candidate) => candidate.id),
                    ),
                    call_ids: uniqueIds(
                        hasProtectedReplay ? input.prefix_dependencies.call_ids : [],
                        blocks.flatMap((candidate) => (candidate.type === 'tool_call' ? [candidate.call_id] : [])),
                    ),
                    request_ids: [],
                },
                ...(hasProtectedReplay ? {} : { dependency_policy: 'discard_on_dependency_change' as const }),
            });
            mappings.push({ canonical_id: replayId, native_id: `${path}/content_order`, kind: 'block' });
        }
        return {
            turns: [
                {
                    id: turnId,
                    kind: 'agent',
                    authority: 'ordinary',
                    ...turnFields(input.options.recorded_at),
                    provenance:
                        input.source === 'imported'
                            ? provenance(path, input.options.source_history_turn_number)
                            : { type: 'received' },
                    metadata: messageGroupMetadata(path),
                    blocks,
                } as ConversationTurn,
            ],
            assets,
            mappings,
            execution_receipts: [],
        };
    }

    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    const executionReceipts: ExecutionReceipt[] = [];
    for (let index = 0; index < nativeBlocks.length; index += 1) {
        const native = nativeBlocks[index];
        const blockPath = `${path}/content/${index}`;
        if ('text' in native && typeof native.text === 'string') {
            const block = await textBlock(input.scope, blockPath, native.text);
            const turnId = await entityId('turn', input.scope, blockPath);
            turns.push({
                id: turnId,
                kind: 'user',
                authority: 'ordinary',
                ...turnFields(input.options.recorded_at),
                provenance: turnProvenance(
                    input.source,
                    blockPath,
                    input.options.source_history_turn_number,
                ) as Exclude<ConversationTurn, { kind: 'agent' }>['provenance'],
                metadata: messageGroupMetadata(blockPath),
                blocks: [block],
            });
            mappings.push(
                { canonical_id: turnId, native_id: blockPath, kind: 'turn' },
                { canonical_id: block.id, native_id: blockPath, kind: 'block' },
            );
        } else if ('toolResult' in native && native.toolResult !== undefined) {
            const converted = await toolResultTurn({
                scope: input.scope,
                path: blockPath,
                native: native.toolResult,
                recorded_at: input.options.recorded_at,
                source: input.source,
                ...(input.options.source_history_turn_number === undefined
                    ? {}
                    : { turn_number: input.options.source_history_turn_number }),
            });
            turns.push(converted.turn);
            assets.push(...converted.assets);
            mappings.push(...converted.mappings);
            if (converted.receipt !== undefined) executionReceipts.push(converted.receipt);
        } else {
            const media = nativeMedia(native);
            if (media === undefined) throw new TypeError(`Bedrock user content ${blockPath} is unsupported`);
            const turnId = await entityId('turn', input.scope, blockPath);
            const converted = await mediaAssetBlock({
                ...media,
                scope: input.scope,
                path: blockPath,
                recorded_at: input.options.recorded_at,
                source: input.source,
                turn_id: turnId,
            });
            turns.push({
                id: turnId,
                kind: 'user',
                authority: 'ordinary',
                ...turnFields(input.options.recorded_at),
                provenance: turnProvenance(
                    input.source,
                    blockPath,
                    input.options.source_history_turn_number,
                ) as Exclude<ConversationTurn, { kind: 'agent' }>['provenance'],
                metadata: messageGroupMetadata(blockPath),
                blocks: [converted.block],
            });
            assets.push(converted.asset);
            mappings.push(
                { canonical_id: turnId, native_id: blockPath, kind: 'turn' },
                { canonical_id: converted.block.id, native_id: blockPath, kind: 'block' },
            );
        }
    }
    return { turns, assets, mappings, execution_receipts: executionReceipts };
}

function requireBedrockReplayProvider(provider: string | undefined): string {
    if (!provider) throw new TypeError('Protected Bedrock replay requires recorded provider provenance');
    return provider;
}

async function importRecords(
    history: BedrockConverseConversation,
    options: ImportBedrockConverseConversationOptions,
    source: SourceKind = 'imported',
    scopeOverride?: string,
    existingCalls: ReadonlySet<string> = new Set(),
    existingResults: ReadonlySet<string> = new Set(),
): Promise<ImportRecords> {
    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    const executionReceipts: ExecutionReceipt[] = [];
    const calls = new Set(existingCalls);
    const results = new Set(existingResults);
    const scope = scopeOverride ?? `${options.conversation_id}:bedrock-import`;

    for (let index = 0; index < (history.system?.length ?? 0); index += 1) {
        const native = history.system?.[index];
        if (native === undefined || !('text' in native) || typeof native.text !== 'string') {
            throw new TypeError(`Bedrock system block system/${index} is unsupported`);
        }
        const path = `system/${index}`;
        const turnId = await entityId('turn', scope, path);
        const block = await textBlock(scope, `${path}/text`, native.text);
        turns.push({
            id: turnId,
            kind: 'program',
            authority: 'system',
            ...turnFields(options.recorded_at),
            provenance: turnProvenance(source, path, options.source_history_turn_number) as Exclude<
                ConversationTurn,
                { kind: 'agent' }
            >['provenance'],
            blocks: [block],
        });
        mappings.push(
            { canonical_id: turnId, native_id: path, kind: 'turn' },
            { canonical_id: block.id, native_id: path, kind: 'block' },
        );
    }

    for (let index = 0; index < (history.messages?.length ?? 0); index += 1) {
        const message = history.messages?.[index];
        if (message === undefined) continue;
        for (const block of message.content ?? []) {
            if ('toolUse' in block && block.toolUse !== undefined) {
                assertNonemptyString(block.toolUse.toolUseId, `Bedrock tool call ID messages/${index}`);
                if (calls.has(block.toolUse.toolUseId)) {
                    throw new TypeError(`Bedrock tool call ID ${block.toolUse.toolUseId} is duplicated`);
                }
                calls.add(block.toolUse.toolUseId);
            }
            if ('toolResult' in block && block.toolResult !== undefined) {
                assertNonemptyString(block.toolResult.toolUseId, `Bedrock tool result ID messages/${index}`);
                if (!calls.has(block.toolResult.toolUseId)) {
                    throw new TypeError(`Bedrock tool result ${block.toolResult.toolUseId} has no prior tool call`);
                }
                if (results.has(block.toolResult.toolUseId)) {
                    throw new TypeError(`Bedrock tool call ${block.toolResult.toolUseId} has multiple results`);
                }
                results.add(block.toolResult.toolUseId);
            }
        }
        const converted = await messageRecords({
            message,
            index,
            scope,
            options,
            source,
            prefix: {
                ...(history.system === undefined ? {} : { system: history.system }),
                messages: history.messages?.slice(0, index) ?? [],
            },
            prefix_dependencies: replayDependenciesForTurns(turns),
        });
        turns.push(...converted.turns);
        assets.push(...converted.assets);
        mappings.push(...converted.mappings);
        executionReceipts.push(...converted.execution_receipts);
    }
    const contextEntries = await Promise.all(
        turns.map(async (turn) => ({
            id: await entityId('context_entry', scope, turn.id),
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
    );
    return {
        turns,
        assets,
        context_entries: contextEntries,
        mappings,
        execution_receipts: executionReceipts,
    };
}

/** Historical document-only compatibility API; new archive imports use the report-returning entry point. */
export async function importBedrockConverseConversation(
    historyInput: unknown,
    options: ImportBedrockConverseConversationOptions,
): Promise<ConversationDocument> {
    return importBedrockHistoryDocument(historyInput, options);
}

/** Pure import with declared origin evidence and explicit completeness/readiness diagnostics. */
export async function importBedrockConverseHistory(
    historyInput: unknown,
    options: NativeConversationImportOptions,
): Promise<NativeConversationImportResult> {
    return guardNativeConversationImport(async () => {
        options = snapshotNativeConversationImportOptions(options);
        newNativeImportDocument(options);
        assertNativeImportInputBounds(historyInput, [Uint8Array.prototype, Buffer.prototype]);
        const historySnapshot = structuredClone(historyInput);
        const importFingerprint = await fingerprintNativeConversationImport(
            bedrockConverseJsonValue(historySnapshot),
            options,
            BEDROCK_CONVERSE_PROTOCOL,
            BEDROCK_CONVERSE_ADAPTER_VERSION,
        );
        const document = await importBedrockHistoryDocument(historySnapshot, options, true, importFingerprint);
        return nativeConversationImportResult(
            document,
            options,
            BEDROCK_CONVERSE_PROTOCOL,
            BEDROCK_CONVERSE_ADAPTER_VERSION,
            [
                {
                    code: 'IMPORT_BYTE_VIEW_PROPERTIES_EXCLUDED',
                    message:
                        'Native byte-view values preserve intrinsic bytes only; own JavaScript annotations are outside imported protocol data.',
                },
            ],
        );
    });
}

async function importBedrockHistoryDocument(
    historyInput: unknown,
    options: ImportBedrockConverseConversationOptions,
    preparationIdentity = false,
    importPayloadFingerprint?: string,
): Promise<ConversationDocument> {
    assertBedrockHistory(historyInput);
    const history = preparationIdentity ? historyInput : structuredClone(historyInput);
    const records = await importRecords(
        history,
        options,
        'imported',
        preparationIdentity ? `${options.conversation_id}:legacy` : undefined,
    );
    const document = createConversationDocument({ id: options.conversation_id, created_at: options.recorded_at });
    return appendConversationRecords(
        document,
        {
            turns: records.turns,
            assets: records.assets,
            context_entries: records.context_entries,
            execution_receipts: records.execution_receipts,
            tool_definitions: [...(options.tool_definitions ?? [])],
            active_tool_definition_ids: (options.tool_definitions ?? []).map((tool) => tool.id),
        },
        {
            expected_revision: document.revision,
            operation_id: preparationIdentity
                ? await entityId('import', options.conversation_id, BEDROCK_CONVERSE_PROTOCOL)
                : await deriveConversationId('operation', options.conversation_id, BEDROCK_CONVERSE_PROTOCOL, 'import'),
            payload_fingerprint: importPayloadFingerprint ?? (await fingerprintJson(bedrockConverseJsonValue(history))),
            recorded_at: options.recorded_at,
        },
    ).document;
}

function bedrockReplay(
    turn: ConversationTurn,
    target?: { provider?: string; model?: string },
): { id: string; payload: BedrockReplayPayload } | undefined {
    const replays = turn.blocks.filter((block) => block.type === 'native_replay');
    const foreign = replays.find(
        (replay) =>
            replay.protocol !== BEDROCK_CONVERSE_PROTOCOL &&
            replay.dependency_policy !== 'discard_on_dependency_change',
    );
    if (foreign !== undefined) {
        throw new TypeError(`Bedrock Converse cannot discard protected ${foreign.protocol} replay block ${foreign.id}`);
    }
    const matching = replays.filter((replay) => replay.protocol === BEDROCK_CONVERSE_PROTOCOL);
    if (matching.length > 1) throw new TypeError(`Bedrock turn ${turn.id} has multiple protected replay blocks`);
    for (const replay of matching) {
        if (
            replay.adapter !== BEDROCK_CONVERSE_ADAPTER_VERSION ||
            (target?.provider !== undefined && replay.compatibility_scope.provider !== target.provider) ||
            replay.compatibility_scope.protocol !== BEDROCK_CONVERSE_PROTOCOL ||
            replay.compatibility_scope.adapter_version !== BEDROCK_CONVERSE_ADAPTER_VERSION ||
            (target?.model !== undefined &&
                replay.compatibility_scope.model !== undefined &&
                replay.compatibility_scope.model !== target.model)
        ) {
            throw new TypeError(`Bedrock Converse replay block ${replay.id} is outside its compatibility scope`);
        }
        assertRecord(replay.payload, `Bedrock replay payload ${replay.id}`);
        if (ownValue(replay.payload, 'type') !== 'bedrock_converse_content_order') {
            throw new TypeError(`Bedrock replay block ${replay.id} has an unsupported payload`);
        }
        const entries = ownValue(replay.payload, 'entries');
        const prefix = ownValue(replay.payload, 'prefix');
        if (prefix !== undefined && !preflightJsonInput(prefix).success) {
            throw new TypeError(`Bedrock replay block ${replay.id} has invalid prefix evidence`);
        }
        if (!Array.isArray(entries) || entries.length === 0) {
            throw new TypeError(`Bedrock replay block ${replay.id} has no ordered content`);
        }
        for (const [index, entry] of entries.entries()) {
            assertRecord(entry, `Bedrock replay entry ${replay.id}/${index}`);
            const kind = ownValue(entry, 'kind');
            if (kind === 'canonical' || kind === 'reasoning' || kind === 'structured_json_fragment') {
                assertNonemptyString(ownValue(entry, 'block_id'), `Bedrock replay block ID ${replay.id}/${index}`);
                if (!preflightJsonInput(ownValue(entry, 'native')).success) {
                    throw new TypeError(`Bedrock replay native evidence ${replay.id}/${index} is invalid`);
                }
                if (kind === 'reasoning') {
                    const signature = ownValue(entry, 'signature');
                    if (signature !== undefined) {
                        assertNonemptyString(signature, `Bedrock replay signature ${replay.id}/${index}`);
                    }
                }
            } else if (kind === 'redacted_reasoning') {
                assertNonemptyString(ownValue(entry, 'data'), `Bedrock replay data ${replay.id}/${index}`);
                if (!preflightJsonInput(ownValue(entry, 'native')).success) {
                    throw new TypeError(`Bedrock replay native evidence ${replay.id}/${index} is invalid`);
                }
            } else {
                throw new TypeError(`Bedrock replay entry ${replay.id}/${index} has unsupported kind ${String(kind)}`);
            }
        }
        const structuredOutput = ownValue(replay.payload, 'structured_output');
        if (structuredOutput !== undefined) parseStructuredOutputEvidence(structuredOutput);
        const requiresSignedReasoning = entries.some((entry) => {
            if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) return false;
            return (
                ownValue(entry, 'kind') === 'redacted_reasoning' ||
                (ownValue(entry, 'kind') === 'reasoning' && ownValue(entry, 'signature') !== undefined)
            );
        });
        const requiresServerToolUse = entries.some((entry) => {
            if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) return false;
            const native = ownValue(entry, 'native');
            if (typeof native !== 'object' || native === null || Array.isArray(native)) return false;
            const toolUse = ownValue(native, 'toolUse');
            if (typeof toolUse !== 'object' || toolUse === null || Array.isArray(toolUse)) return false;
            return bedrockToolUseType(toolUse, replay.id) === 'server_tool_use';
        });
        if ((requiresSignedReasoning || requiresServerToolUse) && replay.dependency_policy !== undefined) {
            throw new TypeError(`Bedrock replay block ${replay.id} cannot discard protected native evidence`);
        }
        if (requiresSignedReasoning && prefix === undefined) {
            throw new TypeError(`Bedrock replay block ${replay.id} is missing signed prefix evidence`);
        }
        if (requiresServerToolUse && prefix === undefined) {
            throw new TypeError(`Bedrock replay block ${replay.id} is missing protected server tool prefix evidence`);
        }
        if (
            requiresSignedReasoning &&
            target?.model !== undefined &&
            !bedrockConverseFamilyCapabilities(target.model).signed_reasoning_replay
        ) {
            const family = bedrockConverseFamilyCapabilities(target.model).family;
            throw new TypeError(
                `Bedrock Converse ${family} model ${target.model} cannot replay signed or redacted reasoning`,
            );
        }
        return { id: replay.id, payload: replay.payload as unknown as BedrockReplayPayload };
    }
    return undefined;
}

function importedMessageGroup(turn: ConversationTurn): string | undefined {
    const metadata = turn.metadata?.bedrock_converse;
    if (typeof metadata === 'object' && metadata !== null && !Array.isArray(metadata)) {
        const messageIndex = metadata.message_index;
        if (typeof messageIndex === 'string') return messageIndex;
    }
    if (turn.provenance.type !== 'imported' || turn.provenance.source !== BEDROCK_CONVERSE_PROTOCOL) return undefined;
    return turn.provenance.native_id?.value.match(/^messages\/(\d+)(?:\/content\/\d+)?$/)?.[1];
}

function appendMessage(
    messages: Message[],
    groups: Array<string | undefined>,
    message: Message,
    group: string | undefined,
): number {
    const prior = messages.at(-1);
    if (group !== undefined && groups.at(-1) === group && prior !== undefined && prior.role === message.role) {
        prior.content = [...(prior.content ?? []), ...(message.content ?? [])];
        return messages.length - 1;
    }
    messages.push(message);
    groups.push(group);
    return messages.length - 1;
}

function assetForBlock(document: ConversationDocument, block: { id: string; asset_id: string }): Asset {
    const asset = document.assets[block.asset_id];
    if (asset === undefined)
        throw new TypeError(`Canonical block ${block.id} references missing asset ${block.asset_id}`);
    return asset;
}

function compileResultContent(
    block: NestedToolResultContentBlock,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): ToolResultContentBlock {
    if (block.type === 'text') return { text: block.text };
    if (block.type === 'json') return { json: structuredClone(block.value) };
    if (block.type === 'image' || block.type === 'document' || block.type === 'video') {
        assertMediaCapability(block.type, target?.model);
        const native = assetToBedrockBlock(assetForBlock(document, block));
        if (block.type === 'image' && native.image !== undefined) return { image: native.image };
        if (block.type === 'document' && native.document !== undefined) return { document: native.document };
        if (block.type === 'video' && native.video !== undefined) return { video: native.video };
        throw new TypeError(`Canonical ${block.type} block ${block.id} references an incompatible asset`);
    }
    throw new TypeError(`Bedrock Converse cannot project canonical ${block.type} inside a tool result`);
}

function compileBlock(
    block: UserContentBlock | AgentContentBlock,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): ContentBlock | undefined {
    if (block.type === 'text') return { text: block.text };
    if (block.type === 'tool_call') {
        if (block.arguments.type === 'invalid') {
            throw new TypeError(`Bedrock Converse cannot project invalid arguments for call ${block.call_id}`);
        }
        if (block.executor === 'provider') {
            throw new TypeError(`Bedrock provider call ${block.call_id} requires protected native replay evidence`);
        }
        return {
            toolUse: {
                toolUseId: block.call_id,
                name: block.tool_name,
                input: toolArgumentsForModel(block.arguments),
            },
        };
    }
    if (block.type === 'image' || block.type === 'document' || block.type === 'audio' || block.type === 'video') {
        assertMediaCapability(block.type, target?.model);
        return assetToBedrockBlock(assetForBlock(document, block));
    }
    if (block.type === 'reasoning') return undefined;
    if (block.type === 'extension' && block.model_projection === 'excluded') return undefined;
    throw new TypeError(`Bedrock Converse cannot project canonical ${block.type} block ${block.id}`);
}

function structuredReplayText(entry: BedrockReplayStructuredJsonEntry, replayId: string): string {
    assertRecord(entry.native, `Bedrock structured replay native block ${replayId}`);
    assertAllowedKeys(entry.native, ['text'], `Bedrock structured replay native block ${replayId}`);
    const text = ownValue(entry.native, 'text');
    if (typeof text !== 'string') {
        throw new TypeError(`Bedrock structured replay block ${replayId} must preserve native text`);
    }
    return text;
}

function preserveReplayToolUseType(
    native: ContentBlock,
    evidence: JsonValue,
    replayId: string,
    block: AgentContentBlock,
): ContentBlock {
    if (!('toolUse' in native) || native.toolUse === undefined) return native;
    assertRecord(evidence, `Bedrock replay native block ${replayId}`);
    const toolUseEvidence = ownValue(evidence, 'toolUse');
    if (toolUseEvidence === undefined) return native;
    assertRecord(toolUseEvidence, `Bedrock replay tool use ${replayId}`);
    const type = bedrockToolUseType(toolUseEvidence, replayId);
    if (block.type !== 'tool_call' || (block.executor === 'provider') !== (type === 'server_tool_use')) {
        throw new TypeError(`Bedrock replay block ${replayId} no longer matches protected tool execution ownership`);
    }
    return type === undefined ? native : ({ toolUse: { ...native.toolUse, type } } as unknown as ContentBlock);
}

function compileReplayBlock(
    block: UserContentBlock | AgentContentBlock,
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): ContentBlock | undefined {
    if (block.type !== 'tool_call' || block.executor !== 'provider') return compileBlock(block, document, target);
    if (block.native_id?.protocol !== BEDROCK_CONVERSE_PROTOCOL) {
        throw new TypeError(`Bedrock Converse cannot replay foreign provider call ${block.call_id}`);
    }
    if (block.arguments.type === 'invalid') {
        throw new TypeError(`Bedrock Converse cannot replay invalid arguments for provider call ${block.call_id}`);
    }
    return {
        toolUse: {
            toolUseId: block.call_id,
            name: block.tool_name,
            input: toolArgumentsForModel(block.arguments),
            type: 'server_tool_use',
        },
    };
}

function compileAgentContent(
    turn: Extract<ConversationTurn, { kind: 'agent' }>,
    document: ConversationDocument,
    prefix: BedrockConverseConversation,
    target?: { provider?: string; model?: string },
): Array<{ native: ContentBlock; block?: AgentContentBlock }> {
    const replay = bedrockReplay(turn, target);
    const semantic = new Map(
        turn.blocks.filter((block) => block.type !== 'native_replay').map((block) => [block.id, block]),
    );
    if (replay === undefined) {
        return turn.blocks.flatMap((block) => {
            if (block.type === 'native_replay') return [];
            const native = compileBlock(block, document, target);
            return native === undefined ? [] : [{ native, block }];
        });
    }
    const hasProtectedReasoning = replay.payload.entries.some(
        (entry) => entry.kind === 'redacted_reasoning' || (entry.kind === 'reasoning' && entry.signature !== undefined),
    );
    const hasProtectedServerToolUse = replay.payload.entries.some((entry) => {
        if (entry.kind !== 'canonical') return false;
        assertRecord(entry.native, `Bedrock replay native block ${replay.id}`);
        const toolUse = ownValue(entry.native, 'toolUse');
        if (toolUse === undefined) return false;
        assertRecord(toolUse, `Bedrock replay tool use ${replay.id}`);
        return bedrockToolUseType(toolUse, replay.id) === 'server_tool_use';
    });
    if (hasProtectedReasoning || hasProtectedServerToolUse) {
        if (replay.payload.prefix === undefined) {
            throw new TypeError(`Bedrock replay block ${replay.id} is missing signed prefix evidence`);
        }
        assertExactReplayEvidence(prefix, replay.payload.prefix, replay.id, 'preceding conversation');
    }
    const structuredEntries = replay.payload.entries.filter(
        (entry): entry is BedrockReplayStructuredJsonEntry => entry.kind === 'structured_json_fragment',
    );
    if (structuredEntries.length > 0) {
        if (replay.payload.structured_output === undefined) {
            throw new TypeError(`Bedrock replay block ${replay.id} has structured fragments without evidence`);
        }
        assertStructuredOutputEvidence(
            turn,
            replay.payload.structured_output,
            structuredEntries.map((entry) => structuredReplayText(entry, replay.id)),
            replay.id,
        );
    } else if (replay.payload.structured_output !== undefined) {
        throw new TypeError(`Bedrock replay block ${replay.id} has structured evidence without fragments`);
    }
    const seen = new Set<string>();
    const projected = replay.payload.entries.map((entry): { native: ContentBlock; block?: AgentContentBlock } => {
        if (entry.kind === 'redacted_reasoning') {
            const native: ContentBlock = { reasoningContent: { redactedContent: base64ToBytes(entry.data) } };
            assertExactReplayEvidence(native, entry.native, replay.id, 'redacted reasoning evidence');
            return { native };
        }
        const block = semantic.get(entry.block_id);
        if (block === undefined) {
            throw new TypeError(`Bedrock replay block ${replay.id} references missing block ${entry.block_id}`);
        }
        if (seen.has(block.id) && entry.kind !== 'structured_json_fragment') {
            throw new TypeError(`Bedrock replay block ${replay.id} repeats block ${block.id}`);
        }
        seen.add(block.id);
        if (entry.kind === 'structured_json_fragment') {
            if (block.type !== 'json') {
                throw new TypeError(
                    `Bedrock replay block ${replay.id} references incompatible structured block ${block.id}`,
                );
            }
            return { native: { text: structuredReplayText(entry, replay.id) }, block };
        }
        if (entry.kind === 'reasoning') {
            if (block.type !== 'reasoning' || block.representation !== 'text') {
                throw new TypeError(
                    `Bedrock replay block ${replay.id} references incompatible reasoning block ${block.id}`,
                );
            }
            const native: ContentBlock = {
                reasoningContent: {
                    reasoningText: {
                        text: block.text,
                        ...(entry.signature === undefined ? {} : { signature: entry.signature }),
                    },
                },
            };
            assertExactReplayEvidence(native, entry.native, replay.id, `reasoning block ${block.id}`);
            return {
                native,
                block,
            };
        }
        const compiled = compileReplayBlock(block, document, target);
        if (compiled === undefined)
            throw new TypeError(`Bedrock replay block ${replay.id} references excluded block ${block.id}`);
        const native = preserveReplayToolUseType(compiled, entry.native, replay.id, block);
        assertExactReplayEvidence(native, entry.native, replay.id, `content block ${block.id}`);
        return { native, block };
    });
    const unreferenced = [...semantic.values()].find(
        (block) =>
            !seen.has(block.id) &&
            !(block.type === 'reasoning' && !hasProtectedReasoning) &&
            !(block.type === 'extension' && block.model_projection === 'excluded'),
    );
    if (unreferenced !== undefined) {
        throw new TypeError(`Bedrock replay block ${replay.id} does not order canonical block ${unreferenced.id}`);
    }
    return projected;
}

export function compileBedrockConverseConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): ReturnType<typeof projectBedrockConverseConversation> {
    return projectBedrockConverseConversation(document, target);
}

function projectBedrockConverseConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
    readOnlyCompatibilityProjection = false,
): CompiledBedrockConversation {
    const system: SystemContentBlock[] = [];
    const messages: Message[] = [];
    const groups: Array<string | undefined> = [];
    const mappings: NativeItemMapping[] = [];
    const calls = new Set<string>();
    const results = new Set<string>();

    const selectedTurns = selectedCanonicalTurns(document, {
        allow_interrupted_with_replay_protocol: BEDROCK_CONVERSE_PROTOCOL,
        allow_interrupted_with_complete_tool_calls: true,
    });
    for (const turn of selectedTurns) {
        if (!readOnlyCompatibilityProjection)
            assertProtectedReplayCompatibility(document, turn, BEDROCK_CONVERSE_PROTOCOL, target);
    }
    for (const turn of selectedTurns) {
        const replay = bedrockReplay(turn, target);
        if (turn.kind === 'program') {
            if (replay !== undefined)
                throw new TypeError(`Bedrock program turn ${turn.id} cannot contain native replay`);
            if (turn.authority === 'developer') {
                throw new TypeError(
                    `Bedrock Converse has no distinct developer-authority projection for turn ${turn.id}`,
                );
            }
            if (turn.authority === 'ordinary') {
                const content: ContentBlock[] = [];
                const projectedBlocks: AgentContentBlock[] = [];
                for (const block of turn.blocks) {
                    if (block.type === 'native_replay') continue;
                    const compiled = compileBlock(block, document, target);
                    if (compiled === undefined) continue;
                    content.push(compiled);
                    projectedBlocks.push(block);
                }
                if (content.length === 0) throw new TypeError(`Bedrock program turn ${turn.id} has no content`);
                const messageIndex = appendMessage(messages, groups, { role: 'user', content }, undefined);
                mappings.push({ canonical_id: turn.id, native_id: `messages/${messageIndex}`, kind: 'turn' });
                projectedBlocks.forEach((block, index) => {
                    mappings.push({
                        canonical_id: block.id,
                        native_id: `messages/${messageIndex}/content/${index}`,
                        kind: 'block',
                    });
                });
                continue;
            }
            const start = system.length;
            for (const block of turn.blocks) {
                if (block.type === 'extension' && block.model_projection === 'excluded') continue;
                if (block.type !== 'text') throw new TypeError('Bedrock Converse system context only supports text');
                system.push({ text: block.text });
                mappings.push({ canonical_id: block.id, native_id: `system/${system.length - 1}`, kind: 'block' });
            }
            if (system.length === start) throw new TypeError(`Bedrock program turn ${turn.id} has no content`);
            mappings.push({ canonical_id: turn.id, native_id: `system/${start}`, kind: 'turn' });
            continue;
        }

        if (turn.kind === 'tool') {
            if (replay !== undefined) throw new TypeError(`Bedrock tool turn ${turn.id} cannot contain native replay`);
            const result = turn.blocks[0];
            if (!calls.has(result.call_id)) {
                throw new TypeError(`Bedrock Converse tool result ${result.call_id} has no selected prior call`);
            }
            if (results.has(result.call_id)) {
                throw new TypeError(`Bedrock Converse tool call ${result.call_id} has multiple selected results`);
            }
            results.add(result.call_id);
            const content = result.content.map((block) => compileResultContent(block, document, target));
            if (content.length === 0) throw new TypeError(`Bedrock result ${result.call_id} has no content`);
            const messageIndex = appendMessage(
                messages,
                groups,
                {
                    role: 'user',
                    content: [
                        {
                            toolResult: {
                                toolUseId: result.call_id,
                                content,
                                ...(result.status === 'unknown' ||
                                turn.metadata?.bedrock_converse_tool_result_status_omitted === true
                                    ? {}
                                    : {
                                          status:
                                              result.status === 'cancelled' || result.status === 'denied'
                                                  ? 'error'
                                                  : result.status,
                                      }),
                            },
                        },
                    ],
                },
                importedMessageGroup(turn),
            );
            const contentIndex = (messages[messageIndex].content?.length ?? 1) - 1;
            mappings.push(
                { canonical_id: turn.id, native_id: `messages/${messageIndex}`, kind: 'turn' },
                {
                    canonical_id: result.id,
                    native_id: `messages/${messageIndex}/content/${contentIndex}`,
                    kind: 'block',
                },
                { canonical_id: result.call_id, native_id: result.call_id, kind: 'call' },
            );
            result.content.forEach((block, index) => {
                mappings.push({
                    canonical_id: block.id,
                    native_id: `messages/${messageIndex}/content/${contentIndex}/toolResult/content/${index}`,
                    kind: 'block',
                });
            });
            continue;
        }

        if (turn.kind !== 'agent' && replay !== undefined) {
            throw new TypeError(`Bedrock user turn ${turn.id} cannot contain native replay`);
        }
        const role = turn.kind === 'agent' ? 'assistant' : 'user';
        const projected =
            turn.kind === 'agent'
                ? compileAgentContent(turn, document, { ...(system.length === 0 ? {} : { system }), messages }, target)
                : turn.blocks.flatMap((block) => {
                      const native = compileBlock(block, document, target);
                      return native === undefined ? [] : [{ native, block }];
                  });
        const content = projected.map((entry) => entry.native);
        for (const { block } of projected) {
            if (block === undefined) continue;
            if (block.type === 'tool_call') {
                if (calls.has(block.call_id))
                    throw new TypeError(`Bedrock tool call ID ${block.call_id} is duplicated`);
                calls.add(block.call_id);
            }
        }
        if (
            content.length === 0 &&
            turn.kind === 'agent' &&
            turn.blocks.every(
                (block) =>
                    block.type === 'reasoning' ||
                    block.type === 'native_replay' ||
                    (block.type === 'extension' && block.model_projection === 'excluded'),
            )
        ) {
            continue;
        }
        if (content.length === 0) throw new TypeError(`Bedrock turn ${turn.id} has no projectable content`);
        const messageIndex = appendMessage(messages, groups, { role, content }, importedMessageGroup(turn));
        const start = (messages[messageIndex].content?.length ?? content.length) - content.length;
        mappings.push({ canonical_id: turn.id, native_id: `messages/${messageIndex}`, kind: 'turn' });
        projected.forEach(({ block }, index) => {
            if (block === undefined) return;
            mappings.push({
                canonical_id: block.id,
                native_id: `messages/${messageIndex}/content/${start + index}`,
                kind: 'block',
            });
            if (block.type === 'tool_call') {
                mappings.push({ canonical_id: block.call_id, native_id: block.call_id, kind: 'call' });
            }
        });
    }
    return { conversation: { messages, ...(system.length === 0 ? {} : { system }) }, mappings };
}

function nativeConversation(value: ConverseRequest | BedrockConverseConversation): BedrockConverseConversation {
    return {
        ...(value.messages === undefined ? {} : { messages: structuredClone(value.messages) }),
        ...(value.system === undefined ? {} : { system: structuredClone(value.system) }),
    };
}

export async function prepareBedrockConverseCanonicalState(input: {
    conversation: unknown;
    prompt: ConverseRequest;
    options: ExecutionOptions;
    provider: string;
}): Promise<Omit<PreparedBedrockConverseConversation, 'payload' | 'receipt' | 'diagnostics'>> {
    const runtime = resolveConversationRuntime(input.options);
    let document = parseCanonicalConversation(input.conversation);
    const toolDefinitions = await resolveCanonicalToolDefinitions(document, input.options.tools);
    if (document === undefined) {
        document = newCanonicalConversation(runtime);
        if (input.conversation !== undefined && input.conversation !== null) {
            assertRecord(input.conversation, 'Bedrock Converse conversation');
            const legacy = nativeConversation(input.conversation as unknown as ConverseRequest);
            if (!isBedrockConverseHistory(legacy, BEDROCK_CONVERSE_PROTOCOL)) {
                throw new TypeError('Conversation is neither canonical nor registered Bedrock Converse history');
            }
            document = (
                await importBedrockConverseHistory(legacy, {
                    conversation_id: runtime.conversation_id,
                    recorded_at: runtime.recorded_at,
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
    const priorCompiled = compileBedrockConverseConversation(document, target).conversation;
    const priorNativeMessageCount = priorCompiled.messages?.length ?? 0;
    const prompt = nativeConversation(input.prompt);
    const promptRecords = await importRecords(
        prompt,
        {
            conversation_id: document.id,
            recorded_at: runtime.recorded_at,
            tool_definitions: toolDefinitions,
            provider: input.provider,
            model: input.options.model,
        },
        'received',
        runtime.input_operation_id,
        new Set(
            document.turns.flatMap((turn) =>
                turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : [])),
            ),
        ),
        new Set(
            document.turns.flatMap((turn) =>
                turn.blocks.flatMap((block) => (block.type === 'tool_result' ? [block.call_id] : [])),
            ),
        ),
    );
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: promptRecords.turns,
            assets: promptRecords.assets,
            context_entries: promptRecords.context_entries,
            item_mappings: promptRecords.mappings,
            execution_receipts: promptRecords.execution_receipts,
        },
        { ...runtime, conversation_id: document.id },
        input.options.tools,
        bedrockConverseJsonValue(prompt),
    );
    const acceptedResponse = acceptedCanonicalResponse(appended.document, runtime.response_operation_id);
    if (
        acceptedResponse !== undefined &&
        (acceptedResponse.generation.request_id !== runtime.request_id ||
            acceptedResponse.generation.provider !== input.provider ||
            acceptedResponse.generation.protocol !== BEDROCK_CONVERSE_PROTOCOL ||
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
    const compiled = compileBedrockConverseConversation(requestDocument, target);
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

export async function finalizeBedrockConversePreparedRequest(
    state: Omit<PreparedBedrockConverseConversation, 'payload' | 'receipt' | 'diagnostics'>,
    payload: ConverseRequest,
): Promise<PreparedBedrockConverseConversation> {
    assertNonemptyString(payload.modelId, 'Bedrock Converse payload modelId');
    const compiled = compileBedrockConverseConversation(state.document, {
        provider: state.provider,
        model: state.requested_model,
    });
    const targetOptions = canonicalToolSelectionTargetOptions(undefined, state.response_selection_policy);
    const receipt = await createRequestReceipt(
        state.document,
        state.runtime,
        {
            provider: state.provider,
            protocol: BEDROCK_CONVERSE_PROTOCOL,
            model: state.requested_model,
            adapter_version: BEDROCK_CONVERSE_ADAPTER_VERSION,
            ...(targetOptions === undefined ? {} : { options: targetOptions }),
        },
        bedrockConverseJsonValue(payload),
        compiled.mappings,
        state.tool_definitions,
    );
    return { ...state, payload, receipt, diagnostics: [] };
}

function safeUsageNumber(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function safeUsageSum(...values: Array<number | undefined>): number | undefined {
    if (values.some((value) => value === undefined)) return undefined;
    const total = (values as number[]).reduce((sum, value) => sum + value, 0);
    return Number.isSafeInteger(total) ? total : undefined;
}

export function bedrockConverseGenerationUsage(usage: TokenUsage | undefined): GenerationUsage | undefined {
    if (usage === undefined) return undefined;
    const inputNew = safeUsageNumber(usage.inputTokens);
    const reportedCacheRead = safeUsageNumber(usage.cacheReadInputTokens);
    const reportedCacheWrite = safeUsageNumber(usage.cacheWriteInputTokens);
    const cacheRead = inputNew === undefined ? reportedCacheRead : (reportedCacheRead ?? 0);
    const cacheWrite = inputNew === undefined ? reportedCacheWrite : (reportedCacheWrite ?? 0);
    const output = safeUsageNumber(usage.outputTokens);
    const input = inputNew === undefined ? undefined : safeUsageSum(inputNew, cacheRead, cacheWrite);
    const total = input === undefined || output === undefined ? undefined : safeUsageSum(input, output);
    const basis = 'bedrock_converse_tokens';
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(inputNew === undefined ? {} : { input_new_tokens: inputNew }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(total === undefined ? {} : { total_tokens: total }),
        ...(cacheRead === undefined ? {} : { cache_read_tokens: cacheRead }),
        ...(cacheWrite === undefined ? {} : { cache_write_tokens: cacheWrite }),
        accounting_provenance: {
            ...(input === undefined ? {} : { input_tokens: { method: 'derived' as const, accounting_basis: basis } }),
            ...(inputNew === undefined
                ? {}
                : { input_new_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(output === undefined
                ? {}
                : { output_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(total === undefined ? {} : { total_tokens: { method: 'derived' as const, accounting_basis: basis } }),
            ...(cacheRead === undefined
                ? {}
                : {
                      cache_read_tokens: {
                          method: reportedCacheRead === undefined ? ('derived' as const) : ('reported' as const),
                          accounting_basis: basis,
                      },
                  }),
            ...(cacheWrite === undefined
                ? {}
                : {
                      cache_write_tokens: {
                          method: reportedCacheWrite === undefined ? ('derived' as const) : ('reported' as const),
                          accounting_basis: basis,
                      },
                  }),
        },
        ...(inputNew === undefined
            ? {}
            : { input_partition: { type: 'complete_disjoint' as const, cache_write_bucket: 'included' as const } }),
        reported_usage: [
            {
                source: 'provider',
                protocol: BEDROCK_CONVERSE_PROTOCOL,
                accounting_basis: basis,
                payload: bedrockConverseJsonValue(usage),
            },
        ],
    };
}

export async function decodeBedrockConverseCanonicalResponse(
    response: ConverseResponse,
    prepared: PreparedBedrockConverseConversation,
    structuredOutput?: CanonicalStructuredOutput,
): Promise<DecodedConversationResponse> {
    assertNonemptyString(response.stopReason, 'Bedrock Converse response stopReason');
    const message = response.output?.message;
    if (message === undefined) throw new Error('Bedrock Converse response has no output message');
    if (message.role !== 'assistant') throw new Error('Bedrock Converse response output role must be assistant');
    if (!Array.isArray(message.content) || message.content.length === 0) {
        throw new Error('Bedrock Converse response output must contain content');
    }
    const messageContent = message.content;
    messageContent.forEach((block, index) => {
        assertContentBlock(block, message.role, `output/content/${index}`);
    });
    const completedAt = prepared.runtime.completed_at ?? new Date().toISOString();
    const records = await messageRecords({
        message,
        index: 0,
        scope: prepared.runtime.response_operation_id,
        options: {
            conversation_id: prepared.document.id,
            recorded_at: completedAt,
            tool_definitions: prepared.tool_definitions,
            provider: prepared.provider,
            model: prepared.requested_model,
        },
        source: 'received',
        prefix: nativeConversation(prepared.native_conversation),
        prefix_dependencies: replayDependenciesForMappings(prepared.receipt.item_mappings),
    });
    const received = records.turns[0];
    if (received?.kind !== 'agent') throw new Error('Bedrock Converse response did not decode to an agent turn');
    const turn: ConversationTurn = {
        ...received,
        id: prepared.response_turn_id,
        status:
            response.stopReason === 'max_tokens' || response.stopReason === 'model_context_window_exceeded'
                ? 'interrupted'
                : response.stopReason === 'guardrail_intervened' || response.stopReason === 'content_filtered'
                  ? 'failed'
                  : 'completed',
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
                          turn_ids: uniqueIds(
                              block.dependencies.turn_ids.map((id) =>
                                  id === received.id ? prepared.response_turn_id : id,
                              ),
                          ),
                          request_ids: [prepared.receipt.request_id],
                      },
                  },
        ),
    } as ConversationTurn;
    const generation: ExecutedGeneration = {
        ...(await createExecutedGeneration({
            id: prepared.generation_id,
            runtime: prepared.runtime,
            receipt: prepared.receipt,
            provider: prepared.provider,
            protocol: BEDROCK_CONVERSE_PROTOCOL,
            adapter_version: BEDROCK_CONVERSE_ADAPTER_VERSION,
            requested_model: prepared.requested_model,
            provider_response_id: (response as ConverseResponse & { $metadata?: { requestId?: string } }).$metadata
                ?.requestId,
            finish_reason: response.stopReason,
            usage: bedrockConverseGenerationUsage(response.usage),
        })),
        status:
            response.stopReason === 'max_tokens' || response.stopReason === 'model_context_window_exceeded'
                ? 'cancelled'
                : response.stopReason === 'guardrail_intervened' || response.stopReason === 'content_filtered'
                  ? 'failed'
                  : 'completed',
    };
    const decoded: DecodedConversationResponse = {
        turns: [turn],
        generation,
        assets: records.assets.map((asset) => ({
            ...asset,
            provenance: { type: 'generated', generation_id: generation.id, source_turn_id: turn.id },
        })),
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(bedrockConverseJsonValue(response)),
    };
    if (structuredOutput === undefined) return decoded;
    return normalizeDecodedStructuredOutput(decoded, structuredOutput, async ({ replay_blocks, binding }) => {
        if (replay_blocks.length > 1) {
            throw new TypeError(`Bedrock structured output turn ${turn.id} has multiple replay blocks`);
        }
        const current = replay_blocks[0];
        const base: NativeReplayBlock =
            current ??
            ({
                id: await entityId('replay', prepared.runtime.response_operation_id, 'structured-output'),
                type: 'native_replay',
                adapter: BEDROCK_CONVERSE_ADAPTER_VERSION,
                protocol: BEDROCK_CONVERSE_PROTOCOL,
                compatibility_scope: {
                    provider: prepared.provider,
                    protocol: BEDROCK_CONVERSE_PROTOCOL,
                    model: prepared.requested_model,
                    adapter_version: BEDROCK_CONVERSE_ADAPTER_VERSION,
                },
                payload: {
                    type: 'bedrock_converse_content_order',
                    entries: [],
                },
                dependencies: {
                    turn_ids: [turn.id],
                    block_ids: binding.source_block_ids,
                    call_ids: [],
                    request_ids: [prepared.receipt.request_id],
                },
            } satisfies NativeReplayBlock);
        const replay = remapStructuredOutputReplayDependencies(base, binding);
        const payload = replay.payload as unknown as BedrockReplayPayload;
        const sources = new Set(binding.source_block_ids);
        let sourceIndex = 0;
        const entries =
            current === undefined
                ? messageContent.flatMap((native): BedrockReplayEntry[] => {
                      if (native.text === undefined) return [];
                      const sourceId = binding.source_block_ids[sourceIndex];
                      const expected = binding.source_texts[sourceIndex];
                      sourceIndex += 1;
                      if (sourceId === undefined || native.text !== expected) {
                          throw new TypeError(
                              'Bedrock structured output replay does not match its source text partition',
                          );
                      }
                      return [
                          {
                              kind: 'structured_json_fragment',
                              block_id: binding.block_id,
                              native: bedrockConverseJsonValue(native),
                          },
                      ];
                  })
                : payload.entries.map((entry): BedrockReplayEntry => {
                      if (entry.kind !== 'canonical' || !sources.has(entry.block_id)) return entry;
                      const expected = binding.source_texts[sourceIndex];
                      sourceIndex += 1;
                      if (
                          structuredReplayText({ ...entry, kind: 'structured_json_fragment' }, replay.id) !== expected
                      ) {
                          throw new TypeError(
                              'Bedrock structured output replay does not match its source text partition',
                          );
                      }
                      return { ...entry, kind: 'structured_json_fragment', block_id: binding.block_id };
                  });
        if (sourceIndex !== binding.source_texts.length) {
            throw new TypeError('Bedrock structured output replay is missing source text partitions');
        }
        return [
            {
                ...replay,
                payload: {
                    ...payload,
                    entries,
                    structured_output: structuredOutputEvidence(binding),
                } as unknown as JsonValue,
            },
        ];
    });
}

export function appendBedrockConverseCanonicalResponse(
    prepared: PreparedBedrockConverseConversation,
    decoded: DecodedConversationResponse,
): ConversationDocument {
    return appendCanonicalDecodedResponse(prepared, decoded, {
        operation_id: prepared.runtime.response_operation_id,
        recorded_at: decoded.generation.timestamps.recorded_at,
    }).document;
}

export function exportLegacyBedrockConverseConversation(
    document: ConversationDocument,
    target?: { provider?: string; model?: string },
): BedrockConverseConversation {
    return projectBedrockConverseConversation(parseConversationDocument(document), target, true).conversation;
}
