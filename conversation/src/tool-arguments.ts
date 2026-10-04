import { type ResolveConversationAsset, readBoundedConversationAsset } from './asset-resolution.js';
import { hashContentBytes, hashUtf8Content } from './content-integrity.js';
import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import { fingerprintJson } from './runtime.js';
import { JsonPathSchema } from './schemas/content.js';
import type {
    Asset,
    ConversationDocument,
    ExternalizedToolArguments,
    JsonObject,
    JsonPath,
    JsonValue,
    NativeReplayBlock,
    OperationReceipt,
    ToolArguments,
    ToolCallBlock,
} from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

const DEFAULT_MAX_HYDRATED_TOOL_ARGUMENT_BYTES = 32 * 1024 * 1024;
const MAX_TOOL_ARGUMENT_HYDRATION_CHUNKS = 4_096;

export interface PreparedToolArgumentExternalization {
    call_id: string;
    input_path: JsonPath;
    content: string;
    content_hash: string;
    byte_length: number;
    exact_arguments_hash: string;
    execution_base: JsonObject;
    replay_archives: PreparedInvalidatedReplayArchive[];
}

export interface PreparedInvalidatedReplayArchive {
    replay_block_id: string;
    content: string;
    content_hash: string;
    byte_length: number;
}

export interface InvalidatedReplayArchiveInput {
    replay_block_id: string;
    asset: Asset;
}

export interface ExternalizeToolArgumentsOptions {
    operation_id: string;
    expected_revision: number;
    recorded_at: string;
    call_id: string;
    input_path: JsonPath;
    model_value: JsonObject;
    exact_arguments_hash: string;
    asset: Asset;
    replay_archives?: InvalidatedReplayArchiveInput[];
}

export interface ExternalizeToolArgumentsResult {
    document: ConversationDocument;
    applied: boolean;
    call_id: string;
    asset_id: string;
}

export interface HydrateToolArgumentsOptions {
    max_bytes?: number;
}

export type ResolveToolArgumentTextAsset = ResolveConversationAsset;

interface LocatedToolCall {
    call: ToolCallBlock;
    turn_index: number;
    block_index: number;
}

function stableJson(value: unknown): string {
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
    return `{${Object.keys(value as object)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${stableJson((value as Record<string, unknown>)[key])}`)
        .join(',')}}`;
}

function ownRecordValue<T>(record: Record<string, T>, key: string): T | undefined {
    return Object.hasOwn(record, key) ? record[key] : undefined;
}

function recordWith<T>(record: Record<string, T>, key: string, value: T): Record<string, T> {
    return Object.fromEntries([...Object.entries(record), [key, value]]);
}

function compareIdentifier(first: string, second: string): number {
    return first < second ? -1 : first > second ? 1 : 0;
}

function checkedNextRevision(revision: number): number {
    const next = revision + 1;
    if (!Number.isSafeInteger(next)) throw new RangeError('Conversation revision exceeds Number.MAX_SAFE_INTEGER');
    return next;
}

function assertSafeByteLimit(maxBytes: number): void {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
        throw new RangeError('max_bytes must be a positive safe integer');
    }
}

function validatedPath(input: JsonPath): JsonPath {
    const parsed = JsonPathSchema.safeParse(input);
    if (!parsed.success) {
        throw new ConversationValidationError(
            'Tool argument hydration path failed schema validation',
            diagnosticsFromZodError(parsed.error),
        );
    }
    return structuredClone(input);
}

function findToolCall(document: ConversationDocument, callId: string): LocatedToolCall {
    let found: LocatedToolCall | undefined;
    for (let turnIndex = 0; turnIndex < document.turns.length; turnIndex += 1) {
        const turn = document.turns[turnIndex];
        for (let blockIndex = 0; blockIndex < turn.blocks.length; blockIndex += 1) {
            const block = turn.blocks[blockIndex];
            if (block.type !== 'tool_call' || block.call_id !== callId) continue;
            if (found !== undefined) throw new Error(`Tool call ${callId} is duplicated`);
            found = { call: block, turn_index: turnIndex, block_index: blockIndex };
        }
    }
    if (found === undefined) throw new Error(`Tool call ${callId} does not exist`);
    return found;
}

function childAt(value: JsonValue, segment: string | number): JsonValue | undefined {
    if (typeof segment === 'number') {
        return Array.isArray(value) && segment < value.length ? value[segment] : undefined;
    }
    return value !== null && typeof value === 'object' && !Array.isArray(value) && Object.hasOwn(value, segment)
        ? value[segment]
        : undefined;
}

function valueAtPath(value: JsonValue, path: JsonPath): JsonValue | undefined {
    let current: JsonValue | undefined = value;
    for (const segment of path) {
        if (current === undefined) return undefined;
        current = childAt(current, segment);
    }
    return current;
}

function parentAtPath(value: JsonValue, path: JsonPath): { parent: JsonObject | JsonValue[]; leaf: string | number } {
    let current: JsonValue = value;
    for (const segment of path.slice(0, -1)) {
        const child = childAt(current, segment);
        if (child === undefined) throw new Error(`Tool argument hydration path ${JSON.stringify(path)} does not exist`);
        current = child;
    }
    if (current === null || typeof current !== 'object') {
        throw new Error(`Tool argument hydration path ${JSON.stringify(path)} has a scalar parent`);
    }
    const leaf = path.at(-1);
    if (leaf === undefined) throw new Error('Tool argument hydration path must not be empty');
    if (typeof leaf === 'number') {
        if (!Array.isArray(current)) {
            throw new Error(`Tool argument hydration path ${JSON.stringify(path)} has the wrong segment kind`);
        }
        if (leaf >= current.length) throw new Error(`Tool argument hydration array index ${leaf} is out of range`);
    } else if (Array.isArray(current)) {
        throw new Error(`Tool argument hydration path ${JSON.stringify(path)} has the wrong segment kind`);
    }
    return { parent: current as JsonObject | JsonValue[], leaf };
}

function executionBase(value: JsonObject, path: JsonPath): JsonObject {
    const base = structuredClone(value);
    const { parent, leaf } = parentAtPath(base, path);
    if (typeof leaf === 'number') {
        (parent as JsonValue[])[leaf] = null;
    } else {
        delete (parent as JsonObject)[leaf];
    }
    return base;
}

function setPath(value: JsonObject, path: JsonPath, content: string): void {
    const { parent, leaf } = parentAtPath(value, path);
    if (typeof leaf === 'number') {
        (parent as JsonValue[])[leaf] = content;
    } else {
        (parent as JsonObject)[leaf] = content;
    }
}

function pathKey(path: JsonPath): string {
    return JSON.stringify(path);
}

function pathsOverlap(first: JsonPath, second: JsonPath): boolean {
    const shared = Math.min(first.length, second.length);
    for (let index = 0; index < shared; index += 1) {
        if (first[index] !== second[index]) return false;
    }
    return true;
}

function assertDistinctHydrationPaths(argumentsValue: ExternalizedToolArguments): void {
    const paths = argumentsValue.hydration.map((entry) => validatedPath(entry.input_path));
    const keys = new Set<string>();
    for (let index = 0; index < paths.length; index += 1) {
        const path = paths[index];
        const key = pathKey(path);
        if (keys.has(key)) throw new Error(`Tool argument hydration path ${key} is duplicated`);
        keys.add(key);
        for (let priorIndex = 0; priorIndex < index; priorIndex += 1) {
            if (pathsOverlap(paths[priorIndex], path)) {
                throw new Error(`Tool argument hydration paths ${pathKey(paths[priorIndex])} and ${key} overlap`);
            }
        }
    }
}

export async function hashUtf8Text(content: string): Promise<{ content_hash: string; byte_length: number }> {
    return hashUtf8Content(content);
}

function assertModelValue(value: JsonObject): void {
    const preflight = preflightJsonInput(value);
    if (!preflight.success) {
        throw new ConversationValidationError(
            'Tool argument model projection failed JSON preflight',
            preflight.diagnostics,
        );
    }
}

function replayBlocksInvalidatedByExternalization(
    document: ConversationDocument,
    located: LocatedToolCall,
): NativeReplayBlock[] {
    const ownerTurn = document.turns[located.turn_index];
    const call = located.call;
    const invalidated: NativeReplayBlock[] = [];
    const turns = [
        ...document.turns,
        ...Object.values(document.compactions).flatMap((compaction) => compaction.replacement_turns),
    ];
    for (const turn of turns) {
        for (const block of turn.blocks) {
            if (
                block.type === 'native_replay' &&
                (block.dependencies.call_ids.includes(call.call_id) ||
                    block.dependencies.block_ids.includes(call.id) ||
                    block.dependencies.turn_ids.includes(ownerTurn.id))
            ) {
                if (block.dependency_policy === 'discard_on_dependency_change') {
                    invalidated.push(block);
                    continue;
                }
                throw new Error(
                    `Tool call ${call.call_id} is protected by native replay ${block.id} and cannot be externalized`,
                );
            }
        }
    }
    return invalidated.sort((first, second) => compareIdentifier(first.id, second.id));
}

export function toolArgumentsForModel(argumentsValue: ToolArguments): JsonValue {
    if (argumentsValue.type === 'invalid') throw new Error('Invalid tool arguments cannot be projected to a model');
    return structuredClone(argumentsValue.type === 'json' ? argumentsValue.value : argumentsValue.model_value);
}

export async function prepareToolArgumentExternalization(
    input: ConversationDocument,
    callId: string,
    inputPath: JsonPath,
    options: { max_bytes?: number } = {},
): Promise<PreparedToolArgumentExternalization> {
    const document = parseConversationDocument(input);
    const path = validatedPath(inputPath);
    const maxBytes = options.max_bytes ?? DEFAULT_MAX_HYDRATED_TOOL_ARGUMENT_BYTES;
    assertSafeByteLimit(maxBytes);
    const located = findToolCall(document, callId);
    const { call } = located;
    const replayBlocks = replayBlocksInvalidatedByExternalization(document, located);
    if (call.arguments.type !== 'json') throw new Error(`Tool call ${callId} does not have inline JSON arguments`);
    if (
        call.arguments.value === null ||
        typeof call.arguments.value !== 'object' ||
        Array.isArray(call.arguments.value)
    ) {
        throw new Error(`Tool call ${callId} arguments must be a JSON object to externalize a field`);
    }
    const content = valueAtPath(call.arguments.value, path);
    if (typeof content !== 'string') {
        throw new Error(`Tool call ${callId} argument path ${JSON.stringify(path)} is not a string`);
    }
    const hashed = await hashUtf8Text(content);
    if (hashed.byte_length > maxBytes) {
        throw new RangeError(`Tool argument content exceeds max_bytes (${hashed.byte_length} > ${maxBytes})`);
    }
    const replayArchives: PreparedInvalidatedReplayArchive[] = [];
    let replayBytes = 0;
    for (const replay of replayBlocks) {
        const replayContent = JSON.stringify(replay);
        const replayHash = await hashUtf8Text(replayContent);
        replayBytes += replayHash.byte_length;
        if (replayBytes > maxBytes) {
            throw new RangeError(`Discardable replay archive exceeds max_bytes (${replayBytes} > ${maxBytes})`);
        }
        replayArchives.push({
            replay_block_id: replay.id,
            content: replayContent,
            ...replayHash,
        });
    }
    return {
        call_id: callId,
        input_path: path,
        content,
        ...hashed,
        exact_arguments_hash: await fingerprintJson(call.arguments.value),
        execution_base: executionBase(call.arguments.value, path),
        replay_archives: replayArchives,
    };
}

function assertExactRetry(
    document: ConversationDocument,
    options: ExternalizeToolArgumentsOptions,
    receipt: OperationReceipt,
): void {
    const suppliedArchives = [...(options.replay_archives ?? [])].sort((first, second) =>
        compareIdentifier(first.replay_block_id, second.replay_block_id),
    );
    const expectedAssetIds = [options.asset.id, ...suppliedArchives.map((archive) => archive.asset.id)];
    if (
        receipt.accepted_asset_ids?.length !== expectedAssetIds.length ||
        receipt.accepted_asset_ids.some((id, index) => id !== expectedAssetIds[index])
    ) {
        throw new Error(`Conversation operation retry cannot resolve accepted asset ${options.asset.id}`);
    }
    const retainedAsset = ownRecordValue(document.assets, options.asset.id);
    if (retainedAsset === undefined || stableJson(retainedAsset) !== stableJson(options.asset)) {
        throw new Error(`Conversation operation retry changes accepted asset ${options.asset.id}`);
    }
    const { call } = findToolCall(document, options.call_id);
    if (
        call.arguments.type !== 'externalized_json' ||
        call.arguments.exact_arguments_hash !== options.exact_arguments_hash ||
        stableJson(call.arguments.model_value) !== stableJson(options.model_value) ||
        call.arguments.hydration.length !== 1 ||
        call.arguments.hydration[0].asset_id !== options.asset.id ||
        stableJson(call.arguments.hydration[0].input_path) !== stableJson(options.input_path)
    ) {
        throw new Error(`Conversation operation retry changes externalized tool call ${options.call_id}`);
    }
    const retainedArchives = call.arguments.invalidated_replay_archives ?? [];
    if (
        retainedArchives.length !== suppliedArchives.length ||
        retainedArchives.some((archive, index) => {
            const supplied = suppliedArchives[index];
            return (
                supplied === undefined ||
                archive.replay_block_id !== supplied.replay_block_id ||
                archive.asset_id !== supplied.asset.id ||
                archive.content_hash !== supplied.asset.content_hash
            );
        })
    ) {
        throw new Error(`Conversation operation retry changes invalidated replay archives for ${options.call_id}`);
    }
    for (const [index, archive] of retainedArchives.entries()) {
        const supplied = suppliedArchives[index];
        const retainedArchiveAsset = ownRecordValue(document.assets, archive.asset_id);
        if (
            supplied === undefined ||
            retainedArchiveAsset === undefined ||
            stableJson(retainedArchiveAsset) !== stableJson(supplied.asset)
        ) {
            throw new Error(`Conversation operation retry changes replay archive ${archive.replay_block_id}`);
        }
        const replayId = archive.replay_block_id;
        if (
            document.turns.some((turn) => turn.blocks.some((block) => block.id === replayId)) ||
            Object.values(document.compactions).some((compaction) =>
                compaction.replacement_turns.some((turn) => turn.blocks.some((block) => block.id === replayId)),
            )
        ) {
            throw new Error(`Conversation operation retry retained invalidated replay block ${replayId}`);
        }
    }
}

async function externalizationPayloadFingerprint(
    options: ExternalizeToolArgumentsOptions,
    inputPath: JsonPath,
    replayArchives: readonly InvalidatedReplayArchiveInput[],
): Promise<string> {
    return fingerprintJson({
        call_id: options.call_id,
        input_path: inputPath,
        model_value: options.model_value,
        exact_arguments_hash: options.exact_arguments_hash,
        asset: options.asset,
        replay_archives: replayArchives.map((archive) => ({
            replay_block_id: archive.replay_block_id,
            asset: archive.asset,
        })),
    });
}

/** Authenticate the original accepted arguments against a later, retained externalization operation. */
export async function assertHistoricalToolArgumentExternalization(
    document: ConversationDocument,
    call: ToolCallBlock,
    originalArguments: JsonObject,
    acceptedRevision: number,
    generationId: string,
): Promise<void> {
    const externalized = call.arguments;
    if (externalized.type !== 'externalized_json' || externalized.hydration.length !== 1) {
        throw new Error(`Accepted tool call ${call.call_id} has no exact externalization witness`);
    }
    const hydration = externalized.hydration[0];
    const path = validatedPath(hydration.input_path);
    const content = valueAtPath(originalArguments, path);
    const asset = ownRecordValue(document.assets, hydration.asset_id);
    if (
        typeof content !== 'string' ||
        !asset ||
        asset.kind !== 'text' ||
        asset.storage.type !== 'external' ||
        (asset.provenance.type === 'generated' && asset.provenance.generation_id !== generationId) ||
        (await fingerprintJson(originalArguments)) !== externalized.exact_arguments_hash ||
        stableJson(executionBase(originalArguments, path)) !== stableJson(externalized.value)
    ) {
        throw new Error(`Accepted tool call ${call.call_id} changed its original arguments`);
    }
    const integrity = await hashUtf8Text(content);
    if (
        integrity.content_hash !== hydration.content_hash ||
        asset.content_hash !== integrity.content_hash ||
        asset.byte_length !== integrity.byte_length
    ) {
        throw new Error(`Accepted tool call ${call.call_id} changed its archived argument bytes`);
    }
    const replayArchives = (externalized.invalidated_replay_archives ?? []).map((reference) => {
        const archived = ownRecordValue(document.assets, reference.asset_id);
        if (archived?.content_hash !== reference.content_hash || archived.storage.type !== 'external') {
            throw new Error(`Accepted tool call ${call.call_id} changed its replay archive`);
        }
        return { replay_block_id: reference.replay_block_id, asset: archived };
    });
    const expectedAssetIds = [asset.id, ...replayArchives.map((archive) => archive.asset.id)];
    const candidates = Object.values(document.operation_receipts).filter(
        (receipt) =>
            receipt.conversation_id === document.id &&
            receipt.base_revision >= acceptedRevision &&
            receipt.accepted_asset_ids?.length === expectedAssetIds.length &&
            receipt.accepted_asset_ids.every((id, index) => id === expectedAssetIds[index]),
    );
    if (candidates.length !== 1) throw new Error(`Accepted tool call ${call.call_id} lacks one exact archive receipt`);
    const receipt = candidates[0];
    const expectedFingerprint = await externalizationPayloadFingerprint(
        {
            operation_id: receipt.id,
            expected_revision: receipt.base_revision,
            recorded_at: receipt.recorded_at,
            call_id: call.call_id,
            input_path: path,
            model_value: externalized.model_value,
            exact_arguments_hash: externalized.exact_arguments_hash,
            asset,
            replay_archives: replayArchives,
        },
        path,
        replayArchives,
    );
    if (
        receipt.payload_fingerprint !== expectedFingerprint ||
        receipt.result_revision !== receipt.base_revision + 1 ||
        receipt.operation_kind !== undefined ||
        receipt.accepted_turn_ids?.length ||
        receipt.accepted_generation_ids?.length
    ) {
        throw new Error(`Accepted tool call ${call.call_id} changed its archive operation`);
    }
}

export async function externalizeToolCallArguments(
    input: ConversationDocument,
    options: ExternalizeToolArgumentsOptions,
): Promise<ExternalizeToolArgumentsResult> {
    const optionsPreflight = preflightJsonInput(options);
    if (!optionsPreflight.success) {
        throw new ConversationValidationError(
            'Tool argument externalization options failed JSON preflight',
            optionsPreflight.diagnostics,
        );
    }
    const document = parseConversationDocument(input);
    const inputPath = validatedPath(options.input_path);
    assertModelValue(options.model_value);
    if (!Number.isSafeInteger(options.expected_revision) || options.expected_revision < 0) {
        throw new RangeError('expected_revision must be a nonnegative safe integer');
    }
    const priorReceipt = ownRecordValue(document.operation_receipts, options.operation_id);
    if (priorReceipt !== undefined) {
        const retained = findToolCall(document, options.call_id).call;
        if (retained.arguments.type !== 'externalized_json') {
            throw new Error(`Conversation operation retry cannot resolve externalized tool call ${options.call_id}`);
        }
        const payloadFingerprint = await externalizationPayloadFingerprint(
            options,
            inputPath,
            [...(options.replay_archives ?? [])].sort((first, second) =>
                compareIdentifier(first.replay_block_id, second.replay_block_id),
            ),
        );
        if (priorReceipt.payload_fingerprint !== payloadFingerprint) {
            throw new Error(`Conversation operation ${options.operation_id} was already used with a different payload`);
        }
        assertExactRetry(document, options, priorReceipt);
        return { document, applied: false, call_id: options.call_id, asset_id: options.asset.id };
    }
    if (options.expected_revision !== document.revision) {
        throw new Error(
            `Conversation revision conflict: expected ${options.expected_revision}, received ${document.revision}`,
        );
    }
    if (ownRecordValue(document.assets, options.asset.id) !== undefined) {
        throw new Error(`Asset ${options.asset.id} already exists`);
    }
    if (options.asset.kind !== 'text' || options.asset.mime_type !== 'text/plain') {
        throw new Error('Externalized text tool arguments require a text/plain asset');
    }
    if (options.asset.content_hash === undefined || options.asset.byte_length === undefined) {
        throw new Error('Externalized tool argument asset requires content_hash and byte_length');
    }
    const located = findToolCall(document, options.call_id);
    const invalidatedReplayBlocks = replayBlocksInvalidatedByExternalization(document, located);
    const suppliedReplayArchives = [...(options.replay_archives ?? [])].sort((first, second) =>
        compareIdentifier(first.replay_block_id, second.replay_block_id),
    );
    if (suppliedReplayArchives.length !== invalidatedReplayBlocks.length) {
        throw new Error(`Tool call ${options.call_id} does not archive every invalidated replay block`);
    }
    const replayArchiveRefs: NonNullable<ExternalizedToolArguments['invalidated_replay_archives']> = [];
    const replayArchiveAssets: Asset[] = [];
    for (let index = 0; index < invalidatedReplayBlocks.length; index += 1) {
        const replay = invalidatedReplayBlocks[index];
        const supplied = suppliedReplayArchives[index];
        if (supplied === undefined || supplied.replay_block_id !== replay.id) {
            throw new Error(`Tool call ${options.call_id} replay archive identity does not match ${replay.id}`);
        }
        const archiveAsset = supplied.asset;
        const serialized = JSON.stringify(replay);
        const archiveHash = await hashUtf8Text(serialized);
        if (
            archiveAsset.kind !== 'document' ||
            archiveAsset.mime_type !== 'application/json' ||
            archiveAsset.storage.type !== 'external' ||
            archiveAsset.content_hash !== archiveHash.content_hash ||
            archiveAsset.byte_length !== archiveHash.byte_length
        ) {
            throw new Error(`Tool call ${options.call_id} replay archive ${replay.id} is not an exact durable copy`);
        }
        if (
            archiveAsset.id === options.asset.id ||
            replayArchiveAssets.some((asset) => asset.id === archiveAsset.id) ||
            ownRecordValue(document.assets, archiveAsset.id) !== undefined
        ) {
            throw new Error(`Replay archive asset ${archiveAsset.id} already exists`);
        }
        replayArchiveAssets.push(archiveAsset);
        replayArchiveRefs.push({
            replay_block_id: replay.id,
            asset_id: archiveAsset.id,
            content_hash: archiveHash.content_hash,
        });
    }
    const payloadFingerprint = await externalizationPayloadFingerprint(options, inputPath, suppliedReplayArchives);
    if (located.call.arguments.type !== 'json') {
        throw new Error(`Tool call ${options.call_id} does not have inline JSON arguments`);
    }
    if (
        located.call.arguments.value === null ||
        typeof located.call.arguments.value !== 'object' ||
        Array.isArray(located.call.arguments.value)
    ) {
        throw new Error(`Tool call ${options.call_id} arguments must be a JSON object to externalize a field`);
    }
    const content = valueAtPath(located.call.arguments.value, inputPath);
    if (typeof content !== 'string') {
        throw new Error(`Tool call ${options.call_id} argument path ${JSON.stringify(inputPath)} is not a string`);
    }
    const contentHash = await hashUtf8Text(content);
    if (
        contentHash.content_hash !== options.asset.content_hash ||
        contentHash.byte_length !== options.asset.byte_length
    ) {
        throw new Error(`Tool call ${options.call_id} content does not match its durable asset`);
    }
    const exactArgumentsHash = await fingerprintJson(located.call.arguments.value);
    if (exactArgumentsHash !== options.exact_arguments_hash) {
        throw new Error(`Tool call ${options.call_id} exact arguments hash does not match`);
    }
    const externalizedArguments: ExternalizedToolArguments = {
        type: 'externalized_json',
        value: executionBase(located.call.arguments.value, inputPath),
        model_value: structuredClone(options.model_value),
        exact_arguments_hash: exactArgumentsHash,
        hydration: [
            {
                type: 'text_asset',
                input_path: inputPath,
                asset_id: options.asset.id,
                content_hash: options.asset.content_hash,
            },
        ],
        ...(replayArchiveRefs.length === 0 ? {} : { invalidated_replay_archives: replayArchiveRefs }),
    };
    const invalidatedReplayIds = new Set(invalidatedReplayBlocks.map((block) => block.id));
    const turns = document.turns.map((turn, turnIndex) => ({
        ...turn,
        blocks: turn.blocks
            .filter((block) => !invalidatedReplayIds.has(block.id))
            .map((block) =>
                turnIndex === located.turn_index && block.id === located.call.id
                    ? { ...located.call, arguments: externalizedArguments }
                    : block,
            ),
    })) as ConversationDocument['turns'];
    const compactions = Object.fromEntries(
        Object.entries(document.compactions).map(([id, compaction]) => [
            id,
            {
                ...compaction,
                replacement_turns: compaction.replacement_turns.map((turn) => ({
                    ...turn,
                    blocks: turn.blocks.filter((block) => !invalidatedReplayIds.has(block.id)),
                })),
            },
        ]),
    ) as ConversationDocument['compactions'];
    const resultRevision = checkedNextRevision(document.revision);
    const receipt: OperationReceipt = {
        id: options.operation_id,
        conversation_id: document.id,
        payload_fingerprint: payloadFingerprint,
        base_revision: document.revision,
        result_revision: resultRevision,
        recorded_at: options.recorded_at,
        accepted_asset_ids: [options.asset.id, ...replayArchiveAssets.map((asset) => asset.id)],
    };
    const updated: ConversationDocument = {
        ...document,
        revision: resultRevision,
        updated_at: options.recorded_at,
        turns,
        compactions,
        assets: Object.fromEntries([
            ...Object.entries(document.assets),
            [options.asset.id, structuredClone(options.asset)],
            ...replayArchiveAssets.map((asset) => [asset.id, structuredClone(asset)]),
        ]),
        operation_receipts: recordWith(document.operation_receipts, receipt.id, receipt),
        context: { ...document.context, revision: resultRevision },
    };
    return {
        document: parseConversationDocument(updated),
        applied: true,
        call_id: options.call_id,
        asset_id: options.asset.id,
    };
}

async function readAssetBytes(
    asset: Asset,
    resolveAsset: ResolveToolArgumentTextAsset,
    maxBytes: number,
): Promise<Uint8Array> {
    return readBoundedConversationAsset(asset, resolveAsset, {
        max_bytes: maxBytes,
        max_chunks: MAX_TOOL_ARGUMENT_HYDRATION_CHUNKS,
        label: 'Tool argument asset',
    });
}

export async function hydrateToolCallArguments(
    input: ConversationDocument,
    callId: string,
    resolveAsset: ResolveToolArgumentTextAsset,
    options: HydrateToolArgumentsOptions = {},
): Promise<JsonObject> {
    const document = parseConversationDocument(input);
    const maxBytes = options.max_bytes ?? DEFAULT_MAX_HYDRATED_TOOL_ARGUMENT_BYTES;
    assertSafeByteLimit(maxBytes);
    const { call } = findToolCall(document, callId);
    if (call.arguments.type === 'invalid') throw new Error(`Tool call ${callId} has invalid arguments`);
    if (call.arguments.type === 'json') {
        const inline = preflightJsonInput(call.arguments.value, { max_bytes: maxBytes });
        if (!inline.success) {
            throw new RangeError(`Tool call ${callId} inline arguments exceed max_bytes`);
        }
        if (
            call.arguments.value === null ||
            typeof call.arguments.value !== 'object' ||
            Array.isArray(call.arguments.value)
        ) {
            throw new Error(`Tool call ${callId} arguments are not a JSON object`);
        }
        return structuredClone(call.arguments.value);
    }
    assertDistinctHydrationPaths(call.arguments);
    const hydrated = structuredClone(call.arguments.value);
    const inline = preflightJsonInput(
        { value: call.arguments.value, model_value: call.arguments.model_value },
        {
            max_bytes: maxBytes,
        },
    );
    if (!inline.success) {
        throw new RangeError(`Tool call ${callId} inline hydration state exceeds max_bytes`);
    }
    let remainingBytes = maxBytes - inline.bytes;
    for (const reference of call.arguments.hydration) {
        const asset = ownRecordValue(document.assets, reference.asset_id);
        if (asset === undefined) throw new Error(`Tool argument asset ${reference.asset_id} does not exist`);
        if (asset.kind !== 'text' || asset.mime_type !== 'text/plain') {
            throw new Error(`Tool argument asset ${asset.id} must be text/plain`);
        }
        if (asset.content_hash === undefined || asset.content_hash !== reference.content_hash) {
            throw new Error(`Tool argument asset ${asset.id} content hash does not match its hydration reference`);
        }
        const bytes = await readAssetBytes(asset, resolveAsset, remainingBytes);
        remainingBytes -= bytes.byteLength;
        if (asset.byte_length !== undefined && asset.byte_length !== bytes.byteLength) {
            throw new Error(`Tool argument asset ${asset.id} byte length does not match`);
        }
        if ((await hashContentBytes(bytes)).content_hash !== reference.content_hash) {
            throw new Error(`Tool argument asset ${asset.id} content hash does not match resolved bytes`);
        }
        let content: string;
        try {
            content = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
        } catch {
            throw new Error(`Tool argument asset ${asset.id} is not valid UTF-8`);
        }
        setPath(hydrated, reference.input_path, content);
    }
    const hydratedPreflight = preflightJsonInput(hydrated, { max_bytes: maxBytes });
    if (!hydratedPreflight.success) {
        throw new RangeError(`Tool call ${callId} reconstructed arguments exceed max_bytes`);
    }
    if ((await fingerprintJson(hydrated)) !== call.arguments.exact_arguments_hash) {
        throw new Error(`Tool call ${callId} reconstructed arguments hash does not match`);
    }
    return hydrated;
}
