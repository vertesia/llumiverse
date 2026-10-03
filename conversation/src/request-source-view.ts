import type { z } from 'zod';
import { canonicalJsonContentBytes, hashContentBytes } from './content-integrity.js';
import { preflightJsonInput } from './json-preflight.js';
import { parseConversationPreparedRequest, parseConversationPreparedRequestRecord } from './prepared-request.js';
import { RequestSourceViewReferenceSchema } from './schemas/execution.js';
import {
    RequestSourceProjectedTurnSchema,
    RequestSourceViewIndexPageSchema,
    RequestSourceViewManifestSchema,
    RequestSourceViewRecordSchema,
    RequestSourceWorkingSetSchema,
    SOURCE_VIEW_MANIFEST_MAX_BYTES,
    SOURCE_VIEW_MAX_INDEX_PAGES,
    SOURCE_VIEW_MAX_SEGMENTS,
    SOURCE_VIEW_SEGMENT_MAX_BYTES,
    SOURCE_VIEW_WORKING_SET_MAX_BYTES,
} from './schemas/request-source-view.js';
import { verifyDerivedBlockLineage } from './source-slice-lineage.js';
import type {
    Asset,
    ConversationPreparedRequest,
    ConversationPreparedRequestRecord,
    ConversationTurn,
} from './types.js';

type Manifest = z.infer<typeof RequestSourceViewManifestSchema>;
type RecordRef = z.infer<typeof RequestSourceViewRecordSchema>;
export type RequestSourceWorkingSet = z.infer<typeof RequestSourceWorkingSetSchema>;
export type RequestSourceProjectedTurn = z.infer<typeof RequestSourceProjectedTurnSchema>;
type WorkingSet = RequestSourceWorkingSet;
type ProjectedTurn = RequestSourceProjectedTurn;

export interface RequestSourceViewArtifact {
    storage_key: string;
    bytes: Uint8Array;
    content_hash: string;
}

export interface PreparedRequestSourceViewArtifacts {
    manifest: RequestSourceViewArtifact;
    segments: RequestSourceViewArtifact[];
    index_pages: RequestSourceViewArtifact[];
    working_set: WorkingSet;
}

export interface RequestSourceViewArtifactReader {
    /** The host authenticates the retained run/account/project scope before invoking this reader. */
    read(storageKey: string): Promise<Uint8Array>;
}

function ownJson<T>(input: T): T {
    if (!preflightJsonInput(input).success) throw new TypeError('Source view input is not bounded JSON');
    return structuredClone(input);
}

function ordinalCompare(first: string, second: string): number {
    return first < second ? -1 : first > second ? 1 : 0;
}

/** JSON tuple encoding keeps identifiers containing separators distinct across runtimes. */
function recordIdentity(kind: string, compactionId: string | undefined, id: string): string {
    return JSON.stringify([kind, compactionId ?? null, id]);
}

function selectedTurn(
    entry: WorkingSet['selected_entries'][number],
    sourceTurns: ReadonlyMap<string, ConversationTurn>,
    replacementTurns: ReadonlyMap<string, ReadonlyMap<string, ConversationTurn>>,
): ConversationTurn {
    const turn =
        entry.type === 'source_turn'
            ? sourceTurns.get(entry.turn_id)
            : replacementTurns.get(entry.compaction_id)?.get(entry.turn_id);
    if (turn === undefined) throw new Error(`Selected source view turn ${entry.turn_id} is unavailable`);
    return turn;
}

function referencedAssetIds(projections: readonly ProjectedTurn[]): Set<string> {
    const ids = new Set<string>();
    for (const projection of projections) {
        for (const block of projection.selected_blocks) {
            if ('asset_id' in block) ids.add(block.asset_id);
            if (block.type === 'tool_call' && block.arguments.type === 'externalized_json') {
                for (const item of block.arguments.hydration) ids.add(item.asset_id);
            }
            if (block.type === 'tool_result') {
                for (const nested of block.content) if ('asset_id' in nested) ids.add(nested.asset_id);
            }
        }
    }
    return ids;
}

function referencedAssets(projections: ProjectedTurn[], prepared: ConversationPreparedRequest): Record<string, Asset> {
    const ids = referencedAssetIds(projections);
    return Object.fromEntries(
        [...ids].sort(ordinalCompare).map((id) => {
            if (!Object.hasOwn(prepared.document.assets, id)) {
                throw new Error(`Selected source view asset ${id} is unavailable`);
            }
            const asset = prepared.document.assets[id];
            if (asset === undefined) throw new Error(`Selected source view asset ${id} is unavailable`);
            return [id, asset];
        }),
    );
}

async function assertSelectedWorkingSetClosure(workingSet: WorkingSet): Promise<void> {
    const projections = new Map<string, ProjectedTurn>();
    for (const projection of workingSet.turns) {
        const key = recordIdentity('source_turn', undefined, projection.header.id);
        if (projections.has(key)) throw new Error('Selected source view repeats a source turn');
        projections.set(key, projection);
    }
    for (const { compaction_id, projection } of workingSet.replacement_turns) {
        const key = recordIdentity('replacement_turn', compaction_id, projection.header.id);
        if (projections.has(key)) throw new Error('Selected source view repeats a replacement turn');
        projections.set(key, projection);
    }
    const selections = new Map<string, Set<string> | undefined>();
    for (const entry of workingSet.context.entries) {
        const key = recordIdentity(
            entry.type,
            entry.type === 'replacement_turn' ? entry.compaction_id : undefined,
            entry.turn_id,
        );
        if (!projections.has(key)) throw new Error('Selected source view omits a context turn');
        if (!selections.has(key))
            selections.set(key, entry.block_ids === undefined ? undefined : new Set(entry.block_ids));
        else if (entry.block_ids === undefined) selections.set(key, undefined);
        else {
            const selected = selections.get(key);
            if (selected !== undefined) for (const id of entry.block_ids) selected.add(id);
        }
    }
    if (projections.size !== selections.size) throw new Error('Selected source view includes an unselected turn');
    const turnIds = new Set<string>();
    const blockIds = new Set<string>();
    const callIds = new Set<string>();
    const globalIds = new Set<string>([workingSet.source.conversation_id]);
    const registerId = (id: string): void => {
        if (globalIds.has(id)) throw new Error('Selected source view repeats a canonical identity');
        globalIds.add(id);
    };
    registerId(workingSet.request_receipt.id);
    for (const entry of workingSet.context.entries) registerId(entry.id);
    const compactionIds = new Set<string>();
    for (const { compaction_id } of workingSet.replacement_turns) {
        if (compactionIds.has(compaction_id)) continue;
        compactionIds.add(compaction_id);
        registerId(compaction_id);
    }
    for (const [key, projection] of projections) {
        const selected = selections.get(key);
        const actualIds = projection.selected_blocks.map((block) => block.id);
        if (
            projection.completeness === 'full_turn' &&
            (await hashContentBytes(canonicalJsonContentBytes(actualIds))).content_hash !==
                projection.source_block_ids_hash
        ) {
            throw new Error('Selected source view full-turn block order differs from its declared source');
        }
        if (
            (selected === undefined && projection.completeness !== 'full_turn') ||
            (selected !== undefined &&
                (actualIds.length !== selected.size ||
                    new Set(actualIds).size !== actualIds.length ||
                    actualIds.some((id) => !selected.has(id))))
        ) {
            throw new Error('Selected source view block projection differs from its context selection');
        }
        registerId(projection.header.id);
        turnIds.add(projection.header.id);
        for (const block of projection.selected_blocks) {
            registerId(block.id);
            blockIds.add(block.id);
            if (block.type === 'tool_call') {
                registerId(block.call_id);
                callIds.add(block.call_id);
            }
            if (block.type === 'tool_result') {
                for (const nested of block.content) {
                    registerId(nested.id);
                    blockIds.add(nested.id);
                }
            }
        }
    }
    for (const mapping of workingSet.request_receipt.item_mappings) {
        const ids = mapping.kind === 'turn' ? turnIds : mapping.kind === 'block' ? blockIds : callIds;
        if (!ids.has(mapping.canonical_id)) throw new Error('Selected source view omits a mapped canonical item');
    }
    const expectedAssetIds = referencedAssetIds([
        ...workingSet.turns,
        ...workingSet.replacement_turns.map(({ projection }) => projection),
    ]);
    const actualAssetIds = Object.keys(workingSet.assets);
    if (expectedAssetIds.size !== actualAssetIds.length || actualAssetIds.some((id) => !expectedAssetIds.has(id))) {
        throw new Error('Selected source view asset records differ from selected content');
    }
    const boundAssetIds = new Set<string>();
    for (const binding of workingSet.request_receipt.asset_versions) {
        if (boundAssetIds.has(binding.asset_id))
            throw new Error('Selected source view repeats an accepted asset version');
        boundAssetIds.add(binding.asset_id);
        const asset = workingSet.assets[binding.asset_id];
        if (asset === undefined || asset.content_hash !== binding.content_hash) {
            throw new Error('Selected source view asset differs from its accepted version');
        }
    }
    for (const [id, asset] of Object.entries(workingSet.assets)) {
        registerId(asset.id);
        if (asset.id !== id || (asset.content_hash !== undefined && !boundAssetIds.has(id))) {
            throw new Error('Selected source view asset differs from its accepted version');
        }
    }
    const activeToolIds = workingSet.context.active_tool_definition_ids;
    const requestToolIds = workingSet.request_receipt.tool_definition_ids;
    const actualToolIds = Object.keys(workingSet.tool_definitions);
    if (
        activeToolIds.length !== requestToolIds.length ||
        activeToolIds.some((id, index) => id !== requestToolIds[index]) ||
        actualToolIds.length !== activeToolIds.length ||
        actualToolIds.some((id) => !activeToolIds.includes(id))
    ) {
        throw new Error('Selected source view tool definitions differ from the accepted request');
    }
    for (const [id, definition] of Object.entries(workingSet.tool_definitions)) {
        registerId(definition.id);
        if (definition.id !== id)
            throw new Error('Selected source view tool definition identity differs from its record');
    }
}

async function selectWorkingSet(prepared: ConversationPreparedRequest): Promise<WorkingSet> {
    const entries = prepared.document.context.entries;
    const sourceTurnIndex = new Map(prepared.document.turns.map((turn) => [turn.id, turn]));
    const replacementTurnIndex = new Map(
        Object.entries(prepared.document.compactions).map(([id, compaction]) => [
            id,
            new Map(compaction.replacement_turns.map((turn) => [turn.id, turn])),
        ]),
    );
    const selections = new Map<string, { turn: ConversationTurn; compaction_id?: string; block_ids?: Set<string> }>();
    for (const entry of entries) {
        const turn = selectedTurn(entry, sourceTurnIndex, replacementTurnIndex);
        const compactionId = entry.type === 'replacement_turn' ? entry.compaction_id : undefined;
        const key = recordIdentity(entry.type, compactionId, turn.id);
        const prior = selections.get(key);
        if (prior === undefined) {
            selections.set(key, {
                turn,
                ...(compactionId === undefined ? {} : { compaction_id: compactionId }),
                ...(entry.block_ids === undefined ? {} : { block_ids: new Set(entry.block_ids) }),
            });
        } else if (prior.block_ids !== undefined) {
            if (entry.block_ids === undefined) prior.block_ids = undefined;
            else for (const id of entry.block_ids) prior.block_ids.add(id);
        }
    }
    const sourceTurns: ProjectedTurn[] = [];
    const replacementTurns: WorkingSet['replacement_turns'] = [];
    for (const selection of selections.values()) {
        const { blocks: originalBlocks, ...header } = selection.turn;
        const selectedWithPositions = originalBlocks
            .map((block, position) => ({ block, position }))
            .filter(({ block }) => selection.block_ids === undefined || selection.block_ids.has(block.id));
        const selectedBlocks = selectedWithPositions.map(({ block }) => block);
        if (selection.block_ids !== undefined && selectedBlocks.length !== selection.block_ids.size) {
            throw new Error('Selected source view names an unavailable block');
        }
        const projection = RequestSourceProjectedTurnSchema.parse({
            completeness: selectedBlocks.length === originalBlocks.length ? 'full_turn' : 'selected_blocks',
            header,
            selected_blocks: selectedBlocks,
            selected_block_positions: selectedWithPositions.map(({ position }) => position),
            source_block_count: originalBlocks.length,
            source_block_ids_hash: (
                await hashContentBytes(canonicalJsonContentBytes(originalBlocks.map((block) => block.id)))
            ).content_hash,
        });
        if (selection.compaction_id === undefined) sourceTurns.push(projection);
        else replacementTurns.push({ compaction_id: selection.compaction_id, projection });
    }
    const toolDefinitions = Object.fromEntries(
        prepared.record.request_receipt.tool_definition_ids.map((id) => {
            if (!Object.hasOwn(prepared.document.tool_definitions, id)) {
                throw new Error(`Selected source view tool ${id} is unavailable`);
            }
            const definition = prepared.document.tool_definitions[id];
            if (definition === undefined) throw new Error(`Selected source view tool ${id} is unavailable`);
            return [id, definition];
        }),
    );
    return RequestSourceWorkingSetSchema.parse({
        completeness: 'selected_content_unverified',
        source: prepared.record.source,
        context: prepared.document.context,
        request_receipt: prepared.record.request_receipt,
        turns: sourceTurns,
        replacement_turns: replacementTurns,
        assets: referencedAssets([...sourceTurns, ...replacementTurns.map((item) => item.projection)], prepared),
        tool_definitions: toolDefinitions,
        selected_entries: entries,
    });
}

/** Build bounded immutable records from a fully validated original before the host publishes its prepared-record CAS. */
export async function createPreparedRequestSourceViewArtifacts(
    input: ConversationPreparedRequest,
    storageKeyPrefix: string,
): Promise<PreparedRequestSourceViewArtifacts> {
    const ownedInput = ownJson(input);
    const prepared = await parseConversationPreparedRequest(ownedInput);
    const selectedIds = [...new Set(prepared.document.context.entries.map((entry) => entry.turn_id))];
    await verifyDerivedBlockLineage(prepared.document, { turn_ids: selectedIds });
    const workingSet = await selectWorkingSet(prepared);
    const records: RecordRef[] = [];
    const packedSegments: Uint8Array[] = [];
    const sources: Array<{ kind: RecordRef['kind']; id: string; value: unknown; compaction_id?: string }> = [
        { kind: 'context', id: 'context', value: workingSet.context },
        ...workingSet.turns.map((projection) => ({
            kind: 'source_turn' as const,
            id: projection.header.id,
            value: projection,
        })),
        ...workingSet.replacement_turns.map(({ compaction_id, projection }) => ({
            kind: 'replacement_turn' as const,
            id: projection.header.id,
            compaction_id,
            value: projection,
        })),
        ...Object.entries(workingSet.assets).map(([id, value]) => ({ kind: 'asset' as const, id, value })),
        ...Object.entries(workingSet.tool_definitions).map(([id, value]) => ({
            kind: 'tool_definition' as const,
            id,
            value,
        })),
    ];
    sources.sort((a, b) =>
        ordinalCompare(recordIdentity(a.kind, a.compaction_id, a.id), recordIdentity(b.kind, b.compaction_id, b.id)),
    );
    let pendingParts: Uint8Array[] = [];
    let pendingBytes = 0;
    let contentBytes = 0;
    const flushSegment = () => {
        if (pendingBytes === 0) return;
        const packed = new Uint8Array(pendingBytes);
        let offset = 0;
        for (const part of pendingParts) {
            packed.set(part, offset);
            offset += part.byteLength;
        }
        packedSegments.push(packed);
        pendingParts = [];
        pendingBytes = 0;
    };
    for (const source of sources) {
        const bytes = canonicalJsonContentBytes(source.value);
        contentBytes += bytes.byteLength;
        if (contentBytes > SOURCE_VIEW_WORKING_SET_MAX_BYTES || bytes.byteLength === 0) {
            throw new RangeError('Selected source view exceeds its bounded working set');
        }
        const parts: RecordRef['parts'] = [];
        for (let position = 0; position < bytes.byteLength; ) {
            if (pendingBytes === SOURCE_VIEW_SEGMENT_MAX_BYTES) flushSegment();
            if (packedSegments.length >= SOURCE_VIEW_MAX_SEGMENTS)
                throw new RangeError('Selected source view has too many content segments');
            const length = Math.min(bytes.byteLength - position, SOURCE_VIEW_SEGMENT_MAX_BYTES - pendingBytes);
            parts.push({ segment_index: packedSegments.length, offset: pendingBytes, length });
            pendingParts.push(bytes.slice(position, position + length));
            pendingBytes += length;
            position += length;
        }
        records.push(
            RequestSourceViewRecordSchema.parse({
                kind: source.kind,
                id: source.id,
                ...(source.compaction_id === undefined ? {} : { compaction_id: source.compaction_id }),
                content_hash: (await hashContentBytes(bytes)).content_hash,
                size_bytes: bytes.byteLength,
                parts,
            }),
        );
    }
    flushSegment();
    if (packedSegments.length > SOURCE_VIEW_MAX_SEGMENTS)
        throw new RangeError('Selected source view has too many content segments');
    const segments: RequestSourceViewArtifact[] = [];
    for (const bytes of packedSegments) {
        const contentHash = (await hashContentBytes(bytes)).content_hash;
        segments.push({
            storage_key: `${storageKeyPrefix}/segments/${contentHash.slice(7)}.jsonpart`,
            bytes,
            content_hash: contentHash,
        });
    }
    const indexPages: RequestSourceViewArtifact[] = [];
    let pageRecords: RecordRef[] = [];
    const flushPage = async () => {
        if (pageRecords.length === 0) return;
        const bytes = canonicalJsonContentBytes({ version: 1, records: pageRecords });
        if (bytes.byteLength > SOURCE_VIEW_MANIFEST_MAX_BYTES)
            throw new RangeError('Selected source view index page exceeds its bound');
        const contentHash = (await hashContentBytes(bytes)).content_hash;
        indexPages.push({
            storage_key: `${storageKeyPrefix}/indexes/${contentHash.slice(7)}.json`,
            bytes,
            content_hash: contentHash,
        });
        pageRecords = [];
    };
    for (const item of records) {
        const candidate = [...pageRecords, item];
        if (canonicalJsonContentBytes({ version: 1, records: candidate }).byteLength > SOURCE_VIEW_MANIFEST_MAX_BYTES) {
            await flushPage();
        }
        pageRecords.push(item);
    }
    await flushPage();
    if (indexPages.length > SOURCE_VIEW_MAX_INDEX_PAGES)
        throw new RangeError('Selected source view has too many index pages');
    const selectedEntryIds = workingSet.selected_entries.map((entry) => entry.id);
    const activeIds = prepared.record.request_receipt.tool_definition_ids;
    const { source_view: _sourceView, ...acceptedRequest } = prepared.record.request_receipt;
    const acceptedRecord = { ...prepared.record, request_receipt: acceptedRequest };
    const manifest: Manifest = RequestSourceViewManifestSchema.parse({
        version: 2,
        completeness: 'selected_content_unverified',
        source: prepared.record.source,
        context_revision: prepared.document.context.revision,
        request_receipt_id: prepared.record.request_receipt.id,
        request_fingerprint: prepared.record.request_receipt.request_fingerprint,
        accepted_request_binding_hash: (await hashContentBytes(canonicalJsonContentBytes(acceptedRequest)))
            .content_hash,
        accepted_prepared_record_binding_hash: (await hashContentBytes(canonicalJsonContentBytes(acceptedRecord)))
            .content_hash,
        validated_document_hash: (await hashContentBytes(canonicalJsonContentBytes(prepared.document))).content_hash,
        validator_profile: 'llumiverse.conversation/full-materialized/2026-09-30.adoption.1',
        context_fingerprint: prepared.record.request_receipt.context_fingerprint,
        selected_entry_ids_hash: (await hashContentBytes(canonicalJsonContentBytes(selectedEntryIds))).content_hash,
        active_tool_definition_ids_hash: (await hashContentBytes(canonicalJsonContentBytes(activeIds))).content_hash,
        record_count: records.length,
        segments: segments.map(({ storage_key, content_hash, bytes }) => ({
            storage_key,
            content_hash,
            size_bytes: bytes.byteLength,
        })),
        index_pages: indexPages.map(({ storage_key, content_hash, bytes }) => ({
            storage_key,
            content_hash,
            size_bytes: bytes.byteLength,
        })),
    });
    const manifestBytes = canonicalJsonContentBytes(manifest);
    const totalBytes =
        manifestBytes.byteLength + contentBytes + indexPages.reduce((sum, page) => sum + page.bytes.byteLength, 0);
    if (manifestBytes.byteLength > SOURCE_VIEW_MANIFEST_MAX_BYTES || totalBytes > SOURCE_VIEW_WORKING_SET_MAX_BYTES) {
        throw new RangeError('Selected source view manifest or working set exceeds its bound');
    }
    const manifestHash = (await hashContentBytes(manifestBytes)).content_hash;
    return {
        manifest: {
            storage_key: `${storageKeyPrefix}/manifests/${manifestHash.slice(7)}.json`,
            bytes: manifestBytes,
            content_hash: manifestHash,
        },
        segments,
        index_pages: indexPages,
        working_set: workingSet,
    };
}

function decodeOwnedJson(bytes: Uint8Array): unknown {
    const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    const canonical = canonicalJsonContentBytes(value);
    if (canonical.byteLength !== bytes.byteLength || canonical.some((byte, index) => byte !== bytes[index])) {
        throw new Error('Selected source view artifact is not canonical JSON');
    }
    return value;
}

async function readOwnedArtifact(
    reader: RequestSourceViewArtifactReader,
    descriptor: { storage_key: string; content_hash: string; size_bytes: number },
): Promise<Uint8Array> {
    const returned = await reader.read(descriptor.storage_key);
    if (!(returned instanceof Uint8Array)) throw new TypeError('Selected source view reader returned nonbytes');
    if (returned.byteLength !== descriptor.size_bytes) {
        throw new Error('Selected source view artifact size or hash mismatch');
    }
    const bytes = Uint8Array.from(returned);
    if (
        bytes.byteLength !== descriptor.size_bytes ||
        (await hashContentBytes(bytes)).content_hash !== descriptor.content_hash
    ) {
        throw new Error('Selected source view artifact size or hash mismatch');
    }
    return bytes;
}

/** Resolve only records named by an authorized, immutable prepared record. No URL or full history fetch occurs. */
export async function resolvePreparedRequestSourceWorkingSet(
    recordInput: ConversationPreparedRequestRecord,
    reader: RequestSourceViewArtifactReader,
): Promise<WorkingSet> {
    const record = parseConversationPreparedRequestRecord(recordInput);
    const reference = RequestSourceViewReferenceSchema.parse(record.request_receipt.source_view);
    if (reference.completeness !== 'selected_content_unverified') {
        throw new Error('Selected source view is not an acknowledged content projection');
    }
    const root = reference.manifest_storage_key.slice(0, reference.manifest_storage_key.lastIndexOf('/manifests/'));
    if (
        !root ||
        reference.manifest_storage_key !== `${root}/manifests/${reference.manifest_content_hash.slice(7)}.json`
    ) {
        throw new Error('Selected source view manifest key does not match its immutable digest');
    }
    const manifestBytes = await readOwnedArtifact(reader, {
        storage_key: reference.manifest_storage_key,
        content_hash: reference.manifest_content_hash,
        size_bytes: reference.manifest_size_bytes,
    });
    if (manifestBytes.byteLength > SOURCE_VIEW_MANIFEST_MAX_BYTES)
        throw new RangeError('Selected manifest exceeds bound');
    const manifest = RequestSourceViewManifestSchema.parse(decodeOwnedJson(manifestBytes));
    if (manifest.version !== 2) {
        throw new Error('Legacy selected source view lacks prepared-record and original-position witnesses');
    }
    const { source_view: _sourceView, ...acceptedRequest } = record.request_receipt;
    const acceptedRecord = { ...record, request_receipt: acceptedRequest };
    if (
        manifest.source.conversation_id !== record.source.conversation_id ||
        manifest.source.revision !== record.source.revision ||
        manifest.source.conversation_id !== reference.source.conversation_id ||
        manifest.source.revision !== reference.source.revision ||
        manifest.context_revision !== reference.context_revision ||
        manifest.request_receipt_id !== record.request_receipt.id ||
        manifest.request_fingerprint !== record.request_receipt.request_fingerprint ||
        manifest.request_fingerprint !== reference.request_fingerprint ||
        manifest.accepted_request_binding_hash !==
            (await hashContentBytes(canonicalJsonContentBytes(acceptedRequest))).content_hash ||
        (manifest.accepted_prepared_record_binding_hash !== undefined &&
            manifest.accepted_prepared_record_binding_hash !==
                (await hashContentBytes(canonicalJsonContentBytes(acceptedRecord))).content_hash) ||
        manifest.context_fingerprint !== record.request_receipt.context_fingerprint ||
        manifest.context_fingerprint !== reference.context_fingerprint ||
        manifest.active_tool_definition_ids_hash !==
            (await hashContentBytes(canonicalJsonContentBytes(record.request_receipt.tool_definition_ids))).content_hash
    ) {
        throw new Error('Selected source view manifest differs from the accepted prepared request');
    }
    const declaredBytes =
        manifestBytes.byteLength +
        manifest.segments.reduce((sum, segment) => sum + segment.size_bytes, 0) +
        manifest.index_pages.reduce((sum, page) => sum + page.size_bytes, 0);
    if (declaredBytes > SOURCE_VIEW_WORKING_SET_MAX_BYTES) {
        throw new RangeError('Selected source view exceeds its bounded working set');
    }
    const records: RecordRef[] = [];
    for (const page of manifest.index_pages) {
        if (
            page.size_bytes > SOURCE_VIEW_MANIFEST_MAX_BYTES ||
            page.storage_key !== `${root}/indexes/${page.content_hash.slice(7)}.json`
        ) {
            throw new Error('Selected source view index page is outside its immutable bound');
        }
        const pageBytes = await readOwnedArtifact(reader, { ...page });
        records.push(...RequestSourceViewIndexPageSchema.parse(decodeOwnedJson(pageBytes)).records);
        if (records.length > manifest.record_count)
            throw new Error('Selected source view index count exceeds manifest');
    }
    if (records.length !== manifest.record_count)
        throw new Error('Selected source view index count differs from manifest');
    const packedSegments: Uint8Array[] = [];
    for (const part of manifest.segments) {
        if (part.storage_key !== `${root}/segments/${part.content_hash.slice(7)}.jsonpart`) {
            throw new Error('Selected source view segment key does not match its immutable digest');
        }
        packedSegments.push(await readOwnedArtifact(reader, { ...part }));
    }
    const segmentCursors = new Array<number>(packedSegments.length).fill(0);
    const keys = new Set<string>();
    const values: Array<{ descriptor: RecordRef; value: unknown }> = [];
    let previousKey: string | undefined;
    for (const descriptor of records) {
        const key = recordIdentity(descriptor.kind, descriptor.compaction_id, descriptor.id);
        if (keys.has(key) || (previousKey !== undefined && ordinalCompare(previousKey, key) >= 0)) {
            throw new Error('Selected source view records are not uniquely ordered');
        }
        previousKey = key;
        keys.add(key);
        if ((descriptor.kind === 'replacement_turn') !== (descriptor.compaction_id !== undefined)) {
            throw new Error('Selected source view has invalid replacement identity');
        }
        const recordBytes = new Uint8Array(descriptor.size_bytes);
        let offset = 0;
        for (const part of descriptor.parts) {
            const segment = packedSegments[part.segment_index];
            if (
                segment === undefined ||
                part.offset !== segmentCursors[part.segment_index] ||
                part.offset + part.length > segment.byteLength ||
                offset + part.length > recordBytes.byteLength
            ) {
                throw new Error('Selected source view part does not match packed content');
            }
            recordBytes.set(segment.subarray(part.offset, part.offset + part.length), offset);
            segmentCursors[part.segment_index] += part.length;
            offset += part.length;
        }
        if (
            offset !== descriptor.size_bytes ||
            (await hashContentBytes(recordBytes)).content_hash !== descriptor.content_hash
        ) {
            throw new Error('Selected source view record size or hash mismatch');
        }
        values.push({ descriptor, value: decodeOwnedJson(recordBytes) });
    }
    if (packedSegments.some((segment, index) => segmentCursors[index] !== segment.byteLength)) {
        throw new Error('Selected source view contains unclaimed packed bytes');
    }
    const contextRecords = values.filter(({ descriptor }) => descriptor.kind === 'context');
    if (contextRecords.length !== 1 || contextRecords[0]?.descriptor.id !== 'context') {
        throw new Error('Selected source view has no unique context record');
    }
    const turns: ProjectedTurn[] = [];
    const replacements: WorkingSet['replacement_turns'] = [];
    const assets = new Map<string, Asset>();
    const toolDefinitions = new Map<string, WorkingSet['tool_definitions'][string]>();
    for (const { descriptor, value } of values) {
        if (descriptor.kind === 'source_turn') turns.push(value as ProjectedTurn);
        if (descriptor.kind === 'replacement_turn') {
            replacements.push({
                compaction_id: descriptor.compaction_id as string,
                projection: value as ProjectedTurn,
            });
        }
        if (descriptor.kind === 'asset') assets.set(descriptor.id, value as Asset);
        if (descriptor.kind === 'tool_definition')
            toolDefinitions.set(descriptor.id, value as WorkingSet['tool_definitions'][string]);
    }
    const context = contextRecords[0].value as WorkingSet['context'];
    const workingSet = RequestSourceWorkingSetSchema.parse({
        completeness: 'selected_content_unverified',
        archive_version: manifest.version,
        accepted_prepared_record_binding_hash: manifest.accepted_prepared_record_binding_hash,
        source: record.source,
        context,
        request_receipt: record.request_receipt,
        turns,
        replacement_turns: replacements,
        assets: Object.fromEntries(assets),
        tool_definitions: Object.fromEntries(toolDefinitions),
        selected_entries: context.entries,
    });
    const selectedEntryIds = workingSet.selected_entries.map((entry) => entry.id);
    const actualEntryHash = (await hashContentBytes(canonicalJsonContentBytes(selectedEntryIds))).content_hash;
    const actualToolHash = (
        await hashContentBytes(canonicalJsonContentBytes(workingSet.context.active_tool_definition_ids))
    ).content_hash;
    if (
        workingSet.context.revision !== manifest.context_revision ||
        new Set(selectedEntryIds).size !== selectedEntryIds.length ||
        new Set(workingSet.context.active_tool_definition_ids).size !==
            workingSet.context.active_tool_definition_ids.length ||
        actualEntryHash !== manifest.selected_entry_ids_hash ||
        actualToolHash !== manifest.active_tool_definition_ids_hash
    ) {
        throw new Error('Selected source view context differs from its immutable manifest');
    }
    await assertSelectedWorkingSetClosure(workingSet);
    for (const { descriptor, value } of values) {
        const actualId =
            descriptor.kind === 'source_turn' || descriptor.kind === 'replacement_turn'
                ? (value as ProjectedTurn).header.id
                : (value as { id?: string }).id;
        if (descriptor.kind !== 'context' && actualId !== descriptor.id) {
            throw new Error('Selected source view record identity mismatch');
        }
    }
    return workingSet;
}
