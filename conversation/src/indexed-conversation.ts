import { z } from 'zod';
import { canonicalJsonContentBytes, canonicalJsonContentString, hashContentBytes } from './content-integrity.js';
import { fingerprintJson } from './identity.js';
import { DEFAULT_JSON_INPUT_LIMITS, preflightJsonInput } from './json-preflight.js';
import {
    buildPagedRecordIndex,
    getPagedRecord,
    type PagedRecordIndexStore,
    type PagedRecordRef,
    type PagedRecordValue,
    putPagedRecord,
    scanPagedRecords,
} from './paged-record-index.js';
import { countUnresolvedProcessingJobs } from './processing-job-status.js';
import { ConversationDeleteChangeSchema } from './schemas/change.js';
import { AssetSchema, ContentBlockSchema, ConversationTurnSchema, ToolDefinitionSchema } from './schemas/content.js';
import { ContextEntrySchema } from './schemas/context-foundation.js';
import { ConversationDeleteOperationSchema } from './schemas/conversation-delete-operation.js';
import { ConversationContextSchema } from './schemas/document.js';
import { ExecutionReceiptSchema, GenerationSchema, OperationReceiptSchema } from './schemas/execution.js';
import {
    INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
    INDEXED_CONVERSATION_DELETE_PROFILE,
    INDEXED_CONVERSATION_PROFILE,
    INDEXED_CONVERSATION_ROOT_MAX_BYTES,
    IndexedConversationContextHeaderSchema,
    type IndexedConversationDeleteCommand,
    IndexedConversationDeleteCommandSchema,
    IndexedConversationDeletedTurnSchema,
    IndexedConversationProcessingHeaderSchema,
    type IndexedConversationRoot,
    IndexedConversationRootSchema,
    IndexedConversationSelectedContextSchema,
    type IndexedConversationTurnHeader,
    IndexedConversationTurnHeaderSchema,
    IndexedConversationTurnLinkSchema,
} from './schemas/indexed-head.js';
import { AppendConversationRecordsOptionsSchema, ConversationRecordBatchSchema } from './schemas/ingestion.js';
import { IdentifierSchema } from './schemas/primitives.js';
import { validateConversationSemantics, validateUsage } from './semantic-validation.js';
import { conversationDocumentFromJson } from './serialization.js';
import { assertToolResultReceiptFingerprint } from './tool-result-integrity.js';
import type {
    AppendConversationRecordsOptions,
    ConversationDocument,
    ConversationRecordBatch,
    ConversationTurn,
    OperationReceipt,
} from './types.js';
import { parseConversationDocument } from './validation.js';

const IndexedProgramRecordsSchema = z.strictObject({
    conversation_id: z.string().min(1),
    expected_revision: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
    operation_id: z.string().min(1),
    recorded_at: z.iso.datetime({ offset: false }),
    turn: ConversationTurnSchema,
    entry: ContextEntrySchema,
    payload_fingerprint: z.string().min(1),
});
export type IndexedProgramRecords = z.infer<typeof IndexedProgramRecordsSchema>;

const IndexedCallStateSchema = z.strictObject({
    call_id: z.string().min(1),
    turn_id: z.string().min(1),
    block_id: z.string().min(1),
    call_fingerprint: z.string().regex(/^sha256:[0-9a-f]{64}$/),
    result_block_id: z.string().min(1).optional(),
    terminal_receipt_id: z.string().min(1).optional(),
});
type IndexedCallState = z.infer<typeof IndexedCallStateSchema>;

/** A bounded canonical batch whose publication remains the host's exact-head CAS. */
export const IndexedRecordBatchCommandSchema = z.strictObject({
    conversation_id: IdentifierSchema,
    batch: ConversationRecordBatchSchema,
    options: AppendConversationRecordsOptionsSchema,
});
export type IndexedRecordBatchCommand = z.infer<typeof IndexedRecordBatchCommandSchema>;

export interface IndexedConversationRecordStore extends PagedRecordIndexStore {
    /** The host derives a run-scoped content-addressed key from kind/hash and owns the bytes. */
    readRecord(value: Extract<PagedRecordValue, { storage: 'record' }>): Promise<Uint8Array>;
    /** Immutable create-only write, with segmentation for bodies over 64KiB. */
    writeRecord(value: Extract<PagedRecordValue, { storage: 'record' }>, bytes: Uint8Array): Promise<void>;
}

export interface StagedIndexedConversationRoot {
    root: IndexedConversationRoot;
    locator: PagedRecordRef;
}

type RecordValue = Extract<PagedRecordValue, { storage: 'record' }>;
type Entry = { key: string; value: PagedRecordValue };

/** One-time legacy import is bounded independently of ordinary 250k-node document operations. */
const INDEXED_MIGRATION_JSON_LIMITS = Object.freeze({
    ...DEFAULT_JSON_INPUT_LIMITS,
    max_nodes: 2_000_000,
});

/** Validate a retained legacy artifact with the same fixed profile as one-time indexed staging. */
export function conversationDocumentFromIndexedMigrationJson(text: string): ConversationDocument {
    return conversationDocumentFromJson(text, { json_input_limits: INDEXED_MIGRATION_JSON_LIMITS });
}

export function indexedOrderedKey(position: number): string {
    if (!Number.isSafeInteger(position) || position < 0) throw new RangeError('Indexed order position is invalid');
    return position.toString().padStart(16, '0');
}

function tupleKey(first: string, second: string): string {
    return JSON.stringify([first, second]);
}

async function stageRecord(
    store: IndexedConversationRecordStore,
    kind: string,
    id: string,
    content: unknown,
): Promise<RecordValue> {
    const bytes = canonicalJsonContentBytes(content);
    const integrity = await hashContentBytes(bytes);
    const value: RecordValue = {
        storage: 'record',
        kind,
        id,
        content_hash: integrity.content_hash,
        size_bytes: integrity.byte_length,
    };
    if (value.size_bytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES) {
        throw new RangeError('Indexed conversation record exceeds its bounded content contract');
    }
    await store.writeRecord(value, Uint8Array.from(bytes));
    const retained = Uint8Array.from(await store.readRecord(value));
    if (
        retained.byteLength !== value.size_bytes ||
        (await hashContentBytes(retained)).content_hash !== value.content_hash
    ) {
        throw new Error('Indexed conversation record failed immutable read-back verification');
    }
    return value;
}

async function loadRecord<Shape extends z.ZodType>(
    store: IndexedConversationRecordStore,
    value: PagedRecordValue | undefined,
    schema: Shape,
): Promise<z.infer<Shape>> {
    if (value?.storage !== 'record') throw new Error('Indexed conversation record is unavailable');
    const bytes = Uint8Array.from(await store.readRecord(value));
    if (bytes.byteLength !== value.size_bytes || (await hashContentBytes(bytes)).content_hash !== value.content_hash) {
        throw new Error('Indexed conversation record differs from its authenticated index');
    }
    return schema.parse(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)));
}

function acceptedResponse(document: ConversationDocument, operationId: string) {
    const receipt = document.operation_receipts[operationId];
    const generationId = receipt?.accepted_generation_ids?.[0];
    const turnId = receipt?.accepted_turn_ids?.[0];
    const generation = generationId === undefined ? undefined : document.generations[generationId];
    const turn = document.turns.find((item) => item.id === turnId);
    if (
        receipt?.accepted_generation_ids?.length !== 1 ||
        receipt.accepted_turn_ids?.length !== 1 ||
        generation?.record_source !== 'executed' ||
        turn?.kind !== 'agent' ||
        !('generation_id' in turn) ||
        turn.generation_id !== generationId ||
        receipt.result_revision > document.revision
    ) {
        throw new Error('Indexed conversation source lacks its exact accepted executed response');
    }
    return {
        operation_id: operationId,
        generation_id: generationId,
        turn_id: turnId,
        accepted_revision: receipt.result_revision,
    };
}

function processingRecordGroups(processing: ConversationDocument['processing']): [string, Record<string, unknown>][] {
    return [
        ['jobs', processing.jobs ?? {}],
        ['resolved_inputs', processing.resolved_inputs ?? {}],
        ['attempts', processing.attempts ?? {}],
        ['outputs', processing.outputs ?? {}],
        ['completions', processing.completions ?? {}],
        ['supersessions', processing.supersessions ?? {}],
        ['coverage_receipts', processing.coverage_receipts ?? {}],
    ];
}

/** A conservative one-pass reverse witness. A marker may reject a closed deletion, but never permit a dependent one. */
function migrationDeleteBlockers(document: ConversationDocument): Set<string> {
    const blocked = new Set<string>();
    const turnIds = new Set(document.turns.map((turn) => turn.id));
    const blockOwner = new Map<string, string>();
    const entryTurn = new Map<string, string>();
    const acceptedTurns = new Map<string, string[]>();
    for (const turn of document.turns) {
        for (const block of turn.blocks) blockOwner.set(block.id, turn.id);
        if (turn.blocks.some((block) => ['tool_call', 'tool_result', 'native_replay'].includes(block.type))) {
            blocked.add(turn.id);
        }
    }
    for (const receipt of Object.values(document.operation_receipts)) {
        if (receipt.operation_kind !== undefined) continue;
        acceptedTurns.set(receipt.id, receipt.accepted_turn_ids ?? []);
        for (const entry of receipt.accepted_context_entries ?? []) entryTurn.set(entry.id, entry.turn_id);
    }
    const mark = (id: string | undefined) => {
        if (id && turnIds.has(id)) blocked.add(id);
    };
    const markBlock = (id: string) => mark(blockOwner.get(id));
    for (const turn of document.turns) {
        mark(turn.parent_turn_id);
        if (turn.provenance.type === 'derived') for (const id of turn.provenance.source_turn_ids) mark(id);
        for (const block of turn.blocks) {
            if (block.type !== 'native_replay') continue;
            for (const id of block.dependencies.turn_ids) mark(id);
            for (const id of block.dependencies.block_ids) markBlock(id);
        }
    }
    for (const compaction of Object.values(document.compactions)) {
        for (const id of compaction.source.turn_ids) mark(id);
        for (const id of compaction.source.block_ids ?? []) markBlock(id);
    }
    for (const asset of Object.values(document.assets)) {
        if (asset.provenance.type === 'received') mark(asset.provenance.source_turn_id);
    }
    for (const generation of Object.values(document.generations)) {
        if (generation.record_source !== 'executed') continue;
        mark(generation.request_receipt.source_tail_turn_id);
        for (const mapping of generation.request_receipt.item_mappings) {
            mark(mapping.canonical_id);
            markBlock(mapping.canonical_id);
        }
    }
    for (const receipt of Object.values(document.execution_receipts)) {
        mark(receipt.result_turn_id);
        mark(receipt.call_source?.turn_id);
    }
    for (const resolved of Object.values(document.processing.resolved_inputs ?? {})) {
        for (const id of resolved.source_turn_ids) mark(id);
        for (const entry of resolved.selected_entries ?? []) mark(entry.turn_id);
        for (const id of resolved.entry_ids) mark(entryTurn.get(id));
    }
    for (const job of Object.values(document.processing.jobs ?? {})) {
        for (const id of acceptedTurns.get(job.source_operation_id) ?? []) mark(id);
        if (job.selection.kind !== 'entries') continue;
        for (const id of job.selection.entry_ids) mark(entryTurn.get(id));
        for (const entry of job.selection.selected_entries ?? []) mark(entry.turn_id);
    }
    return blocked;
}

/** Historical receipts may name content no longer present; reserve those names against future reuse. */
function migrationHistoricalReferences(document: ConversationDocument): Set<string> {
    const referenced = new Set<string>();
    const add = (id: string | undefined) => {
        if (id) referenced.add(id);
    };
    for (const turn of document.turns) {
        add(turn.parent_turn_id);
        if (turn.provenance.type === 'derived') for (const id of turn.provenance.source_turn_ids) add(id);
        for (const block of turn.blocks) {
            if (block.type !== 'native_replay') continue;
            for (const id of block.dependencies.turn_ids) add(id);
            for (const id of block.dependencies.block_ids) add(id);
        }
    }
    for (const compaction of Object.values(document.compactions)) {
        for (const id of compaction.source.turn_ids) add(id);
        for (const id of compaction.source.block_ids ?? []) add(id);
    }
    for (const asset of Object.values(document.assets)) {
        if (asset.provenance.type === 'received') add(asset.provenance.source_turn_id);
    }
    for (const generation of Object.values(document.generations)) {
        if (generation.record_source !== 'executed') continue;
        add(generation.request_receipt.source_tail_turn_id);
        for (const mapping of generation.request_receipt.item_mappings) add(mapping.canonical_id);
        for (const binding of generation.request_receipt.asset_versions) add(binding.asset_id);
    }
    for (const receipt of Object.values(document.execution_receipts)) {
        add(receipt.result_turn_id);
        add(receipt.call_source?.turn_id);
        add(receipt.call_source?.block_id);
    }
    for (const receipt of Object.values(document.operation_receipts)) {
        for (const entry of receipt.accepted_context_entries ?? []) add(entry.turn_id);
    }
    for (const resolved of Object.values(document.processing.resolved_inputs ?? {})) {
        for (const id of resolved.source_turn_ids) add(id);
        for (const entry of resolved.selected_entries ?? []) add(entry.turn_id);
    }
    for (const job of Object.values(document.processing.jobs ?? {})) {
        if (job.selection.kind === 'entries') {
            for (const entry of job.selection.selected_entries ?? []) add(entry.turn_id);
        }
    }
    return referenced;
}

/** One-time conversion of a fully validated legacy snapshot; not an append-time full-history path. */
export async function stageIndexedConversationSnapshot(
    source: ConversationDocument,
    acceptedOperationId: string | undefined,
    store: IndexedConversationRecordStore,
): Promise<StagedIndexedConversationRoot> {
    // Parse and own the complete bounded source before writing any indexed record. This one-time
    // profile admits 100k compact turns while keeping the ordinary append/import limit unchanged.
    const document = parseConversationDocument(source, { json_input_limits: INDEXED_MIGRATION_JSON_LIMITS });
    if (Object.keys(document.deleted_turns ?? {}).length > 0) {
        throw new Error('Indexed migration cannot retain logical-delete tombstone witnesses yet');
    }
    const families: Record<keyof IndexedConversationRoot['directories'], Entry[]> = {
        identifiers: [],
        turns: [],
        blocks: [],
        generations: [],
        generation_acceptances: [],
        operation_receipts: [],
        execution_receipts: [],
        assets: [],
        tool_definitions: [],
        compactions: [],
        processing_records: [],
        open_tool_calls: [],
        tool_call_states: [],
        context_entries: [],
        active_context_order: [],
        turn_order: [],
        turn_acceptances: [],
        block_owners: [],
        deletion_blockers: [],
        turn_links: [],
        deleted_turns: [],
    };
    const idKinds = new Map<string, string>();
    const diagnostics = validateConversationSemantics(document, (id, kind) => idKinds.set(id, kind));
    if (diagnostics.length > 0) throw new Error('Indexed conversation migration source failed semantic validation');
    for (const id of migrationHistoricalReferences(document)) {
        if (!idKinds.has(id)) idKinds.set(id, 'historical_reference');
    }
    for (const [id, kind] of idKinds) {
        families.identifiers.push({ key: id, value: { storage: 'marker', kind, id } });
    }

    const stageFamily = async (family: keyof typeof families, id: string, content: unknown, key = id) => {
        families[family].push({ key, value: await stageRecord(store, family, id, content) });
    };
    const stageTurn = async (turn: ConversationTurn, sourceKind: 'ordinary' | 'replacement', compactionId?: string) => {
        const { blocks, ...header } = turn;
        const blockIds = blocks.map((block) => block.id);
        const blockIdsHash = (await hashContentBytes(canonicalJsonContentBytes(blockIds))).content_hash;
        await stageFamily('turns', turn.id, {
            turn: header,
            source: sourceKind,
            ...(compactionId === undefined ? {} : { compaction_id: compactionId }),
            block_ids: blockIds,
            block_ids_hash: blockIdsHash,
        } satisfies IndexedConversationTurnHeader);
        for (const block of blocks) await stageFamily('blocks', block.id, block);
    };

    for (let index = 0; index < document.turns.length; index += 1) {
        const turn = document.turns[index];
        await stageTurn(turn, 'ordinary');
        await stageFamily(
            'turn_links',
            turn.id,
            IndexedConversationTurnLinkSchema.parse({
                id: turn.id,
                ordinal: index,
                ...(index === 0 ? {} : { previous_turn_id: document.turns[index - 1].id }),
                ...(index === document.turns.length - 1 ? {} : { next_turn_id: document.turns[index + 1].id }),
            }),
        );
        for (const block of turn.blocks) {
            families.block_owners.push({
                key: block.id,
                value: { storage: 'marker', kind: 'block_owner', id: turn.id },
            });
        }
        families.turn_order.push({
            key: indexedOrderedKey(index),
            value: { storage: 'marker', kind: 'turn_order', id: turn.id },
        });
    }
    for (const [id, compaction] of Object.entries(document.compactions)) {
        const { replacement_turns: replacementTurns, ...header } = compaction;
        await stageFamily('compactions', id, header);
        for (const turn of replacementTurns) await stageTurn(turn, 'replacement', id);
    }
    for (const [id, record] of Object.entries(document.generations)) await stageFamily('generations', id, record);
    for (const [id, record] of Object.entries(document.operation_receipts))
        await stageFamily('operation_receipts', id, record);
    for (const [operationId, receipt] of Object.entries(document.operation_receipts)) {
        if (receipt.operation_kind !== undefined) continue;
        for (const turnId of receipt.accepted_turn_ids ?? []) {
            families.turn_acceptances.push({
                key: turnId,
                value: { storage: 'marker', kind: 'turn_acceptance', id: operationId },
            });
        }
    }
    for (const id of migrationDeleteBlockers(document)) {
        families.deletion_blockers.push({ key: id, value: { storage: 'marker', kind: 'delete_blocker', id } });
    }
    for (const [operationId, receipt] of Object.entries(document.operation_receipts)) {
        for (const generationId of receipt.accepted_generation_ids ?? []) {
            families.generation_acceptances.push({
                key: generationId,
                value: { storage: 'marker', kind: 'generation_acceptance', id: operationId },
            });
        }
    }
    for (const [id, record] of Object.entries(document.execution_receipts))
        await stageFamily('execution_receipts', id, record);
    for (const [id, record] of Object.entries(document.assets)) await stageFamily('assets', id, record);
    for (const [id, record] of Object.entries(document.tool_definitions))
        await stageFamily('tool_definitions', id, record);
    for (const [family, records] of processingRecordGroups(document.processing)) {
        for (const [id, record] of Object.entries(records))
            await stageFamily('processing_records', id, record, tupleKey(family, id));
    }
    for (const id of Object.keys(document.processing.jobs ?? {})) {
        if (idKinds.has(id)) throw new Error('Indexed processing job identity conflicts with a canonical record');
        idKinds.set(id, 'processing job');
        families.identifiers.push({ key: id, value: { storage: 'marker', kind: 'processing_job', id } });
    }
    for (let index = 0; index < document.context.entries.length; index += 1) {
        const entry = document.context.entries[index];
        await stageFamily('context_entries', entry.id, entry);
        families.active_context_order.push({
            key: indexedOrderedKey(index),
            value: { storage: 'marker', kind: 'context_order', id: entry.id },
        });
    }
    const completedCalls = new Set<string>();
    for (const turn of document.turns) {
        for (const block of turn.blocks) if (block.type === 'tool_result') completedCalls.add(block.call_id);
    }
    for (const turn of document.turns) {
        for (const block of turn.blocks) {
            if (block.type === 'tool_call' && !completedCalls.has(block.call_id)) {
                await stageFamily('open_tool_calls', block.call_id, {
                    call_id: block.call_id,
                    turn_id: turn.id,
                    block_id: block.id,
                    call_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(block))).content_hash,
                });
            }
        }
    }

    const resultByCall = new Map<string, string>();
    for (const turn of document.turns) {
        for (const block of turn.blocks) {
            if (block.type === 'tool_result') resultByCall.set(block.call_id, block.id);
        }
    }
    const terminalByCall = new Map<string, string>();
    for (const receipt of Object.values(document.execution_receipts)) {
        terminalByCall.set(receipt.call_id, receipt.id);
    }
    for (const turn of document.turns) {
        for (const block of turn.blocks) {
            if (block.type !== 'tool_call') continue;
            await stageFamily(
                'tool_call_states',
                block.call_id,
                IndexedCallStateSchema.parse({
                    call_id: block.call_id,
                    turn_id: turn.id,
                    block_id: block.id,
                    call_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(block))).content_hash,
                    ...(resultByCall.has(block.call_id) ? { result_block_id: resultByCall.get(block.call_id) } : {}),
                    ...(terminalByCall.has(block.call_id)
                        ? { terminal_receipt_id: terminalByCall.get(block.call_id) }
                        : {}),
                }),
            );
        }
    }

    const contextBytes = canonicalJsonContentBytes(document.context.entries).byteLength;
    if (contextBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES) {
        throw new RangeError('Indexed conversation active context exceeds the working-set bound');
    }
    const { entries: _entries, ...contextWithoutEntries } = document.context;
    const contextHeader = IndexedConversationContextHeaderSchema.parse({
        ...contextWithoutEntries,
        active_entry_count: document.context.entries.length,
        active_entry_bytes: contextBytes,
        context_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(document.context))).content_hash,
    });
    const contextHeaderValue = await stageRecord(store, 'context_header', document.id, contextHeader);
    const {
        jobs: _jobs,
        resolved_inputs: _resolvedInputs,
        attempts: _attempts,
        outputs: _outputs,
        completions: _completions,
        supersessions: _supersessions,
        coverage_receipts: _coverageReceipts,
        ...processingHeader
    } = document.processing;
    const processingHeaderValue = await stageRecord(
        store,
        'processing_header',
        document.id,
        IndexedConversationProcessingHeaderSchema.parse({
            ...processingHeader,
            unresolved_job_count: countUnresolvedProcessingJobs(document.processing),
        }),
    );
    const builtDirectories = await Promise.all(
        (Object.keys(families) as (keyof typeof families)[]).map(async (family) => ({
            family,
            root: await buildPagedRecordIndex(store, families[family]),
        })),
    );
    const directories = Object.fromEntries(
        builtDirectories.filter((item) => item.root !== undefined).map((item) => [item.family, item.root]),
    );
    const root = IndexedConversationRootSchema.parse({
        version: 1,
        validator_profile: INDEXED_CONVERSATION_PROFILE,
        delete_index_profile: INDEXED_CONVERSATION_DELETE_PROFILE,
        tool_call_state_complete: true,
        format: document.format,
        schema_version: document.schema_version,
        experimental_revision: document.experimental_revision,
        source: { conversation_id: document.id, revision: document.revision },
        turn_count: document.turns.length,
        live_turn_count: document.turns.length,
        active_tail_turn_id: document.turns.at(-1)?.id ?? null,
        created_at: document.created_at,
        updated_at: document.updated_at,
        ...(document.lineage === undefined ? {} : { lineage: document.lineage }),
        ...(document.metadata === undefined ? {} : { metadata: document.metadata }),
        context_header: { content_hash: contextHeaderValue.content_hash, size_bytes: contextHeaderValue.size_bytes },
        processing_header: {
            content_hash: processingHeaderValue.content_hash,
            size_bytes: processingHeaderValue.size_bytes,
        },
        directories,
        ...(acceptedOperationId === undefined
            ? {}
            : { accepted_response: acceptedResponse(document, acceptedOperationId) }),
    });
    const rootValue = await stageRecord(store, 'root', document.id, root);
    if (rootValue.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES) {
        throw new RangeError('Indexed conversation root exceeds its manifest bound');
    }
    return { root, locator: { content_hash: rootValue.content_hash, size_bytes: rootValue.size_bytes } };
}

/** Maintain the complete reverse-delete witness for every record family this indexed append accepts. */
async function appendDeleteIndex(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    directories: IndexedConversationRoot['directories'],
    operationId: string,
    batch: ConversationRecordBatch,
): Promise<{ live_turn_count?: number; active_tail_turn_id?: string | null }> {
    if (root.delete_index_profile !== INDEXED_CONVERSATION_DELETE_PROFILE) return {};
    if (root.live_turn_count === undefined || root.active_tail_turn_id === undefined) {
        throw new Error('Indexed delete profile lacks its live-turn witnesses');
    }
    const newTurns = new Map((batch.turns ?? []).map((turn) => [turn.id, turn]));
    const newBlockOwners = new Map<string, string>();
    for (const turn of batch.turns ?? []) for (const block of turn.blocks) newBlockOwners.set(block.id, turn.id);
    const blockers = new Set<string>();
    const markTurn = async (id: string | undefined) => {
        if (id === undefined) return;
        if (newTurns.has(id)) {
            blockers.add(id);
            return;
        }
        const existing = await getPagedRecord(store, root.directories.turns, id);
        if (existing?.storage === 'marker' && existing.kind === 'deleted_turn') {
            throw new Error('Indexed append references a logically deleted turn');
        }
        if (existing?.storage === 'record') blockers.add(id);
    };
    const markBlock = async (id: string) => {
        const owner = newBlockOwners.get(id);
        if (owner !== undefined) {
            blockers.add(owner);
            return;
        }
        const descriptor = await getPagedRecord(store, root.directories.blocks, id);
        if (descriptor?.storage === 'marker' && descriptor.kind === 'deleted_block') {
            throw new Error('Indexed append references a logically deleted block');
        }
        const retained = await getPagedRecord(store, root.directories.block_owners, id);
        if (descriptor?.storage === 'record') {
            if (retained?.storage !== 'marker' || retained.kind !== 'block_owner') {
                throw new Error('Indexed delete profile lacks a retained block owner');
            }
            blockers.add(retained.id);
        }
    };
    for (const turn of batch.turns ?? []) {
        await markTurn(turn.parent_turn_id);
        if (turn.blocks.some((block) => ['tool_call', 'tool_result', 'native_replay'].includes(block.type))) {
            blockers.add(turn.id);
        }
    }
    for (const asset of batch.assets ?? []) {
        if (asset.provenance.type === 'received') await markTurn(asset.provenance.source_turn_id);
    }
    for (const generation of batch.generations ?? []) {
        if (generation.record_source !== 'executed') continue;
        await markTurn(generation.request_receipt.source_tail_turn_id);
        for (const mapping of generation.request_receipt.item_mappings) {
            await markTurn(mapping.canonical_id);
            await markBlock(mapping.canonical_id);
        }
    }
    for (const receipt of batch.execution_receipts ?? []) {
        await markTurn(receipt.result_turn_id);
        await markTurn(receipt.call_source?.turn_id);
    }
    for (const id of blockers) {
        if (await getPagedRecord(store, directories.deletion_blockers, id)) continue;
        directories.deletion_blockers = await putPagedRecord(store, directories.deletion_blockers, id, {
            storage: 'marker',
            kind: 'delete_blocker',
            id,
        });
    }
    const turns = batch.turns ?? [];
    for (const turn of turns) {
        directories.turn_acceptances = await putPagedRecord(store, directories.turn_acceptances, turn.id, {
            storage: 'marker',
            kind: 'turn_acceptance',
            id: operationId,
        });
        for (const block of turn.blocks) {
            directories.block_owners = await putPagedRecord(store, directories.block_owners, block.id, {
                storage: 'marker',
                kind: 'block_owner',
                id: turn.id,
            });
        }
    }
    if (turns.length === 0) {
        return { live_turn_count: root.live_turn_count, active_tail_turn_id: root.active_tail_turn_id };
    }
    const previousTail = root.active_tail_turn_id;
    if (previousTail !== null) {
        const previous = await loadRecord(
            store,
            await getPagedRecord(store, directories.turn_links, previousTail),
            IndexedConversationTurnLinkSchema,
        );
        if (previous.id !== previousTail || previous.next_turn_id !== undefined) {
            throw new Error('Indexed live-turn tail link differs from its authenticated root');
        }
        directories.turn_links = await putPagedRecord(
            store,
            directories.turn_links,
            previousTail,
            await stageRecord(store, 'turn_links', previousTail, { ...previous, next_turn_id: turns[0].id }),
            'replace',
        );
    }
    for (const [index, turn] of turns.entries()) {
        const previousTurnId = index === 0 ? previousTail : turns[index - 1].id;
        const link = IndexedConversationTurnLinkSchema.parse({
            id: turn.id,
            ordinal: root.turn_count + index,
            ...(previousTurnId === null ? {} : { previous_turn_id: previousTurnId }),
            ...(index === turns.length - 1 ? {} : { next_turn_id: turns[index + 1].id }),
        });
        directories.turn_links = await putPagedRecord(
            store,
            directories.turn_links,
            turn.id,
            await stageRecord(store, 'turn_links', turn.id, link),
        );
    }
    return {
        live_turn_count: root.live_turn_count + turns.length,
        active_tail_turn_id: turns.at(-1)?.id ?? previousTail,
    };
}

/** Resolve only active entries and their selected blocks; unrelated cold turns are never read. */
export async function loadIndexedActiveContext(store: IndexedConversationRecordStore, root: IndexedConversationRoot) {
    const header = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'context_header',
            id: root.source.conversation_id,
            ...root.context_header,
        },
        IndexedConversationContextHeaderSchema,
    );
    const entries: z.infer<typeof ContextEntrySchema>[] = [];
    let aggregate = 0;
    for await (const ordered of scanPagedRecords(store, root.directories.active_context_order)) {
        if (entries.length >= header.active_entry_count || entries.length >= 100_000) {
            throw new RangeError('Active context has more records than its bounded header');
        }
        const descriptor = await getPagedRecord(store, root.directories.context_entries, ordered.value.id);
        if (
            descriptor?.storage === 'record' &&
            aggregate + descriptor.size_bytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES
        ) {
            throw new RangeError('Active context exceeds bound before record read');
        }
        const entry = await loadRecord(store, descriptor, ContextEntrySchema);
        entries.push(entry);
        aggregate += canonicalJsonContentBytes(entry).byteLength;
        if (aggregate > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES) throw new RangeError('Active context exceeds bound');
    }
    if (
        entries.length !== header.active_entry_count ||
        canonicalJsonContentBytes(entries).byteLength !== header.active_entry_bytes
    ) {
        throw new Error('Indexed active context differs from its retained count or bytes');
    }
    const context = ConversationContextSchema.parse({
        revision: header.revision,
        entries,
        active_tool_definition_ids: header.active_tool_definition_ids,
        protected_entry_ids: header.protected_entry_ids,
        retrieval_requirements: header.retrieval_requirements,
        ...(header.cache_intent === undefined ? {} : { cache_intent: header.cache_intent }),
    });
    if ((await hashContentBytes(canonicalJsonContentBytes(context))).content_hash !== header.context_fingerprint) {
        throw new Error('Indexed active context differs from its authenticated fingerprint');
    }
    return context;
}

/** Load a selected turn header plus requested blocks, preserving original positions. */
export async function loadIndexedProjectedTurn(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    turnId: string,
    selectedBlockIds?: readonly string[],
    reserveRecordBytes?: (byteLength: number) => void,
) {
    let localBytes = 0;
    const reserve =
        reserveRecordBytes ??
        ((byteLength: number) => {
            localBytes += byteLength;
            if (localBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES) {
                throw new RangeError('Indexed projected turn exceeds working-set bound before record read');
            }
        });
    const descriptor = await getPagedRecord(store, root.directories.turns, turnId);
    if (descriptor?.storage === 'record') reserve(descriptor.size_bytes);
    const header = await loadRecord(store, descriptor, IndexedConversationTurnHeaderSchema);
    if ((await hashContentBytes(canonicalJsonContentBytes(header.block_ids))).content_hash !== header.block_ids_hash) {
        throw new Error('Indexed turn block order differs from its retained hash');
    }
    const selected = selectedBlockIds === undefined ? header.block_ids : selectedBlockIds;
    const positions = selected.map((id) => header.block_ids.indexOf(id));
    if (new Set(selected).size !== selected.length || positions.some((position) => position < 0)) {
        throw new Error('Indexed turn selection names a missing or repeated block');
    }
    const ordered = selected
        .map((id, index) => ({ id, position: positions[index] }))
        .sort((a, b) => a.position - b.position);
    const blocks: z.infer<typeof ContentBlockSchema>[] = [];
    for (const item of ordered) {
        const descriptor = await getPagedRecord(store, root.directories.blocks, item.id);
        if (descriptor?.storage === 'record') reserve(descriptor.size_bytes);
        const block = await loadRecord(store, descriptor, ContentBlockSchema);
        if (block.id !== item.id) throw new Error('Indexed selected block identity differs from its turn header');
        blocks.push(block);
    }
    return {
        completeness: blocks.length === header.block_ids.length ? ('full_turn' as const) : ('selected_blocks' as const),
        header: header.turn,
        selected_blocks: blocks,
        selected_block_positions: ordered.map((item) => item.position),
        source_block_count: header.block_ids.length,
        source_block_ids_hash: header.block_ids_hash,
    };
}

/** Load only selected text, active tool definitions, and exact accepted generation witnesses. */
export async function loadIndexedSelectedTextContext(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    rootLocator: PagedRecordRef,
    maxSelectedBytes = INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
) {
    if (
        !Number.isSafeInteger(maxSelectedBytes) ||
        maxSelectedBytes <= 0 ||
        maxSelectedBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES
    ) {
        throw new RangeError('Indexed selected context byte budget is invalid');
    }
    const root = IndexedConversationRootSchema.parse(rootInput);
    const context = await loadIndexedActiveContext(store, root);
    if (context.revision > root.source.revision) {
        throw new Error('Indexed active context revision exceeds its authenticated root');
    }
    const processing = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    if (processing.enabled) throw new Error('Indexed selected preparation requires processing readiness');
    if (processing.unresolved_job_count === undefined && root.directories.processing_records !== undefined) {
        throw new Error('Indexed selected preparation has no accepted processing job-drain witness');
    }
    if ((processing.unresolved_job_count ?? 0) > 0) {
        throw new Error('Indexed selected preparation has accepted processing jobs outstanding');
    }
    const selections = new Map<string, Set<string> | undefined>();
    for (const entry of context.entries) {
        if (entry.type !== 'source_turn') throw new Error('Indexed text preparation needs a compaction witness');
        const prior = selections.get(entry.turn_id);
        if (!selections.has(entry.turn_id)) {
            selections.set(entry.turn_id, entry.block_ids === undefined ? undefined : new Set(entry.block_ids));
        } else if (prior !== undefined) {
            if (entry.block_ids === undefined) selections.set(entry.turn_id, undefined);
            else for (const id of entry.block_ids) prior.add(id);
        }
    }
    const turns = [];
    const generationWitnesses = new Map<
        string,
        { generation: z.infer<typeof GenerationSchema>; acceptance: OperationReceipt }
    >();
    let totalBytes =
        canonicalJsonContentBytes(context).byteLength + rootLocator.size_bytes + root.processing_header.size_bytes;
    let selectedRecords = 0;
    const reserve = (byteLength: number) => {
        selectedRecords += 1;
        totalBytes += byteLength;
        if (selectedRecords > 100_000 || totalBytes > maxSelectedBytes) {
            throw new RangeError('Indexed selected context exceeds working-set bound before record read');
        }
    };
    for (const [turnId, blockIds] of selections) {
        const turn = await loadIndexedProjectedTurn(
            store,
            root,
            turnId,
            blockIds === undefined ? undefined : [...blockIds],
            reserve,
        );
        if (
            turn.header.kind === 'tool' ||
            turn.header.provenance.type === 'derived' ||
            turn.header.provenance.type === 'imported' ||
            turn.header.parent_turn_id !== undefined ||
            turn.header.execution_id !== undefined ||
            turn.header.exchange_id !== undefined ||
            turn.selected_blocks.some((block) => block.type !== 'text')
        ) {
            throw new Error('Indexed text preparation has unsupported selected content or derivation');
        }
        if (turn.header.kind === 'agent' && 'generation_id' in turn.header && turn.header.generation_id !== undefined) {
            const generationDescriptor = await getPagedRecord(
                store,
                root.directories.generations,
                turn.header.generation_id,
            );
            if (generationDescriptor?.storage === 'record') reserve(generationDescriptor.size_bytes);
            const generation = await loadRecord(store, generationDescriptor, GenerationSchema);
            const accepted = await getPagedRecord(store, root.directories.generation_acceptances, generation.id);
            if (accepted?.storage !== 'marker' || accepted.kind !== 'generation_acceptance') {
                throw new Error('Indexed generated turn has no accepted generation operation');
            }
            const acceptanceDescriptor = await getPagedRecord(store, root.directories.operation_receipts, accepted.id);
            if (acceptanceDescriptor?.storage === 'record') reserve(acceptanceDescriptor.size_bytes);
            const acceptance = await loadRecord(store, acceptanceDescriptor, OperationReceiptSchema);
            if (
                generation.record_source !== 'executed' ||
                !acceptance.accepted_generation_ids?.includes(generation.id) ||
                !acceptance.accepted_turn_ids?.includes(turn.header.id) ||
                acceptance.base_revision !== generation.source.revision ||
                acceptance.result_revision > root.source.revision ||
                generation.request_receipt.source.revision !== generation.source.revision ||
                generation.request_receipt.source.conversation_id !== root.source.conversation_id ||
                generation.request_receipt.request_id !== generation.request_id ||
                generation.source.conversation_id !== root.source.conversation_id ||
                generation.source.revision >= root.source.revision
            ) {
                throw new Error('Indexed generated turn differs from its accepted request and response chain');
            }
            generationWitnesses.set(generation.id, { generation, acceptance });
        }
        if (
            turn.header.kind === 'program' &&
            turn.header.provenance.type === 'inserted' &&
            turn.header.provenance.operation_id !== undefined
        ) {
            const receiptDescriptor = await getPagedRecord(
                store,
                root.directories.operation_receipts,
                turn.header.provenance.operation_id,
            );
            if (receiptDescriptor?.storage === 'record') reserve(receiptDescriptor.size_bytes);
            const receipt = await loadRecord(store, receiptDescriptor, OperationReceiptSchema);
            if (
                !receipt.accepted_turn_ids?.includes(turn.header.id) ||
                receipt.result_revision > root.source.revision
            ) {
                throw new Error('Indexed program turn differs from its accepted operation');
            }
        }
        turns.push(turn);
    }
    const toolDefinitions = new Map<string, z.infer<typeof ToolDefinitionSchema>>();
    for (const id of context.active_tool_definition_ids) {
        const descriptor = await getPagedRecord(store, root.directories.tool_definitions, id);
        if (descriptor?.storage === 'record') reserve(descriptor.size_bytes);
        toolDefinitions.set(id, await loadRecord(store, descriptor, ToolDefinitionSchema));
    }
    const tail =
        root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE
            ? root.active_tail_turn_id === null
                ? undefined
                : root.active_tail_turn_id === undefined
                  ? null
                  : { storage: 'marker' as const, kind: 'turn_order', id: root.active_tail_turn_id }
            : root.turn_count === 0
              ? undefined
              : await getPagedRecord(store, root.directories.turn_order, indexedOrderedKey(root.turn_count - 1));
    if (tail === null || (tail !== undefined && (tail.storage !== 'marker' || tail.kind !== 'turn_order'))) {
        throw new Error('Indexed source tail turn is unavailable');
    }
    const selected = IndexedConversationSelectedContextSchema.parse({
        completeness: 'selected_text_pending_admission',
        source: root.source,
        root: rootLocator,
        source_turn_count: root.live_turn_count ?? root.turn_count,
        context,
        turns,
        tool_definitions: Object.fromEntries(toolDefinitions),
        assets: {},
        generation_witnesses: Object.fromEntries(generationWitnesses),
        ...(tail === undefined ? {} : { source_tail_turn_id: tail.id }),
    });
    if (canonicalJsonContentBytes(selected).byteLength > maxSelectedBytes) {
        throw new RangeError('Indexed selected context exceeds the bounded working-set profile');
    }
    return selected;
}

/** A fresh inference after program append must bind that accepted input and have no accepted response. */
export async function assertIndexedFreshProgramInput(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    input: { operation_id: string; result_revision: number; response_operation_id: string },
): Promise<OperationReceipt> {
    const root = IndexedConversationRootSchema.parse(rootInput);
    const receipt = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.operation_receipts, input.operation_id),
        OperationReceiptSchema,
    );
    if (
        receipt.operation_kind !== undefined ||
        receipt.result_revision !== root.source.revision ||
        receipt.result_revision !== input.result_revision ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.accepted_turn_ids?.length !== 1 ||
        receipt.accepted_context_entry_ids?.length !== 1 ||
        receipt.accepted_generation_ids?.length !== 0 ||
        receipt.accepted_execution_receipt_ids?.length !== 0
    ) {
        throw new Error('Indexed fresh preparation is not pinned to its accepted program input');
    }
    const turn = await loadIndexedProjectedTurn(store, root, receipt.accepted_turn_ids[0]);
    if (
        turn.header.kind !== 'program' ||
        turn.header.provenance.type !== 'inserted' ||
        turn.header.provenance.operation_id !== receipt.id
    ) {
        throw new Error('Indexed fresh preparation input is not an ordinary program turn');
    }
    if (await getPagedRecord(store, root.directories.operation_receipts, input.response_operation_id)) {
        throw new Error('Indexed response is already accepted; use exact recovery instead of fresh inference');
    }
    return receipt;
}

/** Stage one ordinary program append against an authenticated indexed root; the caller CASes the locator. */
export async function stageIndexedProgramAppend(
    rootInput: IndexedConversationRoot,
    input: IndexedProgramRecords,
    store: IndexedConversationRecordStore,
): Promise<{ root: IndexedConversationRoot; locator?: PagedRecordRef; receipt: OperationReceipt; applied: boolean }> {
    if (!preflightJsonInput(input, { max_bytes: 512 * 1024 }).success) {
        throw new Error('Indexed program append is not bounded JSON');
    }
    const command = IndexedProgramRecordsSchema.parse(input);
    const root = IndexedConversationRootSchema.parse(rootInput);
    if (
        root.source.conversation_id !== command.conversation_id ||
        command.turn.kind !== 'program' ||
        command.turn.authority !== 'ordinary' ||
        command.turn.provenance.type !== 'inserted' ||
        command.turn.provenance.operation_id !== command.operation_id ||
        command.turn.blocks.length !== 1 ||
        !['text', 'json'].includes(command.turn.blocks[0].type) ||
        command.turn.status !== 'completed' ||
        command.turn.parent_turn_id !== undefined ||
        command.turn.execution_id !== undefined ||
        command.entry.type !== 'source_turn' ||
        command.entry.turn_id !== command.turn.id ||
        command.entry.block_ids !== undefined ||
        command.turn.timestamps.recorded_at !== command.recorded_at ||
        Date.parse(command.recorded_at) < Date.parse(root.created_at) ||
        (command.turn.timestamps.started_at !== undefined &&
            command.turn.timestamps.completed_at !== undefined &&
            Date.parse(command.turn.timestamps.started_at) > Date.parse(command.turn.timestamps.completed_at))
    ) {
        throw new Error('Indexed program append records do not identify one ordinary program turn');
    }
    const fingerprint = await fingerprintJson({
        turns: [command.turn],
        context_entries: [command.entry],
    });
    if (fingerprint !== command.payload_fingerprint) throw new Error('Indexed program append fingerprint differs');
    const prior = await getPagedRecord(store, root.directories.operation_receipts, command.operation_id);
    if (prior !== undefined) {
        const receipt = await loadRecord(store, prior, OperationReceiptSchema);
        if (
            receipt.operation_kind !== undefined ||
            receipt.conversation_id !== root.source.conversation_id ||
            receipt.payload_fingerprint !== fingerprint ||
            receipt.base_revision !== command.expected_revision ||
            receipt.result_revision !== command.expected_revision + 1 ||
            receipt.result_revision > root.source.revision ||
            receipt.recorded_at !== command.recorded_at ||
            JSON.stringify(receipt.accepted_turn_ids) !== JSON.stringify([command.turn.id]) ||
            JSON.stringify(receipt.accepted_context_entry_ids) !== JSON.stringify([command.entry.id]) ||
            (await fingerprintJson(receipt.accepted_context_entries)) !== (await fingerprintJson([command.entry])) ||
            (await fingerprintJson(receipt.accepted_tool_selection)) !==
                (await fingerprintJson({ kind: 'unchanged' })) ||
            (receipt.accepted_generation_ids?.length ?? 0) !== 0 ||
            (receipt.accepted_asset_ids?.length ?? 0) !== 0 ||
            (receipt.accepted_execution_receipt_ids?.length ?? 0) !== 0 ||
            (receipt.accepted_tool_definition_ids?.length ?? 0) !== 0
        ) {
            throw new Error('Indexed program append conflicts with its accepted operation');
        }
        const retained = await loadIndexedAcceptedTurn(store, root, command.turn.id, receipt);
        if (
            retained.completeness !== 'full_turn' ||
            (await fingerprintJson({ ...retained.header, blocks: retained.selected_blocks })) !==
                (await fingerprintJson(command.turn))
        ) {
            throw new Error('Indexed program append records differ from accepted turn');
        }
        const acceptedEntry = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.context_entries, command.entry.id),
            ContextEntrySchema,
        );
        if ((await fingerprintJson(acceptedEntry)) !== (await fingerprintJson(command.entry))) {
            throw new Error('Indexed program append entry differs from accepted record');
        }
        return { root, receipt, applied: false };
    }
    if (root.source.revision !== command.expected_revision) throw new Error('Indexed program append revision conflict');
    if (root.source.revision === Number.MAX_SAFE_INTEGER || root.turn_count === Number.MAX_SAFE_INTEGER) {
        throw new RangeError('Indexed conversation revision or turn count is exhausted');
    }
    const processing = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    if (processing.enabled) throw new Error('Indexed program append requires the processing outbox');
    for (const id of [command.operation_id, command.turn.id, command.turn.blocks[0].id, command.entry.id]) {
        if (await getPagedRecord(store, root.directories.identifiers, id)) {
            throw new Error('Indexed program append identity already exists');
        }
    }
    const context = await loadIndexedActiveContext(store, root);
    if (context.revision > root.source.revision) {
        throw new Error('Indexed program append context revision differs from its authenticated root');
    }
    const nextRevision = root.source.revision + 1;
    const nextContext = ConversationContextSchema.parse({
        ...context,
        revision: nextRevision,
        entries: [...context.entries, command.entry],
    });
    const activeBytes = canonicalJsonContentBytes(nextContext.entries).byteLength;
    if (activeBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES || nextContext.entries.length > 100_000) {
        throw new RangeError('Indexed active context exceeds its bounded profile');
    }
    const receipt = OperationReceiptSchema.parse({
        id: command.operation_id,
        conversation_id: command.conversation_id,
        payload_fingerprint: fingerprint,
        base_revision: command.expected_revision,
        result_revision: nextRevision,
        recorded_at: command.recorded_at,
        accepted_turn_ids: [command.turn.id],
        accepted_generation_ids: [],
        accepted_asset_ids: [],
        accepted_tool_definition_ids: [],
        accepted_execution_receipt_ids: [],
        accepted_context_entry_ids: [command.entry.id],
        accepted_context_entries: [command.entry],
        accepted_tool_selection: { kind: 'unchanged' },
    });
    const block = command.turn.blocks[0];
    const { blocks: _blocks, ...turnHeader } = command.turn;
    const staged = [
        [
            'turns',
            command.turn.id,
            {
                turn: turnHeader,
                source: 'ordinary',
                block_ids: [block.id],
                block_ids_hash: (await hashContentBytes(canonicalJsonContentBytes([block.id]))).content_hash,
            },
        ],
        ['blocks', block.id, block],
        ['context_entries', command.entry.id, command.entry],
        ['operation_receipts', command.operation_id, receipt],
    ] as const;
    const directories = { ...root.directories };
    for (const [family, id, content] of staged) {
        directories[family] = await putPagedRecord(
            store,
            directories[family],
            id,
            await stageRecord(store, family, id, content),
        );
    }
    for (const [id, kind] of [
        [command.operation_id, 'operation_receipt'],
        [command.turn.id, 'turn'],
        [block.id, 'block'],
        [command.entry.id, 'context_entry'],
    ] as const) {
        directories.identifiers = await putPagedRecord(store, directories.identifiers, id, {
            storage: 'marker',
            kind,
            id,
        });
    }
    directories.turn_order = await putPagedRecord(store, directories.turn_order, indexedOrderedKey(root.turn_count), {
        storage: 'marker',
        kind: 'turn_order',
        id: command.turn.id,
    });
    directories.active_context_order = await putPagedRecord(
        store,
        directories.active_context_order,
        indexedOrderedKey(context.entries.length),
        { storage: 'marker', kind: 'context_order', id: command.entry.id },
    );
    const deleteIndex = await appendDeleteIndex(store, root, directories, command.operation_id, {
        turns: [command.turn],
        context_entries: [command.entry],
    });
    const { entries: _entries, ...contextFields } = nextContext;
    const contextHeaderValue = await stageRecord(
        store,
        'context_header',
        root.source.conversation_id,
        IndexedConversationContextHeaderSchema.parse({
            ...contextFields,
            active_entry_count: nextContext.entries.length,
            active_entry_bytes: activeBytes,
            context_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(nextContext))).content_hash,
        }),
    );
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        source: { ...root.source, revision: nextRevision },
        turn_count: root.turn_count + 1,
        ...deleteIndex,
        updated_at: command.recorded_at,
        context_header: { content_hash: contextHeaderValue.content_hash, size_bytes: contextHeaderValue.size_bytes },
        directories,
    });
    const rootValue = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootValue.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES) {
        throw new RangeError('Indexed conversation root exceeds its manifest bound');
    }
    return {
        root: nextRoot,
        locator: { content_hash: rootValue.content_hash, size_bytes: rootValue.size_bytes },
        receipt,
        applied: true,
    };
}

function idsOf(records: readonly { id: string }[] | undefined): string[] {
    return records?.map((record) => record.id) ?? [];
}

function sameIndexedRecord(left: unknown, right: unknown): boolean {
    return canonicalJsonContentString(left) === canonicalJsonContentString(right);
}

function comparableIndexedRecord(kind: string, record: Record<string, unknown>): Record<string, unknown> {
    const result = { ...record };
    if (kind === 'turn' || kind === 'generation') delete result.timestamps;
    if (kind === 'asset') delete result.created_at;
    if (kind === 'execution receipt') delete result.recorded_at;
    if (kind === 'generation' && result.request_receipt && typeof result.request_receipt === 'object') {
        const receipt: Record<string, unknown> = { ...(result.request_receipt as Record<string, unknown>) };
        delete receipt.recorded_at;
        result.request_receipt = receipt;
    }
    return result;
}

async function indexedRecordById<Shape extends z.ZodType>(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    family: keyof IndexedConversationRoot['directories'],
    id: string,
    schema: Shape,
): Promise<z.infer<Shape> | undefined> {
    const descriptor = await getPagedRecord(store, root.directories[family], id);
    return descriptor === undefined ? undefined : loadRecord(store, descriptor, schema);
}

/** A deleted turn is recoverable only through the immutable root authenticated by its current tombstone. */
async function loadIndexedAcceptedTurn(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    turnId: string,
    acceptance: OperationReceipt,
) {
    const descriptor = await getPagedRecord(store, root.directories.turns, turnId);
    if (descriptor?.storage === 'record') return loadIndexedProjectedTurn(store, root, turnId);
    if (descriptor?.storage !== 'marker' || descriptor.kind !== 'deleted_turn') {
        throw new Error('Indexed accepted turn is unavailable');
    }
    if (root.delete_index_profile !== INDEXED_CONVERSATION_DELETE_PROFILE) {
        throw new Error('Indexed deleted turn has no complete deletion profile');
    }
    const tombstone = await indexedRecordById(
        store,
        root,
        'deleted_turns',
        turnId,
        IndexedConversationDeletedTurnSchema,
    );
    if (
        !tombstone ||
        tombstone.deleted_turn.id !== turnId ||
        tombstone.deleted_turn.accepted_operation_id !== acceptance.id
    ) {
        throw new Error('Indexed accepted turn lacks its exact tombstone');
    }
    const deletion = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        tombstone.deleted_turn.operation_id,
        OperationReceiptSchema,
    );
    const detail = deletion?.conversation_delete;
    const ref = detail?.deleted_turns.find((item) => item.id === turnId);
    if (
        deletion?.operation_kind !== 'conversation_delete' ||
        deletion.base_revision !== tombstone.deleted_turn.source_revision ||
        !ref ||
        !sameIndexedRecord(ref, {
            id: turnId,
            fingerprint: tombstone.deleted_turn.fingerprint,
            block_ids: tombstone.deleted_turn.block_ids,
            accepted_operation_id: acceptance.id,
        }) ||
        detail?.source_fingerprint !==
            (await fingerprintJson({
                domain: 'llumiverse.conversation.indexed-delete-source',
                version: 1,
                root: tombstone.predecessor_root,
                turn_ids: detail?.deleted_turns.map((item) => item.id),
            }))
    ) {
        throw new Error('Indexed accepted turn deletion receipt differs from its tombstone');
    }
    const predecessor = await loadRecord(
        store,
        { storage: 'record', kind: 'root', id: root.source.conversation_id, ...tombstone.predecessor_root },
        IndexedConversationRootSchema,
    );
    if (
        predecessor.source.conversation_id !== root.source.conversation_id ||
        predecessor.source.revision !== deletion.base_revision ||
        predecessor.delete_index_profile !== INDEXED_CONVERSATION_DELETE_PROFILE
    ) {
        throw new Error('Indexed accepted turn predecessor differs from its delete receipt');
    }
    const originalAcceptance = await indexedRecordById(
        store,
        predecessor,
        'operation_receipts',
        acceptance.id,
        OperationReceiptSchema,
    );
    if (!originalAcceptance || !sameIndexedRecord(originalAcceptance, acceptance)) {
        throw new Error('Indexed accepted turn predecessor lacks its append receipt');
    }
    const original = await loadIndexedProjectedTurn(store, predecessor, turnId);
    if (
        original.completeness !== 'full_turn' ||
        !sameIndexedRecord(
            original.selected_blocks.map((block) => block.id),
            tombstone.deleted_turn.block_ids,
        ) ||
        (await fingerprintJson({ ...original.header, blocks: original.selected_blocks })) !==
            tombstone.deleted_turn.fingerprint
    ) {
        throw new Error('Indexed accepted turn predecessor differs from its tombstone fingerprint');
    }
    return original;
}

async function acceptedIndexedBatch(
    root: IndexedConversationRoot,
    batch: ConversationRecordBatch,
    options: AppendConversationRecordsOptions,
    receipt: OperationReceipt,
    store: IndexedConversationRecordStore,
): Promise<void> {
    const selections =
        batch.active_tool_definition_ids === undefined
            ? { kind: 'unchanged' }
            : { kind: 'replace', definition_ids: [...batch.active_tool_definition_ids] };
    if (
        receipt.operation_kind !== undefined ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.payload_fingerprint !== options.payload_fingerprint ||
        receipt.base_revision !== options.expected_revision ||
        receipt.result_revision !== options.expected_revision + 1 ||
        receipt.result_revision > root.source.revision ||
        (receipt.accepted_tool_selection === undefined
            ? batch.active_tool_definition_ids !== undefined
            : !sameIndexedRecord(receipt.accepted_tool_selection, selections))
    )
        throw new Error('Indexed append conflicts with its accepted operation');
    const families = [
        ['turns', batch.turns, receipt.accepted_turn_ids],
        ['generations', batch.generations, receipt.accepted_generation_ids],
        ['assets', batch.assets, receipt.accepted_asset_ids],
        ['tool_definitions', batch.tool_definitions, receipt.accepted_tool_definition_ids],
        ['execution_receipts', batch.execution_receipts, receipt.accepted_execution_receipt_ids],
        ['context_entries', batch.context_entries, receipt.accepted_context_entry_ids],
    ] as const;
    for (const [family, records, accepted] of families) {
        if (!sameIndexedRecord(idsOf(records), accepted ?? [])) {
            throw new Error(`Indexed append changes accepted ${family} identities`);
        }
    }
    if (!sameIndexedRecord(batch.context_entries ?? [], receipt.accepted_context_entries ?? [])) {
        throw new Error('Indexed append changes accepted context entries');
    }
    for (const turn of batch.turns ?? []) {
        const retained = await loadIndexedAcceptedTurn(store, root, turn.id, receipt);
        if (
            retained.completeness !== 'full_turn' ||
            !sameIndexedRecord(
                comparableIndexedRecord('turn', { ...retained.header, blocks: retained.selected_blocks }),
                comparableIndexedRecord('turn', turn),
            )
        )
            throw new Error('Indexed append changes accepted turn');
    }
    const schemas = {
        generations: GenerationSchema,
        assets: AssetSchema,
        tool_definitions: ToolDefinitionSchema,
        execution_receipts: ExecutionReceiptSchema,
        context_entries: ContextEntrySchema,
    } as const;
    for (const family of Object.keys(schemas) as (keyof typeof schemas)[]) {
        const comparableKind = {
            generations: 'generation',
            assets: 'asset',
            tool_definitions: 'tool definition',
            execution_receipts: 'execution receipt',
            context_entries: 'context entry',
        }[family];
        for (const record of batch[family] ?? []) {
            const retained = await indexedRecordById(store, root, family, record.id, schemas[family]);
            if (
                retained === undefined ||
                !sameIndexedRecord(
                    comparableIndexedRecord(comparableKind, retained),
                    comparableIndexedRecord(comparableKind, record),
                )
            )
                throw new Error(`Indexed append changes accepted ${family} record`);
        }
    }
}

/**
 * Stage a bounded append without materializing lifetime history. This first indexed batch contract
 * admits only dependency forms it can prove by authenticated point lookup; other forms fail closed.
 * The host alone publishes the returned locator by an exact run-head CAS.
 */
export async function stageIndexedRecordBatch(
    rootInput: IndexedConversationRoot,
    input: IndexedRecordBatchCommand,
    store: IndexedConversationRecordStore,
): Promise<{ root: IndexedConversationRoot; locator?: PagedRecordRef; receipt: OperationReceipt; applied: boolean }> {
    if (!preflightJsonInput(input, { max_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES }).success) {
        throw new Error('Indexed append command is not bounded JSON');
    }
    const parsed = IndexedRecordBatchCommandSchema.parse(input);
    const { batch, options } = parsed;
    const root = IndexedConversationRootSchema.parse(rootInput);
    if (parsed.conversation_id !== root.source.conversation_id) throw new Error('Indexed append conversation differs');
    const prior = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        options.operation_id,
        OperationReceiptSchema,
    );
    if (prior !== undefined) {
        await acceptedIndexedBatch(root, batch, options, prior, store);
        return { root, receipt: prior, applied: false };
    }
    if (options.expected_revision !== root.source.revision) throw new Error('Indexed append revision conflict');
    if (
        root.source.revision === Number.MAX_SAFE_INTEGER ||
        root.turn_count + (batch.turns?.length ?? 0) > Number.MAX_SAFE_INTEGER
    ) {
        throw new RangeError('Indexed append revision or turn count is exhausted');
    }
    if (Date.parse(options.recorded_at) < Date.parse(root.created_at))
        throw new Error('Indexed append timestamp predates source');
    const processing = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    if (processing.enabled) throw new Error('Indexed append requires the processing outbox');
    const newTurns = new Map((batch.turns ?? []).map((turn) => [turn.id, turn]));
    const newGenerations = new Map((batch.generations ?? []).map((generation) => [generation.id, generation]));
    const newAssets = new Map((batch.assets ?? []).map((asset) => [asset.id, asset]));
    const newDefinitions = new Map((batch.tool_definitions ?? []).map((definition) => [definition.id, definition]));
    const newExecution = new Map((batch.execution_receipts ?? []).map((receipt) => [receipt.id, receipt]));
    const newEntries = new Map((batch.context_entries ?? []).map((entry) => [entry.id, entry]));
    for (const [records, count] of [
        [newTurns, batch.turns?.length ?? 0],
        [newGenerations, batch.generations?.length ?? 0],
        [newAssets, batch.assets?.length ?? 0],
        [newDefinitions, batch.tool_definitions?.length ?? 0],
        [newExecution, batch.execution_receipts?.length ?? 0],
        [newEntries, batch.context_entries?.length ?? 0],
    ] as const) {
        if (records.size !== count) throw new Error('Indexed append has duplicate records in one family');
    }
    const globalIds = new Map<string, string>();
    const register = async (id: string, kind: string, allowRetainedDefinition = false) => {
        if (globalIds.has(id)) throw new Error(`Indexed append duplicates ${id}`);
        globalIds.set(id, kind);
        const retained = await getPagedRecord(store, root.directories.identifiers, id);
        if (retained !== undefined && !(allowRetainedDefinition && retained.kind === 'tool definition')) {
            throw new Error(`Indexed append identity ${id} already exists`);
        }
    };
    await register(options.operation_id, 'operation receipt');
    for (const turn of batch.turns ?? []) {
        await register(turn.id, 'turn');
        for (const block of turn.blocks) {
            await register(block.id, 'block');
            if (block.type === 'tool_call') await register(block.call_id, 'tool call');
            if (block.type === 'tool_result') {
                for (const nested of block.content) await register(nested.id, 'block');
            }
        }
    }
    for (const generation of batch.generations ?? []) {
        await register(generation.id, 'generation');
        if (generation.request_receipt) await register(generation.request_receipt.id, 'request receipt');
    }
    for (const asset of batch.assets ?? []) await register(asset.id, 'asset');
    for (const definition of batch.tool_definitions ?? []) await register(definition.id, 'tool definition', true);
    for (const receipt of batch.execution_receipts ?? []) await register(receipt.id, 'execution receipt');
    for (const entry of batch.context_entries ?? []) await register(entry.id, 'context entry');
    if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE) {
        for (const item of batch.generations ?? []) {
            if (item.record_source !== 'executed') continue;
            for (const id of [
                ...item.request_receipt.item_mappings.map((mapping) => mapping.canonical_id),
                ...item.request_receipt.asset_versions.map((binding) => binding.asset_id),
            ]) {
                if (!globalIds.has(id) && !(await getPagedRecord(store, root.directories.identifiers, id))) {
                    globalIds.set(id, 'historical_reference');
                }
            }
        }
    }
    const definition = async (id: string) =>
        newDefinitions.get(id) ?? indexedRecordById(store, root, 'tool_definitions', id, ToolDefinitionSchema);
    const asset = async (id: string) => newAssets.get(id) ?? indexedRecordById(store, root, 'assets', id, AssetSchema);
    const generation = async (id: string) =>
        newGenerations.get(id) ?? indexedRecordById(store, root, 'generations', id, GenerationSchema);
    const turnExists = async (id: string) =>
        newTurns.has(id) || (await getPagedRecord(store, root.directories.turns, id))?.storage === 'record';
    const callStates = new Map<string, IndexedCallState>();
    const readCall = async (id: string): Promise<IndexedCallState | undefined> => {
        if (callStates.has(id)) return callStates.get(id);
        if (!root.tool_call_state_complete) throw new Error('Indexed tool-call lookup is unavailable on this root');
        const existing = await indexedRecordById(store, root, 'tool_call_states', id, IndexedCallStateSchema);
        if (existing) {
            const retainedTurn = await loadIndexedProjectedTurn(store, root, existing.turn_id, [existing.block_id]);
            if (
                retainedTurn.header.kind !== 'agent' ||
                retainedTurn.header.provenance.type !== 'generated' ||
                !('generation_id' in retainedTurn.header) ||
                retainedTurn.header.generation_id === undefined ||
                retainedTurn.selected_blocks[0]?.type !== 'tool_call' ||
                retainedTurn.selected_blocks[0].call_id !== id
            ) {
                throw new Error('Indexed tool-call state differs from its retained turn');
            }
            const generationId = retainedTurn.header.generation_id;
            const accepted = await getPagedRecord(store, root.directories.generation_acceptances, generationId);
            const generationRecord = await indexedRecordById(
                store,
                root,
                'generations',
                generationId,
                GenerationSchema,
            );
            const receipt =
                accepted?.storage === 'marker' && accepted.kind === 'generation_acceptance'
                    ? await indexedRecordById(store, root, 'operation_receipts', accepted.id, OperationReceiptSchema)
                    : undefined;
            if (
                generationRecord?.record_source !== 'executed' ||
                !receipt?.accepted_generation_ids?.includes(generationId) ||
                !receipt.accepted_turn_ids?.includes(existing.turn_id) ||
                receipt.result_revision > root.source.revision
            )
                throw new Error('Indexed tool call lacks an accepted generation chain');
        }
        if (existing) callStates.set(id, existing);
        return existing;
    };
    for (const [index, turn] of (batch.turns ?? []).entries()) {
        if (
            turn.provenance.type === 'derived' ||
            turn.provenance.type === 'imported' ||
            (turn.kind === 'user' && turn.provenance.type !== 'received') ||
            (turn.kind === 'agent' && turn.provenance.type !== 'generated' && turn.provenance.type !== 'received') ||
            (turn.kind === 'tool' &&
                turn.provenance.type !== 'received' &&
                !(turn.provenance.type === 'inserted' && turn.provenance.operation_id === options.operation_id)) ||
            (turn.kind === 'program' &&
                (turn.provenance.type !== 'inserted' || turn.provenance.operation_id !== options.operation_id))
        )
            throw new Error('Indexed append does not support this turn provenance');
        if (turn.parent_turn_id && !(await turnExists(turn.parent_turn_id))) {
            throw new Error('Indexed append parent turn is unavailable');
        }
        if (
            turn.parent_turn_id &&
            newTurns.has(turn.parent_turn_id) &&
            (batch.turns ?? []).findIndex((item) => item.id === turn.parent_turn_id) >= index
        ) {
            throw new Error('Indexed append parent turn must precede its child');
        }
        if (
            turn.execution_id &&
            !newExecution.has(turn.execution_id) &&
            !(await getPagedRecord(store, root.directories.execution_receipts, turn.execution_id))
        ) {
            throw new Error('Indexed append execution receipt is unavailable');
        }
        if (
            turn.timestamps.started_at &&
            turn.timestamps.completed_at &&
            Date.parse(turn.timestamps.started_at) > Date.parse(turn.timestamps.completed_at)
        ) {
            throw new Error('Indexed append turn timestamp order is invalid');
        }
        if (turn.kind === 'agent' && turn.provenance.type === 'generated') {
            if (!('generation_id' in turn) || turn.generation_id === undefined) {
                throw new Error('Indexed generated agent turn has no generation identity');
            }
            const record = await generation(turn.generation_id);
            if (
                !record ||
                (record.status === 'failed' && turn.status !== 'failed') ||
                (record.status === 'cancelled' && turn.status !== 'interrupted') ||
                (record.status === 'completed' && turn.status === 'failed')
            ) {
                throw new Error('Indexed generated turn differs from its generation');
            }
        }
        for (const block of turn.blocks) {
            if (block.type === 'native_replay' || block.type === 'external_reference') {
                throw new Error('Indexed append cannot validate this replay or external reference');
            }
            if (block.type === 'tool_call') {
                if (
                    turn.kind !== 'agent' ||
                    turn.provenance.type !== 'generated' ||
                    !('generation_id' in turn) ||
                    turn.generation_id === undefined ||
                    !newGenerations.has(turn.generation_id)
                ) {
                    throw new Error('Indexed tool call requires an accepted generated agent turn');
                }
                if (block.arguments.type === 'externalized_json') {
                    throw new Error('Indexed append cannot validate externalized tool arguments');
                }
                if (block.definition_id) {
                    const pinned = await definition(block.definition_id);
                    if (!pinned || pinned.name !== block.tool_name)
                        throw new Error('Indexed tool definition is unavailable or mismatched');
                }
                callStates.set(
                    block.call_id,
                    IndexedCallStateSchema.parse({
                        call_id: block.call_id,
                        turn_id: turn.id,
                        block_id: block.id,
                        call_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(block))).content_hash,
                    }),
                );
            }
            if (block.type === 'tool_result') {
                const call = await readCall(block.call_id);
                if (!call || call.result_block_id) throw new Error('Indexed tool result has no open retained call');
                const callBlock =
                    newTurns.get(call.turn_id)?.blocks.find((item) => item.id === call.block_id) ??
                    (await indexedRecordById(store, root, 'blocks', call.block_id, ContentBlockSchema));
                if (
                    callBlock?.type !== 'tool_call' ||
                    callBlock.arguments.type === 'invalid' ||
                    (await hashContentBytes(canonicalJsonContentBytes(callBlock))).content_hash !==
                        call.call_fingerprint
                ) {
                    throw new Error('Indexed tool result call proof is unavailable');
                }
                if (block.content.some((item) => !['text', 'json'].includes(item.type))) {
                    throw new Error('Indexed tool-result content dependency is unsupported');
                }
                if (call.terminal_receipt_id) {
                    const terminal = await indexedRecordById(
                        store,
                        root,
                        'execution_receipts',
                        call.terminal_receipt_id,
                        ExecutionReceiptSchema,
                    );
                    if (!terminal || terminal.call_id !== block.call_id || terminal.status !== block.status) {
                        throw new Error('Indexed tool result differs from accepted terminal receipt');
                    }
                }
                callStates.set(block.call_id, { ...call, result_block_id: block.id });
            }
            if (
                block.type === 'image' ||
                block.type === 'audio' ||
                block.type === 'video' ||
                block.type === 'document'
            ) {
                if (block.selection !== undefined) {
                    throw new Error('Indexed append cannot validate selected media region yet');
                }
                const retained = await asset(block.asset_id);
                if (!retained || retained.kind !== block.type)
                    throw new Error('Indexed media asset is unavailable or mismatched');
            }
        }
    }
    for (const item of batch.assets ?? []) {
        if (item.provenance.type === 'derived') {
            throw new Error('Indexed append cannot validate derived-asset provenance yet');
        }
        if (item.provenance.type === 'generated' && !(await generation(item.provenance.generation_id))) {
            throw new Error('Indexed generated asset has no generation');
        }
        if (
            item.provenance.type === 'received' &&
            item.provenance.source_turn_id &&
            !(await turnExists(item.provenance.source_turn_id))
        ) {
            throw new Error('Indexed received asset has no source turn');
        }
    }
    for (const item of batch.generations ?? []) {
        if (item.record_source !== 'executed') {
            throw new Error('Indexed append requires an executed generation');
        }
        if (item.usage !== undefined) {
            validateUsage(
                item.usage,
                `/generations/${item.id}/usage`,
                (code, path, message) => {
                    throw new Error(`Indexed generation usage ${code} at ${path}: ${message}`);
                },
                item.id,
            );
        }
        const request = item.request_receipt;
        if (
            item.source.conversation_id !== root.source.conversation_id ||
            item.source.revision > root.source.revision ||
            request.source.conversation_id !== item.source.conversation_id ||
            request.source.revision !== item.source.revision ||
            request.request_id !== item.request_id ||
            request.attempt_id !== item.attempt_id ||
            request.target.model !== item.requested_model ||
            request.target.provider !== item.provider ||
            request.target.protocol !== item.protocol ||
            request.target.adapter_version !== item.adapter_version
        )
            throw new Error('Indexed generation does not match its accepted request');
        if (
            request.source_tail_turn_id &&
            !(await getPagedRecord(store, root.directories.turns, request.source_tail_turn_id))
        ) {
            throw new Error('Indexed generation request tail is unavailable');
        }
        for (const binding of request.asset_versions) {
            const retained = await indexedRecordById(store, root, 'assets', binding.asset_id, AssetSchema);
            if (retained?.content_hash && retained.content_hash !== binding.content_hash) {
                throw new Error('Indexed generation asset binding differs from retained content');
            }
        }
        for (const mapping of request.item_mappings) {
            const kind = (await getPagedRecord(store, root.directories.identifiers, mapping.canonical_id))?.kind;
            if (
                kind &&
                ![
                    mapping.kind === 'turn' ? 'turn' : mapping.kind === 'call' ? 'tool call' : 'block',
                    mapping.kind === 'turn' ? 'source turn' : '',
                    mapping.kind === 'turn' ? 'replacement turn' : '',
                ].includes(kind)
            )
                throw new Error('Indexed generation item mapping has another retained kind');
        }
        if (
            item.timestamps.started_at &&
            item.timestamps.completed_at &&
            Date.parse(item.timestamps.started_at) > Date.parse(item.timestamps.completed_at)
        ) {
            throw new Error('Indexed generation timestamp order is invalid');
        }
    }
    if (
        (batch.execution_receipts?.length ||
            batch.turns?.some((turn) =>
                turn.blocks.some((block) => block.type === 'tool_call' || block.type === 'tool_result'),
            )) &&
        !root.tool_call_state_complete
    ) {
        throw new Error('Indexed tool-call lookup is unavailable on this root');
    }
    for (const item of batch.execution_receipts ?? []) {
        const call = await readCall(item.call_id);
        if (!call || call.terminal_receipt_id) throw new Error('Indexed execution receipt has no unterminated call');
        const callBlock =
            newTurns.get(call.turn_id)?.blocks.find((block) => block.id === call.block_id) ??
            (await indexedRecordById(store, root, 'blocks', call.block_id, ContentBlockSchema));
        if (
            callBlock?.type !== 'tool_call' ||
            callBlock.executor !== item.executor ||
            (await hashContentBytes(canonicalJsonContentBytes(callBlock))).content_hash !== call.call_fingerprint
        ) {
            throw new Error('Indexed execution receipt call proof differs');
        }
        if (
            item.call_source &&
            (item.call_source.call_id !== item.call_id ||
                item.call_source.turn_id !== call.turn_id ||
                item.call_source.block_id !== call.block_id ||
                item.call_source.call_fingerprint !== call.call_fingerprint ||
                item.call_source.conversation.conversation_id !== root.source.conversation_id ||
                item.call_source.conversation.revision > root.source.revision)
        )
            throw new Error('Indexed execution receipt source differs from retained call');
        if (item.result_turn_id) {
            const resultTurn = newTurns.get(item.result_turn_id);
            const resultBlock = resultTurn?.blocks.find(
                (block) =>
                    block.type === 'tool_result' && block.call_id === item.call_id && block.status === item.status,
            );
            if (resultBlock?.type !== 'tool_result') {
                throw new Error('Indexed execution receipt result turn is unavailable or mismatched');
            }
            await assertToolResultReceiptFingerprint(resultBlock, item);
        }
        callStates.set(item.call_id, { ...call, terminal_receipt_id: item.id });
    }
    for (const turn of batch.turns ?? []) {
        for (const block of turn.blocks) {
            if (block.type !== 'tool_result') continue;
            const terminal = (batch.execution_receipts ?? []).find((receipt) => receipt.call_id === block.call_id);
            if (!terminal || terminal.result_turn_id !== turn.id || terminal.status !== block.status) {
                throw new Error('Indexed tool result requires its exact terminal execution receipt');
            }
            if (turn.execution_id !== undefined && turn.execution_id !== terminal.id) {
                throw new Error('Indexed tool result execution identity differs');
            }
        }
    }
    const nextRevision = root.source.revision + 1;
    const context = await loadIndexedActiveContext(store, root);
    if (context.revision > root.source.revision) throw new Error('Indexed context revision exceeds root');
    const newSelections = new Map<string, Set<string> | undefined>();
    for (const entry of batch.context_entries ?? []) {
        if (entry.type !== 'source_turn') throw new Error('Indexed append replacement context is unsupported');
        const turn = newTurns.get(entry.turn_id);
        if (!turn) throw new Error('Indexed append context entry must name a new source turn');
        if (entry.block_ids) {
            let previous = -1;
            for (const id of entry.block_ids) {
                const position = turn.blocks.findIndex((block) => block.id === id);
                if (position <= previous) throw new Error('Indexed context block selection is missing or out of order');
                previous = position;
            }
        }
        const prior = newSelections.get(entry.turn_id);
        if (newSelections.has(entry.turn_id)) {
            if (prior === undefined || entry.block_ids === undefined || entry.block_ids.some((id) => prior.has(id))) {
                throw new Error('Indexed append context selections overlap');
            }
            for (const id of entry.block_ids) prior.add(id);
        } else {
            newSelections.set(entry.turn_id, entry.block_ids === undefined ? undefined : new Set(entry.block_ids));
        }
    }
    const activeToolIds = batch.active_tool_definition_ids ?? context.active_tool_definition_ids;
    if (new Set(activeToolIds).size !== activeToolIds.length)
        throw new Error('Indexed active tool selection repeats a definition');
    for (const id of activeToolIds)
        if (!(await definition(id))) {
            throw new Error('Indexed active tool selection has an unavailable definition');
        }
    const nextContext = ConversationContextSchema.parse({
        ...context,
        revision: nextRevision,
        entries: [...context.entries, ...(batch.context_entries ?? [])],
        active_tool_definition_ids: [...activeToolIds],
    });
    const activeBytes = canonicalJsonContentBytes(nextContext.entries).byteLength;
    if (activeBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES || nextContext.entries.length > 100_000) {
        throw new RangeError('Indexed active context exceeds its bounded profile');
    }
    for (const item of batch.tool_definitions ?? []) {
        const retained = await indexedRecordById(store, root, 'tool_definitions', item.id, ToolDefinitionSchema);
        if (retained && !sameIndexedRecord(retained, item)) throw new Error('Indexed tool definition conflicts');
    }
    const receipt = OperationReceiptSchema.parse({
        id: options.operation_id,
        conversation_id: root.source.conversation_id,
        payload_fingerprint: options.payload_fingerprint,
        base_revision: root.source.revision,
        result_revision: nextRevision,
        recorded_at: options.recorded_at,
        accepted_turn_ids: idsOf(batch.turns),
        accepted_generation_ids: idsOf(batch.generations),
        accepted_asset_ids: idsOf(batch.assets),
        accepted_tool_definition_ids: idsOf(batch.tool_definitions),
        accepted_execution_receipt_ids: idsOf(batch.execution_receipts),
        accepted_context_entry_ids: idsOf(batch.context_entries),
        accepted_context_entries: [...(batch.context_entries ?? [])],
        accepted_tool_selection:
            batch.active_tool_definition_ids === undefined
                ? { kind: 'unchanged' }
                : { kind: 'replace', definition_ids: [...batch.active_tool_definition_ids] },
    });
    const directories = { ...root.directories };
    const write = async (
        family: keyof typeof directories,
        id: string,
        value: unknown,
        mode: 'insert' | 'replace' = 'insert',
    ) => {
        directories[family] = await putPagedRecord(
            store,
            directories[family],
            id,
            await stageRecord(store, family, id, value),
            mode,
        );
    };
    for (const turn of batch.turns ?? []) {
        const { blocks, ...header } = turn;
        await write(
            'turns',
            turn.id,
            IndexedConversationTurnHeaderSchema.parse({
                turn: header,
                source: 'ordinary',
                block_ids: blocks.map((block) => block.id),
                block_ids_hash: (await hashContentBytes(canonicalJsonContentBytes(blocks.map((block) => block.id))))
                    .content_hash,
            }),
        );
        for (const block of blocks) await write('blocks', block.id, block);
    }
    for (const item of batch.generations ?? []) {
        await write('generations', item.id, item);
        directories.generation_acceptances = await putPagedRecord(store, directories.generation_acceptances, item.id, {
            storage: 'marker',
            kind: 'generation_acceptance',
            id: options.operation_id,
        });
    }
    for (const item of batch.assets ?? []) await write('assets', item.id, item);
    for (const item of batch.tool_definitions ?? []) {
        const retained = await indexedRecordById(store, root, 'tool_definitions', item.id, ToolDefinitionSchema);
        if (!retained) await write('tool_definitions', item.id, item);
    }
    for (const item of batch.execution_receipts ?? []) await write('execution_receipts', item.id, item);
    for (const item of batch.context_entries ?? []) await write('context_entries', item.id, item);
    await write('operation_receipts', receipt.id, receipt);
    for (const [id, state] of callStates) {
        const retainedCall = await getPagedRecord(store, root.directories.tool_call_states, id);
        await write('tool_call_states', id, state, retainedCall ? 'replace' : 'insert');
        const retainedOpen = await getPagedRecord(store, root.directories.open_tool_calls, id);
        if (state.result_block_id !== undefined) {
            if (retainedOpen) {
                directories.open_tool_calls = await putPagedRecord(
                    store,
                    directories.open_tool_calls,
                    id,
                    { storage: 'marker', kind: 'closed_tool_call', id },
                    'replace',
                );
            }
        } else if (!retainedOpen) {
            await write('open_tool_calls', id, {
                call_id: id,
                turn_id: state.turn_id,
                block_id: state.block_id,
                call_fingerprint: state.call_fingerprint,
            });
        }
    }
    for (const [id, kind] of globalIds) {
        if (kind === 'tool definition' && (await getPagedRecord(store, root.directories.identifiers, id))) continue;
        directories.identifiers = await putPagedRecord(store, directories.identifiers, id, {
            storage: 'marker',
            kind,
            id,
        });
    }
    for (const [index, turn] of (batch.turns ?? []).entries()) {
        directories.turn_order = await putPagedRecord(
            store,
            directories.turn_order,
            indexedOrderedKey(root.turn_count + index),
            { storage: 'marker', kind: 'turn_order', id: turn.id },
        );
    }
    for (const [index, entry] of (batch.context_entries ?? []).entries()) {
        directories.active_context_order = await putPagedRecord(
            store,
            directories.active_context_order,
            indexedOrderedKey(context.entries.length + index),
            { storage: 'marker', kind: 'context_order', id: entry.id },
        );
    }
    const deleteIndex = await appendDeleteIndex(store, root, directories, options.operation_id, batch);
    const { entries: _entries, ...contextFields } = nextContext;
    const contextHeader = await stageRecord(
        store,
        'context_header',
        root.source.conversation_id,
        IndexedConversationContextHeaderSchema.parse({
            ...contextFields,
            active_entry_count: nextContext.entries.length,
            active_entry_bytes: activeBytes,
            context_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(nextContext))).content_hash,
        }),
    );
    const acceptedGeneration =
        batch.generations?.length === 1 && batch.generations[0].record_source === 'executed'
            ? batch.generations[0]
            : undefined;
    const acceptedTurn =
        acceptedGeneration &&
        batch.turns?.length === 1 &&
        batch.turns[0].kind === 'agent' &&
        'generation_id' in batch.turns[0] &&
        batch.turns[0].generation_id === acceptedGeneration.id
            ? batch.turns[0]
            : undefined;
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        source: { ...root.source, revision: nextRevision },
        turn_count: root.turn_count + (batch.turns?.length ?? 0),
        ...deleteIndex,
        updated_at: options.recorded_at,
        context_header: { content_hash: contextHeader.content_hash, size_bytes: contextHeader.size_bytes },
        directories,
        ...(acceptedGeneration && acceptedTurn
            ? {
                  accepted_response: {
                      operation_id: options.operation_id,
                      generation_id: acceptedGeneration.id,
                      turn_id: acceptedTurn.id,
                      accepted_revision: nextRevision,
                  },
              }
            : {}),
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES) {
        throw new RangeError('Indexed conversation root exceeds its manifest bound');
    }
    return {
        root: nextRoot,
        locator: {
            content_hash: rootRecord.content_hash,
            size_bytes: rootRecord.size_bytes,
        },
        receipt,
        applied: true,
    };
}

/** Stage body-free logical deletion using only authenticated index point reads and the active context. */
export async function stageIndexedConversationDelete(
    rootInput: IndexedConversationRoot,
    input: IndexedConversationDeleteCommand,
    store: IndexedConversationRecordStore,
): Promise<{
    root: IndexedConversationRoot;
    locator?: PagedRecordRef;
    receipt: OperationReceipt;
    change: z.infer<typeof ConversationDeleteChangeSchema>;
    applied: boolean;
}> {
    if (!preflightJsonInput(input, { max_bytes: 512 * 1024 }).success) {
        throw new Error('Indexed deletion command is not bounded JSON');
    }
    const command = IndexedConversationDeleteCommandSchema.parse(input);
    const root = IndexedConversationRootSchema.parse(rootInput);
    if (root.delete_index_profile !== INDEXED_CONVERSATION_DELETE_PROFILE) {
        throw new Error('Indexed root has no complete logical-delete witness profile');
    }
    if (command.source.conversation_id !== root.source.conversation_id) {
        throw new Error('Indexed deletion conversation differs from its authenticated root');
    }
    const payloadFingerprint = await fingerprintJson({
        domain: 'llumiverse.conversation.indexed-delete',
        version: 1,
        command,
    });
    const sourceFingerprint = await fingerprintJson({
        domain: 'llumiverse.conversation.indexed-delete-source',
        version: 1,
        root: command.expected_source_root,
        turn_ids: command.turn_ids,
    });
    const prior = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        command.operation_id,
        OperationReceiptSchema,
    );
    if (prior !== undefined) {
        const detail = prior.conversation_delete;
        if (
            prior.operation_kind !== 'conversation_delete' ||
            prior.payload_fingerprint !== payloadFingerprint ||
            prior.base_revision !== command.source.revision ||
            prior.result_revision !== command.source.revision + 1 ||
            prior.result_revision > root.source.revision ||
            prior.recorded_at !== command.recorded_at ||
            !detail ||
            detail.source_fingerprint !== sourceFingerprint ||
            !sameIndexedRecord(detail.source, command.source) ||
            !sameIndexedRecord(
                detail.deleted_turns.map((item) => item.id),
                command.turn_ids,
            )
        ) {
            throw new Error('Indexed deletion retry conflicts with its accepted receipt');
        }
        for (const ref of detail.deleted_turns) {
            const witness = await indexedRecordById(
                store,
                root,
                'deleted_turns',
                ref.id,
                IndexedConversationDeletedTurnSchema,
            );
            const body = await getPagedRecord(store, root.directories.turns, ref.id);
            if (
                !witness ||
                !sameIndexedRecord(witness.predecessor_root, command.expected_source_root) ||
                witness.deleted_turn.operation_id !== prior.id ||
                witness.deleted_turn.source_revision !== prior.base_revision ||
                !sameIndexedRecord(ref, {
                    id: witness.deleted_turn.id,
                    fingerprint: witness.deleted_turn.fingerprint,
                    block_ids: witness.deleted_turn.block_ids,
                    accepted_operation_id: witness.deleted_turn.accepted_operation_id,
                }) ||
                body?.storage !== 'marker' ||
                body.kind !== 'deleted_turn'
            ) {
                throw new Error('Indexed deletion retry lacks its exact retained tombstone');
            }
        }
        return {
            root,
            receipt: prior,
            change: ConversationDeleteChangeSchema.parse({
                operation_id: prior.id,
                conversation_id: prior.conversation_id,
                base_revision: prior.base_revision,
                result_revision: prior.result_revision,
                operations: [detail],
                diagnostics: [],
            }),
            applied: false,
        };
    }
    if (!sameIndexedRecord(command.source, root.source)) {
        throw new Error('Indexed deletion source revision conflict');
    }
    if (await getPagedRecord(store, root.directories.identifiers, command.operation_id)) {
        throw new Error('Indexed deletion operation identity already exists');
    }
    if (root.source.revision === Number.MAX_SAFE_INTEGER) {
        throw new RangeError('Indexed deletion revision is exhausted');
    }
    if (Date.parse(command.recorded_at) < Date.parse(root.created_at)) {
        throw new Error('Indexed deletion timestamp predates its source');
    }
    const rootBytes = canonicalJsonContentBytes(root);
    if (
        rootBytes.byteLength !== command.expected_source_root.size_bytes ||
        (await hashContentBytes(rootBytes)).content_hash !== command.expected_source_root.content_hash
    ) {
        throw new Error('Indexed deletion source root differs from its authenticated locator');
    }
    const retainedRoot = await loadRecord(
        store,
        { storage: 'record', kind: 'root', id: root.source.conversation_id, ...command.expected_source_root },
        IndexedConversationRootSchema,
    );
    if (!sameIndexedRecord(retainedRoot, root)) throw new Error('Indexed deletion source root read-back differs');
    if (root.live_turn_count === undefined || root.active_tail_turn_id === undefined) {
        throw new Error('Indexed deletion root lacks its live-turn count or tail');
    }
    if (
        command.turn_ids.length > root.live_turn_count ||
        (root.live_turn_count === 0) !== (root.active_tail_turn_id === null)
    ) {
        throw new Error('Indexed deletion root has inconsistent live-turn witnesses');
    }
    if (new Set(command.turn_ids).size !== command.turn_ids.length) {
        throw new Error('Indexed deletion repeats a turn ID');
    }
    const selected = new Set(command.turn_ids);
    const context = await loadIndexedActiveContext(store, root);
    if (context.entries.some((entry) => selected.has(entry.turn_id))) {
        throw new Error('Indexed deletion requires prior active-context exclusion');
    }
    const refs: z.infer<typeof ConversationDeleteOperationSchema>['deleted_turns'] = [];
    const links = new Map<string, z.infer<typeof IndexedConversationTurnLinkSchema>>();
    let selectedBytes = 0;
    let selectedRecords = 0;
    const reserve = (bytes: number) => {
        selectedBytes += bytes;
        selectedRecords += 1;
        if (selectedBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES || selectedRecords > 100_000) {
            throw new RangeError('Indexed deletion selected source exceeds its bounded working set');
        }
    };
    let lastOrdinal = -1;
    for (const id of command.turn_ids) {
        if (id === root.accepted_response?.turn_id) {
            throw new Error('Indexed deletion cannot remove the retained accepted response');
        }
        if (await getPagedRecord(store, root.directories.deletion_blockers, id)) {
            throw new Error('Indexed deletion has a retained dependent record');
        }
        const link = await indexedRecordById(store, root, 'turn_links', id, IndexedConversationTurnLinkSchema);
        const header = await indexedRecordById(store, root, 'turns', id, IndexedConversationTurnHeaderSchema);
        const accepted = await getPagedRecord(store, root.directories.turn_acceptances, id);
        if (
            !link ||
            link.id !== id ||
            link.ordinal <= lastOrdinal ||
            !header ||
            header.source !== 'ordinary' ||
            accepted?.storage !== 'marker' ||
            accepted.kind !== 'turn_acceptance'
        ) {
            throw new Error('Indexed deletion has no ordered ordinary accepted turn');
        }
        const ordered = await getPagedRecord(store, root.directories.turn_order, indexedOrderedKey(link.ordinal));
        if (ordered?.storage !== 'marker' || ordered.kind !== 'turn_order' || ordered.id !== id) {
            throw new Error('Indexed deletion turn order differs from its live link');
        }
        const acceptance = await indexedRecordById(
            store,
            root,
            'operation_receipts',
            accepted.id,
            OperationReceiptSchema,
        );
        if (
            !acceptance ||
            acceptance.operation_kind !== undefined ||
            !acceptance.accepted_turn_ids?.includes(id) ||
            acceptance.result_revision > root.source.revision
        ) {
            throw new Error('Indexed deletion turn lacks its accepted append');
        }
        const turn = await loadIndexedProjectedTurn(store, root, id, undefined, reserve);
        if (turn.completeness !== 'full_turn') throw new Error('Indexed deletion source turn is incomplete');
        if (turn.selected_blocks.some((block) => ['tool_call', 'tool_result', 'native_replay'].includes(block.type))) {
            throw new Error('Indexed deletion cannot remove executed or replay-bearing tool facts');
        }
        refs.push({
            id,
            fingerprint: await fingerprintJson({ ...turn.header, blocks: turn.selected_blocks }),
            block_ids: turn.selected_blocks.map((block) => block.id),
            accepted_operation_id: accepted.id,
        });
        links.set(id, link);
        lastOrdinal = link.ordinal;
    }
    const operation = ConversationDeleteOperationSchema.parse({
        version: 1,
        source: command.source,
        source_fingerprint: sourceFingerprint,
        dependency_policy: 'reject',
        deleted_turns: refs,
    });
    const nextRevision = root.source.revision + 1;
    const receipt = OperationReceiptSchema.parse({
        id: command.operation_id,
        conversation_id: root.source.conversation_id,
        payload_fingerprint: payloadFingerprint,
        base_revision: root.source.revision,
        result_revision: nextRevision,
        recorded_at: command.recorded_at,
        operation_kind: 'conversation_delete',
        conversation_delete: operation,
        accepted_turn_ids: [],
        accepted_generation_ids: [],
        accepted_asset_ids: [],
        accepted_tool_definition_ids: [],
        accepted_execution_receipt_ids: [],
        accepted_context_entry_ids: [],
    });
    const directories = { ...root.directories };
    let activeTail = root.active_tail_turn_id;
    for (const ref of refs) {
        const link = links.get(ref.id);
        if (!link) throw new Error('Indexed deletion lost its selected live link');
        const current = await loadRecord(
            store,
            await getPagedRecord(store, directories.turn_links, ref.id),
            IndexedConversationTurnLinkSchema,
        );
        if (current.id !== ref.id || current.ordinal !== link.ordinal) {
            throw new Error('Indexed deletion live link changed while staging');
        }
        const previousId = current.previous_turn_id;
        const nextId = current.next_turn_id;
        if (previousId !== undefined) {
            const previous = await loadRecord(
                store,
                await getPagedRecord(store, directories.turn_links, previousId),
                IndexedConversationTurnLinkSchema,
            );
            if (previous.next_turn_id !== ref.id) throw new Error('Indexed deletion predecessor link differs');
            const { next_turn_id: _discardedNext, ...previousFields } = previous;
            directories.turn_links = await putPagedRecord(
                store,
                directories.turn_links,
                previousId,
                await stageRecord(store, 'turn_links', previousId, {
                    ...previousFields,
                    ...(nextId === undefined ? {} : { next_turn_id: nextId }),
                }),
                'replace',
            );
        }
        if (nextId !== undefined) {
            const next = await loadRecord(
                store,
                await getPagedRecord(store, directories.turn_links, nextId),
                IndexedConversationTurnLinkSchema,
            );
            if (next.previous_turn_id !== ref.id) throw new Error('Indexed deletion successor link differs');
            const { previous_turn_id: _discardedPrevious, ...nextFields } = next;
            directories.turn_links = await putPagedRecord(
                store,
                directories.turn_links,
                nextId,
                await stageRecord(store, 'turn_links', nextId, {
                    ...nextFields,
                    ...(previousId === undefined ? {} : { previous_turn_id: previousId }),
                }),
                'replace',
            );
        }
        directories.turn_links = await putPagedRecord(
            store,
            directories.turn_links,
            ref.id,
            { storage: 'marker', kind: 'deleted_turn_link', id: ref.id },
            'replace',
        );
        directories.turns = await putPagedRecord(
            store,
            directories.turns,
            ref.id,
            { storage: 'marker', kind: 'deleted_turn', id: ref.id },
            'replace',
        );
        directories.turn_order = await putPagedRecord(
            store,
            directories.turn_order,
            indexedOrderedKey(link.ordinal),
            { storage: 'marker', kind: 'deleted_turn_order', id: ref.id },
            'replace',
        );
        for (const blockId of ref.block_ids) {
            directories.blocks = await putPagedRecord(
                store,
                directories.blocks,
                blockId,
                { storage: 'marker', kind: 'deleted_block', id: blockId },
                'replace',
            );
        }
        const tombstone = IndexedConversationDeletedTurnSchema.parse({
            deleted_turn: {
                ...ref,
                operation_id: command.operation_id,
                source_revision: root.source.revision,
            },
            predecessor_root: command.expected_source_root,
        });
        directories.deleted_turns = await putPagedRecord(
            store,
            directories.deleted_turns,
            ref.id,
            await stageRecord(store, 'deleted_turns', ref.id, tombstone),
        );
        if (activeTail === ref.id) activeTail = previousId ?? null;
    }
    directories.operation_receipts = await putPagedRecord(
        store,
        directories.operation_receipts,
        receipt.id,
        await stageRecord(store, 'operation_receipts', receipt.id, receipt),
    );
    directories.identifiers = await putPagedRecord(store, directories.identifiers, receipt.id, {
        storage: 'marker',
        kind: 'operation receipt',
        id: receipt.id,
    });
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        source: { ...root.source, revision: nextRevision },
        updated_at: command.recorded_at,
        live_turn_count: root.live_turn_count - refs.length,
        active_tail_turn_id: activeTail,
        directories,
    });
    const nextRootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (nextRootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES) {
        throw new RangeError('Indexed deletion root exceeds its manifest bound');
    }
    return {
        root: nextRoot,
        locator: { content_hash: nextRootRecord.content_hash, size_bytes: nextRootRecord.size_bytes },
        receipt,
        change: ConversationDeleteChangeSchema.parse({
            operation_id: receipt.id,
            conversation_id: receipt.conversation_id,
            base_revision: receipt.base_revision,
            result_revision: receipt.result_revision,
            operations: [operation],
            diagnostics: [],
        }),
        applied: true,
    };
}
