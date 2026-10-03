import { canonicalJsonContentString } from './content-integrity.js';
import { planContextChange } from './context-change.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { partitionContextSelection } from './context-selection-resolution.js';
import { cacheAfterEdit, nextRevision, preflight, recordRefs } from './conversation-edit-utils.js';
import { applyConversationSliceEdit, planConversationSliceEdit } from './conversation-slice-edit.js';
import { getPendingToolCallIds } from './inspection.js';
import { assembleConversationRecordBatch, deriveConversationId, fingerprintJson } from './runtime.js';
import { ConversationEditChangeSchema } from './schemas/change.js';
import { ContextChangePlanInputSchema } from './schemas/context-change.js';
import {
    ConversationEditPlanInputSchema,
    ConversationEditPlanSchema,
    ConversationEditRequestSchema,
    ConversationEditResultSchema,
} from './schemas/conversation-edit.js';
import { ConversationEditOperationSchema } from './schemas/conversation-edit-operation.js';
import { verifyDerivedBlockLineage } from './source-slice-lineage.js';
import type {
    Asset,
    ContentBlock,
    ContextEntry,
    ConversationDocument,
    ConversationEditAnchor,
    ConversationEditOperation,
    ConversationEditPlan,
    ConversationEditPlanInput,
    ConversationEditResult,
    ConversationRecordBatch,
    ConversationSelection,
    ConversationTurn,
} from './types.js';
import { parseConversationDocument } from './validation.js';

function editChange(
    operationId: string,
    conversationId: string,
    baseRevision: number,
    resultRevision: number,
    operation: ConversationEditOperation,
) {
    return ConversationEditChangeSchema.parse({
        operation_id: operationId,
        conversation_id: conversationId,
        base_revision: baseRevision,
        result_revision: resultRevision,
        operations: [operation],
        diagnostics: [],
    });
}
const same = (a: unknown, b: unknown) => canonicalJsonContentString(a) === canonicalJsonContentString(b);
async function selectionPartition(document: ConversationDocument, selection: ConversationSelection, retained = false) {
    if (
        !retained &&
        (selection.conversation.conversation_id !== document.id ||
            selection.conversation.revision !== document.revision ||
            selection.context_revision !== document.context.revision)
    )
        throw new Error('Edit selection snapshot conflict');
    const { source_fingerprint, ...selected } = selection;
    if (!retained && (await fingerprintJson({ document, selection: selected })) !== source_fingerprint)
        throw new Error('Edit selection source fingerprint conflict');
    const turns = createContextTurnIndex(document);
    const entryIds: string[] = [],
        selectedEntries: ContextEntry[] = [];
    const selectedBlockIds: Record<string, string[]> = {};
    const sourceEntries = retained ? selection.entries.map((item) => item.entry) : document.context.entries;
    const entryIndices = new Map(sourceEntries.map((entry, index) => [entry.id, index]));
    let lastEntryIndex = -1,
        partial = false;
    for (const item of selection.entries) {
        const index = entryIndices.get(item.entry.id);
        if (index === undefined) throw new Error('Edit selection entry unavailable');
        if (index <= lastEntryIndex || !same(sourceEntries[index], item.entry))
            throw new Error('Edit selection entry unavailable or unordered');
        lastEntryIndex = index;
        const actual = resolveContextEntry(turns, item.entry).blocks;
        const blockIndices = new Map(actual.map((block, index) => [block.id, index]));
        let lastBlockIndex = -1;
        const ids: string[] = [];
        for (const block of item.blocks) {
            if (block.kind !== 'whole') throw new Error('Subrange mutation requires explicit source-slice lineage');
            const blockIndex = blockIndices.get(block.block_id);
            if (blockIndex === undefined) throw new Error('Edit selection block unavailable');
            if (
                blockIndex <= lastBlockIndex ||
                actual[blockIndex].type !== block.block_type ||
                (await fingerprintJson(actual[blockIndex])) !== block.block_fingerprint
            )
                throw new Error('Edit selection block unavailable, changed or unordered');
            lastBlockIndex = blockIndex;
            ids.push(block.block_id);
        }
        if (!ids.length && actual.length)
            throw new Error('Edit selection cannot claim an empty cut of a nonempty entry');
        if (ids.length !== actual.length) {
            partial = true;
            Object.defineProperty(selectedBlockIds, item.entry.id, { enumerable: true, value: ids });
        }
        entryIds.push(item.entry.id);
        selectedEntries.push(item.entry);
    }
    const input = {
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        entry_ids: entryIds,
        ...(partial ? { selected_block_ids: selectedBlockIds, selected_entries: selectedEntries } : {}),
    };
    preflight(input);
    const parsed = ContextChangePlanInputSchema.parse(input);
    return { input: parsed, partition: partitionContextSelection(document, parsed, sourceEntries), turns };
}
function anchorIndex(document: ConversationDocument, anchor: ConversationEditAnchor): number {
    if (anchor.kind === 'head') return 0;
    if (anchor.kind === 'tail') return document.context.entries.length;
    const index = document.context.entries.findIndex((entry) => entry.id === anchor.entry_id);
    if (index < 0) throw new Error('Insert anchor is not active');
    return index + (anchor.kind === 'after_entry' ? 1 : 0);
}
/** Protect every complete active call/replay causal interval, not only adjacent call/result pairs. */
function assertInsertionBoundary(document: ConversationDocument, boundary: number): void {
    const turns = createContextTurnIndex(document);
    const locations = {
        turns: new Map<string, { first: number; last: number }>(),
        blocks: new Map<string, { first: number; last: number }>(),
        calls: new Map<string, { first: number; last: number }>(),
        requests: new Map<string, { first: number; last: number }>(),
    };
    const replay: { block: Extract<ContentBlock, { type: 'native_replay' }>; index: number }[] = [];
    const add = (map: Map<string, { first: number; last: number }>, id: string, index: number) => {
        const prior = map.get(id);
        map.set(id, { first: prior?.first ?? index, last: index });
    };
    for (const [index, entry] of document.context.entries.entries()) {
        const { turn, blocks } = resolveContextEntry(turns, entry);
        add(locations.turns, turn.id, index);
        if ('generation_id' in turn && turn.generation_id !== undefined) {
            const generation = document.generations[turn.generation_id];
            if (generation?.record_source === 'executed')
                add(locations.requests, generation.request_receipt.request_id, index);
        }
        const visit = (block: ContentBlock): void => {
            add(locations.blocks, block.id, index);
            if (block.type === 'tool_call' || block.type === 'tool_result') add(locations.calls, block.call_id, index);
            if (block.type === 'native_replay') replay.push({ block, index });
            if (block.type === 'tool_result') for (const nested of block.content) visit(nested);
        };
        for (const block of blocks) visit(block);
    }
    const cuts = (first: number, last: number) => first < boundary && last >= boundary;
    for (const range of locations.calls.values())
        if (cuts(range.first, range.last)) throw new Error('Insert anchor cuts a complete tool causal interval');
    for (const callId of getPendingToolCallIds(document)) {
        const range = locations.calls.get(callId);
        if (range && boundary > range.first)
            throw new Error('Insert anchor cuts an open application tool causal interval');
    }
    for (const { block, index } of replay) {
        const deps = block.dependencies;
        let first = index,
            last = index;
        for (const [ids, map] of [
            [deps.turn_ids, locations.turns],
            [deps.block_ids, locations.blocks],
            [deps.call_ids, locations.calls],
            [deps.request_ids, locations.requests],
        ] as const)
            for (const id of ids) {
                const range = map.get(id);
                if (range) {
                    first = Math.min(first, range.first);
                    last = Math.max(last, range.last);
                }
            }
        if (cuts(first, last)) throw new Error('Insert anchor cuts a protected replay causal interval');
    }
}
function selectedBlocks(selection: ConversationSelection, document: ConversationDocument): ContentBlock[] {
    const turns = createContextTurnIndex(document),
        result: ContentBlock[] = [];
    for (const item of selection.entries) {
        const ids = new Set(item.blocks.map((block) => block.block_id));
        result.push(...resolveContextEntry(turns, item.entry).blocks.filter((block) => ids.has(block.id)));
    }
    return result;
}
function validateAddedAssets(
    document: ConversationDocument,
    newTurns: readonly ConversationTurn[],
    assets: readonly Asset[],
    recordedAt: string,
): void {
    const required = new Set(
        newTurns.flatMap((turn) => turn.blocks.flatMap((block) => ('asset_id' in block ? [block.asset_id] : []))),
    );
    for (const asset of assets) {
        const provenance = asset.provenance;
        if (!required.has(asset.id) || provenance.type !== 'received' || asset.created_at !== recordedAt)
            throw new Error('Inserted asset lacks exact received-record provenance');
        if (provenance.source_turn_id !== undefined && !newTurns.some((turn) => turn.id === provenance.source_turn_id))
            throw new Error('Inserted asset names another source turn');
    }
    const added = new Set(assets.map((asset) => asset.id));
    for (const id of required)
        if (!added.has(id) && !Object.hasOwn(document.assets, id))
            throw new Error('Edited media references an unavailable asset');
}
async function transformSelectedEntries(
    document: ConversationDocument,
    input: ConversationEditPlanInput,
    partition: ReturnType<typeof partitionContextSelection>,
    entryIds: readonly string[],
) {
    const command = input.command;
    if (command.kind === 'insert') throw new Error('Selection transform cannot insert');
    const entries: ContextEntry[] = [],
        createdEntries: ContextEntry[] = [],
        removedEntryIds: string[] = [];
    const protectedIds = new Set(document.context.protected_entry_ids),
        protectedEffect: string[] = [],
        unprotectedEffect: string[] = [];
    let ordinal = 0,
        replacementInserted = false;
    const selectedIds = new Set(entryIds);
    const byEntry = new Map<string, typeof partition.segments>();
    for (const segment of partition.segments) {
        const fragments = byEntry.get(segment.entry.id) ?? [];
        fragments.push(segment);
        byEntry.set(segment.entry.id, fragments);
    }
    for (const id of entryIds) {
        const fragments = byEntry.get(id);
        if (!fragments) throw new Error('Selected partition entry unavailable');
        const split = fragments.length !== 1 || fragments[0].block_ids !== undefined;
        if (command.kind === 'replace' || split) {
            removedEntryIds.push(id);
            protectedIds.delete(id);
        }
    }
    const removed = new Set(removedEntryIds),
        oldPins = new Set(document.context.protected_entry_ids);
    for (const segment of partition.segments) {
        if (!selectedIds.has(segment.entry.id)) {
            entries.push(segment.entry);
            continue;
        }
        if (command.kind === 'replace' && segment.selected) {
            if (!replacementInserted) {
                const replacement: ContextEntry = {
                    id: await deriveConversationId('context_entry', input.operation_id, 'replace'),
                    type: 'source_turn',
                    turn_id: command.replacement_turn.id,
                };
                entries.push(replacement);
                createdEntries.push(replacement);
                replacementInserted = true;
            }
            continue;
        }
        const originalRemoved = removed.has(segment.entry.id);
        const entry: ContextEntry = originalRemoved
            ? {
                  ...segment.entry,
                  id: await deriveConversationId('context_entry', input.operation_id, 'fragment', String(ordinal++)),
                  ...(segment.block_ids ? { block_ids: segment.block_ids } : {}),
              }
            : segment.entry;
        entries.push(entry);
        if (originalRemoved) createdEntries.push(entry);
        const protect =
            command.kind === 'protect' && segment.selected ? command.protected : oldPins.has(segment.entry.id);
        if (protect) protectedIds.add(entry.id);
        else protectedIds.delete(entry.id);
        if (command.kind === 'protect' && segment.selected)
            (command.protected ? protectedEffect : unprotectedEffect).push(entry.id);
    }
    return { entries, createdEntries, removedEntryIds, protectedIds, protectedEffect, unprotectedEffect };
}

async function prepareEdit(document: ConversationDocument, input: ConversationEditPlanInput) {
    if (
        input.conversation.conversation_id !== document.id ||
        input.conversation.revision !== document.revision ||
        input.expected_context_revision !== document.context.revision
    )
        throw new Error('Conversation edit snapshot conflict');
    const sourceFingerprint = await fingerprintJson(document),
        command = input.command;
    const sourceEntryRefs =
        command.kind === 'insert' ? [] : await recordRefs(command.selection.entries.map((item) => item.entry));
    const createdEntries: ContextEntry[] = [],
        createdTurns: ConversationTurn[] = [],
        removedEntryIds: string[] = [];
    let entries = document.context.entries,
        contentChangedAt: number | undefined;
    const protectedIds = new Set(document.context.protected_entry_ids);
    const protectedEffect: string[] = [],
        unprotectedEffect: string[] = [];
    let assets: Asset[] = [];
    if (command.kind === 'insert') {
        const boundary = anchorIndex(document, command.anchor);
        assertInsertionBoundary(document, boundary);
        for (const turn of command.turns) {
            if (
                turn.provenance.operation_id !== input.operation_id ||
                turn.timestamps.recorded_at !== input.recorded_at
            )
                throw new Error('Inserted turn provenance must bind this exact operation/time');
            createdTurns.push(turn);
            createdEntries.push({
                id: await deriveConversationId('context_entry', input.operation_id, 'insert', turn.id),
                type: 'source_turn',
                turn_id: turn.id,
            });
        }
        entries = [...entries.slice(0, boundary), ...createdEntries, ...entries.slice(boundary)];
        contentChangedAt = boundary;
        assets = command.assets ?? [];
    } else {
        const { input: partitionInput, partition } = await selectionPartition(document, command.selection);
        if (command.kind === 'replace') {
            const plan = await planContextChange(document, partitionInput);
            if (command.placement.causal_order !== (partition.ranges > 1 ? 'explicit_disjoint' : 'contiguous'))
                throw new Error('Replace disjoint causal placement must be explicit');
            const turn = command.replacement_turn,
                provenance = turn.provenance;
            if (
                provenance.derivation_id !== input.operation_id ||
                provenance.source_hash !== command.selection.source_fingerprint ||
                !same(provenance.source_turn_ids, plan.source_turn_ids) ||
                !same(provenance.source_block_ids ?? [], plan.source_block_ids)
            )
                throw new Error('Replacement provenance does not bind the exact selected source');
            if (turn.timestamps.recorded_at !== input.recorded_at)
                throw new Error('Replacement turn observation time differs from its edit');
            const chosen = selectedBlocks(command.selection, document);
            if (
                chosen.some(
                    (block) =>
                        block.type !== 'text' &&
                        block.type !== 'json' &&
                        block.type !== 'image' &&
                        block.type !== 'document' &&
                        block.type !== 'audio' &&
                        block.type !== 'video',
                )
            )
                throw new Error('Replacement selects noneditable execution or opaque content');
            createdTurns.push(turn);
            assets = command.assets ?? [];
        }
        const transformed = await transformSelectedEntries(document, input, partition, partitionInput.entry_ids);
        entries = transformed.entries;
        createdEntries.push(...transformed.createdEntries);
        removedEntryIds.push(...transformed.removedEntryIds);
        protectedIds.clear();
        for (const id of transformed.protectedIds) protectedIds.add(id);
        protectedEffect.push(...transformed.protectedEffect);
        unprotectedEffect.push(...transformed.unprotectedEffect);
        const selectedIds = new Set(partitionInput.entry_ids);
        if (removedEntryIds.length)
            contentChangedAt = document.context.entries.findIndex((entry) => selectedIds.has(entry.id));
        if (command.kind === 'replace' && contentChangedAt === undefined)
            throw new Error('Replace has no selected active source');
    }
    validateAddedAssets(document, createdTurns, assets, input.recorded_at);
    const batch: ConversationRecordBatch = { turns: createdTurns, assets };
    const assembled = assembleConversationRecordBatch(document, batch);
    const operation = ConversationEditOperationSchema.parse({
        version: 1,
        kind: command.kind,
        source: input.conversation,
        source_context_revision: input.expected_context_revision,
        source_fingerprint: sourceFingerprint,
        selected_entries: sourceEntryRefs,
        removed_entry_ids: removedEntryIds,
        created_entries: await recordRefs(createdEntries),
        created_turns: await recordRefs(createdTurns),
        created_assets: await recordRefs(assets),
        protected_entry_ids: protectedEffect,
        unprotected_entry_ids: unprotectedEffect,
        ...(command.kind === 'insert' ? { anchor: command.anchor } : {}),
        ...(command.kind === 'replace' ? { fidelity: command.fidelity, placement: command.placement } : {}),
    });
    const cache = cacheAfterEdit(document, entries, contentChangedAt);
    parseConversationDocument({
        ...document,
        revision: nextRevision(document.revision),
        updated_at: input.recorded_at,
        turns: assembled.turns,
        assets: { ...document.assets, ...assembled.assetRecords },
        context: {
            ...document.context,
            revision: nextRevision(document.context.revision),
            entries,
            protected_entry_ids: [...protectedIds],
            ...(cache ? { cache_intent: cache } : {}),
        },
    });
    return { operation, assembled, entries, protectedIds: [...protectedIds], cache };
}

export async function planConversationEdit(
    sourceInput: ConversationDocument,
    input: unknown,
): Promise<ConversationEditPlan> {
    preflight(input);
    if (typeof input === 'object' && input !== null && 'version' in input && input.version === 2)
        return planConversationSliceEdit(sourceInput, input);
    const request = ConversationEditPlanInputSchema.parse(input),
        document = await verifyDerivedBlockLineage(sourceInput);
    const prepared = await prepareEdit(document, request);
    return ConversationEditPlanSchema.parse({ operation: prepared.operation, diagnostics: [] });
}

/** One pure immutable revision; publication/CAS and external side effects remain host-owned. */
export async function applyConversationEdit(
    sourceInput: ConversationDocument,
    requestInput: unknown,
): Promise<ConversationEditResult> {
    preflight(requestInput);
    if (
        typeof requestInput === 'object' &&
        requestInput !== null &&
        'version' in requestInput &&
        requestInput.version === 2
    )
        return applyConversationSliceEdit(sourceInput, requestInput);
    const request = ConversationEditRequestSchema.parse(requestInput),
        document = await verifyDerivedBlockLineage(sourceInput);
    const payloadFingerprint = await fingerprintJson({ domain: 'llumiverse.conversation.edit', version: 1, request });
    const prior = Object.hasOwn(document.operation_receipts, request.operation_id)
        ? document.operation_receipts[request.operation_id]
        : undefined;
    if (prior) {
        if (prior.operation_kind !== 'conversation_edit' || !prior.conversation_edit)
            throw new Error('Edit operation belongs to a different mutation kind');
        if (
            prior.payload_fingerprint !== payloadFingerprint ||
            prior.base_revision !== request.conversation.revision ||
            prior.recorded_at !== request.recorded_at ||
            prior.conversation_edit.source_fingerprint !== request.expected_source_fingerprint
        )
            throw new Error('Edit operation conflicts with its accepted payload');
        const detail = prior.conversation_edit,
            index = createContextTurnIndex(document);
        if (
            detail.kind !== request.command.kind ||
            !same(detail.source, request.conversation) ||
            detail.source_context_revision !== request.expected_context_revision ||
            !same(
                detail.selected_entries,
                request.command.kind === 'insert'
                    ? []
                    : await recordRefs(request.command.selection.entries.map((item) => item.entry)),
            ) ||
            !same(
                detail.kind === 'insert' ? detail.anchor : null,
                request.command.kind === 'insert' ? request.command.anchor : null,
            )
        )
            throw new Error('Edit retry has conflicting retained details');
        if (
            request.command.kind === 'replace' &&
            (detail.kind !== 'replace' ||
                detail.fidelity !== request.command.fidelity ||
                !same(detail.placement, request.command.placement))
        ) {
            throw new Error('Edit retry has conflicting retained replacement fidelity or placement');
        }
        let expectedEntries: ContextEntry[] = [],
            expectedRemoved: string[] = [],
            protectedEffect: string[] = [],
            unprotectedEffect: string[] = [];
        const expectedTurns =
            request.command.kind === 'insert'
                ? request.command.turns
                : request.command.kind === 'replace'
                  ? [request.command.replacement_turn]
                  : [];
        const expectedAssets = request.command.kind === 'protect' ? [] : (request.command.assets ?? []);
        if (request.command.kind === 'insert') {
            for (const turn of request.command.turns)
                expectedEntries.push({
                    id: await deriveConversationId('context_entry', request.operation_id, 'insert', turn.id),
                    type: 'source_turn',
                    turn_id: turn.id,
                });
        } else {
            const retained = await selectionPartition(document, request.command.selection, true);
            const transformed = await transformSelectedEntries(
                document,
                request,
                retained.partition,
                retained.input.entry_ids,
            );
            expectedEntries = transformed.createdEntries;
            expectedRemoved = transformed.removedEntryIds;
            protectedEffect = transformed.protectedEffect;
            unprotectedEffect = transformed.unprotectedEffect;
        }
        if (
            !same(detail.created_entries, await recordRefs(expectedEntries)) ||
            !same(detail.removed_entry_ids, expectedRemoved) ||
            !same(detail.created_turns, await recordRefs(expectedTurns)) ||
            !same(detail.created_assets, await recordRefs(expectedAssets)) ||
            !same(detail.protected_entry_ids, protectedEffect) ||
            !same(detail.unprotected_entry_ids, unprotectedEffect) ||
            !same(
                prior.accepted_turn_ids,
                expectedTurns.map((turn) => turn.id),
            ) ||
            !same(
                prior.accepted_asset_ids,
                expectedAssets.map((asset) => asset.id),
            ) ||
            !same(
                prior.accepted_context_entry_ids,
                expectedEntries.map((entry) => entry.id),
            ) ||
            !same(prior.accepted_generation_ids, []) ||
            !same(prior.accepted_tool_definition_ids, []) ||
            !same(prior.accepted_execution_receipt_ids, [])
        )
            throw new Error('Edit retry has conflicting accepted record/effect details');
        for (const ref of detail.created_entries) {
            const entry = document.context.entries.find((entry) => entry.id === ref.id);
            if (entry && (await fingerprintJson(entry)) !== ref.fingerprint)
                throw new Error('Edit retry changes its accepted entry');
        }
        for (const ref of detail.created_turns) {
            const turn = index.get(ref.id);
            if (!turn || (await fingerprintJson(turn)) !== ref.fingerprint)
                throw new Error('Edit retry changes its accepted turn');
        }
        for (const ref of detail.created_assets) {
            const asset = Object.hasOwn(document.assets, ref.id) ? document.assets[ref.id] : undefined;
            if (!asset || (await fingerprintJson(asset)) !== ref.fingerprint)
                throw new Error('Edit retry changes its accepted asset');
        }
        return ConversationEditResultSchema.parse({
            document,
            change: editChange(prior.id, prior.conversation_id, prior.base_revision, prior.result_revision, detail),
            applied: false,
        });
    }
    const { expected_source_fingerprint: expected, ...input } = request;
    const prepared = await prepareEdit(document, input);
    if (prepared.operation.source_fingerprint !== expected) throw new Error('Edit source fingerprint conflict');
    const revision = nextRevision(document.revision),
        contextRevision = nextRevision(document.context.revision);
    const operation = prepared.operation;
    const updated = parseConversationDocument({
        ...document,
        revision,
        updated_at: request.recorded_at,
        turns: prepared.assembled.turns,
        assets: { ...document.assets, ...prepared.assembled.assetRecords },
        context: {
            ...document.context,
            revision: contextRevision,
            entries: prepared.entries,
            protected_entry_ids: prepared.protectedIds,
            ...(prepared.cache ? { cache_intent: prepared.cache } : {}),
        },
        operation_receipts: {
            ...document.operation_receipts,
            [request.operation_id]: {
                id: request.operation_id,
                conversation_id: document.id,
                payload_fingerprint: payloadFingerprint,
                base_revision: document.revision,
                result_revision: revision,
                recorded_at: request.recorded_at,
                operation_kind: 'conversation_edit',
                conversation_edit: operation,
                accepted_turn_ids: operation.created_turns.map((ref) => ref.id),
                accepted_generation_ids: [],
                accepted_asset_ids: operation.created_assets.map((ref) => ref.id),
                accepted_tool_definition_ids: [],
                accepted_execution_receipt_ids: [],
                accepted_context_entry_ids: operation.created_entries.map((ref) => ref.id),
            },
        },
    });
    return ConversationEditResultSchema.parse({
        document: await verifyDerivedBlockLineage(updated),
        change: editChange(request.operation_id, document.id, document.revision, revision, operation),
        applied: true,
    });
}
