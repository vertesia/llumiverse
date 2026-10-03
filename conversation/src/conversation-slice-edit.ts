import { canonicalJsonContentString } from './content-integrity.js';
import { assertContextMutationDependencyClosure } from './context-change.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { cacheAfterEdit, nextRevision, preflight, recordRefs } from './conversation-edit-utils.js';
import { assertSliceEditTopology, fingerprintSliceEditTopology } from './conversation-slice-topology.js';
import { assembleConversationRecordBatch, deriveConversationId, fingerprintJson } from './runtime.js';
import { ConversationEditChangeSchema } from './schemas/change.js';
import { ConversationTurnSchema } from './schemas/content.js';
import { ConversationEditPlanSchema, ConversationEditResultSchema } from './schemas/conversation-edit.js';
import { ConversationSliceEditOperationSchema } from './schemas/conversation-edit-operation.js';
import {
    ConversationSliceEditPlanInputSchema,
    ConversationSliceEditRequestSchema,
} from './schemas/conversation-slice-edit.js';
import { verifyDerivedBlockLineage } from './source-slice-lineage.js';
import { type SourceSliceSegment, segmentOwnedSourceBlock } from './source-slice-segmentation.js';
import { rejectSourceSlice, type SourceSliceWork, spendSourceSliceWork } from './source-slice-work.js';
import type {
    Asset,
    ContentBlock,
    ContextEntry,
    ConversationDocument,
    ConversationEditPlan,
    ConversationEditResult,
    ConversationSliceEditOperation,
    ConversationSliceEditPlanInput,
    ConversationTurn,
    DerivedBlockLineageGroup,
} from './types.js';
import { parseConversationDocument } from './validation.js';

const same = (a: unknown, b: unknown) => canonicalJsonContentString(a) === canonicalJsonContentString(b);
function change(
    id: string,
    document: ConversationDocument,
    base: number,
    result: number,
    operation: ConversationSliceEditOperation,
) {
    return ConversationEditChangeSchema.parse({
        operation_id: id,
        conversation_id: document.id,
        base_revision: base,
        result_revision: result,
        operations: [operation],
        diagnostics: [],
    });
}
interface EntryPartition {
    entry: ContextEntry;
    turn: ConversationTurn;
    fragments: SourceSliceSegment[];
}
async function partition(
    document: ConversationDocument,
    input: ConversationSliceEditPlanInput,
    retained: boolean,
): Promise<EntryPartition[]> {
    const selection = input.command.selection;
    if (
        !same(selection.conversation, input.conversation) ||
        selection.context_revision !== input.expected_context_revision
    )
        rejectSourceSlice('Slice selection does not bind the requested snapshot');
    if (!retained) {
        if (
            input.conversation.conversation_id !== document.id ||
            input.conversation.revision !== document.revision ||
            input.expected_context_revision !== document.context.revision
        )
            rejectSourceSlice('Slice edit snapshot conflict');
        const { source_fingerprint, ...selected } = selection;
        if (source_fingerprint !== (await fingerprintJson({ document, selection: selected })))
            rejectSourceSlice('Slice selection source fingerprint conflict');
    }
    const index = createContextTurnIndex(document),
        work: SourceSliceWork = { nodes: 0 },
        result: EntryPartition[] = [];
    let lastEntry = -1;
    const sourceEntries = retained ? selection.entries.map((item) => item.entry) : document.context.entries;
    const entryIndices = new Map(sourceEntries.map((entry, index) => [entry.id, index]));
    for (const item of selection.entries) {
        const entryIndex = entryIndices.get(item.entry.id);
        if (entryIndex === undefined) rejectSourceSlice('Slice source entry is unavailable');
        if (entryIndex <= lastEntry || !same(sourceEntries[entryIndex], item.entry))
            rejectSourceSlice('Slice entries are unavailable, changed or unordered');
        lastEntry = entryIndex;
        const { turn, blocks } = resolveContextEntry(index, item.entry);
        const fragments: SourceSliceSegment[] = [];
        let selectedIndex = 0;
        for (const block of blocks) {
            spendSourceSliceWork(work, item.blocks.length + 1);
            const chosen = item.blocks.filter((part) => part.block_id === block.id);
            if (!chosen.length) {
                fragments.push({
                    selected: false,
                    block,
                    source: {
                        source: input.conversation,
                        turn_id: turn.id,
                        block_id: block.id,
                        block_fingerprint: await fingerprintJson(block),
                        selection: { kind: 'whole' },
                    },
                    transform: 'block_copy',
                });
                continue;
            }
            const actualHash = await fingerprintJson(block);
            for (const part of chosen) {
                if (item.blocks[selectedIndex++] !== part || part.block_fingerprint !== actualHash)
                    rejectSourceSlice('Slice blocks are changed, overlapping or unordered');
            }
            if (item.entry.type === 'replacement_turn' && (chosen.length !== 1 || chosen[0].kind !== 'whole'))
                rejectSourceSlice('Compaction replacement remains an indivisible lineage unit');
            fragments.push(
                ...segmentOwnedSourceBlock(
                    block,
                    {
                        source: input.conversation,
                        turn_id: turn.id,
                        block_id: block.id,
                        block_fingerprint: chosen[0].block_fingerprint,
                    },
                    chosen,
                    work,
                ),
            );
        }
        if (selectedIndex !== item.blocks.length || !selectedIndex)
            rejectSourceSlice('Slice selects an unavailable or empty source');
        result.push({ entry: item.entry, turn, fragments });
    }
    return result;
}
function assertEligibility(
    document: ConversationDocument,
    partitions: readonly EntryPartition[],
    input: ConversationSliceEditPlanInput,
): void {
    const selectedIds = new Set(partitions.map((item) => item.entry.id));
    const removed: ContextEntry[] = [],
        retained = document.context.entries.filter((entry) => !selectedIds.has(entry.id));
    const turnIndex = createContextTurnIndex(document);
    for (const item of partitions) {
        const selected = [
            ...new Set(
                item.fragments.filter((fragment) => fragment.selected).map((fragment) => fragment.source.block_id),
            ),
        ];
        const all = resolveContextEntry(turnIndex, item.entry).blocks.map((block) => block.id);
        removed.push({ ...item.entry, block_ids: selected });
        const rest = all.filter((id) => !selected.includes(id));
        if (rest.length) retained.push({ ...item.entry, block_ids: rest });
        if (item.entry.type === 'replacement_turn' && selected.length !== all.length)
            rejectSourceSlice('Compaction replacement remains an indivisible lineage unit');
    }
    // Contextual pins may be changed explicitly; intrinsic authority/replay and causal closure cannot.
    assertContextMutationDependencyClosure(
        input.command.kind === 'protect'
            ? { ...document, context: { ...document.context, protected_entry_ids: [] } }
            : document,
        removed,
        retained,
    );
    if (input.command.kind === 'replace') {
        for (const item of partitions)
            for (const fragment of item.fragments) {
                if (!fragment.selected) continue;
                if (!['text', 'json', 'image', 'document', 'audio', 'video'].includes(fragment.block.type))
                    rejectSourceSlice('Slice replacement cannot transform executable or opaque content');
            }
    }
}
function lineageGroup(segment: SourceSliceSegment, id: string): DerivedBlockLineageGroup {
    if (segment.transform === 'json_projection') {
        if (segment.inverse === undefined)
            rejectSourceSlice('JSON projection requires its exact inverse source mapping');
        return {
            transform: 'json_projection',
            fidelity: 'value_preserving',
            source_slices: [segment.source],
            target_block_ids: [id],
            inverse: segment.inverse,
        };
    }
    if (segment.transform === 'authored_replacement')
        rejectSourceSlice('Source partition cannot manufacture authored lineage');
    return {
        transform: segment.transform,
        fidelity: 'value_preserving',
        source_slices: [segment.source],
        target_block_ids: [id],
    };
}
async function copiedTurn(
    item: EntryPartition,
    segment: SourceSliceSegment,
    input: ConversationSliceEditPlanInput,
    ordinal: number,
): Promise<ConversationTurn> {
    const id = await deriveConversationId('turn', input.operation_id, item.entry.id, segment.block.id, String(ordinal));
    const block: ContentBlock = { ...segment.block, id: await deriveConversationId('block', input.operation_id, id) };
    if (item.turn.authority !== 'ordinary')
        rejectSourceSlice('Partial mutation cannot strip intrinsic instruction authority');
    if (item.turn.kind !== 'user' && item.turn.kind !== 'program' && item.turn.kind !== 'agent')
        rejectSourceSlice('Partial mutation requires ordinary authored or model text/media');
    return ConversationTurnSchema.parse({
        id,
        kind: item.turn.kind,
        ...(item.turn.actor_id === undefined ? {} : { actor_id: item.turn.actor_id }),
        authority: 'ordinary',
        model_visibility: item.turn.model_visibility,
        status: 'completed',
        timestamps: { recorded_at: input.recorded_at },
        blocks: [block],
        provenance: {
            type: 'derived',
            derivation_id: input.operation_id,
            source_hash: input.command.selection.source_fingerprint,
            source_turn_ids: [segment.source.turn_id],
            source_block_ids: [segment.source.block_id],
            block_lineage: { version: 1, groups: [lineageGroup(segment, block.id)] },
        },
    });
}
function validateAssets(
    document: ConversationDocument,
    turns: readonly ConversationTurn[],
    assets: readonly Asset[],
    input: ConversationSliceEditPlanInput,
): void {
    const needed = new Set(
        turns.flatMap((turn) => turn.blocks.flatMap((block) => ('asset_id' in block ? [block.asset_id] : []))),
    );
    const ids = new Set<string>();
    for (const asset of assets) {
        const provenance = asset.provenance;
        if (
            ids.has(asset.id) ||
            !needed.has(asset.id) ||
            asset.created_at !== input.recorded_at ||
            provenance.type !== 'received' ||
            (provenance.source_turn_id !== undefined && !turns.some((turn) => turn.id === provenance.source_turn_id))
        )
            rejectSourceSlice('Slice replacement asset lacks exact received provenance');
        ids.add(asset.id);
    }
    for (const id of needed)
        if (!ids.has(id) && !Object.hasOwn(document.assets, id))
            rejectSourceSlice('Slice references an unavailable original asset');
}
/** Count original source intervals independently of the packed projection's output order. */
function selectionTopology(fragments: readonly SourceSliceSegment[], previousSelected: boolean) {
    let selected = previousSelected,
        ranges = 0;
    const byBlock = new Map<string, SourceSliceSegment[]>();
    for (const fragment of fragments) {
        const group = byBlock.get(fragment.source.block_id) ?? [];
        group.push(fragment);
        byBlock.set(fragment.source.block_id, group);
    }
    for (const group of byBlock.values()) {
        const intervals = group.flatMap((fragment) =>
            fragment.selected && fragment.json_source_interval !== undefined ? [fragment.json_source_interval] : [],
        );
        if (intervals.length) {
            let end = 0;
            for (const interval of intervals) {
                if (interval.start > end) selected = false;
                if (!selected) ranges++;
                selected = true;
                end = interval.end;
            }
            selected = end === intervals[0].total;
        } else {
            for (const fragment of group) {
                if (fragment.selected && !selected) ranges++;
                selected = fragment.selected;
            }
        }
    }
    return { selected, ranges };
}
async function prepare(
    document: ConversationDocument,
    input: ConversationSliceEditPlanInput,
    prior?: ConversationSliceEditOperation,
) {
    if (prior !== undefined) {
        assertSliceEditTopology(prior);
        if (prior.source_topology_fingerprint !== (await fingerprintSliceEditTopology(prior)))
            rejectSourceSlice('Retained slice topology fingerprint conflicts');
    }
    const partitions = await partition(document, input, prior !== undefined);
    if (prior === undefined) assertEligibility(document, partitions, input);
    const byId = new Map(partitions.map((item) => [item.entry.id, item]));
    const selectedSlices = partitions.flatMap((item) =>
        item.fragments.filter((fragment) => fragment.selected).map((fragment) => fragment.source),
    );
    if (!selectedSlices.length || selectedSlices.length > 4096)
        rejectSourceSlice('Slice edit requires bounded selected source records');
    const entries: ContextEntry[] = [],
        createdEntries: ContextEntry[] = [],
        createdTurns: ConversationTurn[] = [],
        remainderTurns: string[] = [];
    const originalPins =
        prior?.source_protected_entry_ids ?? document.context.protected_entry_ids.filter((id) => byId.has(id));
    const protectedIds = new Set(document.context.protected_entry_ids.filter((id) => !byId.has(id))),
        protectedEffect: string[] = [],
        unprotectedEffect: string[] = [];
    let replacement: ConversationTurn | undefined;
    if (input.command.kind === 'replace') {
        const authored = input.command.replacement_turn;
        if (
            authored.provenance.operation_id !== input.operation_id ||
            authored.timestamps.recorded_at !== input.recorded_at
        )
            rejectSourceSlice('Authored replacement does not bind this exact operation/time');
        replacement = {
            ...authored,
            provenance: {
                type: 'derived',
                derivation_id: input.operation_id,
                source_hash: input.command.selection.source_fingerprint,
                source_turn_ids: [...new Set(selectedSlices.map((slice) => slice.turn_id))],
                source_block_ids: [...new Set(selectedSlices.map((slice) => slice.block_id))],
                block_lineage: {
                    version: 1,
                    groups: [
                        {
                            transform: 'authored_replacement',
                            fidelity: input.command.fidelity,
                            target_block_ids: authored.blocks.map((block) => block.id),
                            source_slices: selectedSlices,
                        },
                    ],
                },
            },
        };
    }
    let inserted = false,
        ranges = 0,
        inRange = false;
    const sourceEntries = prior ? input.command.selection.entries.map((item) => item.entry) : document.context.entries;
    const positions =
        prior?.source_entry_positions ??
        input.command.selection.entries.map((item) =>
            document.context.entries.findIndex((entry) => entry.id === item.entry.id),
        );
    let retainedIndex = 0,
        previousPosition = -1;
    for (const entry of sourceEntries) {
        if (prior !== undefined) {
            const position = positions[retainedIndex++];
            if (previousPosition >= 0 && position !== previousPosition + 1) inRange = false;
            previousPosition = position;
        }
        const item = byId.get(entry.id);
        if (!item) {
            entries.push(entry);
            inRange = false;
            continue;
        }
        const topology = selectionTopology(item.fragments, inRange);
        ranges += topology.ranges;
        inRange = topology.selected;
        for (const [ordinal, segment] of item.fragments.entries()) {
            if (segment.selected && replacement) {
                if (!inserted) {
                    const target: ContextEntry = {
                        id: await deriveConversationId('context_entry', input.operation_id, 'replacement'),
                        type: 'source_turn',
                        turn_id: replacement.id,
                    };
                    entries.push(target);
                    createdEntries.push(target);
                    createdTurns.push(replacement);
                    inserted = true;
                }
                continue;
            }
            let target: ContextEntry;
            if (segment.source.selection.kind === 'whole') {
                target = {
                    ...entry,
                    id: await deriveConversationId('context_entry', input.operation_id, entry.id, String(ordinal)),
                    block_ids: [segment.source.block_id],
                };
            } else {
                const turn = await copiedTurn(item, segment, input, ordinal);
                createdTurns.push(turn);
                if (!segment.selected) remainderTurns.push(turn.id);
                target = {
                    id: await deriveConversationId('context_entry', input.operation_id, turn.id),
                    type: 'source_turn',
                    turn_id: turn.id,
                };
            }
            entries.push(target);
            createdEntries.push(target);
            const pin =
                input.command.kind === 'protect' && segment.selected
                    ? input.command.protected
                    : originalPins.includes(entry.id);
            if (pin) protectedIds.add(target.id);
            if (input.command.kind === 'protect' && segment.selected)
                (input.command.protected ? protectedEffect : unprotectedEffect).push(target.id);
        }
    }
    if (
        input.command.kind === 'replace' &&
        input.command.placement.causal_order !== (ranges > 1 ? 'explicit_disjoint' : 'contiguous')
    )
        rejectSourceSlice('Slice replacement requires exact explicit disjoint causal placement');
    const assets = input.command.kind === 'replace' ? (input.command.assets ?? []) : [];
    validateAssets(document, createdTurns, assets, input);
    const sourceEvidence = {
        source: input.conversation,
        source_context_revision: input.expected_context_revision,
        source_fingerprint: prior?.source_fingerprint ?? (await fingerprintJson(document)),
        selected_entries: await recordRefs(input.command.selection.entries.map((item) => item.entry)),
        source_entry_positions: positions,
    };
    const operation = ConversationSliceEditOperationSchema.parse({
        version: 2,
        kind: input.command.kind,
        ...sourceEvidence,
        source_topology_fingerprint: await fingerprintSliceEditTopology(sourceEvidence),
        source_slices: selectedSlices,
        source_protected_entry_ids: originalPins,
        removed_entry_ids: input.command.selection.entries.map((item) => item.entry.id),
        created_entries: await recordRefs(createdEntries),
        created_turns: await recordRefs(createdTurns),
        created_assets: await recordRefs(assets),
        remainder_turn_ids: remainderTurns,
        protected_entry_ids: protectedEffect,
        unprotected_entry_ids: unprotectedEffect,
        ...(input.command.kind === 'replace'
            ? { fidelity: input.command.fidelity, placement: input.command.placement }
            : {}),
    });
    assertSliceEditTopology(operation);
    return { operation, entries, createdEntries, createdTurns, assets, protectedIds: [...protectedIds] };
}
export async function planConversationSliceEdit(
    sourceInput: ConversationDocument,
    input: unknown,
): Promise<ConversationEditPlan> {
    preflight(input);
    const request = ConversationSliceEditPlanInputSchema.parse(input),
        document = await verifyDerivedBlockLineage(sourceInput);
    const prepared = await prepare(document, request);
    return ConversationEditPlanSchema.parse({ operation: prepared.operation, diagnostics: [] });
}
export async function applyConversationSliceEdit(
    sourceInput: ConversationDocument,
    input: unknown,
): Promise<ConversationEditResult> {
    preflight(input);
    const request = ConversationSliceEditRequestSchema.parse(input),
        document = await verifyDerivedBlockLineage(sourceInput);
    const fingerprint = await fingerprintJson({ domain: 'llumiverse.conversation.edit', version: 2, request });
    const prior = Object.hasOwn(document.operation_receipts, request.operation_id)
        ? document.operation_receipts[request.operation_id]
        : undefined;
    if (
        prior &&
        (prior.operation_kind !== 'conversation_edit' ||
            prior.conversation_edit?.version !== 2 ||
            prior.payload_fingerprint !== fingerprint ||
            prior.base_revision !== request.conversation.revision ||
            prior.recorded_at !== request.recorded_at ||
            prior.conversation_edit.source_fingerprint !== request.expected_source_fingerprint)
    )
        rejectSourceSlice('Slice edit operation conflicts with its accepted mutation family or payload');
    const prepared = await prepare(
        document,
        request,
        prior?.conversation_edit?.version === 2 ? prior.conversation_edit : undefined,
    );
    if (prior) {
        if (
            !same(prepared.operation, prior.conversation_edit) ||
            !same(
                prior.accepted_turn_ids,
                prepared.operation.created_turns.map((ref) => ref.id),
            ) ||
            !same(
                prior.accepted_context_entry_ids,
                prepared.operation.created_entries.map((ref) => ref.id),
            ) ||
            !same(
                prior.accepted_asset_ids,
                prepared.operation.created_assets.map((ref) => ref.id),
            ) ||
            (prior.accepted_generation_ids ?? []).length ||
            (prior.accepted_execution_receipt_ids ?? []).length ||
            (prior.accepted_tool_definition_ids ?? []).length
        )
            rejectSourceSlice('Slice retry differs from retained accepted lineage or effects');
        const turns = createContextTurnIndex(document);
        for (const ref of prepared.operation.created_turns) {
            const turn = turns.get(ref.id);
            if (!turn || (await fingerprintJson(turn)) !== ref.fingerprint)
                rejectSourceSlice('Slice retry changes retained target content');
        }
        for (const ref of prepared.operation.created_assets) {
            const asset = Object.hasOwn(document.assets, ref.id) ? document.assets[ref.id] : undefined;
            if (!asset || (await fingerprintJson(asset)) !== ref.fingerprint)
                rejectSourceSlice('Slice retry changes retained asset content');
        }
        for (const ref of prepared.operation.created_entries) {
            const entry = document.context.entries.find((item) => item.id === ref.id);
            if (entry && (await fingerprintJson(entry)) !== ref.fingerprint)
                rejectSourceSlice('Slice retry changes an active retained entry');
        }
        return ConversationEditResultSchema.parse({
            document,
            change: change(prior.id, document, prior.base_revision, prior.result_revision, prepared.operation),
            applied: false,
        });
    }
    if (prepared.operation.source_fingerprint !== request.expected_source_fingerprint)
        rejectSourceSlice('Slice edit source fingerprint conflict');
    const assembled = assembleConversationRecordBatch(document, {
        turns: prepared.createdTurns,
        assets: prepared.assets,
    });
    const revision = nextRevision(document.revision),
        cache = cacheAfterEdit(
            document,
            prepared.entries,
            document.context.entries.findIndex((entry) => prepared.operation.removed_entry_ids.includes(entry.id)),
        );
    const updated = parseConversationDocument({
        ...document,
        revision,
        updated_at: request.recorded_at,
        turns: assembled.turns,
        assets: { ...document.assets, ...assembled.assetRecords },
        context: {
            ...document.context,
            revision: nextRevision(document.context.revision),
            entries: prepared.entries,
            protected_entry_ids: prepared.protectedIds,
            ...(cache ? { cache_intent: cache } : {}),
        },
        operation_receipts: {
            ...document.operation_receipts,
            [request.operation_id]: {
                id: request.operation_id,
                conversation_id: document.id,
                payload_fingerprint: fingerprint,
                base_revision: document.revision,
                result_revision: revision,
                recorded_at: request.recorded_at,
                operation_kind: 'conversation_edit',
                conversation_edit: prepared.operation,
                accepted_turn_ids: prepared.operation.created_turns.map((ref) => ref.id),
                accepted_context_entry_ids: prepared.operation.created_entries.map((ref) => ref.id),
                accepted_asset_ids: prepared.operation.created_assets.map((ref) => ref.id),
                accepted_generation_ids: [],
                accepted_tool_definition_ids: [],
                accepted_execution_receipt_ids: [],
            },
        },
    });
    return ConversationEditResultSchema.parse({
        document: await verifyDerivedBlockLineage(updated),
        change: change(request.operation_id, document, document.revision, revision, prepared.operation),
        applied: true,
    });
}
