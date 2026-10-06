import { z } from 'zod';
import {
    type ContextChangeWorkingSet,
    planContextChangeWorkingSet,
    selectedBlocks,
} from './context-change-working-set.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { ContextChangeRequestSchema } from './schemas/context-change.js';
import { GenerationSchema } from './schemas/execution.js';
import { ConversationRefSchema, IdentifierSchema, TimestampSchema } from './schemas/primitives.js';
import type { CompactionRecord, ContentBlock, ContextChangeRequest, ContextEntry, OperationReceipt } from './types.js';

/** Version two lists the authenticated selected dependency closure, not a lifetime asset inventory.
 * Global immutable asset records remain untouched. Version-one checkpoint records remain unchanged.
 */
export const CHECKPOINT_SELECTED_STRATEGY_VERSION = '2';
export const IndexedCheckpointSummaryCommandSchema = z.strictObject({
    source: ConversationRefSchema,
    operation_id: IdentifierSchema,
    summary: z
        .string()
        .min(1)
        .max(256 * 1024),
    recorded_at: TimestampSchema,
    summary_generation: GenerationSchema.optional(),
});
export type IndexedCheckpointSummaryCommand = z.infer<typeof IndexedCheckpointSummaryCommandSchema>;

function containsBlockIdentity(block: ContentBlock, ids: readonly string[]): boolean {
    return (
        ids.includes(block.id) ||
        (block.type === 'tool_result' && block.content.some((nested) => containsBlockIdentity(nested, ids)))
    );
}

function replayDependencies(blocks: readonly ContentBlock[]) {
    const dependencies: Array<{ turn_ids: string[]; block_ids: string[]; call_ids: string[] }> = [];
    const visit = (block: ContentBlock): void => {
        if (block.type === 'native_replay') dependencies.push(block.dependencies);
        if (block.type === 'tool_result') for (const nested of block.content) visit(nested);
    };
    for (const block of blocks) visit(block);
    return dependencies;
}

export async function buildSelectedCheckpointRequest(
    frame: ContextChangeWorkingSet,
    input: IndexedCheckpointSummaryCommand,
): Promise<ContextChangeRequest> {
    const source = { source: frame.source, context: frame.context, assets: frame.assets, turns: [...frame.turns] };
    if (!preflightJsonInput(source, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new TypeError('Checkpoint active selection exceeds its bounded JSON profile');
    const owned = structuredClone(source);
    frame = { ...owned, turns: new Map(owned.turns) };
    input = IndexedCheckpointSummaryCommandSchema.parse(structuredClone(input));
    if (
        input.source.conversation_id !== frame.source.conversation_id ||
        input.source.revision !== frame.source.revision
    )
        throw new Error('Checkpoint selection differs from its original exact source');
    const resolved = frame.context.entries.map((entry) => {
        const value = frame.turns.get(entry.turn_id);
        if (!value) throw new Error('Checkpoint selection lost an active turn');
        return { entry, turn: value.header, blocks: selectedBlocks(value, entry) };
    });
    // Only the existing same-entry partial edit may retire explicitly discardable replay.
    // Protected or nested replay retains its original entry and dependency closure.
    const semanticBlocks = (blocks: readonly ContentBlock[]) =>
        blocks.filter((block) => block.type !== 'native_replay');
    const removableReplay = (block: ContentBlock, turnId: string, blocks: readonly ContentBlock[]): boolean =>
        block.type === 'native_replay' &&
        block.dependency_policy === 'discard_on_dependency_change' &&
        semanticBlocks(blocks).length > 0 &&
        (block.dependencies.turn_ids.includes(turnId) ||
            semanticBlocks(blocks).some(
                (semantic) =>
                    block.dependencies.block_ids.includes(semantic.id) ||
                    ((semantic.type === 'tool_call' || semantic.type === 'tool_result') &&
                        block.dependencies.call_ids.includes(semantic.call_id)),
            ));
    const protectedIds = new Set(frame.context.protected_entry_ids);
    const selectedResults = new Map<string, string>();
    for (const item of resolved) {
        if (item.turn.kind !== 'tool') continue;
        for (const block of item.blocks)
            if (block.type === 'tool_result') selectedResults.set(block.call_id, item.entry.id);
    }
    const preservedIds = new Set(
        resolved
            .filter(
                ({ entry, turn, blocks }) =>
                    protectedIds.has(entry.id) ||
                    turn.authority === 'system' ||
                    turn.authority === 'developer' ||
                    blocks.some((block) =>
                        block.type === 'native_replay'
                            ? !removableReplay(block, turn.id, blocks)
                            : replayDependencies([block]).length > 0,
                    ) ||
                    (turn.kind === 'agent' &&
                        blocks.some(
                            (block) =>
                                block.type === 'tool_call' &&
                                block.executor === 'application' &&
                                !selectedResults.has(block.call_id),
                        )),
            )
            .map(({ entry }) => entry.id),
    );
    let preservedAdded = true;
    while (preservedAdded) {
        preservedAdded = false;
        for (const item of resolved) {
            if (!preservedIds.has(item.entry.id)) continue;
            if (item.turn.kind === 'tool') {
                const resultCallIds = new Set(
                    item.blocks.filter((block) => block.type === 'tool_result').map((block) => block.call_id),
                );
                for (const candidate of resolved) {
                    if (
                        candidate.turn.kind === 'agent' &&
                        candidate.blocks.some(
                            (block) => block.type === 'tool_call' && resultCallIds.has(block.call_id),
                        ) &&
                        !preservedIds.has(candidate.entry.id)
                    ) {
                        preservedIds.add(candidate.entry.id);
                        preservedAdded = true;
                    }
                }
            }
            for (const dependencies of replayDependencies(item.blocks)) {
                for (const candidate of resolved) {
                    const matchesTurn = dependencies.turn_ids.includes(candidate.turn.id);
                    const matchesBlock = candidate.blocks.some((block) =>
                        containsBlockIdentity(block, dependencies.block_ids),
                    );
                    const matchesCall = candidate.blocks.some(
                        (block) =>
                            (block.type === 'tool_call' || block.type === 'tool_result') &&
                            dependencies.call_ids.includes(block.call_id),
                    );
                    if ((matchesTurn || matchesBlock || matchesCall) && !preservedIds.has(candidate.entry.id)) {
                        preservedIds.add(candidate.entry.id);
                        preservedAdded = true;
                    }
                }
            }
            if (item.turn.kind === 'agent') {
                for (const block of item.blocks) {
                    if (block.type !== 'tool_call') continue;
                    const resultEntryId = selectedResults.get(block.call_id);
                    if (resultEntryId && !preservedIds.has(resultEntryId)) {
                        preservedIds.add(resultEntryId);
                        preservedAdded = true;
                    }
                }
            }
        }
    }
    const eligible = resolved.filter(({ entry }) => !preservedIds.has(entry.id));
    if (eligible.length === 0) throw new Error('Cannot checkpoint canonical context without eligible source turns');

    const selection = {
        expected_revision: frame.source.revision,
        expected_context_revision: frame.context.revision,
        entry_ids: eligible.map(({ entry }) => entry.id),
    };
    const replayEntries = eligible.filter(({ blocks }) => blocks.some((block) => block.type === 'native_replay'));
    const plan = await planContextChangeWorkingSet(
        frame,
        replayEntries.length > 0
            ? {
                  ...selection,
                  selected_entries: eligible.map(({ entry }) => entry),
                  selected_block_ids: Object.fromEntries(
                      replayEntries.map(({ entry, blocks }) => [
                          entry.id,
                          semanticBlocks(blocks).map((block) => block.id),
                      ]),
                  ),
              }
            : selection,
    );
    const configurationFingerprint = await fingerprintJson({
        prompt: 'workflow-checkpoint-summary',
        version: 2,
        retention: 'selected_dependency_closure',
    });
    const inputFingerprint = await fingerprintJson(input);
    const compactionId = await deriveConversationId('compaction', frame.source.conversation_id, input.operation_id);
    const summaryTurnId = await deriveConversationId('turn', compactionId, 'summary');
    const summaryBlockId = await deriveConversationId('block', summaryTurnId, 'text');
    return ContextChangeRequestSchema.parse({
        operation_id: input.operation_id,
        expected_revision: frame.source.revision,
        expected_context_revision: frame.context.revision,
        expected_source_fingerprint: plan.source_fingerprint,
        recorded_at: input.recorded_at,
        entry_ids: plan.entry_ids,
        ...(plan.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: plan.selected_block_ids,
                  selected_entries: plan.selected_entries,
              }),
        proposal: {
            kind: 'replace_with_compaction',
            compaction_id: compactionId,
            strategy: {
                id: 'workflow_checkpoint_summary',
                version: CHECKPOINT_SELECTED_STRATEGY_VERSION,
                configuration_fingerprint: configurationFingerprint,
            },
            replacement_turns: [
                {
                    id: summaryTurnId,
                    kind: 'agent',
                    authority: 'ordinary',
                    status: 'completed',
                    timestamps: { recorded_at: input.recorded_at, completed_at: input.recorded_at },
                    model_visibility: 'include',
                    provenance: {
                        type: 'derived',
                        derivation_id: compactionId,
                        source_turn_ids: plan.source_turn_ids,
                        ...(plan.source_block_ids.length ? { source_block_ids: plan.source_block_ids } : {}),
                        source_hash: plan.source_fingerprint,
                    },
                    blocks: [{ id: summaryBlockId, type: 'text', text: input.summary, format: 'markdown' }],
                },
            ],
            fidelity: 'semantic',
            retained_asset_ids: Object.keys(frame.assets).sort(),
            generation_ids: [],
            ...(input.summary_generation ? { derivation_generation: input.summary_generation } : {}),
            accepted_input_fingerprint: inputFingerprint,
            placement: {
                mode: 'first_selected',
                causal_order: plan.disjoint_ranges > 1 ? 'explicit_disjoint_summary' : 'contiguous',
            },
        },
    });
}

/** Exact immutable retry: reconstruct the accepted request rather than inferring current eligibility.
 * The host separately re-proves the original scheduled summary and current owner/source authority.
 */
export async function assertSelectedCheckpointRetry(
    input: IndexedCheckpointSummaryCommand,
    compaction: CompactionRecord,
    receipt: OperationReceipt,
    selectedEntries?: readonly ContextEntry[],
): Promise<void> {
    input = IndexedCheckpointSummaryCommandSchema.parse(structuredClone(input));
    const contextRevision = compaction.metadata?.source_context_revision;
    const detail = receipt.context_change;
    const compactionId = await deriveConversationId('compaction', input.source.conversation_id, input.operation_id);
    const turnId = await deriveConversationId('turn', compactionId, 'summary');
    const blockId = await deriveConversationId('block', turnId, 'text');
    const entryId = await deriveConversationId('context_entry', compactionId, turnId);
    const configurationFingerprint = await fingerprintJson({
        prompt: 'workflow-checkpoint-summary',
        version: 2,
        retention: 'selected_dependency_closure',
    });
    const turn = compaction.replacement_turns[0];
    const block = turn?.blocks[0];
    if (
        receipt.id !== input.operation_id ||
        receipt.conversation_id !== input.source.conversation_id ||
        receipt.operation_kind !== 'context_change' ||
        receipt.base_revision !== input.source.revision ||
        receipt.result_revision !== input.source.revision + 1 ||
        receipt.recorded_at !== input.recorded_at ||
        detail?.kind !== 'replace_with_compaction' ||
        !detail.placement ||
        compaction.id !== compactionId ||
        compaction.operation_id !== input.operation_id ||
        compaction.strategy.id !== 'workflow_checkpoint_summary' ||
        compaction.strategy.version !== CHECKPOINT_SELECTED_STRATEGY_VERSION ||
        compaction.strategy.configuration_fingerprint !== configurationFingerprint ||
        compaction.fidelity !== 'semantic' ||
        compaction.generation_ids.length !== 0 ||
        compaction.created_at !== input.recorded_at ||
        compaction.replacement_turns.length !== 1 ||
        turn?.id !== turnId ||
        turn.kind !== 'agent' ||
        turn.authority !== 'ordinary' ||
        turn.status !== 'completed' ||
        turn.model_visibility !== 'include' ||
        turn.provenance.type !== 'derived' ||
        turn.provenance.derivation_id !== compactionId ||
        turn.provenance.source_hash !== compaction.source.source_fingerprint ||
        (await fingerprintJson(turn.provenance.source_turn_ids)) !==
            (await fingerprintJson(compaction.source.turn_ids)) ||
        (await fingerprintJson(turn.provenance.source_block_ids ?? [])) !==
            (await fingerprintJson(compaction.source.block_ids ?? [])) ||
        turn.blocks.length !== 1 ||
        block?.id !== blockId ||
        block.type !== 'text' ||
        block.text !== input.summary ||
        block.format !== 'markdown' ||
        compaction.metadata?.applied_revision !== receipt.result_revision ||
        compaction.metadata?.payload_fingerprint !== receipt.payload_fingerprint ||
        compaction.metadata?.accepted_input_fingerprint !== (await fingerprintJson(input)) ||
        typeof contextRevision !== 'number' ||
        !Number.isSafeInteger(contextRevision) ||
        contextRevision < 0 ||
        detail.source_fingerprint !== compaction.source.source_fingerprint ||
        (await fingerprintJson(detail.inserted_entry_ids)) !== (await fingerprintJson([entryId])) ||
        (await fingerprintJson(receipt.accepted_context_entry_ids)) !== (await fingerprintJson([entryId])) ||
        receipt.accepted_turn_ids === undefined ||
        receipt.accepted_turn_ids.length !== 0 ||
        receipt.accepted_generation_ids === undefined ||
        receipt.accepted_generation_ids.length !== 0 ||
        (receipt.accepted_asset_ids?.length ?? 0) !== 0 ||
        (receipt.accepted_tool_definition_ids?.length ?? 0) !== 0 ||
        (receipt.accepted_execution_receipt_ids?.length ?? 0) !== 0
    )
        throw new Error('Checkpoint retry differs from its exact accepted selected mutation');
    const request = ContextChangeRequestSchema.parse({
        operation_id: input.operation_id,
        expected_revision: input.source.revision,
        expected_context_revision: contextRevision,
        expected_source_fingerprint: compaction.source.source_fingerprint,
        recorded_at: input.recorded_at,
        entry_ids: detail.removed_entry_ids,
        ...(detail.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: detail.selected_block_ids,
                  selected_entries: selectedEntries,
              }),
        proposal: {
            kind: 'replace_with_compaction',
            compaction_id: compaction.id,
            strategy: compaction.strategy,
            replacement_turns: compaction.replacement_turns,
            fidelity: 'semantic',
            retained_asset_ids: compaction.retained_asset_ids,
            generation_ids: [],
            ...(input.summary_generation ? { derivation_generation: input.summary_generation } : {}),
            accepted_input_fingerprint: await fingerprintJson(input),
            placement: detail.placement,
        },
    });
    if (
        (await fingerprintJson(request)) !== receipt.payload_fingerprint ||
        (await fingerprintJson(compaction.derivation_generation ?? null)) !==
            (await fingerprintJson(input.summary_generation ?? null))
    )
        throw new Error('Checkpoint retry changed its original summary, generation or selection');
}
