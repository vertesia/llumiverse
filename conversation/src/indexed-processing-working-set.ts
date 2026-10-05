import { canonicalJsonContentString, hashUtf8Content } from './content-integrity.js';
import { applyContextMutationWorkingSet, type ContextMutationResult } from './context-change-transition.js';
import { type ContextChangeWorkingSet, planContextChangeWorkingSet } from './context-change-working-set.js';
import { fingerprintJson } from './identity.js';
import { INDEXED_EXCHANGE_PROCESSOR_ID, INDEXED_EXCHANGE_PROCESSOR_VERSION } from './indexed-exchange-constants.js';
import { preflightJsonInput } from './json-preflight.js';
import type { ProcessorResult } from './processing.js';
import { ContextChangePlanInputSchema, ContextChangeRequestSchema } from './schemas/context-change.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import {
    type IndexedProcessingSelectedContext,
    IndexedProcessingSelectedContextSchema,
} from './schemas/indexed-head.js';
import {
    IndexedProcessingClaimWorkspaceSchema,
    type IndexedProcessingPredecessorEvidence,
    IndexedProcessingPredecessorEvidenceSchema,
} from './schemas/indexed-processing.js';
import { TimestampSchema } from './schemas/primitives.js';
import {
    ProcessingJobSchema,
    ProcessingOutputReceiptSchema,
    ProcessingResolvedInputSchema,
} from './schemas/processing.js';
import { buildTextWorkingSetProposal, selectedTextWorkingSet } from './text-externalization-working-set.js';
import type {
    ContextEntry,
    OperationReceipt,
    ProcessingJob,
    ProcessingOutputReceipt,
    ProcessingResolvedInput,
} from './types.js';

const MAX_WORKING_SET_BYTES = 32 * 1024 * 1024;
const PROFILE = 'llumiverse.conversation/indexed-active-processing/2026-10-04.v1';

function ownSelected(input: unknown): IndexedProcessingSelectedContext {
    if (!preflightJsonInput(input, { max_bytes: MAX_WORKING_SET_BYTES }).success)
        throw new TypeError('Indexed processing selection is not bounded JSON');
    return IndexedProcessingSelectedContextSchema.parse(structuredClone(input));
}

/** Every active entry's complete active blocks are present. Cold unselected originals are not implied
 * by this frame; their headers/count/order hashes remain in the explicit projection witnesses.
 */
export function activeIndexedContextWorkingSet(selected: IndexedProcessingSelectedContext): ContextChangeWorkingSet {
    const projections = [...selected.turns, ...(selected.replacement_turns ?? []).map((item) => item.projection)];
    const index = new Map(projections.map((turn) => [turn.header.id, turn]));
    if (index.size !== projections.length) throw new Error('Indexed processing repeats an active turn projection');
    const activeBlocks = new Map<string, Set<string>>();
    for (const entry of selected.context.entries) {
        const projection = index.get(entry.turn_id);
        if (!projection) throw new Error('Indexed processing has a missing active entry projection');
        const selectedIds = projection.selected_blocks.map((block) => block.id);
        if (entry.block_ids === undefined && projection.completeness !== 'full_turn')
            throw new Error('Indexed processing whole entry has an incomplete projection');
        const ids = entry.block_ids ?? selectedIds;
        if (ids.some((id) => !selectedIds.includes(id)))
            throw new Error('Indexed processing omits an active block dependency');
        const retained = activeBlocks.get(entry.turn_id) ?? new Set<string>();
        for (const id of ids) retained.add(id);
        activeBlocks.set(entry.turn_id, retained);
        if (
            entry.type === 'replacement_turn' &&
            !(selected.replacement_turns ?? []).some(
                (item) => item.compaction_id === entry.compaction_id && item.projection.header.id === entry.turn_id,
            )
        )
            throw new Error('Indexed processing loses its exact compaction replacement witness');
    }
    for (const projection of projections) {
        const ids = projection.selected_blocks.map((block) => block.id);
        const active = activeBlocks.get(projection.header.id);
        if (
            !active ||
            ids.length !== active.size ||
            ids.some((id) => !active.has(id)) ||
            new Set(ids).size !== ids.length ||
            projection.selected_block_positions.some(
                (position, i, all) => position >= projection.source_block_count || (i > 0 && position <= all[i - 1]),
            )
        )
            throw new Error('Indexed processing projection differs from the complete active block set');
    }
    return {
        source: selected.source,
        context: selected.context,
        assets: selected.assets,
        turns: new Map(
            projections.map((projection) => [
                projection.header.id,
                {
                    header: projection.header,
                    active_blocks: projection.selected_blocks,
                },
            ]),
        ),
    };
}

/** A selected-only context identity is explicitly versioned. It never attests unselected turn bodies. */
function contextFingerprint(selected: IndexedProcessingSelectedContext): Promise<string> {
    return fingerprintJson({
        profile: PROFILE,
        context: selected.context,
        turns: selected.turns,
        replacement_turns: selected.replacement_turns ?? [],
    });
}

/** Rebind an immutable accepted exchange selection through accepted partial-compaction
 * remainders. The job keeps its original entry IDs; only a proved active descendant may be
 * used for the next phase. No current entry is selected by turn or block identity alone.
 */
export async function activeIndexedExchangeSelection(
    selected: IndexedProcessingSelectedContext,
    job: ProcessingJob,
): Promise<{ entry_ids: string[]; selected_entries: ContextEntry[]; selected_block_ids: Record<string, string[]> }> {
    if (
        job.processor_id !== INDEXED_EXCHANGE_PROCESSOR_ID ||
        job.processor_version !== INDEXED_EXCHANGE_PROCESSOR_VERSION ||
        job.selection.kind !== 'entries' ||
        !job.selection.selected_entries ||
        !job.selection.selected_block_ids ||
        job.selection.entry_ids.length !== job.selection.selected_entries.length
    )
        throw new Error('Indexed exchange has no exact immutable entry selection');
    const active = new Map(selected.context.entries.map((entry) => [entry.id, entry]));
    const witnessIds = selected.sibling_compaction_ids ?? [];
    if (witnessIds.length > 16 || new Set(witnessIds).size !== witnessIds.length)
        throw new RangeError('Indexed exchange remainder proof exceeds its bounded sibling cohort');
    const witnesses = witnessIds.map((id) => {
        const witness = selected.compaction_witnesses?.[id];
        if (
            !witness ||
            witness.compaction.id !== id ||
            witness.compaction.operation_id !== witness.acceptance.id ||
            witness.acceptance.conversation_id !== selected.source.conversation_id ||
            witness.acceptance.operation_kind !== 'context_change'
        )
            throw new Error('Indexed exchange lacks an accepted sibling compaction witness');
        return witness;
    });
    witnesses.sort((a, b) => a.acceptance.result_revision - b.acceptance.result_revision);
    const entries: ContextEntry[] = [];
    const blocks: Record<string, string[]> = {};
    for (const [index, original] of job.selection.selected_entries.entries()) {
        if (original.id !== job.selection.entry_ids[index] || original.type !== 'source_turn')
            throw new Error('Indexed exchange original entry differs from its accepted job');
        const sourceBlocks = job.selection.selected_block_ids[original.id];
        if (sourceBlocks?.length !== 1 || (original.block_ids && !original.block_ids.includes(sourceBlocks[0])))
            throw new Error('Indexed exchange job has no one exact selected original block');
        const targetBlock = sourceBlocks[0];
        if (
            selected.turns.flatMap((turn) => turn.selected_blocks).filter((block) => block.id === targetBlock)
                .length !== 1
        )
            throw new Error('Indexed exchange original block is not uniquely active');
        let entry: ContextEntry = original;
        let lastRevision = job.enqueue_revision;
        for (const witness of witnesses) {
            const change = witness?.acceptance.context_change;
            if (!change?.removed_entry_ids.includes(entry.id)) continue;
            const remainders = change.remainder_entry_ids ?? [];
            if (
                change.kind !== 'replace_with_compaction' ||
                !remainders.length ||
                !change.selected_block_ids?.[entry.id] ||
                change.selected_block_ids[entry.id].includes(targetBlock) ||
                witness.acceptance.base_revision < lastRevision ||
                witness.acceptance.result_revision > selected.source.revision ||
                remainders.some((remainder) => !change.inserted_entry_ids.includes(remainder))
            )
                throw new Error('Indexed exchange remainder is not an accepted partial compaction');
            const candidates = remainders
                .map((id) => selected.lineage_entry_witnesses?.[id])
                .filter(
                    (candidate): candidate is ContextEntry =>
                        candidate?.type === 'source_turn' &&
                        candidate.turn_id === original.turn_id &&
                        candidate.block_ids?.includes(targetBlock) === true,
                );
            if (candidates.length !== 1 || !change.inserted_entry_ids.includes(candidates[0].id))
                throw new Error('Indexed exchange lacks one accepted target remainder ancestor');
            entry = candidates[0];
            lastRevision = witness.acceptance.result_revision;
        }
        const current = active.get(entry.id);
        if (
            current?.type !== 'source_turn' ||
            current.turn_id !== original.turn_id ||
            (current.block_ids !== undefined && !current.block_ids.includes(targetBlock)) ||
            canonicalJsonContentString(current) !== canonicalJsonContentString(entry)
        )
            throw new Error('Indexed exchange has no unique active selected remainder');
        entries.push(entry);
        blocks[entry.id] = [targetBlock];
    }
    if (new Set(entries.map((entry) => entry.id)).size !== entries.length)
        throw new Error('Indexed exchange rebound two sources to one active entry');
    return { entry_ids: entries.map((entry) => entry.id), selected_entries: entries, selected_block_ids: blocks };
}

/** Derive an ordered stage from its exact accepted predecessor, never from a same-looking
 * active entry. The host separately point-loads the receipt and same-cohort job association. */
export async function indexedPredecessorEntrySelection(
    jobInput: ProcessingJob,
    evidenceInput: IndexedProcessingPredecessorEvidence,
): Promise<Pick<ProcessingResolvedInput, 'entry_ids' | 'selected_block_ids' | 'selected_entries'>> {
    if (!preflightJsonInput({ job: jobInput, evidence: evidenceInput }, { max_bytes: MAX_WORKING_SET_BYTES }).success)
        throw new TypeError('Indexed predecessor evidence is not bounded JSON');
    const job = ProcessingJobSchema.parse(structuredClone(jobInput));
    const evidence = IndexedProcessingPredecessorEvidenceSchema.parse(structuredClone(evidenceInput));
    const {
        job: prior,
        resolution,
        output,
        resolution_receipt: resolvedReceipt,
        attempt,
        completion,
        receipt,
    } = evidence;
    const { output_fingerprint: _outputFingerprint, ...outputPayload } = output;
    if (
        job.selection.kind !== 'predecessor_output' ||
        job.selection_fingerprint !== (await fingerprintJson(job.selection)) ||
        job.configuration_fingerprint !== (await fingerprintJson(job.configuration)) ||
        job.stage_index < 1 ||
        job.selection.job_id !== prior.id ||
        prior.stage_index + 1 !== job.stage_index ||
        prior.processor_index >= job.processor_index ||
        prior.scope !== job.scope ||
        prior.selection_fingerprint !== (await fingerprintJson(prior.selection)) ||
        prior.configuration_fingerprint !== (await fingerprintJson(prior.configuration)) ||
        resolution.target_fingerprint !== prior.target_fingerprint ||
        resolvedReceipt.id !== `processing:resolve:${prior.id}` ||
        resolvedReceipt.conversation_id !== receipt.conversation_id ||
        resolvedReceipt.operation_kind !== 'processing' ||
        resolvedReceipt.processing_operation?.phase !== 'resolve' ||
        resolvedReceipt.processing_operation.job_id !== prior.id ||
        resolvedReceipt.processing_operation.policy_revision !== prior.policy_revision ||
        resolvedReceipt.processing_operation.result_fingerprint !== (await fingerprintJson(resolution)) ||
        resolvedReceipt.payload_fingerprint !== (await fingerprintJson(resolution)) ||
        resolvedReceipt.base_revision !== resolution.source_revision ||
        resolvedReceipt.result_revision !== resolution.source_revision + 1 ||
        resolvedReceipt.base_revision < prior.enqueue_revision ||
        resolvedReceipt.result_revision > receipt.base_revision ||
        receipt.result_revision !== receipt.base_revision + 1 ||
        (output.attempt_token === undefined
            ? attempt !== undefined || output.kind !== 'no_op'
            : !attempt ||
              attempt.job_id !== prior.id ||
              attempt.attempt_token !== output.attempt_token ||
              attempt.resolved_input_fingerprint !== output.resolved_input_fingerprint) ||
        prior.source_operation_id !== job.source_operation_id ||
        prior.enqueue_revision !== job.enqueue_revision ||
        prior.policy_revision !== job.policy_revision ||
        prior.target_fingerprint !== job.target_fingerprint ||
        resolution.job_id !== prior.id ||
        output.job_id !== prior.id ||
        output.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
        output.output_fingerprint !== (await fingerprintJson(outputPayload)) ||
        completion.job_id !== prior.id ||
        completion.output_fingerprint !== output.output_fingerprint ||
        completion.status === 'blocked' ||
        (completion.status === 'applied'
            ? output.kind !== 'proposal' && output.kind !== 'json_minification'
            : completion.status === 'no_op'
              ? output.kind !== 'no_op' && output.kind !== 'json_minification_no_op'
              : (output.kind !== 'failed' && output.kind !== 'unknown_outcome') ||
                prior.required ||
                prior.failure_behavior !== 'skip_with_diagnostic') ||
        (completion.status !== 'applied' &&
            (completion.inserted_entry_ids.length !== 0 || completion.context_change_operation_id !== undefined)) ||
        receipt.result_revision !== completion.result_revision ||
        (completion.status === 'applied'
            ? receipt.operation_kind !== 'context_change' ||
              receipt.id !== completion.context_change_operation_id ||
              receipt.id !== `processing:apply:${prior.id}` ||
              !receipt.context_change ||
              canonicalJsonContentString(receipt.context_change.selected_block_ids ?? null) !==
                  canonicalJsonContentString(resolution.selected_block_ids ?? null) ||
              canonicalJsonContentString(receipt.context_change.removed_entry_ids) !==
                  canonicalJsonContentString(resolution.entry_ids) ||
              canonicalJsonContentString(receipt.context_change.inserted_entry_ids) !==
                  canonicalJsonContentString(completion.inserted_entry_ids) ||
              canonicalJsonContentString(receipt.accepted_context_entry_ids ?? []) !==
                  canonicalJsonContentString(completion.inserted_entry_ids)
            : receipt.operation_kind !== 'processing' ||
              receipt.id !== `processing:complete:${prior.id}` ||
              receipt.processing_operation?.phase !== 'complete' ||
              receipt.processing_operation.job_id !== prior.id ||
              receipt.processing_operation.policy_revision !== prior.policy_revision ||
              receipt.processing_operation.result_fingerprint !== (await fingerprintJson(completion)) ||
              receipt.payload_fingerprint !== (await fingerprintJson(completion)))
    )
        throw new Error('Indexed stage lost its exact accepted predecessor output and completion');
    if (completion.status === 'applied' && output.kind === 'proposal' && receipt.context_change) {
        const appliedHash = receipt.context_change.source_fingerprint;
        const proposal =
            output.proposal.kind === 'replace_with_compaction'
                ? {
                      ...output.proposal,
                      replacement_turns: output.proposal.replacement_turns.map((turn) => {
                          if (
                              turn.provenance.type !== 'derived' ||
                              turn.provenance.source_hash !== resolution.source_fingerprint
                          )
                              throw new Error('Indexed predecessor proposal lost its resolved source fingerprint');
                          return { ...turn, provenance: { ...turn.provenance, source_hash: appliedHash } };
                      }),
                  }
                : output.proposal;
        const request = ContextChangeRequestSchema.parse({
            operation_id: receipt.id,
            recorded_at: receipt.recorded_at,
            expected_revision: receipt.base_revision,
            expected_context_revision: resolution.context_revision,
            expected_source_fingerprint: appliedHash,
            entry_ids: resolution.entry_ids,
            ...(resolution.selected_block_ids === undefined
                ? {}
                : {
                      selected_block_ids: resolution.selected_block_ids,
                      selected_entries: resolution.selected_entries,
                  }),
            proposal,
        });
        if (
            receipt.context_change.kind !== proposal.kind ||
            receipt.payload_fingerprint !== (await fingerprintJson(request))
        )
            throw new Error(
                'Indexed predecessor apply receipt differs from its exact output proposal/source selection',
            );
    }
    return completion.status === 'applied'
        ? { entry_ids: [...completion.inserted_entry_ids] }
        : {
              entry_ids: [...resolution.entry_ids],
              ...(resolution.selected_block_ids === undefined
                  ? {}
                  : { selected_block_ids: resolution.selected_block_ids }),
              ...(resolution.selected_entries === undefined ? {} : { selected_entries: resolution.selected_entries }),
          };
}

/** Data-only stage resolution after the indexed adapter independently authenticates the retained job.
 * This initial transition profile admits registered on-append ordinary text, never tool/program bytes.
 */
export async function resolveIndexedProcessingTextInput(
    selectedInput: unknown,
    jobInput: ProcessingJob,
    recordedAtInput: string,
    predecessorInput?: IndexedProcessingPredecessorEvidence,
): Promise<ProcessingResolvedInput> {
    const selected = ownSelected(selectedInput);
    if (!preflightJsonInput(jobInput).success) throw new TypeError('Indexed processing job is not bounded JSON');
    const job = ProcessingJobSchema.parse(structuredClone(jobInput));
    const recordedAt = TimestampSchema.parse(recordedAtInput);
    if (
        !(
            (job.processor_id === 'externalize-text' && job.processor_version === '1') ||
            (job.processor_id === INDEXED_EXCHANGE_PROCESSOR_ID &&
                job.processor_version === INDEXED_EXCHANGE_PROCESSOR_VERSION)
        ) ||
        (job.processor_id === INDEXED_EXCHANGE_PROCESSOR_ID
            ? job.scope !== 'on_append'
            : job.scope !== 'on_append' && job.scope !== 'manual' && job.scope !== 'on_budget') ||
        (job.stage_index === 0
            ? job.selection.kind !== 'entries' || predecessorInput !== undefined
            : job.selection.kind !== 'predecessor_output' || predecessorInput === undefined) ||
        (job.scope === 'on_budget' && job.target_fingerprint === undefined) ||
        Object.keys(job.configuration).length !== 0 ||
        (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
        (await fingerprintJson(job.selection)) !== job.selection_fingerprint
    )
        throw new Error('Indexed processing requires its exact supported text-stage configuration/selection');
    const frame = activeIndexedContextWorkingSet(selected);
    const context_fingerprint = await contextFingerprint(selected);
    const exchangeSelection =
        job.processor_id === INDEXED_EXCHANGE_PROCESSOR_ID
            ? await activeIndexedExchangeSelection(selected, job)
            : undefined;
    const preceding =
        predecessorInput === undefined ? undefined : await indexedPredecessorEntrySelection(job, predecessorInput);
    if (predecessorInput && predecessorInput.completion.result_revision > selected.source.revision)
        throw new Error('Indexed predecessor completion is newer than its selected source');
    const entriesSelection = job.selection.kind === 'entries' ? job.selection : preceding;
    if (!entriesSelection) throw new Error('Indexed stage has no resolved selected predecessor');
    let entryIds = exchangeSelection?.entry_ids ?? entriesSelection.entry_ids;
    let selectedBlockIds = exchangeSelection?.selected_block_ids ?? entriesSelection.selected_block_ids;
    let selectedEntries = exchangeSelection?.selected_entries ?? entriesSelection.selected_entries;
    if (job.processor_id === 'externalize-text') {
        const activeEntries = new Map(selected.context.entries.map((entry) => [entry.id, entry]));
        const eligibleEntries: ContextEntry[] = [];
        const eligibleBlocks: Record<string, string[]> = {};
        let requiresPartialSelection = entriesSelection.selected_block_ids !== undefined;
        for (const id of entriesSelection.entry_ids) {
            const entry = activeEntries.get(id);
            if (!entry) throw new Error('Indexed predecessor inserted entry is no longer active');
            const turn = frame.turns.get(entry.turn_id);
            if (!turn) throw new Error('Indexed predecessor entry lacks its exact active projection');
            const retainedEntry = entriesSelection.selected_entries?.find((candidate) => candidate.id === id);
            if (retainedEntry && canonicalJsonContentString(retainedEntry) !== canonicalJsonContentString(entry))
                throw new Error('Indexed text selection changed its exact immutable selected entry');
            const selectedIds = entriesSelection.selected_block_ids?.[id] ?? entry.block_ids;
            if (selectedIds?.some((blockId) => !turn.active_blocks.some((block) => block.id === blockId)))
                throw new Error('Indexed text selection omits an accepted selected block dependency');
            const candidateBlocks = turn.active_blocks.filter(
                (block) => selectedIds === undefined || selectedIds.includes(block.id),
            );
            const textIds = turn.active_blocks
                .filter(
                    (block) => block.type === 'text' && (selectedIds === undefined || selectedIds.includes(block.id)),
                )
                .map((block) => block.id);
            if (textIds.length !== candidateBlocks.length) requiresPartialSelection = true;
            if (textIds.length) {
                eligibleEntries.push(entry);
                eligibleBlocks[id] = textIds;
            }
        }
        entryIds = eligibleEntries.map((entry) => entry.id);
        selectedEntries = eligibleEntries.length && requiresPartialSelection ? eligibleEntries : undefined;
        selectedBlockIds = eligibleEntries.length && requiresPartialSelection ? eligibleBlocks : undefined;
    }
    const plan = entryIds.length
        ? await planContextChangeWorkingSet(
              frame,
              ContextChangePlanInputSchema.parse({
                  expected_revision: selected.source.revision,
                  expected_context_revision: selected.context.revision,
                  entry_ids: entryIds,
                  ...(selectedBlockIds === undefined
                      ? {}
                      : {
                            selected_block_ids: selectedBlockIds,
                            selected_entries: selectedEntries,
                        }),
              }),
          )
        : undefined;
    return ProcessingResolvedInputSchema.parse({
        job_id: job.id,
        source_revision: selected.source.revision,
        context_revision: selected.context.revision,
        entry_ids: plan?.entry_ids ?? [],
        ...(selectedBlockIds === undefined
            ? {}
            : {
                  selected_block_ids: selectedBlockIds,
                  selected_entries: selectedEntries,
              }),
        source_fingerprint: plan?.source_fingerprint ?? (await fingerprintJson({ entry_ids: [] })),
        context_fingerprint,
        source_turn_ids: plan?.source_turn_ids ?? [],
        ...(job.target_fingerprint === undefined ? {} : { target_fingerprint: job.target_fingerprint }),
        recorded_at: recordedAt,
    });
}

/** Pure deterministic processor/replay over owned selected records and already accepted archives.
 * The host independently grants retrieval and replays this same builder before its completion CAS.
 */
export async function buildIndexedTextExternalizationProposal(
    input: unknown,
): Promise<Extract<ProcessorResult, { kind: 'proposal' }>> {
    if (!preflightJsonInput(input, { max_bytes: MAX_WORKING_SET_BYTES }).success)
        throw new TypeError('Indexed processor workspace is not bounded JSON');
    const workspace = IndexedProcessingClaimWorkspaceSchema.parse(structuredClone(input));
    const { job, resolution, configuration, attempt, selected, archives, predecessor } = workspace;
    if (
        job.processor_id !== 'externalize-text' ||
        job.processor_version !== '1' ||
        (job.scope !== 'on_append' && job.scope !== 'manual' && job.scope !== 'on_budget') ||
        (job.stage_index === 0
            ? job.selection.kind !== 'entries' || predecessor !== undefined
            : job.selection.kind !== 'predecessor_output' || predecessor === undefined) ||
        (job.scope === 'on_budget' && job.target_fingerprint === undefined) ||
        resolution.target_fingerprint !== job.target_fingerprint ||
        Object.keys(job.configuration).length !== 0 ||
        (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
        (await fingerprintJson(job.selection)) !== job.selection_fingerprint ||
        job.id !== resolution.job_id ||
        job.id !== attempt.job_id ||
        resolution.source_revision > selected.source.revision ||
        resolution.context_revision !== selected.context.revision ||
        attempt.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
        resolution.context_fingerprint !== (await contextFingerprint(selected)) ||
        configuration.id !== job.processor_id ||
        configuration.version !== job.processor_version ||
        configuration.scope !== job.scope ||
        configuration.required !== job.required ||
        configuration.failure_behavior !== job.failure_behavior ||
        canonicalJsonContentString(configuration.config) !== canonicalJsonContentString(job.configuration)
    )
        throw new Error('Indexed processor workspace lost its exact job/resolution/attempt/configuration binding');
    if (predecessor) {
        const expected = await resolveIndexedProcessingTextInput(
            { ...selected, source: { ...selected.source, revision: resolution.source_revision } },
            job,
            resolution.recorded_at,
            predecessor,
        );
        if (canonicalJsonContentString(expected) !== canonicalJsonContentString(resolution))
            throw new Error('Indexed processor workspace differs from its ordered predecessor resolution');
    }
    return buildTextWorkingSetProposal(
        activeIndexedContextWorkingSet(selected),
        selected.tool_definitions,
        workspace.snapshot_at,
        job,
        resolution,
        configuration,
        archives.assets,
        archives.acceptance,
        archives.retrievals,
    );
}

/** The first registered processor is pure: output identity is reconstructed before issuing a scoped
 * transfer URL. Host and worker use these exact owned bytes; no caller timestamp or output authority.
 */
export async function buildIndexedTextExternalizationOutput(input: unknown): Promise<ProcessingOutputReceipt> {
    if (!preflightJsonInput(input, { max_bytes: MAX_WORKING_SET_BYTES }).success)
        throw new TypeError('Indexed processor workspace is not bounded JSON');
    const workspace = IndexedProcessingClaimWorkspaceSchema.parse(structuredClone(input));
    const result = await buildIndexedTextExternalizationProposal(workspace);
    const output = {
        ...result,
        job_id: workspace.job.id,
        resolved_input_fingerprint: await fingerprintJson(workspace.resolution),
        attempt_token: workspace.attempt.attempt_token,
        recorded_at: workspace.snapshot_at,
    };
    return ProcessingOutputReceiptSchema.parse({ ...output, output_fingerprint: await fingerprintJson(output) });
}

/** Exact deterministic completion over authenticated current active dependencies. This result is
 * a context delta/receipt and new compaction only; it is never a partial ConversationDocument.
 * The host independently verifies archived bytes/retrieval grants and global identifier uniqueness,
 * then the indexed store preserves every prior record while publishing the delta under head CAS.
 */
export async function applyIndexedTextExternalizationOutput(
    workspaceInput: unknown,
    outputInput: ProcessingOutputReceipt,
): Promise<ContextMutationResult> {
    const envelope = { workspace: workspaceInput, output: outputInput };
    if (!preflightJsonInput(envelope, { max_bytes: MAX_WORKING_SET_BYTES }).success)
        throw new TypeError('Indexed processing completion is not bounded JSON');
    const workspace = IndexedProcessingClaimWorkspaceSchema.parse(structuredClone(workspaceInput));
    const output = ProcessingOutputReceiptSchema.parse(structuredClone(outputInput));
    const expected = await buildIndexedTextExternalizationOutput(workspace);
    if (canonicalJsonContentString(output) !== canonicalJsonContentString(expected) || output.kind !== 'proposal')
        throw new Error('Indexed completion differs from its exact deterministic archived output');
    const frame = activeIndexedContextWorkingSet(workspace.selected);
    const selection = ContextChangePlanInputSchema.parse({
        expected_revision: frame.source.revision,
        expected_context_revision: frame.context.revision,
        entry_ids: workspace.resolution.entry_ids,
        ...(workspace.resolution.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: workspace.resolution.selected_block_ids,
                  selected_entries: workspace.resolution.selected_entries,
              }),
    });
    const plan = await planContextChangeWorkingSet(frame, selection);
    const proposal = output.proposal;
    if (proposal.kind !== 'replace_with_compaction' || proposal.fidelity !== 'retrievable')
        throw new Error('Indexed text completion requires its exact retrievable compaction');
    const request = ContextChangeRequestSchema.parse({
        operation_id: `processing:apply:${workspace.job.id}`,
        recorded_at: workspace.snapshot_at,
        ...selection,
        expected_source_fingerprint: plan.source_fingerprint,
        proposal: {
            ...proposal,
            replacement_turns: proposal.replacement_turns.map((turn) => {
                if (
                    turn.provenance.type !== 'derived' ||
                    turn.provenance.source_hash !== workspace.resolution.source_fingerprint
                )
                    throw new Error('Indexed proposal is not derived from its exact retained resolution');
                return { ...turn, provenance: { ...turn.provenance, source_hash: plan.source_fingerprint } };
            }),
        },
    });
    return applyContextMutationWorkingSet(
        {
            ...frame,
            assets: {
                ...frame.assets,
                ...Object.fromEntries(workspace.archives.assets.map((asset) => [asset.id, asset])),
            },
        },
        {
            compactions: Object.fromEntries(
                Object.values(workspace.selected.compaction_witnesses ?? {}).map((item) => [
                    item.compaction.id,
                    { id: item.compaction.id },
                ]),
            ),
            operation_receipts: {
                ...workspace.selected.operation_witnesses,
                [workspace.archives.acceptance.id]: workspace.archives.acceptance,
            },
            tool_definitions: workspace.selected.tool_definitions,
        },
        request,
        await fingerprintJson(request),
    );
}

/** Versioned identity of actual active dependency projections. It does not fingerprint cold history. */
export function indexedProcessingContextFingerprint(input: unknown): Promise<string> {
    return contextFingerprint(ownSelected(input));
}

/** Exact owned original text selection for host archival. No storage locator, access grant or asset
 * receipt is invented here; the host independently archives and publishes the returned bytes.
 */
export async function indexedTextExternalizationOriginals(
    selectedInput: unknown,
    jobInput: ProcessingJob,
    resolutionInput: ProcessingResolvedInput,
    retainedResolutionReceiptInput?: OperationReceipt,
    predecessorInput?: IndexedProcessingPredecessorEvidence,
) {
    const envelope = {
        selected: selectedInput,
        job: jobInput,
        resolution: resolutionInput,
        ...(retainedResolutionReceiptInput === undefined ? {} : { receipt: retainedResolutionReceiptInput }),
        ...(predecessorInput === undefined ? {} : { predecessor: predecessorInput }),
    };
    if (!preflightJsonInput(envelope, { max_bytes: MAX_WORKING_SET_BYTES }).success)
        throw new TypeError('Indexed originals are not bounded owned JSON');
    const owned = structuredClone(envelope);
    const selected = ownSelected(owned.selected);
    const job = ProcessingJobSchema.parse(owned.job);
    const resolution = ProcessingResolvedInputSchema.parse(owned.resolution);
    // A later phase root is not a new source for the immutable resolution. Replay at its
    // original revision only with the exact retained resolve receipt; current context/content
    // still participates in the full deterministic resolution comparison below.
    if (resolution.source_revision !== selected.source.revision) {
        const receipt = owned.receipt === undefined ? undefined : OperationReceiptSchema.parse(owned.receipt);
        const identity = await fingerprintJson(resolution);
        if (
            !receipt ||
            resolution.source_revision > selected.source.revision ||
            resolution.context_revision !== selected.context.revision ||
            receipt.id !== `processing:resolve:${job.id}` ||
            receipt.conversation_id !== selected.source.conversation_id ||
            receipt.operation_kind !== 'processing' ||
            receipt.processing_operation?.phase !== 'resolve' ||
            receipt.processing_operation.job_id !== job.id ||
            receipt.processing_operation.policy_revision !== job.policy_revision ||
            receipt.processing_operation.result_fingerprint !== identity ||
            receipt.payload_fingerprint !== identity ||
            receipt.base_revision !== resolution.source_revision ||
            receipt.result_revision !== receipt.base_revision + 1 ||
            receipt.result_revision > selected.source.revision
        )
            throw new Error('Indexed originals lost their exact retained resolution phase');
    }
    const expected = await resolveIndexedProcessingTextInput(
        { ...selected, source: { ...selected.source, revision: resolution.source_revision } },
        job,
        resolution.recorded_at,
        owned.predecessor,
    );
    if (canonicalJsonContentString(expected) !== canonicalJsonContentString(resolution))
        throw new Error('Indexed originals differ from their exact retained resolved selection');
    const texts = selectedTextWorkingSet(activeIndexedContextWorkingSet(selected), resolution);
    if (texts.length > 4096) throw new RangeError('Indexed originals exceed bounded text selection');
    const originals = [];
    let bytes = 0;
    for (const item of texts) {
        const integrity = await hashUtf8Content(item.text);
        bytes += integrity.byte_length;
        if (bytes > MAX_WORKING_SET_BYTES) throw new RangeError('Indexed originals exceed bounded byte selection');
        originals.push({ ...item, ...integrity });
    }
    return originals;
}
