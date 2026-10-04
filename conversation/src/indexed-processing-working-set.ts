import { canonicalJsonContentString, hashUtf8Content } from './content-integrity.js';
import { applyContextMutationWorkingSet, type ContextMutationResult } from './context-change-transition.js';
import { type ContextChangeWorkingSet, planContextChangeWorkingSet } from './context-change-working-set.js';
import { fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import type { ProcessorResult } from './processing.js';
import { ContextChangePlanInputSchema, ContextChangeRequestSchema } from './schemas/context-change.js';
import {
    type IndexedProcessingSelectedContext,
    IndexedProcessingSelectedContextSchema,
} from './schemas/indexed-head.js';
import { IndexedProcessingClaimWorkspaceSchema } from './schemas/indexed-processing.js';
import { TimestampSchema } from './schemas/primitives.js';
import {
    ProcessingJobSchema,
    ProcessingOutputReceiptSchema,
    ProcessingResolvedInputSchema,
} from './schemas/processing.js';
import { buildTextWorkingSetProposal, selectedTextWorkingSet } from './text-externalization-working-set.js';
import type { ProcessingJob, ProcessingOutputReceipt, ProcessingResolvedInput } from './types.js';

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
function activeWorkingSet(selected: IndexedProcessingSelectedContext): ContextChangeWorkingSet {
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

/** Data-only stage resolution after the indexed adapter independently authenticates the retained job.
 * This initial transition profile admits registered on-append ordinary text, never tool/program bytes.
 */
export async function resolveIndexedProcessingTextInput(
    selectedInput: unknown,
    jobInput: ProcessingJob,
    recordedAtInput: string,
): Promise<ProcessingResolvedInput> {
    const selected = ownSelected(selectedInput);
    if (!preflightJsonInput(jobInput).success) throw new TypeError('Indexed processing job is not bounded JSON');
    const job = ProcessingJobSchema.parse(structuredClone(jobInput));
    const recordedAt = TimestampSchema.parse(recordedAtInput);
    if (
        job.processor_id !== 'externalize-text' ||
        job.processor_version !== '1' ||
        job.scope !== 'on_append' ||
        job.selection.kind !== 'entries' ||
        job.stage_index !== 0 ||
        Object.keys(job.configuration).length !== 0 ||
        (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
        (await fingerprintJson(job.selection)) !== job.selection_fingerprint
    )
        throw new Error('Indexed processing requires its exact supported text-stage configuration/selection');
    const frame = activeWorkingSet(selected);
    const context_fingerprint = await contextFingerprint(selected);
    const plan = job.selection.entry_ids.length
        ? await planContextChangeWorkingSet(
              frame,
              ContextChangePlanInputSchema.parse({
                  expected_revision: selected.source.revision,
                  expected_context_revision: selected.context.revision,
                  entry_ids: job.selection.entry_ids,
                  ...(job.selection.selected_block_ids === undefined
                      ? {}
                      : {
                            selected_block_ids: job.selection.selected_block_ids,
                            selected_entries: job.selection.selected_entries,
                        }),
              }),
          )
        : undefined;
    return ProcessingResolvedInputSchema.parse({
        job_id: job.id,
        source_revision: selected.source.revision,
        context_revision: selected.context.revision,
        entry_ids: plan?.entry_ids ?? [],
        ...(job.selection.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: job.selection.selected_block_ids,
                  selected_entries: job.selection.selected_entries,
              }),
        source_fingerprint: plan?.source_fingerprint ?? (await fingerprintJson({ entry_ids: [] })),
        context_fingerprint,
        source_turn_ids: plan?.source_turn_ids ?? [],
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
    const { job, resolution, configuration, attempt, selected, archives } = workspace;
    if (
        job.processor_id !== 'externalize-text' ||
        job.processor_version !== '1' ||
        job.scope !== 'on_append' ||
        job.stage_index !== 0 ||
        job.selection.kind !== 'entries' ||
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
    return buildTextWorkingSetProposal(
        activeWorkingSet(selected),
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
    const frame = activeWorkingSet(workspace.selected);
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
) {
    const envelope = { selected: selectedInput, job: jobInput, resolution: resolutionInput };
    if (!preflightJsonInput(envelope, { max_bytes: MAX_WORKING_SET_BYTES }).success)
        throw new TypeError('Indexed originals are not bounded owned JSON');
    const owned = structuredClone(envelope);
    const selected = ownSelected(owned.selected);
    const job = ProcessingJobSchema.parse(owned.job);
    const resolution = ProcessingResolvedInputSchema.parse(owned.resolution);
    const expected = await resolveIndexedProcessingTextInput(selected, job, resolution.recorded_at);
    if (
        expected.source_fingerprint !== resolution.source_fingerprint ||
        expected.context_fingerprint !== resolution.context_fingerprint ||
        canonicalJsonContentString(expected.entry_ids) !== canonicalJsonContentString(resolution.entry_ids)
    )
        throw new Error('Indexed originals differ from their exact retained resolved selection');
    const texts = selectedTextWorkingSet(activeWorkingSet(selected), resolution);
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
