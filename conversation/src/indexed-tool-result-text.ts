import { canonicalJsonContentBytes, canonicalJsonContentString, hashContentBytes } from './content-integrity.js';
import type { ContextChangeWorkingSet } from './context-change-working-set.js';
import { fingerprintJson } from './identity.js';
import type { ProcessorResult } from './processing.js';
import { ConversationTurnSchema } from './schemas/content.js';
import type { IndexedProcessingSelectedContext } from './schemas/indexed-head.js';
import type {
    IndexedProcessingClaimWorkspace,
    IndexedProcessingPredecessorEvidence,
} from './schemas/indexed-processing.js';
import { ProcessingResolvedInputSchema } from './schemas/processing.js';
import {
    buildToolResultTextWorkingProposal,
    type ToolResultTextSelectionFrame,
    toolResultTextWorkingSelection,
} from './tool-result-text-externalization.js';
import { parseToolResultTextStrategy, supportsToolResultTextProcessingScope } from './tool-result-text-strategy.js';
import type { ConversationTurn, ProcessingJob, ProcessingResolvedInput } from './types.js';

/** Complete originals only. Partial active projections remain explicit active-block dependencies. */
export async function indexedToolResultTextFrame(
    selected: IndexedProcessingSelectedContext,
    frame: ContextChangeWorkingSet,
): Promise<ToolResultTextSelectionFrame> {
    const projections = [...selected.turns, ...(selected.replacement_turns ?? []).map((item) => item.projection)];
    const turns = new Map<string, ConversationTurn>();
    for (const projection of projections)
        if (projection.completeness === 'full_turn') {
            turns.set(
                projection.header.id,
                ConversationTurnSchema.parse({ ...projection.header, blocks: projection.selected_blocks }),
            );
        }
    for (const [id, witness] of Object.entries(selected.tool_result_call_witnesses ?? {})) {
        const projection = selected.turns.find((turn) => turn.header.id === id);
        const { blocks, ...header } = witness;
        if (
            !projection ||
            witness.id !== id ||
            witness.kind !== 'agent' ||
            canonicalJsonContentString(header) !== canonicalJsonContentString(projection.header) ||
            blocks.length !== projection.source_block_count ||
            (await hashContentBytes(canonicalJsonContentBytes(blocks.map((block) => block.id)))).content_hash !==
                projection.source_block_ids_hash ||
            projection.selected_blocks.some(
                (block, index) =>
                    canonicalJsonContentString(block) !==
                    canonicalJsonContentString(blocks[projection.selected_block_positions[index]]),
            )
        )
            throw new Error('Indexed tool-result call witness differs from its authenticated original projection');
        turns.set(id, witness);
    }
    const activeBlocks = new Map(
        frame.context.entries.map((entry) => {
            const active = frame.turns.get(entry.turn_id)?.active_blocks;
            if (!active) throw new Error('Indexed tool-result selection lacks its active replay closure');
            const blocks =
                entry.block_ids === undefined ? active : active.filter((block) => entry.block_ids?.includes(block.id));
            return [entry.id, blocks] as const;
        }),
    );
    return {
        source: frame.source,
        context: frame.context,
        turns,
        active_blocks: activeBlocks,
        execution_receipts: selected.execution_witnesses ?? {},
        projection_witnesses: selected.tool_result_projection_witnesses ?? {},
    };
}

function assertToolResultJob(job: ProcessingJob, predecessor?: IndexedProcessingPredecessorEvidence): void {
    parseToolResultTextStrategy(job);
    if (
        !supportsToolResultTextProcessingScope(job) ||
        job.selection.kind !== 'entries' ||
        job.selection.selected_block_ids !== undefined ||
        job.selection.selected_entries !== undefined ||
        job.target_fingerprint !== undefined
    )
        throw new Error('Indexed tool-result text job changed its whole authenticated result-entry boundary');
    if (job.stage_index === 0) {
        if (predecessor) throw new Error('Indexed first tool-result stage has an unexpected predecessor');
    } else if (
        !predecessor ||
        predecessor.job.stage_index + 1 !== job.stage_index ||
        predecessor.job.source_operation_id !== job.source_operation_id ||
        predecessor.job.enqueue_revision !== job.enqueue_revision ||
        predecessor.job.policy_revision !== job.policy_revision ||
        predecessor.job.scope !== job.scope ||
        predecessor.job.processor_index >= job.processor_index ||
        predecessor.completion.status === 'blocked'
    )
        throw new Error('Indexed tool-result text stage lacks its exact completed preceding stage');
}

export async function resolveIndexedToolResultTextInput(
    selected: IndexedProcessingSelectedContext,
    frame: ContextChangeWorkingSet,
    job: ProcessingJob,
    recordedAt: string,
    contextFingerprint: string,
    predecessor?: IndexedProcessingPredecessorEvidence,
): Promise<ProcessingResolvedInput> {
    assertToolResultJob(job, predecessor);
    if (job.selection.kind !== 'entries') throw new Error('Indexed tool-result selection changed its kind');
    if (
        (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
        (await fingerprintJson(job.selection)) !== job.selection_fingerprint
    )
        throw new Error('Indexed tool-result selection/configuration hash changed');
    const originals = await toolResultTextWorkingSelection(
        await indexedToolResultTextFrame(selected, frame),
        job.selection.entry_ids,
        job,
    );
    return ProcessingResolvedInputSchema.parse({
        job_id: job.id,
        source_revision: selected.source.revision,
        context_revision: selected.context.revision,
        entry_ids: originals.records.map((record) => record.entry.id),
        source_fingerprint: originals.source_fingerprint,
        context_fingerprint: contextFingerprint,
        source_turn_ids: originals.records.map((record) => record.turn.id),
        recorded_at: recordedAt,
    });
}

/** Host/worker share one deterministic proposal over exact accepted archives and original terminal records. */
export async function buildIndexedToolResultTextProposal(
    workspace: IndexedProcessingClaimWorkspace,
    frame: ContextChangeWorkingSet,
    contextFingerprint: string,
): Promise<Extract<ProcessorResult, { kind: 'proposal' }>> {
    const { job, configuration, resolution, attempt, selected, archives, predecessor } = workspace;
    assertToolResultJob(job, predecessor);
    if (
        job.id !== resolution.job_id ||
        job.id !== attempt.job_id ||
        resolution.source_revision > selected.source.revision ||
        resolution.context_revision !== selected.context.revision ||
        resolution.target_fingerprint !== undefined ||
        resolution.selected_block_ids !== undefined ||
        resolution.selected_entries !== undefined ||
        resolution.context_fingerprint !== contextFingerprint ||
        attempt.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
        configuration.id !== job.processor_id ||
        configuration.version !== job.processor_version ||
        configuration.scope !== job.scope ||
        configuration.required !== job.required ||
        configuration.failure_behavior !== job.failure_behavior ||
        canonicalJsonContentString(configuration.config) !== canonicalJsonContentString(job.configuration) ||
        (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
        (await fingerprintJson(job.selection)) !== job.selection_fingerprint ||
        archives.acceptance.base_revision < job.enqueue_revision ||
        archives.acceptance.result_revision > resolution.source_revision
    )
        throw new Error('Indexed tool-result workspace lost its exact job/resolution/attempt/archive binding');
    const sourceFrame = await indexedToolResultTextFrame(selected, frame);
    const originals = await toolResultTextWorkingSelection(sourceFrame, resolution.entry_ids, job);
    if (
        canonicalJsonContentString(originals.records.map((record) => record.entry.id)) !==
            canonicalJsonContentString(resolution.entry_ids) ||
        canonicalJsonContentString(originals.records.map((record) => record.turn.id)) !==
            canonicalJsonContentString(resolution.source_turn_ids)
    )
        throw new Error('Indexed tool-result resolution changed its exact original result entries');
    return buildToolResultTextWorkingProposal(
        sourceFrame,
        { ...selected.assets, ...Object.fromEntries(archives.assets.map((asset) => [asset.id, asset])) },
        selected.tool_definitions,
        archives.acceptance,
        'indexed_assets',
        job,
        resolution,
        archives.retrievals,
    );
}
