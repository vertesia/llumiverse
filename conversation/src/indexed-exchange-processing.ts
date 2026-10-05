import { canonicalJsonContentBytes, canonicalJsonContentString, hashContentBytes } from './content-integrity.js';
import { applyContextMutationWorkingSet, type ContextMutationResult } from './context-change-transition.js';
import { partitionSelection, planContextChangeWorkingSet, selectedRanges } from './context-change-working-set.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { INDEXED_EXCHANGE_PROCESSOR_ID, INDEXED_EXCHANGE_PROCESSOR_VERSION } from './indexed-exchange-constants.js';
import {
    activeIndexedContextWorkingSet,
    activeIndexedExchangeSelection,
    indexedProcessingContextFingerprint,
} from './indexed-processing-working-set.js';
import { preflightJsonInput } from './json-preflight.js';
import { ContextChangePlanInputSchema, ContextChangeRequestSchema } from './schemas/context-change.js';
import { IndexedProcessingClaimWorkspaceSchema } from './schemas/indexed-processing.js';
import { ProcessingOutputReceiptSchema } from './schemas/processing.js';
import type { ProcessingOutputReceipt } from './types.js';

/** Pure processor over accepted selected records and already-published copy assets. The worker
 * cannot select an original locator or issue its own retrieval grant; the host independently
 * replays this exact result after verifying source bytes and destination custody.
 */
export async function buildIndexedExchangeOutput(input: unknown): Promise<ProcessingOutputReceipt> {
    if (!preflightJsonInput(input, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new TypeError('Indexed exchange workspace is not bounded JSON');
    const workspace = IndexedProcessingClaimWorkspaceSchema.parse(structuredClone(input));
    const { selected, job, configuration, resolution, attempt, archives } = workspace;
    if (
        job.processor_id !== INDEXED_EXCHANGE_PROCESSOR_ID ||
        job.processor_version !== INDEXED_EXCHANGE_PROCESSOR_VERSION ||
        job.scope !== 'on_append' ||
        job.stage_index !== 0 ||
        job.selection.kind !== 'entries' ||
        Object.keys(job.configuration).length !== 0 ||
        (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
        (await fingerprintJson(job.selection)) !== job.selection_fingerprint ||
        configuration.id !== job.processor_id ||
        configuration.version !== job.processor_version ||
        configuration.scope !== job.scope ||
        configuration.required !== job.required ||
        configuration.failure_behavior !== job.failure_behavior ||
        canonicalJsonContentString(configuration.config) !== canonicalJsonContentString(job.configuration) ||
        job.id !== resolution.job_id ||
        job.id !== attempt.job_id ||
        resolution.source_revision > selected.source.revision ||
        resolution.context_revision !== selected.context.revision ||
        attempt.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
        archives.assets.length !== 2 ||
        archives.retrievals.length !== 2 ||
        archives.acceptance.conversation_id !== selected.source.conversation_id ||
        archives.acceptance.result_revision > selected.source.revision ||
        archives.acceptance.operation_kind !== undefined ||
        canonicalJsonContentString(archives.acceptance.accepted_asset_ids ?? []) !==
            canonicalJsonContentString(archives.assets.map((asset) => asset.id))
    ) {
        throw new Error('Indexed exchange processor lost its exact accepted job/archive binding');
    }
    const frame = activeIndexedContextWorkingSet(selected);
    const resolvedFrame = { ...frame, source: { ...frame.source, revision: resolution.source_revision } };
    const selection = ContextChangePlanInputSchema.parse({
        expected_revision: resolution.source_revision,
        expected_context_revision: selected.context.revision,
        entry_ids: resolution.entry_ids,
        ...(resolution.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: resolution.selected_block_ids,
                  selected_entries: resolution.selected_entries,
              }),
    });
    const plan = await planContextChangeWorkingSet(resolvedFrame, selection);
    const activeSelection = await activeIndexedExchangeSelection(selected, job);
    if (
        plan.source_fingerprint !== resolution.source_fingerprint ||
        canonicalJsonContentString(resolution.entry_ids) !== canonicalJsonContentString(activeSelection.entry_ids) ||
        canonicalJsonContentString(resolution.source_turn_ids) !== canonicalJsonContentString(plan.source_turn_ids) ||
        canonicalJsonContentString(resolution.selected_block_ids ?? {}) !==
            canonicalJsonContentString(activeSelection.selected_block_ids) ||
        canonicalJsonContentString(resolution.selected_entries ?? []) !==
            canonicalJsonContentString(activeSelection.selected_entries) ||
        resolution.context_fingerprint !== (await indexedProcessingContextFingerprint(selected))
    ) {
        throw new Error('Indexed exchange source differs from its exact resolved selection');
    }
    const ranges = selectedRanges(frame, partitionSelection(frame, selection));
    const sources = ranges.flatMap((range) => range.blocks);
    const call = sources[0];
    const result = sources[1];
    const original = result?.type === 'tool_result' ? result.content[0] : undefined;
    const originalAsset = original?.type === 'external_reference' ? selected.assets[original.asset_id] : undefined;
    const originalRequirement = selected.context.retrieval_requirements.filter(
        (item) =>
            item.asset_id === originalAsset?.id &&
            original?.type === 'external_reference' &&
            canonicalJsonContentString(item.retrieval) === canonicalJsonContentString(original.retrieval),
    );
    const originalPublication =
        originalRequirement.length === 1
            ? selected.operation_witnesses?.[originalRequirement[0]?.accepted_asset_operation_id ?? '']
            : undefined;
    if (
        sources.length !== 2 ||
        call?.type !== 'tool_call' ||
        call.executor !== 'application' ||
        result?.type !== 'tool_result' ||
        result.call_id !== call.call_id ||
        result.status === 'unknown' ||
        result.content.length !== 1 ||
        original?.type !== 'external_reference' ||
        original.original_type !== 'text' ||
        !originalAsset ||
        originalAsset.kind !== 'text' ||
        originalAsset.storage.type !== 'external' ||
        originalAsset.provenance.type !== 'received' ||
        originalAsset.content_hash !== original.content_hash ||
        originalAsset.byte_length === undefined ||
        !originalPublication ||
        originalPublication.operation_kind !== undefined ||
        originalPublication.conversation_id !== selected.source.conversation_id ||
        originalPublication.result_revision > selected.source.revision ||
        originalPublication.accepted_asset_ids?.filter((id) => id === originalAsset.id).length !== 1 ||
        originalPublication.accepted_retrieval_requirements?.filter(
            (item) =>
                item.id === originalRequirement[0]?.id &&
                item.asset_id === originalAsset.id &&
                canonicalJsonContentString(item.retrieval) === canonicalJsonContentString(original.retrieval),
        ).length !== 1
    ) {
        throw new Error('Indexed exchange must select one completed call and its exact archived result');
    }
    const callAsset = archives.assets[0];
    const resultAsset = archives.assets[1];
    const callTurn = [...frame.turns.values()].find((turn) => turn.active_blocks.some((block) => block.id === call.id));
    const callIntegrity = await hashContentBytes(canonicalJsonContentBytes(call));
    if (
        !callTurn ||
        callAsset.kind !== 'text' ||
        callAsset.id === resultAsset.id ||
        callAsset.storage.type !== 'external' ||
        callAsset.provenance.type !== 'received' ||
        callAsset.provenance.source_turn_id !== callTurn.header.id ||
        callAsset.content_hash !== callIntegrity.content_hash ||
        callAsset.byte_length !== callIntegrity.byte_length ||
        resultAsset.kind !== 'text' ||
        resultAsset.storage.type !== 'external' ||
        resultAsset.provenance.type !== 'derived' ||
        resultAsset.provenance.source_asset_id !== originalAsset.id ||
        resultAsset.provenance.transform_id !== 'conversation.archive_rehome' ||
        resultAsset.provenance.transform_version !== '1' ||
        resultAsset.content_hash !== originalAsset.content_hash ||
        resultAsset.byte_length !== originalAsset.byte_length ||
        archives.assets.some((_, index) => {
            const retrieval = archives.retrievals[index];
            const definition = selected.tool_definitions[retrieval?.tool_definition_id ?? ''];
            return (
                retrieval?.version !== 1 ||
                !definition ||
                definition.name !== retrieval.capability ||
                retrieval.arguments.asset_id !== archives.assets[index]?.id ||
                !selected.context.active_tool_definition_ids.includes(definition.id)
            );
        })
    ) {
        throw new Error('Indexed exchange copies differ from their selected source or active reader');
    }
    const compactionId = await deriveConversationId('indexed-exchange-compaction', job.id);
    let sourceOrdinal = 0;
    const replacementTurns = await Promise.all(
        ranges.map(async (range, rangeIndex) => {
            const blocks = await Promise.all(
                range.blocks.map(async (source) => {
                    const index = sourceOrdinal++;
                    const asset = archives.assets[index];
                    const retrieval = archives.retrievals[index];
                    if (!asset || !retrieval) throw new Error('Indexed exchange archive ordering changed');
                    return {
                        id: await deriveConversationId('indexed-exchange-reference', job.id, source.id),
                        type: 'external_reference' as const,
                        original_type: 'text' as const,
                        asset_id: asset.id,
                        description:
                            index === 0
                                ? 'Exact accepted tool call archived for retrieval'
                                : 'Exact accepted tool result archived for retrieval',
                        preview:
                            index === 0
                                ? 'Preview: accepted tool call is archived; retrieve its exact JSON before relying on it.'
                                : 'Preview: accepted tool result is archived; retrieve its exact bytes before relying on it.',
                        content_hash: asset.content_hash,
                        retrieval,
                    };
                }),
            );
            return {
                id: await deriveConversationId('indexed-exchange-turn', job.id, String(rangeIndex)),
                kind: 'agent' as const,
                authority: 'ordinary' as const,
                status: 'completed' as const,
                model_visibility: 'include' as const,
                timestamps: { recorded_at: workspace.snapshot_at },
                provenance: {
                    type: 'derived' as const,
                    derivation_id: compactionId,
                    source_turn_ids: ranges.length === 1 ? resolution.source_turn_ids : range.turn_ids,
                    ...(ranges.length > 1 || range.blocks.length > 1
                        ? { source_block_ids: range.block_ids }
                        : plan.source_block_ids.length
                          ? { source_block_ids: plan.source_block_ids }
                          : {}),
                    source_hash: resolution.source_fingerprint,
                },
                blocks,
            };
        }),
    );
    if (sourceOrdinal !== 2) throw new Error('Indexed exchange replacement does not cover both selected blocks');
    const resultPayload = {
        kind: 'proposal' as const,
        proposal: {
            kind: 'replace_with_compaction' as const,
            compaction_id: compactionId,
            strategy: {
                id: job.processor_id,
                version: job.processor_version,
                configuration_fingerprint: job.configuration_fingerprint,
            },
            replacement_turns: replacementTurns,
            fidelity: 'retrievable' as const,
            retained_asset_ids: archives.assets.map((asset) => asset.id),
            generation_ids: [],
            accepted_asset_operation_id: archives.acceptance.id,
            placement: {
                mode: ranges.length > 1 ? ('per_selected_range' as const) : ('first_selected' as const),
                causal_order: ranges.length > 1 ? ('preserved_disjoint_ranges' as const) : ('contiguous' as const),
            },
        },
        job_id: job.id,
        resolved_input_fingerprint: await fingerprintJson(resolution),
        attempt_token: attempt.attempt_token,
        recorded_at: workspace.snapshot_at,
    };
    return ProcessingOutputReceiptSchema.parse({
        ...resultPayload,
        output_fingerprint: await fingerprintJson(resultPayload),
    });
}

/** Reconstruct the same proposal against the current accepted selected records and retained run-owned
 * archive bytes. This is shared by the indexed completion stage and its host replay fence.
 */
export async function applyIndexedExchangeOutput(
    workspaceInput: unknown,
    outputInput: ProcessingOutputReceipt,
    originalArchiveBytes: ReadonlyMap<string, Uint8Array>,
): Promise<ContextMutationResult> {
    const envelope = { workspace: workspaceInput, output: outputInput };
    if (!preflightJsonInput(envelope, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new TypeError('Indexed exchange completion is not bounded JSON');
    const workspace = IndexedProcessingClaimWorkspaceSchema.parse(structuredClone(workspaceInput));
    const output = ProcessingOutputReceiptSchema.parse(structuredClone(outputInput));
    const expected = await buildIndexedExchangeOutput(workspace);
    if (canonicalJsonContentString(output) !== canonicalJsonContentString(expected) || output.kind !== 'proposal')
        throw new Error('Indexed exchange output differs from its exact deterministic proposal');
    const frame = activeIndexedContextWorkingSet(workspace.selected);
    const selection = ContextChangePlanInputSchema.parse({
        expected_revision: frame.source.revision,
        expected_context_revision: frame.context.revision,
        entry_ids: workspace.resolution.entry_ids,
        selected_block_ids: workspace.resolution.selected_block_ids,
        selected_entries: workspace.resolution.selected_entries,
    });
    const plan = await planContextChangeWorkingSet(frame, selection);
    if (output.proposal.kind !== 'replace_with_compaction' || output.proposal.fidelity !== 'retrievable')
        throw new Error('Indexed exchange output lacks its exact retrievable compaction');
    const request = ContextChangeRequestSchema.parse({
        operation_id: `processing:apply:${workspace.job.id}`,
        recorded_at: workspace.snapshot_at,
        ...selection,
        expected_source_fingerprint: plan.source_fingerprint,
        proposal: {
            ...output.proposal,
            replacement_turns: output.proposal.replacement_turns.map((turn) => {
                if (
                    turn.provenance.type !== 'derived' ||
                    turn.provenance.source_hash !== workspace.resolution.source_fingerprint
                )
                    throw new Error('Indexed exchange proposal lost its retained selected source');
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
            original_archive_bytes: originalArchiveBytes,
        },
        request,
        await fingerprintJson(request),
    );
}
