import { canonicalJsonContentString, hashUtf8Content } from './content-integrity.js';
import type { ContextMutationResult } from './context-change-transition.js';
import { collectAssetIds } from './context-change-working-set.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { cacheAfterContextEdit } from './conversation-edit-utils.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import type { ConversationProcessor, ProcessorResult } from './processing.js';
import { ContextChangeSchema } from './schemas/change.js';
import { CompactionRecordSchema, ConversationContextSchema } from './schemas/document.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import { type ToolResultProjectionWitness, ToolResultProjectionWitnessSchema } from './schemas/indexed-head.js';
import {
    isToolResultTextStrategy,
    parseToolResultTextStrategy,
    selectToolResultTextBlocks,
    supportsToolResultTextProcessingScope,
    TOOL_RESULT_TEXT_PROCESSOR_ID,
    type ToolResultTextStrategy,
} from './tool-result-text-strategy.js';

export {
    TOOL_RESULT_TEXT_CHAINED_PROCESSOR_VERSION,
    TOOL_RESULT_TEXT_PARTIAL_PROCESSOR_VERSION,
    TOOL_RESULT_TEXT_PROCESSOR_ID,
    TOOL_RESULT_TEXT_PROCESSOR_VERSION,
} from './tool-result-text-strategy.js';

import { ConversationTurnSchema, RetrievalCapabilitySchema } from './schemas/content.js';
import { ContextChangeProposalSchema } from './schemas/context-change.js';
import {
    createTextExternalizationProcessor,
    inspectTextExternalizationJobSelection,
    type TextExternalizationRetrievalBinder,
    textExternalizationArchiveInputs,
} from './text-externalization-processor.js';

import { assertToolResultReceiptFingerprint } from './tool-result-integrity.js';
import type {
    Asset,
    ContentBlock,
    ContextEntry,
    ConversationContext,
    ConversationDocument,
    ConversationRef,
    ConversationTurn,
    ExecutionReceipt,
    ExternalReferenceBlock,
    OperationReceipt,
    ProcessingJob,
    ProcessingOutputReceipt,
    ProcessingResolvedInput,
    RetrievalCapability,
    ToolDefinition,
    ToolResultBlock,
} from './types.js';
import { parseConversationDocument } from './validation.js';

/** Separate from v1 ordinary text processing: executable arguments are never selected or rewritten. */

const MAX_BYTES = 32 * 1024 * 1024;
const MAX_BLOCKS = 4096;
export function isToolResultTextProcessor(job: Pick<ProcessingJob, 'processor_id' | 'processor_version'>): boolean {
    return isToolResultTextStrategy(job.processor_id, job.processor_version);
}

export interface SelectedToolResultTextRecord {
    entry: ContextEntry;
    turn: ConversationTurn;
    result: ToolResultBlock;
    receipt: ExecutionReceipt;
    call_turn: ConversationTurn;
    projection_witness?: ToolResultProjectionWitness;
}
export interface ToolResultExternalizationText {
    entry_id: string;
    turn_id: string;
    result_block_id: string;
    block_id: string;
    text: string;
}

/** Explicit dependencies, never a partial ConversationDocument or inferred historical source. */
export interface ToolResultTextSelectionFrame {
    source: ConversationRef;
    context: ConversationContext;
    turns: ReadonlyMap<string, ConversationTurn>;
    active_blocks: ReadonlyMap<string, readonly ContentBlock[]>;
    execution_receipts: Readonly<Record<string, ExecutionReceipt>>;
    projection_witnesses?: Readonly<Record<string, ToolResultProjectionWitness>>;
    /** Materialized source already owned by its caller; only the nominated predecessor is inspected. */
    projection_records?: Pick<ConversationDocument, 'compactions' | 'processing' | 'operation_receipts'>;
}
export function materializedToolResultTextFrame(document: ConversationDocument): ToolResultTextSelectionFrame {
    const turns = createContextTurnIndex(document);
    return {
        source: { conversation_id: document.id, revision: document.revision },
        context: document.context,
        turns,
        active_blocks: new Map(
            document.context.entries.map((entry) => [entry.id, resolveContextEntry(turns, entry).blocks]),
        ),
        execution_receipts: document.execution_receipts,
        projection_records: document,
    };
}
/** A current projection is accepted evidence, never a new terminal result. Indexed hosts
 * supply the point-authenticated witness; materialized hosts prove the exact retained predecessor.
 */
async function toolResultProjectionWitness(
    frame: ToolResultTextSelectionFrame,
    entry: Extract<ContextEntry, { type: 'replacement_turn' }>,
    turn: ConversationTurn,
    terminal: ExecutionReceipt,
): Promise<ToolResultProjectionWitness> {
    let witness = frame.projection_witnesses?.[turn.id];
    if (!witness) {
        const records = frame.projection_records;
        const compaction = records?.compactions[entry.compaction_id];
        const acceptance = compaction ? records?.operation_receipts[compaction.operation_id] : undefined;
        const jobId = compaction?.operation_id.startsWith('processing:apply:')
            ? compaction.operation_id.slice('processing:apply:'.length)
            : '';
        const job = records?.processing.jobs?.[jobId];
        const resolution = records?.processing.resolved_inputs?.[jobId];
        const attempt = records?.processing.attempts?.[jobId];
        const output = records?.processing.outputs?.[jobId];
        const completion = records?.processing.completions?.[jobId];
        const original = terminal.result_turn_id ? frame.turns.get(terminal.result_turn_id) : undefined;
        const originalResult = original?.blocks[0];
        if (
            !compaction ||
            !acceptance ||
            !job ||
            !resolution ||
            !attempt ||
            output?.kind !== 'proposal' ||
            output.proposal.kind !== 'replace_with_compaction' ||
            completion?.status !== 'applied' ||
            !supportsToolResultTextProcessingScope(job) ||
            compaction.strategy.version !== job.processor_version ||
            compaction.strategy.id !== job.processor_id ||
            compaction.strategy.configuration_fingerprint !== job.configuration_fingerprint ||
            job.configuration_fingerprint !== (await fingerprintJson(job.configuration)) ||
            job.selection_fingerprint !== (await fingerprintJson(job.selection)) ||
            resolution.job_id !== job.id ||
            attempt.job_id !== job.id ||
            attempt.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
            completion.job_id !== job.id ||
            completion.output_fingerprint !== output.output_fingerprint ||
            completion.context_change_operation_id !== acceptance.id ||
            completion.result_revision !== acceptance.result_revision ||
            acceptance.operation_kind !== 'context_change' ||
            acceptance.context_change?.source_fingerprint !== resolution.source_fingerprint ||
            acceptance.payload_fingerprint !== (await fingerprintJson(output.proposal)) ||
            output.proposal.compaction_id !== compaction.id ||
            compaction.fidelity !== output.proposal.fidelity ||
            compaction.source.source_fingerprint !== resolution.source_fingerprint ||
            compaction.source.block_ids === undefined ||
            canonicalJsonContentString(compaction.source.block_ids) !==
                canonicalJsonContentString(
                    output.proposal.replacement_turns.flatMap((candidate) =>
                        candidate.provenance.type === 'derived' ? (candidate.provenance.source_block_ids ?? []) : [],
                    ),
                ) ||
            canonicalJsonContentString(compaction.retained_asset_ids) !==
                canonicalJsonContentString(output.proposal.retained_asset_ids) ||
            canonicalJsonContentString(compaction.generation_ids) !==
                canonicalJsonContentString(output.proposal.generation_ids) ||
            compaction.derivation_generation !== undefined ||
            compaction.created_at !== acceptance.recorded_at ||
            compaction.metadata?.payload_fingerprint !== acceptance.payload_fingerprint ||
            compaction.metadata?.applied_revision !== acceptance.result_revision ||
            compaction.metadata?.source_context_revision !== resolution.context_revision ||
            canonicalJsonContentString(compaction.replacement_turns) !==
                canonicalJsonContentString(output.proposal.replacement_turns) ||
            canonicalJsonContentString(compaction.source.turn_ids) !==
                canonicalJsonContentString(resolution.source_turn_ids) ||
            !compaction.replacement_turns.some(
                (candidate) => canonicalJsonContentString(candidate) === canonicalJsonContentString(turn),
            ) ||
            original?.kind !== 'tool' ||
            original.execution_id !== terminal.id ||
            original.blocks.length !== 1 ||
            originalResult?.type !== 'tool_result' ||
            originalResult.call_id !== terminal.call_id
        )
            throw new Error('Chained tool-result selection lacks its exact accepted predecessor and original terminal');
        const predecessorIds = new Set(
            resolution.source_turn_ids.flatMap((sourceId) => {
                const source = frame.turns.get(sourceId);
                return source?.provenance.type === 'derived' ? [source.provenance.derivation_id] : [];
            }),
        );
        const expectedSupersession =
            job.processor_version === '3' && predecessorIds.size === 1 ? [...predecessorIds][0] : undefined;
        if (compaction.supersedes_compaction_id !== expectedSupersession)
            throw new Error('Chained tool-result selection changed its exact accepted predecessor supersession');
        const { output_fingerprint: outputFingerprint, ...outputPayload } = output;
        const expectedAcceptance = await toolResultTextApplyReceipt(
            { conversation_id: frame.source.conversation_id, revision: acceptance.base_revision },
            job,
            resolution,
            output.proposal,
            acceptance.recorded_at,
        );
        if (
            output.job_id !== job.id ||
            output.attempt_token !== attempt.attempt_token ||
            output.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
            outputFingerprint !== (await fingerprintJson(outputPayload)) ||
            acceptance.id !== expectedAcceptance.id ||
            acceptance.conversation_id !== expectedAcceptance.conversation_id ||
            acceptance.result_revision !== expectedAcceptance.result_revision ||
            acceptance.result_revision > frame.source.revision ||
            canonicalJsonContentString(acceptance.context_change) !==
                canonicalJsonContentString(expectedAcceptance.context_change) ||
            canonicalJsonContentString(acceptance.accepted_context_entry_ids) !==
                canonicalJsonContentString(expectedAcceptance.accepted_context_entry_ids) ||
            canonicalJsonContentString(completion.inserted_entry_ids) !==
                canonicalJsonContentString(expectedAcceptance.accepted_context_entry_ids)
        )
            throw new Error('Chained tool-result selection changed its accepted predecessor output fence or receipt');
        await assertToolResultReceiptFingerprint(originalResult, terminal);
        const { replacement_turns: _turns, original_context: _originalContext, ...header } = compaction;
        witness = {
            compaction_id: compaction.id,
            compaction_fingerprint: await fingerprintJson(header),
            projection_fingerprint: await fingerprintJson(turn),
            terminal_execution_id: terminal.id,
            original_result_turn_id: original.id,
            original_result_block_id: originalResult.id,
        };
    }
    const owned = ToolResultProjectionWitnessSchema.parse(witness);
    if (
        owned.compaction_id !== entry.compaction_id ||
        owned.projection_fingerprint !== (await fingerprintJson(turn)) ||
        owned.terminal_execution_id !== terminal.id ||
        owned.original_result_turn_id !== terminal.result_turn_id ||
        turn.provenance.type !== 'derived' ||
        turn.provenance.derivation_id !== entry.compaction_id
    )
        throw new Error('Chained tool-result selection changed its current projection/original terminal binding');
    return owned;
}

function toolResultSupersession(frame: ToolResultTextSelectionFrame, entryIds: readonly string[], job: ProcessingJob) {
    if (job.processor_version !== '3') return {};
    const predecessors = new Set(
        frame.context.entries
            .filter((entry) => entryIds.includes(entry.id))
            .flatMap((entry) => (entry.type === 'replacement_turn' ? [entry.compaction_id] : [])),
    );
    return predecessors.size === 1 ? { supersedes_compaction_id: [...predecessors][0] } : {};
}

async function selectedResults(
    document: ToolResultTextSelectionFrame,
    entryIds: readonly string[],
    strategy?: ToolResultTextStrategy,
): Promise<SelectedToolResultTextRecord[]> {
    if (entryIds.length > MAX_BLOCKS || new Set(entryIds).size !== entryIds.length)
        throw new Error('Tool-result text selection exceeds its unique entry bound');
    const selected = document.context.entries.filter((entry) => entryIds.includes(entry.id));
    if (canonicalJsonContentString(selected.map((entry) => entry.id)) !== canonicalJsonContentString(entryIds))
        throw new Error('Tool-result text selection is not exact active context order');
    const records: SelectedToolResultTextRecord[] = [];
    for (const entry of selected) {
        const turn = document.turns.get(entry.turn_id);
        if (
            (entry.type !== 'source_turn' && strategy?.processor_version !== '3') ||
            !turn ||
            turn.kind !== 'tool' ||
            turn.status !== 'completed' ||
            turn.authority !== 'ordinary' ||
            turn.model_visibility !== 'include' ||
            document.context.protected_entry_ids.includes(entry.id)
        )
            throw new Error('Tool-result text selection contains a protected, unresolved or derived entry');
        // Whole result entries only. Mixing unrelated blocks cannot silently lose model-visible material.
        if (entry.block_ids !== undefined || turn.blocks.length !== 1 || turn.blocks[0]?.type !== 'tool_result')
            throw new Error('Tool-result text selection requires one complete result block');
        const result = turn.blocks[0];
        const receipt = turn.execution_id ? document.execution_receipts[turn.execution_id] : undefined;
        if (
            receipt?.executor !== 'application' ||
            (entry.type === 'source_turn' && receipt.result_turn_id !== turn.id) ||
            receipt.call_id !== result.call_id ||
            result.status === 'unknown' ||
            receipt.status !== result.status ||
            !receipt.call_source ||
            receipt.metadata?.retrieval_excerpt !== undefined ||
            result.content.some((block) => block.type === 'native_replay')
        )
            throw new Error('Tool-result text selection lacks an eligible exact terminal application receipt');
        const projectionWitness =
            entry.type === 'replacement_turn'
                ? await toolResultProjectionWitness(document, entry, turn, receipt)
                : undefined;
        if (!projectionWitness) await assertToolResultReceiptFingerprint(result, receipt);
        const callSource = receipt.call_source;
        const callTurn = document.turns.get(callSource.turn_id);
        const call = callTurn?.blocks.find((block) => block.id === callSource.block_id);
        const callEntries = document.context.entries.filter(
            (item) =>
                item.type === 'source_turn' &&
                item.turn_id === callTurn?.id &&
                (item.block_ids === undefined || item.block_ids.includes(callSource.block_id)),
        );
        if (
            callTurn?.status !== 'completed' ||
            callTurn.authority !== 'ordinary' ||
            callTurn.model_visibility !== 'include' ||
            callTurn.blocks.some(
                (block) => block.type === 'native_replay' && block.dependency_policy !== 'discard_on_dependency_change',
            ) ||
            call?.type !== 'tool_call' ||
            call.executor !== 'application' ||
            call.call_id !== receipt.call_id ||
            callSource.conversation.conversation_id !== document.source.conversation_id ||
            callSource.conversation.revision > document.source.revision ||
            (await fingerprintJson(call)) !== callSource.call_fingerprint ||
            callEntries.length !== 1 ||
            callEntries.some((item) => document.context.protected_entry_ids.includes(item.id))
        )
            throw new Error('Tool-result text selection changed or protected its exact executed call dependency');
        // The executed call and its native replay remain active and byte-for-byte unchanged.
        // Replacing result content cannot silently invalidate replay in another active entry.
        const changedBlocks = new Set([result.id, ...result.content.map((block) => block.id)]);
        for (const active of document.context.entries) {
            const blocks = document.active_blocks.get(active.id);
            if (!blocks) throw new Error('Tool-result selection lacks its complete active replay closure');
            for (const block of blocks) {
                const replays =
                    block.type === 'native_replay'
                        ? [block]
                        : block.type === 'tool_result'
                          ? block.content.filter((nested) => nested.type === 'native_replay')
                          : [];
                for (const replay of replays)
                    if (
                        replay.dependencies.turn_ids.includes(turn.id) ||
                        replay.dependencies.block_ids.some((id) => changedBlocks.has(id))
                    )
                        throw new Error(
                            'Tool-result text selection would invalidate an active native replay dependency',
                        );
            }
        }
        if (!result.content.some((block) => block.type === 'text'))
            throw new Error('Tool-result text selection has no inline text');
        records.push({
            entry,
            turn,
            result,
            receipt,
            call_turn: callTurn,
            ...(projectionWitness ? { projection_witness: projectionWitness } : {}),
        });
    }
    return records;
}

/** Eligibility is conservative: excluded results remain in context, without invoking a processor. */
export async function eligibleToolResultTextEntries(
    document: ConversationDocument,
    accepted: readonly string[],
    strategy?: ToolResultTextStrategy,
): Promise<string[]> {
    return eligibleToolResultTextWorkingEntries(materializedToolResultTextFrame(document), accepted, strategy);
}

export async function eligibleToolResultTextWorkingEntries(
    frame: ToolResultTextSelectionFrame,
    accepted: readonly string[],
    strategy?: ToolResultTextStrategy,
): Promise<string[]> {
    if (strategy) parseToolResultTextStrategy(strategy);
    const ids: string[] = [];
    for (const entry of frame.context.entries) {
        if (!accepted.includes(entry.id) || frame.turns.get(entry.turn_id)?.kind !== 'tool') continue;
        try {
            await selectedResults(frame, [entry.id], strategy);
            ids.push(entry.id);
        } catch {
            /* Protected/unresolved result entries are left unchanged; source integrity is the owning adapter's gate. */
        }
    }
    if (ids.length && strategy)
        return (await toolResultTextWorkingSelection(frame, ids, strategy)).records.map((record) => record.entry.id);
    return ids;
}

export function toolResultTextSelection(
    document: ConversationDocument,
    entryIds: readonly string[],
    strategy?: ToolResultTextStrategy,
) {
    return toolResultTextWorkingSelection(materializedToolResultTextFrame(document), entryIds, strategy);
}

export async function toolResultTextWorkingSelection(
    document: ToolResultTextSelectionFrame,
    entryIds: readonly string[],
    strategy?: ToolResultTextStrategy,
): Promise<{
    records: SelectedToolResultTextRecord[];
    texts: ToolResultExternalizationText[];
    integrities: Awaited<ReturnType<typeof hashUtf8Content>>[];
    source_fingerprint: string;
}> {
    const records = await selectedResults(document, entryIds, strategy);
    const candidates: ToolResultExternalizationText[] = [];
    for (const record of records)
        for (const block of record.result.content) {
            if (block.type === 'text')
                candidates.push({
                    entry_id: record.entry.id,
                    turn_id: record.turn.id,
                    result_block_id: record.result.id,
                    block_id: block.id,
                    text: block.text,
                });
        }
    const texts = selectToolResultTextBlocks(candidates, strategy);
    const chosenEntries = new Set(texts.map((text) => text.entry_id));
    const selectedRecords = records.filter((record) => chosenEntries.has(record.entry.id));
    if (texts.length > MAX_BLOCKS) throw new RangeError('Tool-result text selection exceeds block bound');
    const integrities = await Promise.all(texts.map((item) => hashUtf8Content(item.text)));
    if (integrities.reduce((sum, item) => sum + item.byte_length, 0) > MAX_BYTES)
        throw new RangeError('Tool-result text selection exceeds byte bound');
    const sourceFingerprint = await fingerprintJson({
        kind: 'tool_result_text_selection',
        records: selectedRecords.map((item) => ({
            entry: item.entry,
            turn: item.turn,
            receipt: item.receipt,
            call_turn: item.call_turn,
            ...(item.projection_witness ? { projection_witness: item.projection_witness } : {}),
        })),
        ...(strategy && strategy.processor_version !== '1'
            ? { selected_text_blocks: texts.map(({ text: _text, ...identity }) => identity) }
            : {}),
    });
    return { records: selectedRecords, texts, integrities, source_fingerprint: sourceFingerprint };
}

export async function toolResultExternalizationArchiveInputs(document: ConversationDocument, job: ProcessingJob) {
    if (
        !isToolResultTextProcessor(job) ||
        !supportsToolResultTextProcessingScope(job) ||
        job.selection.kind !== 'entries' ||
        job.selection.selected_block_ids !== undefined
    )
        throw new Error('Tool-result externalization requires its exact whole-entry scheduled selection');
    if (job.stage_index > 0) {
        const predecessors = Object.values(document.processing.jobs ?? {}).filter(
            (candidate) =>
                candidate.source_operation_id === job.source_operation_id &&
                candidate.stage_index === job.stage_index - 1,
        );
        const completion =
            predecessors.length === 1 ? document.processing.completions?.[predecessors[0].id] : undefined;
        if (!completion || completion.status === 'blocked')
            throw new Error('Tool-result text archive requires its completed preceding processing stage');
    }
    const selected = await toolResultTextSelection(document, job.selection.entry_ids, job);
    return {
        texts: selected.texts,
        integrities: selected.integrities,
        payload_fingerprint: await toolResultTextArchiveFingerprint(selected),
    };
}

export function toolResultTextArchiveFingerprint(
    selected: Pick<Awaited<ReturnType<typeof toolResultTextWorkingSelection>>, 'texts' | 'integrities'>,
): Promise<string> {
    return fingerprintJson({
        kind: 'tool_result_text_archive',
        blocks: selected.texts.map(({ text: _text, ...item }, index) => ({ ...item, ...selected.integrities[index] })),
    });
}

function resultTextPreview(text: string): string {
    let result = '';
    for (const scalar of text) {
        if (result.length + scalar.length > 512) break;
        result += scalar;
    }
    return result;
}

type ProposalResult = Extract<ProcessorResult, { kind: 'proposal' }>;
export async function buildToolResultTextExternalizationProposal(
    document: ConversationDocument,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    retrievals: readonly RetrievalCapability[],
): Promise<ProposalResult> {
    const receipt = document.operation_receipts[`processing:archive:${job.id}`];
    if (!receipt) throw new Error('Tool-result text processing requires its durably accepted archive');
    // The materialized predecessor gate remains independent of the pure selected-record builder.
    await toolResultExternalizationArchiveInputs(document, job);
    return buildToolResultTextWorkingProposal(
        materializedToolResultTextFrame(document),
        document.assets,
        document.tool_definitions,
        receipt,
        'tool_result_text',
        job,
        resolution,
        retrievals,
    );
}

export async function buildToolResultTextWorkingProposal(
    frame: ToolResultTextSelectionFrame,
    assets: Readonly<Record<string, Asset>>,
    toolDefinitions: Readonly<Record<string, ToolDefinition>>,
    receipt: OperationReceipt,
    archiveProfile: 'tool_result_text' | 'indexed_assets',
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    retrievals: readonly RetrievalCapability[],
): Promise<ProposalResult> {
    if (!isToolResultTextProcessor(job)) throw new Error('Tool-result text processor configuration is unavailable');
    parseToolResultTextStrategy(job);
    const selected = await toolResultTextWorkingSelection(frame, resolution.entry_ids, job);
    if (selected.source_fingerprint !== resolution.source_fingerprint)
        throw new Error('Tool-result text processing lost its original call/result/receipt selection');
    const assetIds = receipt.accepted_asset_ids ?? [];
    if (
        !receipt ||
        receipt.operation_kind !== undefined ||
        retrievals.length !== selected.texts.length ||
        assetIds.length !== selected.texts.length ||
        new Set(assetIds).size !== assetIds.length ||
        receipt.id !== `processing:archive:${job.id}` ||
        receipt.conversation_id !== frame.source.conversation_id ||
        receipt.result_revision > frame.source.revision ||
        receipt.payload_fingerprint !==
            (archiveProfile === 'indexed_assets'
                ? await fingerprintJson(assetIds.map((id) => assets[id]))
                : await toolResultTextArchiveFingerprint(selected))
    )
        throw new Error('Tool-result text processing requires its exact durably accepted ordered archive');
    const compactionId = await deriveConversationId('tool-result-text-compaction', job.id);
    const replacementTurns: ConversationTurn[] = [];
    let ordinal = 0;
    const chosenTextIds = new Set(selected.texts.map((text) => text.block_id));
    for (const record of selected.records) {
        const content = [];
        for (const block of record.result.content) {
            // v3's exact nested predecessor mapping is this registered ID derivation. IDs must
            // differ across retained projections; every copied block keeps its non-ID payload.
            const id = await deriveConversationId('tool-result-text-block', job.id, block.id);
            if (block.type !== 'text' || !chosenTextIds.has(block.id)) {
                content.push({ ...block, id });
                continue;
            }
            const asset = assets[assetIds[ordinal]];
            const exact = selected.integrities[ordinal];
            const retrieval = RetrievalCapabilitySchema.parse(retrievals[ordinal]);
            const definition = toolDefinitions[retrieval.tool_definition_id ?? ''];
            if (
                asset?.kind !== 'text' ||
                asset.storage.type !== 'external' ||
                asset.content_hash !== exact.content_hash ||
                asset.byte_length !== exact.byte_length ||
                !definition ||
                !frame.context.active_tool_definition_ids.includes(definition.id) ||
                definition.name !== retrieval.capability ||
                retrieval.version !== 1
            )
                throw new Error('Tool-result text archive changed exact bytes or pinned active retrieval definition');
            content.push({
                id,
                type: 'external_reference' as const,
                asset_id: asset.id,
                original_type: 'text' as const,
                content_hash: exact.content_hash,
                preview: resultTextPreview(block.text),
                description: 'Exact original tool-result text is available on demand',
                retrieval,
            });
            ordinal++;
        }
        replacementTurns.push(
            ConversationTurnSchema.parse({
                ...record.turn,
                // execution_id is a lineage link to the immutable original, not a new executed result.
                // The full original block still matches that receipt; only this derived projection differs.
                id: await deriveConversationId('tool-result-text-turn', job.id, record.turn.id),
                timestamps: { recorded_at: resolution.recorded_at },
                provenance: {
                    type: 'derived',
                    derivation_id: compactionId,
                    source_turn_ids: [record.turn.id],
                    source_block_ids: [record.result.id],
                    source_hash: resolution.source_fingerprint,
                },
                blocks: [
                    {
                        ...record.result,
                        id: await deriveConversationId('tool-result-text-result', job.id, record.result.id),
                        content,
                    },
                ],
            }),
        );
    }
    const retainedAssetIds =
        job.processor_version !== '1'
            ? [...new Set([...assetIds, ...replacementTurns.flatMap((turn) => [...collectAssetIds(turn.blocks)])])]
            : assetIds;
    if (retainedAssetIds.some((id) => !assets[id]))
        throw new Error('Tool-result text projection lost an authenticated retained original asset');
    return {
        kind: 'proposal',
        proposal: ContextChangeProposalSchema.parse({
            kind: 'replace_with_compaction',
            compaction_id: compactionId,
            strategy: {
                id: TOOL_RESULT_TEXT_PROCESSOR_ID,
                version: job.processor_version,
                configuration_fingerprint: await fingerprintJson(job.configuration),
            },
            replacement_turns: replacementTurns,
            fidelity: 'retrievable',
            accepted_asset_operation_id: receipt.id,
            retained_asset_ids: retainedAssetIds,
            generation_ids: [],
            placement: { mode: 'per_selected_range', causal_order: 'preserved_disjoint_ranges' },
        }),
    };
}

export function createToolResultTextExternalizationProcessor(
    bindRetrieval: (input: {
        asset: ConversationDocument['assets'][string];
        receipt: ConversationDocument['operation_receipts'][string];
        document: ConversationDocument;
        job: ProcessingJob;
    }) => RetrievalCapability,
): ConversationProcessor {
    return {
        async run({ document, job, resolved_input: resolution }) {
            const receipt = document.operation_receipts[`processing:archive:${job.id}`];
            if (!receipt) throw new Error('Tool-result text processor cannot run before durable archive receipt');
            const retrievals = (receipt.accepted_asset_ids ?? []).map((id) => {
                const asset = document.assets[id];
                if (!asset) throw new Error('Tool-result archive asset is missing');
                return bindRetrieval({ asset, receipt, document, job });
            });
            return buildToolResultTextExternalizationProposal(document, job, resolution, retrievals);
        },
    };
}

/** Only newly archived texts introduce retrieval bindings. Older references keep their own publication. */
export function toolResultTextArchiveRetrievals(
    proposal: Extract<ProposalResult['proposal'], { kind: 'replace_with_compaction' }>,
    archive: OperationReceipt,
): RetrievalCapability[] {
    const ids = archive.accepted_asset_ids ?? [];
    if (new Set(ids).size !== ids.length || ids.some((id) => !proposal.retained_asset_ids.includes(id)))
        throw new Error('Tool-result text projection changed its exact ordered archive assets');
    const references = new Map<string, ExternalReferenceBlock[]>();
    for (const turn of proposal.replacement_turns)
        for (const block of turn.blocks)
            if (block.type === 'tool_result')
                for (const nested of block.content)
                    if (nested.type === 'external_reference') {
                        const retained = references.get(nested.asset_id) ?? [];
                        retained.push(nested);
                        references.set(nested.asset_id, retained);
                    }
    return ids.map((id) => {
        const matches = references.get(id) ?? [];
        if (matches.length !== 1) throw new Error('Tool-result text projection lost its unique archived reference');
        return matches[0].retrieval;
    });
}

/** Dedicated deterministic application. Generic context-edit eligibility remains unchanged. */
export async function applyToolResultTextExternalizationOutput(
    document: ConversationDocument,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    output: Extract<ProcessingOutputReceipt, { kind: 'proposal' }>,
    recordedAt: string,
): Promise<ConversationDocument> {
    const proposal = output.proposal;
    if (proposal.kind !== 'replace_with_compaction' || output.usage !== undefined)
        throw new Error('Tool-result text output is not a deterministic retrievable projection');
    const archive = document.operation_receipts[`processing:archive:${job.id}`];
    if (!archive) throw new Error('Tool-result text output has no retained archive acceptance');
    const retrievals = toolResultTextArchiveRetrievals(proposal, archive);
    const expected = await buildToolResultTextExternalizationProposal(document, job, resolution, retrievals);
    if (canonicalJsonContentString(expected.proposal) !== canonicalJsonContentString(proposal))
        throw new Error('Tool-result text output changed exact original/dependency/archive/projection bytes');
    const mutation = await toolResultTextWorkingMutation(
        materializedToolResultTextFrame(document),
        job,
        resolution,
        proposal,
        recordedAt,
    );
    const { coverage: _coverage, ...processing } = document.processing;
    if (!mutation.compaction) throw new Error('Tool-result text mutation lacks its exact compaction');
    return parseConversationDocument({
        ...document,
        revision: mutation.receipt.result_revision,
        updated_at: recordedAt,
        context: mutation.context,
        compactions: { ...document.compactions, [mutation.compaction.id]: mutation.compaction },
        operation_receipts: { ...document.operation_receipts, [mutation.receipt.id]: mutation.receipt },
        processing: {
            ...processing,
            completions: {
                ...processing.completions,
                [job.id]: {
                    job_id: job.id,
                    output_fingerprint: output.output_fingerprint,
                    status: 'applied',
                    result_revision: mutation.receipt.result_revision,
                    inserted_entry_ids: mutation.change.operations[0].inserted_entry_ids,
                    context_change_operation_id: mutation.receipt.id,
                    recorded_at: recordedAt,
                },
            },
        },
    });
}

/** Exact registered tool-result apply receipt contract. Its caller separately authenticates
 * the accepted source/job and deterministic proposal; constructing this data grants no authority.
 */
export async function toolResultTextApplyReceipt(
    source: ConversationRef,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    proposal: Extract<ProposalResult['proposal'], { kind: 'replace_with_compaction' }>,
    recordedAt: string,
): Promise<OperationReceipt> {
    parseToolResultTextStrategy(job);
    if (
        !isToolResultTextProcessor(job) ||
        resolution.job_id !== job.id ||
        resolution.selected_block_ids !== undefined ||
        resolution.selected_entries !== undefined ||
        proposal.strategy.id !== job.processor_id ||
        proposal.strategy.version !== job.processor_version ||
        proposal.strategy.configuration_fingerprint !== job.configuration_fingerprint ||
        job.configuration_fingerprint !== (await fingerprintJson(job.configuration)) ||
        proposal.compaction_id !== (await deriveConversationId('tool-result-text-compaction', job.id)) ||
        proposal.fidelity !== 'retrievable' ||
        proposal.accepted_asset_operation_id !== `processing:archive:${job.id}` ||
        proposal.generation_ids.length !== 0 ||
        proposal.derivation_generation !== undefined ||
        proposal.placement.mode !== 'per_selected_range' ||
        proposal.placement.causal_order !== 'preserved_disjoint_ranges' ||
        proposal.replacement_turns.length !== resolution.entry_ids.length ||
        resolution.source_turn_ids.length !== resolution.entry_ids.length ||
        proposal.replacement_turns.some(
            (turn, index) =>
                turn.provenance.type !== 'derived' ||
                turn.provenance.derivation_id !== proposal.compaction_id ||
                turn.provenance.source_hash !== resolution.source_fingerprint ||
                canonicalJsonContentString(turn.provenance.source_turn_ids) !==
                    canonicalJsonContentString([resolution.source_turn_ids[index]]),
        )
    )
        throw new Error('Tool-result apply receipt changed its registered proposal/source contract');
    const inserted = await Promise.all(
        resolution.entry_ids.map((id) => deriveConversationId('tool-result-text-entry', job.id, id)),
    );
    return OperationReceiptSchema.parse({
        id: `processing:apply:${job.id}`,
        conversation_id: source.conversation_id,
        base_revision: source.revision,
        result_revision: source.revision + 1,
        recorded_at: recordedAt,
        payload_fingerprint: await fingerprintJson(proposal),
        accepted_turn_ids: [],
        accepted_generation_ids: [],
        accepted_context_entry_ids: inserted,
        operation_kind: 'context_change',
        context_change: {
            kind: 'replace_with_compaction',
            removed_entry_ids: resolution.entry_ids,
            inserted_entry_ids: inserted,
            source_fingerprint: resolution.source_fingerprint,
            placement: proposal.placement,
        },
    });
}

/** Specialized deterministic delta; executable call/receipt and retained original records are never rewritten. */
export async function toolResultTextWorkingMutation(
    frame: ToolResultTextSelectionFrame,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    proposal: Extract<ProposalResult['proposal'], { kind: 'replace_with_compaction' }>,
    recordedAt: string,
): Promise<ContextMutationResult> {
    const replacements = new Map<string, ContextEntry>();
    if (proposal.replacement_turns.length !== resolution.entry_ids.length)
        throw new Error('Tool-result text projection lost its exact selected result entries');
    for (const [index, entryId] of resolution.entry_ids.entries())
        replacements.set(entryId, {
            id: await deriveConversationId('tool-result-text-entry', job.id, entryId),
            type: 'replacement_turn',
            turn_id: proposal.replacement_turns[index].id,
            compaction_id: proposal.compaction_id,
        });
    const operationId = `processing:apply:${job.id}`;
    const inserted = [...replacements.values()].map((item) => item.id);
    const receipt = await toolResultTextApplyReceipt(frame.source, job, resolution, proposal, recordedAt);
    const selectedOriginals = await toolResultTextWorkingSelection(frame, resolution.entry_ids, job);
    const newReferenceIds = new Set(
        await Promise.all(
            selectedOriginals.texts.map((text) =>
                deriveConversationId('tool-result-text-block', job.id, text.block_id),
            ),
        ),
    );
    const requirements = [];
    for (const turn of proposal.replacement_turns)
        for (const block of turn.blocks) {
            if (block.type !== 'tool_result') throw new Error('Tool-result projection changed its result block kind');
            for (const reference of block.content)
                if (reference.type === 'external_reference' && newReferenceIds.has(reference.id)) {
                    requirements.push({
                        id: await deriveConversationId('retrieval_requirement', operationId, reference.asset_id),
                        asset_id: reference.asset_id,
                        retrieval: reference.retrieval,
                        accepted_asset_operation_id: proposal.accepted_asset_operation_id,
                    });
                }
        }
    const entries = frame.context.entries.map((entry) => replacements.get(entry.id) ?? entry);
    const firstChanged = frame.context.entries.findIndex((entry) => replacements.has(entry.id));
    const cache = cacheAfterContextEdit(frame.context, entries, firstChanged < 0 ? undefined : firstChanged);
    return {
        context: ConversationContextSchema.parse({
            ...frame.context,
            revision: frame.context.revision + 1,
            entries,
            ...(cache ? { cache_intent: cache } : {}),
            retrieval_requirements: [...frame.context.retrieval_requirements, ...requirements],
        }),
        compaction: CompactionRecordSchema.parse({
            id: proposal.compaction_id,
            operation_id: operationId,
            strategy: proposal.strategy,
            source: {
                turn_ids: resolution.source_turn_ids,
                block_ids: proposal.replacement_turns.flatMap((turn) =>
                    turn.provenance.type === 'derived' ? (turn.provenance.source_block_ids ?? []) : [],
                ),
                source_fingerprint: resolution.source_fingerprint,
            },
            replacement_turns: proposal.replacement_turns,
            original_context: frame.context,
            ...toolResultSupersession(frame, resolution.entry_ids, job),
            fidelity: 'retrievable',
            retained_asset_ids: proposal.retained_asset_ids,
            generation_ids: [],
            created_at: recordedAt,
            metadata: {
                applied_revision: receipt.result_revision,
                source_context_revision: frame.context.revision,
                payload_fingerprint: receipt.payload_fingerprint,
            },
        }),
        change: ContextChangeSchema.parse({
            operation_id: operationId,
            conversation_id: frame.source.conversation_id,
            base_revision: frame.source.revision,
            result_revision: receipt.result_revision,
            operations: [
                {
                    kind: 'replace_with_compaction',
                    removed_entry_ids: resolution.entry_ids,
                    inserted_entry_ids: inserted,
                    source_fingerprint: resolution.source_fingerprint,
                    placement: proposal.placement,
                },
            ],
            diagnostics: [],
        }),
        receipt,
    };
}

/** Host dispatch retains the historical v1 implementation unchanged for ordinary text. */
export async function canonicalTextExternalizationArchiveInputs(document: ConversationDocument, job: ProcessingJob) {
    return isToolResultTextProcessor(job)
        ? toolResultExternalizationArchiveInputs(document, job)
        : textExternalizationArchiveInputs(document, job);
}
export function createCanonicalTextExternalizationProcessor(
    job: ProcessingJob,
    bind: TextExternalizationRetrievalBinder,
): ConversationProcessor {
    return isToolResultTextProcessor(job)
        ? createToolResultTextExternalizationProcessor(bind)
        : createTextExternalizationProcessor(bind);
}
export async function inspectCanonicalTextExternalizationJobSelection(
    document: ConversationDocument,
    job: ProcessingJob,
) {
    if (!isToolResultTextProcessor(job)) return inspectTextExternalizationJobSelection(document, job);
    const input = await toolResultExternalizationArchiveInputs(document, job);
    return input.texts.length
        ? { kind: 'eligible' as const, texts: input.texts }
        : { kind: 'no_eligible_blocks' as const };
}
