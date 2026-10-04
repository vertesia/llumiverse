import { canonicalJsonContentString, hashUtf8Content } from './content-integrity.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { cacheAfterEdit } from './conversation-edit-utils.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import type { ConversationProcessor, ProcessorResult } from './processing.js';
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
    ContextEntry,
    ConversationDocument,
    ConversationTurn,
    ExecutionReceipt,
    ExternalReferenceBlock,
    ProcessingJob,
    ProcessingOutputReceipt,
    ProcessingResolvedInput,
    RetrievalCapability,
    ToolResultBlock,
} from './types.js';
import { parseConversationDocument } from './validation.js';

/** Separate from v1 ordinary text processing: executable arguments are never selected or rewritten. */
export const TOOL_RESULT_TEXT_PROCESSOR_ID = 'externalize-tool-result-text';
export const TOOL_RESULT_TEXT_PROCESSOR_VERSION = '1';
const MAX_BYTES = 32 * 1024 * 1024;
const MAX_BLOCKS = 4096;
export function isToolResultTextProcessor(job: Pick<ProcessingJob, 'processor_id' | 'processor_version'>): boolean {
    return (
        job.processor_id === TOOL_RESULT_TEXT_PROCESSOR_ID &&
        job.processor_version === TOOL_RESULT_TEXT_PROCESSOR_VERSION
    );
}

interface SelectedResult {
    entry: ContextEntry;
    turn: ConversationTurn;
    result: ToolResultBlock;
    receipt: ExecutionReceipt;
    call_turn: ConversationTurn;
}
export interface ToolResultExternalizationText {
    entry_id: string;
    turn_id: string;
    result_block_id: string;
    block_id: string;
    text: string;
}

async function selectedResults(document: ConversationDocument, entryIds: readonly string[]): Promise<SelectedResult[]> {
    if (entryIds.length > MAX_BLOCKS || new Set(entryIds).size !== entryIds.length)
        throw new Error('Tool-result text selection exceeds its unique entry bound');
    const selected = document.context.entries.filter((entry) => entryIds.includes(entry.id));
    if (canonicalJsonContentString(selected.map((entry) => entry.id)) !== canonicalJsonContentString(entryIds))
        throw new Error('Tool-result text selection is not exact active context order');
    const records: SelectedResult[] = [];
    for (const entry of selected) {
        const turn = document.turns.find((item) => item.id === entry.turn_id);
        if (
            entry.type !== 'source_turn' ||
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
            receipt.result_turn_id !== turn.id ||
            receipt.call_id !== result.call_id ||
            result.status === 'unknown' ||
            receipt.status !== result.status ||
            !receipt.call_source ||
            receipt.metadata?.retrieval_excerpt !== undefined ||
            result.content.some((block) => block.type === 'native_replay')
        )
            throw new Error('Tool-result text selection lacks an eligible exact terminal application receipt');
        await assertToolResultReceiptFingerprint(result, receipt);
        const callSource = receipt.call_source;
        const callTurn = document.turns.find((item) => item.id === callSource.turn_id);
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
            callSource.conversation.conversation_id !== document.id ||
            callSource.conversation.revision > document.revision ||
            (await fingerprintJson(call)) !== callSource.call_fingerprint ||
            callEntries.length !== 1 ||
            callEntries.some((item) => document.context.protected_entry_ids.includes(item.id))
        )
            throw new Error('Tool-result text selection changed or protected its exact executed call dependency');
        // The executed call and its native replay remain active and byte-for-byte unchanged.
        // Replacing result content cannot silently invalidate replay in another active entry.
        const changedBlocks = new Set([result.id, ...result.content.map((block) => block.id)]);
        const turns = createContextTurnIndex(document);
        for (const active of document.context.entries) {
            const { blocks } = resolveContextEntry(turns, active);
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
        records.push({ entry, turn, result, receipt, call_turn: callTurn });
    }
    return records;
}

/** Eligibility is conservative: excluded results remain in context, without invoking a processor. */
export async function eligibleToolResultTextEntries(
    document: ConversationDocument,
    accepted: readonly string[],
): Promise<string[]> {
    const ids: string[] = [];
    for (const entry of document.context.entries) {
        if (!accepted.includes(entry.id)) continue;
        const turn = document.turns.find((item) => item.id === entry.turn_id);
        if (turn?.kind !== 'tool') continue;
        try {
            await selectedResults(document, [entry.id]);
            ids.push(entry.id);
        } catch {
            // Unresolved/protected/received/provider content never becomes executable projection authority.
        }
    }
    return ids;
}

export async function toolResultTextSelection(
    document: ConversationDocument,
    entryIds: readonly string[],
): Promise<{
    records: SelectedResult[];
    texts: ToolResultExternalizationText[];
    integrities: Awaited<ReturnType<typeof hashUtf8Content>>[];
    source_fingerprint: string;
}> {
    const records = await selectedResults(document, entryIds);
    const texts: ToolResultExternalizationText[] = [];
    for (const record of records)
        for (const block of record.result.content) {
            if (block.type === 'text')
                texts.push({
                    entry_id: record.entry.id,
                    turn_id: record.turn.id,
                    result_block_id: record.result.id,
                    block_id: block.id,
                    text: block.text,
                });
        }
    if (texts.length > MAX_BLOCKS) throw new RangeError('Tool-result text selection exceeds block bound');
    const integrities = await Promise.all(texts.map((item) => hashUtf8Content(item.text)));
    if (integrities.reduce((sum, item) => sum + item.byte_length, 0) > MAX_BYTES)
        throw new RangeError('Tool-result text selection exceeds byte bound');
    const sourceFingerprint = await fingerprintJson({
        kind: 'tool_result_text_selection',
        records: records.map((item) => ({
            entry: item.entry,
            turn: item.turn,
            receipt: item.receipt,
            call_turn: item.call_turn,
        })),
    });
    return { records, texts, integrities, source_fingerprint: sourceFingerprint };
}

export async function toolResultExternalizationArchiveInputs(document: ConversationDocument, job: ProcessingJob) {
    if (
        !isToolResultTextProcessor(job) ||
        job.scope !== 'on_append' ||
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
    const selected = await toolResultTextSelection(document, job.selection.entry_ids);
    return {
        texts: selected.texts,
        integrities: selected.integrities,
        payload_fingerprint: await fingerprintJson({
            kind: 'tool_result_text_archive',
            blocks: selected.texts.map(({ text: _text, ...item }, index) => ({
                ...item,
                ...selected.integrities[index],
            })),
        }),
    };
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
    if (!isToolResultTextProcessor(job) || Object.keys(job.configuration).length)
        throw new Error('Tool-result text processor configuration is unavailable');
    const selected = await toolResultTextSelection(document, resolution.entry_ids);
    if (selected.source_fingerprint !== resolution.source_fingerprint)
        throw new Error('Tool-result text processing lost its original call/result/receipt selection');
    const receipt = document.operation_receipts[`processing:archive:${job.id}`];
    const assetIds = receipt?.accepted_asset_ids ?? [];
    if (
        !receipt ||
        receipt.operation_kind !== undefined ||
        retrievals.length !== selected.texts.length ||
        assetIds.length !== selected.texts.length ||
        new Set(assetIds).size !== assetIds.length ||
        receipt.payload_fingerprint !==
            (await toolResultExternalizationArchiveInputs(document, job)).payload_fingerprint
    )
        throw new Error('Tool-result text processing requires its exact durably accepted ordered archive');
    const compactionId = await deriveConversationId('tool-result-text-compaction', job.id);
    const replacementTurns: ConversationTurn[] = [];
    let ordinal = 0;
    for (const record of selected.records) {
        const content = [];
        for (const block of record.result.content) {
            const id = await deriveConversationId('tool-result-text-block', job.id, block.id);
            if (block.type !== 'text') {
                content.push({ ...block, id });
                continue;
            }
            const asset = document.assets[assetIds[ordinal]];
            const exact = selected.integrities[ordinal];
            const retrieval = RetrievalCapabilitySchema.parse(retrievals[ordinal]);
            const definition = document.tool_definitions[retrieval.tool_definition_id ?? ''];
            if (
                asset?.kind !== 'text' ||
                asset.storage.type !== 'external' ||
                asset.content_hash !== exact.content_hash ||
                asset.byte_length !== exact.byte_length ||
                !definition ||
                !document.context.active_tool_definition_ids.includes(definition.id) ||
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
    return {
        kind: 'proposal',
        proposal: ContextChangeProposalSchema.parse({
            kind: 'replace_with_compaction',
            compaction_id: compactionId,
            strategy: {
                id: TOOL_RESULT_TEXT_PROCESSOR_ID,
                version: TOOL_RESULT_TEXT_PROCESSOR_VERSION,
                configuration_fingerprint: await fingerprintJson(job.configuration),
            },
            replacement_turns: replacementTurns,
            fidelity: 'retrievable',
            accepted_asset_operation_id: receipt.id,
            retained_asset_ids: assetIds,
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
    const references: ExternalReferenceBlock[] = [];
    for (const turn of proposal.replacement_turns)
        for (const block of turn.blocks) {
            if (block.type === 'tool_result')
                for (const nested of block.content)
                    if (nested.type === 'external_reference' && proposal.retained_asset_ids.includes(nested.asset_id))
                        references.push(nested);
        }
    const expected = await buildToolResultTextExternalizationProposal(
        document,
        job,
        resolution,
        references.map((item) => item.retrieval),
    );
    if (canonicalJsonContentString(expected.proposal) !== canonicalJsonContentString(proposal))
        throw new Error('Tool-result text output changed exact original/dependency/archive/projection bytes');
    const replacements = new Map<string, ContextEntry>();
    for (const [index, entryId] of resolution.entry_ids.entries())
        replacements.set(entryId, {
            id: await deriveConversationId('tool-result-text-entry', job.id, entryId),
            type: 'replacement_turn',
            turn_id: proposal.replacement_turns[index].id,
            compaction_id: proposal.compaction_id,
        });
    const operationId = `processing:apply:${job.id}`;
    const inserted = [...replacements.values()].map((item) => item.id);
    const receipt = {
        id: operationId,
        conversation_id: document.id,
        base_revision: document.revision,
        result_revision: document.revision + 1,
        recorded_at: recordedAt,
        payload_fingerprint: await fingerprintJson(proposal),
        accepted_turn_ids: [],
        accepted_generation_ids: [],
        accepted_context_entry_ids: inserted,
        operation_kind: 'context_change' as const,
        context_change: {
            kind: 'replace_with_compaction' as const,
            removed_entry_ids: resolution.entry_ids,
            inserted_entry_ids: inserted,
            source_fingerprint: resolution.source_fingerprint,
            placement: proposal.placement,
        },
    };
    const requirements = [];
    for (const reference of references)
        requirements.push({
            id: await deriveConversationId('retrieval_requirement', operationId, reference.asset_id),
            asset_id: reference.asset_id,
            retrieval: reference.retrieval,
            accepted_asset_operation_id: proposal.accepted_asset_operation_id,
        });
    const entries = document.context.entries.map((entry) => replacements.get(entry.id) ?? entry);
    const firstChanged = document.context.entries.findIndex((entry) => replacements.has(entry.id));
    const cache = cacheAfterEdit(document, entries, firstChanged < 0 ? undefined : firstChanged);
    const { coverage: _coverage, ...processing } = document.processing;
    return parseConversationDocument({
        ...document,
        revision: receipt.result_revision,
        updated_at: recordedAt,
        context: {
            ...document.context,
            revision: document.context.revision + 1,
            entries,
            ...(cache ? { cache_intent: cache } : {}),
            retrieval_requirements: [...document.context.retrieval_requirements, ...requirements],
        },
        compactions: {
            ...document.compactions,
            [proposal.compaction_id]: {
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
                fidelity: 'retrievable',
                retained_asset_ids: proposal.retained_asset_ids,
                generation_ids: [],
                created_at: recordedAt,
                metadata: {
                    applied_revision: receipt.result_revision,
                    source_context_revision: document.context.revision,
                    payload_fingerprint: receipt.payload_fingerprint,
                },
            },
        },
        operation_receipts: { ...document.operation_receipts, [operationId]: receipt },
        processing: {
            ...processing,
            completions: {
                ...processing.completions,
                [job.id]: {
                    job_id: job.id,
                    output_fingerprint: output.output_fingerprint,
                    status: 'applied',
                    result_revision: receipt.result_revision,
                    inserted_entry_ids: inserted,
                    context_change_operation_id: operationId,
                    recorded_at: recordedAt,
                },
            },
        },
    });
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
