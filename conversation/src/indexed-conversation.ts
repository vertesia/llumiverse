import { z } from 'zod';
import {
    assertSelectedCheckpointRetry,
    buildSelectedCheckpointRequest,
    IndexedCheckpointSummaryCommandSchema,
} from './checkpoint-context-change.js';
import {
    canonicalJsonContentBytes,
    canonicalJsonContentString,
    hashContentBytes,
    inlineAssetContentIntegrity,
} from './content-integrity.js';
import { applyContextMutationWorkingSet, type ContextMutationResult } from './context-change-transition.js';
import { collectAssetIds, planContextChangeWorkingSet } from './context-change-working-set.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { cacheAfterContextRemoval } from './conversation-edit-utils.js';
import { deletedContentIdentities } from './deleted-content-identities.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { INDEXED_EXCHANGE_PROCESSOR_ID, INDEXED_EXCHANGE_PROCESSOR_VERSION } from './indexed-exchange-constants.js';
import { applyIndexedExchangeOutput } from './indexed-exchange-processing.js';
import { createIndexedProcessingScratchStore } from './indexed-processing-scratch-store.js';
import {
    activeIndexedContextWorkingSet,
    applyIndexedTextExternalizationOutput,
    indexedCompletedJobEntrySelection,
    indexedPredecessorEntrySelection,
    indexedProcessingContextFingerprint,
    resolveIndexedProcessingTextInput,
} from './indexed-processing-working-set.js';
import { indexedToolResultTextFrame } from './indexed-tool-result-text.js';
import { DEFAULT_JSON_INPUT_LIMITS, preflightJsonInput } from './json-preflight.js';
import { createAcceptedOutputFragmentFromRecords } from './output.js';
import {
    buildPagedRecordIndex,
    getPagedRecord,
    getPagedRecords,
    insertPagedRecords,
    PAGED_RECORD_INDEX_MAX_BATCH_BYTES,
    PAGED_RECORD_INDEX_MAX_BATCH_KEYS,
    type PagedRecordIndexStore,
    type PagedRecordRef,
    PagedRecordRefSchema,
    type PagedRecordValue,
    putPagedRecord,
    readPagedRecordRange,
    removePagedRecord,
    scanPagedRecords,
} from './paged-record-index.js';
import { eligibleProcessingAppendRecords } from './processing-append-selection.js';
import { constructProcessingJobs, MAX_PROCESSING_STAGES_PER_OPERATION } from './processing-job-construction.js';
import { countUnresolvedProcessingJobs } from './processing-job-status.js';
import { recoverAcceptedProcessingPolicyCommand } from './processing-policy-recovery.js';
import { createProcessingTransitionReceipt } from './processing-transition-receipt.js';
import { renderContentBlockText } from './rendering.js';
import { MAX_PROCESSOR_CONFIGURATION_BYTES } from './runtime-constants.js';
import { ConversationDeleteChangeSchema } from './schemas/change.js';
import {
    ApplicationToolCallBlockSchema,
    AssetSchema,
    ContentBlockSchema,
    ConversationTurnSchema,
    GeneratedAgentTurnSchema,
    ToolDefinitionSchema,
} from './schemas/content.js';
import { ContextChangePlanInputSchema } from './schemas/context-change.js';
import { ContextEntrySchema } from './schemas/context-foundation.js';
import { ConversationDeleteOperationSchema } from './schemas/conversation-delete-operation.js';
import { ConversationContextSchema } from './schemas/document.js';
import {
    ExecutionReceiptSchema,
    GenerationSchema,
    OperationReceiptSchema,
    ToolCallSourceRefSchema,
} from './schemas/execution.js';
import {
    INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
    INDEXED_CONVERSATION_DELETE_PROFILE,
    INDEXED_CONVERSATION_DELETE_PROFILE_V2,
    INDEXED_CONVERSATION_PROCESSING_PROFILE,
    INDEXED_CONVERSATION_PROFILE,
    INDEXED_CONVERSATION_RESTART_PROFILE,
    INDEXED_CONVERSATION_ROOT_MAX_BYTES,
    INDEXED_PROCESSING_MAX_IO_BYTES,
    INDEXED_PROCESSING_MAX_PAGE_READS,
    INDEXED_PROCESSING_MAX_RECORD_READS,
    INDEXED_PROCESSING_SELECTED_MAX_BLOCKS,
    IndexedConversationCompactionHeaderSchema,
    IndexedConversationContextHeaderSchema,
    type IndexedConversationDeleteCommand,
    IndexedConversationDeleteCommandSchema,
    IndexedConversationDeletedTurnSchema,
    type IndexedConversationDirectories,
    IndexedConversationProcessingHeaderSchema,
    type IndexedConversationRoot,
    IndexedConversationRootSchema,
    IndexedConversationSelectedContextSchema,
    type IndexedConversationTurnHeader,
    IndexedConversationTurnHeaderSchema,
    IndexedConversationTurnLinkSchema,
    type IndexedProcessingCoverageCommand,
    IndexedProcessingCoverageCommandSchema,
    type IndexedProcessingPolicyCommand,
    IndexedProcessingPolicyCommandSchema,
    type IndexedProcessingQueueCommand,
    IndexedProcessingQueueCommandSchema,
    type IndexedProcessingReadinessCoverage,
    IndexedProcessingReadinessCoverageSchema,
    IndexedProcessingSelectedContextSchema,
} from './schemas/indexed-head.js';
import {
    IndexedProcessingArchiveAssetSchema,
    type IndexedProcessingClaimWorkspace,
    IndexedProcessingClaimWorkspaceSchema,
    IndexedProcessingPredecessorEvidenceSchema,
} from './schemas/indexed-processing.js';
import {
    IndexedProcessingClosureCommandSchema,
    IndexedProcessingClosureWitnessSchema,
} from './schemas/indexed-processing-closure.js';
import { IndexedRecordBatchCommandSchema } from './schemas/ingestion.js';
import { ConversationOutputReceiptSchema } from './schemas/output.js';
import { ProcessingPolicyCommandSchema } from './schemas/processing-policy.js';
import { ProcessingQueueAcceptanceInputSchema } from './schemas/processing-queue.js';
import {
    buildToolResultTextWorkingProposal,
    eligibleToolResultTextWorkingEntries,
    isToolResultTextProcessor,
    type ToolResultTextSelectionFrame,
    toolResultTextArchiveRetrievals,
    toolResultTextWorkingSelection,
} from './tool-result-text-externalization.js';
import {
    isToolResultTextStrategy,
    parseToolResultTextStrategy,
    supportsToolResultTextProcessingScope,
} from './tool-result-text-strategy.js';

export { IndexedRecordBatchCommandSchema } from './schemas/ingestion.js';

import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
} from './schemas/primitives.js';
import {
    ProcessingAttemptReceiptSchema,
    ProcessingCompletionReceiptSchema,
    ProcessingJobSchema,
    ProcessingOutputReceiptSchema,
    type ProcessingReadinessCoverageSchema,
    ProcessingResolvedInputSchema,
    ProcessingSupersessionReceiptSchema,
} from './schemas/processing.js';
import { ConversationToolExecutionResultSchema } from './schemas/tool-execution.js';
import { validateConversationSemantics, validateUsage } from './semantic-validation.js';
import { conversationDocumentFromJson } from './serialization.js';
import { validateToolExecutionResult } from './tool-execution.js';
import { assertToolResultReceiptFingerprint } from './tool-result-integrity.js';
import type {
    AppendConversationRecordsOptions,
    Asset,
    CompactionRecord,
    ContextEntry,
    ConversationContext,
    ConversationDocument,
    ConversationRecordBatch,
    ConversationRef,
    ConversationTurn,
    ExecutionReceipt,
    OperationReceipt,
    PendingApplicationToolCall,
    ProcessingCompletionReceipt,
    ProcessingJob,
    ProcessingOutputReceipt,
    ProcessingResolvedInput,
    ProcessorConfiguration,
    ToolDefinition,
    ToolResultBlock,
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

export const IndexedCallStateSchema = z.strictObject({
    call_id: z.string().min(1),
    turn_id: z.string().min(1),
    block_id: z.string().min(1),
    call_fingerprint: z.string().regex(/^sha256:[0-9a-f]{64}$/),
    result_block_id: z.string().min(1).optional(),
    terminal_receipt_id: z.string().min(1).optional(),
});
const IndexedOpenToolCallSchema = IndexedCallStateSchema.omit({ result_block_id: true, terminal_receipt_id: true });
type IndexedCallState = z.infer<typeof IndexedCallStateSchema>;

/** Semantic conflicts in one bounded append; storage/deadline failures retain their original errors. */
export class IndexedRecordAppendConflict extends Error {
    constructor(
        readonly code:
            | 'conversation_identity_conflict'
            | 'revision_conflict'
            | 'operation_conflict'
            | 'record_conflict',
        message: string,
    ) {
        super(message);
        this.name = 'IndexedRecordAppendConflict';
    }
}

export class IndexedRecordAppendValidationError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'IndexedRecordAppendValidationError';
    }
}

export type IndexedRecordBatchCommand = z.infer<typeof IndexedRecordBatchCommandSchema>;

export interface IndexedConversationRecordStore extends PagedRecordIndexStore {
    /** The host derives a run-scoped content-addressed key from kind/hash and owns the bytes. */
    readRecord(value: Extract<PagedRecordValue, { storage: 'record' }>): Promise<Uint8Array>;
    /** Immutable create-only write, with segmentation for bodies over 64KiB. */
    writeRecord(value: Extract<PagedRecordValue, { storage: 'record' }>, bytes: Uint8Array): Promise<void>;
    /** Host custody check for incoming external bytes, before immutable record staging or head publication. */
    assertExternalAssetIntegrity?(asset: Asset): Promise<void>;
    /** Service-owned selected-read verification; a caller receipt alone never proves bytes. */
    assertRetrievalExcerptIntegrity?(
        source: IndexedConversationRoot,
        result: ToolResultBlock,
        receipt: ExecutionReceipt,
    ): Promise<void>;
}

export interface StagedIndexedConversationRoot {
    root: IndexedConversationRoot;
    locator: PagedRecordRef;
}

type RecordValue = Extract<PagedRecordValue, { storage: 'record' }>;
type Entry = { key: string; value: PagedRecordValue };

/** Private indexed commit attestation. It is produced only after deterministic processing
 * verification, never through the caller-controlled processing phase command. Original records
 * remain independently authenticated point descriptors, without reading their cold bodies. */
export const IndexedToolResultOriginalSourceSchema = z.strictObject({
    version: z.literal(1),
    job_id: IdentifierSchema,
    job_fingerprint: ContentHashSchema,
    resolved_input_fingerprint: ContentHashSchema,
    source: z.discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('indexed_root'), root: PagedRecordRefSchema }),
        z.strictObject({ kind: z.literal('materialized_context'), context: ConversationContextSchema }),
    ]),
});

export const IndexedMaterializedProcessingQueueSchema = ProcessingQueueAcceptanceInputSchema.extend({
    version: z.literal(1),
    selected_entries: z.array(ContextEntrySchema).max(INDEXED_PROCESSING_SELECTED_MAX_BLOCKS),
    legacy_reconstructed: z.boolean(),
});

export const IndexedToolResultTerminalValidationSchema = z.strictObject({
    version: z.literal(1),
    terminal_execution_id: IdentifierSchema,
    result_turn_id: IdentifierSchema,
    job_id: IdentifierSchema,
    validation_record_fingerprint: ContentHashSchema,
});

const IndexedToolResultValidationSchema = z.strictObject({
    version: z.literal(1),
    job_id: IdentifierSchema,
    job_fingerprint: ContentHashSchema,
    original_source_record_fingerprint: ContentHashSchema,
    attempt_fingerprint: ContentHashSchema,
    output_fingerprint: ContentHashSchema,
    output_record_fingerprint: ContentHashSchema,
    proposal_fingerprint: ContentHashSchema,
    compaction_id: IdentifierSchema,
    replacement_turn_fingerprints: z.record(IdentifierSchema, ContentHashSchema),
    resolved_input_fingerprint: ContentHashSchema,
    completion_fingerprint: ContentHashSchema,
    compaction_fingerprint: ContentHashSchema,
    acceptance_fingerprint: ContentHashSchema,
    /** Only v3: current projection identity is separate from the immutable terminal original. */
    original_result_bindings: z
        .record(
            IdentifierSchema,
            z.strictObject({
                turn_id: IdentifierSchema,
                block_id: IdentifierSchema,
            }),
        )
        .optional(),
    dependencies: z.array(
        z.strictObject({
            family: z.enum([
                'turns',
                'blocks',
                'context_entries',
                'execution_receipts',
                'operation_receipts',
                'assets',
                'tool_definitions',
                'compactions',
            ]),
            descriptor: z.strictObject({
                storage: z.literal('record'),
                kind: z.string(),
                id: IdentifierSchema,
                content_hash: ContentHashSchema,
                size_bytes: NonnegativeSafeIntegerSchema,
            }),
        }),
    ),
});

/** Call only after validating the exact deterministic proposal against these complete originals. */
async function stageIndexedToolResultValidation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    directories: IndexedConversationDirectories,
    frame: ToolResultTextSelectionFrame,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    output: ProcessingOutputReceipt,
    completion: ProcessingCompletionReceipt,
    compaction: CompactionRecord,
    acceptance: OperationReceipt,
    deferTerminalRelations = false,
): Promise<void> {
    const dependencies: z.infer<typeof IndexedToolResultValidationSchema>['dependencies'] = [];
    const seen = new Set<string>();
    const bind = async (
        family: z.infer<typeof IndexedToolResultValidationSchema>['dependencies'][number]['family'],
        id: string,
        expected?: unknown,
    ) => {
        const key = tupleKey(family, id);
        if (seen.has(key)) return;
        seen.add(key);
        const descriptor = await getPagedRecord(store, root.directories[family], id);
        if (
            descriptor?.storage !== 'record' ||
            descriptor.id !== id ||
            descriptor.kind !== family ||
            (expected !== undefined && descriptor.content_hash !== (await fingerprintJson(expected)))
        )
            throw new Error('Indexed tool-result validation lost its exact original record binding');
        dependencies.push({ family, descriptor });
    };
    const selectedRecords =
        job.processor_version === '3'
            ? (await toolResultTextWorkingSelection(frame, resolution.entry_ids, job)).records
            : undefined;
    const originalResultBindings: Record<string, { turn_id: string; block_id: string }> = {};
    for (const [turnIndex, turnId] of resolution.source_turn_ids.entries()) {
        const turn = frame.turns.get(turnId);
        if (turn?.kind !== 'tool' || !turn.execution_id)
            throw new Error('Indexed tool-result validation lacks a complete original result');
        const receipt = frame.execution_receipts[turn.execution_id];
        if (!receipt?.call_source) throw new Error('Indexed tool-result validation lacks its original call source');
        await bind('execution_receipts', receipt.id, receipt);
        const selectedRecord = selectedRecords?.[turnIndex];
        const predecessor = selectedRecord?.projection_witness;
        if (selectedRecords) {
            const result = turn.blocks[0];
            if (result?.type !== 'tool_result' || !receipt.result_turn_id || selectedRecord?.turn.id !== turn.id)
                throw new Error('Indexed chained validation lost its current selected result');
            originalResultBindings[
                output.kind === 'proposal' && output.proposal.kind === 'replace_with_compaction'
                    ? output.proposal.replacement_turns[turnIndex].id
                    : ''
            ] = {
                turn_id: predecessor?.original_result_turn_id ?? turn.id,
                block_id: predecessor?.original_result_block_id ?? result.id,
            };
        }
        if (predecessor) {
            const prior = await indexedRecordById(
                store,
                root,
                'compactions',
                predecessor.compaction_id,
                IndexedConversationCompactionHeaderSchema,
            );
            if (
                !prior ||
                (await fingerprintJson(prior)) !== predecessor.compaction_fingerprint ||
                !prior.operation_id.startsWith('processing:apply:')
            )
                throw new Error('Indexed chained validation changed its accepted predecessor compaction');
            const priorAcceptance = await indexedRecordById(
                store,
                root,
                'operation_receipts',
                prior.operation_id,
                OperationReceiptSchema,
            );
            if (!priorAcceptance || priorAcceptance.result_revision > resolution.source_revision)
                throw new Error('Indexed chained validation predecessor is newer than its selected source');
            const priorJobId = prior.operation_id.slice('processing:apply:'.length);
            await auditIndexedToolResultValidation(store, root, priorJobId);
            const priorValidation = await indexedProcessingRecord(
                store,
                root,
                'tool_result_validations',
                priorJobId,
                IndexedToolResultValidationSchema,
            );
            if (
                !priorValidation ||
                priorValidation.replacement_turn_fingerprints[turn.id] !== predecessor.projection_fingerprint
            )
                throw new Error('Indexed chained validation lost its accepted predecessor projection');
            await bind('compactions', prior.id, prior);
            for (const [family, id] of [
                ['turns', predecessor.original_result_turn_id],
                ['blocks', predecessor.original_result_block_id],
            ] as const) {
                const bound = priorValidation.dependencies.find(
                    (item) => item.family === family && item.descriptor.id === id,
                );
                const actual = await getPagedRecord(store, root.directories[family], id);
                if (!bound || !sameIndexedRecord(actual, bound.descriptor))
                    throw new Error('Indexed chained validation lost its immutable original terminal descriptors');
                await bind(family, id);
            }
        }
        for (const id of [turn.id, receipt.call_source.turn_id]) {
            const original = frame.turns.get(id);
            if (!original) throw new Error('Indexed tool-result validation lacks a complete original turn');
            const { blocks, ...header } = original;
            const blockIds = blocks.map((block) => block.id);
            await bind('turns', id, {
                turn: header,
                source: id === turn.id && predecessor ? 'replacement' : 'ordinary',
                ...(id === turn.id && predecessor ? { compaction_id: predecessor.compaction_id } : {}),
                block_ids: blockIds,
                block_ids_hash: await fingerprintJson(blockIds),
            });
            for (const block of blocks) await bind('blocks', block.id, block);
            for (const assetId of collectAssetIds(blocks)) await bind('assets', assetId);
        }
    }
    for (const entryId of resolution.entry_ids) {
        const entry = frame.context.entries.find((item) => item.id === entryId);
        if (!entry) throw new Error('Indexed tool-result validation lacks its original context entry');
        await bind('context_entries', entryId, entry);
    }
    const archiveId = `processing:archive:${job.id}`;
    const archive = await indexedRecordById(store, root, 'operation_receipts', archiveId, OperationReceiptSchema);
    if (!archive) throw new Error('Indexed tool-result validation lacks its archive acceptance');
    await bind('operation_receipts', archiveId, archive);
    for (const assetId of archive.accepted_asset_ids ?? []) await bind('assets', assetId);
    for (const definitionId of frame.context.active_tool_definition_ids) await bind('tool_definitions', definitionId);
    if (output.kind !== 'proposal' || output.proposal.kind !== 'replace_with_compaction')
        throw new Error('Indexed tool-result validation lacks its verified deterministic output');
    const replacementTurnFingerprints = Object.fromEntries(
        await Promise.all(
            output.proposal.replacement_turns.map(async (turn) => [turn.id, await fingerprintJson(turn)] as const),
        ),
    );
    const { replacement_turns: _turns, original_context: _originalContext, ...compactionHeader } = compaction;
    const originalSource = await getPagedRecord(
        store,
        directories.processing_records,
        tupleKey('tool_result_sources', job.id),
    );
    if (
        originalSource?.storage !== 'record' ||
        originalSource.kind !== 'processing_records' ||
        originalSource.id !== job.id
    )
        throw new Error('Indexed tool-result validation lacks its authenticated original source descriptor');
    const validation = IndexedToolResultValidationSchema.parse({
        version: 1,
        job_id: job.id,
        job_fingerprint: await fingerprintJson(job),
        original_source_record_fingerprint: originalSource.content_hash,
        attempt_fingerprint: await fingerprintJson(
            await indexedProcessingRecord(store, root, 'attempts', job.id, ProcessingAttemptReceiptSchema),
        ),
        output_fingerprint: output.output_fingerprint,
        output_record_fingerprint: await fingerprintJson(output),
        proposal_fingerprint: await fingerprintJson(output.proposal),
        compaction_id: compaction.id,
        replacement_turn_fingerprints: replacementTurnFingerprints,
        resolved_input_fingerprint: await fingerprintJson(resolution),
        completion_fingerprint: await fingerprintJson(completion),
        compaction_fingerprint: await fingerprintJson(compactionHeader),
        acceptance_fingerprint: await fingerprintJson(acceptance),
        ...(job.processor_version === '3' ? { original_result_bindings: originalResultBindings } : {}),
        dependencies,
    });
    const key = tupleKey('tool_result_validations', job.id);
    const existing = await getPagedRecord(store, directories.processing_records, key);
    if (existing) {
        const prior = await loadRecord(store, existing, IndexedToolResultValidationSchema);
        if (canonicalJsonContentString(prior) !== canonicalJsonContentString(validation))
            throw new Error('Indexed tool-result validation differs from verified original records');
        if (!deferTerminalRelations)
            await stageIndexedToolResultTerminalValidation(store, { ...root, directories }, directories, job.id);
        return;
    }
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        key,
        await stageRecord(store, 'processing_records', job.id, validation),
    );
    if (!deferTerminalRelations)
        await stageIndexedToolResultTerminalValidation(store, { ...root, directories }, directories, job.id);
}

/** Terminal identity chooses only an accepted deterministic witness, never a different execution result. */
export async function auditIndexedToolResultTerminalValidation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    terminalId: string,
): Promise<string | undefined> {
    const marker = await indexedProcessingRecord(
        store,
        root,
        'tool_result_validation_by_terminal',
        terminalId,
        IndexedToolResultTerminalValidationSchema,
    );
    if (!marker) return undefined;
    const descriptor = await getPagedRecord(
        store,
        root.directories.processing_records,
        tupleKey('tool_result_validations', marker.job_id),
    );
    const validation = await indexedProcessingRecord(
        store,
        root,
        'tool_result_validations',
        marker.job_id,
        IndexedToolResultValidationSchema,
    );
    const terminal = await indexedRecordById(store, root, 'execution_receipts', terminalId, ExecutionReceiptSchema);
    if (
        marker.terminal_execution_id !== terminalId ||
        terminal?.id !== terminalId ||
        terminal.result_turn_id !== marker.result_turn_id ||
        descriptor?.storage !== 'record' ||
        descriptor.content_hash !== marker.validation_record_fingerprint ||
        validation?.job_id !== marker.job_id ||
        !validation.dependencies.some(
            (dependency) => dependency.family === 'execution_receipts' && dependency.descriptor.id === terminalId,
        ) ||
        !validation.dependencies.some(
            (dependency) => dependency.family === 'turns' && dependency.descriptor.id === marker.result_turn_id,
        )
    )
        throw new Error('Indexed terminal validation relation changed its exact accepted execution/source');
    await auditIndexedToolResultValidation(store, root, marker.job_id);
    return marker.job_id;
}

async function stageIndexedToolResultTerminalValidation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    directories: IndexedConversationDirectories,
    jobId: string,
): Promise<void> {
    const descriptor = await getPagedRecord(
        store,
        directories.processing_records,
        tupleKey('tool_result_validations', jobId),
    );
    if (descriptor?.storage !== 'record') throw new Error('Indexed terminal mapping lacks its accepted validation');
    const validation = await loadRecord(store, descriptor, IndexedToolResultValidationSchema);
    for (const dependency of validation.dependencies) {
        if (dependency.family !== 'execution_receipts') continue;
        const receipt = await indexedRecordById(
            store,
            root,
            'execution_receipts',
            dependency.descriptor.id,
            ExecutionReceiptSchema,
        );
        if (
            !receipt?.result_turn_id ||
            !validation.dependencies.some(
                (item) => item.family === 'turns' && item.descriptor.id === receipt.result_turn_id,
            )
        )
            throw new Error('Indexed terminal mapping lost its exact original execution/result');
        const key = tupleKey('tool_result_validation_by_terminal', receipt.id);
        const existing = await getPagedRecord(store, directories.processing_records, key);
        if (existing) {
            await auditIndexedToolResultTerminalValidation(store, { ...root, directories }, receipt.id);
            continue;
        }
        const marker = IndexedToolResultTerminalValidationSchema.parse({
            version: 1,
            terminal_execution_id: receipt.id,
            result_turn_id: receipt.result_turn_id,
            job_id: jobId,
            validation_record_fingerprint: descriptor.content_hash,
        });
        directories.processing_records = await putPagedRecord(
            store,
            directories.processing_records,
            key,
            await stageRecord(store, 'processing_records', receipt.id, marker),
        );
    }
}

async function stageIndexedToolResultOriginalSource(
    store: IndexedConversationRecordStore,
    directories: IndexedConversationDirectories,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    source: z.infer<typeof IndexedToolResultOriginalSourceSchema>['source'],
): Promise<void> {
    const original = IndexedToolResultOriginalSourceSchema.parse({
        version: 1,
        job_id: job.id,
        job_fingerprint: await fingerprintJson(job),
        resolved_input_fingerprint: await fingerprintJson(resolution),
        source,
    });
    const key = tupleKey('tool_result_sources', job.id);
    const existing = await getPagedRecord(store, directories.processing_records, key);
    if (existing) {
        if (!sameIndexedRecord(await loadRecord(store, existing, IndexedToolResultOriginalSourceSchema), original))
            throw new Error('Indexed original tool-result source changed its immutable accepted binding');
        return;
    }
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        key,
        await stageRecord(store, 'processing_records', job.id, original),
    );
}

/** Historical source evidence is never replaced by today's active selection. */
export class IndexedToolResultOriginalSourceUnavailableError extends Error {
    constructor(jobId: string) {
        super(`Tool-result compaction ${jobId} requires its authenticated original context for migration/recovery`);
        this.name = 'IndexedToolResultOriginalSourceUnavailableError';
    }
}

export type ToolResultOriginalDocumentResolver = (source: {
    conversation_id: string;
    revision: number;
}) => Promise<ConversationDocument>;

/** Audit an existing immutable commit witness without original-body replay. Used by the
 * explicit historical upgrade, which independently audits canonical record/operation coverage. */
export async function auditIndexedToolResultValidation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    jobId: string,
): Promise<void> {
    const validation = await indexedProcessingRecord(
        store,
        root,
        'tool_result_validations',
        jobId,
        IndexedToolResultValidationSchema,
    );
    const job = await indexedProcessingRecord(store, root, 'jobs', jobId, ProcessingJobSchema);
    const resolution = await indexedProcessingRecord(
        store,
        root,
        'resolved_inputs',
        jobId,
        ProcessingResolvedInputSchema,
    );
    const attempt = await indexedProcessingRecord(store, root, 'attempts', jobId, ProcessingAttemptReceiptSchema);
    const output = await getPagedRecord(store, root.directories.processing_records, tupleKey('outputs', jobId));
    const originalSource = await getPagedRecord(
        store,
        root.directories.processing_records,
        tupleKey('tool_result_sources', jobId),
    );
    const completion = await indexedProcessingRecord(
        store,
        root,
        'completions',
        jobId,
        ProcessingCompletionReceiptSchema,
    );
    if (
        !validation ||
        originalSource?.storage !== 'record' ||
        originalSource.kind !== 'processing_records' ||
        originalSource.id !== jobId ||
        !job ||
        !isToolResultTextProcessor(job) ||
        !resolution ||
        !attempt ||
        output?.storage !== 'record' ||
        output.kind !== 'processing_records' ||
        output.id !== jobId ||
        completion?.status !== 'applied' ||
        !completion.context_change_operation_id
    )
        throw new Error('Indexed validation audit lacks its exact complete processing lineage');
    const acceptance = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        completion.context_change_operation_id,
        OperationReceiptSchema,
    );
    const compaction = await indexedRecordById(
        store,
        root,
        'compactions',
        validation.compaction_id,
        IndexedConversationCompactionHeaderSchema,
    );
    if (
        !acceptance ||
        !compaction ||
        validation.job_id !== jobId ||
        validation.original_source_record_fingerprint !== originalSource.content_hash ||
        validation.job_fingerprint !== (await fingerprintJson(job)) ||
        validation.attempt_fingerprint !== (await fingerprintJson(attempt)) ||
        validation.output_record_fingerprint !== output.content_hash ||
        validation.output_fingerprint !== completion.output_fingerprint ||
        validation.proposal_fingerprint !== acceptance.payload_fingerprint ||
        validation.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
        validation.completion_fingerprint !== (await fingerprintJson(completion)) ||
        validation.compaction_fingerprint !== (await fingerprintJson(compaction)) ||
        validation.acceptance_fingerprint !== (await fingerprintJson(acceptance))
    )
        throw new Error('Indexed validation audit changed its exact committed processing witness');
    if (job.processor_version === '3') {
        const bindings = validation.original_result_bindings;
        if (
            !bindings ||
            !sameIndexedRecord(
                Object.keys(bindings).sort(),
                Object.keys(validation.replacement_turn_fingerprints).sort(),
            ) ||
            Object.values(bindings).some(
                (binding) =>
                    !validation.dependencies.some(
                        (dependency) => dependency.family === 'turns' && dependency.descriptor.id === binding.turn_id,
                    ) ||
                    !validation.dependencies.some(
                        (dependency) => dependency.family === 'blocks' && dependency.descriptor.id === binding.block_id,
                    ),
            )
        )
            throw new Error('Indexed chained validation lost its distinct immutable original result bindings');
    } else if (validation.original_result_bindings !== undefined)
        throw new Error('Indexed original tool-result validation has unexpected chained bindings');
    for (const dependency of validation.dependencies) {
        const actual = await getPagedRecord(store, root.directories[dependency.family], dependency.descriptor.id);
        if (
            actual === undefined ||
            canonicalJsonContentString(actual) !== canonicalJsonContentString(dependency.descriptor)
        )
            throw new Error('Indexed validation audit changed its immutable original record descriptor');
    }
}

/** One accepted predecessor projection. It never resolves the original result/output body. */
async function indexedToolResultProjectionWitness(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    compactionId: string,
    turnId: string,
    terminal: ExecutionReceipt,
) {
    const compaction = await indexedRecordById(
        store,
        root,
        'compactions',
        compactionId,
        IndexedConversationCompactionHeaderSchema,
    );
    if (!compaction?.operation_id.startsWith('processing:apply:') || !terminal.result_turn_id)
        throw new Error('Indexed chained source lacks its registered predecessor and terminal');
    const jobId = compaction.operation_id.slice('processing:apply:'.length);
    await auditIndexedToolResultValidation(store, root, jobId);
    const validation = await indexedProcessingRecord(
        store,
        root,
        'tool_result_validations',
        jobId,
        IndexedToolResultValidationSchema,
    );
    const binding = validation?.original_result_bindings?.[turnId];
    const originalBlock =
        binding?.block_id ??
        validation?.dependencies.find(
            (item) => item.family === 'blocks' && item.descriptor.content_hash === terminal.result_fingerprint,
        )?.descriptor.id;
    if (
        !validation?.replacement_turn_fingerprints[turnId] ||
        !originalBlock ||
        !validation.dependencies.some(
            (dependency) => dependency.family === 'turns' && dependency.descriptor.id === terminal.result_turn_id,
        ) ||
        (binding && binding.turn_id !== terminal.result_turn_id)
    )
        throw new Error('Indexed chained source changed its immutable original terminal identity');
    return {
        compaction_id: compaction.id,
        compaction_fingerprint: await fingerprintJson(compaction),
        projection_fingerprint: validation.replacement_turn_fingerprints[turnId],
        terminal_execution_id: terminal.id,
        original_result_turn_id: terminal.result_turn_id,
        original_result_block_id: originalBlock,
    };
}

/** Explicit upgrade only: repair/audit one job's point terminal relations after all retained
 * missing validations have been regenerated. No original result/output bodies are read here.
 */
export async function recoverIndexedToolResultTerminalRelations(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    jobId: string,
): Promise<IndexedConversationDirectories> {
    const directories = { ...root.directories };
    await auditIndexedToolResultValidation(store, root, jobId);
    await stageIndexedToolResultTerminalValidation(store, { ...root, directories }, directories, jobId);
    return directories;
}

/** Explicit upgrade/recovery only. Rebuild active tool-result validation from complete
 * retained originals under the caller's recovery IO budget; ordinary preparation never does this. */
export async function recoverIndexedToolResultValidations(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    resolveOriginalRoot?: (source: { conversation_id: string; revision: number }) => Promise<PagedRecordRef>,
    retainedCompactionIds?: readonly string[],
    stagingOptions: { defer_terminal_relations?: boolean } = {},
): Promise<IndexedConversationDirectories> {
    const directories = { ...root.directories };
    const context = retainedCompactionIds === undefined ? await loadIndexedActiveContext(store, root) : undefined;
    // Explicit upgrade obligations may include inactive retained compactions. Their IDs only
    // nominate work: the exact accepted source, job, output and replacement are still verified.
    const ids = retainedCompactionIds ?? [
        ...new Set(
            (context?.entries ?? []).flatMap((entry) =>
                entry.type === 'replacement_turn' ? [entry.compaction_id] : [],
            ),
        ),
    ];
    for (const compactionId of ids) {
        const header = await indexedRecordById(
            store,
            root,
            'compactions',
            compactionId,
            IndexedConversationCompactionHeaderSchema,
        );
        if (!header || !isToolResultTextStrategy(header.strategy.id, header.strategy.version)) continue;
        // The genuine accepted replacement header carries its exact operation witness.
        const acceptance = await indexedRecordById(
            store,
            root,
            'operation_receipts',
            header.operation_id,
            OperationReceiptSchema,
        );
        if (!acceptance?.id.startsWith('processing:apply:'))
            throw new Error('Indexed recovery compaction lacks registered processing acceptance');
        const jobId = acceptance.id.slice('processing:apply:'.length);
        if (await getPagedRecord(store, directories.processing_records, tupleKey('tool_result_validations', jobId))) {
            await auditIndexedToolResultValidation(store, { ...root, directories }, jobId);
            if (stagingOptions.defer_terminal_relations !== true)
                await stageIndexedToolResultTerminalValidation(store, { ...root, directories }, directories, jobId);
            continue;
        }
        const job = await indexedProcessingRecord(store, root, 'jobs', jobId, ProcessingJobSchema);
        const resolution = await indexedProcessingRecord(
            store,
            root,
            'resolved_inputs',
            jobId,
            ProcessingResolvedInputSchema,
        );
        const output = await indexedProcessingRecord(store, root, 'outputs', jobId, ProcessingOutputReceiptSchema);
        const completion = await indexedProcessingRecord(
            store,
            root,
            'completions',
            jobId,
            ProcessingCompletionReceiptSchema,
        );
        const archive = await indexedRecordById(
            store,
            root,
            'operation_receipts',
            `processing:archive:${jobId}`,
            OperationReceiptSchema,
        );
        if (
            !job ||
            !isToolResultTextProcessor(job) ||
            !resolution ||
            output?.kind !== 'proposal' ||
            output.proposal.kind !== 'replace_with_compaction' ||
            completion?.status !== 'applied' ||
            !archive ||
            output.proposal.compaction_id !== compactionId ||
            completion.context_change_operation_id !== acceptance.id
        )
            throw new Error('Indexed recovery lacks its complete original processing lineage');
        const attempt = await indexedProcessingRecord(store, root, 'attempts', jobId, ProcessingAttemptReceiptSchema);
        if (
            job.id !== jobId ||
            job.processor_version !== header.strategy.version ||
            job.configuration_fingerprint !== header.strategy.configuration_fingerprint ||
            (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
            (await fingerprintJson(job.selection)) !== job.selection_fingerprint ||
            resolution.job_id !== jobId ||
            resolution.source_fingerprint !== header.source.source_fingerprint ||
            !attempt ||
            attempt.job_id !== jobId ||
            attempt.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
            output.job_id !== jobId ||
            output.attempt_token !== attempt.attempt_token ||
            output.resolved_input_fingerprint !== attempt.resolved_input_fingerprint ||
            output.output_fingerprint !==
                (await fingerprintJson((({ output_fingerprint: _hash, ...payload }) => payload)(output))) ||
            acceptance.operation_kind !== 'context_change' ||
            acceptance.conversation_id !== root.source.conversation_id ||
            acceptance.result_revision > root.source.revision ||
            acceptance.payload_fingerprint !== (await fingerprintJson(output.proposal)) ||
            completion.job_id !== jobId ||
            completion.output_fingerprint !== output.output_fingerprint ||
            completion.result_revision !== acceptance.result_revision ||
            canonicalJsonContentString(completion.inserted_entry_ids) !==
                canonicalJsonContentString(acceptance.accepted_context_entry_ids ?? []) ||
            canonicalJsonContentString(resolution.entry_ids) !==
                canonicalJsonContentString(acceptance.context_change?.removed_entry_ids)
        )
            throw new Error('Indexed recovery changed its original immutable processing evidence');
        let originalSource = await indexedProcessingRecord(
            store,
            root,
            'tool_result_sources',
            jobId,
            IndexedToolResultOriginalSourceSchema,
        );
        if (!originalSource) {
            if (!resolveOriginalRoot) throw new IndexedToolResultOriginalSourceUnavailableError(jobId);
            const originalLocator = PagedRecordRefSchema.parse(
                await resolveOriginalRoot({
                    conversation_id: root.source.conversation_id,
                    revision: resolution.source_revision,
                }),
            );
            originalSource = IndexedToolResultOriginalSourceSchema.parse({
                version: 1,
                job_id: jobId,
                job_fingerprint: await fingerprintJson(job),
                resolved_input_fingerprint: await fingerprintJson(resolution),
                source: { kind: 'indexed_root', root: originalLocator },
            });
        }
        if (
            originalSource.job_id !== jobId ||
            originalSource.job_fingerprint !== (await fingerprintJson(job)) ||
            originalSource.resolved_input_fingerprint !== (await fingerprintJson(resolution))
        )
            throw new Error('Indexed recovery changed its authenticated original source binding');
        let frame: ToolResultTextSelectionFrame;
        let definitions: Record<string, ToolDefinition>;
        if (originalSource.source.kind === 'indexed_root') {
            const originalLocator = originalSource.source.root;
            const originalRoot = await loadRecord(
                store,
                {
                    storage: 'record',
                    kind: 'root',
                    id: root.source.conversation_id,
                    ...originalLocator,
                },
                IndexedConversationRootSchema,
            );
            const originalJob = await indexedProcessingRecord(store, originalRoot, 'jobs', jobId, ProcessingJobSchema);
            if (
                originalRoot.source.conversation_id !== root.source.conversation_id ||
                originalRoot.source.revision < resolution.source_revision ||
                originalRoot.source.revision > acceptance.base_revision ||
                !sameIndexedRecord(originalJob, job)
            )
                throw new Error('Indexed recovery original root is foreign to its accepted job');
            const originalSelected = await loadIndexedProcessingToolResultSelectedContext(
                store,
                originalRoot,
                originalLocator,
            );
            if (
                originalSelected.context.revision !== resolution.context_revision ||
                (await indexedProcessingContextFingerprint(originalSelected)) !== resolution.context_fingerprint
            )
                throw new Error('Indexed recovery original root changed its accepted context closure');
            frame = await indexedToolResultTextFrame(
                originalSelected,
                activeIndexedContextWorkingSet(originalSelected),
            );
            definitions = originalSelected.tool_definitions;
        } else {
            const originalContext = originalSource.source.context;
            const turns = new Map<string, ConversationTurn>();
            for (const item of originalContext.entries) {
                const projected = await loadIndexedProjectedTurn(store, root, item.turn_id);
                if (projected.completeness !== 'full_turn')
                    throw new Error('Indexed migration recovery lacks its complete original active turn');
                turns.set(
                    item.turn_id,
                    ConversationTurnSchema.parse({
                        ...projected.header,
                        blocks: projected.selected_blocks,
                    }),
                );
            }
            if (
                originalContext.revision !== resolution.context_revision ||
                (await fingerprintJson({
                    context: originalContext,
                    entries: originalContext.entries.map((item) => ({ entry: item, turn: turns.get(item.turn_id) })),
                })) !== resolution.context_fingerprint
            )
                throw new Error('Indexed migration recovery changed its accepted original context closure');
            const receipts: Record<string, ExecutionReceipt> = {};
            for (const id of resolution.source_turn_ids) {
                const turn = turns.get(id);
                if (turn?.kind !== 'tool' || !turn.execution_id)
                    throw new Error('Indexed recovery original result is unavailable');
                const receipt = await indexedRecordById(
                    store,
                    root,
                    'execution_receipts',
                    turn.execution_id,
                    ExecutionReceiptSchema,
                );
                if (!receipt?.call_source) throw new Error('Indexed recovery lacks its exact original call source');
                receipts[receipt.id] = receipt;
            }
            const projectionWitnesses: Record<
                string,
                Awaited<ReturnType<typeof indexedToolResultProjectionWitness>>
            > = {};
            for (const entry of originalContext.entries) {
                if (entry.type !== 'replacement_turn' || !resolution.entry_ids.includes(entry.id)) continue;
                const turn = turns.get(entry.turn_id);
                const terminal = turn?.execution_id ? receipts[turn.execution_id] : undefined;
                if (!terminal) throw new Error('Indexed chained recovery lacks its immutable terminal receipt');
                projectionWitnesses[entry.turn_id] = await indexedToolResultProjectionWitness(
                    store,
                    root,
                    entry.compaction_id,
                    entry.turn_id,
                    terminal,
                );
            }
            frame = {
                source: { conversation_id: root.source.conversation_id, revision: acceptance.base_revision },
                projection_witnesses: projectionWitnesses,
                context: originalContext,
                turns,
                active_blocks: new Map(
                    originalContext.entries.map((item) => [item.id, resolveContextEntry(turns, item).blocks]),
                ),
                execution_receipts: receipts,
            };
            definitions = {};
            for (const id of originalContext.active_tool_definition_ids) {
                const definition = await indexedRecordById(store, root, 'tool_definitions', id, ToolDefinitionSchema);
                if (!definition) throw new Error('Indexed recovery lost its original active tool definition');
                definitions[id] = definition;
            }
        }
        // Older indexes omitted removed entries. Restore only the authenticated historical
        // entries held by the exact source closure, never values nominated by today's context.
        for (const id of resolution.entry_ids) {
            const original = frame.context.entries.find((item) => item.id === id);
            if (!original) throw new Error('Indexed recovery lost its accepted original entry');
            const existing = await getPagedRecord(store, directories.context_entries, id);
            if (existing) {
                if (!sameIndexedRecord(await loadRecord(store, existing, ContextEntrySchema), original))
                    throw new Error('Indexed recovery changed its retained original entry');
            } else
                directories.context_entries = await putPagedRecord(
                    store,
                    directories.context_entries,
                    id,
                    await stageRecord(store, 'context_entries', id, original),
                );
        }
        const assets: Record<string, Asset> = {};
        for (const id of new Set([
            ...(archive.accepted_asset_ids ?? []),
            ...[...frame.turns.values()].flatMap((turn) => [...collectAssetIds(turn.blocks)]),
        ])) {
            const asset = await indexedRecordById(store, root, 'assets', id, AssetSchema);
            if (!asset) throw new Error('Indexed recovery original asset is absent');
            assets[id] = asset;
        }
        const indexedArchive = await fingerprintJson((archive.accepted_asset_ids ?? []).map((id) => assets[id]));
        const rebuilt = await buildToolResultTextWorkingProposal(
            frame,
            assets,
            definitions,
            archive,
            archive.payload_fingerprint === indexedArchive ? 'indexed_assets' : 'tool_result_text',
            job,
            resolution,
            toolResultTextArchiveRetrievals(output.proposal, archive),
        );
        if (canonicalJsonContentString(rebuilt.proposal) !== canonicalJsonContentString(output.proposal))
            throw new Error('Indexed recovery changed its deterministic original tool-result transformation');
        const replacementTurns = await Promise.all(
            output.proposal.replacement_turns.map(async (turn) => {
                const projected = await loadIndexedProjectedTurn(store, root, turn.id);
                return ConversationTurnSchema.parse({ ...projected.header, blocks: projected.selected_blocks });
            }),
        );
        const compaction = { ...header, replacement_turns: replacementTurns };
        if (
            canonicalJsonContentString(replacementTurns) !==
            canonicalJsonContentString(output.proposal.replacement_turns)
        )
            throw new Error('Indexed recovery changed its accepted context projection');
        await stageIndexedToolResultOriginalSource(store, directories, job, resolution, originalSource.source);
        await stageIndexedToolResultValidation(
            store,
            { ...root, directories },
            directories,
            frame,
            job,
            resolution,
            output,
            completion,
            compaction,
            acceptance,
            stagingOptions.defer_terminal_relations === true,
        );
    }
    return directories;
}

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

export async function stageRecord(
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

export async function loadRecord<Shape extends z.ZodType>(
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

export const IndexedProcessingOperationJobsSchema = z.strictObject({
    version: z.literal(1),
    operation_id: IdentifierSchema,
    receipt_fingerprint: ContentHashSchema,
    job_ids: z.array(IdentifierSchema).max(16),
});

export async function indexedCoverageIdentity(
    coverage: z.infer<typeof ProcessingReadinessCoverageSchema>,
): Promise<string> {
    return fingerprintJson({
        context_fingerprint: coverage.context_fingerprint,
        policy_revision: coverage.policy_revision,
        target_fingerprint: coverage.target_fingerprint,
        measurement: coverage.measurement,
        required_job_ids: coverage.required_job_ids,
    });
}

function isIndexedProcessingUnresolved(processing: ConversationDocument['processing'], jobId: string): boolean {
    const completion = processing.completions?.[jobId];
    return (!completion || completion.status === 'blocked') && !processing.supersessions?.[jobId];
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

/** v1 readers retain their conservative contract; v2 is independently complete, never inferred from missing families. */
export function hasIndexedDeleteProfile(root: IndexedConversationRoot): boolean {
    return (
        root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE ||
        root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2
    );
}
export function indexedDisplayAnswer(turn: IndexedConversationTurnHeader['turn']): boolean {
    return (
        turn.status === 'completed' &&
        ((turn.kind === 'agent' && turn.authority === 'ordinary') ||
            (turn.kind === 'program' && turn.presentation === 'transcript'))
    );
}
export function indexedDisplayAnswerKey(ordinal: number): string {
    return indexedOrderedKey(Number.MAX_SAFE_INTEGER - ordinal);
}
export interface IndexedDeleteDependency {
    target_turn_id: string;
    kind: 'turn' | 'asset' | 'compaction' | 'job';
    owner_id: string;
}
/** Hash keys keep even maximum-length canonical IDs inside the fixed index key bound. */
export async function indexedDeleteDependencyEntry(dependency: IndexedDeleteDependency) {
    const target = await fingerprintJson({ domain: 'indexed-delete-target/v2', id: dependency.target_turn_id });
    const owner = await fingerprintJson({ kind: dependency.kind, id: dependency.owner_id });
    return {
        key: `${target}:${owner}`,
        value: { storage: 'marker' as const, kind: `delete_dependency_${dependency.kind}`, id: dependency.owner_id },
    };
}
/** All retained forensic generation/execution receipts remain immutable; they are NOT live body dependencies. */
export function indexedSnapshotDeleteDependencies(document: ConversationDocument): IndexedDeleteDependency[] {
    const dependencies: IndexedDeleteDependency[] = [];
    const turns = [
        ...document.turns,
        ...Object.values(document.compactions).flatMap((value) => value.replacement_turns),
    ];
    const owners = new Map(
        turns.flatMap((turn) => deletedContentIdentities(turn.blocks).block_ids.map((id) => [id, turn.id] as const)),
    );
    const callOwners = new Map(
        turns.flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [[block.call_id, turn.id] as const] : [])),
        ),
    );
    const entries = new Map(
        Object.values(document.operation_receipts).flatMap((receipt) =>
            (receipt.accepted_context_entries ?? []).map((entry) => [entry.id, entry.turn_id] as const),
        ),
    );
    const add = (target: string | undefined, kind: IndexedDeleteDependency['kind'], owner: string) => {
        if (target !== undefined) dependencies.push({ target_turn_id: target, kind, owner_id: owner });
    };
    for (const turn of turns) {
        add(turn.parent_turn_id, 'turn', turn.id);
        if (turn.provenance.type === 'derived')
            for (const id of turn.provenance.source_turn_ids) add(id, 'turn', turn.id);
        for (const block of turn.blocks)
            if (block.type === 'native_replay') {
                for (const id of block.dependencies.turn_ids) add(id, 'turn', turn.id);
                for (const id of block.dependencies.block_ids) add(owners.get(id), 'turn', turn.id);
                for (const callId of block.dependencies.call_ids) {
                    add(callOwners.get(callId), 'turn', turn.id);
                }
            }
    }
    for (const receipt of Object.values(document.execution_receipts))
        if (receipt.result_turn_id)
            add(receipt.call_source?.turn_id ?? callOwners.get(receipt.call_id), 'turn', receipt.result_turn_id);
    for (const asset of Object.values(document.assets))
        if (asset.provenance.type === 'received') add(asset.provenance.source_turn_id, 'asset', asset.id);
    for (const compaction of Object.values(document.compactions)) {
        for (const id of compaction.source.turn_ids) add(id, 'compaction', compaction.id);
        for (const id of compaction.source.block_ids ?? []) add(owners.get(id), 'compaction', compaction.id);
    }
    for (const job of Object.values(document.processing.jobs ?? {})) {
        if (job.selection.kind === 'entries') {
            for (const id of job.selection.entry_ids) add(entries.get(id), 'job', job.id);
            for (const entry of job.selection.selected_entries ?? []) add(entry.turn_id, 'job', job.id);
        }
        for (const id of document.operation_receipts[job.source_operation_id]?.accepted_turn_ids ?? [])
            add(id, 'job', job.id);
    }
    return dependencies;
}
async function appendIndexedDeleteDependencies(
    store: IndexedConversationRecordStore,
    directories: IndexedConversationDirectories,
    dependencies: readonly IndexedDeleteDependency[],
) {
    const entries = new Map<string, Awaited<ReturnType<typeof indexedDeleteDependencyEntry>>>();
    for (const dependency of dependencies) {
        const entry = await indexedDeleteDependencyEntry(dependency);
        if (!entries.has(entry.key) && !(await getPagedRecord(store, directories.deletion_dependencies, entry.key)))
            entries.set(entry.key, entry);
    }
    await insertFreshIndexedRecords(store, directories, 'deletion_dependencies', [...entries.values()]);
}
async function assertIndexedDeleteDependencies(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    turnId: string,
    selected: ReadonlySet<string>,
) {
    const prefix = `${await fingerprintJson({ domain: 'indexed-delete-target/v2', id: turnId })}:`;
    const range = await readPagedRecordRange(store, root.directories.deletion_dependencies, {
        after: prefix,
        limit: 256,
        max_page_reads: 64,
        max_bytes: 8 * 1024 * 1024,
    });
    if (range.has_more && range.entries.at(-1)?.key.startsWith(prefix))
        throw new IndexedPresentationCapacityError('Indexed deletion exceeds 256 dependency witnesses per turn');
    for (const entry of range.entries) {
        if (!entry.key.startsWith(prefix)) break;
        const dependency = entry.value;
        if (dependency.storage !== 'marker') throw new Error('Indexed deletion dependency is not a typed marker');
        const kind = dependency.kind.slice('delete_dependency_'.length);
        if (
            !['turn', 'asset', 'compaction', 'job'].includes(kind) ||
            entry.key !== `${prefix}${await fingerprintJson({ kind, id: dependency.id })}`
        )
            throw new Error('Indexed deletion dependency key differs from its typed owner');
        if (dependency.kind === 'delete_dependency_turn') {
            if (selected.has(dependency.id)) continue;
            const body = await getPagedRecord(store, root.directories.turns, dependency.id);
            if (body?.storage === 'marker' && body.kind === 'deleted_turn') {
                const acceptance = await getPagedRecord(store, root.directories.turn_acceptances, dependency.id);
                const receipt =
                    acceptance?.storage === 'marker' && acceptance.kind === 'turn_acceptance'
                        ? await indexedRecordById(
                              store,
                              root,
                              'operation_receipts',
                              acceptance.id,
                              OperationReceiptSchema,
                          )
                        : undefined;
                if (!receipt) throw new Error('Deleted dependent lacks its accepted operation');
                await loadIndexedAcceptedTurn(store, root, dependency.id, receipt);
                continue;
            }
            if (body?.storage !== 'record') throw new Error('Indexed live dependency lost its retained body');
            throw new IndexedConversationDeleteConflict('Indexed deletion leaves a live dependent turn');
        }
        if (dependency.kind === 'delete_dependency_job') {
            const job = await indexedProcessingRecord(store, root, 'jobs', dependency.id, ProcessingJobSchema);
            if (!job) throw new Error('Indexed deletion dependency lost its retained job');
            const completion = await indexedProcessingRecord(
                store,
                root,
                'completions',
                dependency.id,
                ProcessingCompletionReceiptSchema,
            );
            const supersession = await indexedProcessingRecord(
                store,
                root,
                'supersessions',
                dependency.id,
                ProcessingSupersessionReceiptSchema,
            );
            if (
                (completion &&
                    (completion.job_id !== dependency.id || completion.result_revision > root.source.revision)) ||
                (supersession && supersession.job_id !== dependency.id)
            )
                throw new Error('Indexed deletion job dependency has another completion identity');
            if (!supersession && (!completion || completion.status === 'blocked'))
                throw new IndexedConversationDeleteConflict('Indexed deletion leaves unresolved processing');
            continue;
        }
        if (dependency.kind === 'delete_dependency_asset' || dependency.kind === 'delete_dependency_compaction')
            throw new IndexedConversationDeleteConflict('Indexed deletion has retained asset or derivation lineage');
        throw new Error('Indexed deletion dependency has an unknown semantic kind');
    }
}
/** The first reverse LIVE ordinal is the latest display answer, independent of accepted provider-output pointers. */
export async function loadIndexedLastDisplayAnswer(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
) {
    if (root.delete_index_profile !== INDEXED_CONVERSATION_DELETE_PROFILE_V2)
        throw new IndexedConversationDeleteConflict(
            'Indexed last-answer update requires complete v2 display nominations',
        );
    const range = await readPagedRecordRange(store, root.directories.display_answer_order, {
        limit: 1,
        max_page_reads: 64,
        max_bytes: 8 * 1024 * 1024,
    });
    const entry = range.entries[0];
    if (!entry) return undefined;
    const ordinal = Number.MAX_SAFE_INTEGER - Number(entry.key);
    if (
        !Number.isSafeInteger(ordinal) ||
        ordinal < 0 ||
        ordinal >= root.turn_count ||
        entry.value.storage !== 'marker' ||
        entry.value.kind !== 'display_answer'
    )
        throw new Error('Indexed display-answer nomination has another ordinal/family');
    const projected = await loadIndexedProjectedTurn(store, root, entry.value.id);
    const turn = ConversationTurnSchema.parse({ ...projected.header, blocks: projected.selected_blocks });
    const link = await indexedRecordById(store, root, 'turn_links', turn.id, IndexedConversationTurnLinkSchema);
    if (!link || link.ordinal !== ordinal || projected.completeness !== 'full_turn' || !indexedDisplayAnswer(turn))
        throw new Error('Indexed display-answer nomination differs from its live canonical turn');
    return { turn_id: turn.id, ordinal };
}

/** A conservative one-pass reverse witness. A marker may reject a closed deletion, but never permit a dependent one. */
function migrationDeleteBlockers(document: ConversationDocument): Set<string> {
    const blocked = new Set<string>();
    const turns = [
        ...document.turns,
        ...Object.values(document.compactions).flatMap((compaction) => compaction.replacement_turns),
    ];
    const turnIds = new Set(turns.map((turn) => turn.id));
    const blockOwner = new Map<string, string>();
    const entryTurn = new Map<string, string>();
    const acceptedTurns = new Map<string, string[]>();
    for (const turn of turns) {
        for (const id of deletedContentIdentities(turn.blocks).block_ids) blockOwner.set(id, turn.id);
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
    for (const turn of turns) {
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
    resolveOriginalDocument?: ToolResultOriginalDocumentResolver,
): Promise<StagedIndexedConversationRoot> {
    // Parse and own the complete bounded source before writing any indexed record. This one-time
    // profile admits 100k compact turns while keeping the ordinary append/import limit unchanged.
    const document = parseConversationDocument(source, { json_input_limits: INDEXED_MIGRATION_JSON_LIMITS });
    // The explicit snapshot owns full original receipts. Authenticate historical policy commands
    // before writing any new indexed bytes, independently of jobs' remembered configurations.
    const snapshotPolicies = new Map<
        number,
        { command: z.infer<typeof ProcessingPolicyCommandSchema>; receipt: OperationReceipt }
    >();
    for (const receipt of Object.values(document.operation_receipts)) {
        if (receipt.operation_kind !== 'processing' || receipt.processing_operation?.phase !== 'policy') continue;
        const epoch = receipt.processing_operation.policy_revision + 1;
        let command = receipt.processing_operation.policy_command;
        if (command === undefined) {
            let original = document.processing.policy_revision === epoch ? document : undefined;
            if (original === undefined && resolveOriginalDocument !== undefined) {
                const source = { conversation_id: document.id, revision: receipt.result_revision };
                const recovered = parseConversationDocument(await resolveOriginalDocument(source), {
                    json_input_limits: INDEXED_MIGRATION_JSON_LIMITS,
                });
                if (
                    recovered.id !== source.conversation_id ||
                    recovered.revision !== source.revision ||
                    recovered.processing.policy_revision !== epoch ||
                    !sameIndexedRecord(recovered.operation_receipts[receipt.id], receipt)
                )
                    throw new Error(
                        'Indexed historical policy resolver changed its exact authenticated original source',
                    );
                original = recovered;
            }
            if (original !== undefined) {
                const reasons = (receipt.processing_operation.superseded_job_ids ?? []).map(
                    (id) => original.processing.supersessions?.[id],
                );
                const reason = reasons[0]?.reason;
                if (
                    reasons.some(
                        (item) =>
                            item === undefined || item.policy_operation_id !== receipt.id || item.reason !== reason,
                    )
                )
                    throw new Error('Indexed policy original supersession reason has no exact accepted source');
                command = await recoverAcceptedProcessingPolicyCommand(receipt, original.processing, reason);
            }
        }
        if (command === undefined) continue;
        await assertIndexedProcessingPolicyAcceptance(
            { conversation_id: document.id, revision: document.revision },
            command,
            receipt,
        );
        if (
            epoch === document.processing.policy_revision &&
            (command.enabled !== document.processing.enabled ||
                !sameIndexedRecord(command.processors, document.processing.processors) ||
                !sameIndexedRecord(command.budget ?? null, document.processing.budget ?? null))
        )
            throw new Error('Indexed snapshot current policy differs from its retained accepted command');
        if (epoch > document.processing.policy_revision || snapshotPolicies.has(epoch))
            throw new Error('Indexed snapshot policy epoch is impossible or ambiguous');
        snapshotPolicies.set(epoch, { command, receipt });
    }
    const firstPolicy = snapshotPolicies.get(1);
    const capturedGenesis = firstPolicy?.receipt.processing_operation?.policy_genesis;
    let originalGenesis: ConversationDocument | undefined;
    if (
        document.processing.policy_revision > 0 &&
        Object.values(document.processing.jobs ?? {}).some((job) => job.policy_revision === 0)
    ) {
        if (firstPolicy?.receipt.processing_operation?.policy_revision !== 0)
            throw new Error('Indexed materialized genesis requires its genuine first accepted policy command');
        if (capturedGenesis === undefined && resolveOriginalDocument !== undefined) {
            const source = { conversation_id: document.id, revision: firstPolicy.receipt.base_revision };
            originalGenesis = parseConversationDocument(await resolveOriginalDocument(source), {
                json_input_limits: INDEXED_MIGRATION_JSON_LIMITS,
            });
            if (
                originalGenesis.id !== source.conversation_id ||
                originalGenesis.revision !== source.revision ||
                originalGenesis.processing.policy_revision !== 0 ||
                originalGenesis.operation_receipts[firstPolicy.receipt.id] !== undefined
            )
                throw new Error('Indexed materialized genesis resolver changed its exact predecessor source');
        }
        if (capturedGenesis === undefined && originalGenesis === undefined)
            throw new Error('Indexed materialized genesis requires an authenticated original predecessor policy');
    }
    for (const job of Object.values(document.processing.jobs ?? {})) {
        const accepted = snapshotPolicies.get(job.policy_revision);
        const policy =
            job.policy_revision === 0
                ? document.processing.policy_revision === 0
                    ? document.processing
                    : (capturedGenesis?.policy ?? originalGenesis?.processing)
                : accepted?.command;
        const configuration = policy?.processors[job.processor_index];
        if (
            !policy?.enabled ||
            !configuration ||
            (job.policy_revision === 0 &&
                firstPolicy !== undefined &&
                job.enqueue_revision > firstPolicy.receipt.base_revision) ||
            (job.policy_revision === 0 &&
                originalGenesis !== undefined &&
                !sameIndexedRecord(originalGenesis.processing.jobs?.[job.id], job)) ||
            (accepted !== undefined && accepted.receipt.result_revision > job.enqueue_revision) ||
            configuration.id !== job.processor_id ||
            configuration.version !== job.processor_version ||
            configuration.scope !== job.scope ||
            configuration.required !== job.required ||
            configuration.failure_behavior !== job.failure_behavior ||
            (await fingerprintJson(configuration.config)) !== job.configuration_fingerprint
        )
            throw new Error('Indexed snapshot historical job requires authenticated original policy evidence');
    }
    if (Object.keys(document.deleted_turns ?? {}).length > 0) {
        throw new Error('Indexed migration cannot retain logical-delete tombstone witnesses yet');
    }
    // Materialized resolutions use a different context-fingerprint profile. Preserve their
    // exact retained facts, but never schedule an unfinished foreign-profile phase as indexed
    // work. This check precedes every indexed page/record write and owner publication.
    for (const job of Object.values(document.processing.jobs ?? {})) {
        if (
            isIndexedProcessingUnresolved(document.processing, job.id) &&
            (document.processing.resolved_inputs?.[job.id] ||
                document.processing.attempts?.[job.id] ||
                document.processing.outputs?.[job.id] ||
                document.processing.completions?.[job.id])
        )
            throw new Error('Indexed migration requires unresolved materialized phases to drain or be superseded');
    }
    if (document.processing.enabled) {
        // Migration preserves validated pluggable policy records. Native execution separately
        // proves a supported registered profile through assertIndexedCurrentPolicy.
        const turns = createContextTurnIndex(document);
        const blockCount = document.context.entries.reduce(
            (count, entry) =>
                count +
                resolveContextEntry(turns, entry).blocks.reduce(
                    (total, block) => total + 1 + (block.type === 'tool_result' ? block.content.length : 0),
                    0,
                ),
            0,
        );
        if (blockCount > INDEXED_PROCESSING_SELECTED_MAX_BLOCKS)
            throw new RangeError('Indexed processing active dependency closure exceeds its selected-block bound');
    }
    const families: Record<keyof IndexedConversationRoot['directories'], Entry[]> = {
        identifiers: [],
        turns: [],
        blocks: [],
        generations: [],
        generation_acceptances: [],
        accepted_output_order: [],
        operation_receipts: [],
        execution_receipts: [],
        assets: [],
        tool_definitions: [],
        compactions: [],
        processing_records: [],
        processing_pending: [],
        processing_required: [],
        processing_by_operation: [],
        processing_coverage: [],
        open_tool_calls: [],
        tool_call_states: [],
        context_entries: [],
        active_context_order: [],
        turn_order: [],
        turn_acceptances: [],
        block_owners: [],
        deletion_blockers: [],
        deletion_dependencies: [],
        display_answer_order: [],
        turn_links: [],
        deleted_turns: [],
    };
    const idKinds = new Map<string, string>();
    const diagnostics = validateConversationSemantics(document, (id, kind) => idKinds.set(id, kind));
    if (diagnostics.length > 0) throw new Error('Indexed conversation migration source failed semantic validation');
    const outputRevisions = new Set<number>();
    const migrationTurns = new Map(document.turns.map((turn) => [turn.id, turn]));
    for (const receipt of Object.values(document.operation_receipts)) {
        if (
            receipt.operation_kind !== undefined ||
            receipt.accepted_generation_ids?.length !== 1 ||
            receipt.accepted_turn_ids?.length !== 1
        )
            continue;
        const generation = document.generations[receipt.accepted_generation_ids[0]];
        const turn = migrationTurns.get(receipt.accepted_turn_ids[0]);
        if (
            !generation ||
            turn?.kind !== 'agent' ||
            turn.provenance.type !== 'generated' ||
            !('generation_id' in turn) ||
            turn.generation_id !== generation.id
        )
            throw new Error('Indexed accepted history import has another canonical response tuple');
        if (outputRevisions.has(receipt.result_revision))
            throw new Error('Indexed accepted history import has ambiguous response receipts');
        outputRevisions.add(receipt.result_revision);
        families.accepted_output_order.push({
            key: indexedOrderedKey(receipt.result_revision),
            value: { storage: 'marker', kind: 'accepted_output', id: receipt.id },
        });
    }
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
        for (const block of blocks) {
            await stageFamily('blocks', block.id, block);
            for (const id of deletedContentIdentities([block]).block_ids)
                families.block_owners.push({ key: id, value: { storage: 'marker', kind: 'block_owner', id: turn.id } });
        }
    };

    // Cold turns are independent. Keep immutable write/read-back work bounded rather than serializing
    // every remote record round trip; no directory root is published until every batch succeeds.
    const migrationWriteWindow = 32;
    for (let offset = 0; offset < document.turns.length; offset += migrationWriteWindow) {
        const staged: Promise<void>[] = [];
        for (let index = offset; index < Math.min(offset + migrationWriteWindow, document.turns.length); index += 1) {
            const turn = document.turns[index];
            staged.push(
                (async () => {
                    await stageTurn(turn, 'ordinary');
                    await stageFamily(
                        'turn_links',
                        turn.id,
                        IndexedConversationTurnLinkSchema.parse({
                            id: turn.id,
                            ordinal: index,
                            ...(index === 0 ? {} : { previous_turn_id: document.turns[index - 1].id }),
                            ...(index === document.turns.length - 1
                                ? {}
                                : { next_turn_id: document.turns[index + 1].id }),
                        }),
                    );
                })(),
            );
            families.turn_order.push({
                key: indexedOrderedKey(index),
                value: { storage: 'marker', kind: 'turn_order', id: turn.id },
            });
        }
        // A failed write must not leave other writes running after the import rejects.
        for (const result of await Promise.allSettled(staged)) {
            if (result.status === 'rejected') throw result.reason;
        }
    }
    for (const [id, compaction] of Object.entries(document.compactions)) {
        const { replacement_turns: replacementTurns, original_context: _originalContext, ...header } = compaction;
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
    const uniqueDependencies = new Map<string, Awaited<ReturnType<typeof indexedDeleteDependencyEntry>>>();
    for (const dependency of indexedSnapshotDeleteDependencies(document)) {
        const entry = await indexedDeleteDependencyEntry(dependency);
        uniqueDependencies.set(entry.key, entry);
    }
    families.deletion_dependencies.push(...uniqueDependencies.values());
    for (let ordinal = 0; ordinal < document.turns.length; ordinal++)
        if (indexedDisplayAnswer(document.turns[ordinal]))
            families.display_answer_order.push({
                key: indexedDisplayAnswerKey(ordinal),
                value: { storage: 'marker', kind: 'display_answer', id: document.turns[ordinal].id },
            });
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
    let selectedPolicyOperationId: string | undefined;
    for (const [epoch, { command, receipt }] of snapshotPolicies) {
        await stageFamily('processing_records', receipt.id, command, tupleKey('selected_policy_commands', receipt.id));
        await stageFamily(
            'processing_records',
            String(epoch),
            IndexedProcessingPolicyEpochSchema.parse({
                version: 1,
                kind: 'accepted_command',
                policy_revision: epoch,
                operation_id: receipt.id,
                receipt_fingerprint: await fingerprintJson(receipt),
            }),
            tupleKey('policy_epochs', String(epoch)),
        );
        if (epoch === document.processing.policy_revision) selectedPolicyOperationId = receipt.id;
    }
    if (capturedGenesis !== undefined && firstPolicy !== undefined)
        await stageFamily(
            'processing_records',
            '0',
            IndexedProcessingPolicyEpochSchema.parse({
                version: 1,
                kind: 'materialized_genesis',
                policy_revision: 0,
                operation_id: firstPolicy.receipt.id,
                receipt_fingerprint: await fingerprintJson(firstPolicy.receipt),
            }),
            tupleKey('policy_epochs', '0'),
        );
    else if (originalGenesis !== undefined && firstPolicy !== undefined) {
        const {
            jobs: _genesisJobs,
            resolved_inputs: _genesisInputs,
            attempts: _genesisAttempts,
            outputs: _genesisOutputs,
            completions: _genesisCompletions,
            supersessions: _genesisSupersessions,
            coverage_receipts: _genesisCoverageReceipts,
            ...originalHeader
        } = originalGenesis.processing;
        const descriptor = await stageRecord(
            store,
            'processing_header',
            document.id,
            IndexedConversationProcessingHeaderSchema.parse(originalHeader),
        );
        await stageFamily(
            'processing_records',
            '0',
            IndexedProcessingPolicyEpochSchema.parse({
                version: 1,
                kind: 'genesis',
                policy_revision: 0,
                source: { conversation_id: originalGenesis.id, revision: originalGenesis.revision },
                processing_header: { content_hash: descriptor.content_hash, size_bytes: descriptor.size_bytes },
                successor_policy_operation_id: firstPolicy.receipt.id,
                successor_policy_receipt_fingerprint: await fingerprintJson(firstPolicy.receipt),
            }),
            tupleKey('policy_epochs', '0'),
        );
    }
    const operationJobs = new Map<string, ProcessingJob[]>();
    for (const job of Object.values(document.processing.jobs ?? {})) {
        const jobs = operationJobs.get(job.source_operation_id) ?? [];
        jobs.push(job);
        operationJobs.set(job.source_operation_id, jobs);
        const queueReceipt = document.operation_receipts[job.source_operation_id];
        if (queueReceipt?.processing_operation?.phase === 'queue') {
            const captured = queueReceipt.processing_operation.queue_command;
            const knownLegacy =
                isToolResultTextProcessor(job) &&
                supportsToolResultTextProcessingScope(job) &&
                job.scope === 'manual' &&
                job.stage_index === 0 &&
                job.selection.kind === 'entries' &&
                job.selection.selected_block_ids === undefined;
            // Preserve old generic snapshot support. Unknown legacy queue commands are not
            // invented; their later explicit native processing/upgrade audit remains unsupported.
            if (captured || knownLegacy) {
                if (job.selection.kind !== 'entries')
                    throw new Error('Indexed migration materialized queue lacks its accepted entry selection');
                const output = document.processing.outputs?.[job.id];
                const originalContext =
                    output?.kind === 'proposal' && output.proposal.kind === 'replace_with_compaction'
                        ? document.compactions[output.proposal.compaction_id]?.original_context
                        : undefined;
                const context = originalContext ?? document.context;
                const candidates = queueReceipt.processing_operation.queue_selected_entries ?? context.entries;
                const selectedEntries = job.selection.entry_ids.map((id) =>
                    candidates.find((entry) => entry.id === id),
                );
                if (selectedEntries.some((entry) => entry === undefined))
                    throw new Error('Indexed migration materialized queue lacks its exact retained selection closure');
                const entries = selectedEntries.filter((entry): entry is ContextEntry => entry !== undefined);
                const input = ProcessingQueueAcceptanceInputSchema.parse(
                    captured ?? {
                        command: {
                            operation_id: queueReceipt.id,
                            expected_revision: queueReceipt.base_revision,
                            recorded_at: queueReceipt.recorded_at,
                            processor_id: job.processor_id,
                            scope: job.scope,
                            ...(job.target_fingerprint === undefined
                                ? {}
                                : { target_fingerprint: job.target_fingerprint }),
                        },
                        selection: {
                            conversation: { conversation_id: document.id, revision: queueReceipt.base_revision },
                            expected_context_revision: context.revision,
                            selector: { source: { kind: 'turn_ids', turn_ids: entries.map((entry) => entry.turn_id) } },
                        },
                    },
                );
                if (
                    queueReceipt.payload_fingerprint !== (await fingerprintJson(input)) ||
                    (!captured &&
                        (input.selection.selector.source.kind !== 'turn_ids' ||
                            input.selection.selector.filters !== undefined ||
                            !sameIndexedRecord(
                                input.selection.selector.source.turn_ids,
                                entries.map((entry) => entry.turn_id),
                            )))
                )
                    throw new Error(
                        'Indexed migration materialized queue cannot reconstruct its exact accepted command',
                    );
                await stageFamily(
                    'processing_records',
                    queueReceipt.id,
                    IndexedMaterializedProcessingQueueSchema.parse({
                        version: 1,
                        ...input,
                        selected_entries: entries,
                        legacy_reconstructed: captured === undefined,
                    }),
                    tupleKey('materialized_queue_commands', queueReceipt.id),
                );
            }
        }
        if (job.required && !document.processing.supersessions?.[job.id])
            families.processing_required.push({
                key: job.id,
                value: { storage: 'marker', kind: 'processing_required', id: job.id },
            });
        if (isIndexedProcessingUnresolved(document.processing, job.id)) {
            families.processing_pending.push({
                key: job.id,
                value: { storage: 'marker', kind: 'processing_pending', id: job.id },
            });
        }
    }
    for (const [operationId, jobs] of operationJobs) {
        const receipt = document.operation_receipts[operationId];
        if (!receipt) throw new Error('Indexed processing job source has no accepted operation');
        jobs.sort((a, b) => a.stage_index - b.stage_index);
        await stageFamily(
            'processing_by_operation',
            operationId,
            IndexedProcessingOperationJobsSchema.parse({
                version: 1,
                operation_id: operationId,
                receipt_fingerprint: await fingerprintJson(receipt),
                job_ids: jobs.map((job) => job.id),
            }),
        );
    }
    const latestCoverage = new Map<
        string,
        { id: string; coverage: z.infer<typeof ProcessingReadinessCoverageSchema> }
    >();
    for (const [id, coverage] of Object.entries(document.processing.coverage_receipts ?? {})) {
        const identity = await indexedCoverageIdentity(coverage);
        const previous = latestCoverage.get(identity);
        if (!previous || coverage.evaluated_at_revision > previous.coverage.evaluated_at_revision)
            latestCoverage.set(identity, { id, coverage });
    }
    for (const [identity, { id }] of latestCoverage) {
        families.processing_coverage.push({
            key: identity,
            value: { storage: 'marker', kind: 'processing_coverage', id },
        });
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
    // Removed source entries remain point-addressable proof inputs for recovery attestations.
    const retainedEntryIds = new Set(document.context.entries.map((entry) => entry.id));
    for (const receipt of Object.values(document.operation_receipts))
        for (const entry of receipt.accepted_context_entries ?? []) {
            if (retainedEntryIds.has(entry.id)) continue;
            retainedEntryIds.add(entry.id);
            await stageFamily('context_entries', entry.id, entry);
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
            ...(selectedPolicyOperationId === undefined
                ? {}
                : { selected_policy_operation_id: selectedPolicyOperationId, selected_policy_origin: 'materialized' }),
            unresolved_job_count: countUnresolvedProcessingJobs(document.processing),
            job_count: Object.keys(document.processing.jobs ?? {}).length,
            required_job_count: Object.values(document.processing.jobs ?? {}).filter(
                (job) => job.required && !document.processing.supersessions?.[job.id],
            ).length,
            required_unresolved_job_count: Object.values(document.processing.jobs ?? {}).filter(
                (job) => job.required && isIndexedProcessingUnresolved(document.processing, job.id),
            ).length,
            required_blocked_job_count: Object.values(document.processing.jobs ?? {}).filter(
                (job) =>
                    job.required &&
                    document.processing.completions?.[job.id]?.status === 'blocked' &&
                    !document.processing.supersessions?.[job.id],
            ).length,
        }),
    );
    if (document.processing.policy_revision === 0)
        await stageFamily(
            'processing_records',
            '0',
            IndexedProcessingPolicyEpochSchema.parse({
                version: 1,
                kind: 'genesis',
                policy_revision: 0,
                source: { conversation_id: document.id, revision: document.revision },
                processing_header: {
                    content_hash: processingHeaderValue.content_hash,
                    size_bytes: processingHeaderValue.size_bytes,
                },
            }),
            tupleKey('policy_epochs', '0'),
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
    const restart = await indexedSnapshotRestartWitness(document);
    const root = IndexedConversationRootSchema.parse({
        ...restart,
        version: 1,
        validator_profile: INDEXED_CONVERSATION_PROFILE,
        delete_index_profile: INDEXED_CONVERSATION_DELETE_PROFILE_V2,
        accepted_output_index_complete: true,
        tool_call_state_complete: true,
        processing_index_profile: INDEXED_CONVERSATION_PROCESSING_PROFILE,
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
    let originalSourceDocumentBytes = 0;
    // Migration is a one-time recovery boundary. Verify the deterministic transform here,
    // while complete originals are already held, then retain the same body-free commit witness.
    const migratedJobs = Object.values(document.processing.jobs ?? {}).sort((a, b) => {
        const first = document.processing.completions?.[a.id]?.result_revision ?? Number.MAX_SAFE_INTEGER;
        const second = document.processing.completions?.[b.id]?.result_revision ?? Number.MAX_SAFE_INTEGER;
        return first - second || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
    });
    for (const job of migratedJobs) {
        const completion = document.processing.completions?.[job.id];
        // Retained applied jobs remain historical upgrade obligations after their replacement
        // leaves the active context. The owned migration document already bounds this pass.
        if (!isToolResultTextProcessor(job) || completion?.status !== 'applied') continue;
        const resolution = document.processing.resolved_inputs?.[job.id];
        const output = document.processing.outputs?.[job.id];
        const acceptance = document.operation_receipts[completion.context_change_operation_id ?? ''];
        const archive = document.operation_receipts[`processing:archive:${job.id}`];
        if (
            !resolution ||
            output?.kind !== 'proposal' ||
            output.proposal.kind !== 'replace_with_compaction' ||
            !acceptance ||
            !archive
        )
            throw new Error('Indexed tool-result migration lacks its complete processing lineage');
        const compaction = document.compactions[output.proposal.compaction_id];
        if (!compaction) throw new Error('Indexed tool-result migration lacks its accepted compaction');
        const entryIndex = new Map(
            Object.values(document.operation_receipts).flatMap((receipt) =>
                (receipt.accepted_context_entries ?? []).map((entry) => [entry.id, entry] as const),
            ),
        );
        const originalByReplacement = new Map(
            completion.inserted_entry_ids.map(
                (id, index) => [id, entryIndex.get(resolution.entry_ids[index])] as const,
            ),
        );
        // Historical immediate compactions can be inverted only when the entire resulting
        // context and every original active turn reproduce the accepted resolution fingerprint.
        const candidateContext = compaction.original_context ?? {
            ...document.context,
            revision: resolution.context_revision,
            entries: document.context.entries.map((entry) => originalByReplacement.get(entry.id) ?? entry),
            retrieval_requirements: document.context.retrieval_requirements.filter(
                (item) => item.accepted_asset_operation_id !== archive.id,
            ),
        };
        let originalDocument = document;
        let originalContext = candidateContext;
        const sourceContextFingerprint = (sourceDocument: ConversationDocument, sourceContext: ConversationContext) => {
            const originalTurns = createContextTurnIndex(sourceDocument);
            return fingerprintJson({
                context: sourceContext,
                entries: sourceContext.entries.map((entry) => ({ entry, turn: originalTurns.get(entry.turn_id) })),
            });
        };
        if (
            candidateContext.revision !== resolution.context_revision ||
            (await sourceContextFingerprint(originalDocument, originalContext)) !== resolution.context_fingerprint
        ) {
            if (compaction.original_context !== undefined)
                throw new Error('Tool-result migration changed its retained original context');
            if (!resolveOriginalDocument) throw new IndexedToolResultOriginalSourceUnavailableError(job.id);
            originalDocument = parseConversationDocument(
                await resolveOriginalDocument({ conversation_id: document.id, revision: resolution.source_revision }),
                { json_input_limits: INDEXED_MIGRATION_JSON_LIMITS },
            );
            originalSourceDocumentBytes += canonicalJsonContentBytes(originalDocument).byteLength;
            if (originalSourceDocumentBytes > INDEXED_PROCESSING_MAX_IO_BYTES)
                throw new RangeError(
                    'Tool-result migration historical documents exceed the finite recovery byte budget',
                );
            originalContext = originalDocument.context;
            if (
                originalDocument.id !== document.id ||
                originalDocument.revision !== resolution.source_revision ||
                originalDocument.context.revision !== resolution.context_revision ||
                !sameIndexedRecord(originalDocument.processing.jobs?.[job.id], job) ||
                (await sourceContextFingerprint(originalDocument, originalContext)) !== resolution.context_fingerprint
            )
                throw new Error('Tool-result migration resolver returned a foreign original context');
        }
        const turns = createContextTurnIndex(originalDocument);
        const frame: ToolResultTextSelectionFrame = {
            source: { conversation_id: document.id, revision: acceptance.base_revision },
            context: originalContext,
            turns,
            active_blocks: new Map(
                originalContext.entries.map((entry) => [entry.id, resolveContextEntry(turns, entry).blocks]),
            ),
            execution_receipts: originalDocument.execution_receipts,
            projection_records: originalDocument,
        };
        const rebuilt = await buildToolResultTextWorkingProposal(
            frame,
            document.assets,
            document.tool_definitions,
            archive,
            'tool_result_text',
            job,
            resolution,
            toolResultTextArchiveRetrievals(output.proposal, archive),
        );
        if (canonicalJsonContentString(rebuilt.proposal) !== canonicalJsonContentString(output.proposal))
            throw new Error('Indexed tool-result migration changed its deterministic original transformation');
        await stageIndexedToolResultOriginalSource(store, root.directories, job, resolution, {
            kind: 'materialized_context',
            context: originalContext,
        });
        await stageIndexedToolResultValidation(
            store,
            root,
            root.directories,
            frame,
            job,
            resolution,
            output,
            completion,
            compaction,
            acceptance,
        );
    }
    const rootValue = await stageRecord(store, 'root', document.id, root);
    if (rootValue.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES) {
        throw new RangeError('Indexed conversation root exceeds its manifest bound');
    }
    const locator = { content_hash: rootValue.content_hash, size_bytes: rootValue.size_bytes };
    if (document.processing.enabled) await assertIndexedProcessingAcceptanceWorkingSet(store, root, locator);
    return { root, locator };
}

/** Batch fresh index nominations without retaining one obsolete immutable page per inserted key.
 * Partition only the owned commands; never traverse the lifetime index. Every grouped insertion
 * preserves exact collision checks and durable page readback under the caller's operation budget. */
async function insertFreshIndexedRecords(
    store: IndexedConversationRecordStore,
    directories: IndexedConversationRoot['directories'],
    family: keyof IndexedConversationRoot['directories'],
    commands: readonly { key: string; value: PagedRecordValue }[],
): Promise<void> {
    // An absent optional directory must stay omitted when this append has no nominations.
    // Assigning undefined would make the authenticated root non-JSON.
    if (commands.length === 0) return;
    let locator = directories[family];
    let group: { key: string; value: PagedRecordValue }[] = [];
    let bytes = 0;
    const flush = async () => {
        if (group.length === 0) return;
        locator = await insertPagedRecords(store, locator, group);
        group = [];
        bytes = 0;
    };
    for (const command of commands) {
        const commandBytes = canonicalJsonContentBytes(command).byteLength + 1;
        // Reserve the bounded root/envelope JSON overhead inside the existing 8MiB command cap.
        if (group.length >= 4096 || bytes + commandBytes > PAGED_RECORD_INDEX_MAX_BATCH_BYTES - 1024) await flush();
        group.push(command);
        bytes += commandBytes;
    }
    await flush();
    if (locator === undefined) throw new Error('Nonempty indexed insertion has no retained directory');
    directories[family] = locator;
}

/** Maintain the complete reverse-delete witness for every record family this indexed append accepts. */
async function appendDeleteIndex(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    directories: IndexedConversationRoot['directories'],
    operationId: string,
    batch: ConversationRecordBatch,
): Promise<{ live_turn_count?: number; active_tail_turn_id?: string | null }> {
    if (!hasIndexedDeleteProfile(root)) return {};
    if (root.live_turn_count === undefined || root.active_tail_turn_id === undefined) {
        throw new Error('Indexed delete profile lacks its live-turn witnesses');
    }
    const newTurns = new Map((batch.turns ?? []).map((turn) => [turn.id, turn]));
    const newBlockOwners = new Map<string, string>();
    for (const turn of batch.turns ?? [])
        for (const id of deletedContentIdentities(turn.blocks).block_ids) newBlockOwners.set(id, turn.id);
    if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2) {
        const dependencies: IndexedDeleteDependency[] = [];
        for (const turn of batch.turns ?? []) {
            if (turn.parent_turn_id)
                dependencies.push({ target_turn_id: turn.parent_turn_id, kind: 'turn', owner_id: turn.id });
            for (const block of turn.blocks)
                if (block.type === 'native_replay') {
                    const add = (target_turn_id: string) =>
                        dependencies.push({ target_turn_id, kind: 'turn', owner_id: turn.id });
                    for (const id of block.dependencies.turn_ids) add(id);
                    for (const id of block.dependencies.block_ids) {
                        const fresh = newBlockOwners.get(id);
                        const retained =
                            fresh === undefined
                                ? await getPagedRecord(store, root.directories.block_owners, id)
                                : undefined;
                        if (fresh !== undefined) add(fresh);
                        else if (retained?.storage === 'marker' && retained.kind === 'block_owner') add(retained.id);
                        else throw new Error('Indexed replay dependency lost its exact retained block owner');
                    }
                    for (const id of block.dependencies.call_ids) {
                        const fresh = (batch.turns ?? []).find((value) =>
                            value.blocks.some(
                                (candidate) => candidate.type === 'tool_call' && candidate.call_id === id,
                            ),
                        );
                        const retained =
                            fresh === undefined
                                ? await indexedRecordById(store, root, 'tool_call_states', id, IndexedCallStateSchema)
                                : undefined;
                        if (!fresh && !retained)
                            throw new Error('Indexed replay dependency lost its exact retained call owner');
                        if (fresh) add(fresh.id);
                        else if (retained) add(retained.turn_id);
                    }
                }
        }
        for (const receipt of batch.execution_receipts ?? []) {
            if (!receipt.result_turn_id) continue;
            const fresh = (batch.turns ?? []).find((turn) =>
                turn.blocks.some((block) => block.type === 'tool_call' && block.call_id === receipt.call_id),
            );
            const retained = fresh
                ? undefined
                : await indexedRecordById(store, root, 'tool_call_states', receipt.call_id, IndexedCallStateSchema);
            const target = receipt.call_source?.turn_id ?? fresh?.id ?? retained?.turn_id;
            if (!target) throw new Error('Indexed terminal dependency lost its exact original call owner');
            dependencies.push({ target_turn_id: target, kind: 'turn', owner_id: receipt.result_turn_id });
        }
        for (const asset of batch.assets ?? [])
            if (asset.provenance.type === 'received' && asset.provenance.source_turn_id)
                dependencies.push({
                    target_turn_id: asset.provenance.source_turn_id,
                    kind: 'asset',
                    owner_id: asset.id,
                });
        await appendIndexedDeleteDependencies(store, directories, dependencies);
        await insertFreshIndexedRecords(
            store,
            directories,
            'display_answer_order',
            (batch.turns ?? []).flatMap((turn, index) =>
                indexedDisplayAnswer(turn)
                    ? [
                          {
                              key: indexedDisplayAnswerKey(root.turn_count + index),
                              value: { storage: 'marker', kind: 'display_answer', id: turn.id },
                          },
                      ]
                    : [],
            ),
        );
    }
    const blockers = new Set<string>();
    const markTurn = async (id: string | undefined) => {
        if (id === undefined) return;
        if (newTurns.has(id)) {
            blockers.add(id);
            return;
        }
        const existing = await getPagedRecord(store, root.directories.turns, id);
        if (existing?.storage === 'marker' && existing.kind === 'deleted_turn') {
            if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2) return;
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
            if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2) return;
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
    const freshBlockers: { key: string; value: PagedRecordValue }[] = [];
    for (const id of blockers) {
        if (await getPagedRecord(store, directories.deletion_blockers, id)) continue;
        freshBlockers.push({ key: id, value: { storage: 'marker', kind: 'delete_blocker', id } });
    }
    await insertFreshIndexedRecords(store, directories, 'deletion_blockers', freshBlockers);
    const turns = batch.turns ?? [];
    await insertFreshIndexedRecords(
        store,
        directories,
        'turn_acceptances',
        turns.map((turn) => ({ key: turn.id, value: { storage: 'marker', kind: 'turn_acceptance', id: operationId } })),
    );
    await insertFreshIndexedRecords(
        store,
        directories,
        'block_owners',
        turns.flatMap((turn) =>
            deletedContentIdentities(turn.blocks).block_ids.map((id) => ({
                key: id,
                value: { storage: 'marker', kind: 'block_owner', id: turn.id },
            })),
        ),
    );
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
    const links: { key: string; value: PagedRecordValue }[] = [];
    for (const [index, turn] of turns.entries()) {
        const previousTurnId = index === 0 ? previousTail : turns[index - 1].id;
        const link = IndexedConversationTurnLinkSchema.parse({
            id: turn.id,
            ordinal: root.turn_count + index,
            ...(previousTurnId === null ? {} : { previous_turn_id: previousTurnId }),
            ...(index === turns.length - 1 ? {} : { next_turn_id: turns[index + 1].id }),
        });
        links.push({ key: turn.id, value: await stageRecord(store, 'turn_links', turn.id, link) });
    }
    await insertFreshIndexedRecords(store, directories, 'turn_links', links);
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
    // Both the metadata body and the complete ordered-entry body belong to this active window.
    // Charge their authenticated lengths even if there are zero selected turns/entries.
    if (root.context_header.size_bytes + header.active_entry_bytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES)
        throw new RangeError('Indexed active context header and entries exceed the aggregate working-set bound');
    const entries: z.infer<typeof ContextEntrySchema>[] = [];
    let aggregate = 0;
    for await (const ordered of scanPagedRecords(store, root.directories.active_context_order)) {
        if (entries.length >= header.active_entry_count || entries.length >= 100_000) {
            throw new RangeError('Active context has more records than its bounded header');
        }
        const descriptor = await getPagedRecord(store, root.directories.context_entries, ordered.value.id);
        if (
            descriptor?.storage === 'record' &&
            root.context_header.size_bytes + aggregate + descriptor.size_bytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES
        ) {
            throw new RangeError('Active context exceeds bound before record read');
        }
        const entry = await loadRecord(store, descriptor, ContextEntrySchema);
        entries.push(entry);
        aggregate += canonicalJsonContentBytes(entry).byteLength;
        if (root.context_header.size_bytes + aggregate > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES)
            throw new RangeError('Active context exceeds aggregate metadata/entry bound');
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
    reserveRecordBytes?: (byteLength: number, contentHash?: string) => void,
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
    if (descriptor?.storage === 'record') reserve(descriptor.size_bytes, descriptor.content_hash);
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
        if (descriptor?.storage === 'record') reserve(descriptor.size_bytes, descriptor.content_hash);
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

/** Shared bounded resolver; dependent content requires the explicit dependency-complete profile. */
async function loadIndexedSelectedContext(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    rootLocator: PagedRecordRef,
    maxSelectedBytes = INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
    includeDependencies = false,
    includeMediaCompaction = false,
    purpose: 'preparation' | 'processing' = 'preparation',
    allowPendingCalls = false,
) {
    if (
        !Number.isSafeInteger(maxSelectedBytes) ||
        maxSelectedBytes <= 0 ||
        maxSelectedBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES
    ) {
        throw new RangeError('Indexed selected context byte budget is invalid');
    }
    if (
        !preflightJsonInput(rootInput, { max_bytes: INDEXED_CONVERSATION_ROOT_MAX_BYTES }).success ||
        !preflightJsonInput(rootLocator).success
    )
        throw new TypeError('Indexed selected root/locator is not bounded owned JSON');
    const root = IndexedConversationRootSchema.parse(structuredClone(rootInput));
    rootLocator = PagedRecordRefSchema.parse(structuredClone(rootLocator));
    // Header metadata is charged before its record download, independently of index-page limits.
    // An impossible active envelope cannot consume its entire budget before the first selected body.
    if (root.context_header.size_bytes + rootLocator.size_bytes + root.processing_header.size_bytes > maxSelectedBytes)
        throw new RangeError('Indexed selected context headers exceed working-set bound before record read');
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
    if (purpose === 'preparation' && processing.enabled)
        throw new Error('Indexed selected preparation requires processing readiness');
    if (purpose === 'processing') {
        if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
            throw new Error('Indexed processing selection lacks the complete processing profile');
        assertIndexedProcessingCounts(processing);
    }
    if (processing.unresolved_job_count === undefined && root.directories.processing_records !== undefined) {
        throw new Error('Indexed selected preparation has no accepted processing job-drain witness');
    }
    if (purpose === 'preparation' && (processing.unresolved_job_count ?? 0) > 0) {
        throw new Error('Indexed selected preparation has accepted processing jobs outstanding');
    }
    const selections = new Map<string, Set<string> | undefined>();
    const replacementIds = new Map<string, string>();
    for (const entry of context.entries) {
        if (entry.type === 'replacement_turn') {
            if (selections.has(entry.turn_id) && !replacementIds.has(entry.turn_id))
                throw new Error('Indexed selected turn is both an original and a replacement');
            if (!includeMediaCompaction) throw new Error('Indexed text preparation needs a compaction witness');
            const prior = replacementIds.get(entry.turn_id);
            if (prior !== undefined && prior !== entry.compaction_id)
                throw new Error('Indexed replacement has conflicting selected compaction identities');
            replacementIds.set(entry.turn_id, entry.compaction_id);
        } else if (replacementIds.has(entry.turn_id)) {
            throw new Error('Indexed selected turn is both an original and a replacement');
        }
        const prior = selections.get(entry.turn_id);
        if (!selections.has(entry.turn_id)) {
            selections.set(entry.turn_id, entry.block_ids === undefined ? undefined : new Set(entry.block_ids));
        } else if (prior !== undefined) {
            if (entry.block_ids === undefined) selections.set(entry.turn_id, undefined);
            else for (const id of entry.block_ids) prior.add(id);
        }
    }
    const turns: z.infer<typeof IndexedConversationSelectedContextSchema>['turns'] = [];
    const compactionWitnesses = new Map<
        string,
        {
            compaction: z.infer<typeof IndexedConversationCompactionHeaderSchema>;
            acceptance: OperationReceipt;
        }
    >();
    const operationWitnesses = new Map<string, OperationReceipt>();
    const generationWitnesses = new Map<
        string,
        { generation: z.infer<typeof GenerationSchema>; acceptance: OperationReceipt }
    >();
    let totalBytes =
        canonicalJsonContentBytes(context).byteLength + rootLocator.size_bytes + root.processing_header.size_bytes;
    if (totalBytes > maxSelectedBytes)
        throw new RangeError('Indexed selected context metadata exceeds working-set bound before record read');
    let selectedRecords = 0;
    const reservedRecords = new Set<string>();
    const reserve = (byteLength: number, contentHash?: string) => {
        if (contentHash !== undefined) {
            const identity = `${contentHash}:${byteLength}`;
            if (reservedRecords.has(identity)) return;
            reservedRecords.add(identity);
        }
        selectedRecords += 1;
        totalBytes += byteLength;
        if (
            selectedRecords > (purpose === 'processing' ? INDEXED_PROCESSING_MAX_RECORD_READS : 100_000) ||
            totalBytes > maxSelectedBytes
        ) {
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
        const compactionId = replacementIds.get(turnId);
        if (compactionId !== undefined) {
            const turnDescriptor = await getPagedRecord(store, root.directories.turns, turnId);
            if (turnDescriptor?.storage === 'record') reserve(turnDescriptor.size_bytes, turnDescriptor.content_hash);
            const replacementHeader = await loadRecord(store, turnDescriptor, IndexedConversationTurnHeaderSchema);
            if (replacementHeader.source !== 'replacement' || replacementHeader.compaction_id !== compactionId)
                throw new Error('Indexed replacement entry differs from its exact stored compaction identity');
            let witness = compactionWitnesses.get(compactionId);
            if (!witness) {
                const descriptor = await getPagedRecord(store, root.directories.compactions, compactionId);
                if (descriptor?.storage === 'record') reserve(descriptor.size_bytes, descriptor.content_hash);
                const compaction = await loadRecord(store, descriptor, IndexedConversationCompactionHeaderSchema);
                const operation = await getPagedRecord(
                    store,
                    root.directories.operation_receipts,
                    compaction.operation_id,
                );
                if (operation?.storage === 'record') reserve(operation.size_bytes, operation.content_hash);
                const acceptance = await loadRecord(store, operation, OperationReceiptSchema);
                assertIndexedAcceptedCompaction(root, compactionId, compaction, acceptance);
                witness = { compaction, acceptance };
                compactionWitnesses.set(compactionId, witness);
                operationWitnesses.set(acceptance.id, acceptance);
            }
            const provenance = turn.header.provenance;
            if (
                provenance.type !== 'derived' ||
                provenance.derivation_id !== compactionId ||
                provenance.source_hash !== witness.compaction.source.source_fingerprint ||
                provenance.source_turn_ids.length === 0 ||
                provenance.source_turn_ids.some((id) => !witness.compaction.source.turn_ids.includes(id)) ||
                (witness.compaction.source.block_ids !== undefined &&
                    (provenance.source_block_ids ?? []).some(
                        (id) => !witness.compaction.source.block_ids?.includes(id),
                    ))
            )
                throw new Error('Indexed replacement provenance differs from its retained selected-source witness');
        }
        if (
            !includeDependencies &&
            (turn.header.kind === 'tool' ||
                turn.header.provenance.type === 'derived' ||
                turn.header.provenance.type === 'imported' ||
                turn.header.parent_turn_id !== undefined ||
                turn.header.execution_id !== undefined ||
                turn.header.exchange_id !== undefined ||
                turn.selected_blocks.some((block) => block.type !== 'text' && block.type !== 'native_replay'))
        ) {
            throw new Error('Indexed text preparation has unsupported selected content or derivation');
        }
        if (turn.header.kind === 'agent' && 'generation_id' in turn.header && turn.header.generation_id !== undefined) {
            const generationDescriptor = await getPagedRecord(
                store,
                root.directories.generations,
                turn.header.generation_id,
            );
            if (generationDescriptor?.storage === 'record')
                reserve(generationDescriptor.size_bytes, generationDescriptor.content_hash);
            const generation = await loadRecord(store, generationDescriptor, GenerationSchema);
            const accepted = await getPagedRecord(store, root.directories.generation_acceptances, generation.id);
            if (accepted?.storage !== 'marker' || accepted.kind !== 'generation_acceptance') {
                throw new Error('Indexed generated turn has no accepted generation operation');
            }
            const acceptanceDescriptor = await getPagedRecord(store, root.directories.operation_receipts, accepted.id);
            if (acceptanceDescriptor?.storage === 'record')
                reserve(acceptanceDescriptor.size_bytes, acceptanceDescriptor.content_hash);
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
            if (receiptDescriptor?.storage === 'record')
                reserve(receiptDescriptor.size_bytes, receiptDescriptor.content_hash);
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
    if (purpose === 'processing') {
        const blockCount = turns.reduce(
            (count, turn) =>
                count +
                turn.selected_blocks.reduce(
                    (blocks, block) => blocks + 1 + (block.type === 'tool_result' ? block.content.length : 0),
                    0,
                ),
            0,
        );
        if (blockCount > INDEXED_PROCESSING_SELECTED_MAX_BLOCKS)
            throw new RangeError('Indexed processing active dependency closure exceeds its selected-block bound');
    }
    const toolDefinitions = new Map<string, z.infer<typeof ToolDefinitionSchema>>();
    for (const id of context.active_tool_definition_ids) {
        const descriptor = await getPagedRecord(store, root.directories.tool_definitions, id);
        if (descriptor?.storage === 'record') reserve(descriptor.size_bytes, descriptor.content_hash);
        toolDefinitions.set(id, await loadRecord(store, descriptor, ToolDefinitionSchema));
    }
    const assets = new Map<string, z.infer<typeof AssetSchema>>();
    const executionWitnesses = new Map<string, z.infer<typeof ExecutionReceiptSchema>>();
    const projectionWitnesses: Record<string, Awaited<ReturnType<typeof indexedToolResultProjectionWitness>>> = {};
    if (includeDependencies) {
        if (!root.tool_call_state_complete)
            throw new Error('Indexed dependency projection has incomplete call indexes');
        // Required identities come only from selected canonical blocks, never a caller nomination.
        const selectedBlocks = new Map(
            turns.flatMap((turn) =>
                turn.selected_blocks.flatMap((block) =>
                    (block.type === 'tool_result' ? [block, ...block.content] : [block]).map(
                        (item) => [item.id, item] as const,
                    ),
                ),
            ),
        );
        const selectedTurns = new Set(turns.map((turn) => turn.header.id));
        const selectedCalls = new Map(
            turns.flatMap((turn) =>
                turn.selected_blocks.flatMap((block) =>
                    block.type === 'tool_call' ? [[block.call_id, { turn, block }] as const] : [],
                ),
            ),
        );
        const selectedResults = new Map(
            turns.flatMap((turn) =>
                turn.selected_blocks.flatMap((block) =>
                    block.type === 'tool_result' ? [[block.call_id, { turn, block }] as const] : [],
                ),
            ),
        );
        const readDependency = async <Schema extends z.ZodType>(
            family: keyof IndexedConversationDirectories,
            id: string,
            schema: Schema,
        ): Promise<z.output<Schema>> => {
            const descriptor = await getPagedRecord(store, root.directories[family], id);
            if (descriptor?.storage === 'record') reserve(descriptor.size_bytes, descriptor.content_hash);
            return loadRecord(store, descriptor, schema);
        };
        // Derived tool results are context projections only. Their sole terminal authority remains
        // the immutable original accepted result and execution receipt, never the derived IDs.
        // A committed processing witness shares descriptor checks within this root-pinned read.
        const readToolResultLineage = async (jobId: string) => {
            const job = await readDependency('processing_records', tupleKey('jobs', jobId), ProcessingJobSchema);
            const resolution = await readDependency(
                'processing_records',
                tupleKey('resolved_inputs', jobId),
                ProcessingResolvedInputSchema,
            );
            const attempt = await readDependency(
                'processing_records',
                tupleKey('attempts', jobId),
                ProcessingAttemptReceiptSchema,
            );
            const completion = await readDependency(
                'processing_records',
                tupleKey('completions', jobId),
                ProcessingCompletionReceiptSchema,
            );
            const validation = await readDependency(
                'processing_records',
                tupleKey('tool_result_validations', jobId),
                IndexedToolResultValidationSchema,
            );
            const output = await getPagedRecord(store, root.directories.processing_records, tupleKey('outputs', jobId));
            const originalSource = await getPagedRecord(
                store,
                root.directories.processing_records,
                tupleKey('tool_result_sources', jobId),
            );
            return { job, resolution, attempt, completion, validation, output, originalSource };
        };
        const toolResultLineages = new Map<string, ReturnType<typeof readToolResultLineage>>();
        const toolResultValidations = new Map<string, Promise<void>>();
        const readDerivedToolResultOriginal = async (
            result: NonNullable<ReturnType<typeof selectedResults.get>>,
            state: z.infer<typeof IndexedCallStateSchema>,
        ) => {
            const id = replacementIds.get(result.turn.header.id);
            const witness = id === undefined ? undefined : compactionWitnesses.get(id);
            if (
                !witness ||
                !isToolResultTextStrategy(witness.compaction.strategy.id, witness.compaction.strategy.version) ||
                !witness.acceptance.id.startsWith('processing:apply:') ||
                !state.result_block_id ||
                !state.terminal_receipt_id ||
                result.turn.completeness !== 'full_turn' ||
                result.turn.header.provenance.type !== 'derived'
            )
                throw new Error('Indexed derived tool result lacks its exact registered compaction lineage');
            const jobId = witness.acceptance.id.slice('processing:apply:'.length);
            let lineage = toolResultLineages.get(jobId);
            if (!lineage) {
                lineage = readToolResultLineage(jobId);
                toolResultLineages.set(jobId, lineage);
            }
            const { job, resolution, attempt, completion, validation, output, originalSource } = await lineage;
            const terminal = await readDependency(
                'execution_receipts',
                state.terminal_receipt_id,
                ExecutionReceiptSchema,
            );
            const terminalResultTurnId = terminal.result_turn_id;
            if (!terminalResultTurnId)
                throw new Error('Indexed derived tool result lacks its original terminal result turn');
            if (
                job.id !== jobId ||
                !isToolResultTextProcessor(job) ||
                job.processor_version !== witness.compaction.strategy.version ||
                job.configuration_fingerprint !== witness.compaction.strategy.configuration_fingerprint ||
                (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint ||
                (await fingerprintJson(job.selection)) !== job.selection_fingerprint ||
                resolution.job_id !== jobId ||
                resolution.source_fingerprint !== witness.compaction.source.source_fingerprint ||
                attempt.job_id !== jobId ||
                attempt.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
                output?.storage !== 'record' ||
                originalSource?.storage !== 'record' ||
                originalSource.kind !== 'processing_records' ||
                originalSource.id !== jobId ||
                validation.original_source_record_fingerprint !== originalSource.content_hash ||
                output.kind !== 'processing_records' ||
                output.id !== jobId ||
                output.content_hash !== validation.output_record_fingerprint ||
                validation.compaction_id !== witness.compaction.id ||
                witness.acceptance.payload_fingerprint !== validation.proposal_fingerprint ||
                completion.job_id !== jobId ||
                completion.status !== 'applied' ||
                completion.output_fingerprint !== validation.output_fingerprint ||
                completion.context_change_operation_id !== witness.acceptance.id ||
                completion.result_revision !== witness.acceptance.result_revision ||
                canonicalJsonContentString(completion.inserted_entry_ids) !==
                    canonicalJsonContentString(witness.acceptance.accepted_context_entry_ids ?? []) ||
                canonicalJsonContentString(resolution.entry_ids) !==
                    canonicalJsonContentString(witness.acceptance.context_change?.removed_entry_ids) ||
                witness.acceptance.context_change?.source_fingerprint !== resolution.source_fingerprint ||
                (job.processor_version === '3'
                    ? validation.original_result_bindings?.[result.turn.header.id]?.turn_id !== terminalResultTurnId ||
                      validation.original_result_bindings?.[result.turn.header.id]?.block_id !== state.result_block_id
                    : !resolution.source_turn_ids.includes(terminalResultTurnId) ||
                      !witness.compaction.source.turn_ids.includes(terminalResultTurnId) ||
                      !witness.compaction.source.block_ids?.includes(state.result_block_id)) ||
                result.turn.header.execution_id !== terminal.id ||
                terminal.call_id !== state.call_id
            )
                throw new Error(
                    'Indexed derived tool result changed its immutable output/completion/original terminal evidence',
                );
            if (
                validation.replacement_turn_fingerprints[result.turn.header.id] !==
                (await fingerprintJson(
                    ConversationTurnSchema.parse({ ...result.turn.header, blocks: result.turn.selected_blocks }),
                ))
            )
                throw new Error('Indexed derived tool result differs from its exact accepted context projection');
            if (
                validation.job_id !== jobId ||
                validation.job_fingerprint !== (await fingerprintJson(job)) ||
                validation.attempt_fingerprint !== (await fingerprintJson(attempt)) ||
                validation.resolved_input_fingerprint !== attempt.resolved_input_fingerprint ||
                validation.completion_fingerprint !== (await fingerprintJson(completion)) ||
                validation.compaction_fingerprint !== (await fingerprintJson(witness.compaction)) ||
                validation.acceptance_fingerprint !== (await fingerprintJson(witness.acceptance))
            )
                throw new Error('Indexed tool-result validation differs from its committed processing lineage');
            let validated = toolResultValidations.get(jobId);
            if (!validated) {
                validated = (async () => {
                    for (const dependency of validation.dependencies) {
                        const actual = await getPagedRecord(
                            store,
                            root.directories[dependency.family],
                            dependency.descriptor.id,
                        );
                        if (
                            actual === undefined ||
                            canonicalJsonContentString(actual) !== canonicalJsonContentString(dependency.descriptor)
                        )
                            throw new Error('Indexed tool-result validation changed its original record descriptor');
                    }
                })();
                toolResultValidations.set(jobId, validated);
            }
            await validated;
            const originalHeader = await readDependency(
                'turns',
                terminalResultTurnId,
                IndexedConversationTurnHeaderSchema,
            );
            const originalDescriptor = await getPagedRecord(store, root.directories.blocks, state.result_block_id);
            const bound = (family: 'turns' | 'blocks' | 'execution_receipts', recordId: string) =>
                validation.dependencies.find((item) => item.family === family && item.descriptor.id === recordId);
            if (
                originalHeader.source !== 'ordinary' ||
                originalHeader.turn.kind !== 'tool' ||
                originalHeader.turn.provenance.type === 'derived' ||
                originalHeader.turn.execution_id !== terminal.id ||
                originalHeader.block_ids.length !== 1 ||
                originalHeader.block_ids[0] !== state.result_block_id ||
                !bound('turns', terminalResultTurnId) ||
                !bound('blocks', state.result_block_id) ||
                !bound('execution_receipts', terminal.id) ||
                originalDescriptor?.storage !== 'record' ||
                originalDescriptor.content_hash !== terminal.result_fingerprint ||
                terminal.status !== result.block.status
            )
                throw new Error('Indexed derived tool result lost its accepted original result binding');
            if (purpose === 'processing')
                projectionWitnesses[result.turn.header.id] = {
                    compaction_id: witness.compaction.id,
                    compaction_fingerprint: await fingerprintJson(witness.compaction),
                    projection_fingerprint: validation.replacement_turn_fingerprints[result.turn.header.id],
                    terminal_execution_id: terminal.id,
                    original_result_turn_id: terminalResultTurnId,
                    original_result_block_id: state.result_block_id,
                };
            operationWitnesses.set(witness.acceptance.id, witness.acceptance);
            return {
                turn: { header: originalHeader.turn },
                block: {
                    type: 'tool_result' as const,
                    id: state.result_block_id,
                    call_id: terminal.call_id,
                    status: terminal.status,
                },
            };
        };
        for (const turn of turns) {
            if (
                (turn.header.provenance.type === 'derived' && !replacementIds.has(turn.header.id)) ||
                turn.header.provenance.type === 'imported'
            ) {
                throw new Error('Indexed dependency projection requires an explicit compaction/import witness');
            }
            for (const block of turn.selected_blocks) {
                const nested = block.type === 'tool_result' ? block.content : [block];
                for (const content of nested) {
                    if (
                        content.type === 'native_replay' &&
                        content.dependency_policy !== 'discard_on_dependency_change'
                    ) {
                        if (
                            content.dependencies.turn_ids.some((id) => !selectedTurns.has(id)) ||
                            content.dependencies.block_ids.some((id) => !selectedBlocks.has(id)) ||
                            content.dependencies.call_ids.some((id) => !selectedCalls.has(id)) ||
                            content.dependencies.request_ids.some(
                                (id) =>
                                    ![...generationWitnesses.values()].some(
                                        (witness) => witness.generation.request_id === id,
                                    ),
                            )
                        ) {
                            throw new Error('Indexed protected replay has an unavailable selected dependency witness');
                        }
                    }
                    if (content.type === 'external_reference') {
                        if (
                            !includeMediaCompaction ||
                            (content.original_type !== 'text' && content.original_type !== 'json')
                        )
                            throw new Error('Indexed dependency projection needs an externalization receipt witness');
                        const asset = await readDependency('assets', content.asset_id, AssetSchema);
                        const requirement = context.retrieval_requirements.filter(
                            (item) =>
                                item.asset_id === asset.id &&
                                canonicalJsonContentString(item.retrieval) ===
                                    canonicalJsonContentString(content.retrieval),
                        );
                        const definition = toolDefinitions.get(content.retrieval.tool_definition_id ?? '');
                        const operationId =
                            requirement.length === 1 ? requirement[0].accepted_asset_operation_id : undefined;
                        if (
                            !operationId ||
                            !definition ||
                            definition.name !== content.retrieval.capability ||
                            content.retrieval.version !== 1 ||
                            asset.id !== content.asset_id ||
                            asset.kind !== content.original_type ||
                            asset.storage.type !== 'external' ||
                            asset.content_hash !== content.content_hash ||
                            asset.byte_length === undefined ||
                            !asset.content_hash
                        )
                            throw new Error('Indexed external reference lacks its exact asset/read-tool binding');
                        const compactionId = replacementIds.get(turn.header.id);
                        if (
                            compactionId !== undefined &&
                            !compactionWitnesses.get(compactionId)?.compaction.retained_asset_ids.includes(asset.id)
                        )
                            throw new Error(
                                'Indexed replacement reference is absent from its retained compaction assets',
                            );
                        const acceptance = await readDependency(
                            'operation_receipts',
                            operationId,
                            OperationReceiptSchema,
                        );
                        const acceptedRequirement =
                            acceptance.accepted_retrieval_requirements?.filter(
                                (item) =>
                                    item.id === requirement[0]?.id &&
                                    item.asset_id === asset.id &&
                                    canonicalJsonContentString(item.retrieval) ===
                                        canonicalJsonContentString(content.retrieval),
                            ).length === 1;
                        // A compaction introduces a retrieval requirement after the original asset
                        // publication. Its accepted compaction and retained-asset witness, rather
                        // than the earlier asset receipt, proves that new requirement's selection.
                        const retainedCompactionRequirement =
                            compactionId !== undefined &&
                            compactionWitnesses.get(compactionId)?.compaction.retained_asset_ids.includes(asset.id) ===
                                true;
                        if (
                            acceptance.id !== operationId ||
                            acceptance.conversation_id !== root.source.conversation_id ||
                            acceptance.operation_kind !== undefined ||
                            acceptance.result_revision > root.source.revision ||
                            acceptance.accepted_asset_ids?.filter((id) => id === asset.id).length !== 1 ||
                            (!acceptedRequirement && !retainedCompactionRequirement)
                        )
                            throw new Error('Indexed external reference lacks its accepted asset publication');
                        assets.set(asset.id, asset);
                        operationWitnesses.set(acceptance.id, acceptance);
                    }
                    if ('asset_id' in content && !assets.has(content.asset_id)) {
                        const asset = await readDependency('assets', content.asset_id, AssetSchema);
                        if (
                            asset.id !== content.asset_id ||
                            asset.kind !== content.type ||
                            asset.provenance.type === 'derived'
                        ) {
                            throw new Error('Indexed selected media differs from its exact asset identity/provenance');
                        }
                        if (
                            asset.provenance.type === 'generated' &&
                            !generationWitnesses.has(asset.provenance.generation_id)
                        ) {
                            throw new Error('Indexed selected media lacks its accepted generation witness');
                        }
                        const integrity = await inlineAssetContentIntegrity(asset.storage);
                        const external =
                            includeMediaCompaction &&
                            asset.storage.type === 'external' &&
                            asset.content_hash !== undefined &&
                            asset.byte_length !== undefined;
                        if (
                            !external &&
                            (integrity === undefined ||
                                asset.content_hash !== integrity.content_hash ||
                                asset.byte_length !== integrity.byte_length)
                        ) {
                            throw new Error('Indexed selected media bytes differ from their immutable asset binding');
                        }
                        assets.set(asset.id, asset);
                    }
                }
                if (block.type !== 'tool_call' && block.type !== 'tool_result') continue;
                const call = selectedCalls.get(block.call_id);
                const result = selectedResults.get(block.call_id);
                if (!call) throw new Error('Indexed tool result has no selected accepted call');
                if (call.block.arguments.type === 'externalized_json') {
                    throw new Error('Indexed tool arguments require an exact hydration witness');
                }
                const callState = await readDependency('tool_call_states', block.call_id, IndexedCallStateSchema);
                if (
                    callState.call_id !== block.call_id ||
                    callState.turn_id !== call.turn.header.id ||
                    callState.block_id !== call.block.id ||
                    callState.call_fingerprint !==
                        (await hashContentBytes(canonicalJsonContentBytes(call.block))).content_hash ||
                    call.turn.header.kind !== 'agent' ||
                    call.turn.header.provenance.type !== 'generated' ||
                    !('generation_id' in call.turn.header) ||
                    call.turn.header.generation_id === undefined ||
                    !generationWitnesses.has(call.turn.header.generation_id)
                ) {
                    throw new Error('Indexed selected call differs from its accepted generation and exact call index');
                }
                if (call.block.definition_id !== undefined) {
                    let definition = toolDefinitions.get(call.block.definition_id);
                    if (!definition) {
                        definition = await readDependency(
                            'tool_definitions',
                            call.block.definition_id,
                            ToolDefinitionSchema,
                        );
                        toolDefinitions.set(definition.id, definition);
                    }
                    if (definition.id !== call.block.definition_id || definition.name !== call.block.tool_name) {
                        throw new Error('Indexed selected call differs from its pinned tool definition');
                    }
                }
                if (!result) {
                    // Append admission may contain an accepted call before the tool executes. It
                    // still requires an exact pending call index. Provider coverage/ready readers remain strict.
                    if (allowPendingCalls && !callState.result_block_id && !callState.terminal_receipt_id) continue;
                    throw new Error('Indexed selected call requires its exact selected terminal result');
                }
                const terminalResult = replacementIds.has(result.turn.header.id)
                    ? await readDerivedToolResultOriginal(result, callState)
                    : result;
                if (callState.result_block_id !== terminalResult.block.id || !callState.terminal_receipt_id) {
                    throw new Error('Indexed selected result lacks its exact terminal receipt index');
                }
                if (!executionWitnesses.has(callState.terminal_receipt_id)) {
                    const receipt = await readDependency(
                        'execution_receipts',
                        callState.terminal_receipt_id,
                        ExecutionReceiptSchema,
                    );
                    if (
                        receipt.id !== callState.terminal_receipt_id ||
                        receipt.call_id !== block.call_id ||
                        receipt.result_turn_id !== terminalResult.turn.header.id ||
                        receipt.status !== terminalResult.block.status ||
                        receipt.executor !== call.block.executor ||
                        (receipt.executor === 'application' && receipt.call_source === undefined) ||
                        (terminalResult.turn.header.execution_id !== undefined &&
                            terminalResult.turn.header.execution_id !== receipt.id) ||
                        (receipt.call_source !== undefined &&
                            (receipt.call_source.call_id !== block.call_id ||
                                receipt.call_source.turn_id !== call.turn.header.id ||
                                receipt.call_source.block_id !== call.block.id ||
                                receipt.call_source.call_fingerprint !== callState.call_fingerprint ||
                                receipt.call_source.conversation.conversation_id !== root.source.conversation_id ||
                                receipt.call_source.conversation.revision > root.source.revision))
                    ) {
                        throw new Error('Indexed selected result differs from its exact execution/source receipt');
                    }
                    if (!hasIndexedDeleteProfile(root)) {
                        throw new Error('Indexed selected result lacks a complete operation-acceptance index');
                    }
                    const accepted = await getPagedRecord(
                        store,
                        root.directories.turn_acceptances,
                        terminalResult.turn.header.id,
                    );
                    const operation =
                        accepted?.storage === 'marker' && accepted.kind === 'turn_acceptance'
                            ? await readDependency('operation_receipts', accepted.id, OperationReceiptSchema)
                            : undefined;
                    if (
                        !operation?.accepted_turn_ids?.includes(terminalResult.turn.header.id) ||
                        !operation.accepted_execution_receipt_ids?.includes(receipt.id) ||
                        operation.conversation_id !== root.source.conversation_id ||
                        operation.result_revision > root.source.revision
                    ) {
                        throw new Error('Indexed selected result lacks its accepted operation/receipt witness');
                    }
                    operationWitnesses.set(operation.id, operation);
                    if (!replacementIds.has(result.turn.header.id))
                        await assertToolResultReceiptFingerprint(result.block, receipt);
                    executionWitnesses.set(receipt.id, receipt);
                }
            }
        }
    }
    const tail = hasIndexedDeleteProfile(root)
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
    const selectedInput = {
        completeness: includeMediaCompaction
            ? 'selected_media_compaction_pending_admission'
            : includeDependencies
              ? 'selected_dependencies_pending_admission'
              : 'selected_text_pending_admission',
        source: root.source,
        root: rootLocator,
        source_turn_count: root.live_turn_count ?? root.turn_count,
        context,
        turns: turns.filter((turn) => !replacementIds.has(turn.header.id)),
        ...(includeMediaCompaction
            ? {
                  replacement_turns: turns.flatMap((projection) => {
                      const compaction_id = replacementIds.get(projection.header.id);
                      return compaction_id === undefined ? [] : [{ compaction_id, projection }];
                  }),
                  compaction_witnesses: Object.fromEntries(compactionWitnesses),
                  operation_witnesses: Object.fromEntries(operationWitnesses),
              }
            : {}),
        tool_definitions: Object.fromEntries(toolDefinitions),
        assets: Object.fromEntries(assets),
        ...(includeDependencies ? { execution_witnesses: Object.fromEntries(executionWitnesses) } : {}),
        generation_witnesses: Object.fromEntries(generationWitnesses),
        ...(tail === undefined ? {} : { source_tail_turn_id: tail.id }),
    };
    const selected =
        purpose === 'processing'
            ? IndexedProcessingSelectedContextSchema.parse({
                  ...selectedInput,
                  completeness: 'active_processing_dependencies_verified',
                  tool_result_projection_witnesses: projectionWitnesses,
              })
            : IndexedConversationSelectedContextSchema.parse(selectedInput);
    if (canonicalJsonContentBytes(selected).byteLength > maxSelectedBytes) {
        throw new RangeError('Indexed selected context exceeds the bounded working-set profile');
    }
    return selected;
}

/** Selected media/replacement projection with exact accepted witnesses; no byte custody or processing authority. */
export async function loadIndexedSelectedMediaCompactionContext(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    rootLocator: PagedRecordRef,
    maxSelectedBytes = INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
) {
    return IndexedConversationSelectedContextSchema.parse(
        await loadIndexedSelectedContext(store, root, rootLocator, maxSelectedBytes, true, true),
    );
}

/** Original strict text profile retained for historical prepared records. */
export async function loadIndexedSelectedTextContext(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
    maxSelectedBytes = INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
) {
    return IndexedConversationSelectedContextSchema.parse(
        await loadIndexedSelectedContext(store, root, locator, maxSelectedBytes, false),
    );
}

/** Selected tools/media plus exact point-looked-up dependencies; processing remains an explicit unsupported gate. */
export async function loadIndexedSelectedDependencyContext(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
    maxSelectedBytes = INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
) {
    return IndexedConversationSelectedContextSchema.parse(
        await loadIndexedSelectedContext(store, root, locator, maxSelectedBytes, true),
    );
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

/** A single ordinary received text turn may be the next indexed interaction input. */
export async function assertIndexedFreshReceivedTextInput(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    input: { operation_id: string; result_revision: number; response_operation_id: string },
): Promise<OperationReceipt> {
    const root = IndexedConversationRootSchema.parse(rootInput);
    const receipt = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        input.operation_id,
        OperationReceiptSchema,
    );
    if (
        !receipt ||
        receipt.operation_kind !== undefined ||
        receipt.result_revision !== root.source.revision ||
        receipt.result_revision !== input.result_revision ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.accepted_turn_ids?.length !== 1 ||
        receipt.accepted_context_entry_ids?.length !== 1 ||
        (receipt.accepted_generation_ids?.length ?? 0) !== 0 ||
        (receipt.accepted_asset_ids?.length ?? 0) !== 0 ||
        (receipt.accepted_execution_receipt_ids?.length ?? 0) !== 0 ||
        (receipt.accepted_tool_definition_ids?.length ?? 0) !== 0 ||
        receipt.accepted_tool_selection?.kind === 'replace'
    ) {
        throw new Error('Indexed fresh text preparation is not pinned to one received input');
    }
    const turn = await loadIndexedProjectedTurn(store, root, receipt.accepted_turn_ids[0]);
    const entry = await indexedRecordById(
        store,
        root,
        'context_entries',
        receipt.accepted_context_entry_ids[0],
        ContextEntrySchema,
    );
    if (
        turn.header.kind !== 'user' ||
        turn.header.authority !== 'ordinary' ||
        turn.header.status !== 'completed' ||
        turn.header.provenance.type !== 'received' ||
        turn.header.model_visibility !== 'include' ||
        turn.completeness !== 'full_turn' ||
        turn.selected_blocks.length === 0 ||
        turn.selected_blocks.some((block) => block.type !== 'text') ||
        entry?.type !== 'source_turn' ||
        entry.turn_id !== receipt.accepted_turn_ids[0] ||
        entry.block_ids !== undefined
    ) {
        throw new Error('Indexed fresh input is not an ordinary complete received text turn');
    }
    if (await getPagedRecord(store, root.directories.operation_receipts, input.response_operation_id)) {
        throw new Error('Indexed response is already accepted; use exact recovery instead of fresh inference');
    }
    return receipt;
}

/** Dispatch only the two explicitly supported materialized text-input provenances. */
export async function assertIndexedFreshTextInput(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    input: { operation_id: string; result_revision: number; response_operation_id: string },
): Promise<OperationReceipt> {
    const root = IndexedConversationRootSchema.parse(rootInput);
    const receipt = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        input.operation_id,
        OperationReceiptSchema,
    );
    if (receipt?.accepted_turn_ids?.length !== 1) {
        throw new Error('Indexed fresh text input has no single accepted turn');
    }
    const turn = await loadIndexedProjectedTurn(store, root, receipt.accepted_turn_ids[0]);
    if (turn.header.kind === 'program') return assertIndexedFreshProgramInput(store, root, input);
    if (turn.header.kind === 'user') return assertIndexedFreshReceivedTextInput(store, root, input);
    throw new Error('Indexed fresh text input has an unsupported turn role');
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
export async function loadIndexedAcceptedTurn(
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
    if (!hasIndexedDeleteProfile(root)) {
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
            ...(tombstone.deleted_turn.call_ids === undefined ? {} : { call_ids: tombstone.deleted_turn.call_ids }),
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
        !hasIndexedDeleteProfile(predecessor)
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
            predecessor.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2
                ? deletedContentIdentities(original.selected_blocks).block_ids
                : original.selected_blocks.map((block) => block.id),
            tombstone.deleted_turn.block_ids,
        ) ||
        !sameIndexedRecord(
            deletedContentIdentities(original.selected_blocks).call_ids,
            tombstone.deleted_turn.call_ids ?? [],
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
        throw new IndexedRecordAppendConflict(
            'operation_conflict',
            'Indexed append conflicts with its accepted operation',
        );
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
            throw new IndexedRecordAppendConflict(
                'operation_conflict',
                `Indexed append changes accepted ${family} identities`,
            );
        }
    }
    if (!sameIndexedRecord(batch.context_entries ?? [], receipt.accepted_context_entries ?? [])) {
        throw new IndexedRecordAppendConflict('operation_conflict', 'Indexed append changes accepted context entries');
    }
    const acceptedRequirements = receipt.accepted_retrieval_requirements ?? [];
    if (
        !sameIndexedRecord(idsOf(batch.retrieval_requirements), idsOf(acceptedRequirements)) ||
        !sameIndexedRecord(batch.retrieval_requirements ?? [], acceptedRequirements)
    ) {
        throw new IndexedRecordAppendConflict(
            'operation_conflict',
            'Indexed append changes accepted retrieval requirements',
        );
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
            throw new IndexedRecordAppendConflict('operation_conflict', 'Indexed append changes accepted turn');
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
                throw new IndexedRecordAppendConflict(
                    'operation_conflict',
                    `Indexed append changes accepted ${family} record`,
                );
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
    return stageIndexedRecordBatchOwned(rootInput, input, store, false);
}

/** Admit only one host-created application call, with an existing active definition. */
export async function stageIndexedProgramToolCall(
    rootInput: IndexedConversationRoot,
    input: IndexedRecordBatchCommand,
    store: IndexedConversationRecordStore,
): Promise<{ root: IndexedConversationRoot; locator?: PagedRecordRef; receipt: OperationReceipt; applied: boolean }> {
    if (!preflightJsonInput(input, { max_bytes: 512 * 1024 }).success)
        throw new IndexedRecordAppendValidationError('Indexed program call exceeds its bounded operation');
    const command = IndexedRecordBatchCommandSchema.parse(structuredClone(input));
    const root = IndexedConversationRootSchema.parse(rootInput);
    const turn = command.batch.turns?.[0];
    const call = turn?.blocks[0];
    const entry = command.batch.context_entries?.[0];
    if (
        Object.keys(command.batch).some((key) => key !== 'turns' && key !== 'context_entries') ||
        command.batch.turns?.length !== 1 ||
        command.batch.context_entries?.length !== 1 ||
        turn?.kind !== 'program' ||
        turn.authority !== 'ordinary' ||
        turn.status !== 'completed' ||
        turn.provenance.type !== 'inserted' ||
        turn.provenance.operation_id !== command.options.operation_id ||
        turn.timestamps.recorded_at !== command.options.recorded_at ||
        turn.parent_turn_id !== undefined ||
        turn.execution_id !== undefined ||
        turn.blocks.length !== 1 ||
        call?.type !== 'tool_call' ||
        call.executor !== 'application' ||
        call.definition_id === undefined ||
        call.native_id !== undefined ||
        call.arguments.type !== 'json' ||
        call.arguments.value === null ||
        typeof call.arguments.value !== 'object' ||
        Array.isArray(call.arguments.value) ||
        entry?.type !== 'source_turn' ||
        entry.turn_id !== turn.id ||
        entry.block_ids !== undefined
    )
        throw new IndexedRecordAppendValidationError('Indexed program call is not one honest inserted operation');
    if (command.options.payload_fingerprint !== (await fingerprintJson(command.batch)))
        throw new IndexedRecordAppendValidationError('Indexed program operation fingerprint differs');
    const prior = await getPagedRecord(store, root.directories.operation_receipts, command.options.operation_id);
    if (prior === undefined) await assertIndexedActiveCallDefinition(store, root, call);
    return stageIndexedRecordBatchOwned(root, command, store, true);
}

async function stageIndexedRecordBatchOwned(
    rootInput: IndexedConversationRoot,
    input: IndexedRecordBatchCommand,
    store: IndexedConversationRecordStore,
    allowProgramCall: boolean,
): Promise<{ root: IndexedConversationRoot; locator?: PagedRecordRef; receipt: OperationReceipt; applied: boolean }> {
    if (!preflightJsonInput(input, { max_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES }).success) {
        throw new Error('Indexed append command is not bounded JSON');
    }
    const parsed = IndexedRecordBatchCommandSchema.parse(input);
    const { batch, options } = parsed;
    const root = IndexedConversationRootSchema.parse(rootInput);
    if (parsed.conversation_id !== root.source.conversation_id)
        throw new IndexedRecordAppendConflict('conversation_identity_conflict', 'Indexed append conversation differs');
    const prior = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        options.operation_id,
        OperationReceiptSchema,
    );
    if (prior !== undefined) {
        await acceptedIndexedBatch(root, batch, options, prior, store);
        if (
            root.accepted_output_index_complete === true &&
            prior.accepted_turn_ids?.length === 1 &&
            prior.accepted_generation_ids?.length === 1
        ) {
            const nomination = await getPagedRecord(
                store,
                root.directories.accepted_output_order,
                indexedOrderedKey(prior.result_revision),
            );
            if (nomination?.storage !== 'marker' || nomination.kind !== 'accepted_output' || nomination.id !== prior.id)
                throw new Error('Indexed accepted history retry lacks its original ordered nomination');
        }
        return { root, receipt: prior, applied: false };
    }
    if (options.expected_revision !== root.source.revision)
        throw new IndexedRecordAppendConflict('revision_conflict', 'Indexed append revision conflict');
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
    if (processing.enabled && root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed enabled append requires a complete processing outbox index');
    if (processing.enabled) await assertIndexedCurrentPolicy(store, root, processing);
    const wholeExchange = processing.enabled && processing.processors[0]?.id === INDEXED_EXCHANGE_PROCESSOR_ID;
    const assetOnly = batch.assets !== undefined && Object.keys(batch).every((key) => key === 'assets');
    const processingArchive = processing.enabled && assetOnly && options.operation_id.startsWith('processing:archive:');
    const archiveJobId = processingArchive ? options.operation_id.slice('processing:archive:'.length) : undefined;
    const newTurns = new Map((batch.turns ?? []).map((turn) => [turn.id, turn]));
    const acceptedExchangePairs: {
        call_turn_id: string;
        call_block_id: string;
        result_turn_id: string;
        result_block_id: string;
    }[] = [];
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
    const retainedDefinitionIds = new Set<string>();
    const register = (id: string, kind: string, allowRetainedDefinition = false) => {
        if (globalIds.has(id)) throw new Error(`Indexed append duplicates ${id}`);
        globalIds.set(id, kind);
        if (allowRetainedDefinition) retainedDefinitionIds.add(id);
    };
    register(options.operation_id, 'operation receipt');
    for (const turn of batch.turns ?? []) {
        register(turn.id, 'turn');
        for (const block of turn.blocks) {
            register(block.id, 'block');
            if (block.type === 'tool_call') register(block.call_id, 'tool call');
            if (block.type === 'tool_result') {
                for (const nested of block.content) register(nested.id, 'block');
            }
        }
    }
    for (const generation of batch.generations ?? []) {
        register(generation.id, 'generation');
        if (generation.request_receipt) register(generation.request_receipt.id, 'request receipt');
    }
    const verifiedExternalAssetIds = new Set<string>();
    const verifyExternalAsset = async (asset: Asset): Promise<void> => {
        if (verifiedExternalAssetIds.has(asset.id)) return;
        if (!store.assertExternalAssetIntegrity)
            throw new IndexedRecordAppendValidationError('Indexed external media has no authenticated host custody');
        await store.assertExternalAssetIntegrity(structuredClone(asset));
        verifiedExternalAssetIds.add(asset.id);
    };
    for (const asset of batch.assets ?? []) {
        register(asset.id, 'asset');
    }
    for (const definition of batch.tool_definitions ?? []) register(definition.id, 'tool definition', true);
    for (const receipt of batch.execution_receipts ?? []) register(receipt.id, 'execution receipt');
    for (const entry of batch.context_entries ?? []) register(entry.id, 'context entry');
    for (const requirement of batch.retrieval_requirements ?? []) register(requirement.id, 'retrieval requirement');
    const readIdentities = async (ids: readonly string[]) => {
        const retained = new Map<string, PagedRecordValue>();
        const encoder = new TextEncoder();
        let offset = 0;
        while (offset < ids.length) {
            const first = offset;
            // Reserve the small immutable root/envelope overhead, and partition by actual JSON
            // bytes as well as key count. Long already-retained references keep their prior capacity.
            let bytes = 1024;
            while (offset < ids.length && offset - first < PAGED_RECORD_INDEX_MAX_BATCH_KEYS) {
                const next = encoder.encode(JSON.stringify(ids[offset])).byteLength + 1;
                if (offset > first && bytes + next > PAGED_RECORD_INDEX_MAX_BATCH_BYTES) break;
                bytes += next;
                offset++;
            }
            const values = await getPagedRecords(store, root.directories.identifiers, ids.slice(first, offset));
            for (const [id, value] of values) retained.set(id, value);
        }
        return retained;
    };
    const retainedIdentities = await readIdentities([...globalIds.keys()]);
    for (const [id, retained] of retainedIdentities) {
        if (!(retainedDefinitionIds.has(id) && retained.kind === 'tool definition'))
            throw new IndexedRecordAppendConflict('record_conflict', `Indexed append identity ${id} already exists`);
    }
    for (const asset of batch.assets ?? []) {
        // Even an unattached new asset is durable adoption, not previously accepted provenance.
        // Exact operation retries returned above without requiring another byte grant/read.
        if (asset.storage.type === 'external') await verifyExternalAsset(asset);
    }
    if (hasIndexedDeleteProfile(root)) {
        const historicalIds = new Set<string>();
        for (const item of batch.generations ?? []) {
            if (item.record_source !== 'executed') continue;
            for (const id of [
                ...item.request_receipt.item_mappings.map((mapping) => mapping.canonical_id),
                ...item.request_receipt.asset_versions.map((binding) => binding.asset_id),
            ])
                if (!globalIds.has(id)) historicalIds.add(id);
        }
        const retained = await readIdentities([...historicalIds]);
        for (const id of historicalIds) if (!retained.has(id)) globalIds.set(id, 'historical_reference');
    }
    const definition = async (id: string) =>
        newDefinitions.get(id) ?? indexedRecordById(store, root, 'tool_definitions', id, ToolDefinitionSchema);
    const asset = async (id: string) => newAssets.get(id) ?? indexedRecordById(store, root, 'assets', id, AssetSchema);
    const generation = async (id: string) =>
        newGenerations.get(id) ?? indexedRecordById(store, root, 'generations', id, GenerationSchema);
    const turnExists = async (id: string) =>
        newTurns.has(id) || (await getPagedRecord(store, root.directories.turns, id))?.storage === 'record';
    const callStates = new Map<string, IndexedCallState>();
    const callAcceptedRevisions = new Map<string, number>();
    const incomingBlockIds = new Set(
        (batch.turns ?? []).flatMap((turn) =>
            turn.blocks.flatMap((block) =>
                block.type === 'tool_result' ? [block.id, ...block.content.map((content) => content.id)] : [block.id],
            ),
        ),
    );
    const incomingCallIds = new Set(
        (batch.turns ?? []).flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : [])),
        ),
    );
    const readCall = async (id: string): Promise<IndexedCallState | undefined> => {
        if (callStates.has(id)) return callStates.get(id);
        if (!root.tool_call_state_complete) throw new Error('Indexed tool-call lookup is unavailable on this root');
        const existing = await indexedRecordById(store, root, 'tool_call_states', id, IndexedCallStateSchema);
        if (existing) {
            const retainedTurn = await loadIndexedProjectedTurn(store, root, existing.turn_id, [existing.block_id]);
            if (retainedTurn.header.kind === 'program') {
                const selected = await loadIndexedProgramToolCallSelection(store, root, {
                    conversation: root.source,
                    turn_id: existing.turn_id,
                    block_id: existing.block_id,
                    call_id: existing.call_id,
                    call_fingerprint: existing.call_fingerprint,
                });
                callAcceptedRevisions.set(id, selected.operation_receipt.result_revision);
                callStates.set(id, existing);
                return existing;
            }
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
            callAcceptedRevisions.set(id, receipt.result_revision);
        }
        if (existing) callStates.set(id, existing);
        return existing;
    };
    // Tool ingress owns each inserted turn through its execution receipt. The outer batch
    // operation hashes those turns and therefore cannot also supply their provenance ID.
    for (const [index, turn] of (batch.turns ?? []).entries()) {
        const execution = turn.execution_id === undefined ? undefined : newExecution.get(turn.execution_id);
        if (
            turn.provenance.type === 'derived' ||
            turn.provenance.type === 'imported' ||
            (turn.kind === 'user' && turn.provenance.type !== 'received') ||
            (turn.kind === 'agent' && turn.provenance.type !== 'generated' && turn.provenance.type !== 'received') ||
            (turn.kind === 'tool' &&
                turn.provenance.type !== 'received' &&
                !(
                    turn.provenance.type === 'inserted' &&
                    (turn.provenance.operation_id === options.operation_id ||
                        (turn.provenance.operation_id === turn.execution_id &&
                            execution?.executor === 'application' &&
                            execution.result_turn_id === turn.id &&
                            execution.call_source !== undefined &&
                            turn.blocks.length === 1 &&
                            turn.blocks[0]?.type === 'tool_result' &&
                            turn.blocks[0].call_id === execution.call_id &&
                            turn.blocks[0].status === execution.status))
                )) ||
            (turn.kind === 'program' &&
                (turn.provenance.type !== 'inserted' || turn.provenance.operation_id !== options.operation_id))
        )
            throw new IndexedRecordAppendValidationError('Indexed append does not support this turn provenance');
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
            if (block.type === 'external_reference') {
                throw new Error('Indexed append cannot validate this external reference');
            }
            if (block.type === 'native_replay') {
                const record =
                    turn.kind === 'agent' && 'generation_id' in turn && turn.generation_id !== undefined
                        ? newGenerations.get(turn.generation_id)
                        : undefined;
                if (record?.record_source !== 'executed') {
                    throw new Error('Indexed response replay lacks its exact executed generation/dependency receipt');
                }
                const request = record.request_receipt;
                if (
                    block.compatibility_scope.provider !== record.provider ||
                    block.protocol !== record.protocol ||
                    block.compatibility_scope.protocol !== record.protocol ||
                    block.compatibility_scope.adapter_version !== record.adapter_version ||
                    (block.compatibility_scope.model !== undefined &&
                        block.compatibility_scope.model !== record.requested_model) ||
                    block.dependencies.request_ids.some((id) => id !== record.request_id) ||
                    block.dependencies.turn_ids.some(
                        (id) =>
                            !newTurns.has(id) &&
                            !request.item_mappings.some(
                                (mapping) => mapping.kind === 'turn' && mapping.canonical_id === id,
                            ),
                    ) ||
                    block.dependencies.block_ids.some(
                        (id) =>
                            !incomingBlockIds.has(id) &&
                            !request.item_mappings.some(
                                (mapping) => mapping.kind === 'block' && mapping.canonical_id === id,
                            ),
                    ) ||
                    block.dependencies.call_ids.some(
                        (id) =>
                            !incomingCallIds.has(id) &&
                            !request.item_mappings.some(
                                (mapping) => mapping.kind === 'call' && mapping.canonical_id === id,
                            ),
                    )
                ) {
                    throw new Error('Indexed response replay lacks its exact executed generation/dependency receipt');
                }
            }
            if (block.type === 'tool_call') {
                const programCall = allowProgramCall && turn.kind === 'program';
                if (
                    !programCall &&
                    (turn.kind !== 'agent' ||
                        turn.provenance.type !== 'generated' ||
                        !('generation_id' in turn) ||
                        turn.generation_id === undefined ||
                        !newGenerations.has(turn.generation_id))
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
                callAcceptedRevisions.set(block.call_id, root.source.revision + 1);
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
                const readReceipt = (batch.execution_receipts ?? []).find(
                    (receipt) => receipt.call_id === block.call_id && receipt.result_turn_id === turn.id,
                );
                const selectedReadResult =
                    wholeExchange &&
                    block.content.length === 1 &&
                    block.content[0]?.type === 'text' &&
                    readReceipt?.metadata?.retrieval_excerpt !== undefined;
                const hasExternalReference = block.content.some((content) => content.type === 'external_reference');
                if (
                    wholeExchange &&
                    !selectedReadResult &&
                    hasExternalReference &&
                    !(
                        block.content.length === 1 &&
                        block.content[0]?.type === 'external_reference' &&
                        block.content[0].original_type === 'text'
                    )
                )
                    throw new Error('Indexed whole-exchange result requires one exact original archive reference');
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
                if (selectedReadResult && callBlock.tool_name !== 'read_artifact')
                    throw new Error('Indexed inline retrieval result is not the selected read tool');
                if (
                    block.content.length === 1 &&
                    block.content[0]?.type === 'external_reference' &&
                    block.content[0].original_type === 'text'
                ) {
                    acceptedExchangePairs.push({
                        call_turn_id: call.turn_id,
                        call_block_id: call.block_id,
                        result_turn_id: turn.id,
                        result_block_id: block.id,
                    });
                }
                for (const content of block.content) {
                    if (content.type === 'text' || content.type === 'json' || content.type === 'external_reference')
                        continue;
                    if (
                        content.type !== 'image' &&
                        content.type !== 'audio' &&
                        content.type !== 'video' &&
                        content.type !== 'document'
                    ) {
                        throw new Error('Indexed tool-result content dependency is unsupported');
                    }
                    const retained = await asset(content.asset_id);
                    if (content.selection !== undefined || !retained || retained.kind !== content.type) {
                        throw new IndexedRecordAppendValidationError(
                            'Indexed tool-result media lacks its exact whole-asset binding',
                        );
                    }
                    const integrity = await inlineAssetContentIntegrity(retained.storage);
                    const external =
                        retained.storage.type === 'external' &&
                        retained.content_hash !== undefined &&
                        retained.byte_length !== undefined;
                    if (external) await verifyExternalAsset(retained);
                    if (
                        !external &&
                        (!integrity ||
                            retained.content_hash !== integrity.content_hash ||
                            retained.byte_length !== integrity.byte_length)
                    ) {
                        throw new IndexedRecordAppendValidationError(
                            'Indexed tool-result media requires integrity-bound inline custody',
                        );
                    }
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
            const source = await indexedRecordById(store, root, 'assets', item.provenance.source_asset_id, AssetSchema);
            const job = archiveJobId ? await ownedIndexedProcessingJob(store, root, archiveJobId) : undefined;
            const selected = job?.selection.kind === 'entries' ? job.selection : undefined;
            const acceptedSource = job
                ? await indexedRecordById(
                      store,
                      root,
                      'operation_receipts',
                      job.source_operation_id,
                      OperationReceiptSchema,
                  )
                : undefined;
            const resultEntryId = selected?.entry_ids[1];
            const resultEntry = selected?.selected_entries?.[1];
            const resultBlockId = resultEntryId ? selected?.selected_block_ids?.[resultEntryId]?.[0] : undefined;
            const resultBlock = resultBlockId
                ? await indexedRecordById(store, root, 'blocks', resultBlockId, ContentBlockSchema)
                : undefined;
            const reference = resultBlock?.type === 'tool_result' ? resultBlock.content[0] : undefined;
            if (
                !processingArchive ||
                job?.processor_id !== INDEXED_EXCHANGE_PROCESSOR_ID ||
                job.processor_version !== INDEXED_EXCHANGE_PROCESSOR_VERSION ||
                !selected ||
                selected.entry_ids.length !== 2 ||
                resultBlock?.type !== 'tool_result' ||
                resultBlock.content.length !== 1 ||
                reference?.type !== 'external_reference' ||
                !source ||
                reference.asset_id !== source.id ||
                source.provenance.type !== 'received' ||
                resultEntry?.type !== 'source_turn' ||
                source.provenance.source_turn_id !== resultEntry.turn_id ||
                !acceptedSource?.accepted_asset_ids?.includes(source.id) ||
                source.kind !== 'text' ||
                source.storage.type !== 'external' ||
                !source.content_hash ||
                source.byte_length === undefined ||
                reference.content_hash !== source.content_hash ||
                item.kind !== source.kind ||
                item.mime_type !== source.mime_type ||
                item.content_hash !== source.content_hash ||
                item.byte_length !== source.byte_length ||
                item.provenance.transform_id !== 'conversation.archive_rehome' ||
                item.provenance.transform_version !== '1'
            ) {
                throw new Error('Indexed derived archive lacks its exact selected original and immutable bytes');
            }
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
                item.call_source.conversation.revision > root.source.revision ||
                (callAcceptedRevisions.get(item.call_id) ?? 0) > item.call_source.conversation.revision)
        )
            throw new IndexedRecordAppendValidationError('Indexed execution receipt source differs from retained call');
        if (item.result_turn_id) {
            const resultTurn = newTurns.get(item.result_turn_id);
            const resultBlock = resultTurn?.blocks.find(
                (block) =>
                    block.type === 'tool_result' && block.call_id === item.call_id && block.status === item.status,
            );
            if (resultBlock?.type !== 'tool_result') {
                throw new Error('Indexed execution receipt result turn is unavailable or mismatched');
            }
            if (
                item.executor === 'application' &&
                (item.call_source === undefined ||
                    resultTurn?.execution_id !== item.id ||
                    resultTurn.blocks.length !== 1)
            ) {
                throw new IndexedRecordAppendValidationError(
                    'Indexed application result requires its exact source/execution identity',
                );
            }
            if (item.metadata?.retrieval_excerpt) {
                if (!store.assertRetrievalExcerptIntegrity)
                    throw new IndexedRecordAppendValidationError(
                        'Indexed retrieval receipt requires authenticated asset byte verification',
                    );
                await store.assertRetrievalExcerptIntegrity(root, resultBlock, item);
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
    const references = (batch.turns ?? []).flatMap((turn) =>
        turn.blocks.flatMap((block) =>
            block.type === 'tool_result'
                ? block.content.flatMap((content) =>
                      content.type === 'external_reference' ? [{ turn, block, content }] : [],
                  )
                : [],
        ),
    );
    const matchedReferences = new Set<string>();
    for (const requirement of batch.retrieval_requirements ?? []) {
        const retainedAsset = newAssets.get(requirement.asset_id);
        const definitionId = requirement.retrieval.tool_definition_id;
        const selectedDefinition = definitionId ? await definition(definitionId) : undefined;
        const matching = references.filter(
            ({ content }) =>
                content.asset_id === requirement.asset_id &&
                canonicalJsonContentString(content.retrieval) === canonicalJsonContentString(requirement.retrieval),
        );
        const candidate = matching[0];
        const selectedBlocks = candidate && newSelections.get(candidate.turn.id);
        if (
            requirement.accepted_asset_operation_id !== options.operation_id ||
            !retainedAsset ||
            retainedAsset.kind !== 'text' ||
            retainedAsset.storage.type !== 'external' ||
            !retainedAsset.content_hash ||
            retainedAsset.byte_length === undefined ||
            context.retrieval_requirements.some(
                (retained) =>
                    retained.asset_id === requirement.asset_id &&
                    canonicalJsonContentString(retained.retrieval) ===
                        canonicalJsonContentString(requirement.retrieval),
            ) ||
            !selectedDefinition ||
            !activeToolIds.includes(selectedDefinition.id) ||
            selectedDefinition.name !== requirement.retrieval.capability ||
            requirement.retrieval.version !== 1 ||
            matching.length !== 1 ||
            !candidate ||
            candidate.turn.model_visibility !== 'include' ||
            !newSelections.has(candidate.turn.id) ||
            (selectedBlocks !== undefined && !selectedBlocks.has(candidate.block.id)) ||
            candidate.content.original_type !== 'text' ||
            candidate.content.content_hash !== retainedAsset.content_hash ||
            matchedReferences.has(candidate.content.id)
        ) {
            throw new IndexedRecordAppendValidationError(
                'Indexed retrieval requirement lacks exact selected definition, asset and source block',
            );
        }
        matchedReferences.add(candidate.content.id);
    }
    if (matchedReferences.size !== references.length) {
        throw new IndexedRecordAppendValidationError('Indexed accepted retrieval references and requirements differ');
    }
    const nextContext = ConversationContextSchema.parse({
        ...context,
        revision: nextRevision,
        entries: [...context.entries, ...(batch.context_entries ?? [])],
        retrieval_requirements: [...context.retrieval_requirements, ...(batch.retrieval_requirements ?? [])],
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
        ...((batch.retrieval_requirements?.length ?? 0) > 0
            ? { accepted_retrieval_requirements: [...(batch.retrieval_requirements ?? [])] }
            : {}),
        accepted_tool_selection:
            batch.active_tool_definition_ids === undefined
                ? { kind: 'unchanged' }
                : { kind: 'replace', definition_ids: [...batch.active_tool_definition_ids] },
    });
    const selection = eligibleProcessingAppendRecords(
        nextContext.entries,
        newTurns,
        receipt.accepted_context_entry_ids ?? [],
        nextContext.protected_entry_ids,
        processing.enabled,
    );
    // Match materialized append: archive publication feeds an existing job and must not recursively enqueue.
    if (processingArchive) {
        for (const asset of batch.assets ?? []) IndexedProcessingArchiveAssetSchema.parse(asset);
    }
    const processorIndices =
        processing.enabled && !assetOnly
            ? processing.processors.flatMap((processor, index) => (processor.scope === 'on_append' ? [index] : []))
            : [];
    const exchangeProcessor = processing.processors.find(
        (processor) =>
            processor.id === INDEXED_EXCHANGE_PROCESSOR_ID && processor.version === INDEXED_EXCHANGE_PROCESSOR_VERSION,
    );
    const exchangeSelections: {
        entry_ids: string[];
        selected_block_ids: Record<string, string[]>;
        selected_entries: z.infer<typeof ContextEntrySchema>[];
    }[] = [];
    if (processing.enabled && exchangeProcessor && acceptedExchangePairs.length > 0) {
        if (acceptedExchangePairs.length > MAX_PROCESSING_STAGES_PER_OPERATION)
            throw new Error('Indexed exchange append exceeds its bounded completed results');
        for (const pair of acceptedExchangePairs) {
            const callEntries = nextContext.entries.filter(
                (entry) =>
                    entry.type === 'source_turn' &&
                    entry.turn_id === pair.call_turn_id &&
                    (entry.block_ids === undefined || entry.block_ids.includes(pair.call_block_id)),
            );
            const resultEntries = nextContext.entries.filter(
                (entry) =>
                    entry.type === 'source_turn' &&
                    entry.turn_id === pair.result_turn_id &&
                    (entry.block_ids === undefined || entry.block_ids.includes(pair.result_block_id)),
            );
            const callEntry = callEntries[0];
            const resultEntry = resultEntries[0];
            if (
                !callEntry ||
                !resultEntry ||
                callEntries.length !== 1 ||
                resultEntries.length !== 1 ||
                nextContext.protected_entry_ids.includes(callEntry.id) ||
                nextContext.protected_entry_ids.includes(resultEntry.id) ||
                nextContext.entries.indexOf(callEntry) >= nextContext.entries.indexOf(resultEntry) ||
                !(receipt.accepted_context_entry_ids ?? []).includes(resultEntry.id)
            )
                throw new Error('Indexed exchange append lacks one active ordered unprotected executed call/result');
            exchangeSelections.push({
                entry_ids: [callEntry.id, resultEntry.id],
                selected_entries: [callEntry, resultEntry],
                selected_block_ids: {
                    [callEntry.id]: [pair.call_block_id],
                    [resultEntry.id]: [pair.result_block_id],
                },
            });
        }
    }
    const toolResultProcessor = processing.processors.find(
        (processor) => processor.scope === 'on_append' && isToolResultTextStrategy(processor.id, processor.version),
    );
    let toolResultEntryIds: string[] | undefined;
    if (processing.enabled && !assetOnly && toolResultProcessor) {
        const bytes = canonicalJsonContentBytes(root);
        const integrity = await hashContentBytes(bytes);
        const selected = await loadIndexedProcessingToolResultSelectedContext(store, root, {
            content_hash: integrity.content_hash,
            size_bytes: integrity.byte_length,
        });
        const frame = await indexedToolResultTextFrame(selected, activeIndexedContextWorkingSet(selected));
        const turns = new Map([...frame.turns, ...newTurns]);
        for (const receipt of newExecution.values()) {
            if (receipt.executor !== 'application' || !receipt.call_source || turns.has(receipt.call_source.turn_id))
                continue;
            const full = await loadIndexedProjectedTurn(store, root, receipt.call_source.turn_id);
            turns.set(full.header.id, ConversationTurnSchema.parse({ ...full.header, blocks: full.selected_blocks }));
        }
        const activeBlocks = new Map(frame.active_blocks);
        for (const entry of batch.context_entries ?? []) {
            const turn = newTurns.get(entry.turn_id);
            if (!turn) throw new Error('Indexed tool-result selection lost its accepted new turn');
            activeBlocks.set(
                entry.id,
                entry.block_ids === undefined
                    ? turn.blocks
                    : turn.blocks.filter((block) => entry.block_ids?.includes(block.id)),
            );
        }
        const nextFrame: ToolResultTextSelectionFrame = {
            source: { conversation_id: root.source.conversation_id, revision: nextRevision },
            context: nextContext,
            turns,
            active_blocks: activeBlocks,
            execution_receipts: { ...frame.execution_receipts, ...Object.fromEntries(newExecution) },
        };
        toolResultEntryIds = await eligibleToolResultTextWorkingEntries(
            nextFrame,
            receipt.accepted_context_entry_ids ?? [],
            {
                processor_id: toolResultProcessor.id,
                processor_version: toolResultProcessor.version,
                configuration: toolResultProcessor.config,
            },
        );
    }
    const acceptedJobs = await constructProcessingJobs({
        conversation_id: root.source.conversation_id,
        revision: nextRevision,
        source_operation_id: receipt.id,
        policy_revision: processing.policy_revision,
        processors: processing.processors,
        processor_indices: processorIndices,
        entry_ids: selection.entryIds,
        ...(toolResultEntryIds === undefined ? {} : { tool_result_entry_ids: toolResultEntryIds }),
        ...(exchangeSelections.length === 0 ? {} : { exchange_selections: exchangeSelections }),
        ...(selection.selectedBlockIds === undefined ? {} : { selected_block_ids: selection.selectedBlockIds }),
        ...(selection.selectedEntries === undefined ? {} : { selected_entries: selection.selectedEntries }),
    });
    for (const job of acceptedJobs) {
        if (globalIds.has(job.id) || (await getPagedRecord(store, root.directories.identifiers, job.id)))
            throw new Error('Indexed append processing job identity conflicts');
    }
    const directories = { ...root.directories };
    const fresh = new Map<keyof typeof directories, { key: string; value: PagedRecordValue }[]>();
    const nominate = (family: keyof typeof directories, key: string, value: PagedRecordValue) => {
        const commands = fresh.get(family) ?? [];
        commands.push({ key, value });
        fresh.set(family, commands);
    };
    const flushFamily = async (family: keyof typeof directories) => {
        const commands = fresh.get(family);
        if (!commands) return;
        await insertFreshIndexedRecords(store, directories, family, commands);
        fresh.delete(family);
    };
    const write = async (
        family: keyof typeof directories,
        id: string,
        value: unknown,
        mode: 'insert' | 'replace' = 'insert',
    ) => {
        const record = await stageRecord(store, family, id, value);
        if (mode === 'insert') nominate(family, id, record);
        else {
            await flushFamily(family);
            directories[family] = await putPagedRecord(store, directories[family], id, record, mode);
        }
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
        nominate('generation_acceptances', item.id, {
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
    if (
        root.accepted_output_index_complete === true &&
        receipt.accepted_generation_ids?.length === 1 &&
        receipt.accepted_turn_ids?.length === 1
    )
        await insertFreshIndexedRecords(store, directories, 'accepted_output_order', [
            {
                key: indexedOrderedKey(receipt.result_revision),
                value: { storage: 'marker', kind: 'accepted_output', id: receipt.id },
            },
        ]);
    if (root.processing_index_profile === INDEXED_CONVERSATION_PROCESSING_PROFILE) {
        if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2) {
            const dependencies: IndexedDeleteDependency[] = [];
            for (const job of acceptedJobs) {
                const targets = new Set(batch.turns?.map((turn) => turn.id) ?? []);
                if (job.selection.kind === 'entries')
                    for (const id of job.selection.entry_ids) {
                        const entry = nextContext.entries.find((value) => value.id === id);
                        if (!entry) throw new Error('Indexed new job lost its selected context dependency');
                        targets.add(entry.turn_id);
                    }
                for (const target_turn_id of targets)
                    dependencies.push({ target_turn_id, kind: 'job', owner_id: job.id });
            }
            await appendIndexedDeleteDependencies(store, directories, dependencies);
        }
        for (const job of acceptedJobs) {
            directories.processing_records = await putPagedRecord(
                store,
                directories.processing_records,
                tupleKey('jobs', job.id),
                await stageRecord(store, 'processing_records', job.id, job),
            );
            if (job.required)
                directories.processing_required = await putPagedRecord(store, directories.processing_required, job.id, {
                    storage: 'marker',
                    kind: 'processing_required',
                    id: job.id,
                });
            directories.processing_pending = await putPagedRecord(store, directories.processing_pending, job.id, {
                storage: 'marker',
                kind: 'processing_pending',
                id: job.id,
            });
            directories.identifiers = await putPagedRecord(store, directories.identifiers, job.id, {
                storage: 'marker',
                kind: 'processing_job',
                id: job.id,
            });
        }
        await write(
            'processing_by_operation',
            receipt.id,
            IndexedProcessingOperationJobsSchema.parse({
                version: 1,
                operation_id: receipt.id,
                receipt_fingerprint: await fingerprintJson(receipt),
                job_ids: acceptedJobs.map((job) => job.id),
            }),
        );
    }

    for (const [id, state] of callStates) {
        const retainedCall = await getPagedRecord(store, root.directories.tool_call_states, id);
        await write('tool_call_states', id, state, retainedCall ? 'replace' : 'insert');
        const retainedOpen = await getPagedRecord(store, root.directories.open_tool_calls, id);
        if (state.result_block_id !== undefined) {
            if (retainedOpen) {
                if (root.restart_index_profile === INDEXED_CONVERSATION_RESTART_PROFILE) {
                    const remaining = await removePagedRecord(store, directories.open_tool_calls, id);
                    if (!remaining.applied) throw new Error('Indexed terminal acceptance lost its exact open call');
                    if (remaining.root === undefined) delete directories.open_tool_calls;
                    else directories.open_tool_calls = remaining.root;
                } else {
                    directories.open_tool_calls = await putPagedRecord(
                        store,
                        directories.open_tool_calls,
                        id,
                        { storage: 'marker', kind: 'closed_tool_call', id },
                        'replace',
                    );
                }
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
    // All fresh record/marker families are nominated once; existing call-state replacements
    // retain their point-checked sequential semantics above.
    for (const family of [...fresh.keys()]) await flushFamily(family);
    const identifiers: { key: string; value: PagedRecordValue }[] = [];
    for (const [id, kind] of globalIds) {
        if (kind === 'tool definition' && (await getPagedRecord(store, root.directories.identifiers, id))) continue;
        identifiers.push({ key: id, value: { storage: 'marker', kind, id } });
    }
    await insertFreshIndexedRecords(store, directories, 'identifiers', identifiers);
    await insertFreshIndexedRecords(
        store,
        directories,
        'turn_order',
        (batch.turns ?? []).map((turn, index) => ({
            key: indexedOrderedKey(root.turn_count + index),
            value: { storage: 'marker', kind: 'turn_order', id: turn.id },
        })),
    );
    await insertFreshIndexedRecords(
        store,
        directories,
        'active_context_order',
        (batch.context_entries ?? []).map((entry, index) => ({
            key: indexedOrderedKey(context.entries.length + index),
            value: { storage: 'marker', kind: 'context_order', id: entry.id },
        })),
    );
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
    let nextProcessingHeader = root.processing_header;
    if (root.processing_index_profile === INDEXED_CONVERSATION_PROCESSING_PROFILE) {
        if (
            processing.job_count === undefined ||
            processing.unresolved_job_count === undefined ||
            processing.required_unresolved_job_count === undefined ||
            processing.required_job_count === undefined ||
            processing.required_blocked_job_count === undefined
        )
            throw new Error('Indexed processing profile has no complete counts');
        const increment = acceptedJobs.length;
        const requiredIncrement = acceptedJobs.filter((job) => job.required).length;
        const { coverage: _previousCoverage, ...retainedProcessing } = processing;
        const header = IndexedConversationProcessingHeaderSchema.parse({
            ...retainedProcessing,
            job_count: processing.job_count + increment,
            unresolved_job_count: processing.unresolved_job_count + increment,
            required_unresolved_job_count: processing.required_unresolved_job_count + requiredIncrement,
            required_job_count: processing.required_job_count + requiredIncrement,
        });
        const headerRecord = await stageRecord(store, 'processing_header', root.source.conversation_id, header);
        nextProcessingHeader = { content_hash: headerRecord.content_hash, size_bytes: headerRecord.size_bytes };
    }
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        ...(root.restart_index_profile === INDEXED_CONVERSATION_RESTART_PROFILE
            ? {
                  ...((receipt.accepted_generation_ids?.length ?? 0) > 0 ||
                  batch.turns?.some(indexedRawRestartResponseTurn)
                      ? { restart_response: { operation_id: receipt.id, result_revision: receipt.result_revision } }
                      : {}),
                  ...((receipt.accepted_execution_receipt_ids?.length ?? 0) > 0 ||
                  batch.turns?.some((turn) => turn.kind === 'tool')
                      ? { restart_tool_input: { operation_id: receipt.id, result_revision: receipt.result_revision } }
                      : {}),
              }
            : {}),
        source: { ...root.source, revision: nextRevision },
        turn_count: root.turn_count + (batch.turns?.length ?? 0),
        ...deleteIndex,
        updated_at: options.recorded_at,
        processing_header: nextProcessingHeader,
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
    const locator = { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes };
    // Staged immutable writes are not acceptance. An oversized active dependency closure must
    // fail before the host receives a publishable root/receipt or commits any enabled job.
    if (processing.enabled) await assertIndexedProcessingAcceptanceWorkingSet(store, nextRoot, locator);
    return { root: nextRoot, locator, receipt, applied: true };
}

/** Stage body-free logical deletion using only authenticated index point reads and the active context. */
/** A proven invalid delete nomination or unsupported dependency policy; storage failures remain unwrapped. */
export class IndexedConversationDeleteConflict extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'IndexedConversationDeleteConflict';
    }
}

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
    if (!hasIndexedDeleteProfile(root)) {
        throw new IndexedConversationDeleteConflict('Indexed root has no complete logical-delete witness profile');
    }
    if (command.source.conversation_id !== root.source.conversation_id) {
        throw new IndexedConversationDeleteConflict(
            'Indexed deletion conversation differs from its authenticated root',
        );
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
            throw new IndexedConversationDeleteConflict('Indexed deletion retry conflicts with its accepted receipt');
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
                    ...(witness.deleted_turn.call_ids === undefined ? {} : { call_ids: witness.deleted_turn.call_ids }),
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
        throw new IndexedConversationDeleteConflict('Indexed deletion source revision conflict');
    }
    if (await getPagedRecord(store, root.directories.identifiers, command.operation_id)) {
        throw new IndexedConversationDeleteConflict('Indexed deletion operation identity already exists');
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
    if (command.turn_ids.length > root.live_turn_count)
        throw new IndexedConversationDeleteConflict('Indexed deletion nominates more turns than the live source');
    if ((root.live_turn_count === 0) !== (root.active_tail_turn_id === null)) {
        throw new Error('Indexed deletion root has inconsistent live-turn witnesses');
    }
    if (new Set(command.turn_ids).size !== command.turn_ids.length) {
        throw new IndexedConversationDeleteConflict('Indexed deletion repeats a turn ID');
    }
    const selected = new Set(command.turn_ids);
    const context = await loadIndexedActiveContext(store, root);
    const excludedEntries = context.entries.filter((entry) => selected.has(entry.turn_id));
    if (excludedEntries.length > 0 && command.context_policy !== 'exclude')
        throw new IndexedConversationDeleteConflict('Indexed deletion requires prior active-context exclusion');
    if (excludedEntries.some((entry) => context.protected_entry_ids.includes(entry.id)))
        throw new IndexedConversationDeleteConflict('Indexed deletion cannot exclude protected entries');
    if (excludedEntries.length > 0 && context.cache_intent?.mode === 'required')
        throw new IndexedConversationDeleteConflict('Context change would invalidate required cache intent');
    const excludedEntryIds = new Set(excludedEntries.map((entry) => entry.id));
    const cacheIntent = cacheAfterContextRemoval(context, excludedEntryIds);
    if (excludedEntries.length > 0 && context.revision === Number.MAX_SAFE_INTEGER)
        throw new IndexedPresentationCapacityError('Indexed deletion context revision is exhausted');
    const refs: z.infer<typeof ConversationDeleteOperationSchema>['deleted_turns'] = [];
    const links = new Map<string, z.infer<typeof IndexedConversationTurnLinkSchema>>();
    let selectedBytes = 0;
    let selectedRecords = 0;
    const reserve = (bytes: number) => {
        selectedBytes += bytes;
        selectedRecords += 1;
        if (selectedBytes > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES || selectedRecords > 100_000) {
            throw new IndexedPresentationCapacityError(
                'Indexed deletion selected source exceeds its bounded working set',
            );
        }
    };
    let lastOrdinal = -1;
    for (const id of command.turn_ids) {
        const descriptor = await getPagedRecord(store, root.directories.turns, id);
        if (!descriptor || (descriptor.storage === 'marker' && descriptor.kind === 'deleted_turn'))
            throw new IndexedConversationDeleteConflict('Indexed deletion turn is unavailable or already deleted');
        if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2)
            await assertIndexedDeleteDependencies(store, root, id, selected);
        else if (await getPagedRecord(store, root.directories.deletion_blockers, id))
            throw new IndexedConversationDeleteConflict('Indexed v1 deletion has a retained dependent record');
        const link = await indexedRecordById(store, root, 'turn_links', id, IndexedConversationTurnLinkSchema);
        const header = await indexedRecordById(store, root, 'turns', id, IndexedConversationTurnHeaderSchema);
        const accepted = await getPagedRecord(store, root.directories.turn_acceptances, id);
        if (link && link.ordinal <= lastOrdinal)
            throw new IndexedConversationDeleteConflict('Indexed deletion turn IDs must follow source order');
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
        if (
            excludedEntries.some((entry) => entry.turn_id === id) &&
            context.retrieval_requirements.some((requirement) =>
                collectAssetIds(turn.selected_blocks).has(requirement.asset_id),
            )
        )
            throw new IndexedConversationDeleteConflict(
                'Indexed deletion cannot discard an unresolved retrieval obligation',
            );
        for (const block of turn.selected_blocks) {
            if (
                root.delete_index_profile !== INDEXED_CONVERSATION_DELETE_PROFILE_V2 &&
                ['tool_call', 'tool_result', 'native_replay'].includes(block.type)
            )
                throw new IndexedConversationDeleteConflict(
                    'Indexed v1 deletion cannot remove executed or replay-bearing tool facts',
                );
            if (block.type === 'tool_call') {
                const state = await indexedRecordById(
                    store,
                    root,
                    'tool_call_states',
                    block.call_id,
                    IndexedCallStateSchema,
                );
                if (
                    !state ||
                    state.turn_id !== id ||
                    state.block_id !== block.id ||
                    state.call_fingerprint !== (await fingerprintJson(block))
                )
                    throw new Error('Indexed deletion lost its retained original tool call state');
                if (!state.terminal_receipt_id)
                    throw new IndexedConversationDeleteConflict(
                        'Indexed deletion cannot remove an unresolved tool call',
                    );
                await loadIndexedTerminalCallWitness(store, root, state);
            }
            if (block.type === 'tool_result') {
                const state = await indexedRecordById(
                    store,
                    root,
                    'tool_call_states',
                    block.call_id,
                    IndexedCallStateSchema,
                );
                if (!state?.terminal_receipt_id || state.result_block_id !== block.id)
                    throw new Error('Indexed deletion result lost its terminal call/receipt binding');
                await loadIndexedTerminalCallWitness(store, root, state);
            }
        }
        refs.push({
            id,
            fingerprint: await fingerprintJson({ ...turn.header, blocks: turn.selected_blocks }),
            block_ids: deletedContentIdentities(turn.selected_blocks).block_ids,
            ...(deletedContentIdentities(turn.selected_blocks).call_ids.length === 0
                ? {}
                : { call_ids: deletedContentIdentities(turn.selected_blocks).call_ids }),
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
        ...(excludedEntries.length === 0
            ? {}
            : { excluded_context_entry_ids: excludedEntries.map((entry) => entry.id) }),
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
                (await getPagedRecord(store, directories.blocks, blockId)) === undefined ? 'insert' : 'replace',
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
        if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2) {
            const link = links.get(ref.id);
            if (!link) throw new Error('Deleted answer nomination lost its actual ordinal');
            const removed = await removePagedRecord(
                store,
                directories.display_answer_order,
                indexedDisplayAnswerKey(link.ordinal),
            );
            if (removed.root === undefined) delete directories.display_answer_order;
            else directories.display_answer_order = removed.root;
        }
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
    let contextLocator = root.context_header;
    let processingLocator = root.processing_header;
    if (excludedEntries.length > 0) {
        const nextContext = {
            ...context,
            revision: context.revision + 1,
            ...(context.cache_intent === undefined ? {} : { cache_intent: cacheIntent }),
            entries: context.entries.filter((entry) => !selected.has(entry.turn_id)),
        };
        const { entries, ...header } = nextContext;
        const nextHeader = await stageRecord(
            store,
            'context_header',
            root.source.conversation_id,
            IndexedConversationContextHeaderSchema.parse({
                ...header,
                active_entry_count: entries.length,
                active_entry_bytes: canonicalJsonContentBytes(entries).byteLength,
                context_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(nextContext))).content_hash,
            }),
        );
        contextLocator = { content_hash: nextHeader.content_hash, size_bytes: nextHeader.size_bytes };
        const order = await buildPagedRecordIndex(
            store,
            entries.map((entry, index) => ({
                key: indexedOrderedKey(index),
                value: { storage: 'marker' as const, kind: 'context_order', id: entry.id },
            })),
        );
        if (order === undefined) delete directories.active_context_order;
        else directories.active_context_order = order;
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
        const { coverage: _coverage, ...withoutCoverage } = processing;
        const processingRecord = await stageRecord(
            store,
            'processing_header',
            root.source.conversation_id,
            withoutCoverage,
        );
        processingLocator = { content_hash: processingRecord.content_hash, size_bytes: processingRecord.size_bytes };
    }
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        context_header: contextLocator,
        processing_header: processingLocator,
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

interface IndexedProcessingReadProfile {
    recordReads: number;
    pageReads: number;
    bytes: number;
}
function boundedIndexedProcessingReader(
    store: IndexedConversationRecordStore,
    profile?: IndexedProcessingReadProfile,
): IndexedConversationRecordStore {
    let pageReads = 0;
    let recordReads = 0;
    let bytes = 0;
    const charge = (size: number, family: 'page' | 'record') => {
        if (family === 'page') pageReads++;
        else recordReads++;
        bytes += size;
        if (profile) Object.assign(profile, { recordReads, pageReads, bytes });
        if (
            pageReads > INDEXED_PROCESSING_MAX_PAGE_READS ||
            recordReads > INDEXED_PROCESSING_MAX_RECORD_READS ||
            bytes > INDEXED_PROCESSING_MAX_IO_BYTES
        )
            throw new RangeError('Indexed processing inspection exceeds its working-set bound');
    };
    const read = store.read;
    const readRecord = store.readRecord;
    const write = store.write;
    const writeRecord = store.writeRecord;
    const assertIntegrity = store.assertExternalAssetIntegrity;
    const assertRetrieval = store.assertRetrievalExcerptIntegrity;
    // The cache belongs to this single operation. Integrity-addressed keys may share immutable
    // bytes, while fresh copies prevent either the underlying store or a caller mutating cache data.
    const pages = new Map<string, Promise<Uint8Array>>();
    const records = new Map<string, Promise<Uint8Array>>();
    const cachedRead = async (
        ref: PagedRecordRef,
        family: 'page' | 'record',
        cache: Map<string, Promise<Uint8Array>>,
        load: () => Promise<Uint8Array>,
    ) => {
        const key = `${ref.content_hash}:${ref.size_bytes}`;
        let pending = cache.get(key);
        if (!pending) {
            charge(ref.size_bytes, family);
            pending = load().then((bytes) => Uint8Array.from(bytes));
            cache.set(key, pending);
        }
        return Uint8Array.from(await pending);
    };
    return {
        ...(assertIntegrity === undefined
            ? {}
            : { assertExternalAssetIntegrity: (asset) => assertIntegrity.call(store, asset) }),
        ...(assertRetrieval === undefined
            ? {}
            : {
                  assertRetrievalExcerptIntegrity: (source, result, receipt) =>
                      assertRetrieval.call(store, source, result, receipt),
              }),
        write(bytes, ref) {
            return write.call(store, bytes, ref);
        },
        writeRecord(ref, bytes) {
            return writeRecord.call(store, ref, bytes);
        },
        async read(ref) {
            return cachedRead(ref, 'page', pages, () => read.call(store, ref));
        },
        async readRecord(ref) {
            return cachedRead(ref, 'record', records, () => readRecord.call(store, ref));
        },
    };
}

/** Reserve conservative completion headroom before acceptance. A selected text block may add
 * an archive asset, replacement block and replacement header while unrelated active dependencies remain selected.
 * Phase/archive/compaction receipts also need point reads. Limits apply simultaneously; the block
 * ceiling is not a promise that every maximum-size shape fits every other resource ceiling. */
async function assertIndexedProcessingAcceptanceWorkingSet(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
): Promise<void> {
    const profile = { recordReads: 0, pageReads: 0, bytes: 0 };
    const selected = IndexedProcessingSelectedContextSchema.parse(
        await loadIndexedSelectedContext(
            boundedIndexedProcessingReader(store, profile),
            root,
            locator,
            INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
            true,
            true,
            'processing',
            true,
        ),
    );
    const textCount = selected.turns.reduce(
        (count, turn) =>
            count +
            (turn.header.authority === 'ordinary' && (turn.header.kind === 'user' || turn.header.kind === 'agent')
                ? turn.selected_blocks.filter((block) => block.type === 'text').length
                : 0),
        0,
    );
    // The shared indexed archive schema bounds each asset and retrieval binding to two KiB.
    // Sixteen KiB per text also reserves replacement/provenance/index/receipt evidence. The exact
    // claim/workspace and output still undergo their independent 32MiB validation.
    const completionRecordReserve = textCount === 0 ? 16 : 16 + textCount * 3;
    const completionByteReserve = 32 * 1024 + textCount * 16 * 1024;
    if (
        profile.recordReads + completionRecordReserve > INDEXED_PROCESSING_MAX_RECORD_READS ||
        profile.bytes + completionByteReserve > INDEXED_PROCESSING_MAX_IO_BYTES ||
        canonicalJsonContentBytes(selected).byteLength + completionByteReserve > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES
    )
        throw new RangeError('Indexed processing active dependency closure has no bounded completion headroom');
}

async function indexedProcessingRecord<Shape extends z.ZodType>(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    family: string,
    jobId: string,
    schema: Shape,
): Promise<z.infer<Shape> | undefined> {
    const record = await getPagedRecord(store, root.directories.processing_records, tupleKey(family, jobId));
    if (record === undefined) return undefined;
    if (record.storage !== 'record' || record.kind !== 'processing_records' || record.id !== jobId)
        throw new Error('Indexed processing record identity differs from its exact family/key');
    return loadRecord(store, record, schema);
}

export async function ownedIndexedProcessingJob(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    jobId: string,
): Promise<ProcessingJob> {
    const job = await indexedProcessingRecord(store, root, 'jobs', jobId, ProcessingJobSchema);
    if (!job || job.id !== jobId || job.enqueue_revision > root.source.revision)
        throw new Error('Indexed pending job lacks its retained canonical source');
    const acceptance = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        job.source_operation_id,
        OperationReceiptSchema,
    );
    const relation = await indexedRecordById(
        store,
        root,
        'processing_by_operation',
        job.source_operation_id,
        IndexedProcessingOperationJobsSchema,
    );
    if (
        !acceptance ||
        !relation ||
        acceptance.conversation_id !== root.source.conversation_id ||
        relation.operation_id !== acceptance.id ||
        relation.receipt_fingerprint !== (await fingerprintJson(acceptance)) ||
        relation.job_ids.filter((id) => id === job.id).length !== 1 ||
        acceptance.result_revision !== job.enqueue_revision ||
        job.configuration_fingerprint !== (await fingerprintJson(job.configuration)) ||
        job.selection_fingerprint !== (await fingerprintJson(job.selection))
    )
        throw new Error('Indexed processing job differs from its accepted operation/configuration/selection');
    if (acceptance.processing_operation?.phase === 'queue') {
        const command = await indexedProcessingRecord(
            store,
            root,
            'selected_queue_commands',
            acceptance.id,
            IndexedProcessingQueueCommandSchema,
        );
        if (command === undefined) {
            const materialized = await indexedProcessingRecord(
                store,
                root,
                'materialized_queue_commands',
                acceptance.id,
                IndexedMaterializedProcessingQueueSchema,
            );
            const selector = materialized?.selection.selector;
            const captured = acceptance.processing_operation.queue_command;
            if (
                !materialized ||
                job.stage_index !== 0 ||
                relation.job_ids.length !== 1 ||
                (materialized.legacy_reconstructed
                    ? !isToolResultTextProcessor(job) ||
                      !supportsToolResultTextProcessingScope(job) ||
                      job.scope !== 'manual' ||
                      selector?.source.kind !== 'turn_ids' ||
                      selector.filters !== undefined ||
                      captured !== undefined ||
                      acceptance.processing_operation.job_id !== undefined
                    : !captured ||
                      acceptance.processing_operation.job_id !== job.id ||
                      !sameIndexedRecord(captured, {
                          command: materialized.command,
                          selection: materialized.selection,
                      }) ||
                      !sameIndexedRecord(
                          acceptance.processing_operation.queue_selected_entries,
                          materialized.selected_entries,
                      )) ||
                acceptance.processing_operation.policy_revision !== job.policy_revision ||
                job.selection.kind !== 'entries' ||
                materialized.command.operation_id !== acceptance.id ||
                materialized.command.expected_revision !== acceptance.base_revision ||
                materialized.command.recorded_at !== acceptance.recorded_at ||
                materialized.command.processor_id !== job.processor_id ||
                materialized.command.scope !== job.scope ||
                materialized.command.target_fingerprint !== job.target_fingerprint ||
                materialized.selection.conversation.conversation_id !== root.source.conversation_id ||
                materialized.selection.conversation.revision !== acceptance.base_revision ||
                !sameIndexedRecord(
                    materialized.selected_entries.map((entry) => entry.id),
                    job.selection.entry_ids,
                ) ||
                (materialized.legacy_reconstructed &&
                    selector?.source.kind === 'turn_ids' &&
                    !sameIndexedRecord(
                        materialized.selected_entries.map((entry) => entry.turn_id),
                        selector.source.turn_ids,
                    )) ||
                acceptance.payload_fingerprint !==
                    (await fingerprintJson({ command: materialized.command, selection: materialized.selection }))
            )
                throw new Error('Indexed materialized job lost its exact accepted queue command');
            for (const entry of materialized.selected_entries) {
                const descriptor = await getPagedRecord(store, root.directories.context_entries, entry.id);
                if (descriptor?.storage !== 'record' || descriptor.content_hash !== (await fingerprintJson(entry)))
                    throw new Error('Indexed materialized queue changed its retained selected entry binding');
            }
            return job;
        }
        if (
            !command ||
            command.operation_id !== acceptance.id ||
            command.expected_revision !== acceptance.base_revision ||
            command.processor_id !== job.processor_id ||
            command.scope !== job.scope ||
            command.target_fingerprint !== job.target_fingerprint ||
            acceptance.processing_operation.job_id !== job.id ||
            acceptance.payload_fingerprint !== (await fingerprintJson(command)) ||
            job.selection.kind !== 'entries' ||
            canonicalJsonContentString(command.selected_entry_ids) !==
                canonicalJsonContentString(job.selection.entry_ids) ||
            canonicalJsonContentString(command.selected_block_ids ?? null) !==
                canonicalJsonContentString(job.selection.selected_block_ids ?? null)
        )
            throw new Error('Indexed selected job lost its exact accepted queue command');
    }
    return job;
}

/** A point-addressed accepted policy epoch, never a processor list reconstructed from job data.
 * Genesis pins the genuine immutable original header; no policy operation is invented for it. */
export const IndexedProcessingPolicyEpochSchema = z.discriminatedUnion('kind', [
    z.strictObject({
        version: z.literal(1),
        kind: z.literal('materialized_genesis'),
        policy_revision: z.literal(0),
        operation_id: IdentifierSchema,
        receipt_fingerprint: ContentHashSchema,
    }),
    z.strictObject({
        version: z.literal(1),
        kind: z.literal('accepted_command'),
        policy_revision: NonnegativeSafeIntegerSchema,
        operation_id: IdentifierSchema,
        receipt_fingerprint: ContentHashSchema,
    }),
    z.strictObject({
        version: z.literal(1),
        kind: z.literal('genesis'),
        policy_revision: z.literal(0),
        source: ConversationRefSchema,
        processing_header: PagedRecordRefSchema,
        successor_policy_operation_id: IdentifierSchema.optional(),
        successor_policy_receipt_fingerprint: ContentHashSchema.optional(),
    }),
]);

export async function assertIndexedProcessingPolicyAcceptance(
    source: z.infer<typeof ConversationRefSchema>,
    command: z.infer<typeof ProcessingPolicyCommandSchema>,
    receipt: OperationReceipt,
): Promise<void> {
    if (
        command.operation_id !== receipt.id ||
        receipt.conversation_id !== source.conversation_id ||
        receipt.operation_kind !== 'processing' ||
        receipt.processing_operation?.phase !== 'policy' ||
        receipt.base_revision !== command.expected_revision ||
        receipt.result_revision !== command.expected_revision + 1 ||
        receipt.result_revision > source.revision ||
        receipt.recorded_at !== command.recorded_at ||
        receipt.payload_fingerprint !== (await fingerprintJson(command)) ||
        !sameIndexedRecord(receipt.processing_operation.superseded_job_ids ?? [], command.supersede_job_ids ?? []) ||
        (receipt.processing_operation.policy_command !== undefined &&
            !sameIndexedRecord(receipt.processing_operation.policy_command, command))
    )
        throw new Error('Indexed retained policy command lost its exact accepted receipt');
    const genesis = receipt.processing_operation?.policy_genesis;
    if (
        genesis !== undefined &&
        (receipt.processing_operation?.policy_revision !== 0 ||
            genesis.source.conversation_id !== receipt.conversation_id ||
            genesis.source.revision !== receipt.base_revision ||
            genesis.policy.policy_revision !== 0)
    )
        throw new Error('Indexed materialized genesis changed its genuine predecessor source');
}

/** Explicit upgrade/native publication helper. One accepted command supplies one exact epoch. */
export async function stageIndexedAcceptedProcessingPolicy(
    store: IndexedConversationRecordStore,
    source: z.infer<typeof ConversationRefSchema>,
    directoriesInput: IndexedConversationDirectories,
    command: z.infer<typeof ProcessingPolicyCommandSchema>,
    receipt: OperationReceipt,
): Promise<IndexedConversationDirectories> {
    await assertIndexedProcessingPolicyAcceptance(source, command, receipt);
    const oldPolicyRevision = receipt.processing_operation?.policy_revision;
    if (oldPolicyRevision === undefined) throw new Error('Indexed policy acceptance has no genuine epoch');
    const epoch = IndexedProcessingPolicyEpochSchema.parse({
        version: 1,
        kind: 'accepted_command',
        policy_revision: oldPolicyRevision + 1,
        operation_id: receipt.id,
        receipt_fingerprint: await fingerprintJson(receipt),
    });
    const directories = { ...directoriesInput };
    for (const [family, id, value] of [
        ['selected_policy_commands', receipt.id, command],
        ['policy_epochs', String(epoch.policy_revision), epoch],
    ] as const) {
        const key = tupleKey(family, id);
        const existing = await getPagedRecord(store, directories.processing_records, key);
        const hash = await fingerprintJson(value);
        if (
            existing !== undefined &&
            (existing.storage !== 'record' ||
                existing.kind !== 'processing_records' ||
                existing.id !== id ||
                existing.content_hash !== hash)
        )
            throw new Error('Indexed accepted policy epoch has conflicting immutable evidence');
        if (existing === undefined)
            directories.processing_records = await putPagedRecord(
                store,
                directories.processing_records,
                key,
                await stageRecord(store, 'processing_records', id, value),
            );
    }
    if (receipt.processing_operation?.policy_genesis !== undefined) {
        const genesis = IndexedProcessingPolicyEpochSchema.parse({
            version: 1,
            kind: 'materialized_genesis',
            policy_revision: 0,
            operation_id: receipt.id,
            receipt_fingerprint: await fingerprintJson(receipt),
        });
        const key = tupleKey('policy_epochs', '0');
        const existing = await getPagedRecord(store, directories.processing_records, key);
        if (
            existing !== undefined &&
            (existing.storage !== 'record' ||
                existing.kind !== 'processing_records' ||
                existing.id !== '0' ||
                existing.content_hash !== (await fingerprintJson(genesis)))
        )
            throw new Error('Indexed materialized genesis has conflicting immutable evidence');
        if (existing === undefined)
            directories.processing_records = await putPagedRecord(
                store,
                directories.processing_records,
                key,
                await stageRecord(store, 'processing_records', '0', genesis),
            );
    }
    return directories;
}

/** Bounded exact policy epoch audit, independent of current readiness counters. */
export async function auditIndexedProcessingPolicyEpoch(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    policyRevision: number,
) {
    let epoch = await indexedProcessingRecord(
        store,
        root,
        'policy_epochs',
        String(policyRevision),
        IndexedProcessingPolicyEpochSchema,
    );
    if (epoch === undefined) {
        const current = await loadRecord(
            store,
            {
                storage: 'record',
                kind: 'processing_header',
                id: root.source.conversation_id,
                ...root.processing_header,
            },
            IndexedConversationProcessingHeaderSchema,
        );
        if (policyRevision === 0 && current.policy_revision === 0 && current.selected_policy_operation_id === undefined)
            epoch = IndexedProcessingPolicyEpochSchema.parse({
                version: 1,
                kind: 'genesis',
                policy_revision: 0,
                source: root.source,
                processing_header: root.processing_header,
            });
        else if (current.policy_revision === policyRevision && current.selected_policy_operation_id !== undefined) {
            const receipt = await indexedRecordById(
                store,
                root,
                'operation_receipts',
                current.selected_policy_operation_id,
                OperationReceiptSchema,
            );
            if (!receipt) throw new Error('Indexed current policy has no genuine acceptance');
            epoch = IndexedProcessingPolicyEpochSchema.parse({
                version: 1,
                kind: 'accepted_command',
                policy_revision: policyRevision,
                operation_id: receipt.id,
                receipt_fingerprint: await fingerprintJson(receipt),
            });
        }
    }
    if (epoch === undefined || epoch.policy_revision !== policyRevision)
        throw new Error('Indexed historical policy epoch requires authenticated original policy evidence');
    if (epoch.kind === 'materialized_genesis') {
        const receipt = await indexedRecordById(
            store,
            root,
            'operation_receipts',
            epoch.operation_id,
            OperationReceiptSchema,
        );
        const command = await indexedProcessingRecord(
            store,
            root,
            'selected_policy_commands',
            epoch.operation_id,
            ProcessingPolicyCommandSchema,
        );
        const genesis = receipt?.processing_operation?.policy_genesis;
        if (
            !receipt ||
            !command ||
            !genesis ||
            epoch.receipt_fingerprint !== (await fingerprintJson(receipt)) ||
            receipt.processing_operation?.policy_revision !== 0
        )
            throw new Error('Indexed materialized genesis lost its exact accepted first policy receipt');
        await assertIndexedProcessingPolicyAcceptance(root.source, command, receipt);
        return { ...genesis.policy, first_transition_revision: receipt.base_revision };
    }
    if (epoch.kind === 'genesis') {
        if (
            epoch.source.conversation_id !== root.source.conversation_id ||
            epoch.source.revision > root.source.revision
        )
            throw new Error('Indexed genesis policy witness belongs to a different accepted source');
        const header = await loadRecord(
            store,
            {
                storage: 'record',
                kind: 'processing_header',
                id: root.source.conversation_id,
                ...epoch.processing_header,
            },
            IndexedConversationProcessingHeaderSchema,
        );
        if (header.policy_revision !== 0 || header.selected_policy_operation_id !== undefined)
            throw new Error('Indexed genesis policy witness is not the genuine initial header');
        let firstTransitionRevision: number | undefined;
        if (
            epoch.successor_policy_operation_id !== undefined ||
            epoch.successor_policy_receipt_fingerprint !== undefined
        ) {
            if (!epoch.successor_policy_operation_id || !epoch.successor_policy_receipt_fingerprint)
                throw new Error('Indexed recovered genesis has incomplete first-policy evidence');
            const receipt = await indexedRecordById(
                store,
                root,
                'operation_receipts',
                epoch.successor_policy_operation_id,
                OperationReceiptSchema,
            );
            const command = await indexedProcessingRecord(
                store,
                root,
                'selected_policy_commands',
                epoch.successor_policy_operation_id,
                ProcessingPolicyCommandSchema,
            );
            if (
                !receipt ||
                !command ||
                receipt.processing_operation?.policy_revision !== 0 ||
                receipt.base_revision !== epoch.source.revision ||
                epoch.successor_policy_receipt_fingerprint !== (await fingerprintJson(receipt))
            )
                throw new Error('Indexed recovered genesis changed its exact accepted first-policy source');
            await assertIndexedProcessingPolicyAcceptance(root.source, command, receipt);
            firstTransitionRevision = receipt.base_revision;
        }
        return {
            policy_revision: 0,
            enabled: header.enabled,
            processors: header.processors,
            ...(firstTransitionRevision === undefined ? {} : { first_transition_revision: firstTransitionRevision }),
            ...(header.budget === undefined ? {} : { budget: header.budget }),
        };
    }
    const command = await indexedProcessingRecord(
        store,
        root,
        'selected_policy_commands',
        epoch.operation_id,
        ProcessingPolicyCommandSchema,
    );
    const receipt = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        epoch.operation_id,
        OperationReceiptSchema,
    );
    if (
        !command ||
        !receipt ||
        epoch.receipt_fingerprint !== (await fingerprintJson(receipt)) ||
        receipt.processing_operation?.policy_revision === undefined ||
        receipt.processing_operation.policy_revision + 1 !== policyRevision
    )
        throw new Error('Indexed historical policy epoch differs from its exact accepted command');
    await assertIndexedProcessingPolicyAcceptance(root.source, command, receipt);
    return {
        policy_revision: policyRevision,
        enabled: command.enabled,
        processors: command.processors,
        accepted_revision: receipt.result_revision,
        ...(command.budget === undefined ? {} : { budget: command.budget }),
    };
}

/** Point-read exact canonical processing evidence. This is a bounded data reader, not a job claim,
 * processor capability or provider admission. Hosts still prove run/namespace/current-task custody.
 */
export async function loadIndexedProcessingJobState(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    jobIdInput: string,
) {
    const input = { root: rootInput, job_id: jobIdInput };
    if (!preflightJsonInput(input).success) throw new TypeError('Indexed job inspection is not bounded JSON');
    const { root, job_id: jobId } = z
        .strictObject({ root: IndexedConversationRootSchema, job_id: IdentifierSchema })
        .parse(structuredClone(input));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed job inspection requires complete processing indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const job = await ownedIndexedProcessingJob(store, root, jobId);
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    assertIndexedProcessingCounts(header);
    return auditIndexedProcessingJobEvidence(store, root, job, header);
}

/** Shared immutable evidence audit only. This does not certify complete indexes, counters or readiness.
 * The ordinary inspection entry point validates those independently before invoking this function.
 */
export async function auditIndexedProcessingJobEvidence(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    job: ProcessingJob,
    header: z.infer<typeof IndexedConversationProcessingHeaderSchema>,
) {
    const jobId = job.id;
    const policy = await auditIndexedProcessingPolicyEpoch(store, root, job.policy_revision);
    const configuration = policy.processors[job.processor_index];
    if (
        !policy.enabled ||
        policy.policy_revision > header.policy_revision ||
        ('first_transition_revision' in policy &&
            policy.first_transition_revision !== undefined &&
            job.enqueue_revision > policy.first_transition_revision) ||
        ('accepted_revision' in policy &&
            policy.accepted_revision !== undefined &&
            policy.accepted_revision > job.enqueue_revision) ||
        !configuration ||
        configuration.id !== job.processor_id ||
        configuration.version !== job.processor_version ||
        configuration.scope !== job.scope ||
        configuration.required !== job.required ||
        configuration.failure_behavior !== job.failure_behavior ||
        policy.policy_revision !== job.policy_revision ||
        (await fingerprintJson(configuration.config)) !== job.configuration_fingerprint
    )
        throw new Error('Indexed job is not bound to its retained policy stage');
    const resolution = await indexedProcessingRecord(
        store,
        root,
        'resolved_inputs',
        jobId,
        ProcessingResolvedInputSchema,
    );
    const resolutionReceipt =
        resolution === undefined
            ? undefined
            : await indexedRecordById(
                  store,
                  root,
                  'operation_receipts',
                  `processing:resolve:${jobId}`,
                  OperationReceiptSchema,
              );
    if (resolution !== undefined) {
        const identity = await fingerprintJson(resolution);
        if (
            !resolutionReceipt ||
            resolutionReceipt.conversation_id !== root.source.conversation_id ||
            resolutionReceipt.operation_kind !== 'processing' ||
            resolutionReceipt.processing_operation?.phase !== 'resolve' ||
            resolutionReceipt.processing_operation.job_id !== job.id ||
            resolutionReceipt.processing_operation.policy_revision !== job.policy_revision ||
            (resolutionReceipt.processing_operation.result_fingerprint !== undefined &&
                resolutionReceipt.processing_operation.result_fingerprint !== identity) ||
            resolutionReceipt.payload_fingerprint !== identity ||
            resolutionReceipt.base_revision !== resolution.source_revision ||
            resolutionReceipt.result_revision !== resolution.source_revision + 1 ||
            resolutionReceipt.result_revision > root.source.revision
        )
            throw new Error('Indexed job resolution lost its exact immutable phase receipt');
    }
    const attempt = await indexedProcessingRecord(store, root, 'attempts', jobId, ProcessingAttemptReceiptSchema);
    const output = await indexedProcessingRecord(store, root, 'outputs', jobId, ProcessingOutputReceiptSchema);
    const completion = await indexedProcessingRecord(
        store,
        root,
        'completions',
        jobId,
        ProcessingCompletionReceiptSchema,
    );
    const supersession = await indexedProcessingRecord(
        store,
        root,
        'supersessions',
        jobId,
        ProcessingSupersessionReceiptSchema,
    );
    return {
        job,
        configuration,
        header,
        ...(resolution === undefined ? {} : { resolution, resolution_receipt: resolutionReceipt }),
        ...(attempt === undefined ? {} : { attempt }),
        ...(output === undefined ? {} : { output }),
        ...(completion === undefined ? {} : { completion }),
        ...(supersession === undefined ? {} : { supersession }),
    };
}

/** Point-addressed completed predecessor from the same immutable accepted append cohort.
 * Neither an active entry with the same text nor an unrelated completed job can stand in. */
export async function loadIndexedProcessingPredecessorEvidence(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    jobIdInput: string,
) {
    const nomination = { root: rootInput, job_id: jobIdInput };
    if (!preflightJsonInput(nomination).success)
        throw new TypeError('Indexed predecessor nomination is not bounded JSON');
    const { root, job_id: jobId } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            job_id: IdentifierSchema,
        })
        .parse(structuredClone(nomination));
    const store = boundedIndexedProcessingReader(storeInput);
    const job = await ownedIndexedProcessingJob(store, root, jobId);
    if (job.stage_index < 1) return undefined;
    let predecessorId: string;
    if (job.selection.kind === 'predecessor_output') predecessorId = job.selection.job_id;
    else if (isToolResultTextProcessor(job)) {
        const accepted = await loadIndexedProcessingAppendAcceptance(store, root, job.source_operation_id);
        const preceding = accepted.jobs.filter((candidate) => candidate.stage_index + 1 === job.stage_index);
        if (preceding.length !== 1) throw new Error('Indexed tool-result stage has no exact accepted predecessor');
        predecessorId = preceding[0].id;
    } else return undefined;
    const prior = await loadIndexedProcessingJobState(store, root, predecessorId);
    if (!prior.resolution || !prior.resolution_receipt || !prior.output || !prior.completion || prior.supersession)
        throw new Error('Indexed stage predecessor has not durably completed');
    const receiptId = prior.completion.context_change_operation_id ?? `processing:complete:${prior.job.id}`;
    const receipt = await indexedRecordById(store, root, 'operation_receipts', receiptId, OperationReceiptSchema);
    if (
        !receipt ||
        receipt.result_revision > root.source.revision ||
        receipt.conversation_id !== root.source.conversation_id
    )
        throw new Error('Indexed stage predecessor lost its accepted completion receipt');
    const evidence = IndexedProcessingPredecessorEvidenceSchema.parse({
        job: prior.job,
        resolution: prior.resolution,
        output: prior.output,
        resolution_receipt: prior.resolution_receipt,
        ...(prior.attempt === undefined ? {} : { attempt: prior.attempt }),
        completion: prior.completion,
        receipt,
    });
    if (isToolResultTextProcessor(job)) {
        if (
            prior.job.source_operation_id !== job.source_operation_id ||
            prior.job.enqueue_revision !== job.enqueue_revision ||
            prior.job.policy_revision !== job.policy_revision ||
            prior.job.scope !== job.scope ||
            prior.job.processor_index >= job.processor_index
        )
            throw new Error('Indexed tool-result preceding stage changed its exact accepted cohort');
        await indexedCompletedJobEntrySelection(evidence);
    } else await indexedPredecessorEntrySelection(job, evidence);
    return evidence;
}

/** Exact asset append witness, point-loaded independently of the claim workspace or active preview. */
export async function loadIndexedProcessingArchiveState(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    jobIdInput: string,
) {
    const input = { root: rootInput, job_id: jobIdInput };
    if (!preflightJsonInput(input).success) throw new TypeError('Indexed archive inspection is not bounded JSON');
    const { root, job_id: jobId } = z
        .strictObject({ root: IndexedConversationRootSchema, job_id: IdentifierSchema })
        .parse(structuredClone(input));
    const store = boundedIndexedProcessingReader(storeInput);
    const receipt = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        `processing:archive:${jobId}`,
        OperationReceiptSchema,
    );
    if (!receipt) return undefined;
    const ids = receipt.accepted_asset_ids ?? [];
    if (
        receipt.operation_kind !== undefined ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.result_revision > root.source.revision ||
        ids.length < 1 ||
        ids.length > 4096 ||
        new Set(ids).size !== ids.length
    )
        throw new Error('Indexed archive receipt has a foreign or incomplete asset binding');
    const assets = [];
    for (const id of ids) {
        const asset = await indexedRecordById(store, root, 'assets', id, AssetSchema);
        if (
            asset?.kind !== 'text' ||
            asset.storage.type !== 'external' ||
            asset.content_hash === undefined ||
            asset.byte_length === undefined
        )
            throw new Error('Indexed archive receipt has a missing immutable text asset');
        assets.push(asset);
    }
    return { receipt, assets };
}

function assertIndexedProcessingCounts(header: z.infer<typeof IndexedConversationProcessingHeaderSchema>) {
    if (
        header.job_count === undefined ||
        header.unresolved_job_count === undefined ||
        header.required_unresolved_job_count === undefined ||
        header.required_job_count === undefined ||
        header.required_blocked_job_count === undefined ||
        header.unresolved_job_count > header.job_count ||
        header.required_unresolved_job_count > header.unresolved_job_count ||
        header.required_job_count > header.job_count ||
        header.required_unresolved_job_count > header.required_job_count ||
        header.required_blocked_job_count > header.required_unresolved_job_count
    )
        throw new Error('Indexed processing profile lacks its complete bounded counts');
    return {
        job_count: header.job_count,
        unresolved_job_count: header.unresolved_job_count,
        required_unresolved_job_count: header.required_unresolved_job_count,
        required_job_count: header.required_job_count,
        required_blocked_job_count: header.required_blocked_job_count,
    };
}

/** Bounded current-root discovery only. It creates no claim, output, readiness or provider authority. */
export async function loadIndexedPendingProcessingJobs(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    optionsInput: { cursor?: string; limit?: number } = {},
) {
    const input = { root: rootInput, options: optionsInput };
    if (!preflightJsonInput(input).success) throw new TypeError('Indexed pending discovery is not bounded JSON');
    const { root, options } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            options: z.strictObject({
                cursor: z.string().min(1).max(2048).optional(),
                limit: z.number().int().min(1).max(16).default(16),
            }),
        })
        .parse(input);
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed pending discovery requires the complete processing index profile');
    const store = boundedIndexedProcessingReader(storeInput);
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    const counts = assertIndexedProcessingCounts(header);
    const page = await readPagedRecordRange(store, root.directories.processing_pending, {
        ...(options.cursor === undefined ? {} : { after: options.cursor }),
        limit: options.limit,
    });
    if (
        (root.directories.processing_pending === undefined) !== (counts.unresolved_job_count === 0) ||
        (options.cursor === undefined && page.entries.length === 0 && counts.unresolved_job_count !== 0)
    )
        throw new Error('Indexed pending index and complete header count disagree');
    const jobs = [];
    for (const entry of page.entries) {
        if (
            entry.value.storage !== 'marker' ||
            entry.value.kind !== 'processing_pending' ||
            entry.value.id !== entry.key
        )
            throw new Error('Indexed pending index has a foreign job marker');
        const job = await ownedIndexedProcessingJob(store, root, entry.key);
        const completion = await indexedProcessingRecord(
            store,
            root,
            'completions',
            job.id,
            ProcessingCompletionReceiptSchema,
        );
        const supersession = await indexedProcessingRecord(
            store,
            root,
            'supersessions',
            job.id,
            ProcessingSupersessionReceiptSchema,
        );
        if (
            supersession ||
            (completion &&
                (completion.job_id !== job.id ||
                    completion.status !== 'blocked' ||
                    completion.result_revision > root.source.revision))
        )
            throw new Error('Indexed pending index contains a completed or superseded job');
        jobs.push({ job, ...(completion === undefined ? {} : { completion }) });
    }
    return {
        source: root.source,
        policy_revision: header.policy_revision,
        enabled: header.enabled,
        ...counts,
        jobs,
        has_more: page.has_more,
        ...(page.next_cursor === undefined ? {} : { next_cursor: page.next_cursor }),
    };
}

/** Exact append receipt and its own immutable jobs, separate from all outstanding obligations. */
export async function loadIndexedProcessingAppendAcceptance(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    operationIdInput: string,
) {
    const input = { root: rootInput, operation_id: operationIdInput };
    if (!preflightJsonInput(input).success) throw new TypeError('Indexed append acceptance is not bounded JSON');
    const { root, operation_id: operationId } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            operation_id: IdentifierSchema,
        })
        .parse(input);
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed append acceptance requires the complete processing index profile');
    const store = boundedIndexedProcessingReader(storeInput);
    const receipt = await indexedRecordById(store, root, 'operation_receipts', operationId, OperationReceiptSchema);
    if (
        !receipt ||
        receipt.operation_kind !== undefined ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.result_revision > root.source.revision
    )
        throw new Error('Indexed append acceptance has no exact accepted append receipt');
    const association = await indexedRecordById(
        store,
        root,
        'processing_by_operation',
        operationId,
        IndexedProcessingOperationJobsSchema,
    );
    if (
        association &&
        (association.operation_id !== receipt.id ||
            association.receipt_fingerprint !== (await fingerprintJson(receipt)))
    )
        throw new Error('Indexed append acceptance job association differs from its exact receipt');
    const jobs = [];
    for (const jobId of association?.job_ids ?? []) jobs.push(await ownedIndexedProcessingJob(store, root, jobId));
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    const counts = assertIndexedProcessingCounts(header);
    return {
        receipt,
        jobs,
        processing: {
            status: header.enabled ? (counts.required_blocked_job_count ? 'blocked' : 'pending') : 'ready',
            job_ids: jobs.map((job) => job.id),
        },
    };
}

/** Authenticated active data for a separate processor, with explicit processing-only completeness.
 * It cannot bypass native readiness. The reader bounds all active dependencies, never cold history.
 */
export async function loadIndexedProcessingSelectedContext(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
) {
    // Archive-only input can precede tool execution. Exact accepted call/index proof remains required.
    return loadIndexedProcessingContext(store, root, locator, true);
}

async function loadIndexedProcessingContext(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
    allowPendingCalls: boolean,
) {
    return IndexedProcessingSelectedContextSchema.parse(
        await loadIndexedSelectedContext(
            boundedIndexedProcessingReader(store),
            root,
            locator,
            INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
            true,
            true,
            'processing',
            allowPendingCalls,
        ),
    );
}

/** Same bounded active closure, plus genuine complete call originals only when their active projection is partial. */
export async function loadIndexedProcessingToolResultSelectedContext(
    storeInput: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
) {
    const store = boundedIndexedProcessingReader(storeInput);
    const selected = await loadIndexedProcessingSelectedContext(store, root, locator);
    const witnesses: Record<string, ConversationTurn> = {};
    for (const receipt of Object.values(selected.execution_witnesses ?? {})) {
        const source = receipt.call_source;
        if (!source || receipt.executor !== 'application') continue;
        const projection = selected.turns.find((turn) => turn.header.id === source.turn_id);
        if (!projection || projection.completeness === 'full_turn' || witnesses[source.turn_id]) continue;
        const complete = await loadIndexedProjectedTurn(store, root, source.turn_id);
        witnesses[source.turn_id] = ConversationTurnSchema.parse({
            ...complete.header,
            blocks: complete.selected_blocks,
        });
    }
    const result = IndexedProcessingSelectedContextSchema.parse({ ...selected, tool_result_call_witnesses: witnesses });
    if (canonicalJsonContentBytes(result).byteLength > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES)
        throw new RangeError('Indexed tool-result call witnesses exceed the active working-set bound');
    await indexedToolResultTextFrame(result, activeIndexedContextWorkingSet(result));
    return result;
}

/** Add only immutable completed sibling transitions from this job's accepted append cohort.
 * Historical replacement entries may no longer be active, but their point-addressed records and
 * completion/compaction receipts remain retained. No lifetime compaction scan is permitted.
 */
export async function loadIndexedProcessingSelectedContextForJob(
    storeInput: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
    jobId: string,
) {
    const store = boundedIndexedProcessingReader(storeInput);
    const job = await ownedIndexedProcessingJob(store, root, jobId);
    const relation = await indexedRecordById(
        store,
        root,
        'processing_by_operation',
        job.source_operation_id,
        IndexedProcessingOperationJobsSchema,
    );
    if (relation?.job_ids.filter((id) => id === jobId).length !== 1 || relation.job_ids.length > 16)
        throw new Error('Indexed exchange lacks its bounded accepted sibling-job set');
    const selected = await loadIndexedProcessingSelectedContext(store, root, locator);
    const compactionWitnesses = { ...selected.compaction_witnesses };
    const lineageEntryWitnesses: Record<string, z.infer<typeof ContextEntrySchema>> = {};
    const siblingCompactionIds: string[] = [];
    let remainderCount = 0;
    for (const siblingId of relation.job_ids) {
        if (siblingId === jobId) continue;
        const sibling = await ownedIndexedProcessingJob(store, root, siblingId);
        if (
            sibling.source_operation_id !== job.source_operation_id ||
            sibling.processor_id !== 'externalize-whole-exchange' ||
            sibling.processor_version !== '1'
        )
            throw new Error('Indexed exchange sibling differs from its accepted whole-exchange cohort');
        const completion = await indexedProcessingRecord(
            store,
            root,
            'completions',
            siblingId,
            ProcessingCompletionReceiptSchema,
        );
        if (!completion) continue;
        const output = await indexedProcessingRecord(store, root, 'outputs', siblingId, ProcessingOutputReceiptSchema);
        const compactionId = await deriveConversationId('indexed-exchange-compaction', siblingId);
        const receipt = completion.context_change_operation_id
            ? await indexedRecordById(
                  store,
                  root,
                  'operation_receipts',
                  completion.context_change_operation_id,
                  OperationReceiptSchema,
              )
            : undefined;
        const compaction = await indexedRecordById(
            store,
            root,
            'compactions',
            compactionId,
            IndexedConversationCompactionHeaderSchema,
        );
        const outputPayload =
            output?.kind === 'proposal' ? (({ output_fingerprint: _hash, ...value }) => value)(output) : undefined;
        if (
            completion.status !== 'applied' ||
            !output ||
            output.kind !== 'proposal' ||
            output.proposal.kind !== 'replace_with_compaction' ||
            output.proposal.compaction_id !== compactionId ||
            completion.output_fingerprint !== output.output_fingerprint ||
            !outputPayload ||
            (await fingerprintJson(outputPayload)) !== output.output_fingerprint ||
            !receipt ||
            receipt.id !== `processing:apply:${siblingId}` ||
            receipt.conversation_id !== root.source.conversation_id ||
            receipt.operation_kind !== 'context_change' ||
            receipt.context_change?.kind !== 'replace_with_compaction' ||
            receipt.context_change.removed_entry_ids.length !== 2 ||
            receipt.result_revision !== completion.result_revision ||
            receipt.result_revision > root.source.revision ||
            !compaction ||
            compaction.id !== compactionId ||
            compaction.operation_id !== receipt.id ||
            compaction.source.source_fingerprint !== receipt.context_change.source_fingerprint ||
            compaction.metadata?.applied_revision !== receipt.result_revision ||
            compaction.metadata?.payload_fingerprint !== receipt.payload_fingerprint
        )
            throw new Error('Indexed exchange sibling lacks its exact accepted completion lineage');
        compactionWitnesses[compactionId] = { compaction, acceptance: receipt };
        siblingCompactionIds.push(compactionId);
        for (const entryId of receipt.context_change.remainder_entry_ids ?? []) {
            remainderCount++;
            if (remainderCount > 32)
                throw new RangeError('Indexed exchange accepted remainder lineage exceeds its bounded cohort');
            if (!receipt.context_change.inserted_entry_ids.includes(entryId))
                throw new Error('Indexed exchange remainder is absent from its accepted insertion');
            const entry = await indexedRecordById(store, root, 'context_entries', entryId, ContextEntrySchema);
            if (!entry || entry.id !== entryId || entry.type !== 'source_turn')
                throw new Error('Indexed exchange accepted remainder entry is unavailable');
            lineageEntryWitnesses[entryId] = entry;
        }
    }
    const augmented = IndexedProcessingSelectedContextSchema.parse({
        ...selected,
        compaction_witnesses: compactionWitnesses,
        lineage_entry_witnesses: lineageEntryWitnesses,
        sibling_compaction_ids: siblingCompactionIds,
    });
    if (canonicalJsonContentBytes(augmented).byteLength > INDEXED_CONVERSATION_ACTIVE_MAX_BYTES)
        throw new RangeError('Indexed exchange selected lineage exceeds its bounded working set');
    return augmented;
}

const IndexedProcessingPhaseCommandSchema = z.discriminatedUnion('phase', [
    z.strictObject({ phase: z.literal('resolve'), value: ProcessingResolvedInputSchema }),
    z.strictObject({ phase: z.literal('attempt'), value: ProcessingAttemptReceiptSchema }),
    z.strictObject({ phase: z.literal('output'), value: ProcessingOutputReceiptSchema }),
]);
export type IndexedProcessingPhaseCommand = z.infer<typeof IndexedProcessingPhaseCommandSchema>;
export interface StagedIndexedProcessingPhase extends StagedIndexedConversationRoot {
    receipt: OperationReceipt;
    applied: boolean;
}

/** Durable phase records are keyed by the original immutable job, independent of HTTP attempts.
 * The host still owns current scheduler/task proof and exact current-root publication CAS.
 * First publication is validated against the real active projection; exact retained retry reads
 * only the original phase record/receipt and never invokes a processor or resets an attempt.
 */
export async function stageIndexedProcessingPhase(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    rootLocatorInput: PagedRecordRef,
    commandInput: IndexedProcessingPhaseCommand,
): Promise<StagedIndexedProcessingPhase> {
    const envelope = { root: rootInput, locator: rootLocatorInput, command: commandInput };
    if (!preflightJsonInput(envelope, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new TypeError('Indexed processing phase is not bounded JSON');
    const { root, locator, command } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            command: IndexedProcessingPhaseCommandSchema,
        })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed processing phase requires its complete processing indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const job = await ownedIndexedProcessingJob(store, root, command.value.job_id);
    const operationId = `processing:${command.phase}:${job.id}`;
    const payloadFingerprint = await fingerprintJson(command.value);
    const family =
        command.phase === 'resolve' ? 'resolved_inputs' : command.phase === 'attempt' ? 'attempts' : 'outputs';
    const phaseRecord = await getPagedRecord(store, root.directories.processing_records, tupleKey(family, job.id));
    const retained = await indexedRecordById(store, root, 'operation_receipts', operationId, OperationReceiptSchema);
    if (retained || phaseRecord) {
        if (
            !retained ||
            !phaseRecord ||
            phaseRecord.storage !== 'record' ||
            phaseRecord.kind !== 'processing_records' ||
            phaseRecord.id !== job.id
        )
            throw new Error('Indexed processing retry lacks its exact phase record and operation receipt');
        const retainedValue = await loadRecord(store, phaseRecord, z.unknown());
        if (
            retained.conversation_id !== root.source.conversation_id ||
            retained.operation_kind !== 'processing' ||
            retained.processing_operation?.phase !== command.phase ||
            retained.processing_operation.job_id !== job.id ||
            retained.payload_fingerprint !== payloadFingerprint ||
            retained.result_revision !== retained.base_revision + 1 ||
            retained.result_revision > root.source.revision ||
            (await fingerprintJson(retainedValue)) !== payloadFingerprint
        )
            throw new Error('Indexed processing retry differs from its immutable phase evidence');
        return { root, locator, receipt: retained, applied: false };
    }
    const completion = await indexedProcessingRecord(
        store,
        root,
        'completions',
        job.id,
        ProcessingCompletionReceiptSchema,
    );
    const supersession = await indexedProcessingRecord(
        store,
        root,
        'supersessions',
        job.id,
        ProcessingSupersessionReceiptSchema,
    );
    if (completion || supersession)
        throw new Error('Indexed processing cannot create a phase for a completed or superseded job');
    const header = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    assertIndexedProcessingCounts(header);
    if (!header.enabled) throw new Error('Indexed processing first publication requires activated policy');
    const configuration = header.processors[job.processor_index];
    if (
        !configuration ||
        configuration.id !== job.processor_id ||
        configuration.version !== job.processor_version ||
        configuration.scope !== job.scope ||
        configuration.required !== job.required ||
        configuration.failure_behavior !== job.failure_behavior ||
        header.policy_revision !== job.policy_revision ||
        (await fingerprintJson(configuration.config)) !== job.configuration_fingerprint
    )
        throw new Error('Indexed processing phase differs from its exact accepted policy stage');
    let recordedAt: string;
    if (command.phase === 'resolve') {
        const selected =
            job.processor_id === INDEXED_EXCHANGE_PROCESSOR_ID
                ? await loadIndexedProcessingSelectedContextForJob(store, root, locator, job.id)
                : isToolResultTextProcessor(job)
                  ? await loadIndexedProcessingToolResultSelectedContext(store, root, locator)
                  : await loadIndexedProcessingSelectedContext(store, root, locator);
        const predecessor = await loadIndexedProcessingPredecessorEvidence(store, root, job.id);
        const resolution = await resolveIndexedProcessingTextInput(
            selected,
            job,
            command.value.recorded_at,
            predecessor,
        );
        if (!sameIndexedRecord(resolution, command.value))
            throw new Error('Indexed resolution is not derived from the exact selected source');
        recordedAt = command.value.recorded_at;
    } else {
        const resolution = await indexedProcessingRecord(
            store,
            root,
            'resolved_inputs',
            job.id,
            ProcessingResolvedInputSchema,
        );
        if (!resolution || (await fingerprintJson(resolution)) !== command.value.resolved_input_fingerprint)
            throw new Error('Indexed processing phase lost its exact durable resolution');
        if (command.phase === 'attempt') {
            const selected =
                job.processor_id === INDEXED_EXCHANGE_PROCESSOR_ID
                    ? await loadIndexedProcessingSelectedContextForJob(store, root, locator, job.id)
                    : isToolResultTextProcessor(job)
                      ? await loadIndexedProcessingToolResultSelectedContext(store, root, locator)
                      : await loadIndexedProcessingSelectedContext(store, root, locator);
            const predecessor = await loadIndexedProcessingPredecessorEvidence(store, root, job.id);
            const current = await resolveIndexedProcessingTextInput(selected, job, resolution.recorded_at, predecessor);
            if (current.context_fingerprint !== resolution.context_fingerprint)
                throw new Error('Indexed attempt lost its active dependency context');
            recordedAt = command.value.started_at;
        } else {
            const attempt = await indexedProcessingRecord(
                store,
                root,
                'attempts',
                job.id,
                ProcessingAttemptReceiptSchema,
            );
            if (
                command.value.attempt_token === undefined
                    ? attempt !== undefined
                    : attempt?.attempt_token !== command.value.attempt_token
            )
                throw new Error('Indexed output lost its exact durable attempt');
            const { output_fingerprint: fingerprint, ...payload } = command.value;
            if ((await fingerprintJson(payload)) !== fingerprint)
                throw new Error('Indexed processing output fingerprint changed');
            recordedAt = command.value.recorded_at;
        }
    }
    const receipt = createProcessingTransitionReceipt({
        source: root.source,
        operation_id: operationId,
        recorded_at: recordedAt,
        payload_fingerprint: payloadFingerprint,
        processing_operation: {
            phase: command.phase,
            job_id: job.id,
            policy_revision: header.policy_revision,
            result_fingerprint: payloadFingerprint,
        },
    });
    const directories = { ...root.directories };
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        tupleKey(family, job.id),
        await stageRecord(store, 'processing_records', job.id, command.value),
    );
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
        source: { ...root.source, revision: receipt.result_revision },
        updated_at: recordedAt,
        directories,
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed processing phase root exceeds its manifest bound');
    return {
        root: nextRoot,
        locator: { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes },
        receipt,
        applied: true,
    };
}

export interface StagedIndexedProcessingCompletion extends StagedIndexedProcessingPhase {
    completion: z.infer<typeof ProcessingCompletionReceiptSchema>;
}

/** Shared immutable context mutation publication. This helper neither creates nor settles jobs.
 * Every caller has already applied the pure working-set mutation and owns its exact receipt.
 */
async function stageIndexedContextMutationRecords(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    previousContext: ConversationContext,
    mutation: ContextMutationResult,
): Promise<IndexedConversationDirectories> {
    const compaction = mutation.compaction;
    if (!compaction) throw new Error('Indexed context mutation lost its exact new compaction');
    // Replacement entities must retain their semantic identity kinds for later selected preparation.
    const newIdentities = [
        { id: compaction.id, kind: 'compaction' },
        { id: mutation.receipt.id, kind: 'operation receipt' },
        ...compaction.replacement_turns.flatMap((turn) => [
            { id: turn.id, kind: 'replacement turn' },
            ...deletedContentIdentities(turn.blocks).block_ids.map((id) => ({ id, kind: 'block' })),
        ]),
        ...mutation.change.operations[0].inserted_entry_ids.map((id) => ({ id, kind: 'context entry' })),
        ...mutation.context.retrieval_requirements
            .filter((item) => !previousContext.retrieval_requirements.some((old) => old.id === item.id))
            .map((item) => ({ id: item.id, kind: 'retrieval requirement' })),
    ];
    const newIds = newIdentities.map(({ id }) => id);
    if (new Set(newIds).size !== newIds.length)
        throw new Error('Indexed context mutation creates duplicate record identities');
    for (const id of newIds)
        if (await getPagedRecord(store, root.directories.identifiers, id))
            throw new Error('Indexed context mutation identity already belongs to a retained record');
    const directories = { ...root.directories };
    const insertions = new Map<keyof IndexedConversationDirectories, { key: string; value: PagedRecordValue }[]>();
    const nominate = (family: keyof IndexedConversationDirectories, key: string, value: PagedRecordValue) => {
        const commands = insertions.get(family) ?? [];
        commands.push({ key, value });
        insertions.set(family, commands);
    };
    const write = async (family: keyof IndexedConversationDirectories, id: string, value: unknown, key = id) => {
        nominate(family, key, await stageRecord(store, family, id, value));
    };
    const { replacement_turns: replacementTurns, original_context: _originalContext, ...compactionHeader } = compaction;
    await write('compactions', compaction.id, compactionHeader);
    for (const turn of replacementTurns) {
        const { blocks, ...header } = turn;
        const blockIds = blocks.map((block) => block.id);
        await write(
            'turns',
            turn.id,
            IndexedConversationTurnHeaderSchema.parse({
                turn: header,
                source: 'replacement',
                compaction_id: compaction.id,
                block_ids: blockIds,
                block_ids_hash: (await hashContentBytes(canonicalJsonContentBytes(blockIds))).content_hash,
            }),
        );
        for (const block of blocks) {
            await write('blocks', block.id, block);
            for (const id of deletedContentIdentities([block]).block_ids)
                nominate('block_owners', id, { storage: 'marker', kind: 'block_owner', id: turn.id });
        }
    }
    // Complete reverse delete witnesses: compaction originals and replacement block owners remain
    // protected even when subsequent provider receipts refer only to the replacement.
    const protectedTurns = new Set([...compaction.source.turn_ids, ...replacementTurns.map((turn) => turn.id)]);
    for (const blockId of compaction.source.block_ids ?? []) {
        const owner = await getPagedRecord(store, root.directories.block_owners, blockId);
        if (owner?.storage !== 'marker' || owner.kind !== 'block_owner')
            throw new Error('Indexed compaction source block loses its immutable owner');
        protectedTurns.add(owner.id);
    }
    if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2)
        await appendIndexedDeleteDependencies(
            store,
            directories,
            [...protectedTurns].map((target_turn_id) => ({
                target_turn_id,
                kind: 'compaction',
                owner_id: compaction.id,
            })),
        );
    for (const turnId of protectedTurns)
        if (!(await getPagedRecord(store, directories.deletion_blockers, turnId)))
            nominate('deletion_blockers', turnId, {
                storage: 'marker',
                kind: 'delete_blocker',
                id: turnId,
            });
    for (const entry of mutation.context.entries) {
        const existing = await getPagedRecord(store, directories.context_entries, entry.id);
        if (!existing) await write('context_entries', entry.id, entry);
        else {
            const actual = await loadRecord(store, existing, ContextEntrySchema);
            if (!sameIndexedRecord(actual, entry)) throw new Error('Indexed retained context entry differs');
        }
    }
    directories.active_context_order = await buildPagedRecordIndex(
        store,
        mutation.context.entries.map((entry, i) => ({
            key: indexedOrderedKey(i),
            value: { storage: 'marker' as const, kind: 'context_order', id: entry.id },
        })),
    );
    for (const { id, kind } of newIdentities)
        nominate('identifiers', id, {
            storage: 'marker',
            kind,
            id,
        });

    await write('operation_receipts', mutation.receipt.id, mutation.receipt);
    for (const [family, commands] of insertions) await insertFreshIndexedRecords(store, directories, family, commands);
    return directories;
}

/** Settle an already durably stored pure text output. Global identifiers and archive records are
 * looked up individually; originals stay immutable and cold. A completion removes only this job's
 * unresolved marker, retains its full resolution/attempt/output/receipts, and invalidates readiness.
 */
export async function stageIndexedTextProcessingCompletion(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    workspaceInput: IndexedProcessingClaimWorkspace,
    originalArchiveBytes: ReadonlyMap<string, Uint8Array> = new Map(),
): Promise<StagedIndexedProcessingCompletion> {
    const envelope = { root: rootInput, locator: locatorInput, workspace: workspaceInput };
    if (!preflightJsonInput(envelope, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new TypeError('Indexed text completion is not bounded JSON');
    const { root, locator, workspace } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            workspace: IndexedProcessingClaimWorkspaceSchema,
        })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed completion needs its complete processing indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const job = await ownedIndexedProcessingJob(store, root, workspace.job.id);
    const resolution = await indexedProcessingRecord(
        store,
        root,
        'resolved_inputs',
        job.id,
        ProcessingResolvedInputSchema,
    );
    const attempt = await indexedProcessingRecord(store, root, 'attempts', job.id, ProcessingAttemptReceiptSchema);
    const output = await indexedProcessingRecord(store, root, 'outputs', job.id, ProcessingOutputReceiptSchema);
    if (
        !sameIndexedRecord(job, workspace.job) ||
        !resolution ||
        !attempt ||
        !output ||
        !sameIndexedRecord(resolution, workspace.resolution) ||
        !sameIndexedRecord(attempt, workspace.attempt) ||
        attempt.resolved_input_fingerprint !== output.resolved_input_fingerprint ||
        output.attempt_token !== attempt.attempt_token
    )
        throw new Error('Indexed completion lost its exact durable job/resolution/attempt/output');
    const { output_fingerprint: outputFingerprint, ...outputPayload } = output;
    if ((await fingerprintJson(outputPayload)) !== outputFingerprint)
        throw new Error('Indexed completion output integrity changed');
    const retained = await indexedProcessingRecord(
        store,
        root,
        'completions',
        job.id,
        ProcessingCompletionReceiptSchema,
    );
    if (retained) {
        const receiptId = retained.context_change_operation_id ?? `processing:complete:${job.id}`;
        const receipt = await indexedRecordById(store, root, 'operation_receipts', receiptId, OperationReceiptSchema);
        if (
            !receipt ||
            receipt.conversation_id !== root.source.conversation_id ||
            retained.output_fingerprint !== outputFingerprint ||
            retained.result_revision !== receipt.result_revision ||
            receipt.result_revision > root.source.revision ||
            canonicalJsonContentString(retained.inserted_entry_ids) !==
                canonicalJsonContentString(receipt.accepted_context_entry_ids ?? []) ||
            (retained.status === 'applied'
                ? receipt.operation_kind !== 'context_change'
                : receipt.processing_operation?.phase !== 'complete')
        )
            throw new Error('Indexed completion retry differs from its immutable result receipt');
        return { root, locator, receipt, completion: retained, applied: false };
    }
    if (await indexedProcessingRecord(store, root, 'supersessions', job.id, ProcessingSupersessionReceiptSchema))
        throw new Error('Indexed completion job has been superseded');
    const selected =
        job.processor_id === 'externalize-whole-exchange'
            ? await loadIndexedProcessingSelectedContextForJob(store, root, locator, job.id)
            : isToolResultTextProcessor(job)
              ? await loadIndexedProcessingToolResultSelectedContext(store, root, locator)
              : await loadIndexedProcessingSelectedContext(store, root, locator);
    const currentWorkspace = { ...workspace, selected };
    // Replay validates unchanged active-context identity against retained resolution, plus full
    // deterministic output equality. Merely having a job/output marker never proves the delta.
    for (const asset of workspace.archives.assets) {
        const accepted = await indexedRecordById(store, root, 'assets', asset.id, AssetSchema);
        if (!accepted || !sameIndexedRecord(accepted, asset))
            throw new Error('Indexed completion archive differs from its actual accepted asset');
    }
    const archiveReceipt = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        workspace.archives.acceptance.id,
        OperationReceiptSchema,
    );
    if (!archiveReceipt || !sameIndexedRecord(archiveReceipt, workspace.archives.acceptance))
        throw new Error('Indexed completion archive receipt differs from its actual retained acceptance');
    if (job.processor_id !== INDEXED_EXCHANGE_PROCESSOR_ID && originalArchiveBytes.size !== 0)
        throw new Error('Indexed text completion has unexpected external archive bytes');
    const mutation =
        job.processor_id === INDEXED_EXCHANGE_PROCESSOR_ID &&
        job.processor_version === INDEXED_EXCHANGE_PROCESSOR_VERSION
            ? await applyIndexedExchangeOutput(currentWorkspace, output, originalArchiveBytes)
            : await applyIndexedTextExternalizationOutput(currentWorkspace, output);
    const directories = await stageIndexedContextMutationRecords(store, root, selected.context, mutation);
    const completion = ProcessingCompletionReceiptSchema.parse({
        job_id: job.id,
        output_fingerprint: outputFingerprint,
        status: 'applied',
        result_revision: mutation.receipt.result_revision,
        inserted_entry_ids: mutation.change.operations[0].inserted_entry_ids,
        context_change_operation_id: mutation.receipt.id,
        recorded_at: workspace.snapshot_at,
    });
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        tupleKey('completions', job.id),
        await stageRecord(store, 'processing_records', job.id, completion),
    );
    if (isToolResultTextProcessor(job)) {
        if (!mutation.compaction) throw new Error('Indexed tool-result completion lost its compaction');
        await stageIndexedToolResultOriginalSource(store, directories, job, resolution, {
            kind: 'indexed_root',
            root: locator,
        });
        await stageIndexedToolResultValidation(
            store,
            root,
            directories,
            await indexedToolResultTextFrame(selected, activeIndexedContextWorkingSet(selected)),
            job,
            resolution,
            output,
            completion,
            mutation.compaction,
            mutation.receipt,
        );
    }
    const removed = await removePagedRecord(store, directories.processing_pending, job.id);
    if (
        !removed.applied ||
        removed.removed?.storage !== 'marker' ||
        removed.removed.kind !== 'processing_pending' ||
        removed.removed.id !== job.id
    )
        throw new Error('Indexed completion lost its exact unresolved marker');
    if (removed.root === undefined) delete directories.processing_pending;
    else directories.processing_pending = removed.root;
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    const counts = assertIndexedProcessingCounts(header);
    if (counts.unresolved_job_count < 1 || (job.required && counts.required_unresolved_job_count < 1))
        throw new Error('Indexed completion cannot decrement an absent unresolved obligation');
    const { coverage: _coverage, ...processing } = header;
    const processingHeader = await stageRecord(
        store,
        'processing_header',
        root.source.conversation_id,
        IndexedConversationProcessingHeaderSchema.parse({
            ...processing,
            unresolved_job_count: counts.unresolved_job_count - 1,
            required_unresolved_job_count: counts.required_unresolved_job_count - (job.required ? 1 : 0),
        }),
    );
    const { entries: _entries, ...contextFields } = mutation.context;
    const contextHeader = await stageRecord(
        store,
        'context_header',
        root.source.conversation_id,
        IndexedConversationContextHeaderSchema.parse({
            ...contextFields,
            active_entry_count: mutation.context.entries.length,
            active_entry_bytes: canonicalJsonContentBytes(mutation.context.entries).byteLength,
            context_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(mutation.context))).content_hash,
        }),
    );
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        directories,
        source: { ...root.source, revision: completion.result_revision },
        updated_at: workspace.snapshot_at,
        context_header: { content_hash: contextHeader.content_hash, size_bytes: contextHeader.size_bytes },
        processing_header: { content_hash: processingHeader.content_hash, size_bytes: processingHeader.size_bytes },
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed processing completion root exceeds its manifest bound');
    await loadIndexedProcessingSelectedContext(store, nextRoot, {
        content_hash: rootRecord.content_hash,
        size_bytes: rootRecord.size_bytes,
    });
    return {
        root: nextRoot,
        locator: { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes },
        receipt: mutation.receipt,
        completion,
        applied: true,
    };
}

/** Complete only the exact no-eligible-blocks output for an immutable empty text-stage selection.
 * Archive-only append jobs are genuine canonical jobs; they do not require fabricated archives,
 * provider admission or another processor invocation just to settle their zero-content obligation.
 */
export async function stageIndexedProcessingNoOpCompletion(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    jobIdInput: string,
): Promise<StagedIndexedProcessingCompletion> {
    const envelope = { root: rootInput, locator: locatorInput, job_id: jobIdInput };
    if (!preflightJsonInput(envelope).success) throw new TypeError('Indexed no-op completion is not bounded JSON');
    const {
        root,
        locator,
        job_id: jobId,
    } = z
        .strictObject({ root: IndexedConversationRootSchema, locator: PagedRecordRefSchema, job_id: IdentifierSchema })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed no-op completion requires complete processing indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const job = await ownedIndexedProcessingJob(store, root, jobId);
    const resolution = await indexedProcessingRecord(
        store,
        root,
        'resolved_inputs',
        jobId,
        ProcessingResolvedInputSchema,
    );
    const output = await indexedProcessingRecord(store, root, 'outputs', jobId, ProcessingOutputReceiptSchema);
    const attempt = await indexedProcessingRecord(store, root, 'attempts', jobId, ProcessingAttemptReceiptSchema);
    if (
        job.processor_id !== 'externalize-text' ||
        job.processor_version !== '1' ||
        (job.scope !== 'on_append' && job.scope !== 'manual' && job.scope !== 'on_budget') ||
        (job.stage_index === 0 ? job.selection.kind !== 'entries' : job.selection.kind !== 'predecessor_output') ||
        !resolution ||
        resolution.entry_ids.length !== 0 ||
        resolution.source_turn_ids.length !== 0 ||
        !output ||
        output.kind !== 'no_op' ||
        output.reason !== 'no_eligible_blocks' ||
        output.attempt_token !== undefined ||
        attempt
    )
        throw new Error('Indexed no-op completion requires its exact empty text selection and unattempted output');
    const predecessor = await loadIndexedProcessingPredecessorEvidence(store, root, job.id);
    const selected = await loadIndexedProcessingSelectedContext(store, root, locator);
    const expectedResolution = await resolveIndexedProcessingTextInput(
        { ...selected, source: { ...selected.source, revision: resolution.source_revision } },
        job,
        resolution.recorded_at,
        predecessor,
    );
    if (!sameIndexedRecord(expectedResolution, resolution))
        throw new Error('Indexed no-op completion differs from its exact accepted ordered resolution');
    const { output_fingerprint: outputFingerprint, ...payload } = output;
    if (
        (await fingerprintJson(payload)) !== outputFingerprint ||
        (await fingerprintJson(resolution)) !== output.resolved_input_fingerprint
    )
        throw new Error('Indexed no-op completion lost its retained output/resolution identity');
    const operationId = `processing:complete:${jobId}`;
    const retained = await indexedProcessingRecord(
        store,
        root,
        'completions',
        jobId,
        ProcessingCompletionReceiptSchema,
    );
    const prior = await indexedRecordById(store, root, 'operation_receipts', operationId, OperationReceiptSchema);
    if (retained || prior) {
        if (
            !retained ||
            !prior ||
            retained.status !== 'no_op' ||
            retained.output_fingerprint !== outputFingerprint ||
            retained.result_revision !== prior.result_revision ||
            prior.result_revision > root.source.revision ||
            prior.conversation_id !== root.source.conversation_id ||
            prior.processing_operation?.phase !== 'complete' ||
            prior.processing_operation.job_id !== jobId ||
            prior.payload_fingerprint !== (await fingerprintJson(retained))
        )
            throw new Error('Indexed no-op retry differs from its exact retained completion');
        return { root, locator, receipt: prior, completion: retained, applied: false };
    }
    if (await indexedProcessingRecord(store, root, 'supersessions', jobId, ProcessingSupersessionReceiptSchema))
        throw new Error('Indexed no-op job has been superseded');
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    const counts = assertIndexedProcessingCounts(header);
    if (counts.unresolved_job_count < 1 || (job.required && counts.required_unresolved_job_count < 1))
        throw new Error('Indexed no-op cannot remove an absent obligation');
    const completion = ProcessingCompletionReceiptSchema.parse({
        job_id: jobId,
        output_fingerprint: outputFingerprint,
        status: 'no_op',
        result_revision: root.source.revision + 1,
        inserted_entry_ids: [],
        recorded_at: output.recorded_at,
    });
    const completionFingerprint = await fingerprintJson(completion);
    const receipt = createProcessingTransitionReceipt({
        source: root.source,
        operation_id: operationId,
        payload_fingerprint: completionFingerprint,
        recorded_at: output.recorded_at,
        processing_operation: {
            phase: 'complete',
            job_id: jobId,
            policy_revision: header.policy_revision,
            result_fingerprint: completionFingerprint,
        },
    });
    const directories = { ...root.directories };
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        tupleKey('completions', jobId),
        await stageRecord(store, 'processing_records', jobId, completion),
    );
    directories.operation_receipts = await putPagedRecord(
        store,
        directories.operation_receipts,
        operationId,
        await stageRecord(store, 'operation_receipts', operationId, receipt),
    );
    directories.identifiers = await putPagedRecord(store, directories.identifiers, operationId, {
        storage: 'marker',
        kind: 'operation receipt',
        id: operationId,
    });
    const removed = await removePagedRecord(store, directories.processing_pending, jobId);
    if (
        !removed.applied ||
        removed.removed?.storage !== 'marker' ||
        removed.removed.kind !== 'processing_pending' ||
        removed.removed.id !== jobId
    )
        throw new Error('Indexed no-op lost its exact unresolved marker');
    if (removed.root === undefined) delete directories.processing_pending;
    else directories.processing_pending = removed.root;
    const { coverage: _coverage, ...processing } = header;
    const nextHeader = await stageRecord(
        store,
        'processing_header',
        root.source.conversation_id,
        IndexedConversationProcessingHeaderSchema.parse({
            ...processing,
            unresolved_job_count: counts.unresolved_job_count - 1,
            required_unresolved_job_count: counts.required_unresolved_job_count - (job.required ? 1 : 0),
        }),
    );
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        directories,
        source: { ...root.source, revision: receipt.result_revision },
        updated_at: output.recorded_at,
        processing_header: { content_hash: nextHeader.content_hash, size_bytes: nextHeader.size_bytes },
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed no-op completion root exceeds manifest bound');
    await loadIndexedProcessingSelectedContext(store, nextRoot, {
        content_hash: rootRecord.content_hash,
        size_bytes: rootRecord.size_bytes,
    });
    return {
        root: nextRoot,
        locator: { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes },
        receipt,
        completion,
        applied: true,
    };
}

export type { IndexedProcessingCoverageCommand } from './schemas/indexed-head.js';
export { IndexedProcessingCoverageCommandSchema } from './schemas/indexed-head.js';
export interface StagedIndexedProcessingCoverage extends StagedIndexedProcessingPhase {
    coverage: IndexedProcessingReadinessCoverage;
}

/** Queue one exact selected current-context job. The host supplies an independently verified
 * model target and owns the physical head CAS; the pure transition retains the selected plan,
 * policy and job set in the same immutable revision as its accepted queue receipt.
 */
export async function stageIndexedProcessingQueue(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    commandInput: IndexedProcessingQueueCommand,
): Promise<StagedIndexedProcessingPhase & { job: ProcessingJob }> {
    const envelope = { root: rootInput, locator: locatorInput, command: commandInput };
    if (!preflightJsonInput(envelope, { max_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES }).success)
        throw new TypeError('Indexed processing queue is not bounded JSON');
    const { root, locator, command } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            command: IndexedProcessingQueueCommandSchema,
        })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed processing queue requires complete durable outbox indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const fingerprint = await fingerprintJson(command);
    const prior = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        command.operation_id,
        OperationReceiptSchema,
    );
    if (prior) {
        const retainedCommand = await indexedProcessingRecord(
            store,
            root,
            'selected_queue_commands',
            command.operation_id,
            IndexedProcessingQueueCommandSchema,
        );
        const association = await indexedRecordById(
            store,
            root,
            'processing_by_operation',
            command.operation_id,
            IndexedProcessingOperationJobsSchema,
        );
        if (
            prior.operation_kind !== 'processing' ||
            prior.processing_operation?.phase !== 'queue' ||
            prior.conversation_id !== root.source.conversation_id ||
            prior.result_revision !== command.expected_revision + 1 ||
            prior.payload_fingerprint !== fingerprint ||
            prior.base_revision !== command.expected_revision ||
            prior.recorded_at !== command.recorded_at ||
            prior.result_revision > root.source.revision ||
            association?.receipt_fingerprint !== (await fingerprintJson(prior)) ||
            association.job_ids.length !== 1 ||
            prior.processing_operation.job_id !== association.job_ids[0] ||
            !retainedCommand ||
            canonicalJsonContentString(retainedCommand) !== canonicalJsonContentString(command)
        )
            throw new Error('Indexed processing queue retry conflicts with its accepted job set');
        const job = await ownedIndexedProcessingJob(store, root, association.job_ids[0]);
        return { root, locator, receipt: prior, job, applied: false };
    }
    if (root.source.revision !== command.expected_revision)
        throw new Error('Indexed processing queue source revision conflict');
    if (Date.parse(command.recorded_at) < Date.parse(root.updated_at))
        throw new Error('Indexed processing queue timestamp predates current source');
    if (command.scope === 'on_budget' && !command.target_fingerprint)
        throw new Error('Indexed on-budget queue requires exact measured target identity');
    const header = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    await assertIndexedCurrentPolicy(store, root, header);
    const counts = assertIndexedRequiredIdentity(root, header);
    if (!header.enabled || counts.unresolved_job_count >= 256)
        throw new Error('Indexed processing queue has no enabled bounded policy capacity');
    const processorIndex = header.processors.findIndex(
        (processor) => processor.id === command.processor_id && processor.scope === command.scope,
    );
    const processor = header.processors[processorIndex];
    if (!processor) throw new Error('Indexed processing queue processor is absent from the accepted policy');
    const toolResult = isToolResultTextStrategy(processor.id, processor.version);
    if (
        toolResult &&
        (!supportsToolResultTextProcessingScope({
            processor_id: processor.id,
            processor_version: processor.version,
            scope: processor.scope,
        }) ||
            command.selected_block_ids !== undefined ||
            command.target_fingerprint !== undefined)
    )
        throw new Error('Manual tool-result queue requires registered v2 whole result entries');
    const selected = toolResult
        ? await loadIndexedProcessingToolResultSelectedContext(store, root, locator)
        : await loadIndexedProcessingSelectedContext(store, root, locator);
    if (selected.context.revision !== command.expected_context_revision)
        throw new Error('Indexed processing queue context revision conflict');
    const entries = new Map(selected.context.entries.map((entry) => [entry.id, entry]));
    const nominatedEntries = command.selected_entry_ids.map((id) => entries.get(id));
    if (nominatedEntries.some((entry) => !entry))
        throw new Error('Indexed processing queue selects an unavailable current entry');
    const selectedEntries = nominatedEntries.filter((entry): entry is NonNullable<typeof entry> => entry !== undefined);
    const workingSet = activeIndexedContextWorkingSet(selected);
    const selectedResults = toolResult
        ? await toolResultTextWorkingSelection(
              await indexedToolResultTextFrame(selected, workingSet),
              command.selected_entry_ids,
              { processor_id: processor.id, processor_version: processor.version, configuration: processor.config },
          )
        : undefined;
    if (
        selectedResults &&
        (!selectedResults.texts.length ||
            !sameIndexedRecord(
                selectedResults.records.map((record) => record.entry.id),
                command.selected_entry_ids,
            ))
    )
        throw new Error('Manual tool-result queue must select exactly its eligible text-bearing result entries');
    const plan = selectedResults
        ? undefined
        : await planContextChangeWorkingSet(
              workingSet,
              ContextChangePlanInputSchema.parse({
                  expected_revision: root.source.revision,
                  expected_context_revision: command.expected_context_revision,
                  entry_ids: command.selected_entry_ids,
                  ...(command.selected_block_ids === undefined
                      ? {}
                      : {
                            selected_block_ids: command.selected_block_ids,
                            selected_entries: selectedEntries,
                        }),
              }),
          );
    const jobs = await constructProcessingJobs({
        conversation_id: root.source.conversation_id,
        revision: root.source.revision + 1,
        source_operation_id: command.operation_id,
        policy_revision: header.policy_revision,
        processors: header.processors,
        processor_indices: [processorIndex],
        entry_ids: plan?.entry_ids ?? command.selected_entry_ids,
        ...(plan?.selected_block_ids === undefined ? {} : { selected_block_ids: plan.selected_block_ids }),
        ...(plan?.selected_entries === undefined ? {} : { selected_entries: plan.selected_entries }),
        ...(command.target_fingerprint === undefined ? {} : { target_fingerprint: command.target_fingerprint }),
    });
    const job = jobs[0];
    if (jobs.length !== 1 || !job)
        throw new Error('Indexed processing queue requires one exact registered selected job');
    if (
        (await getPagedRecord(store, root.directories.identifiers, command.operation_id)) ||
        (await getPagedRecord(store, root.directories.identifiers, job.id))
    )
        throw new Error('Indexed processing queue identity is already accepted');
    const receipt = createProcessingTransitionReceipt({
        source: root.source,
        operation_id: command.operation_id,
        payload_fingerprint: fingerprint,
        recorded_at: command.recorded_at,
        processing_operation: { phase: 'queue', policy_revision: header.policy_revision, job_id: job.id },
    });
    const directories = { ...root.directories };
    if (root.delete_index_profile === INDEXED_CONVERSATION_DELETE_PROFILE_V2)
        await appendIndexedDeleteDependencies(
            store,
            directories,
            selectedEntries.map((entry) => ({ target_turn_id: entry.turn_id, kind: 'job', owner_id: job.id })),
        );

    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        tupleKey('jobs', job.id),
        await stageRecord(store, 'processing_records', job.id, job),
    );
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        tupleKey('selected_queue_commands', receipt.id),
        await stageRecord(store, 'processing_records', receipt.id, command),
    );
    directories.processing_pending = await putPagedRecord(store, directories.processing_pending, job.id, {
        storage: 'marker',
        kind: 'processing_pending',
        id: job.id,
    });
    if (job.required)
        directories.processing_required = await putPagedRecord(store, directories.processing_required, job.id, {
            storage: 'marker',
            kind: 'processing_required',
            id: job.id,
        });
    directories.processing_by_operation = await putPagedRecord(
        store,
        directories.processing_by_operation,
        receipt.id,
        await stageRecord(
            store,
            'processing_by_operation',
            receipt.id,
            IndexedProcessingOperationJobsSchema.parse({
                version: 1,
                operation_id: receipt.id,
                receipt_fingerprint: await fingerprintJson(receipt),
                job_ids: [job.id],
            }),
        ),
    );
    directories.operation_receipts = await putPagedRecord(
        store,
        directories.operation_receipts,
        receipt.id,
        await stageRecord(store, 'operation_receipts', receipt.id, receipt),
    );
    directories.identifiers = await insertPagedRecords(store, directories.identifiers, [
        { key: receipt.id, value: { storage: 'marker', kind: 'operation receipt', id: receipt.id } },
        { key: job.id, value: { storage: 'marker', kind: 'processing_job', id: job.id } },
    ]);
    const { coverage: _coverage, ...processing } = header;
    const nextHeader = await stageRecord(
        store,
        'processing_header',
        root.source.conversation_id,
        IndexedConversationProcessingHeaderSchema.parse({
            ...processing,
            job_count: counts.job_count + 1,
            unresolved_job_count: counts.unresolved_job_count + 1,
            required_job_count: counts.required_job_count + (job.required ? 1 : 0),
            required_unresolved_job_count: counts.required_unresolved_job_count + (job.required ? 1 : 0),
        }),
    );
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        source: { ...root.source, revision: receipt.result_revision },
        updated_at: command.recorded_at,
        directories,
        processing_header: { content_hash: nextHeader.content_hash, size_bytes: nextHeader.size_bytes },
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed processing queue root exceeds its manifest bound');
    return {
        root: nextRoot,
        locator: { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes },
        receipt,
        job,
        applied: true,
    };
}

/** Read-only status after a scheduled control ACK. The service still proves the actual task,
 * owner and current physical head; this point reader proves the accepted control operation.
 */
export async function loadIndexedProcessingControlAcceptance(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    sourceInput: { conversation_id: string; revision: number },
    operationIdInput: string,
    phaseInput: 'policy' | 'queue',
    payloadFingerprintInput: string,
) {
    const envelope = {
        root: rootInput,
        source: sourceInput,
        operation_id: operationIdInput,
        phase: phaseInput,
        payload_fingerprint: payloadFingerprintInput,
    };
    if (!preflightJsonInput(envelope).success) throw new TypeError('Indexed control observation is not bounded JSON');
    const {
        root,
        source,
        operation_id: operationId,
        phase,
        payload_fingerprint: payloadFingerprint,
    } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            source: z.strictObject({ conversation_id: IdentifierSchema, revision: NonnegativeSafeIntegerSchema }),
            operation_id: IdentifierSchema,
            phase: z.enum(['policy', 'queue']),
            payload_fingerprint: ContentHashSchema,
        })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed control observation requires complete durable outbox indexes');
    if (root.source.conversation_id !== source.conversation_id || root.source.revision < source.revision + 1)
        throw new Error('Indexed control changed its accepted source');
    const store = boundedIndexedProcessingReader(storeInput);
    const receipt = await indexedRecordById(store, root, 'operation_receipts', operationId, OperationReceiptSchema);
    if (
        !receipt ||
        receipt.id !== operationId ||
        receipt.conversation_id !== source.conversation_id ||
        receipt.base_revision !== source.revision ||
        receipt.result_revision !== source.revision + 1 ||
        receipt.operation_kind !== 'processing' ||
        receipt.processing_operation?.phase !== phase ||
        receipt.payload_fingerprint !== payloadFingerprint
    )
        throw new Error('Indexed control lacks its exact accepted scheduled operation');
    if (phase === 'queue') {
        const command = await indexedProcessingRecord(
            store,
            root,
            'selected_queue_commands',
            operationId,
            IndexedProcessingQueueCommandSchema,
        );
        const relation = await indexedRecordById(
            store,
            root,
            'processing_by_operation',
            operationId,
            IndexedProcessingOperationJobsSchema,
        );
        if (
            !command ||
            command.operation_id !== operationId ||
            receipt.payload_fingerprint !== (await fingerprintJson(command)) ||
            relation?.operation_id !== operationId ||
            relation.receipt_fingerprint !== (await fingerprintJson(receipt)) ||
            relation.job_ids.length !== 1 ||
            relation.job_ids[0] !== receipt.processing_operation?.job_id
        )
            throw new Error('Indexed control lost its exact queued job association');
        await ownedIndexedProcessingJob(store, root, relation.job_ids[0]);
    } else if (receipt.processing_operation?.job_id !== undefined) {
        throw new Error('Indexed policy control unexpectedly nominated a processing job');
    }
    const pending = await loadIndexedPendingProcessingJobs(store, root, { limit: 16 });
    return {
        receipt,
        source: root.source,
        remaining_processing:
            pending.required_blocked_job_count > 0
                ? ('blocked' as const)
                : pending.unresolved_job_count > 0
                  ? ('pending' as const)
                  : ('none' as const),
    };
}

/** Point-addressed scheduled manual/budget job origin. A current pending marker alone cannot
 * authorize processing; the original queue command, receipt and one-job relation all agree. */
export async function loadIndexedProcessingQueuedJobAcceptance(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    jobIdInput: string,
) {
    const envelope = { root: rootInput, job_id: jobIdInput };
    if (!preflightJsonInput(envelope).success) throw new TypeError('Indexed queue origin is not bounded JSON');
    const { root, job_id: jobId } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            job_id: IdentifierSchema,
        })
        .parse(structuredClone(envelope));
    const store = boundedIndexedProcessingReader(storeInput);
    const job = await ownedIndexedProcessingJob(store, root, jobId);
    if (job.scope !== 'manual' && job.scope !== 'on_budget')
        throw new Error('Indexed selected job is not a manual or measured-budget queue');
    const command = await indexedProcessingRecord(
        store,
        root,
        'selected_queue_commands',
        job.source_operation_id,
        IndexedProcessingQueueCommandSchema,
    );
    const receipt = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        job.source_operation_id,
        OperationReceiptSchema,
    );
    if (
        !command ||
        !receipt ||
        receipt.operation_kind !== 'processing' ||
        receipt.processing_operation?.phase !== 'queue' ||
        receipt.processing_operation.job_id !== job.id ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.result_revision !== job.enqueue_revision ||
        receipt.payload_fingerprint !== (await fingerprintJson(command))
    )
        throw new Error('Indexed selected job has no exact accepted queue origin');
    return { job, command, receipt };
}

export type { IndexedProcessingQueueCommand } from './schemas/indexed-head.js';
export { IndexedProcessingQueueCommandSchema } from './schemas/indexed-head.js';

/** The owning host resolves each selected-capable processor before policy publication. The
 * callback is a private service capability, never a serialized part of the policy command.
 */
export async function stageIndexedProcessingPolicy(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    commandInput: IndexedProcessingPolicyCommand,
    assertSelectedProcessorSupported: (configuration: ProcessorConfiguration) => Promise<void>,
): Promise<StagedIndexedProcessingPhase> {
    const envelope = { root: rootInput, locator: locatorInput, command: commandInput };
    if (!preflightJsonInput(envelope, { max_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES }).success)
        throw new TypeError('Indexed processing policy is not bounded JSON');
    const { root, locator, command } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            command: IndexedProcessingPolicyCommandSchema,
        })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed processing policy requires complete durable outbox indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const fingerprint = await fingerprintJson(command);
    const prior = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        command.operation_id,
        OperationReceiptSchema,
    );
    if (prior) {
        if (
            prior.operation_kind !== 'processing' ||
            prior.processing_operation?.phase !== 'policy' ||
            prior.conversation_id !== root.source.conversation_id ||
            prior.result_revision !== command.expected_revision + 1 ||
            prior.payload_fingerprint !== fingerprint ||
            prior.base_revision !== command.expected_revision ||
            prior.recorded_at !== command.recorded_at ||
            prior.result_revision > root.source.revision ||
            canonicalJsonContentString(prior.processing_operation.superseded_job_ids ?? []) !==
                canonicalJsonContentString(command.supersede_job_ids ?? [])
        )
            throw new Error('Indexed processing policy retry conflicts with its accepted transition');
        return { root, locator, receipt: prior, applied: false };
    }
    if (root.source.revision !== command.expected_revision)
        throw new Error('Indexed processing policy source revision conflict');
    if (Date.parse(command.recorded_at) < Date.parse(root.updated_at))
        throw new Error('Indexed processing policy timestamp predates current source');
    if (command.processors.length > MAX_PROCESSING_STAGES_PER_OPERATION)
        throw new RangeError('Indexed processing policy exceeds its ordered stage bound');
    for (const configuration of command.processors) {
        if (canonicalJsonContentBytes(configuration.config).byteLength > MAX_PROCESSOR_CONFIGURATION_BYTES)
            throw new RangeError(`Indexed processor ${configuration.id} configuration exceeds durable bound`);
        if (command.enabled) await assertSelectedProcessorSupported(configuration);
    }
    const header = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    if (header.budget !== undefined && command.budget === undefined)
        throw new Error('Indexed policy must retain its accepted existing budget');
    const counts = assertIndexedRequiredIdentity(root, header);
    const supersede = new Set(command.supersede_job_ids ?? []);
    if (supersede.size !== (command.supersede_job_ids?.length ?? 0))
        throw new Error('Indexed policy supersession repeats a job');
    if (supersede.size && !command.supersession_reason)
        throw new Error('Indexed policy supersession requires an exact reason');
    if (supersede.size !== counts.unresolved_job_count)
        throw new Error('Indexed policy transition requires explicit supersession of every unresolved job');
    let unresolved = counts.unresolved_job_count;
    let requiredUnresolved = counts.required_unresolved_job_count;
    let requiredCount = counts.required_job_count;
    let requiredBlocked = counts.required_blocked_job_count;
    const directories = { ...root.directories };
    for (const jobId of supersede) {
        const job = await ownedIndexedProcessingJob(store, root, jobId);
        if (await indexedProcessingRecord(store, root, 'supersessions', jobId, ProcessingSupersessionReceiptSchema))
            throw new Error('Indexed policy cannot supersede an already settled job');
        const completion = await indexedProcessingRecord(
            store,
            root,
            'completions',
            jobId,
            ProcessingCompletionReceiptSchema,
        );
        if (completion && completion.status !== 'blocked')
            throw new Error('Indexed policy cannot supersede a successfully completed job');
        const attempt = await indexedProcessingRecord(store, root, 'attempts', jobId, ProcessingAttemptReceiptSchema);
        const output = await indexedProcessingRecord(store, root, 'outputs', jobId, ProcessingOutputReceiptSchema);
        if (attempt && !output) throw new Error('Indexed policy cannot supersede an unresolved external attempt');
        const pending = await removePagedRecord(store, directories.processing_pending, jobId);
        if (
            !pending.applied ||
            pending.removed?.storage !== 'marker' ||
            pending.removed.kind !== 'processing_pending' ||
            pending.removed.id !== jobId
        )
            throw new Error('Indexed policy supersession loses its exact pending job marker');
        if (pending.root === undefined) delete directories.processing_pending;
        else directories.processing_pending = pending.root;
        unresolved -= 1;
        if (job.required) {
            requiredUnresolved -= 1;
            if (completion?.status === 'blocked') requiredBlocked -= 1;
            const required = await removePagedRecord(store, directories.processing_required, jobId);
            if (
                !required.applied ||
                required.removed?.storage !== 'marker' ||
                required.removed.kind !== 'processing_required' ||
                required.removed.id !== jobId
            )
                throw new Error('Indexed policy supersession loses its exact required job marker');
            if (required.root === undefined) delete directories.processing_required;
            else directories.processing_required = required.root;
            requiredCount -= 1;
        }
        const supersession = ProcessingSupersessionReceiptSchema.parse({
            job_id: jobId,
            policy_operation_id: command.operation_id,
            reason: command.supersession_reason,
            recorded_at: command.recorded_at,
        });
        directories.processing_records = await putPagedRecord(
            store,
            directories.processing_records,
            tupleKey('supersessions', jobId),
            await stageRecord(store, 'processing_records', jobId, supersession),
        );
    }
    if (unresolved < 0 || requiredUnresolved < 0 || requiredCount < 0 || requiredBlocked < 0)
        throw new Error('Indexed policy supersession exceeds durable processing counts');
    if (unresolved !== 0 || directories.processing_pending !== undefined)
        throw new Error('Indexed policy transition requires explicit pending-job supersession');
    const receipt = createProcessingTransitionReceipt({
        source: root.source,
        operation_id: command.operation_id,
        payload_fingerprint: fingerprint,
        recorded_at: command.recorded_at,
        processing_operation: {
            phase: 'policy',
            policy_revision: header.policy_revision,
            superseded_job_ids: [...supersede],
        },
    });
    if (await getPagedRecord(store, directories.identifiers, receipt.id))
        throw new Error('Indexed policy operation identity is already accepted');
    directories.operation_receipts = await putPagedRecord(
        store,
        directories.operation_receipts,
        receipt.id,
        await stageRecord(store, 'operation_receipts', receipt.id, receipt),
    );
    if (header.selected_policy_operation_id !== undefined) {
        const previousCommand = await indexedProcessingRecord(
            store,
            root,
            'selected_policy_commands',
            header.selected_policy_operation_id,
            ProcessingPolicyCommandSchema,
        );
        const previousReceipt = await indexedRecordById(
            store,
            root,
            'operation_receipts',
            header.selected_policy_operation_id,
            OperationReceiptSchema,
        );
        if (
            !previousCommand ||
            !previousReceipt ||
            previousReceipt.processing_operation?.policy_revision === undefined ||
            previousReceipt.processing_operation.policy_revision + 1 !== header.policy_revision ||
            previousCommand.enabled !== header.enabled ||
            !sameIndexedRecord(previousCommand.processors, header.processors) ||
            !sameIndexedRecord(previousCommand.budget ?? null, header.budget ?? null)
        )
            throw new Error('Indexed policy transition lost its actual predecessor command');
        Object.assign(
            directories,
            await stageIndexedAcceptedProcessingPolicy(
                store,
                root.source,
                directories,
                previousCommand,
                previousReceipt,
            ),
        );
    }
    if (
        header.policy_revision === 0 &&
        (await getPagedRecord(store, directories.processing_records, tupleKey('policy_epochs', '0'))) !== undefined
    ) {
        const originalPolicy = await auditIndexedProcessingPolicyEpoch(store, root, 0);
        if (
            originalPolicy.enabled !== header.enabled ||
            !sameIndexedRecord(originalPolicy.processors, header.processors) ||
            !sameIndexedRecord(originalPolicy.budget ?? null, header.budget ?? null)
        )
            throw new Error('Indexed initial policy transition changed its genuine genesis evidence');
    }
    if (
        header.policy_revision === 0 &&
        header.selected_policy_operation_id === undefined &&
        (await getPagedRecord(store, directories.processing_records, tupleKey('policy_epochs', '0'))) === undefined
    )
        directories.processing_records = await putPagedRecord(
            store,
            directories.processing_records,
            tupleKey('policy_epochs', '0'),
            await stageRecord(
                store,
                'processing_records',
                '0',
                IndexedProcessingPolicyEpochSchema.parse({
                    version: 1,
                    kind: 'genesis',
                    policy_revision: 0,
                    source: root.source,
                    processing_header: root.processing_header,
                }),
            ),
        );
    Object.assign(
        directories,
        await stageIndexedAcceptedProcessingPolicy(
            store,
            { ...root.source, revision: receipt.result_revision },
            directories,
            command,
            receipt,
        ),
    );
    directories.identifiers = await putPagedRecord(store, directories.identifiers, receipt.id, {
        storage: 'marker',
        kind: 'operation receipt',
        id: receipt.id,
    });
    const {
        budget: _budget,
        coverage: _coverage,
        selected_policy_operation_id: _selectedPolicy,
        ...processing
    } = header;
    const nextHeader = await stageRecord(
        store,
        'processing_header',
        root.source.conversation_id,
        IndexedConversationProcessingHeaderSchema.parse({
            ...processing,
            enabled: command.enabled,
            policy_revision: header.policy_revision + 1,
            processors: command.processors,
            selected_policy_operation_id: receipt.id,
            selected_policy_origin: 'native_registry',
            ...(command.budget === undefined ? {} : { budget: command.budget }),
            unresolved_job_count: unresolved,
            required_unresolved_job_count: requiredUnresolved,
            required_job_count: requiredCount,
            required_blocked_job_count: requiredBlocked,
        }),
    );
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        directories,
        source: { ...root.source, revision: receipt.result_revision },
        updated_at: command.recorded_at,
        processing_header: { content_hash: nextHeader.content_hash, size_bytes: nextHeader.size_bytes },
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed policy root exceeds its manifest bound');
    return {
        root: nextRoot,
        locator: { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes },
        receipt,
        applied: true,
    };
}

export type { IndexedProcessingPolicyCommand } from './schemas/indexed-head.js';
export { IndexedProcessingPolicyCommandSchema } from './schemas/indexed-head.js';

export async function indexedReadinessIdentity(coverage: IndexedProcessingReadinessCoverage): Promise<string> {
    return fingerprintJson({
        profile: coverage.profile,
        context_fingerprint: coverage.context_fingerprint,
        policy_revision: coverage.policy_revision,
        target_fingerprint: coverage.target_fingerprint,
        measurement: coverage.measurement,
        required_job_count: coverage.required_job_count,
        ...(coverage.required_jobs_root === undefined ? {} : { required_jobs_root: coverage.required_jobs_root }),
    });
}

/** Finite built-in selected policy profile shared with native hosts and snapshot adoption. */
export function supportsIndexedRegisteredProcessingPolicy(
    header: Pick<z.infer<typeof IndexedConversationProcessingHeaderSchema>, 'enabled' | 'processors'>,
): boolean {
    const exchangeOnly =
        header.processors.length === 1 &&
        header.processors[0].id === INDEXED_EXCHANGE_PROCESSOR_ID &&
        header.processors[0].version === '1' &&
        header.processors[0].scope === 'on_append' &&
        Object.keys(header.processors[0].config).length === 0;
    const orderedText =
        header.processors.length > 0 &&
        header.processors.length <= MAX_PROCESSING_STAGES_PER_OPERATION &&
        header.processors.every(
            (processor) =>
                processor.id === 'externalize-text' &&
                processor.version === '1' &&
                (processor.scope === 'on_append' || processor.scope === 'manual' || processor.scope === 'on_budget') &&
                Object.keys(processor.config).length === 0,
        );
    const onAppend = header.processors.filter((processor) => processor.scope === 'on_append');
    const orderedToolResults =
        header.processors.length > 0 &&
        header.processors.length <= MAX_PROCESSING_STAGES_PER_OPERATION &&
        header.processors.some((processor) => isToolResultTextStrategy(processor.id, processor.version)) &&
        header.processors.every((processor) => {
            if (isToolResultTextStrategy(processor.id, processor.version)) {
                parseToolResultTextStrategy({
                    processor_id: processor.id,
                    processor_version: processor.version,
                    configuration: processor.config,
                });
                return (
                    supportsToolResultTextProcessingScope({
                        processor_id: processor.id,
                        processor_version: processor.version,
                        scope: processor.scope,
                    }) &&
                    (processor.scope === 'manual' || processor === onAppend[onAppend.length - 1])
                );
            }
            return (
                processor.id === 'externalize-text' &&
                processor.version === '1' &&
                (processor.scope === 'on_append' || processor.scope === 'manual' || processor.scope === 'on_budget') &&
                Object.keys(processor.config).length === 0
            );
        });
    return header.enabled && (exchangeOnly || orderedText || orderedToolResults);
}

/** Initial activation accepts disabled processing or the finite registered execution profile.
 * A disabled policy schedules no work. This predicate grants no accepted-source, count or readiness authority. */
export function supportsIndexedInheritedProcessingPolicy(
    header: Pick<z.infer<typeof IndexedConversationProcessingHeaderSchema>, 'enabled' | 'processors'>,
): boolean {
    return !header.enabled || supportsIndexedRegisteredProcessingPolicy(header);
}

function assertSupportedIndexedReadinessPolicy(
    header: z.infer<typeof IndexedConversationProcessingHeaderSchema>,
): void {
    if (header.enabled && !supportsIndexedRegisteredProcessingPolicy(header))
        throw new Error('Indexed readiness requires a registered bounded ordered text or whole-exchange policy');
}

/** Authenticate current immutable policy data without granting native execution capability.
 * Storage-only migration may retain pluggable policy data that its native host cannot execute.
 */
export async function authenticateIndexedCurrentPolicy(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    header: z.infer<typeof IndexedConversationProcessingHeaderSchema>,
): Promise<'materialized' | 'native_registry' | undefined> {
    const integrity = await hashContentBytes(canonicalJsonContentBytes(header));
    if (
        integrity.byte_length !== root.processing_header.size_bytes ||
        integrity.content_hash !== root.processing_header.content_hash
    )
        throw new Error('Indexed current policy header differs from its immutable root descriptor');
    const operationId = header.selected_policy_operation_id;
    if (!operationId) return undefined;
    const command = await indexedProcessingRecord(
        store,
        root,
        'selected_policy_commands',
        operationId,
        IndexedProcessingPolicyCommandSchema,
    );
    const receipt = await indexedRecordById(store, root, 'operation_receipts', operationId, OperationReceiptSchema);
    if (
        !command ||
        !receipt ||
        command.enabled !== header.enabled ||
        command.operation_id !== receipt.id ||
        receipt.operation_kind !== 'processing' ||
        receipt.processing_operation?.phase !== 'policy' ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.base_revision !== command.expected_revision ||
        receipt.result_revision !== command.expected_revision + 1 ||
        receipt.result_revision > root.source.revision ||
        receipt.recorded_at !== command.recorded_at ||
        receipt.processing_operation.policy_revision + 1 !== header.policy_revision ||
        receipt.payload_fingerprint !== (await fingerprintJson(command)) ||
        canonicalJsonContentString(command.processors) !== canonicalJsonContentString(header.processors) ||
        canonicalJsonContentString(command.budget ?? null) !== canonicalJsonContentString(header.budget ?? null)
    )
        throw new Error('Indexed selected processor policy lost its accepted registered command');
    // Materialized acceptance retains portable custom policy data, but does not certify the
    // native execution registry. A genuine indexed transition separately validates its host
    // registry before CAS, preserving registered extension policies in the portable core.
    return header.selected_policy_origin === 'materialized' ||
        receipt.processing_operation?.policy_command !== undefined
        ? 'materialized'
        : 'native_registry';
}

/** Native readiness requires finite built-in capability for materialized policy acceptance.
 * Exact native transitions preserve their separately host-validated registered extensions.
 */
export async function assertIndexedCurrentPolicy(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    header: z.infer<typeof IndexedConversationProcessingHeaderSchema>,
): Promise<void> {
    const origin = await authenticateIndexedCurrentPolicy(store, root, header);
    if (origin !== 'native_registry') assertSupportedIndexedReadinessPolicy(header);
}

function assertIndexedRequiredIdentity(
    root: IndexedConversationRoot,
    header: z.infer<typeof IndexedConversationProcessingHeaderSchema>,
) {
    const counts = assertIndexedProcessingCounts(header);
    if ((root.directories.processing_required === undefined) !== (counts.required_job_count === 0))
        throw new Error('Indexed required-job identity root differs from its complete count');
    return counts;
}

/** Only the complete bounded pending index can keep an over-budget target in `pending`.
 * A remembered policy name or an unrelated target job does not cover current model bytes.
 */
async function hasExactPendingBudgetJob(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    counts: ReturnType<typeof assertIndexedProcessingCounts>,
    targetFingerprint: string,
): Promise<boolean> {
    const page = await readPagedRecordRange(store, root.directories.processing_pending, { limit: 256 });
    if (page.has_more || page.entries.length !== counts.unresolved_job_count)
        throw new Error('Indexed budget pending index differs from its complete bounded count');
    let matching = false;
    for (const entry of page.entries) {
        if (
            entry.value.storage !== 'marker' ||
            entry.value.kind !== 'processing_pending' ||
            entry.value.id !== entry.key
        )
            throw new Error('Indexed budget pending page contains a foreign job identity');
        const job = await ownedIndexedProcessingJob(store, root, entry.key);
        if (job.required && job.scope === 'on_budget' && job.target_fingerprint === targetFingerprint) {
            const completion = await indexedProcessingRecord(
                store,
                root,
                'completions',
                job.id,
                ProcessingCompletionReceiptSchema,
            );
            if (!completion) matching = true;
        }
    }
    return matching;
}

/** Actual native measurement is supplied by the authenticated host compiler/count capability.
 * This data function cannot produce that capability. No completed-job or receipt history is scanned:
 * the persistent required-job root/count binds exact obligations independently of their pending set.
 */
export async function stageIndexedProcessingCoverage(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    commandInput: IndexedProcessingCoverageCommand,
): Promise<StagedIndexedProcessingCoverage> {
    const envelope = { root: rootInput, locator: locatorInput, command: commandInput };
    if (!preflightJsonInput(envelope).success) throw new TypeError('Indexed readiness input is not bounded JSON');
    const { root, locator, command } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            command: IndexedProcessingCoverageCommandSchema,
        })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed readiness requires its complete processing index profile');
    const store = boundedIndexedProcessingReader(storeInput);
    const requestFingerprint = await fingerprintJson(command);
    const prior = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        command.operation_id,
        OperationReceiptSchema,
    );
    if (prior) {
        const coverage = await indexedProcessingRecord(
            store,
            root,
            'indexed_coverage',
            command.operation_id,
            IndexedProcessingReadinessCoverageSchema,
        );
        if (
            !coverage ||
            prior.operation_kind !== 'processing' ||
            prior.processing_operation?.phase !== 'coverage' ||
            prior.payload_fingerprint !== requestFingerprint ||
            prior.base_revision !== command.expected_revision ||
            prior.result_revision !== coverage.evaluated_at_revision ||
            prior.result_revision > root.source.revision ||
            prior.conversation_id !== root.source.conversation_id ||
            prior.processing_operation.result_fingerprint !== (await fingerprintJson(coverage))
        )
            throw new Error('Indexed coverage retry differs from its exact retained evaluation');
        return { root, locator, receipt: prior, coverage, applied: false };
    }
    if (root.source.revision !== command.expected_revision)
        throw new Error('Indexed readiness source revision conflict');
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    await assertIndexedCurrentPolicy(store, root, header);
    const counts = assertIndexedRequiredIdentity(root, header);
    const selected = await loadIndexedProcessingContext(store, root, locator, false);
    const overBudget = header.budget !== undefined && command.measured_input_tokens > header.budget.max_input_tokens;
    const pendingBudget =
        overBudget && (await hasExactPendingBudgetJob(store, root, counts, command.target_fingerprint));
    const coverage = IndexedProcessingReadinessCoverageSchema.parse({
        version: 1,
        profile: INDEXED_CONVERSATION_PROCESSING_PROFILE,
        context_fingerprint: await indexedProcessingContextFingerprint(selected),
        policy_revision: header.policy_revision,
        target_fingerprint: command.target_fingerprint,
        measurement: {
            input_tokens: command.measured_input_tokens,
            tokenizer_id: command.tokenizer_id,
            fingerprint: command.measurement_fingerprint,
        },
        required_job_count: counts.required_job_count,
        ...(root.directories.processing_required === undefined
            ? {}
            : { required_jobs_root: root.directories.processing_required }),
        status:
            counts.required_blocked_job_count > 0 || (overBudget && !pendingBudget)
                ? 'blocked'
                : counts.unresolved_job_count > 0 || overBudget
                  ? 'pending'
                  : 'ready',
        evaluated_at_revision: root.source.revision + 1,
        recorded_at: command.recorded_at,
    });
    const receipt = createProcessingTransitionReceipt({
        source: root.source,
        operation_id: command.operation_id,
        payload_fingerprint: requestFingerprint,
        recorded_at: command.recorded_at,
        processing_operation: {
            phase: 'coverage',
            policy_revision: header.policy_revision,
            result_fingerprint: await fingerprintJson(coverage),
        },
    });
    const identity = await indexedReadinessIdentity(coverage);
    const directories = { ...root.directories };
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        tupleKey('indexed_coverage', receipt.id),
        await stageRecord(store, 'processing_records', receipt.id, coverage),
    );
    const existingIdentity = await getPagedRecord(store, directories.processing_coverage, identity);
    if (
        existingIdentity &&
        (existingIdentity.storage !== 'marker' || existingIdentity.kind !== 'indexed_processing_coverage')
    )
        throw new Error('Indexed coverage identity is occupied by a foreign record');
    directories.processing_coverage = await putPagedRecord(
        store,
        directories.processing_coverage,
        identity,
        { storage: 'marker', kind: 'indexed_processing_coverage', id: receipt.id },
        existingIdentity ? 'replace' : 'insert',
    );
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
        directories,
        source: { ...root.source, revision: receipt.result_revision },
        updated_at: command.recorded_at,
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed readiness root exceeds manifest bound');
    return {
        root: nextRoot,
        locator: { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes },
        receipt,
        coverage,
        applied: true,
    };
}

/** Processing-only witnesses cannot cross the strict native preparation boundary. */
export function indexedPreparationSelection(input: unknown) {
    const {
        lineage_entry_witnesses: _lineageEntries,
        sibling_compaction_ids: _siblingCompactions,
        tool_result_call_witnesses: _callWitnesses,
        tool_result_projection_witnesses: _projectionWitnesses,
        ...selected
    } = IndexedProcessingSelectedContextSchema.parse(input);
    return IndexedConversationSelectedContextSchema.parse({
        ...selected,
        completeness: 'selected_media_compaction_pending_admission',
    });
}

/** Common drained-processing fence for native preparation and a pending selected retrieval call.
 * Neither projection grants admission or byte access; callers must separately prove their exact
 * native request or selected call, owner and current physical head.
 */
async function loadIndexedSettledSelectedContext(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    purpose: 'preparation' | 'retrieval',
) {
    const input = { root: rootInput, locator: locatorInput };
    if (!preflightJsonInput(input).success) throw new TypeError('Indexed dry preparation is not bounded JSON');
    const { root, locator } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
        })
        .parse(structuredClone(input));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed dry preparation requires complete processing indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const header = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    // Processing is opt-in. Disabled preparation still proves the immutable policy and any
    // retained policy command, then passes the same complete pending/count fence below.
    // Retrieval independently owns call/requirement/custody authority.
    await assertIndexedCurrentPolicy(store, root, header);
    const counts = assertIndexedRequiredIdentity(root, header);
    if (
        counts.unresolved_job_count !== 0 ||
        counts.required_blocked_job_count !== 0 ||
        root.directories.processing_pending !== undefined
    )
        throw new Error('Indexed dry preparation still has unresolved processing obligations');
    if (purpose === 'retrieval') {
        // A retrieval call is accepted before its result exists. This view retains the same
        // drained-processing fence but verifies its pending call index instead of inventing a result.
        const selected = await loadIndexedSelectedContext(
            store,
            root,
            locator,
            INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
            true,
            true,
            'processing',
            true,
        );
        return indexedPreparationSelection(selected);
    }
    const selected = await loadIndexedProcessingContext(store, root, locator, false);
    return indexedPreparationSelection(selected);
}

/** Read-only summary derivation from a settled exact source. Pending application calls retain
 * their authenticated dependency/index witnesses; this processing-only view grants no preparation,
 * native measurement, provider transport, or readiness authority. */
export async function loadIndexedCheckpointSummarySelectedContext(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
) {
    const input = { root: rootInput, locator: locatorInput };
    if (!preflightJsonInput(input).success) throw new TypeError('Indexed checkpoint source is not bounded JSON');
    const { root, locator } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
        })
        .parse(structuredClone(input));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new IndexedPresentationNominationConflict(
            'Indexed checkpoint source requires complete processing indexes',
        );
    const store = boundedIndexedProcessingReader(storeInput);
    const header = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: root.source.conversation_id,
            ...root.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    const counts = assertIndexedRequiredIdentity(root, header);
    if (
        counts.unresolved_job_count !== 0 ||
        counts.required_blocked_job_count !== 0 ||
        root.directories.processing_pending !== undefined
    )
        throw new IndexedPresentationNominationConflict(
            'Indexed checkpoint source still has unresolved processing obligations',
        );
    if (header.enabled) await assertIndexedCurrentPolicy(store, root, header);
    return loadIndexedProcessingContext(store, root, locator, true);
}

/** Dry native preparation after the independently drained outbox. The host still counts the
 * native request and publishes exact target/measurement coverage before dispatch.
 */
export async function loadIndexedSettledProcessingSelectedContext(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
) {
    return loadIndexedSettledSelectedContext(store, root, locator, 'preparation');
}

/** A drained processing projection for an accepted read call whose terminal result is still pending.
 * It grants no execution or bytes: the caller must bind the exact selected call, active reader,
 * copied archive integrity and current owner/head before returning a bounded excerpt.
 */
export async function loadIndexedSettledRetrievalSelectedContext(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    locator: PagedRecordRef,
) {
    return loadIndexedSettledSelectedContext(store, root, locator, 'retrieval');
}

export interface IndexedReadySelectedContext {
    selection: z.infer<typeof IndexedConversationSelectedContextSchema>;
    coverage: IndexedProcessingReadinessCoverage;
}

/** Target/count-specific read barrier. An earlier ready evaluation cannot cover new jobs, context,
 * policy, target or native count. Coverage-only successors can reuse the same immutable index entry.
 * Hosts additionally recheck actual current execution/ownership and prepared bytes before transport.
 */
export async function loadIndexedReadySelectedContext(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    bindingInput: {
        target_fingerprint: string;
        measured_input_tokens: number;
        tokenizer_id: string;
        measurement_fingerprint: string;
    },
): Promise<IndexedReadySelectedContext> {
    const envelope = { root: rootInput, locator: locatorInput, binding: bindingInput };
    if (!preflightJsonInput(envelope).success) throw new TypeError('Indexed readiness barrier is not bounded JSON');
    const { root, locator, binding } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            binding: IndexedProcessingCoverageCommandSchema.omit({
                operation_id: true,
                expected_revision: true,
                recorded_at: true,
            }),
        })
        .parse(structuredClone(envelope));
    if (root.processing_index_profile !== INDEXED_CONVERSATION_PROCESSING_PROFILE)
        throw new Error('Indexed readiness barrier requires complete processing indexes');
    const store = boundedIndexedProcessingReader(storeInput);
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    await assertIndexedCurrentPolicy(store, root, header);
    const counts = assertIndexedRequiredIdentity(root, header);
    if (
        counts.unresolved_job_count !== 0 ||
        counts.required_blocked_job_count !== 0 ||
        root.directories.processing_pending !== undefined
    )
        throw new Error('Indexed readiness still has unresolved processing obligations');
    const selected = await loadIndexedProcessingContext(store, root, locator, false);
    const draft = IndexedProcessingReadinessCoverageSchema.parse({
        version: 1,
        profile: INDEXED_CONVERSATION_PROCESSING_PROFILE,
        context_fingerprint: await indexedProcessingContextFingerprint(selected),
        policy_revision: header.policy_revision,
        target_fingerprint: binding.target_fingerprint,
        measurement: {
            input_tokens: binding.measured_input_tokens,
            tokenizer_id: binding.tokenizer_id,
            fingerprint: binding.measurement_fingerprint,
        },
        required_job_count: counts.required_job_count,
        ...(root.directories.processing_required === undefined
            ? {}
            : { required_jobs_root: root.directories.processing_required }),
        status: 'ready',
        evaluated_at_revision: root.source.revision,
        recorded_at: root.updated_at,
    });
    const candidate = await getPagedRecord(
        store,
        root.directories.processing_coverage,
        await indexedReadinessIdentity(draft),
    );
    if (candidate?.storage !== 'marker' || candidate.kind !== 'indexed_processing_coverage')
        throw new Error('Indexed ready coverage for exact context/target/count is unavailable');
    const coverage = await indexedProcessingRecord(
        store,
        root,
        'indexed_coverage',
        candidate.id,
        IndexedProcessingReadinessCoverageSchema,
    );
    const receipt = await indexedRecordById(store, root, 'operation_receipts', candidate.id, OperationReceiptSchema);
    if (
        coverage?.status !== 'ready' ||
        !receipt ||
        receipt.operation_kind !== 'processing' ||
        receipt.processing_operation?.phase !== 'coverage' ||
        receipt.result_revision !== coverage.evaluated_at_revision ||
        receipt.result_revision > root.source.revision ||
        receipt.processing_operation.result_fingerprint !== (await fingerprintJson(coverage)) ||
        (await indexedReadinessIdentity(coverage)) !== (await indexedReadinessIdentity(draft))
    )
        throw new Error('Indexed ready coverage lost its exact retained receipt or obligation identity');
    return {
        selection: indexedPreparationSelection(selected),
        coverage,
    };
}

/** A generic bounded host closure witness in the existing processing index. Opaque binding JSON
 * is integrity evidence only: the portable core never interprets Temporal/tenant/registration
 * fields or grants a host action. The witness binds the PREVIOUS root, avoiding a Merkle cycle.
 */
export async function loadIndexedProcessingClosureWitness(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    operationIdInput: string,
) {
    const envelope = { root: rootInput, operation_id: operationIdInput };
    if (!preflightJsonInput(envelope).success) throw new TypeError('Indexed closure lookup is not bounded JSON');
    const root = IndexedConversationRootSchema.parse(structuredClone(rootInput));
    const operationId = IdentifierSchema.parse(operationIdInput);
    const store = boundedIndexedProcessingReader(storeInput);
    const descriptor = await getPagedRecord(
        store,
        root.directories.processing_records,
        tupleKey('closures', operationId),
    );
    if (!descriptor) return undefined;
    if (descriptor.storage !== 'record' || descriptor.kind !== 'processing_records' || descriptor.id !== operationId)
        throw new Error('Indexed closure point lookup has a foreign descriptor');
    const witness = await loadRecord(store, descriptor, IndexedProcessingClosureWitnessSchema);
    const marker = await getPagedRecord(store, root.directories.identifiers, operationId);
    if (
        witness.operation_id !== operationId ||
        witness.predecessor.source.conversation_id !== root.source.conversation_id ||
        witness.result_revision !==
            witness.predecessor.source.revision + (witness.publication === 'retention' ? 0 : 1) ||
        witness.result_revision > root.source.revision ||
        witness.binding_fingerprint !== (await fingerprintJson(witness.binding)) ||
        marker?.storage !== 'marker' ||
        marker.kind !== 'indexed processing closure' ||
        marker.id !== operationId
    )
        throw new Error('Indexed closure witness lost its exact immutable binding/identifier');
    return witness;
}

export async function stageIndexedProcessingClosureWitness(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    commandInput: z.infer<typeof IndexedProcessingClosureCommandSchema>,
) {
    const envelope = { root: rootInput, locator: locatorInput, command: commandInput };
    if (!preflightJsonInput(envelope, { max_bytes: 512 * 1024 }).success)
        throw new TypeError('Indexed closure publication is not bounded owned JSON');
    const root = IndexedConversationRootSchema.parse(structuredClone(rootInput));
    const locator = PagedRecordRefSchema.parse({ ...locatorInput });
    const command = IndexedProcessingClosureCommandSchema.parse(structuredClone(commandInput));
    const store = boundedIndexedProcessingReader(storeInput);
    const rootIntegrity = await hashContentBytes(canonicalJsonContentBytes(root));
    if (rootIntegrity.content_hash !== locator.content_hash || rootIntegrity.byte_length !== locator.size_bytes)
        throw new Error('Indexed closure predecessor locator differs from the immutable root');
    const retained = await loadIndexedProcessingClosureWitness(store, root, command.operation_id);
    const bindingFingerprint = await fingerprintJson(command.binding);
    if (retained) {
        if (
            retained.publication !== command.publication ||
            retained.predecessor.source.revision !== command.expected_revision ||
            retained.binding_fingerprint !== bindingFingerprint ||
            retained.recorded_at !== command.recorded_at
        )
            throw new Error('Indexed closure retry changes its original binding/source/time');
        return { root, locator, witness: retained, applied: false };
    }
    if (root.source.revision !== command.expected_revision)
        throw new Error('Indexed closure publication lost its exact current source');
    const pending = await loadIndexedPendingProcessingJobs(store, root, { limit: 1 });
    if (pending.has_more || pending.jobs.length || pending.unresolved_job_count || pending.required_blocked_job_count)
        throw new Error('Indexed closure cannot retire unresolved processing obligations');
    const witness = IndexedProcessingClosureWitnessSchema.parse({
        version: 1,
        ...(command.publication === undefined ? {} : { publication: command.publication }),
        operation_id: command.operation_id,
        predecessor: { source: root.source, root: locator },
        result_revision: root.source.revision + (command.publication === 'retention' ? 0 : 1),
        binding_fingerprint: bindingFingerprint,
        binding: command.binding,
        recorded_at: command.recorded_at,
    });
    const directories = { ...root.directories };
    directories.processing_records = await putPagedRecord(
        store,
        directories.processing_records,
        tupleKey('closures', command.operation_id),
        await stageRecord(store, 'processing_records', command.operation_id, witness),
    );
    directories.identifiers = await putPagedRecord(store, directories.identifiers, command.operation_id, {
        storage: 'marker',
        kind: 'indexed processing closure',
        id: command.operation_id,
    });
    const next = IndexedConversationRootSchema.parse({
        ...root,
        directories,
        source: { ...root.source, revision: witness.result_revision },
        // Retention changes only the physical indexes. It is never a canonical content mutation.
        updated_at: command.publication === 'retention' ? root.updated_at : command.recorded_at,
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, next);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed closing root exceeds its manifest bound');
    return {
        root: next,
        locator: { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes },
        witness,
        applied: true,
    };
}

/** Reconstruct the exact closing root from its immutable predecessor and the witness selected by
 * the authenticated current index. Scratch writes have no external effects. A later epoch/head is
 * never returned in place of the historical close acknowledgement.
 */
export async function recoverIndexedProcessingClosureRoot(
    store: IndexedConversationRecordStore,
    currentRoot: IndexedConversationRoot,
    operationId: string,
) {
    const witness = await loadIndexedProcessingClosureWitness(store, currentRoot, operationId);
    if (!witness) return undefined;
    const scratch = createIndexedProcessingScratchStore(store, async () => {
        throw new Error('Indexed closure reconstruction cannot grant or publish an asset');
    });
    const body = await scratch.store.readRecord({
        storage: 'record',
        kind: 'root',
        id: witness.predecessor.source.conversation_id,
        ...witness.predecessor.root,
    });
    const raw: unknown = JSON.parse(new TextDecoder().decode(body));
    if (!preflightJsonInput(raw, { max_bytes: INDEXED_CONVERSATION_ROOT_MAX_BYTES }).success)
        throw new TypeError('Indexed closure predecessor is not bounded JSON');
    const predecessor = IndexedConversationRootSchema.parse(raw);
    if ((await fingerprintJson(predecessor.source)) !== (await fingerprintJson(witness.predecessor.source)))
        throw new Error('Indexed closure predecessor has a foreign source');
    const recovered = await stageIndexedProcessingClosureWitness(scratch.store, predecessor, witness.predecessor.root, {
        ...(witness.publication === undefined ? {} : { publication: witness.publication }),
        operation_id: witness.operation_id,
        expected_revision: witness.predecessor.source.revision,
        recorded_at: witness.recorded_at,
        binding: witness.binding,
    });
    if (!recovered.applied || (await fingerprintJson(recovered.witness)) !== (await fingerprintJson(witness)))
        throw new Error('Indexed closure reconstruction differs from its retained witness');
    return { root: recovered.root, locator: recovered.locator, witness };
}

/** A selected-call nomination conflict; transport and durable storage errors retain their own type. */
export class IndexedToolCallSelectionConflict extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'IndexedToolCallSelectionConflict';
    }
}

/** Active definition metadata comes from the same immutable context header. No turn/history scan. */
export async function loadIndexedActiveToolDefinitions(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
) {
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'context_header', id: root.source.conversation_id, ...root.context_header },
        IndexedConversationContextHeaderSchema,
    );
    if (header.active_tool_definition_ids.length > 4096)
        throw new RangeError('Indexed active tool definitions exceed the selected dependency bound');
    const definitions: z.infer<typeof ToolDefinitionSchema>[] = [];
    for (const id of header.active_tool_definition_ids) {
        const definition = await indexedRecordById(store, root, 'tool_definitions', id, ToolDefinitionSchema);
        if (!definition || definition.id !== id)
            throw new IndexedToolCallSelectionConflict('Indexed active definition lost its exact identity');
        definitions.push(definition);
    }
    return definitions;
}

async function assertIndexedActiveCallDefinition(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    call: Pick<z.infer<typeof ApplicationToolCallBlockSchema>, 'definition_id' | 'tool_name'>,
) {
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'context_header', id: root.source.conversation_id, ...root.context_header },
        IndexedConversationContextHeaderSchema,
    );
    if (header.active_tool_definition_ids.length > 4096)
        throw new RangeError('Indexed program catalog exceeds its selected dependency bound');
    const definition =
        call.definition_id !== undefined && header.active_tool_definition_ids.includes(call.definition_id)
            ? await indexedRecordById(store, root, 'tool_definitions', call.definition_id, ToolDefinitionSchema)
            : undefined;
    if (!definition || definition.id !== call.definition_id || definition.name !== call.tool_name)
        throw new IndexedToolCallSelectionConflict('Indexed program call has no exact active definition');
    return definition;
}

/** Select a genuine program operation through bounded call/turn/acceptance point lookups. */
export async function loadIndexedProgramToolCallSelection(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    sourceInput: z.infer<typeof ToolCallSourceRefSchema>,
) {
    if (!preflightJsonInput(sourceInput, { max_bytes: 16 * 1024 }).success)
        throw new RangeError('Indexed program source exceeds its nomination bound');
    const root = IndexedConversationRootSchema.parse(rootInput);
    const source = ToolCallSourceRefSchema.parse(structuredClone(sourceInput));
    if (
        source.conversation.conversation_id !== root.source.conversation_id ||
        source.conversation.revision !== root.source.revision ||
        root.tool_call_state_complete !== true
    )
        throw new IndexedToolCallSelectionConflict('Indexed program call requires its exact current source');
    const state = await indexedRecordById(store, root, 'tool_call_states', source.call_id, IndexedCallStateSchema);
    const open = await indexedRecordById(store, root, 'open_tool_calls', source.call_id, IndexedOpenToolCallSchema);
    if (
        !state ||
        !open ||
        state.call_id !== source.call_id ||
        state.turn_id !== source.turn_id ||
        state.block_id !== source.block_id ||
        state.call_fingerprint !== source.call_fingerprint ||
        state.result_block_id !== undefined ||
        state.terminal_receipt_id !== undefined ||
        !sameIndexedRecord(open, {
            call_id: state.call_id,
            turn_id: state.turn_id,
            block_id: state.block_id,
            call_fingerprint: state.call_fingerprint,
        })
    )
        throw new IndexedToolCallSelectionConflict('Indexed program call differs from its exact open-call index');
    const turn = await loadIndexedProjectedTurn(store, root, source.turn_id);
    const call = turn.selected_blocks[0];
    const binding = await getPagedRecord(store, root.directories.turn_acceptances, source.turn_id);
    const receipt =
        binding?.storage === 'marker' && binding.kind === 'turn_acceptance'
            ? await indexedRecordById(store, root, 'operation_receipts', binding.id, OperationReceiptSchema)
            : undefined;
    const entryId = receipt?.accepted_context_entry_ids?.[0];
    const entry =
        entryId === undefined
            ? undefined
            : await indexedRecordById(store, root, 'context_entries', entryId, ContextEntrySchema);
    if (
        turn.completeness !== 'full_turn' ||
        turn.header.id !== source.turn_id ||
        turn.header.kind !== 'program' ||
        turn.header.authority !== 'ordinary' ||
        turn.header.status !== 'completed' ||
        turn.header.provenance.type !== 'inserted' ||
        turn.header.parent_turn_id !== undefined ||
        turn.header.execution_id !== undefined ||
        turn.selected_blocks.length !== 1 ||
        call?.type !== 'tool_call' ||
        call.executor !== 'application' ||
        call.id !== source.block_id ||
        call.call_id !== source.call_id ||
        call.definition_id === undefined ||
        call.native_id !== undefined ||
        call.arguments.type !== 'json' ||
        call.arguments.value === null ||
        typeof call.arguments.value !== 'object' ||
        Array.isArray(call.arguments.value) ||
        (await fingerprintJson(call)) !== source.call_fingerprint ||
        !receipt ||
        receipt.operation_kind !== undefined ||
        receipt.id !== binding?.id ||
        receipt.id !== turn.header.provenance.operation_id ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.result_revision !== receipt.base_revision + 1 ||
        receipt.result_revision > root.source.revision ||
        receipt.recorded_at !== turn.header.timestamps.recorded_at ||
        !sameIndexedRecord(receipt.accepted_turn_ids, [source.turn_id]) ||
        receipt.accepted_context_entry_ids?.length !== 1 ||
        receipt.accepted_generation_ids === undefined ||
        receipt.accepted_asset_ids === undefined ||
        receipt.accepted_execution_receipt_ids === undefined ||
        receipt.accepted_tool_definition_ids === undefined ||
        receipt.accepted_tool_selection === undefined ||
        !sameIndexedRecord(receipt.accepted_generation_ids, []) ||
        !sameIndexedRecord(receipt.accepted_asset_ids, []) ||
        !sameIndexedRecord(receipt.accepted_execution_receipt_ids, []) ||
        !sameIndexedRecord(receipt.accepted_tool_definition_ids, []) ||
        !sameIndexedRecord(receipt.accepted_tool_selection, { kind: 'unchanged' }) ||
        (receipt.accepted_retrieval_requirements?.length ?? 0) !== 0 ||
        entry?.type !== 'source_turn' ||
        entry.turn_id !== source.turn_id ||
        entry.block_ids !== undefined ||
        !sameIndexedRecord(receipt.accepted_context_entries, [entry]) ||
        receipt.payload_fingerprint !==
            (await fingerprintJson({
                turns: [{ ...turn.header, blocks: turn.selected_blocks }],
                context_entries: [entry],
            }))
    )
        throw new IndexedToolCallSelectionConflict('Indexed program call lost its exact inserted operation proof');
    // Retained entry bytes do not establish active membership after an ordered context edit.
    // This metadata-only working set is bounded by the existing aggregate active-context profile;
    // it does not materialize turns, model output, assets or generations.
    const active = await loadIndexedActiveContext(store, root);
    if (!active.entries.some((candidate) => sameIndexedRecord(candidate, entry)))
        throw new IndexedToolCallSelectionConflict('Indexed program call source is no longer active in context');
    const definition = await assertIndexedActiveCallDefinition(store, root, call);
    const selected = {
        source,
        call: ApplicationToolCallBlockSchema.parse(call),
        assets: {},
        definition,
        operation_receipt: receipt,
    };
    if (!preflightJsonInput(selected, { max_bytes: 512 * 1024 }).success)
        throw new RangeError('Indexed program selection exceeds its working-set bound');
    return selected;
}

/** Select one exact accepted generated application call and only its hydration dependencies. */
export async function loadIndexedToolCallSelection(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    sourceInput: z.infer<typeof ToolCallSourceRefSchema>,
) {
    if (!preflightJsonInput(sourceInput, { max_bytes: 16 * 1024 }).success)
        throw new RangeError('Indexed tool source exceeds the nomination bound');
    const source = ToolCallSourceRefSchema.parse(structuredClone(sourceInput));
    if (
        source.conversation.conversation_id !== root.source.conversation_id ||
        source.conversation.revision !== root.source.revision ||
        root.tool_call_state_complete !== true
    )
        throw new IndexedToolCallSelectionConflict('Indexed tool call requires its exact complete accepted source');
    const state = await indexedRecordById(store, root, 'tool_call_states', source.call_id, IndexedCallStateSchema);
    if (
        !state ||
        state.turn_id !== source.turn_id ||
        state.block_id !== source.block_id ||
        state.call_id !== source.call_id ||
        state.call_fingerprint !== source.call_fingerprint
    )
        throw new IndexedToolCallSelectionConflict('Indexed tool call differs from its exact accepted call index');
    const turn = await loadIndexedProjectedTurn(store, root, source.turn_id, [source.block_id]);
    const call = turn.selected_blocks[0];
    if (
        call?.type !== 'tool_call' ||
        call.executor !== 'application' ||
        call.call_id !== source.call_id ||
        call.id !== source.block_id ||
        (await hashContentBytes(canonicalJsonContentBytes(call))).content_hash !== source.call_fingerprint ||
        turn.header.id !== source.turn_id ||
        turn.header.kind !== 'agent' ||
        turn.header.provenance.type !== 'generated' ||
        !('generation_id' in turn.header) ||
        turn.header.generation_id === undefined
    )
        throw new IndexedToolCallSelectionConflict('Indexed selected call changed its accepted generated content');
    const generation = await indexedRecordById(store, root, 'generations', turn.header.generation_id, GenerationSchema);
    const accepted = await getPagedRecord(store, root.directories.generation_acceptances, turn.header.generation_id);
    if (!generation || accepted?.storage !== 'marker' || accepted.kind !== 'generation_acceptance')
        throw new IndexedToolCallSelectionConflict('Indexed tool call has no exact accepted generation');
    const receipt = await indexedRecordById(store, root, 'operation_receipts', accepted.id, OperationReceiptSchema);
    if (
        !receipt ||
        generation.id !== turn.header.generation_id ||
        generation.record_source !== 'executed' ||
        receipt.id !== accepted.id ||
        receipt.conversation_id !== root.source.conversation_id ||
        !receipt.accepted_generation_ids?.includes(generation.id) ||
        !receipt.accepted_turn_ids?.includes(source.turn_id) ||
        receipt.base_revision !== generation.source.revision ||
        receipt.result_revision > root.source.revision ||
        receipt.result_revision !== receipt.base_revision + 1 ||
        generation.source.conversation_id !== root.source.conversation_id ||
        generation.request_receipt.source.conversation_id !== root.source.conversation_id ||
        generation.request_receipt.source.revision !== generation.source.revision ||
        generation.request_receipt.request_id !== generation.request_id
    )
        throw new IndexedToolCallSelectionConflict(
            'Indexed tool call differs from its accepted response/request chain',
        );
    const context = await loadRecord(
        store,
        { storage: 'record', kind: 'context_header', id: root.source.conversation_id, ...root.context_header },
        IndexedConversationContextHeaderSchema,
    );
    const definition =
        call.definition_id === undefined || !context.active_tool_definition_ids.includes(call.definition_id)
            ? undefined
            : await indexedRecordById(store, root, 'tool_definitions', call.definition_id, ToolDefinitionSchema);
    if (
        call.definition_id !== undefined &&
        (!definition || definition.id !== call.definition_id || definition.name !== call.tool_name)
    )
        throw new IndexedToolCallSelectionConflict(
            'Indexed call definition is absent from its accepted active catalog',
        );
    const assets: Record<string, Asset> = {};
    if (call.arguments.type === 'externalized_json') {
        if (call.arguments.hydration.length > 4096)
            throw new RangeError('Indexed tool hydration exceeds the dependency bound');
        for (const reference of call.arguments.hydration) {
            if (Object.hasOwn(assets, reference.asset_id)) continue;
            const asset = await indexedRecordById(store, root, 'assets', reference.asset_id, AssetSchema);
            if (!asset || asset.id !== reference.asset_id || asset.content_hash !== reference.content_hash)
                throw new IndexedToolCallSelectionConflict('Indexed tool hydration lost its exact accepted asset');
            Object.defineProperty(assets, asset.id, { value: asset, enumerable: true });
        }
    }
    const selected = { source, call, assets, ...(definition === undefined ? {} : { definition }) };
    if (!preflightJsonInput(selected, { max_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES }).success)
        throw new RangeError('Indexed tool call selection exceeds its working-set bound');
    return selected;
}

/** Retained terminal facts remain verifiable after either body is deleted; this grants no execution authority. */
async function loadIndexedTerminalCallWitness(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    state: z.infer<typeof IndexedCallStateSchema>,
) {
    if (!root.tool_call_state_complete || !state.terminal_receipt_id || !state.result_block_id)
        throw new Error('Indexed terminal call has an incomplete retained fence');
    const acceptedTurn = async (turnId: string) => {
        const accepted = await getPagedRecord(store, root.directories.turn_acceptances, turnId);
        const operation =
            accepted?.storage === 'marker' && accepted.kind === 'turn_acceptance'
                ? await indexedRecordById(store, root, 'operation_receipts', accepted.id, OperationReceiptSchema)
                : undefined;
        if (
            !operation ||
            operation.operation_kind !== undefined ||
            operation.conversation_id !== root.source.conversation_id ||
            !operation.accepted_turn_ids?.includes(turnId) ||
            operation.result_revision > root.source.revision ||
            operation.result_revision !== operation.base_revision + 1
        )
            throw new Error('Indexed terminal fact lost its exact accepted append');
        return { operation, turn: await loadIndexedAcceptedTurn(store, root, turnId, operation) };
    };
    const original = await acceptedTurn(state.turn_id);
    const call = original.turn.selected_blocks.find((block) => block.id === state.block_id);
    if (
        call?.type !== 'tool_call' ||
        call.call_id !== state.call_id ||
        (await fingerprintJson(call)) !== state.call_fingerprint ||
        original.turn.header.kind !== 'agent'
    )
        throw new Error('Indexed terminal fact differs from its original call');
    // Imported closed facts can be removed as historical bodies, without inventing a
    // runnable generation/request. Execution selection retains its separate strict guard.
    const archival = original.turn.header.provenance.type === 'imported';
    const verifyGeneration = async (accepted: Awaited<ReturnType<typeof acceptedTurn>>, imported: boolean) => {
        const header = accepted.turn.header;
        if (
            header.kind !== 'agent' ||
            (imported ? header.provenance.type !== 'imported' : header.provenance.type !== 'generated')
        )
            throw new Error('Indexed terminal call lost its accepted generation/request chain');
        if (!('generation_id' in header) || header.generation_id === undefined) {
            if (imported) return;
            throw new Error('Indexed terminal call lost its accepted generation/request chain');
        }
        const generation = await indexedRecordById(store, root, 'generations', header.generation_id, GenerationSchema);
        const generationAcceptance = await getPagedRecord(
            store,
            root.directories.generation_acceptances,
            header.generation_id,
        );
        if (
            !generation ||
            generation.id !== header.generation_id ||
            generationAcceptance?.storage !== 'marker' ||
            generationAcceptance.kind !== 'generation_acceptance' ||
            generationAcceptance.id !== accepted.operation.id ||
            !accepted.operation.accepted_generation_ids?.includes(generation.id) ||
            generation.source.conversation_id !== root.source.conversation_id ||
            generation.source.revision !== accepted.operation.base_revision
        )
            throw new Error('Indexed terminal call lost its accepted generation/request chain');
        if (imported) {
            if (generation.record_source !== 'imported')
                throw new Error('Indexed imported terminal fact changed its recorded generation provenance');
        } else if (
            generation.record_source !== 'executed' ||
            generation.request_receipt.source.conversation_id !== root.source.conversation_id ||
            generation.request_receipt.source.revision !== generation.source.revision ||
            generation.request_receipt.request_id !== generation.request_id
        )
            throw new Error('Indexed terminal call lost its accepted generation/request chain');
    };
    await verifyGeneration(original, archival);
    const receipt = await indexedRecordById(
        store,
        root,
        'execution_receipts',
        state.terminal_receipt_id,
        ExecutionReceiptSchema,
    );
    if (
        !receipt?.result_turn_id ||
        receipt.id !== state.terminal_receipt_id ||
        receipt.call_id !== call.call_id ||
        receipt.executor !== call.executor
    )
        throw new Error('Indexed terminal call lost its original execution receipt');
    const source = receipt.call_source;
    if (
        (call.executor === 'application' && !source && !archival) ||
        (source &&
            (source.conversation.conversation_id !== root.source.conversation_id ||
                source.conversation.revision < original.operation.result_revision ||
                source.conversation.revision > root.source.revision ||
                source.turn_id !== state.turn_id ||
                source.block_id !== state.block_id ||
                source.call_id !== state.call_id ||
                source.call_fingerprint !== state.call_fingerprint))
    )
        throw new Error('Indexed terminal receipt changed its exact original call source');
    const accepted = await acceptedTurn(receipt.result_turn_id);
    const result = accepted.turn.selected_blocks.find((block) => block.id === state.result_block_id);
    if (
        result?.type !== 'tool_result' ||
        result.call_id !== state.call_id ||
        result.status !== receipt.status ||
        !accepted.operation.accepted_execution_receipt_ids?.includes(receipt.id)
    )
        throw new Error('Indexed terminal result lost its exact accepted receipt');
    if (accepted.turn.header.kind === 'tool') {
        if (
            accepted.turn.header.execution_id !== receipt.id &&
            !(
                archival &&
                accepted.turn.header.execution_id === undefined &&
                (accepted.turn.header.provenance.type === 'received' ||
                    accepted.turn.header.provenance.type === 'imported')
            )
        )
            throw new Error('Indexed terminal result turn has another execution identity');
    } else if (call.executor === 'provider' && accepted.turn.header.kind === 'agent') {
        await verifyGeneration(accepted, archival);
    } else throw new Error('Indexed terminal result has no authentic result turn');
    await assertToolResultReceiptFingerprint(result, receipt);
}

/** Point-check the current duplicate-result fence without scanning turns or receipt history. */
export async function loadIndexedToolCallTerminalResult(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    source: z.infer<typeof ToolCallSourceRefSchema>,
) {
    if (
        root.source.conversation_id !== source.conversation.conversation_id ||
        root.source.revision < source.conversation.revision ||
        root.tool_call_state_complete !== true
    )
        throw new IndexedToolCallSelectionConflict('Indexed current call fence differs from its accepted source');
    const state = await indexedRecordById(store, root, 'tool_call_states', source.call_id, IndexedCallStateSchema);
    if (
        !state ||
        state.call_id !== source.call_id ||
        state.turn_id !== source.turn_id ||
        state.block_id !== source.block_id ||
        state.call_fingerprint !== source.call_fingerprint
    )
        throw new IndexedToolCallSelectionConflict('Indexed current call fence changed its original call binding');
    if (state.result_block_id === undefined && state.terminal_receipt_id === undefined) return false;
    if (state.result_block_id === undefined || state.terminal_receipt_id === undefined)
        throw new IndexedToolCallSelectionConflict('Indexed current call has an incomplete terminal result fence');

    const receipt = await indexedRecordById(
        store,
        root,
        'execution_receipts',
        state.terminal_receipt_id,
        ExecutionReceiptSchema,
    );
    if (!receipt?.result_turn_id)
        throw new IndexedToolCallSelectionConflict('Indexed terminal receipt lacks its exact result turn');
    const accepted = await getPagedRecord(store, root.directories.turn_acceptances, receipt.result_turn_id);
    const operation =
        accepted?.storage === 'marker' && accepted.kind === 'turn_acceptance'
            ? await indexedRecordById(store, root, 'operation_receipts', accepted.id, OperationReceiptSchema)
            : undefined;
    if (!operation) throw new Error('Indexed terminal result lost its accepted operation');
    // A compacted result's committed validation proves its immutable block/terminal hashes.
    // Resolve only the original operation's finite processing cohort, never active/cold history.
    let validatedOriginal = false;
    if (root.processing_index_profile === INDEXED_CONVERSATION_PROCESSING_PROFILE) {
        const terminalValidationJob = await auditIndexedToolResultTerminalValidation(store, root, receipt.id);
        const cohort =
            terminalValidationJob === undefined
                ? await indexedRecordById(
                      store,
                      root,
                      'processing_by_operation',
                      operation.id,
                      IndexedProcessingOperationJobsSchema,
                  )
                : undefined;
        if (
            cohort &&
            (cohort.operation_id !== operation.id || cohort.receipt_fingerprint !== (await fingerprintJson(operation)))
        )
            throw new IndexedToolCallSelectionConflict('Indexed terminal cohort changed its accepted operation');
        const validationJobs = terminalValidationJob === undefined ? (cohort?.job_ids ?? []) : [terminalValidationJob];
        if (validationJobs.length) {
            for (const jobId of validationJobs) {
                const job = await indexedProcessingRecord(store, root, 'jobs', jobId, ProcessingJobSchema);
                if (!job) throw new IndexedToolCallSelectionConflict('Indexed terminal cohort lost its original job');
                if (!isToolResultTextProcessor(job)) continue;
                const completion = await indexedProcessingRecord(
                    store,
                    root,
                    'completions',
                    jobId,
                    ProcessingCompletionReceiptSchema,
                );
                if (completion?.status !== 'applied') continue;
                const resolution = await indexedProcessingRecord(
                    store,
                    root,
                    'resolved_inputs',
                    jobId,
                    ProcessingResolvedInputSchema,
                );
                if (!resolution?.source_turn_ids.includes(receipt.result_turn_id)) continue;
                if (terminalValidationJob !== jobId) await auditIndexedToolResultValidation(store, root, jobId);
                const validation = await indexedProcessingRecord(
                    store,
                    root,
                    'tool_result_validations',
                    jobId,
                    IndexedToolResultValidationSchema,
                );
                const descriptor = await getPagedRecord(store, root.directories.blocks, state.result_block_id);
                const bound = (family: 'turns' | 'blocks' | 'execution_receipts', id: string) =>
                    validation?.dependencies.some((item) => item.family === family && item.descriptor.id === id);
                if (
                    !bound('turns', receipt.result_turn_id) ||
                    !bound('blocks', state.result_block_id) ||
                    !bound('execution_receipts', receipt.id) ||
                    descriptor?.storage !== 'record' ||
                    descriptor.content_hash !== receipt.result_fingerprint
                )
                    throw new IndexedToolCallSelectionConflict(
                        'Indexed terminal validation lost its original result hash',
                    );
                validatedOriginal = true;
                break;
            }
        }
    }
    const originalHeader = validatedOriginal
        ? await indexedRecordById(store, root, 'turns', receipt.result_turn_id, IndexedConversationTurnHeaderSchema)
        : undefined;
    if (
        validatedOriginal &&
        (originalHeader?.source !== 'ordinary' ||
            originalHeader.block_ids.length !== 1 ||
            originalHeader.block_ids[0] !== state.result_block_id)
    )
        throw new IndexedToolCallSelectionConflict('Indexed terminal validation lost its exact original turn header');
    const originalTurn = validatedOriginal
        ? undefined
        : await loadIndexedAcceptedTurn(store, root, receipt.result_turn_id, operation);
    const resultHeader = originalHeader?.turn ?? originalTurn?.header;
    if (!resultHeader) throw new IndexedToolCallSelectionConflict('Indexed terminal original turn header is absent');
    const originalResult = originalTurn?.selected_blocks.find((block) => block.id === state.result_block_id);
    const result = validatedOriginal
        ? { type: 'tool_result' as const, id: state.result_block_id, call_id: receipt.call_id, status: receipt.status }
        : originalResult;
    if (
        result?.type !== 'tool_result' ||
        result.id !== state.result_block_id ||
        result.call_id !== source.call_id ||
        !receipt ||
        receipt.id !== state.terminal_receipt_id ||
        receipt.call_id !== source.call_id ||
        receipt.executor !== 'application' ||
        receipt.status !== result.status ||
        !receipt.call_source ||
        canonicalJsonContentString(receipt.call_source) !== canonicalJsonContentString(source)
    )
        throw new IndexedToolCallSelectionConflict('Indexed terminal result lost its exact execution/source witness');
    if (!receipt.result_turn_id)
        throw new IndexedToolCallSelectionConflict('Indexed terminal receipt lacks its exact result turn');
    if (
        resultHeader.kind !== 'tool' ||
        resultHeader.execution_id !== receipt.id ||
        !operation?.accepted_turn_ids?.includes(receipt.result_turn_id) ||
        !operation.accepted_execution_receipt_ids?.includes(receipt.id) ||
        operation.conversation_id !== root.source.conversation_id ||
        operation.result_revision > root.source.revision
    )
        throw new IndexedToolCallSelectionConflict('Indexed terminal result lacks its accepted operation/turn witness');
    if (originalResult?.type === 'tool_result') await assertToolResultReceiptFingerprint(originalResult, receipt);
    return true;
}

/** Historical presentation is selected by accepted IDs, never by rehydrating a full document. */
const INDEXED_PRESENTATION_MAX_RECORD_BYTES = 32 * 1024 * 1024;
const INDEXED_TOPIC_MAX_TURNS = 4096;
const INDEXED_TOPIC_MAX_TEXT_BYTES = 16 * 1024 * 1024;

/** A caller nominated a different historical receipt, turn or head. Storage failures do not use this type. */
export class IndexedPresentationNominationConflict extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'IndexedPresentationNominationConflict';
    }
}

/** The exact source exceeds the explicit selected-read or topic profile; nothing is truncated. */
export class IndexedPresentationCapacityError extends RangeError {
    constructor(message: string) {
        super(message);
        this.name = 'IndexedPresentationCapacityError';
    }
}

function indexedPresentationReadStore(underlying: IndexedConversationRecordStore): IndexedConversationRecordStore {
    let bytes = 0;
    let pages = 0;
    let records = 0;
    const charge = (size: number, kind: 'page' | 'record') => {
        if (!Number.isSafeInteger(size) || size < 1)
            throw new Error('Indexed historical presentation has an invalid immutable descriptor');
        if (size > INDEXED_PRESENTATION_MAX_RECORD_BYTES - bytes)
            throw new IndexedPresentationCapacityError(
                'Indexed historical presentation exceeds its bounded selected-record profile',
            );
        if (kind === 'page' ? ++pages > 4096 : ++records > 8192)
            throw new IndexedPresentationCapacityError(
                'Indexed historical presentation exceeds its bounded selected-read profile',
            );
        bytes += size;
    };
    return {
        async read(ref) {
            charge(ref.size_bytes, 'page');
            return underlying.read(ref);
        },
        async readRecord(ref) {
            charge(ref.size_bytes, 'record');
            return underlying.readRecord(ref);
        },
        async write() {
            throw new Error('Indexed historical presentation is read-only');
        },
        async writeRecord() {
            throw new Error('Indexed historical presentation is read-only');
        },
    };
}

async function presentationRecord<Shape extends z.ZodType>(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    family: keyof IndexedConversationRoot['directories'],
    id: string,
    schema: Shape,
    nominated = false,
): Promise<z.infer<Shape>> {
    const descriptor = await getPagedRecord(store, root.directories[family], id);
    if (descriptor?.storage !== 'record') {
        if (nominated) throw new IndexedPresentationNominationConflict('Indexed historical receipt was not accepted');
        throw new Error('Indexed historical presentation dependency is missing');
    }
    return loadRecord(store, descriptor, schema);
}

/** Missing completeness is never interpreted as an empty accepted history. */
export class IndexedAcceptedOutputHistoryUpgradeRequired extends Error {
    constructor() {
        super('Indexed accepted-output history requires an authenticated complete snapshot upgrade');
        this.name = 'IndexedAcceptedOutputHistoryUpgradeRequired';
    }
}

/** Body-free accepted history, ordered by source revision. Omitted nominations consume the page
 * and advance its cursor; this never scans farther to fill a page after deletion/import filtering. */
export async function loadIndexedAcceptedOutputHistoryPage(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    options: { snapshot_revision: number; after_revision?: number; limit: number },
) {
    const root = IndexedConversationRootSchema.parse(rootInput);
    const { snapshot_revision: snapshotRevision, after_revision: afterRevision, limit } = options;
    if (
        !Number.isSafeInteger(snapshotRevision) ||
        snapshotRevision < 0 ||
        snapshotRevision > root.source.revision ||
        !Number.isSafeInteger(limit) ||
        limit < 1 ||
        limit > 100 ||
        (afterRevision !== undefined &&
            (!Number.isSafeInteger(afterRevision) || afterRevision < 0 || afterRevision > snapshotRevision))
    )
        throw new TypeError('Canonical accepted-output history cursor is invalid for the pinned snapshot');
    if (root.accepted_output_index_complete !== true) throw new IndexedAcceptedOutputHistoryUpgradeRequired();
    const store = indexedPresentationReadStore(storeInput);
    const range = await readPagedRecordRange(store, root.directories.accepted_output_order, {
        ...(afterRevision === undefined ? {} : { after: indexedOrderedKey(afterRevision) }),
        limit: limit + 1,
    });
    for (const entry of range.entries) {
        const revision = Number(entry.key);
        if (
            !Number.isSafeInteger(revision) ||
            revision < 1 ||
            revision > root.source.revision ||
            entry.key !== indexedOrderedKey(revision) ||
            entry.value.storage !== 'marker' ||
            entry.value.kind !== 'accepted_output'
        )
            throw new Error('Indexed accepted history has another ordered nomination');
    }
    const ordered = range.entries.filter((entry) => Number(entry.key) <= snapshotRevision);
    const page = ordered.slice(0, limit);
    const references: {
        source: ConversationRef;
        receipt: z.infer<typeof ConversationOutputReceiptSchema>;
    }[] = [];
    const omissions: { operation_id: string; revision: number; reason: 'logically_deleted' | 'imported' }[] = [];
    for (const entry of page) {
        const revision = Number(entry.key);
        if (
            entry.key !== indexedOrderedKey(revision) ||
            entry.value.storage !== 'marker' ||
            entry.value.kind !== 'accepted_output'
        )
            throw new Error('Indexed accepted history has another ordered nomination');
        const receipt = await presentationRecord(
            store,
            root,
            'operation_receipts',
            entry.value.id,
            OperationReceiptSchema,
        );
        if (
            receipt.id !== entry.value.id ||
            receipt.conversation_id !== root.source.conversation_id ||
            receipt.operation_kind !== undefined ||
            receipt.result_revision !== revision ||
            receipt.base_revision + 1 !== revision ||
            receipt.accepted_turn_ids?.length !== 1 ||
            receipt.accepted_generation_ids?.length !== 1
        )
            throw new Error('Indexed accepted history nomination differs from its original receipt');
        const turnId = receipt.accepted_turn_ids[0];
        const generationId = receipt.accepted_generation_ids[0];
        const generationAcceptance = await getPagedRecord(store, root.directories.generation_acceptances, generationId);
        const turnAcceptance = await getPagedRecord(store, root.directories.turn_acceptances, turnId);
        if (
            generationAcceptance?.storage !== 'marker' ||
            generationAcceptance.kind !== 'generation_acceptance' ||
            generationAcceptance.id !== receipt.id ||
            turnAcceptance?.storage !== 'marker' ||
            turnAcceptance.kind !== 'turn_acceptance' ||
            turnAcceptance.id !== receipt.id
        )
            throw new Error('Indexed accepted history has another turn/generation acceptance');
        const turnDescriptor = await getPagedRecord(store, root.directories.turns, turnId);
        if (turnDescriptor?.storage === 'marker' && turnDescriptor.kind === 'deleted_turn') {
            const tombstone = await presentationRecord(
                store,
                root,
                'deleted_turns',
                turnId,
                IndexedConversationDeletedTurnSchema,
            );
            if (
                !hasIndexedDeleteProfile(root) ||
                turnDescriptor.id !== turnId ||
                tombstone.deleted_turn.id !== turnId ||
                tombstone.deleted_turn.accepted_operation_id !== receipt.id
            )
                throw new Error('Indexed accepted history lacks its exact tombstone');
            const deletion = await presentationRecord(
                store,
                root,
                'operation_receipts',
                tombstone.deleted_turn.operation_id,
                OperationReceiptSchema,
            );
            const detail = deletion.conversation_delete;
            const ref = detail?.deleted_turns.find((item) => item.id === turnId);
            if (
                deletion.id !== tombstone.deleted_turn.operation_id ||
                deletion.conversation_id !== root.source.conversation_id ||
                deletion.operation_kind !== 'conversation_delete' ||
                deletion.base_revision !== tombstone.deleted_turn.source_revision ||
                deletion.result_revision !== deletion.base_revision + 1 ||
                deletion.result_revision > root.source.revision ||
                !ref ||
                !sameIndexedRecord(ref, {
                    id: turnId,
                    fingerprint: tombstone.deleted_turn.fingerprint,
                    block_ids: tombstone.deleted_turn.block_ids,
                    ...(tombstone.deleted_turn.call_ids === undefined
                        ? {}
                        : { call_ids: tombstone.deleted_turn.call_ids }),
                    accepted_operation_id: receipt.id,
                }) ||
                detail?.source_fingerprint !==
                    (await fingerprintJson({
                        domain: 'llumiverse.conversation.indexed-delete-source',
                        version: 1,
                        root: tombstone.predecessor_root,
                        turn_ids: detail?.deleted_turns.map((item) => item.id),
                    }))
            )
                throw new Error('Indexed accepted history tombstone differs from its original deletion');
            omissions.push({ operation_id: receipt.id, revision, reason: 'logically_deleted' });
            continue;
        }
        const turn = await presentationRecord(store, root, 'turns', turnId, IndexedConversationTurnHeaderSchema);
        const generation = await presentationRecord(store, root, 'generations', generationId, GenerationSchema);
        if (
            turn.source !== 'ordinary' ||
            turn.turn.id !== turnId ||
            turn.turn.kind !== 'agent' ||
            turn.turn.provenance.type !== 'generated' ||
            !('generation_id' in turn.turn) ||
            turn.turn.generation_id !== generationId ||
            generation.id !== generationId
        )
            throw new Error('Indexed accepted history has another canonical response tuple');
        if (generation.record_source === 'imported') {
            omissions.push({ operation_id: receipt.id, revision, reason: 'imported' });
            continue;
        }
        if (
            generation.source.conversation_id !== receipt.conversation_id ||
            generation.source.revision !== receipt.base_revision
        )
            throw new Error('Indexed accepted history generation has another original source');
        references.push({
            source: { conversation_id: receipt.conversation_id, revision },
            receipt: ConversationOutputReceiptSchema.parse({
                id: receipt.id,
                conversation_id: receipt.conversation_id,
                base_revision: receipt.base_revision,
                result_revision: revision,
                recorded_at: receipt.recorded_at,
                accepted_turn_ids: receipt.accepted_turn_ids,
                accepted_generation_ids: receipt.accepted_generation_ids,
                ...(receipt.accepted_asset_ids === undefined ? {} : { accepted_asset_ids: receipt.accepted_asset_ids }),
            }),
        });
    }
    const last = page.at(-1);
    return {
        references,
        omissions,
        ...(ordered.length > limit && last ? { next_after_revision: Number(last.key) } : {}),
    };
}

/** Exact accepted output, including generation usage and the original include-thoughts decision. */
async function loadIndexedAcceptedOutputRecords(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    receiptInput: unknown,
    retained = false,
) {
    const root = IndexedConversationRootSchema.parse(rootInput);
    const nominated = ConversationOutputReceiptSchema.parse(receiptInput);
    if (
        nominated.conversation_id !== root.source.conversation_id ||
        (retained
            ? nominated.result_revision > root.source.revision
            : nominated.result_revision !== root.source.revision)
    )
        throw new IndexedPresentationNominationConflict(
            'Indexed accepted output nomination differs from its historical root',
        );
    const boundedStore = indexedPresentationReadStore(store);
    const receipt = await presentationRecord(
        boundedStore,
        root,
        'operation_receipts',
        nominated.id,
        OperationReceiptSchema,
        true,
    );
    if (receipt.accepted_turn_ids?.length !== 1 || receipt.accepted_generation_ids?.length !== 1)
        throw new Error('Indexed accepted output has no exact generation and turn');
    const generationId = receipt.accepted_generation_ids[0];
    const accepted = await getPagedRecord(boundedStore, root.directories.generation_acceptances, generationId);
    if (accepted?.storage !== 'marker' || accepted.kind !== 'generation_acceptance' || accepted.id !== receipt.id)
        throw new Error('Indexed accepted output generation has no matching acceptance index');
    const generation = await presentationRecord(boundedStore, root, 'generations', generationId, GenerationSchema);
    if (generation.record_source !== 'executed') throw new Error('Indexed accepted output generation was not executed');
    // Retained projection must use the live turn index. Historical tombstone recovery would
    // resurrect output deliberately removed from the selected current root.
    const turnDescriptor = await getPagedRecord(boundedStore, root.directories.turns, receipt.accepted_turn_ids[0]);
    if (turnDescriptor?.storage === 'marker' && turnDescriptor.kind === 'deleted_turn')
        throw new IndexedPresentationNominationConflict('Indexed accepted output turn was logically deleted');
    if (retained) {
        const turnAcceptance = await getPagedRecord(
            boundedStore,
            root.directories.turn_acceptances,
            receipt.accepted_turn_ids[0],
        );
        if (
            turnAcceptance?.storage !== 'marker' ||
            turnAcceptance.kind !== 'turn_acceptance' ||
            turnAcceptance.id !== receipt.id
        )
            throw new Error('Indexed retained output turn has no matching acceptance index');
    }
    const projection = await loadIndexedProjectedTurn(boundedStore, root, receipt.accepted_turn_ids[0]);
    if (projection.completeness !== 'full_turn') throw new Error('Indexed accepted output turn is incomplete');
    const turn = GeneratedAgentTurnSchema.parse({ ...projection.header, blocks: projection.selected_blocks });
    const assetIds = receipt.accepted_asset_ids ?? [];
    if (assetIds.length > 512)
        throw new IndexedPresentationCapacityError('Indexed accepted output has too many selected assets');
    const assets: Record<string, Asset> = {};
    for (const id of assetIds) assets[id] = await presentationRecord(boundedStore, root, 'assets', id, AssetSchema);
    const fragment = createAcceptedOutputFragmentFromRecords({
        source: retained
            ? { conversation_id: receipt.conversation_id, revision: receipt.result_revision }
            : root.source,
        receipt,
        turn,
        generation,
        assets,
    });
    if ((await fingerprintJson(fragment.receipt)) !== (await fingerprintJson(nominated)))
        throw new IndexedPresentationNominationConflict(
            'Indexed accepted output receipt differs from its retained acceptance',
        );
    return {
        fragment,
        turn,
        root,
        boundedStore,
        include_reasoning: generation.model_options?.include_thoughts === true,
    };
}

/** Public presentation continues to omit private canonical call identity fields. */
export async function loadIndexedAcceptedOutputPresentation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    receipt: unknown,
) {
    const output = await loadIndexedAcceptedOutputRecords(store, root, receipt);
    return { fragment: output.fragment, include_reasoning: output.include_reasoning };
}

/** Bounded retained output anchored to the actual selected physical root. The fragment keeps
 * its original accepted source; deleted turns are rejected through the current live index.
 * Hosts separately authenticate that original revision and revalidate current custody. */
export async function loadIndexedRetainedAcceptedOutputPresentation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    receipt: unknown,
) {
    const output = await loadIndexedAcceptedOutputRecords(store, root, receipt, true);
    return { fragment: output.fragment, include_reasoning: output.include_reasoning };
}

/** Private execution continuation fingerprints the complete accepted canonical call, including
 * definition/native identifiers which the public output fragment deliberately omits. */
export async function loadIndexedAcceptedOutputContinuation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    receipt: unknown,
) {
    const output = await loadIndexedAcceptedOutputRecords(store, root, receipt);
    const pending: PendingApplicationToolCall[] = [];
    if (output.turn.status === 'completed' && output.fragment.generation.status === 'completed') {
        for (const block of output.turn.blocks) {
            if (block.type !== 'tool_call' || block.executor !== 'application') continue;
            if (block.arguments.type === 'invalid')
                throw new Error('Indexed accepted application call has invalid arguments');
            if (pending.length >= 256)
                throw new IndexedPresentationCapacityError(
                    'Indexed accepted continuation exceeds its pending-call profile',
                );
            const source = {
                conversation: output.fragment.source,
                turn_id: output.turn.id,
                block_id: block.id,
                call_id: block.call_id,
                call_fingerprint: await fingerprintJson(block),
            };
            const state = await indexedRecordById(
                output.boundedStore,
                output.root,
                'tool_call_states',
                block.call_id,
                IndexedCallStateSchema,
            );
            if (
                !state ||
                state.turn_id !== source.turn_id ||
                state.block_id !== source.block_id ||
                state.call_id !== source.call_id ||
                state.call_fingerprint !== source.call_fingerprint
            )
                throw new Error('Indexed accepted continuation differs from its durable canonical call state');
            pending.push({
                source,
                call: {
                    call_id: block.call_id,
                    tool_name: block.tool_name,
                    ...(block.definition_id === undefined ? {} : { definition_id: block.definition_id }),
                    executor: 'application',
                },
            });
        }
    }
    return { fragment: output.fragment, pending_tool_calls: pending };
}

/** One exact ordinary terminal program turn, bound to its accepted append receipt. */
export function loadIndexedTerminalProgramPresentation(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    receiptInput: unknown,
    turnId: string,
) {
    return loadIndexedTerminalProgramRecords(store, rootInput, receiptInput, turnId, false);
}

/** Current live indexes fence retained terminal content without fabricating a historical root.
 * Hosts must independently prove the original accepted revision descriptor and current custody. */
export function loadIndexedRetainedTerminalProgramPresentation(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    receiptInput: unknown,
    turnId: string,
) {
    return loadIndexedTerminalProgramRecords(store, rootInput, receiptInput, turnId, true);
}

async function loadIndexedTerminalProgramRecords(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    receiptInput: unknown,
    turnId: string,
    retained: boolean,
) {
    const root = IndexedConversationRootSchema.parse(rootInput);
    const nominated = OperationReceiptSchema.parse(receiptInput);
    if (
        nominated.conversation_id !== root.source.conversation_id ||
        (retained
            ? nominated.result_revision > root.source.revision
            : nominated.result_revision !== root.source.revision)
    )
        throw new IndexedPresentationNominationConflict(
            'Indexed terminal output nomination differs from its historical root',
        );
    const boundedStore = indexedPresentationReadStore(store);
    const receipt = await presentationRecord(
        boundedStore,
        root,
        'operation_receipts',
        nominated.id,
        OperationReceiptSchema,
        true,
    );
    if (
        (await fingerprintJson(receipt)) !== (await fingerprintJson(nominated)) ||
        receipt.accepted_turn_ids?.length !== 1 ||
        receipt.accepted_turn_ids[0] !== turnId
    )
        throw new IndexedPresentationNominationConflict(
            'Indexed terminal output differs from its nominated turn and receipt',
        );
    // A retained nomination must use the live current index. A tombstone or absent
    // turn is a nomination conflict, never permission to recover its historical body.
    const descriptor = await getPagedRecord(boundedStore, root.directories.turns, turnId);
    if (descriptor?.storage !== 'record')
        throw new IndexedPresentationNominationConflict(
            'Indexed terminal output is not the exact live program acceptance',
        );
    const projection = await loadIndexedProjectedTurn(boundedStore, root, turnId);
    const turn = ConversationTurnSchema.parse({ ...projection.header, blocks: projection.selected_blocks });
    if (
        projection.completeness !== 'full_turn' ||
        (receipt.accepted_generation_ids?.length ?? 0) !== 0 ||
        turn.kind !== 'program' ||
        turn.authority !== 'ordinary' ||
        turn.model_visibility !== 'exclude' ||
        turn.presentation !== 'transcript' ||
        turn.provenance.type !== 'inserted' ||
        turn.provenance.operation_id !== receipt.id ||
        turn.blocks.length !== 1 ||
        (turn.blocks[0]?.type !== 'text' && turn.blocks[0]?.type !== 'json')
    )
        throw new Error('Indexed terminal output differs from its accepted server-created program turn');
    return { receipt, turn };
}

/** Exact historical topic text. The 4096-turn/16MiB profile fails closed instead of silently
 * truncating or using the display transcript, which can omit external references. */
export async function renderIndexedConversationTopicText(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
): Promise<string> {
    return renderIndexedConversationHistoricalText(store, rootInput, false);
}

/** Exact search/enrichment projection, excluding private reasoning/replay and non-transcript programs. */
export async function renderIndexedConversationSearchText(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
): Promise<string> {
    return renderIndexedConversationHistoricalText(store, rootInput, true);
}

/** Exact last N visible user/agent messages. Cold history is never loaded merely to select a tail.
 * A long suffix of skipped turns or oversized selected bytes fails explicitly, rather than returning a partial tail. */
export async function renderIndexedConversationRecentMessages(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    limit: number,
): Promise<{ role: 'user' | 'assistant'; content: string }[]> {
    if (!Number.isSafeInteger(limit) || limit < 0 || limit > 100)
        throw new IndexedPresentationCapacityError('Indexed recent messages require a limit from 0 to 100');
    const root = IndexedConversationRootSchema.parse(rootInput);
    const count = root.live_turn_count;
    const tail = root.active_tail_turn_id;
    if (count === undefined || tail === undefined || (count === 0) !== (tail === null))
        throw new Error('Indexed recent source lacks its exact live-turn profile');
    if (limit === 0) return [];
    const bounded = indexedPresentationReadStore(store);
    const messages: { role: 'user' | 'assistant'; content: string }[] = [];
    let id: string | undefined = tail ?? undefined;
    let nextId: string | undefined;
    let lastOrdinal = Number.MAX_SAFE_INTEGER;
    let visited = 0;
    let textBytes = 0;
    while (id !== undefined && messages.length < limit) {
        if (visited >= count) throw new Error('Indexed recent live-turn chain exceeds its retained count');
        if (++visited > 4096)
            throw new IndexedPresentationCapacityError(
                'Indexed recent selection exceeds its bounded skipped-turn suffix',
            );
        const link = await presentationRecord(bounded, root, 'turn_links', id, IndexedConversationTurnLinkSchema);
        if (link.id !== id || link.next_turn_id !== nextId || link.ordinal >= lastOrdinal)
            throw new Error('Indexed recent live-turn order differs from its retained links');
        const header = await presentationRecord(bounded, root, 'turns', id, IndexedConversationTurnHeaderSchema);
        if (header.turn.id !== id) throw new Error('Indexed recent turn differs from its retained link');
        if (header.turn.kind === 'user' || header.turn.kind === 'agent') {
            const projection = await loadIndexedProjectedTurn(bounded, root, id);
            if (projection.completeness !== 'full_turn') throw new Error('Indexed recent turn is incomplete');
            const turn = ConversationTurnSchema.parse({ ...projection.header, blocks: projection.selected_blocks });
            const content = turn.blocks
                .filter((block) => block.type !== 'tool_call')
                .map(indexedSearchBlockText)
                .filter(Boolean)
                .join('\n');
            if (content.trim()) {
                textBytes += new TextEncoder().encode(content).byteLength;
                if (textBytes > INDEXED_TOPIC_MAX_TEXT_BYTES)
                    throw new IndexedPresentationCapacityError(
                        'Indexed recent text exceeds its 16MiB exact-render profile',
                    );
                messages.push({ role: turn.kind === 'agent' ? 'assistant' : 'user', content });
            }
        }
        nextId = id;
        lastOrdinal = link.ordinal;
        id = link.previous_turn_id;
    }
    if (id === undefined && visited !== count) throw new Error('Indexed recent source lacks retained live turns');
    return messages.reverse();
}

function indexedSearchBlockText(block: ConversationTurn['blocks'][number]): string {
    if (block.type === 'reasoning' || block.type === 'native_replay' || block.type === 'extension') return '';
    if (block.type === 'tool_result')
        return `[TOOL RESULT]: ${block.call_id} → ${block.content.map(indexedSearchBlockText).filter(Boolean).join(' ')}`;
    return renderContentBlockText(block);
}

async function renderIndexedConversationHistoricalText(
    store: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    search: boolean,
): Promise<string> {
    const root = IndexedConversationRootSchema.parse(rootInput);
    const count = root.live_turn_count;
    const tail = root.active_tail_turn_id;
    if (count === undefined || tail === undefined || (count === 0) !== (tail === null))
        throw new Error('Indexed topic source lacks its complete live-turn profile');
    if (count > INDEXED_TOPIC_MAX_TURNS)
        throw new IndexedPresentationCapacityError('Indexed topic source exceeds its bounded live-turn profile');
    const boundedStore = indexedPresentationReadStore(store);
    const reversed: string[] = [];
    let id: string | undefined = tail ?? undefined;
    let nextId: string | undefined;
    let lastOrdinal = Number.MAX_SAFE_INTEGER;
    while (id !== undefined) {
        if (reversed.length >= count) throw new Error('Indexed topic live-turn chain exceeds its retained count');
        const link = await presentationRecord(boundedStore, root, 'turn_links', id, IndexedConversationTurnLinkSchema);
        if (link.id !== id || link.next_turn_id !== nextId || link.ordinal >= lastOrdinal)
            throw new Error('Indexed topic live-turn order differs from its retained links');
        reversed.push(id);
        nextId = id;
        lastOrdinal = link.ordinal;
        id = link.previous_turn_id;
    }
    if (reversed.length !== count) throw new Error('Indexed topic source lacks retained live turns');
    const lines: string[] = [];
    let textBytes = 0;
    for (const turnId of reversed.reverse()) {
        const projection = await loadIndexedProjectedTurn(boundedStore, root, turnId);
        if (projection.completeness !== 'full_turn') throw new Error('Indexed topic source turn is incomplete');
        const turn = ConversationTurnSchema.parse({ ...projection.header, blocks: projection.selected_blocks });
        if (search && turn.kind === 'program' && turn.presentation !== 'transcript') continue;
        const content = turn.blocks
            .map(search ? indexedSearchBlockText : renderContentBlockText)
            .filter(Boolean)
            .join(' ');
        if (!content) continue;
        const role = turn.kind === 'agent' ? 'ASSISTANT' : turn.kind.toUpperCase();
        const line = `[${role}]: ${content}`;
        textBytes += new TextEncoder().encode(line).byteLength + (lines.length ? 2 : 0);
        if (textBytes > INDEXED_TOPIC_MAX_TEXT_BYTES)
            throw new IndexedPresentationCapacityError('Indexed topic text exceeds its 16MiB exact-render profile');
        lines.push(line);
    }
    return lines.join('\n\n');
}

/** Restart indexes are maintained by the same canonical snapshot/append publication, never
 * derived from a filtered history page or a host content mirror. */
function indexedRawRestartResponseTurn(turn: ConversationTurn): boolean {
    return (
        turn.kind === 'agent' &&
        (turn.provenance.type === 'generated' ||
            turn.provenance.type === 'imported' ||
            ('generation_id' in turn && turn.generation_id !== undefined))
    );
}

async function indexedSnapshotRestartWitness(document: ConversationDocument) {
    const receipts = Object.values(document.operation_receipts);
    const byTurn = new Map<string, OperationReceipt[]>();
    const byGeneration = new Map<string, OperationReceipt[]>();
    const byExecution = new Map<string, OperationReceipt[]>();
    const index = (map: Map<string, OperationReceipt[]>, ids: string[] | undefined, receipt: OperationReceipt) => {
        for (const id of ids ?? []) {
            const bindings = map.get(id) ?? [];
            bindings.push(receipt);
            map.set(id, bindings);
        }
    };
    for (const receipt of receipts) {
        index(byTurn, receipt.accepted_turn_ids, receipt);
        index(byGeneration, receipt.accepted_generation_ids, receipt);
        index(byExecution, receipt.accepted_execution_receipt_ids, receipt);
    }
    const generated = new Set(document.turns.filter(indexedRawRestartResponseTurn).map((turn) => turn.id));
    const tools = new Set(document.turns.filter((turn) => turn.kind === 'tool').map((turn) => turn.id));
    const turns = new Map(document.turns.map((turn) => [turn.id, turn]));
    const outputs = receipts
        .filter(
            (receipt) =>
                (receipt.accepted_generation_ids?.length ?? 0) > 0 ||
                receipt.accepted_turn_ids?.some((id) => generated.has(id)),
        )
        .sort((a, b) => b.result_revision - a.result_revision);
    const inputs = receipts
        .filter(
            (receipt) =>
                (receipt.accepted_execution_receipt_ids?.length ?? 0) > 0 ||
                receipt.accepted_turn_ids?.some((id) => tools.has(id)),
        )
        .sort((a, b) => b.result_revision - a.result_revision);
    // Semantic validation checks references that exist, but does not require every raw
    // generated/tool turn to have an append receipt. An old valid output must never hide
    // a newer unnominated imported/generated turn or unreceipted tool input on migration.
    for (const turn of document.turns) {
        if (!generated.has(turn.id) && !tools.has(turn.id)) continue;
        const bindings = byTurn.get(turn.id) ?? [];
        if (bindings.length !== 1) return {};
        const receipt = bindings[0];
        if (receipt.operation_kind !== undefined || receipt.result_revision !== receipt.base_revision + 1) return {};
        if (turn.kind === 'agent') {
            const generationId = 'generation_id' in turn ? turn.generation_id : undefined;
            const generation = generationId ? document.generations[generationId] : undefined;
            if (
                !generation ||
                generation.source.conversation_id !== document.id ||
                generation.source.revision !== receipt.base_revision ||
                receipt.accepted_turn_ids?.length !== 1 ||
                receipt.accepted_generation_ids?.length !== 1 ||
                receipt.accepted_generation_ids[0] !== generation.id
            )
                return {};
        } else if (turn.kind === 'tool') {
            const execution = turn.execution_id ? document.execution_receipts[turn.execution_id] : undefined;
            if (
                !execution ||
                execution.result_turn_id !== turn.id ||
                receipt.accepted_execution_receipt_ids?.filter((id) => id === execution.id).length !== 1
            )
                return {};
        }
    }
    // One-time complete migration checks. Steady-state append already authenticates these
    // exact canonical associations before publishing either new pointer in the same root.
    for (const generation of Object.values(document.generations)) {
        if (generation.record_source !== 'executed') continue;
        const bindings = byGeneration.get(generation.id) ?? [];
        if (
            bindings.length !== 1 ||
            bindings[0].base_revision !== generation.source.revision ||
            bindings[0].accepted_generation_ids?.length !== 1 ||
            bindings[0].accepted_turn_ids?.length !== 1
        )
            return {}; // Explicit older/incomplete profile; restart will reject, never guess.
    }
    for (const execution of Object.values(document.execution_receipts)) {
        if (execution.executor !== 'application') continue;
        const bindings = byExecution.get(execution.id) ?? [];
        if (bindings.length !== 1) return {};
    }
    if (
        (outputs[0] && outputs[1]?.result_revision === outputs[0].result_revision) ||
        (inputs[0] && inputs[1]?.result_revision === inputs[0].result_revision)
    )
        return {};
    for (const receipt of inputs.filter((input) => !outputs[0] || input.result_revision > outputs[0].result_revision)) {
        const ids = receipt.accepted_turn_ids ?? [];
        const executions = new Set(receipt.accepted_execution_receipt_ids ?? []);
        if (ids.length === 0 || executions.size !== ids.length || (receipt.accepted_generation_ids?.length ?? 0) !== 0)
            return {};
        const matched = new Set<string>();
        for (const id of ids) {
            const turn = turns.get(id);
            const execution =
                turn?.kind === 'tool' && turn.execution_id ? document.execution_receipts[turn.execution_id] : undefined;
            if (
                turn?.kind !== 'tool' ||
                !execution ||
                execution.executor !== 'application' ||
                !executions.has(execution.id) ||
                matched.has(execution.id) ||
                !execution.call_source
            )
                return {};
            const result = ConversationToolExecutionResultSchema.safeParse({
                source: execution.call_source,
                turn,
                execution_receipt: execution,
            });
            if (!result.success) return {};
            await validateToolExecutionResult(document, result.data);
            matched.add(execution.id);
        }
    }
    return {
        restart_index_profile: INDEXED_CONVERSATION_RESTART_PROFILE,
        ...(outputs[0]
            ? { restart_response: { operation_id: outputs[0].id, result_revision: outputs[0].result_revision } }
            : {}),
        ...(inputs[0]
            ? { restart_tool_input: { operation_id: inputs[0].id, result_revision: inputs[0].result_revision } }
            : {}),
    };
}
export class IndexedRestartSourceUnavailable extends Error {
    constructor(
        readonly reason: 'upgrade_required' | 'imported' | 'logically_deleted' | 'invalid_acceptance',
        message: string,
    ) {
        super(message);
        this.name = 'IndexedRestartSourceUnavailable';
    }
}
/** Bounded restart authority from complete raw nominations, exact immutable receipts and
 * the current live call index. Original output and tool source revisions are never rewritten. */
export async function loadIndexedRestartEvidence(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
) {
    const root = IndexedConversationRootSchema.parse(rootInput);
    if (root.restart_index_profile !== INDEXED_CONVERSATION_RESTART_PROFILE || root.tool_call_state_complete !== true)
        throw new IndexedRestartSourceUnavailable(
            'upgrade_required',
            'Indexed restart requires an authenticated complete restart profile',
        );
    const store = indexedPresentationReadStore(storeInput);
    const nomination = root.restart_response;
    if (!nomination) return { kind: 'no_output' as const, source: root.source };
    const receipt = await presentationRecord(
        store,
        root,
        'operation_receipts',
        nomination.operation_id,
        OperationReceiptSchema,
    );
    if (
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.result_revision !== receipt.base_revision + 1 ||
        receipt.result_revision !== nomination.result_revision ||
        receipt.result_revision > root.source.revision ||
        receipt.accepted_generation_ids?.length !== 1 ||
        receipt.accepted_turn_ids?.length !== 1
    )
        throw new IndexedRestartSourceUnavailable(
            'invalid_acceptance',
            'Newest raw restart response has no exact accepted generation and turn',
        );
    const generation = await presentationRecord(
        store,
        root,
        'generations',
        receipt.accepted_generation_ids[0],
        GenerationSchema,
    );
    if (generation.record_source !== 'executed')
        throw new IndexedRestartSourceUnavailable(
            'imported',
            'Newest raw restart acceptance is imported, not executed',
        );
    const live = await getPagedRecord(store, root.directories.turns, receipt.accepted_turn_ids[0]);
    if (live?.storage === 'marker' && live.kind === 'deleted_turn')
        throw new IndexedRestartSourceUnavailable(
            'logically_deleted',
            'Newest raw restart acceptance was logically deleted',
        );
    const output = await loadIndexedRetainedAcceptedOutputPresentation(
        store,
        root,
        ConversationOutputReceiptSchema.parse({
            id: receipt.id,
            conversation_id: receipt.conversation_id,
            base_revision: receipt.base_revision,
            result_revision: receipt.result_revision,
            recorded_at: receipt.recorded_at,
            accepted_turn_ids: receipt.accepted_turn_ids,
            accepted_generation_ids: receipt.accepted_generation_ids,
            ...(receipt.accepted_asset_ids === undefined ? {} : { accepted_asset_ids: receipt.accepted_asset_ids }),
        }),
    );
    let materialized_input: { operation_id: string; result_revision: number } | undefined;
    const input = root.restart_tool_input;
    if (input && input.result_revision > receipt.result_revision) {
        const accepted = await presentationRecord(
            store,
            root,
            'operation_receipts',
            input.operation_id,
            OperationReceiptSchema,
        );
        const turns = accepted.accepted_turn_ids ?? [];
        const executions = new Set(accepted.accepted_execution_receipt_ids ?? []);
        if (
            accepted.conversation_id !== root.source.conversation_id ||
            accepted.result_revision !== input.result_revision ||
            input.result_revision > root.source.revision ||
            turns.length === 0 ||
            turns.length > 256 ||
            executions.size !== turns.length ||
            (accepted.accepted_generation_ids?.length ?? 0) !== 0
        )
            throw new IndexedRestartSourceUnavailable(
                'invalid_acceptance',
                'Newest restart tool input has no exact accepted execution batch',
            );
        const matched = new Set<string>();
        for (const id of turns) {
            const descriptor = await getPagedRecord(store, root.directories.turns, id);
            if (descriptor?.storage === 'marker' && descriptor.kind === 'deleted_turn')
                throw new IndexedRestartSourceUnavailable(
                    'logically_deleted',
                    'Newest raw restart tool input was logically deleted',
                );
            const selected = await loadIndexedProjectedTurn(store, root, id);
            if (
                selected.header.kind !== 'tool' ||
                !selected.header.execution_id ||
                !executions.has(selected.header.execution_id) ||
                matched.has(selected.header.execution_id)
            )
                throw new IndexedRestartSourceUnavailable(
                    'invalid_acceptance',
                    'Restart tool input has an unbound result turn',
                );
            const execution = await presentationRecord(
                store,
                root,
                'execution_receipts',
                selected.header.execution_id,
                ExecutionReceiptSchema,
            );
            const binding = await getPagedRecord(store, root.directories.turn_acceptances, id);
            if (
                execution.result_turn_id !== id ||
                execution.id !== selected.header.execution_id ||
                !execution.call_source ||
                execution.executor !== 'application' ||
                binding?.storage !== 'marker' ||
                binding.kind !== 'turn_acceptance' ||
                binding.id !== accepted.id ||
                !(await loadIndexedToolCallTerminalResult(store, root, execution.call_source))
            )
                throw new IndexedRestartSourceUnavailable(
                    'invalid_acceptance',
                    'Restart tool input lacks exact call/result/receipt acceptance',
                );
            matched.add(execution.id);
        }
        materialized_input = { operation_id: accepted.id, result_revision: accepted.result_revision };
    }
    const open = await readPagedRecordRange(store, root.directories.open_tool_calls, { limit: 256 });
    if (open.has_more) throw new IndexedPresentationCapacityError('Indexed restart exceeds its 256 pending-call bound');
    const pending: PendingApplicationToolCall[] = [];
    for (const item of open.entries) {
        const original = await indexedRecordById(store, root, 'open_tool_calls', item.key, IndexedOpenToolCallSchema);
        const state = await indexedRecordById(store, root, 'tool_call_states', item.key, IndexedCallStateSchema);
        if (
            !original ||
            original.call_id !== item.key ||
            !state ||
            original.call_id !== state.call_id ||
            original.turn_id !== state.turn_id ||
            original.block_id !== state.block_id ||
            original.call_fingerprint !== state.call_fingerprint
        )
            throw new Error('Restart open-call nomination differs from its complete durable original');
        if (!state || state.result_block_id !== undefined || state.terminal_receipt_id !== undefined)
            throw new Error('Complete restart open-call index has another terminal state');
        const projected = await loadIndexedProjectedTurn(store, root, state.turn_id, [state.block_id]);
        const call = projected.selected_blocks[0];
        if (
            call?.type !== 'tool_call' ||
            call.call_id !== state.call_id ||
            (await fingerprintJson(call)) !== state.call_fingerprint
        )
            throw new Error('Restart pending call differs from its immutable original block');
        if (call.executor !== 'application') continue;
        if (
            call.arguments.type === 'invalid' ||
            projected.header.kind !== 'agent' ||
            projected.header.provenance.type !== 'generated' ||
            !('generation_id' in projected.header) ||
            !projected.header.generation_id
        )
            throw new IndexedRestartSourceUnavailable(
                'invalid_acceptance',
                'Restart pending application call has no valid generated source',
            );
        const binding = await getPagedRecord(
            store,
            root.directories.generation_acceptances,
            projected.header.generation_id,
        );
        if (binding?.storage !== 'marker' || binding.kind !== 'generation_acceptance')
            throw new Error('Restart pending call lost its original acceptance');
        const accepted = await presentationRecord(
            store,
            root,
            'operation_receipts',
            binding.id,
            OperationReceiptSchema,
        );
        const generation = await presentationRecord(
            store,
            root,
            'generations',
            projected.header.generation_id,
            GenerationSchema,
        );
        if (
            generation.record_source !== 'executed' ||
            accepted.conversation_id !== root.source.conversation_id ||
            generation.source.conversation_id !== root.source.conversation_id ||
            accepted.result_revision !== accepted.base_revision + 1 ||
            !accepted.accepted_turn_ids?.includes(state.turn_id) ||
            !accepted.accepted_generation_ids?.includes(generation.id) ||
            accepted.base_revision !== generation.source.revision ||
            accepted.result_revision > root.source.revision
        )
            throw new Error('Restart pending call has another original accepted request/generation tuple');
        pending.push({
            source: {
                conversation: { conversation_id: root.source.conversation_id, revision: accepted.result_revision },
                turn_id: state.turn_id,
                block_id: state.block_id,
                call_id: state.call_id,
                call_fingerprint: state.call_fingerprint,
            },
            call: {
                call_id: call.call_id,
                tool_name: call.tool_name,
                executor: 'application',
                ...(call.definition_id === undefined ? {} : { definition_id: call.definition_id }),
            },
        });
    }
    const definitions = await loadIndexedActiveToolDefinitions(store, root);
    return {
        kind: 'accepted_output' as const,
        source: root.source,
        accepted: output.fragment,
        pending,
        ...(materialized_input === undefined ? {} : { materialized_input }),
        active_tool_names: definitions.map((definition) => definition.name),
    };
}

export interface StagedIndexedCheckpointSummary {
    root: IndexedConversationRoot;
    locator: PagedRecordRef;
    receipt: OperationReceipt;
    compaction: CompactionRecord;
    applied: boolean;
}

/** One ordinary semantic checkpoint from a complete settled selected working set. The host
 * authenticates its scheduled activity, retained accepted output and genuine summary-fork result.
 * This operation never fabricates a processing job, readiness receipt, or materialized document.
 */
export async function stageIndexedCheckpointSummary(
    storeInput: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locatorInput: PagedRecordRef,
    commandInput: unknown,
): Promise<StagedIndexedCheckpointSummary> {
    const envelope = { root: rootInput, locator: locatorInput, command: commandInput };
    if (!preflightJsonInput(envelope, { max_bytes: 2 * 1024 * 1024 }).success)
        throw new TypeError('Indexed checkpoint intent exceeds its bounded JSON profile');
    const { root, locator, command } = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            command: IndexedCheckpointSummaryCommandSchema,
        })
        .parse(structuredClone(envelope));
    const store = boundedIndexedProcessingReader(storeInput);
    if (root.source.conversation_id !== command.source.conversation_id)
        throw new IndexedPresentationNominationConflict('Checkpoint source belongs to another conversation');
    const retained = await indexedRecordById(
        store,
        root,
        'operation_receipts',
        command.operation_id,
        OperationReceiptSchema,
    );
    if (retained) {
        if (retained.result_revision > root.source.revision)
            throw new Error('Checkpoint receipt is newer than its authentic retained root');
        const id = await deriveConversationId('compaction', root.source.conversation_id, command.operation_id);
        const header = await indexedRecordById(
            store,
            root,
            'compactions',
            id,
            IndexedConversationCompactionHeaderSchema,
        );
        if (!header) throw new Error('Checkpoint retry lost its exact retained compaction header');
        const replacements = [];
        const expectedTurnId = await deriveConversationId('turn', id, 'summary');
        for (const turnId of [expectedTurnId]) {
            const projection = await loadIndexedProjectedTurn(store, root, turnId);
            if (projection.completeness !== 'full_turn')
                throw new Error('Checkpoint retry lost its complete replacement');
            replacements.push(
                ConversationTurnSchema.parse({ ...projection.header, blocks: projection.selected_blocks }),
            );
        }
        const compaction = { ...header, replacement_turns: replacements };
        const selectedEntries =
            retained.context_change?.selected_block_ids === undefined
                ? undefined
                : await Promise.all(
                      retained.context_change.removed_entry_ids.map(async (entryId) => {
                          const entry = await indexedRecordById(
                              store,
                              root,
                              'context_entries',
                              entryId,
                              ContextEntrySchema,
                          );
                          if (!entry) throw new Error('Checkpoint retry lost its exact original selected entry');
                          return entry;
                      }),
                  );
        await assertSelectedCheckpointRetry(command, compaction, retained, selectedEntries);
        return { root, locator, receipt: retained, compaction, applied: false };
    }
    if (root.source.revision !== command.source.revision)
        throw new IndexedPresentationNominationConflict('Checkpoint predecessor changed before publication');
    const selected = await loadIndexedCheckpointSummarySelectedContext(store, root, locator);
    const frame = activeIndexedContextWorkingSet(selected);
    const request = await buildSelectedCheckpointRequest(frame, command);
    const mutation = await applyContextMutationWorkingSet(
        frame,
        {
            compactions: Object.fromEntries(
                Object.entries(selected.compaction_witnesses ?? []).map(([id, witness]) => [
                    id,
                    { id: witness.compaction.id },
                ]),
            ),
            tool_definitions: selected.tool_definitions,
            operation_receipts: selected.operation_witnesses ?? {},
        },
        request,
        await fingerprintJson(request),
    );
    const compaction = mutation.compaction;
    if (!compaction) throw new Error('Semantic checkpoint produced no compaction');
    const directories = await stageIndexedContextMutationRecords(store, root, selected.context, mutation);
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    const { coverage: _coverage, ...processing } = header;
    const processingHeader = await stageRecord(
        store,
        'processing_header',
        root.source.conversation_id,
        IndexedConversationProcessingHeaderSchema.parse(processing),
    );
    const { entries: _entries, ...contextFields } = mutation.context;
    const contextHeader = await stageRecord(
        store,
        'context_header',
        root.source.conversation_id,
        IndexedConversationContextHeaderSchema.parse({
            ...contextFields,
            active_entry_count: mutation.context.entries.length,
            active_entry_bytes: canonicalJsonContentBytes(mutation.context.entries).byteLength,
            context_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(mutation.context))).content_hash,
        }),
    );
    const nextRoot = IndexedConversationRootSchema.parse({
        ...root,
        directories,
        source: { ...root.source, revision: mutation.receipt.result_revision },
        updated_at: command.recorded_at,
        context_header: { content_hash: contextHeader.content_hash, size_bytes: contextHeader.size_bytes },
        processing_header: { content_hash: processingHeader.content_hash, size_bytes: processingHeader.size_bytes },
    });
    const rootRecord = await stageRecord(store, 'root', root.source.conversation_id, nextRoot);
    if (rootRecord.size_bytes > INDEXED_CONVERSATION_ROOT_MAX_BYTES)
        throw new RangeError('Indexed checkpoint root exceeds its manifest bound');
    const nextLocator = { content_hash: rootRecord.content_hash, size_bytes: rootRecord.size_bytes };
    await loadIndexedCheckpointSummarySelectedContext(store, nextRoot, nextLocator);
    return { root: nextRoot, locator: nextLocator, receipt: mutation.receipt, compaction, applied: true };
}

/** Shared immutable compaction acceptance proof; no preparation/readiness authority. */
export function assertIndexedAcceptedCompaction(
    root: IndexedConversationRoot,
    compactionId: string,
    compaction: z.infer<typeof IndexedConversationCompactionHeaderSchema>,
    acceptance: z.infer<typeof OperationReceiptSchema>,
): void {
    if (
        compaction.id !== compactionId ||
        acceptance.id !== compaction.operation_id ||
        acceptance.conversation_id !== root.source.conversation_id ||
        acceptance.operation_kind !== 'context_change' ||
        acceptance.context_change?.kind !== 'replace_with_compaction' ||
        acceptance.context_change.source_fingerprint !== compaction.source.source_fingerprint ||
        acceptance.result_revision !== acceptance.base_revision + 1 ||
        compaction.created_at !== acceptance.recorded_at ||
        acceptance.result_revision > root.source.revision ||
        compaction.metadata?.applied_revision !== acceptance.result_revision ||
        compaction.metadata?.payload_fingerprint !== acceptance.payload_fingerprint
    )
        throw new Error('Indexed replacement lacks its exact accepted compaction operation');
}
