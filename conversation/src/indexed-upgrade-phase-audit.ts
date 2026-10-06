import { fingerprintJson } from './identity.js';
import {
    auditIndexedProcessingJobEvidence,
    type IndexedConversationRecordStore,
    indexedCoverageIdentity,
    indexedReadinessIdentity,
    loadRecord,
    ownedIndexedProcessingJob,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import {
    IndexedConversationProcessingHeaderSchema,
    type IndexedConversationRoot,
    IndexedProcessingQueueCommandSchema,
    IndexedProcessingReadinessCoverageSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';
import {
    ProcessingAttemptReceiptSchema,
    ProcessingCompletionReceiptSchema,
    ProcessingOutputReceiptSchema,
    ProcessingReadinessCoverageSchema,
    ProcessingResolvedInputSchema,
    ProcessingSupersessionReceiptSchema,
} from './schemas/processing.js';

/** Every retained phase must point back to an audited job; orphan records cannot hide in history. */
export async function advanceIndexedUpgradePhaseAudit(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.processing_records, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, ...next } = progress;
        return { ...next, audit_family: 10 };
    }
    let tuple: unknown;
    try {
        tuple = JSON.parse(entry.key);
    } catch (cause: unknown) {
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade phase key is invalid', cause);
    }
    if (!Array.isArray(tuple) || tuple.length !== 2 || typeof tuple[0] !== 'string' || typeof tuple[1] !== 'string')
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade phase identity is invalid');
    const [family, id] = tuple;
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    const readRoot = { ...root, directories: progress.directories };
    const check = async (
        jobId: string,
        actual: unknown,
        key: 'resolution' | 'attempt' | 'output' | 'completion' | 'supersession',
    ) => {
        if (jobId !== id)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade phase key differs from original job');
        const job = await ownedIndexedProcessingJob(store, readRoot, jobId);
        const evidence = await auditIndexedProcessingJobEvidence(store, readRoot, job, header);
        if (evidence[key] === undefined || (await fingerprintJson(evidence[key])) !== (await fingerprintJson(actual)))
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade phase has no exact original audited job linkage',
            );
    };
    if (family === 'resolved_inputs') {
        const value = await loadRecord(store, entry.value, ProcessingResolvedInputSchema);
        await check(value.job_id, value, 'resolution');
    } else if (family === 'attempts') {
        const value = await loadRecord(store, entry.value, ProcessingAttemptReceiptSchema);
        await check(value.job_id, value, 'attempt');
    } else if (family === 'outputs') {
        const value = await loadRecord(store, entry.value, ProcessingOutputReceiptSchema);
        await check(value.job_id, value, 'output');
    } else if (family === 'completions') {
        const value = await loadRecord(store, entry.value, ProcessingCompletionReceiptSchema);
        await check(value.job_id, value, 'completion');
    } else if (family === 'supersessions') {
        const value = await loadRecord(store, entry.value, ProcessingSupersessionReceiptSchema);
        await check(value.job_id, value, 'supersession');
    } else if (family === 'selected_queue_commands') {
        const command = await loadRecord(store, entry.value, IndexedProcessingQueueCommandSchema);
        const relation = await getPagedRecord(store, progress.directories.processing_by_operation, id);
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, id),
            OperationReceiptSchema,
        );
        if (
            command.operation_id !== id ||
            relation === undefined ||
            receipt.operation_kind !== 'processing' ||
            receipt.processing_operation?.phase !== 'queue' ||
            receipt.base_revision !== command.expected_revision
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade retained queue command lacks exact accepted job cohort',
            );
        // Each job's shared evidence audit already reconstructs the exact command/receipt fingerprint.
    } else if (family === 'coverage_receipts' || family === 'indexed_coverage') {
        const coverage =
            family === 'coverage_receipts'
                ? await loadRecord(store, entry.value, ProcessingReadinessCoverageSchema)
                : await loadRecord(store, entry.value, IndexedProcessingReadinessCoverageSchema);
        if (coverage.evaluated_at_revision > root.source.revision || coverage.policy_revision > header.policy_revision)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade coverage has an impossible original source/policy',
            );
        if (family === 'indexed_coverage') {
            const native = IndexedProcessingReadinessCoverageSchema.parse(coverage);
            const receipt = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.operation_receipts, id),
                OperationReceiptSchema,
            );
            if (
                receipt.operation_kind !== 'processing' ||
                receipt.processing_operation?.phase !== 'coverage' ||
                receipt.result_revision !== native.evaluated_at_revision ||
                receipt.processing_operation.result_fingerprint !== (await fingerprintJson(native))
            )
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade native coverage lacks exact original accepted receipt',
                );
        }
        const identity =
            family === 'coverage_receipts'
                ? await indexedCoverageIdentity(ProcessingReadinessCoverageSchema.parse(coverage))
                : await indexedReadinessIdentity(IndexedProcessingReadinessCoverageSchema.parse(coverage));
        const directories = { ...progress.directories };
        const previous = await getPagedRecord(store, directories.processing_coverage, identity);
        let replace = previous === undefined;
        if (previous !== undefined) {
            const older =
                family === 'coverage_receipts'
                    ? await loadRecord(
                          store,
                          await getPagedRecord(
                              store,
                              root.directories.processing_records,
                              JSON.stringify([family, previous.id]),
                          ),
                          ProcessingReadinessCoverageSchema,
                      )
                    : await loadRecord(
                          store,
                          await getPagedRecord(
                              store,
                              root.directories.processing_records,
                              JSON.stringify([family, previous.id]),
                          ),
                          IndexedProcessingReadinessCoverageSchema,
                      );
            if (older.evaluated_at_revision === coverage.evaluated_at_revision && previous.id !== id)
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade coverage has ambiguous same-revision identities',
                );
            replace = older.evaluated_at_revision < coverage.evaluated_at_revision;
        }
        if (replace)
            directories.processing_coverage = await putPagedRecord(
                store,
                directories.processing_coverage,
                identity,
                {
                    storage: 'marker',
                    kind: family === 'coverage_receipts' ? 'processing_coverage' : 'indexed_processing_coverage',
                    id,
                },
                previous === undefined ? 'insert' : 'replace',
            );
        return { ...progress, directories, cursor: entry.key };
    } else if (family !== 'jobs')
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade phase family is unsupported');
    return { ...progress, cursor: entry.key };
}
