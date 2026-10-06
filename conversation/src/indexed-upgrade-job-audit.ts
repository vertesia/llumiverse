import { fingerprintJson } from './identity.js';
import {
    auditIndexedProcessingJobEvidence,
    type IndexedConversationRecordStore,
    loadRecord,
    ownedIndexedProcessingJob,
} from './indexed-conversation.js';
import { indexedCompletedJobEntrySelection } from './indexed-processing-working-set.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import { IndexedConversationProcessingHeaderSchema, type IndexedConversationRoot } from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

/** A second pass uses reconstructed cohorts and the unchanged original root/header; no fake profile. */
export async function advanceIndexedUpgradeJobAudit(
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
        return { ...next, audit_family: 1 };
    }
    let tuple: unknown;
    try {
        tuple = JSON.parse(entry.key);
    } catch (cause: unknown) {
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade processing audit key is invalid', cause);
    }
    if (!Array.isArray(tuple) || tuple.length !== 2 || typeof tuple[0] !== 'string' || typeof tuple[1] !== 'string')
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade processing audit key is not an exact family/id',
        );
    const [family, id] = tuple;
    if (family !== 'jobs') return { ...progress, cursor: entry.key };
    const readRoot = { ...root, directories: progress.directories };
    const job = await ownedIndexedProcessingJob(store, readRoot, id);
    const header = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    const evidence = await auditIndexedProcessingJobEvidence(store, readRoot, job, header);
    if ((evidence.attempt || evidence.output || evidence.completion) && !evidence.resolution)
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade processing phase lacks its original resolution',
        );
    for (const [phase, value] of [
        ['attempt', evidence.attempt],
        ['output', evidence.output],
    ] as const) {
        if (value === undefined) continue;
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, `processing:${phase}:${job.id}`),
            OperationReceiptSchema,
        );
        const identity = await fingerprintJson(value);
        if (
            receipt.operation_kind !== 'processing' ||
            receipt.processing_operation?.phase !== phase ||
            receipt.processing_operation.job_id !== job.id ||
            receipt.processing_operation.policy_revision !== job.policy_revision ||
            receipt.conversation_id !== root.source.conversation_id ||
            receipt.result_revision > root.source.revision ||
            receipt.result_revision !== receipt.base_revision + 1 ||
            receipt.payload_fingerprint !== identity ||
            (receipt.processing_operation.result_fingerprint !== undefined &&
                receipt.processing_operation.result_fingerprint !== identity)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade processing phase lacks exact original accepted receipt',
            );
    }
    if (evidence.completion) {
        if (!evidence.output || !evidence.resolution || !evidence.resolution_receipt || evidence.supersession)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade completed job lacks exact original phase chain',
            );
        if (evidence.completion.status === 'blocked')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade blocked completion has no supported exact native phase proof',
            );
        const receiptId = evidence.completion.context_change_operation_id ?? `processing:complete:${job.id}`;
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, receiptId),
            OperationReceiptSchema,
        );
        await indexedCompletedJobEntrySelection({
            job,
            resolution: evidence.resolution,
            resolution_receipt: evidence.resolution_receipt,
            output: evidence.output,
            completion: evidence.completion,
            receipt,
            ...(evidence.attempt === undefined ? {} : { attempt: evidence.attempt }),
        });
    }
    if (evidence.supersession)
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade superseded legacy job requires its exact policy-transition phase audit',
        );
    const directories = { ...progress.directories };
    const counts = { ...progress.counts, jobs: progress.counts.jobs + 1 };
    if (!evidence.completion) {
        directories.processing_pending = await putPagedRecord(store, directories.processing_pending, job.id, {
            storage: 'marker',
            kind: 'processing_pending',
            id: job.id,
        });
        counts.unresolved_jobs += 1;
        if (job.required) counts.required_unresolved_jobs += 1;
    }
    if (job.required) {
        directories.processing_required = await putPagedRecord(store, directories.processing_required, job.id, {
            storage: 'marker',
            kind: 'processing_required',
            id: job.id,
        });
        counts.required_jobs += 1;
    }
    return { ...progress, directories, counts, cursor: entry.key };
}
