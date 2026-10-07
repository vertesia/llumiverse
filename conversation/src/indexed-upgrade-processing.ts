import { fingerprintJson } from './identity.js';
import {
    type IndexedConversationRecordStore,
    IndexedProcessingOperationJobsSchema,
    loadRecord,
    stageIndexedAcceptedProcessingPolicy,
    stageRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import type { IndexedConversationRoot } from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';
import { ProcessingJobSchema } from './schemas/processing.js';
import { ProcessingPolicyCommandSchema } from './schemas/processing-policy.js';

/** Build operation cohorts first. A later pass audits every exact phase before counting obligations. */
export async function advanceIndexedUpgradeProcessing(
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
        return { ...next, phase: 'audit', audit_family: 0 };
    }
    let tuple: unknown;
    try {
        tuple = JSON.parse(entry.key);
    } catch (cause: unknown) {
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade processing key is not a tuple', cause);
    }
    if (
        !Array.isArray(tuple) ||
        tuple.length !== 2 ||
        typeof tuple[0] !== 'string' ||
        typeof tuple[1] !== 'string' ||
        entry.value.storage !== 'record' ||
        entry.value.kind !== 'processing_records' ||
        entry.value.id !== tuple[1]
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade processing family/key differs from its retained record',
        );
    const [family, id] = tuple;
    if (
        ![
            'jobs',
            'resolved_inputs',
            'attempts',
            'outputs',
            'completions',
            'supersessions',
            'coverage_receipts',
            'indexed_coverage',
            'selected_queue_commands',
            'materialized_queue_commands',
            'selected_policy_commands',
            'policy_epochs',
            'tool_result_validations',
            'tool_result_sources',
            'tool_result_validation_by_terminal',
        ].includes(family)
    )
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade processing has an unsupported family');
    if (family === 'selected_policy_commands') {
        const command = await loadRecord(store, entry.value, ProcessingPolicyCommandSchema);
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, id),
            OperationReceiptSchema,
        );
        const directories = await stageIndexedAcceptedProcessingPolicy(
            store,
            root.source,
            progress.directories,
            command,
            receipt,
        );
        return { ...progress, directories, cursor: entry.key };
    }
    if (family !== 'jobs') return { ...progress, cursor: entry.key };
    const job = await loadRecord(store, entry.value, ProcessingJobSchema);
    const receipt = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.operation_receipts, job.source_operation_id),
        OperationReceiptSchema,
    );
    if (
        job.id !== id ||
        job.enqueue_revision !== receipt.result_revision ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.result_revision > root.source.revision ||
        job.configuration_fingerprint !== (await fingerprintJson(job.configuration)) ||
        job.selection_fingerprint !== (await fingerprintJson(job.selection))
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade job differs from original enqueue/configuration/selection acceptance',
        );
    const directories = { ...progress.directories };
    const existing = await getPagedRecord(store, directories.processing_by_operation, receipt.id);
    const previous =
        existing === undefined ? undefined : await loadRecord(store, existing, IndexedProcessingOperationJobsSchema);
    const fingerprint = await fingerprintJson(receipt);
    if (
        previous &&
        (previous.operation_id !== receipt.id ||
            previous.receipt_fingerprint !== fingerprint ||
            previous.job_ids.includes(job.id))
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade processing cohort changed or repeated an original job',
        );
    const jobs = [job];
    for (const previousId of previous?.job_ids ?? []) {
        const retained = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.processing_records, JSON.stringify(['jobs', previousId])),
            ProcessingJobSchema,
        );
        if (retained.id !== previousId || retained.source_operation_id !== receipt.id)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade cohort contains a foreign original job');
        jobs.push(retained);
    }
    if (jobs.length > 16 || new Set(jobs.map((item) => item.stage_index)).size !== jobs.length)
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade cohort exceeds ordered stages or repeats a stage',
        );
    jobs.sort((a, b) => a.stage_index - b.stage_index);
    const relation = IndexedProcessingOperationJobsSchema.parse({
        version: 1,
        operation_id: receipt.id,
        receipt_fingerprint: fingerprint,
        job_ids: jobs.map((item) => item.id),
    });
    directories.processing_by_operation = await putPagedRecord(
        store,
        directories.processing_by_operation,
        receipt.id,
        await stageRecord(store, 'processing_by_operation', receipt.id, relation),
        existing === undefined ? 'insert' : 'replace',
    );
    return { ...progress, directories, cursor: entry.key };
}
