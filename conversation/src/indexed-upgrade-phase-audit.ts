import { fingerprintJson } from './identity.js';
import {
    auditIndexedProcessingJobEvidence,
    auditIndexedProcessingPolicyEpoch,
    auditIndexedToolResultTerminalValidation,
    auditIndexedToolResultValidation,
    type IndexedConversationRecordStore,
    IndexedMaterializedProcessingQueueSchema,
    IndexedProcessingOperationJobsSchema,
    IndexedProcessingPolicyEpochSchema,
    IndexedToolResultOriginalSourceSchema,
    IndexedToolResultTerminalValidationSchema,
    indexedCoverageIdentity,
    indexedOrderedKey,
    indexedReadinessIdentity,
    loadRecord,
    ownedIndexedProcessingJob,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { ExecutionReceiptSchema, OperationReceiptSchema } from './schemas/execution.js';
import {
    IndexedConversationProcessingHeaderSchema,
    type IndexedConversationRoot,
    IndexedProcessingPolicyCommandSchema,
    IndexedProcessingQueueCommandSchema,
    IndexedProcessingReadinessCoverageSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';
import {
    ProcessingAttemptReceiptSchema,
    ProcessingCompletionReceiptSchema,
    ProcessingJobSchema,
    ProcessingOutputReceiptSchema,
    ProcessingReadinessCoverageSchema,
    ProcessingResolvedInputSchema,
    ProcessingSupersessionReceiptSchema,
} from './schemas/processing.js';

import { isToolResultTextProcessor } from './tool-result-text-externalization.js';

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
    const auditOrDeferValidation = async (jobId = id): Promise<IndexedConversationUpgradeProgress> => {
        const validation = await getPagedRecord(
            store,
            readRoot.directories.processing_records,
            JSON.stringify(['tool_result_validations', jobId]),
        );
        if (validation !== undefined) await auditIndexedToolResultValidation(store, readRoot, jobId);
        const completion = await loadRecord(
            store,
            await getPagedRecord(
                store,
                readRoot.directories.processing_records,
                JSON.stringify(['completions', jobId]),
            ),
            ProcessingCompletionReceiptSchema,
        );
        if (completion.job_id !== jobId || completion.status !== 'applied')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade terminal obligation lacks applied completion',
            );
        const key = `${indexedOrderedKey(completion.result_revision)}:${jobId}`;
        const scratch = { ...progress.scratch };
        const existing = await getPagedRecord(store, scratch.missing_tool_result_validations, key);
        if (
            existing !== undefined &&
            (existing.storage !== 'marker' ||
                existing.kind !== 'upgrade_missing_tool_result_validation' ||
                existing.id !== jobId)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade terminal obligation changed its accepted job',
            );
        if (existing === undefined)
            scratch.missing_tool_result_validations = await putPagedRecord(
                store,
                scratch.missing_tool_result_validations,
                key,
                { storage: 'marker', kind: 'upgrade_missing_tool_result_validation', id: jobId },
            );
        return { ...progress, scratch, cursor: entry.key };
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
    } else if (family === 'policy_epochs') {
        const epoch = await loadRecord(store, entry.value, IndexedProcessingPolicyEpochSchema);
        if (String(epoch.policy_revision) !== id || epoch.policy_revision > header.policy_revision)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade policy epoch differs from its original key',
            );
        await auditIndexedProcessingPolicyEpoch(store, readRoot, epoch.policy_revision);
    } else if (family === 'selected_policy_commands') {
        const command = await loadRecord(store, entry.value, IndexedProcessingPolicyCommandSchema);
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, id),
            OperationReceiptSchema,
        );
        if (
            command.operation_id !== id ||
            receipt.id !== id ||
            receipt.conversation_id !== root.source.conversation_id ||
            receipt.operation_kind !== 'processing' ||
            receipt.processing_operation?.phase !== 'policy' ||
            receipt.processing_operation.policy_revision >= header.policy_revision ||
            receipt.base_revision !== command.expected_revision ||
            receipt.result_revision !== command.expected_revision + 1 ||
            receipt.result_revision > root.source.revision ||
            receipt.recorded_at !== command.recorded_at ||
            receipt.payload_fingerprint !== (await fingerprintJson(command)) ||
            (await fingerprintJson(receipt.processing_operation.superseded_job_ids ?? [])) !==
                (await fingerprintJson(command.supersede_job_ids ?? []))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade retained policy command lacks its exact acceptance',
            );
    } else if (family === 'materialized_queue_commands') {
        const command = await loadRecord(store, entry.value, IndexedMaterializedProcessingQueueSchema);
        const relation = await loadRecord(
            store,
            await getPagedRecord(store, progress.directories.processing_by_operation, id),
            IndexedProcessingOperationJobsSchema,
        );
        if (command.command.operation_id !== id || relation.job_ids.length !== 1)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed materialized queue lacks its exact accepted job cohort',
            );
        await ownedIndexedProcessingJob(store, readRoot, relation.job_ids[0]);
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
    } else if (family === 'tool_result_validation_by_terminal') {
        const marker = await loadRecord(store, entry.value, IndexedToolResultTerminalValidationSchema);
        const validation = await getPagedRecord(
            store,
            readRoot.directories.processing_records,
            JSON.stringify(['tool_result_validations', marker.job_id]),
        );
        if (validation !== undefined) await auditIndexedToolResultTerminalValidation(store, readRoot, id);
        else {
            const terminal = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.execution_receipts, id),
                ExecutionReceiptSchema,
            );
            const job = await ownedIndexedProcessingJob(store, readRoot, marker.job_id);
            const evidence = await auditIndexedProcessingJobEvidence(store, readRoot, job, header);
            if (
                marker.terminal_execution_id !== id ||
                terminal.id !== id ||
                terminal.result_turn_id !== marker.result_turn_id ||
                !isToolResultTextProcessor(job) ||
                evidence.completion?.status !== 'applied' ||
                (!evidence.resolution?.source_turn_ids.includes(marker.result_turn_id) && job.processor_version !== '3')
            )
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade terminal relation lacks its applied original job',
                );
            // Rebuilding that exact validation also audits every retained terminal relation;
            // its content fingerprint cannot be replaced by the recovered witness.
            return auditOrDeferValidation(marker.job_id);
        }
    } else if (family === 'tool_result_sources') {
        const original = await loadRecord(store, entry.value, IndexedToolResultOriginalSourceSchema);
        const job = await ownedIndexedProcessingJob(store, readRoot, id);
        const evidence = await auditIndexedProcessingJobEvidence(store, readRoot, job, header);
        if (
            original.job_id !== id ||
            original.job_fingerprint !== (await fingerprintJson(job)) ||
            !evidence.resolution ||
            original.resolved_input_fingerprint !== (await fingerprintJson(evidence.resolution))
        )
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade changed its original source binding');
        if (evidence.completion?.status !== 'applied')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade original source lacks applied processing',
            );
        return auditOrDeferValidation();
    } else if (family === 'tool_result_validations') {
        await auditIndexedToolResultValidation(store, readRoot, id);
    } else if (family === 'jobs') {
        const job = await loadRecord(store, entry.value, ProcessingJobSchema);
        if (isToolResultTextProcessor(job)) {
            const evidence = await auditIndexedProcessingJobEvidence(store, readRoot, job, header);
            if (evidence.completion?.status === 'applied') return auditOrDeferValidation();
        }
    } else throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade phase family is unsupported');
    return { ...progress, cursor: entry.key };
}
