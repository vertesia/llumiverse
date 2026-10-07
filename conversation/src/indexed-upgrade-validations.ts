import {
    auditIndexedToolResultValidation,
    type IndexedConversationRecordStore,
    indexedOrderedKey,
    loadRecord,
    recoverIndexedToolResultTerminalRelations,
    recoverIndexedToolResultValidations,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import type { IndexedConversationRoot } from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';
import { ProcessingCompletionReceiptSchema, ProcessingOutputReceiptSchema } from './schemas/processing.js';

/** One accepted completion per retained step, in chronological order. Witness regeneration
 * precedes relation auditing, so two compactions of the same original never force two closures
 * into one budget or depend on lexical job-ID order. Neither pass materializes lifetime history.
 */
export async function advanceIndexedUpgradeValidation(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, progress.scratch.missing_tool_result_validations, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const obligation = page.entries[0];
    const terminalPhase = progress.phase === 'tool_result_terminal_validations';
    if (!obligation) {
        const { cursor: _cursor, ...next } = progress;
        if (!terminalPhase) return { ...next, phase: 'tool_result_terminal_validations' };
        const { missing_tool_result_validations: _missing, ...scratch } = progress.scratch;
        return { ...next, scratch, phase: 'complete' };
    }
    if (obligation.value.storage !== 'marker' || obligation.value.kind !== 'upgrade_missing_tool_result_validation')
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade validation obligation is invalid');
    const jobId = obligation.value.id;
    const completion = await loadRecord(
        store,
        await getPagedRecord(store, progress.directories.processing_records, JSON.stringify(['completions', jobId])),
        ProcessingCompletionReceiptSchema,
    );
    if (
        completion.job_id !== jobId ||
        completion.status !== 'applied' ||
        obligation.key !== `${indexedOrderedKey(completion.result_revision)}:${jobId}`
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade validation obligation changed its accepted completion',
        );
    let directories = progress.directories;
    if (terminalPhase)
        directories = await recoverIndexedToolResultTerminalRelations(store, { ...root, directories }, jobId);
    else {
        if (
            (await getPagedRecord(
                store,
                directories.processing_records,
                JSON.stringify(['tool_result_validations', jobId]),
            )) === undefined
        ) {
            const output = await loadRecord(
                store,
                await getPagedRecord(store, directories.processing_records, JSON.stringify(['outputs', jobId])),
                ProcessingOutputReceiptSchema,
            );
            if (
                output.job_id !== jobId ||
                output.kind !== 'proposal' ||
                output.proposal.kind !== 'replace_with_compaction'
            )
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade missing validation lacks its proposal',
                );
            directories = await recoverIndexedToolResultValidations(
                store,
                { ...root, directories },
                undefined,
                [output.proposal.compaction_id],
                { defer_terminal_relations: true },
            );
        }
        await auditIndexedToolResultValidation(store, { ...root, directories }, jobId);
    }
    return { ...progress, directories, cursor: obligation.key };
}
