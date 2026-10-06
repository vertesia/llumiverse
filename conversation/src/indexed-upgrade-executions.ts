import { fingerprintJson } from './identity.js';
import {
    IndexedCallStateSchema,
    type IndexedConversationRecordStore,
    loadIndexedAcceptedTurn,
    loadRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { ExecutionReceiptSchema, OperationReceiptSchema } from './schemas/execution.js';
import type { IndexedConversationRoot } from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';
import { assertToolResultReceiptFingerprint } from './tool-result-integrity.js';

/** Audit one original execution, including exact accepted original call/result after deletion. */
export async function advanceIndexedUpgradeExecution(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.execution_receipts, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, ...next } = progress;
        return { ...next, phase: 'turns' };
    }
    const receipt = await loadRecord(store, entry.value, ExecutionReceiptSchema);
    const accepted = await getPagedRecord(store, progress.scratch.execution_acceptances, receipt.id);
    if (
        entry.key !== receipt.id ||
        entry.value.kind !== 'execution_receipts' ||
        accepted?.storage !== 'marker' ||
        accepted.kind !== 'upgrade_execution_acceptance'
    )
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade execution lacks its exact accepted input');
    const operation = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.operation_receipts, accepted.id),
        OperationReceiptSchema,
    );
    if (
        !operation.accepted_execution_receipt_ids?.includes(receipt.id) ||
        operation.operation_kind !== undefined ||
        operation.conversation_id !== root.source.conversation_id ||
        operation.result_revision > root.source.revision ||
        !receipt.result_turn_id ||
        !operation.accepted_turn_ids?.includes(receipt.result_turn_id)
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade execution differs from its original append receipt',
        );
    const state = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.tool_call_states, receipt.call_id),
        IndexedCallStateSchema,
    );
    const callAcceptance = await getPagedRecord(store, progress.directories.turn_acceptances, state.turn_id);
    if (callAcceptance?.storage !== 'marker' || callAcceptance.kind !== 'turn_acceptance')
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade executed call lacks its accepted original turn',
        );
    const callOperation = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.operation_receipts, callAcceptance.id),
        OperationReceiptSchema,
    );
    const callTurn = await loadIndexedAcceptedTurn(store, root, state.turn_id, callOperation);
    const resultTurn = await loadIndexedAcceptedTurn(store, root, receipt.result_turn_id, operation);
    const call = callTurn.selected_blocks.find((block) => block.id === state.block_id);
    const result = resultTurn.selected_blocks.find((block) => block.id === state.result_block_id);
    if (
        call?.type !== 'tool_call' ||
        result?.type !== 'tool_result' ||
        callTurn.header.kind !== 'agent' ||
        state.call_id !== receipt.call_id ||
        call.call_id !== receipt.call_id ||
        result.call_id !== receipt.call_id ||
        state.terminal_receipt_id !== receipt.id ||
        state.call_fingerprint !== (await fingerprintJson(call)) ||
        receipt.executor !== call.executor ||
        receipt.status !== result.status ||
        resultTurn.header.kind !== 'tool' ||
        (receipt.call_source !== undefined &&
            (receipt.call_source.call_id !== call.call_id ||
                receipt.call_source.turn_id !== state.turn_id ||
                receipt.call_source.block_id !== call.id ||
                receipt.call_source.call_fingerprint !== state.call_fingerprint ||
                receipt.call_source.conversation.conversation_id !== root.source.conversation_id ||
                receipt.call_source.conversation.revision < callOperation.result_revision ||
                receipt.call_source.conversation.revision > operation.base_revision))
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade execution differs from its exact original call/result/source',
        );
    await assertToolResultReceiptFingerprint(result, receipt);
    const scratch = { ...progress.scratch };
    const previous = await getPagedRecord(store, scratch.call_executions, receipt.call_id);
    if (
        previous &&
        (previous.storage !== 'marker' || previous.kind !== 'upgrade_call_execution' || previous.id !== receipt.id)
    )
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade call has multiple terminal executions');
    if (!previous)
        scratch.call_executions = await putPagedRecord(store, scratch.call_executions, receipt.call_id, {
            storage: 'marker',
            kind: 'upgrade_call_execution',
            id: receipt.id,
        });
    return { ...progress, scratch, cursor: entry.key };
}
