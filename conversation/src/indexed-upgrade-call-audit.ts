import { fingerprintJson } from './identity.js';
import {
    IndexedCallStateSchema,
    type IndexedConversationRecordStore,
    loadIndexedAcceptedTurn,
    loadRecord,
    stageRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import type { IndexedConversationRoot } from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

/** Reverse state coverage must prove original call bytes, including retained deleted originals. */
export async function advanceIndexedUpgradeCallAudit(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.tool_call_states, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, ...next } = progress;
        return { ...next, audit_family: 2 };
    }
    const state = await loadRecord(store, entry.value, IndexedCallStateSchema);
    if (entry.key !== state.call_id || entry.value.kind !== 'tool_call_states')
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade call state has a different immutable identity',
        );
    const acceptance = await getPagedRecord(store, progress.directories.turn_acceptances, state.turn_id);
    if (acceptance?.storage !== 'marker' || acceptance.kind !== 'turn_acceptance')
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade orphan call state lacks original acceptance',
        );
    const receipt = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.operation_receipts, acceptance.id),
        OperationReceiptSchema,
    );
    const original = await loadIndexedAcceptedTurn(store, root, state.turn_id, receipt);
    const calls = original.selected_blocks.filter((block) => block.id === state.block_id);
    const call = calls[0];
    if (
        original.header.kind !== 'agent' ||
        calls.length !== 1 ||
        call?.type !== 'tool_call' ||
        call.call_id !== state.call_id ||
        state.call_fingerprint !== (await fingerprintJson(call))
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade call state differs from accepted original call',
        );
    const execution = await getPagedRecord(store, progress.scratch.call_executions, state.call_id);
    if (
        (state.terminal_receipt_id === undefined) !== (execution === undefined) ||
        (execution !== undefined &&
            (execution.storage !== 'marker' ||
                execution.kind !== 'upgrade_call_execution' ||
                execution.id !== state.terminal_receipt_id)) ||
        (state.terminal_receipt_id === undefined) !== (state.result_block_id === undefined)
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade call state lacks complete original terminal execution coverage',
        );
    const directories = { ...progress.directories };
    if (state.terminal_receipt_id === undefined) {
        directories.open_tool_calls = await putPagedRecord(
            store,
            directories.open_tool_calls,
            state.call_id,
            await stageRecord(store, 'open_tool_calls', state.call_id, {
                call_id: state.call_id,
                turn_id: state.turn_id,
                block_id: state.block_id,
                call_fingerprint: state.call_fingerprint,
            }),
        );
    }
    return { ...progress, directories, cursor: entry.key };
}
