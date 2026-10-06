import { fingerprintJson } from './identity.js';
import { IndexedCallStateSchema, type IndexedConversationRecordStore, loadRecord } from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import type { IndexedConversationRoot } from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

/** Old pending nominations may be absent, but every retained nomination must be authentic. */
export async function advanceIndexedUpgradeOpenCallAudit(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.open_tool_calls, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, ...next } = progress;
        return { ...next, audit_family: 3 };
    }
    const open = await loadRecord(
        store,
        entry.value,
        IndexedCallStateSchema.omit({ result_block_id: true, terminal_receipt_id: true }),
    );
    const state = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.tool_call_states, entry.key),
        IndexedCallStateSchema,
    );
    const nomination = await getPagedRecord(store, progress.directories.open_tool_calls, entry.key);
    if (
        entry.value.kind !== 'open_tool_calls' ||
        entry.key !== open.call_id ||
        state.terminal_receipt_id !== undefined ||
        state.result_block_id !== undefined ||
        (await fingerprintJson(open)) !== (await fingerprintJson(state)) ||
        nomination === undefined
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade retained pending call nomination is stale or corrupt',
        );
    return { ...progress, cursor: entry.key };
}
