import {
    type IndexedConversationRecordStore,
    indexedDisplayAnswer,
    indexedDisplayAnswerKey,
    indexedOrderedKey,
    loadRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import {
    type IndexedConversationRoot,
    IndexedConversationTurnHeaderSchema,
    IndexedConversationTurnLinkSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

/** Original ordinals remain lifetime identities; live links and tail are independently audited. */
export async function advanceIndexedUpgradeOrder(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.turn_order, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        if (
            progress.counts.source_records !== progress.counts.audited_records ||
            progress.counts.source_records !== root.turn_count ||
            (root.live_turn_count !== undefined && root.live_turn_count !== progress.counts.live_turns) ||
            (root.active_tail_turn_id !== undefined &&
                root.active_tail_turn_id !== (progress.previous_live_turn_id ?? null))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade original turn count/live count/tail differs from complete order',
            );
        if (progress.previous_live_turn_id !== undefined) {
            const tail = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.turn_links, progress.previous_live_turn_id),
                IndexedConversationTurnLinkSchema,
            );
            if (tail.next_turn_id !== undefined)
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade live tail has an unaudited successor',
                );
        }
        const { cursor: _cursor, ...next } = progress;
        return { ...next, phase: 'processing' };
    }
    const ordinal = progress.counts.source_records;
    if (entry.key !== indexedOrderedKey(ordinal) || entry.value.storage !== 'marker')
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade turn order has a missing/repeated original ordinal',
        );
    const scratch = { ...progress.scratch };
    if ((await getPagedRecord(store, scratch.ordered_turns, entry.value.id)) !== undefined)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade repeats an original turn in lifetime order');
    scratch.ordered_turns = await putPagedRecord(store, scratch.ordered_turns, entry.value.id, {
        storage: 'marker',
        kind: 'upgrade_ordered_turn',
        id: entry.key,
    });
    const counts = { ...progress.counts, source_records: ordinal + 1 };
    const descriptor = await getPagedRecord(store, root.directories.turns, entry.value.id);
    if (entry.value.kind === 'deleted_turn_order') {
        if (descriptor?.storage !== 'marker' || descriptor.kind !== 'deleted_turn' || descriptor.id !== entry.value.id)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade deleted order lacks exact audited tombstone',
            );
        return { ...progress, scratch, counts, cursor: entry.key };
    }
    if (entry.value.kind !== 'turn_order')
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade order has an unsupported record kind');
    const header = await loadRecord(store, descriptor, IndexedConversationTurnHeaderSchema);
    const link = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.turn_links, entry.value.id),
        IndexedConversationTurnLinkSchema,
    );
    if (
        header.turn.id !== entry.value.id ||
        header.source !== 'ordinary' ||
        link.id !== entry.value.id ||
        link.ordinal !== ordinal ||
        link.previous_turn_id !== progress.previous_live_turn_id
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade live original turn differs from its exact order/link',
        );
    if (progress.previous_live_turn_id !== undefined) {
        const previous = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.turn_links, progress.previous_live_turn_id),
            IndexedConversationTurnLinkSchema,
        );
        if (previous.next_turn_id !== link.id)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade live links do not agree in both directions',
            );
    }
    const directories = { ...progress.directories };
    if (indexedDisplayAnswer(header.turn))
        directories.display_answer_order = await putPagedRecord(
            store,
            directories.display_answer_order,
            indexedDisplayAnswerKey(ordinal),
            { storage: 'marker', kind: 'display_answer', id: header.turn.id },
        );
    counts.live_turns += 1;
    return {
        ...progress,
        directories,
        scratch,
        counts,
        cursor: entry.key,
        previous_live_turn_id: link.id,
        previous_live_ordinal: ordinal,
    };
}
