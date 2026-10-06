import { fingerprintJson } from './identity.js';
import {
    assertIndexedCurrentPolicy,
    type IndexedConversationRecordStore,
    loadIndexedAcceptedTurn,
    loadIndexedActiveContext,
    loadIndexedActiveToolDefinitions,
    loadRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord } from './paged-record-index.js';
import { GenerationSchema, OperationReceiptSchema } from './schemas/execution.js';
import {
    IndexedConversationCompactionHeaderSchema,
    IndexedConversationProcessingHeaderSchema,
    type IndexedConversationRoot,
    IndexedConversationTurnHeaderSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

/** The existing bounded active reader checks the exact ordered count/bytes/header fingerprint.
 * This phase is separately charged against the finite active-window IO budget, including
 * original turn cross-references. It returns no document and creates no readiness proof.
 */
export async function auditIndexedUpgradeActiveWindow(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const context = await loadIndexedActiveContext(store, root);
    const ids = new Set<string>();
    for (const entry of context.entries) {
        if (ids.has(entry.id))
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade active context repeats an entry identity',
            );
        ids.add(entry.id);
        const descriptor = await getPagedRecord(store, root.directories.turns, entry.turn_id);
        const header = await loadRecord(store, descriptor, IndexedConversationTurnHeaderSchema);
        if (
            header.turn.id !== entry.turn_id ||
            (entry.type === 'source_turn'
                ? header.source !== 'ordinary'
                : header.source !== 'replacement' || header.compaction_id !== entry.compaction_id)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade active entry lacks its exact live original turn',
            );
        if (entry.type === 'replacement_turn') {
            const compaction = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.compactions, entry.compaction_id),
                IndexedConversationCompactionHeaderSchema,
            );
            if (compaction.id !== entry.compaction_id)
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade active replacement lacks its exact compaction',
                );
        }
        if (
            entry.block_ids !== undefined &&
            (new Set(entry.block_ids).size !== entry.block_ids.length ||
                entry.block_ids.some((id) => !header.block_ids.includes(id)))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade active entry selects unknown/repeated original blocks',
            );
    }
    if (context.protected_entry_ids.some((id) => !ids.has(id)))
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade protected context identity is not active');
    const definitions = await loadIndexedActiveToolDefinitions(store, root);
    if (new Set(definitions.map((definition) => definition.id)).size !== definitions.length)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade active definitions repeat an identity');
    const processing = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    if (processing.selected_policy_operation_id !== undefined)
        await assertIndexedCurrentPolicy(store, root, processing);
    if (root.accepted_response !== undefined) {
        const nomination = root.accepted_response;
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, nomination.operation_id),
            OperationReceiptSchema,
        );
        const original = await loadIndexedAcceptedTurn(store, root, nomination.turn_id, receipt);
        const header = original.header;
        const generation = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.generations, nomination.generation_id),
            GenerationSchema,
        );
        if (
            receipt.operation_kind !== undefined ||
            receipt.id !== nomination.operation_id ||
            receipt.result_revision !== nomination.accepted_revision ||
            receipt.result_revision > root.source.revision ||
            (await fingerprintJson(receipt.accepted_turn_ids)) !== (await fingerprintJson([nomination.turn_id])) ||
            (await fingerprintJson(receipt.accepted_generation_ids)) !==
                (await fingerprintJson([nomination.generation_id])) ||
            header.id !== nomination.turn_id ||
            header.kind !== 'agent' ||
            !('generation_id' in header) ||
            header.generation_id !== nomination.generation_id ||
            generation.id !== nomination.generation_id ||
            generation.source.conversation_id !== root.source.conversation_id ||
            generation.source.revision > receipt.base_revision
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade accepted response differs from original immutable tuple',
            );
    }
    return { ...progress, phase: 'complete' };
}
