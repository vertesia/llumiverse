import type { z } from 'zod';
import { type IndexedConversationRecordStore, loadIndexedAcceptedTurn, loadRecord } from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { ToolDefinitionSchema } from './schemas/content.js';
import { ContextEntrySchema } from './schemas/context-foundation.js';
import { GenerationSchema, OperationReceiptSchema } from './schemas/execution.js';
import {
    IndexedConversationCompactionHeaderSchema,
    IndexedConversationDeletedTurnSchema,
    type IndexedConversationRoot,
    IndexedConversationTurnHeaderSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

const families = [
    'generations',
    'tool_definitions',
    'context_entries',
    'turn_acceptances',
    'generation_acceptances',
    'block_owners',
    'deleted_turns',
    'identifiers',
] as const;

/** Reverse scans are independently paged; no absent directory is treated as completeness. */
export async function advanceIndexedUpgradeRecordAudit(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const family = families[(progress.audit_family ?? 0) - 10];
    if (family === undefined)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade reverse family is invalid');
    const page = await readPagedRecordRange(store, root.directories[family], {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (entry === undefined) {
        const { cursor: _cursor, ...next } = progress;
        return family === 'identifiers'
            ? { ...next, phase: 'active_window' }
            : { ...next, audit_family: (progress.audit_family ?? 0) + 1 };
    }
    if (family === 'generations') {
        const generation = await loadRecord(store, entry.value, GenerationSchema);
        const acceptance = await getPagedRecord(store, progress.directories.generation_acceptances, generation.id);
        if (
            entry.key !== generation.id ||
            entry.value.kind !== family ||
            acceptance?.storage !== 'marker' ||
            acceptance.kind !== 'generation_acceptance' ||
            generation.source.conversation_id !== root.source.conversation_id ||
            generation.source.revision > root.source.revision
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade generation lacks its retained original source/acceptance',
            );
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, acceptance.id),
            OperationReceiptSchema,
        );
        if (
            receipt.operation_kind !== undefined ||
            !receipt.accepted_generation_ids?.includes(generation.id) ||
            (generation.record_source === 'executed'
                ? receipt.base_revision !== generation.source.revision
                : generation.source.revision > receipt.base_revision)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade generation acceptance differs from original append',
            );
        if (
            generation.record_source === 'executed' &&
            (generation.request_receipt.source.conversation_id !== generation.source.conversation_id ||
                generation.request_receipt.source.revision !== generation.source.revision ||
                generation.request_receipt.request_id !== generation.request_id ||
                generation.request_receipt.attempt_id !== generation.attempt_id)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade executed generation has different original request identity',
            );
    } else if (family === 'tool_definitions') {
        const tool = await loadRecord(store, entry.value, ToolDefinitionSchema);
        if (tool.id !== entry.key || entry.value.kind !== family)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade tool definition differs from original identity',
            );
    } else if (family === 'context_entries') {
        const context = await loadRecord(store, entry.value, ContextEntrySchema);
        const turnDescriptor = await getPagedRecord(store, root.directories.turns, context.turn_id);
        let header: Pick<
            z.infer<typeof IndexedConversationTurnHeaderSchema>,
            'turn' | 'source' | 'block_ids' | 'compaction_id'
        >;
        if (turnDescriptor?.storage === 'marker' && turnDescriptor.kind === 'deleted_turn') {
            const acceptance = await getPagedRecord(store, progress.directories.turn_acceptances, context.turn_id);
            if (acceptance?.storage !== 'marker' || acceptance.kind !== 'turn_acceptance')
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade retained context lacks its original deleted owner',
                );
            const receipt = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.operation_receipts, acceptance.id),
                OperationReceiptSchema,
            );
            const original = await loadIndexedAcceptedTurn(store, root, context.turn_id, receipt);
            header = {
                turn: original.header,
                source: 'ordinary' as const,
                block_ids: original.selected_blocks.map((block) => block.id),
                compaction_id: undefined,
            };
        } else header = await loadRecord(store, turnDescriptor, IndexedConversationTurnHeaderSchema);

        if (
            context.id !== entry.key ||
            entry.value.kind !== family ||
            header.turn.id !== context.turn_id ||
            (context.type === 'source_turn'
                ? header.source !== 'ordinary'
                : header.source !== 'replacement' || header.compaction_id !== context.compaction_id)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade retained context entry lacks its original owner',
            );
        if (context.type === 'replacement_turn') {
            const compaction = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.compactions, context.compaction_id),
                IndexedConversationCompactionHeaderSchema,
            );
            if (compaction.id !== context.compaction_id)
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade replacement entry references different compaction',
                );
        }
        if (context.block_ids?.some((id) => !header.block_ids.includes(id)))
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade retained context entry references missing original block',
            );
    } else if (family === 'turn_acceptances' || family === 'generation_acceptances') {
        if (
            entry.value.storage !== 'marker' ||
            entry.value.kind !== (family === 'turn_acceptances' ? 'turn_acceptance' : 'generation_acceptance')
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade retained acceptance is not an exact marker',
            );
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, entry.value.id),
            OperationReceiptSchema,
        );
        if (
            receipt.operation_kind !== undefined ||
            !(family === 'turn_acceptances' ? receipt.accepted_turn_ids : receipt.accepted_generation_ids)?.includes(
                entry.key,
            )
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade retained acceptance points to a different append identity',
            );
    } else if (family === 'block_owners') {
        if (entry.value.storage !== 'marker' || entry.value.kind !== 'block_owner')
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade retained block owner is not exact');
        const descriptor = await getPagedRecord(store, root.directories.turns, entry.value.id);
        if (descriptor?.storage === 'record') {
            const header = await loadRecord(store, descriptor, IndexedConversationTurnHeaderSchema);
            if (!header.block_ids.includes(entry.key)) {
                // Nested original bodies are loaded one parent at a time by the earlier block audit.
                const nested = await getPagedRecord(store, progress.scratch.audited_identifiers, entry.key);
                if (
                    nested?.storage !== 'marker' ||
                    nested.kind !== 'upgrade_nested_block' ||
                    nested.id !== entry.value.id
                )
                    throw new IndexedConversationUpgradeEvidenceError(
                        'Indexed upgrade orphan block owner has no original nested identity',
                    );
            }
        } else if (descriptor?.storage === 'marker' && descriptor.kind === 'deleted_turn') {
            const acceptance = await getPagedRecord(store, progress.directories.turn_acceptances, entry.value.id);
            if (acceptance?.storage !== 'marker' || acceptance.kind !== 'turn_acceptance')
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade deleted owner lacks original accepted turn',
                );
            const receipt = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.operation_receipts, acceptance.id),
                OperationReceiptSchema,
            );
            const original = await loadIndexedAcceptedTurn(store, root, entry.value.id, receipt);
            if (
                !original.selected_blocks.some(
                    (block) =>
                        block.id === entry.key ||
                        (block.type === 'tool_result' && block.content.some((nested) => nested.id === entry.key)),
                )
            )
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade deleted owner lacks exact original block identity',
                );
        } else
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade orphan owner references missing original turn',
            );
    } else if (family === 'deleted_turns') {
        const tombstone = await loadRecord(store, entry.value, IndexedConversationDeletedTurnSchema);
        if (tombstone.deleted_turn.id !== entry.key)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade tombstone identity differs');
        const receipt = await loadRecord(
            store,
            await getPagedRecord(
                store,
                root.directories.operation_receipts,
                tombstone.deleted_turn.accepted_operation_id,
            ),
            OperationReceiptSchema,
        );
        await loadIndexedAcceptedTurn(store, root, entry.key, receipt);
    } else if (entry.value.storage !== 'marker' || entry.value.id !== entry.key || entry.value.kind.length === 0)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade original identifier marker is malformed');
    return { ...progress, cursor: entry.key };
}
