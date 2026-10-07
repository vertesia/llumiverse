import { fingerprintJson } from './identity.js';
import {
    assertIndexedAcceptedCompaction,
    type IndexedConversationRecordStore,
    loadIndexedAcceptedTurn,
    loadRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { ContentBlockSchema } from './schemas/content.js';
import { GenerationSchema, OperationReceiptSchema } from './schemas/execution.js';
import {
    IndexedConversationCompactionHeaderSchema,
    type IndexedConversationRoot,
    IndexedConversationTurnHeaderSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

/** One original turn/header or one contained block per retained step; never a lifetime turn array. */
export async function advanceIndexedUpgradeTurn(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.turns, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, item_cursor: _itemCursor, ...next } = progress;
        return { ...next, phase: 'blocks' };
    }
    if (entry.value.storage === 'marker' && entry.value.kind === 'deleted_turn') {
        const accepted = await getPagedRecord(store, progress.directories.turn_acceptances, entry.key);
        if (accepted?.storage !== 'marker' || accepted.kind !== 'turn_acceptance')
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade deleted turn lacks original acceptance');
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, accepted.id),
            OperationReceiptSchema,
        );
        const original = await loadIndexedAcceptedTurn(store, root, entry.key, receipt);
        const identities: { id: string; kind: 'block' | 'tool call' }[] = [];
        for (const block of original.selected_blocks) {
            identities.push({ id: block.id, kind: 'block' });
            if (block.type === 'tool_call') identities.push({ id: block.call_id, kind: 'tool call' });
            if (block.type === 'tool_result')
                for (const nested of block.content) identities.push({ id: nested.id, kind: 'block' });
        }
        if (new Set(identities.map((value) => value.id)).size !== identities.length)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade deleted original repeats a reserved content identity',
            );
        const position = progress.item_cursor ?? 0;
        if (position > identities.length)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade deleted original identity cursor is invalid',
            );
        const identity = identities[position];
        if (identity !== undefined) {
            const directories = { ...progress.directories };
            const retained = await getPagedRecord(store, directories.identifiers, identity.id);
            if (
                retained !== undefined &&
                (retained.storage !== 'marker' ||
                    retained.id !== identity.id ||
                    (retained.kind !== identity.kind &&
                        !(identity.kind === 'block' && retained.kind === 'deleted block')))
            )
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade deleted original conflicts with reserved namespace',
                );
            if (retained === undefined)
                directories.identifiers = await putPagedRecord(store, directories.identifiers, identity.id, {
                    storage: 'marker',
                    kind: identity.kind,
                    id: identity.id,
                });
            if (identity.kind === 'block') {
                const owner = await getPagedRecord(store, directories.block_owners, identity.id);
                if (
                    owner !== undefined &&
                    (owner.storage !== 'marker' || owner.kind !== 'block_owner' || owner.id !== entry.key)
                )
                    throw new IndexedConversationUpgradeEvidenceError(
                        'Indexed upgrade deleted content conflicts with original owner',
                    );
                if (owner === undefined)
                    directories.block_owners = await putPagedRecord(store, directories.block_owners, identity.id, {
                        storage: 'marker',
                        kind: 'block_owner',
                        id: entry.key,
                    });
            }
            return { ...progress, directories, item_cursor: position + 1 };
        }
        const { item_cursor: _itemCursor, ...next } = progress;
        return {
            ...next,
            counts: { ...progress.counts, audited_records: progress.counts.audited_records + 1 },
            cursor: entry.key,
        };
    }
    const header = await loadRecord(store, entry.value, IndexedConversationTurnHeaderSchema);
    if (
        header.turn.id !== entry.key ||
        entry.value.kind !== 'turns' ||
        header.block_ids_hash !== (await fingerprintJson(header.block_ids))
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade turn identity/block order differs from original bytes',
        );
    if (header.source === 'replacement') {
        if (header.compaction_id === undefined)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade replacement lacks original compaction identity',
            );
        const compaction = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.compactions, header.compaction_id),
            IndexedConversationCompactionHeaderSchema,
        );
        const acceptance = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, compaction.operation_id),
            OperationReceiptSchema,
        );
        assertIndexedAcceptedCompaction(root, header.compaction_id, compaction, acceptance);
        const provenance = header.turn.provenance;
        if (
            provenance.type !== 'derived' ||
            provenance.derivation_id !== compaction.id ||
            provenance.source_hash !== compaction.source.source_fingerprint ||
            provenance.source_turn_ids.length === 0 ||
            provenance.source_turn_ids.some((id) => !compaction.source.turn_ids.includes(id)) ||
            (compaction.source.block_ids !== undefined &&
                (provenance.source_block_ids ?? []).some((id) => !compaction.source.block_ids?.includes(id)))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade replacement differs from original derivation witness',
            );
    }
    const position = progress.item_cursor ?? 0;
    if (position > header.block_ids.length)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade turn block cursor is invalid');
    if (position === 0 && header.source === 'ordinary') {
        const accepted = await getPagedRecord(store, progress.directories.turn_acceptances, header.turn.id);
        if (accepted?.storage !== 'marker' || accepted.kind !== 'turn_acceptance')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade ordinary turn lacks immutable original acceptance',
            );
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, accepted.id),
            OperationReceiptSchema,
        );
        if (
            !receipt.accepted_turn_ids?.includes(header.turn.id) ||
            receipt.operation_kind !== undefined ||
            receipt.conversation_id !== root.source.conversation_id ||
            receipt.result_revision > root.source.revision
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade ordinary turn differs from exact original acceptance',
            );
        if (header.turn.kind === 'agent' && header.turn.provenance.type === 'generated') {
            if (!('generation_id' in header.turn) || !header.turn.generation_id)
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade generated turn lacks original generation',
                );
            const generation = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.generations, header.turn.generation_id),
                GenerationSchema,
            );
            const generationAcceptance = await getPagedRecord(
                store,
                progress.directories.generation_acceptances,
                generation.id,
            );
            if (
                generationAcceptance?.storage !== 'marker' ||
                generationAcceptance.kind !== 'generation_acceptance' ||
                generationAcceptance.id !== receipt.id ||
                !receipt.accepted_generation_ids?.includes(generation.id) ||
                generation.source.conversation_id !== root.source.conversation_id ||
                generation.source.revision !== receipt.base_revision
            )
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade generated turn has inconsistent original generation acceptance',
                );
        }
    }
    const blockId = header.block_ids[position];
    if (blockId !== undefined) {
        const block = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.blocks, blockId),
            ContentBlockSchema,
        );
        const owner = await getPagedRecord(store, root.directories.block_owners, blockId);
        if (
            block.id !== blockId ||
            owner?.storage !== 'marker' ||
            owner.kind !== 'block_owner' ||
            owner.id !== header.turn.id
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade block differs from its exact original owner',
            );
        return { ...progress, item_cursor: position + 1 };
    }
    const { item_cursor: _itemCursor, ...next } = progress;
    return {
        ...next,
        counts: {
            ...progress.counts,
            audited_records: progress.counts.audited_records + (header.source === 'ordinary' ? 1 : 0),
        },
        cursor: entry.key,
    };
}
