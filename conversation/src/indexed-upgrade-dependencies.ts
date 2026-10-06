import {
    assertIndexedAcceptedCompaction,
    IndexedCallStateSchema,
    type IndexedConversationRecordStore,
    indexedDeleteDependencyEntry,
    loadRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { AssetSchema, ContentBlockSchema } from './schemas/content.js';
import { ExecutionReceiptSchema, GenerationSchema, OperationReceiptSchema } from './schemas/execution.js';
import {
    IndexedConversationCompactionHeaderSchema,
    type IndexedConversationRoot,
    IndexedConversationTurnHeaderSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';
import { ProcessingJobSchema } from './schemas/processing.js';

const families = ['turns', 'blocks', 'execution_receipts', 'assets', 'compactions', 'processing_records'] as const;
interface Target {
    kind: 'turn' | 'block' | 'call' | 'entry' | 'request' | 'asset' | 'generation';
    id: string;
}

/** Rebuild v2 live dependency identities from the same original record families as snapshot.
 * At most one target is resolved/emitted per retained effect cursor; forensic receipts are
 * not mistaken for live body owners, and no partial array write can advance progress.
 */
export async function advanceIndexedUpgradeDependencies(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const familyIndex = (progress.audit_family ?? 0) - 3;
    const family = families[familyIndex];
    if (family === undefined)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency family is invalid');
    const page = await readPagedRecordRange(store, root.directories[family], {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, item_cursor: _itemCursor, ...next } = progress;
        return { ...next, audit_family: (progress.audit_family ?? 0) + 1 };
    }
    const targets: Target[] = [];
    let kind: 'turn' | 'asset' | 'compaction' | 'job' = 'turn';
    let owner = entry.key;
    const add = (id: string | undefined, refKind: Target['kind'] = 'turn') => {
        if (id !== undefined) targets.push({ kind: refKind, id });
    };
    if (family === 'turns') {
        if (entry.value.storage === 'record') {
            const header = await loadRecord(store, entry.value, IndexedConversationTurnHeaderSchema);
            if (header.turn.id !== entry.key)
                throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency turn identity differs');
            add(header.turn.parent_turn_id);
            if (header.turn.provenance.type === 'derived')
                for (const id of header.turn.provenance.source_turn_ids) add(id);
        } else if (entry.value.kind !== 'deleted_turn' || entry.value.id !== entry.key)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency turn tombstone is invalid');
    } else if (family === 'blocks') {
        if (entry.value.storage === 'record') {
            const block = await loadRecord(store, entry.value, ContentBlockSchema);
            if (block.id !== entry.key)
                throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency block identity differs');
            const blockOwner = await getPagedRecord(store, root.directories.block_owners, block.id);
            if (blockOwner?.storage !== 'marker' || blockOwner.kind !== 'block_owner')
                throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency block owner is missing');
            owner = blockOwner.id;
            if (block.type === 'native_replay') {
                for (const id of block.dependencies.turn_ids) add(id);
                for (const id of block.dependencies.block_ids) add(id, 'block');
                for (const id of block.dependencies.call_ids) add(id, 'call');
                for (const id of block.dependencies.request_ids) add(id, 'request');
            }
        } else if (entry.value.kind !== 'deleted_block' || entry.value.id !== entry.key)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency block tombstone is invalid');
    } else if (family === 'execution_receipts') {
        const receipt = await loadRecord(store, entry.value, ExecutionReceiptSchema);
        if (receipt.id !== entry.key)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency execution identity differs');
        if (receipt.result_turn_id !== undefined) {
            owner = receipt.result_turn_id;
            if (receipt.call_source !== undefined) add(receipt.call_source.turn_id);
            else add(receipt.call_id, 'call');
        }
    } else if (family === 'assets') {
        const asset = await loadRecord(store, entry.value, AssetSchema);
        if (asset.id !== entry.key)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency asset identity differs');
        kind = 'asset';
        if (asset.provenance.type === 'received') add(asset.provenance.source_turn_id);
    } else if (family === 'compactions') {
        const compaction = await loadRecord(store, entry.value, IndexedConversationCompactionHeaderSchema);
        if (compaction.id !== entry.key)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency compaction identity differs');
        const acceptance = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, compaction.operation_id),
            OperationReceiptSchema,
        );
        assertIndexedAcceptedCompaction(root, compaction.id, compaction, acceptance);
        kind = 'compaction';
        for (const id of compaction.source.turn_ids) add(id);
        for (const id of compaction.source.block_ids ?? []) add(id, 'block');
        for (const id of compaction.retained_asset_ids) add(id, 'asset');
        for (const id of compaction.generation_ids) add(id, 'generation');
    } else {
        let tuple: unknown;
        try {
            tuple = JSON.parse(entry.key);
        } catch (cause: unknown) {
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade dependency processing key is invalid',
                cause,
            );
        }
        if (!Array.isArray(tuple) || tuple.length !== 2 || typeof tuple[0] !== 'string' || typeof tuple[1] !== 'string')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade dependency processing identity is invalid',
            );
        if (tuple[0] === 'jobs') {
            const job = await loadRecord(store, entry.value, ProcessingJobSchema);
            if (job.id !== tuple[1])
                throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency job identity differs');
            owner = job.id;
            kind = 'job';
            if (job.selection.kind === 'entries') {
                for (const id of job.selection.entry_ids) add(id, 'entry');
                for (const selected of job.selection.selected_entries ?? []) add(selected.turn_id);
            }
            const receipt = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.operation_receipts, job.source_operation_id),
                OperationReceiptSchema,
            );
            for (const id of receipt.accepted_turn_ids ?? []) add(id);
        }
    }
    const position = progress.item_cursor ?? 0;
    if (position > targets.length)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency effect cursor is invalid');
    const target = targets[position];
    if (target === undefined) {
        const { item_cursor: _itemCursor, ...next } = progress;
        return { ...next, cursor: entry.key };
    }
    if (target.kind === 'asset') {
        const asset = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.assets, target.id),
            AssetSchema,
        );
        if (asset.id !== target.id)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade compaction retains different original asset',
            );
        return { ...progress, item_cursor: position + 1 };
    }
    if (target.kind === 'generation') {
        const generation = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.generations, target.id),
            GenerationSchema,
        );
        if (generation.id !== target.id)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade compaction retains different original generation',
            );
        return { ...progress, item_cursor: position + 1 };
    }
    if (target.kind === 'request') {
        const nomination = await getPagedRecord(store, progress.scratch.request_generations, target.id);
        if (nomination?.storage !== 'marker' || nomination.kind !== 'upgrade_request_generation')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade replay references missing original request',
            );
        const generation = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.generations, nomination.id),
            GenerationSchema,
        );
        if (generation.id !== nomination.id || generation.request_id !== target.id)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade replay request differs from original generation',
            );
        return { ...progress, item_cursor: position + 1 };
    }
    let targetTurn = target.id;
    if (target.kind === 'block') {
        const value = await getPagedRecord(store, progress.directories.block_owners, target.id);
        if (value?.storage !== 'marker' || value.kind !== 'block_owner')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade dependency references unknown block owner',
            );
        targetTurn = value.id;
    } else if (target.kind === 'call') {
        const state = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.tool_call_states, target.id),
            IndexedCallStateSchema,
        );
        if (state.call_id !== target.id)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency references different call');
        targetTurn = state.turn_id;
    } else if (target.kind === 'entry') {
        const value = await getPagedRecord(store, progress.scratch.entry_turns, target.id);
        if (value?.storage !== 'marker' || value.kind !== 'upgrade_entry_turn')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade job selection lacks retained original entry witness',
            );
        targetTurn = value.id;
    }
    if ((await getPagedRecord(store, root.directories.turns, targetTurn)) === undefined)
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade dependency references missing original turn',
        );
    const dependency = await indexedDeleteDependencyEntry({ target_turn_id: targetTurn, kind, owner_id: owner });
    const directories = { ...progress.directories };
    const retained = await getPagedRecord(store, directories.deletion_dependencies, dependency.key);
    if (retained === undefined)
        directories.deletion_dependencies = await putPagedRecord(
            store,
            directories.deletion_dependencies,
            dependency.key,
            dependency.value,
        );
    else if (retained.storage !== 'marker' || retained.kind !== dependency.value.kind || retained.id !== owner)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade dependency reconstruction conflicts');
    return { ...progress, directories, item_cursor: position + 1 };
}
