import { fingerprintJson } from './identity.js';
import {
    assertIndexedAcceptedCompaction,
    IndexedCallStateSchema,
    type IndexedConversationRecordStore,
    loadIndexedAcceptedTurn,
    loadRecord,
} from './indexed-conversation.js';
import { auditIndexedUpgradeContent } from './indexed-upgrade-content.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { ContentBlockSchema } from './schemas/content.js';
import { ExecutionReceiptSchema, OperationReceiptSchema } from './schemas/execution.js';
import {
    IndexedConversationCompactionHeaderSchema,
    type IndexedConversationRoot,
    IndexedConversationTurnHeaderSchema,
} from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';
import { isToolResultTextStrategy } from './tool-result-text-strategy.js';

/** Reverse audit catches orphan blocks/calls which a forward turn walk cannot establish. */
export async function advanceIndexedUpgradeBlock(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.blocks, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, ...next } = progress;
        return { ...next, phase: 'turn_order' };
    }
    if (entry.value.storage === 'marker' && entry.value.kind === 'deleted_block') {
        const owner = await getPagedRecord(store, root.directories.block_owners, entry.key);
        if (entry.value.id !== entry.key || owner?.storage !== 'marker' || owner.kind !== 'block_owner')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade deleted block lacks exact original owner',
            );
        const accepted = await getPagedRecord(store, progress.directories.turn_acceptances, owner.id);
        if (accepted?.storage !== 'marker' || accepted.kind !== 'turn_acceptance')
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade deleted block lacks original turn acceptance',
            );
        const receipt = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, accepted.id),
            OperationReceiptSchema,
        );
        const original = await loadIndexedAcceptedTurn(store, root, owner.id, receipt);
        if (original.selected_blocks.filter((block) => block.id === entry.key).length !== 1)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade deleted block differs from retained original turn',
            );
        return { ...progress, cursor: entry.key };
    }
    const block = await loadRecord(store, entry.value, ContentBlockSchema);
    const owner = await getPagedRecord(store, root.directories.block_owners, block.id);
    if (
        entry.key !== block.id ||
        entry.value.kind !== 'blocks' ||
        owner?.storage !== 'marker' ||
        owner.kind !== 'block_owner'
    )
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade block has no exact original owner');
    const turn = await loadRecord(
        store,
        await getPagedRecord(store, root.directories.turns, owner.id),
        IndexedConversationTurnHeaderSchema,
    );
    if (turn.turn.id !== owner.id || turn.block_ids.filter((id) => id === block.id).length !== 1)
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade orphan block is not in its original turn order',
        );
    const scratch = { ...progress.scratch };
    const directories = { ...progress.directories };
    const nested = block.type === 'tool_result' ? block.content : [];
    const position = progress.item_cursor ?? 0;
    if (position > nested.length)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade nested original block cursor is invalid');
    const child = nested[position];
    if (child !== undefined) {
        await auditIndexedUpgradeContent(store, root, child);
        if ((await getPagedRecord(store, root.directories.blocks, child.id)) !== undefined)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade nested identity also names an original top-level block',
            );

        const previous = await getPagedRecord(store, scratch.audited_identifiers, child.id);
        if (previous !== undefined)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade nested block identity has multiple original occurrences',
            );
        if (previous === undefined)
            scratch.audited_identifiers = await putPagedRecord(store, scratch.audited_identifiers, child.id, {
                storage: 'marker',
                kind: 'upgrade_nested_block',
                id: turn.turn.id,
            });
        const retained = await getPagedRecord(store, directories.block_owners, child.id);
        if (
            retained !== undefined &&
            (retained.storage !== 'marker' || retained.kind !== 'block_owner' || retained.id !== turn.turn.id)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade nested block owner conflicts with original namespace',
            );
        if (retained === undefined)
            directories.block_owners = await putPagedRecord(store, directories.block_owners, child.id, {
                storage: 'marker',
                kind: 'block_owner',
                id: turn.turn.id,
            });
        const identity = await getPagedRecord(store, directories.identifiers, child.id);
        if (
            identity !== undefined &&
            (identity.storage !== 'marker' || identity.kind !== 'block' || identity.id !== child.id)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade nested block identity conflicts with canonical namespace',
            );
        if (identity === undefined)
            directories.identifiers = await putPagedRecord(store, directories.identifiers, child.id, {
                storage: 'marker',
                kind: 'block',
                id: child.id,
            });
        return { ...progress, scratch, directories, item_cursor: position + 1 };
    }
    await auditIndexedUpgradeContent(store, root, block);
    if (block.type === 'tool_call') {
        const state = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.tool_call_states, block.call_id),
            IndexedCallStateSchema,
        );
        if (
            state.call_id !== block.call_id ||
            state.turn_id !== turn.turn.id ||
            state.block_id !== block.id ||
            state.call_fingerprint !== (await fingerprintJson(block))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade call differs from original exact call state',
            );
        const terminal = await getPagedRecord(store, scratch.call_executions, block.call_id);
        if (
            (state.terminal_receipt_id === undefined) !== (terminal === undefined) ||
            (terminal !== undefined &&
                (terminal.storage !== 'marker' ||
                    terminal.kind !== 'upgrade_call_execution' ||
                    terminal.id !== state.terminal_receipt_id))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade call terminal state lacks original execution coverage',
            );
    }
    if (block.type === 'tool_result' && turn.source === 'replacement') {
        if (!turn.compaction_id)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade derived result lacks its compaction');
        const compaction = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.compactions, turn.compaction_id),
            IndexedConversationCompactionHeaderSchema,
        );
        const acceptance = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.operation_receipts, compaction.operation_id),
            OperationReceiptSchema,
        );
        assertIndexedAcceptedCompaction(root, turn.compaction_id, compaction, acceptance);
        const state = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.tool_call_states, block.call_id),
            IndexedCallStateSchema,
        );
        const terminal = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.execution_receipts, state.terminal_receipt_id ?? ''),
            ExecutionReceiptSchema,
        );
        let originalIdentityBound =
            !!terminal.result_turn_id && compaction.source.turn_ids.includes(terminal.result_turn_id);
        if (!originalIdentityBound && compaction.strategy.version === '3' && turn.turn.provenance.type === 'derived') {
            // Authenticate the immediate current projection; deterministic original binding is
            // regenerated/audited one job at a time before this upgrade can publish readiness.
            for (const id of turn.turn.provenance.source_turn_ids) {
                if (!compaction.source.turn_ids.includes(id)) continue;
                const source = await loadRecord(
                    store,
                    await getPagedRecord(store, root.directories.turns, id),
                    IndexedConversationTurnHeaderSchema,
                );
                if (
                    source.source !== 'replacement' ||
                    source.turn.kind !== 'tool' ||
                    source.turn.execution_id !== terminal.id ||
                    !source.compaction_id
                )
                    continue;
                const predecessor = await loadRecord(
                    store,
                    await getPagedRecord(store, root.directories.compactions, source.compaction_id),
                    IndexedConversationCompactionHeaderSchema,
                );
                const accepted = await loadRecord(
                    store,
                    await getPagedRecord(store, root.directories.operation_receipts, predecessor.operation_id),
                    OperationReceiptSchema,
                );
                assertIndexedAcceptedCompaction(root, predecessor.id, predecessor, accepted);
                if (
                    accepted.result_revision >= acceptance.result_revision ||
                    (compaction.supersedes_compaction_id !== undefined &&
                        compaction.supersedes_compaction_id !== predecessor.id)
                )
                    throw new IndexedConversationUpgradeEvidenceError(
                        'Indexed chained compaction changed its predecessor ordering',
                    );
                originalIdentityBound = true;
            }
        }
        if (
            !isToolResultTextStrategy(compaction.strategy.id, compaction.strategy.version) ||
            turn.turn.kind !== 'tool' ||
            turn.turn.execution_id !== terminal.id ||
            terminal.id !== state.terminal_receipt_id ||
            terminal.call_id !== block.call_id ||
            terminal.status !== block.status ||
            state.result_block_id === undefined ||
            !terminal.result_turn_id ||
            !originalIdentityBound
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade derived result differs from its immutable original terminal lineage',
            );
        // Its deterministic compaction validation is independently audited/recovered in the
        // processing phases. A derived result never becomes an original terminal candidate.
    } else if (block.type === 'tool_result') {
        const previous = await getPagedRecord(store, scratch.call_results, block.call_id);
        if (
            previous &&
            (previous.storage !== 'marker' || previous.kind !== 'upgrade_call_result' || previous.id !== block.id)
        )
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade call has multiple original results');
        const state = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.tool_call_states, block.call_id),
            IndexedCallStateSchema,
        );
        if (state.result_block_id !== block.id || state.terminal_receipt_id === undefined)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade result lacks exact original terminal call state',
            );
        if (!previous)
            scratch.call_results = await putPagedRecord(store, scratch.call_results, block.call_id, {
                storage: 'marker',
                kind: 'upgrade_call_result',
                id: block.id,
            });
    }
    const { item_cursor: _itemCursor, ...next } = progress;
    return { ...next, scratch, directories, cursor: entry.key };
}
