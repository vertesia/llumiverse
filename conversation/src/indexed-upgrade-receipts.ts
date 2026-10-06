import {
    type IndexedConversationRecordStore,
    indexedOrderedKey,
    loadIndexedAcceptedTurn,
    loadRecord,
} from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord, putPagedRecord, readPagedRecordRange } from './paged-record-index.js';
import { GenerationSchema, OperationReceiptSchema } from './schemas/execution.js';
import { type IndexedConversationRoot, IndexedConversationTurnHeaderSchema } from './schemas/indexed-head.js';
import type { IndexedConversationUpgradeProgress } from './schemas/indexed-upgrade.js';

/** One accepted receipt identity is processed at a time; effects have their own retained cursor. */
export async function advanceIndexedUpgradeReceipt(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    progress: IndexedConversationUpgradeProgress,
): Promise<IndexedConversationUpgradeProgress> {
    const page = await readPagedRecordRange(store, root.directories.operation_receipts, {
        ...(progress.cursor === undefined ? {} : { after: progress.cursor }),
        limit: 1,
    });
    const entry = page.entries[0];
    if (!entry) {
        const { cursor: _cursor, item_cursor: _itemCursor, ...next } = progress;
        return { ...next, phase: 'executions' };
    }
    const receipt = await loadRecord(store, entry.value, OperationReceiptSchema);
    if (
        entry.key !== receipt.id ||
        entry.value.kind !== 'operation_receipts' ||
        receipt.conversation_id !== root.source.conversation_id ||
        receipt.result_revision > root.source.revision ||
        receipt.result_revision !== receipt.base_revision + 1
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade receipt differs from its original source/key/revision',
        );
    const effects: {
        family: 'turn_acceptances' | 'generation_acceptances' | 'execution_acceptances' | 'entry_turns';
        id: string;
        kind: string;
        owner?: string;
    }[] = [];
    if (receipt.operation_kind === undefined) {
        for (const id of receipt.accepted_turn_ids ?? [])
            effects.push({ family: 'turn_acceptances', id, kind: 'turn_acceptance' });
        for (const id of receipt.accepted_generation_ids ?? [])
            effects.push({ family: 'generation_acceptances', id, kind: 'generation_acceptance' });
        for (const id of receipt.accepted_execution_receipt_ids ?? [])
            effects.push({ family: 'execution_acceptances', id, kind: 'upgrade_execution_acceptance' });
        for (const accepted of receipt.accepted_context_entries ?? [])
            effects.push({
                family: 'entry_turns',
                id: accepted.id,
                kind: 'upgrade_entry_turn',
                owner: accepted.turn_id,
            });
    } else if (
        (receipt.accepted_execution_receipt_ids?.length ?? 0) > 0 ||
        (receipt.accepted_generation_ids?.length ?? 0) > 0 ||
        (receipt.accepted_turn_ids?.length ?? 0) > 0
    ) {
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade authoring receipt claims append outputs');
    }
    const directories = { ...progress.directories };
    const scratch = { ...progress.scratch };
    const position = progress.item_cursor ?? 0;
    if (position > effects.length)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade receipt effect cursor is invalid');
    const effect = effects[position];
    if (effect) {
        const directory =
            effect.family === 'execution_acceptances'
                ? scratch.execution_acceptances
                : effect.family === 'entry_turns'
                  ? scratch.entry_turns
                  : directories[effect.family];
        const existing = await getPagedRecord(store, directory, effect.id);
        if (
            existing &&
            (existing.storage !== 'marker' ||
                existing.kind !== effect.kind ||
                existing.id !== (effect.owner ?? receipt.id))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade append identity has multiple acceptance receipts',
            );
        if (!existing) {
            const nextDirectory = await putPagedRecord(store, directory, effect.id, {
                storage: 'marker',
                kind: effect.kind,
                id: effect.owner ?? receipt.id,
            });
            if (effect.family === 'execution_acceptances') scratch.execution_acceptances = nextDirectory;
            else if (effect.family === 'entry_turns') scratch.entry_turns = nextDirectory;
            else directories[effect.family] = nextDirectory;
        }
        if (effect.family === 'generation_acceptances') {
            const generation = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.generations, effect.id),
                GenerationSchema,
            );
            if (generation.id !== effect.id)
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade generation acceptance references different original',
                );
            if (
                generation.request_id !== undefined &&
                (await getPagedRecord(store, scratch.request_generations, generation.request_id)) === undefined
            )
                scratch.request_generations = await putPagedRecord(
                    store,
                    scratch.request_generations,
                    generation.request_id,
                    { storage: 'marker', kind: 'upgrade_request_generation', id: generation.id },
                );
        }
        return { ...progress, directories, scratch, item_cursor: position + 1 };
    }
    let restartResponse = progress.restart_response;
    let restartToolInput = progress.restart_tool_input;
    if (receipt.operation_kind === undefined && (receipt.accepted_generation_ids?.length ?? 0) > 0) {
        const key = indexedOrderedKey(receipt.result_revision);
        const existing = await getPagedRecord(store, scratch.response_revisions, key);
        if (existing && existing.id !== receipt.id)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade response revision is ambiguous');
        if (!existing)
            scratch.response_revisions = await putPagedRecord(store, scratch.response_revisions, key, {
                storage: 'marker',
                kind: 'upgrade_response_revision',
                id: receipt.id,
            });
        if (!restartResponse || receipt.result_revision > restartResponse.result_revision)
            restartResponse = { operation_id: receipt.id, result_revision: receipt.result_revision };
        if (receipt.accepted_generation_ids?.length === 1 && receipt.accepted_turn_ids?.length === 1) {
            const generationId = receipt.accepted_generation_ids[0];
            const turnId = receipt.accepted_turn_ids[0];
            const generation = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.generations, generationId),
                GenerationSchema,
            );
            // Deleted originals are audited in the deletion phase; their authoritative original
            // header remains reachable through the tombstone's immutable predecessor, not guessed here.
            const descriptor = await getPagedRecord(store, root.directories.turns, turnId);
            if (descriptor?.storage === 'record') {
                const turn = await loadRecord(store, descriptor, IndexedConversationTurnHeaderSchema);
                if (turn.turn.kind === 'agent' && turn.turn.provenance.type === 'generated') {
                    if (
                        !('generation_id' in turn.turn) ||
                        turn.turn.generation_id !== generation.id ||
                        generation.source.conversation_id !== root.source.conversation_id ||
                        generation.source.revision !== receipt.base_revision
                    )
                        throw new IndexedConversationUpgradeEvidenceError(
                            'Indexed upgrade response lacks exact original generation/turn acceptance',
                        );
                    directories.accepted_output_order = await putPagedRecord(
                        store,
                        directories.accepted_output_order,
                        key,
                        { storage: 'marker', kind: 'accepted_output', id: receipt.id },
                    );
                }
            } else if (descriptor?.storage === 'marker' && descriptor.kind === 'deleted_turn') {
                const original = await loadIndexedAcceptedTurn(store, root, turnId, receipt);
                if (original.header.kind === 'agent' && original.header.provenance.type === 'generated') {
                    if (
                        !('generation_id' in original.header) ||
                        original.header.generation_id !== generation.id ||
                        generation.source.conversation_id !== root.source.conversation_id ||
                        generation.source.revision !== receipt.base_revision
                    )
                        throw new IndexedConversationUpgradeEvidenceError(
                            'Indexed upgrade deleted response lacks its original exact generation acceptance',
                        );
                    directories.accepted_output_order = await putPagedRecord(
                        store,
                        directories.accepted_output_order,
                        key,
                        { storage: 'marker', kind: 'accepted_output', id: receipt.id },
                    );
                }
            } else {
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade accepted response original turn is missing',
                );
            }
        }
    }
    if (receipt.operation_kind === undefined && (receipt.accepted_execution_receipt_ids?.length ?? 0) > 0) {
        const key = indexedOrderedKey(receipt.result_revision);
        const existing = await getPagedRecord(store, scratch.input_revisions, key);
        if (existing && existing.id !== receipt.id)
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade tool-input revision is ambiguous');
        if (!existing)
            scratch.input_revisions = await putPagedRecord(store, scratch.input_revisions, key, {
                storage: 'marker',
                kind: 'upgrade_input_revision',
                id: receipt.id,
            });
        if (!restartToolInput || receipt.result_revision > restartToolInput.result_revision)
            restartToolInput = { operation_id: receipt.id, result_revision: receipt.result_revision };
    }
    const { item_cursor: _itemCursor, restart_response: _response, restart_tool_input: _input, ...next } = progress;
    return {
        ...next,
        directories,
        scratch,
        cursor: entry.key,
        ...(restartResponse === undefined ? {} : { restart_response: restartResponse }),
        ...(restartToolInput === undefined ? {} : { restart_tool_input: restartToolInput }),
    };
}
