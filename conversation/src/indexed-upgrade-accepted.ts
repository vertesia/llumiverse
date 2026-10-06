import { fingerprintJson } from './identity.js';
import { type IndexedConversationRecordStore, loadRecord } from './indexed-conversation.js';
import { finishIndexedConversationUpgrade } from './indexed-upgrade-finish.js';
import { createIndexedUpgradeStepStore } from './indexed-upgrade-io.js';
import {
    IndexedConversationUpgradeEvidenceError,
    loadIndexedUpgradePredecessor,
    readIndexedUpgradeProgress,
} from './indexed-upgrade-progress.js';
import { getPagedRecord, type PagedRecordRef, putPagedRecord } from './paged-record-index.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import {
    INDEXED_CONVERSATION_DELETE_PROFILE_V2,
    INDEXED_CONVERSATION_PROCESSING_PROFILE,
    INDEXED_CONVERSATION_RESTART_PROFILE,
    IndexedConversationProcessingHeaderSchema,
    type IndexedConversationRoot,
    IndexedConversationRootSchema,
} from './schemas/indexed-head.js';
import { IndexedConversationUpgradeCommandSchema } from './schemas/indexed-upgrade.js';

/** Point-replay an already accepted upgrade at its exact N+1 root. This reads original
 * facts through audited rebuilt indices; it never labels a historical N root complete. */
export async function loadIndexedUpgradeAcceptedSource(
    underlying: IndexedConversationRecordStore,
    rootInput: IndexedConversationRoot,
    locator: PagedRecordRef,
    operationId: string,
) {
    // Reconstruct only two insertion paths in bounded ephemeral pages. No durable
    // write or alternate authority is produced by this accepted-source reader.
    const pages = new Map<string, Uint8Array>();
    const { store } = createIndexedUpgradeStepStore({
        async read(ref) {
            const staged = pages.get(ref.content_hash);
            return staged === undefined ? underlying.read(ref) : Uint8Array.from(staged);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        readRecord: (ref) => underlying.readRecord(ref),
        async writeRecord() {
            throw new IndexedConversationUpgradeEvidenceError('Accepted upgrade observation cannot stage records');
        },
    });
    const root = IndexedConversationRootSchema.parse(rootInput);
    const descriptor = await getPagedRecord(store, root.directories.operation_receipts, operationId);
    if (!descriptor) throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade has no exact accepted receipt');
    const receipt = await loadRecord(store, descriptor, OperationReceiptSchema);
    if (
        receipt.operation_kind !== 'indexed_upgrade' ||
        !receipt.indexed_upgrade ||
        receipt.id !== operationId ||
        receipt.result_revision !== root.source.revision ||
        receipt.conversation_id !== root.source.conversation_id
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade observation differs from its exact accepted source',
        );
    const command = IndexedConversationUpgradeCommandSchema.parse({
        version: 1,
        profile: receipt.indexed_upgrade.profile,
        operation_id: receipt.id,
        source: receipt.indexed_upgrade.source,
        predecessor_root: receipt.indexed_upgrade.predecessor_root,
        recorded_at: receipt.recorded_at,
    });
    const completeLocator = receipt.indexed_upgrade.completed_progress;
    const exactReceipt = OperationReceiptSchema.parse({
        id: command.operation_id,
        conversation_id: command.source.conversation_id,
        base_revision: command.source.revision,
        result_revision: command.source.revision + 1,
        payload_fingerprint: receipt.payload_fingerprint,
        recorded_at: command.recorded_at,
        accepted_turn_ids: [],
        operation_kind: 'indexed_upgrade',
        indexed_upgrade: receipt.indexed_upgrade,
    });
    if ((await fingerprintJson(exactReceipt)) !== (await fingerprintJson(receipt)))
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade receipt has unaudited content effects');

    const retained = await finishIndexedConversationUpgrade(store, root, locator, command, completeLocator);
    if (retained.applied)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade observation cannot publish');
    const progress = await readIndexedUpgradeProgress(store, command, completeLocator);
    const reconstructedReceipts = await putPagedRecord(
        store,
        progress.directories.operation_receipts,
        receipt.id,
        descriptor,
    );
    const reconstructedIdentifiers = await putPagedRecord(store, progress.directories.identifiers, receipt.id, {
        storage: 'marker',
        kind: 'operation receipt',
        id: receipt.id,
    });
    if (
        (await fingerprintJson(reconstructedReceipts)) !==
            (await fingerprintJson(root.directories.operation_receipts)) ||
        (await fingerprintJson(reconstructedIdentifiers)) !== (await fingerprintJson(root.directories.identifiers))
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade changed audited receipt or identifier directories',
        );
    const original = await loadIndexedUpgradePredecessor(store, command);
    const oldHeader = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: original.source.conversation_id,
            ...original.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    const { coverage: _coverage, ...header } = oldHeader;
    const expectedHeader = IndexedConversationProcessingHeaderSchema.parse({
        ...header,
        job_count: progress.counts.jobs,
        unresolved_job_count: progress.counts.unresolved_jobs,
        required_job_count: progress.counts.required_jobs,
        required_unresolved_job_count: progress.counts.required_unresolved_jobs,
        required_blocked_job_count: progress.counts.required_blocked_jobs,
    });
    const actualHeader = await loadRecord(
        store,
        { storage: 'record', kind: 'processing_header', id: root.source.conversation_id, ...root.processing_header },
        IndexedConversationProcessingHeaderSchema,
    );
    if ((await fingerprintJson(actualHeader)) !== (await fingerprintJson(expectedHeader)))
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade changed original policy or audited processing counters',
        );
    for (const [family, ref] of Object.entries(progress.directories)) {
        if (family === 'operation_receipts' || family === 'identifiers') continue;
        if (
            (await fingerprintJson(root.directories[family as keyof typeof root.directories] ?? null)) !==
            (await fingerprintJson(ref))
        )
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade changed an audited original directory');
    }
    for (const family of Object.keys(root.directories)) {
        if (family === 'operation_receipts' || family === 'identifiers') continue;
        if (!(family in progress.directories))
            throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade introduced an unaudited directory');
    }
    const marker = await getPagedRecord(store, root.directories.identifiers, receipt.id);
    if (marker?.storage !== 'marker' || marker.kind !== 'operation receipt' || marker.id !== receipt.id)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade lost its exact accepted operation identity');
    const { restart_response: _response, restart_tool_input: _toolInput, ...rest } = original;
    const expectedRoot = IndexedConversationRootSchema.parse({
        ...rest,
        source: { conversation_id: command.source.conversation_id, revision: command.source.revision + 1 },
        updated_at: command.recorded_at,
        delete_index_profile: INDEXED_CONVERSATION_DELETE_PROFILE_V2,
        processing_index_profile: INDEXED_CONVERSATION_PROCESSING_PROFILE,
        restart_index_profile: INDEXED_CONVERSATION_RESTART_PROFILE,
        accepted_output_index_complete: true,
        tool_call_state_complete: true,
        live_turn_count: progress.counts.live_turns,
        active_tail_turn_id: progress.previous_live_turn_id ?? null,
        processing_header: root.processing_header,
        directories: root.directories,
        ...(progress.restart_response === undefined ? {} : { restart_response: progress.restart_response }),
        ...(progress.restart_tool_input === undefined ? {} : { restart_tool_input: progress.restart_tool_input }),
    });
    if ((await fingerprintJson(expectedRoot)) !== (await fingerprintJson(root)))
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade changed original content, context or source facts',
        );
    return { receipt, original, original_locator: command.predecessor_root, progress };
}
