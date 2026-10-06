import { fingerprintJson } from './identity.js';
import { type IndexedConversationRecordStore, loadRecord, stageRecord } from './indexed-conversation.js';
import { createIndexedUpgradeStepStore, INDEXED_UPGRADE_STEP_LIMITS } from './indexed-upgrade-io.js';
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
import {
    type IndexedConversationUpgradeCommand,
    IndexedConversationUpgradeCommandSchema,
} from './schemas/indexed-upgrade.js';

/** Only the authenticated host nominates its retained complete staging pointer. Publication
 * is an ordinary N+1 CAS, never a second locator for historical N or a readiness assertion.
 */
export async function finishIndexedConversationUpgrade(
    underlying: IndexedConversationRecordStore,
    currentInput: IndexedConversationRoot,
    currentLocator: PagedRecordRef,
    commandInput: IndexedConversationUpgradeCommand,
    completeLocator: PagedRecordRef,
) {
    const { store } = createIndexedUpgradeStepStore(underlying, INDEXED_UPGRADE_STEP_LIMITS);
    const command = IndexedConversationUpgradeCommandSchema.parse(commandInput);
    const current = IndexedConversationRootSchema.parse(currentInput);
    const complete = await readIndexedUpgradeProgress(store, command, completeLocator);
    if (complete.phase !== 'complete')
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade cannot publish incomplete evidence');
    const payload = await fingerprintJson({ command, completed_progress: completeLocator });
    const retained = await getPagedRecord(store, current.directories.operation_receipts, command.operation_id);
    if (retained !== undefined) {
        const receipt = await loadRecord(store, retained, OperationReceiptSchema);
        if (
            current.source.conversation_id !== command.source.conversation_id ||
            current.source.revision < receipt.result_revision ||
            receipt.id !== command.operation_id ||
            receipt.operation_kind !== 'indexed_upgrade' ||
            receipt.payload_fingerprint !== payload ||
            receipt.conversation_id !== command.source.conversation_id ||
            receipt.base_revision !== command.source.revision ||
            receipt.result_revision !== command.source.revision + 1 ||
            (await fingerprintJson(receipt.indexed_upgrade)) !==
                (await fingerprintJson({
                    version: 1,
                    profile: command.profile,
                    source: command.source,
                    predecessor_root: command.predecessor_root,
                    completed_progress: completeLocator,
                }))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade retry differs from exact retained acceptance',
            );
        return { root: current, locator: currentLocator, receipt, applied: false };
    }
    if (
        current.source.conversation_id !== command.source.conversation_id ||
        current.source.revision !== command.source.revision ||
        (await fingerprintJson(currentLocator)) !== (await fingerprintJson(command.predecessor_root)) ||
        current.source.revision >= Number.MAX_SAFE_INTEGER
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade publication lost its exact original predecessor',
        );
    if ((await getPagedRecord(store, complete.directories.identifiers, command.operation_id)) !== undefined)
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade operation reuses an existing original identity',
        );
    const original = await loadIndexedUpgradePredecessor(store, command);
    if ((await fingerprintJson(original)) !== (await fingerprintJson(current)))
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade predecessor bytes differ from current publication',
        );
    const originalHeader = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'processing_header',
            id: original.source.conversation_id,
            ...original.processing_header,
        },
        IndexedConversationProcessingHeaderSchema,
    );
    for (const [field, count] of [
        ['job_count', complete.counts.jobs],
        ['unresolved_job_count', complete.counts.unresolved_jobs],
        ['required_job_count', complete.counts.required_jobs],
        ['required_unresolved_job_count', complete.counts.required_unresolved_jobs],
        ['required_blocked_job_count', complete.counts.required_blocked_jobs],
    ] as const)
        if (originalHeader[field] !== undefined && originalHeader[field] !== count)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade original processing counter differs from independently audited records',
            );
    const { coverage: _coverage, ...header } = originalHeader;
    const updatedHeader = IndexedConversationProcessingHeaderSchema.parse({
        ...header,
        job_count: complete.counts.jobs,
        unresolved_job_count: complete.counts.unresolved_jobs,
        required_job_count: complete.counts.required_jobs,
        required_unresolved_job_count: complete.counts.required_unresolved_jobs,
        required_blocked_job_count: complete.counts.required_blocked_jobs,
    });
    const descriptor = await stageRecord(store, 'processing_header', original.source.conversation_id, updatedHeader);
    const receipt = OperationReceiptSchema.parse({
        id: command.operation_id,
        conversation_id: command.source.conversation_id,
        base_revision: command.source.revision,
        result_revision: command.source.revision + 1,
        payload_fingerprint: payload,
        recorded_at: command.recorded_at,
        accepted_turn_ids: [],
        operation_kind: 'indexed_upgrade',
        indexed_upgrade: {
            version: 1,
            profile: command.profile,
            source: command.source,
            predecessor_root: command.predecessor_root,
            completed_progress: completeLocator,
        },
    });
    const directories = { ...complete.directories };
    directories.operation_receipts = await putPagedRecord(
        store,
        directories.operation_receipts,
        receipt.id,
        await stageRecord(store, 'operation_receipts', receipt.id, receipt),
    );
    directories.identifiers = await putPagedRecord(store, directories.identifiers, receipt.id, {
        storage: 'marker',
        kind: 'operation receipt',
        id: receipt.id,
    });
    const { restart_response: _response, restart_tool_input: _toolInput, ...rest } = original;
    const root = IndexedConversationRootSchema.parse({
        ...rest,
        source: { conversation_id: command.source.conversation_id, revision: command.source.revision + 1 },
        updated_at: command.recorded_at,
        delete_index_profile: INDEXED_CONVERSATION_DELETE_PROFILE_V2,
        processing_index_profile: INDEXED_CONVERSATION_PROCESSING_PROFILE,
        restart_index_profile: INDEXED_CONVERSATION_RESTART_PROFILE,
        accepted_output_index_complete: true,
        tool_call_state_complete: true,
        live_turn_count: complete.counts.live_turns,
        active_tail_turn_id: complete.previous_live_turn_id ?? null,
        processing_header: { content_hash: descriptor.content_hash, size_bytes: descriptor.size_bytes },
        directories,
        ...(complete.restart_response === undefined ? {} : { restart_response: complete.restart_response }),
        ...(complete.restart_tool_input === undefined ? {} : { restart_tool_input: complete.restart_tool_input }),
    });
    const rootDescriptor = await stageRecord(store, 'root', root.source.conversation_id, root);
    return {
        root,
        locator: { content_hash: rootDescriptor.content_hash, size_bytes: rootDescriptor.size_bytes },
        receipt,
        applied: true,
    };
}
