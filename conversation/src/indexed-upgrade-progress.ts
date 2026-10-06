import { canonicalJsonContentBytes } from './content-integrity.js';
import { fingerprintJson } from './identity.js';
import { type IndexedConversationRecordStore, loadRecord, stageRecord } from './indexed-conversation.js';
import { createIndexedUpgradeStepStore } from './indexed-upgrade-io.js';
import type { PagedRecordRef } from './paged-record-index.js';
import { type IndexedConversationRoot, IndexedConversationRootSchema } from './schemas/indexed-head.js';
import {
    type IndexedConversationUpgradeCommand,
    IndexedConversationUpgradeCommandSchema,
    type IndexedConversationUpgradeProgress,
    IndexedConversationUpgradeProgressSchema,
} from './schemas/indexed-upgrade.js';

export class IndexedConversationUpgradeEvidenceError extends TypeError {
    constructor(message: string, cause?: unknown) {
        super(message, cause === undefined ? undefined : { cause });
        this.name = 'IndexedConversationUpgradeEvidenceError';
    }
}

/** Own and verify the exact root supplied by the authenticated host. No latest-head lookup occurs here. */
export async function loadIndexedUpgradePredecessor(
    store: IndexedConversationRecordStore,
    command: IndexedConversationUpgradeCommand,
): Promise<IndexedConversationRoot> {
    const root = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'root',
            id: command.source.conversation_id,
            ...command.predecessor_root,
        },
        IndexedConversationRootSchema,
    );
    if (
        root.source.conversation_id !== command.source.conversation_id ||
        root.source.revision !== command.source.revision
    )
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade predecessor has another exact source');
    return root;
}

export async function stageIndexedUpgradeProgress(
    store: IndexedConversationRecordStore,
    progress: IndexedConversationUpgradeProgress,
): Promise<PagedRecordRef> {
    const owned = IndexedConversationUpgradeProgressSchema.parse(progress);
    if (canonicalJsonContentBytes(owned).byteLength > 256 * 1024)
        throw new RangeError('Indexed upgrade progress exceeds its manifest byte bound');
    const value = await stageRecord(store, 'indexed_upgrade_progress', owned.command.operation_id, owned);
    return { content_hash: value.content_hash, size_bytes: value.size_bytes };
}

export async function readIndexedUpgradeProgress(
    store: IndexedConversationRecordStore,
    command: IndexedConversationUpgradeCommand,
    locator: PagedRecordRef,
): Promise<IndexedConversationUpgradeProgress> {
    const ownedCommand = IndexedConversationUpgradeCommandSchema.parse(command);
    const progress = await loadRecord(
        store,
        {
            storage: 'record',
            kind: 'indexed_upgrade_progress',
            id: ownedCommand.operation_id,
            ...locator,
        },
        IndexedConversationUpgradeProgressSchema,
    );
    if (
        (await fingerprintJson(progress.command)) !== (await fingerprintJson(ownedCommand)) ||
        progress.command_fingerprint !== (await fingerprintJson(ownedCommand))
    )
        throw new IndexedConversationUpgradeEvidenceError(
            'Indexed upgrade progress changed its exact command/predecessor',
        );
    return progress;
}

/** Beginning stages no accepted root and creates no reusable processing/readiness proof. */
export async function beginIndexedConversationUpgrade(
    underlying: IndexedConversationRecordStore,
    input: IndexedConversationUpgradeCommand,
): Promise<{ progress: IndexedConversationUpgradeProgress; locator: PagedRecordRef }> {
    const command = IndexedConversationUpgradeCommandSchema.parse(structuredClone(input));
    const { store } = createIndexedUpgradeStepStore(underlying);
    const original = await loadIndexedUpgradePredecessor(store, command);
    // These genuine first-v1 families retain deletion/call originals. A missing legacy witness
    // is not repaired by asserting a profile. Their phased rebuild/audit must finish first.
    const directories = { ...original.directories };
    delete directories.accepted_output_order;
    delete directories.processing_pending;
    delete directories.processing_required;
    delete directories.processing_by_operation;
    delete directories.processing_coverage;
    delete directories.open_tool_calls;
    delete directories.deletion_dependencies;
    delete directories.display_answer_order;
    const progress = IndexedConversationUpgradeProgressSchema.parse({
        version: 1,
        profile: command.profile,
        command,
        command_fingerprint: await fingerprintJson(command),
        step: 0,
        phase: 'receipts',
        directories,
        scratch: {},
        counts: {
            live_turns: 0,
            jobs: 0,
            unresolved_jobs: 0,
            required_jobs: 0,
            required_unresolved_jobs: 0,
            required_blocked_jobs: 0,
            source_records: 0,
            audited_records: 0,
        },
    });
    return { progress, locator: await stageIndexedUpgradeProgress(store, progress) };
}
