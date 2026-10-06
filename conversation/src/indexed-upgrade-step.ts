import type { IndexedConversationRecordStore } from './indexed-conversation.js';
import { auditIndexedUpgradeActiveWindow } from './indexed-upgrade-active.js';
import { advanceIndexedUpgradeBlock } from './indexed-upgrade-blocks.js';
import { advanceIndexedUpgradeCallAudit } from './indexed-upgrade-call-audit.js';
import { advanceIndexedUpgradeDependencies } from './indexed-upgrade-dependencies.js';
import { advanceIndexedUpgradeExecution } from './indexed-upgrade-executions.js';
import {
    createIndexedUpgradeStepStore,
    INDEXED_UPGRADE_ACTIVE_LIMITS,
    type IndexedUpgradeStepIo,
} from './indexed-upgrade-io.js';
import { advanceIndexedUpgradeJobAudit } from './indexed-upgrade-job-audit.js';
import { advanceIndexedUpgradeOpenCallAudit } from './indexed-upgrade-open-call-audit.js';
import { advanceIndexedUpgradeOrder } from './indexed-upgrade-order.js';
import { advanceIndexedUpgradePhaseAudit } from './indexed-upgrade-phase-audit.js';
import { advanceIndexedUpgradeProcessing } from './indexed-upgrade-processing.js';
import {
    IndexedConversationUpgradeEvidenceError,
    loadIndexedUpgradePredecessor,
    readIndexedUpgradeProgress,
    stageIndexedUpgradeProgress,
} from './indexed-upgrade-progress.js';
import { advanceIndexedUpgradeReceipt } from './indexed-upgrade-receipts.js';
import { advanceIndexedUpgradeRecordAudit } from './indexed-upgrade-record-audit.js';
import { advanceIndexedUpgradeTurn } from './indexed-upgrade-turns.js';
import type { PagedRecordRef } from './paged-record-index.js';
import type {
    IndexedConversationUpgradeCommand,
    IndexedConversationUpgradeProgress,
} from './schemas/indexed-upgrade.js';

/** The authenticated host owns the current progress nomination; callers cannot skip cursors/phases.
 * Every nested read, staging write and readback spends the same finite step budget.
 */
export async function advanceIndexedConversationUpgrade(
    underlying: IndexedConversationRecordStore,
    command: IndexedConversationUpgradeCommand,
    previousLocator: PagedRecordRef,
): Promise<{
    progress: IndexedConversationUpgradeProgress;
    locator: PagedRecordRef;
    usage: Readonly<IndexedUpgradeStepIo>;
}> {
    // The manifest is always first read under the ordinary bound; only its owned phase can
    // nominate the separately bounded active-window reader.
    const initial = createIndexedUpgradeStepStore(underlying);
    const nominated = await readIndexedUpgradeProgress(initial.store, command, previousLocator);
    const { store, usage } =
        nominated.phase === 'active_window'
            ? createIndexedUpgradeStepStore(underlying, INDEXED_UPGRADE_ACTIVE_LIMITS, initial.usage)
            : initial;
    const root = await loadIndexedUpgradePredecessor(store, command);
    const previous = nominated;
    let next: IndexedConversationUpgradeProgress;
    switch (previous.phase) {
        case 'receipts':
            next = await advanceIndexedUpgradeReceipt(store, root, previous);
            break;
        case 'executions':
            next = await advanceIndexedUpgradeExecution(store, root, previous);
            break;
        case 'turns':
            next = await advanceIndexedUpgradeTurn(store, root, previous);
            break;
        case 'blocks':
            next = await advanceIndexedUpgradeBlock(store, root, previous);
            break;
        case 'turn_order':
            next = await advanceIndexedUpgradeOrder(store, root, previous);
            break;
        case 'processing':
            next = await advanceIndexedUpgradeProcessing(store, root, previous);
            break;
        case 'audit':
            if (previous.audit_family === undefined)
                throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade reverse-directory phase is invalid');
            if (previous.audit_family === 0) next = await advanceIndexedUpgradeJobAudit(store, root, previous);
            else if (previous.audit_family === 1) next = await advanceIndexedUpgradeCallAudit(store, root, previous);
            else if (previous.audit_family === 2)
                next = await advanceIndexedUpgradeOpenCallAudit(store, root, previous);
            else if (previous.audit_family >= 3 && previous.audit_family <= 8)
                next = await advanceIndexedUpgradeDependencies(store, root, previous);
            else if (previous.audit_family === 9) next = await advanceIndexedUpgradePhaseAudit(store, root, previous);
            else if (previous.audit_family >= 10 && previous.audit_family <= 17)
                next = await advanceIndexedUpgradeRecordAudit(store, root, previous);
            else
                throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade reverse-directory phase is invalid');
            break;
        case 'active_window':
            next = await auditIndexedUpgradeActiveWindow(store, root, previous);
            break;
        case 'complete':
            return { progress: previous, locator: previousLocator, usage };
    }
    if (previous.step >= Number.MAX_SAFE_INTEGER)
        throw new IndexedConversationUpgradeEvidenceError('Indexed upgrade progress step overflows its exact identity');
    next = { ...next, step: previous.step + 1, predecessor_progress: previousLocator };
    return { progress: next, locator: await stageIndexedUpgradeProgress(store, next), usage };
}
