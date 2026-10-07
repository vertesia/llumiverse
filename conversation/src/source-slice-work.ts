import { ConversationValidationError } from './diagnostics.js';
import { DEFAULT_JSON_INPUT_LIMITS } from './json-preflight.js';

/** Ephemeral shared budget; never persisted as conversation state or caller-supplied authority. */
export interface SourceSliceWork {
    nodes: number;
}
export function rejectSourceSlice(message: string): never {
    throw new ConversationValidationError(message, [
        { code: 'DERIVED_PROVENANCE_MISMATCH', stage: 'semantic', path: '/', message },
    ]);
}
export function spendSourceSliceWork(work: SourceSliceWork, amount = 1): void {
    work.nodes += amount;
    if (work.nodes > DEFAULT_JSON_INPUT_LIMITS.max_nodes) {
        rejectSourceSlice('Source-slice lineage exceeds bounded materialized work');
    }
}

export function rejectUnsupportedSliceFormat(): never {
    const message = 'Partial text mutation requires a registered format boundary validator';
    throw new ConversationValidationError(message, [
        { code: 'SELECTION_FORMAT_UNSUPPORTED', stage: 'semantic', path: '/', message },
    ]);
}
