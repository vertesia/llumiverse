import { fingerprintJson } from './identity.js';
import { rejectSourceSlice } from './source-slice-work.js';
import type { ConversationSliceEditOperation } from './types.js';

/** Immutable source ordering, captured by the edit publisher rather than supplied by its caller. */
export function assertSliceEditTopology(operation: ConversationSliceEditOperation): void {
    const positions = operation.source_entry_positions;
    if (positions.length !== operation.selected_entries.length || positions.length === 0) {
        rejectSourceSlice('Slice topology positions must bind every selected source entry');
    }
    for (let index = 0; index < positions.length; index++) {
        if (
            !Number.isSafeInteger(positions[index]) ||
            positions[index] < 0 ||
            (index > 0 && positions[index] <= positions[index - 1])
        ) {
            rejectSourceSlice('Slice topology positions must be strictly increasing safe source ordinals');
        }
    }
}
export function fingerprintSliceEditTopology(
    operation: Pick<
        ConversationSliceEditOperation,
        'source' | 'source_context_revision' | 'source_fingerprint' | 'selected_entries' | 'source_entry_positions'
    >,
): Promise<string> {
    return fingerprintJson({
        domain: 'llumiverse.conversation.slice-topology',
        version: 1,
        source: operation.source,
        source_context_revision: operation.source_context_revision,
        source_fingerprint: operation.source_fingerprint,
        selected_entries: operation.selected_entries,
        source_entry_positions: operation.source_entry_positions,
    });
}
