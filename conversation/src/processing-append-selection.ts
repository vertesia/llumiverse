import { resolveContextEntry } from './context-entry-resolution.js';
import type { ContextEntry, ConversationTurn } from './types.js';

/** Selection-only adapter: all entries/turns supplied here must already be validated and owned.
 * This is not a partial ConversationDocument and establishes no dependency or publication authority.
 */
export function eligibleProcessingAppendRecords(
    entries: readonly ContextEntry[],
    turns: ReadonlyMap<string, ConversationTurn>,
    acceptedEntryIds: readonly string[],
    protectedEntryIds: readonly string[],
    textOnly = false,
) {
    const accepted = new Set(acceptedEntryIds);
    const protectedIds = new Set(protectedEntryIds);
    const entryIds: string[] = [];
    const selectedEntries: ContextEntry[] = [];
    const selectedBlockIds: Record<string, string[]> = {};
    let partial = false;
    for (const entry of entries) {
        if (!accepted.has(entry.id) || protectedIds.has(entry.id)) continue;
        const { turn, blocks } = resolveContextEntry(turns, entry);
        if (
            turn.status !== 'completed' ||
            turn.kind === 'program' ||
            turn.kind === 'tool' ||
            turn.authority !== 'ordinary'
        )
            continue;
        const eligible = blocks.filter((block) => block.type === 'text' || (!textOnly && block.type === 'json'));
        if (!eligible.length) continue;
        entryIds.push(entry.id);
        selectedEntries.push(entry);
        if (eligible.length !== blocks.length) {
            Object.defineProperty(selectedBlockIds, entry.id, {
                value: eligible.map((block) => block.id),
                enumerable: true,
            });
            partial = true;
        }
    }
    return {
        entryIds,
        selectedBlockIds: partial ? selectedBlockIds : undefined,
        selectedEntries: partial ? selectedEntries : undefined,
    };
}
