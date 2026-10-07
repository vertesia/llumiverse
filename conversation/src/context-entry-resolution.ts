import type { ContentBlock, ContextEntry, ConversationDocument, ConversationTurn } from './types.js';

/** Ephemeral index of canonical records; never serialized as another conversation representation. */
export function createContextTurnIndex(document: ConversationDocument): ReadonlyMap<string, ConversationTurn> {
    return new Map([
        ...document.turns.map((turn) => [turn.id, turn] as const),
        ...Object.values(document.compactions).flatMap((record) =>
            record.replacement_turns.map((turn) => [turn.id, turn] as const),
        ),
    ]);
}

export function resolveContextEntry(
    turns: ReadonlyMap<string, ConversationTurn>,
    entry: ContextEntry,
): { turn: ConversationTurn; blocks: ContentBlock[] } {
    const turn = turns.get(entry.turn_id);
    if (!turn) throw new Error(`Context entry ${entry.id} references unavailable turn ${entry.turn_id}`);
    const ids = entry.block_ids === undefined ? undefined : new Set(entry.block_ids);
    return { turn, blocks: turn.blocks.filter((block) => ids === undefined || ids.has(block.id)) };
}
