import type {
    ConversationDocument,
    ConversationInspection,
    ConversationTurn,
    Generation,
    TurnKindCounts,
} from './types.js';

export interface ResolvedConversationTurn {
    turn: ConversationTurn;
    source: 'history' | 'compaction';
    compaction_id?: string;
    generation?: Generation;
}

function emptyKindCounts(): TurnKindCounts {
    return { user: 0, agent: 0, tool: 0, program: 0 };
}

function findGeneration(document: ConversationDocument, turn: ConversationTurn): Generation | undefined {
    if (turn.kind !== 'agent' || !('generation_id' in turn) || turn.generation_id === undefined) {
        return undefined;
    }
    return Object.hasOwn(document.generations, turn.generation_id)
        ? document.generations[turn.generation_id]
        : undefined;
}

function indexConversationTurns(document: ConversationDocument): Map<string, ResolvedConversationTurn> {
    const turns = new Map<string, ResolvedConversationTurn>();
    for (const turn of document.turns) {
        if (!turns.has(turn.id)) {
            turns.set(turn.id, {
                turn,
                source: 'history',
                generation: findGeneration(document, turn),
            });
        }
    }
    for (const compaction of Object.values(document.compactions)) {
        for (const turn of compaction.replacement_turns) {
            if (!turns.has(turn.id)) {
                turns.set(turn.id, {
                    turn,
                    source: 'compaction',
                    compaction_id: compaction.id,
                    generation: findGeneration(document, turn),
                });
            }
        }
    }
    return turns;
}

export function getConversationTurn(
    document: ConversationDocument,
    turnId: string,
): ResolvedConversationTurn | undefined {
    for (const turn of document.turns) {
        if (turn.id === turnId) {
            return {
                turn,
                source: 'history',
                generation: findGeneration(document, turn),
            };
        }
    }
    for (const compaction of Object.values(document.compactions)) {
        for (const turn of compaction.replacement_turns) {
            if (turn.id === turnId) {
                return {
                    turn,
                    source: 'compaction',
                    compaction_id: compaction.id,
                    generation: findGeneration(document, turn),
                };
            }
        }
    }
    return undefined;
}

export function getPendingToolCallIds(document: ConversationDocument): string[] {
    const resolved = new Set<string>();
    for (const turn of document.turns) {
        if (turn.kind === 'tool') {
            resolved.add(turn.blocks[0].call_id);
        }
    }
    for (const receipt of Object.values(document.execution_receipts)) {
        resolved.add(receipt.call_id);
    }

    const pending: string[] = [];
    const seen = new Set<string>();
    for (const turn of document.turns) {
        if (turn.kind !== 'agent') {
            continue;
        }
        for (const block of turn.blocks) {
            if (
                block.type === 'tool_call' &&
                block.executor === 'application' &&
                !resolved.has(block.call_id) &&
                !seen.has(block.call_id)
            ) {
                pending.push(block.call_id);
                seen.add(block.call_id);
            }
        }
    }
    return pending;
}

export function inspectConversation(document: ConversationDocument): ConversationInspection {
    const sourceTurnsByKind = emptyKindCounts();
    for (const turn of document.turns) {
        sourceTurnsByKind[turn.kind] += 1;
    }

    const turnsById = indexConversationTurns(document);
    const contextTurns = new Map<string, ConversationTurn>();
    for (const entry of document.context.entries) {
        const resolved = turnsById.get(entry.turn_id);
        if (resolved !== undefined && !contextTurns.has(entry.turn_id)) {
            contextTurns.set(entry.turn_id, resolved.turn);
        }
    }
    const contextTurnsByKind = emptyKindCounts();
    for (const turn of contextTurns.values()) {
        contextTurnsByKind[turn.kind] += 1;
    }

    return {
        source_turn_count: document.turns.length,
        context_turn_count: contextTurns.size,
        source_turns_by_kind: sourceTurnsByKind,
        context_turns_by_kind: contextTurnsByKind,
        generation_count: Object.keys(document.generations).length,
        pending_tool_call_ids: getPendingToolCallIds(document),
    };
}
