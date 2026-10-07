import { type ConversationTurn, isGeneratedAgentTurn } from '@llumiverse/conversation';

export function inspectNarrowedTurn(turn: ConversationTurn): string {
    if (turn.kind === 'tool') {
        return `${turn.blocks[0].call_id}:${turn.blocks[0].status}`;
    }
    if (turn.kind === 'user') {
        return turn.provenance.type;
    }
    if (turn.kind === 'program') {
        return turn.authority;
    }
    if (isGeneratedAgentTurn(turn)) {
        return turn.generation_id;
    }
    return turn.provenance.type;
}
