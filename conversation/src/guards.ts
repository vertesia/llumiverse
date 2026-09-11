import type { AgentTurn, ConversationTurn, GeneratedAgentTurn, ProgramTurn, ToolTurn, UserTurn } from './types.js';

export function isUserTurn(turn: ConversationTurn): turn is UserTurn {
    return turn.kind === 'user';
}

export function isAgentTurn(turn: ConversationTurn): turn is AgentTurn {
    return turn.kind === 'agent';
}

export function isGeneratedAgentTurn(turn: ConversationTurn): turn is GeneratedAgentTurn {
    return turn.kind === 'agent' && turn.provenance.type === 'generated';
}

export function isToolTurn(turn: ConversationTurn): turn is ToolTurn {
    return turn.kind === 'tool';
}

export function isProgramTurn(turn: ConversationTurn): turn is ProgramTurn {
    return turn.kind === 'program';
}
