import { isGeneratedAgentTurn } from './guards.js';
import { fingerprintJson } from './runtime.js';
import type { ApplicationToolCallBlock, ConversationDocument, PendingApplicationToolCall } from './types.js';

export interface AcceptedToolUseIdentity {
    id: string;
    tool_name: string;
}

export interface AcceptedExecutedResponse {
    operation_id: string;
    result_revision: number;
    turn: Extract<ConversationDocument['turns'][number], { kind: 'agent'; provenance: { type: 'generated' } }>;
}

/** Select one accepted application response without treating unrelated imported receipts as execution provenance. */
export function selectAcceptedExecutedResponse(
    document: ConversationDocument,
    acceptedToolUses?: readonly AcceptedToolUseIdentity[],
): AcceptedExecutedResponse {
    const candidates = Object.values(document.operation_receipts).flatMap((receipt) => {
        if (receipt.accepted_turn_ids?.length !== 1 || receipt.accepted_generation_ids?.length !== 1) return [];
        const turn = document.turns.find((candidate) => candidate.id === receipt.accepted_turn_ids?.[0]);
        const generationId = receipt.accepted_generation_ids[0];
        const generation = Object.hasOwn(document.generations, generationId)
            ? document.generations[generationId]
            : undefined;
        if (!turn || !isGeneratedAgentTurn(turn) || turn.generation_id !== generationId) return [];
        const calls = turn.blocks.filter(
            (block): block is ApplicationToolCallBlock =>
                block.type === 'tool_call' && block.executor === 'application',
        );
        // Imported generations can have operation receipts but never authorize application execution.
        if (generation?.record_source !== 'executed') return [];
        return [{ operation_id: receipt.id, receipt, turn, generation, calls }];
    });
    const latestRevision = Math.max(...candidates.map(({ receipt }) => receipt.result_revision));
    const selected = candidates.filter(({ receipt }) => receipt.result_revision === latestRevision);
    if (selected.length !== 1) {
        throw new Error('Accepted canonical tool calls do not identify one executed response operation');
    }
    const [{ operation_id, receipt, turn, generation, calls }] = selected;
    if (
        turn.status !== 'completed' ||
        generation.status !== 'completed' ||
        generation.source.conversation_id !== document.id ||
        generation.source.revision !== receipt.base_revision
    ) {
        throw new Error('Accepted canonical response receipt has inconsistent generation provenance');
    }
    if (
        acceptedToolUses !== undefined &&
        (calls.length !== acceptedToolUses.length ||
            calls.some(
                (call, index) =>
                    call.call_id !== acceptedToolUses[index]?.id ||
                    call.tool_name !== acceptedToolUses[index]?.tool_name,
            ))
    ) {
        throw new Error('Accepted tool-use mirror does not match the latest canonical response operation');
    }
    return { operation_id, result_revision: receipt.result_revision, turn };
}

/** Derive runnable calls only from the exact accepted executed response named by the host result boundary. */
export async function derivePendingApplicationToolCalls(
    document: ConversationDocument,
    acceptedToolUses?: readonly AcceptedToolUseIdentity[],
): Promise<PendingApplicationToolCall[]> {
    const completedCalls = new Set(Object.values(document.execution_receipts).map((receipt) => receipt.call_id));
    for (const turn of document.turns) {
        for (const block of turn.blocks) {
            if (block.type === 'tool_result') completedCalls.add(block.call_id);
        }
    }
    const selected = selectAcceptedExecutedResponse(document, acceptedToolUses);
    const pending: PendingApplicationToolCall[] = [];
    for (const block of selected.turn.blocks) {
        if (block.type !== 'tool_call' || block.executor !== 'application' || completedCalls.has(block.call_id))
            continue;
        if (block.arguments.type === 'invalid') {
            throw new Error(`Accepted canonical tool call ${block.call_id} has invalid arguments`);
        }
        pending.push({
            source: {
                conversation: { conversation_id: document.id, revision: document.revision },
                turn_id: selected.turn.id,
                block_id: block.id,
                call_id: block.call_id,
                call_fingerprint: await fingerprintJson(block),
            },
            call: {
                call_id: block.call_id,
                tool_name: block.tool_name,
                ...(block.definition_id ? { definition_id: block.definition_id } : {}),
                executor: 'application',
            },
        });
    }
    return pending;
}
