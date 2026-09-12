import {
    type ConversationDocument,
    createConversationDocument,
    createGeneratedAgentTurn,
    createTextBlock,
    createToolTurn,
    createUserTurn,
    type GeneratedAgentTurn,
    type ImportedGeneration,
    type TextBlock,
    type ToolCallBlock,
    type ToolTurn,
    type UserTurn,
} from '../src/index.js';

export const RECORDED_AT = '2026-09-11T00:00:00.000Z';

export function emptyDocument(id = 'conversation'): ConversationDocument {
    return createConversationDocument({ id, created_at: RECORDED_AT });
}

export function textBlock(id: string, text = id): TextBlock {
    return createTextBlock({ id, text, format: 'plain' });
}

export function userTurn(id: string, blockId = `${id}-text`): UserTurn {
    return createUserTurn({
        id,
        authority: 'ordinary',
        blocks: [textBlock(blockId)],
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        provenance: { type: 'received' },
        model_visibility: 'include',
    });
}

export function generatedAgentTurn(
    id: string,
    generationId: string,
    blocks: GeneratedAgentTurn['blocks'] = [textBlock(`${id}-text`)],
): GeneratedAgentTurn {
    return createGeneratedAgentTurn({
        id,
        generation_id: generationId,
        authority: 'ordinary',
        blocks,
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        provenance: { type: 'generated' },
        model_visibility: 'include',
    });
}

export function toolCallBlock(
    id: string,
    callId: string,
    executor: 'application' | 'provider' = 'application',
): ToolCallBlock {
    return {
        id,
        type: 'tool_call',
        call_id: callId,
        tool_name: 'read',
        executor,
        arguments: { type: 'json', value: { path: '/tmp/example' } },
    };
}

export function toolResultTurn(
    id: string,
    callId: string,
    status: 'success' | 'error' | 'cancelled' | 'denied' = 'success',
): ToolTurn {
    return createToolTurn({
        id,
        authority: 'ordinary',
        blocks: [
            {
                id: `${id}-result`,
                type: 'tool_result',
                call_id: callId,
                status,
                content: [textBlock(`${id}-content`, status)],
            },
        ],
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        provenance: { type: 'received' },
        model_visibility: 'include',
    });
}

export function importedGeneration(id: string): ImportedGeneration {
    return {
        id,
        record_source: 'imported',
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        source: { conversation_id: 'conversation', revision: 0 },
        missing_metadata: ['requested_model', 'provider'],
    };
}
