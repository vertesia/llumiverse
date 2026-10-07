import {
    applyContextChange,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    planContextChange,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { prepareOpenAIChatCanonicalContext } from '../openai/openai-chat-conversation-adapter.js';

const recordedAt = '2026-09-30T00:00:00.000Z';

function sourceDocument() {
    const document = createConversationDocument({ id: 'conversation:context-edit', created_at: recordedAt });
    document.turns.push(
        createUserTurn({
            id: 'turn:old',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: recordedAt },
            model_visibility: 'include',
            provenance: { type: 'received' },
            blocks: [createTextBlock({ id: 'block:old', text: 'Old material retained', format: 'plain' })],
        }),
        createUserTurn({
            id: 'turn:recent',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: recordedAt },
            model_visibility: 'include',
            provenance: { type: 'received' },
            blocks: [createTextBlock({ id: 'block:recent', text: 'Recent request', format: 'plain' })],
        }),
    );
    document.context.entries = [
        { id: 'context:old', type: 'source_turn', turn_id: 'turn:old' },
        { id: 'context:recent', type: 'source_turn', turn_id: 'turn:recent' },
    ];
    return document;
}

function runtime(conversationId: string, suffix: string) {
    return {
        conversation_id: conversationId,
        request_id: `request:${suffix}`,
        attempt_id: `attempt:${suffix}`,
        input_operation_id: `input:${suffix}`,
        response_operation_id: `response:${suffix}`,
        recorded_at: recordedAt,
        purpose: 'interaction' as const,
    };
}

describe('context change reaches native preparation', () => {
    it('retains source history while the next OpenAI Chat request selects only the edited context', async () => {
        const source = sourceDocument();
        const before = await prepareOpenAIChatCanonicalContext({
            options: { model: 'gpt-test', conversation: source, conversation_runtime: runtime(source.id, 'before') },
            provider: 'openai',
        });
        const plan = await planContextChange(source, {
            expected_revision: source.revision,
            expected_context_revision: source.context.revision,
            entry_ids: ['context:old'],
        });
        const applied = await applyContextChange(source, {
            operation_id: 'change:exclude-old',
            expected_revision: source.revision,
            expected_context_revision: source.context.revision,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: recordedAt,
            entry_ids: ['context:old'],
            proposal: { kind: 'exclude' },
        });
        const after = await prepareOpenAIChatCanonicalContext({
            options: {
                model: 'gpt-test',
                conversation: applied.document,
                conversation_runtime: runtime(source.id, 'after'),
            },
            provider: 'openai',
        });

        expect(applied.document.turns).toEqual(source.turns);
        expect(applied.document.generations).toEqual(source.generations);
        expect(JSON.stringify(before.native_conversation.messages)).toContain('Old material retained');
        expect(JSON.stringify(after.native_conversation.messages)).not.toContain('Old material retained');
        expect(JSON.stringify(after.native_conversation.messages)).toContain('Recent request');
        expect(after.document.revision).toBe(source.revision + 1);
        expect(after.document.operation_receipts['change:exclude-old'].operation_kind).toBe('context_change');
    });
});
