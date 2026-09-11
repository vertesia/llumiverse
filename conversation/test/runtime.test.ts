import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    ConversationValidationError,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
} from '../src/index.js';

const firstRecordedAt = '2026-09-11T00:00:00.000Z';
const retryRecordedAt = '2026-09-11T00:01:00.000Z';

function userTurn(text: string, recordedAt = firstRecordedAt) {
    return createUserTurn({
        id: 'turn:user:1',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: recordedAt },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [createTextBlock({ id: 'block:user:1', text, format: 'plain' })],
    });
}

describe('canonical ingestion runtime', () => {
    it('recovers an old accepted operation after later context/tool changes and fresh retry timestamps', () => {
        const initial = createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt });
        const first = appendConversationRecords(
            initial,
            {
                turns: [userTurn('hello')],
                context_entries: [{ id: 'context:user:1', type: 'source_turn', turn_id: 'turn:user:1' }],
                active_tool_definition_ids: [],
            },
            {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:first',
                recorded_at: firstRecordedAt,
            },
        );
        const advanced = appendConversationRecords(
            first.document,
            {
                tool_definitions: [
                    {
                        id: 'tool:lookup:v1',
                        name: 'lookup',
                        version: 'sha256:lookup-v1',
                        input_schema: { type: 'object' },
                    },
                ],
                active_tool_definition_ids: ['tool:lookup:v1'],
            },
            {
                expected_revision: 1,
                operation_id: 'operation:input:2',
                payload_fingerprint: 'sha256:second',
                recorded_at: retryRecordedAt,
            },
        );

        const retried = appendConversationRecords(
            advanced.document,
            {
                turns: [userTurn('hello', retryRecordedAt)],
                context_entries: [{ id: 'context:user:1', type: 'source_turn', turn_id: 'turn:user:1' }],
                active_tool_definition_ids: [],
            },
            {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:first',
                recorded_at: retryRecordedAt,
            },
        );

        expect(retried.applied).toBe(false);
        expect(retried.document).toEqual(advanced.document);
        expect(retried.document.context.active_tool_definition_ids).toEqual(['tool:lookup:v1']);
        expect(retried.accepted_turn_ids).toEqual(['turn:user:1']);
    });

    it('rejects changed semantic records even when IDs and a caller fingerprint are reused', () => {
        const initial = createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt });
        const accepted = appendConversationRecords(
            initial,
            { turns: [userTurn('original')] },
            {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:claimed',
                recorded_at: firstRecordedAt,
            },
        );

        expect(() =>
            appendConversationRecords(
                accepted.document,
                { turns: [userTurn('changed')] },
                {
                    expected_revision: 0,
                    operation_id: 'operation:input:1',
                    payload_fingerprint: 'sha256:claimed',
                    recorded_at: retryRecordedAt,
                },
            ),
        ).toThrow('retry changes accepted turn turn:user:1');
    });

    it('preflights accessors before reading them and schema-validates strict options on retries', () => {
        const initial = createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt });
        let getterReads = 0;
        const batch = Object.defineProperty({}, 'turns', {
            enumerable: true,
            get() {
                getterReads += 1;
                return [];
            },
        });

        expect(() =>
            appendConversationRecords(initial, batch as never, {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:first',
                recorded_at: firstRecordedAt,
            }),
        ).toThrow(ConversationValidationError);
        expect(getterReads).toBe(0);

        const accepted = appendConversationRecords(
            initial,
            {},
            {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:first',
                recorded_at: firstRecordedAt,
            },
        );
        expect(() =>
            appendConversationRecords(accepted.document, {}, {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:first',
                recorded_at: retryRecordedAt,
                extra: true,
            } as never),
        ).toThrow();
    });
});
