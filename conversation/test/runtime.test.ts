import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    applyContextChange,
    ConversationValidationError,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    parseConversationDocument,
    planContextChange,
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

    it('recovers accepted entries after exclusion and JSON reload without trusting a reused caller fingerprint', async () => {
        const batch = {
            turns: [userTurn('original')],
            context_entries: [
                {
                    id: 'accepted-entry',
                    type: 'source_turn' as const,
                    turn_id: 'turn:user:1',
                    block_ids: ['block:user:1'],
                },
            ],
        };
        const options = {
            expected_revision: 0,
            operation_id: 'accepted',
            payload_fingerprint: 'sha256:caller',
            recorded_at: firstRecordedAt,
        };
        const first = appendConversationRecords(
            createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt }),
            batch,
            options,
        );
        const plan = await planContextChange(first.document, {
            expected_revision: 1,
            expected_context_revision: 1,
            entry_ids: ['accepted-entry'],
        });
        const excluded = await applyContextChange(first.document, {
            operation_id: 'exclude',
            expected_revision: 1,
            expected_context_revision: 1,
            entry_ids: ['accepted-entry'],
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: retryRecordedAt,
            proposal: { kind: 'exclude' },
        });
        const loaded = parseConversationDocument(JSON.parse(JSON.stringify(excluded.document)));
        expect(loaded.context.entries).toEqual([]);
        const retry = appendConversationRecords(loaded, batch, options);
        expect(retry.applied).toBe(false);
        expect(retry.change).toEqual(first.change);
        expect(retry.document).toEqual(loaded);
        expect(() =>
            appendConversationRecords(loaded, { ...batch, turns: [userTurn('conflicting')] }, options),
        ).toThrow('changes accepted turn');
        expect(() =>
            appendConversationRecords(
                loaded,
                { ...batch, context_entries: [{ ...batch.context_entries[0], block_ids: ['another'] }] },
                options,
            ),
        ).toThrow();
        const missing = structuredClone(loaded);
        delete missing.operation_receipts.accepted.accepted_context_entries;
        expect(() => appendConversationRecords(missing, batch, options)).toThrow(
            'cannot resolve accepted context entries',
        );
        const drift = structuredClone(loaded);
        drift.operation_receipts.accepted.accepted_context_entries![0].block_ids = ['missing'];
        expect(() => parseConversationDocument(drift)).toThrow('validation');
        const reordered = structuredClone(loaded);
        reordered.operation_receipts.accepted.accepted_context_entry_ids = ['another'];
        expect(() => parseConversationDocument(reordered)).toThrow('validation');
    });

    it('binds omitted versus explicit active-tool requests without restoring a later selection', () => {
        const options = {
            expected_revision: 0,
            operation_id: 'tools',
            payload_fingerprint: 'sha256:same',
            recorded_at: firstRecordedAt,
        };
        const definition = { id: 'def', name: 'lookup', version: '1', input_schema: { type: 'object' } };
        const batch = { tool_definitions: [definition], active_tool_definition_ids: ['def'] };
        const first = appendConversationRecords(
            createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt }),
            batch,
            options,
        );
        const later = appendConversationRecords(
            first.document,
            { active_tool_definition_ids: [] },
            { ...options, expected_revision: 1, operation_id: 'tools:later' },
        );
        const retry = appendConversationRecords(later.document, batch, options);
        expect(retry.applied).toBe(false);
        expect(retry.document.context.active_tool_definition_ids).toEqual([]);
        expect(retry.change).toEqual(first.change);
        expect(() =>
            appendConversationRecords(later.document, { ...batch, active_tool_definition_ids: [] }, options),
        ).toThrow('changes accepted active tool selection');
        expect(() => appendConversationRecords(later.document, { tool_definitions: [definition] }, options)).toThrow(
            'changes accepted active tool selection',
        );
        const unchanged = appendConversationRecords(
            createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt }),
            {},
            options,
        );
        expect(() =>
            appendConversationRecords(unchanged.document, { active_tool_definition_ids: [] }, options),
        ).toThrow('changes accepted active tool selection');
        const empty = appendConversationRecords(
            createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt }),
            { active_tool_definition_ids: [] },
            options,
        );
        expect(() => appendConversationRecords(empty.document, {}, options)).toThrow(
            'changes accepted active tool selection',
        );
        const historical = structuredClone(first.document);
        delete historical.operation_receipts.tools.accepted_tool_selection;
        // The current catalog happens to match the request, but is not evidence of the old operation's selection.
        expect(historical.context.active_tool_definition_ids).toEqual(batch.active_tool_definition_ids);
        expect(() => appendConversationRecords(historical, batch, options)).toThrow(
            'Historical append receipt cannot prove',
        );
        const historicalAfterLater = structuredClone(later.document);
        delete historicalAfterLater.operation_receipts.tools.accepted_tool_selection;
        expect(() => appendConversationRecords(historicalAfterLater, batch, options)).toThrow(
            'Historical append receipt cannot prove',
        );
        const drift = structuredClone(first.document);
        drift.operation_receipts.tools.accepted_tool_selection = { kind: 'replace', definition_ids: ['missing'] };
        expect(() => parseConversationDocument(drift)).toThrow('validation');
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

    it.each(['timestamps', 'created_at', 'recorded_at'])(
        'rejects changed JSON content under the %s property on retry',
        (property) => {
            const initial = createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt });
            const jsonTurn = (value: string) =>
                createUserTurn({
                    ...userTurn(''),
                    blocks: [{ id: 'block:user:1', type: 'json', value: { records: [{ [property]: value }] } }],
                });
            const options = {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:claimed',
                recorded_at: firstRecordedAt,
            };
            const accepted = appendConversationRecords(initial, { turns: [jsonTurn('original')] }, options);

            expect(() =>
                appendConversationRecords(
                    accepted.document,
                    { turns: [jsonTurn('changed')] },
                    { ...options, recorded_at: retryRecordedAt },
                ),
            ).toThrow('retry changes accepted turn turn:user:1');
        },
    );

    it.each(['timestamps', 'created_at', 'recorded_at'])(
        'rejects changed tool schema properties named %s on retry',
        (property) => {
            const initial = createConversationDocument({ id: 'conversation:1', created_at: firstRecordedAt });
            const definition = (type: string) => ({
                id: 'tool:lookup:v1',
                name: 'lookup',
                version: 'sha256:lookup-v1',
                input_schema: { type: 'object', properties: { [property]: { type } } },
            });
            const options = {
                expected_revision: 0,
                operation_id: 'operation:input:1',
                payload_fingerprint: 'sha256:claimed',
                recorded_at: firstRecordedAt,
            };
            const accepted = appendConversationRecords(initial, { tool_definitions: [definition('string')] }, options);

            expect(() =>
                appendConversationRecords(
                    accepted.document,
                    { tool_definitions: [definition('number')] },
                    { ...options, recorded_at: retryRecordedAt },
                ),
            ).toThrow('retry changes accepted tool definition tool:lookup:v1');
        },
    );

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
