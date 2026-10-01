import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    assertAcceptedResponseMatchesPreparedRequest,
    type ConversationPreparedRequest,
    createConversationDocument,
    deriveConversationId,
    parseConversationPreparedRequest,
} from '../src/index.js';

const RECORDED_AT = '2026-09-30T00:00:00.000Z';

async function preparedRequest(): Promise<ConversationPreparedRequest> {
    const document = appendConversationRecords(
        createConversationDocument({ id: 'conversation-1', created_at: RECORDED_AT }),
        {
            turns: [
                {
                    id: 'user-turn',
                    kind: 'user',
                    authority: 'ordinary',
                    blocks: [{ id: 'user-text', type: 'text', text: 'hello', format: 'plain' }],
                    status: 'completed',
                    timestamps: { recorded_at: RECORDED_AT },
                    provenance: { type: 'received' },
                    model_visibility: 'include',
                },
            ],
            context_entries: [{ id: 'user-context', type: 'source_turn', turn_id: 'user-turn' }],
        },
        {
            expected_revision: 0,
            operation_id: 'input-operation',
            payload_fingerprint: 'sha256:input',
            recorded_at: RECORDED_AT,
        },
    ).document;
    const runtime = {
        conversation_id: document.id,
        request_id: 'request-1',
        attempt_id: 'attempt-1',
        input_operation_id: 'input-operation',
        response_operation_id: 'response-operation',
        recorded_at: RECORDED_AT,
        purpose: 'conversation',
    };
    return {
        document,
        record: {
            source: { conversation_id: document.id, revision: document.revision },
            runtime,
            request_receipt: {
                id: await deriveConversationId('request_receipt', runtime.request_id, runtime.attempt_id),
                request_id: runtime.request_id,
                attempt_id: runtime.attempt_id,
                source: { conversation_id: document.id, revision: document.revision },
                source_tail_turn_id: 'user-turn',
                context_fingerprint: 'sha256:context',
                tool_set_fingerprint: 'sha256:tools',
                request_fingerprint: 'sha256:request',
                target: {
                    provider: 'test',
                    protocol: 'test.generate',
                    model: 'test-model',
                    adapter_version: '1',
                },
                tool_definition_ids: [],
                asset_versions: [],
                item_mappings: [
                    { canonical_id: 'user-turn', native_id: 'contents/0', kind: 'turn' },
                    { canonical_id: 'user-text', native_id: 'contents/0/parts/0', kind: 'block' },
                ],
                recorded_at: RECORDED_AT,
            },
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        },
    };
}

describe('prepared canonical request evidence', () => {
    it('parses exact pre-provider request evidence without mutating the caller document', async () => {
        const prepared = await preparedRequest();
        const before = structuredClone(prepared);

        await expect(parseConversationPreparedRequest(prepared)).resolves.toEqual(prepared);
        expect(prepared).toEqual(before);
    });

    it.each([
        [
            'source revision',
            (value: ConversationPreparedRequest) => {
                value.record.source.revision = 0;
            },
        ],
        [
            'request identity',
            (value: ConversationPreparedRequest) => {
                value.record.runtime.request_id = 'changed';
            },
        ],
        [
            'selected mapping',
            (value: ConversationPreparedRequest) => {
                value.record.request_receipt.item_mappings[0].canonical_id = 'missing-turn';
            },
        ],
    ])('rejects changed %s before provider transport', async (_label, mutate) => {
        const prepared = await preparedRequest();
        mutate(prepared);
        await expect(parseConversationPreparedRequest(prepared)).rejects.toThrow();
    });

    it('binds an accepted response to the exact prepared receipt and response identities', async () => {
        const prepared = await preparedRequest();
        const generation = {
            id: prepared.record.generation_id,
            record_source: 'executed' as const,
            request_id: prepared.record.runtime.request_id,
            attempt_id: prepared.record.runtime.attempt_id,
            purpose: prepared.record.runtime.purpose,
            requested_model: 'test-model',
            provider: 'test',
            protocol: 'test.generate',
            adapter_version: '1',
            status: 'completed' as const,
            timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
            source: prepared.record.source,
            request_receipt: prepared.record.request_receipt,
        };
        const turn = {
            id: prepared.record.response_turn_id,
            kind: 'agent' as const,
            authority: 'ordinary' as const,
            blocks: [{ id: 'answer', type: 'text' as const, text: 'done', format: 'plain' as const }],
            status: 'completed' as const,
            timestamps: { recorded_at: RECORDED_AT },
            generation_id: generation.id,
            provenance: { type: 'generated' as const },
            model_visibility: 'include' as const,
        };
        const accepted = appendConversationRecords(
            prepared.document,
            { turns: [turn], generations: [generation] },
            {
                expected_revision: prepared.document.revision,
                operation_id: prepared.record.runtime.response_operation_id,
                payload_fingerprint: 'sha256:response',
                recorded_at: RECORDED_AT,
            },
        ).document;

        await expect(assertAcceptedResponseMatchesPreparedRequest(accepted, prepared)).resolves.toEqual({
            generation,
            turn,
        });
        const changed = structuredClone(accepted);
        const changedGeneration = changed.generations[generation.id];
        if (changedGeneration?.record_source !== 'executed') throw new Error('Expected executed generation fixture');
        changedGeneration.request_receipt.request_fingerprint = 'sha256:changed';
        await expect(assertAcceptedResponseMatchesPreparedRequest(changed, prepared)).rejects.toThrow(
            'durably prepared request',
        );
    });
});
