import type { ResolvedConversationRuntimeContext } from '@llumiverse/conversation';
import { createConversationDocument, setProcessingPolicy } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { appendCanonicalPrompt, type CanonicalPromptRecords } from './canonical-runtime.js';

const at = '2026-10-02T00:00:00.000Z';
const runtime: ResolvedConversationRuntimeContext = {
    conversation_id: 'ingress:processing',
    request_id: 'request:ingress',
    attempt_id: 'attempt:ingress',
    input_operation_id: 'input:ingress',
    response_operation_id: 'response:ingress',
    recorded_at: at,
    started_at: at,
    purpose: 'interaction',
};
const records: CanonicalPromptRecords = {
    turns: [
        {
            id: 'turn:ingress',
            kind: 'user',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            model_visibility: 'include',
            provenance: { type: 'received' },
            blocks: [{ id: 'block:ingress', type: 'text', format: 'plain', text: 'durably accepted input' }],
        },
    ],
    context_entries: [{ id: 'entry:ingress', type: 'source_turn', turn_id: 'turn:ingress' }],
    assets: [],
    item_mappings: [],
};

describe('processing-aware canonical provider ingress', () => {
    it('stages accepted input and one durable job without executing a processor or inference', async () => {
        const policy = await setProcessingPolicy(
            createConversationDocument({ id: runtime.conversation_id, created_at: at }),
            {
                operation_id: 'policy:ingress',
                expected_revision: 0,
                recorded_at: at,
                enabled: true,
                processors: [
                    {
                        id: 'processor:ingress',
                        version: 'v1',
                        scope: 'on_append',
                        required: true,
                        failure_behavior: 'block',
                        config: {},
                    },
                ],
            },
        );
        const accepted = await appendCanonicalPrompt(policy.document, records, runtime, undefined, {
            text: 'same input',
        });
        expect(accepted.document.turns).toEqual(records.turns);
        expect(accepted.document.context.entries).toEqual(records.context_entries);
        const jobs = Object.values(accepted.document.processing.jobs ?? {});
        expect(jobs).toHaveLength(1);
        expect(jobs[0]).toMatchObject({
            source_operation_id: runtime.input_operation_id,
            processor_id: 'processor:ingress',
            selection: { kind: 'entries', entry_ids: ['entry:ingress'] },
        });
        expect(accepted.document.processing.resolved_inputs).toBeUndefined();
        expect(accepted.document.processing.attempts).toBeUndefined();
        expect(accepted.document.processing.outputs).toBeUndefined();
        expect(accepted.document.generations).toEqual({});
        const retried = await appendCanonicalPrompt(
            JSON.parse(JSON.stringify(accepted.document)),
            records,
            runtime,
            undefined,
            { text: 'same input' },
        );
        expect(retried).toEqual(accepted);
        await expect(
            appendCanonicalPrompt(accepted.document, records, runtime, undefined, { text: 'changed input' }),
        ).rejects.toThrow('already used with a different payload');
    });
});
