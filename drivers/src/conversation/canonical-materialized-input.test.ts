import {
    appendConversationRecords,
    type ConversationDocument,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    appendCanonicalPrompt,
    type CanonicalPromptRecords,
    canonicalToolDefinitions,
    type ResolvedConversationRuntimeContext,
} from './canonical-runtime.js';

const RECORDED_AT = '2026-09-30T00:00:00.000Z';
const INPUT_OPERATION_ID = 'operation:materialized-input';

const emptyRecords = (): CanonicalPromptRecords => ({
    turns: [],
    assets: [],
    context_entries: [],
    item_mappings: [],
});

function runtime(document: ConversationDocument): ResolvedConversationRuntimeContext {
    return {
        conversation_id: document.id,
        request_id: 'request:next-model-call',
        attempt_id: 'attempt:next-model-call',
        input_operation_id: 'operation:unused-input',
        response_operation_id: 'operation:next-response',
        recorded_at: RECORDED_AT,
        purpose: 'interaction',
        materialized_input: {
            operation_id: INPUT_OPERATION_ID,
            result_revision: document.revision,
        },
    };
}

function materializedDocument(): ConversationDocument {
    const initial = createConversationDocument({ id: 'conversation:materialized', created_at: RECORDED_AT });
    const turn = createUserTurn({
        id: 'turn:materialized-input',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [
            createTextBlock({ id: 'block:materialized-a', text: 'first', format: 'plain' }),
            createTextBlock({ id: 'block:materialized-b', text: 'second', format: 'plain' }),
        ],
    });
    return appendConversationRecords(
        initial,
        {
            turns: [turn],
            context_entries: [{ id: 'context:materialized-input', type: 'source_turn', turn_id: turn.id }],
            active_tool_definition_ids: [],
        },
        {
            expected_revision: initial.revision,
            operation_id: INPUT_OPERATION_ID,
            payload_fingerprint: 'sha256:materialized-input',
            recorded_at: RECORDED_AT,
        },
    ).document;
}

describe('materialized canonical input proof', () => {
    it('reuses an exactly accepted current-head input without appending it again', async () => {
        const document = materializedDocument();
        const options = runtime(document);

        const first = await appendCanonicalPrompt(document, emptyRecords(), options, [], { prompt: [] });
        const retried = await appendCanonicalPrompt(document, emptyRecords(), options, [], { prompt: [] });

        expect(first.document).toBe(document);
        expect(retried.document).toBe(document);
        expect(first.document.revision).toBe(1);
        expect(first.document.turns).toHaveLength(1);
        expect(first.tool_definitions).toEqual([]);
    });

    it('rejects a proof for the wrong operation or revision', async () => {
        const document = materializedDocument();
        const wrongOperation = runtime(document);
        if (wrongOperation.materialized_input === undefined) throw new Error('Expected materialized proof');
        wrongOperation.materialized_input.operation_id = 'operation:unknown';
        await expect(appendCanonicalPrompt(document, emptyRecords(), wrongOperation, [], null)).rejects.toThrow(
            /accepted input-only operation receipt/,
        );

        const wrongRevision = runtime(document);
        if (wrongRevision.materialized_input === undefined) throw new Error('Expected materialized proof');
        wrongRevision.materialized_input.result_revision -= 1;
        await expect(appendCanonicalPrompt(document, emptyRecords(), wrongRevision, [], null)).rejects.toThrow(
            /accepted input-only operation receipt/,
        );
    });

    it('rejects new prompt records alongside an already materialized input', async () => {
        const document = materializedDocument();
        const records = emptyRecords();
        const retainedTurn = document.turns[0];
        if (retainedTurn === undefined) throw new Error('Expected a materialized input turn');
        records.turns.push(retainedTurn);

        await expect(appendCanonicalPrompt(document, records, runtime(document), [], null)).rejects.toThrow(
            /cannot include new prompt records/,
        );
    });

    it('requires the accepted context identity and complete turn selection', async () => {
        const document = materializedDocument();
        const missingAcceptedContext = structuredClone(document);
        const receipt = missingAcceptedContext.operation_receipts[INPUT_OPERATION_ID];
        if (receipt === undefined) throw new Error('Expected input receipt');
        receipt.accepted_context_entry_ids = [];
        await expect(
            appendCanonicalPrompt(missingAcceptedContext, emptyRecords(), runtime(missingAcceptedContext), [], null),
        ).rejects.toThrow(/not selected by an accepted context entry/);

        const partial = structuredClone(document);
        const [entry] = partial.context.entries;
        if (entry?.type !== 'source_turn') throw new Error('Expected source-turn context entry');
        entry.block_ids = ['block:materialized-a'];
        await expect(appendCanonicalPrompt(partial, emptyRecords(), runtime(partial), [], null)).rejects.toThrow(
            /only partially selected/,
        );
    });

    it('records and exactly retries an explicit tool-set change after the materialized input', async () => {
        const definitions = await canonicalToolDefinitions([
            { name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } },
        ]);
        const original = materializedDocument();
        const proof = runtime(original);
        const tools = [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' as const } }];

        const first = await appendCanonicalPrompt(original, emptyRecords(), proof, tools, null);
        expect(first.document.revision).toBe(original.revision + 1);
        expect(first.document.context.active_tool_definition_ids).toEqual(
            definitions.map((definition) => definition.id),
        );
        expect(first.document.operation_receipts[proof.input_operation_id]).toMatchObject({
            base_revision: original.revision,
            result_revision: first.document.revision,
            accepted_tool_definition_ids: definitions.map((definition) => definition.id),
            accepted_turn_ids: [],
        });

        const retried = await appendCanonicalPrompt(first.document, emptyRecords(), proof, tools, null);
        expect(retried.document).toEqual(first.document);
        expect(retried.document.revision).toBe(first.document.revision);
    });
});
