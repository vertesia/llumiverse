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
    prepareCanonicalContext,
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

function activeToolDocument(): ConversationDocument {
    const initial = createConversationDocument({ id: 'conversation:active-tools', created_at: RECORDED_AT });
    return appendConversationRecords(
        initial,
        {
            tool_definitions: [
                {
                    id: 'tool-definition:lookup:v1',
                    name: 'lookup',
                    version: 'v1',
                    input_schema: { type: 'object' },
                    result_capabilities: ['json'],
                },
                {
                    id: 'tool-definition:write:v2',
                    name: 'write',
                    version: 'v2',
                    input_schema: { type: 'object' },
                    result_capabilities: ['text', 'document'],
                },
            ],
            active_tool_definition_ids: ['tool-definition:write:v2', 'tool-definition:lookup:v1'],
        },
        {
            expected_revision: initial.revision,
            operation_id: 'operation:active-tools',
            payload_fingerprint: 'sha256:active-tools',
            recorded_at: RECORDED_AT,
        },
    ).document;
}

function freshRuntime(document: ConversationDocument, suffix: string): ResolvedConversationRuntimeContext {
    return {
        conversation_id: document.id,
        request_id: `request:${suffix}`,
        attempt_id: `attempt:${suffix}`,
        input_operation_id: `operation:${suffix}`,
        response_operation_id: `response:${suffix}`,
        recorded_at: RECORDED_AT,
        purpose: 'interaction',
    };
}

describe('canonical active tool precedence', () => {
    it('prepares an exact retained context without an empty append or legacy tool authority', async () => {
        const document = activeToolDocument();
        const prepared = await prepareCanonicalContext({
            options: {
                model: 'test-model',
                conversation: document,
                conversation_runtime: freshRuntime(document, 'direct-context'),
            },
            provider: 'test-provider',
            protocol: 'test.protocol',
            adapter_version: 'test.v1',
        });

        expect(prepared.document).toEqual(document);
        expect(prepared.document).not.toBe(document);
        expect(prepared.document.revision).toBe(document.revision);
        expect(prepared.document.operation_receipts).toEqual(document.operation_receipts);
        expect(prepared.request_document).toEqual(prepared.document);
        expect(prepared.tool_definitions).toEqual([
            document.tool_definitions['tool-definition:write:v2'],
            document.tool_definitions['tool-definition:lookup:v1'],
        ]);
        expect(prepared.tool_definitions[0]).not.toBe(document.tool_definitions['tool-definition:write:v2']);
    });

    it('rejects a missing active definition while preparing a retained context', async () => {
        const document = activeToolDocument();
        delete document.tool_definitions['tool-definition:write:v2'];

        await expect(
            prepareCanonicalContext({
                options: {
                    model: 'test-model',
                    conversation: document,
                    conversation_runtime: freshRuntime(document, 'missing-context-tool'),
                },
                provider: 'test-provider',
                protocol: 'test.protocol',
                adapter_version: 'test.v1',
            }),
        ).rejects.toThrow('Conversation document validation failed');
    });

    it('preserves the exact ordered canonical catalog when legacy tools are omitted', async () => {
        const document = activeToolDocument();
        const appended = await appendCanonicalPrompt(
            document,
            emptyRecords(),
            freshRuntime(document, 'preserve-tools'),
            undefined,
            null,
        );

        expect(appended.tool_definitions).toEqual([
            document.tool_definitions['tool-definition:write:v2'],
            document.tool_definitions['tool-definition:lookup:v1'],
        ]);
        expect(appended.document.context.active_tool_definition_ids).toEqual([
            'tool-definition:write:v2',
            'tool-definition:lookup:v1',
        ]);
        expect(appended.tool_definitions[0]).not.toBe(document.tool_definitions['tool-definition:write:v2']);
    });

    it('keeps explicit legacy arrays as exact replacement and clear operations', async () => {
        const document = activeToolDocument();
        const replacement = [{ name: 'search', description: 'Search', input_schema: { type: 'object' as const } }];
        const replacementDefinitions = await canonicalToolDefinitions(replacement);
        const replaced = await appendCanonicalPrompt(
            document,
            emptyRecords(),
            freshRuntime(document, 'replace-tools'),
            replacement,
            null,
        );
        expect(replaced.tool_definitions).toEqual(replacementDefinitions);
        expect(replaced.document.context.active_tool_definition_ids).toEqual(
            replacementDefinitions.map((definition) => definition.id),
        );

        const cleared = await appendCanonicalPrompt(
            document,
            emptyRecords(),
            freshRuntime(document, 'clear-tools'),
            [],
            null,
        );
        expect(cleared.tool_definitions).toEqual([]);
        expect(cleared.document.context.active_tool_definition_ids).toEqual([]);
    });

    it('rejects an active identity whose canonical definition is unavailable', async () => {
        const document = activeToolDocument();
        delete document.tool_definitions['tool-definition:write:v2'];

        await expect(
            appendCanonicalPrompt(document, emptyRecords(), freshRuntime(document, 'missing-tool'), undefined, null),
        ).rejects.toThrow('Active canonical tool definition tool-definition:write:v2 is missing');
    });
});

describe('materialized canonical input proof', () => {
    it('validates a retained materialized input without appending or changing its operation receipt', async () => {
        const document = materializedDocument();
        const proof = runtime(document);
        const prepared = await prepareCanonicalContext({
            options: { model: 'test-model', conversation: document, conversation_runtime: proof },
            provider: 'test-provider',
            protocol: 'test.protocol',
            adapter_version: 'test.v1',
        });

        expect(prepared.document.revision).toBe(document.revision);
        expect(prepared.document.operation_receipts[INPUT_OPERATION_ID]).toEqual(
            document.operation_receipts[INPUT_OPERATION_ID],
        );
        expect(prepared.document.operation_receipts[proof.input_operation_id]).toBeUndefined();
    });

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

        const omittedCompatibilityTools = await appendCanonicalPrompt(
            first.document,
            emptyRecords(),
            proof,
            undefined,
            null,
        );
        expect(omittedCompatibilityTools.document).toEqual(first.document);
        expect(omittedCompatibilityTools.tool_definitions).toEqual(definitions);
    });
});
