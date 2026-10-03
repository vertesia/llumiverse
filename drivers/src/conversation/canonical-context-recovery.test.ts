import {
    appendConversationRecords,
    type ConversationPreparedRequestRecord,
    fingerprintJson,
    parseConversationDocument,
    type ToolCallBlock,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionContextInputOptions,
    canonicalRetainedPreparedRequest,
    canonicalToolDefinitions,
    resolveCanonicalExecutionContextOptions,
    withCanonicalRetainedPreparedRequest,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import {
    type OpenAIChatCompletionsPayload,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsResponse,
} from '../openai/openai_chat_completions.js';

const at = '2026-09-11T00:00:00.000Z';

class RecoveryProtocol extends OpenAIChatCompletionsProtocol<undefined> {
    readonly requests: OpenAIChatCompletionsPayload[] = [];

    constructor() {
        super({ modelName: 'test/model' });
    }

    protected async postChatCompletion(
        _driver: undefined,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<OpenAIChatCompletionsResponse> {
        this.requests.push(structuredClone(payload));
        const first = this.requests.length === 1;
        return {
            id: `native:${this.requests.length}`,
            model: 'test/model',
            object: 'chat.completion',
            created: Date.parse(at) / 1000,
            choices: [
                {
                    index: 0,
                    message: first
                        ? {
                              role: 'assistant',
                              content: null,
                              tool_calls: [
                                  {
                                      id: 'call:lookup',
                                      type: 'function',
                                      function: { name: 'lookup', arguments: '{"key":"quarterly"}' },
                                  },
                              ],
                          }
                        : { role: 'assistant', content: 'Retained result accepted.' },
                    finish_reason: first ? 'tool_calls' : 'stop',
                    logprobs: null,
                },
            ],
            usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
        };
    }

    protected async postChatCompletionStream(): Promise<ReadableStream> {
        throw new Error('Finite recovery fixture does not stream transport');
    }
}

async function acceptedFixture() {
    const protocol = new RecoveryProtocol();
    const first = await protocol.requestCanonicalTextCompletion(
        undefined,
        {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Read the report.' }],
        },
        {
            model: 'test/model',
            conversation_runtime: {
                conversation_id: 'conversation:retained-context',
                request_id: 'request:call',
                attempt_id: 'attempt:call',
                input_operation_id: 'input:call',
                response_operation_id: 'response:call',
                recorded_at: at,
            },
            tools: [{ name: 'lookup', input_schema: { type: 'object', properties: { key: { type: 'string' } } } }],
        },
    );
    const callTurn = first.conversation.turns.find((turn) => turn.id === first.accepted_output.turn.id);
    if (callTurn === undefined) throw new Error('Fixture lacks retained canonical call turn');
    const call = callTurn.blocks.find((block): block is ToolCallBlock => block.type === 'tool_call');
    if (call === undefined) throw new Error('Fixture has no actual application call');
    const result = {
        id: 'block:lookup:result',
        type: 'tool_result' as const,
        call_id: call.call_id,
        status: 'success' as const,
        content: [
            { id: 'block:lookup:text', type: 'text' as const, format: 'plain' as const, text: 'Quarterly result.' },
        ],
    };
    const input = appendConversationRecords(
        first.conversation,
        {
            turns: [
                {
                    id: 'turn:lookup:result',
                    kind: 'tool',
                    execution_id: 'execution:lookup',
                    authority: 'ordinary',
                    model_visibility: 'include',
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    provenance: { type: 'inserted', operation_id: 'operation:lookup' },
                    blocks: [result],
                },
            ],
            execution_receipts: [
                {
                    id: 'execution:lookup',
                    executor: 'application',
                    call_id: call.call_id,
                    status: 'success',
                    result_turn_id: 'turn:lookup:result',
                    result_fingerprint: await fingerprintJson(result),
                    recorded_at: at,
                    call_source: {
                        conversation: { conversation_id: first.conversation.id, revision: first.conversation.revision },
                        turn_id: callTurn.id,
                        block_id: call.id,
                        call_id: call.call_id,
                        call_fingerprint: await fingerprintJson(call),
                    },
                },
            ],
            context_entries: [{ id: 'context:lookup:result', type: 'source_turn', turn_id: 'turn:lookup:result' }],
        },
        {
            expected_revision: first.conversation.revision,
            operation_id: 'operation:lookup',
            payload_fingerprint: await fingerprintJson(result),
            recorded_at: at,
        },
    ).document;
    const extra = await canonicalToolDefinitions([{ name: 'finalize', input_schema: { type: 'object' } }]);
    const source = appendConversationRecords(
        input,
        {
            turns: [
                {
                    id: 'turn:program',
                    kind: 'program',
                    authority: 'ordinary',
                    model_visibility: 'include',
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    provenance: { type: 'inserted', operation_id: 'operation:program' },
                    blocks: [{ id: 'block:program', type: 'text', format: 'plain', text: 'Use the retained report.' }],
                },
            ],
            context_entries: [{ id: 'context:program', type: 'source_turn', turn_id: 'turn:program' }],
            tool_definitions: extra,
            active_tool_definition_ids: [
                ...input.context.active_tool_definition_ids,
                ...extra.map((definition) => definition.id),
            ],
        },
        {
            expected_revision: input.revision,
            operation_id: 'operation:program',
            payload_fingerprint: 'program:catalog',
            recorded_at: at,
        },
    ).document;
    let stored: ConversationPreparedRequestRecord | undefined;
    const options: CanonicalExecutionContextInputOptions = {
        model: 'test/model',
        conversation: source,
        conversation_runtime: {
            conversation_id: source.id,
            request_id: 'request:resume',
            attempt_id: 'attempt:resume',
            input_operation_id: 'input:unused',
            response_operation_id: 'response:resume',
            purpose: 'conversation',
            recorded_at: at,
            materialized_input: { operation_id: 'operation:lookup', result_revision: input.revision },
        },
        on_canonical_request_prepared: async (prepared) => {
            stored = structuredClone(prepared.record);
            stored.request_receipt.source_view = {
                version: 1,
                completeness: 'selected_content_unverified',
                source: stored.source,
                context_revision: prepared.document.context.revision,
                manifest_storage_key: 'authorized/view.json',
                manifest_content_hash: `sha256:${'a'.repeat(64)}`,
                manifest_size_bytes: 10,
                context_fingerprint: stored.request_receipt.context_fingerprint,
                request_fingerprint: stored.request_receipt.request_fingerprint,
            };
            return stored;
        },
    };
    const accepted = await protocol.requestCanonicalContextCompletion(
        undefined,
        resolveCanonicalExecutionContextOptions(options),
    );
    if (stored === undefined) throw new Error('Fixture did not persist a genuine prepared record');
    return {
        protocol,
        accepted,
        record: stored,
        retry: {
            ...options,
            conversation: parseConversationDocument(JSON.parse(JSON.stringify(accepted.conversation))),
        },
    };
}

describe('proof-bearing native context accepted recovery', () => {
    it('retains a real tool proof across program/catalog successors and JSON retry with no transport/accounting', async () => {
        const { protocol, accepted, record, retry } = await acceptedFixture();
        const publish = vi.fn();
        const recover = vi.fn();
        expect(Object.hasOwn(accepted.conversation.operation_receipts, 'input:unused')).toBe(false);
        expect(record.source.revision).toBeGreaterThan(record.runtime.materialized_input?.result_revision ?? 0);
        await expect(
            protocol.requestCanonicalContextCompletion(undefined, resolveCanonicalExecutionContextOptions(retry)),
        ).rejects.toThrow(/materialized tool-set operation/);
        const owned = withCanonicalRetainedPreparedRequest(
            {
                ...retry,
                on_canonical_request_prepared: publish,
                load_recovered_canonical_output: recover,
            },
            JSON.parse(JSON.stringify(record)),
        );
        const response = await protocol.requestCanonicalContextCompletion(
            undefined,
            resolveCanonicalExecutionContextOptions(owned),
        );
        expect(response.accepted_output).toEqual(accepted.accepted_output);
        expect(response.conversation.generations).toEqual(accepted.conversation.generations);
        expect(response.conversation.operation_receipts).toEqual(accepted.conversation.operation_receipts);
        expect(protocol.requests).toHaveLength(2);
        expect(publish).not.toHaveBeenCalled();
        expect(recover).toHaveBeenCalledTimes(1);
    });

    it.each([
        'proof',
        'request',
        'input_operation',
        'response_operation',
        'attempt',
        'source',
        'generation',
        'target',
        'source_view',
    ] as const)('rejects changed %s evidence without additional transport', async (change) => {
        const { protocol, record, retry } = await acceptedFixture();
        const altered = structuredClone(record);
        if (change === 'proof')
            altered.runtime.materialized_input = {
                operation_id: 'operation:program',
                result_revision: altered.source.revision,
            };
        if (change === 'request') altered.runtime.request_id = 'different';
        if (change === 'input_operation') altered.runtime.input_operation_id = 'different';
        if (change === 'response_operation') altered.runtime.response_operation_id = 'different';
        if (change === 'attempt') altered.runtime.attempt_id = 'different';
        if (change === 'source') altered.source.revision -= 1;
        if (change === 'generation') altered.generation_id = 'different';
        if (change === 'target') altered.request_receipt.target.model = 'different';
        if (change === 'source_view') {
            const view = altered.request_receipt.source_view;
            if (view === undefined) throw new Error('Fixture lacks retained view');
            view.manifest_storage_key = 'changed/view.json';
        }
        const options = withCanonicalRetainedPreparedRequest(retry, altered);
        await expect(
            protocol.requestCanonicalContextCompletion(undefined, resolveCanonicalExecutionContextOptions(options)),
        ).rejects.toThrow();
        expect(protocol.requests).toHaveLength(2);
    });

    it.each(['runtime_proof', 'call_source', 'response_operation'] as const)(
        'rejects changed current %s while retained evidence stays exact',
        async (change) => {
            const { protocol, record, retry } = await acceptedFixture();
            if (change === 'runtime_proof') {
                retry.conversation_runtime.materialized_input = {
                    operation_id: 'operation:program',
                    result_revision: record.source.revision,
                };
            }
            if (change === 'call_source') {
                const execution = retry.conversation.execution_receipts['execution:lookup'];
                if (execution?.call_source === undefined) throw new Error('Missing retained execution source');
                execution.call_source.call_fingerprint = 'changed';
            }
            if (change === 'response_operation') retry.conversation_runtime.response_operation_id = 'different';
            const options = withCanonicalRetainedPreparedRequest(retry, record);
            await expect(
                protocol.requestCanonicalContextCompletion(undefined, resolveCanonicalExecutionContextOptions(options)),
            ).rejects.toThrow();
            expect(protocol.requests).toHaveLength(2);
        },
    );

    it('owns the selected record and runtime before asynchronous request hashing', async () => {
        const { protocol, record, retry, accepted } = await acceptedFixture();
        const options = withCanonicalRetainedPreparedRequest(retry, record);
        const pending = protocol.requestCanonicalContextCompletion(
            undefined,
            resolveCanonicalExecutionContextOptions(options),
        );
        record.runtime.request_id = 'mutated';
        retry.conversation_runtime.request_id = 'mutated';
        const hidden = canonicalRetainedPreparedRequest(options);
        if (hidden === undefined) throw new Error('Missing private evidence');
        hidden.request_receipt.target.model = 'mutated';
        expect((await pending).accepted_output).toEqual(accepted.accepted_output);
        expect(protocol.requests).toHaveLength(2);
    });
});
