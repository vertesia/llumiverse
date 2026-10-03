import { createHash } from 'node:crypto';
import {
    appendConversationRecords,
    type ConversationDocument,
    type ConversationStreamEvent,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    parseConversationDocument,
    processingContextFingerprint,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    legacyCompletionFromCanonicalExecution,
    PromptRole,
    Providers,
    resolveCanonicalExecutionContextOptions,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';
import {
    exportLegacyOpenAIResponsesConversation,
    finalizeOpenAIResponsesPreparedRequest,
    OPENAI_RESPONSES_ADAPTER_VERSION,
    OPENAI_RESPONSES_PROTOCOL,
    prepareOpenAIResponsesCanonicalContext,
    prepareOpenAIResponsesCanonicalState,
} from './openai-responses-conversation-adapter.js';

class TestOpenAIResponsesDriver extends OpenAIResponsesDriverBase {
    provider: Providers.openai = Providers.openai;
    service: OpenAI;

    constructor(create: (request: unknown, options?: unknown) => Promise<unknown>) {
        super({});
        this.service = { responses: { create } } as unknown as OpenAI;
    }
}

type ResponseStatus = OpenAI.Responses.Response['status'];

const toolDefinition: NonNullable<ExecutionOptions['tools']>[number] = {
    name: 'lookup_weather',
    description: 'Look up weather',
    input_schema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
        additionalProperties: false,
    },
};

function messageItem(id: string, text: string, status: 'completed' | 'incomplete' = 'completed') {
    return {
        type: 'message' as const,
        id,
        role: 'assistant' as const,
        status,
        content: [
            {
                type: 'output_text' as const,
                text,
                annotations: [],
                logprobs: [],
            },
        ],
    };
}

function response(input: {
    id: string;
    output: OpenAI.Responses.ResponseOutputItem[];
    model?: string;
    status?: ResponseStatus;
    inputTokens?: number;
    outputTokens?: number;
    cachedTokens?: number;
    cacheWriteTokens?: number;
    reasoningTokens?: number;
}): OpenAI.Responses.Response {
    const status = input.status ?? 'completed';
    const inputTokens = input.inputTokens ?? 11;
    const outputTokens = input.outputTokens ?? 7;
    return {
        id: input.id,
        object: 'response',
        created_at: 1,
        model: input.model ?? 'gpt-5',
        service_tier: 'default',
        status,
        output: input.output,
        output_text: '',
        parallel_tool_calls: true,
        tool_choice: 'auto',
        tools: [],
        error: status === 'failed' ? { code: 'server_error', message: 'provider failed' } : null,
        incomplete_details: status === 'incomplete' ? { reason: 'max_output_tokens' } : null,
        instructions: null,
        metadata: null,
        temperature: null,
        top_p: null,
        usage: {
            input_tokens: inputTokens,
            output_tokens: outputTokens,
            total_tokens: inputTokens + outputTokens,
            input_tokens_details: {
                cached_tokens: input.cachedTokens ?? 0,
                ...(input.cacheWriteTokens === undefined ? {} : { cache_write_tokens: input.cacheWriteTokens }),
            },
            output_tokens_details: { reasoning_tokens: input.reasoningTokens ?? 0 },
        },
    } as unknown as OpenAI.Responses.Response;
}

function runtimeOptions(input: {
    flow: string;
    operation: string;
    attempt: string;
    recordedAt: string;
    conversation?: ConversationDocument;
    model?: string;
    materializedInput?: { operation_id: string; result_revision: number };
}): CanonicalExecutionInputOptions {
    return {
        model: input.model ?? 'gpt-5',
        ...(input.conversation === undefined ? {} : { conversation: input.conversation }),
        conversation_runtime: {
            conversation_id: `conversation:${input.flow}`,
            request_id: `request:${input.flow}:${input.operation}`,
            attempt_id: `attempt:${input.flow}:${input.attempt}`,
            input_operation_id: `input:${input.flow}:${input.operation}`,
            response_operation_id: `response:${input.flow}:${input.operation}`,
            recorded_at: input.recordedAt,
            started_at: input.recordedAt,
            ...(input.materializedInput === undefined ? {} : { materialized_input: input.materializedInput }),
        },
    };
}

function materializedInputDocument() {
    const recordedAt = '2026-09-12T00:00:00.000Z';
    const document = createConversationDocument({
        id: 'conversation:materialized-driver-retry',
        created_at: recordedAt,
    });
    const turn = createUserTurn({
        id: 'turn:materialized-driver-input',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: recordedAt },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [createTextBlock({ id: 'block:materialized-driver-input', text: 'Answer once.', format: 'plain' })],
    });
    return appendConversationRecords(
        document,
        {
            turns: [turn],
            context_entries: [{ id: 'context:materialized-driver-input', type: 'source_turn', turn_id: turn.id }],
            active_tool_definition_ids: [],
        },
        {
            expected_revision: document.revision,
            operation_id: 'operation:materialized-driver-input',
            payload_fingerprint: 'sha256:materialized-driver-input',
            recorded_at: recordedAt,
        },
    ).document;
}

function latestGeneratedJson(value: unknown): unknown {
    const document = parseConversationDocument(value);
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind !== 'agent' || turn.provenance.type !== 'generated') continue;
        return turn.blocks.find((block) => block.type === 'json')?.value;
    }
    return undefined;
}

async function consume(stream: AsyncIterable<string>): Promise<string> {
    let text = '';
    for await (const chunk of stream) text += chunk;
    return text;
}

async function collectCanonicalEvents(
    stream: AsyncIterable<ConversationStreamEvent>,
): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

function acceptedOutputWithoutProviderTimestamps(value: unknown): unknown {
    return JSON.parse(JSON.stringify(value), (key, item) =>
        key === 'recorded_at' || key === 'completed_at' ? '<provider-completed-at>' : item,
    );
}

describe('OpenAI Responses canonical lifecycle', () => {
    it('publishes the exact counted native body and measurement in the prepared receipt', async () => {
        const document = materializedInputDocument();
        const create = vi.fn(async (_request: unknown) =>
            response({
                id: 'response:counted-body',
                model: 'gpt-5',
                output: [messageItem('message:counted-body', 'Done.')],
            }),
        );
        const publish = vi.fn(
            async (..._args: Parameters<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>) => undefined,
        );
        const driver = new TestOpenAIResponsesDriver(create);
        await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow: 'materialized-driver-retry',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T00:02:00.000Z',
                conversation: document,
                materializedInput: {
                    operation_id: 'operation:materialized-driver-input',
                    result_revision: document.revision,
                },
            }),
            conversation: document,
            on_canonical_request_projected: async (projection) => ({
                counted_request_fingerprint: await fingerprintJson(projection.native_request),
                measurement: {
                    input_tokens: 42,
                    method: 'estimated',
                    tokenizer: 'full-native-json-bpe-v1',
                    tokenizer_version: '1.0.22',
                    adapter: projection.target.protocol,
                    adapter_version: projection.target.adapter_version,
                    source_fingerprint: await processingContextFingerprint(projection.document),
                    target_model: projection.target.model,
                    measured_at: '2026-09-12T00:02:00.000Z',
                },
            }),
            on_canonical_request_prepared: publish,
        });
        expect(publish).toHaveBeenCalledOnce();
        const prepared = publish.mock.calls[0]?.[0];
        const projection = publish.mock.calls[0]?.[1];
        expect(prepared?.record.request_receipt.measurement).toMatchObject({ input_tokens: 42 });
        expect(prepared?.record.request_receipt.measurement).toEqual(projection?.measurement);
        expect(projection?.counted_request_fingerprint).toBe(
            await fingerprintJson(JSON.parse(JSON.stringify(create.mock.calls[0]?.[0]))),
        );
    });

    it('exposes the exact final execute and stream bodies before publication, with callback isolation', async () => {
        const document = materializedInputDocument();
        const sent: unknown[] = [];
        const create = vi.fn(async (request: unknown) => {
            sent.push(structuredClone(request));
            return response({
                id: 'response:projected-body',
                model: 'gpt-4o-2024-08-06',
                output: [messageItem('message:projected-body', 'Done.')],
            });
        });
        const driver = new TestOpenAIResponsesDriver(create);
        const observed: unknown[] = [];
        const publish = vi.fn(async () => undefined);
        await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow: 'materialized-driver-retry',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T00:03:00.000Z',
                conversation: document,
                materializedInput: {
                    operation_id: 'operation:materialized-driver-input',
                    result_revision: document.revision,
                },
            }),
            conversation: document,
            on_canonical_request_projected: async (projection) => {
                observed.push(structuredClone(projection.native_request));
                const body = projection.native_request as { model?: string };
                body.model = 'mutated-by-callback';
                return undefined;
            },
            on_canonical_request_prepared: publish,
        });
        expect(observed[0]).toEqual(sent[0]);
        expect(sent[0]).toMatchObject({ model: 'gpt-5', stream: false });
        expect(publish).toHaveBeenCalledOnce();

        const terminal = response({
            id: 'response:projected-stream',
            model: 'gpt-4o-2024-08-06',
            output: [messageItem('message:projected-stream', 'Done.')],
        });
        const streamCreate = vi.fn(async (request: unknown) => {
            sent.push(structuredClone(request));
            return (async function* () {
                yield { type: 'response.completed' as const, sequence_number: 1, response: terminal };
            })();
        });
        const streamDriver = new TestOpenAIResponsesDriver(streamCreate);
        const stream = await streamDriver.streamCanonicalContextEvents(
            {
                ...runtimeOptions({
                    flow: 'materialized-driver-retry',
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-09-12T00:04:00.000Z',
                    conversation: document,
                    materializedInput: {
                        operation_id: 'operation:materialized-driver-input',
                        result_revision: document.revision,
                    },
                }),
                conversation: document,
                on_canonical_request_projected: async (projection) => {
                    observed.push(structuredClone(projection.native_request));
                    return undefined;
                },
                on_canonical_request_prepared: publish,
            },
            undefined,
            { stream_id: 'stream:projected-body' },
        );
        await collectCanonicalEvents(stream);
        expect(observed[1]).toEqual(sent[1]);
        expect(sent[1]).toMatchObject({ model: 'gpt-5', stream: true });
        expect(publish).toHaveBeenCalledTimes(2);
    });

    it('rejects a post-input budget callback before prepared publication or provider transport', async () => {
        const document = materializedInputDocument();
        const create = vi.fn();
        const publish = vi.fn(async () => undefined);
        const driver = new TestOpenAIResponsesDriver(create);
        await expect(
            driver.executeCanonicalContext({
                ...runtimeOptions({
                    flow: 'materialized-driver-retry',
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-09-12T00:05:00.000Z',
                    conversation: document,
                    materializedInput: {
                        operation_id: 'operation:materialized-driver-input',
                        result_revision: document.revision,
                    },
                }),
                conversation: document,
                on_canonical_request_projected: async (projection) => {
                    expect(projection.native_request).toMatchObject({ stream: false });
                    throw new Error('Post-input request exceeds the target budget');
                },
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow('Post-input request exceeds the target budget');
        expect(publish).not.toHaveBeenCalled();
        expect(create).not.toHaveBeenCalled();
    });

    it('dry-projects the exact configured Responses body sent by a fresh context execution', async () => {
        const create = vi.fn(async (_request: unknown) =>
            response({
                id: 'response:model-switch-parity',
                model: 'gpt-4o-2024-08-06',
                output: [messageItem('message:model-switch-parity', 'Done.')],
            }),
        );
        class AliasedResponsesDriver extends TestOpenAIResponsesDriver {
            override getResponsesRequestModel(model: string): string {
                return `deployment/${model}`;
            }
        }
        const driver = new AliasedResponsesDriver(create);
        const document = materializedInputDocument();
        const modelOptions = {
            _option_id: 'openai-text',
            max_tokens: 64,
            extra_body: { metadata: { model_switch: 'count-this-field' } },
        } as const;
        const target = {
            provider: Providers.openai,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            model: 'gpt-4o-2024-08-06',
            adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
            options: modelOptions,
        };
        const projected = await driver.projectCanonicalModelSwitchRequest(document, target, 'execute');
        expect(projected.status).toBe('compiled');
        if (projected.status !== 'compiled') throw new Error('Expected configured Responses projection');
        await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow: 'materialized-driver-retry',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T00:01:00.000Z',
                conversation: document,
                model: target.model,
                materializedInput: {
                    operation_id: 'operation:materialized-driver-input',
                    result_revision: document.revision,
                },
            }),
            conversation: document,
            model_options: modelOptions,
        });
        expect(create).toHaveBeenCalledOnce();
        expect(projected.native_request).toEqual(create.mock.calls[0]?.[0]);
        expect(projected.native_request).toMatchObject({
            model: 'deployment/gpt-4o-2024-08-06',
            metadata: { model_switch: 'count-this-field' },
        });

        const resolved = await driver.resolveCanonicalModelSwitchTarget(target.model, modelOptions);
        expect(resolved).toEqual(target);
        const projectedStream = await driver.projectCanonicalModelSwitchRequest(document, target, 'stream');
        if (projectedStream.status !== 'compiled') throw new Error('Expected configured Responses stream projection');
        const terminal = response({
            id: 'response:model-switch-stream-parity',
            model: 'gpt-4o-2024-08-06',
            output: [messageItem('message:model-switch-stream-parity', 'Done.')],
        });
        const streamCreate = vi.fn(async (_request: unknown) =>
            (async function* () {
                yield { type: 'response.completed' as const, sequence_number: 1, response: terminal };
            })(),
        );
        const streamDriver = new AliasedResponsesDriver(streamCreate);
        const stream = await streamDriver.streamCanonicalContextEvents(
            {
                ...runtimeOptions({
                    flow: 'materialized-driver-retry',
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-09-12T00:02:00.000Z',
                    conversation: document,
                    model: target.model,
                    materializedInput: {
                        operation_id: 'operation:materialized-driver-input',
                        result_revision: document.revision,
                    },
                }),
                conversation: document,
                model_options: modelOptions,
            },
            undefined,
            { stream_id: 'stream:model-switch-parity' },
        );
        await collectCanonicalEvents(stream);
        expect(streamCreate).toHaveBeenCalledOnce();
        expect(projectedStream.native_request).toEqual(streamCreate.mock.calls[0]?.[0]);
        expect(projectedStream.native_request).toMatchObject({
            model: 'deployment/gpt-4o-2024-08-06',
            stream: true,
            max_output_tokens: 64,
        });
    });

    it('rejects provider-side continuation and extra-body source overrides during a dry switch', async () => {
        const driver = new TestOpenAIResponsesDriver(
            vi.fn(async () => {
                throw new Error('Dry projection must not send provider transport');
            }),
        );
        const document = materializedInputDocument();
        const base = {
            provider: Providers.openai,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            model: 'gpt-4o-2024-08-06',
            adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
        };
        for (const field of ['previous_response_id', 'conversation', 'input', 'tools'] as const) {
            const projected = await driver.projectCanonicalModelSwitchRequest(
                document,
                { ...base, options: { _option_id: 'openai-text', extra_body: { [field]: 'opaque' } } },
                'execute',
            );
            expect(projected).toMatchObject({ status: 'unsupported' });
        }
    });

    it('does not authorize implicit history stripping as a compatible model switch', async () => {
        const at = '2026-09-12T00:00:00.000Z';
        const source = createConversationDocument({ id: 'conversation:model-switch-history', created_at: at });
        const older = createUserTurn({
            id: 'turn:model-switch-history:older',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            model_visibility: 'include',
            provenance: { type: 'received' },
            blocks: [
                createTextBlock({
                    id: 'block:model-switch-history:older',
                    text: '<heartbeat>old status</heartbeat>',
                    format: 'plain',
                }),
            ],
        });
        const newer = createUserTurn({
            id: 'turn:model-switch-history:newer',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            model_visibility: 'include',
            provenance: { type: 'received' },
            blocks: [createTextBlock({ id: 'block:model-switch-history:newer', text: 'Continue.', format: 'plain' })],
        });
        const document = appendConversationRecords(
            source,
            {
                turns: [older, newer],
                context_entries: [
                    { id: 'entry:model-switch-history:older', type: 'source_turn', turn_id: older.id },
                    { id: 'entry:model-switch-history:newer', type: 'source_turn', turn_id: newer.id },
                ],
            },
            {
                expected_revision: 0,
                operation_id: 'operation:model-switch-history',
                payload_fingerprint: 'sha256:model-switch-history',
                recorded_at: at,
            },
        ).document;
        const driver = new TestOpenAIResponsesDriver(
            vi.fn(async () => {
                throw new Error('Dry projection must not send provider transport');
            }),
        );
        const projected = await driver.projectCanonicalModelSwitchRequest(
            document,
            {
                provider: Providers.openai,
                protocol: OPENAI_RESPONSES_PROTOCOL,
                model: 'gpt-4o-2024-08-06',
                adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
            },
            'execute',
            { stripHeartbeatsAfterTurns: 0 },
        );
        expect(projected).toMatchObject({ status: 'unsupported', reason: expect.stringContaining('history') });
    });

    it('emits native-positioned structured-output events with the same accepted output as the legacy stream boundary', async () => {
        const final = response({
            id: 'response:typed-parity',
            output: [messageItem('message:typed-parity', '{"answer":"Tokyo"}')],
        });
        const create = vi.fn(async () =>
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:typed-parity',
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 1,
                    delta: '{"answer":',
                    logprobs: [],
                };
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:typed-parity',
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 2,
                    delta: '"Tokyo"}',
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 3, response: final };
            })(),
        );
        const executionOptions: CanonicalExecutionInputOptions = {
            ...runtimeOptions({
                flow: 'typed-parity',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
        };
        const typedDriver = new TestOpenAIResponsesDriver(create);
        const typed = await typedDriver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Return JSON.' }],
            executionOptions,
            undefined,
            { stream_id: 'stream:responses:typed-parity' },
        );
        const events = await collectCanonicalEvents(typed);
        const acceptedEvent = events.find((event) => event.type === 'response_accepted');

        expect(events.filter((event) => event.type === 'draft_text_delta')).toMatchObject([
            {
                text: '{"answer":',
                native_position: {
                    protocol: 'openai.responses',
                    path: ['output', 0, 'content', 0],
                    native_item_id: 'message:typed-parity',
                },
            },
            { text: '"Tokyo"}' },
        ]);
        expect(acceptedEvent).toMatchObject({
            reconciliations: [
                {
                    disposition: 'structured_output',
                    committed_block_ids: [typed.completion?.accepted_output.turn.blocks[0]?.id],
                },
            ],
        });
        expect(typed.completion?.accepted_output.turn.blocks).toMatchObject([
            { type: 'json', value: { answer: 'Tokyo' } },
        ]);

        const legacyDriver = new TestOpenAIResponsesDriver(create);
        const legacy = await legacyDriver.streamCanonical(
            [{ role: PromptRole.user, content: 'Return JSON.' }],
            executionOptions,
        );
        await consume(legacy);
        expect(acceptedOutputWithoutProviderTimestamps(typed.completion?.accepted_output)).toEqual(
            acceptedOutputWithoutProviderTimestamps(legacy.completion?.accepted_output),
        );
    });

    it('normalizes every canonical output-text partition even when legacy presentation omits an empty part', async () => {
        const output = [
            {
                type: 'message',
                id: 'message:canonical-partitions',
                role: 'assistant',
                status: 'completed',
                content: [
                    { type: 'output_text', text: '', annotations: [], logprobs: [] },
                    { type: 'output_text', text: '{"answer":"Tokyo"}', annotations: [], logprobs: [] },
                ],
            },
        ] as OpenAI.Responses.ResponseOutputItem[];
        const create = vi.fn(async () => response({ id: 'response:canonical-partitions', output }));
        const driver = new TestOpenAIResponsesDriver(create);

        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({
                flow: 'canonical-partitions',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:15.000Z',
            }),
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
        });

        expect(result.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } }),
        );
        expect(latestGeneratedJson(result.conversation)).toEqual({ answer: 'Tokyo' });
        expect(create).toHaveBeenCalledOnce();
    });

    it('recovers an accepted typed stream as one terminal event without publishing or calling transport again', async () => {
        const final = response({
            id: 'response:typed-recovery',
            output: [messageItem('message:typed-recovery', 'Accepted once.')],
        });
        const create = vi.fn(async () =>
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:typed-recovery',
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 1,
                    delta: 'Accepted once.',
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 2, response: final };
            })(),
        );
        const publish = vi.fn(async () => undefined);
        const firstOptions: CanonicalExecutionInputOptions = {
            ...runtimeOptions({
                flow: 'typed-recovery',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            on_canonical_request_prepared: publish,
        };
        const driver = new TestOpenAIResponsesDriver(create);
        const first = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer once.' }],
            firstOptions,
            undefined,
            { stream_id: 'stream:responses:typed-recovery:first' },
        );
        await collectCanonicalEvents(first);
        if (first.completion === undefined) throw new Error('Expected initial typed completion');

        const recovered = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer once.' }],
            {
                ...runtimeOptions({
                    flow: 'typed-recovery',
                    operation: 'generate',
                    attempt: 'retry',
                    recordedAt: '2026-09-12T01:01:00.000Z',
                    conversation: JSON.parse(JSON.stringify(first.completion.conversation)),
                }),
                on_canonical_request_prepared: publish,
            },
            undefined,
            { stream_id: 'stream:responses:typed-recovery:delivery-2' },
        );
        const recoveredEvents = await collectCanonicalEvents(recovered);

        expect(recoveredEvents).toHaveLength(1);
        expect(recoveredEvents[0]).toMatchObject({
            type: 'response_accepted',
            origin: 'accepted_recovery',
            stream_id: 'stream:responses:typed-recovery:delivery-2',
            sequence: 0,
        });
        expect(recovered.completion?.accepted_output).toEqual(first.completion.accepted_output);
        expect(create).toHaveBeenCalledTimes(1);
        expect(publish).toHaveBeenCalledTimes(1);
    });

    it('retains the authoritative accepted response when bounded final-event delivery fails', async () => {
        const final = response({
            id: 'response:typed-delivery-bound',
            output: [messageItem('message:typed-delivery-bound', 'Accepted despite delivery failure.')],
        });
        const driver = new TestOpenAIResponsesDriver(
            vi.fn(async () =>
                (async function* () {
                    yield {
                        type: 'response.output_text.delta' as const,
                        item_id: 'message:typed-delivery-bound',
                        output_index: 0,
                        content_index: 0,
                        sequence_number: 1,
                        delta: 'Accepted despite delivery failure.',
                        logprobs: [],
                    };
                    yield { type: 'response.completed' as const, sequence_number: 2, response: final };
                })(),
            ),
        );
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer.' }],
            runtimeOptions({
                flow: 'typed-delivery-bound',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            undefined,
            { stream_id: 'stream:responses:typed-delivery-bound', max_events: 5 },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'CANONICAL_EVENT_DELIVERY_FAILED' },
        });
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Accepted despite delivery failure.' }),
        );
        expect(stream.completion?.accepted_output.generation.status).toBe('completed');
    });

    it('retains authoritative terminal text while rejecting divergent preview reconciliation', async () => {
        const final = response({
            id: 'response:typed-divergent-preview',
            output: [messageItem('message:typed-divergent-preview', 'Authoritative terminal text.')],
        });
        const driver = new TestOpenAIResponsesDriver(
            vi.fn(async () =>
                (async function* () {
                    yield {
                        type: 'response.output_text.delta' as const,
                        item_id: 'message:typed-divergent-preview',
                        output_index: 0,
                        content_index: 0,
                        sequence_number: 1,
                        delta: 'Different preview text.',
                        logprobs: [],
                    };
                    yield { type: 'response.completed' as const, sequence_number: 2, response: final };
                })(),
            ),
        );
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer.' }],
            runtimeOptions({
                flow: 'typed-divergent-preview',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            undefined,
            { stream_id: 'stream:responses:typed-divergent-preview' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'CANONICAL_EVENT_DELIVERY_FAILED' },
        });
        expect(events.some((event) => event.type === 'response_accepted')).toBe(false);
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Authoritative terminal text.' }),
        );
    });

    it('retains a terminal-only accepted response when bounded draft synthesis cannot be delivered', async () => {
        const final = response({
            id: 'response:typed-terminal-only-bound',
            output: [messageItem('message:typed-terminal-only-bound', 'Terminal only.')],
        });
        const driver = new TestOpenAIResponsesDriver(
            vi.fn(async () =>
                (async function* () {
                    yield { type: 'response.completed' as const, sequence_number: 1, response: final };
                })(),
            ),
        );
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer.' }],
            runtimeOptions({
                flow: 'typed-terminal-only-bound',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            undefined,
            { stream_id: 'stream:responses:typed-terminal-only-bound', max_events: 2 },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events).toHaveLength(2);
        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'CANONICAL_EVENT_DELIVERY_FAILED' },
        });
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Terminal only.' }),
        );
    });

    it('fails the typed prepared-request barrier before opening the provider stream', async () => {
        const create = vi.fn(async () => {
            throw new Error('transport must not be called');
        });
        const driver = new TestOpenAIResponsesDriver(create);

        await expect(
            driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'Answer.' }],
                {
                    ...runtimeOptions({
                        flow: 'typed-barrier',
                        operation: 'generate',
                        attempt: 'first',
                        recordedAt: '2026-09-12T01:00:00.000Z',
                    }),
                    on_canonical_request_prepared: async () => {
                        throw new Error('durability barrier failed');
                    },
                },
                undefined,
                { stream_id: 'stream:responses:typed-barrier' },
            ),
        ).rejects.toThrow('durability barrier failed');
        expect(create).not.toHaveBeenCalled();
    });

    it.each([
        ['zero event budget', { max_events: 0 }],
        ['zero delivery buffer', { max_buffered_events: 0 }],
        [
            'fresh resume cursor',
            { resume_after: { stream_id: 'stream:prior', event_id: 'stream:prior#0', sequence: 0 } },
        ],
    ] as const)(
        'rejects %s before publication, abort-listener registration, or transport',
        async (_label, invalidOpen) => {
            const create = vi.fn(async () => {
                throw new Error('transport must not be called');
            });
            const publish = vi.fn(async () => undefined);
            const controller = new AbortController();
            const addAbortListener = vi.spyOn(controller.signal, 'addEventListener');
            const removeAbortListener = vi.spyOn(controller.signal, 'removeEventListener');
            const driver = new TestOpenAIResponsesDriver(create);

            await expect(
                driver.streamCanonicalEvents(
                    [{ role: PromptRole.user, content: 'Answer.' }],
                    {
                        ...runtimeOptions({
                            flow: `typed-invalid-open-${_label}`,
                            operation: 'generate',
                            attempt: 'first',
                            recordedAt: '2026-09-12T01:00:00.000Z',
                        }),
                        on_canonical_request_prepared: publish,
                    },
                    controller.signal,
                    { stream_id: `stream:responses:typed-invalid-open:${_label}`, ...invalidOpen },
                ),
            ).rejects.toThrow();
            expect(publish).not.toHaveBeenCalled();
            expect(create).not.toHaveBeenCalled();
            expect(addAbortListener).not.toHaveBeenCalled();
            expect(removeAbortListener).not.toHaveBeenCalled();
        },
    );

    it('emits display reasoning while keeping encrypted provider replay out of typed events', async () => {
        const reasoningItem = {
            type: 'reasoning' as const,
            id: 'reasoning:typed-protected',
            summary: [{ type: 'summary_text' as const, text: 'Visible summary.' }],
            encrypted_content: 'opaque-protected-replay-payload',
            status: 'completed' as const,
        };
        const final = response({
            id: 'response:typed-protected',
            output: [reasoningItem, messageItem('message:typed-protected', 'Answer.')],
            reasoningTokens: 4,
        });
        const create = vi.fn(async () =>
            (async function* () {
                yield {
                    type: 'response.reasoning_summary_text.delta' as const,
                    item_id: reasoningItem.id,
                    output_index: 0,
                    summary_index: 0,
                    sequence_number: 1,
                    delta: 'Visible summary.',
                };
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:typed-protected',
                    output_index: 1,
                    content_index: 0,
                    sequence_number: 2,
                    delta: 'Answer.',
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 3, response: final };
            })(),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Explain.' }],
            {
                ...runtimeOptions({
                    flow: 'typed-protected',
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-09-12T01:00:00.000Z',
                }),
                model_options: { _option_id: 'openai-thinking', include_thoughts: false },
            },
            undefined,
            { stream_id: 'stream:responses:typed-protected' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events).toContainEqual(
            expect.objectContaining({ type: 'draft_reasoning_delta', text: 'Visible summary.' }),
        );
        expect(JSON.stringify(events)).not.toContain(reasoningItem.encrypted_content);
        expect(stream.completion?.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'reasoning', text: 'Visible summary.' }),
                expect.objectContaining({ type: 'text', text: 'Answer.' }),
            ]),
        );
        if (stream.completion === undefined) throw new Error('Expected protected-reasoning typed completion');
        expect(legacyCompletionFromCanonicalExecution(stream.completion, { include_reasoning: false }).result).toEqual([
            { type: 'text', value: 'Answer.' },
        ]);
    });
    it('recovers a saved response against its older materialized-input proof without another transport call', async () => {
        const create = vi.fn(async () =>
            response({
                id: 'response:materialized-driver-retry',
                output: [messageItem('message:materialized-driver-retry', 'Accepted once.')],
            }),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const materialized = materializedInputDocument();
        const proof = {
            operation_id: 'operation:materialized-driver-input',
            result_revision: materialized.revision,
        };
        const first = await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow: 'materialized-driver-retry',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T00:01:00.000Z',
                conversation: materialized,
                materializedInput: proof,
            }),
            conversation: materialized,
        });
        expect(first.conversation.revision).toBe(materialized.revision + 1);

        const retained = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;
        const retry = await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow: 'materialized-driver-retry',
                operation: 'generate',
                attempt: 'retry',
                recordedAt: '2026-09-12T00:02:00.000Z',
                conversation: retained,
                materializedInput: proof,
            }),
            conversation: retained,
        });
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(retry.conversation).toEqual(first.conversation);
        expect(create).toHaveBeenCalledOnce();
        const generation = Object.values(first.conversation.generations).find(
            (candidate) =>
                candidate.record_source === 'executed' && candidate.id === first.accepted_output.generation.id,
        );
        expect(generation?.record_source).toBe('executed');
        expect(generation?.record_source === 'executed' ? generation.request_receipt.source : undefined).toEqual({
            conversation_id: materialized.id,
            revision: materialized.revision,
        });
    });

    it('executes directly into canonical output and recovers an accepted retry without transport', async () => {
        const native = response({
            id: 'response:canonical-direct',
            output: [messageItem('message:canonical-direct', '{"answer":"Direct answer."}')],
        });
        const create = vi.fn(async () => native);
        const publishPreparedRequest = vi.fn(async () => undefined);
        const driver = new TestOpenAIResponsesDriver(create);
        const segments = [{ role: PromptRole.user, content: 'Answer directly.' }];
        const options = {
            ...runtimeOptions({
                flow: 'canonical-direct',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            result_schema: {
                type: 'object' as const,
                properties: { answer: { type: 'string' as const } },
                required: ['answer'],
                additionalProperties: false,
            },
            on_canonical_request_prepared: publishPreparedRequest,
        };

        const first = await driver.executeCanonical(segments, options);
        expect(first.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([expect.objectContaining({ type: 'json', value: { answer: 'Direct answer.' } })]),
        );
        expect(first.accepted_output.generation.usage).toMatchObject({ input_tokens: 11, output_tokens: 7 });
        expect(first.service_tier).toBe('default');
        expect(first).not.toHaveProperty('prompt');

        const retry = await driver.executeCanonical(segments, {
            ...runtimeOptions({
                flow: 'canonical-direct',
                operation: 'generate',
                attempt: 'retry',
                recordedAt: '2026-09-12T01:00:01.000Z',
                conversation: JSON.parse(JSON.stringify(first.conversation)),
            }),
            result_schema: options.result_schema,
            on_canonical_request_prepared: publishPreparedRequest,
        });
        expect(retry.accepted_output).toEqual(first.accepted_output);
        await expect(
            driver.executeCanonical(segments, {
                ...runtimeOptions({
                    flow: 'canonical-direct',
                    operation: 'generate',
                    attempt: 'changed-options',
                    recordedAt: '2026-09-12T01:00:02.000Z',
                    conversation: first.conversation,
                }),
                result_schema: options.result_schema,
                model_options: { _option_id: 'openai-thinking', max_tokens: 123 },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(create).toHaveBeenCalledOnce();
        expect(publishPreparedRequest).toHaveBeenCalledOnce();
    });

    it.each(['sync', 'stream'] as const)(
        'prevents %s transport when prepared-request publication fails',
        async (mode) => {
            const create = vi.fn(async () =>
                response({
                    id: `response:publication-${mode}`,
                    output: [messageItem(`message:publication-${mode}`, 'Must not be returned.')],
                }),
            );
            const driver = new TestOpenAIResponsesDriver(create);
            const options: CanonicalExecutionInputOptions = {
                ...runtimeOptions({
                    flow: `publication-${mode}`,
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-09-12T01:00:30.000Z',
                }),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability barrier failed');
                },
            };
            const execution =
                mode === 'sync'
                    ? driver.executeCanonical([{ role: PromptRole.user, content: 'Answer.' }], options)
                    : driver.streamCanonical([{ role: PromptRole.user, content: 'Answer.' }], options);

            await expect(execution).rejects.toThrow('durability barrier failed');
            expect(create).not.toHaveBeenCalled();
        },
    );

    it('streams directly into a failed canonical outcome for invalid required structured output', async () => {
        const final = response({
            id: 'response:canonical-invalid-stream',
            output: [messageItem('message:canonical-invalid-stream', '{"wrong":true}')],
        });
        const create = vi.fn(async () =>
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:canonical-invalid-stream',
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 1,
                    delta: '{"wrong":true}',
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 2, response: final };
            })(),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const stream = await driver.streamCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({
                flow: 'canonical-invalid-stream',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:01:00.000Z',
            }),
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
        });

        expect(await consume(stream)).toBe('{"wrong":true}');
        expect(stream.completion?.accepted_output.generation.status).toBe('failed');
        expect(stream.completion?.accepted_output.turn.status).toBe('failed');
        expect(stream.completion?.service_tier).toBe('default');
        expect(stream.completion?.accepted_output.generation.usage).toMatchObject({
            input_tokens: 11,
            output_tokens: 7,
        });
    });

    it.each(['sync', 'stream'] as const)(
        'preserves an incomplete cutoff over a complete function call for direct canonical %s execution',
        async (mode) => {
            const callItem = {
                type: 'function_call' as const,
                id: 'item:cutoff-call',
                call_id: 'call:cutoff-call',
                name: 'lookup_weather',
                arguments: '{"city":"Tokyo"}',
                status: 'completed' as const,
            };
            const incomplete = response({
                id: `response:cutoff-${mode}`,
                output: [callItem],
                status: 'incomplete',
            });
            const create = vi.fn(async (request: unknown) => {
                if (!(request as { stream?: boolean }).stream) return incomplete;
                return (async function* () {
                    yield {
                        type: 'response.output_item.added' as const,
                        output_index: 0,
                        sequence_number: 1,
                        item: callItem,
                    };
                    yield { type: 'response.incomplete' as const, sequence_number: 2, response: incomplete };
                })();
            });
            const driver = new TestOpenAIResponsesDriver(create);
            const options = {
                ...runtimeOptions({
                    flow: `direct-cutoff-${mode}`,
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-09-12T01:02:00.000Z',
                }),
                tools: [toolDefinition],
            };
            const result =
                mode === 'sync'
                    ? await driver.executeCanonical([{ role: PromptRole.user, content: 'Call the tool.' }], options)
                    : await (async () => {
                          const stream = await driver.streamCanonical(
                              [{ role: PromptRole.user, content: 'Call the tool.' }],
                              options,
                          );
                          await consume(stream);
                          if (stream.completion === undefined) throw new Error('Expected canonical stream completion');
                          return stream.completion;
                      })();

            expect(result.accepted_output.turn.status).toBe('interrupted');
            expect(result.accepted_output.generation).toMatchObject({
                status: 'cancelled',
                finish_reason: 'length',
            });
            expect(result.accepted_output.turn.blocks).toContainEqual(
                expect.objectContaining({
                    type: 'tool_call',
                    call_id: 'call:cutoff-call',
                    tool_name: 'lookup_weather',
                }),
            );
        },
    );

    it('validates structured sync output, retains exact native data, and recovers a persisted retry', async () => {
        const rawText = '{ "answer" : "Tokyo", "note" : null }';
        const output = [
            {
                ...messageItem('message:structured', rawText),
                content: [
                    {
                        type: 'output_text' as const,
                        text: rawText,
                        annotations: [
                            {
                                type: 'url_citation' as const,
                                start_index: 13,
                                end_index: 20,
                                title: 'Tokyo',
                                url: 'https://example.com/tokyo',
                            },
                        ],
                        logprobs: [],
                    },
                ],
            },
        ] satisfies OpenAI.Responses.ResponseOutputItem[];
        const nativeResponse = response({
            id: 'response:structured',
            output,
            inputTokens: 100,
            outputTokens: 20,
            cachedTokens: 25,
            cacheWriteTokens: 5,
            reasoningTokens: 7,
        });
        const create = vi.fn(async (_request: unknown, _options?: unknown) => nativeResponse);
        const driver = new TestOpenAIResponsesDriver(create);
        const segments = [{ role: PromptRole.user, content: 'Return the city as JSON.' }];
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' }, note: { type: 'null' } },
            required: ['answer', 'note'],
            additionalProperties: false,
        };
        const firstOptions = {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T00:00:00.000Z',
            }),
            result_schema: resultSchema,
        };

        const first = await driver.execute(segments, firstOptions);

        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo', note: null } }]);
        expect(first.token_usage).toEqual({
            prompt: 100,
            prompt_cached: 25,
            prompt_cache_write: 5,
            prompt_new: 70,
            result: 20,
            total: 120,
        });
        expect(latestGeneratedJson(first.conversation)).toEqual({ answer: 'Tokyo', note: null });
        const document = parseConversationDocument(JSON.parse(JSON.stringify(first.conversation)));
        const generation = Object.values(document.generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        expect(generation).toMatchObject({
            provider: 'openai',
            protocol: OPENAI_RESPONSES_PROTOCOL,
            requested_model: 'gpt-5',
            resolved_model: 'gpt-5',
            provider_response_id: 'response:structured',
            usage: {
                input_tokens: 100,
                output_tokens: 20,
                reasoning_tokens: 7,
                cache_read_tokens: 25,
                cache_write_tokens: 5,
                input_new_tokens: 70,
                total_tokens: 120,
            },
        });
        expect(exportLegacyOpenAIResponsesConversation(document).at(-1)).toEqual(output[0]);

        const retried = await driver.execute(segments, {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'retry',
                recordedAt: '2026-09-12T00:05:00.000Z',
                conversation: document,
            }),
            result_schema: resultSchema,
        });

        expect(retried.result).toEqual(first.result);
        expect(retried.token_usage).toEqual(first.token_usage);
        expect(retried.service_tier).toBe(first.service_tier);
        expect(retried.conversation).toEqual(document);
        expect(create).toHaveBeenCalledTimes(1);
    });

    it('normalizes split streamed JSON around reasoning and recovers its exact native output', async () => {
        const firstFragment = '```json\n{"answer":';
        const secondFragment = '"Tokyo"}\n```';
        const firstMessage = messageItem('message:structured-stream:first', firstFragment);
        const reasoningItem = {
            type: 'reasoning' as const,
            id: 'reasoning:structured-stream',
            summary: [{ type: 'summary_text' as const, text: 'Check the requested shape.' }],
            encrypted_content: 'opaque-structured-stream-reasoning',
            status: 'completed' as const,
        };
        const secondMessage = messageItem('message:structured-stream:second', secondFragment);
        const final = response({
            id: 'response:structured-stream',
            output: [firstMessage, reasoningItem, secondMessage],
            reasoningTokens: 3,
        });
        const create = vi.fn(async (_request: unknown, _options?: unknown) =>
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: firstMessage.id,
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 1,
                    delta: firstFragment,
                    logprobs: [],
                };
                yield {
                    type: 'response.reasoning_summary_text.delta' as const,
                    item_id: reasoningItem.id,
                    output_index: 1,
                    summary_index: 0,
                    sequence_number: 2,
                    delta: 'Check the requested shape.',
                };
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: secondMessage.id,
                    output_index: 2,
                    content_index: 0,
                    sequence_number: 3,
                    delta: secondFragment,
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 4, response: final };
            })(),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const segments = [{ role: PromptRole.user, content: 'Return the city as JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'structured-stream',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T00:10:00.000Z',
            }),
            model_options: { _option_id: 'openai-thinking' },
            result_schema,
        });

        await consume(first);
        expect(first.completion?.result).toEqual([
            { type: 'json', value: { answer: 'Tokyo' } },
            { type: 'thoughts', value: 'Check the requested shape.' },
        ]);
        expect(latestGeneratedJson(first.completion?.conversation)).toEqual({ answer: 'Tokyo' });
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(first.completion?.conversation)));
        expect(exportLegacyOpenAIResponsesConversation(persisted).slice(-3)).toEqual(final.output);

        const retried = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'structured-stream',
                operation: 'generate',
                attempt: 'retry',
                recordedAt: '2026-09-12T00:15:00.000Z',
                conversation: persisted,
            }),
            model_options: { _option_id: 'openai-thinking' },
            result_schema,
        });
        await consume(retried);
        expect(retried.completion?.result).toEqual(first.completion?.result);
        expect(retried.completion?.conversation).toEqual(persisted);
        expect(create).toHaveBeenCalledTimes(1);
    });

    it('rejects a function-call result without a call association before provider execution', async () => {
        const options = runtimeOptions({
            flow: 'missing-call-id',
            operation: 'continue',
            attempt: 'first',
            recordedAt: '2026-09-12T00:10:00.000Z',
        });

        await expect(
            prepareOpenAIResponsesCanonicalState({
                conversation: [
                    { type: 'function_call_output', output: 'orphaned result' },
                ] as unknown as OpenAI.Responses.ResponseInputItem[],
                prompt: [],
                options,
                provider: Providers.openai,
            }),
        ).rejects.toThrow('function_call_output at items/0 has no call_id');
    });

    it('preserves tool result status internally, projects rich continuation, and recovers its accepted response', async () => {
        const callItem = {
            type: 'function_call' as const,
            id: 'item:lookup',
            call_id: 'call:lookup',
            name: 'lookup_weather',
            arguments: '{ "city" : "Tokyo", "units" : null }',
            status: 'completed' as const,
        };
        const responses = [
            response({ id: 'response:tool-call', output: [callItem] }),
            response({ id: 'response:tool-result', output: [messageItem('message:tool-result', 'Try again later.')] }),
        ];
        const create = vi.fn(async (_request: unknown, _options?: unknown) => {
            const next = responses.shift();
            if (!next) throw new Error('Unexpected provider call');
            return next;
        });
        const driver = new TestOpenAIResponsesDriver(create);
        const tools = [toolDefinition];
        const first = await driver.execute([{ role: PromptRole.user, content: 'Weather in Tokyo?' }], {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'ask',
                attempt: 'ask',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            tools,
        });
        expect(first.tool_use).toEqual([
            { id: 'call:lookup', tool_name: 'lookup_weather', tool_input: { city: 'Tokyo', units: null } },
        ]);

        const resultSegments = [
            {
                role: PromptRole.tool,
                content: '{"error":"weather service unavailable"}',
                tool_use_id: 'call:lookup',
                tool_result_status: 'error' as const,
            },
        ];
        const second = await driver.execute(resultSegments, {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'answer',
                attempt: 'answer',
                recordedAt: '2026-09-12T01:01:00.000Z',
                conversation: parseConversationDocument(first.conversation),
            }),
            tools,
        });
        const secondRequest = create.mock.calls[1]?.[0] as { input?: Array<Record<string, unknown>> };
        const projectedResult = secondRequest.input?.find((item) => item.type === 'function_call_output');
        expect(projectedResult).toEqual({
            type: 'function_call_output',
            call_id: 'call:lookup',
            output: '{"error":"weather service unavailable"}',
        });
        expect(JSON.stringify(secondRequest)).not.toContain('_llumiverse_tool_result_status');

        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(second.conversation)));
        const toolTurn = persisted.turns.find((turn) => turn.kind === 'tool');
        expect(toolTurn?.blocks[0]).toMatchObject({ type: 'tool_result', call_id: 'call:lookup', status: 'error' });
        expect(
            Object.values(persisted.execution_receipts).find(
                (receipt) => receipt.executor === 'application' && receipt.call_id === 'call:lookup',
            ),
        ).toMatchObject({ status: 'error' });

        const retried = await driver.execute(resultSegments, {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'answer',
                attempt: 'answer-retry',
                recordedAt: '2026-09-12T01:05:00.000Z',
                conversation: persisted,
            }),
            tools,
        });
        expect(retried.result).toEqual(second.result);
        expect(retried.conversation).toEqual(persisted);
        expect(create).toHaveBeenCalledTimes(2);

        // This is a complete validated context whose optional opaque native replay was removed by
        // source policy. Its application call/result pair remains, in the original order.
        const portable = structuredClone(persisted);
        for (const turn of portable.turns) {
            if (turn.kind === 'agent') turn.blocks = turn.blocks.filter((block) => block.type !== 'native_replay');
            if (turn.kind === 'tool') {
                turn.blocks[0].content = turn.blocks[0].content.filter((block) => block.type !== 'native_replay');
            }
        }
        const portableDocument = parseConversationDocument(portable);
        const target = {
            provider: Providers.openai,
            protocol: OPENAI_RESPONSES_PROTOCOL,
            model: 'gpt-5',
            adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
        };
        const projected = await driver.projectCanonicalModelSwitchRequest(portableDocument, target, 'execute');
        if (projected.status !== 'compiled') throw new Error('Expected complete tool-pair projection');
        responses.push(
            response({
                id: 'response:tool-pair-switch',
                output: [messageItem('message:tool-pair-switch', 'The tool failed.')],
            }),
        );
        await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow: 'tools',
                operation: 'tool-pair-switch',
                attempt: 'first',
                recordedAt: '2026-09-12T01:06:00.000Z',
                conversation: portableDocument,
                model: target.model,
            }),
            conversation: portableDocument,
        });
        expect(projected.native_request).toEqual(create.mock.calls[2]?.[0]);
        expect(projected.native_request).toMatchObject({
            input: expect.arrayContaining([
                expect.objectContaining({ type: 'function_call', call_id: 'call:lookup' }),
                expect.objectContaining({ type: 'function_call_output', call_id: 'call:lookup' }),
            ]),
        });
    });

    it('streams reasoning and text through core, finalizes authoritative output, and recovers without transport', async () => {
        const reasoningItem = {
            type: 'reasoning' as const,
            id: 'reasoning:stream',
            summary: [{ type: 'summary_text' as const, text: 'Check the evidence.' }],
            encrypted_content: 'opaque-stream-reasoning',
            status: 'completed' as const,
        };
        const final = response({
            id: 'response:stream',
            output: [reasoningItem, messageItem('message:stream', 'It is clear.')],
            reasoningTokens: 4,
        });
        const create = vi.fn(async (_request: unknown, _options?: unknown) =>
            (async function* () {
                yield {
                    type: 'response.reasoning_summary_text.delta' as const,
                    item_id: 'reasoning:stream',
                    output_index: 0,
                    summary_index: 0,
                    sequence_number: 1,
                    delta: 'Check the evidence.',
                };
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:stream',
                    output_index: 1,
                    content_index: 0,
                    sequence_number: 2,
                    delta: 'It is clear.',
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 3, response: final };
            })(),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const segments = [{ role: PromptRole.user, content: 'Explain.' }];
        const firstOptions = {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T02:00:00.000Z',
            }),
            model_options: { _option_id: 'openai-thinking' as const },
        };

        const first = await driver.stream(segments, firstOptions);
        expect(await consume(first)).toBe('Check the evidence.\nIt is clear.');
        expect(first.completion?.result).toEqual([
            { type: 'thoughts', value: 'Check the evidence.' },
            { type: 'text', value: 'It is clear.' },
        ]);
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(first.completion?.conversation)));
        expect(JSON.stringify(persisted)).toContain('opaque-stream-reasoning');

        const retried = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'retry',
                recordedAt: '2026-09-12T02:05:00.000Z',
                conversation: persisted,
            }),
            model_options: { _option_id: 'openai-thinking' as const },
        });
        expect(await consume(retried)).toBe('Check the evidence.\nIt is clear.');
        expect(retried.completion?.conversation).toEqual(persisted);
        expect(create).toHaveBeenCalledTimes(1);
    });

    it.each([
        ['truncated', undefined],
        ['failed', 7],
    ] as const)('does not persist a canonical response when a stream is %s', async (failure, billedOutputTokens) => {
        const failed = response({
            id: 'response:failed-stream',
            output: [messageItem('message:failed-stream', 'partial')],
            status: 'failed',
            outputTokens: billedOutputTokens ?? 0,
        });
        const create = vi.fn(async (_request: unknown, _options?: unknown) =>
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:failed-stream',
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 1,
                    delta: 'partial',
                    logprobs: [],
                };
                if (failure === 'failed') {
                    yield { type: 'response.failed' as const, sequence_number: 2, response: failed };
                }
            })(),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({
                flow: `stream-${failure}`,
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T03:00:00.000Z',
            }),
        });

        await expect(consume(stream)).rejects.toThrow(
            failure === 'failed' ? 'provider failed' : 'ended without a final response',
        );
        expect(stream.completion?.conversation).toBeUndefined();
        if (billedOutputTokens !== undefined) {
            expect(stream.completion?.token_usage?.result).toBe(billedOutputTokens);
        }
    });

    it('retains an incomplete native turn and replays it for a later Responses continuation', async () => {
        const partialItem = messageItem('message:partial', 'First,', 'incomplete');
        const partial = response({ id: 'response:partial', output: [partialItem], status: 'incomplete' });
        const completed = response({
            id: 'response:continued',
            output: [messageItem('message:continued', 'the rest follows.')],
        });
        const create = vi.fn(async (request: unknown, _options?: unknown) => {
            if ((request as { stream?: boolean }).stream) {
                return (async function* () {
                    yield {
                        type: 'response.output_text.delta' as const,
                        item_id: 'message:partial',
                        output_index: 0,
                        content_index: 0,
                        sequence_number: 1,
                        delta: 'First,',
                        logprobs: [],
                    };
                    yield { type: 'response.incomplete' as const, sequence_number: 2, response: partial };
                })();
            }
            return completed;
        });
        const driver = new TestOpenAIResponsesDriver(create);
        const first = await driver.stream([{ role: PromptRole.user, content: 'Start an explanation.' }], {
            ...runtimeOptions({
                flow: 'partial',
                operation: 'start',
                attempt: 'start',
                recordedAt: '2026-09-12T04:00:00.000Z',
            }),
        });
        await consume(first);
        expect(first.completion?.finish_reason).toBe('length');
        const partialDocument = parseConversationDocument(first.completion?.conversation);
        expect(partialDocument.turns.at(-1)?.status).toBe('interrupted');

        const continuation = await driver.execute([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({
                flow: 'partial',
                operation: 'continue',
                attempt: 'continue',
                recordedAt: '2026-09-12T04:01:00.000Z',
                conversation: partialDocument,
            }),
        });
        const continuationRequest = create.mock.calls[1]?.[0] as { input?: unknown[] };
        expect(continuationRequest.input).toEqual(expect.arrayContaining([partialItem]));
        expect(continuation.result).toEqual([{ type: 'text', value: 'the rest follows.' }]);
    });

    it('applies request-only history retention and prompt transforms without rewriting canonical source', async () => {
        const nativeHistory = {
            _arrayConversation: [{ role: 'user' as const, content: '<heartbeat>old status</heartbeat>' }],
            _llumiverse_meta: { turnNumber: 20 },
        };
        const currentPrompt = [
            {
                type: 'message' as const,
                role: 'system' as const,
                content: [
                    { type: 'input_text' as const, text: 'Inspect this image.' },
                    {
                        type: 'input_image' as const,
                        image_url: 'data:image/png;base64,aW1hZ2U=',
                        detail: 'auto' as const,
                    },
                ],
            },
        ];
        const create = vi.fn(async (_request: unknown, _options?: unknown) =>
            response({ id: 'response:projection', output: [messageItem('message:projection', 'Done.')], model: 'o1' }),
        );
        const driver = new TestOpenAIResponsesDriver(create);

        const completion = await driver.requestTextCompletion(currentPrompt, {
            ...runtimeOptions({
                flow: 'projection',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T05:00:00.000Z',
                model: 'o1',
            }),
            conversation: nativeHistory,
            stripHeartbeatsAfterTurns: 1,
            model_options: { _option_id: 'openai-thinking', image_detail: 'high' },
        });
        const request = create.mock.calls[0]?.[0] as { input?: unknown[] };
        expect(request.input).toEqual([
            { role: 'user', content: '[Heartbeat removed from conversation history]' },
            {
                role: 'developer',
                content: [
                    { type: 'input_text', text: 'Inspect this image.' },
                    { type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=', detail: 'high' },
                ],
            },
        ]);

        const projected = exportLegacyOpenAIResponsesConversation(parseConversationDocument(completion.conversation));
        expect(projected.slice(0, 2)).toEqual([
            { role: 'user', content: '<heartbeat>old status</heartbeat>' },
            {
                role: 'system',
                content: [
                    { type: 'input_text', text: 'Inspect this image.' },
                    { type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=', detail: 'auto' },
                ],
            },
        ]);
    });

    it('hydrates an authenticated received image only in the native Responses body, retaining external source identity', async () => {
        const png = Buffer.from(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==',
            'base64',
        );
        const at = '2026-09-12T06:00:00.000Z';
        const initial = createConversationDocument({ id: 'conversation:received-image', created_at: at });
        const asset = {
            id: 'asset:received-image',
            kind: 'image' as const,
            mime_type: 'image/png',
            byte_length: png.byteLength,
            content_hash: `sha256:${createHash('sha256').update(png).digest('hex')}`,
            storage: {
                type: 'external' as const,
                resolver: 'vertesia.agent_artifact',
                locator: { storage_id: 'owner', artifact_path: 'archive/assets/one' },
            },
            provenance: { type: 'received' as const },
            created_at: at,
        };
        const appended = appendConversationRecords(
            initial,
            {
                turns: [
                    {
                        id: 'turn:received-image',
                        kind: 'user' as const,
                        authority: 'ordinary' as const,
                        status: 'completed' as const,
                        timestamps: { recorded_at: at },
                        model_visibility: 'include' as const,
                        blocks: [
                            createTextBlock({
                                id: 'block:received-image:text',
                                text: 'Inspect this.',
                                format: 'plain',
                            }),
                            { id: 'block:received-image:image', type: 'image' as const, asset_id: asset.id },
                        ],
                        provenance: { type: 'inserted' as const, operation_id: 'operation:received-image' },
                    },
                ],
                assets: [asset],
                context_entries: [
                    { id: 'context:received-image', type: 'source_turn' as const, turn_id: 'turn:received-image' },
                ],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:received-image',
                payload_fingerprint: await fingerprintJson({ received: 'image' }),
                recorded_at: at,
            },
        ).document;
        const resolve = vi.fn(async function* () {
            yield png;
        });
        const runtime = runtimeOptions({
            flow: 'received-image',
            operation: 'respond',
            attempt: 'first',
            recordedAt: at,
            conversation: appended,
            materializedInput: { operation_id: 'operation:received-image', result_revision: appended.revision },
        });
        const options = {
            ...runtime,
            conversation: appended,
            resolve_canonical_asset: resolve,
        };
        const dry = await prepareOpenAIResponsesCanonicalContext({
            options: resolveCanonicalExecutionContextOptions(options),
            provider: Providers.openai,
        });
        expect(dry.native_conversation).toMatchObject([
            {
                role: 'user',
                content: [
                    { type: 'input_text', text: 'Inspect this.' },
                    { type: 'input_image', image_url: `data:image/png;base64,${png.toString('base64')}` },
                ],
            },
        ]);
        const changedSource = structuredClone(dry);
        changedSource.document.assets[asset.id].content_hash = `sha256:${'0'.repeat(64)}`;
        await expect(
            finalizeOpenAIResponsesPreparedRequest(changedSource, { model: 'gpt-5', input: [], stream: false }),
        ).rejects.toThrow('canonical source changed after native image projection');
        const create = vi.fn(async (_request: unknown) =>
            response({ id: 'response:received-image', output: [messageItem('message:received-image', 'Done.')] }),
        );
        const publish = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(async () => undefined);
        const driver = new TestOpenAIResponsesDriver(create);
        const first = await driver.executeCanonicalContext({ ...options, on_canonical_request_prepared: publish });
        expect(resolve).toHaveBeenCalledTimes(2);
        expect(create).toHaveBeenCalledOnce();
        await expect(
            driver.executeCanonicalContext({
                ...options,
                conversation: first.conversation,
                resolve_canonical_asset: async function* () {
                    yield await Promise.reject<Buffer>(new Error('received image unavailable'));
                },
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow('received image unavailable');
        expect(create).toHaveBeenCalledOnce();
        expect(create.mock.calls[0]?.[0]).toMatchObject({ input: dry.native_conversation });
        const prepared = publish.mock.calls[0]?.[0];
        expect(prepared?.record.request_receipt.request_fingerprint).toBe(
            await fingerprintJson(JSON.parse(JSON.stringify(create.mock.calls[0]?.[0]))),
        );
        expect(prepared?.document.assets[asset.id]).toEqual(asset);
        expect(prepared?.record.request_receipt.asset_versions).toContainEqual({
            asset_id: asset.id,
            content_hash: asset.content_hash,
        });
        const streamCreate = vi.fn(async (_request: unknown) =>
            (async function* () {
                yield {
                    type: 'response.completed' as const,
                    sequence_number: 1,
                    response: response({
                        id: 'response:received-image:stream',
                        output: [messageItem('message:received-image:stream', 'Done.')],
                    }),
                };
            })(),
        );
        const streamDriver = new TestOpenAIResponsesDriver(streamCreate);
        const stream = await streamDriver.streamCanonicalContextEvents(
            { ...options, on_canonical_request_prepared: publish },
            undefined,
            { stream_id: 'stream:received-image' },
        );
        await collectCanonicalEvents(stream);
        expect(streamCreate.mock.calls[0]?.[0]).toMatchObject({ input: dry.native_conversation, stream: true });
        expect(publish).toHaveBeenCalledTimes(2);
        await expect(
            driver.executeCanonicalContext({
                ...options,
                resolve_canonical_asset: async function* () {
                    yield Buffer.from('wrong image bytes');
                },
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow('does not match resolved bytes');
        expect(create).toHaveBeenCalledOnce();
    });
});
