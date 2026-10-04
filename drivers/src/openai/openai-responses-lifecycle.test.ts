import { createHash } from 'node:crypto';
import type { CanonicalProjectedRequestMeasurement } from '@llumiverse/common';
import { deriveCanonicalProjectedMeasurementIdentity } from '@llumiverse/common/schemas';
import {
    appendConversationRecords,
    appendToolExecutionResult,
    applyContextChange,
    type ConversationDocument,
    type ConversationPreparedRequest,
    type ConversationPreparedRequestRecord,
    type ConversationStreamEvent,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    parseConversationDocument,
    planContextChange,
    processingContextFingerprint,
    recordProcessingCoverage,
    toolArgumentsForModel,
} from '@llumiverse/conversation';
import {
    assertCanonicalFailedExecutionMatchesPreparedRecord,
    type CanonicalExecutionInputOptions,
    canonicalFailedExecution,
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
    compileOpenAIResponsesConversation,
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

    it('compacts an actual accepted answer without reactivating its discardable native representation', async () => {
        const originalText = 'Actual accepted terminal answer with immutable native payload.';
        const create = vi.fn(async () =>
            response({
                id: 'response:terminal-compaction',
                output: [messageItem('message:terminal-compaction', originalText)],
            }),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const first = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Answer once.' }],
            runtimeOptions({
                flow: 'terminal-compaction',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
        );
        const answer = first.conversation.turns.find((turn) => turn.id === first.accepted_output.turn.id);
        if (!answer) throw new Error('Actual retained answer absent');
        const entry = first.conversation.context.entries.find((candidate) => candidate.turn_id === answer.id);
        const text = answer.blocks.find((block) => block.type === 'text');
        const replay = answer.blocks.find((block) => block.type === 'native_replay');
        if (!entry || !text || replay?.type !== 'native_replay') throw new Error('Actual answer/replay absent');
        expect(replay.dependency_policy).toBe('discard_on_dependency_change');
        const selection = {
            expected_revision: first.conversation.revision,
            expected_context_revision: first.conversation.context.revision,
            entry_ids: [entry.id],
            selected_entries: [entry],
            selected_block_ids: { [entry.id]: [text.id] },
        };
        const plan = await planContextChange(first.conversation, selection);
        expect(plan.discarded_replay_block_ids).toEqual([replay.id]);
        const replacement = {
            id: 'turn:terminal-summary',
            kind: 'agent' as const,
            authority: 'ordinary' as const,
            status: 'completed' as const,
            model_visibility: 'include' as const,
            timestamps: { recorded_at: '2026-09-12T01:01:00.000Z' },
            blocks: [
                createTextBlock({ id: 'block:terminal-summary', text: 'Compacted accepted answer.', format: 'plain' }),
            ],
            provenance: {
                type: 'derived' as const,
                derivation_id: 'compaction:terminal',
                source_turn_ids: plan.source_turn_ids,
                source_block_ids: plan.source_block_ids,
                source_hash: plan.source_fingerprint,
            },
        };
        const request = {
            ...selection,
            operation_id: 'context:terminal-compaction',
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: '2026-09-12T01:01:00.000Z',
            proposal: {
                kind: 'replace_with_compaction' as const,
                compaction_id: 'compaction:terminal',
                strategy: { id: 'test-compaction', version: '1', configuration_fingerprint: 'sha256:config' },
                replacement_turns: [replacement],
                fidelity: 'heuristic' as const,
                retained_asset_ids: [],
                generation_ids: [],
                placement: { mode: 'first_selected' as const, causal_order: 'contiguous' as const },
            },
        };
        const edited = await applyContextChange(first.conversation, request);
        expect(edited.document.turns).toEqual(first.conversation.turns);
        expect(edited.document.generations).toEqual(first.conversation.generations);
        for (const [id, receipt] of Object.entries(first.conversation.operation_receipts))
            expect(edited.document.operation_receipts[id]).toEqual(receipt);
        const projected = compileOpenAIResponsesConversation(edited.document, { provider: 'openai', model: 'gpt-5' });
        expect(JSON.stringify(projected.conversation)).toContain('Compacted accepted answer.');
        expect(JSON.stringify(projected.conversation)).not.toContain(originalText);
        expect((await applyContextChange(edited.document, request)).document).toEqual(edited.document);
        expect(create).toHaveBeenCalledOnce();
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

    it('appends a fresh authored prompt after hydrating a retained received image for execute and typed stream', async () => {
        const png = Buffer.from(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==',
            'base64',
        );
        const at = '2026-09-12T06:00:00.000Z';
        const initial = createConversationDocument({ id: 'conversation:authored-received-image', created_at: at });
        const asset = {
            id: 'asset:authored-received-image',
            kind: 'image' as const,
            mime_type: 'image/png',
            byte_length: png.byteLength,
            content_hash: `sha256:${createHash('sha256').update(png).digest('hex')}`,
            storage: {
                type: 'external' as const,
                resolver: 'vertesia.agent_artifact',
                locator: { storage_id: 'owner', artifact_path: 'archive/assets/authored' },
            },
            provenance: { type: 'received' as const },
            created_at: at,
        };
        const retained = appendConversationRecords(
            initial,
            {
                turns: [
                    {
                        id: 'turn:authored-received-image',
                        kind: 'user' as const,
                        authority: 'ordinary' as const,
                        status: 'completed' as const,
                        timestamps: { recorded_at: at },
                        model_visibility: 'include' as const,
                        blocks: [
                            createTextBlock({
                                id: 'block:authored-received-image:text',
                                text: 'Inspect this retained picture.',
                                format: 'plain',
                            }),
                            { id: 'block:authored-received-image:image', type: 'image' as const, asset_id: asset.id },
                        ],
                        provenance: { type: 'inserted' as const, operation_id: 'operation:authored-received-image' },
                    },
                ],
                assets: [asset],
                context_entries: [
                    {
                        id: 'context:authored-received-image',
                        type: 'source_turn' as const,
                        turn_id: 'turn:authored-received-image',
                    },
                ],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:authored-received-image',
                payload_fingerprint: await fingerprintJson({ received: 'image' }),
                recorded_at: at,
            },
        ).document;
        const expectedImage = `data:image/png;base64,${png.toString('base64')}`;
        const executeCreate = vi.fn(async (_request: unknown) =>
            response({ id: 'response:authored-image', output: [messageItem('message:authored-image', 'Done.')] }),
        );
        const executePublish = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(
            async () => undefined,
        );
        const executeResolve = vi.fn(async function* () {
            yield png;
        });
        const executeDriver = new TestOpenAIResponsesDriver(executeCreate);
        const execute = await executeDriver.executeCanonical(
            [{ role: PromptRole.user, content: 'And now describe its color.' }],
            {
                ...runtimeOptions({
                    flow: 'authored-received-image',
                    operation: 'execute',
                    attempt: 'first',
                    recordedAt: at,
                    conversation: retained,
                }),
                on_canonical_request_prepared: executePublish,
            },
            undefined,
            { resolve_canonical_asset: executeResolve },
        );
        expect(executeResolve).toHaveBeenCalledOnce();
        expect(executeCreate).toHaveBeenCalledOnce();
        expect(executeCreate.mock.calls[0]?.[0]).toMatchObject({
            input: [
                {
                    role: 'user',
                    content: [
                        { type: 'input_text', text: 'Inspect this retained picture.' },
                        { type: 'input_image', image_url: expectedImage },
                    ],
                },
                { role: 'user', content: 'And now describe its color.' },
            ],
        });
        expect(executePublish.mock.calls[0]?.[0]?.record.request_receipt.request_fingerprint).toBe(
            await fingerprintJson(JSON.parse(JSON.stringify(executeCreate.mock.calls[0]?.[0]))),
        );
        expect(executePublish.mock.calls[0]?.[0]?.document.assets[asset.id]).toEqual(asset);
        expect(execute.conversation.assets[asset.id]).toEqual(asset);
        const streamCreate = vi.fn(async (_request: unknown) =>
            (async function* () {
                yield {
                    type: 'response.completed' as const,
                    sequence_number: 1,
                    response: response({
                        id: 'response:authored-image:stream',
                        output: [messageItem('message:authored-image:stream', 'Done.')],
                    }),
                };
            })(),
        );
        const streamResolve = vi.fn(async function* () {
            yield png;
        });
        const streamDriver = new TestOpenAIResponsesDriver(streamCreate);
        const stream = await streamDriver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'And now describe its color.' }],
            runtimeOptions({
                flow: 'authored-received-image',
                operation: 'stream',
                attempt: 'first',
                recordedAt: at,
                conversation: retained,
            }),
            undefined,
            { stream_id: 'stream:authored-received-image' },
            { resolve_canonical_asset: streamResolve },
        );
        await collectCanonicalEvents(stream);
        expect(streamResolve).toHaveBeenCalledOnce();
        expect(streamCreate.mock.calls[0]?.[0]).toMatchObject({
            input: [
                {
                    role: 'user',
                    content: [
                        { type: 'input_text', text: 'Inspect this retained picture.' },
                        { type: 'input_image', image_url: expectedImage },
                    ],
                },
                { role: 'user', content: 'And now describe its color.' },
            ],
            stream: true,
        });
        await expect(
            executeDriver.executeCanonical(
                [{ role: PromptRole.user, content: 'And now describe its color.' }],
                runtimeOptions({
                    flow: 'authored-received-image',
                    operation: 'unresolved',
                    attempt: 'first',
                    recordedAt: at,
                    conversation: retained,
                }),
            ),
        ).rejects.toThrow('has no host resolver');
        expect(executeCreate).toHaveBeenCalledOnce();
        const cancelled = new AbortController();
        const cancelResolve = vi.fn(async function* () {
            cancelled.abort(new Error('authored image cancelled'));
            yield png;
        });
        await expect(
            executeDriver.executeCanonical(
                [{ role: PromptRole.user, content: 'And now describe its color.' }],
                runtimeOptions({
                    flow: 'authored-received-image',
                    operation: 'cancelled',
                    attempt: 'first',
                    recordedAt: at,
                    conversation: retained,
                }),
                cancelled.signal,
                { resolve_canonical_asset: cancelResolve },
            ),
        ).rejects.toThrow('authored image cancelled');
        expect(cancelResolve).toHaveBeenCalledOnce();
        expect(executeCreate).toHaveBeenCalledOnce();
        expect(executePublish).toHaveBeenCalledOnce();
    });

    it('hydrates a selected image nested in a tool result without changing its canonical asset', async () => {
        const png = Buffer.from(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==',
            'base64',
        );
        const at = '2026-09-12T06:00:00.000Z';
        const initial = createConversationDocument({ id: 'conversation:nested-received-image', created_at: at });
        const callBlock = {
            id: 'block:nested-call',
            type: 'tool_call' as const,
            call_id: 'call:nested-image',
            tool_name: 'lookup_weather',
            executor: 'application' as const,
            arguments: { type: 'json' as const, value: { city: 'Tokyo' } },
        };
        const withCall = appendConversationRecords(
            initial,
            {
                turns: [
                    {
                        id: 'turn:nested-call',
                        kind: 'agent' as const,
                        authority: 'ordinary' as const,
                        status: 'completed' as const,
                        timestamps: { recorded_at: at },
                        model_visibility: 'include' as const,
                        blocks: [callBlock],
                        provenance: { type: 'imported' as const, source: 'test' },
                    },
                ],
                context_entries: [
                    { id: 'context:nested-call', type: 'source_turn' as const, turn_id: 'turn:nested-call' },
                ],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:nested-call',
                payload_fingerprint: await fingerprintJson({ call: 'nested-image' }),
                recorded_at: at,
            },
        ).document;
        const asset = {
            id: 'asset:nested-received-image',
            kind: 'image' as const,
            mime_type: 'image/png',
            byte_length: png.byteLength,
            content_hash: `sha256:${createHash('sha256').update(png).digest('hex')}`,
            storage: {
                type: 'external' as const,
                resolver: 'vertesia.agent_artifact',
                locator: { storage_id: 'owner', artifact_path: 'archive/assets/nested' },
            },
            provenance: { type: 'received' as const, source_turn_id: 'turn:nested-result' },
            created_at: at,
        };
        const resultBlock = {
            id: 'block:nested-result',
            type: 'tool_result' as const,
            call_id: callBlock.call_id,
            status: 'success' as const,
            content: [{ id: 'block:nested-image', type: 'image' as const, asset_id: asset.id }],
        };
        const source = {
            conversation: { conversation_id: withCall.id, revision: withCall.revision },
            turn_id: 'turn:nested-call',
            block_id: callBlock.id,
            call_id: callBlock.call_id,
            call_fingerprint: await fingerprintJson(callBlock),
        };
        const accepted = await appendToolExecutionResult(
            withCall,
            {
                source,
                turn: {
                    id: 'turn:nested-result',
                    kind: 'tool' as const,
                    authority: 'ordinary' as const,
                    status: 'completed' as const,
                    timestamps: { recorded_at: at },
                    model_visibility: 'include' as const,
                    blocks: [resultBlock],
                    execution_id: 'execution:nested-image',
                    provenance: { type: 'received' as const },
                },
                assets: [asset],
                execution_receipt: {
                    id: 'execution:nested-image',
                    call_id: callBlock.call_id,
                    executor: 'application' as const,
                    status: 'success' as const,
                    result_turn_id: 'turn:nested-result',
                    result_fingerprint: await fingerprintJson(resultBlock),
                    recorded_at: at,
                    call_source: source,
                },
            },
            {
                expected_revision: withCall.revision,
                operation_id: 'operation:nested-result',
                recorded_at: at,
            },
        );
        const resolve = vi.fn(async function* () {
            yield png;
        });
        const prepared = await prepareOpenAIResponsesCanonicalContext({
            options: resolveCanonicalExecutionContextOptions({
                ...runtimeOptions({
                    flow: 'nested-received-image',
                    operation: 'respond',
                    attempt: 'first',
                    recordedAt: at,
                    conversation: accepted.document,
                }),
                conversation: accepted.document,
                resolve_canonical_asset: resolve,
            }),
            provider: Providers.openai,
        });
        expect(resolve).toHaveBeenCalledOnce();
        expect(prepared.native_conversation).toContainEqual(
            expect.objectContaining({
                type: 'function_call_output',
                call_id: callBlock.call_id,
                output: [
                    expect.objectContaining({
                        type: 'input_image',
                        image_url: `data:image/png;base64,${png.toString('base64')}`,
                    }),
                ],
            }),
        );
        expect(prepared.document.assets[asset.id]).toEqual(asset);
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

describe('received canonical Responses failure evidence', () => {
    it.each(['finite', 'stream'] as const)(
        'retains actual %s failure bytes without accepting a response',
        async (transport) => {
            for (const hasOutput of [false, true]) {
                const native = response({
                    id: `response:failed:${transport}:${hasOutput}`,
                    status: 'failed',
                    output: hasOutput ? [messageItem('failed:message', 'Received partial output.', 'incomplete')] : [],
                });
                let record: ConversationPreparedRequestRecord | undefined;
                let source: ConversationDocument | undefined;
                const options: CanonicalExecutionInputOptions = {
                    ...runtimeOptions({
                        flow: `failed:${transport}:${hasOutput}`,
                        operation: 'generate',
                        attempt: 'first',
                        recordedAt: '2026-10-04T00:00:00.000Z',
                    }),
                    on_canonical_request_prepared: async (prepared) => {
                        record = prepared.record;
                        source = prepared.document;
                    },
                };
                const create = vi.fn(async () =>
                    transport === 'finite'
                        ? native
                        : (async function* () {
                              yield { type: 'response.failed' as const, sequence_number: 1, response: native };
                          })(),
                );
                const driver = new TestOpenAIResponsesDriver(create);
                let failure: unknown;
                if (transport === 'finite') {
                    try {
                        await driver.executeCanonical([{ role: PromptRole.user, content: 'Continue.' }], options);
                    } catch (error: unknown) {
                        failure = error;
                    }
                } else {
                    const stream = await driver.streamCanonicalEvents(
                        [{ role: PromptRole.user, content: 'Continue.' }],
                        options,
                        undefined,
                        { stream_id: `stream:failed:${hasOutput}` },
                    );
                    const events = await collectCanonicalEvents(stream);
                    expect(events.some((event) => event.type === 'response_accepted')).toBe(false);
                    expect(stream.terminal_event).toMatchObject({ type: 'stream_terminated', outcome: 'failed' });
                    expect(stream.completion).toBeUndefined();
                    failure = stream.failure;
                    await stream.closed;
                }
                expect(failure).toBeDefined();
                const evidence = canonicalFailedExecution(failure);
                if (!evidence || !record || !source)
                    throw new Error('Actual provider failure lost its durably prepared evidence');
                await assertCanonicalFailedExecutionMatchesPreparedRecord(evidence, record);
                const generation = evidence.decoded_response.generation;
                expect(generation).toMatchObject({
                    status: 'failed',
                    provider_response_id: native.id,
                    finish_reason: 'server_error',
                    usage: { input_tokens: 11, output_tokens: 7, total_tokens: 18 },
                    request_receipt: record.request_receipt,
                    metadata: { openai_responses_failure: native },
                });
                expect(evidence.prepared_request).toEqual(record);
                expect(source.generations[record.generation_id]).toBeUndefined();
                expect(source.operation_receipts[record.runtime.response_operation_id]).toBeUndefined();
                const failedTurns = evidence.decoded_response.turns;
                expect(failedTurns).toHaveLength(hasOutput ? 1 : 0);
                expect(
                    failedTurns.every((turn) => turn.status === 'failed' && turn.model_visibility === 'exclude'),
                ).toBe(true);
                expect(
                    source.context.entries.some(
                        (entry) =>
                            entry.type === 'source_turn' && failedTurns.some((turn) => turn.id === entry.turn_id),
                    ),
                ).toBe(false);
                expect(create).toHaveBeenCalledOnce();
                const changed = { ...record, runtime: { ...record.runtime, attempt_id: 'foreign:attempt' } };
                await expect(assertCanonicalFailedExecutionMatchesPreparedRecord(evidence, changed)).rejects.toThrow();
                expect(canonicalFailedExecution(new Error('caller metadata', { cause: { evidence } }))).toBeUndefined();
            }
        },
    );

    it('retains absent usage as unavailable rather than a fabricated zero', async () => {
        const native = { ...response({ id: 'failed:unknown-usage', status: 'failed', output: [] }), usage: null };
        const driver = new TestOpenAIResponsesDriver(async () => native);
        let failure: unknown;
        try {
            await driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Continue.' }],
                runtimeOptions({
                    flow: 'unknown-usage',
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-10-04T00:00:00.000Z',
                }),
            );
        } catch (error: unknown) {
            failure = error;
        }
        const evidence = canonicalFailedExecution(failure);
        if (!evidence) throw new Error('Received failed response has no execution evidence');
        const generation = evidence.decoded_response.generation;
        expect(generation?.usage).toBeUndefined();
    });

    it.each(['finite', 'stream'] as const)(
        'retains %s partial function arguments as invalid failed evidence, never repaired calls',
        async (transport) => {
            const native = response({
                id: `failed:partial-tool:${transport}`,
                status: 'failed',
                output: [
                    {
                        type: 'function_call',
                        id: 'partial:call',
                        call_id: 'call:partial',
                        name: 'read_artifact',
                        arguments: '{"path":',
                        status: 'incomplete',
                    },
                ],
            });
            const driver = new TestOpenAIResponsesDriver(async () =>
                transport === 'finite'
                    ? native
                    : (async function* () {
                          yield { type: 'response.failed' as const, sequence_number: 1, response: native };
                      })(),
            );
            const options = runtimeOptions({
                flow: `partial-tool:${transport}`,
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-10-04T00:00:00.000Z',
            });
            let failure: unknown;
            if (transport === 'finite') {
                try {
                    await driver.executeCanonical([{ role: PromptRole.user, content: 'Continue.' }], options);
                } catch (error: unknown) {
                    failure = error;
                }
            } else {
                const stream = await driver.streamCanonicalEvents(
                    [{ role: PromptRole.user, content: 'Continue.' }],
                    options,
                    undefined,
                    { stream_id: `stream:partial-tool:${transport}` },
                );
                const events = await collectCanonicalEvents(stream);
                expect(events.some((event) => event.type === 'response_accepted')).toBe(false);
                expect(stream.completion).toBeUndefined();
                failure = stream.failure;
                await stream.closed;
            }
            const evidence = canonicalFailedExecution(failure);
            if (!evidence) throw new Error('Received partial failed native payload lost its provenance');
            expect(evidence.decoded_response.turns).toHaveLength(1);
            const failedTurn = evidence.decoded_response.turns[0];
            expect(failedTurn).toMatchObject({ kind: 'agent', status: 'failed', model_visibility: 'exclude' });
            if (failedTurn.kind !== 'agent') throw new Error('Actual failed response lost its agent turn');
            const call = failedTurn.blocks.find((block) => block.type === 'tool_call');
            if (call?.type !== 'tool_call') throw new Error('Actual partial failed call is unavailable');
            expect(call.arguments).toMatchObject({ type: 'invalid', raw: '{"path":' });
            expect(() => toolArgumentsForModel(call.arguments)).toThrow('Invalid tool arguments');
            expect(failedTurn.blocks.some((block) => block.type === 'native_replay')).toBe(true);
            expect('accepted_output' in evidence).toBe(false);

            expect(evidence.decoded_response.generation).toMatchObject({
                status: 'failed',
                finish_reason: 'server_error',
                usage: { input_tokens: 11, output_tokens: 7, total_tokens: 18 },
                metadata: { openai_responses_failure: native },
            });
            expect(evidence.decoded_response.payload_fingerprint).toBe(await fingerprintJson(native));
        },
    );

    it('does not lower the normal prepared-source bound for a paid failed native result', async () => {
        const native = response({ id: 'failed:large-prepared-source', status: 'failed', output: [] });
        const driver = new TestOpenAIResponsesDriver(async () => native);
        let source: ConversationDocument | undefined;
        let failure: unknown;
        try {
            await driver.executeCanonical([{ role: PromptRole.user, content: 'x'.repeat(17 * 1024 * 1024) }], {
                ...runtimeOptions({
                    flow: 'large-prepared',
                    operation: 'generate',
                    attempt: 'first',
                    recordedAt: '2026-10-04T00:00:00.000Z',
                }),
                on_canonical_request_prepared: async (prepared) => {
                    source = prepared.document;
                },
            });
        } catch (error: unknown) {
            failure = error;
        }
        const evidence = canonicalFailedExecution(failure);
        if (!source || !evidence) throw new Error('Valid large preparation lost failed execution facts');
        expect(JSON.stringify(source).length).toBeGreaterThan(16 * 1024 * 1024);
        expect(JSON.stringify(evidence).length).toBeLessThan(64 * 1024);
        expect(evidence.decoded_response.generation.usage?.total_tokens).toBe(18);
        expect(evidence.decoded_response.turns).toHaveLength(0);
        expect(source.generations[evidence.prepared_request.generation_id]).toBeUndefined();
    });

    it('retains received failed facts from an enabled processing source without changing policy or creating append work', async () => {
        const original = materializedInputDocument();
        const create = vi.fn(async () => response({ id: 'failed:processing-enabled', status: 'failed', output: [] }));
        const driver = new TestOpenAIResponsesDriver(create);
        const captures: ConversationPreparedRequest[] = [];
        let counted: CanonicalProjectedRequestMeasurement | undefined;
        const options = runtimeOptions({
            flow: 'materialized-driver-retry',
            operation: 'failed-processing',
            attempt: 'first',
            recordedAt: '2026-09-12T00:02:00.000Z',
            conversation: original,
        });
        await expect(
            driver.executeCanonicalContext({
                ...options,
                conversation: original,
                on_canonical_request_projected: async (projection) => {
                    counted = {
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
                        readiness: { profile: 'test-processing', context_limit: 1000, output_reserve_tokens: 100 },
                    };
                    return counted;
                },
                on_canonical_request_prepared: async (prepared) => {
                    captures.push(prepared);
                    throw new Error('Captured dry native preparation');
                },
            }),
        ).rejects.toThrow('Captured dry native preparation');
        expect(create).not.toHaveBeenCalled();
        const dry = captures[0];
        if (!dry || !counted || counted.measurement.tokenizer === undefined)
            throw new Error('Actual native count/preparation is unavailable');
        const enabled = structuredClone(original);
        enabled.processing.enabled = true;
        const ready = (
            await recordProcessingCoverage(enabled, {
                expected_revision: enabled.revision,
                operation_id: 'processing:failed-native:coverage',
                target_fingerprint: await fingerprintJson(dry.record.request_receipt.target),
                measured_input_tokens: counted.measurement.input_tokens,
                tokenizer_id: counted.measurement.tokenizer,
                measurement_fingerprint: await deriveCanonicalProjectedMeasurementIdentity(
                    counted,
                    dry.record.request_receipt.target,
                ),
                recorded_at: '2026-09-12T00:02:00.000Z',
            })
        ).document;
        const before = structuredClone(ready);
        let failure: unknown;
        try {
            await driver.executeCanonicalContext({
                ...options,
                conversation: ready,
                on_canonical_request_projected: async () => counted,
                on_canonical_request_prepared: async (prepared) => {
                    captures.push(prepared);
                },
            });
        } catch (error: unknown) {
            failure = error;
        }
        const evidence = canonicalFailedExecution(failure);
        if (!evidence) throw new Error('Processing-enabled received failure lost authentic evidence');
        expect(create).toHaveBeenCalledOnce();
        expect(ready).toEqual(before);
        expect(captures.at(-1)?.document.processing.enabled).toBe(true);
        expect(evidence.decoded_response.generation.usage?.total_tokens).toBe(18);
        expect(evidence.decoded_response.turns).toEqual([]);
        expect(Object.keys(ready.processing.jobs ?? {})).toEqual([]);
    });

    it('does not grant failure evidence to a truncated transport without a received failed response', async () => {
        const driver = new TestOpenAIResponsesDriver(async () =>
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    sequence_number: 1,
                    output_index: 0,
                    content_index: 0,
                    item_id: 'draft:only',
                    delta: 'unconfirmed',
                    logprobs: [],
                };
            })(),
        );
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Continue.' }],
            runtimeOptions({
                flow: 'truncated-failure',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-10-04T00:00:00.000Z',
            }),
            undefined,
            { stream_id: 'stream:truncated' },
        );
        await collectCanonicalEvents(stream);
        expect(stream.completion).toBeUndefined();
        expect(canonicalFailedExecution(stream.failure)).toBeUndefined();
        await stream.closed;
    });
});

it.each(['finite', 'stream'] as const)(
    'retains %s truly unconvertible received output as generation-only failure',
    async (transport) => {
        const native = {
            ...response({ id: `failed:unconvertible:${transport}`, status: 'failed', output: [] }),
            // A received terminal can be incomplete beyond the SDK's declared successful item shape.
            output: [{ type: 'function_call', id: 'partial:item', arguments: '{"path":', status: 'incomplete' }],
        };
        const driver = new TestOpenAIResponsesDriver(async () =>
            transport === 'finite'
                ? native
                : (async function* () {
                      yield { type: 'response.failed', sequence_number: 1, response: native };
                  })(),
        );
        const options = runtimeOptions({
            flow: `unconvertible:${transport}`,
            operation: 'generate',
            attempt: 'first',
            recordedAt: '2026-10-04T00:00:00.000Z',
        });
        let failure: unknown;
        if (transport === 'finite') {
            try {
                await driver.executeCanonical([{ role: PromptRole.user, content: 'Continue.' }], options);
            } catch (error: unknown) {
                failure = error;
            }
        } else {
            const stream = await driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'Continue.' }],
                options,
                undefined,
                { stream_id: `stream:unconvertible:${transport}` },
            );
            const events = await collectCanonicalEvents(stream);
            expect(events.some((event) => event.type === 'response_accepted')).toBe(false);
            expect(stream.completion).toBeUndefined();
            failure = stream.failure;
            await stream.closed;
        }
        const evidence = canonicalFailedExecution(failure);
        if (!evidence) throw new Error('Actual incomplete terminal lost its real failed generation');
        expect(evidence.decoded_response.turns).toEqual([]);
        expect(evidence.decoded_response.payload_fingerprint).toBe(await fingerprintJson(native));
        expect(evidence.decoded_response.generation).toMatchObject({
            status: 'failed',
            finish_reason: 'server_error',
            usage: { input_tokens: 11, output_tokens: 7, total_tokens: 18 },
            metadata: { openai_responses_failure: native },
        });
    },
);
