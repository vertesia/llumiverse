import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import {
    type ConversationDocument,
    type ConversationStreamEvent,
    fingerprintJson,
    parseConversationDocument,
    resolveToolExecutionRequest,
} from '@llumiverse/conversation';
import { type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from '../anthropic/index.js';
import {
    type ClaudePrompt,
    executeClaudeCompletion,
    formatClaudePrompt,
    pruneClaudeThinking,
    streamClaudeCompletion,
} from './claude-messages.js';
import { exportLegacyClaudeMessagesConversation } from './claude-messages-conversation-adapter.js';
import { claudeFinishReason, logClaudeTruncation } from './claude-stop-reason.js';

function sdkStream(events: RawMessageStreamEvent[], finalMessage: Message) {
    return {
        async *[Symbol.asyncIterator]() {
            for (const event of events) yield event;
        },
        async finalMessage() {
            return finalMessage;
        },
        abort() {},
    };
}

const finalToolMessage = {
    id: 'msg-1',
    type: 'message',
    role: 'assistant',
    model: 'claude-sonnet-4-6',
    content: [
        { type: 'thinking', thinking: 'plan', signature: 'signed-thinking' },
        { type: 'redacted_thinking', data: 'encrypted-redaction' },
        { type: 'text', text: 'checking', citations: null },
        { type: 'tool_use', id: 'call-1', name: 'lookup', input: { city: 'Paris' } },
    ],
    stop_reason: 'tool_use',
    stop_sequence: null,
    usage: { input_tokens: 2, output_tokens: 3 },
} as unknown as Message;

function clientFor(message: Message, events: RawMessageStreamEvent[] = []) {
    return { messages: { stream: () => sdkStream(events, message) } } as never;
}

const prompt: ClaudePrompt = { messages: [{ role: 'user', content: 'question' }] };
const expectedFinalToolContent = [
    { type: 'thinking', thinking: 'plan', signature: 'signed-thinking' },
    { type: 'redacted_thinking', data: 'encrypted-redaction' },
    { type: 'text', text: 'checking' },
    { type: 'tool_use', id: 'call-1', name: 'lookup', input: { city: 'Paris' } },
];

function legacyConversation(value: unknown): ClaudePrompt {
    return exportLegacyClaudeMessagesConversation(value as ConversationDocument);
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

function canonicalOptions(attempt: string, recordedAt: string, conversation?: unknown): ExecutionOptions {
    return {
        model: 'claude-sonnet-4-6',
        ...(conversation === undefined ? {} : { conversation }),
        conversation_runtime: {
            conversation_id: 'conversation:claude-structured',
            request_id: 'request:claude-structured',
            attempt_id: attempt,
            input_operation_id: 'input:claude-structured',
            response_operation_id: 'response:claude-structured',
            recorded_at: recordedAt,
            started_at: recordedAt,
        },
    };
}

describe('Claude native reasoning replay', () => {
    it('emits structured typed events with native content indices, exact recovery, and legacy acceptance parity', async () => {
        const rawText = '{"answer":"Tokyo"}';
        const finalMessage = {
            id: 'msg-typed-structured',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{ type: 'text', text: rawText }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 2, output_tokens: 3, service_tier: 'priority' },
        } as unknown as Message;
        const nativeEvents = [
            { type: 'message_start', message: { ...finalMessage, content: [], stop_reason: null } },
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '', citations: null },
            },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '{"answer":' } },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '"Tokyo"}' } },
            { type: 'content_block_stop', index: 0 },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
            { type: 'message_stop' },
        ] as RawMessageStreamEvent[];
        const providerCall = vi.fn(() => sdkStream(nativeEvents, finalMessage));
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const executionOptions = {
            ...canonicalOptions('attempt:typed:structured', '2026-09-11T00:00:00.000Z'),
            result_schema: resultSchema,
        };
        const typed = await driver.streamCanonicalEvents(segments, executionOptions, undefined, {
            stream_id: 'stream:claude:typed-structured',
        });
        const events = await collectCanonicalEvents(typed);

        expect(events.filter((event) => event.type === 'draft_text_delta')).toMatchObject([
            {
                text: '{"answer":',
                native_position: { protocol: 'anthropic.messages', path: ['content', 0] },
            },
            { text: '"Tokyo"}' },
        ]);
        expect(events.find((event) => event.type === 'response_accepted')).toMatchObject({
            origin: 'live_transport',
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

        const legacyDriver = new AnthropicDriver({ apiKey: 'test' });
        legacyDriver.client = { messages: { stream: providerCall } } as never;
        const legacy = await legacyDriver.streamCanonical(segments, executionOptions);
        for await (const _chunk of legacy) {
            // Drain the explicit legacy preview boundary.
        }
        expect(acceptedOutputWithoutProviderTimestamps(typed.completion?.accepted_output)).toEqual(
            acceptedOutputWithoutProviderTimestamps(legacy.completion?.accepted_output),
        );

        if (typed.completion === undefined) throw new Error('Expected typed Claude completion');
        const recovered = await driver.streamCanonicalEvents(
            segments,
            {
                ...canonicalOptions(
                    'attempt:typed:structured:retry',
                    '2026-09-11T00:01:00.000Z',
                    JSON.parse(JSON.stringify(typed.completion.conversation)),
                ),
                result_schema: resultSchema,
            },
            undefined,
            { stream_id: 'stream:claude:typed-structured:delivery-2' },
        );
        const recoveredEvents = await collectCanonicalEvents(recovered);
        expect(recoveredEvents).toEqual([
            expect.objectContaining({
                type: 'response_accepted',
                origin: 'accepted_recovery',
                stream_id: 'stream:claude:typed-structured:delivery-2',
                sequence: 0,
            }),
        ]);
        expect(recovered.completion?.accepted_output).toEqual(typed.completion.accepted_output);
        expect(providerCall).toHaveBeenCalledTimes(2);
    });

    it('keeps signed and redacted reasoning replay out of typed display events', async () => {
        const finalMessage = {
            ...finalToolMessage,
            id: 'msg-typed-reasoning',
            content: [
                { type: 'thinking', thinking: 'visible plan', signature: 'secret-signature' },
                { type: 'redacted_thinking', data: 'opaque-redaction' },
                { type: 'text', text: 'answer' },
            ],
            stop_reason: 'end_turn',
        } as unknown as Message;
        const nativeEvents = [
            { type: 'message_start', message: { ...finalMessage, content: [], stop_reason: null } },
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'thinking', thinking: '', signature: '' },
            },
            { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'visible plan' } },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'signature_delta', signature: 'secret-signature' },
            },
            { type: 'content_block_stop', index: 0 },
            {
                type: 'content_block_start',
                index: 1,
                content_block: { type: 'redacted_thinking', data: 'opaque-redaction' },
            },
            { type: 'content_block_stop', index: 1 },
            {
                type: 'content_block_start',
                index: 2,
                content_block: { type: 'text', text: '', citations: null },
            },
            { type: 'content_block_delta', index: 2, delta: { type: 'text_delta', text: 'answer' } },
            { type: 'content_block_stop', index: 2 },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
            { type: 'message_stop' },
        ] as RawMessageStreamEvent[];
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: () => sdkStream(nativeEvents, finalMessage) } } as never;
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Explain.' }],
            {
                ...canonicalOptions('attempt:typed:reasoning', '2026-09-11T00:00:00.000Z'),
                model_options: { _option_id: 'anthropic-claude', include_thoughts: false },
            },
            undefined,
            { stream_id: 'stream:claude:typed-reasoning' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events).toContainEqual(expect.objectContaining({ type: 'draft_reasoning_delta', text: 'visible plan' }));
        expect(JSON.stringify(events)).not.toContain('secret-signature');
        expect(JSON.stringify(events)).not.toContain('opaque-redaction');
        expect(stream.completion?.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'reasoning', text: 'visible plan' }),
                expect.objectContaining({ type: 'text', text: 'answer' }),
            ]),
        );
    });

    it('reconciles streamed tool JSON fragments to the authoritative Claude call identity', async () => {
        const finalMessage = {
            ...finalToolMessage,
            id: 'msg-typed-tool',
            content: [{ type: 'tool_use', id: 'call-typed', name: 'lookup', input: { city: 'Tokyo' } }],
        } as unknown as Message;
        const nativeEvents = [
            { type: 'message_start', message: { ...finalMessage, content: [], stop_reason: null } },
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'tool_use', id: 'call-typed', name: 'lookup', input: {} },
            },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'input_json_delta', partial_json: '{"city":' },
            },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'input_json_delta', partial_json: '"Tokyo"}' },
            },
            { type: 'content_block_stop', index: 0 },
            {
                type: 'message_delta',
                delta: { stop_reason: 'tool_use', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
            { type: 'message_stop' },
        ] as RawMessageStreamEvent[];
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: () => sdkStream(nativeEvents, finalMessage) } } as never;
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Look up Tokyo.' }],
            {
                ...canonicalOptions('attempt:typed:tool', '2026-09-11T00:00:00.000Z'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
            undefined,
            { stream_id: 'stream:claude:typed-tool' },
        );
        const events = await collectCanonicalEvents(stream);
        const call = stream.completion?.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');

        expect(events.filter((event) => event.type === 'draft_tool_arguments_delta')).toMatchObject([
            { arguments: { encoding: 'json_fragment', fragment: '{"city":' } },
            { arguments: { encoding: 'json_fragment', fragment: '"Tokyo"}' } },
        ]);
        expect(call).toMatchObject({
            type: 'tool_call',
            call_id: 'call-typed',
            tool_name: 'lookup',
            executor: 'application',
            arguments: { type: 'json', value: { city: 'Tokyo' } },
        });
        expect(events.find((event) => event.type === 'response_accepted')).toMatchObject({
            reconciliations: [{ disposition: 'direct', committed_block_ids: [call?.id] }],
        });
    });

    it.each([
        ['initial snapshot', { city: 'Tokyo' }],
        ['terminal snapshot', {}],
    ] as const)('emits a typed tool argument snapshot for %s without JSON fragments', async (_label, initialInput) => {
        const finalMessage = {
            ...finalToolMessage,
            id: `msg-typed-tool-${_label}`,
            content: [{ type: 'tool_use', id: 'call-snapshot', name: 'lookup', input: { city: 'Tokyo' } }],
        } as unknown as Message;
        const nativeEvents = [
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'tool_use', id: 'call-snapshot', name: 'lookup', input: initialInput },
            },
            { type: 'content_block_stop', index: 0 },
            {
                type: 'message_delta',
                delta: { stop_reason: 'tool_use', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
        ] as RawMessageStreamEvent[];
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: () => sdkStream(nativeEvents, finalMessage) } } as never;
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Look up Tokyo.' }],
            {
                ...canonicalOptions(`attempt:typed:tool-${_label}`, '2026-09-11T00:00:00.000Z'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
            undefined,
            { stream_id: `stream:claude:typed-tool-${_label}` },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events.filter((event) => event.type === 'draft_tool_arguments_delta')).toEqual([
            expect.objectContaining({
                arguments: { encoding: 'json_value_snapshot', value: { city: 'Tokyo' } },
            }),
        ]);
        expect(events.at(-1)?.type).toBe('response_accepted');
    });

    it('retains the terminal tool input but rejects malformed streamed argument fragments', async () => {
        const finalMessage = {
            ...finalToolMessage,
            id: 'msg-typed-tool-malformed-preview',
            content: [{ type: 'tool_use', id: 'call-malformed-preview', name: 'lookup', input: { city: 'Tokyo' } }],
        } as unknown as Message;
        const nativeEvents = [
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'tool_use', id: 'call-malformed-preview', name: 'lookup', input: {} },
            },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'input_json_delta', partial_json: '{"city":' },
            },
            { type: 'content_block_stop', index: 0 },
            {
                type: 'message_delta',
                delta: { stop_reason: 'tool_use', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
        ] as RawMessageStreamEvent[];
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: () => sdkStream(nativeEvents, finalMessage) } } as never;
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Look up Tokyo.' }],
            {
                ...canonicalOptions('attempt:typed:tool-malformed-preview', '2026-09-11T00:00:00.000Z'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
            undefined,
            { stream_id: 'stream:claude:typed-tool-malformed-preview' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'CANONICAL_EVENT_DELIVERY_FAILED' },
        });
        expect(events.some((event) => event.type === 'response_accepted')).toBe(false);
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({
                type: 'tool_call',
                arguments: { type: 'json', value: { city: 'Tokyo' } },
            }),
        );
    });

    it.each(['max_tokens', 'model_context_window_exceeded'] as const)(
        'keeps sync, legacy stream, and typed %s tool calls identically non-executable',
        async (stopReason) => {
            const finalMessage = {
                ...finalToolMessage,
                id: `msg-typed-tool-${stopReason}`,
                content: [{ type: 'tool_use', id: 'call-interrupted', name: 'lookup', input: { city: 'Tokyo' } }],
                stop_reason: stopReason,
            } as unknown as Message;
            const nativeEvents = [
                {
                    type: 'content_block_start',
                    index: 0,
                    content_block: { type: 'tool_use', id: 'call-interrupted', name: 'lookup', input: {} },
                },
                {
                    type: 'content_block_delta',
                    index: 0,
                    delta: { type: 'input_json_delta', partial_json: '{"city":"Tokyo"}' },
                },
                { type: 'content_block_stop', index: 0 },
                {
                    type: 'message_delta',
                    delta: { stop_reason: stopReason, stop_sequence: null },
                    usage: { output_tokens: 3 },
                },
            ] as RawMessageStreamEvent[];
            const segments = [{ role: PromptRole.user, content: 'Look up Tokyo.' }];
            const options = {
                ...canonicalOptions(`attempt:tool-${stopReason}`, '2026-09-11T00:00:00.000Z'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            } satisfies ExecutionOptions;
            const syncDriver = new AnthropicDriver({ apiKey: 'test' });
            syncDriver.client = { messages: { stream: () => sdkStream([], finalMessage) } } as never;
            const sync = await syncDriver.executeCanonical(segments, options);

            const legacyDriver = new AnthropicDriver({ apiKey: 'test' });
            legacyDriver.client = { messages: { stream: () => sdkStream(nativeEvents, finalMessage) } } as never;
            const legacy = await legacyDriver.streamCanonical(segments, options);
            for await (const _chunk of legacy) {
                // Drain the legacy string boundary so terminal decode runs.
            }

            const typedDriver = new AnthropicDriver({ apiKey: 'test' });
            typedDriver.client = { messages: { stream: () => sdkStream(nativeEvents, finalMessage) } } as never;
            const stream = await typedDriver.streamCanonicalEvents(segments, options, undefined, {
                stream_id: `stream:claude:typed-tool-${stopReason}`,
            });
            const events = await collectCanonicalEvents(stream);
            const call = stream.completion?.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');

            expect(call).toMatchObject({
                type: 'tool_call',
                executor: 'application',
                arguments: { type: 'invalid', error: expect.stringContaining(stopReason) },
            });
            expect(stream.completion?.accepted_output.turn.status).toBe('interrupted');
            expect(stream.completion?.accepted_output.generation.status).toBe('cancelled');
            expect(events).toContainEqual(
                expect.objectContaining({ type: 'draft_block_finished', outcome: 'malformed' }),
            );
            expect(events).toContainEqual(expect.objectContaining({ type: 'draft_finished', outcome: 'interrupted' }));
            expect(events.at(-1)?.type).toBe('response_accepted');
            expect(acceptedOutputWithoutProviderTimestamps(legacy.completion?.accepted_output)).toEqual(
                acceptedOutputWithoutProviderTimestamps(sync.accepted_output),
            );
            expect(acceptedOutputWithoutProviderTimestamps(stream.completion?.accepted_output)).toEqual(
                acceptedOutputWithoutProviderTimestamps(sync.accepted_output),
            );

            if (stream.completion === undefined || call?.type !== 'tool_call') {
                throw new Error('Expected interrupted canonical Claude call');
            }
            const sourceTurn = stream.completion.conversation.turns.find(
                (turn) => turn.id === stream.completion?.accepted_output.turn.id,
            );
            const sourceCall = sourceTurn?.blocks.find((block) => block.id === call.id);
            if (sourceCall?.type !== 'tool_call')
                throw new Error('Expected retained interrupted canonical Claude call');
            await expect(
                resolveToolExecutionRequest(
                    stream.completion.conversation,
                    {
                        conversation: {
                            conversation_id: stream.completion.conversation.id,
                            revision: stream.completion.conversation.revision,
                        },
                        turn_id: stream.completion.accepted_output.turn.id,
                        block_id: sourceCall.id,
                        call_id: sourceCall.call_id,
                        call_fingerprint: await fingerprintJson(sourceCall),
                    },
                    async function* () {
                        // Interrupted inline arguments must fail before any asset resolution.
                    },
                ),
            ).rejects.toThrow(/invalid arguments/);
        },
    );

    it('retains the authoritative Claude response when preview reconciliation rejects divergent text', async () => {
        const finalMessage = {
            ...finalToolMessage,
            id: 'msg-typed-divergent',
            content: [{ type: 'text', text: 'Authoritative terminal text.' }],
            stop_reason: 'end_turn',
        } as unknown as Message;
        const nativeEvents = [
            { type: 'message_start', message: { ...finalMessage, content: [], stop_reason: null } },
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '', citations: null },
            },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'text_delta', text: 'Different preview text.' },
            },
            { type: 'content_block_stop', index: 0 },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
            { type: 'message_stop' },
        ] as RawMessageStreamEvent[];
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: () => sdkStream(nativeEvents, finalMessage) } } as never;
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer.' }],
            canonicalOptions('attempt:typed:divergent', '2026-09-11T00:00:00.000Z'),
            undefined,
            { stream_id: 'stream:claude:typed-divergent' },
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

    it('rejects typed stream bounds before publication, listener registration, or Claude transport', async () => {
        const providerCall = vi.fn(() => sdkStream([], finalToolMessage));
        const publish = vi.fn(async () => undefined);
        const controller = new AbortController();
        const addAbortListener = vi.spyOn(controller.signal, 'addEventListener');
        const removeAbortListener = vi.spyOn(controller.signal, 'removeEventListener');
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;

        await expect(
            driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'Answer.' }],
                {
                    ...canonicalOptions('attempt:typed:invalid-open', '2026-09-11T00:00:00.000Z'),
                    on_canonical_request_prepared: publish,
                },
                controller.signal,
                { stream_id: 'stream:claude:typed-invalid-open', max_buffered_events: 0 },
            ),
        ).rejects.toThrow();
        expect(publish).not.toHaveBeenCalled();
        expect(providerCall).not.toHaveBeenCalled();
        expect(addAbortListener).not.toHaveBeenCalled();
        expect(removeAbortListener).not.toHaveBeenCalled();
    });

    it('fails the typed prepared-request barrier before opening the Claude transport', async () => {
        const providerCall = vi.fn(() => sdkStream([], finalToolMessage));
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;

        await expect(
            driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'Answer.' }],
                {
                    ...canonicalOptions('attempt:typed:barrier', '2026-09-11T00:00:00.000Z'),
                    on_canonical_request_prepared: async () => {
                        throw new Error('durability barrier failed');
                    },
                },
                undefined,
                { stream_id: 'stream:claude:typed-barrier' },
            ),
        ).rejects.toThrow('durability barrier failed');
        expect(providerCall).not.toHaveBeenCalled();
    });

    it('aborts a pending native Claude read and emits one cancellation terminal', async () => {
        let finishPendingRead: (() => void) | undefined;
        const abort = vi.fn(() => finishPendingRead?.());
        const nativeEvents = [
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '', citations: null },
            },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'started' } },
        ] as RawMessageStreamEvent[];
        const providerCall = vi.fn(() => ({
            [Symbol.asyncIterator]() {
                let index = 0;
                return {
                    next: async (): Promise<IteratorResult<RawMessageStreamEvent>> => {
                        const event = nativeEvents[index];
                        index += 1;
                        if (event !== undefined) return { done: false, value: event };
                        return new Promise((resolve) => {
                            finishPendingRead = () => resolve({ done: true, value: undefined });
                        });
                    },
                    return: async () => ({ done: true, value: undefined }),
                };
            },
            finalMessage: async () => finalToolMessage,
            abort,
        }));
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Start.' }],
            canonicalOptions('attempt:typed:cancel', '2026-09-11T00:00:00.000Z'),
            undefined,
            { stream_id: 'stream:claude:typed-cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_block_started' }, done: false });
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_text_delta' }, done: false });

        const terminal = await stream.cancel();

        expect(abort).toHaveBeenCalledOnce();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(iterator.next()).resolves.toMatchObject({ value: terminal, done: false });
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
        expect(stream.completion).toBeUndefined();
    });
    it('rejects video input before calling the Claude provider', async () => {
        await expect(
            formatClaudePrompt(
                [
                    {
                        role: PromptRole.user,
                        content: 'Inspect this video.',
                        files: [
                            {
                                name: 'clip.mp4',
                                mime_type: 'video/mp4',
                                getStream: async () => new Blob(['video']).stream(),
                                getURI: async () => 'gs://bucket/clip.mp4',
                                getURL: async () => 'https://example.test/clip.mp4',
                            },
                        ],
                    },
                ],
                { model: 'claude-sonnet-4-6' },
            ),
        ).rejects.toThrow('Claude does not support video input: clip.mp4');
    });

    it('sends the same Anthropic request with or without the optional family ID', async () => {
        const stream = vi.fn(() => sdkStream([], finalToolMessage));
        const client = { messages: { stream } } as never;
        const model_options = { max_tokens: 1024, temperature: 0.2, cache_enabled: false };
        for (const tagged of [false, true]) {
            await executeClaudeCompletion(client, prompt, {
                model: 'claude-3-haiku-20240307',
                model_options: tagged ? { ...model_options, _option_id: 'anthropic-claude' } : model_options,
            });
        }
        expect(stream).toHaveBeenCalledTimes(2);
        expect(stream.mock.calls[0]).toEqual(stream.mock.calls[1]);
        expect(stream).toHaveBeenCalledWith(
            expect.objectContaining({ model: 'claude-3-haiku-20240307', max_tokens: 1024, temperature: 0.2 }),
            undefined,
        );
    });

    it('normalizes both Claude truncation stop reasons to length', () => {
        expect(claudeFinishReason('max_tokens')).toBe('length');
        expect(claudeFinishReason('model_context_window_exceeded')).toBe('length');
    });

    it('records a Claude max-token terminal as interrupted and cancelled', async () => {
        const truncated = {
            ...finalToolMessage,
            id: 'msg-truncated',
            content: [{ type: 'text', text: 'partial', citations: null }],
            stop_reason: 'max_tokens',
        } as unknown as Message;
        const completion = await executeClaudeCompletion(clientFor(truncated), prompt, {
            model: 'claude-sonnet-4-6',
        });
        const document = parseConversationDocument(completion.conversation);

        expect(document.turns.at(-1)?.status).toBe('interrupted');
        expect(Object.values(document.generations).at(-1)?.status).toBe('cancelled');
    });

    it('logs distinct provider-native truncation causes after normalization', () => {
        const logger = {
            debug: vi.fn(),
            info: vi.fn(),
            warn: vi.fn(),
            error: vi.fn(),
        };

        logClaudeTruncation(logger, 'max_tokens', { provider: 'anthropic', model: 'claude-sonnet-4-6' });
        logClaudeTruncation(logger, 'model_context_window_exceeded', {
            provider: 'vertexai',
            model: 'claude-sonnet-4-6',
        });

        expect(logger.warn).toHaveBeenNthCalledWith(
            1,
            expect.objectContaining({
                finish_reason: 'length',
                provider_finish_reason: 'max_tokens',
            }),
            '[Claude] Completion stopped at the output token limit',
        );
        expect(logger.warn).toHaveBeenNthCalledWith(
            2,
            expect.objectContaining({
                finish_reason: 'length',
                provider_finish_reason: 'model_context_window_exceeded',
            }),
            '[Claude] Completion exceeded the model context window',
        );
    });

    it('persists ordered native blocks for blocking responses without exposing thoughts by default', async () => {
        const completion = await executeClaudeCompletion(clientFor(finalToolMessage), prompt, {
            model: 'claude-sonnet-4-6',
        });

        expect(completion.result).toEqual([{ type: 'text', value: 'checking' }]);
        expect(legacyConversation(completion.conversation)).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: expectedFinalToolContent }]),
        });
    });

    it('persists reasoning on the latest completed blocking turn', async () => {
        const finalMessage = {
            ...finalToolMessage,
            content: [
                { type: 'thinking', thinking: 'final plan', signature: 'final-signature' },
                { type: 'text', text: 'final answer', citations: null },
            ],
            stop_reason: 'end_turn',
        } as unknown as Message;
        const completion = await executeClaudeCompletion(clientFor(finalMessage), prompt, {
            model: 'claude-sonnet-4-6',
        });

        expect(completion.result).toEqual([{ type: 'text', value: 'final answer' }]);
        expect(legacyConversation(completion.conversation)).toMatchObject({
            messages: expect.arrayContaining([
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'final plan', signature: 'final-signature' },
                        { type: 'text', text: 'final answer' },
                    ],
                },
            ]),
        });
    });

    it('uses the SDK final message for streaming persistence and replays it on the next tool turn', async () => {
        const events = [
            { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'plan' } },
            { type: 'content_block_delta', index: 2, delta: { type: 'text_delta', text: 'checking', citations: null } },
        ] as RawMessageStreamEvent[];
        const stream = await streamClaudeCompletion(clientFor(finalToolMessage, events), prompt, {
            model: 'claude-sonnet-4-6',
        });
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.();

        expect(results).toEqual([{ type: 'text', value: 'checking' }]);
        expect(legacyConversation(conversation)).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: expectedFinalToolContent }]),
        });

        const nextMessage = {
            ...finalToolMessage,
            id: 'msg-2',
            content: [{ type: 'text', text: 'done', citations: null }],
            stop_reason: 'end_turn',
        } as unknown as Message;
        const nextStream = vi.fn(() => sdkStream([], nextMessage));
        await executeClaudeCompletion(
            { messages: { stream: nextStream } } as never,
            {
                messages: [
                    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'sunny' }] },
                ],
            },
            {
                model: 'claude-sonnet-4-6',
                conversation: JSON.parse(JSON.stringify(conversation)),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
        );

        expect(nextStream).toHaveBeenCalledWith(
            expect.objectContaining({
                messages: expect.arrayContaining([{ role: 'assistant', content: expectedFinalToolContent }]),
            }),
            undefined,
        );
    });

    it('validates structured output through the full Claude driver and retries without another provider call', async () => {
        const rawText = '{ "answer" : "Tokyo" }';
        const finalMessage = {
            id: 'msg-structured',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{ type: 'text', text: rawText }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 2, output_tokens: 3 },
        } as unknown as Message;
        const providerCall = vi.fn(() => sdkStream([], finalMessage));
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];

        const first = await driver.execute(segments, {
            ...canonicalOptions('attempt:first', '2026-09-11T00:00:00.000Z'),
            result_schema: resultSchema,
        });
        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo' } }]);
        expect(latestGeneratedJson(first.conversation)).toEqual({ answer: 'Tokyo' });
        expect(legacyConversation(first.conversation).messages.at(-1)).toEqual({
            role: 'assistant',
            content: finalMessage.content,
        });

        const retried = await driver.execute(segments, {
            ...canonicalOptions('attempt:retry', '2026-09-11T00:01:00.000Z', first.conversation),
            result_schema: resultSchema,
        });
        expect(retried.result).toEqual(first.result);
        expect(retried.conversation).toEqual(first.conversation);
        expect(providerCall).toHaveBeenCalledOnce();
    });

    it('returns a direct canonical sync response and retries an accepted operation without transport', async () => {
        const rawText = '{"answer":"Tokyo"}';
        const finalMessage = {
            id: 'msg-direct-structured',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{ type: 'text', text: rawText }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 7, output_tokens: 4, service_tier: 'standard' },
        } as unknown as Message;
        const providerCall = vi.fn(() => sdkStream([], finalMessage));
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];
        const first = await driver.executeCanonical(segments, {
            ...canonicalOptions('attempt:direct:first', '2026-09-11T00:10:00.000Z'),
            result_schema: resultSchema,
        });

        expect(first.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } }),
        );
        expect(first.accepted_output.generation).toMatchObject({
            status: 'completed',
            usage: { input_new_tokens: 7, output_tokens: 4 },
        });
        expect(first.service_tier).toBe('standard');
        expect(legacyConversation(first.conversation).messages.at(-1)).toEqual({
            role: 'assistant',
            content: finalMessage.content,
        });

        const retried = await driver.executeCanonical(segments, {
            ...canonicalOptions('attempt:direct:retry', '2026-09-11T00:11:00.000Z', first.conversation),
            result_schema: resultSchema,
        });
        expect(retried.conversation).toEqual(first.conversation);
        expect(retried.service_tier).toBe(first.service_tier);
        await expect(
            driver.executeCanonical(segments, {
                ...canonicalOptions('attempt:direct:changed-options', '2026-09-11T00:12:00.000Z', first.conversation),
                result_schema: resultSchema,
                model_options: { _option_id: 'anthropic-claude', temperature: 0.2 },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(providerCall).toHaveBeenCalledOnce();
    });

    it('marks invalid direct canonical structured output failed for sync and stream', async () => {
        const finalMessage = {
            id: 'msg-direct-invalid',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{ type: 'text', text: '{"wrong":42}' }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: {
                input_tokens: 2,
                output_tokens: 3,
                cache_read_input_tokens: 1,
                cache_creation_input_tokens: 0,
                service_tier: 'priority',
            },
        } as unknown as Message;
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];
        const syncDriver = new AnthropicDriver({ apiKey: 'test' });
        syncDriver.client = { messages: { stream: () => sdkStream([], finalMessage) } } as never;
        const sync = await syncDriver.executeCanonical(segments, {
            ...canonicalOptions('attempt:direct:invalid:sync', '2026-09-11T00:20:00.000Z'),
            result_schema: resultSchema,
        });
        expect(sync.accepted_output.generation.status).toBe('failed');
        expect(sync.accepted_output.turn.status).toBe('failed');
        expect(sync.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: '{"wrong":42}' }),
        );

        const events = [
            { type: 'message_start', message: { ...finalMessage, content: [], stop_reason: null } },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'text_delta', text: '{"wrong":42}' },
            },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
        ] as RawMessageStreamEvent[];
        const streamDriver = new AnthropicDriver({ apiKey: 'test' });
        streamDriver.client = { messages: { stream: () => sdkStream(events, finalMessage) } } as never;
        const stream = await streamDriver.streamCanonical(segments, {
            ...canonicalOptions('attempt:direct:invalid:stream', '2026-09-11T00:21:00.000Z'),
            result_schema: resultSchema,
        });
        for await (const _chunk of stream) {
            // Drain the preview so canonical finalization runs.
        }
        expect(stream.completion?.accepted_output.generation.status).toBe('failed');
        expect(stream.completion?.accepted_output.turn.status).toBe('failed');
        expect(stream.completion?.accepted_output.generation.usage).toMatchObject({
            input_tokens: 3,
            output_tokens: 3,
            cache_read_tokens: 1,
        });
        expect(stream.completion?.service_tier).toBe('priority');
    });

    it('aborts a pending direct canonical Claude read before waiting for iterator return', async () => {
        let providerSignal: AbortSignal | undefined;
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = {
            messages: {
                stream: (_payload: unknown, requestOptions?: { signal?: AbortSignal }) => {
                    providerSignal = requestOptions?.signal;
                    return {
                        [Symbol.asyncIterator]() {
                            return {
                                next: () =>
                                    new Promise<IteratorResult<RawMessageStreamEvent>>((resolve) => {
                                        providerSignal?.addEventListener(
                                            'abort',
                                            () => resolve({ done: true, value: undefined }),
                                            { once: true },
                                        );
                                    }),
                                return: async () => ({ done: true, value: undefined }),
                            };
                        },
                        finalMessage: async () => finalToolMessage,
                        abort() {},
                    };
                },
            },
        } as never;
        const stream = await driver.streamCanonical(
            [{ role: PromptRole.user, content: 'Wait.' }],
            canonicalOptions('attempt:direct:cancel', '2026-09-11T00:30:00.000Z'),
        );
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();
        await stream.cancel();
        await expect(pending).resolves.toMatchObject({ done: true });
        expect(providerSignal?.aborted).toBe(true);
        expect(stream.completion).toBeUndefined();
    });

    it('persists streamed structured output as canonical JSON and recovers without another provider call', async () => {
        const rawText = '{"answer":"Tokyo"}';
        const finalMessage = {
            id: 'msg-stream-structured',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{ type: 'text', text: rawText }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 2, output_tokens: 3 },
        } as unknown as Message;
        const events = [
            {
                type: 'message_start',
                message: { ...finalMessage, content: [], stop_reason: null },
            },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: rawText } },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
        ] as RawMessageStreamEvent[];
        const providerCall = vi.fn(() => sdkStream(events, finalMessage));
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const stream = await driver.stream(segments, {
            ...canonicalOptions('attempt:stream', '2026-09-11T00:00:00.000Z'),
            result_schema: resultSchema,
        });

        for await (const _chunk of stream) {
            // Consume the public stream so core finalization and schema validation run.
        }
        expect(stream.completion?.result).toEqual([{ type: 'json', value: { answer: 'Tokyo' } }]);
        expect(latestGeneratedJson(stream.completion?.conversation)).toEqual({ answer: 'Tokyo' });
        expect(legacyConversation(stream.completion?.conversation).messages.at(-1)).toEqual({
            role: 'assistant',
            content: finalMessage.content,
        });

        const persisted = JSON.parse(JSON.stringify(stream.completion?.conversation));
        const retried = await driver.stream(segments, {
            ...canonicalOptions('attempt:stream:retry', '2026-09-11T00:01:00.000Z', persisted),
            result_schema: resultSchema,
        });
        for await (const _chunk of retried) {
            // Consume the recovered public stream to finalize the same accepted response.
        }
        expect(retried.completion?.result).toEqual(stream.completion?.result);
        expect(retried.completion?.conversation).toEqual(persisted);
        expect(providerCall).toHaveBeenCalledOnce();
    });

    it('prunes only completed historical reasoning and keeps an active tool chain intact', () => {
        const conversation: ClaudePrompt = {
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'old', signature: 'old-signature' },
                        { type: 'text', text: 'old answer' },
                    ],
                },
                { role: 'user', content: 'next' },
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'active', signature: 'active-signature' },
                        { type: 'tool_use', id: 'call-1', name: 'lookup', input: {} },
                    ],
                },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'done' }] },
            ],
        };

        const pruned = pruneClaudeThinking(conversation);
        expect(pruned.messages[0]).toEqual({ role: 'assistant', content: [{ type: 'text', text: 'old answer' }] });
        expect(pruned.messages[2]).toEqual(conversation.messages[2]);
        expect(pruneClaudeThinking(pruned)).toEqual(pruned);
    });
});
