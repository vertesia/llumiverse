import { describe, expect, it } from 'vitest';
import {
    assertConversationStreamDecodeEvidence,
    assertStructuredOutputTransformationProof,
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    ConversationStreamAccumulator,
    type ConversationStreamEvent,
    ConversationStreamEventSchema,
    type ConversationStreamIdentity,
    conversationStreamCursor,
    conversationStreamEventId,
    createStructuredOutputTransformationProof,
    type DecodedConversationResponse,
    type NativeStreamPosition,
} from '../src/index.js';

const identity: ConversationStreamIdentity = {
    stream_id: 'stream-live',
    request_id: 'request-1',
    attempt_id: 'attempt-1',
    response_operation_id: 'response-1',
    generation_id: 'generation-1',
    draft_turn_id: 'draft-turn-1',
};

type StreamEnvelopeKey =
    | keyof ConversationStreamIdentity
    | 'format'
    | 'schema_version'
    | 'experimental_revision'
    | 'event_id'
    | 'sequence';
type ConversationStreamEventBody = ConversationStreamEvent extends infer Event
    ? Event extends ConversationStreamEvent
        ? Omit<Event, StreamEnvelopeKey>
        : never
    : never;

function event<T extends ConversationStreamEventBody>(
    sequence: number,
    value: T,
    streamIdentity: ConversationStreamIdentity = identity,
): ConversationStreamEvent {
    return ConversationStreamEventSchema.parse({
        format: CONVERSATION_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        ...streamIdentity,
        ...value,
        sequence,
        event_id: conversationStreamEventId(streamIdentity.stream_id, sequence),
    });
}

const textPosition: NativeStreamPosition = { protocol: 'test.protocol', path: ['output', 0, 'content', 0] };
const reasoningPosition: NativeStreamPosition = { protocol: 'test.protocol', path: ['output', 0, 'content', 1] };
const toolPosition: NativeStreamPosition = { protocol: 'test.protocol', path: ['output', 1] };

describe('ConversationStreamAccumulator', () => {
    it('retains ordered interleaved native drafts without parsing partial tool arguments', () => {
        const accumulator = new ConversationStreamAccumulator(identity);
        const events: ConversationStreamEvent[] = [
            event(0, { type: 'draft_started', origin: 'live_transport' }),
            event(1, {
                type: 'draft_block_started',
                draft_block_id: 'text-draft',
                native_position: textPosition,
                block: { type: 'text' },
            }),
            event(2, {
                type: 'draft_text_delta',
                draft_block_id: 'text-draft',
                native_position: textPosition,
                text: 'hello',
            }),
            event(3, {
                type: 'draft_block_started',
                draft_block_id: 'reasoning-draft',
                native_position: reasoningPosition,
                block: { type: 'reasoning', visibility: 'display' },
            }),
            event(4, {
                type: 'draft_reasoning_delta',
                draft_block_id: 'reasoning-draft',
                native_position: reasoningPosition,
                text: 'visible thought',
            }),
            event(5, {
                type: 'draft_block_started',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                block: { type: 'tool_call', executor: 'provider', tool_name: 'web_search' },
            }),
            event(6, {
                type: 'draft_tool_call_identity',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                call_id: 'native-call-1',
            }),
            event(7, {
                type: 'draft_tool_arguments_delta',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                arguments: { encoding: 'json_fragment', fragment: '{"query":"ready"}' },
            }),
        ];

        for (const item of events) accumulator.append(item);

        expect(accumulator.retained_events).toEqual(events);
        expect(accumulator.draft_snapshot()).toEqual([
            {
                draft_block_id: 'text-draft',
                native_position: textPosition,
                type: 'text',
                text: 'hello',
                finished: false,
            },
            {
                draft_block_id: 'reasoning-draft',
                native_position: reasoningPosition,
                type: 'reasoning',
                text: 'visible thought',
                finished: false,
            },
            {
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                type: 'tool_call',
                executor: 'provider',
                call_id: 'native-call-1',
                tool_name: 'web_search',
                finished: false,
            },
        ]);
    });

    it('preserves split Unicode surrogate pairs and split JSON escapes exactly', () => {
        const accumulator = new ConversationStreamAccumulator(identity);
        const high = '\ud83d';
        const low = '\ude00';
        const firstJson = '{"emoji":"\\uD8';
        const secondJson = '3D\\uDE00"}';
        const events: ConversationStreamEvent[] = [
            event(0, { type: 'draft_started', origin: 'live_transport' }),
            event(1, {
                type: 'draft_block_started',
                draft_block_id: 'text-draft',
                native_position: textPosition,
                block: { type: 'text' },
            }),
            event(2, {
                type: 'draft_text_delta',
                draft_block_id: 'text-draft',
                native_position: textPosition,
                text: high,
            }),
            event(3, {
                type: 'draft_text_delta',
                draft_block_id: 'text-draft',
                native_position: textPosition,
                text: low,
            }),
            event(4, {
                type: 'draft_block_started',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                block: { type: 'tool_call', executor: 'application' },
            }),
            event(5, {
                type: 'draft_tool_arguments_delta',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                arguments: { encoding: 'json_fragment', fragment: firstJson },
            }),
            event(6, {
                type: 'draft_tool_arguments_delta',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                arguments: { encoding: 'json_fragment', fragment: secondJson },
            }),
        ];

        for (const item of events) accumulator.append(item);
        const roundTripped = JSON.parse(JSON.stringify(accumulator.retained_events)) as ConversationStreamEvent[];
        const text = roundTripped
            .filter((item) => item.type === 'draft_text_delta')
            .map((item) => item.text)
            .join('');
        const argumentsJson = roundTripped
            .filter((item) => item.type === 'draft_tool_arguments_delta')
            .map((item) => (item.arguments.encoding === 'json_fragment' ? item.arguments.fragment : ''))
            .join('');

        expect(text).toBe('😀');
        expect(JSON.parse(argumentsJson)).toEqual({ emoji: '😀' });
        expect(roundTripped).toEqual(events);
        expect(accumulator.draft_snapshot()).toContainEqual(
            expect.objectContaining({ draft_block_id: 'text-draft', text: '😀' }),
        );
    });

    it('requires an explicit matching retained log for resume and makes exact duplicates idempotent', () => {
        const retained = [
            event(0, { type: 'draft_started', origin: 'live_transport' }),
            event(1, { type: 'draft_finished', outcome: 'interrupted' }),
        ];
        const accumulator = new ConversationStreamAccumulator(identity, {
            retained_events: retained,
            resume_after: conversationStreamCursor(retained[1]),
        });

        expect(accumulator.append(retained[0])).toEqual({ event: retained[0], applied: false });
        expect(accumulator.eventsAfter(conversationStreamCursor(retained[1]))).toEqual([]);
        expect(
            () =>
                new ConversationStreamAccumulator(identity, {
                    resume_after: conversationStreamCursor(retained[1]),
                }),
        ).toThrow('not present');
        expect(
            () =>
                new ConversationStreamAccumulator(identity, {
                    retained_events: retained,
                    resume_after: { ...conversationStreamCursor(retained[1]), stream_id: 'other-stream' },
                }),
        ).toThrow('wrong stream_id');
        expect(() => accumulator.append({ ...retained[1], outcome: 'failed' })).toThrow('conflicts');
        expect(() => accumulator.append(event(3, { type: 'stream_terminated', outcome: 'failed' }))).toThrow(
            'expected sequence 2',
        );
    });

    it('isolates retained state from caller mutation and rejects accessors before schema traversal', () => {
        const started = event(0, { type: 'draft_started', origin: 'live_transport' });
        const accumulator = new ConversationStreamAccumulator({ ...identity }, { retained_events: [started] });
        const exposed = accumulator.retained_events;

        expect(Object.isFrozen(accumulator.identity)).toBe(true);
        expect(Object.isFrozen(exposed)).toBe(true);
        expect(Object.isFrozen(exposed[0])).toBe(true);
        expect(Reflect.set(exposed[0], 'event_id', 'tampered')).toBe(false);
        expect(accumulator.retained_events[0].event_id).toBe('stream-live#0');

        let getterCalls = 0;
        const malicious = Object.defineProperty({}, 'type', {
            enumerable: true,
            get() {
                getterCalls += 1;
                return 'draft_started';
            },
        });
        expect(() => accumulator.append(malicious)).toThrow('bounded JSON preflight');
        expect(getterCalls).toBe(0);
        const cyclic: Record<string, unknown> = {};
        cyclic.self = cyclic;
        expect(() => accumulator.append(cyclic)).toThrow('bounded JSON preflight');

        const identityWithAccessor = Object.defineProperty({ ...identity }, 'request_id', {
            enumerable: true,
            get() {
                getterCalls += 1;
                return 'request-1';
            },
        });
        expect(() => new ConversationStreamAccumulator(identityWithAccessor)).toThrow('identity failed bounded');
        expect(getterCalls).toBe(0);

        const cursorWithAccessor = Object.defineProperty(
            { stream_id: identity.stream_id, event_id: started.event_id, sequence: 0 },
            'event_id',
            {
                enumerable: true,
                get() {
                    getterCalls += 1;
                    return started.event_id;
                },
            },
        );
        expect(() => accumulator.eventsAfter(cursorWithAccessor)).toThrow('cursor failed bounded');
        expect(getterCalls).toBe(0);
    });

    it('rejects deltas after draft finish and inconsistent cumulative tool identity', () => {
        const accumulator = new ConversationStreamAccumulator(identity);
        accumulator.append(event(0, { type: 'draft_started', origin: 'live_transport' }));
        accumulator.append(
            event(1, {
                type: 'draft_block_started',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                block: { type: 'tool_call', executor: 'application', call_id: 'call' },
            }),
        );
        accumulator.append(
            event(2, {
                type: 'draft_tool_call_identity',
                draft_block_id: 'tool-draft',
                native_position: toolPosition,
                call_id: 'call-1',
                tool_name: 'look',
            }),
        );
        expect(() =>
            accumulator.append(
                event(3, {
                    type: 'draft_tool_call_identity',
                    draft_block_id: 'tool-draft',
                    native_position: toolPosition,
                    call_id: 'call-12',
                    tool_name: 'different',
                }),
            ),
        ).toThrow('changes tool_name');
        expect(accumulator.draft_snapshot()).toContainEqual(
            expect.objectContaining({ call_id: 'call-1', tool_name: 'look' }),
        );

        const finished = new ConversationStreamAccumulator(identity);
        finished.append(event(0, { type: 'draft_started', origin: 'live_transport' }));
        finished.append(
            event(1, {
                type: 'draft_block_started',
                draft_block_id: 'text-draft',
                native_position: textPosition,
                block: { type: 'text' },
            }),
        );
        finished.append(event(2, { type: 'draft_finished', outcome: 'interrupted' }));
        expect(() =>
            finished.append(
                event(3, {
                    type: 'draft_text_delta',
                    draft_block_id: 'text-draft',
                    native_position: textPosition,
                    text: 'late',
                }),
            ),
        ).toThrow('draft already finished');
    });

    it('delivers accepted recovery once on a fresh stream and rejects live identity reuse', () => {
        const recoveredIdentity = { ...identity, stream_id: 'stream-recovery' };
        const accumulator = new ConversationStreamAccumulator(recoveredIdentity);
        const accepted = event(
            0,
            {
                type: 'response_accepted',
                origin: 'accepted_recovery',
                conversation: { conversation_id: 'conversation-1', revision: 3 },
                operation_receipt_id: 'response-1',
                committed_turn_id: 'turn-1',
                turn_status: 'completed',
                generation_status: 'completed',
                committed_block_ids: ['block-1'],
                accepted_asset_ids: [],
                reconciliations: [],
            },
            recoveredIdentity,
        );

        accumulator.append(accepted);
        expect(accumulator.terminal_event).toEqual(accepted);
        expect(() => accumulator.append(accepted)).not.toThrow();
        expect(() =>
            accumulator.append(event(1, { type: 'stream_terminated', outcome: 'failed' }, recoveredIdentity)),
        ).toThrow('after its terminal');
    });

    it('serializes one cancellation terminal and rejects all later events', () => {
        const accumulator = new ConversationStreamAccumulator(identity);
        accumulator.append(event(0, { type: 'draft_started', origin: 'live_transport' }));
        const terminal = event(1, { type: 'stream_terminated', outcome: 'cancelled' });
        accumulator.append(terminal);

        expect(accumulator.terminal_event).toEqual(terminal);
        expect(() => accumulator.append(terminal)).not.toThrow();
        expect(() => accumulator.append(event(2, { type: 'usage_snapshot', usage: { input_tokens: 1 } }))).toThrow(
            'after its terminal',
        );
    });

    it('requires native drafts to reconcile exactly once with accepted canonical blocks', () => {
        const prefix = (accumulator: ConversationStreamAccumulator) => {
            accumulator.append(event(0, { type: 'draft_started', origin: 'live_transport' }));
            accumulator.append(
                event(1, {
                    type: 'draft_block_started',
                    draft_block_id: 'text-draft',
                    native_position: textPosition,
                    block: { type: 'text' },
                }),
            );
            accumulator.append(
                event(2, {
                    type: 'draft_block_finished',
                    draft_block_id: 'text-draft',
                    native_position: textPosition,
                    outcome: 'native_complete',
                }),
            );
            accumulator.append(event(3, { type: 'draft_finished', outcome: 'completed' }));
        };
        const acceptedBody = {
            type: 'response_accepted' as const,
            origin: 'live_transport' as const,
            conversation: { conversation_id: 'conversation-1', revision: 1 },
            operation_receipt_id: 'response-1',
            committed_turn_id: 'turn-1',
            turn_status: 'completed' as const,
            generation_status: 'completed' as const,
            committed_block_ids: ['text-1'],
            accepted_asset_ids: [],
        };

        const valid = new ConversationStreamAccumulator(identity);
        prefix(valid);
        valid.append(
            event(4, {
                ...acceptedBody,
                reconciliations: [
                    {
                        draft_block_ids: ['text-draft'],
                        native_positions: [textPosition],
                        committed_block_ids: ['text-1'],
                        disposition: 'direct',
                    },
                ],
            }),
        );
        expect(valid.terminal_event?.type).toBe('response_accepted');

        const missing = new ConversationStreamAccumulator(identity);
        prefix(missing);
        expect(() => missing.append(event(4, { ...acceptedBody, reconciliations: [] }))).toThrow(
            'does not reconcile every',
        );

        const wrongPosition = new ConversationStreamAccumulator(identity);
        prefix(wrongPosition);
        expect(() =>
            wrongPosition.append(
                event(4, {
                    ...acceptedBody,
                    reconciliations: [
                        {
                            draft_block_ids: ['text-draft'],
                            native_positions: [toolPosition],
                            committed_block_ids: ['text-1'],
                            disposition: 'direct',
                        },
                    ],
                }),
            ),
        ).toThrow('native positions');

        const emptyDirect = new ConversationStreamAccumulator(identity);
        prefix(emptyDirect);
        expect(() =>
            emptyDirect.append(
                event(4, {
                    ...acceptedBody,
                    reconciliations: [
                        {
                            draft_block_ids: ['text-draft'],
                            native_positions: [textPosition],
                            committed_block_ids: [],
                            disposition: 'direct',
                        },
                    ],
                }),
            ),
        ).toThrow('exactly one draft and one committed block');

        const omittedClaim = new ConversationStreamAccumulator(identity);
        prefix(omittedClaim);
        expect(() =>
            omittedClaim.append(
                event(4, {
                    ...acceptedBody,
                    reconciliations: [
                        {
                            draft_block_ids: ['text-draft'],
                            native_positions: [textPosition],
                            committed_block_ids: ['text-1'],
                            disposition: 'omitted_invalid',
                        },
                    ],
                }),
            ),
        ).toThrow('cannot claim committed blocks');

        const duplicateCommitted = new ConversationStreamAccumulator(identity);
        duplicateCommitted.append(event(0, { type: 'draft_started', origin: 'live_transport' }));
        for (const [index, position] of [textPosition, reasoningPosition].entries()) {
            duplicateCommitted.append(
                event(index * 2 + 1, {
                    type: 'draft_block_started',
                    draft_block_id: `text-draft-${index}`,
                    native_position: position,
                    block: { type: 'text' },
                }),
            );
            duplicateCommitted.append(
                event(index * 2 + 2, {
                    type: 'draft_block_finished',
                    draft_block_id: `text-draft-${index}`,
                    native_position: position,
                    outcome: 'native_complete',
                }),
            );
        }
        duplicateCommitted.append(event(5, { type: 'draft_finished', outcome: 'completed' }));
        expect(() =>
            duplicateCommitted.append(
                event(6, {
                    ...acceptedBody,
                    reconciliations: [
                        {
                            draft_block_ids: ['text-draft-0'],
                            native_positions: [textPosition],
                            committed_block_ids: ['text-1'],
                            disposition: 'direct',
                        },
                        {
                            draft_block_ids: ['text-draft-1'],
                            native_positions: [reasoningPosition],
                            committed_block_ids: ['text-1'],
                            disposition: 'direct',
                        },
                    ],
                }),
            ),
        ).toThrow('reconciled more than once');
    });

    it('enforces event, count, and serialized JSON byte budgets', () => {
        const oversized = event(0, { type: 'stream_terminated', outcome: 'failed' });
        expect(() => new ConversationStreamAccumulator(identity, { max_event_bytes: 10 }).append(oversized)).toThrow(
            'max_event_bytes',
        );

        const countBound = new ConversationStreamAccumulator(identity, { max_events: 1 });
        countBound.append(event(0, { type: 'draft_started', origin: 'live_transport' }));
        expect(() => countBound.append(event(1, { type: 'stream_terminated', outcome: 'cancelled' }))).toThrow(
            'max_events',
        );

        const first = event(0, { type: 'draft_started', origin: 'live_transport' });
        const firstBytes = new TextEncoder().encode(JSON.stringify(first)).byteLength;
        const totalBound = new ConversationStreamAccumulator(identity, {
            max_event_bytes: firstBytes + 1_000,
            max_total_bytes: firstBytes,
        });
        totalBound.append(first);
        expect(() => totalBound.append(event(1, { type: 'stream_terminated', outcome: 'cancelled' }))).toThrow(
            'max_total_bytes',
        );

        const reserved = new ConversationStreamAccumulator(identity, {
            max_event_bytes: firstBytes + 1_000,
            max_total_bytes: firstBytes + 99,
            reserved_terminal_bytes: 100,
        });
        expect(() => reserved.append(first)).toThrow('max_total_bytes');
        expect(reserved.retained_events).toEqual([]);
    });
});

describe('structured output stream reconciliation', () => {
    it('allows one native tool item to map its block and call identities', async () => {
        const nativePosition = { protocol: 'test.protocol', path: ['output', 0] };
        const decoded = {
            turns: [
                {
                    id: 'turn-1',
                    blocks: [
                        {
                            id: 'block-1',
                            type: 'tool_call',
                            call_id: 'call-1',
                            tool_name: 'lookup',
                            executor: 'application',
                            arguments: { type: 'json', value: { query: 'value' } },
                        },
                    ],
                },
            ],
            stream_evidence: {
                item_mappings: [
                    { canonical_id: 'block-1', native_position: nativePosition, kind: 'block' },
                    { canonical_id: 'call-1', native_position: nativePosition, kind: 'call' },
                ],
                transformations: [],
            },
        } as unknown as DecodedConversationResponse;

        await expect(assertConversationStreamDecodeEvidence(decoded)).resolves.toBeUndefined();

        const duplicateBlock = structuredClone(decoded);
        const duplicateBlocks = duplicateBlock.turns[0]?.blocks as unknown as Array<unknown>;
        duplicateBlocks.push({
            id: 'block-2',
            type: 'tool_call',
            call_id: 'call-2',
            tool_name: 'lookup',
            executor: 'application',
            arguments: { type: 'json', value: { query: 'duplicate' } },
        });
        duplicateBlock.stream_evidence?.item_mappings.push({
            canonical_id: 'block-2',
            native_position: nativePosition,
            kind: 'block',
        });
        await expect(assertConversationStreamDecodeEvidence(duplicateBlock)).rejects.toThrow(
            'one native position to more than one block',
        );
    });

    it('binds proof to the actual decoded source and normalized result blocks', async () => {
        const sourceBlocks = [
            { id: 'text-1', type: 'text' as const, text: '{"answer":', format: 'plain' as const },
            { id: 'text-2', type: 'text' as const, text: '42}', format: 'plain' as const },
        ];
        const resultBlock = { id: 'json-1', type: 'json' as const, value: { answer: 42 } };
        const proof = await createStructuredOutputTransformationProof({
            id: 'transform-1',
            source_blocks: sourceBlocks,
            result_block: resultBlock,
        });

        await expect(
            assertStructuredOutputTransformationProof(proof, sourceBlocks, resultBlock),
        ).resolves.toBeUndefined();
        await expect(
            assertStructuredOutputTransformationProof(
                proof,
                [{ ...sourceBlocks[0], text: '{"answer":43}' }, sourceBlocks[1]],
                resultBlock,
            ),
        ).rejects.toThrow('does not match decoded blocks');
        await expect(
            assertStructuredOutputTransformationProof(proof, sourceBlocks, {
                ...resultBlock,
                value: { answer: 43 },
            }),
        ).rejects.toThrow('does not match decoded blocks');
    });

    it('resolves transformation proof from the concrete decoded record batch', async () => {
        const sourceBlocks = [
            { id: 'source-1', type: 'text' as const, text: '{"ok":', format: 'plain' as const },
            { id: 'source-2', type: 'text' as const, text: 'true}', format: 'plain' as const },
        ];
        const resultBlock = { id: 'result-1', type: 'json' as const, value: { ok: true } };
        const proof = await createStructuredOutputTransformationProof({
            id: 'transformation-1',
            source_blocks: sourceBlocks,
            result_block: resultBlock,
        });
        const decoded = {
            turns: [{ id: 'turn-1', blocks: [resultBlock] }],
            stream_evidence: {
                item_mappings: [
                    ...sourceBlocks.map((block, index) => ({
                        canonical_id: block.id,
                        native_position: { protocol: 'test.protocol', path: ['output', index] },
                        kind: 'block' as const,
                    })),
                    {
                        canonical_id: 'result-1',
                        native_position: { protocol: 'test.protocol', path: ['normalized', 0] },
                        kind: 'block',
                    },
                ],
                transformations: [proof],
            },
        } as unknown as DecodedConversationResponse;

        await expect(assertConversationStreamDecodeEvidence(decoded)).resolves.toBeUndefined();
        const changed = structuredClone(decoded);
        const transformation = changed.stream_evidence?.transformations[0];
        if (transformation === undefined) throw new Error('Expected transformation');
        transformation.source_texts[0] = '{"ok":false';
        await expect(assertConversationStreamDecodeEvidence(changed)).rejects.toThrow('does not match decoded blocks');
    });
});
