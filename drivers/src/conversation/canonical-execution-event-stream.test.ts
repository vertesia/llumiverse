import {
    appendConversationRecords,
    type ConversationStreamEvent,
    createConversationDocument,
    type DecodedConversationResponse,
    type NativeStreamPosition,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamTerminalEvent,
    createCanonicalExecutionResponse,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import {
    type CanonicalNativeStreamWriter,
    canonicalNativeExecutionEventStream,
} from './canonical-execution-event-stream.js';

const RECORDED_AT = '2026-09-30T00:00:00.000Z';
const identity = {
    request_id: 'request',
    attempt_id: 'attempt',
    response_operation_id: 'response-operation',
    generation_id: 'generation',
    draft_turn_id: 'agent-turn',
};
const position: NativeStreamPosition = { protocol: 'test.protocol', path: ['output', 0] };

async function* nativeSource<T>(...values: T[]): AsyncIterable<T> {
    yield* values;
}

function requestReceipt(source: { conversation_id: string; revision: number }) {
    return {
        id: 'request-receipt',
        request_id: identity.request_id,
        attempt_id: identity.attempt_id,
        source,
        context_fingerprint: 'sha256:context',
        tool_set_fingerprint: 'sha256:tools',
        request_fingerprint: 'sha256:request',
        target: {
            provider: 'provider',
            protocol: position.protocol,
            model: 'model',
            adapter_version: 'adapter',
        },
        tool_definition_ids: [],
        asset_versions: [],
        item_mappings: [],
        recorded_at: RECORDED_AT,
    };
}

function acceptedResponse(text = 'answer'): CanonicalExecutionResponse {
    const initial = createConversationDocument({ id: 'conversation', created_at: RECORDED_AT });
    const receipt = requestReceipt({ conversation_id: initial.id, revision: initial.revision });
    const generation = {
        id: identity.generation_id,
        record_source: 'executed' as const,
        request_id: identity.request_id,
        attempt_id: identity.attempt_id,
        purpose: 'interaction',
        requested_model: 'model',
        provider: 'provider',
        protocol: position.protocol,
        adapter_version: 'adapter',
        status: 'completed' as const,
        finish_reason: 'stop',
        timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
        source: { conversation_id: initial.id, revision: initial.revision },
        usage: {
            input_tokens: 5,
            output_tokens: 3,
            total_tokens: 8,
            accounting_provenance: {
                input_tokens: { method: 'reported' as const, accounting_basis: 'provider' },
                output_tokens: { method: 'reported' as const, accounting_basis: 'provider' },
                total_tokens: { method: 'derived' as const, accounting_basis: 'provider' },
            },
        },
        request_receipt: receipt,
    };
    const turn = {
        id: identity.draft_turn_id,
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks: [{ id: 'text', type: 'text' as const, text, format: 'plain' as const }],
        status: 'completed' as const,
        timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
        provenance: { type: 'generated' as const },
        model_visibility: 'include' as const,
        generation_id: generation.id,
    };
    const document = appendConversationRecords(
        initial,
        { turns: [turn], generations: [generation] },
        {
            expected_revision: 0,
            operation_id: identity.response_operation_id,
            payload_fingerprint: 'sha256:response',
            recorded_at: RECORDED_AT,
        },
    ).document;
    return createCanonicalExecutionResponse(document, identity.response_operation_id);
}

function decoded(text = 'answer'): DecodedConversationResponse {
    return {
        turns: [
            {
                id: identity.draft_turn_id,
                kind: 'agent',
                authority: 'ordinary',
                blocks: [{ id: 'text', type: 'text', text, format: 'plain' }],
                status: 'completed',
                timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
                provenance: { type: 'generated' },
                model_visibility: 'include',
                generation_id: identity.generation_id,
            },
        ],
        generation: {
            id: identity.generation_id,
            record_source: 'executed',
            request_id: identity.request_id,
            attempt_id: identity.attempt_id,
            purpose: 'interaction',
            requested_model: 'model',
            provider: 'provider',
            protocol: position.protocol,
            adapter_version: 'adapter',
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
            source: { conversation_id: 'conversation', revision: 0 },
            usage: {
                input_tokens: 5,
                output_tokens: 3,
                total_tokens: 8,
                accounting_provenance: {
                    input_tokens: { method: 'reported', accounting_basis: 'provider' },
                    output_tokens: { method: 'reported', accounting_basis: 'provider' },
                    total_tokens: { method: 'derived', accounting_basis: 'provider' },
                },
            },
            request_receipt: requestReceipt({ conversation_id: 'conversation', revision: 0 }),
        },
        assets: [],
        diagnostics: [],
        payload_fingerprint: 'sha256:response',
        stream_evidence: {
            item_mappings: [{ canonical_id: 'text', native_position: position, kind: 'block' }],
            transformations: [],
        },
    };
}

async function emitText(text: string, writer: CanonicalNativeStreamWriter) {
    await writer.startBlock({ draft_block_id: 'draft-text', native_position: position, block: { type: 'text' } });
    await writer.text({ draft_block_id: 'draft-text', native_position: position, text });
    await writer.finishBlock({
        draft_block_id: 'draft-text',
        native_position: position,
        outcome: 'native_complete',
    });
}

describe('canonical native execution event stream', () => {
    it('waits for provider cleanup before delivering accepted response', async () => {
        let releaseClose!: () => void;
        const close = vi.fn(
            () =>
                new Promise<void>((resolve) => {
                    releaseClose = resolve;
                }),
        );
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:accepted' },
            openSource: () => nativeSource('answer'),
            map: emitText,
            finalize: async (writer) => {
                await writer.finish({ outcome: 'completed', finish_reason: 'stop' });
                return {
                    decoded: decoded(),
                    response: acceptedResponse(),
                    reconciliations: [
                        {
                            draft_block_ids: ['draft-text'],
                            native_positions: [position],
                            committed_block_ids: ['text'],
                            disposition: 'direct',
                        },
                    ],
                };
            },
            abort: vi.fn(),
            close,
        });
        const events: string[] = [];
        const consumption = (async () => {
            for await (const event of stream) events.push(event.type);
        })();

        await vi.waitFor(() => expect(close).toHaveBeenCalledOnce());
        expect(events).not.toContain('response_accepted');
        releaseClose();
        await consumption;

        expect(events.at(-1)).toBe('response_accepted');
        expect(stream.completion).toBeDefined();
    });

    it('preserves a blocked event before cancellation terminal without a sequence gap', async () => {
        let releaseClose!: () => void;
        let mapBlocked!: () => void;
        const blockingMap = new Promise<void>((resolve) => {
            mapBlocked = resolve;
        });
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:backpressure', max_buffered_events: 1 },
            openSource: () => nativeSource('answer'),
            map: async (_event, writer) => {
                const blocked = writer.startBlock({
                    draft_block_id: 'draft-text',
                    native_position: position,
                    block: { type: 'text' },
                });
                mapBlocked();
                await blocked;
            },
            finalize: async () => {
                throw new Error('unreachable');
            },
            abort: vi.fn(),
            close: () =>
                new Promise<void>((resolve) => {
                    releaseClose = resolve;
                }),
        });
        const iterator = stream[Symbol.asyncIterator]();
        await blockingMap;
        const cancellation = stream.cancel();
        await vi.waitFor(() => expect(releaseClose).toBeTypeOf('function'));
        releaseClose();
        await cancellation;

        const events: ConversationStreamEvent[] = [];
        while (true) {
            const next = await iterator.next();
            if (next.done) break;
            events.push(next.value);
        }
        expect(events.map((event) => event.sequence)).toEqual(events.map((_event, index) => index));
        expect(events.map((event) => event.type)).toEqual([
            'draft_started',
            'draft_block_started',
            'stream_terminated',
        ]);
    });

    it('does not open a provider transport when cancelled from draft_started', async () => {
        const openSource = vi.fn(() => nativeSource('answer'));
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:cancel-before-open' },
            openSource,
            map: vi.fn(),
            finalize: vi.fn(),
            abort: vi.fn(),
            close: vi.fn(),
        });
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });

        await expect(stream.cancel()).resolves.toMatchObject({ outcome: 'cancelled' });
        expect(openSource).not.toHaveBeenCalled();
        await expect(iterator.next()).resolves.toMatchObject({
            value: { type: 'stream_terminated', outcome: 'cancelled' },
            done: false,
        });
    });

    it('awaits and cleans a late provider handle before terminal delivery and lease release', async () => {
        let openingStarted!: () => void;
        const started = new Promise<void>((resolve) => {
            openingStarted = resolve;
        });
        let resolveSource!: (source: AsyncIterable<string>) => void;
        const source = new Promise<AsyncIterable<string>>((resolve) => {
            resolveSource = resolve;
        });
        const iteratorReturn = vi.fn(async () => ({ value: undefined, done: true }) as IteratorResult<string>);
        const lateSource = {
            [Symbol.asyncIterator]() {
                return { next: async () => new Promise<IteratorResult<string>>(() => {}), return: iteratorReturn };
            },
        };
        const close = vi.fn();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:late-open' },
            openSource: () => {
                openingStarted();
                return source;
            },
            map: vi.fn(),
            finalize: vi.fn(),
            abort: vi.fn(),
            close,
        });
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        await started;

        const terminalRead = iterator.next();
        const cancellation = stream.cancel();
        await Promise.resolve();
        expect(close).not.toHaveBeenCalled();
        expect(stream.terminal_event).toBeUndefined();

        resolveSource(lateSource);
        await expect(terminalRead).resolves.toMatchObject({
            value: { type: 'stream_terminated', outcome: 'cancelled' },
            done: false,
        });
        await expect(cancellation).resolves.toMatchObject({ outcome: 'cancelled' });
        expect(iteratorReturn).toHaveBeenCalledOnce();
        expect(close).toHaveBeenCalledOnce();
    });

    it('retains a finalized canonical response when reconciliation fails', async () => {
        const response = acceptedResponse();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:delivery-failure' },
            openSource: () => nativeSource('answer'),
            map: emitText,
            finalize: async (writer) => {
                await writer.finish({ outcome: 'completed' });
                return { decoded: decoded('different'), response, reconciliations: [] };
            },
            abort: vi.fn(),
            close: vi.fn(),
        });
        const events = [];
        for await (const event of stream) events.push(event);

        expect(stream.completion).toBe(response);
        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'CANONICAL_EVENT_DELIVERY_FAILED' },
        });
    });

    it('rejects bounds that cannot hold the reserved terminal before opening the source', () => {
        const openSource = vi.fn(() => nativeSource());
        expect(() =>
            canonicalNativeExecutionEventStream({
                identity,
                open: { stream_id: 'stream:small', max_event_bytes: 1, max_total_bytes: 1 },
                openSource,
                map: vi.fn(),
                finalize: vi.fn(),
                abort: vi.fn(),
                close: vi.fn(),
            }),
        ).toThrow('max_event_bytes');
        expect(openSource).not.toHaveBeenCalled();

        expect(() =>
            canonicalNativeExecutionEventStream({
                identity,
                open: {
                    stream_id: 'stream:resume',
                    resume_after: { stream_id: 'stream:resume', sequence: 0, event_id: 'event' },
                },
                openSource,
                map: vi.fn(),
                finalize: vi.fn(),
                abort: vi.fn(),
                close: vi.fn(),
            }),
        ).toThrow('cannot resume');
        expect(openSource).not.toHaveBeenCalled();
    });

    it('waits for delayed cleanup on provider failure before delivering its terminal', async () => {
        let releaseClose!: () => void;
        const close = vi.fn(
            () =>
                new Promise<void>((resolve) => {
                    releaseClose = resolve;
                }),
        );
        const source = {
            [Symbol.asyncIterator]() {
                return {
                    next: async () => {
                        throw new Error('private provider failure');
                    },
                };
            },
        };
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:failure' },
            openSource: () => source,
            map: vi.fn(),
            finalize: vi.fn(),
            abort: vi.fn(),
            close,
        });
        const events: Array<{ type: string }> = [];
        const consumption = (async () => {
            for await (const event of stream) events.push(event);
        })();

        await vi.waitFor(() => expect(close).toHaveBeenCalledOnce());
        expect(events.map((event) => event.type)).toEqual(['draft_started']);
        releaseClose();
        await consumption;

        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            diagnostic: { code: 'PROVIDER_STREAM_FAILED' },
        });
        expect(JSON.stringify(events)).not.toContain('private provider failure');
    });

    it('delivers cancellation after throwing abort, iterator return, and close cleanup', async () => {
        let nextStarted!: () => void;
        const started = new Promise<void>((resolve) => {
            nextStarted = resolve;
        });
        const source = {
            [Symbol.asyncIterator]() {
                return {
                    next: () => {
                        nextStarted();
                        return new Promise<IteratorResult<string>>(() => {});
                    },
                    return: async () => {
                        throw new Error('private iterator cleanup failure');
                    },
                };
            },
        };
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:throwing-cleanup' },
            openSource: () => source,
            map: vi.fn(),
            finalize: vi.fn(),
            abort: () => {
                throw new Error('private abort failure');
            },
            close: () => {
                throw new Error('private close failure');
            },
        });
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        await started;

        await expect(stream.cancel()).resolves.toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(iterator.next()).resolves.toMatchObject({
            value: { type: 'stream_terminated', outcome: 'cancelled' },
            done: false,
        });
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
    });

    it('retains a response that finishes decoding after cancellation won the terminal race', async () => {
        let finalizeStarted!: () => void;
        const started = new Promise<void>((resolve) => {
            finalizeStarted = resolve;
        });
        let releaseFinalize!: () => void;
        const finalizeBarrier = new Promise<void>((resolve) => {
            releaseFinalize = resolve;
        });
        const response = acceptedResponse();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:cancel-finalize' },
            openSource: () => nativeSource<string>(),
            map: vi.fn(),
            finalize: async () => {
                finalizeStarted();
                await finalizeBarrier;
                return { decoded: decoded(), response, reconciliations: [] };
            },
            abort: vi.fn(),
            close: vi.fn(),
        });
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        await started;

        await expect(stream.cancel()).resolves.toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        releaseFinalize();
        await vi.waitFor(() => expect(stream.completion).toBe(response));

        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'stream_terminated' }, done: false });
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
    });

    it('joins cancellation to a failure settlement already cleaning the provider', async () => {
        let releaseReturn!: () => void;
        const returnBarrier = new Promise<void>((resolve) => {
            releaseReturn = resolve;
        });
        const iteratorReturn = vi.fn(async () => {
            await returnBarrier;
            return { value: undefined, done: true } as IteratorResult<string>;
        });
        const source = {
            [Symbol.asyncIterator]() {
                return {
                    next: async () => ({ value: 'answer', done: false }) as IteratorResult<string>,
                    return: iteratorReturn,
                };
            },
        };
        const abort = vi.fn();
        const close = vi.fn();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:failure-cancel-race' },
            openSource: () => source,
            map: () => {
                throw new Error('private map failure');
            },
            finalize: vi.fn(),
            abort,
            close,
        });
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        const terminalRead = iterator.next();
        await vi.waitFor(() => expect(iteratorReturn).toHaveBeenCalledOnce());

        const cancellation = stream.cancel();
        expect(stream.terminal_event).toBeUndefined();
        releaseReturn();
        const [read, cancelled] = await Promise.all([terminalRead, cancellation]);

        expect(read).toMatchObject({ value: { type: 'stream_terminated', outcome: 'failed' }, done: false });
        expect(cancelled).toBe(read.value);
        expect(abort).toHaveBeenCalledOnce();
        expect(iteratorReturn).toHaveBeenCalledOnce();
        expect(close).toHaveBeenCalledOnce();
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
    });

    it('claims cancellation before a reentrant abort callback can settle again', async () => {
        let reentrant: Promise<CanonicalStreamTerminalEvent> | undefined;
        let stream!: CanonicalExecutionEventStream;
        const abort = vi.fn(() => {
            reentrant = stream.cancel();
        });
        const close = vi.fn();
        stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:reentrant-abort' },
            openSource: () => nativeSource<string>(),
            map: vi.fn(),
            finalize: vi.fn(),
            abort,
            close,
        });

        const cancellation = stream.cancel();
        const terminal = await cancellation;

        expect(reentrant).toBe(cancellation);
        await expect(reentrant).resolves.toBe(terminal);
        expect(abort).toHaveBeenCalledOnce();
        expect(close).toHaveBeenCalledOnce();
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: terminal, done: false });
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
    });

    it('joins cancellation to accepted settlement after provider finalization', async () => {
        let releaseClose!: () => void;
        const close = vi.fn(
            () =>
                new Promise<void>((resolve) => {
                    releaseClose = resolve;
                }),
        );
        const abort = vi.fn();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:accepted-cancel-race' },
            openSource: () => nativeSource('answer'),
            map: emitText,
            finalize: async (writer) => {
                await writer.finish({ outcome: 'completed', finish_reason: 'stop' });
                return {
                    decoded: decoded(),
                    response: acceptedResponse(),
                    reconciliations: [
                        {
                            draft_block_ids: ['draft-text'],
                            native_positions: [position],
                            committed_block_ids: ['text'],
                            disposition: 'direct',
                        },
                    ],
                };
            },
            abort,
            close,
        });
        const events: ConversationStreamEvent[] = [];
        const consumption = (async () => {
            for await (const event of stream) events.push(event);
        })();
        await vi.waitFor(() => expect(close).toHaveBeenCalledOnce());

        const cancellation = stream.cancel();
        releaseClose();
        const terminal = await cancellation;
        await consumption;

        expect(terminal.type).toBe('response_accepted');
        expect(events.at(-1)).toBe(terminal);
        expect(abort).not.toHaveBeenCalled();
        expect(close).toHaveBeenCalledOnce();
    });
});
