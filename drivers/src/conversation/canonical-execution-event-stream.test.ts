import {
    appendConversationRecords,
    type ConversationStreamEvent,
    createConversationDocument,
    type DecodedConversationResponse,
    type NativeStreamPosition,
} from '@llumiverse/conversation';
import {
    CANONICAL_FORBIDDEN_TOOL_CALL,
    CANONICAL_REQUIRED_TOOL_CALL_MISSING,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    type CanonicalStreamTerminalEvent,
    CanonicalToolSelectionViolationError,
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
    it('delivers accepted response while retaining ownership through provider cleanup', async () => {
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
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        const consumption = (async () => {
            for await (const event of stream) events.push(event.type);
        })();

        await vi.waitFor(() => expect(close).toHaveBeenCalledOnce());
        await vi.waitFor(() => expect(events.at(-1)).toBe('response_accepted'));
        expect(closed).toBe(false);
        releaseClose();
        await Promise.all([consumption, stream.closed]);

        expect(events.at(-1)).toBe('response_accepted');
        expect(stream.completion).toBeDefined();
        expect(closed).toBe(true);
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
        await expect(cancellation).resolves.toMatchObject({ outcome: 'cancelled' });

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
        releaseClose();
        await stream.closed;
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
        expect(stream.execution_started).toBe(false);
        await expect(iterator.next()).resolves.toMatchObject({
            value: { type: 'stream_terminated', outcome: 'cancelled' },
            done: false,
        });
    });

    it('delivers cancellation before a late provider handle while retaining cleanup ownership', async () => {
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
        expect(stream.execution_started).toBe(true);

        const terminalRead = iterator.next();
        const cancellation = stream.cancel();
        await Promise.resolve();
        expect(close).toHaveBeenCalledOnce();
        await expect(terminalRead).resolves.toMatchObject({
            value: { type: 'stream_terminated', outcome: 'cancelled' },
            done: false,
        });
        await expect(cancellation).resolves.toMatchObject({ outcome: 'cancelled' });
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);

        resolveSource(lateSource);
        await stream.closed;
        expect(iteratorReturn).toHaveBeenCalledOnce();
        expect(close).toHaveBeenCalledOnce();
        expect(closed).toBe(true);
    });

    it('settles cancellation while a provider open never resolves and retains cleanup ownership', async () => {
        let openingStarted!: () => void;
        const started = new Promise<void>((resolve) => {
            openingStarted = resolve;
        });
        const close = vi.fn();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:never-open' },
            openSource: () => {
                openingStarted();
                return new Promise<AsyncIterable<string>>(() => undefined);
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
        await expect(stream.cancel()).resolves.toMatchObject({ outcome: 'cancelled' });
        await expect(terminalRead).resolves.toMatchObject({
            value: { type: 'stream_terminated', outcome: 'cancelled' },
            done: false,
        });
        expect(close).toHaveBeenCalledOnce();
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
    });

    it('settles cancellation while iterator return never resolves and retains cleanup ownership', async () => {
        let nextStarted!: () => void;
        const started = new Promise<void>((resolve) => {
            nextStarted = resolve;
        });
        const close = vi.fn();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:never-return' },
            openSource: () => ({
                [Symbol.asyncIterator]() {
                    return {
                        next: () => {
                            nextStarted();
                            return new Promise<IteratorResult<string>>(() => undefined);
                        },
                        return: () => new Promise<IteratorResult<string>>(() => undefined),
                    };
                },
            }),
            map: vi.fn(),
            finalize: vi.fn(),
            abort: vi.fn(),
            close,
        });
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        await started;

        await expect(stream.cancel()).resolves.toMatchObject({ outcome: 'cancelled' });
        await expect(iterator.next()).resolves.toMatchObject({
            value: { type: 'stream_terminated', outcome: 'cancelled' },
            done: false,
        });
        expect(close).toHaveBeenCalledOnce();
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
    });

    it('settles cancellation while provider close never resolves and retains cleanup ownership', async () => {
        let nextStarted!: () => void;
        const started = new Promise<void>((resolve) => {
            nextStarted = resolve;
        });
        const iteratorReturn = vi.fn(async () => ({ value: undefined, done: true }) as IteratorResult<string>);
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:never-close' },
            openSource: () => ({
                [Symbol.asyncIterator]() {
                    return {
                        next: () => {
                            nextStarted();
                            return new Promise<IteratorResult<string>>(() => undefined);
                        },
                        return: iteratorReturn,
                    };
                },
            }),
            map: vi.fn(),
            finalize: vi.fn(),
            abort: vi.fn(),
            close: () => new Promise<void>(() => undefined),
        });
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        await started;

        await expect(iterator.return?.()).resolves.toEqual({ value: undefined, done: true });
        expect(stream.terminal_event).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await vi.waitFor(() => expect(iteratorReturn).toHaveBeenCalledOnce());
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
    });

    it('retains a finalized canonical response when reconciliation fails', async () => {
        const response = acceptedResponse();
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:delivery-failure' },
            classifyFailure: () => true,
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
        expect((events.at(-1) as { diagnostic?: object }).diagnostic).not.toHaveProperty('retryable');
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

    it('retains classified provider failure within the reserved terminal budget', async () => {
        const providerFailureSource = () => ({
            [Symbol.asyncIterator]() {
                return {
                    next: async () => {
                        throw new Error('private provider failure');
                    },
                };
            },
        });
        const collectFailure = async (open: CanonicalStreamOpenOptions) => {
            const stream = canonicalNativeExecutionEventStream({
                identity,
                open,
                classifyFailure: () => false,
                openSource: providerFailureSource,
                map: vi.fn(),
                finalize: vi.fn(),
                abort: vi.fn(),
                close: vi.fn(),
            });
            const events: ConversationStreamEvent[] = [];
            for await (const event of stream) events.push(event);
            return events;
        };
        const streamId = 'stream:classified-bounds';
        const baseline = await collectFailure({ stream_id: streamId, max_events: 2 });
        const eventBytes = (event: ConversationStreamEvent) =>
            new TextEncoder().encode(JSON.stringify(event)).byteLength;
        const terminal = baseline.at(-1);
        if (terminal?.type !== 'stream_terminated') throw new Error('Expected a classified failed terminal');
        const terminalBytes = eventBytes(terminal);
        const totalBytes = baseline.reduce((total, event) => total + eventBytes(event), 0);

        const boundedStream = (open: CanonicalStreamOpenOptions) =>
            canonicalNativeExecutionEventStream({
                identity,
                open,
                classifyFailure: () => false,
                openSource: providerFailureSource,
                map: vi.fn(),
                finalize: vi.fn(),
                abort: vi.fn(),
                close: vi.fn(),
            });
        expect(() =>
            boundedStream({
                stream_id: streamId,
                max_events: 2,
                max_event_bytes: terminalBytes - 1,
                max_total_bytes: totalBytes,
            }),
        ).toThrow('max_event_bytes');
        expect(() =>
            boundedStream({
                stream_id: streamId,
                max_events: 2,
                max_total_bytes: terminalBytes - 1,
            }),
        ).toThrow('max_total_bytes');

        expect(baseline.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'PROVIDER_STREAM_FAILED', retryable: false },
        });
    });

    it('delivers provider failure while retaining ownership through delayed cleanup', async () => {
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
            classifyFailure: () => true,
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
        await consumption;
        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            diagnostic: { code: 'PROVIDER_STREAM_FAILED', retryable: true },
        });
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
        releaseClose();
        await stream.closed;

        expect(closed).toBe(true);
        expect(JSON.stringify(events)).not.toContain('private provider failure');
    });

    it('keeps failed provider classification unknown when the classifier cannot classify it', async () => {
        const source = {
            [Symbol.asyncIterator]() {
                return {
                    next: async () => {
                        throw new Error('private unknown failure');
                    },
                };
            },
        };
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:unknown-failure' },
            classifyFailure: () => {
                throw new Error('private classifier failure');
            },
            openSource: () => source,
            map: vi.fn(),
            finalize: vi.fn(),
            abort: vi.fn(),
            close: vi.fn(),
        });
        const events: ConversationStreamEvent[] = [];
        for await (const event of stream) events.push(event);

        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            diagnostic: { code: 'PROVIDER_STREAM_FAILED' },
        });
        expect((events.at(-1) as { diagnostic?: object }).diagnostic).not.toHaveProperty('retryable');
        expect(JSON.stringify(events)).not.toContain('private');
    });

    it('classifies a pre-ingestion selection violation as permanent without exposing an accepted completion', async () => {
        const stream = canonicalNativeExecutionEventStream({
            identity,
            open: { stream_id: 'stream:selection-violation' },
            openSource: () => nativeSource('terminal'),
            map: vi.fn(),
            finalize: async () => {
                throw new CanonicalToolSelectionViolationError(decoded());
            },
            classifyFailure: () => true,
            abort: vi.fn(),
            close: vi.fn(),
        });
        const events: ConversationStreamEvent[] = [];
        for await (const event of stream) events.push(event);

        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: CANONICAL_REQUIRED_TOOL_CALL_MISSING, retryable: false },
        });
        expect(events.some((event) => event.type === 'response_accepted')).toBe(false);
        expect(stream.completion).toBeUndefined();
    });

    it('keeps forbidden tool calls nonrecoverable and reserves the longest selection terminal exactly', async () => {
        const collect = async (
            code: typeof CANONICAL_REQUIRED_TOOL_CALL_MISSING | typeof CANONICAL_FORBIDDEN_TOOL_CALL,
            limits: Pick<CanonicalStreamOpenOptions, 'max_event_bytes' | 'max_total_bytes'> = {},
        ) => {
            const streamId = `stream:${code.toLowerCase()}`;
            const stream = canonicalNativeExecutionEventStream({
                identity,
                open: { stream_id: streamId, max_events: 2, ...limits },
                openSource: () => nativeSource('terminal'),
                map: vi.fn(),
                finalize: async () => {
                    throw new CanonicalToolSelectionViolationError(decoded(), code);
                },
                classifyFailure: () => true,
                abort: vi.fn(),
                close: vi.fn(),
            });
            const events: ConversationStreamEvent[] = [];
            for await (const event of stream) events.push(event);
            return events;
        };
        const required = await collect(CANONICAL_REQUIRED_TOOL_CALL_MISSING);
        const forbidden = await collect(CANONICAL_FORBIDDEN_TOOL_CALL);
        expect(forbidden.at(-1)).toMatchObject({
            type: 'stream_terminated',
            diagnostic: { code: CANONICAL_FORBIDDEN_TOOL_CALL, retryable: false },
        });

        const bytes = (event: ConversationStreamEvent) => new TextEncoder().encode(JSON.stringify(event)).byteLength;
        const longer = [required.at(-1), forbidden.at(-1)]
            .filter((event): event is ConversationStreamEvent => event !== undefined)
            .sort((left, right) => bytes(right) - bytes(left))[0];
        if (longer?.type !== 'stream_terminated' || longer.diagnostic === undefined) {
            throw new Error('Expected a selection terminal');
        }
        const code = longer.diagnostic.code;
        if (code !== CANONICAL_REQUIRED_TOOL_CALL_MISSING && code !== CANONICAL_FORBIDDEN_TOOL_CALL) {
            throw new Error('Expected a selection diagnostic code');
        }
        const selected = code === CANONICAL_REQUIRED_TOOL_CALL_MISSING ? required : forbidden;
        const exactBytes = bytes(longer);
        const exact = await collect(code, {
            max_event_bytes: exactBytes,
            max_total_bytes: selected.reduce((total, event) => total + bytes(event), 0),
        });
        expect(exact.at(-1)).toEqual(longer);
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
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
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
        await stream.closed;

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
        const [read, cancelled] = await Promise.all([terminalRead, cancellation]);

        expect(read).toMatchObject({ value: { type: 'stream_terminated', outcome: 'failed' }, done: false });
        expect(cancelled).toBe(read.value);
        expect(abort).toHaveBeenCalledOnce();
        expect(iteratorReturn).toHaveBeenCalledOnce();
        expect(close).toHaveBeenCalledOnce();
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
        releaseReturn();
        await stream.closed;
        expect(closed).toBe(true);
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
        await stream.closed;
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
        const terminal = await cancellation;
        await consumption;

        expect(terminal.type).toBe('response_accepted');
        expect(events.at(-1)).toBe(terminal);
        expect(abort).not.toHaveBeenCalled();
        expect(close).toHaveBeenCalledOnce();
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
        releaseClose();
        await stream.closed;
        expect(closed).toBe(true);
    });
});
