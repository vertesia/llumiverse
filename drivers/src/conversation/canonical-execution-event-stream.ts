import { LlumiverseError } from '@llumiverse/common';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    CONVERSATION_STREAM_MAX_EVENT_BYTES,
    CONVERSATION_STREAM_MAX_EVENTS,
    CONVERSATION_STREAM_MAX_TOTAL_BYTES,
    ConversationStreamAccumulator,
    type ConversationStreamDraftBlock,
    type ConversationStreamEvent,
    type ConversationStreamIdentity,
    type ConversationStreamReconciliation,
    conversationStreamEventId,
    type DecodedConversationResponse,
    type GenerationUsage,
    type JsonValue,
    type NativeStreamPosition,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    CanonicalStreamEventChannel,
    type CanonicalStreamOpenOptions,
    type CanonicalStreamTerminalEvent,
    finalizeCanonicalExecutionStreamResponse,
} from '@llumiverse/core';

type DraftOutcome = Extract<ConversationStreamEvent, { type: 'draft_finished' }>['outcome'];
type BlockOutcome = Extract<ConversationStreamEvent, { type: 'draft_block_finished' }>['outcome'];
type WithoutEnvelope<Event> = Event extends ConversationStreamEvent
    ? Omit<Event, keyof ReturnType<typeof envelope>>
    : never;
type ConversationStreamEventInput = WithoutEnvelope<ConversationStreamEvent>;

interface CanonicalNativeStreamPreparedReconciliation {
    decoded: DecodedConversationResponse;
    reconciliations: ConversationStreamReconciliation[];
    deliver_final_events?(writer: CanonicalNativeStreamWriter): Promise<void>;
}

export interface CanonicalNativeStreamFinalization {
    decoded: DecodedConversationResponse;
    response: CanonicalExecutionResponse;
    reconciliations?: ConversationStreamReconciliation[];
    result_schema?: object;
    deliver_final_events?(writer: CanonicalNativeStreamWriter): Promise<void>;
    prepare_reconciliation?(writer: CanonicalNativeStreamWriter): Promise<CanonicalNativeStreamPreparedReconciliation>;
}

type ReconciledCanonicalNativeStreamFinalization = CanonicalNativeStreamFinalization & {
    reconciliations: ConversationStreamReconciliation[];
};

export interface CanonicalNativeEventStreamOptions<NativeEvent> {
    identity: Omit<ConversationStreamIdentity, 'stream_id'>;
    open: CanonicalStreamOpenOptions;
    openSource(): AsyncIterable<NativeEvent> | Promise<AsyncIterable<NativeEvent>>;
    map(event: NativeEvent, writer: CanonicalNativeStreamWriter): void | Promise<void>;
    finalize(writer: CanonicalNativeStreamWriter): Promise<CanonicalNativeStreamFinalization>;
    classifyFailure?(error: unknown): boolean | undefined;
    abort(): void;
    close(): void | Promise<void>;
}

function envelope(identity: ConversationStreamIdentity, sequence: number) {
    return {
        format: CONVERSATION_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        ...identity,
        event_id: conversationStreamEventId(identity.stream_id, sequence),
        sequence,
    } as const;
}

function terminatedEvent(
    identity: ConversationStreamIdentity,
    sequence: number,
    outcome: 'cancelled' | 'failed',
    failureKind: 'provider' | 'delivery' = 'provider',
    retryable?: boolean,
): CanonicalStreamTerminalEvent {
    return {
        ...envelope(identity, sequence),
        type: 'stream_terminated',
        outcome,
        ...(outcome === 'failed'
            ? {
                  diagnostic: {
                      code: failureKind === 'delivery' ? 'CANONICAL_EVENT_DELIVERY_FAILED' : 'PROVIDER_STREAM_FAILED',
                      message:
                          failureKind === 'delivery'
                              ? 'Canonical event delivery failed'
                              : 'Provider stream ended before canonical response acceptance',
                      ...(retryable === undefined ? {} : { retryable }),
                  },
              }
            : {}),
    };
}

function serializedBytes(value: unknown): number {
    return new TextEncoder().encode(JSON.stringify(value)).byteLength;
}

function nextTask(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

function deferred<T>(): { promise: Promise<T>; resolve(value: T): void } {
    let resolvePromise: ((value: T) => void) | undefined;
    const promise = new Promise<T>((resolve) => {
        resolvePromise = resolve;
    });
    if (resolvePromise === undefined) throw new Error('Failed to create deferred promise');
    return { promise, resolve: resolvePromise };
}

export class CanonicalNativeStreamWriter {
    constructor(
        private readonly accumulator: ConversationStreamAccumulator,
        private readonly channel: CanonicalStreamEventChannel,
    ) {}

    get identity(): Readonly<ConversationStreamIdentity> {
        return this.accumulator.identity;
    }

    get draft_snapshot() {
        return this.accumulator.draft_snapshot();
    }

    async start(): Promise<void> {
        await this.emit({ type: 'draft_started', origin: 'live_transport' });
    }

    async startBlock(input: {
        draft_block_id: string;
        native_position: NativeStreamPosition;
        block: ConversationStreamDraftBlock;
    }): Promise<void> {
        await this.emit({ type: 'draft_block_started', ...input });
    }

    async text(input: { draft_block_id: string; native_position: NativeStreamPosition; text: string }): Promise<void> {
        if (input.text.length === 0) return;
        await this.emit({ type: 'draft_text_delta', ...input });
    }

    async reasoning(input: {
        draft_block_id: string;
        native_position: NativeStreamPosition;
        text: string;
    }): Promise<void> {
        if (input.text.length === 0) return;
        await this.emit({ type: 'draft_reasoning_delta', ...input });
    }

    async toolIdentity(input: {
        draft_block_id: string;
        native_position: NativeStreamPosition;
        call_id?: string;
        tool_name?: string;
    }): Promise<void> {
        if (input.call_id === undefined && input.tool_name === undefined) return;
        await this.emit({ type: 'draft_tool_call_identity', ...input });
    }

    async toolArgumentsFragment(input: {
        draft_block_id: string;
        native_position: NativeStreamPosition;
        fragment: string;
    }): Promise<void> {
        if (input.fragment.length === 0) return;
        const { fragment, ...draft } = input;
        await this.emit({
            type: 'draft_tool_arguments_delta',
            ...draft,
            arguments: { encoding: 'json_fragment', fragment },
        });
    }

    async toolArgumentsSnapshot(input: {
        draft_block_id: string;
        native_position: NativeStreamPosition;
        value: JsonValue;
    }): Promise<void> {
        const { value, ...draft } = input;
        await this.emit({
            type: 'draft_tool_arguments_delta',
            ...draft,
            arguments: { encoding: 'json_value_snapshot', value },
        });
    }

    async finishBlock(input: {
        draft_block_id: string;
        native_position: NativeStreamPosition;
        outcome: BlockOutcome;
    }): Promise<void> {
        await this.emit({ type: 'draft_block_finished', ...input });
    }

    async usage(usage: GenerationUsage): Promise<void> {
        await this.emit({ type: 'usage_snapshot', usage });
    }

    async finish(input: { outcome: DraftOutcome; finish_reason?: string; service_tier?: string }): Promise<void> {
        await this.emit({ type: 'draft_finished', ...input });
    }

    private async emit(input: ConversationStreamEventInput): Promise<ConversationStreamEvent> {
        const sequence = this.accumulator.next_sequence;
        const event = {
            ...envelope(this.accumulator.identity, sequence),
            ...input,
        } as ConversationStreamEvent;
        const appended = this.accumulator.append(event).event;
        await this.channel.emit(appended);
        return appended;
    }
}

/** One-consumer native stream that emits canonical drafts before accepting the authoritative terminal decode. */
export class CanonicalNativeExecutionEventStream<NativeEvent> implements CanonicalExecutionEventStream {
    completion: CanonicalExecutionResponse | undefined;
    readonly closed: Promise<void>;
    private readonly accumulator: ConversationStreamAccumulator;
    private readonly channel: CanonicalStreamEventChannel;
    private readonly closeDeferred = deferred<void>();
    private readonly writer: CanonicalNativeStreamWriter;
    private opening: Promise<AsyncIterable<NativeEvent> | undefined> | undefined;
    private iterator: AsyncIterator<NativeEvent> | undefined;
    private iteratorCleanup: Promise<void> | undefined;
    private runCompletion: Promise<void> | undefined;
    private executionStarted = false;
    private started = false;
    private settled = false;
    private settlement: Promise<CanonicalStreamTerminalEvent> | undefined;
    private closing: Promise<void> | undefined;
    private cleanup: Promise<void> | undefined;

    constructor(private readonly options: CanonicalNativeEventStreamOptions<NativeEvent>) {
        this.closed = this.closeDeferred.promise;
        if ((options.open.retained_events?.length ?? 0) > 0 || options.open.resume_after !== undefined) {
            throw new Error('A fresh provider transport cannot resume a retained canonical stream');
        }
        const identity = { ...options.identity, stream_id: options.open.stream_id };
        const maxEventBytes = options.open.max_event_bytes ?? CONVERSATION_STREAM_MAX_EVENT_BYTES;
        const maxEvents = options.open.max_events ?? CONVERSATION_STREAM_MAX_EVENTS;
        const maxTotalBytes = options.open.max_total_bytes ?? CONVERSATION_STREAM_MAX_TOTAL_BYTES;
        const terminalSequence = Math.max(0, maxEvents - 1);
        const reservedTerminalBytes = Math.max(
            serializedBytes(terminatedEvent(identity, terminalSequence, 'cancelled')),
            serializedBytes(terminatedEvent(identity, terminalSequence, 'failed')),
            serializedBytes(terminatedEvent(identity, terminalSequence, 'failed', 'provider', false)),
            serializedBytes(terminatedEvent(identity, terminalSequence, 'failed', 'provider', true)),
            serializedBytes(terminatedEvent(identity, terminalSequence, 'failed', 'delivery')),
        );
        if (maxEventBytes < reservedTerminalBytes) {
            throw new RangeError('max_event_bytes cannot hold a canonical stream terminal event');
        }
        if (maxTotalBytes < reservedTerminalBytes) {
            throw new RangeError('max_total_bytes cannot hold a canonical stream terminal event');
        }
        if (maxEvents < 1) throw new RangeError('max_events cannot hold a canonical stream terminal event');
        this.accumulator = new ConversationStreamAccumulator(identity, {
            max_event_bytes: options.open.max_event_bytes,
            max_events: options.open.max_events,
            max_total_bytes: options.open.max_total_bytes,
            reserved_terminal_bytes: reservedTerminalBytes,
            reserved_terminal_events: 1,
        });
        this.channel = new CanonicalStreamEventChannel(options.open.max_buffered_events, async () => {
            await this.cancel();
        });
        this.writer = new CanonicalNativeStreamWriter(this.accumulator, this.channel);
    }

    get terminal_event(): CanonicalStreamTerminalEvent | undefined {
        const terminal = this.accumulator.terminal_event;
        return terminal?.type === 'response_accepted' || terminal?.type === 'stream_terminated' ? terminal : undefined;
    }

    get execution_started(): boolean {
        return this.executionStarted;
    }

    cancel(): Promise<CanonicalStreamTerminalEvent> {
        return this.beginTermination('cancelled');
    }

    [Symbol.asyncIterator](): AsyncIterator<ConversationStreamEvent> {
        if (!this.started) {
            this.started = true;
            if (!this.settled) {
                this.runCompletion = this.run().catch((error: unknown) => {
                    this.settled = true;
                    this.channel.fail(error);
                });
            }
        }
        return this.channel[Symbol.asyncIterator]();
    }

    private async run(): Promise<void> {
        try {
            await this.writer.start();
            if (this.settled) return;
            // Let the draft_started consumer request cancellation before an inference transport is opened.
            await nextTask();
            if (this.settled) return;
            this.opening = Promise.resolve().then(async () => {
                if (this.settled) return undefined;
                this.executionStarted = true;
                return this.options.openSource();
            });
            const source = await this.opening;
            if (source === undefined) return;
            if (this.settled) return;
            this.iterator = source[Symbol.asyncIterator]();
            while (!this.settled) {
                const next = await this.iterator.next();
                if (next.done) break;
                await this.options.map(next.value, this.writer);
            }
            if (this.settled) return;
            const accepted = await this.options.finalize(this.writer);
            this.completion = accepted.response;
            if (this.settled) return;
            const prepared = await accepted.prepare_reconciliation?.(this.writer);
            const finalized = prepared === undefined ? accepted : { ...accepted, ...prepared };
            const reconciliations = finalized.reconciliations;
            if (reconciliations === undefined) {
                throw new Error('Canonical native stream finalization has no reconciliations');
            }
            await finalized.deliver_final_events?.(this.writer);
            if (this.settled) return;
            await this.beginAcceptance({ ...finalized, reconciliations });
        } catch (error: unknown) {
            if (this.settlement === undefined) {
                try {
                    const failureKind = this.completion === undefined ? 'provider' : 'delivery';
                    await this.beginTermination(
                        'failed',
                        failureKind,
                        failureKind === 'provider' ? this.safeFailureClassification(error) : undefined,
                    );
                } catch (settlementError: unknown) {
                    this.settled = true;
                    this.channel.fail(settlementError ?? error);
                }
            }
        }
    }

    private beginAcceptance(
        finalized: ReconciledCanonicalNativeStreamFinalization,
    ): Promise<CanonicalStreamTerminalEvent> {
        if (this.settlement !== undefined) return this.settlement;
        this.settled = true;
        let resolve!: (terminal: CanonicalStreamTerminalEvent) => void;
        let reject!: (error: unknown) => void;
        const settlement = new Promise<CanonicalStreamTerminalEvent>((resolvePromise, rejectPromise) => {
            resolve = resolvePromise;
            reject = rejectPromise;
        });
        this.settlement = settlement;
        this.startCleanup();
        void this.acceptInternal(finalized).then(resolve, reject);
        return settlement;
    }

    private async acceptInternal(
        finalized: ReconciledCanonicalNativeStreamFinalization,
    ): Promise<CanonicalStreamTerminalEvent> {
        try {
            const accepted = await finalizeCanonicalExecutionStreamResponse({
                accumulator: this.accumulator,
                decoded: finalized.decoded,
                response: finalized.response,
                reconciliations: finalized.reconciliations,
                ...(finalized.result_schema === undefined ? {} : { result_schema: finalized.result_schema }),
            });
            await this.channel.terminate(accepted);
            return accepted;
        } catch {
            const existing = this.terminal_event;
            if (existing !== undefined) {
                await this.channel.terminate(existing);
                return existing;
            }
            return this.terminateInternal('failed', 'delivery');
        }
    }

    private beginTermination(
        outcome: 'cancelled' | 'failed',
        failureKind: 'provider' | 'delivery' = 'provider',
        retryable?: boolean,
    ): Promise<CanonicalStreamTerminalEvent> {
        if (this.settlement !== undefined) return this.settlement;
        this.settled = true;
        let resolve!: (terminal: CanonicalStreamTerminalEvent) => void;
        let reject!: (error: unknown) => void;
        const settlement = new Promise<CanonicalStreamTerminalEvent>((resolvePromise, rejectPromise) => {
            resolve = resolvePromise;
            reject = rejectPromise;
        });
        this.settlement = settlement;
        try {
            this.options.abort();
        } catch {
            // The bounded terminal still settles while cleanup retains transport ownership.
        }
        this.startCleanup();
        void this.terminateInternal(outcome, failureKind, retryable).then(resolve, reject);
        return settlement;
    }

    private safeFailureClassification(error: unknown): boolean | undefined {
        if (LlumiverseError.isLlumiverseError(error)) return error.retryable;
        try {
            return this.options.classifyFailure?.(error);
        } catch {
            return undefined;
        }
    }

    private async terminateInternal(
        outcome: 'cancelled' | 'failed',
        failureKind: 'provider' | 'delivery',
        retryable?: boolean,
    ): Promise<CanonicalStreamTerminalEvent> {
        const terminal = this.appendTerminal(outcome, failureKind, retryable);
        await this.channel.terminate(terminal);
        return terminal;
    }

    private appendTerminal(
        outcome: 'cancelled' | 'failed',
        failureKind: 'provider' | 'delivery' = 'provider',
        retryable?: boolean,
    ): CanonicalStreamTerminalEvent {
        const event = terminatedEvent(
            this.accumulator.identity,
            this.accumulator.next_sequence,
            outcome,
            failureKind,
            retryable,
        );
        return this.accumulator.append(event).event as CanonicalStreamTerminalEvent;
    }

    private closeOnce(): Promise<void> {
        this.closing ??= Promise.resolve().then(() => this.options.close());
        return this.closing;
    }

    private startCleanup(): void {
        if (this.cleanup !== undefined) return;
        this.cleanup = Promise.allSettled([this.cleanupIterator(), this.closeOnce()]).then(() => undefined);
        const runCompletion = this.runCompletion;
        void Promise.allSettled(runCompletion === undefined ? [this.cleanup] : [this.cleanup, runCompletion]).then(
            () => {
                this.closeDeferred.resolve();
            },
        );
        void this.cleanup.catch(() => {
            // Promise.allSettled keeps cleanup non-rejecting; retain a defensive handler.
        });
    }

    private cleanupIterator(): Promise<void> {
        this.iteratorCleanup ??= (async () => {
            try {
                const source = await this.opening;
                if (source !== undefined && this.iterator === undefined) {
                    this.iterator = source[Symbol.asyncIterator]();
                }
                await this.iterator?.return?.();
            } catch {
                // Transport opening and iterator cleanup are best-effort after cancellation or failure.
            }
        })();
        return this.iteratorCleanup;
    }
}

export function canonicalNativeExecutionEventStream<NativeEvent>(
    options: CanonicalNativeEventStreamOptions<NativeEvent>,
): CanonicalExecutionEventStream {
    return new CanonicalNativeExecutionEventStream(options);
}
