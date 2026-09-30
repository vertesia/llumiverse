import {
    assertConversationStreamDecodeEvidence,
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    CONVERSATION_STREAM_MAX_EVENT_BYTES,
    CONVERSATION_STREAM_MAX_EVENTS,
    CONVERSATION_STREAM_MAX_TOTAL_BYTES,
    ConversationStreamAccumulator,
    type ConversationStreamAccumulatorOptions,
    type ConversationStreamCursor,
    type ConversationStreamEvent,
    type ConversationStreamIdentity,
    type ConversationStreamReconciliation,
    canonicalJsonContentString,
    conversationStreamEventId,
    type DecodedConversationResponse,
} from '@llumiverse/conversation';
import {
    CanonicalAcceptedOutputRecovered,
    type CanonicalExecutionResponse,
    type CanonicalExecutionStream,
    canonicalExecutionPreview,
} from './CanonicalExecution.js';
import { normalizeCompletionResult } from './validation.js';

export type CanonicalStreamTerminalEvent = Extract<
    ConversationStreamEvent,
    { type: 'response_accepted' | 'stream_terminated' }
>;

export interface CanonicalStreamOpenOptions
    extends Pick<
        ConversationStreamAccumulatorOptions,
        'max_event_bytes' | 'max_events' | 'max_total_bytes' | 'retained_events'
    > {
    stream_id: string;
    resume_after?: ConversationStreamCursor;
    max_buffered_events?: number;
}

export interface CanonicalExecutionEventStream extends AsyncIterable<ConversationStreamEvent> {
    readonly completion: CanonicalExecutionResponse | undefined;
    readonly terminal_event: CanonicalStreamTerminalEvent | undefined;
    /**
     * True once execution may have performed provider work: the finite execute callback or native openSource was
     * entered. Retained replay, explicit accepted recovery, and failure before execution leave this false.
     */
    readonly execution_started: boolean;
    /** Fulfills after provider execution and owned transport cleanup have finished. */
    readonly closed: Promise<void>;
    cancel(): Promise<CanonicalStreamTerminalEvent>;
}

interface Deferred<T> {
    promise: Promise<T>;
    resolve(value: T): void;
    reject(reason: unknown): void;
}

function deferred<T>(): Deferred<T> {
    let resolvePromise: ((value: T) => void) | undefined;
    let rejectPromise: ((reason: unknown) => void) | undefined;
    const promise = new Promise<T>((resolve, reject) => {
        resolvePromise = resolve;
        rejectPromise = reject;
    });
    if (resolvePromise === undefined || rejectPromise === undefined)
        throw new Error('Failed to create deferred promise');
    return { promise, resolve: resolvePromise, reject: rejectPromise };
}

function boundedQueueSize(value: number | undefined): number {
    const resolved = value ?? 32;
    if (!Number.isSafeInteger(resolved) || resolved <= 0) {
        throw new RangeError('max_buffered_events must be a positive safe integer');
    }
    return resolved;
}

/**
 * Single-consumer bounded event channel. A terminal event bypasses the normal capacity wait so cancellation cannot
 * deadlock behind an unread buffer. Earlier events remain ordered before that terminal.
 */
export class CanonicalStreamEventChannel implements AsyncIterable<ConversationStreamEvent> {
    private readonly buffered: ConversationStreamEvent[] = [];
    private readonly blockedProducers: Array<{ event: ConversationStreamEvent; completion: Deferred<void> }> = [];
    private pendingConsumer: Deferred<IteratorResult<ConversationStreamEvent>> | undefined;
    private terminal: CanonicalStreamTerminalEvent | undefined;
    private terminalDelivered = false;
    private failure: unknown;
    private closed = false;
    private iteratorCreated = false;

    constructor(
        private readonly maxBufferedEvents = 32,
        private readonly onReturn?: () => Promise<void>,
    ) {
        boundedQueueSize(maxBufferedEvents);
    }

    get terminal_event(): CanonicalStreamTerminalEvent | undefined {
        return this.terminal;
    }

    async emit(event: ConversationStreamEvent): Promise<void> {
        if (event.type === 'response_accepted' || event.type === 'stream_terminated') {
            throw new Error('Canonical stream terminal events must use terminate()');
        }
        if (this.terminal !== undefined || this.closed) throw new Error('Canonical stream event channel is terminated');
        if (this.pendingConsumer !== undefined && this.buffered.length === 0) {
            const consumer = this.pendingConsumer;
            this.pendingConsumer = undefined;
            consumer.resolve({ value: event, done: false });
            return;
        }
        if (this.buffered.length >= this.maxBufferedEvents) {
            // Native producers are deliberately single-threaded. Keeping one blocked event bounds retained
            // delivery memory to max_buffered_events + one event + one terminal while allowing cancellation
            // to preserve the already-assigned sequence without waiting for a consumer.
            if (this.blockedProducers.length >= 1) {
                throw new Error('Canonical stream event channel supports one blocked producer');
            }
            const completion = deferred<void>();
            this.blockedProducers.push({ event, completion });
            await completion.promise;
            return;
        }
        this.buffered.push(event);
    }

    async terminate(event: CanonicalStreamTerminalEvent): Promise<void> {
        if (this.terminal !== undefined) {
            if (canonicalJsonContentString(this.terminal) !== canonicalJsonContentString(event)) {
                throw new Error('Canonical stream event channel already has a different terminal event');
            }
            return;
        }
        this.terminal = event;
        this.closed = true;
        for (const producer of this.blockedProducers.splice(0)) {
            this.buffered.push(producer.event);
            producer.completion.resolve(undefined);
        }
        if (this.pendingConsumer !== undefined && this.buffered.length === 0) {
            const consumer = this.pendingConsumer;
            this.pendingConsumer = undefined;
            this.terminalDelivered = true;
            consumer.resolve({ value: event, done: false });
            return;
        }
        this.buffered.push(event);
    }

    close(): void {
        if (this.terminal !== undefined) return;
        this.closed = true;
        for (const producer of this.blockedProducers.splice(0)) {
            producer.completion.reject(new Error('Canonical stream event channel is closed'));
        }
        if (this.pendingConsumer !== undefined && this.buffered.length === 0) {
            const consumer = this.pendingConsumer;
            this.pendingConsumer = undefined;
            consumer.resolve({ value: undefined, done: true });
        }
    }

    /** Abandon this delivery without claiming that buffered retained events were replayed. */
    abandon(): void {
        if (this.terminalDelivered) return;
        this.closed = true;
        this.buffered.splice(0);
        for (const producer of this.blockedProducers.splice(0)) {
            producer.completion.reject(new Error('Canonical stream event delivery was abandoned'));
        }
        if (this.pendingConsumer !== undefined) {
            const consumer = this.pendingConsumer;
            this.pendingConsumer = undefined;
            consumer.resolve({ value: undefined, done: true });
        }
    }

    fail(error: unknown): void {
        if (this.terminal !== undefined) return;
        this.failure = error;
        this.closed = true;
        this.buffered.splice(0);
        for (const producer of this.blockedProducers.splice(0)) producer.completion.reject(error);
        if (this.pendingConsumer !== undefined) {
            const consumer = this.pendingConsumer;
            this.pendingConsumer = undefined;
            consumer.reject(error);
        }
    }

    [Symbol.asyncIterator](): AsyncIterator<ConversationStreamEvent> {
        if (this.iteratorCreated) throw new Error('Canonical stream event channel can only be consumed once');
        this.iteratorCreated = true;
        return {
            next: () => this.next(),
            return: async () => {
                await this.onReturn?.();
                return { value: undefined, done: true };
            },
        };
    }

    private async next(): Promise<IteratorResult<ConversationStreamEvent>> {
        const event = this.buffered.shift();
        if (event !== undefined) {
            const producer = this.blockedProducers.shift();
            if (producer !== undefined) {
                this.buffered.push(producer.event);
                producer.completion.resolve(undefined);
            }
            if (event.type === 'response_accepted' || event.type === 'stream_terminated') this.terminalDelivered = true;
            return { value: event, done: false };
        }
        if (this.failure !== undefined) throw this.failure;
        if (this.closed || this.terminalDelivered) return { value: undefined, done: true };
        if (this.pendingConsumer !== undefined) throw new Error('Canonical stream event channel has a pending read');
        this.pendingConsumer = deferred<IteratorResult<ConversationStreamEvent>>();
        return this.pendingConsumer.promise;
    }
}

function streamEnvelope(identity: ConversationStreamIdentity, sequence: number) {
    return {
        format: CONVERSATION_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        ...identity,
        sequence,
        event_id: conversationStreamEventId(identity.stream_id, sequence),
    };
}

function serializedEventBytes(event: ConversationStreamEvent): number {
    return new TextEncoder().encode(JSON.stringify(event)).byteLength;
}

function terminatedEvent(
    identity: ConversationStreamIdentity,
    sequence: number,
    outcome: 'cancelled' | 'failed',
    failureKind: 'execution' | 'delivery' = 'execution',
): Extract<ConversationStreamEvent, { type: 'stream_terminated' }> {
    return {
        ...streamEnvelope(identity, sequence),
        type: 'stream_terminated',
        outcome,
        ...(outcome === 'failed'
            ? failureKind === 'delivery'
                ? {
                      diagnostic: {
                          code: 'CANONICAL_EVENT_DELIVERY_FAILED',
                          message: 'Canonical event delivery failed',
                      },
                  }
                : { diagnostic: { code: 'CANONICAL_EXECUTION_FAILED', message: 'Canonical execution failed' } }
            : {}),
    };
}

function acceptedEvent(
    identity: ConversationStreamIdentity,
    sequence: number,
    response: CanonicalExecutionResponse,
    origin: 'live_transport' | 'accepted_recovery',
    reconciliations: ConversationStreamReconciliation[] = [],
): CanonicalStreamTerminalEvent {
    const output = response.accepted_output;
    if (
        output.receipt.id !== identity.response_operation_id ||
        output.generation.id !== identity.generation_id ||
        output.generation.request_id !== identity.request_id ||
        output.generation.attempt_id !== identity.attempt_id ||
        output.turn.id !== identity.draft_turn_id
    ) {
        throw new Error('Canonical execution response does not match its stream identity');
    }
    return {
        ...streamEnvelope(identity, sequence),
        type: 'response_accepted',
        origin,
        conversation: { conversation_id: response.conversation.id, revision: response.conversation.revision },
        operation_receipt_id: output.receipt.id,
        committed_turn_id: output.turn.id,
        turn_status: output.turn.status,
        generation_status: output.generation.status,
        committed_block_ids: output.turn.blocks.map((block) => block.id),
        accepted_asset_ids: Object.keys(output.assets),
        reconciliations,
    };
}

function decodedBlocks(decoded: DecodedConversationResponse) {
    return new Map(decoded.turns.flatMap((turn) => turn.blocks.map((block) => [block.id, block] as const)));
}

function assertDirectDraftBinding(
    accumulator: ConversationStreamAccumulator,
    decoded: DecodedConversationResponse,
    reconciliations: readonly ConversationStreamReconciliation[],
): void {
    const blocks = decodedBlocks(decoded);
    const mappings = new Map(
        (decoded.stream_evidence?.item_mappings ?? [])
            .filter((mapping) => mapping.kind === 'block')
            .map((mapping) => [mapping.canonical_id, mapping.native_position]),
    );
    const reconciliationByDraft = new Map(
        reconciliations.flatMap((reconciliation) =>
            reconciliation.draft_block_ids.map((draftId) => [draftId, reconciliation] as const),
        ),
    );
    for (const draft of accumulator.draft_snapshot()) {
        const reconciliation = reconciliationByDraft.get(draft.draft_block_id);
        if (reconciliation?.disposition !== 'direct') continue;
        const block = blocks.get(reconciliation.committed_block_ids[0] ?? '');
        if (block === undefined || block.type !== draft.type) {
            throw new Error(`Direct stream reconciliation changes draft block type ${draft.draft_block_id}`);
        }
        const mappedPosition = mappings.get(block.id);
        if (
            mappedPosition === undefined ||
            reconciliation.native_positions.length !== 1 ||
            canonicalJsonContentString(mappedPosition) !==
                canonicalJsonContentString(reconciliation.native_positions[0])
        ) {
            throw new Error(`Direct stream reconciliation changes native position ${draft.draft_block_id}`);
        }
        if (draft.type === 'tool_call') {
            if (
                block.type !== 'tool_call' ||
                block.executor !== draft.executor ||
                (draft.call_id !== undefined && block.call_id !== draft.call_id) ||
                (draft.tool_name !== undefined && block.tool_name !== draft.tool_name)
            ) {
                throw new Error(`Tool stream reconciliation changes identity or executor ${draft.draft_block_id}`);
            }
        } else if (
            (draft.type === 'text' || draft.type === 'reasoning') &&
            (block.type !== draft.type || block.text !== draft.text)
        ) {
            throw new Error(`Direct stream reconciliation changes emitted text ${draft.draft_block_id}`);
        }
    }
}

async function assertStructuredReconciliation(
    accumulator: ConversationStreamAccumulator,
    decoded: DecodedConversationResponse,
    reconciliations: readonly ConversationStreamReconciliation[],
    resultSchema: object | undefined,
): Promise<void> {
    const structured = reconciliations.filter((item) => item.disposition === 'structured_output');
    if (structured.length === 0) return;
    if (resultSchema === undefined) throw new Error('Structured-output stream finalization requires its result schema');
    const evidence = decoded.stream_evidence;
    if (evidence === undefined) throw new Error('Structured-output stream finalization requires decode evidence');
    const blocks = decodedBlocks(decoded);
    const transformations = new Map(evidence.transformations.map((proof) => [proof.id, proof]));
    const mappings = new Map(
        evidence.item_mappings
            .filter((mapping) => mapping.kind === 'block')
            .map((mapping) => [mapping.canonical_id, mapping.native_position]),
    );
    const draftsByPosition = new Map(
        accumulator
            .draft_snapshot()
            .map((draft) => [canonicalJsonContentString(draft.native_position), draft] as const),
    );
    for (const reconciliation of structured) {
        const proof = transformations.get(reconciliation.transformation_id ?? '');
        if (proof === undefined) {
            throw new Error(`Structured-output reconciliation has no decode proof ${reconciliation.transformation_id}`);
        }
        if (proof.result_block_id !== reconciliation.committed_block_ids[0]) {
            throw new Error('Structured-output proof result does not match its committed block');
        }
        const sourceBlocks = proof.source_block_ids.map((id, index) => ({
            id,
            type: 'text' as const,
            text: proof.source_texts[index] ?? '',
            format: 'plain' as const,
        }));
        const resultBlock = blocks.get(proof.result_block_id);
        if (resultBlock?.type !== 'json') throw new Error('Structured-output result is not decoded JSON');
        const normalized = normalizeCompletionResult(
            sourceBlocks.map((block) => ({ type: 'text' as const, value: block.text })),
            resultSchema,
        );
        const normalizedJson =
            normalized.status === 'valid' ? normalized.result.find((part) => part.type === 'json') : undefined;
        if (
            normalizedJson?.type !== 'json' ||
            canonicalJsonContentString(normalizedJson.value) !== canonicalJsonContentString(resultBlock.value)
        ) {
            throw new Error('Structured-output reconciliation does not match the shared normalizer result');
        }
        const sourcePositions = proof.source_block_ids.map((id) => mappings.get(id));
        if (sourcePositions.some((position) => position === undefined)) {
            throw new Error('Structured-output decode evidence does not map every source block');
        }
        const expected = sourcePositions.map((position) => canonicalJsonContentString(position)).sort();
        const actual = reconciliation.native_positions.map((position) => canonicalJsonContentString(position)).sort();
        if (canonicalJsonContentString(expected) !== canonicalJsonContentString(actual)) {
            throw new Error('Structured-output reconciliation positions do not match decode evidence');
        }
        for (const [index, position] of sourcePositions.entries()) {
            if (position === undefined) throw new Error('Structured-output source position is missing');
            const draft = draftsByPosition.get(canonicalJsonContentString(position));
            if (
                draft === undefined ||
                !reconciliation.draft_block_ids.includes(draft.draft_block_id) ||
                draft.type !== 'text'
            ) {
                throw new Error('Structured-output source is not bound to a text draft at its native position');
            }
            if (draft.text !== proof.source_texts[index]) {
                throw new Error('Structured-output source text differs from emitted draft text');
            }
        }
    }
}

function assertAcceptedResponseMatchesDecode(
    decoded: DecodedConversationResponse,
    response: CanonicalExecutionResponse,
): void {
    const output = response.accepted_output;
    if (
        decoded.generation.id !== output.generation.id ||
        decoded.generation.request_id !== output.generation.request_id ||
        decoded.generation.attempt_id !== output.generation.attempt_id ||
        decoded.generation.status !== output.generation.status
    ) {
        throw new Error('Accepted stream generation does not match decoded generation');
    }
    const turn = decoded.turns.find((candidate) => candidate.id === output.turn.id);
    if (turn === undefined || turn.status !== output.turn.status) {
        throw new Error('Accepted stream turn does not match decoded turn');
    }
    const blocks = new Map(turn.blocks.map((block) => [block.id, block]));
    for (const accepted of output.turn.blocks) {
        const decodedBlock = blocks.get(accepted.id);
        if (decodedBlock === undefined) throw new Error(`Accepted stream block ${accepted.id} is absent from decode`);
        const comparable = structuredClone(decodedBlock) as Record<string, unknown>;
        if (decodedBlock.type === 'tool_call') {
            Reflect.deleteProperty(comparable, 'native_id');
            Reflect.deleteProperty(comparable, 'definition_id');
        }
        if (canonicalJsonContentString(comparable) !== canonicalJsonContentString(accepted)) {
            throw new Error(`Accepted stream block ${accepted.id} differs from decoded block`);
        }
    }
}

function assertTerminalOnlyBlocksHaveDecodeMappings(
    accumulator: ConversationStreamAccumulator,
    decoded: DecodedConversationResponse,
    response: CanonicalExecutionResponse,
    reconciliations: readonly ConversationStreamReconciliation[],
): void {
    if (accumulator.draft_snapshot().length > 0 || response.accepted_output.turn.blocks.length === 0) return;
    if (reconciliations.length > 0) {
        throw new Error('Zero-draft stream finalization cannot introduce native draft reconciliations');
    }
    const mappedBlockIds = new Set(
        decoded.stream_evidence?.item_mappings
            .filter((mapping) => mapping.kind === 'block')
            .map((mapping) => mapping.canonical_id) ?? [],
    );
    for (const block of response.accepted_output.turn.blocks) {
        if (!mappedBlockIds.has(block.id)) {
            throw new Error(`Terminal-only accepted block ${block.id} has no native decode mapping`);
        }
    }
}

export async function finalizeCanonicalExecutionStreamResponse(input: {
    accumulator: ConversationStreamAccumulator;
    decoded: DecodedConversationResponse;
    response: CanonicalExecutionResponse;
    reconciliations: ConversationStreamReconciliation[];
    result_schema?: object;
    origin?: 'live_transport' | 'accepted_recovery';
}): Promise<Extract<ConversationStreamEvent, { type: 'response_accepted' }>> {
    await assertConversationStreamDecodeEvidence(input.decoded);
    assertAcceptedResponseMatchesDecode(input.decoded, input.response);
    assertTerminalOnlyBlocksHaveDecodeMappings(input.accumulator, input.decoded, input.response, input.reconciliations);
    assertDirectDraftBinding(input.accumulator, input.decoded, input.reconciliations);
    await assertStructuredReconciliation(input.accumulator, input.decoded, input.reconciliations, input.result_schema);
    const event = acceptedEvent(
        input.accumulator.identity,
        input.accumulator.next_sequence,
        input.response,
        input.origin ?? 'live_transport',
        input.reconciliations,
    );
    const appended = input.accumulator.append(event).event;
    if (appended.type !== 'response_accepted') throw new Error('Canonical stream finalization lost its accepted event');
    return appended;
}

export interface FallbackCanonicalExecutionEventStreamOptions extends CanonicalStreamOpenOptions {
    origin?: 'live_transport' | 'accepted_recovery';
}

/** Finite typed delivery for a canonical sync response. It never reconstructs native draft events from preview text. */
export class FallbackCanonicalExecutionEventStream implements CanonicalExecutionEventStream {
    completion: CanonicalExecutionResponse | undefined;
    readonly closed: Promise<void>;
    private readonly abortController = new AbortController();
    private readonly accumulator: ConversationStreamAccumulator;
    private readonly channel: CanonicalStreamEventChannel;
    private readonly closeDeferred = deferred<void>();
    private readonly replayEvents: readonly ConversationStreamEvent[];
    private readonly retainedDelivery: boolean;
    private executionStarted = false;
    private started = false;
    private settled = false;

    constructor(
        identity: Omit<ConversationStreamIdentity, 'stream_id'>,
        private readonly execute: (signal: AbortSignal) => Promise<CanonicalExecutionResponse>,
        private readonly options: FallbackCanonicalExecutionEventStreamOptions,
    ) {
        this.closed = this.closeDeferred.promise;
        const streamIdentity = { ...identity, stream_id: options.stream_id };
        this.retainedDelivery = options.retained_events !== undefined;
        const maxEventBytes = options.max_event_bytes ?? CONVERSATION_STREAM_MAX_EVENT_BYTES;
        const maxEvents = options.max_events ?? CONVERSATION_STREAM_MAX_EVENTS;
        const maxTotalBytes = options.max_total_bytes ?? CONVERSATION_STREAM_MAX_TOTAL_BYTES;
        const terminalSequence = Math.max(0, maxEvents - 1);
        const reservedTerminalBytes = Math.max(
            serializedEventBytes(terminatedEvent(streamIdentity, terminalSequence, 'cancelled')),
            serializedEventBytes(terminatedEvent(streamIdentity, terminalSequence, 'failed')),
            serializedEventBytes(terminatedEvent(streamIdentity, terminalSequence, 'failed', 'delivery')),
        );
        if (!this.retainedDelivery && maxEventBytes < reservedTerminalBytes) {
            throw new RangeError('max_event_bytes cannot hold a canonical stream terminal event');
        }
        if (!this.retainedDelivery && maxTotalBytes < reservedTerminalBytes) {
            throw new RangeError('max_total_bytes cannot hold a canonical stream terminal event');
        }
        if (!this.retainedDelivery && maxEvents < 1) {
            throw new RangeError('max_events cannot hold a canonical stream terminal event');
        }
        this.accumulator = new ConversationStreamAccumulator(streamIdentity, {
            retained_events: options.retained_events,
            resume_after: options.resume_after,
            max_event_bytes: options.max_event_bytes,
            max_events: options.max_events,
            max_total_bytes: options.max_total_bytes,
            ...(!this.retainedDelivery
                ? { reserved_terminal_bytes: reservedTerminalBytes, reserved_terminal_events: 1 }
                : {}),
        });
        this.replayEvents = this.accumulator.eventsAfter(options.resume_after);
        if (options.retained_events !== undefined && this.accumulator.terminal_event === undefined) {
            throw new Error('Finite canonical fallback cannot resume a retained nonterminal native stream');
        }
        this.channel = new CanonicalStreamEventChannel(boundedQueueSize(options.max_buffered_events), async () => {
            await this.cancel();
        });
    }

    get terminal_event(): CanonicalStreamTerminalEvent | undefined {
        const event = this.accumulator.terminal_event;
        return event?.type === 'response_accepted' || event?.type === 'stream_terminated' ? event : undefined;
    }

    get execution_started(): boolean {
        return this.executionStarted;
    }

    async cancel(): Promise<CanonicalStreamTerminalEvent> {
        const terminal = this.terminal_event;
        if (terminal !== undefined) {
            if (this.retainedDelivery && !this.settled) {
                this.settled = true;
                this.channel.abandon();
            } else {
                await this.channel.terminate(terminal);
            }
            if (!this.started) this.closeDeferred.resolve();
            return terminal;
        }
        this.abortController.abort();
        const cancelled = await this.settleTerminated('cancelled');
        if (!this.started) this.closeDeferred.resolve();
        return cancelled;
    }

    [Symbol.asyncIterator](): AsyncIterator<ConversationStreamEvent> {
        if (!this.started) {
            this.started = true;
            void this.run();
        }
        return this.channel[Symbol.asyncIterator]();
    }

    private async run(): Promise<void> {
        try {
            if (this.replayEvents.length > 0 || this.accumulator.terminal_event !== undefined) {
                for (const event of this.replayEvents) {
                    if (event.type === 'response_accepted' || event.type === 'stream_terminated') {
                        await this.channel.terminate(event);
                    } else {
                        await this.channel.emit(event);
                    }
                }
                if (this.replayEvents.length === 0) this.channel.close();
                return;
            }
            if (this.abortController.signal.aborted) return;
            if (this.options.origin !== 'accepted_recovery') this.executionStarted = true;
            const response = await this.execute(this.abortController.signal);
            if (this.abortController.signal.aborted || this.settled) {
                this.completion = response;
                return;
            }
            const event = acceptedEvent(
                this.accumulator.identity,
                this.accumulator.next_sequence,
                response,
                this.options.origin ?? 'live_transport',
            );
            this.completion = response;
            this.accumulator.append(event);
            this.settled = true;
            await this.channel.terminate(event);
        } catch (error: unknown) {
            if (CanonicalAcceptedOutputRecovered.is(error)) {
                this.settled = true;
                this.channel.fail(error);
                return;
            }
            if (!this.settled) {
                try {
                    await this.settleTerminated('failed', this.completion === undefined ? 'execution' : 'delivery');
                } catch (settlementError: unknown) {
                    this.settled = true;
                    this.channel.fail(settlementError);
                }
            }
        } finally {
            this.closeDeferred.resolve();
        }
    }

    private async settleTerminated(
        outcome: 'cancelled' | 'failed',
        failureKind: 'execution' | 'delivery' = 'execution',
    ): Promise<CanonicalStreamTerminalEvent> {
        const existing = this.terminal_event;
        if (existing !== undefined) return existing;
        const event = terminatedEvent(this.accumulator.identity, this.accumulator.next_sequence, outcome, failureKind);
        this.accumulator.append(event);
        this.settled = true;
        await this.channel.terminate(event);
        return event;
    }
}

/** Explicit compatibility projection. Canonical events and the final canonical response remain authoritative. */
export class LegacyCanonicalExecutionEventProjection implements CanonicalExecutionStream {
    private iteratorCreated = false;

    constructor(
        private readonly source: CanonicalExecutionEventStream,
        private readonly includeReasoning = false,
    ) {}

    get completion(): CanonicalExecutionResponse | undefined {
        return this.source.completion;
    }

    async cancel(): Promise<void> {
        await this.source.cancel();
    }

    [Symbol.asyncIterator](): AsyncIterator<string> {
        if (this.iteratorCreated) throw new Error('Canonical execution stream can only be consumed once');
        this.iteratorCreated = true;
        const self = this;
        return (async function* () {
            let projectedDraft = false;
            for await (const event of self.source) {
                if (event.type === 'draft_text_delta') {
                    projectedDraft = true;
                    yield event.text;
                } else if (event.type === 'draft_reasoning_delta' && self.includeReasoning) {
                    projectedDraft = true;
                    yield event.text;
                } else if (
                    event.type === 'response_accepted' &&
                    !projectedDraft &&
                    self.source.completion !== undefined
                ) {
                    const preview = canonicalExecutionPreview(self.source.completion, self.includeReasoning);
                    if (preview.length > 0) yield preview;
                }
            }
        })();
    }
}
