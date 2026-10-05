import { canonicalJsonContentString } from './json-content-runtime.js';
import { preflightJsonInput } from './json-preflight.js';
import type {
    ConversationStreamCursor,
    ConversationStreamEvent,
    ConversationStreamIdentity,
    NativeStreamPosition,
} from './types.js';

export { canonicalJsonContentString } from './json-content-runtime.js';
export { preflightJsonInput } from './json-preflight.js';

export interface ConversationStreamRuntimeValidators {
    identity(input: unknown): ConversationStreamIdentity;
    cursor(input: unknown): ConversationStreamCursor;
    event(input: unknown): ConversationStreamEvent;
}

export const CONVERSATION_STREAM_MAX_EVENT_BYTES = 256 * 1024;
export const CONVERSATION_STREAM_MAX_EVENTS = 100_000;
export const CONVERSATION_STREAM_MAX_TOTAL_BYTES = 32 * 1024 * 1024;

export interface ConversationStreamAccumulatorOptions {
    retained_events?: readonly ConversationStreamEvent[];
    resume_after?: ConversationStreamCursor;
    max_event_bytes?: number;
    max_events?: number;
    max_total_bytes?: number;
    reserved_terminal_events?: number;
    reserved_terminal_bytes?: number;
}

export interface ConversationStreamDraftSnapshot {
    draft_block_id: string;
    native_position: NativeStreamPosition;
    type: 'text' | 'reasoning' | 'tool_call' | 'image' | 'audio' | 'video' | 'document';
    executor?: 'application' | 'provider';
    call_id?: string;
    tool_name?: string;
    text?: string;
    finished: boolean;
}

function positiveLimit(value: number | undefined, fallback: number, label: string): number {
    const resolved = value ?? fallback;
    if (!Number.isSafeInteger(resolved) || resolved <= 0)
        throw new RangeError(`${label} must be a positive safe integer`);
    return resolved;
}

function nonnegativeLimit(value: number | undefined, fallback: number, label: string): number {
    const resolved = value ?? fallback;
    if (!Number.isSafeInteger(resolved) || resolved < 0)
        throw new RangeError(`${label} must be a nonnegative safe integer`);
    return resolved;
}

export function conversationStreamEventId(streamId: string, sequence: number): string {
    if (streamId.length === 0) throw new TypeError('stream_id must not be empty');
    if (!Number.isSafeInteger(sequence) || sequence < 0)
        throw new RangeError('sequence must be a nonnegative safe integer');
    return `${streamId}#${sequence}`;
}

export function conversationStreamCursor(event: ConversationStreamEvent): ConversationStreamCursor {
    return { stream_id: event.stream_id, event_id: event.event_id, sequence: event.sequence };
}

function deepFreeze<T>(value: T): Readonly<T> {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
        for (const child of Object.values(value)) deepFreeze(child);
        Object.freeze(value);
    }
    return value;
}

function immutableClone<T>(value: T): Readonly<T> {
    return deepFreeze(structuredClone(value));
}

function assertSmallJsonContract(value: unknown, label: string): void {
    const preflight = preflightJsonInput(value, {
        max_depth: 4,
        max_nodes: 64,
        max_bytes: 64 * 1024,
        max_string_bytes: 16 * 1024,
        max_array_length: 32,
        max_object_properties: 32,
    });
    if (!preflight.success) throw new TypeError(`${label} failed bounded JSON preflight`);
}

function positionKey(position: NativeStreamPosition): string {
    return canonicalJsonContentString(position);
}

function sameEvent(first: ConversationStreamEvent, second: ConversationStreamEvent): boolean {
    return canonicalJsonContentString(first) === canonicalJsonContentString(second);
}

function assertIdentity(event: ConversationStreamEvent, identity: ConversationStreamIdentity): void {
    for (const key of [
        'stream_id',
        'request_id',
        'attempt_id',
        'response_operation_id',
        'generation_id',
        'draft_turn_id',
    ] as const) {
        if (event[key] !== identity[key]) throw new Error(`Conversation stream event changes ${key}`);
    }
}

/**
 * Bounded request-scoped draft validator. Retained events are explicit input; this class has no global stream registry.
 */
export class ConversationStreamAccumulatorRuntime {
    private readonly retained: ConversationStreamEvent[] = [];
    private readonly drafts = new Map<string, ConversationStreamDraftSnapshot>();
    private readonly positions = new Map<string, string>();
    private readonly maxEventBytes: number;
    private readonly maxEvents: number;
    private readonly maxTotalBytes: number;
    private readonly reservedTerminalEvents: number;
    private readonly reservedTerminalBytes: number;
    private totalBytes = 0;
    private started = false;
    private draftFinished = false;
    private terminal: ConversationStreamEvent | undefined;
    readonly identity: Readonly<ConversationStreamIdentity>;

    constructor(
        identityInput: ConversationStreamIdentity,
        private readonly validators: ConversationStreamRuntimeValidators,
        options: ConversationStreamAccumulatorOptions = {},
    ) {
        assertSmallJsonContract(identityInput, 'Conversation stream identity');
        this.identity = immutableClone(this.validators.identity(identityInput));
        this.maxEventBytes = positiveLimit(
            options.max_event_bytes,
            CONVERSATION_STREAM_MAX_EVENT_BYTES,
            'max_event_bytes',
        );
        this.maxEvents = positiveLimit(options.max_events, CONVERSATION_STREAM_MAX_EVENTS, 'max_events');
        this.maxTotalBytes = positiveLimit(
            options.max_total_bytes,
            CONVERSATION_STREAM_MAX_TOTAL_BYTES,
            'max_total_bytes',
        );
        this.reservedTerminalBytes = nonnegativeLimit(options.reserved_terminal_bytes, 0, 'reserved_terminal_bytes');
        if (this.reservedTerminalBytes > this.maxTotalBytes) {
            throw new RangeError('reserved_terminal_bytes exceeds max_total_bytes');
        }
        this.reservedTerminalEvents = nonnegativeLimit(options.reserved_terminal_events, 0, 'reserved_terminal_events');
        if (this.reservedTerminalEvents > this.maxEvents) {
            throw new RangeError('reserved_terminal_events exceeds max_events');
        }
        for (const event of options.retained_events ?? []) this.append(event);
        if (options.resume_after !== undefined) this.assertResumeCursor(options.resume_after);
    }

    get next_sequence(): number {
        return this.retained.length;
    }

    get terminal_event(): Readonly<ConversationStreamEvent> | undefined {
        return this.terminal;
    }

    get retained_events(): readonly ConversationStreamEvent[] {
        return Object.freeze([...this.retained]);
    }

    draft_snapshot(): readonly ConversationStreamDraftSnapshot[] {
        return Object.freeze([...this.drafts.values()].map((draft) => immutableClone(draft)));
    }

    eventsAfter(cursor?: ConversationStreamCursor): readonly ConversationStreamEvent[] {
        if (cursor === undefined) return Object.freeze([...this.retained]);
        this.assertResumeCursor(cursor);
        return Object.freeze(this.retained.slice(cursor.sequence + 1));
    }

    append(input: unknown): { event: ConversationStreamEvent; applied: boolean } {
        const preflight = preflightJsonInput(input, {
            max_bytes: this.maxEventBytes,
            max_string_bytes: this.maxEventBytes,
        });
        if (!preflight.success) {
            throw new TypeError('Conversation stream event exceeds max_event_bytes or failed bounded JSON preflight');
        }
        let parsed: ConversationStreamEvent;
        try {
            parsed = this.validators.event(input);
        } catch (cause) {
            throw new TypeError('Conversation stream event failed schema validation', { cause });
        }
        const event = immutableClone(parsed) as ConversationStreamEvent;
        assertIdentity(event, this.identity);
        const expectedEventId = conversationStreamEventId(event.stream_id, event.sequence);
        if (event.event_id !== expectedEventId) throw new Error('Conversation stream event_id does not match sequence');

        if (event.sequence < this.retained.length) {
            const retained = this.retained[event.sequence];
            if (retained === undefined || !sameEvent(retained, event)) {
                throw new Error(`Conversation stream sequence ${event.sequence} conflicts with retained event`);
            }
            return { event: retained, applied: false };
        }
        if (event.sequence !== this.retained.length) {
            throw new Error(
                `Conversation stream expected sequence ${this.retained.length}, received ${event.sequence}`,
            );
        }
        if (this.terminal !== undefined) throw new Error('Conversation stream cannot emit after its terminal event');
        const isTerminal = event.type === 'response_accepted' || event.type === 'stream_terminated';
        const reservedEvents = isTerminal ? 0 : this.reservedTerminalEvents;
        if (this.retained.length + 1 + reservedEvents > this.maxEvents) {
            throw new Error('Conversation stream exceeds max_events');
        }
        const bytes = preflight.bytes;
        const reservedBytes = isTerminal ? 0 : this.reservedTerminalBytes;
        if (this.totalBytes + bytes + reservedBytes > this.maxTotalBytes)
            throw new Error('Conversation stream exceeds max_total_bytes');

        this.assertTransition(event);
        this.retained.push(event);
        this.totalBytes += bytes;
        if (event.type === 'response_accepted' || event.type === 'stream_terminated') this.terminal = event;
        return { event, applied: true };
    }

    private assertResumeCursor(cursor: ConversationStreamCursor): void {
        assertSmallJsonContract(cursor, 'Conversation stream cursor');
        const parsed = this.validators.cursor(cursor);
        if (parsed.stream_id !== this.identity.stream_id)
            throw new Error('Conversation stream cursor has wrong stream_id');
        const retained = this.retained[parsed.sequence];
        if (retained === undefined || retained.event_id !== parsed.event_id) {
            throw new Error('Conversation stream resume cursor is not present in the retained event log');
        }
    }

    private draftFor(event: Extract<ConversationStreamEvent, { draft_block_id: string }>) {
        const draft = this.drafts.get(event.draft_block_id);
        if (draft === undefined)
            throw new Error(`Conversation stream draft block ${event.draft_block_id} has not started`);
        if (this.draftFinished) throw new Error('Conversation stream draft already finished');
        if (positionKey(draft.native_position) !== positionKey(event.native_position)) {
            throw new Error(`Conversation stream draft block ${event.draft_block_id} changed native position`);
        }
        if (draft.finished) throw new Error(`Conversation stream draft block ${event.draft_block_id} already finished`);
        return draft;
    }

    private assertTransition(event: ConversationStreamEvent): void {
        if (event.type === 'response_accepted' && event.origin === 'accepted_recovery') {
            if (this.retained.length !== 0 || this.started) {
                throw new Error('Accepted recovery must be the only event in a fresh delivery stream');
            }
            if (event.reconciliations.length > 0) {
                throw new Error('Accepted recovery cannot introduce native draft reconciliations');
            }
            return;
        }
        if (event.type === 'draft_started') {
            if (this.retained.length !== 0 || this.started)
                throw new Error('Conversation stream draft already started');
            this.started = true;
            return;
        }
        if (event.type === 'response_accepted' && !this.started) {
            // Finite sync/audio/image delivery has no provisional native drafts.
            if (event.reconciliations.length > 0) {
                throw new Error('Finite response acceptance cannot introduce native draft reconciliations');
            }
            return;
        }
        if (event.type === 'stream_terminated' && !this.started) return;
        if (!this.started) throw new Error('Conversation stream event precedes draft_started');

        if (event.type === 'draft_block_started') {
            if (this.draftFinished) throw new Error('Conversation stream draft already finished');
            if (this.drafts.has(event.draft_block_id)) throw new Error(`Duplicate draft block ${event.draft_block_id}`);
            const key = positionKey(event.native_position);
            if (this.positions.has(key)) throw new Error('Conversation stream native position is already assigned');
            this.positions.set(key, event.draft_block_id);
            this.drafts.set(event.draft_block_id, {
                draft_block_id: event.draft_block_id,
                native_position: structuredClone(event.native_position),
                type: event.block.type,
                ...(event.block.type === 'tool_call'
                    ? {
                          executor: event.block.executor,
                          ...(event.block.call_id === undefined ? {} : { call_id: event.block.call_id }),
                          ...(event.block.tool_name === undefined ? {} : { tool_name: event.block.tool_name }),
                      }
                    : {}),
                ...(event.block.type === 'text' || event.block.type === 'reasoning' ? { text: '' } : {}),
                finished: false,
            });
            return;
        }
        if (event.type === 'draft_text_delta') {
            const draft = this.draftFor(event);
            if (draft.type !== 'text') throw new Error('Text delta targets a non-text draft block');
            draft.text = `${draft.text ?? ''}${event.text}`;
            return;
        }
        if (event.type === 'draft_reasoning_delta') {
            const draft = this.draftFor(event);
            if (draft.type !== 'reasoning') {
                throw new Error('Reasoning delta targets a non-reasoning draft block');
            }
            draft.text = `${draft.text ?? ''}${event.text}`;
            return;
        }
        if (event.type === 'draft_tool_call_identity') {
            if (event.call_id === undefined && event.tool_name === undefined) {
                throw new Error('Tool call identity event has no identity value');
            }
            const draft = this.draftFor(event);
            if (draft.type !== 'tool_call') {
                throw new Error('Tool identity targets a non-tool draft block');
            }
            const updates: Partial<Pick<ConversationStreamDraftSnapshot, 'call_id' | 'tool_name'>> = {};
            for (const key of ['call_id', 'tool_name'] as const) {
                const next = event[key];
                const prior = draft[key];
                if (next === undefined) continue;
                if (prior !== undefined && !next.startsWith(prior)) {
                    throw new Error(`Tool call identity changes ${key}`);
                }
                updates[key] = next;
            }
            Object.assign(draft, updates);
            return;
        }
        if (event.type === 'draft_tool_arguments_delta') {
            if (this.draftFor(event).type !== 'tool_call') {
                throw new Error('Tool arguments target a non-tool draft block');
            }
            return;
        }
        if (event.type === 'draft_block_finished') {
            this.draftFor(event).finished = true;
            return;
        }
        if (event.type === 'draft_finished') {
            if (this.draftFinished) throw new Error('Conversation stream draft already finished');
            if (event.outcome === 'completed' && [...this.drafts.values()].some((draft) => !draft.finished)) {
                throw new Error('Completed conversation stream draft has unfinished blocks');
            }
            this.draftFinished = true;
            return;
        }
        if (event.type === 'usage_snapshot') {
            if (this.draftFinished) throw new Error('Usage snapshot follows draft_finished');
            return;
        }
        if (event.type === 'response_accepted') {
            if (!this.draftFinished) throw new Error('Live response acceptance requires draft_finished');
            this.assertReconciliations(event);
            return;
        }
        if (event.type === 'stream_terminated') return;
    }

    private assertReconciliations(event: Extract<ConversationStreamEvent, { type: 'response_accepted' }>): void {
        const reconciledDrafts = new Set<string>();
        const reconciledCommittedBlocks = new Set<string>();
        const committedBlocks = new Set(event.committed_block_ids);
        if (committedBlocks.size !== event.committed_block_ids.length) {
            throw new Error('Accepted response contains duplicate committed block IDs');
        }
        if (this.drafts.size === 0) {
            if (event.reconciliations.length > 0) {
                throw new Error('Zero-draft response acceptance cannot introduce native draft reconciliations');
            }
            return;
        }
        for (const reconciliation of event.reconciliations) {
            if (reconciliation.disposition === 'structured_output') {
                if (reconciliation.transformation_id === undefined || reconciliation.committed_block_ids.length !== 1) {
                    throw new Error('Structured-output reconciliation requires one result block and transformation');
                }
            } else if (reconciliation.transformation_id !== undefined) {
                throw new Error('Only structured-output reconciliation may name a transformation');
            }
            if (
                reconciliation.disposition === 'direct' &&
                (reconciliation.draft_block_ids.length !== 1 || reconciliation.committed_block_ids.length !== 1)
            ) {
                throw new Error('Direct reconciliation requires exactly one draft and one committed block');
            }
            if (
                (reconciliation.disposition === 'omitted_invalid' || reconciliation.disposition === 'replay_only') &&
                reconciliation.committed_block_ids.length > 0
            ) {
                throw new Error(`${reconciliation.disposition} reconciliation cannot claim committed blocks`);
            }
            const expectedPositions = new Set<string>();
            for (const draftId of reconciliation.draft_block_ids) {
                if (reconciledDrafts.has(draftId))
                    throw new Error(`Draft block ${draftId} is reconciled more than once`);
                const draft = this.drafts.get(draftId);
                if (draft === undefined) throw new Error(`Reconciliation references unknown draft block ${draftId}`);
                reconciledDrafts.add(draftId);
                expectedPositions.add(positionKey(draft.native_position));
            }
            const actualPositions = new Set(reconciliation.native_positions.map(positionKey));
            if (
                actualPositions.size !== reconciliation.native_positions.length ||
                actualPositions.size !== expectedPositions.size ||
                [...actualPositions].some((position) => !expectedPositions.has(position))
            ) {
                throw new Error('Reconciliation native positions do not match its draft blocks');
            }
            for (const blockId of reconciliation.committed_block_ids) {
                if (!committedBlocks.has(blockId)) {
                    throw new Error(`Reconciliation references unaccepted committed block ${blockId}`);
                }
                if (reconciledCommittedBlocks.has(blockId)) {
                    throw new Error(`Committed block ${blockId} is reconciled more than once`);
                }
                reconciledCommittedBlocks.add(blockId);
            }
        }
        if (reconciledDrafts.size !== this.drafts.size) {
            throw new Error('Accepted response does not reconcile every native draft block');
        }
        if (reconciledCommittedBlocks.size !== committedBlocks.size) {
            throw new Error('Accepted response does not reconcile every committed block');
        }
    }
}
