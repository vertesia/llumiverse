/**
 * Claude message threads (beta `message-threads-2026-08-12`).
 *
 * A thread keeps the conversation on the provider between requests: a create request sends the whole conversation
 * with `thread: { type: 'create' }`, and a continue request sends only the messages added since the stored response,
 * with `thread: { type: 'continue', previous_message_id }` and without `system` or `tools`. Threads lower request
 * size and time to first token; pricing, rate limits and prompt caching are unchanged.
 *
 * The conversation llumiverse returns stays complete and is the source of truth. The thread state rides along in
 * `_llumiverse_meta.claude_thread`, and every request has a full-conversation fallback:
 * - a continue is sent only when the stored state still describes the start of this request: same endpoint, same
 *   fixed fields (fingerprint), recent enough, and the first `stored_messages` messages hash to `prefix_hash`. A
 *   history rewrite (stripped images or heartbeats, pruned thinking, an edited or branched conversation) therefore
 *   starts a new thread from the local copy;
 * - a continue the provider rejects (expired, already continued, in flight) is replayed as a create;
 * - a request the provider cannot thread is sent without a thread, and later requests skip threads.
 *
 * Docs: https://platform.claude.com/docs/en/build-with-claude/message-threads
 */

import { createHash } from 'node:crypto';
import type { AnthropicBeta, BetaMessageThreadParam } from '@anthropic-ai/sdk/resources/beta.js';
import type { Message, MessageParam } from '@anthropic-ai/sdk/resources/index.js';
import {
    type ExecutionOptions,
    getConversationMeta,
    isClaudeVersionGTE,
    type Logger,
    Providers,
    setConversationMeta,
} from '@llumiverse/core';
import { CLAUDE_CUT_OFF_STOP_REASONS } from './claude-stop-reason.js';

export const MESSAGE_THREADS_BETA: AnthropicBeta = 'message-threads-2026-08-12';

/**
 * Providers whose Claude endpoint offers message threads. Amazon Bedrock and Vertex AI (Google managed) do not;
 * Bedrock Mantle is Amazon Bedrock.
 */
const THREAD_PROVIDERS = new Set<string>([Providers.anthropic, Providers.azure_foundry]);

/** A thread expires 5 minutes after its last request; past this, a continue would fail, so create directly. */
const THREAD_IDLE_LIMIT_MS = 4.5 * 60 * 1000;

/** Clock used for thread expiry. Replaceable in tests. */
export const claudeThreadClock = { now: (): number => Date.now() };

/** Thread errors after which a continue is replayed as a create. */
const REPLAYABLE_CONTINUE_ERRORS = new Set([
    'thread_not_found',
    'thread_already_continued',
    'thread_continue_in_flight',
    'thread_fingerprint_mismatch',
]);

/** Thread state stored in `_llumiverse_meta.claude_thread`. JSON-safe. */
export interface ClaudeThreadState {
    /** The latest response stored on the provider. */
    previous_message_id: string;
    /** How many leading messages of the sent conversation the provider holds, the reply included. */
    stored_messages: number;
    /** Hash of those messages, as sent and as received. */
    prefix_hash: string;
    /** Hash of the fields a thread cannot change, plus `system` and `tools`, which only the create sends. */
    fingerprint: string;
    /** Provider and base URL: a thread is bound to one endpoint and key. */
    endpoint: string;
    /** When the request that stored the thread was sent (epoch ms). */
    last_used_at: number;
    /** The create that started this thread replaced a continue the provider rejected. */
    replayed?: boolean;
}

interface ClaudeThreadMeta {
    claude_thread?: ClaudeThreadState;
    /** Set when the provider cannot thread this conversation; later requests send it whole. */
    claude_thread_disabled?: boolean;
}

/** Whether the provider and model offer message threads: Claude 4.5 and later, on the Claude API and Foundry. */
export function canUseClaudeThreads(model: string, provider: string): boolean {
    return THREAD_PROVIDERS.has(provider) && isClaudeVersionGTE(model, 4, 5);
}

/**
 * Whether this request may use a thread: the caller opted in, the provider and model offer threads, the
 * conversation has not disabled them, and there is no compaction, which cannot be combined with a thread.
 */
export function usesClaudeThreads(
    options: Pick<
        ExecutionOptions,
        'message_thread' | 'compaction' | 'compact_conversation' | 'conversation' | 'model'
    >,
    provider: string,
): boolean {
    return (
        options.message_thread === true &&
        !options.compaction &&
        !options.compact_conversation &&
        canUseClaudeThreads(options.model, provider) &&
        !readThreadMeta(options.conversation).claude_thread_disabled
    );
}

function readThreadMeta(conversation: unknown): ClaudeThreadMeta {
    return getConversationMeta(conversation) as ClaudeThreadMeta;
}

/**
 * The provider's error code (`error.details.error_code`). Matched by shape, not class: platform SDKs can bundle
 * another copy of the Anthropic SDK.
 */
export function claudeThreadErrorCode(error: unknown): string | undefined {
    if (!error || typeof error !== 'object') return undefined;
    const direct = (error as { errorCode?: unknown }).errorCode;
    if (typeof direct === 'string') return direct;
    const nested = (error as { error?: { error?: { details?: { error_code?: unknown } } } }).error?.error?.details
        ?.error_code;
    return typeof nested === 'string' ? nested : undefined;
}

// ============================================================================
// Hashing
// ============================================================================

/** JSON with sorted keys, without `cache_control`, and with `null` treated as absent. */
function canonicalJson(value: unknown): string {
    if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item ?? null)).join(',')}]`;
    if (value && typeof value === 'object') {
        const entries = Object.entries(value as Record<string, unknown>)
            .filter(([key, item]) => item !== undefined && item !== null && key !== 'cache_control')
            .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
        return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(',')}}`;
    }
    return JSON.stringify(value) ?? 'null';
}

function hashOf(value: unknown): string {
    return createHash('sha256').update(canonicalJson(value)).digest('hex');
}

// ============================================================================
// Requests
// ============================================================================

interface ThreadRequestOptions {
    headers?: Record<string, string>;
    signal?: AbortSignal;
    timeout?: number;
}

/** The request fields this module reads or changes; the rest of the payload passes through. */
export interface ClaudeThreadPayload {
    model: string;
    messages: MessageParam[];
    system?: unknown;
    tools?: unknown;
    tool_choice?: unknown;
    thinking?: unknown;
    context_management?: unknown;
    output_config?: unknown;
    cache_control?: unknown;
    thread?: BetaMessageThreadParam;
}

export type ClaudeThreadAttemptKind = 'continue' | 'create' | 'unthreaded';

export interface ClaudeThreadAttempt<P extends ClaudeThreadPayload> {
    kind: ClaudeThreadAttemptKind;
    payload: P;
    requestOptions: ThreadRequestOptions | undefined;
}

export interface ClaudeThreadSettings {
    options: ExecutionOptions;
    provider: string;
    /** Identifies the endpoint and key: the client's base URL. */
    baseURL: string | undefined;
    /** Top-level `cache_control` sent in place of message breakpoints when prompt caching is on. */
    cacheControl: unknown;
    logger?: Logger;
}

function withoutMessageCacheControl(messages: MessageParam[]): MessageParam[] {
    return messages.map((message) => {
        if (!Array.isArray(message.content)) return message;
        if (!message.content.some((block) => (block as { cache_control?: unknown }).cache_control !== undefined)) {
            return message;
        }
        const content = message.content.map((block) => {
            const { cache_control: _cacheControl, ...rest } = block as typeof block & { cache_control?: unknown };
            return rest;
        });
        return { ...message, content } as MessageParam;
    });
}

function withBeta(requestOptions: ThreadRequestOptions | undefined, beta: string): ThreadRequestOptions {
    const current = requestOptions?.headers?.['anthropic-beta'];
    const betas = current ? current.split(',').map((value) => value.trim()) : [];
    if (!betas.includes(beta)) betas.push(beta);
    // One comma-separated header, as getClaudePayload sends it.
    return { ...requestOptions, headers: { ...requestOptions?.headers, 'anthropic-beta': betas.join(',') } };
}

function sortedBetas(requestOptions: ThreadRequestOptions | undefined): string[] {
    const header = requestOptions?.headers?.['anthropic-beta'];
    return header
        ? header
              .split(',')
              .map((value) => value.trim())
              .filter(Boolean)
              .sort()
        : [];
}

/**
 * One Claude request with its thread: the attempts to send in order, the fallback between them, and the state to
 * store with the returned conversation. Created for every request; inactive when the request does not use threads.
 */
export class ClaudeThreadRequest<P extends ClaudeThreadPayload> {
    readonly active: boolean;
    private readonly previous: ClaudeThreadState | undefined;
    private readonly endpoint: string;
    private fingerprint = '';
    private disabled: boolean;
    private replayed = false;
    private sentAt = 0;
    private sent: ClaudeThreadAttempt<P> | undefined;

    constructor(
        private readonly payload: P,
        private readonly requestOptions: ThreadRequestOptions | undefined,
        private readonly settings: ClaudeThreadSettings,
    ) {
        const meta = readThreadMeta(settings.options.conversation);
        this.previous = meta.claude_thread;
        this.disabled = meta.claude_thread_disabled === true;
        this.active = usesClaudeThreads(settings.options, settings.provider);
        this.endpoint = `${settings.provider}|${settings.baseURL ?? ''}`;
        if (this.active) {
            const threaded = this.threadedBody({ type: 'create' });
            this.fingerprint = hashOf({
                model: payload.model,
                tool_choice: payload.tool_choice,
                thinking: payload.thinking,
                context_management: payload.context_management,
                output_config: payload.output_config,
                cache_control: threaded.cache_control,
                betas: sortedBetas(withBeta(requestOptions, MESSAGE_THREADS_BETA)),
                system: payload.system,
                tools: payload.tools,
            });
        }
    }

    /** The first attempt: a continue when the stored thread still matches, else a create; unthreaded when inactive. */
    first(): ClaudeThreadAttempt<P> {
        if (!this.active) return this.unthreaded();
        const stored = this.previous;
        if (stored && this.canContinue(stored)) {
            return {
                kind: 'continue',
                payload: this.threadedBody(
                    { type: 'continue', previous_message_id: stored.previous_message_id },
                    stored.stored_messages,
                ),
                requestOptions: withBeta(this.requestOptions, MESSAGE_THREADS_BETA),
            };
        }
        return this.create();
    }

    /**
     * The attempt that replaces a failed one, or undefined to rethrow the error. Only thread errors fall back; they
     * are returned before the model runs, so the failed attempt used no tokens.
     */
    fallback(failed: ClaudeThreadAttempt<P>, error: unknown): ClaudeThreadAttempt<P> | undefined {
        if (failed.kind === 'unthreaded') return undefined;
        const code = claudeThreadErrorCode(error);
        if (!code?.startsWith('thread_')) return undefined;
        const log = (next: ClaudeThreadAttemptKind) =>
            this.settings.logger?.debug(
                { provider: this.settings.provider, model: this.settings.options.model, error_code: code, next },
                '[Claude] Message thread request rejected; sending the full conversation',
            );
        if (code === 'thread_unsupported_request') {
            this.disabled = true;
            log('unthreaded');
            return this.unthreaded();
        }
        if (failed.kind === 'continue' && REPLAYABLE_CONTINUE_ERRORS.has(code)) {
            // A thread created by a replay that is already gone again is not being stored: stop threading.
            if (code === 'thread_not_found' && this.previous?.replayed) {
                this.disabled = true;
                log('unthreaded');
                return this.unthreaded();
            }
            this.replayed = true;
            log('create');
            return this.create();
        }
        return undefined;
    }

    /** Send the attempts in order until one is accepted. */
    async send<T>(open: (attempt: ClaudeThreadAttempt<P>) => Promise<T>): Promise<T> {
        let attempt: ClaudeThreadAttempt<P> | undefined = this.first();
        for (;;) {
            this.sentAt = claudeThreadClock.now();
            this.sent = attempt;
            try {
                return await open(attempt);
            } catch (error) {
                attempt = this.fallback(attempt, error);
                if (!attempt) throw error;
            }
        }
    }

    /**
     * Store the thread state with the finalized conversation. `reply` is the response as the next request will send
     * it (sanitized); the prefix hash covers the conversation as sent before any history rewrite, so a rewrite makes
     * the next request start a new thread.
     */
    record<C>(
        conversation: C,
        result: Pick<Message, 'id' | 'stop_reason' | 'content' | 'usage'>,
        reply: MessageParam[],
    ): C {
        const clear = (): C =>
            setConversationMeta(conversation, {
                ...getConversationMeta(conversation),
                claude_thread: undefined,
                claude_thread_disabled: this.disabled ? true : undefined,
            }) as C;
        const sent = this.sent;
        if (!this.active || this.disabled || !sent || sent.kind === 'unthreaded') return clear();
        if (!isStoredResponse(result) || reply.length !== 1) return clear();
        const messages = [...this.payload.messages, ...reply];
        const state: ClaudeThreadState = {
            previous_message_id: result.id,
            stored_messages: messages.length,
            prefix_hash: hashOf(messages),
            fingerprint: this.fingerprint,
            endpoint: this.endpoint,
            last_used_at: this.sentAt,
            ...(sent.kind === 'create' && this.replayed && { replayed: true }),
        };
        return setConversationMeta(conversation, {
            ...getConversationMeta(conversation),
            claude_thread: state,
            claude_thread_disabled: undefined,
        }) as C;
    }

    private canContinue(stored: ClaudeThreadState): boolean {
        const age = claudeThreadClock.now() - stored.last_used_at;
        return (
            stored.endpoint === this.endpoint &&
            stored.fingerprint === this.fingerprint &&
            age >= 0 &&
            age < THREAD_IDLE_LIMIT_MS &&
            stored.stored_messages > 0 &&
            stored.stored_messages < this.payload.messages.length &&
            hashOf(this.payload.messages.slice(0, stored.stored_messages)) === stored.prefix_hash
        );
    }

    private create(): ClaudeThreadAttempt<P> {
        return {
            kind: 'create',
            payload: this.threadedBody({ type: 'create' }),
            requestOptions: withBeta(this.requestOptions, MESSAGE_THREADS_BETA),
        };
    }

    private unthreaded(): ClaudeThreadAttempt<P> {
        return { kind: 'unthreaded', payload: this.payload, requestOptions: this.requestOptions };
    }

    /**
     * A threaded request body. Message breakpoints are replaced by one top-level `cache_control` (when caching is
     * on): a thread request takes at most 4 breakpoints, and the create's `system` and `tools` breakpoints count
     * again on every continue. A continue sends only the new messages and leaves out `system` and `tools`, which the
     * API rejects there even as null.
     */
    private threadedBody(thread: BetaMessageThreadParam, from = 0): P {
        const { system, tools, ...rest } = this.payload;
        const body = {
            ...(from > 0 ? rest : { ...rest, system, tools }),
            messages: withoutMessageCacheControl(this.payload.messages.slice(from)),
            thread,
            ...(this.settings.cacheControl !== undefined && { cache_control: this.settings.cacheControl }),
        };
        return body as P;
    }
}

/**
 * Whether the provider stored this response for a continue. A refusal must not be continued; a paused turn, a tool
 * call cut off at the token limit, and a reply written by a fallback model are not stored.
 */
function isStoredResponse(result: Pick<Message, 'stop_reason' | 'content' | 'usage'>): boolean {
    const stopReason = result.stop_reason ?? '';
    if (stopReason === 'refusal' || stopReason === 'pause_turn') return false;
    if (result.content.at(-1)?.type === 'tool_use' && CLAUDE_CUT_OFF_STOP_REASONS.has(stopReason)) return false;
    const iterations = (result.usage as { iterations?: Array<{ type?: string }> | null }).iterations;
    return !iterations?.some((iteration) => iteration?.type === 'fallback_message');
}
