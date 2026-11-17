import { APIError } from '@anthropic-ai/sdk/error';
import type { Message, MessageParam, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import { type ExecutionOptions, getConversationMeta, Providers, type ToolDefinition } from '@llumiverse/core';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { type ClaudePrompt, executeClaudeCompletion, streamClaudeCompletion } from './claude-messages.js';
import {
    type ClaudeThreadState,
    canUseClaudeThreads,
    claudeThreadClock,
    claudeThreadErrorCode,
    MESSAGE_THREADS_BETA,
} from './claude-threads.js';
import { INLINE_TOOLS_BETA, MID_CONVERSATION_TOOL_CHANGES_BETA } from './claude-tool-changes.js';

const MODEL = 'claude-sonnet-5-5';
const BASE_URL = 'https://api.anthropic.com';
const schema = { type: 'object', properties: { q: { type: 'string' } } } as const;
const tool = (name: string): ToolDefinition => ({ name, description: `${name} tool`, input_schema: schema });

interface SentRequest {
    payload: Record<string, unknown> & { messages: MessageParam[] };
    betas: string[];
}

type Reply = Message | Error;

function reply(id: string, content: unknown[], stop_reason = 'end_turn', usage: Record<string, unknown> = {}): Message {
    return {
        id,
        type: 'message',
        role: 'assistant',
        model: MODEL,
        content,
        stop_reason,
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1, ...usage },
    } as unknown as Message;
}

const text = (id: string, value = `answer ${id}`) =>
    reply(id, [{ type: 'text', text: value, citations: null }], 'end_turn');
const toolCall = (id: string, name = 'lookup') =>
    reply(
        id,
        [
            { type: 'thinking', thinking: 'Look it up.', signature: `sig-${id}` },
            { type: 'tool_use', id: `call-${id}`, name, input: { q: 'x' }, caller: null },
        ],
        'tool_use',
    );

/** The SDK error for a thread failure, built from the body the API returns. */
function threadError(code: string, status = code === 'thread_not_found' ? 404 : 400): Error {
    return APIError.generate(
        status,
        { type: 'error', error: { type: 'invalid_request_error', message: code, details: { error_code: code } } },
        undefined,
        new Headers(),
    );
}

/** Events for a reply: the stream content does not matter here, the final message is what gets stored. */
function eventsFor(message: Message): RawMessageStreamEvent[] {
    return [
        { type: 'message_start', message: { ...message, content: [] } },
        ...message.content.flatMap((block, index) => [
            { type: 'content_block_start', index, content_block: block },
            { type: 'content_block_stop', index },
        ]),
        { type: 'message_delta', delta: { stop_reason: message.stop_reason, stop_sequence: null }, usage: {} },
        { type: 'message_stop' },
    ] as unknown as RawMessageStreamEvent[];
}

/** A client that answers each request with the next scripted reply and records what it was sent. */
function mockClient(replies: Reply[], baseURL = BASE_URL) {
    const sent: SentRequest[] = [];
    const stream = vi.fn((payload: SentRequest['payload'], options?: { headers?: Record<string, string> }) => {
        sent.push({
            payload: JSON.parse(JSON.stringify(payload)),
            betas: options?.headers?.['anthropic-beta']?.split(',') ?? [],
        });
        const next = replies.shift();
        if (!next) throw new Error('unexpected request');
        if (next instanceof Error) {
            return {
                // biome-ignore lint/correctness/useYield: the request fails before its first event
                async *[Symbol.asyncIterator](): AsyncGenerator<RawMessageStreamEvent> {
                    throw next;
                },
                finalMessage: () => Promise.reject(next),
                abort() {},
            };
        }
        return {
            async *[Symbol.asyncIterator]() {
                yield* eventsFor(next);
            },
            finalMessage: async () => next,
            abort() {},
        };
    });
    return { client: { baseURL, messages: { stream } }, sent };
}

type Mode = 'execute' | 'stream';

/** One turn through the real execute or stream function; returns the stored (JSON round-tripped) conversation. */
async function runTurn(
    mode: Mode,
    client: ReturnType<typeof mockClient>['client'],
    prompt: ClaudePrompt,
    options: Partial<ExecutionOptions>,
    provider: string = Providers.anthropic,
): Promise<ClaudePrompt> {
    const fullOptions: ExecutionOptions = { model: MODEL, message_thread: true, ...options };
    if (mode === 'execute') {
        const completion = await executeClaudeCompletion(client as never, prompt, fullOptions, undefined, provider);
        return JSON.parse(JSON.stringify(completion.conversation));
    }
    const completion = await streamClaudeCompletion(client as never, prompt, fullOptions, undefined, provider);
    for await (const _chunk of completion) {
        // drain
    }
    return JSON.parse(JSON.stringify(await completion.finalizeConversation?.()));
}

const user = (value: string): ClaudePrompt => ({
    messages: [{ role: 'user', content: [{ type: 'text', text: value }] }],
});
const withSystem = (value: string): ClaudePrompt => ({
    ...user(value),
    system: [{ type: 'text', text: 'Be brief.' }],
});
const toolResult = (id: string): ClaudePrompt => ({
    messages: [{ role: 'user', content: [{ type: 'tool_result', tool_use_id: `call-${id}`, content: 'ok' }] }],
});

function threadState(conversation: unknown): ClaudeThreadState | undefined {
    return getConversationMeta(conversation).claude_thread as ClaudeThreadState | undefined;
}

function cacheBreakpoints(payload: unknown): number {
    return (JSON.stringify(payload).match(/"cache_control"/g) ?? []).length;
}

let now = 1_000_000;
beforeEach(() => {
    now = 1_000_000;
    vi.spyOn(claudeThreadClock, 'now').mockImplementation(() => now);
});
afterEach(() => {
    vi.restoreAllMocks();
});

describe.each<Mode>(['execute', 'stream'])('Claude message threads (%s)', (mode) => {
    it('creates a thread, then continues it with only the new messages', async () => {
        const { client, sent } = mockClient([text('msg-1'), toolCall('msg-2'), text('msg-3')]);
        const tools = [tool('lookup')];

        const c1 = await runTurn(mode, client, withSystem('Hello'), { tools });
        expect(sent[0].payload.thread).toEqual({ type: 'create' });
        expect(sent[0].payload.system).toEqual([{ type: 'text', text: 'Be brief.' }]);
        expect(sent[0].betas).toContain(MESSAGE_THREADS_BETA);
        expect(threadState(c1)).toMatchObject({ previous_message_id: 'msg-1', stored_messages: 2 });

        now += 60_000;
        const c2 = await runTurn(mode, client, user('Look something up'), { tools, conversation: c1 });
        expect(sent[1].payload.thread).toEqual({ type: 'continue', previous_message_id: 'msg-1' });
        expect(sent[1].payload.messages).toEqual(user('Look something up').messages);
        expect('system' in sent[1].payload).toBe(false);
        expect('tools' in sent[1].payload).toBe(false);
        expect(sent[1].payload.tool_choice).toEqual(sent[0].payload.tool_choice);
        expect(sent[1].betas).toEqual(sent[0].betas);
        expect(threadState(c2)).toMatchObject({ previous_message_id: 'msg-2', stored_messages: 4 });

        // The reply had thinking and a tool call: its stored form must hash as the next request sends it.
        const c3 = await runTurn(mode, client, toolResult('msg-2'), { tools, conversation: c2 });
        expect(sent[2].payload.thread).toEqual({ type: 'continue', previous_message_id: 'msg-2' });
        expect(sent[2].payload.messages).toEqual(toolResult('msg-2').messages);
        expect(threadState(c3)).toMatchObject({ previous_message_id: 'msg-3', stored_messages: 6 });
        // The returned conversation stays complete.
        expect(c3.messages).toHaveLength(6);
    });

    it('creates a new thread once the stored one is about to expire', async () => {
        const { client, sent } = mockClient([text('msg-1'), text('msg-2')]);
        const c1 = await runTurn(mode, client, user('Hello'), {});
        now += 5 * 60_000;
        const c2 = await runTurn(mode, client, user('Again'), { conversation: c1 });
        expect(sent[1].payload.thread).toEqual({ type: 'create' });
        expect(sent[1].payload.messages).toHaveLength(3);
        expect(threadState(c2)).toMatchObject({ previous_message_id: 'msg-2', stored_messages: 4 });
        expect(threadState(c2)?.replayed).toBeUndefined();
    });

    it('creates a new thread on another endpoint', async () => {
        const first = mockClient([text('msg-1')]);
        const c1 = await runTurn(mode, first.client, user('Hello'), {});
        const second = mockClient([text('msg-2')], 'https://proxy.example.com');
        await runTurn(mode, second.client, user('Again'), { conversation: c1 });
        expect(second.sent[0].payload.thread).toEqual({ type: 'create' });
    });

    it.each([
        'thread_not_found',
        'thread_already_continued',
        'thread_continue_in_flight',
        'thread_fingerprint_mismatch',
    ])('replays the full conversation as a create after %s', async (code) => {
        const { client, sent } = mockClient([text('msg-1'), threadError(code), text('msg-2')]);
        const c1 = await runTurn(mode, client, user('Hello'), {});
        const c2 = await runTurn(mode, client, user('Again'), { conversation: c1 });
        expect(sent[1].payload.thread).toMatchObject({ type: 'continue' });
        expect(sent[2].payload.thread).toEqual({ type: 'create' });
        expect(sent[2].payload.messages).toHaveLength(3);
        expect(threadState(c2)).toMatchObject({ previous_message_id: 'msg-2', replayed: true });
    });

    it('sends the conversation without a thread, and stops threading, when the request cannot be threaded', async () => {
        const { client, sent } = mockClient([threadError('thread_unsupported_request'), text('msg-1'), text('msg-2')]);
        const c1 = await runTurn(mode, client, user('Hello'), {});
        expect(sent[0].payload.thread).toEqual({ type: 'create' });
        expect(sent[1].payload.thread).toBeUndefined();
        expect(sent[1].betas).not.toContain(MESSAGE_THREADS_BETA);
        expect(threadState(c1)).toBeUndefined();
        expect(getConversationMeta(c1).claude_thread_disabled).toBe(true);

        const c2 = await runTurn(mode, client, user('Again'), { conversation: c1 });
        expect(sent).toHaveLength(3);
        expect(sent[2].payload.thread).toBeUndefined();
        expect(getConversationMeta(c2).claude_thread_disabled).toBe(true);
    });

    it('stops threading when a thread created by a replay is not found either', async () => {
        const { client, sent } = mockClient([
            text('msg-1'),
            threadError('thread_not_found'),
            text('msg-2'),
            threadError('thread_not_found'),
            text('msg-3'),
        ]);
        const c1 = await runTurn(mode, client, user('Hello'), {});
        const c2 = await runTurn(mode, client, user('Again'), { conversation: c1 });
        expect(threadState(c2)?.replayed).toBe(true);
        const c3 = await runTurn(mode, client, user('Once more'), { conversation: c2 });
        expect(sent[3].payload.thread).toMatchObject({ type: 'continue', previous_message_id: 'msg-2' });
        expect(sent[4].payload.thread).toBeUndefined();
        expect(sent[4].payload.messages).toHaveLength(5);
        expect(getConversationMeta(c3).claude_thread_disabled).toBe(true);
    });

    it('rethrows errors that are not thread errors', async () => {
        const overloaded = APIError.generate(
            529,
            { type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } },
            undefined,
            new Headers(),
        );
        const { client, sent } = mockClient([text('msg-1'), overloaded]);
        const c1 = await runTurn(mode, client, user('Hello'), {});
        await expect(runTurn(mode, client, user('Again'), { conversation: c1 })).rejects.toBe(overloaded);
        expect(sent).toHaveLength(2);
    });

    it.each([
        ['a refusal', reply('msg-2', [{ type: 'text', text: 'No.' }], 'refusal')],
        ['a paused turn', reply('msg-2', [{ type: 'text', text: 'Searching.' }], 'pause_turn')],
        [
            'a tool call cut off at the token limit',
            reply('msg-2', [{ type: 'tool_use', id: 'call-x', name: 'lookup', input: {} }], 'max_tokens'),
        ],
        [
            'a reply from a fallback model',
            reply('msg-2', [{ type: 'text', text: 'Hi.' }], 'end_turn', {
                iterations: [{ type: 'fallback_message', input_tokens: 1, output_tokens: 1 }],
            }),
        ],
    ])('does not continue after %s', async (_label, response) => {
        const { client, sent } = mockClient([text('msg-1'), response, text('msg-3')]);
        const tools = [tool('lookup')];
        const c1 = await runTurn(mode, client, user('Hello'), { tools });
        const c2 = await runTurn(mode, client, user('Again'), { tools, conversation: c1 });
        expect(threadState(c2)).toBeUndefined();
        await runTurn(mode, client, user('Once more'), { tools, conversation: c2 });
        expect(sent[2].payload.thread).toEqual({ type: 'create' });
    });

    it('starts a new thread when llumiverse rewrites the history it holds', async () => {
        const { client, sent } = mockClient([text('msg-1'), text('msg-2')]);
        const withImage: ClaudePrompt = {
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'What is this?' },
                        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'A'.repeat(4000) } },
                    ],
                },
            ],
        };
        const c1 = await runTurn(mode, client, withImage, { stripImagesAfterTurns: 0 });
        expect(JSON.stringify(c1.messages[0])).not.toContain('"image"');
        await runTurn(mode, client, user('Again'), { conversation: c1, stripImagesAfterTurns: 0 });
        expect(sent[1].payload.thread).toEqual({ type: 'create' });
    });

    it('replaces message breakpoints with one top-level cache_control', async () => {
        const { client, sent } = mockClient([text('msg-1'), text('msg-2')]);
        const options = {
            tools: [tool('lookup')],
            prompt_cache_key: 'agent-run-1',
            model_options: { _option_id: 'anthropic-claude', cache_ttl: '1h' },
        } as Partial<ExecutionOptions>;
        const c1 = await runTurn(mode, client, withSystem('Hello'), options);
        await runTurn(mode, client, user('Again'), { ...options, conversation: c1 });
        for (const { payload } of sent) {
            expect(payload.cache_control).toEqual({ type: 'ephemeral', ttl: '1h' });
            expect(cacheBreakpoints(payload.messages)).toBe(0);
            expect(cacheBreakpoints(payload)).toBeLessThanOrEqual(4);
        }
        expect(sent[1].payload.thread).toMatchObject({ type: 'continue' });
        expect(cacheBreakpoints(sent[1].payload)).toBe(1);
    });

    it('keeps the inline-tools beta from the first request, so a tool unlock continues the thread', async () => {
        const { client, sent } = mockClient([toolCall('msg-1', 'learn'), text('msg-2')]);
        const c1 = await runTurn(mode, client, user('Hello'), { tools: [tool('learn')] });
        expect(sent[0].betas).toContain(INLINE_TOOLS_BETA);
        expect(sent[0].betas).not.toContain(MID_CONVERSATION_TOOL_CHANGES_BETA);

        await runTurn(mode, client, toolResult('msg-1'), { tools: [tool('learn'), tool('read')], conversation: c1 });
        expect(sent[1].payload.thread).toEqual({ type: 'continue', previous_message_id: 'msg-1' });
        expect(sent[1].betas).toEqual(sent[0].betas);
        expect(sent[1].payload.messages.at(-1)).toMatchObject({
            role: 'system',
            content: [{ type: 'tool_addition', tool: { type: 'tool_definition' } }],
        });
    });

    it('keeps the by-reference tool-change beta from the first request on Foundry', async () => {
        const { client, sent } = mockClient([text('msg-1')]);
        await runTurn(mode, client, user('Hello'), { tools: [tool('learn')] }, Providers.azure_foundry);
        expect(sent[0].payload.thread).toEqual({ type: 'create' });
        expect(sent[0].betas).toContain(MID_CONVERSATION_TOOL_CHANGES_BETA);
    });

    it.each([Providers.vertexai, Providers.bedrock_mantle, Providers.bedrock])(
        'ignores the option on %s',
        async (provider) => {
            const { client, sent } = mockClient([text('msg-1')]);
            const c1 = await runTurn(mode, client, user('Hello'), { tools: [tool('learn')] }, provider);
            expect(sent[0].payload.thread).toBeUndefined();
            expect(sent[0].betas).not.toContain(MESSAGE_THREADS_BETA);
            expect(sent[0].betas).not.toContain(MID_CONVERSATION_TOOL_CHANGES_BETA);
            expect(threadState(c1)).toBeUndefined();
        },
    );

    it('is off with compaction', async () => {
        const { client, sent } = mockClient([text('msg-1')]);
        await runTurn(mode, client, user('Hello'), { compaction: {} });
        expect(sent[0].payload.thread).toBeUndefined();
        expect(sent[0].betas).not.toContain(MESSAGE_THREADS_BETA);
    });

    it('is off unless requested', async () => {
        const { client, sent } = mockClient([text('msg-1')]);
        const c1 = await runTurn(mode, client, user('Hello'), { message_thread: undefined });
        expect(sent[0].payload.thread).toBeUndefined();
        expect(threadState(c1)).toBeUndefined();
    });
});

describe('canUseClaudeThreads', () => {
    it.each([
        ['claude-sonnet-4-5', Providers.anthropic, true],
        ['claude-opus-5-5', Providers.azure_foundry, true],
        ['claude-sonnet-7-0', Providers.anthropic, true],
        ['claude-sonnet-4', Providers.anthropic, false],
        ['claude-3-7-sonnet-20250219', Providers.anthropic, false],
        ['claude-sonnet-5-5', Providers.vertexai, false],
        ['claude-sonnet-5-5', Providers.bedrock, false],
        ['claude-sonnet-5-5', Providers.bedrock_mantle, false],
    ])('%s on %s: %s', (model, provider, expected) => {
        expect(canUseClaudeThreads(model, provider)).toBe(expected);
    });
});

describe('claudeThreadErrorCode', () => {
    it('reads the code from any copy of the SDK error', () => {
        expect(claudeThreadErrorCode(threadError('thread_not_found'))).toBe('thread_not_found');
        expect(
            claudeThreadErrorCode({ error: { error: { details: { error_code: 'thread_already_continued' } } } }),
        ).toBe('thread_already_continued');
        expect(claudeThreadErrorCode(new Error('boom'))).toBeUndefined();
        expect(claudeThreadErrorCode(undefined)).toBeUndefined();
    });
});
