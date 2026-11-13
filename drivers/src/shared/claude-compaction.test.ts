import type { MessageParam } from '@anthropic-ai/sdk/resources/messages.js';
import type { CompletionChunkObject, ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { trimClaudeConversationToCompaction } from './claude-compaction.js';
import {
    anthropicUsageToTokenUsage,
    type ClaudePrompt,
    getClaudePayload,
    streamClaudeCompletion,
} from './claude-messages.js';

const MODEL = 'claude-opus-4-8';
const summary = { type: 'compaction', content: 'Summary of the conversation so far.', encrypted_content: 'opaque' };

function payloadFor(compaction: ExecutionOptions['compaction']) {
    return getClaudePayload({ model: MODEL, compaction }, { messages: [{ role: 'user', content: 'Hello' }] });
}

describe('Claude compaction request', () => {
    it('sends the compaction edit with its beta header', () => {
        const { payload, requestOptions } = payloadFor({ trigger_tokens: 60000, instructions: 'Keep the file names.' });

        expect(payload.context_management).toEqual({
            edits: [
                {
                    type: 'compact_20260112',
                    trigger: { type: 'input_tokens', value: 60000 },
                    instructions: 'Keep the file names.',
                },
            ],
        });
        expect(requestOptions?.headers?.['anthropic-beta']?.split(',')).toContain('compact-2026-01-12');
    });

    it('leaves the trigger and prompt to the API when they are omitted', () => {
        expect(payloadFor({}).payload.context_management).toEqual({ edits: [{ type: 'compact_20260112' }] });
    });

    it('preserves a trigger below the documented minimum for the API to validate', () => {
        const edit = payloadFor({ trigger_tokens: 0 }).payload.context_management?.edits?.[0];
        expect(edit).toMatchObject({ trigger: { type: 'input_tokens', value: 0 } });
    });

    it('sends neither the field nor the beta without compaction', () => {
        const { payload, requestOptions } = payloadFor(undefined);
        expect(payload).not.toHaveProperty('context_management');
        expect(requestOptions?.headers?.['anthropic-beta'] ?? '').not.toContain('compact-2026-01-12');
    });
});

describe('Claude on-demand compaction request', () => {
    it('asks for a summary with its beta header and prompt', () => {
        const { payload, requestOptions } = getClaudePayload(
            { model: MODEL, compact_conversation: { instructions: 'Keep the file names.' } },
            { messages: [{ role: 'user', content: 'Hello' }] },
        );

        expect(payload.compaction).toEqual({ type: 'summarize', instructions: 'Keep the file names.' });
        expect(requestOptions?.headers?.['anthropic-beta']?.split(',')).toContain('compact-2026-09-04');
    });

    it('leaves the prompt to the API when it is omitted', () => {
        const { payload } = getClaudePayload(
            { model: MODEL, compact_conversation: {} },
            { messages: [{ role: 'user', content: 'Hello' }] },
        );
        expect(payload.compaction).toEqual({ type: 'summarize' });
    });

    it('leaves out the threshold settings, which the API rejects alongside it, and says so', () => {
        const warn = vi.fn();
        const { payload, requestOptions } = getClaudePayload(
            { model: MODEL, compact_conversation: {}, compaction: { trigger_tokens: 60000 } },
            { messages: [{ role: 'user', content: 'Hello' }] },
            'anthropic',
            'stream',
            { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() },
        );

        expect(payload).not.toHaveProperty('context_management');
        expect(requestOptions?.headers?.['anthropic-beta']?.split(',')).not.toContain('compact-2026-01-12');
        expect(warn).toHaveBeenCalledWith(
            { model: MODEL, option_names: ['compaction'], reason: 'claude_on_demand_compaction' },
            expect.any(String),
        );
    });

    it('sends no compaction field on a normal request', () => {
        expect(payloadFor(undefined).payload).not.toHaveProperty('compaction');
    });
});

describe('trimClaudeConversationToCompaction', () => {
    const user = (text: string): MessageParam => ({ role: 'user', content: text });

    it('starts the conversation at the latest summary', () => {
        const conversation = {
            messages: [
                user('first'),
                { role: 'assistant', content: [{ type: 'text', text: 'old' }] },
                user('second'),
                {
                    role: 'assistant',
                    content: [{ type: 'text', text: 'before' }, summary, { type: 'text', text: 'after' }],
                },
                user('third'),
            ] as MessageParam[],
            _llumiverse_meta: { turnNumber: 3 },
        };

        expect(trimClaudeConversationToCompaction(conversation)).toEqual({
            messages: [{ role: 'assistant', content: [summary, { type: 'text', text: 'after' }] }, user('third')],
            _llumiverse_meta: { turnNumber: 3 },
        });
    });

    it('keeps only the newest of two summaries', () => {
        const newer = { ...summary, content: 'Newer summary.' };
        const conversation = {
            messages: [
                { role: 'assistant', content: [summary, { type: 'text', text: 'a' }] },
                user('next'),
                { role: 'assistant', content: [newer, { type: 'text', text: 'b' }] },
            ] as MessageParam[],
        };

        expect(trimClaudeConversationToCompaction(conversation).messages).toEqual([
            { role: 'assistant', content: [newer, { type: 'text', text: 'b' }] },
        ]);
    });

    it('ignores a failed compaction, which has no summary', () => {
        const conversation = {
            messages: [
                user('first'),
                {
                    role: 'assistant',
                    content: [
                        { ...summary, content: null },
                        { type: 'text', text: 'reply' },
                    ],
                },
            ] as MessageParam[],
        };

        expect(trimClaudeConversationToCompaction(conversation)).toBe(conversation);
    });

    it('returns the conversation unchanged when the summary already comes first', () => {
        const conversation = { messages: [{ role: 'assistant', content: [summary] }, user('next')] as MessageParam[] };
        expect(trimClaudeConversationToCompaction(conversation)).toBe(conversation);
    });
});

describe('Claude compaction usage', () => {
    it('adds the compaction iterations, which the top-level usage leaves out', () => {
        expect(
            anthropicUsageToTokenUsage({
                input_tokens: 23000,
                output_tokens: 1000,
                cache_read_input_tokens: 100,
                cache_creation_input_tokens: 0,
                iterations: [
                    {
                        type: 'compaction',
                        input_tokens: 180000,
                        output_tokens: 3500,
                        cache_read_input_tokens: 50,
                        cache_creation_input_tokens: 0,
                    },
                    { type: 'message', input_tokens: 23000, output_tokens: 1000 },
                ],
            }),
        ).toEqual({
            prompt_new: 203000,
            prompt: 203150,
            result: 4500,
            total: 207650,
            prompt_cached: 150,
            prompt_cache_write: 0,
            prompt_cache_write_1h: undefined,
        });
    });
});

describe('streamed Claude compaction', () => {
    const SUMMARY = 'The user is building a scraper.';
    const usage = { input_tokens: 2000, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 };
    const iterations = [
        { type: 'compaction', input_tokens: 160000, output_tokens: 900, cache_read_input_tokens: 0 },
        { type: 'message', input_tokens: 2000, output_tokens: 40 },
    ];
    const events = [
        { type: 'message_start', message: { id: 'msg-1', usage } },
        { type: 'content_block_start', index: 0, content_block: { type: 'compaction', content: null } },
        {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'compaction_delta', content: SUMMARY, encrypted_content: 'opaque' },
        },
        { type: 'content_block_stop', index: 0 },
        { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Here is the next step.' } },
        { type: 'content_block_stop', index: 1 },
        {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn' },
            usage: { output_tokens: 40, iterations },
        },
    ];
    // What the stable SDK accumulator returns: it ignores `compaction_delta`, so the block has no summary.
    const finalMessage = {
        id: 'msg-1',
        type: 'message',
        role: 'assistant',
        model: MODEL,
        content: [
            { type: 'compaction', content: null },
            { type: 'text', text: 'Here is the next step.' },
        ],
        stop_reason: 'end_turn',
        usage: { ...usage, output_tokens: 40, iterations },
    };
    const client = {
        messages: {
            stream: () => ({
                async *[Symbol.asyncIterator]() {
                    yield* events;
                },
                abort: () => undefined,
                finalMessage: () => Promise.resolve(finalMessage),
            }),
        },
    };

    it('stores the streamed summary first in the conversation and counts its tokens', async () => {
        const history: ClaudePrompt = {
            messages: [
                { role: 'user', content: 'Help me build a scraper.' },
                { role: 'assistant', content: [{ type: 'text', text: 'Sure.' }] },
            ],
        };
        const completion = await streamClaudeCompletion(
            client as never,
            { messages: [{ role: 'user', content: 'Add rate limiting.' }] },
            { model: MODEL, conversation: history, compaction: {} },
        );
        const chunks: CompletionChunkObject[] = [];
        for await (const chunk of completion) chunks.push(chunk as CompletionChunkObject);

        const text = chunks.flatMap((chunk) => chunk.result ?? []).map((result) => result.value);
        expect(text.join('')).toBe('Here is the next step.');
        expect(chunks.at(-1)?.token_usage).toMatchObject({ prompt: 162000, result: 940 });

        const conversation = (await completion.finalizeConversation?.()) as ClaudePrompt;
        expect(conversation.messages).toEqual([
            {
                role: 'assistant',
                content: [
                    { type: 'compaction', content: SUMMARY, encrypted_content: 'opaque' },
                    { type: 'text', text: 'Here is the next step.' },
                ],
            },
        ]);
    });
});

describe('streamed Claude on-demand compaction', () => {
    const SUMMARY = 'The user is building a scraper with rate limiting.';
    const block = { type: 'compaction', content: SUMMARY, signature: 'signed' };
    const usage = { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 };
    const iterations = [{ type: 'compaction', input_tokens: 90000, output_tokens: 700 }];
    // The block arrives whole in `content_block_start`, with no delta.
    const events = [
        { type: 'message_start', message: { id: 'msg-2', usage } },
        { type: 'content_block_start', index: 0, content_block: block },
        { type: 'content_block_stop', index: 0 },
        { type: 'message_delta', delta: { stop_reason: 'compaction' }, usage: { output_tokens: 0, iterations } },
    ];
    const finalMessage = {
        id: 'msg-2',
        type: 'message',
        role: 'assistant',
        model: MODEL,
        content: [block],
        stop_reason: 'compaction',
        usage: { ...usage, iterations },
    };
    const stream = vi.fn((_payload: unknown, _options: unknown) => ({
        async *[Symbol.asyncIterator]() {
            yield* events;
        },
        abort: () => undefined,
        finalMessage: () => Promise.resolve(finalMessage),
    }));

    it('reports the summary, counts its tokens and continues from the block alone', async () => {
        const history: ClaudePrompt = {
            messages: [
                { role: 'user', content: 'Help me build a scraper.' },
                {
                    role: 'assistant',
                    content: [{ type: 'tool_use', id: 'call-1', name: 'fetch_page', input: { url: 'https://x' } }],
                },
            ],
        };
        const completion = await streamClaudeCompletion(
            { messages: { stream } } as never,
            { messages: [] },
            { model: MODEL, conversation: history, compact_conversation: {} },
        );
        const chunks: CompletionChunkObject[] = [];
        for await (const chunk of completion) chunks.push(chunk as CompletionChunkObject);

        const payload = stream.mock.calls[0][0] as { compaction?: unknown; messages: MessageParam[] };
        expect(payload.compaction).toEqual({ type: 'summarize' });
        // The pending tool call is closed by a user turn, which the API requires before it summarizes.
        expect(payload.messages.at(-1)?.role).toBe('user');

        const text = chunks.flatMap((chunk) => chunk.result ?? []).map((result) => result.value);
        expect(text.join('')).toBe(SUMMARY);
        expect(chunks.at(-1)).toMatchObject({
            finish_reason: 'compaction',
            token_usage: { prompt: 90000, result: 700 },
        });

        const conversation = (await completion.finalizeConversation?.()) as ClaudePrompt;
        expect(conversation.messages).toEqual([{ role: 'assistant', content: [block] }]);
    });
});
