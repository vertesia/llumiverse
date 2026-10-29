/**
 * A Claude `refusal` stop can arrive after a tool call has started. The completion must report the refusal rather
 * than `tool_use`, so the caller can tell the turn was refused, while the refused turn and its tool calls stay on
 * record in the result and the conversation.
 */

import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import { describe, expect, it } from 'vitest';
import { type ClaudePrompt, executeClaudeCompletion, streamClaudeCompletion } from './claude-messages.js';

const prompt: ClaudePrompt = { messages: [{ role: 'user', content: 'What was the total spend in 2025?' }] };
const stopDetails = { type: 'refusal', category: 'reasoning_extraction', explanation: 'Declined.' };

function message(stopReason: string, content: unknown[]): Message {
    return {
        id: 'msg-1',
        type: 'message',
        role: 'assistant',
        model: 'claude-test',
        content,
        stop_reason: stopReason,
        stop_details: stopReason === 'refusal' ? stopDetails : null,
        stop_sequence: null,
        usage: { input_tokens: 2, output_tokens: 3 },
    } as unknown as Message;
}

const preamble = { type: 'text', text: 'Searching the 2025 expenses.', citations: null };
const searchCall = { type: 'tool_use', id: 'toolu_search', name: 'search', input: { query: 'expenses 2025' } };
const thinkCall = { type: 'tool_use', id: 'toolu_think', name: 'think', input: { message_to_human: 'Planning' } };

function clientFor(final: Message, events: RawMessageStreamEvent[] = []) {
    const stream = () => ({
        async *[Symbol.asyncIterator]() {
            for (const event of events) yield event;
        },
        async finalMessage() {
            return final;
        },
    });
    return { messages: { stream } } as never;
}

function lastMessage(conversation: unknown) {
    return (conversation as ClaudePrompt).messages.at(-1);
}

describe('Claude refusal after tool calls', () => {
    it('reports refusal, not tool_use, and keeps the calls and the turn on record', async () => {
        const refused = message('refusal', [preamble, searchCall, thinkCall]);
        const completion = await executeClaudeCompletion(clientFor(refused), prompt, { model: 'claude-test' });

        expect(completion.finish_reason).toBe('refusal');
        expect(completion.tool_use?.map((tool) => tool.id)).toEqual(['toolu_search', 'toolu_think']);
        expect(lastMessage(completion.conversation)).toEqual({ role: 'assistant', content: refused.content });
    });

    it('reports length, not tool_use, when the token limit stops the response after tool calls', async () => {
        const truncated = message('max_tokens', [preamble, searchCall]);
        const completion = await executeClaudeCompletion(clientFor(truncated), prompt, { model: 'claude-test' });

        expect(completion.finish_reason).toBe('length');
        expect(completion.tool_use?.map((tool) => tool.id)).toEqual(['toolu_search']);
    });

    it('reports tool_use for a normal tool_use stop', async () => {
        const normal = message('tool_use', [preamble, searchCall]);
        const completion = await executeClaudeCompletion(clientFor(normal), prompt, { model: 'claude-test' });

        expect(completion.finish_reason).toBe('tool_use');
        expect(completion.tool_use?.map((tool) => tool.id)).toEqual(['toolu_search']);
    });

    it('keeps the refused turn in a streamed conversation', async () => {
        const refused = message('refusal', [preamble, thinkCall]);
        const events = [
            { type: 'message_start', message: { ...refused, content: [], stop_reason: null, stop_details: null } },
            {
                type: 'message_delta',
                delta: { stop_reason: 'refusal', stop_sequence: null, stop_details: stopDetails },
                usage: { output_tokens: 3 },
            },
            { type: 'message_stop' },
        ] as unknown as RawMessageStreamEvent[];
        const stream = await streamClaudeCompletion(clientFor(refused, events), prompt, { model: 'claude-test' });
        for await (const _chunk of stream) {
            // drain
        }

        expect(lastMessage(await stream.finalizeConversation?.())).toEqual({
            role: 'assistant',
            content: refused.content,
        });
    });
});
