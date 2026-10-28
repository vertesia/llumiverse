/**
 * A Claude `refusal` stop can arrive mid-stream, after a tool call has started, so the tool input may be cut off.
 * The refused turn must be discarded: its tool calls are never returned for execution, and the assistant message is
 * not kept in the conversation sent on the next turn.
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
// The input the API reports for a call the refusal cut off: whatever was streamed before the stop.
const cutOffThinkCall = { type: 'tool_use', id: 'toolu_think', name: 'think', input: { message_to_human: 'Planning' } };

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

function lastRole(conversation: unknown): string | undefined {
    return (conversation as ClaudePrompt).messages.at(-1)?.role;
}

describe('Claude refusal discards the refused turn', () => {
    it('returns no tool calls and reports refusal for a non-streaming refusal after tool calls', async () => {
        const refused = message('refusal', [preamble, searchCall, cutOffThinkCall]);
        const completion = await executeClaudeCompletion(clientFor(refused), prompt, { model: 'claude-test' });

        expect(completion.finish_reason).toBe('refusal');
        expect(completion.tool_use).toBeUndefined();
        expect(lastRole(completion.conversation)).toBe('user');
        expect(JSON.stringify(completion.conversation)).not.toContain('toolu_');
    });

    it('still returns tool calls and keeps the assistant turn for a normal tool_use stop', async () => {
        const normal = message('tool_use', [preamble, searchCall]);
        const completion = await executeClaudeCompletion(clientFor(normal), prompt, { model: 'claude-test' });

        expect(completion.finish_reason).toBe('tool_use');
        expect(completion.tool_use).toEqual([
            expect.objectContaining({
                id: 'toolu_search',
                tool_name: 'search',
                tool_input: { query: 'expenses 2025' },
            }),
        ]);
        expect(lastRole(completion.conversation)).toBe('assistant');
    });

    it('leaves the refused turn out of a streamed conversation', async () => {
        const refused = message('refusal', [preamble, cutOffThinkCall]);
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

        const conversation = await stream.finalizeConversation?.();
        expect(lastRole(conversation)).toBe('user');
        expect(JSON.stringify(conversation)).not.toContain('toolu_think');
    });
});
