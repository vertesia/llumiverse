import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import { describe, expect, it, vi } from 'vitest';
import {
    type ClaudePrompt,
    claudeFinishReason,
    executeClaudeCompletion,
    pruneClaudeThinking,
    streamClaudeCompletion,
} from './claude-messages.js';

function sdkStream(events: RawMessageStreamEvent[], finalMessage: Message) {
    return {
        async *[Symbol.asyncIterator]() {
            for (const event of events) yield event;
        },
        async finalMessage() {
            return finalMessage;
        },
    };
}

const finalToolMessage = {
    id: 'msg-1',
    type: 'message',
    role: 'assistant',
    model: 'claude-sonnet-4-6',
    content: [
        { type: 'thinking', thinking: 'plan', signature: 'signed-thinking' },
        { type: 'redacted_thinking', data: 'encrypted-redaction' },
        { type: 'text', text: 'checking', citations: null },
        { type: 'tool_use', id: 'call-1', name: 'lookup', input: { city: 'Paris' } },
    ],
    stop_reason: 'tool_use',
    stop_sequence: null,
    usage: { input_tokens: 2, output_tokens: 3 },
} as unknown as Message;

function clientFor(message: Message, events: RawMessageStreamEvent[] = []) {
    return { messages: { stream: () => sdkStream(events, message) } } as never;
}

const prompt: ClaudePrompt = { messages: [{ role: 'user', content: 'question' }] };

describe('Claude native reasoning replay', () => {
    it('normalizes both Claude truncation stop reasons to length', () => {
        expect(claudeFinishReason('max_tokens')).toBe('length');
        expect(claudeFinishReason('model_context_window_exceeded')).toBe('length');
    });

    it('persists ordered native blocks for blocking responses without exposing thoughts by default', async () => {
        const completion = await executeClaudeCompletion(clientFor(finalToolMessage), prompt, {
            model: 'claude-sonnet-4-6',
        });

        expect(completion.result).toEqual([{ type: 'text', value: 'checking' }]);
        expect(completion.conversation).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: finalToolMessage.content }]),
        });
    });

    it('persists reasoning on the latest completed blocking turn', async () => {
        const finalMessage = {
            ...finalToolMessage,
            content: [
                { type: 'thinking', thinking: 'final plan', signature: 'final-signature' },
                { type: 'text', text: 'final answer', citations: null },
            ],
            stop_reason: 'end_turn',
        } as unknown as Message;
        const completion = await executeClaudeCompletion(clientFor(finalMessage), prompt, {
            model: 'claude-sonnet-4-6',
        });

        expect(completion.result).toEqual([{ type: 'text', value: 'final answer' }]);
        expect(completion.conversation).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: finalMessage.content }]),
        });
    });

    it('uses the SDK final message for streaming persistence and replays it on the next tool turn', async () => {
        const events = [
            { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'plan' } },
            { type: 'content_block_delta', index: 2, delta: { type: 'text_delta', text: 'checking', citations: null } },
        ] as RawMessageStreamEvent[];
        const stream = await streamClaudeCompletion(clientFor(finalToolMessage, events), prompt, {
            model: 'claude-sonnet-4-6',
        });
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.();

        expect(results).toEqual([{ type: 'text', value: 'checking' }]);
        expect(conversation).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: finalToolMessage.content }]),
        });

        const nextMessage = {
            ...finalToolMessage,
            id: 'msg-2',
            content: [{ type: 'text', text: 'done', citations: null }],
            stop_reason: 'end_turn',
        } as unknown as Message;
        const nextStream = vi.fn(() => sdkStream([], nextMessage));
        await executeClaudeCompletion(
            { messages: { stream: nextStream } } as never,
            {
                messages: [
                    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'sunny' }] },
                ],
            },
            {
                model: 'claude-sonnet-4-6',
                conversation: JSON.parse(JSON.stringify(conversation)),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
        );

        expect(nextStream).toHaveBeenCalledWith(
            expect.objectContaining({
                messages: expect.arrayContaining([{ role: 'assistant', content: finalToolMessage.content }]),
            }),
            undefined,
        );
    });

    it('prunes only completed historical reasoning and keeps an active tool chain intact', () => {
        const conversation: ClaudePrompt = {
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'old', signature: 'old-signature' },
                        { type: 'text', text: 'old answer' },
                    ],
                },
                { role: 'user', content: 'next' },
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'active', signature: 'active-signature' },
                        { type: 'tool_use', id: 'call-1', name: 'lookup', input: {} },
                    ],
                },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'done' }] },
            ],
        };

        const pruned = pruneClaudeThinking(conversation);
        expect(pruned.messages[0]).toEqual({ role: 'assistant', content: [{ type: 'text', text: 'old answer' }] });
        expect(pruned.messages[2]).toEqual(conversation.messages[2]);
        expect(pruneClaudeThinking(pruned)).toEqual(pruned);
    });
});
