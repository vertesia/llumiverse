import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import { describe, expect, it } from 'vitest';
import {
    type ClaudePrompt,
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
    model: 'claude-sonnet-4-5',
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
    return {
        messages: {
            stream: () => sdkStream(events, message),
        },
    } as never;
}

const prompt: ClaudePrompt = { messages: [{ role: 'user', content: 'question' }] };

describe('shared Claude reasoning replay codec', () => {
    it('preserves ordered thinking, redacted thinking, text, and tool use in blocking responses', async () => {
        const visible = await executeClaudeCompletion(clientFor(finalToolMessage), prompt, {
            model: 'claude-sonnet-4-5',
        });
        expect(visible.result).toEqual([
            { type: 'thoughts', value: 'plan' },
            { type: 'text', value: 'checking' },
        ]);
        expect(visible.conversation).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: finalToolMessage.content }]),
        });

        const hidden = await executeClaudeCompletion(clientFor(finalToolMessage), prompt, {
            model: 'claude-sonnet-4-5',
            model_options: { _option_id: 'text-fallback', include_thoughts: false },
        });
        expect(hidden.result).toEqual([{ type: 'text', value: 'checking' }]);
        expect(hidden.conversation).toEqual(visible.conversation);
    });

    it('uses the SDK final message to reconstruct fragmented streaming reasoning and signatures', async () => {
        const events = [
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'thinking_delta', thinking: 'pl' },
            },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'thinking_delta', thinking: 'an' },
            },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'signature_delta', signature: 'signed-thinking' },
            },
            {
                type: 'content_block_delta',
                index: 2,
                delta: { type: 'text_delta', text: 'checking', citations: null },
            },
        ] as RawMessageStreamEvent[];
        const stream = await streamClaudeCompletion(clientFor(finalToolMessage, events), prompt, {
            model: 'claude-sonnet-4-5',
        });
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.({ result: results });

        expect(results).toEqual([
            { type: 'thoughts', value: 'pl' },
            { type: 'thoughts', value: 'an' },
            { type: 'text', value: 'checking' },
        ]);
        expect(conversation).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: finalToolMessage.content }]),
        });
    });

    it('prunes only complete historical thinking blocks and is serialization-stable', () => {
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
                {
                    role: 'user',
                    content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'done' }],
                },
            ],
        };

        const once = pruneClaudeThinking(conversation);
        const twice = pruneClaudeThinking(once);
        expect(once.messages[0]).toEqual({
            role: 'assistant',
            content: [{ type: 'text', text: 'old answer' }],
        });
        expect(once.messages[2]).toEqual(conversation.messages[2]);
        expect(JSON.stringify(twice)).toBe(JSON.stringify(once));
    });
});
