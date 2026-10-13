import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import { describe, expect, it, vi } from 'vitest';
import { type ClaudePrompt, executeClaudeCompletion, streamClaudeCompletion } from './claude-messages.js';

const dropped = [{ type: 'thinking_dropped', path: 'messages.1.content.0', reason: 'prefix_binding_mismatch' }];

function messageWith(input_transformations: unknown[]): Message {
    return {
        id: 'msg-1',
        type: 'message',
        role: 'assistant',
        model: 'claude-sonnet-5-5',
        content: [{ type: 'text', text: 'done', citations: null }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: 2, output_tokens: 3 },
        input_transformations,
    } as unknown as Message;
}

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

function testLogger() {
    return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

const prompt: ClaudePrompt = { messages: [{ role: 'user', content: 'question' }] };
const droppedLog = expect.objectContaining({
    provider: 'anthropic',
    model: 'claude-sonnet-5-5',
    transformation_count: 1,
    transformations: { 'thinking_dropped:prefix_binding_mismatch': 1 },
    first_path: 'messages.1.content.0',
});

describe('Claude thinking block binding', () => {
    it('sends the binding beta header and logs blocks the API dropped', async () => {
        const stream = vi.fn(() => sdkStream([], messageWith(dropped)));
        const logger = testLogger();
        await executeClaudeCompletion(
            { messages: { stream } } as never,
            prompt,
            { model: 'claude-sonnet-5-5' },
            logger,
        );

        expect(stream).toHaveBeenCalledWith(
            expect.objectContaining({
                thinking: expect.objectContaining({ block_binding: { prefix_mismatch_behavior: 'drop_block' } }),
            }),
            { headers: { 'anthropic-beta': 'thinking-binding-controls-2026-08-01' } },
        );
        expect(logger.warn).toHaveBeenCalledWith(
            droppedLog,
            '[Claude] Replayed thinking blocks no longer match the request',
        );
    });

    it('does not log when no block was transformed', async () => {
        const logger = testLogger();
        const client = { messages: { stream: () => sdkStream([], messageWith([])) } } as never;
        await executeClaudeCompletion(client, prompt, { model: 'claude-sonnet-5-5' }, logger);
        expect(logger.warn).not.toHaveBeenCalled();
    });

    it('logs drops reported on message_start when streaming', async () => {
        const logger = testLogger();
        const message = messageWith(dropped);
        const events = [{ type: 'message_start', message }] as RawMessageStreamEvent[];
        const client = { messages: { stream: () => sdkStream(events, message) } } as never;
        const stream = await streamClaudeCompletion(client, prompt, { model: 'claude-sonnet-5-5' }, logger);
        for await (const _chunk of stream) {
            // drain
        }
        expect(logger.warn).toHaveBeenCalledTimes(1);
        expect(logger.warn).toHaveBeenCalledWith(
            droppedLog,
            '[Claude] Replayed thinking blocks no longer match the request',
        );
    });
});
