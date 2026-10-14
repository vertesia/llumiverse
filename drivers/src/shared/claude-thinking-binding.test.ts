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

function startEvents(message: Message): RawMessageStreamEvent[] {
    return [{ type: 'message_start', message }];
}

/** The final message_delta carries the array only after a server-side model fallback, replacing message_start's. */
function closingEvents(input_transformations?: unknown[]): RawMessageStreamEvent[] {
    return [
        {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: 3 },
            ...(input_transformations && { input_transformations }),
        } as unknown as RawMessageStreamEvent,
        { type: 'message_stop' },
    ];
}

function fallbackEvents(input_transformations: unknown[]): RawMessageStreamEvent[] {
    return [...startEvents(messageWith([])), ...closingEvents(input_transformations)];
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
        const message = messageWith(dropped);
        const stream = vi.fn(() => sdkStream(startEvents(message), message));
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

    it('ignores input transformation types it does not recognise', async () => {
        const logger = testLogger();
        const message = messageWith([{ type: 'future_transformation', path: 'messages.0' }]);
        const client = { messages: { stream: () => sdkStream(startEvents(message), message) } } as never;
        await executeClaudeCompletion(client, prompt, { model: 'claude-sonnet-5-5' }, logger);
        expect(logger.warn).not.toHaveBeenCalled();
    });

    it('does not log when no block was transformed', async () => {
        const logger = testLogger();
        const message = messageWith([]);
        const client = { messages: { stream: () => sdkStream(startEvents(message), message) } } as never;
        await executeClaudeCompletion(client, prompt, { model: 'claude-sonnet-5-5' }, logger);
        expect(logger.warn).not.toHaveBeenCalled();
    });

    it('logs drops reported after a server-side fallback', async () => {
        const logger = testLogger();
        // The stable SDK accumulator drops the delta's array, so the final message reports none.
        const client = { messages: { stream: () => sdkStream(fallbackEvents(dropped), messageWith([])) } } as never;
        await executeClaudeCompletion(client, prompt, { model: 'claude-sonnet-5-5' }, logger);
        expect(logger.warn).toHaveBeenCalledTimes(1);
        expect(logger.warn).toHaveBeenCalledWith(
            droppedLog,
            '[Claude] Replayed thinking blocks no longer match the request',
        );
    });

    it.each([
        ['message_start', (message: Message) => [...startEvents(message), ...closingEvents()]],
        ['a fallback message_delta', () => fallbackEvents(dropped)],
    ])('logs drops reported on %s once when streaming', async (_source, events) => {
        const logger = testLogger();
        const message = messageWith(dropped);
        const client = { messages: { stream: () => sdkStream(events(message), message) } } as never;
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
