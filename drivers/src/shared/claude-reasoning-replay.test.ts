import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import { type ConversationDocument, parseConversationDocument } from '@llumiverse/conversation';
import { type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from '../anthropic/index.js';
import {
    type ClaudePrompt,
    executeClaudeCompletion,
    pruneClaudeThinking,
    streamClaudeCompletion,
} from './claude-messages.js';
import { exportLegacyClaudeMessagesConversation } from './claude-messages-conversation-adapter.js';
import { claudeFinishReason, logClaudeTruncation } from './claude-stop-reason.js';

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
const expectedFinalToolContent = [
    { type: 'thinking', thinking: 'plan', signature: 'signed-thinking' },
    { type: 'redacted_thinking', data: 'encrypted-redaction' },
    { type: 'text', text: 'checking' },
    { type: 'tool_use', id: 'call-1', name: 'lookup', input: { city: 'Paris' } },
];

function legacyConversation(value: unknown): ClaudePrompt {
    return exportLegacyClaudeMessagesConversation(value as ConversationDocument);
}

function latestGeneratedText(value: unknown): string | undefined {
    const document = parseConversationDocument(value);
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind !== 'agent' || turn.provenance.type !== 'generated') continue;
        return turn.blocks.find((block) => block.type === 'text')?.text;
    }
    return undefined;
}

function canonicalOptions(attempt: string, recordedAt: string, conversation?: unknown): ExecutionOptions {
    return {
        model: 'claude-sonnet-4-6',
        ...(conversation === undefined ? {} : { conversation }),
        conversation_runtime: {
            conversation_id: 'conversation:claude-structured',
            request_id: 'request:claude-structured',
            attempt_id: attempt,
            input_operation_id: 'input:claude-structured',
            response_operation_id: 'response:claude-structured',
            recorded_at: recordedAt,
            started_at: recordedAt,
        },
    };
}

describe('Claude native reasoning replay', () => {
    it('normalizes both Claude truncation stop reasons to length', () => {
        expect(claudeFinishReason('max_tokens')).toBe('length');
        expect(claudeFinishReason('model_context_window_exceeded')).toBe('length');
    });

    it('logs distinct provider-native truncation causes after normalization', () => {
        const logger = {
            debug: vi.fn(),
            info: vi.fn(),
            warn: vi.fn(),
            error: vi.fn(),
        };

        logClaudeTruncation(logger, 'max_tokens', { provider: 'anthropic', model: 'claude-sonnet-4-6' });
        logClaudeTruncation(logger, 'model_context_window_exceeded', {
            provider: 'vertexai',
            model: 'claude-sonnet-4-6',
        });

        expect(logger.warn).toHaveBeenNthCalledWith(
            1,
            expect.objectContaining({
                finish_reason: 'length',
                provider_finish_reason: 'max_tokens',
            }),
            '[Claude] Completion stopped at the output token limit',
        );
        expect(logger.warn).toHaveBeenNthCalledWith(
            2,
            expect.objectContaining({
                finish_reason: 'length',
                provider_finish_reason: 'model_context_window_exceeded',
            }),
            '[Claude] Completion exceeded the model context window',
        );
    });

    it('persists ordered native blocks for blocking responses without exposing thoughts by default', async () => {
        const completion = await executeClaudeCompletion(clientFor(finalToolMessage), prompt, {
            model: 'claude-sonnet-4-6',
        });

        expect(completion.result).toEqual([{ type: 'text', value: 'checking' }]);
        expect(legacyConversation(completion.conversation)).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: expectedFinalToolContent }]),
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
        expect(legacyConversation(completion.conversation)).toMatchObject({
            messages: expect.arrayContaining([
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'final plan', signature: 'final-signature' },
                        { type: 'text', text: 'final answer' },
                    ],
                },
            ]),
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
        expect(legacyConversation(conversation)).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: expectedFinalToolContent }]),
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
                messages: expect.arrayContaining([{ role: 'assistant', content: expectedFinalToolContent }]),
            }),
            undefined,
        );
    });

    it('validates structured output through the full Claude driver and retries without another provider call', async () => {
        const rawText = '{ "answer" : "Tokyo" }';
        const finalMessage = {
            id: 'msg-structured',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{ type: 'text', text: rawText }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 2, output_tokens: 3 },
        } as unknown as Message;
        const providerCall = vi.fn(() => sdkStream([], finalMessage));
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: providerCall } } as never;
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];

        const first = await driver.execute(segments, {
            ...canonicalOptions('attempt:first', '2026-09-11T00:00:00.000Z'),
            result_schema: resultSchema,
        });
        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo' } }]);
        expect(latestGeneratedText(first.conversation)).toBe(rawText);

        const retried = await driver.execute(segments, {
            ...canonicalOptions('attempt:retry', '2026-09-11T00:01:00.000Z', first.conversation),
            result_schema: resultSchema,
        });
        expect(retried.result).toEqual(first.result);
        expect(retried.conversation).toEqual(first.conversation);
        expect(providerCall).toHaveBeenCalledOnce();
    });

    it('validates streamed structured output through the full Claude driver while retaining source text', async () => {
        const rawText = '{"answer":"Tokyo"}';
        const finalMessage = {
            id: 'msg-stream-structured',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{ type: 'text', text: rawText }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 2, output_tokens: 3 },
        } as unknown as Message;
        const events = [
            {
                type: 'message_start',
                message: { ...finalMessage, content: [], stop_reason: null },
            },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: rawText } },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
        ] as RawMessageStreamEvent[];
        const driver = new AnthropicDriver({ apiKey: 'test' });
        driver.client = { messages: { stream: () => sdkStream(events, finalMessage) } } as never;
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Return the city.' }], {
            ...canonicalOptions('attempt:stream', '2026-09-11T00:00:00.000Z'),
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
        });

        for await (const _chunk of stream) {
            // Consume the public stream so core finalization and schema validation run.
        }
        expect(stream.completion?.result).toEqual([{ type: 'json', value: { answer: 'Tokyo' } }]);
        expect(latestGeneratedText(stream.completion?.conversation)).toBe(rawText);
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
