import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import type { ConversationStreamEvent } from '@llumiverse/conversation';
import { type CompletionChunkObject, type ExecutionOptions, Providers } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { ClaudePrompt } from '../../shared/claude-messages.js';
import type { VertexAIDriver } from '../index.js';
import { ClaudeModelDefinition } from './claude.js';

function createAsyncStream(events: unknown[]): AsyncIterable<unknown> {
    return (async function* () {
        for (const event of events) {
            yield event;
        }
    })();
}

async function collectChunks(stream: AsyncIterable<CompletionChunkObject>): Promise<CompletionChunkObject[]> {
    const chunks: CompletionChunkObject[] = [];
    for await (const chunk of stream) {
        chunks.push(chunk);
    }
    return chunks;
}

async function collectCanonicalEvents(
    stream: AsyncIterable<ConversationStreamEvent>,
): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('ClaudeModelDefinition streaming spacing', () => {
    it('delegates Vertex Claude typed streaming with Anthropic native positions', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const message = {
            id: 'msg-vertex-typed',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-5',
            content: [{ type: 'text', text: 'Hello from Vertex.' }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 2, output_tokens: 3 },
        } as unknown as Message;
        const events = [
            {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '', citations: null },
            },
            {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'text_delta', text: 'Hello from Vertex.' },
            },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: { output_tokens: 3 },
            },
        ] as RawMessageStreamEvent[];
        const streamRequest = vi.fn(() => ({
            async *[Symbol.asyncIterator]() {
                for (const event of events) yield event;
            },
            finalMessage: async () => message,
            abort() {},
        }));
        const driver = {
            provider: Providers.vertexai,
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            getAnthropicClient: async () => ({ messages: { stream: streamRequest } }),
        } as unknown as VertexAIDriver;
        const stream = await modelDef.requestCanonicalTextCompletionEventStream(
            driver,
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Say hello.' }] }] },
            {
                model: 'publishers/anthropic/models/claude-sonnet-4-5',
                conversation_runtime: {
                    conversation_id: 'conversation:vertex-typed',
                    request_id: 'request:vertex-typed',
                    attempt_id: 'attempt:vertex-typed',
                    input_operation_id: 'input:vertex-typed',
                    response_operation_id: 'response:vertex-typed',
                    recorded_at: '2026-09-30T00:00:00.000Z',
                },
            },
            undefined,
            { stream_id: 'stream:vertex:claude:typed' },
        );
        const typedEvents = await collectCanonicalEvents(stream);

        expect(typedEvents).toContainEqual(
            expect.objectContaining({
                type: 'draft_text_delta',
                text: 'Hello from Vertex.',
                native_position: { protocol: 'anthropic.messages', path: ['content', 0] },
            }),
        );
        expect(typedEvents.at(-1)?.type).toBe('response_accepted');
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            provider: Providers.vertexai,
            protocol: 'anthropic.messages',
            requested_model: 'claude-sonnet-4-5',
        });
        expect(streamRequest).toHaveBeenCalledOnce();
    });

    it('does not leak deferred spacing when tool use follows thinking', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            getAnthropicClient: async () => ({
                messages: {
                    stream: async () =>
                        createAsyncStream([
                            {
                                type: 'content_block_delta',
                                delta: { type: 'thinking_delta', thinking: 'Thinking...' },
                            },
                            {
                                type: 'content_block_delta',
                                delta: { type: 'signature_delta' },
                            },
                            {
                                type: 'content_block_start',
                                content_block: { type: 'tool_use', id: 'tool-1', name: 'get_weather' },
                            },
                            {
                                type: 'content_block_delta',
                                delta: { type: 'input_json_delta', partial_json: '{"city":"Paris"}' },
                            },
                            {
                                type: 'content_block_stop',
                            },
                        ]),
                },
            }),
        } as unknown as VertexAIDriver;

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'Weather?' }] }],
        } as unknown as ClaudePrompt;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
                include_thoughts: true,
            },
        } as ExecutionOptions;

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        const chunks = await collectChunks(stream);

        const textOutput = chunks
            .flatMap((chunk) => chunk.result ?? [])
            .map((part) => part.value)
            .join('');
        const toolChunks = chunks.flatMap((chunk) => chunk.tool_use ?? []);

        expect(textOutput).toBe('Thinking...');
        expect(toolChunks).toHaveLength(2);
        expect(toolChunks[0]).toMatchObject({ id: 'tool-1', tool_name: 'get_weather', tool_input: '' });
        expect(toolChunks[1]).toMatchObject({ id: 'tool-1', tool_name: '', tool_input: '{"city":"Paris"}' });
    });

    it('keeps thinking and answer text as separate results without injected spacing', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            getAnthropicClient: async () => ({
                messages: {
                    stream: async () =>
                        createAsyncStream([
                            {
                                type: 'content_block_delta',
                                delta: { type: 'thinking_delta', thinking: 'Thinking...' },
                            },
                            {
                                type: 'content_block_delta',
                                delta: { type: 'signature_delta' },
                            },
                            {
                                type: 'content_block_delta',
                                delta: { type: 'text_delta', text: 'Answer' },
                            },
                        ]),
                },
            }),
        } as unknown as VertexAIDriver;

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'Question?' }] }],
        } as unknown as ClaudePrompt;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
                include_thoughts: true,
            },
        } as ExecutionOptions;

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        const chunks = await collectChunks(stream);

        const textParts = chunks.flatMap((chunk) => chunk.result ?? []).map((part) => part.value);
        expect(textParts).toEqual(['Thinking...', 'Answer']);
    });

    it('does not reintroduce deferred spacing when text arrives after a tool call', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            getAnthropicClient: async () => ({
                messages: {
                    stream: async () =>
                        createAsyncStream([
                            {
                                type: 'content_block_delta',
                                delta: { type: 'thinking_delta', thinking: 'Thinking...' },
                            },
                            {
                                type: 'content_block_delta',
                                delta: { type: 'signature_delta' },
                            },
                            {
                                type: 'content_block_start',
                                content_block: { type: 'tool_use', id: 'tool-1', name: 'get_weather' },
                            },
                            {
                                type: 'content_block_delta',
                                delta: { type: 'input_json_delta', partial_json: '{"city":"Paris"}' },
                            },
                            {
                                type: 'content_block_stop',
                            },
                            {
                                type: 'content_block_delta',
                                delta: { type: 'text_delta', text: 'Answer after tool' },
                            },
                        ]),
                },
            }),
        } as unknown as VertexAIDriver;

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'Weather?' }] }],
        } as unknown as ClaudePrompt;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
                include_thoughts: true,
            },
        } as ExecutionOptions;

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        const chunks = await collectChunks(stream);

        const textParts = chunks.flatMap((chunk) => chunk.result ?? []).map((part) => part.value);
        expect(textParts).toEqual(['Thinking...', 'Answer after tool']);
    });
});
