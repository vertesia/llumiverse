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
        const streamRequest = vi.fn((_params: unknown) => ({
            async *[Symbol.asyncIterator]() {
                for (const event of events) yield event;
            },
            finalMessage: async () => message,
            abort() {},
        }));
        const driver = {
            provider: Providers.vertexai,
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            getVertexRegion: () => 'us-central1',
            getAnthropicClient: async () => ({ messages: { stream: streamRequest } }),
        } as unknown as VertexAIDriver;
        const publish = vi.fn(async () => undefined);
        const requestedModel = 'locations/global/publishers/anthropic/models/claude-sonnet-4-5';
        const runtime: NonNullable<ExecutionOptions['conversation_runtime']> = {
            conversation_id: 'conversation:vertex-typed',
            request_id: 'request:vertex-typed',
            attempt_id: 'attempt:vertex-typed:first',
            input_operation_id: 'input:vertex-typed',
            response_operation_id: 'response:vertex-typed',
            recorded_at: '2026-09-30T00:00:00.000Z',
        };
        const options: ExecutionOptions = {
            model: requestedModel,
            on_canonical_request_prepared: publish,
            conversation_runtime: runtime,
        };
        const stream = await modelDef.requestCanonicalTextCompletionEventStream(
            driver,
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Say hello.' }] }] },
            options,
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
            requested_model: requestedModel,
        });
        const retainedGeneration = Object.values(stream.completion?.conversation.generations ?? {}).at(-1);
        if (retainedGeneration?.request_receipt === undefined) throw new Error('Expected retained request receipt');
        expect(retainedGeneration.request_receipt.target).toMatchObject({
            model: requestedModel,
            options: { region: 'global' },
        });
        expect(streamRequest.mock.calls[0]?.[0]).toMatchObject({ model: 'claude-sonnet-4-5' });
        expect(publish).toHaveBeenCalledOnce();

        if (stream.completion === undefined) throw new Error('Expected accepted Vertex Claude response');
        const recovered = await modelDef.requestCanonicalTextCompletionEventStream(
            driver,
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Say hello.' }] }] },
            {
                ...options,
                conversation: JSON.parse(JSON.stringify(stream.completion.conversation)),
                conversation_runtime: {
                    ...runtime,
                    attempt_id: 'attempt:vertex-typed:retry',
                },
            },
            undefined,
            { stream_id: 'stream:vertex:claude:typed:retry' },
        );
        expect(await collectCanonicalEvents(recovered)).toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        expect(streamRequest).toHaveBeenCalledOnce();

        await expect(
            modelDef.requestCanonicalTextCompletionEventStream(
                driver,
                { messages: [{ role: 'user', content: [{ type: 'text', text: 'Say hello.' }] }] },
                {
                    ...options,
                    model: 'locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5',
                    conversation: stream.completion.conversation,
                    conversation_runtime: {
                        ...runtime,
                        attempt_id: 'attempt:vertex-typed:changed-region',
                    },
                },
                undefined,
                { stream_id: 'stream:vertex:claude:typed:changed-region' },
            ),
        ).rejects.toThrow(/incompatible request identity|outside its compatibility scope/);
        expect(streamRequest).toHaveBeenCalledOnce();

        const routedModel = 'publishers/anthropic/models/claude-sonnet-4-5';
        const routedRuntime: NonNullable<ExecutionOptions['conversation_runtime']> = {
            conversation_id: 'conversation:vertex-typed-route',
            request_id: 'request:vertex-typed-route',
            attempt_id: 'attempt:vertex-typed-route:first',
            input_operation_id: 'input:vertex-typed-route',
            response_operation_id: 'response:vertex-typed-route',
            recorded_at: '2026-09-30T00:01:00.000Z',
        };
        const routedOptions: ExecutionOptions = {
            model: routedModel,
            conversation_runtime: routedRuntime,
        };
        const routedDriver = { ...driver, getVertexRegion: () => 'global' } as unknown as VertexAIDriver;
        const routed = await modelDef.requestCanonicalTextCompletionEventStream(
            routedDriver,
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Route this.' }] }] },
            routedOptions,
            undefined,
            { stream_id: 'stream:vertex:claude:routed' },
        );
        await collectCanonicalEvents(routed);
        if (routed.completion === undefined) throw new Error('Expected routed Vertex Claude response');
        const routedGeneration = Object.values(routed.completion.conversation.generations).at(-1);
        if (routedGeneration?.request_receipt === undefined) throw new Error('Expected routed request receipt');
        expect(routedGeneration.request_receipt.target.options).toEqual({
            region: 'global',
        });

        const changedRouteDriver = { ...driver, getVertexRegion: () => 'us-east5' } as unknown as VertexAIDriver;
        await expect(
            modelDef.requestCanonicalTextCompletionEventStream(
                changedRouteDriver,
                { messages: [{ role: 'user', content: [{ type: 'text', text: 'Route this.' }] }] },
                {
                    ...routedOptions,
                    conversation: routed.completion.conversation,
                    conversation_runtime: {
                        ...routedRuntime,
                        attempt_id: 'attempt:vertex-typed-route:changed',
                    },
                },
                undefined,
                { stream_id: 'stream:vertex:claude:routed:changed' },
            ),
        ).rejects.toThrow(/incompatible request routing/);
        expect(streamRequest).toHaveBeenCalledTimes(2);

        const controller = new AbortController();
        const addAbortListener = vi.spyOn(controller.signal, 'addEventListener');
        const removeAbortListener = vi.spyOn(controller.signal, 'removeEventListener');
        await expect(
            modelDef.requestCanonicalTextCompletionStream(
                changedRouteDriver,
                { messages: [{ role: 'user', content: [{ type: 'text', text: 'Route this.' }] }] },
                {
                    ...routedOptions,
                    conversation: routed.completion.conversation,
                    conversation_runtime: {
                        ...routedRuntime,
                        attempt_id: 'attempt:vertex-typed-route:changed-direct',
                    },
                },
                controller.signal,
            ),
        ).rejects.toThrow(/incompatible request routing/);
        expect(addAbortListener).not.toHaveBeenCalled();
        expect(removeAbortListener).not.toHaveBeenCalled();
        expect(streamRequest).toHaveBeenCalledTimes(2);
    });

    it('does not leak deferred spacing when tool use follows thinking', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            getVertexRegion: () => 'us-central1',
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
            getVertexRegion: () => 'us-central1',
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
            getVertexRegion: () => 'us-central1',
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
