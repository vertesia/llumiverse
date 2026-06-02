import type { CompletionChunkObject, ExecutionOptions } from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import { describe, expect, it, vi } from 'vitest';
import type { ClaudePrompt } from '../../shared/claude-messages.js';
import type { VertexAIDriver } from '../index.js';
import { ClaudeModelDefinition } from './claude.js';

function createSseStream(events: unknown[]): ReadableStream<ServerSentEvent> {
    return new ReadableStream<ServerSentEvent>({
        start(controller) {
            for (const event of events) {
                controller.enqueue({ type: 'event', data: JSON.stringify(event) });
            }
            controller.close();
        },
    });
}

async function collectChunks(stream: AsyncIterable<CompletionChunkObject>): Promise<CompletionChunkObject[]> {
    const chunks: CompletionChunkObject[] = [];
    for await (const chunk of stream) {
        chunks.push(chunk);
    }
    return chunks;
}

describe('ClaudeModelDefinition streaming spacing', () => {
    it('does not leak deferred spacing when tool use follows thinking', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const streamVertexModel = vi.fn(async () =>
            createSseStream([
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
        );
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            options: { region: 'us-central1' },
            streamVertexModel,
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
        expect(streamVertexModel).toHaveBeenCalledWith(
            'publishers/anthropic/models/claude-sonnet-4-5',
            'streamRawPredict',
            expect.objectContaining({
                anthropic_version: 'vertex-2023-10-16',
                stream: true,
            }),
            expect.objectContaining({
                region: 'us-east5',
            }),
        );
    });

    it('flushes deferred spacing into the first text delta after thinking', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            options: { region: 'us-central1' },
            streamVertexModel: async () =>
                createSseStream([
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
        expect(textParts).toEqual(['Thinking...', '\n\nAnswer']);
    });

    it('does not reintroduce deferred spacing when text arrives after a tool call', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {} },
            options: { region: 'us-central1' },
            streamVertexModel: async () =>
                createSseStream([
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

    it('requestTextCompletion uses streamRawPredict and aggregates the final completion', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const streamVertexModel = vi.fn(async () =>
            createSseStream([
                {
                    type: 'message_start',
                    message: {
                        usage: { input_tokens: 7, output_tokens: 0 },
                    },
                },
                {
                    type: 'content_block_delta',
                    delta: { type: 'text_delta', text: 'Hello' },
                },
                {
                    type: 'content_block_delta',
                    delta: { type: 'text_delta', text: ' there' },
                },
                {
                    type: 'message_delta',
                    delta: { stop_reason: 'end_turn' },
                    usage: { output_tokens: 3 },
                },
            ]),
        );
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} },
            options: { region: 'us-central1' },
            streamVertexModel,
        } as unknown as VertexAIDriver;

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'Question?' }] }],
        } as unknown as ClaudePrompt;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
            },
        } as ExecutionOptions;

        const completion = await modelDef.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: 'Hello there' }]);
        expect(completion.finish_reason).toBe('stop');
        expect(completion.token_usage).toEqual({ prompt: 7, prompt_new: 7, result: 3, total: 10 });
        expect(completion.conversation).toBeDefined();
        expect(streamVertexModel).toHaveBeenCalledWith(
            'publishers/anthropic/models/claude-sonnet-4-5',
            'streamRawPredict',
            expect.objectContaining({
                anthropic_version: 'vertex-2023-10-16',
                stream: true,
            }),
            expect.objectContaining({
                region: 'us-east5',
            }),
        );
    });

    it('keeps streamed assistant text as content blocks for prompt cache markers', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const streamVertexModel = vi.fn(async () =>
            createSseStream([
                {
                    type: 'message_start',
                    message: {
                        usage: { input_tokens: 7, output_tokens: 0 },
                    },
                },
                {
                    type: 'content_block_delta',
                    delta: { type: 'text_delta', text: 'OK' },
                },
                {
                    type: 'message_delta',
                    delta: { stop_reason: 'end_turn' },
                    usage: { output_tokens: 1 },
                },
            ]),
        );
        const driver = {
            logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} },
            options: { region: 'us-central1' },
            streamVertexModel,
        } as unknown as VertexAIDriver;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
                cache_enabled: true,
            },
        } as ExecutionOptions;

        const first = await modelDef.requestTextCompletion(
            driver,
            {
                messages: [{ role: 'user', content: [{ type: 'text', text: 'First prompt' }] }],
                system: [{ type: 'text', text: 'System prompt' }],
            } as unknown as ClaudePrompt,
            options,
        );
        const firstConversation = first.conversation as ClaudePrompt;
        expect(firstConversation.messages[1]?.content).toEqual([{ type: 'text', text: 'OK' }]);

        const second = await modelDef.requestTextCompletion(
            driver,
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Second prompt' }] }] } as ClaudePrompt,
            { ...options, conversation: firstConversation },
        );
        await modelDef.requestTextCompletion(
            driver,
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Third prompt' }] }] } as ClaudePrompt,
            { ...options, conversation: second.conversation },
        );

        const streamCalls = streamVertexModel.mock.calls as unknown as Array<
            [string, string, { messages: ClaudePrompt['messages'] }]
        >;
        const thirdPayload = streamCalls[2]?.[2];
        expect(thirdPayload).toBeDefined();
        if (!thirdPayload) {
            throw new Error('Expected third Vertex Claude request payload');
        }

        const pivotMessage = thirdPayload.messages[thirdPayload.messages.length - 2];
        expect(Array.isArray(pivotMessage.content)).toBe(true);

        const pivotContent = pivotMessage.content as Array<{ cache_control?: unknown }>;
        expect(pivotContent[pivotContent.length - 1]?.cache_control).toEqual({ type: 'ephemeral' });
    });
});
