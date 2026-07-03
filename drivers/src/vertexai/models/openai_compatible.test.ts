import type { CompletionChunkObject, ExecutionOptions } from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { OpenAICompatibleModelDefinition, type OpenAIPrompt } from './openai_compatible.js';

function createSSEStream(events: ServerSentEvent[]): ReadableStream<ServerSentEvent> {
    return new ReadableStream<ServerSentEvent>({
        start(controller) {
            for (const event of events) {
                controller.enqueue(event);
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

class TestVertexAIDriver extends VertexAIDriver {
    constructor(private readonly fetchMock: typeof fetch) {
        super({ project: 'test-project', region: 'us-central1' });
        this.googleAuth = {
            getAccessToken: async () => 'test-token',
        } as VertexAIDriver['googleAuth'];
    }

    protected override getDriverFetch(): typeof fetch {
        return this.fetchMock;
    }
}

describe('OpenAICompatibleModelDefinition', () => {
    it('uses a stable streaming tool-call key and preserves the real OpenAI tool id', async () => {
        const modelDef = new OpenAICompatibleModelDefinition({ modelName: 'xai/grok-4.1', region: 'global' });
        const stream = createSSEStream([
            {
                type: 'event',
                data: JSON.stringify({
                    id: 'chatcmpl-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'xai/grok-4.1',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                tool_calls: [
                                    {
                                        index: 0,
                                        id: 'call_real_id',
                                        type: 'function',
                                        function: { name: 'get_weather', arguments: '{"location"' },
                                    },
                                ],
                            },
                        },
                    ],
                }),
            },
            {
                type: 'event',
                data: JSON.stringify({
                    id: 'chatcmpl-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'xai/grok-4.1',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                tool_calls: [{ index: 0, function: { arguments: ':"Paris"' } }],
                            },
                        },
                    ],
                }),
            },
            {
                type: 'event',
                data: JSON.stringify({
                    id: 'chatcmpl-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'xai/grok-4.1',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                tool_calls: [{ index: 0, function: { arguments: '}' } }],
                            },
                            finish_reason: 'tool_calls',
                        },
                    ],
                }),
            },
        ]);
        const driver = {
            getFetchClientForRegion: () => ({
                post: async () => stream,
            }),
        } as unknown as VertexAIDriver;
        const prompt: OpenAIPrompt = { _is_openai_compat: true, messages: [{ role: 'user', content: 'Weather?' }] };
        const options: ExecutionOptions = {
            model: 'publishers/xai/models/grok-4.1',
            model_options: { _option_id: 'text-fallback' },
        };

        const chunks = await collectChunks(await modelDef.requestTextCompletionStream(driver, prompt, options));
        const toolChunks = chunks.flatMap((chunk) => chunk.tool_use ?? []);

        expect(toolChunks).toHaveLength(3);
        expect(toolChunks.map((tool) => tool.id)).toEqual(['tool_0', 'tool_0', 'tool_0']);
        expect(toolChunks[0]).toMatchObject({
            _actual_id: 'call_real_id',
            tool_name: 'get_weather',
            tool_input: '{"location"',
        });
        expect(toolChunks.map((tool) => tool.tool_input).join('')).toBe('{"location":"Paris"}');
    });

    it('uses the driver fetch implementation for region override clients', async () => {
        const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }))) as unknown as typeof fetch;
        const driver = new TestVertexAIDriver(fetchMock);
        const client = driver.getFetchClientForRegion('global');

        await client.post('endpoints/openapi/chat/completions', { payload: { model: 'xai/grok-4.1' } });

        expect(fetchMock).toHaveBeenCalledTimes(1);
        const request = vi.mocked(fetchMock).mock.calls[0][0] as Request;
        expect(request.url).toContain(
            'https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/openapi/chat/completions',
        );
    });

    it('builds streaming conversation when there is no prior OpenAI-compatible conversation', () => {
        const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }))) as unknown as typeof fetch;
        const driver = new TestVertexAIDriver(fetchMock);
        const prompt: OpenAIPrompt = { _is_openai_compat: true, messages: [{ role: 'user', content: 'Hello' }] };

        const conversation = driver.buildStreamingConversation(
            prompt,
            [{ type: 'text', value: 'Hi there' }],
            undefined,
            { model: 'publishers/xai/models/grok-4.1' },
        ) as OpenAIPrompt;

        expect(conversation.messages).toEqual([
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi there' },
        ]);
    });
});
