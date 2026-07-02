import type { CompletionChunkObject, ExecutionOptions } from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import { describe, expect, it, vi } from 'vitest';
import type { OpenAIChatCompletionsPrompt } from '../../openai/openai_chat_completions.js';
import { VertexAIDriver } from '../index.js';
import { OpenAIChatCompletionsModelDefinition } from './openai_chat_completions.js';

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

describe('OpenAIChatCompletionsModelDefinition', () => {
    it('uses a stable streaming tool-call key and preserves the real OpenAI tool id', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({ modelName: 'xai/grok-4.1', region: 'global' });
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
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Weather?' }],
        };
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
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };

        const conversation = driver.buildStreamingConversation(
            prompt,
            [{ type: 'text', value: 'Hi there' }],
            undefined,
            { model: 'publishers/xai/models/grok-4.1' },
        ) as OpenAIChatCompletionsPrompt;

        expect(conversation.messages).toEqual([
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi there' },
        ]);
    });

    it('uses reasoning fields as a fallback when non-streaming content is blank', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'zai-org/glm-5-maas',
            region: 'global',
        });
        const driver = {
            getFetchClientForRegion: () => ({
                post: async () => ({
                    id: 'chatcmpl-1',
                    object: 'chat.completion',
                    created: 1,
                    model: 'zai-org/glm-5-maas',
                    choices: [
                        {
                            index: 0,
                            message: {
                                role: 'assistant',
                                content: null,
                                reasoning: 'fallback text',
                            },
                            finish_reason: 'stop',
                        },
                    ],
                }),
            }),
        } as unknown as VertexAIDriver;
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const completion = await modelDef.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: 'fallback text' }]);
    });

    it('reads text from non-streaming OpenAI content arrays', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'deepseek-ai/deepseek-v3.2-maas',
            region: 'global',
        });
        const driver = {
            getFetchClientForRegion: () => ({
                post: async () => ({
                    id: 'chatcmpl-1',
                    object: 'chat.completion',
                    created: 1,
                    model: 'deepseek-ai/deepseek-v3.2-maas',
                    choices: [
                        {
                            index: 0,
                            message: {
                                role: 'assistant',
                                content: [
                                    { type: 'text', text: 'first' },
                                    { type: 'text', text: 'second' },
                                ],
                                reasoning_content: 'hidden reasoning',
                            },
                            finish_reason: 'stop',
                        },
                    ],
                }),
            }),
        } as unknown as VertexAIDriver;
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/deepseek-ai/models/deepseek-v3.2-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const completion = await modelDef.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: 'first\nsecond' }]);
    });

    it('prefers normal content over reasoning fields in non-streaming responses', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'zai-org/glm-5-maas',
            region: 'global',
        });
        const driver = {
            getFetchClientForRegion: () => ({
                post: async () => ({
                    id: 'chatcmpl-1',
                    object: 'chat.completion',
                    created: 1,
                    model: 'zai-org/glm-5-maas',
                    choices: [
                        {
                            index: 0,
                            message: {
                                role: 'assistant',
                                content: 'visible content',
                                reasoning_content: 'hidden reasoning',
                            },
                            finish_reason: 'stop',
                        },
                    ],
                }),
            }),
        } as unknown as VertexAIDriver;
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const completion = await modelDef.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: 'visible content' }]);
    });

    it('uses response_format with normalized JSON schema for structured Vertex MaaS output', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'zai-org/glm-5-maas',
            region: 'global',
        });
        const post = vi.fn(async () => ({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'zai-org/glm-5-maas',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: '{"answer":"Paris"}',
                    },
                    finish_reason: 'stop',
                },
            ],
        }));
        const driver = {
            getFetchClientForRegion: () => ({ post }),
        } as unknown as VertexAIDriver;
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
            },
        };

        await modelDef.requestTextCompletion(driver, prompt, options);

        expect(post).toHaveBeenCalledWith('endpoints/openapi/chat/completions', {
            payload: expect.objectContaining({
                response_format: {
                    type: 'json_schema',
                    json_schema: {
                        name: 'output',
                        strict: true,
                        schema: {
                            type: 'object',
                            properties: { answer: { type: 'string' } },
                            required: ['answer'],
                            additionalProperties: false,
                        },
                    },
                },
            }),
        });
    });

    it('reads text from streaming OpenAI content arrays', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'deepseek-ai/deepseek-v3.2-maas',
            region: 'global',
        });
        const stream = createSSEStream([
            {
                type: 'event',
                data: JSON.stringify({
                    id: 'chatcmpl-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'deepseek-ai/deepseek-v3.2-maas',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                content: [{ type: 'text', text: 'hello' }],
                            },
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
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/deepseek-ai/models/deepseek-v3.2-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const chunks = await collectChunks(await modelDef.requestTextCompletionStream(driver, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([{ type: 'text', value: 'hello' }]);
    });

    it('does not emit streaming reasoning fallback after normal content arrives', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'zai-org/glm-5-maas',
            region: 'global',
        });
        const stream = createSSEStream([
            {
                type: 'event',
                data: JSON.stringify({
                    id: 'chatcmpl-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'zai-org/glm-5-maas',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                reasoning_content: 'hidden',
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
                    model: 'zai-org/glm-5-maas',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                content: 'visible',
                            },
                            finish_reason: 'stop',
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
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const chunks = await collectChunks(await modelDef.requestTextCompletionStream(driver, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([{ type: 'text', value: 'visible' }]);
    });

    it('uses buffered streaming reasoning as a fallback when no content deltas arrive', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'zai-org/glm-5-maas',
            region: 'global',
        });
        const stream = createSSEStream([
            {
                type: 'event',
                data: JSON.stringify({
                    id: 'chatcmpl-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'zai-org/glm-5-maas',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                reasoning: 'fallback',
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
                    model: 'zai-org/glm-5-maas',
                    choices: [
                        {
                            index: 0,
                            delta: {
                                reasoning: ' text',
                            },
                            finish_reason: 'stop',
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
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const chunks = await collectChunks(await modelDef.requestTextCompletionStream(driver, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([{ type: 'text', value: 'fallback text' }]);
    });
});
