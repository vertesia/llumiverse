import { type CompletionChunkObject, type ExecutionOptions, PromptRole } from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { withVertexOpenAIClient } from '../../../test/__helpers__/vertex-openai-client.js';
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
            getRequestHeaders: async () => new Headers({ authorization: 'Bearer test-token' }),
        } as unknown as VertexAIDriver['googleAuth'];
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
        const driver = withVertexOpenAIClient({
            getFetchClientForRegion: () => ({
                post: async () => stream,
            }),
        });
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
        const driver = withVertexOpenAIClient({
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
        });
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const completion = await modelDef.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([{ type: 'thoughts', value: 'fallback text' }]);
    });

    it('reads text from non-streaming OpenAI content arrays', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'deepseek-ai/deepseek-v3.2-maas',
            region: 'global',
        });
        const driver = withVertexOpenAIClient({
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
        });
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/deepseek-ai/models/deepseek-v3.2-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const completion = await modelDef.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([
            { type: 'thoughts', value: 'hidden reasoning' },
            { type: 'text', value: 'first\nsecond' },
        ]);
    });

    it('keeps normal content and reasoning fields separate', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'zai-org/glm-5-maas',
            region: 'global',
        });
        const driver = withVertexOpenAIClient({
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
        });
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const completion = await modelDef.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([
            { type: 'thoughts', value: 'hidden reasoning' },
            { type: 'text', value: 'visible content' },
        ]);
    });

    it('uses strict response_format for structured Vertex GPT output', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'openai/gpt-oss-120b-maas',
            region: 'global',
        });
        const post = vi.fn(async () => ({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'openai/gpt-oss-120b-maas',
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
        const driver = withVertexOpenAIClient({
            getFetchClientForRegion: () => ({ post }),
        });
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/openai/models/gpt-oss-120b-maas',
            model_options: { _option_id: 'text-fallback' },
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
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

    it('uses provider-compatible tool schemas for Vertex MaaS requests', async () => {
        const modelDef = new OpenAIChatCompletionsModelDefinition({
            modelName: 'meta/llama-3.3-70b-instruct-maas',
            region: 'us-central1',
        });
        const post = vi.fn(async () => ({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'meta/llama-3.3-70b-instruct-maas',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: 'Done',
                    },
                    finish_reason: 'stop',
                },
            ],
        }));
        const driver = withVertexOpenAIClient({
            getFetchClientForRegion: () => ({ post }),
        });
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Use a tool if needed' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/us-central1/publishers/meta/models/llama-3.3-70b-instruct-maas',
            model_options: { _option_id: 'text-fallback' },
            tools: [
                {
                    name: 'think',
                    description: 'Think before answering',
                    input_schema: {
                        type: 'object',
                        properties: {
                            thought: { type: 'string' },
                            message_to_human: { type: 'string' },
                        },
                        required: ['thought'],
                    },
                },
            ],
        };

        await modelDef.requestTextCompletion(driver, prompt, options);

        expect(post).toHaveBeenCalledWith('endpoints/openapi/chat/completions', {
            payload: expect.objectContaining({
                tools: [
                    {
                        type: 'function',
                        function: {
                            name: 'think',
                            description: 'Think before answering',
                            parameters: {
                                type: 'object',
                                properties: {
                                    thought: { type: 'string' },
                                    message_to_human: { type: 'string' },
                                },
                                required: ['thought'],
                            },
                        },
                    },
                ],
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
        const driver = withVertexOpenAIClient({
            getFetchClientForRegion: () => ({
                post: async () => stream,
            }),
        });
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
        const driver = withVertexOpenAIClient({
            getFetchClientForRegion: () => ({
                post: async () => stream,
            }),
        });
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const chunks = await collectChunks(await modelDef.requestTextCompletionStream(driver, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([
            { type: 'thoughts', value: 'hidden' },
            { type: 'text', value: 'visible' },
        ]);
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
        const driver = withVertexOpenAIClient({
            getFetchClientForRegion: () => ({
                post: async () => stream,
            }),
        });
        const prompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'Hello' }],
        };
        const options: ExecutionOptions = {
            model: 'locations/global/publishers/zai-org/models/glm-5-maas',
            model_options: { _option_id: 'text-fallback' },
        };

        const chunks = await collectChunks(await modelDef.requestTextCompletionStream(driver, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([
            { type: 'thoughts', value: 'fallback' },
            { type: 'thoughts', value: ' text' },
        ]);
    });
});

describe('Vertex MaaS Chat Completions through the OpenAI SDK', () => {
    function completionResponse() {
        return new Response(
            JSON.stringify({
                id: 'chatcmpl-1',
                object: 'chat.completion',
                created: 1,
                model: 'm',
                choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
                usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
    }

    async function send(model: string) {
        const fetchMock = vi.fn(async () => completionResponse());
        const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);
        await driver.execute([{ role: PromptRole.user, content: 'hello' }], {
            model,
            model_options: { _option_id: 'text-fallback' },
        });
        const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
        return { url: String(url), headers: new Headers(init.headers), body: JSON.parse(String(init.body)) };
    }

    afterEach(() => vi.unstubAllEnvs());

    it('posts to the regional endpoint with the API version and Google auth, and no OpenAI account headers', async () => {
        vi.stubEnv('OPENAI_ORG_ID', 'org-from-env');
        vi.stubEnv('OPENAI_PROJECT_ID', 'proj_from_env');
        const { url, headers, body } = await send(
            'locations/us-east5/publishers/meta/models/llama-4-scout-17b-16e-instruct-maas',
        );
        expect(url).toBe(
            'https://us-east5-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-east5/endpoints/openapi/chat/completions',
        );
        expect(headers.get('authorization')).toBe('Bearer test-token');
        expect(headers.get('openai-organization')).toBeNull();
        expect(headers.get('openai-project')).toBeNull();
        expect(body.model).toBe('meta/llama-4-scout-17b-16e-instruct-maas');
        // Llama's safety settings stay in the literal `extra_body` field the MaaS endpoint reads.
        expect(body.extra_body).toMatchObject({ google: { model_safety_settings: { enabled: false } } });
        expect(body).not.toHaveProperty('google');
    });

    it('uses the global host for global MaaS models', async () => {
        const { url } = await send('locations/global/publishers/zai-org/models/glm-5.2-maas');
        expect(url).toBe(
            'https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/openapi/chat/completions',
        );
    });

    function sseResponse(chunks: unknown[]) {
        const body = [...chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`), 'data: [DONE]\n\n'].join('');
        return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    }

    async function stream(model: string) {
        const chunk = (delta: object, finish_reason: string | null = null) => ({
            id: 'chatcmpl-1',
            object: 'chat.completion.chunk',
            created: 1,
            model: 'm',
            choices: [{ index: 0, delta, finish_reason }],
        });
        const fetchMock = vi.fn(async () =>
            sseResponse([
                chunk({ role: 'assistant', content: 'o' }),
                chunk({ content: 'k' }, 'stop'),
                { ...chunk({}), choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } },
            ]),
        );
        const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);
        const completionStream = await driver.stream([{ role: PromptRole.user, content: 'hello' }], {
            model,
            model_options: { _option_id: 'text-fallback' },
        });
        let text = '';
        for await (const part of completionStream) text += part;
        const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
        return { url: String(url), body: JSON.parse(String(init.body)), text, completion: completionStream.completion };
    }

    it('streams MaaS models through the SDK and asks for usage', async () => {
        const { url, body, text, completion } = await stream('locations/global/publishers/zai-org/models/glm-5.2-maas');
        expect(url).toBe(
            'https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/openapi/chat/completions',
        );
        expect(body).toMatchObject({ stream: true, stream_options: { include_usage: true } });
        expect(text).toBe('ok');
        expect(completion?.token_usage).toMatchObject({ prompt: 3, result: 2 });
    });

    it('streams regional Mistral models from their streamRawPredict path', async () => {
        const { url, body, text } = await stream(
            'locations/europe-west4/publishers/mistralai/models/mistral-small-2503',
        );
        expect(url).toBe(
            'https://europe-west4-aiplatform.googleapis.com/v1/projects/test-project/locations/europe-west4/publishers/mistralai/models/mistral-small-2503:streamRawPredict',
        );
        expect(body.stream).toBe(true);
        expect(text).toBe('ok');
    });

    it('reports a provider error with its status code', async () => {
        const fetchMock = vi.fn(
            async () =>
                new Response(JSON.stringify({ error: { code: 429, message: 'Resource exhausted' } }), {
                    status: 429,
                    headers: { 'Content-Type': 'application/json' },
                }),
        );
        const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);
        await expect(
            driver.execute([{ role: PromptRole.user, content: 'hello' }], {
                model: 'locations/global/publishers/zai-org/models/glm-5.2-maas',
                model_options: { _option_id: 'text-fallback' },
            }),
        ).rejects.toMatchObject({ code: 429, retryable: true });
    });

    it('posts regional Mistral models to their publisher rawPredict path', async () => {
        const { url } = await send('locations/europe-west4/publishers/mistralai/models/mistral-small-2503');
        expect(url).toBe(
            'https://europe-west4-aiplatform.googleapis.com/v1/projects/test-project/locations/europe-west4/publishers/mistralai/models/mistral-small-2503:rawPredict',
        );
    });
});
