import { type CompletionChunkObject, type ExecutionOptions, getConversationMeta } from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import { describe, expect, it } from 'vitest';
import {
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsResponse,
    stripOpenAIChatCompletionsThinkBlocksFromCompletion,
} from './openai_chat_completions.js';

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

class TestOpenAIChatCompletionsProtocol extends OpenAIChatCompletionsProtocol<undefined> {
    payloads: OpenAIChatCompletionsPayload[] = [];

    constructor(
        private readonly response?: OpenAIChatCompletionsResponse,
        private readonly stream?: ReadableStream,
    ) {
        super({ modelName: 'test/model' });
    }

    protected async postChatCompletion(
        _driver: undefined,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<OpenAIChatCompletionsResponse> {
        this.payloads.push(payload);
        if (!this.response) {
            throw new Error('Missing test response');
        }
        return this.response;
    }

    protected async postChatCompletionStream(
        _driver: undefined,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<ReadableStream> {
        this.payloads.push(payload);
        if (!this.stream) {
            throw new Error('Missing test stream');
        }
        return this.stream;
    }
}

const prompt: OpenAIChatCompletionsPrompt = {
    _is_openai_chat_completions: true,
    messages: [{ role: 'user', content: 'Hello' }],
};

const options: ExecutionOptions = {
    model: 'test/model',
    model_options: { _option_id: 'text-fallback' },
};

describe('OpenAIChatCompletionsProtocol', () => {
    it('reads text from non-streaming content arrays', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: [
                            { type: 'text', text: 'first' },
                            { type: 'text', text: 'second' },
                        ],
                        reasoning_content: 'hidden',
                    },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: 'first\nsecond' }]);
    });

    it('uses reasoning as a non-streaming fallback only when content is absent', async () => {
        const fallbackModel = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: null, reasoning: 'fallback text' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const contentModel = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-2',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'visible', reasoning_content: 'hidden' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await expect(fallbackModel.requestTextCompletion(undefined, prompt, options)).resolves.toMatchObject({
            result: [{ type: 'text', value: 'fallback text' }],
        });
        await expect(contentModel.requestTextCompletion(undefined, prompt, options)).resolves.toMatchObject({
            result: [{ type: 'text', value: 'visible' }],
        });
    });

    it('uses reasoning as a non-streaming fallback when content is blank after think-block stripping', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: '<think>hidden</think>\n\n',
                        reasoning: 'fallback text',
                    },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: 'fallback text' }]);
    });

    it('strips provider-emitted think blocks from completed non-streaming output and conversation', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: '<think>hidden reasoning</think>\n\n{"answer":"Paris"}',
                    },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: '{"answer":"Paris"}' }]);
        expect(completion.conversation).toMatchObject({
            messages: expect.arrayContaining([{ role: 'assistant', content: '{"answer":"Paris"}' }]),
        });
    });

    it('applies conversation stripping and turn metadata to non-streaming completions', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const imagePrompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'What is this?' },
                        {
                            type: 'image_url',
                            image_url: { url: 'data:image/png;base64,aW1hZ2U=', detail: 'auto' },
                        },
                    ],
                },
            ],
        };

        const completion = await model.requestTextCompletion(undefined, imagePrompt, {
            ...options,
            stripImagesAfterTurns: 0,
        });
        const conversation = completion.conversation as OpenAIChatCompletionsPrompt;

        expect(getConversationMeta(conversation).turnNumber).toBe(1);
        expect(conversation.messages[0].content).toEqual([
            { type: 'text', text: 'What is this?' },
            { type: 'text', text: '[Image removed from conversation history]' },
        ]);
    });

    it('strips provider-emitted think blocks from final accumulated completion objects', () => {
        const completion = stripOpenAIChatCompletionsThinkBlocksFromCompletion({
            result: [{ type: 'text', value: '<think>hidden</think>\n\nfinal answer' }],
            conversation: {
                _is_openai_chat_completions: true,
                messages: [
                    { role: 'user', content: 'Hello' },
                    { role: 'assistant', content: '<think>hidden</think>\n\nfinal answer' },
                ],
            },
        });

        expect(completion.result).toEqual([{ type: 'text', value: 'final answer' }]);
        expect(completion.conversation).toMatchObject({
            messages: [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'final answer' },
            ],
        });
    });

    it('reads streaming content arrays', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { content: [{ type: 'text', text: 'hello' }] } }],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([{ type: 'text', value: 'hello' }]);
    });

    it('uses buffered streaming reasoning as a fallback only when no content arrives', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { reasoning_content: 'fallback' } }],
                    }),
                },
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { reasoning: ' text' }, finish_reason: 'stop' }],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([{ type: 'text', value: 'fallback text' }]);
    });

    it('uses streaming reasoning fallback when content is blank after think-block stripping', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { reasoning: 'fallback text' } }],
                    }),
                },
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { content: '<think>hidden</think>' }, finish_reason: 'stop' }],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([{ type: 'text', value: 'fallback text' }]);
    });

    it('normalizes non-streaming tool-call finish reasons', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            {
                                id: 'call_1',
                                type: 'function',
                                function: { name: 'get_weather', arguments: '{"location":"Paris"}' },
                            },
                        ],
                    },
                    finish_reason: 'tool_calls',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.finish_reason).toBe('tool_use');
    });

    it('preserves streaming function tool-call chunks', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
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
                        model: 'test/model',
                        choices: [
                            {
                                index: 0,
                                delta: { tool_calls: [{ index: 0, function: { arguments: ':"Paris"}' } }] },
                                finish_reason: 'tool_calls',
                            },
                        ],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));
        const toolChunks = chunks.flatMap((chunk) => chunk.tool_use ?? []);

        expect(toolChunks.map((tool) => tool.id)).toEqual(['tool_0', 'tool_0']);
        expect(toolChunks[0]).toMatchObject({
            _actual_id: 'call_real_id',
            tool_name: 'get_weather',
            tool_input: '{"location"',
        });
        expect(toolChunks.map((tool) => tool.tool_input).join('')).toBe('{"location":"Paris"}');
        expect(chunks[chunks.length - 1].finish_reason).toBe('tool_use');
    });

    it('normalizes tool schemas and structured-output schemas for Chat Completions payloads', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: '{"answer":"Paris"}' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
            },
            tools: [
                {
                    name: 'get_weather',
                    description: 'Get weather',
                    input_schema: {
                        type: 'object',
                        properties: { location: { type: 'string' } },
                    },
                },
            ],
        });

        expect(model.payloads[0].response_format).toEqual({
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
        });
        const toolDefinition = model.payloads[0].tools?.[0];
        expect(toolDefinition?.type).toBe('function');
        if (toolDefinition?.type !== 'function') {
            throw new Error('Expected function tool definition');
        }
        expect(toolDefinition.function).toEqual({
            name: 'get_weather',
            description: 'Get weather',
            strict: true,
            parameters: {
                type: 'object',
                properties: { location: { type: 'string' } },
                required: ['location'],
                additionalProperties: false,
            },
        });
    });
});
