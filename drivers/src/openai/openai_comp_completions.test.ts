import type { CompletionChunkObject, ExecutionOptions } from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import { describe, expect, it } from 'vitest';
import {
    OpenAICompletionsModelDefinitionBase,
    type OpenAICompletionsPayload,
    type OpenAICompletionsPrompt,
    type OpenAICompletionsResponse,
} from './openai_comp_completions.js';

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

class TestCompletionsModel extends OpenAICompletionsModelDefinitionBase<undefined> {
    payloads: OpenAICompletionsPayload[] = [];

    constructor(
        private readonly response?: OpenAICompletionsResponse,
        private readonly stream?: ReadableStream,
    ) {
        super({ modelName: 'test/model' });
    }

    protected async postChatCompletion(
        _driver: undefined,
        payload: OpenAICompletionsPayload,
    ): Promise<OpenAICompletionsResponse> {
        this.payloads.push(payload);
        if (!this.response) {
            throw new Error('Missing test response');
        }
        return this.response;
    }

    protected async postChatCompletionStream(
        _driver: undefined,
        payload: OpenAICompletionsPayload,
    ): Promise<ReadableStream> {
        this.payloads.push(payload);
        if (!this.stream) {
            throw new Error('Missing test stream');
        }
        return this.stream;
    }
}

const prompt: OpenAICompletionsPrompt = {
    _is_openai_compat: true,
    messages: [{ role: 'user', content: 'Hello' }],
};

const options: ExecutionOptions = {
    model: 'test/model',
    model_options: { _option_id: 'text-fallback' },
};

describe('OpenAICompletionsModelDefinitionBase', () => {
    it('reads text from non-streaming content arrays', async () => {
        const model = new TestCompletionsModel({
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
        const fallbackModel = new TestCompletionsModel({
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
        const contentModel = new TestCompletionsModel({
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

    it('reads streaming content arrays', async () => {
        const model = new TestCompletionsModel(
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
        const model = new TestCompletionsModel(
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

    it('preserves streaming function tool-call chunks', async () => {
        const model = new TestCompletionsModel(
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
    });
});
