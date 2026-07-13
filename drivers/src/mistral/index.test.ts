import { PromptRole } from '@llumiverse/core';
import { InvalidRequestError, RequestTimeoutError } from '@mistralai/mistralai/models/errors';
import { describe, expect, it, vi } from 'vitest';
import { MistralAIDriver } from './index.js';

describe('MistralAIDriver official SDK transport', () => {
    it('preserves signed thinking for replay after JSON roundtrip while projecting thoughts by default', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const response = {
            id: 'mistral-signed',
            object: 'chat.completion',
            created: 1,
            model: 'magistral',
            choices: [
                {
                    index: 0,
                    finishReason: 'stop',
                    message: {
                        role: 'assistant' as const,
                        content: [
                            {
                                type: 'thinking' as const,
                                thinking: [{ type: 'text' as const, text: 'native plan' }],
                                signature: 'signed-thinking',
                                closed: true,
                            },
                            { type: 'text' as const, text: 'answer' },
                        ],
                    },
                },
            ],
            usage: { promptTokens: 2, completionTokens: 3, totalTokens: 5 },
        };
        const complete = vi.fn(async (_request: unknown) => response);
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });

        const first = await driver.requestTextCompletion(
            { messages: [{ role: 'user', content: 'question' }] },
            { model: 'magistral', stripTextMaxTokens: 1, stripImagesAfterTurns: 0 },
        );
        expect(first.result).toEqual([
            { type: 'thoughts', value: 'native plan' },
            { type: 'text', value: 'answer' },
        ]);

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        await driver.requestTextCompletion(
            { messages: [{ role: 'user', content: 'continue' }] },
            { model: 'magistral', conversation: persisted },
        );
        expect((complete.mock.calls[1][0] as { messages: unknown[] }).messages).toContainEqual(
            response.choices[0].message,
        );

        const hidden = await driver.requestTextCompletion(
            { messages: [{ role: 'user', content: 'hidden' }] },
            {
                model: 'magistral',
                model_options: { _option_id: 'text-fallback', include_thoughts: false },
            },
        );
        expect(hidden.result).toEqual([{ type: 'text', value: 'answer' }]);
        expect(hidden.conversation).toMatchObject({
            messages: expect.arrayContaining([response.choices[0].message]),
        });
    });

    it('reconstructs fragmented signed thinking in the native streaming assistant message', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        Object.defineProperty(driver.client.chat, 'stream', {
            value: vi.fn(async () =>
                (async function* () {
                    yield {
                        data: {
                            id: 'chunk-1',
                            model: 'magistral',
                            choices: [
                                {
                                    index: 0,
                                    delta: {
                                        content: [
                                            {
                                                type: 'thinking',
                                                thinking: [{ type: 'text', text: 'plan' }],
                                            },
                                        ],
                                    },
                                },
                            ],
                        },
                    };
                    yield {
                        data: {
                            id: 'chunk-2',
                            model: 'magistral',
                            choices: [
                                {
                                    index: 0,
                                    finishReason: 'stop',
                                    delta: {
                                        content: [
                                            {
                                                type: 'thinking',
                                                thinking: [{ type: 'text', text: ' more' }],
                                                signature: 'stream-signature',
                                                closed: true,
                                            },
                                            { type: 'text', text: 'answer' },
                                        ],
                                    },
                                },
                            ],
                        },
                    };
                })(),
            ),
        });

        const stream = await driver.requestTextCompletionStream(
            { messages: [{ role: 'user', content: 'question' }] },
            { model: 'magistral' },
        );
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.({ result: results });

        expect(results).toEqual([
            { type: 'thoughts', value: 'plan' },
            { type: 'thoughts', value: ' more' },
            { type: 'text', value: 'answer' },
        ]);
        expect(conversation).toMatchObject({
            messages: expect.arrayContaining([
                expect.objectContaining({
                    role: 'assistant',
                    content: [
                        {
                            type: 'thinking',
                            thinking: [
                                { type: 'text', text: 'plan' },
                                { type: 'text', text: ' more' },
                            ],
                            signature: 'stream-signature',
                            closed: true,
                        },
                        { type: 'text', text: 'answer' },
                    ],
                }),
            ]),
        });
    });

    it('maps canonical Chat requests and preserves typed content, tools, usage, and the native response', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key', endpoint_url: 'https://mistral.example.test' });
        const response = {
            id: 'mistral-1',
            object: 'chat.completion',
            created: 1,
            model: 'mistral-large',
            choices: [
                {
                    index: 0,
                    finishReason: 'tool_calls',
                    message: {
                        role: 'assistant',
                        content: [{ type: 'text', text: 'hello' }],
                        toolCalls: [
                            {
                                id: 'call_1',
                                index: 0,
                                type: 'function',
                                function: { name: 'lookup', arguments: { city: 'Paris' } },
                            },
                        ],
                    },
                },
            ],
            usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
        };
        const complete = vi.fn(async () => response);
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], {
            model: 'mistral-large',
        });
        const completion = await driver.requestTextCompletion(prompt, {
            model: 'mistral-large',
            include_original_response: true,
            model_options: {
                _option_id: 'text-fallback',
                max_tokens: 12,
                top_p: 0.8,
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop_sequence: ['END'],
            },
            tools: [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } }],
        });

        expect(complete).toHaveBeenCalledWith(
            expect.objectContaining({
                maxTokens: 12,
                topP: 0.8,
                presencePenalty: 0.1,
                frequencyPenalty: 0.2,
                stop: ['END'],
                stream: false,
            }),
        );
        expect(completion.result).toEqual([{ type: 'text', value: 'hello' }]);
        expect(completion.tool_use?.[0]).toEqual({
            id: 'call_1',
            tool_name: 'lookup',
            tool_input: { city: 'Paris' },
        });
        expect(completion.original_response).toBe(response);
    });

    it('uses the SDK for models, validation, embeddings, endpoint override, and fetch wiring', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key', endpoint_url: 'https://mistral.example.test' });
        const list = vi.fn(async () => ({
            object: 'list',
            data: [
                {
                    id: 'mistral-large',
                    name: 'Mistral Large',
                    description: 'Large model',
                    ownedBy: 'mistralai',
                    type: 'base',
                },
            ],
        }));
        Object.defineProperty(driver.client.models, 'list', { value: list });
        Object.defineProperty(driver.client.embeddings, 'create', {
            value: vi.fn(async () => ({
                id: 'emb-1',
                object: 'list',
                model: 'mistral-embed',
                data: [{ index: 0, embedding: [0.1, 0.2] }],
                usage: { promptTokens: 2, totalTokens: 2 },
            })),
        });

        await expect(driver.validateConnection()).resolves.toBe(true);
        await expect(driver.listModels()).resolves.toEqual([
            expect.objectContaining({ id: 'mistral-large', name: 'Mistral Large', owner: 'mistralai' }),
        ]);
        await expect(
            driver.generateEmbeddings({ model: 'mistral-embed', inputs: [{ type: 'text', text: 'hello' }] }),
        ).resolves.toEqual({
            model: 'mistral-embed',
            results: [{ outputs: [{ values: [0.1, 0.2], modality: 'text' }] }],
            usage: { input_tokens: 2, input_text_tokens: 2 },
        });
        expect(list).toHaveBeenCalledTimes(2);
    });

    it('classifies Mistral transport and request errors', () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const context = { provider: driver.provider, model: 'mistral-large', operation: 'execute' as const };

        expect(driver.formatLlumiverseError(new RequestTimeoutError('timed out'), context)).toMatchObject({
            name: 'RequestTimeoutError',
            retryable: true,
        });
        expect(driver.formatLlumiverseError(new InvalidRequestError('invalid request'), context)).toMatchObject({
            name: 'InvalidRequestError',
            retryable: false,
        });
    });

    it('normalizes Mistral embedding transport errors', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        Object.defineProperty(driver.client.embeddings, 'create', {
            value: vi.fn(async () => {
                throw new RequestTimeoutError('timed out');
            }),
        });

        await expect(
            driver.generateEmbeddings({ model: 'mistral-embed', inputs: [{ type: 'text', text: 'hello' }] }),
        ).rejects.toMatchObject({
            name: 'RequestTimeoutError',
            retryable: true,
            context: { provider: driver.provider, model: 'mistral-embed', operation: 'execute' },
            originalError: expect.any(RequestTimeoutError),
        });
    });

    it('preserves array-shaped assistant and tool content at the Mistral SDK boundary', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const complete = vi.fn(async (_request: unknown) => ({
            id: 'mistral-1',
            object: 'chat.completion',
            created: 1,
            model: 'mistral-large',
            choices: [{ index: 0, finishReason: 'stop', message: { role: 'assistant', content: 'ok' } }],
            usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        }));
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });

        await driver.requestTextCompletion(
            {
                _is_openai_chat_completions: true,
                messages: [
                    {
                        role: 'assistant',
                        content: [
                            { type: 'text', text: 'working' },
                            { type: 'image_url', image_url: { url: 'https://example.test/context.png' } },
                        ],
                        tool_calls: [
                            {
                                id: 'call_1',
                                type: 'function',
                                function: { name: 'lookup', arguments: '{}' },
                            },
                        ],
                    },
                    {
                        role: 'tool',
                        tool_call_id: 'call_1',
                        content: [
                            { type: 'text', text: 'result' },
                            { type: 'image_url', image_url: { url: 'https://example.test/image.png' } },
                        ],
                    },
                ],
            },
            { model: 'mistral-large', tools: [{ name: 'lookup', input_schema: { type: 'object' } }] },
        );

        const request = complete.mock.calls[0][0] as { messages: unknown[] };
        expect(request.messages).toEqual([
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'working' },
                    { type: 'image_url', imageUrl: 'https://example.test/context.png' },
                ],
                toolCalls: [
                    {
                        id: 'call_1',
                        index: 0,
                        type: 'function',
                        function: { name: 'lookup', arguments: '{}' },
                    },
                ],
            },
            {
                role: 'tool',
                toolCallId: 'call_1',
                content: [
                    { type: 'text', text: 'result' },
                    { type: 'image_url', imageUrl: 'https://example.test/image.png' },
                ],
            },
        ]);
    });
});
