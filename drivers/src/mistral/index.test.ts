import { PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { MistralAIDriver } from './index.js';

describe('MistralAIDriver official SDK transport', () => {
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

    it('replays signed thinking content after blocking and persisted conversation turns', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const signedContent = [
            {
                type: 'thinking' as const,
                thinking: [{ type: 'text' as const, text: 'private reasoning' }],
                signature: 'signed-thinking',
                closed: true,
            },
            { type: 'text' as const, text: 'visible answer' },
        ];
        const complete = vi
            .fn()
            .mockResolvedValueOnce({
                id: 'mistral-1',
                object: 'chat.completion',
                created: 1,
                model: 'magistral',
                choices: [
                    {
                        index: 0,
                        finishReason: 'stop',
                        message: { role: 'assistant', content: signedContent },
                    },
                ],
                usage: { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
            })
            .mockResolvedValueOnce({
                id: 'mistral-2',
                object: 'chat.completion',
                created: 2,
                model: 'magistral',
                choices: [
                    {
                        index: 0,
                        finishReason: 'stop',
                        message: { role: 'assistant', content: 'continued' },
                    },
                ],
                usage: { promptTokens: 4, completionTokens: 1, totalTokens: 5 },
            });
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });

        const firstPrompt = await driver.createPrompt([{ role: PromptRole.user, content: 'First turn' }], {
            model: 'magistral',
        });
        const first = await driver.requestTextCompletion(firstPrompt, { model: 'magistral' });
        const persistedConversation = JSON.parse(JSON.stringify(first.conversation)) as typeof first.conversation;
        const secondPrompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Second turn' }], {
            model: 'magistral',
        });
        await driver.requestTextCompletion(secondPrompt, {
            model: 'magistral',
            conversation: persistedConversation,
        });

        expect(complete).toHaveBeenNthCalledWith(
            2,
            expect.objectContaining({
                messages: expect.arrayContaining([{ role: 'assistant', content: signedContent }]),
            }),
        );
    });

    it('replays signed thinking content after streaming conversation turns', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const thinking = {
            type: 'thinking' as const,
            thinking: [{ type: 'text' as const, text: 'stream reasoning' }],
            signature: 'stream-signature',
            closed: true,
        };
        async function* events() {
            yield {
                data: {
                    id: 'chunk-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'magistral',
                    choices: [
                        {
                            index: 0,
                            finishReason: null,
                            delta: { role: 'assistant', content: [thinking] },
                        },
                    ],
                },
            };
            yield {
                data: {
                    id: 'chunk-2',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'magistral',
                    choices: [
                        {
                            index: 0,
                            finishReason: 'stop',
                            delta: { role: 'assistant', content: [{ type: 'text' as const, text: 'answer' }] },
                        },
                    ],
                },
            };
        }
        Object.defineProperty(driver.client.chat, 'stream', { value: vi.fn(async () => events()) });

        const stream = await driver.stream([{ role: PromptRole.user, content: 'First turn' }], { model: 'magistral' });
        for await (const _chunk of stream) {
            // Consume the stream to populate the final completion.
        }

        const complete = vi.fn(async () => ({
            id: 'mistral-2',
            object: 'chat.completion',
            created: 2,
            model: 'magistral',
            choices: [
                {
                    index: 0,
                    finishReason: 'stop',
                    message: { role: 'assistant', content: 'continued' },
                },
            ],
            usage: { promptTokens: 4, completionTokens: 1, totalTokens: 5 },
        }));
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });
        const secondPrompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Second turn' }], {
            model: 'magistral',
        });
        await driver.requestTextCompletion(secondPrompt, {
            model: 'magistral',
            conversation: JSON.parse(JSON.stringify(stream.completion?.conversation)),
        });

        expect(complete).toHaveBeenCalledWith(
            expect.objectContaining({
                messages: expect.arrayContaining([
                    {
                        role: 'assistant',
                        content: [thinking, { type: 'text', text: 'answer' }],
                    },
                ]),
            }),
        );
    });
});
