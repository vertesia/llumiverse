import { ModelType, PromptRole, Providers } from '@llumiverse/core';
import { APIConnectionTimeoutError } from 'together-ai/error';
import { describe, expect, it, vi } from 'vitest';
import { TogetherAIDriver } from './index.js';

describe('TogetherAIDriver', () => {
    it('maps Together array-shaped model catalog responses', async () => {
        const driver = new TogetherAIDriver({ apiKey: 'test-key' });
        const list = vi.fn(async () => [
            {
                id: 'Qwen/Qwen3.5-9B',
                display_name: 'Qwen3.5 9B',
                organization: 'Qwen',
                type: 'chat',
            },
            {
                id: 'togethercomputer/m2-bert-80M-8k-retrieval',
                display_name: 'M2 BERT Retrieval',
                organization: 'Together',
                type: 'embedding',
            },
        ]);
        driver.service = { models: { list } } as unknown as TogetherAIDriver['service'];

        const models = await driver.listModels();

        expect(list).toHaveBeenCalledOnce();
        expect(models).toEqual([
            expect.objectContaining({
                id: 'Qwen/Qwen3.5-9B',
                name: 'Qwen3.5 9B',
                owner: 'Qwen',
                provider: Providers.togetherai,
                type: ModelType.Chat,
                tool_support: true,
            }),
        ]);
    });

    it('keeps direct embedding generation support without listing embedding models', async () => {
        const driver = new TogetherAIDriver({ apiKey: 'test-key' });
        const create = vi.fn(async () => ({
            data: [
                { index: 1, embedding: [0.3, 0.4] },
                { index: 0, embedding: [0.1, 0.2] },
            ],
            usage: { prompt_tokens: 3, total_tokens: 3 },
        }));
        driver.service = {
            embeddings: { create },
        } as unknown as TogetherAIDriver['service'];

        const result = await driver.generateEmbeddings({
            model: 'togethercomputer/m2-bert-80M-8k-retrieval',
            dimensions: 128,
            inputs: [
                { type: 'text', text: 'first' },
                { type: 'text', text: 'second' },
            ],
        });

        expect(create).toHaveBeenCalledWith({
            input: ['first', 'second'],
            model: 'togethercomputer/m2-bert-80M-8k-retrieval',
            dimensions: 128,
            encoding_format: 'float',
        });
        expect(result.results).toEqual([
            { outputs: [{ values: [0.1, 0.2], modality: 'text' }] },
            { outputs: [{ values: [0.3, 0.4], modality: 'text' }] },
        ]);
        expect(result.usage).toEqual({ input_tokens: 3, input_text_tokens: 3 });
    });

    it('keeps provider indexes stable for fragmented parallel streaming tool calls', async () => {
        const driver = new TogetherAIDriver({ apiKey: 'test-key' });
        async function* chunks() {
            const base = { object: 'chat.completion.chunk' as const, created: 1, model: 'test-model' };
            yield {
                ...base,
                id: 'chunk-1',
                choices: [
                    {
                        index: 0,
                        finish_reason: null,
                        delta: {
                            role: 'assistant' as const,
                            tool_calls: [
                                {
                                    index: 0,
                                    id: 'call_a',
                                    type: 'function' as const,
                                    function: { name: 'first_tool', arguments: '{"value":' },
                                },
                            ],
                        },
                    },
                ],
            };
            yield {
                ...base,
                id: 'chunk-2',
                choices: [
                    {
                        index: 0,
                        finish_reason: null,
                        delta: {
                            role: 'assistant' as const,
                            tool_calls: [
                                {
                                    index: 1,
                                    id: 'call_b',
                                    type: 'function' as const,
                                    function: { name: 'second_tool', arguments: '{"value":' },
                                },
                            ],
                        },
                    },
                ],
            };
            yield {
                ...base,
                id: 'chunk-3',
                choices: [
                    {
                        index: 0,
                        finish_reason: null,
                        delta: {
                            role: 'assistant' as const,
                            tool_calls: [
                                {
                                    index: 0,
                                    id: 'call_a',
                                    type: 'function' as const,
                                    function: { name: '', arguments: '1}' },
                                },
                            ],
                        },
                    },
                ],
            };
            yield {
                ...base,
                id: 'chunk-4',
                choices: [
                    {
                        index: 0,
                        finish_reason: 'tool_calls' as const,
                        delta: {
                            role: 'assistant' as const,
                            tool_calls: [
                                {
                                    index: 1,
                                    id: 'call_b',
                                    type: 'function' as const,
                                    function: { name: '', arguments: '2}' },
                                },
                            ],
                        },
                    },
                ],
            };
        }
        driver.service = {
            chat: { completions: { create: vi.fn(async () => chunks()) } },
        } as unknown as TogetherAIDriver['service'];

        const stream = await driver.stream([{ role: PromptRole.user, content: 'Use both tools' }], {
            model: 'test-model',
            tools: [
                { name: 'first_tool', input_schema: { type: 'object' } },
                { name: 'second_tool', input_schema: { type: 'object' } },
            ],
        });
        for await (const _chunk of stream) {
            // Consume the stream to populate the final completion.
        }

        expect(stream.completion?.tool_use).toEqual([
            { id: 'call_a', tool_name: 'first_tool', tool_input: { value: 1 } },
            { id: 'call_b', tool_name: 'second_tool', tool_input: { value: 2 } },
        ]);
    });

    it('classifies Together transport timeouts as retryable llumiverse errors', () => {
        const driver = new TogetherAIDriver({ apiKey: 'test-key' });
        const error = driver.formatLlumiverseError(new APIConnectionTimeoutError({ message: 'timed out' }), {
            provider: driver.provider,
            model: 'test-model',
            operation: 'execute',
        });

        expect(error).toMatchObject({ name: 'APIConnectionTimeoutError', retryable: true });
    });

    it('retains tool text and image references when Together requires string tool content', async () => {
        const driver = new TogetherAIDriver({ apiKey: 'test-key' });
        const create = vi.fn(async (_request: unknown) => ({
            id: 'together-1',
            object: 'chat.completion',
            created: 1,
            model: 'test-model',
            choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }],
        }));
        driver.service = { chat: { completions: { create } } } as unknown as TogetherAIDriver['service'];

        await driver.requestTextCompletion(
            {
                _is_openai_chat_completions: true,
                messages: [
                    {
                        role: 'assistant',
                        content: null,
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
            { model: 'test-model', tools: [{ name: 'lookup', input_schema: { type: 'object' } }] },
        );

        const request = create.mock.calls[0][0] as { messages: unknown[] };
        expect(request.messages[1]).toEqual({
            role: 'tool',
            tool_call_id: 'call_1',
            content: 'result\n[Image: https://example.test/image.png]',
        });
    });
});
