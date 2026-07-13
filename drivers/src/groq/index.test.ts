import { PromptRole } from '@llumiverse/core';
import { APIConnectionTimeoutError } from 'groq-sdk/error';
import { describe, expect, it, vi } from 'vitest';
import { GroqDriver } from './index.js';

function setGroqCreate(driver: GroqDriver, create: ReturnType<typeof vi.fn>): void {
    Object.defineProperty(driver.client.chat.completions, 'create', { value: create });
}

describe('GroqDriver shared Chat Completions transport', () => {
    it('maps Groq request extensions and preserves the native response', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key', endpoint_url: 'https://groq.example.test' });
        const response = {
            id: 'groq-1',
            object: 'chat.completion',
            created: 1,
            model: 'deepseek-r1-distill-llama-70b',
            choices: [
                {
                    index: 0,
                    finish_reason: 'tool_calls',
                    logprobs: null,
                    message: {
                        role: 'assistant',
                        content: null,
                        reasoning: 'reasoning',
                        tool_calls: [
                            {
                                id: 'call_actual',
                                type: 'function',
                                function: { name: 'lookup', arguments: '{"city":"Paris"}' },
                            },
                        ],
                    },
                },
            ],
            usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
        };
        const create = vi.fn(async () => response);
        setGroqCreate(driver, create);
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Weather?' }], {
            model: response.model,
        });

        const completion = await driver.requestTextCompletion(prompt, {
            model: response.model,
            include_original_response: true,
            model_options: {
                _option_id: 'groq-deepseek-thinking',
                max_tokens: 32,
                reasoning_format: 'parsed',
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop_sequence: ['END'],
            },
            tools: [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } }],
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                max_completion_tokens: 32,
                reasoning_format: 'parsed',
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop: ['END'],
                stream: false,
            }),
        );
        expect(completion.tool_use).toEqual([
            { id: 'call_actual', tool_name: 'lookup', tool_input: { city: 'Paris' } },
        ]);
        expect(completion.original_response).toBe(response);
    });

    it('emits fragmented tool calls with the provider ID and x_groq usage', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key' });
        async function* chunks() {
            yield {
                id: 'chunk-1',
                object: 'chat.completion.chunk',
                created: 1,
                model: 'llama',
                choices: [
                    {
                        index: 0,
                        finish_reason: null,
                        delta: {
                            tool_calls: [
                                {
                                    index: 0,
                                    id: 'call_actual',
                                    type: 'function',
                                    function: { name: 'lookup', arguments: '{"city"' },
                                },
                            ],
                        },
                    },
                ],
            };
            yield {
                id: 'chunk-2',
                object: 'chat.completion.chunk',
                created: 1,
                model: 'llama',
                choices: [
                    {
                        index: 0,
                        finish_reason: 'tool_calls',
                        delta: {
                            tool_calls: [
                                {
                                    index: 0,
                                    type: 'function',
                                    function: { name: '', arguments: ':"Paris"}' },
                                },
                            ],
                        },
                    },
                ],
                x_groq: { usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 } },
            };
        }
        setGroqCreate(
            driver,
            vi.fn(async () => chunks()),
        );
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Weather?' }], { model: 'llama' });
        const stream = await driver.requestTextCompletionStream(prompt, {
            model: 'llama',
            model_options: { _option_id: 'text-fallback' },
        });
        const emitted = [];
        for await (const chunk of stream) emitted.push(chunk);

        expect(emitted[0].tool_use?.[0]).toEqual(
            expect.objectContaining({ id: 'tool_0', _actual_id: 'call_actual', tool_name: 'lookup' }),
        );
        expect(emitted.at(-1)?.token_usage).toEqual({ prompt: 2, result: 1, total: 3 });
    });

    it('validates with model listing and reports the missing embedding transport', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key' });
        Object.defineProperty(driver.client.models, 'list', {
            value: vi.fn(async () => ({ data: [{ id: 'llama', owned_by: 'groq' }] })),
        });
        await expect(driver.validateConnection()).resolves.toBe(true);
        await expect(driver.generateEmbeddings({ inputs: [{ type: 'text', text: 'hello' }] })).rejects.toThrow(
            'does not expose an embeddings transport',
        );
    });

    it('classifies Groq transport timeouts as retryable llumiverse errors', () => {
        const driver = new GroqDriver({ apiKey: 'test-key' });
        const error = driver.formatLlumiverseError(new APIConnectionTimeoutError({ message: 'timed out' }), {
            provider: driver.provider,
            model: 'llama',
            operation: 'execute',
        });

        expect(error).toMatchObject({ name: 'APIConnectionTimeoutError', retryable: true });
    });

    it('normalizes shared protocol errors through execute', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key' });
        setGroqCreate(
            driver,
            vi.fn(async () => ({
                id: 'groq-empty',
                object: 'chat.completion',
                created: 1,
                model: 'llama',
                choices: [],
            })),
        );

        await expect(
            driver.execute([{ role: PromptRole.user, content: 'Hello' }], { model: 'llama' }),
        ).rejects.toMatchObject({
            name: 'Error',
            context: { provider: driver.provider, model: 'llama', operation: 'execute' },
            originalError: expect.objectContaining({
                message: 'Chat Completions response is not valid: no data',
            }),
        });
    });

    it('preserves array-shaped tool results at the Groq SDK boundary', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key' });
        const create = vi.fn(async (_request: unknown) => ({
            id: 'groq-1',
            object: 'chat.completion',
            created: 1,
            model: 'llama',
            choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }],
        }));
        setGroqCreate(driver, create);

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
            { model: 'llama', tools: [{ name: 'lookup', input_schema: { type: 'object' } }] },
        );

        const request = create.mock.calls[0][0] as { messages: unknown[] };
        expect(request.messages[1]).toEqual({
            role: 'tool',
            tool_call_id: 'call_1',
            content: [
                { type: 'text', text: 'result' },
                { type: 'image_url', image_url: { url: 'https://example.test/image.png' } },
            ],
        });
    });
});
