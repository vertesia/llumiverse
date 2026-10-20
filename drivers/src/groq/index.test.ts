import { ModelOptionsSchema } from '@llumiverse/common/schemas';
import { PromptRole } from '@llumiverse/core';
import { APIConnectionTimeoutError } from 'groq-sdk/error';
import { describe, expect, it, vi } from 'vitest';
import { GroqDriver } from './index.js';

function setGroqCreate(driver: GroqDriver, create: ReturnType<typeof vi.fn>): void {
    Object.defineProperty(driver.client.chat.completions, 'create', { value: create });
}

describe('GroqDriver shared Chat Completions transport', () => {
    it.each([
        [false, false],
        [false, true],
        [true, false],
        [true, true],
    ])('preserves extensions with streaming=%s and supplied options=%s', async (streaming, supplied) => {
        const extraBody = {
            reasoning_effort: 'low',
            service_tier: 'flex',
            temperature: 0.5,
            reasoning_format: 'hidden',
        };
        const modelOptions = ModelOptionsSchema.parse({
            _option_id: 'openai-text',
            seed: 0,
            ...(supplied ? { effort: 'high', service_tier: 'priority', temperature: 0 } : {}),
            extra_body: extraBody,
        });
        const driver = new GroqDriver({ apiKey: 'test-key' });
        async function* chunks() {
            yield {
                id: 'chunk-1',
                object: 'chat.completion.chunk',
                created: 1,
                model: 'test-model',
                choices: [{ index: 0, finish_reason: 'stop', delta: { content: 'ok' } }],
            };
        }
        const create = vi.fn(async () =>
            streaming
                ? chunks()
                : {
                      id: 'completion-1',
                      object: 'chat.completion',
                      created: 1,
                      model: 'test-model',
                      choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }],
                  },
        );
        setGroqCreate(driver, create);
        const options = { model: 'openai/gpt-oss-120b', model_options: modelOptions };
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], options);
        if (streaming) {
            const result = await driver.requestTextCompletionStream(prompt, options);
            for await (const _chunk of result) {
                /* Consume the mocked provider stream. */
            }
        } else await driver.requestTextCompletion(prompt, options);
        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                reasoning_effort: supplied ? 'high' : 'low',
                service_tier: supplied ? 'priority' : 'flex',
                temperature: supplied ? 0 : 0.5,
                seed: 0,
                stream: streaming,
                reasoning_format: 'hidden',
            }),
        );
        expect(extraBody).toEqual({
            reasoning_effort: 'low',
            service_tier: 'flex',
            temperature: 0.5,
            reasoning_format: 'hidden',
        });
    });

    it.each(['high', 'none', 'max'] as const)(
        'forwards explicit effort %s and seed without local filtering',
        async (effort) => {
            const seed = effort === 'high' ? 42 : effort === 'none' ? 0 : undefined;
            const modelOptions = {
                _option_id: 'openai-text' as const,
                effort,
                seed,
                extra_body: { include_reasoning: false, seed: 27 },
            };
            const driver = new GroqDriver({ apiKey: 'test-key' });
            const create = vi.fn(async (_request: unknown) => ({
                id: 'groq-1',
                object: 'chat.completion',
                created: 1,
                model: 'test-model',
                choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }],
            }));
            setGroqCreate(driver, create);
            await driver.requestTextCompletion(
                { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'hello' }] },
                {
                    model: 'openai/gpt-oss-120b',
                    model_options: modelOptions,
                },
            );
            expect(create).toHaveBeenCalledWith(
                expect.objectContaining({ reasoning_effort: effort, seed: seed ?? 27, include_reasoning: false }),
            );
        },
    );

    it('forwards a longer per-execution timeout to the SDK request', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key' });
        const create = vi.fn(async (_request: unknown, _options?: unknown) => ({
            id: 'groq-timeout',
            object: 'chat.completion',
            created: 1,
            model: 'llama',
            choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'done' } }],
        }));
        setGroqCreate(driver, create);

        await driver.requestTextCompletion(
            { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'hello' }] },
            {
                model: 'llama',
                httpTimeout: { headersTimeout: 1_200_000, bodyTimeout: 1_800_000 },
            },
        );

        expect(create.mock.calls[0][1]).toEqual({ signal: undefined, timeout: 1_800_000 });
    });

    it('uses central capabilities when listing provider models', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key' });
        const list = vi.fn(async () => ({
            data: [
                { id: 'llama-4-scout', owned_by: 'Meta' },
                { id: 'text-embedding-test', owned_by: 'Groq' },
                { id: 'whisper-large-v3', owned_by: 'Groq' },
                { id: 'whispering-chat-v1', owned_by: 'Groq' },
                { id: 'canopylabs/orpheus-v1-english', owned_by: 'Canopy Labs' },
                { id: 'meta-llama/llama-prompt-guard-2-86m', owned_by: 'Meta' },
            ],
        }));
        Object.defineProperty(driver.client.models, 'list', { value: list });

        const models = await driver.listModels();

        expect(models).toHaveLength(3);
        expect(models).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: 'llama-4-scout',
                    input_modalities: ['text', 'image'],
                    output_modalities: ['text'],
                    tool_support: true,
                }),
                expect.objectContaining({ id: 'meta-llama/llama-prompt-guard-2-86m' }),
                expect.objectContaining({ id: 'whispering-chat-v1' }),
            ]),
        );
        expect(models.find((model) => model.id.includes('prompt-guard'))?.tool_support).toBe(false);
    });

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

    describe('structured outputs', () => {
        const resultSchema = {
            type: 'object' as const,
            properties: { color: { type: 'string' as const } },
            required: ['color'],
            additionalProperties: false,
        };

        async function requestFor(model: string, withTools = false) {
            const driver = new GroqDriver({ apiKey: 'test-key' });
            const create = vi.fn(async () => ({
                id: 'groq-so',
                object: 'chat.completion',
                created: 1,
                model,
                choices: [
                    {
                        index: 0,
                        finish_reason: 'stop',
                        logprobs: null,
                        message: { role: 'assistant', content: '{"color":"blue"}' },
                    },
                ],
                usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            }));
            setGroqCreate(driver, create);
            const options = {
                model,
                result_schema: resultSchema,
                ...(withTools && {
                    tools: [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' as const } }],
                }),
            };
            const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Sky?' }], options);
            await driver.requestTextCompletion(prompt, options);
            return (create.mock.calls as unknown[][])[0][0] as { response_format?: unknown; messages: unknown[] };
        }

        it.each(['openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'qwen/qwen3.8-27b', 'qwen/qwen3.9-32b'])(
            'sends a strict json_schema response_format for %s',
            async (model) => {
                const request = await requestFor(model);
                expect(request.response_format).toEqual({
                    type: 'json_schema',
                    json_schema: { name: 'output', strict: true, schema: expect.objectContaining({ type: 'object' }) },
                });
            },
        );

        it('keeps gpt-oss-safeguard in best-effort mode', async () => {
            const request = await requestFor('openai/gpt-oss-safeguard-20b');
            expect(request.response_format).toMatchObject({ type: 'json_schema', json_schema: { strict: false } });
        });

        // qwen3-32b is Qwen 3, not 3.32: the dash starts the size.
        it.each(['qwen/qwen3.6-27b', 'qwen/qwen3-32b', 'meta-llama/llama-4-scout-17b-16e-instruct'])(
            'leaves %s on prompt instructions',
            async (model) => {
                const request = await requestFor(model);
                expect(request.response_format).toBeUndefined();
            },
        );

        it('omits response_format when tools are offered, which Groq rejects', async () => {
            const request = await requestFor('openai/gpt-oss-20b', true);
            expect(request.response_format).toBeUndefined();
        });

        it('keeps the schema instruction in the prompt alongside response_format', async () => {
            const request = await requestFor('openai/gpt-oss-20b');
            expect(JSON.stringify(request.messages)).toContain('<response_schema>');
        });
    });

    it('reports the prompt tokens served from the Groq prompt cache', async () => {
        const driver = new GroqDriver({ apiKey: 'test-key', endpoint_url: 'https://groq.example.test' });
        const create = vi.fn(async () => ({
            id: 'groq-2',
            object: 'chat.completion',
            created: 1,
            model: 'moonshotai/kimi-k2-instruct',
            choices: [
                { index: 0, finish_reason: 'stop', logprobs: null, message: { role: 'assistant', content: 'ok' } },
            ],
            usage: {
                prompt_tokens: 1000,
                completion_tokens: 5,
                total_tokens: 1005,
                prompt_tokens_details: { cached_tokens: 800 },
            },
        }));
        setGroqCreate(driver, create);
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hi' }], {
            model: 'moonshotai/kimi-k2-instruct',
        });

        const completion = await driver.requestTextCompletion(prompt, { model: 'moonshotai/kimi-k2-instruct' });

        expect(completion.token_usage).toEqual({
            prompt: 1000,
            prompt_cached: 800,
            prompt_new: 200,
            result: 5,
            total: 1005,
        });
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
        expect(emitted.at(-1)?.token_usage).toEqual({ prompt: 2, prompt_new: 2, result: 1, total: 3 });
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

    it.each([undefined, 'auto', 'low', 'high', 'original'] as const)(
        'preserves user image detail %s at the Groq SDK boundary',
        async (detail) => {
            const driver = new GroqDriver({ apiKey: 'test-key' });
            const create = vi.fn(async (_request: unknown) => ({
                id: 'groq-1',
                object: 'chat.completion',
                created: 1,
                model: 'llama',
                choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }],
            }));
            setGroqCreate(driver, create);
            const content = [
                { type: 'text' as const, text: 'Describe this image' },
                {
                    type: 'image_url' as const,
                    image_url: { url: 'https://example.test/image.png', ...(detail === undefined ? {} : { detail }) },
                },
            ];

            await driver.requestTextCompletion(
                { _is_openai_chat_completions: true, messages: [{ role: 'user', content }] },
                { model: 'llama' },
            );

            expect(create).toHaveBeenCalledWith(expect.objectContaining({ messages: [{ role: 'user', content }] }));
        },
    );

    it('preserves tool image attachments at the Groq SDK boundary', async () => {
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
                            {
                                type: 'image_url',
                                image_url: { url: 'https://example.test/image.png', detail: 'original' },
                            },
                        ],
                    },
                ],
            },
            { model: 'llama', tools: [{ name: 'lookup', input_schema: { type: 'object' } }] },
        );

        const request = create.mock.calls[0][0] as { messages: unknown[] };
        expect(request.messages.slice(1)).toEqual([
            {
                role: 'tool',
                tool_call_id: 'call_1',
                content: 'result\n[Image 1 attached below]',
            },
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'Image 1 from tool result call_1:' },
                    { type: 'image_url', image_url: { url: 'https://example.test/image.png', detail: 'original' } },
                ],
            },
        ]);
    });
});
