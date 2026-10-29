import { ModelOptionsSchema } from '@llumiverse/common/schemas';
import { Base64DataSource, PromptRole, Providers } from '@llumiverse/core';
import OpenAI from 'openai';
import { APIError } from 'openai/error';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';

class TestResponsesDriver extends OpenAIResponsesDriverBase {
    provider: OpenAIResponsesDriverBase['provider'];
    service: OpenAI;

    constructor(
        create: (request: unknown) => Promise<unknown>,
        provider: OpenAIResponsesDriverBase['provider'] = Providers.openai,
    ) {
        super({});
        this.provider = provider;
        this.service = { responses: { create } } as unknown as OpenAI;
    }
}

const reasoningItem = {
    id: 'reason-1',
    type: 'reasoning' as const,
    summary: [{ type: 'summary_text' as const, text: 'visible plan' }],
    encrypted_content: 'encrypted-replay-state',
    status: 'completed' as const,
};
const messageItem = {
    id: 'msg-1',
    type: 'message' as const,
    role: 'assistant' as const,
    status: 'completed' as const,
    content: [{ type: 'output_text' as const, text: 'answer', annotations: [], logprobs: [] }],
};

function response() {
    return {
        id: 'response-1',
        object: 'response',
        created_at: 1,
        model: 'gpt-5',
        service_tier: 'priority',
        status: 'completed',
        output: [reasoningItem, messageItem],
        output_text: 'answer',
        parallel_tool_calls: true,
        tool_choice: 'auto',
        tools: [],
        error: null,
        incomplete_details: null,
        instructions: null,
        metadata: null,
        temperature: null,
        top_p: null,
        usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3, input_tokens_details: { cached_tokens: 0 } },
    } as unknown as OpenAI.Responses.Response;
}

describe('OpenAI Responses reasoning', () => {
    it('uses prompt-schema fallback for GLM 5.3 on OpenAI-compatible Responses', async () => {
        const create = vi.fn(async () => ({
            ...response(),
            model: 'z-ai/glm-5.3',
            output: [
                {
                    ...messageItem,
                    content: [{ type: 'output_text', text: '{"value":"ok"}', annotations: [], logprobs: [] }],
                },
            ],
        }));
        const driver = new TestResponsesDriver(create, Providers.openai_compatible);
        const options = {
            model: 'z-ai/glm-5.3',
            result_schema: {
                type: 'object' as const,
                properties: { value: { type: 'string' as const } },
                required: ['value'],
                additionalProperties: false,
            },
        };
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Return the value.' }], options);

        await driver.requestTextCompletion(prompt, options);

        expect(JSON.stringify(prompt)).toContain('<response_schema>');
        expect(create).toHaveBeenCalledWith(expect.not.objectContaining({ text: expect.anything() }));
    });

    it.each(
        ['gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-sol', 'gpt-6.1-terra'].flatMap((model) =>
            [false, true].map((streaming) => ({ model, streaming })),
        ),
    )(
        'uses Responses with native schemas and reasoning replay for $model when streaming=$streaming',
        async ({ model, streaming }) => {
            const toolCall: OpenAI.Responses.ResponseFunctionToolCall = {
                type: 'function_call',
                call_id: 'call-1',
                name: 'lookup',
                arguments: '{"city":"Paris"}',
            };
            const final = { ...response(), model, output: [reasoningItem, toolCall] };
            const create = vi.fn(async (request: unknown) =>
                (request as { stream?: boolean }).stream
                    ? (async function* () {
                          yield { type: 'response.completed', sequence_number: 1, response: final };
                      })()
                    : final,
            );
            const driver = new TestResponsesDriver(create);
            const options = {
                model,
                model_options: ModelOptionsSchema.parse({
                    _option_id: 'openai-thinking',
                    effort: 'max',
                    verbosity: 'low',
                    include_thoughts: false,
                    max_tokens: 128_000,
                    service_tier: 'fast',
                }),
                tools: [
                    {
                        name: 'lookup',
                        description: 'Look up a city.',
                        input_schema: {
                            type: 'object',
                            properties: { city: { type: 'string' } },
                            required: ['city'],
                            additionalProperties: false,
                        },
                    },
                ],
                result_schema: {
                    type: 'object' as const,
                    properties: { answer: { type: 'string' as const } },
                    required: ['answer'],
                    additionalProperties: false,
                },
            };
            const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Look up Paris.' }], options);
            expect(JSON.stringify(prompt)).not.toContain('<response_schema>');
            let conversation: unknown;
            if (streaming) {
                const stream = await driver.requestTextCompletionStream(prompt, options);
                for await (const chunk of stream) expect(chunk.result).toEqual([]);
                conversation = await stream.finalizeConversation?.();
            } else {
                const completion = await driver.requestTextCompletion(prompt, options);
                expect(completion.result).toEqual([]);
                expect(completion.tool_use).toEqual([
                    { id: 'call-1', tool_name: 'lookup', tool_input: { city: 'Paris' } },
                ]);
                conversation = completion.conversation;
            }
            expect(create.mock.calls[0][0]).toMatchObject({
                model,
                reasoning: { effort: 'max', summary: 'auto' },
                include: ['reasoning.encrypted_content'],
                max_output_tokens: 128_000,
                service_tier: 'fast',
                tools: [expect.objectContaining({ type: 'function', name: 'lookup', strict: true })],
                text: { verbosity: 'low', format: expect.objectContaining({ type: 'json_schema', strict: true }) },
            });
            await driver.requestTextCompletion(
                [{ type: 'function_call_output', call_id: 'call-1', output: 'Paris is in France.' }],
                {
                    ...options,
                    conversation: JSON.parse(JSON.stringify(conversation)),
                },
            );
            expect(create.mock.calls[1][0]).toMatchObject({ input: expect.arrayContaining([reasoningItem, toolCall]) });
        },
    );

    it.each(['error', 'response.error', 'response.failed'] as const)(
        'propagates a streamed %s with provider error details',
        async (type) => {
            const error = { code: 'server_error' as const, message: 'Provider failed after generation started.' };
            const event =
                type === 'response.failed'
                    ? { type, sequence_number: 2, response: { ...response(), status: 'failed', error } }
                    : { type, sequence_number: 2, ...error, param: null };
            const create = vi.fn(async () =>
                (async function* () {
                    yield { type: 'response.output_text.delta', delta: 'partial', sequence_number: 1 };
                    yield event;
                })(),
            );
            const driver = new TestResponsesDriver(create);
            const stream = await driver.requestTextCompletionStream([], { model: 'gpt-6.1-terra' });
            const consume = async () => {
                for await (const _chunk of stream) {
                    /* Consume stream. */
                }
            };
            const failure = await consume().catch((err: unknown) => err);
            if (type !== 'response.failed') expect(failure).toBeInstanceOf(APIError);
            expect(failure).toMatchObject({ ...error, message: expect.stringContaining(error.message) });
            expect(
                driver.formatLlumiverseError(failure, {
                    provider: Providers.openai,
                    model: 'gpt-6.1-terra',
                    operation: 'stream',
                }).retryable,
            ).toBe(true);
        },
    );

    it('rejects failed blocking responses even if they contain partial output', async () => {
        const error = { code: 'server_error' as const, message: 'Provider failed.' };
        const driver = new TestResponsesDriver(vi.fn(async () => ({ ...response(), status: 'failed', error })));
        await expect(driver.requestTextCompletion([], { model: 'gpt-6-sol' })).rejects.toMatchObject({
            ...error,
            message: expect.stringContaining(error.message),
        });
    });

    it.each(
        (
            [
                Providers.openai,
                Providers.azure_openai,
                Providers.xai,
                Providers.bedrock_mantle,
                Providers.openai_compatible,
            ] as const
        )
            .flatMap((provider) => [
                { provider, code: 'server_error' as const, retryable: true },
                { provider, code: 'insufficient_quota' as const, retryable: false },
            ])
            .flatMap((scenario) => [false, true].map((streaming) => ({ ...scenario, streaming }))),
    )(
        'classifies $provider/$code through the SDK when streaming=$streaming',
        async ({ provider, code, retryable, streaming }) => {
            const error = { code, message: 'Provider failure after partial output.' };
            const failed = { ...response(), status: 'failed', error };
            const events = [
                { type: 'response.output_text.delta', sequence_number: 1, delta: 'partial' },
                { type: 'response.failed', sequence_number: 2, response: failed },
            ];
            const fetchMock = vi.fn(
                async () =>
                    new Response(
                        streaming
                            ? events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join('')
                            : JSON.stringify(failed),
                        { headers: { 'content-type': streaming ? 'text/event-stream' : 'application/json' } },
                    ),
            );
            const driver = new TestResponsesDriver(async () => failed, provider);
            driver.service = new OpenAI({ apiKey: 'test-key', fetch: fetchMock, maxRetries: 0 });
            const options = { model: 'gpt-6.1-terra', model_options: { include_thoughts: false } };
            const prompt = [{ role: PromptRole.user, content: 'question' }];
            const expected = {
                retryable,
                context: { provider, model: options.model, operation: streaming ? 'stream' : 'execute' },
                originalError: expect.objectContaining({ ...error, message: expect.stringContaining(error.message) }),
            };
            if (streaming) {
                const stream = await driver.stream(prompt, options);
                const iterator = stream[Symbol.asyncIterator]();
                await expect(iterator.next()).resolves.toMatchObject({ done: false, value: 'partial' });
                const consume = async () => {
                    while (!(await iterator.next()).done) {
                        /* Consume terminal usage before the provider failure is raised. */
                    }
                };
                await expect(consume()).rejects.toMatchObject(expected);
                // Retain failed-turn usage without exposing a successful result or replayable conversation.
                expect(stream.completion).toMatchObject({
                    finish_reason: 'failed',
                    token_usage: { prompt: 1, result: 2, total: 3 },
                });
                expect(stream.completion?.conversation).toBeUndefined();
            } else {
                await expect(driver.execute(prompt, options)).rejects.toMatchObject(expected);
            }
            expect(fetchMock).toHaveBeenCalledOnce();
            driver.destroy();
        },
    );

    it.each(['minimax/minimax-m3', 'minimax/minimax-m4'])(
        'supplements native structured output for %s on OpenAI-compatible Responses',
        async (model) => {
            const create = vi.fn(async (_request: unknown) => response());
            const driver = new TestResponsesDriver(create, Providers.openai_compatible);
            const options = {
                model,
                result_schema: { type: 'object' as const, properties: { value: { type: 'string' as const } } },
            };
            const prompt = await driver.createPrompt(
                [{ role: PromptRole.user, content: 'Return the value.' }],
                options,
            );

            await driver.requestTextCompletion(prompt, options);

            expect(JSON.stringify(prompt)).toContain('<response_schema>');
            expect(create).toHaveBeenCalledWith(
                expect.objectContaining({
                    text: expect.objectContaining({ format: expect.objectContaining({ type: 'json_schema' }) }),
                }),
            );
        },
    );

    it('keeps native structured output for MiniMax on OpenAI', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create, Providers.openai);
        const options = {
            model: 'minimax/minimax-m3',
            result_schema: { type: 'object' as const, properties: { value: { type: 'string' as const } } },
        };
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Return the value.' }], options);

        await driver.requestTextCompletion(prompt, options);

        expect(JSON.stringify(prompt)).not.toContain('<response_schema>');
        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                text: expect.objectContaining({ format: expect.objectContaining({ type: 'json_schema' }) }),
            }),
        );
    });

    it.each([
        ['required', 'required'],
        ['any', 'required'],
    ] as const)('forwards explicit %s tool choice as %s', async (configured, expected) => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-5.6-sol',
            model_options: { _option_id: 'openai-thinking', tool_choice: configured },
            tools: [{ name: 'think', description: 'Think', input_schema: { type: 'object' } }],
        });

        expect(create).toHaveBeenCalledWith(expect.objectContaining({ tool_choice: expected }));
    });

    it('forces one named tool without changing the visible tool definitions', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-5.6-sol',
            model_options: {
                _option_id: 'openai-thinking',
                tool_choice: 'required',
                required_tool_name: 'write_artifact',
                parallel_tool_calls: false,
            } as Parameters<typeof driver.requestTextCompletion>[1]['model_options'] & {
                required_tool_name: string;
                parallel_tool_calls: false;
            },
            tools: [
                { name: 'read_artifact', description: 'Read', input_schema: { type: 'object' } },
                { name: 'write_artifact', description: 'Write', input_schema: { type: 'object' } },
            ],
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                tools: expect.arrayContaining([
                    expect.objectContaining({ name: 'read_artifact' }),
                    expect.objectContaining({ name: 'write_artifact' }),
                ]),
                tool_choice: { type: 'function', name: 'write_artifact' },
                parallel_tool_calls: false,
            }),
        );
    });

    it('rejects a required tool choice when no tools are available', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await expect(
            driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
                model: 'gpt-5.6-sol',
                model_options: { _option_id: 'openai-thinking', tool_choice: 'required' },
                tools: [],
            }),
        ).rejects.toMatchObject({
            name: 'ToolChoiceConfigurationError',
            retryable: false,
            code: 400,
            message: expect.stringContaining('required tool choice was requested, but no tools are available'),
        });
        expect(create).not.toHaveBeenCalled();
    });

    it.each(
        ([Providers.openai, Providers.azure_openai, Providers.openai_compatible] as const).flatMap((provider) =>
            [42, 0, undefined].flatMap((seed) => [false, true].map((streaming) => ({ provider, seed, streaming }))),
        ),
    )('preserves seed=$seed for $provider when streaming=$streaming', async ({ provider, seed, streaming }) => {
        const create = vi.fn(async (request: unknown) =>
            (request as { stream?: boolean }).stream
                ? (async function* () {
                      yield { type: 'response.completed', sequence_number: 1, response: response() };
                  })()
                : response(),
        );
        const driver = new TestResponsesDriver(create, provider);
        const warn = vi.fn();
        driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
        const model_options = ModelOptionsSchema.parse({ _option_id: 'openai-text', seed });
        const original = structuredClone(model_options);
        const options = { model: 'gpt-4o', model_options };
        const prompt = [{ type: 'message' as const, role: 'user' as const, content: 'question' }];
        if (streaming) {
            for await (const _chunk of await driver.requestTextCompletionStream(prompt, options)) {
                /* Consume stream. */
            }
        } else await driver.requestTextCompletion(prompt, options);
        const request = create.mock.calls[0][0];
        if (seed === undefined) expect(request).not.toHaveProperty('seed');
        else expect(request).toHaveProperty('seed', seed);
        expect(model_options).toEqual(original);
        expect(warn).not.toHaveBeenCalled();
    });

    it('does not mistake an embedded o-series substring for an OpenAI reasoning model', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create, Providers.openai_compatible);
        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'custom-o1-compatible',
            model_options: { _option_id: 'openai-text', temperature: 0 },
        });
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ temperature: 0, reasoning: undefined }));
    });

    it.each(
        ([Providers.openai, Providers.azure_openai, Providers.openai_compatible] as const).flatMap((provider) =>
            ['gpt-4o', 'gpt-4o-mini', 'gpt-5.4-mini'].flatMap((model) =>
                [false, true].map((streaming) => ({ provider, model, streaming })),
            ),
        ),
    )(
        'preserves saved Chat-option compatibility for $provider/$model when streaming=$streaming',
        async ({ provider, model, streaming }) => {
            const create = vi.fn(async (request: unknown) =>
                (request as { stream?: boolean }).stream
                    ? (async function* () {
                          yield { type: 'response.completed', sequence_number: 1, response: response() };
                      })()
                    : response(),
            );
            const driver = new TestResponsesDriver(create, provider);
            const warn = vi.fn();
            driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
            const model_options = {
                _option_id: 'openai-text' as const,
                stop_sequence: ['END'],
                presence_penalty: 0,
                frequency_penalty: 0.3,
                max_tokens: 2048,
            };
            const original = structuredClone(model_options);
            const options = { model, model_options };
            const prompt = [{ type: 'message' as const, role: 'user' as const, content: 'question' }];
            if (streaming) {
                for await (const _chunk of await driver.requestTextCompletionStream(prompt, options)) {
                    /* Consume stream. */
                }
            } else {
                await driver.requestTextCompletion(prompt, options);
            }
            const request = create.mock.calls[0][0];
            expect(request).not.toHaveProperty('stop');
            expect(request).not.toHaveProperty('presence_penalty');
            expect(request).not.toHaveProperty('frequency_penalty');
            expect(request).toMatchObject({ max_output_tokens: 2048 });
            expect(model_options).toEqual(original);
            expect(warn).toHaveBeenCalledExactlyOnceWith(
                {
                    model,
                    option_names: ['stop_sequence', 'presence_penalty', 'frequency_penalty'],
                    reason: 'openai_responses_chat_options',
                },
                'Model option compatibility exception changed caller input',
            );
        },
    );

    it('leaves omitted Chat controls absent without warning', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);
        const warn = vi.fn();
        driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-4o',
        });
        const request = create.mock.calls[0][0];
        expect(request).not.toHaveProperty('stop');
        expect(request).not.toHaveProperty('presence_penalty');
        expect(request).not.toHaveProperty('frequency_penalty');
        expect(request).toMatchObject({ max_output_tokens: undefined });
        expect(warn).not.toHaveBeenCalled();
    });

    it.each([false, true])(
        'logs input precedence and cache compatibility exceptions when stream=%s',
        async (streaming) => {
            const create = vi.fn(async (request: unknown) =>
                (request as { stream?: boolean }).stream
                    ? (async function* () {
                          yield { type: 'response.completed', sequence_number: 1, response: response() };
                      })()
                    : response(),
            );
            const driver = new TestResponsesDriver(create);
            const warn = vi.fn();
            driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
            const options = {
                model: 'gpt-5.6-sol',
                model_options: {
                    _option_id: 'openai-thinking' as const,
                    effort: 'low' as const,
                    reasoning_effort: 'high' as const,
                    prompt_cache_retention: '24h' as const,
                    extra_body: { model: 'override' },
                },
            };
            const prompt = [{ type: 'message' as const, role: 'user' as const, content: 'question' }];
            if (streaming) {
                for await (const _chunk of await driver.requestTextCompletionStream(prompt, options)) {
                    /* Consume stream. */
                }
            } else {
                await driver.requestTextCompletion(prompt, options);
            }
            expect(warn.mock.calls.map(([fields]) => fields)).toEqual([
                { model: options.model, option_names: ['reasoning_effort'], reason: 'openai_effort_alias_precedence' },
                { model: options.model, option_names: ['prompt_cache_retention'], reason: 'openai_cache_retention' },
                { model: options.model, option_names: ['model'], reason: 'openai_extra_body_precedence' },
            ]);
        },
    );

    it('warns about existing reasoning-model sampling omissions', async () => {
        const driver = new TestResponsesDriver(vi.fn(async () => response()));
        const warn = vi.fn();
        driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
        const model_options = { temperature: 0, top_p: 0.8 };
        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-5',
            model_options,
        });
        expect(warn).toHaveBeenCalledExactlyOnceWith(
            { model: 'gpt-5', option_names: ['temperature', 'top_p'], reason: 'openai_reasoning_sampling' },
            'Model option compatibility exception changed caller input',
        );
    });

    it('returns the processing tier reported by OpenAI', async () => {
        const driver = new TestResponsesDriver(vi.fn(async () => response()));

        const completion = await driver.requestTextCompletion(
            [{ type: 'message', role: 'user', content: 'question' }],
            { model: 'gpt-5' },
        );

        expect(completion.service_tier).toBe('priority');
    });

    it('forwards a longer per-execution timeout to the SDK request', async () => {
        const create = vi.fn(async (_request: unknown, _options?: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-5',
            httpTimeout: { headersTimeout: 1_200_000, bodyTimeout: 1_800_000 },
        });

        expect(create.mock.calls[0][1]).toEqual({ signal: undefined, timeout: 1_800_000 });
    });

    it.each([
        ['effort', { effort: 'high' as const }],
        ['reasoning_effort', { reasoning_effort: 'high' as const }],
    ])('passes explicit %s through an OpenAI-compatible endpoint', async (_name, effortOption) => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create, Providers.openai_compatible);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'custom-reasoning-model',
            model_options: { _option_id: 'openai-text', ...effortOption, temperature: 0.7 },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({ reasoning: { effort: 'high', summary: 'auto' }, temperature: 0.7 }),
        );
    });

    it.each([
        ['gpt-6-astra', 'none'],
        ['gpt-6-sol', 'minimal'],
    ] as const)('passes effort %s through unchanged for %s', async (model, effort) => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model,
            model_options: { _option_id: 'openai-thinking', effort },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({ reasoning: expect.objectContaining({ effort, summary: 'auto' }) }),
        );
    });

    it('merges provider-specific extra body fields without allowing core request overrides', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create, Providers.openai_compatible);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'openrouter/model',
            model_options: {
                _option_id: 'openai-text',
                max_tokens: 256,
                extra_body: {
                    provider: { sort: 'throughput', allow_fallbacks: false },
                    baseten: { performance: 'max' },
                    model: 'must-not-override',
                    stream: true,
                },
            },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'openrouter/model',
                stream: false,
                max_output_tokens: 256,
                provider: { sort: 'throughput', allow_fallbacks: false },
                baseten: { performance: 'max' },
            }),
        );
        expect(create.mock.calls[0][0]).not.toHaveProperty('extra_body');
    });

    it.each(['gpt-5.4', 'gpt-5.5'])('uses the model default reasoning context for %s', async (model) => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model,
            model_options: { _option_id: 'openai-thinking' },
        });

        expect(create.mock.calls[0][0]).toMatchObject({ reasoning: { summary: 'auto' } });
        expect((create.mock.calls[0][0] as { reasoning: Record<string, unknown> }).reasoning).not.toHaveProperty(
            'context',
        );
    });

    it.each(['gpt-5.6', 'gpt-5.6-sol', 'gpt-5.7', 'gpt-6-astra'])(
        'leaves persisted reasoning context at the API default for %s',
        async (model) => {
            const create = vi.fn(async (_request: unknown) => response());
            const driver = new TestResponsesDriver(create);

            await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
                model,
                model_options: { _option_id: 'openai-thinking' },
            });

            const reasoning = (create.mock.calls[0][0] as { reasoning: Record<string, unknown> }).reasoning;
            expect(reasoning).not.toHaveProperty('context');
        },
    );

    it.each(['current_turn', 'all_turns', 'auto'] as const)(
        'passes an explicit reasoning_context option through for supported models: %s',
        async (reasoning_context) => {
            const create = vi.fn(async (_request: unknown) => response());
            const driver = new TestResponsesDriver(create);

            await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
                model: 'gpt-6-astra',
                model_options: { _option_id: 'openai-thinking', reasoning_context },
            });

            expect(create.mock.calls[0][0]).toMatchObject({ reasoning: { context: reasoning_context } });
        },
    );

    it('passes an explicit reasoning_context option through for streaming requests', async () => {
        const create = vi.fn(async (_request: unknown) =>
            (async function* () {
                yield { type: 'response.completed', sequence_number: 1, response: response() };
            })(),
        );
        const driver = new TestResponsesDriver(create);

        const stream = await driver.requestTextCompletionStream(
            [{ type: 'message', role: 'user', content: 'question' }],
            {
                model: 'gpt-5.6',
                model_options: { _option_id: 'openai-thinking', reasoning_context: 'current_turn' },
            },
        );
        for await (const _chunk of stream) {
            // Consume the provider stream.
        }

        expect(create.mock.calls[0][0]).toMatchObject({ reasoning: { context: 'current_turn' } });
    });

    it('passes explicit reasoning_context through for provider validation', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-5.5',
            model_options: { _option_id: 'openai-thinking', reasoning_context: 'all_turns' },
        });
        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({ reasoning: { context: 'all_turns', summary: 'auto' } }),
        );
    });

    it('does not request cross-turn reasoning controls for models without documented support', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking' },
        });

        expect(create.mock.calls[0][0]).toMatchObject({ reasoning: { summary: 'auto' } });
        expect((create.mock.calls[0][0] as { reasoning: Record<string, unknown> }).reasoning).not.toHaveProperty(
            'context',
        );
    });

    it('projects reasoning by default and replays the exact encrypted output item after JSON roundtrip', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);
        const prompt = [{ type: 'message', role: 'user', content: 'question' }] as OpenAI.Responses.ResponseInputItem[];

        const first = await driver.requestTextCompletion(prompt, {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking' },
        });
        expect(first.result).toEqual([
            { type: 'thoughts', value: 'visible plan' },
            { type: 'text', value: 'answer' },
        ]);
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ include: ['reasoning.encrypted_content'] }));

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'continue' }], {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking' },
            conversation: persisted,
        });
        expect(create.mock.calls[1][0]).toMatchObject({ input: expect.arrayContaining([reasoningItem]) });

        const hidden = await driver.requestTextCompletion(prompt, {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking', include_thoughts: false },
        });
        expect(hidden.result).toEqual([{ type: 'text', value: 'answer' }]);
        expect(JSON.stringify(hidden.conversation)).toContain('encrypted-replay-state');
    });

    it('streams reasoning separately and finalizes from the authoritative response output', async () => {
        const final = response();
        const create = vi.fn(async () =>
            (async function* () {
                yield {
                    type: 'response.reasoning_summary_text.delta',
                    item_id: 'reason-1',
                    output_index: 0,
                    summary_index: 0,
                    sequence_number: 1,
                    delta: 'visible plan',
                };
                yield {
                    type: 'response.output_text.delta',
                    item_id: 'msg-1',
                    output_index: 1,
                    content_index: 0,
                    sequence_number: 2,
                    delta: 'answer',
                    logprobs: [],
                };
                yield { type: 'response.completed', sequence_number: 3, response: final };
            })(),
        );
        const driver = new TestResponsesDriver(create);
        const stream = await driver.requestTextCompletionStream(
            [{ type: 'message', role: 'user', content: 'question' }],
            { model: 'gpt-5', model_options: { _option_id: 'openai-thinking' } },
        );
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.();

        expect(results).toEqual([
            { type: 'thoughts', value: 'visible plan' },
            { type: 'text', value: 'answer' },
        ]);
        expect(JSON.stringify(conversation)).toContain('encrypted-replay-state');
    });

    it('prunes adjacent conversation content while preserving encrypted reasoning items', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);
        const prompt = [
            { type: 'message' as const, role: 'user' as const, content: 'old tool output that should be truncated' },
        ] as OpenAI.Responses.ResponseInputItem[];

        const completion = await driver.requestTextCompletion(prompt, {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking' },
            stripImagesAfterTurns: 0,
            stripTextMaxTokens: 1,
        });

        const serialized = JSON.stringify(completion.conversation);
        expect(serialized).toContain('encrypted-replay-state');
        expect(serialized).toContain('[Content truncated - exceeded token limit]');

        const imageCompletion = await driver.requestTextCompletion(
            [
                {
                    type: 'message',
                    role: 'user',
                    content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,aW1hZ2U=' } }],
                } as unknown as OpenAI.Responses.ResponseInputItem,
            ],
            {
                model: 'gpt-5',
                model_options: { _option_id: 'openai-thinking' },
                stripImagesAfterTurns: 0,
            },
        );
        expect(JSON.stringify(imageCompletion.conversation)).toContain('[Image removed from conversation history]');

        const heartbeatCompletion = await driver.requestTextCompletion(
            [{ type: 'message', role: 'user', content: '<heartbeat>old status</heartbeat>' }],
            {
                model: 'gpt-5',
                model_options: { _option_id: 'openai-thinking' },
                stripHeartbeatsAfterTurns: 0,
            },
        );
        expect(JSON.stringify(heartbeatCompletion.conversation)).toContain(
            '[Heartbeat removed from conversation history]',
        );
    });

    it.each([false, true])('forwards OpenAI prompt cache controls when stream=%s', async (streaming) => {
        const create = vi.fn(async (request: unknown) =>
            (request as { stream?: boolean }).stream
                ? (async function* () {
                      yield { type: 'response.completed', sequence_number: 1, response: response() };
                  })()
                : response(),
        );
        const driver = new TestResponsesDriver(create);
        const options = {
            model: 'gpt-5',
            model_options: {
                _option_id: 'openai-thinking' as const,
                prompt_cache_key: 'agent-cache-key',
                prompt_cache_retention: '24h' as const,
            },
        };

        if (streaming) {
            const stream = await driver.requestTextCompletionStream(
                [{ type: 'message', role: 'user', content: 'question' }],
                options,
            );
            for await (const _chunk of stream) {
                // Consume the provider stream.
            }
        } else {
            await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], options);
        }

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                prompt_cache_key: 'agent-cache-key',
                prompt_cache_retention: '24h',
            }),
        );
    });

    it.each([false, true])('preserves GPT-5.6+ cache controls when stream=%s', async (streaming) => {
        const create = vi.fn(async (request: unknown) =>
            (request as { stream?: boolean }).stream
                ? (async function* () {
                      yield { type: 'response.completed', sequence_number: 1, response: response() };
                  })()
                : response(),
        );
        const driver = new TestResponsesDriver(create);
        const options = {
            model: 'gpt-6-astra',
            model_options: {
                _option_id: 'openai-thinking' as const,
                prompt_cache_retention: '24h' as const,
            },
        };

        if (streaming) {
            const stream = await driver.requestTextCompletionStream(
                [{ type: 'message', role: 'user', content: 'question' }],
                options,
            );
            for await (const _chunk of stream) {
                // Consume the provider stream.
            }
        } else {
            await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], options);
        }

        const request = create.mock.calls[0][0] as Record<string, unknown>;
        expect(request.prompt_cache_options).toEqual({ ttl: '30m' });
        expect(request.prompt_cache_retention).toBe('24h');
    });

    it.each([false, true])('passes explicit in-memory cache retention through when stream=%s', async (streaming) => {
        const create = vi.fn(async (request: unknown) =>
            (request as { stream?: boolean }).stream
                ? (async function* () {
                      yield { type: 'response.completed', sequence_number: 1, response: response() };
                  })()
                : response(),
        );
        const driver = new TestResponsesDriver(create);
        for (const model of ['gpt-5.5', 'gpt-5.6', 'gpt-6-astra']) {
            const options = {
                model,
                model_options: {
                    _option_id: 'openai-thinking' as const,
                    prompt_cache_retention: 'in_memory' as const,
                },
            };

            const request = async () => {
                if (streaming) {
                    const stream = await driver.requestTextCompletionStream(
                        [{ type: 'message', role: 'user', content: 'question' }],
                        options,
                    );
                    for await (const _chunk of stream) {
                        // Consume the provider stream.
                    }
                } else {
                    await driver.requestTextCompletion(
                        [{ type: 'message', role: 'user', content: 'question' }],
                        options,
                    );
                }
            };

            await request();
            expect(create).toHaveBeenLastCalledWith(expect.objectContaining({ prompt_cache_retention: 'in_memory' }));
        }
        expect(create).toHaveBeenCalledTimes(3);
    });

    it.each([false, true])('forwards the Flex service tier when stream=%s', async (streaming) => {
        const create = vi.fn(async (request: unknown) =>
            (request as { stream?: boolean }).stream
                ? (async function* () {
                      yield { type: 'response.completed', sequence_number: 1, response: response() };
                  })()
                : response(),
        );
        const driver = new TestResponsesDriver(create);
        const options = {
            model: 'gpt-5.6-sol',
            model_options: { _option_id: 'openai-thinking' as const, service_tier: 'flex' },
        };

        if (streaming) {
            const stream = await driver.requestTextCompletionStream(
                [{ type: 'message', role: 'user', content: 'question' }],
                options,
            );
            for await (const _chunk of stream) {
                // Consume the provider stream.
            }
        } else {
            await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], options);
        }

        expect(create).toHaveBeenCalledWith(expect.objectContaining({ service_tier: 'flex' }));
    });

    it.each([Providers.openai, Providers.azure_openai] as const)(
        'forwards service tier names without a driver allowlist for %s',
        async (provider) => {
            const create = vi.fn(async (_request: unknown) => response());
            const driver = new TestResponsesDriver(create, provider);

            await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
                model: 'gpt-5.6-sol',
                model_options: { _option_id: 'openai-thinking', service_tier: 'future-tier' },
            });

            expect(create).toHaveBeenCalledWith(expect.objectContaining({ service_tier: 'future-tier' }));
        },
    );
});

describe('Responses image generation', () => {
    const image = {
        id: 'img-1',
        type: 'image_generation_call' as const,
        status: 'completed' as const,
        result: 'YWJj',
        output_format: 'webp' as const,
    };
    it('combines native image and function tools and retains follow-up state and file IDs', async () => {
        const create = vi.fn(async (_request: unknown) => ({
            ...response(),
            output: [
                messageItem,
                image,
                { type: 'function_call', id: 'fn-1', call_id: 'call-1', name: 'lookup', arguments: '{}' },
            ],
        }));
        const driver = new TestResponsesDriver(create);
        const config = {
            _option_id: 'openai-thinking' as const,
            image_generation: { model: 'gpt-image-2.5-flare', force: true },
        };
        const first = await driver.requestTextCompletion(
            [
                {
                    role: 'user',
                    content: [
                        { type: 'input_text', text: 'Edit this' },
                        { type: 'input_image', file_id: 'file-1', detail: 'auto' },
                    ],
                },
            ],
            {
                model: 'gpt-5',
                model_options: config,
                stripTextMaxTokens: 1,
                tools: [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } }],
            },
        );
        expect(first.result).toContainEqual({ type: 'image', value: 'data:image/webp;base64,YWJj' });
        expect(first.tool_use?.[0].tool_name).toBe('lookup');
        expect(create.mock.calls[0][0]).toMatchObject({
            tools: [
                expect.objectContaining({ type: 'function', name: 'lookup' }),
                expect.objectContaining({ type: 'image_generation', model: 'gpt-image-2.5-flare' }),
            ],
            tool_choice: { type: 'image_generation' },
        });
        const persisted = JSON.parse(JSON.stringify(first.conversation));
        await driver.requestTextCompletion([{ role: 'user', content: 'Make it blue' }], {
            model: 'gpt-5',
            model_options: config,
            conversation: persisted,
        });
        expect(create.mock.calls[1][0]).toMatchObject({ input: expect.arrayContaining([image]) });
        expect(JSON.stringify(create.mock.calls[1][0])).toContain('file-1');
    });
    it('strips generated payloads according to retention without corrupting replay state', async () => {
        const create = vi.fn(async (_request: unknown) => ({ ...response(), output: [image] }));
        const driver = new TestResponsesDriver(create);
        const completion = await driver.requestTextCompletion([{ role: 'user', content: 'Draw' }], {
            model: 'gpt-5',
            stripImagesAfterTurns: 0,
            model_options: { image_generation: { model: 'gpt-image-2.5-flare' } },
        });
        expect(JSON.stringify(completion.conversation)).toContain('item_reference');
        expect(JSON.stringify(completion.conversation)).not.toContain('YWJj');
    });
    it.each([Providers.openai, Providers.azure_openai, Providers.openai_compatible] as const)(
        'does not add image headers or tool choice to ordinary text requests for %s',
        async (provider) => {
            const create = vi.fn(async (_request: unknown, _options?: unknown) => response());
            const driver = new TestResponsesDriver(create, provider);
            await driver.requestTextCompletion([{ role: 'user', content: 'Hello' }], { model: 'gpt-5' });
            expect(create.mock.calls[0][1]).toBeUndefined();
            const request = JSON.parse(JSON.stringify(create.mock.calls[0][0]));
            expect(request).not.toHaveProperty('tool_choice');
            expect(request).not.toHaveProperty('tools');
        },
    );
    it('adds Azure deployment headers', async () => {
        const create = vi.fn(async (_request: unknown, _options?: unknown) => ({ ...response(), output: [image] }));
        const driver = new TestResponsesDriver(create, Providers.azure_openai);
        await driver.requestTextCompletion([{ role: 'user', content: 'Draw' }], {
            model: 'gpt-5',
            model_options: { image_generation: { model: 'gpt-image-2.5-flare' } },
        });
        expect(create.mock.calls[0][1]).toMatchObject({
            headers: {
                'x-ms-oai-image-generation-deployment': 'gpt-image-2.5-flare',
                api_version: 'preview',
            },
        });
    });
    it('returns streamed final images beside text without previews', async () => {
        const create = vi.fn(async (_request: unknown) =>
            (async function* () {
                yield { type: 'response.output_text.delta', delta: 'answer' };
                yield { type: 'response.image_generation_call.partial_image', partial_image_b64: 'preview' };
                yield { type: 'response.output_item.done', item: image };
                yield { type: 'response.output_item.done', item: image };
                yield { type: 'response.completed', response: { ...response(), output: [messageItem, image] } };
            })(),
        );
        const driver = new TestResponsesDriver(create);
        const results = [];
        for await (const chunk of await driver.requestTextCompletionStream([{ role: 'user', content: 'Draw' }], {
            model: 'gpt-5',
            model_options: { image_generation: { model: 'gpt-image-2.5-flare' } },
        }))
            results.push(...chunk.result);
        expect(results).toEqual([
            { type: 'text', value: 'answer' },
            { type: 'image', value: 'data:image/webp;base64,YWJj' },
        ]);
    });
});

it('maps PromptRole.mask to the Responses tool without treating it as a reference', async () => {
    const create = vi.fn(async (_request: unknown) => response());
    const driver = new TestResponsesDriver(create);
    const options = {
        model: 'gpt-5',
        model_options: { _option_id: 'openai-thinking' as const, image_generation: { model: 'gpt-image-2.5-flare' } },
    };
    const prompt = await driver.createPrompt(
        [
            { role: PromptRole.user, content: 'Edit', files: [new Base64DataSource('a.png', 'image/png', 'YQ==')] },
            { role: PromptRole.mask, content: '', files: [new Base64DataSource('mask.png', 'image/png', 'Yg==')] },
        ],
        options,
    );
    await driver.requestTextCompletion(prompt, options);
    expect(create.mock.calls[0][0]).toMatchObject({
        tools: [
            expect.objectContaining({
                type: 'image_generation',
                input_image_mask: expect.objectContaining({ image_url: 'data:image/png;base64,Yg==' }),
            }),
        ],
    });
    expect(JSON.stringify((create.mock.calls[0][0] as { input: unknown }).input)).not.toContain('Yg==');
});
