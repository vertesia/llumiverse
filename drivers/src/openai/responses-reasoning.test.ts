import { Base64DataSource, PromptRole, Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';

class TestResponsesDriver extends OpenAIResponsesDriverBase {
    provider: Providers.openai | Providers.azure_openai | Providers.openai_compatible;
    service: OpenAI;

    constructor(
        create: (request: unknown) => Promise<unknown>,
        provider: Providers.openai | Providers.azure_openai | Providers.openai_compatible = Providers.openai,
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
