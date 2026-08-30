import { PromptRole, Providers } from '@llumiverse/core';
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

    it.each(['gpt-5.4', 'gpt-5.5', 'gpt-5.6', 'gpt-5.6-sol', 'gpt-5.7'])(
        'uses current-turn reasoning context for %s',
        async (model) => {
            const create = vi.fn(async (_request: unknown) => response());
            const driver = new TestResponsesDriver(create);

            await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
                model,
                model_options: { _option_id: 'openai-thinking' },
            });

            expect(create).toHaveBeenCalledWith(
                expect.objectContaining({
                    reasoning: expect.objectContaining({ context: 'current_turn' }),
                }),
            );
        },
    );

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
