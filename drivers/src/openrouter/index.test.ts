import { Base64DataSource, type ExecutionOptions, ModelType, PromptRole, Providers } from '@llumiverse/core';
import { RequestTimeoutError } from '@openrouter/sdk/models/errors';
import { describe, expect, it, vi } from 'vitest';
import { OpenRouterDriver } from './index.js';

function setService(driver: OpenRouterDriver, service: unknown): void {
    driver.service = service as OpenRouterDriver['service'];
}

describe('OpenRouterDriver native SDK transport', () => {
    it('requires an API key', () => {
        expect(() => new OpenRouterDriver({ apiKey: '' })).toThrow('apiKey is required');
    });

    it('supplements native structured output only for GLM 5.3', async () => {
        const driver = new OpenRouterDriver({ apiKey: 'test-key' });
        const resultSchema = { type: 'object' as const, properties: { answer: { type: 'string' as const } } };
        const segments = [{ role: PromptRole.user, content: 'Answer.' }];

        const glmPrompt = await driver.createPrompt(segments, { model: 'z-ai/glm-5.3', result_schema: resultSchema });
        const grokPrompt = await driver.createPrompt(segments, { model: 'x-ai/grok-4.6', result_schema: resultSchema });

        expect(glmPrompt.messages.some((message) => JSON.stringify(message).includes('<response_schema>'))).toBe(true);
        expect(grokPrompt.messages.some((message) => JSON.stringify(message).includes('<response_schema>'))).toBe(
            false,
        );
    });

    it('maps chat, structured output, tools, timeout, and the native response', async () => {
        const driver = new OpenRouterDriver({ apiKey: 'test-key' });
        const response = {
            id: 'generation-1',
            object: 'chat.completion' as const,
            created: 1,
            model: 'anthropic/claude-sonnet-5',
            serviceTier: 'priority',
            systemFingerprint: null,
            choices: [
                {
                    index: 0,
                    finishReason: 'tool_calls',
                    logprobs: null,
                    message: {
                        role: 'assistant' as const,
                        content: null,
                        reasoning: 'checking',
                        toolCalls: [
                            {
                                id: 'call_1',
                                type: 'function' as const,
                                function: { name: 'lookup', arguments: '{"city":"Paris"}' },
                            },
                        ],
                    },
                },
            ],
            usage: { promptTokens: 4, completionTokens: 3, totalTokens: 7 },
        };
        const send = vi.fn(async (_request: unknown, _options?: unknown) => response);
        setService(driver, { chat: { send } });

        const completion = await driver.requestTextCompletion(
            {
                _is_openai_chat_completions: true,
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'Weather?' },
                            { type: 'image_url', image_url: { url: 'https://example.test/map.png' } },
                        ],
                    },
                ],
            },
            {
                model: response.model,
                httpTimeout: { headersTimeout: 1_200_000, bodyTimeout: 1_800_000 },
                include_original_response: true,
                model_options: {
                    _option_id: 'openrouter-text',
                    max_tokens: 64,
                    effort: 'high',
                    service_tier: 'priority',
                    provider_sort: 'throughput',
                    provider_order: ['google-vertex', 'amazon-bedrock'],
                    provider_allow_fallbacks: false,
                    tool_choice: 'required',
                    required_tool_name: 'lookup',
                    parallel_tool_calls: false,
                } as ExecutionOptions['model_options'] & {
                    required_tool_name: string;
                    parallel_tool_calls: false;
                },
                result_schema: {
                    type: 'object',
                    properties: { answer: { type: 'string' } },
                    required: ['answer'],
                },
                tools: [{ name: 'lookup', description: 'Lookup weather', input_schema: { type: 'object' } }],
            },
        );

        const [request, requestOptions] = send.mock.calls[0];
        expect(request).toMatchObject({
            chatRequest: {
                model: response.model,
                maxTokens: 64,
                reasoningEffort: 'high',
                serviceTier: 'priority',
                stream: false,
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'Weather?' },
                            { type: 'image_url', imageUrl: { url: 'https://example.test/map.png' } },
                        ],
                    },
                ],
                responseFormat: {
                    type: 'json_schema',
                    jsonSchema: expect.objectContaining({ name: 'output', strict: false }),
                },
                provider: {
                    sort: 'throughput',
                    order: ['google-vertex', 'amazon-bedrock'],
                    allowFallbacks: false,
                },
                tools: [
                    {
                        type: 'function',
                        function: expect.objectContaining({ name: 'lookup' }),
                    },
                ],
                toolChoice: { type: 'function', function: { name: 'lookup' } },
                parallelToolCalls: false,
            },
        });
        expect(requestOptions).toEqual({ timeoutMs: 1_800_000 });
        expect(completion).toMatchObject({
            finish_reason: 'tool_use',
            token_usage: { prompt: 4, result: 3, total: 7 },
            tool_use: [{ id: 'call_1', tool_name: 'lookup', tool_input: { city: 'Paris' } }],
            original_response: response,
        });
    });

    it('normalizes streaming reasoning, tool calls, usage, and cancellation', async () => {
        const driver = new OpenRouterDriver({ apiKey: 'test-key' });
        const cancel = vi.fn(async () => undefined);
        const chunks = [
            {
                id: 'chunk-1',
                object: 'chat.completion.chunk' as const,
                created: 1,
                model: 'openai/gpt-5.6-sol',
                choices: [
                    {
                        index: 0,
                        finishReason: null,
                        delta: {
                            role: 'assistant' as const,
                            reasoning: 'thinking',
                            toolCalls: [
                                {
                                    index: 0,
                                    id: 'call_actual',
                                    type: 'function' as const,
                                    function: { name: 'lookup', arguments: '{"city"' },
                                },
                            ],
                        },
                    },
                ],
            },
            {
                id: 'chunk-2',
                object: 'chat.completion.chunk' as const,
                created: 1,
                model: 'openai/gpt-5.6-sol',
                choices: [
                    {
                        index: 0,
                        finishReason: 'tool_calls',
                        delta: {
                            toolCalls: [{ index: 0, function: { name: '', arguments: ':"Paris"}' } }],
                        },
                    },
                ],
                usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
            },
        ];
        const nativeStream = {
            cancel,
            async *[Symbol.asyncIterator]() {
                yield* chunks;
            },
        };
        const send = vi.fn(async (_request: unknown, _options?: unknown) => nativeStream);
        setService(driver, { chat: { send } });

        const options: ExecutionOptions = {
            model: 'openai/gpt-5.6-sol',
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
            },
            tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
        };
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Weather?' }], options);
        for await (const _chunk of stream) {
            // Consume all native chunks so the final completion and conversation are assembled.
        }

        expect(send.mock.calls[0][0]).toMatchObject({
            chatRequest: {
                stream: true,
                streamOptions: { includeUsage: true },
                messages: [{ role: 'user', content: 'Weather?' }],
                responseFormat: {
                    type: 'json_schema',
                    jsonSchema: expect.objectContaining({ name: 'output' }),
                },
            },
        });
        expect(stream.completion?.result).toContainEqual({ type: 'thoughts', value: 'thinking' });
        expect(stream.completion?.token_usage).toEqual({ prompt: 2, prompt_new: 2, result: 1, total: 3 });
        expect(stream.completion?.tool_use).toEqual([
            { id: 'call_actual', tool_name: 'lookup', tool_input: { city: 'Paris' } },
        ]);
    });

    it('maps the native model catalog and excludes dedicated inference models', async () => {
        const driver = new OpenRouterDriver({ apiKey: 'test-key' });
        const models = [
            openRouterModel({
                id: 'future-lab/vision-chat-v1',
                name: 'Vision Chat',
                inputModalities: ['text', 'image'],
                outputModalities: ['text'],
                supportedParameters: ['tools'],
            }),
            openRouterModel({
                id: 'openai/text-embedding-3-small',
                name: 'Embedding',
                inputModalities: ['text'],
                outputModalities: ['embeddings'],
            }),
            openRouterModel({
                id: 'openai/gpt-image-1',
                name: 'Image',
                inputModalities: ['text'],
                outputModalities: ['image'],
            }),
        ];
        const list = vi.fn(async () => ({ result: { data: models } }));
        setService(driver, { models: { list } });

        await expect(driver.listModels()).resolves.toEqual([
            expect.objectContaining({
                id: 'future-lab/vision-chat-v1',
                name: 'Vision Chat',
                owner: 'future-lab',
                provider: Providers.openrouter,
                type: ModelType.Text,
                input_modalities: ['text', 'image'],
                output_modalities: ['text'],
                is_multimodal: true,
                tool_support: true,
            }),
        ]);
        expect(list).toHaveBeenCalledWith({ limit: 1_000 });
    });

    it('uses the native embeddings transport and preserves input order', async () => {
        const driver = new OpenRouterDriver({ apiKey: 'test-key' });
        const generate = vi.fn(async () => ({
            object: 'list' as const,
            model: 'openai/text-embedding-3-small',
            data: [
                { object: 'embedding' as const, index: 1, embedding: [0.3, 0.4] },
                { object: 'embedding' as const, index: 0, embedding: [0.1, 0.2] },
                { object: 'embedding' as const, index: 2, embedding: [0.5, 0.6] },
            ],
            usage: {
                promptTokens: 5,
                totalTokens: 5,
                promptTokensDetails: { textTokens: 3, imageTokens: 2 },
            },
        }));
        setService(driver, { embeddings: { generate } });

        const result = await driver.generateEmbeddings({
            model: 'openai/text-embedding-3-small',
            dimensions: 128,
            task_type: 'query',
            inputs: [
                { type: 'text', text: 'first' },
                { type: 'text', text: 'second' },
                { type: 'image', source: new Base64DataSource('pixel.png', 'image/png', 'aW1hZ2U=') },
            ],
        });

        expect(generate).toHaveBeenCalledWith({
            requestBody: {
                model: 'openai/text-embedding-3-small',
                input: [
                    { content: [{ type: 'text', text: 'first' }] },
                    { content: [{ type: 'text', text: 'second' }] },
                    { content: [{ type: 'image_url', imageUrl: { url: 'data:image/png;base64,aW1hZ2U=' } }] },
                ],
                inputType: 'query',
                dimensions: 128,
                encodingFormat: 'float',
            },
        });
        expect(result).toEqual({
            model: 'openai/text-embedding-3-small',
            results: [
                { outputs: [{ values: [0.1, 0.2], modality: 'text' }] },
                { outputs: [{ values: [0.3, 0.4], modality: 'text' }] },
                { outputs: [{ values: [0.5, 0.6], modality: 'image' }] },
            ],
            usage: { input_tokens: 5, input_text_tokens: 3, input_image_tokens: 2 },
        });
    });

    it('classifies native request timeouts as retryable', () => {
        const driver = new OpenRouterDriver({ apiKey: 'test-key' });
        const error = driver.formatLlumiverseError(new RequestTimeoutError('timed out'), {
            provider: driver.provider,
            model: 'openai/gpt-5.6-sol',
            operation: 'execute',
        });

        expect(error).toMatchObject({ name: 'RequestTimeoutError', retryable: true });
    });
});

interface OpenRouterModelFixture {
    id: string;
    name: string;
    inputModalities: string[];
    outputModalities: string[];
    supportedParameters?: string[];
}

function openRouterModel(fixture: OpenRouterModelFixture): Record<string, unknown> {
    return {
        id: fixture.id,
        name: fixture.name,
        description: `${fixture.name} description`,
        architecture: {
            inputModalities: fixture.inputModalities,
            outputModalities: fixture.outputModalities,
        },
        supportedParameters: fixture.supportedParameters ?? [],
    };
}
