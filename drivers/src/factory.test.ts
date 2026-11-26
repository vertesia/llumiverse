import { type Driver, PromptRole, Providers } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { OpenAICompatibleApi } from './driver-options.js';
import { createDriver, type DriverFactoryOptions, isDriverFactoryProvider } from './factory.js';
import { GroqDriver } from './groq/index.js';
import { GroqResponsesDriver } from './groq/responses.js';
import { OpenAIChatCompletionsDriver, OpenAIResponsesDriver } from './openai/drivers.js';
import { OpenRouterDriver } from './openrouter/index.js';
import { OpenRouterResponsesDriver } from './openrouter/responses.js';
import { TogetherAIDriver } from './togetherai/index.js';
import { TogetherAIResponsesDriver } from './togetherai/responses.js';
import { xAIChatCompletionsDriver } from './xai/chat_completions.js';
import { xAIDriver } from './xai/index.js';

const OPENAI_COMPATIBLE_ENDPOINT = 'https://compatible.example/v1';

/** The URL of the request an `openai_compatible` driver sends for a one-message prompt. */
async function requestedUrl(driver: Driver): Promise<string> {
    const fetch = vi.fn(async () => new Response(JSON.stringify({ error: { message: 'stub' } }), { status: 400 }));
    (driver as unknown as { service: { fetch: typeof globalThis.fetch } }).service.fetch = fetch;
    await driver.execute([{ role: PromptRole.user, content: 'Hello' }], { model: 'some-model' }).catch(() => undefined);
    expect(fetch).toHaveBeenCalledTimes(1);
    return String((fetch.mock.calls[0] as unknown[])[0]);
}

describe('createDriver', () => {
    it('covers every provider', () => {
        for (const provider of Object.values(Providers)) {
            expect(isDriverFactoryProvider(provider), provider).toBe(true);
        }
    });

    it('creates the driver of the requested provider', async () => {
        await expect(createDriver('test', {})).resolves.toMatchObject({ provider: 'test' });
        await expect(createDriver('openai', { apiKey: 'key' })).resolves.toMatchObject({ provider: Providers.openai });
    });

    it('rejects an unknown provider', async () => {
        expect(isDriverFactoryProvider('toString')).toBe(false);
        await expect(createDriver('unknown' as 'test', {})).rejects.toThrow('Unknown driver provider: unknown');
    });

    describe('openai_compatible', () => {
        const options = { apiKey: 'key', endpoint: OPENAI_COMPATIBLE_ENDPOINT };

        it.each([
            { openai_api: undefined, driverClass: OpenAIResponsesDriver, path: '/responses' },
            { openai_api: 'responses' as const, driverClass: OpenAIResponsesDriver, path: '/responses' },
            {
                openai_api: 'chat_completions' as const,
                driverClass: OpenAIChatCompletionsDriver,
                path: '/chat/completions',
            },
        ])('openai_api $openai_api calls $path', async ({ openai_api, driverClass, path }) => {
            const driver = await createDriver('openai_compatible', { ...options, openai_api });
            try {
                expect(driver).toBeInstanceOf(driverClass);
                expect(driver).toMatchObject({ provider: Providers.openai_compatible });
                expect(await requestedUrl(driver)).toBe(`${OPENAI_COMPATIBLE_ENDPOINT}${path}`);
            } finally {
                driver.destroy();
            }
        });

        it.each([
            { model: 'Inkling-2-Mini-GGUF', openai_api: 'responses' as const },
            { model: 'Inkling-2-Mini-GGUF', openai_api: 'chat_completions' as const },
            // A model the directory does not know still gets its tools.
            { model: 'acme-local-chat-v1', openai_api: 'responses' as const },
            { model: 'acme-local-chat-v1', openai_api: 'chat_completions' as const },
        ])('sends the tools of a $model request through $openai_api', async ({ model, openai_api }) => {
            const driver = await createDriver('openai_compatible', { ...options, openai_api });
            const fetch = vi.fn(
                async () => new Response(JSON.stringify({ error: { message: 'stub' } }), { status: 400 }),
            );
            (driver as unknown as { service: { fetch: typeof globalThis.fetch } }).service.fetch = fetch;
            try {
                await driver
                    .execute([{ role: PromptRole.user, content: 'Weather in Paris?' }], {
                        model,
                        tools: [
                            {
                                name: 'get_weather',
                                description: 'Get the weather for a city',
                                input_schema: { type: 'object', properties: { city: { type: 'string' } } },
                            },
                        ],
                    })
                    .catch(() => undefined);
                const init = (fetch.mock.calls[0] as unknown[])[1] as RequestInit;
                const body = JSON.parse(String(init.body)) as {
                    tools?: { name?: string; function?: { name: string } }[];
                };
                expect(body.tools?.map((tool) => tool.name ?? tool.function?.name)).toEqual(['get_weather']);
            } finally {
                driver.destroy();
            }
        });

        it('rejects an unknown openai_api', async () => {
            await expect(
                createDriver('openai_compatible', { ...options, openai_api: 'completions' as OpenAICompatibleApi }),
            ).rejects.toThrow('Unsupported openai_api for the openai_compatible driver: completions');
        });
    });

    describe('openai_api on providers whose driver calls either API', () => {
        type SwitchableProvider = 'openrouter' | 'groq' | 'togetherai' | 'xai';
        const cases: {
            provider: SwitchableProvider;
            options: DriverFactoryOptions[SwitchableProvider];
            defaultClass: abstract new (...args: never[]) => Driver;
            chatClass: abstract new (...args: never[]) => Driver;
            responsesClass: abstract new (...args: never[]) => Driver;
            /** The URL of each API's requests, for the drivers that send them through the OpenAI SDK. */
            urls: Partial<Record<'responses' | 'chat_completions', string>>;
        }[] = [
            {
                provider: 'openrouter',
                options: { apiKey: 'key', endpoint: 'https://router.example/api/v1' },
                defaultClass: OpenRouterDriver,
                chatClass: OpenRouterDriver,
                responsesClass: OpenRouterResponsesDriver,
                urls: { responses: 'https://router.example/api/v1/responses' },
            },
            {
                provider: 'groq',
                options: { apiKey: 'key', endpoint_url: 'https://groq.example' },
                defaultClass: GroqDriver,
                chatClass: GroqDriver,
                responsesClass: GroqResponsesDriver,
                urls: { responses: 'https://groq.example/openai/v1/responses' },
            },
            {
                provider: 'togetherai',
                options: { apiKey: 'key', endpoint: 'https://together.example' },
                defaultClass: TogetherAIDriver,
                chatClass: TogetherAIDriver,
                responsesClass: TogetherAIResponsesDriver,
                urls: { responses: 'https://together.example/v1/responses' },
            },
            {
                provider: 'xai',
                options: { apiKey: 'key', endpoint: 'https://xai.example/v1' },
                defaultClass: xAIDriver,
                chatClass: xAIChatCompletionsDriver,
                responsesClass: xAIDriver,
                urls: {
                    responses: 'https://xai.example/v1/responses',
                    chat_completions: 'https://xai.example/v1/chat/completions',
                },
            },
        ];

        it.each(cases)('$provider keeps its current driver when openai_api is omitted', async (testCase) => {
            const driver = await createDriver(testCase.provider, testCase.options);
            try {
                expect(driver).toBeInstanceOf(testCase.defaultClass);
                expect(driver).toMatchObject({ provider: testCase.provider });
            } finally {
                driver.destroy();
            }
        });

        it.each(
            cases.flatMap((testCase) =>
                (['responses', 'chat_completions'] as const).map((openai_api) => ({ ...testCase, openai_api })),
            ),
        )('$provider with openai_api $openai_api', async (testCase) => {
            const driver = await createDriver(testCase.provider, {
                ...testCase.options,
                openai_api: testCase.openai_api,
            });
            try {
                expect(driver).toBeInstanceOf(
                    testCase.openai_api === 'responses' ? testCase.responsesClass : testCase.chatClass,
                );
                expect(driver).toMatchObject({ provider: testCase.provider });
                const url = testCase.urls[testCase.openai_api];
                if (url) expect(await requestedUrl(driver)).toBe(url);
            } finally {
                driver.destroy();
            }
        });

        it.each(cases)('$provider rejects an unknown openai_api', async (testCase) => {
            await expect(
                createDriver(testCase.provider, {
                    ...testCase.options,
                    openai_api: 'completions' as OpenAICompatibleApi,
                }),
            ).rejects.toThrow(`Unsupported openai_api for the ${testCase.provider} driver: completions`);
        });

        it('sends OpenRouter routing preferences and attribution on Responses in the wire format', async () => {
            const driver = await createDriver('openrouter', {
                apiKey: 'key',
                appTitle: 'Test App',
                httpReferer: 'https://app.example',
                openai_api: 'responses',
            });
            const { body, headers } = await capturedRequest(driver, {
                model: 'openai/gpt-5.6',
                model_options: {
                    _option_id: 'openrouter-text',
                    provider_require_parameters: true,
                    provider_allow_fallbacks: false,
                    provider_order: ['anthropic'],
                },
            });
            try {
                expect(body.provider).toEqual({
                    require_parameters: true,
                    allow_fallbacks: false,
                    order: ['anthropic'],
                });
                expect(headers.get('X-OpenRouter-Title')).toBe('Test App');
                expect(headers.get('HTTP-Referer')).toBe('https://app.example');
            } finally {
                driver.destroy();
            }
        });

        it('leaves the fields Groq does not support out of its Responses requests', async () => {
            const driver = await createDriver('groq', { apiKey: 'key', openai_api: 'responses' });
            try {
                const { body } = await capturedRequest(driver, {
                    model: 'openai/gpt-oss-120b',
                    prompt_cache_key: 'cache-key',
                    model_options: { _option_id: 'openai-thinking', reasoning_effort: 'low' },
                });
                expect(body).not.toHaveProperty('prompt_cache_key');
                expect(body).not.toHaveProperty('include');
                expect(body.model).toBe('openai/gpt-oss-120b');
            } finally {
                driver.destroy();
            }
        });

        it('refuses a Responses driver without an API key instead of using the OpenAI key of the environment', async () => {
            vi.stubEnv('OPENAI_API_KEY', 'server-openai-key');
            // The Groq SDK falls back to its own variable, so only the Responses client's check stands in the way.
            vi.stubEnv('GROQ_API_KEY', 'server-groq-key');
            try {
                await expect(
                    createDriver('groq', { apiKey: undefined as unknown as string, openai_api: 'responses' }),
                ).rejects.toThrow('apiKey is required');
            } finally {
                vi.unstubAllEnvs();
            }
        });

        it("leaves a Groq Responses caller's extra_body fields for Groq to validate", async () => {
            const driver = await createDriver('groq', { apiKey: 'key', openai_api: 'responses' });
            try {
                const { body } = await capturedRequest(driver, {
                    model: 'openai/gpt-oss-120b',
                    model_options: { _option_id: 'openai-text', extra_body: { truncation: 'auto' } },
                });
                expect(body.truncation).toBe('auto');
            } finally {
                driver.destroy();
            }
        });

        it('reports a Groq reasoning_format the Responses API has no field for', async () => {
            const driver = await createDriver('groq', { apiKey: 'key', openai_api: 'responses' });
            const warn = vi.fn();
            (driver as unknown as { logger: unknown }).logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
            try {
                const { body } = await capturedRequest(driver, {
                    model: 'deepseek-r1-distill-llama-70b',
                    model_options: { _option_id: 'groq-deepseek-thinking', reasoning_format: 'hidden' },
                });
                expect(body).not.toHaveProperty('reasoning_format');
                expect(warn).toHaveBeenCalledWith(
                    expect.objectContaining({
                        option_names: ['reasoning_format'],
                        reason: 'groq_reasoning_format_without_responses_field',
                    }),
                    expect.any(String),
                );
            } finally {
                driver.destroy();
            }
        });

        it('runs image models through the provider driver on Responses', async () => {
            const driver = await createDriver('openrouter', { apiKey: 'key', openai_api: 'responses' });
            const imageResult = { result: [{ type: 'image', value: 'https://image.example/1.png' }] };
            const providerDriver = { execute: vi.fn(async () => imageResult), destroy: vi.fn() };
            Object.defineProperty(driver, 'providerDriver', { value: providerDriver });
            const fetch = vi.fn();
            (driver as unknown as { service: { fetch: typeof globalThis.fetch } }).service.fetch = fetch;
            try {
                await expect(
                    driver.execute([{ role: PromptRole.user, content: 'A cat' }], { model: 'openai/gpt-image-1' }),
                ).resolves.toBe(imageResult);
                expect(providerDriver.execute).toHaveBeenCalledTimes(1);
                expect(fetch).not.toHaveBeenCalled();
            } finally {
                driver.destroy();
            }
        });

        it('keeps model listing and teardown with the provider driver on Responses', async () => {
            const driver = await createDriver('openrouter', { apiKey: 'key', openai_api: 'responses' });
            const providerDriver = {
                listModels: vi.fn(async () => []),
                validateConnection: vi.fn(async () => true),
                destroy: vi.fn(),
            };
            Object.defineProperty(driver, 'providerDriver', { value: providerDriver });
            await expect(driver.listModels()).resolves.toEqual([]);
            await expect(driver.validateConnection()).resolves.toBe(true);
            driver.destroy();
            expect(providerDriver.listModels).toHaveBeenCalledTimes(1);
            expect(providerDriver.validateConnection).toHaveBeenCalledTimes(1);
            expect(providerDriver.destroy).toHaveBeenCalledTimes(1);
        });
    });
});

/** The body and headers of the request a driver sends through its OpenAI SDK client for a one-message prompt. */
async function capturedRequest(
    driver: Driver,
    options: Omit<Parameters<Driver['execute']>[1], 'model'> & { model: string },
): Promise<{ body: Record<string, unknown>; headers: Headers }> {
    const fetch = vi.fn(async () => new Response(JSON.stringify({ error: { message: 'stub' } }), { status: 400 }));
    (driver as unknown as { service: { fetch: typeof globalThis.fetch } }).service.fetch = fetch;
    await driver.execute([{ role: PromptRole.user, content: 'Hello' }], options).catch(() => undefined);
    expect(fetch).toHaveBeenCalledTimes(1);
    const init = (fetch.mock.calls[0] as unknown[])[1] as RequestInit;
    return { body: JSON.parse(String(init.body)) as Record<string, unknown>, headers: new Headers(init.headers) };
}
