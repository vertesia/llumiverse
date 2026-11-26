import { type Driver, PromptRole, Providers } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { OpenAICompatibleApi } from './driver-options.js';
import { createDriver, isDriverFactoryProvider } from './factory.js';
import { OpenAIChatCompletionsDriver, OpenAIResponsesDriver } from './openai/drivers.js';

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
            ).rejects.toThrow('Unsupported openai_api for an OpenAI-compatible driver: completions');
        });
    });
});
