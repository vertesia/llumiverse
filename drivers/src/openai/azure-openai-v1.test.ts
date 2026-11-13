import { PromptRole } from '@llumiverse/core';
import { AzureOpenAI, toFile, toStreamingFile } from 'openai';
import { afterEach, describe, expect, expectTypeOf, it, vi } from 'vitest';
import { stubCoreDriverFetch } from '../../test/__helpers__/test-utils.js';
import { AzureOpenAIDriver, AzureOpenAIV1Client } from './azure_openai.js';

const responsesBody = {
    id: 'response-1',
    object: 'response',
    created_at: 1,
    model: 'chat',
    status: 'completed',
    output: [
        {
            id: 'message-1',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [{ type: 'output_text', text: 'Hello', annotations: [], logprobs: [] }],
        },
    ],
    usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
};
const chatBody = {
    id: 'chat-1',
    object: 'chat.completion',
    created: 1,
    model: 'gpt-6-luna-2026-09-22',
    choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'Hi' } }],
};
const embeddingsBody = { object: 'list', data: [{ object: 'embedding', index: 0, embedding: [0.5] }], model: 'e' };

function capture(driver: AzureOpenAIDriver) {
    const requests: { url: string; body?: Record<string, unknown>; headers: Headers }[] = [];
    const fetch = vi.fn<typeof globalThis.fetch>(async (url, init) => {
        const target = String(url);
        requests.push({
            url: target,
            body: init?.body ? JSON.parse(String(init.body)) : undefined,
            headers: new Headers(init?.headers),
        });
        const body = target.includes('/responses')
            ? responsesBody
            : target.includes('/embeddings')
              ? embeddingsBody
              : chatBody;
        return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
    });
    driver.service = driver.service.withOptions({ fetch });
    return requests;
}

describe('Azure OpenAI v1 API', () => {
    it('uses the version-less v1 path with the api-key header by default', async () => {
        const driver = new AzureOpenAIDriver({ endpoint: 'https://example.openai.azure.com/', apiKey: 'test-key' });
        const requests = capture(driver);
        try {
            expect(driver.service).toBeInstanceOf(AzureOpenAIV1Client);
            expect(driver.service.baseURL).toBe('https://example.openai.azure.com/openai/v1');
            const result = await driver.execute([{ role: PromptRole.user, content: 'Hello' }], { model: 'chat' });
            expect(requests[0].url).toBe('https://example.openai.azure.com/openai/v1/responses');
            expect(requests[0].body).toMatchObject({ model: 'chat' });
            expect(requests[0].headers.get('api-key')).toBe('test-key');
            expect(requests[0].headers.has('authorization')).toBe(false);
            expect(result.result).toEqual([{ type: 'text', value: 'Hello' }]);
        } finally {
            await driver.destroy();
        }
    });

    it.each([
        'https://example.openai.azure.com/openai',
        'https://example.openai.azure.com/openai/v1/',
        'https://example.openai.azure.com///',
    ])('accepts an endpoint that already names the API: %s', (endpoint) => {
        const driver = new AzureOpenAIDriver({ endpoint, apiKey: 'test-key' });
        expect(driver.service.baseURL).toBe('https://example.openai.azure.com/openai/v1');
        expect(driver.getImageService().baseURL).toBe('https://example.openai.azure.com/openai/v1');
    });

    it('sends an Entra token as a bearer token', async () => {
        const azureADTokenProvider = vi.fn(async () => 'entra-token');
        const driver = new AzureOpenAIDriver({ endpoint: 'https://example.openai.azure.com', azureADTokenProvider });
        const requests = capture(driver);
        try {
            await driver.execute([{ role: PromptRole.user, content: 'Hello' }], { model: 'chat' });
            expect(requests[0].headers.get('authorization')).toBe('Bearer entra-token');
            expect(requests[0].headers.has('api-key')).toBe(false);
            expect(azureADTokenProvider).toHaveBeenCalled();
        } finally {
            await driver.destroy();
        }
    });

    it('sends the configured deployment as the model of deployment-scoped requests', async () => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test-key',
            deployment: 'production',
        });
        const requests = capture(driver);
        try {
            expect(await driver.listModels()).toEqual([expect.objectContaining({ id: 'gpt-6-luna-2026-09-22' })]);
            await driver.generateEmbeddings({
                model: 'text-embedding-3-large',
                inputs: [{ type: 'text', text: 'hello' }],
            });
            await driver.execute([{ role: PromptRole.user, content: 'Hello' }], { model: 'chat' });
            expect(requests.map((request) => [request.url, request.body?.model])).toEqual([
                ['https://example.openai.azure.com/openai/v1/chat/completions', 'production'],
                ['https://example.openai.azure.com/openai/v1/embeddings', 'production'],
                // Responses keep the request's model, as on the legacy client.
                ['https://example.openai.azure.com/openai/v1/responses', 'chat'],
            ]);
        } finally {
            await driver.destroy();
        }
    });

    it('keeps the configured deployment on derived clients', () => {
        const client = new AzureOpenAIV1Client({
            baseURL: 'https://example.openai.azure.com/openai/v1',
            apiKey: 'test-key',
            azureDeployment: 'production',
        });
        expect((client.withOptions({ timeout: 1 }) as AzureOpenAIV1Client).deploymentName).toBe('production');
    });

    it('keeps the dated API when the caller sets an apiVersion', async () => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test-key',
            apiVersion: '2025-04-01-preview',
            deployment: 'production',
        });
        const requests = capture(driver);
        try {
            expect(driver.service).toBeInstanceOf(AzureOpenAI);
            await driver.listModels();
            expect(requests[0].url).toBe(
                'https://example.openai.azure.com/openai/deployments/production/chat/completions?api-version=2025-04-01-preview',
            );
        } finally {
            await driver.destroy();
        }
    });

    it('does not follow redirects on requests that carry the api-key', async () => {
        const inits: RequestInit[] = [];
        const fetch = vi.fn<typeof globalThis.fetch>(async (_url, init) => {
            inits.push(init ?? {});
            return new Response(JSON.stringify(responsesBody), { headers: { 'content-type': 'application/json' } });
        });
        stubCoreDriverFetch(AzureOpenAIDriver, fetch);
        const driver = new AzureOpenAIDriver({ endpoint: 'https://example.openai.azure.com', apiKey: 'test-key' });
        try {
            await driver.execute([{ role: PromptRole.user, content: 'Hello' }], { model: 'chat' });
            await driver.getImageService().models.list();
            expect(inits).toHaveLength(2);
            for (const init of inits) expect(init.redirect).toBe('manual');
        } finally {
            await driver.destroy();
        }
    });

    it('keeps the default redirect policy on Entra-authenticated requests', async () => {
        const inits: RequestInit[] = [];
        stubCoreDriverFetch(AzureOpenAIDriver, async (_url, init) => {
            inits.push(init ?? {});
            return new Response(JSON.stringify(responsesBody), { headers: { 'content-type': 'application/json' } });
        });
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            azureADTokenProvider: async () => 'entra-token',
        });
        try {
            await driver.execute([{ role: PromptRole.user, content: 'Hello' }], { model: 'chat' });
            expect(inits[0].redirect).toBeUndefined();
        } finally {
            await driver.destroy();
        }
    });

    it('sends the configured deployment as the model of multipart uploads', async () => {
        const bodies: string[] = [];
        const client = new AzureOpenAIV1Client({
            baseURL: 'https://example.openai.azure.com/openai/v1',
            apiKey: 'test-key',
            azureDeployment: 'custom-whisper',
            fetch: async (_url, init) => {
                const text = await new Response(init?.body).text();
                // The SDK probes FormData support with an empty request first.
                if (text) bodies.push(text);
                return Response.json({ text: 'Hello.', created: 1, data: [] });
            },
        });
        const wav = () => new Blob([new Uint8Array([82, 73, 70, 70])]);
        await client.audio.transcriptions.create({ model: 'whisper-1', file: await toFile(wav(), 'a.wav') });
        await client.audio.transcriptions.create({
            model: 'whisper-1',
            file: toStreamingFile(wav().stream(), 'a.wav', { type: 'audio/wav' }),
        });
        await client.audio.translations.create({ model: 'whisper-1', file: await toFile(wav(), 'a.wav') });
        await client.images.edit({ model: 'gpt-image-1', prompt: 'p', image: await toFile(wav(), 'a.png') });
        expect(bodies).toHaveLength(4);
        for (const body of bodies) {
            expect(body).toMatch(/name="model"\r\n\r\ncustom-whisper\r\n/);
            expect(body).not.toMatch(/whisper-1|gpt-image-1/);
        }
    });

    it('types the service by how the driver was built', () => {
        const injected = new AzureOpenAI({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test-key',
            apiVersion: '2024-10-21',
            deployment: 'legacy',
        });
        const legacy = new AzureOpenAIDriver(injected);
        const pinned = new AzureOpenAIDriver<AzureOpenAI>({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test-key',
            apiVersion: '2025-04-01-preview',
            deployment: 'production',
        });
        const endpoint = 'https://example.openai.azure.com';
        // Compile-time checks only: these constructions are type errors, and one would throw without an endpoint.
        const mismatchedClientTypes = () => {
            // @ts-expect-error The legacy client type needs the pinned apiVersion that builds it.
            new AzureOpenAIDriver<AzureOpenAI>({ endpoint, apiKey: 'test-key' });
            // @ts-expect-error An empty apiVersion builds the v1 client, not the legacy one.
            new AzureOpenAIDriver<AzureOpenAI>({ endpoint, apiKey: 'test-key', apiVersion: '' });
            // @ts-expect-error A dated apiVersion builds the legacy client, not the v1 one.
            new AzureOpenAIDriver<AzureOpenAIV1Client>({ endpoint, apiKey: 'test-key', apiVersion: '2024-10-21' });
            // @ts-expect-error Without an endpoint the driver builds the legacy client from AZURE_OPENAI_ENDPOINT.
            new AzureOpenAIDriver<AzureOpenAIV1Client>({ apiKey: 'test-key' });
        };
        expectTypeOf(mismatchedClientTypes).toBeFunction();
        const selectedV1 = new AzureOpenAIDriver<AzureOpenAIV1Client>({ endpoint, apiKey: 'test-key' });
        expect(selectedV1.service).toBeInstanceOf(AzureOpenAIV1Client);
        expectTypeOf(selectedV1.service.apiVersion).toEqualTypeOf<undefined>();
        const v1 = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test-key',
            deployment: 'production',
        });
        expectTypeOf(legacy.service).toEqualTypeOf<AzureOpenAI>();
        expectTypeOf(pinned.service).toEqualTypeOf<AzureOpenAI>();
        expectTypeOf(pinned.service.apiVersion).toEqualTypeOf<string>();
        expectTypeOf(v1.service.deploymentName).toEqualTypeOf<string | undefined>();
        expectTypeOf(v1.service.apiVersion).toEqualTypeOf<string | undefined>();
        expect(legacy.service).toBe(injected);
        expect(pinned.service).toBeInstanceOf(AzureOpenAI);
        expect([legacy.service.deploymentName, legacy.service.apiVersion]).toEqual(['legacy', '2024-10-21']);
        expect([pinned.service.deploymentName, pinned.service.apiVersion]).toEqual([
            'production',
            '2025-04-01-preview',
        ]);
        expect([v1.service.deploymentName, v1.service.apiVersion]).toEqual(['production', undefined]);
    });

    it('decides prompt caching by the source model of a composite ID, not the deployment alias', async () => {
        const driver = new AzureOpenAIDriver({ endpoint: 'https://example.openai.azure.com', apiKey: 'test-key' });
        const requests = capture(driver);
        const conversation = [
            { role: PromptRole.user, content: 'First question' },
            { role: PromptRole.assistant, content: 'First answer' },
            { role: PromptRole.user, content: 'Second question' },
        ];
        try {
            for (const cache_ttl of ['12h', '24h'] as const) {
                await driver.execute(conversation, {
                    model: 'gpt-6-luna::gpt-5.4',
                    prompt_cache_key: 'key',
                    model_options: { _option_id: 'openai-thinking', cache_ttl },
                });
            }
            const [unsupported, supported] = requests.map((request) => request.body ?? {});
            // GPT-5.4 takes neither the 5.6+ lifetimes nor explicit breakpoints.
            expect(unsupported.prompt_cache_options).toBeUndefined();
            expect(unsupported.prompt_cache_retention).toBeUndefined();
            expect(JSON.stringify(unsupported.input)).not.toContain('prompt_cache_breakpoint');
            expect(supported).toMatchObject({ model: 'gpt-6-luna', prompt_cache_retention: '24h' });
            expect(supported.prompt_cache_options).toBeUndefined();
        } finally {
            await driver.destroy();
        }
    });

    it('decides prompt caching by the sourceModel of a deployment named like another model', async () => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test-key',
            sourceModel: 'gpt-5.4',
        });
        const requests = capture(driver);
        try {
            await driver.execute(
                [
                    { role: PromptRole.user, content: 'First question' },
                    { role: PromptRole.assistant, content: 'First answer' },
                    { role: PromptRole.user, content: 'Second question' },
                ],
                {
                    model: 'gpt-6-luna',
                    prompt_cache_key: 'key',
                    model_options: { _option_id: 'openai-thinking', cache_ttl: '24h' },
                },
            );
            const body = requests[0].body ?? {};
            expect(body).toMatchObject({ model: 'gpt-6-luna', prompt_cache_retention: '24h' });
            expect(body.prompt_cache_options).toBeUndefined();
            expect(JSON.stringify(body.input)).not.toContain('prompt_cache_breakpoint');
        } finally {
            await driver.destroy();
        }
    });

    it('applies sourceModel to the configured deployment only', async () => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test-key',
            deployment: 'primary',
            sourceModel: 'gpt-5.6-luna',
        });
        const requests = capture(driver);
        const conversation = [
            { role: PromptRole.user, content: 'First question' },
            { role: PromptRole.assistant, content: 'First answer' },
            { role: PromptRole.user, content: 'Second question' },
        ];
        try {
            for (const [model, cache_ttl] of [
                ['gpt-5.4', '24h'],
                ['primary', '12h'],
            ] as const) {
                await driver.execute(conversation, {
                    model,
                    prompt_cache_key: 'key',
                    model_options: { _option_id: 'openai-thinking', cache_ttl },
                });
            }
            const [other, primary] = requests.map((request) => request.body ?? {});
            // Another deployment keeps its own model's cache parameters.
            expect(other).toMatchObject({ model: 'gpt-5.4', prompt_cache_retention: '24h' });
            expect(other.prompt_cache_options).toBeUndefined();
            expect(JSON.stringify(other.input)).not.toContain('prompt_cache_breakpoint');
            // The configured deployment follows its sourceModel.
            expect(primary).toMatchObject({ model: 'primary', prompt_cache_options: { mode: 'explicit', ttl: '12h' } });
        } finally {
            await driver.destroy();
        }
    });
});

afterEach(() => {
    vi.restoreAllMocks();
});
