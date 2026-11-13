import { PromptRole } from '@llumiverse/core';
import { AzureOpenAI, toFile, toStreamingFile } from 'openai';
import { afterEach, describe, expect, it, vi } from 'vitest';
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
});

afterEach(() => {
    vi.restoreAllMocks();
});
