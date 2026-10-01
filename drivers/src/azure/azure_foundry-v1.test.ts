import type { TokenCredential } from '@azure/identity';
import { Base64DataSource, PromptRole } from '@llumiverse/core';
import OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { exposePrivate } from '../../test/__helpers__/test-utils.js';
import { AzureFoundryDriver } from './azure_foundry.js';

type Internals = {
    getDriverFetch: () => typeof fetch;
    getInferenceClient: () => OpenAI;
    getInferenceProtocolDriver: () => { destroy: () => void };
    getOpenAIProtocolDriver: () => {
        getDriverFetch: () => typeof fetch;
        getImageService: () => OpenAI;
        destroy: () => void;
    };
};

function setup(apiVersion?: string) {
    const getToken = vi.fn<TokenCredential['getToken']>(async () => ({
        token: 'test-token',
        expiresOnTimestamp: Date.now() + 3_600_000,
    }));
    const driver = new AzureFoundryDriver({
        endpoint: 'https://foundry.example.test/api/projects/project',
        azureADTokenProvider: { getToken },
        apiVersion,
    });
    const requests: { url: string; body: Record<string, unknown>; headers: Headers; signal?: AbortSignal | null }[] =
        [];
    const respond = vi.fn<(body: Record<string, unknown>) => Response>(
        () => new Response('{}', { headers: { 'content-type': 'application/json' } }),
    );
    const fetch = vi.fn<typeof globalThis.fetch>(async (url, init) => {
        requests.push({
            url: String(url),
            body: JSON.parse(String(init?.body)),
            headers: new Headers(init?.headers),
            signal: init?.signal,
        });
        return respond(requests[requests.length - 1].body);
    });
    const internals = exposePrivate<Internals>(driver);
    vi.spyOn(internals, 'getDriverFetch').mockReturnValue(fetch);
    return { driver, internals, requests, fetch, respond, getToken };
}

const chatResponse = {
    id: 'chat-1',
    object: 'chat.completion',
    created: 1,
    model: 'chat',
    choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'Hello' } }],
    usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
};

describe('Foundry OpenAI v1 transport', () => {
    it.each([undefined, 'explicit-version'])('uses the native SDK and preserves API version %s', async (apiVersion) => {
        const { driver, internals, respond, requests, getToken } = setup(apiVersion);
        vi.spyOn(driver.service.deployments, 'get').mockResolvedValue({
            type: 'ModelDeployment',
            name: 'chat',
            modelPublisher: 'Meta',
            modelName: 'Llama-5',
            modelVersion: '1',
            capabilities: { chat_completion: 'true' },
        });
        respond.mockImplementation(() => {
            return new Response(JSON.stringify(chatResponse), { headers: { 'content-type': 'application/json' } });
        });
        try {
            const result = await driver.execute([{ role: PromptRole.user, content: 'Hello' }], {
                model: 'chat::Llama-5',
                include_original_response: true,
            });
            expect(internals.getInferenceClient()).toBeInstanceOf(OpenAI);
            expect(requests[0].url).toBe(
                'https://foundry.example.test/api/projects/project/openai/v1/chat/completions' +
                    (apiVersion ? `?api-version=${apiVersion}` : ''),
            );
            expect(requests[0].body).toMatchObject({ model: 'chat', stream: false });
            expect(requests[0].headers.get('authorization')).toBe('Bearer test-token');
            expect(requests[0].headers.has('x-ms-oai-image-generation-deployment')).toBe(false);
            expect(getToken.mock.calls[0][0]).toContain('https://ai.azure.com/.default');
            expect(result.result).toEqual([{ type: 'text', value: 'Hello' }]);
            expect(result.token_usage).toEqual({ prompt: 2, result: 1, total: 3 });
            expect(result.original_response).toEqual(chatResponse);
        } finally {
            driver.destroy();
        }
    });

    it('preserves injected resource endpoints for images and embeddings', async () => {
        const { driver, internals, fetch, respond, requests } = setup();
        vi.spyOn(driver.service, 'getOpenAIClient').mockReturnValue({
            baseURL: 'https://gateway.example.test/custom/openai/v1',
        } as ReturnType<typeof driver.service.getOpenAIClient>);
        vi.spyOn(internals.getOpenAIProtocolDriver(), 'getDriverFetch').mockReturnValue(fetch);
        respond.mockImplementation(
            ({ input }) =>
                new Response(
                    JSON.stringify(input ? { data: [{ index: 0, embedding: [1] }] } : { data: [{ b64_json: 'YQ==' }] }),
                    {
                        headers: { 'content-type': 'application/json' },
                    },
                ),
        );
        try {
            await driver.execute([{ role: PromptRole.user, content: 'A garden' }], { model: 'garden::gpt-image-2' });
            await driver.generateEmbeddings({ model: 'embedding', inputs: [{ type: 'text', text: 'Hello' }] });
            expect(requests.map((request) => request.url)).toEqual([
                'https://gateway.example.test/custom/openai/v1/images/generations',
                'https://gateway.example.test/custom/openai/v1/embeddings',
            ]);
        } finally {
            driver.destroy();
        }
    });

    it('streams non-OpenAI text and cancels the SDK stream on early exit', async () => {
        const { driver, internals, respond, requests } = setup();
        vi.spyOn(driver.service.deployments, 'get').mockResolvedValue({
            type: 'ModelDeployment',
            name: 'chat',
            modelPublisher: 'Meta',
            modelName: 'Llama-5',
            modelVersion: '1',
            capabilities: { chat_completion: 'true' },
        });
        respond.mockImplementation(() => {
            return new Response(
                new ReadableStream({
                    start(controller) {
                        const chunk = {
                            id: 'chat-1',
                            object: 'chat.completion.chunk',
                            created: 1,
                            model: 'chat',
                            choices: [
                                { index: 0, finish_reason: null, delta: { role: 'assistant', content: 'Hello' } },
                            ],
                        };
                        controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
                    },
                }),
                { headers: { 'content-type': 'text/event-stream' } },
            );
        });
        try {
            const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], {
                model: 'chat::Llama-5',
            });
            const stream = await driver.requestTextCompletionStream(prompt, { model: 'chat::Llama-5' });
            for await (const chunk of stream) {
                expect(chunk).toMatchObject({ result: [{ type: 'text', value: 'Hello' }] });
                break;
            }
            expect(requests[0].body).toMatchObject({
                model: 'chat',
                stream: true,
            });
            expect(requests[0].body).not.toHaveProperty('stream_options');
            await vi.waitFor(() => expect(requests[0].signal?.aborted).toBe(true));
            expect(internals.getInferenceClient().maxRetries).toBe(0);
        } finally {
            driver.destroy();
        }
    });

    it('passes cancellation and per-request deadlines to the Chat SDK', async () => {
        const { driver, internals } = setup();
        vi.spyOn(driver.service.deployments, 'get').mockResolvedValue({
            type: 'ModelDeployment',
            name: 'chat',
            modelPublisher: 'Meta',
            modelName: 'Llama-5',
            modelVersion: '1',
            capabilities: { chat_completion: 'true' },
        });
        const create = vi
            .spyOn(internals.getInferenceClient().chat.completions, 'create')
            .mockResolvedValue(chatResponse as OpenAI.Chat.ChatCompletion);
        const controller = new AbortController();
        const options = { model: 'chat::Llama-5', httpTimeout: { headersTimeout: 100, bodyTimeout: 200 } };
        try {
            const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], options);
            await driver.requestTextCompletion(prompt, options, controller.signal);
            expect(create.mock.calls[0][1]).toMatchObject({ signal: controller.signal, timeout: 200 });
        } finally {
            driver.destroy();
        }
    });

    it('releases both protocol adapters and the image transport on destruction', () => {
        const { driver, internals } = setup();
        const responses = internals.getOpenAIProtocolDriver();
        responses.getImageService();
        const chat = internals.getInferenceProtocolDriver();
        const destroyResponses = vi.spyOn(responses, 'destroy');
        const destroyChat = vi.spyOn(chat, 'destroy');
        driver.destroy();
        expect(destroyResponses).toHaveBeenCalledOnce();
        expect(destroyChat).toHaveBeenCalledOnce();
    });

    it('preserves embedding order and uses deployment IDs on the v1 endpoint', async () => {
        const { driver, respond, requests } = setup();
        respond.mockImplementation(({ input }) => {
            const data =
                Array.isArray(input) && input.length === 2
                    ? [
                          { index: 1, embedding: [2] },
                          { index: 0, embedding: [1] },
                      ]
                    : [{ index: 0, embedding: [3] }];
            return new Response(JSON.stringify({ object: 'list', model: 'embedding', data }), {
                headers: { 'content-type': 'application/json' },
            });
        });
        try {
            const result = await driver.generateEmbeddings({
                model: 'embedding::cohere-embed-v4',
                inputs: [
                    { type: 'text', text: 'first' },
                    { type: 'image', source: new Base64DataSource('reference.png', 'image/png', 'YQ==') },
                    { type: 'text', text: 'second' },
                ],
            });
            expect(requests.map((request) => request.body)).toEqual([
                { input: ['first', 'second'], model: 'embedding', encoding_format: 'float' },
                { input: ['YQ=='], model: 'embedding', encoding_format: 'float' },
            ]);
            expect(requests.every((request) => request.url.endsWith('/openai/v1/embeddings'))).toBe(true);
            expect(result.results.map((item) => item.outputs[0].values)).toEqual([[1], [3], [2]]);
        } finally {
            driver.destroy();
        }
    });

    it.each([undefined, 'explicit-version'])(
        'uses native image authentication and preserves only explicit API version %s',
        async (apiVersion) => {
            const { driver, internals, fetch, respond, requests } = setup(apiVersion);
            vi.spyOn(internals.getOpenAIProtocolDriver(), 'getDriverFetch').mockReturnValue(fetch);
            respond.mockImplementation(() => {
                return new Response(JSON.stringify({ data: [{ b64_json: 'YQ==' }], output_format: 'webp' }), {
                    headers: { 'content-type': 'application/json' },
                });
            });
            try {
                const result = await driver.execute([{ role: PromptRole.user, content: 'A garden' }], {
                    model: 'garden::gpt-image-2.5-flare',
                    model_options: {
                        _option_id: 'openai-gpt-image',
                        width: 2048,
                        height: 1024,
                        image_quality: 'max',
                        output_format: 'webp',
                    },
                });
                expect(requests[0].url).toBe(
                    'https://foundry.example.test/openai/v1/images/generations' +
                        (apiVersion ? `?api-version=${apiVersion}` : ''),
                );
                expect(requests[0].body).toMatchObject({ model: 'garden', size: '2048x1024', quality: 'max' });
                expect(requests[0].headers.get('authorization')).toBe('Bearer test-token');
                expect(result.result).toEqual([{ type: 'image', value: 'data:image/webp;base64,YQ==' }]);
            } finally {
                driver.destroy();
            }
        },
    );
});
