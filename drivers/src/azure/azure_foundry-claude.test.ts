import type { AnthropicFoundry } from '@anthropic-ai/foundry-sdk';
import type { Message } from '@anthropic-ai/sdk/resources/messages.js';
import type { TokenCredential } from '@azure/identity';
import { Base64DataSource, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { exposePrivate } from '../../test/__helpers__/test-utils.js';
import type { ClaudePrompt } from '../shared/claude-messages.js';
import { AzureFoundryDriver } from './azure_foundry.js';

type Internals = {
    getDriverFetch: () => typeof fetch;
    getAnthropicClient: () => AnthropicFoundry;
};

function response(content: Message['content'] = [{ type: 'text', text: 'Green', citations: null }]) {
    const events: object[] = [
        {
            type: 'message_start',
            message: {
                id: 'msg-test',
                type: 'message',
                role: 'assistant',
                model: 'deployment',
                content: [],
                stop_reason: null,
                stop_sequence: null,
                usage: {
                    input_tokens: 3,
                    output_tokens: 0,
                    cache_read_input_tokens: 2,
                    cache_creation_input_tokens: 1,
                },
            },
        },
    ];
    content.forEach((block, index) => {
        events.push({
            type: 'content_block_start',
            index,
            content_block: block.type === 'text' ? { ...block, text: '' } : block,
        });
        if (block.type === 'text') {
            events.push({ type: 'content_block_delta', index, delta: { type: 'text_delta', text: block.text } });
        }
        events.push({ type: 'content_block_stop', index });
    });
    events.push(
        {
            type: 'message_delta',
            delta: {
                stop_reason: content.some((block) => block.type === 'tool_use') ? 'tool_use' : 'end_turn',
                stop_sequence: null,
            },
            usage: { output_tokens: 4 },
        },
        { type: 'message_stop' },
    );
    return new Response(
        events
            .map((event) => `event: ${(event as { type: string }).type}\ndata: ${JSON.stringify(event)}\n\n`)
            .join(''),
        {
            headers: { 'content-type': 'text/event-stream' },
        },
    );
}

function setup(sourceModel?: string) {
    const getToken = vi.fn<TokenCredential['getToken']>(async () => ({
        token: 'test-token',
        expiresOnTimestamp: Date.now() + 3600000,
    }));
    const driver = new AzureFoundryDriver({
        endpoint: 'https://foundry.example.test/api/projects/project',
        azureADTokenProvider: { getToken },
        apiVersion: 'openai-version',
        sourceModel,
    });
    const requests: { url: string; body: Record<string, unknown>; headers: Headers; signal?: AbortSignal | null }[] =
        [];
    const respond = vi.fn(() => response());
    const fetch = vi.fn<typeof globalThis.fetch>(async (url, init) => {
        requests.push({
            url: String(url),
            body: JSON.parse(String(init?.body)),
            headers: new Headers(init?.headers),
            signal: init?.signal,
        });
        return respond();
    });
    const internals = exposePrivate<Internals>(driver);
    vi.spyOn(internals, 'getDriverFetch').mockReturnValue(fetch);
    const metadata = vi.spyOn(driver.service.deployments, 'get').mockResolvedValue({
        type: 'ModelDeployment',
        name: 'deployment',
        modelName: 'claude-opus-5-5',
        modelPublisher: 'Anthropic',
        modelVersion: '2',
        capabilities: { chat_completion: 'false' },
    });
    return { driver, internals, requests, fetch, respond, getToken, metadata };
}

const segments = [{ role: PromptRole.user, content: 'What color is grass?' }];

describe('Foundry Claude Messages', () => {
    it.each(['deployment::claude-opus-5-5', 'deployment::claude-sonnet-5-5-20260901', 'deployment::claude-opus-6'])(
        'uses deployment names, native transport and source model rules for %s',
        async (model) => {
            const { driver, requests, metadata, getToken, internals } = setup();
            try {
                const result = await driver.execute(segments, {
                    model,
                    include_original_response: true,
                    model_options: { _option_id: 'anthropic-claude', cache_enabled: true },
                });
                expect(metadata).not.toHaveBeenCalled();
                expect(requests[0].url).toBe('https://foundry.example.test/anthropic/v1/messages');
                expect(requests[0].headers.get('authorization')).toBe('Bearer test-token');
                expect(requests[0].headers.get('anthropic-version')).toBe('2023-06-01');
                expect(requests[0].headers.has('x-ms-oai-image-generation-deployment')).toBe(false);
                expect(requests[0].body).toMatchObject({ model: 'deployment', stream: true });
                expect(requests[0].body).not.toHaveProperty('response_format');
                expect(requests[0].body).not.toHaveProperty('temperature');
                expect(getToken).toHaveBeenCalledWith(['https://ai.azure.com/.default'], expect.anything());
                expect(internals.getAnthropicClient().maxRetries).toBe(0);
                expect(result.result).toContainEqual({ type: 'text', value: 'Green' });
                expect(result.original_response).toMatchObject({ id: 'msg-test' });
                expect(result.token_usage).toMatchObject({
                    prompt_new: 3,
                    prompt: 6,
                    result: 4,
                    prompt_cached: 2,
                    prompt_cache_write: 1,
                });
            } finally {
                driver.destroy();
            }
        },
    );

    it.each([undefined, 'claude-opus-5-5'])('resolves opaque names with source hint %s', async (source) => {
        const { driver, metadata, requests } = setup(source);
        try {
            await driver.execute(segments, { model: 'deployment' });
            await driver.execute(segments, { model: 'deployment' });
            expect(metadata).toHaveBeenCalledTimes(source ? 0 : 1);
            expect(requests.every((request) => request.body.model === 'deployment')).toBe(true);
        } finally {
            driver.destroy();
        }
    });

    it.each([
        ['gpt-replacement', undefined],
        ['gpt-image-replacement', undefined],
        ['gpt-replacement', 'claude-opus-6'],
        ['gpt-image-replacement', 'claude-opus-6'],
    ] as const)('uses the Claude source rather than deployment name %s with hint %s', async (model, source) => {
        const { driver, metadata, requests } = setup(source);
        try {
            const result = await driver.execute(segments, { model });
            expect(result.prompt).toHaveProperty('messages');
            const stream = await driver.stream(segments, { model });
            const chunks = [];
            for await (const chunk of stream) chunks.push(chunk);
            expect(chunks.join('')).toContain('Green');
            expect(stream.completion?.prompt).toHaveProperty('messages');
            expect(metadata).toHaveBeenCalledTimes(source ? 0 : 1);
            expect(requests).toHaveLength(2);
            expect(requests.every((request) => request.body.model === model)).toBe(true);
            expect(requests.every((request) => !('temperature' in request.body))).toBe(true);
            expect(requests.every((request) => request.url.endsWith('/anthropic/v1/messages'))).toBe(true);
        } finally {
            driver.destroy();
        }
    });

    it('gives a qualified source priority over a conflicting driver hint', async () => {
        const { driver, metadata, requests } = setup('gpt-5');
        try {
            await driver.execute(segments, { model: 'gpt-replacement::claude-opus-6' });
            expect(metadata).not.toHaveBeenCalled();
            expect(requests[0].body.model).toBe('gpt-replacement');
            expect(requests[0].url).toContain('/anthropic/v1/messages');
        } finally {
            driver.destroy();
        }
    });

    it('uses an OpenAI source hint for a Claude-named deployment', async () => {
        const { driver, metadata } = setup('gpt-5');
        metadata.mockResolvedValue({
            type: 'ModelDeployment',
            name: 'claude-replacement',
            modelName: 'gpt-5',
            modelPublisher: 'OpenAI',
            modelVersion: '1',
            capabilities: { chat_completion: 'true' },
        });
        try {
            expect(await driver.createPrompt(segments, { model: 'claude-replacement' })).toBeInstanceOf(Array);
            expect(await driver.isOpenAIDeployment('claude-replacement')).toBe(true);
            expect(metadata).toHaveBeenCalledOnce();
        } finally {
            driver.destroy();
        }
    });

    it.each(['execute', 'stream', 'createPrompt'] as const)(
        'cancels publisher discovery during %s without caching the failed lookup',
        async (operation) => {
            const { driver, metadata, requests } = setup();
            const controller = new AbortController();
            metadata.mockImplementationOnce(
                async (_name, options) =>
                    new Promise((_resolve, reject) => {
                        options?.abortSignal?.addEventListener('abort', () => reject(controller.signal.reason), {
                            once: true,
                        });
                    }),
            );
            const reason = new DOMException('Cancelled', 'AbortError');
            try {
                const pending = driver[operation](
                    segments,
                    { model: 'gpt-replacement', httpTimeout: { headersTimeout: 40, bodyTimeout: 40 } },
                    controller.signal,
                );
                const rejected = expect(pending).rejects.toBe(reason);
                await vi.waitFor(() => expect(metadata).toHaveBeenCalledOnce());
                expect(metadata).toHaveBeenCalledWith('gpt-replacement', {
                    abortSignal: controller.signal,
                    requestOptions: { timeout: 40 },
                });
                controller.abort(reason);
                await rejected;
                expect(requests).toHaveLength(0);
                await driver.execute(segments, { model: 'gpt-replacement' });
                expect(metadata).toHaveBeenCalledTimes(2);
                expect(requests).toHaveLength(1);
            } finally {
                controller.abort();
                driver.destroy();
            }
        },
    );

    it.each(['execute', 'stream', 'createPrompt'] as const)(
        'skips discovery for an already aborted %s',
        async (operation) => {
            const { driver, metadata, requests } = setup();
            const controller = new AbortController();
            controller.abort();
            try {
                await expect(driver[operation](segments, { model: 'deployment' }, controller.signal)).rejects.toBe(
                    controller.signal.reason,
                );
                expect(metadata).not.toHaveBeenCalled();
                expect(requests).toHaveLength(0);
            } finally {
                driver.destroy();
            }
        },
    );

    it('does not cache failed publisher lookups', async () => {
        const { driver, metadata } = setup();
        metadata.mockRejectedValueOnce(new Error('lookup failed'));
        try {
            await expect(driver.createPrompt(segments, { model: 'deployment' })).rejects.toThrow('lookup failed');
            await driver.execute(segments, { model: 'deployment' });
            expect(metadata).toHaveBeenCalledTimes(2);
        } finally {
            driver.destroy();
        }
    });

    it('uses Claude schema guidance and handles ordered image references', async () => {
        const { driver, requests, respond } = setup();
        respond.mockImplementation(() => response([{ type: 'text', text: '{"color":"green"}', citations: null }]));
        try {
            await driver.execute(
                [
                    {
                        role: PromptRole.user,
                        content: 'Describe these',
                        files: [
                            new Base64DataSource('first.png', 'image/png', 'AQID'),
                            new Base64DataSource('second.png', 'image/png', 'BAUG'),
                        ],
                    },
                ],
                {
                    model: 'deployment::claude-opus-5-5',
                    result_schema: { type: 'object', properties: { color: { type: 'string' } }, required: ['color'] },
                },
            );
            expect(JSON.stringify(requests[0].body.system)).toContain('color');
            expect(requests[0].body).not.toHaveProperty('output_config.format');
            const prompt = requests[0].body.messages as ClaudePrompt['messages'];
            expect(prompt[0].content).toEqual(
                expect.arrayContaining([
                    expect.objectContaining({ type: 'image', source: expect.objectContaining({ data: 'AQID' }) }),
                    expect.objectContaining({ type: 'image', source: expect.objectContaining({ data: 'BAUG' }) }),
                ]),
            );
        } finally {
            driver.destroy();
        }
    });

    it('preserves signed thinking and tool results across turns', async () => {
        const { driver, respond, requests } = setup();
        respond.mockImplementationOnce(() =>
            response([
                { type: 'thinking', thinking: 'Consider the color', signature: 'signed-reasoning' },
                {
                    type: 'tool_use',
                    caller: { type: 'direct' },
                    id: 'call-1',
                    name: 'lookup',
                    input: { object: 'grass' },
                },
            ]),
        );
        const options = {
            model: 'deployment::claude-opus-5-5',
            model_options: { _option_id: 'anthropic-claude' as const, include_thoughts: true, cache_enabled: true },
            tools: [
                {
                    name: 'lookup',
                    input_schema: { type: 'object' as const, properties: { object: { type: 'string' } } },
                },
            ],
            stripImagesAfterTurns: 2,
        };
        try {
            const first = await driver.execute(segments, options);
            expect(first.tool_use?.[0]).toMatchObject({ tool_name: 'lookup', tool_input: { object: 'grass' } });
            expect(first.result).toContainEqual({ type: 'thoughts', value: 'Consider the color' });
            const continuation: ClaudePrompt = {
                messages: [
                    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'green' }] },
                ],
            };
            await driver.requestTextCompletion(continuation, { ...options, conversation: first.conversation });
            expect(JSON.stringify(requests[1].body)).toContain('signed-reasoning');
            expect(JSON.stringify(requests[1].body)).toContain('tool_result');
            expect(JSON.stringify(requests[1].body)).toContain('cache_control');
        } finally {
            driver.destroy();
        }
    });

    it('streams and finalizes Claude conversations', async () => {
        const { driver } = setup();
        try {
            const options = { model: 'deployment::claude-opus-5-5' };
            const prompt = await driver.createPrompt(segments, options);
            const stream = await driver.requestTextCompletionStream(prompt, options);
            const chunks = [];
            for await (const chunk of stream) chunks.push(chunk);
            expect(chunks.some((chunk) => chunk.result.length > 0)).toBe(true);
            expect(await stream.finalizeConversation?.()).toMatchObject({
                messages: expect.arrayContaining([expect.objectContaining({ role: 'assistant' })]),
            });
            expect(driver.formatDebugPrompt(prompt)).toHaveProperty('messages');
        } finally {
            driver.destroy();
        }
    });

    it('normalizes provider errors without retries', async () => {
        const { driver, respond, fetch } = setup();
        respond.mockImplementation(
            () =>
                new Response(
                    JSON.stringify({ type: 'error', error: { type: 'permission_error', message: 'Access denied' } }),
                    { status: 403, headers: { 'content-type': 'application/json' } },
                ),
        );
        try {
            await expect(driver.execute(segments, { model: 'deployment::claude-opus-5-5' })).rejects.toMatchObject({
                code: 403,
                context: expect.objectContaining({ provider: 'azure_foundry' }),
            });
            expect(fetch).toHaveBeenCalledOnce();
        } finally {
            driver.destroy();
        }
    });

    it('enforces a request deadline without retrying', async () => {
        const { driver, fetch } = setup();
        fetch.mockImplementation(
            async (_url, init) =>
                new Promise((_resolve, reject) => {
                    init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {
                        once: true,
                    });
                }),
        );
        try {
            await expect(
                driver.execute(segments, {
                    model: 'deployment::claude-opus-5-5',
                    httpTimeout: { headersTimeout: 20, bodyTimeout: 20 },
                }),
            ).rejects.toMatchObject({ name: 'APIConnectionTimeoutError', retryable: true });
            expect(fetch).toHaveBeenCalledOnce();
        } finally {
            driver.destroy();
        }
    });

    it('aborts the SDK response when a streaming consumer exits early', async () => {
        const { driver, fetch } = setup();
        let signal: AbortSignal | null | undefined;
        fetch.mockImplementation(async (_url, init) => {
            signal = init?.signal;
            const event = {
                type: 'message_start',
                message: {
                    id: 'msg-test',
                    type: 'message',
                    role: 'assistant',
                    model: 'deployment',
                    content: [],
                    stop_reason: null,
                    stop_sequence: null,
                    usage: { input_tokens: 1, output_tokens: 0 },
                },
            };
            return new Response(
                new ReadableStream({
                    start(controller) {
                        controller.enqueue(
                            new TextEncoder().encode(`event: message_start\ndata: ${JSON.stringify(event)}\n\n`),
                        );
                    },
                }),
                { headers: { 'content-type': 'text/event-stream' } },
            );
        });
        try {
            const options = { model: 'deployment::claude-opus-5-5' };
            const prompt = await driver.createPrompt(segments, options);
            for await (const _chunk of await driver.requestTextCompletionStream(prompt, options)) break;
            await vi.waitFor(() => expect(signal?.aborted).toBe(true));
        } finally {
            driver.destroy();
        }
    });

    it('propagates cancellation and per-request deadlines to the SDK', async () => {
        const { driver, fetch, requests, internals } = setup();
        fetch.mockImplementation(
            async (_url, init) =>
                new Promise((_resolve, reject) => {
                    init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {
                        once: true,
                    });
                }),
        );
        const controller = new AbortController();
        try {
            const options = { model: 'deployment::claude-opus-5-5' };
            const prompt = await driver.createPrompt(segments, options);
            const post = vi.spyOn(internals.getAnthropicClient().messages, 'stream');
            const pending = driver.requestTextCompletion(
                prompt,
                { ...options, httpTimeout: { headersTimeout: 1000, bodyTimeout: 1000 } },
                controller.signal,
            );
            const rejected = expect(pending).rejects.toThrow();
            await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());
            expect(post).toHaveBeenCalledWith(
                expect.anything(),
                expect.objectContaining({ signal: controller.signal, timeout: 1000 }),
            );
            controller.abort();
            await rejected;
            expect(requests).toHaveLength(0);
        } finally {
            controller.abort();
            driver.destroy();
        }
    });
});
