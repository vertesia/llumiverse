import type { TokenCredential } from '@azure/identity';
import { PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { exposePrivate } from '../../test/__helpers__/test-utils.js';
import { AzureFoundryDriver } from './azure_foundry.js';

type Internals = { getDriverFetch: () => typeof fetch };

const ENDPOINT = 'https://foundry.example.test/api/projects/project';
const segments = [{ role: PromptRole.user, content: 'Hello' }];

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
    model: 'chat',
    choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'Hello' } }],
    usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
};

function claudeStream(): Response {
    const events = [
        {
            type: 'message_start',
            message: {
                id: 'msg-1',
                type: 'message',
                role: 'assistant',
                model: 'claude',
                content: [],
                stop_reason: null,
                stop_sequence: null,
                usage: { input_tokens: 2, output_tokens: 0 },
            },
        },
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } },
        { type: 'content_block_stop', index: 0 },
        { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 1 } },
        { type: 'message_stop' },
    ];
    return new Response(events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''), {
        headers: { 'content-type': 'text/event-stream' },
    });
}

function setup(options: { tokenProvider?: boolean; sourceModel?: string } = {}) {
    const getToken = vi.fn<TokenCredential['getToken']>(async () => ({
        token: 'test-token',
        expiresOnTimestamp: Date.now() + 3_600_000,
    }));
    const driver = new AzureFoundryDriver({
        endpoint: ENDPOINT,
        apiKey: 'test-key',
        ...(options.tokenProvider ? { azureADTokenProvider: { getToken } } : {}),
        sourceModel: options.sourceModel,
    });
    const requests: { url: string; method: string; body?: Record<string, unknown>; headers: Headers }[] = [];
    const fetch = vi.fn<typeof globalThis.fetch>(async (url, init) => {
        const target = String(url);
        requests.push({
            url: target,
            method: init?.method ?? 'GET',
            body: init?.body ? JSON.parse(String(init.body)) : undefined,
            headers: new Headers(init?.headers),
        });
        if (target.includes('/anthropic/')) return claudeStream();
        const body = target.endsWith('/responses')
            ? responsesBody
            : target.endsWith('/models')
              ? { object: 'list', data: [] }
              : chatBody;
        return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
    });
    vi.spyOn(exposePrivate<Internals>(driver), 'getDriverFetch').mockReturnValue(fetch);
    const deploymentGet = vi.spyOn(driver.service.deployments, 'get').mockResolvedValue({
        type: 'ModelDeployment',
        name: 'chat',
        modelPublisher: 'OpenAI',
        modelName: 'gpt-6-luna',
        modelVersion: '1',
        capabilities: { chat_completion: 'true' },
    });
    return { driver, requests, getToken, deploymentGet };
}

function expectApiKeyHeaders(headers: Headers) {
    expect(headers.get('api-key')).toBe('test-key');
    expect(headers.has('authorization')).toBe(false);
}

describe('Microsoft Foundry API key authentication', () => {
    it('sends Responses requests to the resource endpoint with the api-key header', async () => {
        const { driver, requests, getToken, deploymentGet } = setup();
        try {
            const result = await driver.execute(segments, { model: 'chat::gpt-6-luna' });
            expect(requests[0].url).toBe('https://foundry.example.test/openai/v1/responses');
            expect(requests[0].body).toMatchObject({ model: 'chat' });
            expectApiKeyHeaders(requests[0].headers);
            expect(result.result).toEqual([{ type: 'text', value: 'Hello' }]);
            expect(deploymentGet).not.toHaveBeenCalled();
            expect(getToken).not.toHaveBeenCalled();
        } finally {
            driver.destroy();
        }
    });

    it('sends partner models to Chat Completions with the api-key header', async () => {
        const { driver, requests, deploymentGet } = setup();
        try {
            await driver.execute(segments, { model: 'deepseek::DeepSeek-V4-Pro' });
            expect(requests[0].url).toBe('https://foundry.example.test/openai/v1/chat/completions');
            expect(requests[0].body).toMatchObject({ model: 'deepseek' });
            expectApiKeyHeaders(requests[0].headers);
            expect(deploymentGet).not.toHaveBeenCalled();
        } finally {
            driver.destroy();
        }
    });

    it('sends Claude requests with the x-api-key header', async () => {
        const { driver, requests, getToken } = setup();
        try {
            await driver.execute(segments, {
                model: 'claude::claude-sonnet-5-5',
                model_options: { _option_id: 'anthropic-claude' },
            });
            expect(requests[0].url).toBe('https://foundry.example.test/anthropic/v1/messages');
            expect(requests[0].headers.get('x-api-key')).toBe('test-key');
            expect(requests[0].headers.has('authorization')).toBe(false);
            expect(requests[0].body).toMatchObject({ model: 'claude' });
            expect(getToken).not.toHaveBeenCalled();
        } finally {
            driver.destroy();
        }
    });

    it.each([
        ['gpt-6-luna', 'responses'],
        ['gpt-5.3-codex', 'responses'],
        ['codex-mini', 'responses'],
        ['gpt-oss-120b', 'chat/completions'],
        ['grok-4.7', 'chat/completions'],
        ['my-llama', 'chat/completions'],
    ])('routes the bare deployment %s by its name', async (model, path) => {
        const { driver, requests, deploymentGet } = setup();
        try {
            await driver.execute(segments, { model });
            expect(requests[0].url).toBe(`https://foundry.example.test/openai/v1/${path}`);
            expect(deploymentGet).not.toHaveBeenCalled();
        } finally {
            driver.destroy();
        }
    });

    it('routes a bare deployment by the sourceModel option', async () => {
        const { driver, requests } = setup({ sourceModel: 'gpt-6-luna' });
        try {
            await driver.execute(segments, { model: 'production' });
            expect(requests[0].url).toBe('https://foundry.example.test/openai/v1/responses');
            expect(requests[0].body).toMatchObject({ model: 'production' });
        } finally {
            driver.destroy();
        }
    });

    it('validates the key against the resource model list and explains why listing needs Entra ID', async () => {
        const { driver, requests } = setup();
        try {
            await expect(driver.validateConnection()).resolves.toBe(true);
            expect(requests[0]).toMatchObject({ url: 'https://foundry.example.test/openai/v1/models', method: 'GET' });
            expectApiKeyHeaders(requests[0].headers);
            await expect(driver.listModels()).rejects.toThrow(/requires Microsoft Entra ID/);
        } finally {
            driver.destroy();
        }
    });

    it('keeps Entra ID for the project API when both credentials are set', async () => {
        const { driver, requests, deploymentGet, getToken } = setup({ tokenProvider: true });
        try {
            await driver.execute(segments, { model: 'chat' });
            expect(deploymentGet).toHaveBeenCalledOnce();
            expect(requests[0].url).toBe('https://foundry.example.test/openai/v1/responses');
            expectApiKeyHeaders(requests[0].headers);
            expect(getToken).not.toHaveBeenCalled();
        } finally {
            driver.destroy();
        }
    });
});
