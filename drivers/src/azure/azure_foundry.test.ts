import type { TokenCredential } from '@azure/identity';
import { PromptRole } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { exposePrivate } from '../../test/__helpers__/test-utils.js';
import { AzureFoundryDriver } from './azure_foundry.js';

const credential: TokenCredential = {
    getToken: vi.fn(async () => ({ token: 'test-token', expiresOnTimestamp: Date.now() + 60_000 })),
};

type FoundryInternals = {
    inferenceProtocolDriver: {
        service: object;
    };
};

function createDriver(): AzureFoundryDriver {
    return new AzureFoundryDriver({
        endpoint: 'https://foundry.example.test',
        azureADTokenProvider: credential,
    });
}

describe('AzureFoundryDriver protocol composition', () => {
    it('does not cache failed deployment lookups or silently route them to Chat', async () => {
        const driver = createDriver();
        const get = vi
            .fn()
            .mockRejectedValueOnce(new Error('lookup failed'))
            .mockResolvedValue({ modelPublisher: 'OpenAI' });
        driver.service = { deployments: { get } } as unknown as AzureFoundryDriver['service'];

        await expect(driver.isOpenAIDeployment('deployment::gpt-5')).rejects.toThrow('lookup failed');
        await expect(driver.isOpenAIDeployment('deployment::gpt-5')).resolves.toBe(true);
        await expect(driver.isOpenAIDeployment('deployment::gpt-5')).resolves.toBe(true);
        expect(get).toHaveBeenCalledTimes(2);
    });

    it('uses shared Chat behavior for non-OpenAI deployments and sends stream false', async () => {
        const driver = createDriver();
        const deploymentGet = vi.fn(async () => ({ modelPublisher: 'Meta' }));
        driver.service = { deployments: { get: deploymentGet } } as unknown as AzureFoundryDriver['service'];
        const nativeResponse = {
            id: 'foundry-1',
            created: 1,
            model: 'llama-deployment',
            choices: [
                {
                    index: 0,
                    finish_reason: 'tool_calls',
                    message: {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            {
                                id: 'call_1',
                                type: 'function',
                                function: { name: 'lookup', arguments: '{"city":"Paris"}' },
                            },
                        ],
                    },
                },
            ],
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
        };
        const post = vi.fn(async () => ({ status: '200', body: nativeResponse }));
        const path = vi.fn(() => ({ post }));
        const inferenceAdapter = exposePrivate<FoundryInternals>(driver).inferenceProtocolDriver;
        Object.defineProperty(inferenceAdapter, 'service', { value: { path } });
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Weather?' }], {
            model: 'llama-deployment::llama',
        });

        const completion = await driver.requestTextCompletion(prompt, {
            model: 'llama-deployment::llama',
            include_original_response: true,
            model_options: {
                _option_id: 'text-fallback',
                max_tokens: 16,
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop_sequence: ['END'],
            },
            tools: [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } }],
        });

        expect(path).toHaveBeenCalledWith('/chat/completions');
        expect(post).toHaveBeenCalledWith({
            body: expect.objectContaining({
                model: 'llama-deployment',
                stream: false,
                max_tokens: 16,
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop: ['END'],
                tools: [
                    {
                        type: 'function',
                        function: {
                            name: 'lookup',
                            description: 'Lookup',
                            parameters: expect.objectContaining({ type: 'object' }),
                        },
                    },
                ],
            }),
        });
        expect(completion.tool_use?.[0]).toEqual({
            id: 'call_1',
            tool_name: 'lookup',
            tool_input: { city: 'Paris' },
        });
        expect(completion.original_response).toBe(nativeResponse);
    });

    it('memoizes the OpenAI Responses adapter and deployment decision', async () => {
        const driver = createDriver();
        const deploymentGet = vi.fn(async () => ({ modelPublisher: 'OpenAI' }));
        const response = {
            id: 'response-1',
            object: 'response',
            created_at: 1,
            model: 'gpt-deployment',
            status: 'completed',
            output: [
                {
                    id: 'message-1',
                    type: 'message',
                    role: 'assistant',
                    status: 'completed',
                    content: [{ type: 'output_text', text: 'ok', annotations: [], logprobs: [] }],
                },
            ],
            output_text: 'ok',
            error: null,
            incomplete_details: null,
            instructions: null,
            metadata: {},
            parallel_tool_calls: true,
            temperature: 1,
            tool_choice: 'auto',
            tools: [],
            top_p: 1,
            usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
        };
        const create = vi.fn(async () => response);
        const openAIClient = { responses: { create } } as unknown as OpenAI;
        const getOpenAIClient = vi.fn(() => openAIClient);
        driver.service = {
            deployments: { get: deploymentGet },
            getOpenAIClient,
        } as unknown as AzureFoundryDriver['service'];
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], {
            model: 'gpt-deployment::gpt-5',
        });
        const options = {
            model: 'gpt-deployment::gpt-5',
            model_options: { _option_id: 'text-fallback' as const },
        };

        await expect(driver.requestTextCompletion(prompt, options)).resolves.toEqual(
            expect.objectContaining({ result: [{ type: 'text', value: 'ok' }] }),
        );
        await expect(driver.requestTextCompletion(prompt, options)).resolves.toEqual(
            expect.objectContaining({ result: [{ type: 'text', value: 'ok' }] }),
        );
        expect(deploymentGet).toHaveBeenCalledOnce();
        expect(getOpenAIClient).toHaveBeenCalledOnce();
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ model: 'gpt-deployment', stream: false }));
    });
});
