import type { TokenCredential } from '@azure/identity';
import { parseConversationDocument } from '@llumiverse/conversation';
import { PromptRole } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { exposePrivate } from '../../test/__helpers__/test-utils.js';
import type { OpenAIChatCompletionsPayload } from '../openai/openai_chat_completions.js';
import { prepareOpenAIChatCanonicalState } from '../openai/openai-chat-conversation-adapter.js';
import { prepareOpenAIResponsesCanonicalState } from '../openai/openai-responses-conversation-adapter.js';
import { AzureFoundryDriver, toAzureInferenceRequest } from './azure_foundry.js';

const credential: TokenCredential = {
    getToken: vi.fn(async () => ({ token: 'test-token', expiresOnTimestamp: Date.now() + 60_000 })),
};

type FoundryInternals = {
    inferenceClient: object;
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

function canonicalOptions(model: string, id: string, operation = 'next') {
    return {
        model,
        conversation_runtime: {
            conversation_id: id,
            request_id: `${id}:${operation}:request`,
            attempt_id: `${id}:${operation}:attempt`,
            input_operation_id: `${id}:${operation}:input`,
            response_operation_id: `${id}:${operation}:response`,
            recorded_at: '2026-09-12T00:00:00.000Z',
        },
    };
}

describe('AzureFoundryDriver protocol composition', () => {
    it('accepts canonical history through the parent Chat execution path', async () => {
        const driver = createDriver();
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
        } as unknown as AzureFoundryDriver['service'];
        const post = vi.fn(async () => ({
            status: '200',
            body: {
                id: 'foundry-chat-canonical',
                created: 1,
                model: 'llama-deployment',
                choices: [
                    {
                        index: 0,
                        finish_reason: 'stop',
                        message: { role: 'assistant', content: 'chat answer' },
                    },
                ],
                usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
            },
        }));
        const inferenceAdapter = exposePrivate<FoundryInternals>(driver).inferenceProtocolDriver;
        Object.defineProperty(inferenceAdapter, 'service', { value: { path: vi.fn(() => ({ post })) } });
        const model = 'llama-deployment::llama';
        const id = 'foundry-chat';
        const state = await prepareOpenAIChatCanonicalState({
            conversation: { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'prior' }] },
            prompt: { _is_openai_chat_completions: true, messages: [] },
            options: canonicalOptions(model, id, 'seed'),
            provider: 'azure_foundry',
        });

        const completion = await driver.execute([{ role: PromptRole.user, content: 'next' }], {
            ...canonicalOptions(model, id),
            conversation: state.document,
        });

        expect(parseConversationDocument(completion.conversation).id).toBe(id);
        expect(post).toHaveBeenCalledOnce();
    });

    it('preserves tool result status through Chat ingestion without sending private evidence', async () => {
        const driver = createDriver();
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
        } as unknown as AzureFoundryDriver['service'];
        const responses = [
            {
                id: 'foundry-tool-call',
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
                                    id: 'call:lookup',
                                    type: 'function',
                                    function: { name: 'lookup', arguments: '{"city":"Tokyo"}' },
                                },
                            ],
                        },
                    },
                ],
                usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
            },
            {
                id: 'foundry-tool-result',
                created: 2,
                model: 'llama-deployment',
                choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'unavailable' } }],
                usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 },
            },
        ];
        const post = vi.fn(async (_request: unknown) => ({ status: '200', body: responses.shift() }));
        const inferenceAdapter = exposePrivate<FoundryInternals>(driver).inferenceProtocolDriver;
        Object.defineProperty(inferenceAdapter, 'service', { value: { path: vi.fn(() => ({ post })) } });
        const model = 'llama-deployment::llama';
        const id = 'foundry-tool-status';

        const first = await driver.execute([{ role: PromptRole.user, content: 'Weather?' }], {
            ...canonicalOptions(model, id, 'ask'),
            tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
        });
        const second = await driver.execute(
            [
                {
                    role: PromptRole.tool,
                    content: '{"error":"offline"}',
                    tool_use_id: 'call:lookup',
                    tool_result_status: 'error',
                },
            ],
            {
                ...canonicalOptions(model, id, 'answer'),
                conversation: first.conversation,
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
        );

        const persisted = parseConversationDocument(second.conversation);
        expect(persisted.turns.find((turn) => turn.kind === 'tool')?.blocks[0]).toMatchObject({
            type: 'tool_result',
            call_id: 'call:lookup',
            status: 'error',
        });
        expect(JSON.stringify(post.mock.calls[1]?.[0])).not.toContain('tool_result_status');
        expect(JSON.stringify(post.mock.calls[1]?.[0])).not.toContain('_llumiverse_tool_result_status');
    });

    it('accepts canonical history through the parent Responses streaming path', async () => {
        const driver = createDriver();
        const model = 'gpt-deployment::gpt-5';
        const id = 'foundry-responses';
        const response = {
            id: 'foundry-response-canonical',
            object: 'response',
            created_at: 1,
            model: 'gpt-deployment',
            status: 'completed',
            output: [
                {
                    id: 'message-canonical',
                    type: 'message',
                    role: 'assistant',
                    status: 'completed',
                    content: [{ type: 'output_text', text: 'response answer', annotations: [], logprobs: [] }],
                },
            ],
            output_text: 'response answer',
            error: null,
            incomplete_details: null,
            instructions: null,
            metadata: {},
            parallel_tool_calls: true,
            temperature: 1,
            tool_choice: 'auto',
            tools: [],
            top_p: 1,
            usage: {
                input_tokens: 2,
                input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
                output_tokens: 2,
                output_tokens_details: { reasoning_tokens: 0 },
                total_tokens: 4,
            },
        } satisfies OpenAI.Responses.Response;
        const create = vi.fn((_request: OpenAI.Responses.ResponseCreateParamsStreaming, _options?: unknown) =>
            Promise.resolve(
                (async function* () {
                    yield { type: 'response.completed', sequence_number: 1, response };
                })(),
            ),
        );
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'OpenAI' })) },
            getOpenAIClient: vi.fn(() => ({ responses: { create } })),
        } as unknown as AzureFoundryDriver['service'];
        const state = await prepareOpenAIResponsesCanonicalState({
            conversation: [{ type: 'message', role: 'user', content: 'prior' }],
            prompt: [],
            options: canonicalOptions(model, id, 'seed'),
            provider: 'azure_foundry',
        });

        const stream = await driver.stream([{ role: PromptRole.user, content: 'next' }], {
            ...canonicalOptions(model, id),
            conversation: state.document,
        });
        for await (const _chunk of stream) {
            // Consume the parent stream so terminal canonical finalization runs.
        }

        const conversation = parseConversationDocument(stream.completion?.conversation);
        expect(conversation.id).toBe(id);
        expect(Object.values(conversation.generations)).toEqual(
            expect.arrayContaining([expect.objectContaining({ requested_model: model })]),
        );
        expect(create.mock.calls[0]?.[0]).toEqual(expect.objectContaining({ model: 'gpt-deployment', stream: true }));
        expect(create).toHaveBeenCalledOnce();
    });

    it('preserves required tool choice when adapting an OpenAI chat request', () => {
        const body = toAzureInferenceRequest(
            {
                model: 'deployment',
                messages: [{ role: 'user', content: 'Act now.' }],
                tools: [
                    {
                        type: 'function',
                        function: { name: 'write_artifact', parameters: { type: 'object', properties: {} } },
                    },
                ],
                tool_choice: { type: 'function', function: { name: 'write_artifact' } },
                parallel_tool_calls: false,
                stream: false,
            } satisfies OpenAIChatCompletionsPayload,
            false,
        );

        expect(body.tool_choice).toEqual({ type: 'function', function: { name: 'write_artifact' } });
        expect(body.parallel_tool_calls).toBe(false);
    });
    it('parses string capability flags and excludes dedicated endpoint deployments from inference listing', async () => {
        const driver = createDriver();
        const deployments = [
            modelDeployment('future-chat', 'Future-Chat-7', {}),
            modelDeployment('explicit-chat', 'Llama-5', { chat_completion: 'true' }),
            modelDeployment('not-chat', 'Llama-4', { chat_completion: 'false' }),
            modelDeployment('embedding', 'text-embedding-4', { chat_completion: 'true' }),
            modelDeployment('speech', 'gpt-4o-mini-tts', { chat_completion: 'true' }),
        ];
        driver.service = {
            deployments: {
                list: () => ({
                    async *byPage() {
                        yield deployments;
                    },
                }),
            },
        } as unknown as AzureFoundryDriver['service'];

        expect((await driver.listModels()).map((model) => model.id)).toEqual([
            'explicit-chat::Llama-5',
            'future-chat::Future-Chat-7',
        ]);
    });

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
                _option_id: 'azure-foundry-chat',
                max_tokens: 16,
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop_sequence: ['END'],
                seed: 42,
            },
            tools: [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } }],
        });

        expect(path).toHaveBeenCalledWith('/chat/completions');
        expect(post).toHaveBeenCalledWith({
            timeout: 900_000,
            headers: { 'extra-parameters': 'pass-through' },
            body: expect.objectContaining({
                model: 'llama-deployment',
                stream: false,
                max_tokens: 16,
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop: ['END'],
                seed: 42,
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
            model_options: {
                _option_id: 'openai-thinking' as const,
                effort: 'high' as const,
                temperature: 0.3,
                top_p: 0.7,
            },
            tools: [{ name: 'lookup', input_schema: { type: 'object' as const } }],
            result_schema: {
                type: 'object' as const,
                properties: { answer: { type: 'string' as const } },
                required: ['answer'],
            },
        };

        await expect(driver.requestTextCompletion(prompt, options)).resolves.toEqual(
            expect.objectContaining({ result: [{ type: 'text', value: 'ok' }] }),
        );
        await expect(driver.requestTextCompletion(prompt, options)).resolves.toEqual(
            expect.objectContaining({ result: [{ type: 'text', value: 'ok' }] }),
        );
        expect(deploymentGet).toHaveBeenCalledOnce();
        expect(getOpenAIClient).toHaveBeenCalledOnce();
        expect(getOpenAIClient).toHaveBeenCalledWith({
            fetch: expect.any(Function),
            timeout: 900_000,
        });
        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'gpt-deployment',
                stream: false,
                reasoning: { effort: 'high', summary: 'auto' },
                include: ['reasoning.encrypted_content'],
                temperature: undefined,
                top_p: undefined,
                tools: [expect.objectContaining({ type: 'function', name: 'lookup' })],
                text: expect.objectContaining({
                    format: expect.objectContaining({ type: 'json_schema', name: 'format_output' }),
                }),
            }),
        );
        expect(
            driver.buildStreamingConversation(prompt, [{ type: 'text', value: 'streamed' }], undefined, options),
        ).toEqual(
            expect.objectContaining({
                _arrayConversation: expect.arrayContaining([
                    expect.objectContaining({ type: 'message', role: 'assistant', content: 'streamed' }),
                ]),
            }),
        );
    });

    it('delegates Chat streaming conversation reconstruction and result cleanup', async () => {
        const driver = createDriver();
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
        } as unknown as AzureFoundryDriver['service'];
        const options = {
            model: 'llama-deployment::llama',
            tools: [{ name: 'lookup', input_schema: { type: 'object' as const } }],
        };
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Weather?' }], options);
        await driver.isOpenAIDeployment(options.model);

        const conversation = driver.buildStreamingConversation(
            prompt,
            [{ type: 'text', value: '<think>hidden</think>Checking' }],
            [{ id: 'call_1', tool_name: 'lookup', tool_input: { city: 'Paris' } }],
            options,
        );
        const completion = { result: [{ type: 'text' as const, value: '<think>hidden</think>Answer' }] };
        driver.validateResult(completion, options);

        expect(conversation).toEqual(
            expect.objectContaining({
                _is_openai_chat_completions: true,
                messages: expect.arrayContaining([
                    expect.objectContaining({
                        role: 'assistant',
                        content: 'Checking',
                        tool_calls: [
                            {
                                id: 'call_1',
                                type: 'function',
                                function: { name: 'lookup', arguments: '{"city":"Paris"}' },
                            },
                        ],
                    }),
                ]),
            }),
        );
        expect(completion.result).toEqual([{ type: 'text', value: '<think>hidden</think>Answer' }]);
    });

    it('preserves Azure HTTP status and retryability in LlumiverseError', async () => {
        const driver = createDriver();
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
        } as unknown as AzureFoundryDriver['service'];
        const post = vi.fn(async () => ({
            status: '503',
            body: { error: { code: 'ServiceUnavailable', message: 'Temporarily unavailable' } },
        }));
        const inferenceAdapter = exposePrivate<FoundryInternals>(driver).inferenceProtocolDriver;
        Object.defineProperty(inferenceAdapter, 'service', { value: { path: vi.fn(() => ({ post })) } });

        await expect(
            driver.execute([{ role: PromptRole.user, content: 'Hello' }], {
                model: 'llama-deployment::llama',
            }),
        ).rejects.toMatchObject({
            name: 'AzureFoundryHTTPError',
            code: 503,
            retryable: true,
            originalError: expect.objectContaining({
                status: 503,
                body: { error: { code: 'ServiceUnavailable', message: 'Temporarily unavailable' } },
            }),
        });
    });

    it('preserves Azure embedding HTTP status and retryability in LlumiverseError', async () => {
        const driver = createDriver();
        const post = vi.fn(async () => ({
            status: '503',
            body: { error: { code: 'ServiceUnavailable', message: 'Temporarily unavailable' } },
            headers: { get: vi.fn(() => undefined) },
            request: { url: 'https://foundry.example.test/embeddings' },
        }));
        Object.defineProperty(exposePrivate<FoundryInternals>(driver), 'inferenceClient', {
            value: { path: vi.fn(() => ({ post })) },
        });

        await expect(
            driver.generateEmbeddings({
                model: 'embedding-deployment::embedding-model',
                inputs: [{ type: 'text', text: 'Hello' }],
            }),
        ).rejects.toMatchObject({
            name: 'AzureFoundryHTTPError',
            code: 503,
            retryable: true,
            context: {
                provider: driver.provider,
                model: 'embedding-deployment::embedding-model',
                operation: 'execute',
            },
            originalError: expect.objectContaining({
                status: 503,
                body: { error: { code: 'ServiceUnavailable', message: 'Temporarily unavailable' } },
            }),
        });
    });
});

function modelDeployment(name: string, modelName: string, capabilities: Record<string, string>) {
    return {
        type: 'ModelDeployment' as const,
        name,
        modelName,
        modelVersion: '1',
        modelPublisher: 'Meta',
        capabilities,
    };
}
