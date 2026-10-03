import type { TokenCredential } from '@azure/identity';
import { type ConversationStreamEvent, parseConversationDocument } from '@llumiverse/conversation';
import { type CanonicalExecutionInputOptions, PromptRole } from '@llumiverse/core';
import OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { exposePrivate } from '../../test/__helpers__/test-utils.js';
import { type OpenAIChatCompletionsPayload, toOpenAINonStreamingPayload } from '../openai/openai_chat_completions.js';
import { prepareOpenAIChatCanonicalState } from '../openai/openai-chat-conversation-adapter.js';
import { prepareOpenAIResponsesCanonicalState } from '../openai/openai-responses-conversation-adapter.js';
import { AzureFoundryDriver } from './azure_foundry.js';

const credential: TokenCredential = {
    getToken: vi.fn(async () => ({ token: 'test-token', expiresOnTimestamp: Date.now() + 60_000 })),
};

type FoundryInternals = {
    getDriverFetch: () => typeof fetch;
    canStream: (options: import('@llumiverse/core').ExecutionOptions) => Promise<boolean>;
    getInferenceClient: () => OpenAI;
    getResourceClient: () => OpenAI;
    getInferenceProtocolDriver: () => { service: OpenAI };
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

async function collectCanonicalEvents(stream: AsyncIterable<ConversationStreamEvent>) {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('AzureFoundryDriver protocol composition', () => {
    it.each(['missing_runtime', 'cancelled'] as const)(
        'rejects %s typed audio before looking up the Foundry deployment',
        async (mode) => {
            const driver = createDriver();
            const get = vi.fn(async () => ({ modelPublisher: 'OpenAI' }));
            driver.service = { deployments: { get } } as unknown as AzureFoundryDriver['service'];
            const model = 'speech-deployment::gpt-4o-mini-tts';
            const options =
                mode === 'missing_runtime'
                    ? ({ model } as unknown as CanonicalExecutionInputOptions)
                    : canonicalOptions(model, 'audio-cancelled');

            await expect(
                driver.streamCanonicalEvents(
                    [{ role: PromptRole.user, content: 'hello' }],
                    options,
                    mode === 'cancelled' ? AbortSignal.abort(new Error('audio already cancelled')) : undefined,
                    { stream_id: `stream:foundry:preflight:${mode}` },
                ),
            ).rejects.toThrow(
                mode === 'missing_runtime'
                    ? 'Invalid input: expected object, received undefined'
                    : 'audio already cancelled',
            );
            expect(get).not.toHaveBeenCalled();
        },
    );

    it.each(['legacy', 'canonical'] as const)(
        'accepts canonical history through the parent Chat %s execution path',
        async (mode) => {
            const driver = createDriver();
            const nativeResponse = {
                id: 'foundry-chat-canonical',
                object: 'chat.completion',
                created: 1,
                model: 'llama-deployment',
                choices: [
                    {
                        index: 0,
                        finish_reason: 'stop',
                        logprobs: null,
                        message: { role: 'assistant', content: 'chat answer', refusal: null },
                    },
                ],
                usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
            } satisfies OpenAI.Chat.ChatCompletion;
            const requests: OpenAI.Chat.ChatCompletionCreateParams[] = [];
            const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
                requests.push(JSON.parse(String(init?.body)) as OpenAI.Chat.ChatCompletionCreateParams);
                return new Response(JSON.stringify(nativeResponse), {
                    headers: { 'content-type': 'application/json' },
                });
            });
            vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
            driver.service = {
                deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
                getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
            } as unknown as AzureFoundryDriver['service'];
            const model = 'llama-deployment::llama';
            const id = 'foundry-chat';
            const state = await prepareOpenAIChatCanonicalState({
                conversation: { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'prior' }] },
                prompt: { _is_openai_chat_completions: true, messages: [] },
                options: canonicalOptions(model, id, 'seed'),
                provider: 'azure_foundry',
            });

            const segments = [{ role: PromptRole.user, content: 'next' }];
            const options = { ...canonicalOptions(model, id), conversation: state.document };
            expect(await driver.supportsCanonicalExecution(options)).toBe(true);
            const completion =
                mode === 'canonical'
                    ? await driver.executeCanonical(segments, options)
                    : await driver.execute(segments, options);

            const document = parseConversationDocument(completion.conversation);
            expect(document.id).toBe(id);
            if (mode === 'canonical') {
                expect(Object.values(document.generations)).toContainEqual(
                    expect.objectContaining({
                        provider: 'azure_foundry',
                        requested_model: model,
                        resolved_model: 'llama-deployment',
                    }),
                );
                const retry = await driver.executeCanonical(segments, {
                    ...options,
                    conversation: JSON.parse(JSON.stringify(completion.conversation)),
                });
                expect(retry.conversation).toEqual(completion.conversation);
                await expect(
                    driver.executeCanonical(segments, {
                        ...options,
                        conversation: document,
                        model_options: { _option_id: 'text-fallback', temperature: 0.123 },
                    }),
                ).rejects.toThrow();
            }
            expect(requests).toEqual([expect.objectContaining({ model: 'llama-deployment' })]);
            expect(fetchMock).toHaveBeenCalledOnce();
        },
    );

    it('streams a canonical Chat response through the parent without losing deployment identity', async () => {
        const driver = createDriver();
        const chunk = {
            id: 'foundry-stream',
            object: 'chat.completion.chunk',
            model: 'llama-deployment',
            created: 1,
            choices: [
                {
                    index: 0,
                    delta: { role: 'assistant', content: 'stream answer' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
            usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
        } satisfies OpenAI.Chat.ChatCompletionChunk;
        const requests: OpenAI.Chat.ChatCompletionCreateParams[] = [];
        const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
            requests.push(JSON.parse(String(init?.body)) as OpenAI.Chat.ChatCompletionCreateParams);
            return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, {
                headers: { 'content-type': 'text/event-stream' },
            });
        });
        vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
        } as unknown as AzureFoundryDriver['service'];
        const options = canonicalOptions('llama-deployment::llama', 'foundry-stream');
        const stream = await driver.streamCanonical([{ role: PromptRole.user, content: 'hello' }], options);
        for await (const _chunk of stream) {
            // Completion becomes authoritative only after the native stream terminates.
        }
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({
                type: 'text',
                text: 'stream answer',
            }),
        );
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            requested_model: options.model,
            resolved_model: 'llama-deployment',
            provider: 'azure_foundry',
        });
        expect(requests).toEqual([expect.objectContaining({ model: 'llama-deployment', stream: true })]);
        expect(fetchMock).toHaveBeenCalledOnce();
    });

    it('delegates public typed canonical Chat events after the prepared-request barrier', async () => {
        const driver = createDriver();
        let prepared = false;
        const chunk = {
            id: 'foundry-typed-chat',
            object: 'chat.completion.chunk',
            model: 'llama-deployment',
            created: 1,
            choices: [
                {
                    index: 0,
                    delta: { role: 'assistant', content: 'typed answer' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
            usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
        } satisfies OpenAI.Chat.ChatCompletionChunk;
        const fetchMock = vi.fn(async () => {
            expect(prepared).toBe(true);
            return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, {
                headers: { 'content-type': 'text/event-stream' },
            });
        });
        vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
        } as unknown as AzureFoundryDriver['service'];
        const options = {
            ...canonicalOptions('llama-deployment::llama', 'foundry-typed-chat'),
            on_canonical_request_prepared: vi.fn(async () => {
                prepared = true;
            }),
        };

        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'hello' }],
            options,
            undefined,
            { stream_id: 'stream:foundry:chat' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events).toContainEqual(expect.objectContaining({ type: 'draft_text_delta', text: 'typed answer' }));
        expect(events.at(-1)).toMatchObject({ type: 'response_accepted', stream_id: 'stream:foundry:chat' });
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            provider: 'azure_foundry',
            requested_model: options.model,
            resolved_model: 'llama-deployment',
        });
        expect(options.on_canonical_request_prepared).toHaveBeenCalledOnce();
        expect(fetchMock).toHaveBeenCalledOnce();
    });

    it('does not open the Foundry Chat transport when typed request publication is rejected', async () => {
        const driver = createDriver();
        const fetchMock = vi.fn();
        vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
        } as unknown as AzureFoundryDriver['service'];
        await expect(
            driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'hello' }],
                {
                    ...canonicalOptions('llama-deployment::llama', 'foundry-typed-chat-rejected'),
                    on_canonical_request_prepared: async () => {
                        throw new Error('publication rejected');
                    },
                },
                undefined,
                { stream_id: 'stream:foundry:chat:rejected' },
            ),
        ).rejects.toThrow('publication rejected');
        expect(fetchMock).not.toHaveBeenCalled();
    });

    it('preserves tool result status through Chat ingestion without sending private evidence', async () => {
        const driver = createDriver();
        const responses: OpenAI.Chat.ChatCompletion[] = [
            {
                id: 'foundry-tool-call',
                object: 'chat.completion',
                created: 1,
                model: 'llama-deployment',
                choices: [
                    {
                        index: 0,
                        finish_reason: 'tool_calls',
                        logprobs: null,
                        message: {
                            role: 'assistant',
                            content: null,
                            refusal: null,
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
                object: 'chat.completion',
                created: 2,
                model: 'llama-deployment',
                choices: [
                    {
                        index: 0,
                        finish_reason: 'stop',
                        logprobs: null,
                        message: { role: 'assistant', content: 'unavailable', refusal: null },
                    },
                ],
                usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 },
            },
        ];
        const requests: OpenAI.Chat.ChatCompletionCreateParams[] = [];
        const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
            requests.push(JSON.parse(String(init?.body)) as OpenAI.Chat.ChatCompletionCreateParams);
            return new Response(JSON.stringify(responses.shift()), {
                headers: { 'content-type': 'application/json' },
            });
        });
        vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Meta' })) },
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
        } as unknown as AzureFoundryDriver['service'];
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
        expect(JSON.stringify(requests[1])).not.toContain('tool_result_status');
        expect(JSON.stringify(requests[1])).not.toContain('_llumiverse_tool_result_status');
    });

    it.each(['legacy', 'canonical', 'canonical-sync', 'canonical-typed'] as const)(
        'accepts canonical history through the parent Responses %s path',
        async (mode) => {
            const driver = createDriver();
            const model = 'gpt-deployment::gpt-5';
            const id = 'foundry-responses';
            const response = {
                id: 'foundry-response-canonical',
                object: 'response',
                access_programs: null,
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
            const requests: OpenAI.Responses.ResponseCreateParams[] = [];
            const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
                const request = JSON.parse(String(init?.body)) as OpenAI.Responses.ResponseCreateParams;
                requests.push(request);
                return request.stream
                    ? new Response(
                          `data: ${JSON.stringify({ type: 'response.completed', sequence_number: 1, response })}\n\ndata: [DONE]\n\n`,
                          { headers: { 'content-type': 'text/event-stream' } },
                      )
                    : new Response(JSON.stringify(response), {
                          headers: { 'content-type': 'application/json' },
                      });
            });
            vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
            driver.service = {
                deployments: { get: vi.fn(async () => ({ modelPublisher: 'OpenAI' })) },
                getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
            } as unknown as AzureFoundryDriver['service'];
            const state = await prepareOpenAIResponsesCanonicalState({
                conversation: [{ type: 'message', role: 'user', content: 'prior' }],
                prompt: [],
                options: canonicalOptions(model, id, 'seed'),
                provider: 'azure_foundry',
            });

            const segments = [{ role: PromptRole.user, content: 'next' }];
            const options = { ...canonicalOptions(model, id), conversation: state.document };
            expect(await driver.supportsCanonicalExecution(options)).toBe(true);
            let completedConversation: unknown;
            if (mode === 'canonical-sync') {
                completedConversation = (await driver.executeCanonical(segments, options)).conversation;
            } else if (mode === 'canonical-typed') {
                const stream = await driver.streamCanonicalEvents(segments, options, undefined, {
                    stream_id: 'stream:foundry:responses',
                });
                const events = await collectCanonicalEvents(stream);
                expect(events.at(-1)).toMatchObject({
                    type: 'response_accepted',
                    stream_id: 'stream:foundry:responses',
                });
                completedConversation = stream.completion?.conversation;
            } else {
                const stream =
                    mode === 'canonical'
                        ? await driver.streamCanonical(segments, options)
                        : await driver.stream(segments, options);
                for await (const _chunk of stream) {
                    // Consume the parent stream so terminal canonical finalization runs.
                }
                completedConversation = stream.completion?.conversation;
            }

            const conversation = parseConversationDocument(completedConversation);
            expect(conversation.id).toBe(id);
            expect(Object.values(conversation.generations)).toEqual(
                expect.arrayContaining([expect.objectContaining({ requested_model: model })]),
            );
            expect(requests[0]).toEqual(
                expect.objectContaining({ model: 'gpt-deployment', stream: mode !== 'canonical-sync' }),
            );
            expect(fetchMock).toHaveBeenCalledOnce();
        },
    );

    it('preserves required tool choice when adapting an OpenAI chat request', () => {
        const body = toOpenAINonStreamingPayload({
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
        } satisfies OpenAIChatCompletionsPayload);

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
            modelDeployment('flux', 'FLUX.1-Kontext-pro', {}),
            modelDeployment('future-flux', 'FLUX-9-pro', { chat_completion: 'true' }),
            {
                ...modelDeployment('anthropic', 'claude-future', { chat_completion: 'true' }),
                modelPublisher: 'Anthropic',
            },
            modelDeployment('speech', 'gpt-4o-mini-tts', { chat_completion: 'true' }),
            {
                ...modelDeployment('image', 'gpt-image-2.5-flare', { chat_completion: 'false' }),
                modelPublisher: 'OpenAI',
            },
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
            'image::gpt-image-2.5-flare',
        ]);
    });

    it.each([
        ['image::dall-e-3', false],
        ['gpt-image-2::gpt-image-2', true],
        ['custom-image', true],
        ['chat::gpt-4.1-mini', true],
    ])('preserves image and text streaming capability for %s', async (model, expected) => {
        const driver = new AzureFoundryDriver({
            endpoint: 'https://foundry.example.test',
            azureADTokenProvider: credential,
            sourceModel: 'gpt-image-2',
        });
        try {
            expect(await exposePrivate<FoundryInternals>(driver).canStream({ model })).toBe(expected);
            expect(
                await exposePrivate<FoundryInternals>(driver).canStream({
                    model,
                    model_options: { _option_id: 'openai-text', image_generation: { model: 'gpt-image-2' } },
                }),
            ).toBe(false);
        } finally {
            driver.destroy();
        }
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
        driver.service = {
            deployments: { get: deploymentGet },
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
        } as unknown as AzureFoundryDriver['service'];
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
        const client = exposePrivate<FoundryInternals>(driver).getInferenceClient();
        const post = vi
            .spyOn(client.chat.completions, 'create')
            .mockResolvedValue(nativeResponse as OpenAI.Chat.ChatCompletion);
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

        expect(post).toHaveBeenCalledWith(
            expect.objectContaining({
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
            undefined,
        );
        expect(completion.tool_use?.[0]).toEqual({
            id: 'call_1',
            tool_name: 'lookup',
            tool_input: { city: 'Paris' },
        });
        expect(completion.original_response).toBe(nativeResponse);
    });

    it('streams non-OpenAI inference without unsupported usage options', async () => {
        const driver = createDriver();
        driver.service = {
            deployments: { get: vi.fn(async () => ({ modelPublisher: 'Mistral AI' })) },
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
        } as unknown as AzureFoundryDriver['service'];
        const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
            const body = JSON.parse(String(init?.body));
            expect(body).not.toHaveProperty('stream_options');
            expect(body).toMatchObject({ model: 'chat-deployment', stream: true });
            return new Response(
                `data: ${JSON.stringify({
                    id: 'chat-1',
                    object: 'chat.completion.chunk',
                    created: 1,
                    model: 'chat-deployment',
                    choices: [{ index: 0, delta: { content: 'Green' }, finish_reason: null }],
                })}\n\ndata: [DONE]\n\n`,
                { headers: { 'content-type': 'text/event-stream' } },
            );
        });
        vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
        const options = { model: 'chat-deployment::mistral' };
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'What color is grass?' }], options);
        const results = [];
        for await (const chunk of await driver.requestTextCompletionStream(prompt, options)) {
            results.push(...chunk.result);
        }
        expect(results).toContainEqual({ type: 'text', value: 'Green' });
        expect(fetchMock).toHaveBeenCalledOnce();
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
        const requests: { body: unknown; headers: Headers; url: string }[] = [];
        const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
            requests.push({
                body: JSON.parse(String(init?.body)),
                headers: new Headers(init?.headers),
                url: String(input),
            });
            return new Response(JSON.stringify(response), { headers: { 'content-type': 'application/json' } });
        });
        vi.spyOn(exposePrivate<FoundryInternals>(driver), 'getDriverFetch').mockReturnValue(fetchMock);
        const openAIClient = { baseURL: 'https://foundry.example.test/openai/v1' };
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
        expect(requests).toHaveLength(2);
        expect(requests[0].url).toBe('https://foundry.example.test/openai/v1/responses');
        expect(requests[0].headers.get('authorization')).toBe('Bearer test-token');
        expect(requests[0].headers.has('x-ms-oai-image-generation-deployment')).toBe(false);
        expect(requests[0].body).not.toHaveProperty('temperature');
        expect(requests[0].body).not.toHaveProperty('top_p');
        expect(requests[0].body).toEqual(
            expect.objectContaining({
                model: 'gpt-deployment',
                stream: false,
                reasoning: { effort: 'high', summary: 'auto' },
                include: ['reasoning.encrypted_content'],
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
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
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
            getOpenAIClient: () => ({ baseURL: 'https://foundry.example.test/openai/v1' }),
        } as unknown as AzureFoundryDriver['service'];
        const error = new OpenAI.InternalServerError(
            503,
            { code: 'ServiceUnavailable', message: 'Temporarily unavailable' },
            'Temporarily unavailable',
            new Headers(),
        );
        vi.spyOn(
            exposePrivate<FoundryInternals>(driver).getInferenceClient().chat.completions,
            'create',
        ).mockRejectedValue(error);

        await expect(
            driver.execute([{ role: PromptRole.user, content: 'Hello' }], {
                model: 'llama-deployment::llama',
            }),
        ).rejects.toMatchObject({
            name: 'InternalServerError',
            code: 503,
            retryable: true,
            originalError: expect.objectContaining({
                status: 503,
                error: { code: 'ServiceUnavailable', message: 'Temporarily unavailable' },
            }),
        });
    });

    it('preserves Azure embedding HTTP status and retryability in LlumiverseError', async () => {
        const driver = createDriver();
        const error = new OpenAI.InternalServerError(
            503,
            { code: 'ServiceUnavailable', message: 'Temporarily unavailable' },
            'Temporarily unavailable',
            new Headers(),
        );
        vi.spyOn(exposePrivate<FoundryInternals>(driver).getResourceClient().embeddings, 'create').mockRejectedValue(
            error,
        );

        await expect(
            driver.generateEmbeddings({
                model: 'embedding-deployment::embedding-model',
                inputs: [{ type: 'text', text: 'Hello' }],
            }),
        ).rejects.toMatchObject({
            name: 'InternalServerError',
            code: 503,
            retryable: true,
            context: {
                provider: driver.provider,
                model: 'embedding-deployment::embedding-model',
                operation: 'execute',
            },
            originalError: expect.objectContaining({
                status: 503,
                error: { code: 'ServiceUnavailable', message: 'Temporarily unavailable' },
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
