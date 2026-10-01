import {
    type ConversationDocument,
    type ConversationStreamEvent,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { toOpenAISDKMessage } from '../openai/openai_chat_completions.js';
import { compileOpenAIChatCompletionsConversation } from '../openai/openai-chat-conversation-adapter.js';
import { MistralAIDriver, mistralRequestFromOpenAI, normalizeMistralStream } from './index.js';

const MODEL = 'mistral-small-latest';

function runtime(flow: string, attempt = 'first', conversation?: ConversationDocument): CanonicalExecutionInputOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        model_options: {
            _option_id: 'mistral-text',
            max_tokens: 128,
            temperature: 0.2,
            random_seed: 42,
            safe_prompt: true,
            parallel_tool_calls: false,
            prompt_mode: 'reasoning',
        },
        conversation_runtime: {
            conversation_id: `conversation:mistral:${flow}`,
            request_id: `request:mistral:${flow}`,
            attempt_id: `attempt:mistral:${flow}:${attempt}`,
            input_operation_id: `input:mistral:${flow}`,
            response_operation_id: `response:mistral:${flow}`,
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:01.000Z',
        },
    };
}

function signedResponse() {
    return {
        id: 'mistral-response-1',
        object: 'chat.completion',
        created: 1,
        model: MODEL,
        choices: [
            {
                index: 0,
                finishReason: 'stop',
                message: {
                    role: 'assistant' as const,
                    content: [
                        {
                            type: 'thinking' as const,
                            thinking: [{ type: 'text' as const, text: 'private plan' }],
                            signature: 'signed-private-plan',
                            closed: true,
                        },
                        { type: 'text' as const, text: 'Visible answer' },
                    ],
                },
            },
        ],
        usage: {
            promptTokens: 4,
            completionTokens: 5,
            totalTokens: 9,
            promptAudioSeconds: 2,
            serviceTier: 'priority',
            vendorUnits: { input: 4 },
        },
    };
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Mistral canonical lifecycle', () => {
    it('binds the exact Mistral SDK request, preserves signed replay, and exact-retries without transport', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const complete = vi.fn(async (_request: unknown) => signedResponse());
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });
        let prepared = 0;
        const options = runtime('sync');
        const first = await driver.executeCanonical([{ role: PromptRole.user, content: 'Answer carefully.' }], {
            ...options,
            on_canonical_request_prepared: async () => {
                prepared += 1;
                expect(complete).not.toHaveBeenCalled();
            },
        });

        expect(complete).toHaveBeenCalledOnce();
        expect(complete.mock.calls[0]?.[0]).toEqual(
            expect.objectContaining({
                model: MODEL,
                messages: [{ role: 'user', content: 'Answer carefully.' }],
                maxTokens: 128,
                temperature: 0.2,
                randomSeed: 42,
                safePrompt: true,
                parallelToolCalls: false,
                promptMode: 'reasoning',
                stream: false,
            }),
        );
        expect(prepared).toBe(1);
        expect(first.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'text', text: 'Visible answer' }),
                expect.objectContaining({ type: 'reasoning', text: 'private plan' }),
            ]),
        );
        const firstGeneration = first.conversation.generations[first.accepted_output.generation.id];
        expect(firstGeneration?.usage).toMatchObject({
            input_tokens: 4,
            output_tokens: 5,
            total_tokens: 9,
            reported_usage: [
                expect.objectContaining({
                    payload: expect.objectContaining({
                        provider_usage: {
                            source: 'mistral_sdk_usage_info',
                            omitted_token_count_semantics: 'sdk_default_zero',
                            payload: expect.objectContaining({
                                promptAudioSeconds: 2,
                                serviceTier: 'priority',
                                vendorUnits: { input: 4 },
                            }),
                        },
                    }),
                }),
            ],
        });
        expect(JSON.stringify(first.accepted_output)).not.toContain('signed-private-plan');

        const document = parseConversationDocument(first.conversation);
        const retainedTurn = document.turns.find((turn) => turn.id === first.accepted_output.turn.id);
        const replay = retainedTurn?.blocks.find((block) => block.type === 'native_replay');
        expect(replay).toMatchObject({
            type: 'native_replay',
            compatibility_scope: { provider: 'mistralai' },
            payload: {
                provider_replay: {
                    provider: 'mistralai',
                    protocol: 'mistral.chat.completions',
                    adapter_version: '2026-09-30.canonical.1',
                },
            },
        });
        expect(JSON.stringify(replay)).toContain('signed-private-plan');
        expect(() =>
            compileOpenAIChatCompletionsConversation(document, {
                provider: 'openai',
                model: MODEL,
            }),
        ).toThrow(/outside its compatibility scope|cannot be projected/i);
        expect(
            toOpenAISDKMessage({
                role: 'assistant',
                content: 'Visible answer',
                provider_replay: {
                    provider: 'mistralai',
                    protocol: 'mistral.chat.completions',
                    adapter_version: '2026-09-30.canonical.1',
                    payload: { type: 'mistral_assistant_content', content: [] },
                },
            }),
        ).not.toHaveProperty('provider_replay');
        const generation = document.generations[first.accepted_output.generation.id];
        expect(generation.request_receipt?.target.options).toMatchObject({
            transport: 'mistral_sdk',
            model: MODEL,
            maxTokens: 128,
            temperature: 0.2,
            randomSeed: 42,
            safePrompt: true,
            parallelToolCalls: false,
            promptMode: 'reasoning',
            stream: false,
        });
        expect(JSON.stringify(generation.request_receipt?.target.options)).not.toContain('Answer carefully.');

        const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;
        const retry = await driver.executeCanonical([{ role: PromptRole.user, content: 'Answer carefully.' }], {
            ...runtime('sync', 'retry', persisted),
            on_canonical_request_prepared: async () => {
                prepared += 1;
            },
        });
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(complete).toHaveBeenCalledOnce();
        expect(prepared).toBe(1);

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Answer carefully.' }], {
                ...runtime('sync', 'changed', persisted),
                model_options: { ...runtime('sync').model_options, random_seed: 7 },
            }),
        ).rejects.toThrow(/fingerprint|incompatible|request/i);
        expect(complete).toHaveBeenCalledOnce();
    });

    it('emits typed reasoning without opaque signatures and retains exact signed replay at acceptance', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const streamCall = vi.fn(async () =>
            (async function* () {
                yield {
                    data: {
                        id: 'mistral-stream-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: MODEL,
                        choices: [
                            {
                                index: 0,
                                finishReason: null,
                                delta: {
                                    role: 'assistant',
                                    content: [
                                        {
                                            type: 'thinking',
                                            thinking: [{ type: 'text', text: 'private ' }],
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                };
                yield {
                    data: {
                        id: 'mistral-stream-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: MODEL,
                        usage: { promptTokens: 4, completionTokens: 5, totalTokens: 9 },
                        choices: [
                            {
                                index: 0,
                                finishReason: 'stop',
                                delta: {
                                    content: [
                                        {
                                            type: 'thinking',
                                            thinking: [{ type: 'text', text: 'plan' }],
                                            signature: 'stream-signature',
                                            closed: true,
                                        },
                                        { type: 'text', text: 'Visible answer' },
                                    ],
                                },
                            },
                        ],
                    },
                };
            })(),
        );
        Object.defineProperty(driver.client.chat, 'stream', { value: streamCall });

        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer carefully.' }],
            runtime('typed'),
            undefined,
            { stream_id: 'stream:mistral:typed' },
        );
        const events = await collect(stream);
        expect(events[0]).toMatchObject({ type: 'draft_started', origin: 'live_transport' });
        expect(events).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'draft_reasoning_delta', text: 'private ' }),
                expect.objectContaining({ type: 'draft_reasoning_delta', text: 'plan' }),
                expect.objectContaining({ type: 'draft_text_delta', text: 'Visible answer' }),
                expect.objectContaining({ type: 'response_accepted' }),
            ]),
        );
        expect(JSON.stringify(events)).not.toContain('stream-signature');
        expect(JSON.stringify(stream.completion?.accepted_output)).not.toContain('stream-signature');
        expect(JSON.stringify(stream.completion?.conversation)).toContain('stream-signature');
        expect(streamCall).toHaveBeenCalledOnce();
    });

    it('serializes long signed replay once and restores it exactly for an accepted continuation', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const reasoningFragments = Array.from({ length: 48 }, (_, index) => `${index}:`.padEnd(1_024, 'x'));
        const nativeStream = async function* () {
            for (const fragment of reasoningFragments) {
                yield {
                    data: {
                        id: 'mistral-long-replay-1',
                        object: 'chat.completion.chunk' as const,
                        created: 1,
                        model: MODEL,
                        choices: [
                            {
                                index: 0,
                                finishReason: null,
                                delta: {
                                    role: 'assistant' as const,
                                    content: [
                                        {
                                            type: 'thinking' as const,
                                            thinking: [{ type: 'text' as const, text: fragment }],
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                };
            }
            yield {
                data: {
                    id: 'mistral-long-replay-1',
                    object: 'chat.completion.chunk' as const,
                    created: 1,
                    model: MODEL,
                    usage: { promptTokens: 3, completionTokens: 50, totalTokens: 53 },
                    choices: [
                        {
                            index: 0,
                            finishReason: 'tool_calls' as const,
                            delta: {
                                content: [
                                    {
                                        type: 'thinking' as const,
                                        thinking: [{ type: 'text' as const, text: 'final' }],
                                        signature: 'long-stream-signature',
                                        closed: true,
                                    },
                                    { type: 'text' as const, text: 'Visible answer' },
                                ],
                                toolCalls: [
                                    {
                                        index: 0,
                                        id: 'call_long_replay',
                                        type: 'function' as const,
                                        function: { name: 'lookup', arguments: '{"key":"typed"}' },
                                    },
                                ],
                            },
                        },
                    ],
                },
            };
        };

        const normalized = [];
        for await (const chunk of normalizeMistralStream(nativeStream())) normalized.push(chunk);
        const replayChunks = normalized.filter((chunk) => chunk.choices[0]?.delta.provider_replay !== undefined);
        expect(replayChunks).toHaveLength(1);
        expect(normalized.reduce((total, chunk) => total + JSON.stringify(chunk).length, 0)).toBeLessThan(180_000);

        const streamCall = vi.fn(async () => nativeStream());
        Object.defineProperty(driver.client.chat, 'stream', { value: streamCall });
        const longOptions = {
            ...runtime('long-replay'),
            tools: [
                {
                    name: 'lookup',
                    input_schema: {
                        type: 'object' as const,
                        properties: { key: { type: 'string' as const } },
                        required: ['key'],
                    },
                },
            ],
        } satisfies CanonicalExecutionInputOptions;
        const first = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Think carefully.' }],
            longOptions,
            undefined,
            { stream_id: 'stream:mistral:long-replay' },
        );
        await collect(first);
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(first.completion?.conversation)));
        expect(JSON.stringify(persisted)).toContain('long-stream-signature');
        const compiled = compileOpenAIChatCompletionsConversation(persisted, {
            provider: 'mistralai',
            model: MODEL,
        });
        const compiledAssistant = compiled.conversation.messages.find((message) => message.role === 'assistant');
        expect(JSON.stringify(compiledAssistant?.provider_replay)).toContain('long-stream-signature');
        const continuationRequest = mistralRequestFromOpenAI(
            {
                model: MODEL,
                stream: false,
                messages: [
                    ...compiled.conversation.messages,
                    { role: 'tool', tool_call_id: 'call_long_replay', content: 'lookup result' },
                ],
            },
            longOptions,
        );
        const replayedAssistant = continuationRequest.messages.find((message) => message.role === 'assistant');
        expect(replayedAssistant?.content).toEqual([
            {
                type: 'thinking',
                thinking: [
                    ...reasoningFragments.map((text) => ({ type: 'text', text })),
                    { type: 'text', text: 'final' },
                ],
                signature: 'long-stream-signature',
                closed: true,
            },
            { type: 'text', text: 'Visible answer' },
        ]);
    });

    it('preserves native tool arguments as executable canonical JSON', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const complete = vi.fn(async (_request: unknown) => ({
            id: 'mistral-tool-1',
            object: 'chat.completion',
            created: 1,
            model: MODEL,
            choices: [
                {
                    index: 0,
                    finishReason: 'tool_calls',
                    message: {
                        role: 'assistant' as const,
                        content: null,
                        toolCalls: [
                            {
                                id: 'call_lookup',
                                index: 0,
                                type: 'function' as const,
                                function: { name: 'lookup', arguments: { key: 'typed' } },
                            },
                        ],
                    },
                },
            ],
            usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
        }));
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });

        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Use the tool.' }], {
            ...runtime('tool'),
            tools: [
                {
                    name: 'lookup',
                    description: 'Lookup a key.',
                    input_schema: {
                        type: 'object',
                        properties: { key: { type: 'string' } },
                        required: ['key'],
                    },
                },
            ],
            model_options: {
                ...runtime('tool').model_options,
                tool_choice: 'required',
                required_tool_name: 'lookup',
            } as ExecutionOptions['model_options'] & { required_tool_name: string },
        });

        expect(result.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({
                type: 'tool_call',
                call_id: 'call_lookup',
                tool_name: 'lookup',
                executor: 'application',
                arguments: { type: 'json', value: { key: 'typed' } },
            }),
        ]);
        expect(complete.mock.calls[0]?.[0]).toMatchObject({
            toolChoice: { type: 'function', function: { name: 'lookup' } },
            tools: [
                {
                    type: 'function',
                    function: {
                        name: 'lookup',
                        description: 'Lookup a key.',
                        parameters: expect.objectContaining({ type: 'object' }),
                    },
                },
            ],
        });
    });

    it('does not prefix split streamed tool arguments when the opening delta omits arguments', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const streamCall = vi.fn(async () =>
            (async function* () {
                yield {
                    data: {
                        id: 'mistral-tool-stream-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: MODEL,
                        choices: [
                            {
                                index: 0,
                                finishReason: null,
                                delta: {
                                    role: 'assistant',
                                    toolCalls: [
                                        {
                                            index: 0,
                                            id: 'call_split',
                                            type: 'function',
                                            function: { name: 'lookup' },
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                };
                yield {
                    data: {
                        id: 'mistral-tool-stream-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: MODEL,
                        choices: [
                            {
                                index: 0,
                                finishReason: null,
                                delta: {
                                    toolCalls: [{ index: 0, function: { name: '', arguments: '{"key":' } }],
                                },
                            },
                        ],
                    },
                };
                yield {
                    data: {
                        id: 'mistral-tool-stream-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: MODEL,
                        usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
                        choices: [
                            {
                                index: 0,
                                finishReason: 'tool_calls',
                                delta: {
                                    toolCalls: [{ index: 0, function: { name: '', arguments: '"typed"}' } }],
                                },
                            },
                        ],
                    },
                };
            })(),
        );
        Object.defineProperty(driver.client.chat, 'stream', { value: streamCall });

        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Use the tool.' }],
            {
                ...runtime('split-tool'),
                tools: [
                    {
                        name: 'lookup',
                        input_schema: {
                            type: 'object',
                            properties: { key: { type: 'string' } },
                            required: ['key'],
                        },
                    },
                ],
                model_options: {
                    ...runtime('split-tool').model_options,
                    tool_choice: 'required',
                    required_tool_name: 'lookup',
                } as ExecutionOptions['model_options'] & { required_tool_name: string },
            },
            undefined,
            { stream_id: 'stream:mistral:split-tool' },
        );
        const events = await collect(stream);

        expect(JSON.stringify(events)).not.toContain('{}{"key"');
        expect(stream.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({
                type: 'tool_call',
                call_id: 'call_split',
                tool_name: 'lookup',
                arguments: { type: 'json', value: { key: 'typed' } },
            }),
        ]);
    });

    it('retains partial Mistral usage evidence with explicit SDK default semantics', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const complete = vi.fn(async (_request: unknown) => ({
            id: 'mistral-partial-usage-1',
            object: 'chat.completion',
            created: 1,
            model: MODEL,
            choices: [
                {
                    index: 0,
                    finishReason: 'stop',
                    message: { role: 'assistant' as const, content: 'done' },
                },
            ],
            usage: { promptTokens: 7, promptAudioSeconds: 3, serviceTier: 'standard', billedUnits: 11 },
        }));
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });

        const result = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Report usage.' }],
            runtime('partial-usage'),
        );

        const generation = result.conversation.generations[result.accepted_output.generation.id];
        expect(generation?.usage).toMatchObject({
            input_tokens: 7,
            output_tokens: 0,
            total_tokens: 7,
            reported_usage: [
                expect.objectContaining({
                    payload: expect.objectContaining({
                        provider_usage: {
                            source: 'mistral_sdk_usage_info',
                            omitted_token_count_semantics: 'sdk_default_zero',
                            payload: expect.objectContaining({
                                promptTokens: 7,
                                promptAudioSeconds: 3,
                                serviceTier: 'standard',
                                billedUnits: 11,
                            }),
                        },
                    }),
                }),
            ],
        });
    });

    it('cancels the Mistral stream through the owned SDK signal and waits for cleanup', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        let transportSignal: AbortSignal | undefined;
        const streamCall = vi.fn(async (_request: unknown, requestOptions?: { signal?: AbortSignal }) => {
            transportSignal = requestOptions?.signal;
            return (async function* () {
                yield {
                    data: {
                        id: 'mistral-cancel-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: MODEL,
                        choices: [
                            {
                                index: 0,
                                finishReason: null,
                                delta: { role: 'assistant', content: 'partial' },
                            },
                        ],
                    },
                };
                await new Promise<never>((_resolve, reject) => {
                    transportSignal?.addEventListener(
                        'abort',
                        () => reject(new DOMException('provider request aborted', 'AbortError')),
                        { once: true },
                    );
                });
            })();
        });
        Object.defineProperty(driver.client.chat, 'stream', { value: streamCall });

        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            runtime('cancel'),
            undefined,
            { stream_id: 'stream:mistral:cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({
            value: expect.objectContaining({ type: 'draft_started' }),
        });
        await vi.waitFor(() => expect(streamCall).toHaveBeenCalledOnce());
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        expect(transportSignal?.aborted).toBe(true);
        await stream.closed;
        expect(stream.completion).toBeUndefined();
    });

    it('does not start transport when prepared publication fails', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const complete = vi.fn(async () => signedResponse());
        const streamCall = vi.fn();
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });
        Object.defineProperty(driver.client.chat, 'stream', { value: streamCall });

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Do not send.' }], {
                ...runtime('barrier'),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability barrier failed');
                },
            }),
        ).rejects.toThrow('durability barrier failed');
        expect(complete).not.toHaveBeenCalled();

        await expect(
            driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'Do not stream.' }],
                {
                    ...runtime('barrier-typed'),
                    on_canonical_request_prepared: async () => {
                        throw new Error('typed durability barrier failed');
                    },
                },
                undefined,
                { stream_id: 'stream:mistral:barrier' },
            ),
        ).rejects.toThrow('typed durability barrier failed');
        expect(streamCall).not.toHaveBeenCalled();
    });

    it('rejects native assistant content that cannot be replayed losslessly', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key' });
        const complete = vi.fn(async () => ({
            ...signedResponse(),
            choices: [
                {
                    index: 0,
                    finishReason: 'stop',
                    message: {
                        role: 'assistant' as const,
                        content: [{ type: 'image_url', imageUrl: 'data:image/png;base64,AA==' }],
                    },
                },
            ],
        }));
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });

        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Return unsupported native output.' }],
                runtime('unsupported-native-output'),
            ),
        ).rejects.toThrow(/unsupported payload/i);
        expect(complete).toHaveBeenCalledOnce();
    });
});
