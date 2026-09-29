import {
    type ContentBlock,
    type ConversationDocument,
    createConversationDocument,
    externalizeToolCallArguments,
    parseConversationDocument,
    prepareToolArgumentExternalization,
} from '@llumiverse/conversation';
import {
    type AIModel,
    type CompletionChunkObject,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    legacyCompletionFromCanonicalExecution,
    type ModelSearchPayload,
    PromptRole,
} from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import {
    normalizeOpenAIChatCompletionsResponse,
    normalizeOpenAIChatCompletionsStream,
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsProtocolOptions,
    type OpenAIChatCompletionsResponse,
    openAIChatCompletionsStreamToSSE,
    parseOpenAIChatCompletionsToolCalls,
    prepareOpenAIChatCompletionsConversation,
} from './openai_chat_completions.js';
import {
    exportLegacyOpenAIChatCompletionsConversation,
    prepareOpenAIChatCanonicalState,
} from './openai-chat-conversation-adapter.js';

const streamChunk = {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk' as const,
    created: 1,
    model: 'test/model',
    choices: [],
};

describe('openAIChatCompletionsStreamToSSE', () => {
    it('closes after forwarding a normally completed provider stream', async () => {
        async function* providerStream() {
            yield streamChunk;
        }

        const reader = openAIChatCompletionsStreamToSSE(providerStream()).getReader();

        await expect(reader.read()).resolves.toEqual({
            done: false,
            value: { type: 'event', data: JSON.stringify(streamChunk) },
        });
        await expect(reader.read()).resolves.toEqual({ done: true, value: undefined });
    });

    it('propagates provider errors to the consumer', async () => {
        const providerError = new Error('provider stream failed');
        const providerStream = {
            [Symbol.asyncIterator]() {
                return {
                    next: vi.fn().mockRejectedValue(providerError),
                };
            },
        };

        const reader = openAIChatCompletionsStreamToSSE(providerStream).getReader();

        await expect(reader.read()).rejects.toBe(providerError);
    });

    it('aborts a pending provider read when the consumer cancels', async () => {
        const abortController = new AbortController();
        const providerStream = {
            async *[Symbol.asyncIterator]() {
                yield streamChunk;
                await new Promise<never>((_resolve, reject) => {
                    abortController.signal.addEventListener(
                        'abort',
                        () => reject(new DOMException('provider request aborted', 'AbortError')),
                        { once: true },
                    );
                });
            },
        };
        const normalized = normalizeOpenAIChatCompletionsStream(providerStream);
        const reader = openAIChatCompletionsStreamToSSE(normalized, () => abortController.abort()).getReader();

        await expect(reader.read()).resolves.toEqual({
            done: false,
            value: { type: 'event', data: JSON.stringify(streamChunk) },
        });
        await reader.cancel('consumer stopped');

        expect(abortController.signal.aborted).toBe(true);
    });
});

function createSSEStream(events: ServerSentEvent[]): ReadableStream<ServerSentEvent> {
    return new ReadableStream<ServerSentEvent>({
        start(controller) {
            for (const event of events) {
                controller.enqueue(event);
            }
            controller.close();
        },
    });
}

async function collectChunks(stream: AsyncIterable<CompletionChunkObject>): Promise<CompletionChunkObject[]> {
    const chunks: CompletionChunkObject[] = [];
    for await (const chunk of stream) {
        chunks.push(chunk);
    }
    return chunks;
}

function latestGeneratedText(value: unknown): string | undefined {
    const document = parseConversationDocument(value);
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind !== 'agent' || turn.provenance.type !== 'generated') continue;
        return turn.blocks.find((block) => block.type === 'text')?.text;
    }
    return undefined;
}

function latestGeneratedJson(value: unknown): unknown {
    const document = parseConversationDocument(value);
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind !== 'agent' || turn.provenance.type !== 'generated') continue;
        return turn.blocks.find((block) => block.type === 'json')?.value;
    }
    return undefined;
}

function latestExecutedGeneration(value: unknown) {
    const document = parseConversationDocument(value);
    return Object.values(document.generations).find((generation) => generation.record_source === 'executed');
}

class TestOpenAIChatCompletionsProtocol extends OpenAIChatCompletionsProtocol<undefined> {
    payloads: OpenAIChatCompletionsPayload[] = [];

    constructor(
        private readonly response?: OpenAIChatCompletionsResponse,
        private readonly stream?: ReadableStream,
        protocolOptions: Partial<OpenAIChatCompletionsProtocolOptions> = {},
    ) {
        super({ modelName: 'test/model', ...protocolOptions });
    }

    protected async postChatCompletion(
        _driver: undefined,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<OpenAIChatCompletionsResponse> {
        this.payloads.push(payload);
        if (!this.response) {
            throw new Error('Missing test response');
        }
        return this.response;
    }

    protected async postChatCompletionStream(
        _driver: undefined,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<ReadableStream> {
        this.payloads.push(payload);
        if (!this.stream) {
            throw new Error('Missing test stream');
        }
        return this.stream;
    }
}

class TestOpenAIChatCompletionsDriver extends OpenAIChatCompletionsDriverBase {
    readonly provider = 'openai_compatible';
    readonly payloads: OpenAIChatCompletionsPayload[] = [];

    constructor(
        private readonly response?: OpenAIChatCompletionsResponse,
        private readonly responseStream?: ReadableStream,
    ) {
        super({});
    }

    async _postChatCompletion(payload: OpenAIChatCompletionsPayload): Promise<OpenAIChatCompletionsResponse> {
        this.payloads.push(payload);
        if (this.response === undefined) throw new Error('Missing test response');
        return this.response;
    }

    async _postChatCompletionStream(payload: OpenAIChatCompletionsPayload): Promise<ReadableStream> {
        this.payloads.push(payload);
        if (this.responseStream === undefined) throw new Error('Missing test stream');
        return this.responseStream;
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return [];
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return { model: 'test/model', results: [] };
    }
}

const prompt: OpenAIChatCompletionsPrompt = {
    _is_openai_chat_completions: true,
    messages: [{ role: 'user', content: 'Hello' }],
};

const options: ExecutionOptions = {
    model: 'test/model',
    model_options: { _option_id: 'text-fallback' },
};

function legacyConversation(value: unknown): OpenAIChatCompletionsPrompt {
    return exportLegacyOpenAIChatCompletionsConversation(value as ConversationDocument);
}

function canonicalOptions(attempt: string, recordedAt: string, conversation?: unknown): ExecutionOptions {
    return {
        model: 'test/model',
        ...(conversation === undefined ? {} : { conversation }),
        conversation_runtime: {
            conversation_id: 'conversation:structured-output',
            request_id: 'request:structured-output',
            attempt_id: attempt,
            input_operation_id: 'input:structured-output',
            response_operation_id: 'response:structured-output',
            recorded_at: recordedAt,
            started_at: recordedAt,
        },
    };
}

describe('OpenAIChatCompletionsProtocol', () => {
    it('stores Chat input audio as a durable canonical asset and replays exact bytes after JSON persistence', async () => {
        const audio = { type: 'input_audio' as const, input_audio: { data: 'UklGRg==', format: 'wav' as const } };
        const state = await prepareOpenAIChatCanonicalState({
            conversation: undefined,
            prompt: {
                _is_openai_chat_completions: true,
                messages: [{ role: 'user', content: [{ type: 'text', text: 'Transcribe.' }, audio] }],
            },
            options: canonicalOptions('attempt:audio', '2026-09-11T00:00:00.000Z'),
            provider: 'openai',
        });
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(state.document)));
        let audioBlock: Extract<ContentBlock, { type: 'audio' }> | undefined;
        for (const turn of persisted.turns) {
            for (const block of turn.blocks as readonly ContentBlock[]) {
                if (block.type === 'audio') audioBlock = block;
            }
        }
        expect(audioBlock).toBeDefined();
        if (audioBlock?.type !== 'audio') throw new Error('Expected canonical audio block');
        expect(persisted.assets[audioBlock.asset_id]).toMatchObject({
            kind: 'audio',
            mime_type: 'audio/wav',
            storage: { type: 'inline_base64', data: 'UklGRg==' },
        });
        expect(exportLegacyOpenAIChatCompletionsConversation(persisted).messages[0]?.content).toEqual([
            { type: 'text', text: 'Transcribe.' },
            audio,
        ]);
    });

    it('preserves compatible-provider prompt cache usage', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-cache',
            object: 'chat.completion',
            created: 1,
            model: 'compatible-model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
            usage: {
                prompt_tokens: 1_000,
                completion_tokens: 20,
                total_tokens: 1_020,
                prompt_tokens_details: { cached_tokens: 800, audio_tokens: 0 },
                completion_tokens_details: {
                    accepted_prediction_tokens: 0,
                    audio_tokens: 0,
                    reasoning_tokens: 0,
                    rejected_prediction_tokens: 0,
                },
            },
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.token_usage).toMatchObject({
            prompt: 1_000,
            prompt_cached: 800,
            prompt_new: 200,
            result: 20,
            total: 1_020,
        });
        expect(latestExecutedGeneration(completion.conversation)?.usage).toMatchObject({
            input_tokens: 1_000,
            output_tokens: 20,
            reasoning_tokens: 0,
            total_tokens: 1_020,
            cache_read_tokens: 800,
            input_new_tokens: 200,
            accounting_provenance: {
                reasoning_tokens: { method: 'reported', accounting_basis: 'openai_chat_tokens' },
            },
        });
    });

    it('preserves compatible reasoning fields at the OpenAI SDK transport boundary', async () => {
        const response = {
            id: 'chatcmpl-1',
            object: 'chat.completion' as const,
            created: 1,
            model: 'compatible-model',
            choices: [
                {
                    index: 0,
                    finish_reason: 'stop' as const,
                    logprobs: null,
                    message: {
                        role: 'assistant' as const,
                        content: null,
                        refusal: null,
                        annotations: [],
                        reasoning_content: 'blocking reasoning',
                    },
                },
            ],
        };
        async function* stream() {
            yield {
                id: 'chatcmpl-1',
                object: 'chat.completion.chunk' as const,
                created: 1,
                model: 'compatible-model',
                choices: [
                    {
                        index: 0,
                        finish_reason: 'stop' as const,
                        logprobs: null,
                        delta: { role: 'assistant' as const, reasoning: 'streaming reasoning' },
                    },
                ],
            };
        }

        expect(normalizeOpenAIChatCompletionsResponse(response).choices[0].message.reasoning_content).toBe(
            'blocking reasoning',
        );
        const chunks = [];
        for await (const chunk of normalizeOpenAIChatCompletionsStream(stream())) chunks.push(chunk);
        expect(chunks[0].choices[0].delta.reasoning).toBe('streaming reasoning');
    });

    it('combines static and per-execution extra body fields for the transport', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            {
                id: 'chatcmpl-1',
                object: 'chat.completion',
                created: 1,
                model: 'test/model',
                choices: [
                    {
                        index: 0,
                        message: { role: 'assistant', content: 'ok' },
                        finish_reason: 'stop',
                        logprobs: null,
                    },
                ],
            },
            undefined,
            { defaultMaxTokens: 64, extraBody: { google: { model_safety_settings: { enabled: false } } } },
        );

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model_options: {
                _option_id: 'openai-text',
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop_sequence: ['END'],
                extra_body: {
                    provider: { sort: 'latency' },
                    baseten: { performance: 'max' },
                },
            },
        });

        expect(model.payloads[0]).toEqual(
            expect.objectContaining({
                max_tokens: 64,
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop: ['END'],
                extra_body: {
                    google: { model_safety_settings: { enabled: false } },
                    provider: { sort: 'latency' },
                    baseten: { performance: 'max' },
                },
            }),
        );
    });

    it('forwards the Flex service tier to the transport', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-flex',
            object: 'chat.completion',
            created: 1,
            model: 'gpt-5.6-sol',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model: 'gpt-5.6-sol',
            model_options: { _option_id: 'openai-thinking', service_tier: 'flex' },
        });

        expect(model.payloads[0].service_tier).toBe('flex');
    });

    it('ignores SDK custom tool calls that are not function tools', () => {
        const customToolCall = {
            id: 'custom-1',
            type: 'custom',
            custom: { name: 'shell', input: 'echo hello' },
        } satisfies OpenAI.Chat.ChatCompletionMessageCustomToolCall;

        expect(parseOpenAIChatCompletionsToolCalls([customToolCall])).toBeUndefined();
    });

    it('keeps the default tool choice implicit', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            tools: [{ name: 'think', description: 'Think', input_schema: { type: 'object' } }],
        });

        expect(model.payloads[0].tool_choice).toBeUndefined();
    });

    it.each([
        ['required', 'required'],
        ['any', 'required'],
    ] as const)('forwards explicit %s tool choice as %s', async (configured, expected) => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model_options: { _option_id: 'text-fallback', tool_choice: configured },
            tools: [{ name: 'think', description: 'Think', input_schema: { type: 'object' } }],
        });

        expect(model.payloads[0].tool_choice).toBe(expected);
    });

    it('forces one named tool without changing the visible tool definitions', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model_options: {
                _option_id: 'text-fallback',
                tool_choice: 'required',
                required_tool_name: 'write_artifact',
                parallel_tool_calls: false,
            } as ExecutionOptions['model_options'] & { required_tool_name: string; parallel_tool_calls: false },
            tools: [
                { name: 'read_artifact', description: 'Read', input_schema: { type: 'object' } },
                { name: 'write_artifact', description: 'Write', input_schema: { type: 'object' } },
            ],
        });

        expect(model.payloads[0].tools).toHaveLength(2);
        expect(model.payloads[0].tool_choice).toEqual({
            type: 'function',
            function: { name: 'write_artifact' },
        });
        expect(model.payloads[0].parallel_tool_calls).toBe(false);
    });

    it('reads text from non-streaming content arrays', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: [
                            { type: 'text', text: 'first' },
                            { type: 'text', text: 'second' },
                        ],
                        reasoning_content: 'hidden',
                    },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.result).toEqual([
            { type: 'thoughts', value: 'hidden' },
            { type: 'text', value: 'first\nsecond' },
        ]);
    });

    it('hides thoughts only when explicitly disabled and keeps native replay fields', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-thoughts',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: 'answer',
                        reasoning_content: 'signed-or-native-reasoning',
                    },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const explicit = await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model_options: { _option_id: 'text-fallback', include_thoughts: true },
        });
        expect(explicit.result).toEqual([
            { type: 'thoughts', value: 'signed-or-native-reasoning' },
            { type: 'text', value: 'answer' },
        ]);

        const hidden = await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model_options: { _option_id: 'text-fallback', include_thoughts: false },
        });
        expect(hidden.result).toEqual([{ type: 'text', value: 'answer' }]);
        expect(legacyConversation(hidden.conversation)).toMatchObject({
            messages: expect.arrayContaining([
                expect.objectContaining({ reasoning_content: 'signed-or-native-reasoning' }),
            ]),
        });
    });

    it('keeps exact DeepSeek R1 reasoning in canonical source while excluding it from the next request', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-r1',
            object: 'chat.completion',
            created: 1,
            model: 'deepseek-r1-0528-maas',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'answer', reasoning_content: 'visible reasoning' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model: 'deepseek-ai/deepseek-r1-0528-maas',
        });

        expect(completion.result).toEqual([
            { type: 'thoughts', value: 'visible reasoning' },
            { type: 'text', value: 'answer' },
        ]);
        expect(JSON.stringify(completion.conversation)).toContain('visible reasoning');
        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            model: 'deepseek-ai/deepseek-r1-0528-maas',
            conversation: completion.conversation,
        });
        expect(JSON.stringify(model.payloads[1].messages)).not.toContain('visible reasoning');
    });

    it('replays DeepSeek V3.2 reasoning only within the current user tool turn', () => {
        const currentTurn = prepareOpenAIChatCompletionsConversation(
            {
                messages: [
                    { role: 'user', content: 'first question' },
                    {
                        role: 'assistant',
                        content: null,
                        reasoning_content: 'current reasoning',
                        tool_calls: [
                            {
                                id: 'call-1',
                                type: 'function',
                                function: { name: 'lookup', arguments: '{}' },
                            },
                        ],
                    },
                    { role: 'tool', tool_call_id: 'call-1', content: 'result' },
                ],
            },
            {
                model: 'deepseek-ai/deepseek-v3.2-maas',
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
        );
        expect(currentTurn.messages[1].reasoning_content).toBe('current reasoning');

        const nextTurn = prepareOpenAIChatCompletionsConversation(
            { messages: [...currentTurn.messages, { role: 'user', content: 'next question' }] },
            {
                model: 'deepseek-ai/deepseek-v3.2-maas',
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
        );
        expect(nextTurn.messages[1].reasoning_content).toBeUndefined();
    });

    it('keeps DeepSeek V4 tool reasoning history immutable until checkpoint', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-v4',
            object: 'chat.completion',
            created: 1,
            model: 'deepseek-v4-pro',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: null,
                        reasoning_content: 'tool reasoning',
                        tool_calls: [
                            {
                                id: 'call-1',
                                type: 'function',
                                function: { name: 'lookup', arguments: '{}' },
                            },
                        ],
                    },
                    finish_reason: 'tool_calls',
                    logprobs: null,
                },
            ],
        });
        const imagePrompt: OpenAIChatCompletionsPrompt = {
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'question' },
                        { type: 'image_url', image_url: { url: 'data:image/png;base64,aW1hZ2U=' } },
                    ],
                },
            ],
        };

        const completion = await model.requestTextCompletion(undefined, imagePrompt, {
            ...options,
            model: 'deepseek-v4-pro',
            tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            stripImagesAfterTurns: 0,
        });
        const conversation = legacyConversation(completion.conversation);

        expect(conversation.messages[0]).toEqual(imagePrompt.messages[0]);
        expect(conversation.messages[1].reasoning_content).toBe('tool reasoning');
    });

    it('returns native reasoning separately whether or not answer content is present', async () => {
        const fallbackModel = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: null, reasoning: 'fallback text' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const contentModel = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-2',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'visible', reasoning_content: 'hidden' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await expect(fallbackModel.requestTextCompletion(undefined, prompt, options)).resolves.toMatchObject({
            result: [{ type: 'thoughts', value: 'fallback text' }],
        });
        await expect(contentModel.requestTextCompletion(undefined, prompt, options)).resolves.toMatchObject({
            result: [
                { type: 'thoughts', value: 'hidden' },
                { type: 'text', value: 'visible' },
            ],
        });
    });

    it('projects both native reasoning and embedded think blocks as thoughts', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: '<think>hidden</think>\n\n',
                        reasoning: 'fallback text',
                    },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.result).toEqual([
            { type: 'thoughts', value: 'fallback text' },
            { type: 'thoughts', value: 'hidden' },
        ]);
    });

    it('projects provider think blocks while preserving native conversation content', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: '<think>hidden reasoning</think>\n\n{"answer":"Paris"}',
                    },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.result).toEqual([
            { type: 'thoughts', value: 'hidden reasoning' },
            { type: 'text', value: '{"answer":"Paris"}' },
        ]);
        expect(legacyConversation(completion.conversation)).toMatchObject({
            messages: expect.arrayContaining([
                expect.objectContaining({
                    role: 'assistant',
                    content: '<think>hidden reasoning</think>\n\n{"answer":"Paris"}',
                }),
            ]),
        });
    });

    it('applies stripping to the request projection while preserving canonical source media', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const imagePrompt: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'What is this?' },
                        {
                            type: 'image_url',
                            image_url: { url: 'data:image/png;base64,aW1hZ2U=', detail: 'auto' },
                        },
                    ],
                },
            ],
        };

        const completion = await model.requestTextCompletion(undefined, imagePrompt, {
            ...options,
            stripImagesAfterTurns: 0,
        });
        expect(JSON.stringify(legacyConversation(completion.conversation).messages[0].content)).toContain('aW1hZ2U=');

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            conversation: completion.conversation,
            stripImagesAfterTurns: 0,
        });
        expect(model.payloads[1].messages[0].content).toBe('What is this?\n[Image removed from conversation history]');
    });

    it('preserves imported history age when applying the host retention projection', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-aged-history',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const oldText = 'x'.repeat(32_001);
        const agedHistory = {
            _is_openai_chat_completions: true as const,
            _llumiverse_meta: { turnNumber: 20 },
            messages: [
                {
                    role: 'user' as const,
                    content: [
                        { type: 'text' as const, text: 'old image' },
                        {
                            type: 'image_url' as const,
                            image_url: { url: 'data:image/png;base64,b2xkLWltYWdl', detail: 'auto' as const },
                        },
                    ],
                },
                { role: 'assistant' as const, content: oldText },
                { role: 'user' as const, content: '<heartbeat>old status</heartbeat>' },
            ],
        };

        await model.requestTextCompletion(
            undefined,
            { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'current request' }] },
            {
                ...canonicalOptions('attempt:aged', '2026-09-11T00:00:00.000Z', agedHistory),
                stripImagesAfterTurns: 5,
                stripTextMaxTokens: 8_000,
                stripHeartbeatsAfterTurns: 1,
            },
        );

        expect(model.payloads[0].messages).toEqual([
            {
                role: 'user',
                content: 'old image\n[Image removed from conversation history]',
            },
            {
                role: 'assistant',
                content: `${oldText.slice(0, 32_000)}\n\n[Content truncated - exceeded token limit]`,
            },
            { role: 'user', content: '[Heartbeat removed from conversation history]' },
            { role: 'user', content: 'current request' },
        ]);
    });

    it('does not treat every compatible reasoning field as an immutable history chain', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-reasoning',
            object: 'chat.completion',
            created: 1,
            model: 'compatible-model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok', reasoning_content: 'reasoning projection' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const imagePrompt: OpenAIChatCompletionsPrompt = {
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'question' },
                        { type: 'image_url', image_url: { url: 'data:image/png;base64,aW1hZ2U=' } },
                    ],
                },
            ],
        };

        const completion = await model.requestTextCompletion(undefined, imagePrompt, {
            ...options,
            stripImagesAfterTurns: 0,
            stripTextMaxTokens: 1,
        });
        expect(JSON.stringify(completion.conversation)).toContain('aW1hZ2U=');
        expect(JSON.stringify(completion.conversation)).toContain('reasoning projection');
        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            conversation: completion.conversation,
            stripImagesAfterTurns: 0,
            stripTextMaxTokens: 1,
        });
        expect(JSON.stringify(model.payloads[1].messages)).not.toContain('aW1hZ2U=');
        expect(JSON.stringify(model.payloads[1].messages)).toContain('Content truncated');
    });

    it('reads streaming content arrays', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { content: [{ type: 'text', text: 'hello' }] } }],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([{ type: 'text', value: 'hello' }]);
    });

    it('streams reasoning fields as thoughts', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { reasoning_content: 'fallback' } }],
                    }),
                },
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { reasoning: ' text' }, finish_reason: 'stop' }],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([
            { type: 'thoughts', value: 'fallback' },
            { type: 'thoughts', value: ' text' },
        ]);
    });

    it('finalizes streaming conversation from native reasoning and tool-call deltas', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [
                            {
                                index: 0,
                                delta: {
                                    reasoning_content: 'plan',
                                    tool_calls: [
                                        {
                                            index: 0,
                                            id: 'call-1',
                                            type: 'function',
                                            function: { name: 'lookup', arguments: '{"city":' },
                                        },
                                    ],
                                },
                            },
                        ],
                    }),
                },
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [
                            {
                                index: 0,
                                finish_reason: 'tool_calls',
                                delta: {
                                    content: 'checking',
                                    tool_calls: [{ index: 0, function: { arguments: '"Paris"}' } }],
                                },
                            },
                        ],
                    }),
                },
            ]),
        );

        const stream = await model.requestTextCompletionStream(undefined, prompt, options);
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.();

        expect(legacyConversation(conversation)).toMatchObject({
            messages: expect.arrayContaining([
                expect.objectContaining({
                    role: 'assistant',
                    content: 'checking',
                    reasoning_content: 'plan',
                    tool_calls: [
                        {
                            id: 'call-1',
                            type: 'function',
                            function: { name: 'lookup', arguments: '{"city":"Paris"}' },
                        },
                    ],
                }),
            ]),
        });
    });

    it('streams native and embedded reasoning as thoughts', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { reasoning: 'fallback text' } }],
                    }),
                },
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { content: '<think>hidden</think>' }, finish_reason: 'stop' }],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));

        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([
            { type: 'thoughts', value: 'fallback text' },
            { type: 'thoughts', value: 'hidden' },
        ]);
    });

    it('normalizes non-streaming tool-call finish reasons', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            {
                                id: 'call_1',
                                type: 'function',
                                function: { name: 'get_weather', arguments: '{"location":"Paris"}' },
                            },
                        ],
                    },
                    finish_reason: 'tool_calls',
                    logprobs: null,
                },
            ],
        });

        const completion = await model.requestTextCompletion(undefined, prompt, options);

        expect(completion.finish_reason).toBe('tool_use');
    });

    it('injects synthetic tool results for interrupted prior Chat Completions tool calls', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const toolCall = {
            id: 'call_1',
            type: 'function' as const,
            function: { name: 'lookup', arguments: '{"city":"Paris"}' },
        };

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            conversation: {
                _is_openai_chat_completions: true,
                messages: [{ role: 'assistant', content: null, tool_calls: [toolCall] }],
            },
            tools: [{ name: 'lookup', description: 'Lookup weather', input_schema: { type: 'object' } }],
        });

        expect(model.payloads[0].messages).toEqual([
            { role: 'assistant', content: null, tool_calls: [toolCall] },
            {
                role: 'tool',
                tool_call_id: 'call_1',
                content: '[Tool interrupted: The user stopped the operation before "lookup" could execute.]',
            },
            { role: 'user', content: 'Hello' },
        ]);
    });

    it('temporarily accepts legacy array-shaped conversations', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            conversation: [{ role: 'assistant', content: 'legacy response' }],
        });

        expect(model.payloads[0].messages).toEqual([
            { role: 'assistant', content: 'legacy response' },
            { role: 'user', content: 'Hello' },
        ]);
    });

    it('converts prior tool calls and results to text when no tools are provided', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'ok' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            conversation: {
                _is_openai_chat_completions: true,
                messages: [
                    {
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
                    { role: 'tool', tool_call_id: 'call_1', content: '15 degrees celsius, sunny' },
                ],
            },
            tools: [],
        });

        expect(model.payloads[0].messages).toEqual([
            { role: 'assistant', content: '[Tool call: lookup({"city":"Paris"})]' },
            { role: 'user', content: '[Tool result: 15 degrees celsius, sunny]' },
            { role: 'user', content: 'Hello' },
        ]);
    });

    it('rejects a required tool choice when no tools are available', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [],
        });

        await expect(
            model.requestTextCompletion(undefined, prompt, {
                ...options,
                tools: [],
                model_options: { _option_id: 'text-fallback', tool_choice: 'required' },
            }),
        ).rejects.toMatchObject({
            name: 'ToolChoiceConfigurationError',
            retryable: false,
            code: 400,
            message: expect.stringContaining('required tool choice was requested, but no tools are available'),
        });
        expect(model.payloads).toHaveLength(0);
    });

    it('throws when a non-streaming response has no content or tool calls', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: null },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await expect(model.requestTextCompletion(undefined, prompt, options)).rejects.toThrow(
            'Chat Completions response is not valid: no data',
        );
    });

    it('preserves streaming function tool-call chunks', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [
                            {
                                index: 0,
                                delta: {
                                    tool_calls: [
                                        {
                                            index: 0,
                                            id: 'call_real_id',
                                            type: 'function',
                                            function: { name: 'get_weather', arguments: '{"location"' },
                                        },
                                    ],
                                },
                            },
                        ],
                    }),
                },
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [
                            {
                                index: 0,
                                delta: { tool_calls: [{ index: 0, function: { arguments: ':"Paris"}' } }] },
                                finish_reason: 'tool_calls',
                            },
                        ],
                    }),
                },
            ]),
        );

        const chunks = await collectChunks(await model.requestTextCompletionStream(undefined, prompt, options));
        const toolChunks = chunks.flatMap((chunk) => chunk.tool_use ?? []);

        expect(toolChunks.map((tool) => tool.id)).toEqual(['tool_0', 'tool_0']);
        expect(toolChunks[0]).toMatchObject({
            _actual_id: 'call_real_id',
            tool_name: 'get_weather',
            tool_input: '{"location"',
        });
        expect(toolChunks.map((tool) => tool.tool_input).join('')).toBe('{"location":"Paris"}');
        expect(chunks[chunks.length - 1].finish_reason).toBe('tool_use');
    });

    it('keeps empty argument deltas as strings and preserves a length stop', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [
                            {
                                index: 0,
                                delta: {
                                    tool_calls: [
                                        {
                                            index: 0,
                                            id: 'call_real_id',
                                            type: 'function',
                                            function: { name: 'write_artifact', arguments: '{"name"' },
                                        },
                                    ],
                                },
                            },
                        ],
                    }),
                },
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-1',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [
                            {
                                index: 0,
                                delta: { tool_calls: [{ index: 0, function: { arguments: '' } }] },
                                finish_reason: 'length',
                            },
                        ],
                    }),
                },
            ]),
        );

        const stream = await model.requestTextCompletionStream(undefined, prompt, options);
        const chunks = await collectChunks(stream);
        expect(chunks.flatMap((chunk) => chunk.tool_use ?? []).map((tool) => tool.tool_input)).toEqual(['{"name"', '']);
        expect(chunks.at(-1)?.finish_reason).toBe('length');
        const document = parseConversationDocument(await stream.finalizeConversation?.());
        expect(document.turns.at(-1)?.status).toBe('interrupted');
        expect(Object.values(document.generations).at(-1)?.status).toBe('cancelled');
    });

    it('normalizes tool schemas and structured-output schemas for Chat Completions payloads', async () => {
        const model = new TestOpenAIChatCompletionsProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: '{"answer":"Paris"}' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });

        await model.requestTextCompletion(undefined, prompt, {
            ...options,
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
            tools: [
                {
                    name: 'get_weather',
                    description: 'Get weather',
                    input_schema: {
                        type: 'object',
                        properties: { location: { type: 'string' } },
                        required: ['location'],
                        additionalProperties: false,
                    },
                },
            ],
        });

        expect(model.payloads[0].response_format).toEqual({
            type: 'json_schema',
            json_schema: {
                name: 'output',
                strict: true,
                schema: {
                    type: 'object',
                    properties: { answer: { type: 'string' } },
                    required: ['answer'],
                    additionalProperties: false,
                },
            },
        });
        const toolDefinition = model.payloads[0].tools?.[0];
        expect(toolDefinition?.type).toBe('function');
        if (toolDefinition?.type !== 'function') {
            throw new Error('Expected function tool definition');
        }
        expect(toolDefinition.function).toEqual({
            name: 'get_weather',
            description: 'Get weather',
            strict: true,
            parameters: {
                type: 'object',
                properties: { location: { type: 'string' } },
                required: ['location'],
                additionalProperties: false,
            },
        });
    });

    it('rejects a truncated stream before persisting a completed canonical response', async () => {
        const model = new TestOpenAIChatCompletionsProtocol(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-truncated',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { content: 'partial' } }],
                    }),
                },
            ]),
        );

        const stream = await model.requestTextCompletionStream(undefined, prompt, options);
        await collectChunks(stream);
        if (stream.finalizeConversation === undefined) throw new Error('Expected canonical stream finalizer');
        await expect(stream.finalizeConversation()).rejects.toThrow(
            'Chat Completions stream ended without a terminal finish reason',
        );
    });

    it('validates structured output through the full driver and recovers it without a second provider call', async () => {
        const rawText = '{ "answer" : "Tokyo" }';
        const driver = new TestOpenAIChatCompletionsDriver({
            id: 'chatcmpl-structured',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: rawText },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
            usage: {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
                prompt_tokens_details: { cached_tokens: 25, cache_write_tokens: 5 },
                cost: 0.0012,
                is_byok: false,
            },
        });
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];
        const first = await driver.execute(segments, {
            ...canonicalOptions('attempt:first', '2026-09-11T00:00:00.000Z'),
            result_schema: resultSchema,
        });

        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo' } }]);
        expect(first.token_usage).toEqual({
            prompt: 100,
            prompt_cached: 25,
            prompt_cache_write: 5,
            prompt_new: 70,
            provider_cost_usd: 0.0012,
            result: 20,
            total: 120,
        });
        expect(latestGeneratedJson(first.conversation)).toEqual({ answer: 'Tokyo' });
        expect(legacyConversation(first.conversation).messages.at(-1)?.content).toBe(rawText);

        const retried = await driver.execute(segments, {
            ...canonicalOptions('attempt:retry', '2026-09-11T00:01:00.000Z', first.conversation),
            result_schema: resultSchema,
        });
        expect(retried.result).toEqual(first.result);
        expect(retried.token_usage).toEqual(first.token_usage);
        expect(retried.conversation).toEqual(first.conversation);
        expect(driver.payloads).toHaveLength(1);
    });

    it('executes directly into canonical JSON and recovers without a second provider call', async () => {
        const driver = new TestOpenAIChatCompletionsDriver({
            id: 'chatcmpl-canonical-direct',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            service_tier: 'priority',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: '{"answer":"Tokyo"}' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
            usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        });
        const segments = [{ role: PromptRole.user, content: 'Return JSON.' }];
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.executeCanonical(segments, {
            ...canonicalOptions('attempt:canonical:first', '2026-09-11T00:00:00.000Z'),
            result_schema: resultSchema,
        });

        expect(first.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } })]),
        );
        expect(first.accepted_output.generation.usage).toMatchObject({ input_tokens: 4, output_tokens: 2 });
        expect(first.service_tier).toBe('priority');

        const retry = await driver.executeCanonical(segments, {
            ...canonicalOptions(
                'attempt:canonical:retry',
                '2026-09-11T00:01:00.000Z',
                JSON.parse(JSON.stringify(first.conversation)),
            ),
            result_schema: resultSchema,
        });
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(driver.payloads).toHaveLength(1);
    });

    it('recovers an accepted tool call from a later externalized head without another provider call', async () => {
        const driver = new TestOpenAIChatCompletionsDriver({
            id: 'chatcmpl-canonical-tool',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            {
                                id: 'call-write',
                                type: 'function',
                                function: {
                                    name: 'write_artifact',
                                    arguments: '{ "path" : "notes.txt", "content" : "exact retained content" }',
                                },
                            },
                        ],
                    },
                    finish_reason: 'tool_calls',
                    logprobs: null,
                },
            ],
            usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        });
        const seed = createConversationDocument({
            id: 'conversation:structured-output',
            created_at: '2026-09-11T00:00:00.000Z',
        });
        const segments = [{ role: PromptRole.user, content: 'Write the artifact.' }];
        const executionOptions = {
            ...canonicalOptions(
                'attempt:externalized:first',
                '2026-09-11T00:00:00.000Z',
                parseConversationDocument(seed),
            ),
            tools: [
                {
                    name: 'write_artifact',
                    input_schema: {
                        type: 'object' as const,
                        properties: { path: { type: 'string' }, content: { type: 'string' } },
                        required: ['path', 'content'],
                        additionalProperties: false,
                    },
                },
            ],
        } satisfies ExecutionOptions;
        const first = await driver.executeCanonical(segments, executionOptions);
        const prepared = await prepareToolArgumentExternalization(first.conversation, 'call-write', ['content']);
        const replayArchives = prepared.replay_archives.map((archive, index) => ({
            replay_block_id: archive.replay_block_id,
            asset: {
                id: `asset:call-write:replay:${index}`,
                kind: 'document' as const,
                mime_type: 'application/json',
                storage: {
                    type: 'external' as const,
                    resolver: 'test.artifact',
                    locator: { artifact_path: `tool-inputs/call-write-replay-${index}.json` },
                },
                provenance: { type: 'imported' as const, source: 'test' },
                byte_length: archive.byte_length,
                content_hash: archive.content_hash,
                created_at: '2026-09-11T00:01:00.000Z',
            },
        }));
        expect(replayArchives).toHaveLength(1);
        const externalized = await externalizeToolCallArguments(first.conversation, {
            operation_id: 'externalize:call-write',
            expected_revision: first.conversation.revision,
            recorded_at: '2026-09-11T00:01:00.000Z',
            call_id: 'call-write',
            input_path: ['content'],
            model_value: { path: 'notes.txt', content: '[stored externally]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: {
                id: 'asset:call-write',
                kind: 'text',
                mime_type: 'text/plain',
                storage: {
                    type: 'external',
                    resolver: 'test.artifact',
                    locator: { artifact_path: 'tool-inputs/call-write.txt' },
                },
                provenance: { type: 'imported', source: 'test' },
                byte_length: prepared.byte_length,
                content_hash: prepared.content_hash,
                created_at: '2026-09-11T00:01:00.000Z',
            },
            replay_archives: replayArchives,
        });

        const retry = await driver.executeCanonical(segments, {
            ...executionOptions,
            conversation: JSON.parse(JSON.stringify(externalized.document)),
            conversation_runtime: {
                ...executionOptions.conversation_runtime,
                attempt_id: 'attempt:externalized:delivery-2',
                recorded_at: '2026-09-11T00:02:00.000Z',
                started_at: '2026-09-11T00:02:00.000Z',
            } as NonNullable<ExecutionOptions['conversation_runtime']>,
            load_recovered_canonical_output: async () => first.accepted_output,
        });

        expect(retry.conversation).toEqual(externalized.document);
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(retry.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([expect.objectContaining({ type: 'tool_call', call_id: 'call-write' })]),
        );
        expect(legacyCompletionFromCanonicalExecution(retry).result).toEqual([]);
        let externalizedCall: ContentBlock | undefined;
        for (const turn of retry.conversation.turns) {
            externalizedCall = (turn.blocks as ContentBlock[]).find(
                (block) => block.type === 'tool_call' && block.call_id === 'call-write',
            );
            if (externalizedCall !== undefined) break;
        }
        expect(externalizedCall).toMatchObject({
            type: 'tool_call',
            call_id: 'call-write',
            arguments: { type: 'externalized_json', exact_arguments_hash: prepared.exact_arguments_hash },
        });
        expect(driver.payloads).toHaveLength(1);
    });

    it('streams invalid required JSON into an explicit failed canonical outcome', async () => {
        const driver = new TestOpenAIChatCompletionsDriver(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-canonical-invalid-stream',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        service_tier: 'priority',
                        choices: [{ index: 0, delta: { content: '{"wrong":true}' }, finish_reason: 'stop' }],
                        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
                    }),
                },
            ]),
        );
        const stream = await driver.streamCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...canonicalOptions('attempt:canonical:invalid-stream', '2026-09-11T00:00:00.000Z'),
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
        });
        for await (const _chunk of stream) {
            // Drain through canonical finalization.
        }

        expect(stream.completion?.accepted_output.generation.status).toBe('failed');
        expect(stream.completion?.accepted_output.turn.status).toBe('failed');
        expect(stream.completion?.service_tier).toBe('priority');
    });

    it('continues an interrupted turn with a complete call while keeping a malformed trailing call non-executable', async () => {
        const tools: NonNullable<ExecutionOptions['tools']> = [
            {
                name: 'lookup_weather',
                input_schema: {
                    type: 'object',
                    properties: { city: { type: 'string' } },
                    required: ['city'],
                    additionalProperties: false,
                },
            },
        ];
        const firstDriver = new TestOpenAIChatCompletionsDriver({
            id: 'chatcmpl-cutoff-calls',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            {
                                id: 'call-complete',
                                type: 'function',
                                function: { name: 'lookup_weather', arguments: '{"city":"Tokyo"}' },
                            },
                            {
                                id: 'call-truncated',
                                type: 'function',
                                function: { name: 'lookup_weather', arguments: '{"city":' },
                            },
                        ],
                    },
                    finish_reason: 'length',
                    logprobs: null,
                },
            ],
        });
        const first = await firstDriver.executeCanonical([{ role: PromptRole.user, content: 'Check Tokyo.' }], {
            ...canonicalOptions('attempt:cutoff:first', '2026-09-11T00:02:00.000Z'),
            tools,
        });

        expect(first.accepted_output.generation).toMatchObject({ status: 'cancelled', finish_reason: 'length' });
        const firstDocument = parseConversationDocument(first.conversation);
        expect(firstDocument.turns.at(-1)?.status).toBe('interrupted');
        expect(
            first.accepted_output.turn.blocks.find(
                (block) => block.type === 'tool_call' && block.call_id === 'call-complete',
            ),
        ).toMatchObject({ arguments: { type: 'json', value: { city: 'Tokyo' } } });
        expect(
            firstDocument.turns
                .at(-1)
                ?.blocks.find((block) => block.type === 'tool_call' && block.call_id === 'call-truncated'),
        ).toMatchObject({ arguments: { type: 'invalid', raw: '{"city":' } });

        const secondDriver = new TestOpenAIChatCompletionsDriver({
            id: 'chatcmpl-after-cutoff-tool',
            object: 'chat.completion',
            created: 2,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'It is sunny.' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        await secondDriver.execute(
            [
                {
                    role: PromptRole.tool,
                    content: '{"condition":"sunny"}',
                    tool_use_id: 'call-complete',
                },
            ],
            {
                model: 'test/model',
                conversation: first.conversation,
                tools,
                conversation_runtime: {
                    conversation_id: firstDocument.id,
                    request_id: 'request:cutoff:continue',
                    attempt_id: 'attempt:cutoff:continue',
                    input_operation_id: 'input:cutoff:continue',
                    response_operation_id: 'response:cutoff:continue',
                    recorded_at: '2026-09-11T00:03:00.000Z',
                    started_at: '2026-09-11T00:03:00.000Z',
                },
            },
        );

        expect(secondDriver.payloads[0]?.messages).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    role: 'assistant',
                    tool_calls: expect.arrayContaining([
                        expect.objectContaining({ id: 'call-complete' }),
                        expect.objectContaining({ id: 'call-truncated' }),
                    ]),
                }),
                { role: 'tool', tool_call_id: 'call-complete', content: '{"condition":"sunny"}' },
            ]),
        );

        const invalidResultDriver = new TestOpenAIChatCompletionsDriver();
        await expect(
            invalidResultDriver.execute(
                [
                    {
                        role: PromptRole.tool,
                        content: 'must not execute',
                        tool_use_id: 'call-truncated',
                    },
                ],
                {
                    model: 'test/model',
                    conversation: first.conversation,
                    tools,
                    conversation_runtime: {
                        conversation_id: firstDocument.id,
                        request_id: 'request:cutoff:invalid-result',
                        attempt_id: 'attempt:cutoff:invalid-result',
                        input_operation_id: 'input:cutoff:invalid-result',
                        response_operation_id: 'response:cutoff:invalid-result',
                        recorded_at: '2026-09-11T00:04:00.000Z',
                    },
                },
            ),
        ).rejects.toThrow('Conversation document validation failed');
        expect(invalidResultDriver.payloads).toHaveLength(0);
    });

    it.each([
        ['omitted cache details', undefined],
        ['reported zero cache counts', { cached_tokens: 0, cache_write_tokens: 0 }],
    ] as const)('preserves %s across accepted-response recovery', async (_label, promptTokensDetails) => {
        const driver = new TestOpenAIChatCompletionsDriver({
            id: 'chatcmpl-usage-parity',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'done' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
            usage: {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
                ...(promptTokensDetails === undefined ? {} : { prompt_tokens_details: promptTokensDetails }),
            },
        });
        const segments = [{ role: PromptRole.user, content: 'Answer.' }];
        const first = await driver.execute(
            segments,
            canonicalOptions('attempt:usage:first', '2026-09-11T01:00:00.000Z'),
        );
        const retried = await driver.execute(
            segments,
            canonicalOptions('attempt:usage:retry', '2026-09-11T01:01:00.000Z', first.conversation),
        );

        expect(first.token_usage).toMatchObject({ prompt: 10, prompt_new: 10, result: 2, total: 12 });
        expect(retried.token_usage).toEqual(first.token_usage);
        expect(driver.payloads).toHaveLength(1);
    });

    it('persists streamed structured output as canonical JSON and recovers without another provider call', async () => {
        const rawText = '{"answer":"Tokyo"}';
        const driver = new TestOpenAIChatCompletionsDriver(
            undefined,
            createSSEStream([
                {
                    type: 'event',
                    data: JSON.stringify({
                        id: 'chatcmpl-stream-structured',
                        object: 'chat.completion.chunk',
                        created: 1,
                        model: 'test/model',
                        choices: [{ index: 0, delta: { content: rawText }, finish_reason: 'stop' }],
                    }),
                },
            ]),
        );
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Return the city.' }], {
            ...canonicalOptions('attempt:stream', '2026-09-11T00:00:00.000Z'),
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
        });

        for await (const _chunk of stream) {
            // Consume the public stream so core finalization and schema validation run.
        }
        expect(stream.completion?.result).toEqual([{ type: 'json', value: { answer: 'Tokyo' } }]);
        expect(latestGeneratedJson(stream.completion?.conversation)).toEqual({ answer: 'Tokyo' });
        expect(legacyConversation(stream.completion?.conversation).messages.at(-1)?.content).toBe(rawText);

        const recovered = await driver.stream([{ role: PromptRole.user, content: 'Return the city.' }], {
            ...canonicalOptions(
                'attempt:stream:retry',
                '2026-09-11T00:01:00.000Z',
                JSON.parse(JSON.stringify(stream.completion?.conversation)),
            ),
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
                additionalProperties: false,
            },
        });
        for await (const _chunk of recovered) {
            // Consume the recovered public stream so finalization runs.
        }
        expect(recovered.completion?.result).toEqual(stream.completion?.result);
        expect(recovered.completion?.conversation).toEqual(stream.completion?.conversation);
        expect(driver.payloads).toHaveLength(1);
    });

    it('keeps invalid structured output as canonical source and reports the full-driver validation error', async () => {
        const rawText = '{"answer":{}}';
        const driver = new TestOpenAIChatCompletionsDriver({
            id: 'chatcmpl-invalid-structured',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: rawText },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        });
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const segments = [{ role: PromptRole.user, content: 'Return the city.' }];
        const completion = await driver.execute(segments, {
            ...canonicalOptions('attempt:invalid', '2026-09-11T00:00:00.000Z'),
            result_schema: resultSchema,
        });

        expect(completion.error).toMatchObject({ code: 'validation_error' });
        expect(latestGeneratedText(completion.conversation)).toBe(rawText);
        const recovered = await driver.execute(segments, {
            ...canonicalOptions('attempt:invalid:retry', '2026-09-11T00:01:00.000Z', completion.conversation),
            result_schema: resultSchema,
        });
        expect(recovered.error).toEqual(completion.error);
        expect(recovered.conversation).toEqual(completion.conversation);
        expect(driver.payloads).toHaveLength(1);
    });
});
