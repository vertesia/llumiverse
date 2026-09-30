import { FinishReason, GenerateContentResponse, GoogleGenAI } from '@google/genai';
import { type ConversationStreamEvent, fingerprintJson, parseConversationDocument } from '@llumiverse/conversation';
import { type DataSource, type ExecutionOptions, isCanonicalAcceptedRecovery, PromptRole } from '@llumiverse/core';
import OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from '../anthropic/index.js';
import { AzureFoundryDriver } from '../azure/azure_foundry.js';
import { formatConversePrompt } from '../bedrock/converse.js';
import { BedrockMantleDriver } from '../bedrock-mantle/index.js';
import { executeOpenAIAudioCanonical } from '../openai/audio.js';
import { OpenAIDriver } from '../openai/openai.js';
import { OpenAIChatCompletionsDriver } from '../openai/openai_chat_completions.js';
import {
    isOpenAIChatCompletionsHistory,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    prepareOpenAIChatCanonicalState,
} from '../openai/openai-chat-conversation-adapter.js';
import { VertexAIDriver } from '../vertexai/index.js';

vi.mock('@aws/bedrock-token-generator', () => ({ getTokenProvider: vi.fn(() => async () => 'test') }));
const bytes = new Uint8Array([1, 2, 3, 4]);
const base64 = Buffer.from(bytes).toString('base64');
const contentHash = 'sha256:9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a';
function file(uri = 'gs://bucket/recording.wav'): DataSource {
    return {
        name: 'recording.wav',
        mime_type: 'audio/wav',
        getURI: vi.fn(async () => uri),
        getURL: vi.fn(async () => 'https://example.test/recording.wav'),
        getStream: vi.fn(async () => new Blob([bytes]).stream()),
    };
}
const prompt = [{ role: PromptRole.user, content: 'Hello.' }];
const store = vi.fn<NonNullable<ExecutionOptions['store_audio']>>(async (stream) => {
    expect(new Uint8Array(await new Response(stream).arrayBuffer())).toEqual(bytes);
    return 'gs://bucket/speech.pcm';
});
function chatResponse(model: string, audio?: Pick<OpenAI.Chat.Completions.ChatCompletionAudio, 'data' | 'transcript'>) {
    return {
        id: 'test',
        object: 'chat.completion' as const,
        created: 1,
        model,
        choices: [
            {
                index: 0,
                message: {
                    role: 'assistant' as const,
                    content: 'A greeting.',
                    audio: audio && { id: 'audio', expires_at: 1, ...audio },
                    refusal: null,
                },
                finish_reason: 'stop' as const,
                logprobs: null,
            },
        ],
    };
}

async function collectCanonicalEvents(stream: AsyncIterable<ConversationStreamEvent>) {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('primary provider file audio', () => {
    it('routes Foundry speech using the deployment name and its SDK client', async () => {
        const service = new OpenAI({ apiKey: 'test' });
        const create = vi
            .spyOn(service.audio.speech, 'create')
            .mockImplementation(() => Promise.resolve(new Response(bytes)) as never);
        const driver = new AzureFoundryDriver({
            endpoint: 'https://example.test',
            azureADTokenProvider: { getToken: async () => ({ token: 'test', expiresOnTimestamp: Date.now() + 60000 }) },
        });
        driver.service = {
            endpoint: 'https://example.test',
            deployments: { get: async () => ({ modelPublisher: 'OpenAI' }) },
            getOpenAIClient: () => service,
        } as unknown as AzureFoundryDriver['service'];
        const result = await driver.execute(prompt, {
            model: 'speech-deployment::gpt-4o-mini-tts',
            store_audio: store,
        });
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ model: 'speech-deployment' }), expect.anything());
        expect(result.result[0].type).toBe('audio');
        expect(result.prompt).toEqual([]);

        const canonical = await driver.executeCanonical(prompt, {
            model: 'speech-deployment::gpt-4o-mini-tts',
            store_audio: store,
            conversation_runtime: {
                conversation_id: 'conversation:foundry-speech-sync',
                request_id: 'request:foundry-speech-sync',
                attempt_id: 'attempt:foundry-speech-sync',
                input_operation_id: 'input:foundry-speech-sync',
                response_operation_id: 'response:foundry-speech-sync',
                recorded_at: '2026-09-29T01:00:00.000Z',
            },
        });
        const canonicalAudio = canonical.accepted_output.turn.blocks.find((block) => block.type === 'audio');
        if (canonicalAudio?.type !== 'audio') throw new Error('Expected Foundry canonical audio output');
        expect(canonical.accepted_output.generation).toMatchObject({
            provider: 'azure_foundry',
            protocol: 'openai.audio.speech',
            requested_model: 'speech-deployment::gpt-4o-mini-tts',
            resolved_model: 'speech-deployment',
        });
        expect(canonical.accepted_output.assets[canonicalAudio.asset_id]).toMatchObject({
            storage: { type: 'external', resolver: 'url', locator: { url: 'gs://bucket/speech.pcm' } },
            media: { container: 'mp3' },
        });

        const stream = await driver.streamCanonical(prompt, {
            model: 'speech-deployment::gpt-4o-mini-tts',
            store_audio: store,
            conversation_runtime: {
                conversation_id: 'conversation:foundry-speech-stream',
                request_id: 'request:foundry-speech-stream',
                attempt_id: 'attempt:foundry-speech-stream',
                input_operation_id: 'input:foundry-speech-stream',
                response_operation_id: 'response:foundry-speech-stream',
                recorded_at: '2026-09-29T01:01:00.000Z',
            },
        });
        const chunks: string[] = [];
        for await (const chunk of stream) chunks.push(chunk);
        expect(chunks).toEqual(['[Audio]']);
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            provider: 'azure_foundry',
            protocol: 'openai.audio.speech',
            requested_model: 'speech-deployment::gpt-4o-mini-tts',
            resolved_model: 'speech-deployment',
        });
        const streamedAudio = stream.completion?.accepted_output.turn.blocks.find((block) => block.type === 'audio');
        if (streamedAudio?.type !== 'audio') throw new Error('Expected streamed Foundry canonical audio output');
        expect(stream.completion?.accepted_output.assets[streamedAudio.asset_id]?.media).toMatchObject({
            container: 'mp3',
        });
        const runtime = {
            conversation_id: 'conversation:foundry-speech-typed',
            request_id: 'request:foundry-speech-typed',
            attempt_id: 'attempt:foundry-speech-typed:first',
            input_operation_id: 'input:foundry-speech-typed',
            response_operation_id: 'response:foundry-speech-typed',
            recorded_at: '2026-09-29T01:02:00.000Z',
        };
        const publish = vi.fn(async () => undefined);
        const typed = await driver.streamCanonicalEvents(
            prompt,
            {
                model: 'speech-deployment::gpt-4o-mini-tts',
                store_audio: store,
                conversation_runtime: runtime,
                on_canonical_request_prepared: publish,
            },
            undefined,
            { stream_id: 'stream:foundry:speech:first' },
        );
        expect(await collectCanonicalEvents(typed)).toEqual([
            expect.objectContaining({
                type: 'response_accepted',
                origin: 'live_transport',
                stream_id: 'stream:foundry:speech:first',
            }),
        ]);
        expect(isCanonicalAcceptedRecovery(typed.completion)).toBe(false);
        const retry = await driver.streamCanonicalEvents(
            prompt,
            {
                model: 'speech-deployment::gpt-4o-mini-tts',
                store_audio: store,
                conversation: JSON.parse(JSON.stringify(typed.completion?.conversation)),
                conversation_runtime: {
                    ...runtime,
                    attempt_id: 'attempt:foundry-speech-typed:retry',
                    recorded_at: '2026-09-29T01:03:00.000Z',
                },
                on_canonical_request_prepared: publish,
            },
            undefined,
            { stream_id: 'stream:foundry:speech:retry' },
        );
        expect(await collectCanonicalEvents(retry)).toEqual([
            expect.objectContaining({
                type: 'response_accepted',
                origin: 'accepted_recovery',
                stream_id: 'stream:foundry:speech:retry',
            }),
        ]);
        expect(retry.completion?.accepted_output).toEqual(typed.completion?.accepted_output);
        expect(isCanonicalAcceptedRecovery(retry.completion)).toBe(true);
        expect(isCanonicalAcceptedRecovery(JSON.parse(JSON.stringify(retry.completion)))).toBe(false);
        expect(publish).toHaveBeenCalledOnce();
        const mismatched = await driver.streamCanonicalEvents(
            prompt,
            {
                model: 'speech-deployment::gpt-4o-mini-tts',
                model_options: { _option_id: 'openai-speech', voice: 'echo' },
                store_audio: store,
                conversation: typed.completion?.conversation,
                conversation_runtime: runtime,
                on_canonical_request_prepared: publish,
            },
            undefined,
            { stream_id: 'stream:foundry:speech:mismatch' },
        );
        expect(await collectCanonicalEvents(mismatched)).toEqual([
            expect.objectContaining({ type: 'stream_terminated', outcome: 'failed' }),
        ]);
        expect(create).toHaveBeenCalledTimes(4);
        expect(create.mock.calls.map(([request]) => request.model)).toEqual([
            'speech-deployment',
            'speech-deployment',
            'speech-deployment',
            'speech-deployment',
        ]);
    });

    it.each(['responses', 'chat'] as const)(
        'reaches finite typed OpenAI speech execution through the public %s driver',
        async (protocol) => {
            const driver =
                protocol === 'responses'
                    ? new OpenAIDriver({ apiKey: 'test' })
                    : new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://example.test/v1' });
            const create = vi
                .spyOn(driver.service.audio.speech, 'create')
                .mockImplementation(() => Promise.resolve(new Response(bytes)) as never);
            const publish = vi.fn(async () => undefined);
            const stream = await driver.streamCanonicalEvents(
                prompt,
                {
                    model: 'gpt-4o-mini-tts',
                    store_audio: store,
                    conversation_runtime: {
                        conversation_id: `conversation:openai-speech-${protocol}`,
                        request_id: `request:openai-speech-${protocol}`,
                        attempt_id: `attempt:openai-speech-${protocol}`,
                        input_operation_id: `input:openai-speech-${protocol}`,
                        response_operation_id: `response:openai-speech-${protocol}`,
                        recorded_at: '2026-09-29T01:04:00.000Z',
                    },
                    on_canonical_request_prepared: publish,
                },
                undefined,
                { stream_id: `stream:openai:speech:${protocol}` },
            );

            expect(await collectCanonicalEvents(stream)).toEqual([
                expect.objectContaining({
                    type: 'response_accepted',
                    origin: 'live_transport',
                    stream_id: `stream:openai:speech:${protocol}`,
                }),
            ]);
            expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
                expect.objectContaining({ type: 'audio' }),
            );
            expect(publish).toHaveBeenCalledOnce();
            expect(create).toHaveBeenCalledOnce();
        },
    );

    it.each(['responses', 'chat'] as const)(
        'requires explicit runtime identity before opening a typed %s audio stream',
        async (protocol) => {
            const driver =
                protocol === 'responses'
                    ? new OpenAIDriver({ apiKey: 'test' })
                    : new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://example.test/v1' });
            const create = vi.spyOn(driver.service.audio.speech, 'create');

            await expect(
                driver.streamCanonicalEvents(prompt, { model: 'gpt-4o-mini-tts', store_audio: store }, undefined, {
                    stream_id: `stream:audio:no-runtime:${protocol}`,
                }),
            ).rejects.toThrow('Canonical typed streaming requires conversation_runtime');
            expect(create).not.toHaveBeenCalled();
        },
    );

    it.each(['responses', 'chat'] as const)(
        'rejects an already cancelled typed %s audio stream before returning a producer',
        async (protocol) => {
            const driver =
                protocol === 'responses'
                    ? new OpenAIDriver({ apiKey: 'test' })
                    : new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://example.test/v1' });
            const create = vi.spyOn(driver.service.audio.speech, 'create');
            const options: ExecutionOptions = {
                model: 'gpt-4o-mini-tts',
                store_audio: store,
                conversation_runtime: {
                    request_id: `request:audio:aborted:${protocol}`,
                    attempt_id: `attempt:audio:aborted:${protocol}`,
                    input_operation_id: `input:audio:aborted:${protocol}`,
                    response_operation_id: `response:audio:aborted:${protocol}`,
                    recorded_at: '2026-09-29T01:08:00.000Z',
                },
            };

            await expect(
                driver.streamCanonicalEvents(
                    prompt,
                    options,
                    AbortSignal.abort(new Error('cancelled before audio open')),
                    { stream_id: `stream:audio:aborted:${protocol}` },
                ),
            ).rejects.toThrow('cancelled before audio open');
            expect(create).not.toHaveBeenCalled();
        },
    );

    it('fails finite typed audio at the prepared-request barrier before opening transport', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi
            .spyOn(driver.service.audio.speech, 'create')
            .mockImplementation(() => Promise.resolve(new Response(bytes)) as never);
        const stream = await driver.streamCanonicalEvents(
            prompt,
            {
                model: 'gpt-4o-mini-tts',
                store_audio: store,
                conversation_runtime: {
                    conversation_id: 'conversation:openai-speech-barrier',
                    request_id: 'request:openai-speech-barrier',
                    attempt_id: 'attempt:openai-speech-barrier',
                    input_operation_id: 'input:openai-speech-barrier',
                    response_operation_id: 'response:openai-speech-barrier',
                    recorded_at: '2026-09-29T01:05:00.000Z',
                },
                on_canonical_request_prepared: async () => {
                    throw new Error('publication rejected');
                },
            },
            undefined,
            { stream_id: 'stream:openai:speech:barrier' },
        );

        expect(await collectCanonicalEvents(stream)).toEqual([
            expect.objectContaining({ type: 'stream_terminated', outcome: 'failed' }),
        ]);
        expect(create).not.toHaveBeenCalled();
    });

    it('rejects unsupported finite typed audio input before transport', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.audio.speech, 'create');

        await expect(
            driver.streamCanonicalEvents(
                prompt,
                {
                    model: 'gpt-4o-mini-tts',
                    store_audio: store,
                    tools: [{ name: 'unexpected', input_schema: { type: 'object' } }],
                    conversation_runtime: {
                        conversation_id: 'conversation:openai-speech-unsupported',
                        request_id: 'request:openai-speech-unsupported',
                        attempt_id: 'attempt:openai-speech-unsupported',
                        input_operation_id: 'input:openai-speech-unsupported',
                        response_operation_id: 'response:openai-speech-unsupported',
                        recorded_at: '2026-09-29T01:06:00.000Z',
                    },
                },
                undefined,
                { stream_id: 'stream:openai:speech:unsupported' },
            ),
        ).rejects.toThrow('do not accept tools');
        expect(create).not.toHaveBeenCalled();
    });

    it('settles typed audio cancellation while retaining transport ownership until the request settles', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        let resolveTransport!: (response: Response) => void;
        const create = vi.spyOn(driver.service.audio.speech, 'create').mockImplementation(
            () =>
                new Promise<Response>((resolve) => {
                    resolveTransport = resolve;
                }) as never,
        );
        const stream = await driver.streamCanonicalEvents(
            prompt,
            {
                model: 'gpt-4o-mini-tts',
                store_audio: store,
                conversation_runtime: {
                    conversation_id: 'conversation:openai-speech-cancel',
                    request_id: 'request:openai-speech-cancel',
                    attempt_id: 'attempt:openai-speech-cancel',
                    input_operation_id: 'input:openai-speech-cancel',
                    response_operation_id: 'response:openai-speech-cancel',
                    recorded_at: '2026-09-29T01:07:00.000Z',
                },
            },
            undefined,
            { stream_id: 'stream:openai:speech:cancel' },
        );
        const pending = stream[Symbol.asyncIterator]().next();
        await vi.waitFor(() => expect(create).toHaveBeenCalledOnce());

        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toMatchObject({ value: terminal });
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);

        resolveTransport(new Response(bytes));
        await stream.closed;
        expect(closed).toBe(true);
        expect(stream.terminal_event).toEqual(terminal);
        expect(stream.completion).toBeUndefined();
    });

    it.each([
        ['openai', true],
        ['compatible', true],
        ['openai', false],
        ['compatible', false],
    ] as const)(
        'uses Chat audio for %s with attachment=%s and persists the generated output',
        async (provider, hasAttachment) => {
            const driver =
                provider === 'openai'
                    ? new OpenAIDriver({ apiKey: 'test' })
                    : new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://example.test/v1' });
            const create = vi
                .spyOn(driver.service.chat.completions, 'create')
                .mockResolvedValue(chatResponse('gpt-audio', { data: base64, transcript: 'A greeting.' }));
            const result = await driver.execute(
                [{ role: PromptRole.user, content: 'Describe', files: hasAttachment ? [file()] : [] }],
                {
                    model: 'gpt-audio',
                    include_original_response: true,
                    store_audio: store,
                },
            );
            expect(create).toHaveBeenCalledWith(
                expect.objectContaining({
                    modalities: ['text', 'audio'],
                    messages: [
                        {
                            role: 'user',
                            content: [
                                { type: 'text', text: 'Describe' },
                                ...(hasAttachment
                                    ? [{ type: 'input_audio', input_audio: { data: base64, format: 'wav' } }]
                                    : []),
                            ],
                        },
                    ],
                }),
                expect.anything(),
            );
            expect(store).toHaveBeenCalled();
            expect(result.result).toContainEqual(
                expect.objectContaining({ type: 'audio', value: 'gs://bucket/speech.pcm', mime_type: 'audio/wav' }),
            );
            expect(result.original_response).toMatchObject({ choices: [{ message: { content: 'A greeting.' } }] });
            expect(JSON.stringify(result)).not.toContain(base64);
            expect(result.conversation).toBeUndefined();
        },
    );

    it.each(['text', 'audio_bytes', 'voice', 'tools'] as const)(
        'executes OpenAI audio understanding durably and rejects changed %s on accepted retry',
        async (changed) => {
            const driver = new OpenAIDriver({ apiKey: 'test' });
            const create = vi
                .spyOn(driver.service.chat.completions, 'create')
                .mockResolvedValue(chatResponse('gpt-audio', { data: base64, transcript: 'A greeting.' }));
            const runtime = {
                conversation_id: 'conversation:openai-audio',
                request_id: 'request:openai-audio',
                attempt_id: 'attempt:openai-audio:first',
                input_operation_id: 'input:openai-audio',
                response_operation_id: 'response:openai-audio',
                recorded_at: '2026-09-29T00:10:00.000Z',
            };
            const first = await driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Describe', files: [file()] }],
                { model: 'gpt-audio', store_audio: store, conversation_runtime: runtime },
            );
            expect(isCanonicalAcceptedRecovery(first)).toBe(false);
            const document = parseConversationDocument(first.conversation);
            expect(Object.values(document.assets)).toEqual(
                expect.arrayContaining([
                    expect.objectContaining({
                        kind: 'audio',
                        storage: { type: 'inline_base64', data: base64 },
                        byte_length: bytes.byteLength,
                        content_hash: contentHash,
                        provenance: expect.objectContaining({ type: 'received' }),
                    }),
                    expect.objectContaining({
                        kind: 'audio',
                        storage: {
                            type: 'external',
                            resolver: 'url',
                            locator: { url: 'gs://bucket/speech.pcm' },
                        },
                        byte_length: bytes.byteLength,
                        content_hash: contentHash,
                        media: { container: 'wav' },
                        provenance: expect.objectContaining({ type: 'generated' }),
                    }),
                ]),
            );
            expect(first.accepted_output.turn.blocks).toEqual(
                expect.arrayContaining([
                    expect.objectContaining({ type: 'text', text: 'A greeting.' }),
                    expect.objectContaining({ type: 'audio' }),
                ]),
            );

            if (changed === 'text') {
                const mismatchedFile = file();
                await expect(
                    driver.executeCanonical([{ role: PromptRole.user, content: 'Describe', files: [mismatchedFile] }], {
                        model: 'gpt-audio',
                        store_audio: store,
                        conversation: first.conversation,
                        conversation_runtime: {
                            ...runtime,
                            conversation_id: 'conversation:other-audio',
                        },
                    }),
                ).rejects.toThrow('conversation_runtime.conversation_id does not match');
                expect(mismatchedFile.getStream).not.toHaveBeenCalled();
                expect(create).toHaveBeenCalledOnce();
            }

            const retry = await driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Describe', files: [file()] }],
                {
                    model: 'gpt-audio',
                    store_audio: store,
                    conversation: JSON.parse(JSON.stringify(first.conversation)),
                    conversation_runtime: {
                        ...runtime,
                        attempt_id: 'attempt:openai-audio:retry',
                        recorded_at: '2026-09-29T00:11:00.000Z',
                    },
                },
            );
            expect(retry.accepted_output).toEqual(first.accepted_output);
            expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
            expect(create).toHaveBeenCalledOnce();

            await expect(
                driver.executeCanonical([{ role: PromptRole.user, content: 'Describe', files: [file()] }], {
                    model: 'gpt-audio',
                    store_audio: store,
                    conversation: first.conversation,
                    conversation_runtime: { ...runtime, request_id: 'request:other' },
                }),
            ).rejects.toThrow('incompatible request identity');
            expect(create).toHaveBeenCalledOnce();

            const changedFile = file();
            if (changed === 'audio_bytes') {
                changedFile.getStream = async () => new Blob([new Uint8Array([9, 8, 7, 6])]).stream();
            }
            await expect(
                driver.executeCanonical(
                    [
                        {
                            role: PromptRole.user,
                            content: changed === 'text' ? 'Translate' : 'Describe',
                            files: [changedFile],
                        },
                    ],
                    {
                        model: 'gpt-audio',
                        store_audio: store,
                        conversation: first.conversation,
                        conversation_runtime: runtime,
                        ...(changed === 'voice'
                            ? { model_options: { _option_id: 'openai-audio' as const, voice: 'echo' as const } }
                            : {}),
                        ...(changed === 'tools'
                            ? {
                                  tools: [
                                      {
                                          name: 'unexpected',
                                          description: 'Unexpected tool',
                                          input_schema: { type: 'object' as const, properties: {} },
                                      },
                                  ],
                              }
                            : {}),
                    },
                ),
            ).rejects.toThrow();
            expect(create).toHaveBeenCalledOnce();
        },
    );

    it('stores OpenAI speech synthesis as a content-bound canonical asset', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.audio.speech, 'create').mockResolvedValue(new Response(bytes));
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Hello.' }], {
            model: 'gpt-4o-mini-tts',
            store_audio: store,
            conversation_runtime: {
                conversation_id: 'conversation:openai-speech',
                request_id: 'request:openai-speech',
                attempt_id: 'attempt:openai-speech',
                input_operation_id: 'input:openai-speech',
                response_operation_id: 'response:openai-speech',
                recorded_at: '2026-09-29T00:15:00.000Z',
            },
        });
        const audioBlock = result.accepted_output.turn.blocks.find((block) => block.type === 'audio');
        if (audioBlock?.type !== 'audio') throw new Error('Expected canonical audio block');
        expect(result.accepted_output.assets[audioBlock.asset_id]).toMatchObject({
            kind: 'audio',
            storage: {
                type: 'external',
                resolver: 'url',
                locator: { url: 'gs://bucket/speech.pcm' },
            },
            byte_length: bytes.byteLength,
            content_hash: contentHash,
            media: { container: 'mp3', codec: 'mp3' },
        });
        expect(JSON.stringify(result.conversation)).not.toContain(base64);
        expect(create).toHaveBeenCalledOnce();
    });

    it('binds the resolved OpenAI request model across accepted audio retries', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi
            .spyOn(driver.service.chat.completions, 'create')
            .mockResolvedValue(chatResponse('deployment-a', { data: base64, transcript: 'A greeting.' }));
        const runtime = {
            conversation_id: 'conversation:openai-audio-deployment',
            request_id: 'request:openai-audio-deployment',
            attempt_id: 'attempt:openai-audio-deployment',
            input_operation_id: 'input:openai-audio-deployment',
            response_operation_id: 'response:openai-audio-deployment',
            recorded_at: '2026-09-29T00:16:00.000Z',
        };
        const first = await executeOpenAIAudioCanonical({
            service: driver.service,
            segments: [{ role: PromptRole.user, content: 'Describe' }],
            options: { model: 'gpt-audio', store_audio: store, conversation_runtime: runtime },
            provider: driver.provider,
            request_model: 'deployment-a',
        });

        await expect(
            executeOpenAIAudioCanonical({
                service: driver.service,
                segments: [{ role: PromptRole.user, content: 'Describe' }],
                options: {
                    model: 'gpt-audio',
                    store_audio: store,
                    conversation: first.conversation,
                    conversation_runtime: runtime,
                },
                provider: driver.provider,
                request_model: 'deployment-b',
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(create).toHaveBeenCalledOnce();
    });

    it('rejects canonical artifact audio until a host resolver contract is available', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        vi.spyOn(driver.service.audio.speech, 'create').mockResolvedValue(new Response(bytes));
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Hello.' }], {
                model: 'gpt-4o-mini-tts',
                store_audio: async (stream) => {
                    await new Response(stream).arrayBuffer();
                    return 'artifact:speech';
                },
                conversation_runtime: {
                    conversation_id: 'conversation:openai-artifact-speech',
                    request_id: 'request:openai-artifact-speech',
                    attempt_id: 'attempt:openai-artifact-speech',
                    input_operation_id: 'input:openai-artifact-speech',
                    response_operation_id: 'response:openai-artifact-speech',
                    recorded_at: '2026-09-29T00:17:00.000Z',
                },
            }),
        ).rejects.toThrow('cannot resolve durable URI artifact:speech');
    });

    it('rejects empty and multiple-file audio chat inputs before calling the provider', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.chat.completions, 'create');
        const options = { model: 'gpt-audio', store_audio: store };
        await expect(driver.execute([{ role: PromptRole.user, content: '  ' }], options)).rejects.toThrow(
            'requires text or an audio file',
        );
        await expect(
            driver.execute([{ role: PromptRole.user, content: '', files: [file(), file()] }], options),
        ).rejects.toThrow('at most one audio file');
        expect(create).not.toHaveBeenCalled();
    });

    it('requests diarized JSON and retains speaker segments', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = driver.service.withOptions({
            fetch: async (_url, init) => {
                const body = await new Response(init?.body).text();
                expect(body).toContain('diarized_json');
                expect(body).toContain('chunking_strategy');
                return Response.json({
                    text: 'Hello',
                    segments: [{ id: '0', speaker: 'A', start: 0, end: 1, text: 'Hello' }],
                });
            },
        });
        const result = await driver.execute([{ role: PromptRole.user, content: '', files: [file()] }], {
            model: 'gpt-4o-transcribe-diarize',
        });
        expect(result.result).toEqual([
            { type: 'text', value: 'Hello' },
            { type: 'json', value: { segments: [{ id: '0', speaker: 'A', start: 0, end: 1, text: 'Hello' }] } },
        ]);
    });

    it('records OpenAI diarized transcription as canonical text and JSON', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = driver.service.withOptions({
            fetch: async () =>
                Response.json({
                    text: 'Hello',
                    segments: [{ id: '0', speaker: 'A', start: 0, end: 1, text: 'Hello' }],
                }),
        });
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: '', files: [file()] }], {
            model: 'gpt-4o-transcribe-diarize',
            conversation_runtime: {
                conversation_id: 'conversation:openai-transcription',
                request_id: 'request:openai-transcription',
                attempt_id: 'attempt:openai-transcription',
                input_operation_id: 'input:openai-transcription',
                response_operation_id: 'response:openai-transcription',
                recorded_at: '2026-09-29T00:20:00.000Z',
            },
        });
        expect(result.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'text', text: 'Hello' }),
                expect.objectContaining({
                    type: 'json',
                    value: { segments: [{ id: '0', speaker: 'A', start: 0, end: 1, text: 'Hello' }] },
                }),
            ]),
        );
    });

    it('stores Vertex PCM and delivers the fallback completion without inline audio', async () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'global' });
        const client = new GoogleGenAI({ apiKey: 'test' });
        const response = new GenerateContentResponse();
        response.candidates = [
            { content: { parts: [{ inlineData: { mimeType: 'audio/L16;codec=pcm;rate=24000', data: base64 } }] } },
        ];
        const generate = vi.spyOn(client.models, 'generateContent').mockResolvedValue(response);
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
        const stream = await driver.stream(prompt, {
            model: 'gemini-3.1-flash-tts-preview',
            store_audio: store,
            include_original_response: true,
        });
        for await (const _chunk of stream) {
            /* consume finite completion */
        }
        expect(generate).toHaveBeenCalledWith(
            expect.objectContaining({ config: expect.objectContaining({ responseModalities: ['AUDIO'] }) }),
        );
        expect(stream.completion?.result[0]).toMatchObject({
            type: 'audio',
            value: 'gs://bucket/speech.pcm',
            sample_rate: 24000,
            channels: 1,
            byte_order: 'little',
        });
        expect(JSON.stringify(stream.completion)).not.toContain(base64);
        expect(stream.completion?.original_response).toBeDefined();
        expect(stream.completion?.conversation).toBeUndefined();
    });

    it.each(['text', 'voice', 'tools'] as const)(
        'stores Vertex PCM durably and rejects changed %s on accepted retry',
        async (changed) => {
            const driver = new VertexAIDriver({ project: 'test', region: 'global' });
            const client = new GoogleGenAI({ apiKey: 'test' });
            const response = new GenerateContentResponse();
            response.responseId = 'gemini-audio-response';
            response.modelVersion = 'gemini-3.1-flash-tts-preview';
            response.candidates = [
                {
                    finishReason: FinishReason.STOP,
                    content: { parts: [{ inlineData: { mimeType: 'audio/L16;codec=pcm;rate=24000', data: base64 } }] },
                },
            ];
            const generate = vi.spyOn(client.models, 'generateContent').mockResolvedValue(response);
            vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
            const runtime = {
                conversation_id: 'conversation:gemini-audio',
                request_id: 'request:gemini-audio',
                attempt_id: 'attempt:gemini-audio:first',
                input_operation_id: 'input:gemini-audio',
                response_operation_id: 'response:gemini-audio',
                recorded_at: '2026-09-29T01:00:00.000Z',
            };
            const first = await driver.executeCanonical(prompt, {
                model: 'gemini-3.1-flash-tts-preview',
                store_audio: store,
                conversation_runtime: runtime,
            });
            expect(isCanonicalAcceptedRecovery(first)).toBe(false);
            const audioBlock = first.accepted_output.turn.blocks.find((block) => block.type === 'audio');
            if (audioBlock?.type !== 'audio') throw new Error('Expected canonical audio block');
            expect(first.accepted_output.assets[audioBlock.asset_id]).toMatchObject({
                kind: 'audio',
                mime_type: 'audio/L16;codec=pcm;rate=24000',
                storage: {
                    type: 'external',
                    resolver: 'url',
                    locator: { url: 'gs://bucket/speech.pcm' },
                },
                byte_length: bytes.byteLength,
                content_hash: contentHash,
                media: {
                    container: 'raw',
                    codec: 'pcm',
                    sample_rate: 24000,
                    channels: 1,
                    sample_encoding: 'int16',
                    byte_order: 'little',
                },
            });
            const persisted = parseConversationDocument(first.conversation);
            expect(persisted.assets[audioBlock.asset_id]).toMatchObject({
                metadata: {
                    audio_result: {
                        value: 'gs://bucket/speech.pcm',
                        codec: 'pcm',
                        sample_rate: 24000,
                        channels: 1,
                    },
                },
            });
            expect(JSON.stringify(first.accepted_output)).not.toContain(base64);
            expect(JSON.stringify(first.conversation)).not.toContain(base64);

            const retry = await driver.executeCanonical(prompt, {
                model: 'gemini-3.1-flash-tts-preview',
                store_audio: store,
                conversation: JSON.parse(JSON.stringify(first.conversation)),
                conversation_runtime: {
                    ...runtime,
                    attempt_id: 'attempt:gemini-audio:retry',
                    recorded_at: '2026-09-29T01:01:00.000Z',
                },
            });
            expect(retry.accepted_output).toEqual(first.accepted_output);
            expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
            expect(generate).toHaveBeenCalledOnce();

            const typedRetryController = new AbortController();
            const recoveredStream = await driver.streamCanonicalEvents(
                prompt,
                {
                    model: 'gemini-3.1-flash-tts-preview',
                    store_audio: store,
                    conversation: JSON.parse(JSON.stringify(first.conversation)),
                    conversation_runtime: {
                        ...runtime,
                        attempt_id: 'attempt:gemini-audio:typed-retry',
                        recorded_at: '2026-09-29T01:02:00.000Z',
                    },
                },
                typedRetryController.signal,
                { stream_id: `stream:gemini-audio:${changed}:retry` },
            );
            expect(await collectCanonicalEvents(recoveredStream)).toEqual([
                expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
            ]);
            expect(isCanonicalAcceptedRecovery(recoveredStream.completion)).toBe(true);
            expect(generate).toHaveBeenCalledOnce();

            await expect(
                driver.executeCanonical(prompt, {
                    model: 'gemini-3.1-flash-tts-preview',
                    store_audio: store,
                    conversation: first.conversation,
                    conversation_runtime: { ...runtime, request_id: 'request:other' },
                }),
            ).rejects.toThrow('incompatible request identity');
            expect(generate).toHaveBeenCalledOnce();

            await expect(
                driver.executeCanonical(
                    changed === 'text' ? [{ role: PromptRole.user, content: 'Different speech' }] : prompt,
                    {
                        model: 'gemini-3.1-flash-tts-preview',
                        store_audio: store,
                        conversation: first.conversation,
                        conversation_runtime: runtime,
                        ...(changed === 'voice'
                            ? { model_options: { _option_id: 'vertexai-gemini' as const, speech_voice: 'Puck' } }
                            : {}),
                        ...(changed === 'tools'
                            ? {
                                  tools: [
                                      {
                                          name: 'unexpected',
                                          description: 'Unexpected tool',
                                          input_schema: { type: 'object' as const, properties: {} },
                                      },
                                  ],
                              }
                            : {}),
                    },
                ),
            ).rejects.toThrow();
            expect(generate).toHaveBeenCalledOnce();
        },
    );

    it('exact-recovers Gemini file audio from typed to sync with transport signals excluded from identity', async () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'global' });
        const client = new GoogleGenAI({ apiKey: 'test' });
        const response = new GenerateContentResponse();
        response.responseId = 'gemini-audio-cross-projection';
        response.modelVersion = 'gemini-3.1-flash-tts-preview';
        response.candidates = [
            {
                finishReason: FinishReason.STOP,
                content: { parts: [{ inlineData: { mimeType: 'audio/L16;codec=pcm;rate=24000', data: base64 } }] },
            },
        ];
        const generate = vi.spyOn(client.models, 'generateContent').mockResolvedValue(response);
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
        const publish = vi.fn(async () => undefined);
        const runtime = {
            conversation_id: 'conversation:gemini-audio-cross-projection',
            request_id: 'request:gemini-audio-cross-projection',
            attempt_id: 'attempt:gemini-audio-cross-projection:first',
            input_operation_id: 'input:gemini-audio-cross-projection',
            response_operation_id: 'response:gemini-audio-cross-projection',
            recorded_at: '2026-09-29T02:00:00.000Z',
        };
        const firstController = new AbortController();
        const first = await driver.streamCanonicalEvents(
            prompt,
            {
                model: 'gemini-3.1-flash-tts-preview',
                store_audio: store,
                conversation_runtime: runtime,
                on_canonical_request_prepared: publish,
            },
            firstController.signal,
            { stream_id: 'stream:gemini-audio-cross-projection:first' },
        );
        expect(await collectCanonicalEvents(first)).toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'live_transport' }),
        ]);
        expect(isCanonicalAcceptedRecovery(first.completion)).toBe(false);
        if (first.completion === undefined) throw new Error('Expected accepted Gemini audio response');

        const retry = await driver.executeCanonical(prompt, {
            model: 'gemini-3.1-flash-tts-preview',
            store_audio: store,
            conversation: JSON.parse(JSON.stringify(first.completion.conversation)),
            conversation_runtime: {
                ...runtime,
                attempt_id: 'attempt:gemini-audio-cross-projection:retry',
                recorded_at: '2026-09-29T02:01:00.000Z',
            },
            on_canonical_request_prepared: publish,
        });
        expect(retry.accepted_output).toEqual(first.completion.accepted_output);
        expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
        expect(generate).toHaveBeenCalledOnce();
        expect(store).toHaveBeenCalledOnce();
        expect(publish).toHaveBeenCalledOnce();

        const legacySignalBound = JSON.parse(JSON.stringify(first.completion.conversation));
        const generation = legacySignalBound.generations[first.completion.accepted_output.generation.id];
        const dispatchedPayload = JSON.parse(JSON.stringify(generate.mock.calls[0]?.[0]));
        expect(dispatchedPayload.config.abortSignal).toEqual({});
        generation.request_receipt.request_fingerprint = await fingerprintJson(dispatchedPayload);
        await expect(
            driver.executeCanonical(prompt, {
                model: 'gemini-3.1-flash-tts-preview',
                store_audio: store,
                conversation: legacySignalBound,
                conversation_runtime: {
                    ...runtime,
                    attempt_id: 'attempt:gemini-audio-cross-projection:legacy-signal',
                    recorded_at: '2026-09-29T02:01:30.000Z',
                },
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(generate).toHaveBeenCalledOnce();
        expect(store).toHaveBeenCalledOnce();
        expect(publish).toHaveBeenCalledOnce();
    });

    it('does not dispatch Gemini file audio when cancellation wins during the prepared barrier', async () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'global' });
        const client = new GoogleGenAI({ apiKey: 'test' });
        const generate = vi.spyOn(client.models, 'generateContent');
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
        const controller = new AbortController();
        const publish = vi.fn(async () => controller.abort(new Error('cancel before Gemini dispatch')));

        await expect(
            driver.executeCanonical(
                prompt,
                {
                    model: 'gemini-3.1-flash-tts-preview',
                    store_audio: store,
                    conversation_runtime: {
                        conversation_id: 'conversation:gemini-audio-barrier-cancel',
                        request_id: 'request:gemini-audio-barrier-cancel',
                        attempt_id: 'attempt:gemini-audio-barrier-cancel',
                        input_operation_id: 'input:gemini-audio-barrier-cancel',
                        response_operation_id: 'response:gemini-audio-barrier-cancel',
                        recorded_at: '2026-09-29T02:02:00.000Z',
                    },
                    on_canonical_request_prepared: publish,
                },
                controller.signal,
            ),
        ).rejects.toThrow('cancel before Gemini dispatch');
        expect(publish).toHaveBeenCalledOnce();
        expect(generate).not.toHaveBeenCalled();
        expect(store).not.toHaveBeenCalled();
    });

    it('passes Vertex GCS transcription as fileData and returns transcript metadata', async () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'global' });
        const client = new GoogleGenAI({ apiKey: 'test' });
        const response = new GenerateContentResponse();
        response.candidates = [
            {
                content: {
                    parts: [
                        {
                            text: 'Hello',
                            audioTranscription: {
                                text: 'Hello',
                                speakerLabel: 'A',
                                words: [{ word: 'Hello', startOffset: '0s', endOffset: '1s' }],
                            },
                        },
                    ],
                },
            },
        ];
        const generate = vi.spyOn(client.models, 'generateContent').mockResolvedValue(response);
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
        const audio = file();
        const result = await driver.execute(
            [
                { role: PromptRole.system, content: 'Preserve punctuation.' },
                { role: PromptRole.system, content: 'Keep speaker labels.' },
                { role: PromptRole.user, content: '', files: [audio] },
            ],
            {
                model: 'gemini-3.5-transcribe-preview',
                model_options: { _option_id: 'vertexai-gemini', transcription_diarization: true },
            },
        );
        expect(generate).toHaveBeenCalledWith(
            expect.objectContaining({
                contents: [
                    {
                        role: 'user',
                        parts: [{ fileData: { fileUri: 'gs://bucket/recording.wav', mimeType: 'audio/wav' } }],
                    },
                ],
                config: expect.objectContaining({
                    systemInstruction: {
                        role: 'user',
                        parts: [{ text: 'Preserve punctuation.' }, { text: 'Keep speaker labels.' }],
                    },
                    audioTranscriptionConfig: expect.objectContaining({ diarization: true }),
                }),
            }),
        );
        expect(audio.getStream).not.toHaveBeenCalled();
        expect(audio.getURL).not.toHaveBeenCalled();
        expect(result.result[0]).toEqual({ type: 'text', value: 'Hello' });
        expect(result.result).toHaveLength(2);
    });

    it('decodes Vertex transcription text and metadata as canonical replay-safe blocks', async () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'global' });
        const client = new GoogleGenAI({ apiKey: 'test' });
        const response = new GenerateContentResponse();
        response.responseId = 'gemini-transcription-response';
        response.modelVersion = 'gemini-3.5-transcribe-preview';
        response.candidates = [
            {
                finishReason: FinishReason.STOP,
                content: {
                    parts: [
                        {
                            audioTranscription: {
                                text: 'Hello',
                                speakerLabel: 'A',
                                words: [{ word: 'Hello', startOffset: '0s', endOffset: '1s' }],
                            },
                        },
                    ],
                },
            },
        ];
        vi.spyOn(client.models, 'generateContent').mockResolvedValue(response);
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: '', files: [file()] }], {
            model: 'gemini-3.5-transcribe-preview',
            conversation_runtime: {
                conversation_id: 'conversation:gemini-transcription',
                request_id: 'request:gemini-transcription',
                attempt_id: 'attempt:gemini-transcription',
                input_operation_id: 'input:gemini-transcription',
                response_operation_id: 'response:gemini-transcription',
                recorded_at: '2026-09-29T02:00:00.000Z',
            },
        });
        expect(result.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'text', text: 'Hello' }),
                expect.objectContaining({
                    type: 'json',
                    value: {
                        text: 'Hello',
                        language_code: null,
                        speaker_label: 'A',
                        words: [{ word: 'Hello', start_offset: '0s', end_offset: '1s' }],
                    },
                }),
            ]),
        );
        expect(JSON.stringify(result.conversation)).toContain('audioTranscription');
    });

    it('preserves S3 audio references in Converse without reading the file', async () => {
        const audio = file('s3://bucket/recording.wav');
        const result = await formatConversePrompt([{ role: PromptRole.user, content: 'Describe', files: [audio] }], {
            model: 'mistral.voxtral-small-24b-2507',
        });
        expect(JSON.stringify(result)).toContain('s3Location');
        expect(JSON.stringify(result)).toContain('s3://bucket/recording.wav');
        expect(audio.getStream).not.toHaveBeenCalled();
        expect(audio.getURL).not.toHaveBeenCalled();
    });

    it('sends Mantle Voxtral input_audio through Chat and persists an exact canonical audio asset', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const create = vi
            .spyOn(driver.service.chat.completions, 'create')
            .mockResolvedValue(chatResponse('mistral.voxtral-small-24b-2507'));
        const result = await driver.execute([{ role: PromptRole.user, content: 'Describe', files: [file()] }], {
            model: 'mistral.voxtral-small-24b-2507',
            include_original_response: true,
        });
        expect(JSON.stringify(create.mock.calls[0][0])).toContain(base64);
        expect(JSON.stringify(result.conversation)).toContain(base64);
        const document = parseConversationDocument(result.conversation);
        expect(Object.values(document.assets)).toContainEqual(
            expect.objectContaining({
                kind: 'audio',
                mime_type: 'audio/wav',
                storage: { type: 'inline_base64', data: base64 },
            }),
        );
        expect(result.result[0]).toMatchObject({ type: 'text', value: 'A greeting.' });
    });

    it('retries an accepted audio input once without duplicating its prompt', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const create = vi
            .spyOn(driver.service.chat.completions, 'create')
            .mockResolvedValue(chatResponse('mistral.voxtral-small-24b-2507'));
        const segments = [{ role: PromptRole.user, content: 'Describe', files: [file()] }];
        const runtime = {
            conversation_id: 'conversation:audio-retry',
            request_id: 'request:audio-retry',
            attempt_id: 'attempt:audio-first',
            input_operation_id: 'input:audio-retry',
            response_operation_id: 'response:audio-retry',
            recorded_at: '2026-09-29T00:00:00.000Z',
        };
        const options: ExecutionOptions = {
            model: 'mistral.voxtral-small-24b-2507',
            conversation_runtime: runtime,
        };
        const prompt = await driver.createPrompt(segments, options);
        if (!isOpenAIChatCompletionsHistory(prompt, OPENAI_CHAT_COMPLETIONS_PROTOCOL) || Array.isArray(prompt)) {
            throw new Error('Expected a Chat Completions prompt');
        }
        const acceptedInput = await prepareOpenAIChatCanonicalState({
            conversation: undefined,
            prompt,
            options,
            provider: driver.provider,
        });
        expect(JSON.stringify(acceptedInput.document)).toContain(base64);

        const result = await driver.execute(segments, {
            ...options,
            conversation: acceptedInput.document,
            conversation_runtime: {
                ...runtime,
                attempt_id: 'attempt:audio-retry',
                recorded_at: '2026-09-29T00:01:00.000Z',
            },
        });

        const request = create.mock.calls[0][0];
        expect(request.messages).toHaveLength(1);
        expect(JSON.stringify(request.messages[0])).toContain(base64);
        const document = parseConversationDocument(result.conversation);
        expect(document.turns.filter((turn) => turn.kind === 'user')).toHaveLength(1);
        expect(JSON.stringify(document)).toContain(base64);
        expect(create).toHaveBeenCalledOnce();
    });

    it('rejects native Claude audio explicitly before inference', async () => {
        const driver = new AnthropicDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.client.messages, 'create');
        await expect(
            driver.execute([{ role: PromptRole.user, content: 'Describe', files: [file()] }], {
                model: 'claude-sonnet-4-6',
            }),
        ).rejects.toThrow('does not support audio input');
        expect(create).not.toHaveBeenCalled();
    });
});
