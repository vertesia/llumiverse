import {
    type ConversationPreparedRequest,
    createConversationDocument,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    hashContentBytes,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    type DataSource,
    type ExecutionOptions,
    type ExecutionResponse,
    getModelCapabilities,
    PromptRole,
    Providers,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { boundedAudioStream } from './audio.js';
import { OpenAIDriver } from './openai.js';
import type { ChatCompletionsUsage } from './usage.js';

const bytes = new Uint8Array([82, 73, 70, 70, 1, 2, 3]);
function source(): DataSource {
    return {
        name: 'recording.wav',
        mime_type: 'audio/wav',
        getURI: vi.fn(async () => 'gs://bucket/recording.wav'),
        getURL: vi.fn(async () => 'https://example.com/recording.wav'),
        getStream: vi.fn(async () => new Blob([bytes]).stream()),
    };
}
function speechOptions(store_audio?: ExecutionOptions['store_audio']) {
    return { model: 'gpt-4o-mini-tts', store_audio };
}
function canonicalRuntime(flow: string) {
    return {
        conversation_id: `conversation:openai-audio:${flow}`,
        request_id: `request:openai-audio:${flow}`,
        attempt_id: `attempt:openai-audio:${flow}`,
        input_operation_id: `input:openai-audio:${flow}`,
        response_operation_id: `response:openai-audio:${flow}`,
        recorded_at: '2026-09-30T00:00:00.000Z',
    };
}
const prompt = [{ role: PromptRole.user, content: 'Hello from a file.' }];

async function retainedAudioDocument(
    flow: string,
    storage:
        | { type: 'inline_base64'; data: string }
        | { type: 'external'; resolver: string; locator: { uri: string } } = {
        type: 'inline_base64',
        data: Buffer.from(bytes).toString('base64'),
    },
) {
    const runtime = canonicalRuntime(flow);
    const document = createConversationDocument({ id: runtime.conversation_id, created_at: runtime.recorded_at });
    const assetId = `asset:openai-audio:${flow}`;
    const blockId = `block:openai-audio:${flow}`;
    const turn = createUserTurn({
        id: `turn:openai-audio:${flow}`,
        authority: 'ordinary',
        blocks: [
            createTextBlock({ id: `block:prompt:${flow}`, text: 'Transcribe exactly.', format: 'plain' }),
            { id: blockId, type: 'audio', asset_id: assetId },
        ],
        status: 'completed',
        timestamps: { recorded_at: runtime.recorded_at },
        provenance: { type: 'received' },
        model_visibility: 'include',
    });
    const integrity = storage.type === 'inline_base64' ? await hashContentBytes(bytes) : undefined;
    document.turns.push(turn);
    document.assets[assetId] = {
        id: assetId,
        kind: 'audio',
        mime_type: 'audio/wav',
        storage,
        provenance: { type: 'received', source_turn_id: turn.id },
        created_at: runtime.recorded_at,
        ...(integrity === undefined ? {} : integrity),
    };
    document.context.entries.push({ id: `context:openai-audio:${flow}`, type: 'source_turn', turn_id: turn.id });
    return { document, runtime };
}

describe('OpenAI file audio', () => {
    it('uses the installed SDK multipart encoder and returns a transcript without audio history', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        let multipart = '';
        driver.service = driver.service.withOptions({
            fetch: async (url, init) => {
                expect(String(url)).toContain('/audio/transcriptions');
                multipart = await new Response(init?.body).text();
                return Response.json({
                    text: 'Hello.',
                    usage: { type: 'tokens', input_tokens: 8, output_tokens: 2, total_tokens: 10 },
                });
            },
        });
        const file = source();
        const result = await driver.execute([{ role: PromptRole.user, content: '', files: [file] }], {
            model: 'gpt-transcribe',
            model_options: { _option_id: 'openai-transcription', language: 'en' },
        });
        expect(multipart).toContain('filename="recording.wav"');
        expect(multipart).toContain('audio/wav');
        expect(multipart).toContain('gpt-transcribe');
        expect(result.result).toEqual([{ type: 'text', value: 'Hello.' }]);
        expect(result.token_usage).toEqual({ prompt: 8, result: 2, total: 10 });
        expect(result.prompt).toEqual([]);
        expect(result.conversation).toBeUndefined();
        expect(file.getURL).not.toHaveBeenCalled();
    });

    it('records installed SDK transcription output and usage as canonical evidence', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        let multipart = '';
        driver.service = driver.service.withOptions({
            fetch: async (url, init) => {
                expect(String(url)).toContain('/audio/transcriptions');
                multipart = await new Response(init?.body).text();
                return Response.json({
                    text: 'Canonical transcript.',
                    usage: {
                        type: 'tokens',
                        input_tokens: 8,
                        output_tokens: 2,
                        total_tokens: 10,
                        input_token_details: { audio_tokens: 7, text_tokens: 1 },
                    },
                });
            },
        });

        const result = await driver.executeCanonical([{ role: PromptRole.user, content: '', files: [source()] }], {
            model: 'gpt-transcribe',
            model_options: { _option_id: 'openai-transcription', language: 'en' },
            conversation_runtime: canonicalRuntime('transcription-evidence'),
        });

        expect(multipart).toContain('filename="recording.wav"');
        expect(result).not.toHaveProperty('result');
        expect(result.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Canonical transcript.', format: 'plain' }),
        ]);
        expect(result.accepted_output.generation.usage).toEqual({
            input_tokens: 8,
            output_tokens: 2,
            total_tokens: 10,
            accounting_provenance: {
                input_tokens: { method: 'reported', accounting_basis: 'openai_transcription_tokens' },
                output_tokens: { method: 'reported', accounting_basis: 'openai_transcription_tokens' },
                total_tokens: { method: 'reported', accounting_basis: 'openai_transcription_tokens' },
            },
        });
        const retainedGeneration = parseConversationDocument(result.conversation).generations[
            result.accepted_output.generation.id
        ];
        expect(retainedGeneration?.usage?.reported_usage).toEqual([
            {
                source: 'provider',
                protocol: 'openai.audio.transcription',
                accounting_basis: 'openai_transcription_tokens',
                payload: {
                    type: 'tokens',
                    input_tokens: 8,
                    output_tokens: 2,
                    total_tokens: 10,
                    input_token_details: { audio_tokens: 7, text_tokens: 1 },
                },
            },
        ]);
    });

    it('executes retained inline audio context and exact-recovers without another provider request', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        let requests = 0;
        let multipart = '';
        driver.service = driver.service.withOptions({
            fetch: async (url, init) => {
                requests += 1;
                expect(String(url)).toContain('/audio/transcriptions');
                multipart = await new Response(init?.body).text();
                return Response.json({
                    text: 'Retained transcript.',
                    usage: { type: 'tokens', input_tokens: 5, output_tokens: 2, total_tokens: 7 },
                });
            },
        });
        const { document, runtime } = await retainedAudioDocument('retained-transcription');
        const program = createProgramTurn({
            id: 'turn:openai-audio:retained-transcription:program',
            authority: 'ordinary',
            blocks: [
                createTextBlock({
                    id: 'block:openai-audio:retained-transcription:program',
                    text: 'Use the retained recording.',
                    format: 'plain',
                }),
            ],
            status: 'completed',
            timestamps: { recorded_at: runtime.recorded_at },
            provenance: { type: 'received' },
            model_visibility: 'include',
        });
        document.turns.unshift(program);
        document.context.entries.unshift({
            id: 'context:openai-audio:retained-transcription:program',
            type: 'source_turn',
            turn_id: program.id,
        });
        const prepared = vi.fn(async (_request: ConversationPreparedRequest) => undefined);
        const options = {
            model: 'gpt-transcribe',
            conversation: document,
            conversation_runtime: runtime,
            on_canonical_request_prepared: prepared,
        };

        expect(await driver.supportsCanonicalContextExecution(options)).toBe(true);
        const first = await driver.executeCanonicalContext(options);
        expect(multipart).toContain('audio/wav');
        expect(multipart).toContain('Use the retained recording.\nTranscribe exactly.');
        expect(multipart).toContain('Transcribe exactly.');
        expect(first.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Retained transcript.' }),
        ]);
        expect(prepared).toHaveBeenCalledOnce();
        expect(requests).toBe(1);

        const recovered = await driver.executeCanonicalContext({ ...options, conversation: first.conversation });
        expect(recovered.accepted_output.receipt).toEqual(first.accepted_output.receipt);
        expect(recovered.accepted_output.generation).toEqual(first.accepted_output.generation);
        expect(requests).toBe(1);
    });

    it('rejects unresolved retained audio before publication or provider transport', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.audio.transcriptions, 'create');
        const prepared = vi.fn(async (_request: ConversationPreparedRequest) => undefined);
        const { document, runtime } = await retainedAudioDocument('external-transcription', {
            type: 'external',
            resolver: 'url',
            locator: { uri: 'gs://bucket/input.wav' },
        });

        await expect(
            driver.executeCanonicalContext({
                model: 'gpt-transcribe',
                conversation: document,
                conversation_runtime: runtime,
                on_canonical_request_prepared: prepared,
            }),
        ).rejects.toThrow('must use inline_base64 storage');
        expect(prepared).not.toHaveBeenCalled();
        expect(create).not.toHaveBeenCalled();
    });

    it.each([
        {
            name: 'caption',
            mutate(document: Awaited<ReturnType<typeof retainedAudioDocument>>['document']) {
                const block = document.turns[0]?.blocks.find((candidate) => candidate.type === 'audio');
                if (block?.type !== 'audio') throw new Error('Missing retained audio block');
                block.caption = 'Do not discard this caption';
            },
            expected: /cannot preserve audio block .* caption/,
        },
        {
            name: 'selection',
            mutate(document: Awaited<ReturnType<typeof retainedAudioDocument>>['document']) {
                const block = document.turns[0]?.blocks.find((candidate) => candidate.type === 'audio');
                if (block?.type !== 'audio') throw new Error('Missing retained audio block');
                block.selection = { type: 'time_range', start_seconds: 1, end_seconds: 2 };
            },
            expected: /cannot preserve audio block .* selection/,
        },
        {
            name: 'elevated user authority',
            mutate(document: Awaited<ReturnType<typeof retainedAudioDocument>>['document']) {
                const turn = document.turns[0];
                if (turn?.kind !== 'user') throw new Error('Missing retained audio user turn');
                turn.authority = 'system';
            },
            expected: /cannot preserve user turn .* authority system/,
        },
        {
            name: 'system program authority',
            mutate(document: Awaited<ReturnType<typeof retainedAudioDocument>>['document']) {
                const program = createProgramTurn({
                    id: 'turn:openai-audio:system-program',
                    authority: 'system',
                    blocks: [
                        createTextBlock({
                            id: 'block:openai-audio:system-program',
                            text: 'System instruction',
                            format: 'plain',
                        }),
                    ],
                    status: 'completed',
                    timestamps: { recorded_at: '2026-09-30T00:00:00.000Z' },
                    provenance: { type: 'received' },
                    model_visibility: 'include',
                });
                document.turns.unshift(program);
                document.context.entries.unshift({
                    id: 'context:openai-audio:system-program',
                    type: 'source_turn',
                    turn_id: program.id,
                });
            },
            expected: /cannot preserve program turn .* authority system/,
        },
    ])('rejects retained $name before publication or provider transport', async ({ name, mutate, expected }) => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.audio.transcriptions, 'create');
        const prepared = vi.fn(async (_request: ConversationPreparedRequest) => undefined);
        const { document, runtime } = await retainedAudioDocument(`projection-${name.replaceAll(' ', '-')}`);
        mutate(document);
        const parsed = parseConversationDocument(document);

        await expect(
            driver.executeCanonicalContext({
                model: 'gpt-transcribe',
                conversation: parsed,
                conversation_runtime: runtime,
                on_canonical_request_prepared: prepared,
            }),
        ).rejects.toThrow(expected);
        expect(prepared).not.toHaveBeenCalled();
        expect(create).not.toHaveBeenCalled();
    });

    it('retains duration-billed provider usage without inventing token accounting', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = driver.service.withOptions({
            fetch: async () =>
                Response.json({
                    text: 'Duration transcript.',
                    usage: { type: 'duration', seconds: 1.25 },
                }),
        });
        const legacy = await driver.execute([{ role: PromptRole.user, content: '', files: [source()] }], {
            model: 'gpt-transcribe',
        });
        expect(legacy.result).toEqual([{ type: 'text', value: 'Duration transcript.' }]);
        expect(legacy.token_usage).toBeUndefined();

        const canonical = await driver.executeCanonical([{ role: PromptRole.user, content: '', files: [source()] }], {
            model: 'gpt-transcribe',
            conversation_runtime: canonicalRuntime('transcription-duration-evidence'),
        });
        expect(canonical.accepted_output.generation.usage).toBeUndefined();
        const retainedGeneration = parseConversationDocument(canonical.conversation).generations[
            canonical.accepted_output.generation.id
        ];
        expect(retainedGeneration?.usage).toEqual({
            reported_usage: [
                {
                    source: 'provider',
                    protocol: 'openai.audio.transcription',
                    accounting_basis: 'openai_transcription_duration',
                    payload: { type: 'duration', seconds: 1.25 },
                },
            ],
        });
    });

    it.each(['mp3', 'wav'] as const)(
        'stores complete %s before returning a reference, including the SSE fallback',
        async (format) => {
            const driver = new OpenAIDriver({ apiKey: 'test' });
            const create = vi.spyOn(driver.service.audio.speech, 'create').mockResolvedValue(new Response(bytes));
            const store = vi.fn<NonNullable<ExecutionOptions['store_audio']>>(async (stream) => {
                expect(new Uint8Array(await new Response(stream).arrayBuffer())).toEqual(bytes);
                return `gs://bucket/speech.${format}`;
            });
            const stream = await driver.stream(prompt, {
                ...speechOptions(store),
                model_options: {
                    _option_id: 'openai-speech',
                    voice: 'coral',
                    response_format: format,
                },
            });
            let preview = '';
            for await (const chunk of stream) preview += chunk;
            expect(create).toHaveBeenCalledWith(
                expect.objectContaining({
                    input: 'Hello from a file.',
                    voice: 'coral',
                    response_format: format,
                }),
                expect.anything(),
            );
            expect(store).toHaveBeenCalledOnce();
            expect(stream.completion?.result).toEqual([
                expect.objectContaining({
                    type: 'audio',
                    value: `gs://bucket/speech.${format}`,
                    mime_type: format === 'mp3' ? 'audio/mpeg' : 'audio/wav',
                    container: format,
                }),
            ]);
            expect(preview).toContain('[Audio: gs://');
            expect(JSON.stringify(stream.completion)).not.toContain('base64');
            expect(stream.completion?.conversation).toBeUndefined();
        },
    );

    it('stores gpt-audio output as WAV without retaining provider base64 data', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.chat.completions, 'create').mockResolvedValue({
            choices: [
                {
                    finish_reason: 'stop',
                    message: {
                        content: null,
                        audio: { data: Buffer.from(bytes).toString('base64'), transcript: 'Spoken response.' },
                    },
                },
            ],
        } as never);
        const store = vi.fn<NonNullable<ExecutionOptions['store_audio']>>(async (stream) => {
            expect(new Uint8Array(await new Response(stream).arrayBuffer())).toEqual(bytes);
            return 'gs://bucket/response.wav';
        });
        const result = await driver.execute([{ ...prompt[0], files: [source()] }], {
            model: 'gpt-audio-1.5',
            store_audio: store,
            model_options: { _option_id: 'openai-audio', voice: 'marin', response_format: 'wav' },
        });
        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                modalities: ['text', 'audio'],
                audio: { voice: 'marin', format: 'wav' },
            }),
            expect.anything(),
        );
        expect(store).toHaveBeenCalledOnce();
        expect(result.result).toEqual([
            { type: 'text', value: 'Spoken response.' },
            expect.objectContaining({ type: 'audio', value: 'gs://bucket/response.wav', mime_type: 'audio/wav' }),
        ]);
        expect(JSON.stringify(result)).not.toContain('base64');
        expect(result.conversation).toBeUndefined();
    });

    it.each(['blocking', 'fallback'] as const)('preserves PCM format and usage in %s audio results', async (mode) => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const usage = {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            prompt_tokens_details: { cached_tokens: 4, cache_write_tokens: 1 },
            cost: 0.125,
            is_byok: false,
        } satisfies ChatCompletionsUsage;
        vi.spyOn(driver.service.chat.completions, 'create').mockResolvedValue({
            id: 'completion',
            object: 'chat.completion',
            created: 1,
            model: 'gpt-audio',
            choices: [
                {
                    index: 0,
                    finish_reason: 'stop',
                    logprobs: null,
                    message: {
                        role: 'assistant',
                        content: null,
                        refusal: null,
                        audio: {
                            id: 'audio',
                            expires_at: 1,
                            data: Buffer.from(bytes).toString('base64'),
                            transcript: 'Hello',
                        },
                    },
                },
            ],
            usage,
        });
        const options: ExecutionOptions = {
            model: 'gpt-audio',
            model_options: { _option_id: 'openai-audio', response_format: 'pcm16' },
            include_original_response: true,
            store_audio: async (stream) => {
                await new Response(stream).arrayBuffer();
                return 'gs://bucket/output.pcm';
            },
        };
        const segments = [{ ...prompt[0], files: [source()] }];
        let result: ExecutionResponse | undefined;
        if (mode === 'blocking') result = await driver.execute(segments, options);
        else {
            const stream = await driver.stream(segments, options);
            for await (const _chunk of stream) {
                /* consume */
            }
            result = stream.completion;
        }
        expect(result?.token_usage).toEqual({
            prompt: 10,
            result: 20,
            total: 30,
            prompt_cached: 4,
            prompt_cache_write: 1,
            prompt_new: 5,
            provider_cost_usd: 0.125,
        });
        expect(result?.result).toContainEqual({
            type: 'audio',
            value: 'gs://bucket/output.pcm',
            mime_type: 'audio/pcm',
            container: 'raw',
            codec: 'pcm',
            sample_rate: 24000,
            channels: 1,
            sample_encoding: 'int16',
            byte_order: 'little',
        });
        expect(JSON.stringify(result)).not.toContain(Buffer.from(bytes).toString('base64'));
        driver.destroy();
    });

    it('fails before provider work for missing storage, oversized text, and audio-to-audio input', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.audio.speech, 'create');
        await expect(driver.execute(prompt, speechOptions())).rejects.toThrow('storage sink');
        await expect(
            driver.execute([{ role: PromptRole.user, content: 'a'.repeat(4097) }], speechOptions()),
        ).rejects.toThrow('4096');
        await expect(driver.execute([{ ...prompt[0], files: [source()] }], speechOptions())).rejects.toThrow(
            'text only',
        );
        await expect(
            driver.execute(prompt, {
                ...speechOptions(),
                model_options: {
                    _option_id: 'openai-speech',
                    response_format: 'pcm',
                },
            } as unknown as ExecutionOptions),
        ).rejects.toThrow();
        expect(create).not.toHaveBeenCalled();
    });

    it('fails closed on upload errors or transient URLs', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        vi.spyOn(driver.service.audio.speech, 'create')
            .mockResolvedValueOnce(new Response(bytes))
            .mockResolvedValueOnce(new Response(bytes));
        await expect(
            driver.execute(
                prompt,
                speechOptions(async () => {
                    throw new Error('storage unavailable');
                }),
            ),
        ).rejects.toThrow('storage unavailable');
        await expect(
            driver.execute(
                prompt,
                speechOptions(async (stream) => {
                    await new Response(stream).arrayBuffer();
                    return 'https://example.com/signed.mp3';
                }),
            ),
        ).rejects.toThrow('durable object URI');
    });

    it('enforces a streaming byte bound and cancels the source', async () => {
        const cancel = vi.fn();
        const input = new ReadableStream<Uint8Array>({
            start(controller) {
                controller.enqueue(new Uint8Array(6));
            },
            cancel,
        });
        await expect(new Response(boundedAudioStream(input, 5)).arrayBuffer()).rejects.toThrow('byte limit');
        expect(cancel).toHaveBeenCalled();
        await expect(new Response(boundedAudioStream(new Blob([]).stream(), 5)).arrayBuffer()).rejects.toThrow('empty');
    });

    it('cancels pending SDK work and does no work when already aborted', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const controller = new AbortController();
        controller.abort();
        const create = vi.spyOn(driver.service.audio.speech, 'create');
        await expect(driver.execute(prompt, speechOptions(), controller.signal)).rejects.toThrow();
        expect(create).not.toHaveBeenCalled();
        let receivedSignal: AbortSignal | null | undefined;
        driver.service = driver.service.withOptions({
            fetch: (_url, options) =>
                new Promise((_resolve, reject) => {
                    receivedSignal = options?.signal;
                    receivedSignal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
                }),
        });
        const stream = await driver.stream(
            prompt,
            speechOptions(async () => 'gs://bucket/file.mp3'),
        );
        const pending = stream[Symbol.asyncIterator]().next();
        await vi.waitFor(() => expect(receivedSignal).toBeDefined());
        await stream.cancel();
        await pending;
        expect(receivedSignal?.aborted).toBe(true);
        expect(stream.completion).toBeUndefined();
    });

    it('rejects unsupported operation combinations before provider work', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.audio.transcriptions, 'create');
        await expect(driver.execute(prompt, { model: 'whisper-1' })).rejects.toThrow('exactly one');
        await expect(
            driver.execute([{ ...prompt[0], files: [source(), source()] }], { model: 'whisper-1' }),
        ).rejects.toThrow('exactly one');
        await expect(driver.execute(prompt, { model: 'whisper-1', result_schema: { type: 'object' } })).rejects.toThrow(
            'result schemas',
        );
        await expect(
            driver.execute(prompt, {
                model: 'tts-1',
                store_audio: async () => 'gs://bucket/speech.mp3',
                model_options: { _option_id: 'openai-speech', instructions: 'Whisper' },
            }),
        ).rejects.toThrow('do not support speech instructions');
        expect(create).not.toHaveBeenCalled();
    });

    it('validates file count, role, and MIME before reading canonical audio sources', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const chatCreate = vi.spyOn(driver.service.chat.completions, 'create');
        const transcriptionCreate = vi.spyOn(driver.service.audio.transcriptions, 'create');
        const speechCreate = vi.spyOn(driver.service.audio.speech, 'create');
        const first = source();
        const second = source();
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Describe', files: [first, second] }], {
                model: 'gpt-audio',
                store_audio: async () => 'gs://bucket/output.wav',
                conversation_runtime: canonicalRuntime('file-count'),
            }),
        ).rejects.toThrow('at most one');
        expect(first.getStream).not.toHaveBeenCalled();
        expect(second.getStream).not.toHaveBeenCalled();

        const assistantFile = source();
        await expect(
            driver.executeCanonical([{ role: PromptRole.assistant, content: '', files: [assistantFile] }], {
                model: 'gpt-transcribe',
                conversation_runtime: canonicalRuntime('file-role'),
            }),
        ).rejects.toThrow('only user and system');
        expect(assistantFile.getStream).not.toHaveBeenCalled();

        const unsupported = source();
        unsupported.mime_type = 'application/pdf';
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: '', files: [unsupported] }], {
                model: 'gpt-transcribe',
                conversation_runtime: canonicalRuntime('file-mime'),
            }),
        ).rejects.toThrow('does not support application/pdf');
        expect(unsupported.getStream).not.toHaveBeenCalled();

        const speechFile = source();
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Speak', files: [speechFile] }], {
                model: 'gpt-4o-mini-tts',
                store_audio: async () => 'gs://bucket/output.mp3',
                conversation_runtime: canonicalRuntime('speech-file'),
            }),
        ).rejects.toThrow('text only');
        expect(speechFile.getStream).not.toHaveBeenCalled();
        expect(chatCreate).not.toHaveBeenCalled();
        expect(transcriptionCreate).not.toHaveBeenCalled();
        expect(speechCreate).not.toHaveBeenCalled();
    });

    it('awaits canonical request publication and does not call the provider when publication fails', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.service.audio.speech, 'create');
        const publish = vi.fn(async (prepared: ConversationPreparedRequest) => {
            expect(prepared.record.request_receipt.target).toMatchObject({
                provider: Providers.openai,
                protocol: 'openai.audio.speech',
                model: 'gpt-4o-mini-tts',
            });
            throw new Error('durable publication failed');
        });

        await expect(
            driver.executeCanonical(prompt, {
                ...speechOptions(async () => 'gs://bucket/output.mp3'),
                conversation_runtime: canonicalRuntime('publication'),
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow('durable publication failed');
        expect(publish).toHaveBeenCalledOnce();
        expect(create).not.toHaveBeenCalled();
    });

    it('preserves provider status and retryability from the installed SDK', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = driver.service.withOptions({
            fetch: async () =>
                Response.json(
                    {
                        error: { message: 'Rate limited', type: 'rate_limit_error' },
                    },
                    { status: 429 },
                ),
        });
        await expect(
            driver.execute(
                prompt,
                speechOptions(async () => 'gs://bucket/speech.mp3'),
            ),
        ).rejects.toMatchObject({ code: 429, retryable: true, context: { provider: 'openai' } });
    });

    it('keeps a leased audio stream callable when the driver is evicted before consumption', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        vi.spyOn(driver.service.audio.speech, 'create').mockResolvedValue(new Response(bytes));
        const stream = await driver.stream(
            prompt,
            speechOptions(async (input) => {
                await new Response(input).arrayBuffer();
                return 'gs://bucket/speech.mp3';
            }),
        );
        driver.destroy();
        for await (const _chunk of stream) {
            /* consume the bounded result */
        }
        expect(stream.completion?.result[0].type).toBe('audio');
    });

    it('reports audio output only for the implemented provider operation', () => {
        expect(getModelCapabilities('gpt-4o-mini-tts', Providers.openai).output.audio).toBe(true);
        expect(getModelCapabilities('gpt-audio', Providers.openai).output.audio).toBe(true);
        expect(getModelCapabilities('gpt-4o-mini-tts', Providers.azure_openai).output.audio).toBe(false);
        expect(getModelCapabilities('gpt-transcribe', Providers.openai).input.audio).toBe(true);
    });
});
