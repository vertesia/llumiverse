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
function speechOptions(store_audio?: ExecutionOptions['store_audio']): ExecutionOptions {
    return { model: 'gpt-4o-mini-tts', store_audio };
}
const prompt = [{ role: PromptRole.user, content: 'Hello from a file.' }];

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
            usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
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
        expect(result?.token_usage).toEqual({ prompt: 10, result: 20, total: 30 });
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
