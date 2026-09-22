import { boundedAudioStream, storeAudioResult } from '../shared/audio.js';

export { boundedAudioStream } from '../shared/audio.js';

import {
    OpenAiAudioOptionsSchema,
    OpenAiSpeechOptionsSchema,
    OpenAiTranscriptionOptionsSchema,
} from '@llumiverse/common/schemas';
import {
    type AudioResult,
    type Completion,
    type ExecutionOptions,
    type ExecutionResponse,
    type PromptSegment,
    Providers,
    resolveModelProfile,
} from '@llumiverse/core';
import type { AbstractDriver } from '@llumiverse/core/driver';
import type OpenAI from 'openai';
import { toStreamingFile } from 'openai';

export function openAIAudioTask(model: string): 'transcription' | 'speech' | 'understanding' | undefined {
    const { family } = resolveModelProfile(model.split('::').pop() ?? model, Providers.openai);
    if (/(?:realtime|live)/i.test(model)) return undefined;
    return family === 'audio'
        ? 'understanding'
        : family === 'transcription' || family === 'speech'
          ? family
          : undefined;
}

export async function executeOpenAIAudio(
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    requestOptions: { signal?: AbortSignal; timeout?: number } | undefined,
    requestModel = options.model,
): Promise<Completion> {
    const signal = requestOptions?.signal;
    signal?.throwIfAborted();
    if (options.conversation || options.tools?.length || options.result_schema || options.format) {
        throw new Error(
            'File audio operations do not accept conversation, tools, result schemas, or custom formatting',
        );
    }
    const files = segments.flatMap((segment) => segment.files ?? []);
    const text = segments
        .map((segment) => segment.content ?? '')
        .join('\n')
        .trim();
    if (segments.some((segment) => segment.role === 'tool' || segment.role === 'assistant')) {
        throw new Error('File audio operations accept only user and system input');
    }
    if (openAIAudioTask(options.model) === 'understanding') {
        if (files.length !== 1) throw new Error('Audio understanding requires exactly one audio file');
        const content: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [{ type: 'text', text }];
        for (const file of files) {
            const format =
                file.mime_type === 'audio/wav' || file.mime_type === 'audio/x-wav'
                    ? 'wav'
                    : file.mime_type === 'audio/mpeg' || file.mime_type === 'audio/mp3'
                      ? 'mp3'
                      : undefined;
            if (!format) throw new Error('OpenAI audio chat requires MP3 or WAV input');
            const stream = boundedAudioStream(await file.getStream(), 25_000_000, signal);
            const data = Buffer.from(await new Response(stream).arrayBuffer()).toString('base64');
            content.push({ type: 'input_audio', input_audio: { data, format } });
        }
        const params = OpenAiAudioOptionsSchema.parse(options.model_options ?? { _option_id: 'openai-audio' });
        if (!options.store_audio) throw new Error('Audio generation requires a durable audio storage sink');
        const format = params.response_format ?? 'wav';
        const result = await service.chat.completions.create(
            {
                model: requestModel,
                messages: [{ role: 'user', content }],
                modalities: ['text', 'audio'],
                audio: { voice: params.voice ?? 'alloy', format },
            },
            requestOptions,
        );
        signal?.throwIfAborted();
        const message = result.choices[0]?.message;
        if (!message?.audio?.data) throw new Error('OpenAI audio chat returned no audio data');
        const audio = await storeAudioResult(
            new Blob([Buffer.from(message.audio.data, 'base64')]).stream(),
            {
                mime_type: {
                    wav: 'audio/wav',
                    mp3: 'audio/mpeg',
                    flac: 'audio/flac',
                    opus: 'audio/ogg',
                    pcm16: 'audio/pcm',
                }[format],
                container: format === 'opus' ? 'ogg' : format === 'pcm16' ? 'raw' : format,
                ...(format === 'mp3' ? { codec: 'mp3' } : {}),
                ...(format === 'pcm16' ? { codec: 'pcm', sample_encoding: 'int16', byte_order: 'little' } : {}),
            },
            options,
            signal,
        );
        return {
            result: [
                ...(message.content || message.audio.transcript
                    ? [{ type: 'text' as const, value: message.content ?? message.audio.transcript }]
                    : []),
                audio,
            ],
            finish_reason: result.choices[0]?.finish_reason,
        };
    }
    if (openAIAudioTask(options.model) === 'transcription') {
        const params = OpenAiTranscriptionOptionsSchema.parse(
            options.model_options ?? {
                _option_id: 'openai-transcription',
            },
        );
        if (files.length !== 1) throw new Error('Transcription requires exactly one audio file');
        // The endpoint owns codec validation; MP4 and WebM recordings may carry a video MIME type.
        const file = files[0];
        const stream = boundedAudioStream(await file.getStream(), 25_000_000, signal);
        try {
            if (options.model.includes('diarize')) {
                if (text) throw new Error('Diarized transcription does not accept a prompt');
                const result = (await service.audio.transcriptions.create(
                    {
                        model: requestModel,
                        file: toStreamingFile(stream, file.name, { type: file.mime_type }),
                        response_format: 'diarized_json',
                        chunking_strategy: 'auto',
                        language: params.language,
                    },
                    requestOptions,
                )) as OpenAI.Audio.TranscriptionDiarized; // SDK overload omits its exported diarized response type.
                signal?.throwIfAborted();
                return {
                    result: [
                        { type: 'text', value: result.text },
                        {
                            type: 'json',
                            value: {
                                segments: result.segments.map((segment) => ({
                                    id: segment.id,
                                    speaker: segment.speaker,
                                    start: segment.start,
                                    end: segment.end,
                                    text: segment.text,
                                })),
                            },
                        },
                    ],
                    finish_reason: 'stop',
                };
            }
            const result = await service.audio.transcriptions.create(
                {
                    model: requestModel,
                    file: toStreamingFile(stream, file.name, { type: file.mime_type }),
                    response_format: 'json',
                    language: params.language,
                    ...(text && { prompt: text }),
                },
                requestOptions,
            );
            signal?.throwIfAborted();
            return { result: [{ type: 'text', value: result.text }], finish_reason: 'stop' };
        } finally {
            if (!stream.locked) await stream.cancel().catch(() => undefined);
        }
    }
    const params = OpenAiSpeechOptionsSchema.parse(options.model_options ?? { _option_id: 'openai-speech' });
    if (files.length) throw new Error('Speech synthesis accepts text only');
    if (!text || text.length > 4096) throw new Error('Speech synthesis requires 1–4096 characters');
    if (!options.store_audio) throw new Error('Speech synthesis requires a durable audio storage sink');
    if (params.instructions && /^tts-1(?:-|$)/.test(options.model)) {
        throw new Error('tts-1 models do not support speech instructions');
    }
    const format = params.response_format ?? 'mp3';
    const response = await service.audio.speech.create(
        {
            model: requestModel,
            input: text,
            voice: params.voice ?? 'alloy',
            response_format: format,
            speed: params.speed,
            instructions: params.instructions,
        } satisfies OpenAI.Audio.SpeechCreateParams,
        requestOptions,
    );
    if (!response.body) throw new Error('Speech endpoint returned no audio body');
    const metadata: Omit<AudioResult, 'type' | 'value'> = {
        mime_type: {
            mp3: 'audio/mpeg',
            wav: 'audio/wav',
            opus: 'audio/ogg',
            aac: 'audio/aac',
            flac: 'audio/flac',
            pcm: 'audio/pcm',
        }[format],
        container: format === 'opus' ? 'ogg' : format === 'pcm' ? 'raw' : format,
        ...(format === 'mp3' ? { codec: 'mp3' } : {}),
        ...(format === 'pcm'
            ? { codec: 'pcm', sample_rate: 24000, channels: 1, sample_encoding: 'int16', byte_order: 'little' }
            : {}),
    };
    return { result: [await storeAudioResult(response.body, metadata, options, signal)], finish_reason: 'stop' };
}

/** Execute a finite SDK audio request without retaining multipart input in a prompt or conversation. */
export async function executeOpenAIAudioRequest<PromptT>(
    driver: Pick<AbstractDriver, 'createExecutionHttpAgentScope' | 'formatLlumiverseError' | 'provider'>,
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    emptyPrompt: PromptT,
    signal?: AbortSignal,
    requestModel = options.model,
    requestOptions?: { signal?: AbortSignal; timeout?: number },
): Promise<ExecutionResponse<PromptT>> {
    const start = Date.now();
    const scope = driver.createExecutionHttpAgentScope(options, signal !== undefined);
    const abort = () => void scope.abort();
    if (signal?.aborted) abort();
    else signal?.addEventListener('abort', abort, { once: true });
    try {
        const completion = await scope.run(() =>
            executeOpenAIAudio(service, segments, options, requestOptions ?? { signal }, requestModel),
        );
        return { ...completion, prompt: emptyPrompt, execution_time: Date.now() - start };
    } catch (error) {
        throw driver.formatLlumiverseError(error, {
            provider: driver.provider,
            model: options.model,
            operation: 'execute',
        });
    } finally {
        signal?.removeEventListener('abort', abort);
        await scope.close();
    }
}
