import { OpenAiSpeechOptionsSchema, OpenAiTranscriptionOptionsSchema } from '@llumiverse/common/schemas';
import {
    type AudioResult,
    type Completion,
    type ExecutionOptions,
    type PromptSegment,
    Providers,
    resolveModelProfile,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { toStreamingFile } from 'openai';

export function openAIAudioTask(model: string): 'transcription' | 'speech' | undefined {
    const { family } = resolveModelProfile(model, Providers.openai);
    return family === 'transcription' || family === 'speech' ? family : undefined;
}

/** Consume lazily, enforce the byte budget even without Content-Length, and close the source on failure. */
export function boundedAudioStream(
    source: ReadableStream<Uint8Array | string>,
    maximumBytes: number,
    signal?: AbortSignal,
): ReadableStream<Uint8Array> {
    let bytes = 0;
    return source.pipeThrough(
        new TransformStream<Uint8Array | string, Uint8Array>({
            transform(chunk, controller) {
                signal?.throwIfAborted();
                if (typeof chunk === 'string') throw new Error('Audio sources must contain binary bytes');
                bytes += chunk.byteLength;
                if (bytes > maximumBytes) throw new Error(`Audio exceeds the ${maximumBytes} byte limit`);
                controller.enqueue(chunk);
            },
            flush() {
                if (bytes === 0) throw new Error('Audio file is empty');
            },
        }),
        { signal },
    );
}

export async function executeOpenAIAudio(
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    requestOptions: { signal?: AbortSignal; timeout?: number } | undefined,
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
            const result = await service.audio.transcriptions.create(
                {
                    model: options.model,
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
            model: options.model,
            input: text,
            voice: params.voice ?? 'alloy',
            response_format: format,
            speed: params.speed,
            instructions: params.instructions,
        } satisfies OpenAI.Audio.SpeechCreateParams,
        requestOptions,
    );
    if (!response.body) throw new Error('Speech endpoint returned no audio body');
    // Complete files only. Raw PCM is deliberately excluded until its metadata contract is selected.
    const metadata: Omit<AudioResult, 'type' | 'value'> = {
        mime_type: format === 'mp3' ? 'audio/mpeg' : 'audio/wav',
        container: format,
        ...(format === 'mp3' && { codec: 'mp3' }),
    };
    const stream = boundedAudioStream(response.body, 50_000_000, signal);
    try {
        const value = await options.store_audio(stream, metadata, signal);
        signal?.throwIfAborted();
        if (!/^(?:gs|s3):\/\/[^/]+\/.+/.test(value)) {
            throw new Error('Audio storage must return a durable object URI');
        }
        return { result: [{ type: 'audio', value, ...metadata }], finish_reason: 'stop' };
    } finally {
        if (!stream.locked) await stream.cancel().catch(() => undefined);
    }
}
