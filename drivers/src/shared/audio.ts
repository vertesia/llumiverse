import type { AudioResult, ExecutionOptions } from '@llumiverse/core';

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

export async function storeAudioResult(
    source: ReadableStream<Uint8Array>,
    metadata: Omit<AudioResult, 'type' | 'value'>,
    options: ExecutionOptions,
    signal?: AbortSignal,
): Promise<AudioResult> {
    if (!options.store_audio) throw new Error('Speech synthesis requires a durable audio storage sink');
    const stream = boundedAudioStream(source, 50_000_000, signal);
    try {
        const value = await options.store_audio(stream, metadata, signal);
        signal?.throwIfAborted();
        if (!/^(?:gs|s3):\/\/[^/]+\/.+/.test(value)) {
            throw new Error('Audio storage must return a durable object URI');
        }
        return { type: 'audio', value, ...metadata };
    } finally {
        if (!stream.locked) await stream.cancel().catch(() => undefined);
    }
}
