import type {
    CanonicalExecutionResponse,
    CanonicalExecutionStream,
    CompletionChunkObject,
    DriverCompletionStream,
} from '@llumiverse/core';

export interface CanonicalFinalizingDriverStream extends DriverCompletionStream {
    finalizeCanonicalExecution(): Promise<CanonicalExecutionResponse>;
}

function preview(chunk: CompletionChunkObject): string {
    return chunk.result
        .map((result) => {
            switch (result.type) {
                case 'text':
                case 'thoughts':
                    return result.value;
                case 'json':
                    return JSON.stringify(result.value);
                case 'image':
                    return '[Image]';
                case 'audio':
                    return '[Audio]';
                case 'video':
                    return '[Video]';
                default: {
                    const _exhaustive: never = result;
                    return String(_exhaustive);
                }
            }
        })
        .join('');
}

export function canonicalExecutionStreamFromDriver(
    source: CanonicalFinalizingDriverStream,
    lifecycle: { abort(): void; close(): void | Promise<void> },
): CanonicalExecutionStream {
    const sourceIterator = source[Symbol.asyncIterator]();
    let completion: CanonicalExecutionResponse | undefined;
    let iteratorCreated = false;
    let cancelled = false;
    let chunks = 0;
    let closing: Promise<void> | undefined;
    let cancellation: Promise<void> | undefined;
    const startedAt = Date.now();

    function closeOnce(): Promise<void> {
        closing ??= Promise.resolve().then(() => lifecycle.close());
        return closing;
    }

    function abortAndClose(): Promise<void> {
        cancellation ??= Promise.resolve().then(async () => {
            try {
                lifecycle.abort();
                await sourceIterator.return?.();
            } finally {
                await closeOnce();
            }
        });
        return cancellation;
    }

    return {
        get completion() {
            return completion;
        },
        cancel() {
            cancelled = true;
            return abortAndClose();
        },
        [Symbol.asyncIterator]() {
            if (iteratorCreated) throw new Error('Canonical execution stream can only be consumed once');
            iteratorCreated = true;
            return (async function* () {
                try {
                    if (cancelled) return;
                    while (true) {
                        const next = await sourceIterator.next();
                        if (cancelled) return;
                        if (next.done) break;
                        const value = preview(next.value);
                        if (value.length > 0) {
                            chunks += 1;
                            yield value;
                        }
                    }
                    const accepted = await source.finalizeCanonicalExecution();
                    if (cancelled) return;
                    completion = {
                        ...accepted,
                        execution_time: accepted.execution_time ?? Date.now() - startedAt,
                        chunks,
                    };
                    await closeOnce();
                } finally {
                    if (completion === undefined) {
                        cancelled = true;
                        await abortAndClose();
                    }
                }
            })();
        },
    };
}
