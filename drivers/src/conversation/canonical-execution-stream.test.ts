import type { CompletionChunkObject } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import {
    type CanonicalFinalizingDriverStream,
    canonicalExecutionStreamFromDriver,
} from './canonical-execution-stream.js';

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((accept) => {
        resolve = accept;
    });
    return { promise, resolve };
}

function sourceFrom(iterator: AsyncIterator<CompletionChunkObject>) {
    const finalize = vi.fn(async () => {
        throw new Error('Response must not be accepted');
    });
    return {
        source: {
            [Symbol.asyncIterator]: () => iterator,
            finalizeCanonicalExecution: finalize,
        } satisfies CanonicalFinalizingDriverStream,
        finalize,
    };
}

describe('canonical stream cancellation', () => {
    it('does not accept a response when abort makes a pending read end cleanly', async () => {
        const pending = deferred<IteratorResult<CompletionChunkObject>>();
        const started = deferred<void>();
        const { source, finalize } = sourceFrom({
            next: () => {
                started.resolve();
                return pending.promise;
            },
            return: vi.fn(async () => ({ done: true as const, value: undefined })),
        });
        const close = vi.fn();
        const stream = canonicalExecutionStreamFromDriver(source, {
            abort: () => pending.resolve({ done: true, value: undefined }),
            close,
        });
        const read = stream[Symbol.asyncIterator]().next();
        await started.promise;
        await stream.cancel();

        await expect(read).resolves.toEqual({ done: true, value: undefined });
        expect(finalize).not.toHaveBeenCalled();
        expect(stream.completion).toBeUndefined();
        expect(close).toHaveBeenCalledTimes(1);
    });

    it('does not start the provider iterator after cancellation before consumption', async () => {
        const next = vi.fn(async () => ({ done: true as const, value: undefined }));
        const { source, finalize } = sourceFrom({ next });
        const stream = canonicalExecutionStreamFromDriver(source, { abort: vi.fn(), close: vi.fn() });
        await stream.cancel();

        await expect(stream[Symbol.asyncIterator]().next()).resolves.toEqual({ done: true, value: undefined });
        expect(next).not.toHaveBeenCalled();
        expect(finalize).not.toHaveBeenCalled();
    });

    it('makes concurrent cancellation callers await the same resource cleanup', async () => {
        const released = deferred<void>();
        const returned = vi.fn(async () => {
            await released.promise;
            return { done: true as const, value: undefined };
        });
        const { source } = sourceFrom({
            next: async () => ({ done: true, value: undefined }),
            return: returned,
        });
        const close = vi.fn();
        const abort = vi.fn();
        const stream = canonicalExecutionStreamFromDriver(source, { abort, close });
        let secondFinished = false;
        const first = stream.cancel();
        const second = stream.cancel().then(() => {
            secondFinished = true;
        });
        await Promise.resolve();
        expect(secondFinished).toBe(false);
        released.resolve();
        await Promise.all([first, second]);
        expect(abort).toHaveBeenCalledTimes(1);
        expect(returned).toHaveBeenCalledTimes(1);
        expect(close).toHaveBeenCalledTimes(1);
    });

    it('closes provider resources without accepting output when the consumer stops early', async () => {
        const returned = vi.fn(async () => ({ done: true as const, value: undefined }));
        const { source, finalize } = sourceFrom({
            next: async () => ({ done: false, value: { result: [{ type: 'text', value: 'preview' }] } }),
            return: returned,
        });
        const abort = vi.fn();
        const close = vi.fn();
        const stream = canonicalExecutionStreamFromDriver(source, { abort, close });
        for await (const text of stream) {
            expect(text).toBe('preview');
            break;
        }
        expect(finalize).not.toHaveBeenCalled();
        expect(stream.completion).toBeUndefined();
        expect(abort).toHaveBeenCalledTimes(1);
        expect(returned).toHaveBeenCalledTimes(1);
        expect(close).toHaveBeenCalledTimes(1);
    });
});
