import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type CompletionStream,
    type DriverCompletionStream,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ModelSearchPayload,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/common';
import { describe, expect, it, vi } from 'vitest';
import { DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS } from './CompletionStream.js';
import { AbstractDriver } from './Driver.js';

class LifecycleTestDriver extends AbstractDriver<DriverOptions, string> {
    provider = 'lifecycle-test';
    completion = Promise.resolve<Completion>({ result: [{ type: 'text', value: 'done' }] });
    imageCompletion = Promise.resolve<Completion>({ result: [{ type: 'image', value: 'image-data' }] });
    models = Promise.resolve<AIModel[]>([]);
    embeddings = Promise.resolve<EmbeddingsResult>({ results: [], model: 'test-model' });
    imageModel = false;
    streaming = true;
    completionSignal?: AbortSignal;
    waitForCompletionAbort = false;
    completionStreamSignal?: AbortSignal;
    completionStream: DriverCompletionStream = {
        async *[Symbol.asyncIterator]() {
            yield { result: [{ type: 'text', value: 'first' }] } satisfies CompletionChunkObject;
        },
    };

    constructor(
        private readonly cleanup: () => void,
        options: DriverOptions = {},
    ) {
        super(options);
    }

    async requestTextCompletion(
        _prompt: string,
        _options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        this.completionSignal = signal;
        if (this.waitForCompletionAbort) {
            return new Promise((_resolve, reject) => {
                signal?.addEventListener('abort', () => reject(signal.reason), {
                    once: true,
                });
            });
        }
        return this.completion;
    }

    async requestTextCompletionStream(
        _prompt: string,
        _options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        this.completionStreamSignal = signal;
        return this.completionStream;
    }

    async requestImageGeneration(_prompt: string, _options: ExecutionOptions): Promise<Completion> {
        return this.imageCompletion;
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return this.models;
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return this.embeddings;
    }

    protected override destroyProviderResources(): void {
        this.cleanup();
    }

    protected override isImageModel(_model: string): boolean {
        return this.imageModel;
    }

    protected override canStream(_options?: ExecutionOptions, _signal?: AbortSignal): Promise<boolean> {
        return Promise.resolve(this.streaming);
    }
}

class OverriddenStreamDriver extends LifecycleTestDriver {
    readonly cancelStream = vi.fn().mockResolvedValue(undefined);

    override async stream(
        _segments: PromptSegment[],
        _options: ExecutionOptions,
        _signal?: AbortSignal,
    ): Promise<CompletionStream<string>> {
        return {
            completion: undefined,
            cancel: this.cancelStream,
            async *[Symbol.asyncIterator]() {
                yield 'custom';
            },
        };
    }
}

class PendingStreamCreationDriver extends LifecycleTestDriver {
    protected override canStream(_options?: ExecutionOptions, signal?: AbortSignal): Promise<boolean> {
        return new Promise((_resolve, reject) => {
            signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
        });
    }
}

class ThrowingIteratorDriver extends LifecycleTestDriver {
    readonly cancelStream = vi.fn().mockResolvedValue(undefined);

    override async stream(_segments: PromptSegment[], _options: ExecutionOptions): Promise<CompletionStream<string>> {
        return {
            completion: undefined,
            cancel: this.cancelStream,
            [Symbol.asyncIterator]() {
                throw new Error('iterator creation failed');
            },
        };
    }
}

const segments = [{ role: PromptRole.user, content: 'hello' }];
const options = { model: 'test-model' };

function holdStreamCancellation(driver: OverriddenStreamDriver): () => void {
    let release!: () => void;
    driver.cancelStream.mockImplementationOnce(
        () =>
            new Promise<void>((resolve) => {
                release = resolve;
            }),
    );
    return () => release();
}

describe('AbstractDriver lifecycle', () => {
    it('defers destruction until an in-flight execution finishes', async () => {
        let resolveCompletion!: (completion: Completion) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.completion = new Promise((resolve) => {
            resolveCompletion = resolve;
        });

        const execution = driver.execute(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveCompletion({ result: [{ type: 'text', value: 'done' }] });
        await execution;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('defers destruction until an in-flight image generation finishes', async () => {
        let resolveImage!: (completion: Completion) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.imageModel = true;
        driver.imageCompletion = new Promise((resolve) => {
            resolveImage = resolve;
        });

        const execution = driver.execute(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveImage({ result: [{ type: 'image', value: 'image-data' }] });
        await execution;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('defers destruction until a stream is cancelled', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.completionStream = {
            async *[Symbol.asyncIterator]() {
                yield { result: [{ type: 'text', value: 'first' }] } satisfies CompletionChunkObject;
                yield { result: [{ type: 'text', value: 'second' }] } satisfies CompletionChunkObject;
            },
        };
        const stream = await driver.stream(segments, options);
        const iterator = stream[Symbol.asyncIterator]();

        await iterator.next();
        driver.destroy();
        expect(cleanup).not.toHaveBeenCalled();

        await iterator.return?.();
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('holds the stream lease between creation and delayed consumption', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);

        const stream = await driver.stream(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        const chunks: string[] = [];
        for await (const chunk of stream) {
            chunks.push(chunk);
        }
        expect(chunks).toEqual(['first']);
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('transfers lifecycle ownership for overridden stream implementations', async () => {
        const cleanup = vi.fn();
        const driver = new OverriddenStreamDriver(cleanup);

        const stream = await driver.stream(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        await stream.cancel();
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('coalesces concurrent cancellation of a leased stream', async () => {
        const driver = new OverriddenStreamDriver(vi.fn());
        const releaseCancellation = holdStreamCancellation(driver);
        const stream = await driver.stream(segments, options);

        const first = stream.cancel();
        const second = stream.cancel();

        expect(second).toBe(first);
        expect(driver.cancelStream).toHaveBeenCalledOnce();
        releaseCancellation();
        await Promise.all([first, second]);
    });

    it('coalesces abort-signal and explicit cancellation', async () => {
        const driver = new OverriddenStreamDriver(vi.fn());
        const releaseCancellation = holdStreamCancellation(driver);
        const controller = new AbortController();
        const stream = await driver.stream(segments, options, controller.signal);

        controller.abort();
        const explicitCancellation = stream.cancel();

        expect(driver.cancelStream).toHaveBeenCalledOnce();
        releaseCancellation();
        await explicitCancellation;
    });

    it('coalesces stream-start timeout and explicit cancellation', async () => {
        vi.useFakeTimers();
        try {
            const driver = new OverriddenStreamDriver(vi.fn(), { streamStartTimeoutMs: 100 });
            const releaseCancellation = holdStreamCancellation(driver);
            const stream = await driver.stream(segments, options);

            await vi.advanceTimersByTimeAsync(101);
            const explicitCancellation = stream.cancel();

            expect(driver.cancelStream).toHaveBeenCalledOnce();
            releaseCancellation();
            await explicitCancellation;
            const iterator = stream[Symbol.asyncIterator]();
            await expect(iterator.next()).rejects.toThrow('Completion stream was not consumed within 100ms');
        } finally {
            vi.useRealTimers();
        }
    });

    it('releases lifecycle ownership when an overridden stream iterator throws during creation', async () => {
        const cleanup = vi.fn();
        const driver = new ThrowingIteratorDriver(cleanup);
        const stream = await driver.stream(segments, options);

        expect(() => stream[Symbol.asyncIterator]()).toThrow('iterator creation failed');
        driver.destroy();

        await vi.waitFor(() => expect(driver.cancelStream).toHaveBeenCalledOnce());
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('releases an abandoned stream lease after its start timeout', async () => {
        vi.useFakeTimers();
        try {
            const cleanup = vi.fn();
            const driver = new LifecycleTestDriver(cleanup, { streamStartTimeoutMs: 100 });

            const stream = await driver.stream(segments, options);
            driver.destroy();
            expect(cleanup).not.toHaveBeenCalled();

            await vi.advanceTimersByTimeAsync(101);

            expect(cleanup).toHaveBeenCalledOnce();
            const iterator = stream[Symbol.asyncIterator]();
            await expect(iterator.next()).rejects.toThrow('Completion stream was not consumed within 100ms');
        } finally {
            vi.useRealTimers();
        }
    });

    it('cancels an overridden stream when its start lease expires', async () => {
        vi.useFakeTimers();
        try {
            const driver = new OverriddenStreamDriver(vi.fn(), { streamStartTimeoutMs: 100 });
            await driver.stream(segments, options);

            await vi.advanceTimersByTimeAsync(101);

            expect(driver.cancelStream).toHaveBeenCalledOnce();
        } finally {
            vi.useRealTimers();
        }
    });

    it('keeps the default abandoned-stream lease beyond the request boundary', async () => {
        expect(DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS).toBe(900_000);

        vi.useFakeTimers();
        try {
            const cleanup = vi.fn();
            const driver = new LifecycleTestDriver(cleanup);

            await driver.stream(segments, options);
            driver.destroy();

            await vi.advanceTimersByTimeAsync(5 * 60_000);
            expect(cleanup).not.toHaveBeenCalled();

            await vi.advanceTimersByTimeAsync(10 * 60_000);
            expect(cleanup).toHaveBeenCalledOnce();
        } finally {
            vi.useRealTimers();
        }
    });

    it('allows an unconsumed stream to be cancelled explicitly', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);

        const stream = await driver.stream(segments, options);
        await stream.cancel?.();
        driver.destroy();

        expect(cleanup).toHaveBeenCalledOnce();
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).rejects.toThrow('Completion stream was cancelled before consumption');
    });

    it('cancels the provider while a public iterator next call is pending', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.completionStream = {
            [Symbol.asyncIterator]() {
                return {
                    next: async () => {
                        await new Promise<void>((resolve) => {
                            driver.completionStreamSignal?.addEventListener('abort', () => resolve(), { once: true });
                        });
                        return { done: true, value: undefined };
                    },
                };
            },
        };

        const stream = await driver.stream(segments, options);
        const iterator = stream[Symbol.asyncIterator]();
        const read = iterator.next();
        await stream.cancel();

        expect(driver.completionStreamSignal?.aborted).toBe(true);
        await expect(read).resolves.toEqual({ done: true, value: undefined });
    });

    it('cancels a pending fallback provider request', async () => {
        const driver = new LifecycleTestDriver(vi.fn());
        driver.streaming = false;
        driver.waitForCompletionAbort = true;

        const stream = await driver.stream(segments, options);
        const read = stream[Symbol.asyncIterator]().next();
        await stream.cancel();

        expect(driver.completionSignal?.aborted).toBe(true);
        await expect(read).resolves.toEqual({ done: true, value: undefined });
    });

    it('cancels stream creation before a provider stream exists', async () => {
        const cleanup = vi.fn();
        const driver = new PendingStreamCreationDriver(cleanup);
        const controller = new AbortController();

        const stream = driver.stream(segments, options, controller.signal);
        controller.abort();

        await expect(stream).rejects.toBe(controller.signal.reason);
        driver.destroy();
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('keeps the stream signal active after creation', async () => {
        const driver = new LifecycleTestDriver(vi.fn());
        const controller = new AbortController();
        const stream = await driver.stream(segments, options, controller.signal);
        const read = stream[Symbol.asyncIterator]().next();

        controller.abort();

        await expect(read).resolves.toEqual({ done: true, value: undefined });
        expect(driver.completionStreamSignal?.aborted).toBe(true);
    });

    it('rejects invalid stream start timeouts', () => {
        expect(() => new LifecycleTestDriver(vi.fn(), { streamStartTimeoutMs: 0 })).toThrow(
            'streamStartTimeoutMs must be a positive integer no greater than 2147483647',
        );
        expect(() => new LifecycleTestDriver(vi.fn(), { streamStartTimeoutMs: 1.5 })).toThrow(RangeError);
        expect(() => new LifecycleTestDriver(vi.fn(), { streamStartTimeoutMs: 2_147_483_648 })).toThrow(RangeError);
    });

    it('defers destruction until model listing finishes', async () => {
        let resolveModels!: (models: AIModel[]) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.models = new Promise((resolve) => {
            resolveModels = resolve;
        });

        const listing = driver.listModels();
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveModels([]);
        await listing;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('defers destruction until embedding generation finishes', async () => {
        let resolveEmbeddings!: (result: EmbeddingsResult) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.embeddings = new Promise((resolve) => {
            resolveEmbeddings = resolve;
        });

        const generation = driver.generateEmbeddings({ inputs: [{ type: 'text', text: 'hello' }] });
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveEmbeddings({ results: [], model: 'test-model' });
        await generation;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('destroys immediately when idle and only once', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);

        driver.destroy();
        driver.destroy();

        expect(cleanup).toHaveBeenCalledOnce();
        await expect(driver.listModels()).rejects.toThrow('Cannot use destroyed lifecycle-test driver');
        await expect(driver.generateEmbeddings({ inputs: [{ type: 'text', text: 'hello' }] })).rejects.toThrow(
            'Cannot use destroyed lifecycle-test driver',
        );
    });
});
