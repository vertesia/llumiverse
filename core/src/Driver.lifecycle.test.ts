import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type DriverCompletionStream,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ModelSearchPayload,
    PromptRole,
} from '@llumiverse/common';
import { describe, expect, it, vi } from 'vitest';
import { AbstractDriver } from './Driver.js';

class LifecycleTestDriver extends AbstractDriver<DriverOptions, string> {
    provider = 'lifecycle-test';
    completion = Promise.resolve<Completion>({ result: [{ type: 'text', value: 'done' }] });
    imageCompletion = Promise.resolve<Completion>({ result: [{ type: 'image', value: 'image-data' }] });
    models = Promise.resolve<AIModel[]>([]);
    embeddings = Promise.resolve<EmbeddingsResult>({ results: [], model: 'test-model' });
    imageModel = false;
    completionStream: DriverCompletionStream = (async function* () {
        yield { result: [{ type: 'text', value: 'first' }] } satisfies CompletionChunkObject;
    })();

    constructor(
        private readonly cleanup: () => void,
        options: DriverOptions = {},
    ) {
        super(options);
    }

    async requestTextCompletion(_prompt: string, _options: ExecutionOptions): Promise<Completion> {
        return this.completion;
    }

    async requestTextCompletionStream(_prompt: string, _options: ExecutionOptions): Promise<DriverCompletionStream> {
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
}

const segments = [{ role: PromptRole.user, content: 'hello' }];
const options = { model: 'test-model' };

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
        driver.completionStream = (async function* () {
            yield { result: [{ type: 'text', value: 'first' }] } satisfies CompletionChunkObject;
            yield { result: [{ type: 'text', value: 'second' }] } satisfies CompletionChunkObject;
        })();
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

    it('rejects invalid stream start timeouts', () => {
        expect(() => new LifecycleTestDriver(vi.fn(), { streamStartTimeoutMs: 0 })).toThrow(
            'streamStartTimeoutMs must be a positive finite number',
        );
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
        expect(() => driver.acquireOperation()).toThrow('Cannot use destroyed lifecycle-test driver');
        await expect(driver.listModels()).rejects.toThrow('Cannot use destroyed lifecycle-test driver');
        await expect(driver.generateEmbeddings({ inputs: [{ type: 'text', text: 'hello' }] })).rejects.toThrow(
            'Cannot use destroyed lifecycle-test driver',
        );
    });
});
