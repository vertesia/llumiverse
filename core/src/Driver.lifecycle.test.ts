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
    imageModel = false;
    completionStream: DriverCompletionStream = (async function* () {
        yield { result: [{ type: 'text', value: 'first' }] } satisfies CompletionChunkObject;
    })();

    constructor(private readonly cleanup: () => void) {
        super({});
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
        return [];
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Not implemented');
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

    it('destroys immediately when idle and only once', () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);

        driver.destroy();
        driver.destroy();

        expect(cleanup).toHaveBeenCalledOnce();
        expect(() => driver.acquireOperation()).toThrow('Cannot use destroyed lifecycle-test driver');
    });
});
