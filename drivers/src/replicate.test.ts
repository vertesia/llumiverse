import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
    predictions: {
        create: vi.fn(),
        get: vi.fn(),
        cancel: vi.fn(),
    },
    sources: [] as Array<{
        close: ReturnType<typeof vi.fn>;
        emit(type: string, event?: { data: string }): void;
        options: { fetch?: typeof fetch };
    }>,
}));

vi.mock('replicate', () => ({
    default: class {
        predictions = mocks.predictions;
    },
}));

vi.mock('eventsource', () => ({
    EventSource: class {
        close = vi.fn();
        private readonly listeners = new Map<string, (event: { data: string }) => void>();

        constructor(
            _url: string,
            readonly options: { fetch?: typeof fetch },
        ) {
            mocks.sources.push(this);
        }

        addEventListener(type: string, listener: (event: { data: string }) => void) {
            this.listeners.set(type, listener);
        }

        emit(type: string, event = { data: '' }) {
            this.listeners.get(type)?.(event);
        }
    },
}));

import { ReplicateDriver } from './replicate.js';

const options = { model: 'owner/model:version' };

describe('Replicate cancellation', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.sources.length = 0;
        mocks.predictions.cancel.mockResolvedValue({ status: 'canceled' });
    });

    it('aborts fallback polling and cancels the remote prediction', async () => {
        mocks.predictions.create.mockResolvedValue({ id: 'prediction', status: 'starting' });
        mocks.predictions.get.mockImplementation(
            (_id: string, request?: { signal?: AbortSignal }) =>
                new Promise((_resolve, reject) => {
                    request?.signal?.addEventListener('abort', () => reject(request.signal?.reason), { once: true });
                }),
        );
        const driver = new ReplicateDriver({ apiKey: 'key' });
        const controller = new AbortController();

        const completion = driver.requestTextCompletion('prompt', options, controller.signal);
        await vi.waitFor(() => expect(mocks.predictions.get).toHaveBeenCalled());
        controller.abort();

        await expect(completion).rejects.toBe(controller.signal.reason);
        expect(mocks.predictions.cancel).toHaveBeenCalledWith('prediction');
    });

    it('uses the managed fetch and closes a failed SSE connection', async () => {
        mocks.predictions.create.mockResolvedValue({
            id: 'prediction',
            status: 'processing',
            urls: { stream: 'https://replicate.test/stream' },
        });
        const driver = new ReplicateDriver({ apiKey: 'key' });
        const stream = await driver.requestTextCompletionStream('prompt', options);
        const read = stream[Symbol.asyncIterator]().next();
        const source = mocks.sources[0];

        expect(source.options.fetch).toBeTypeOf('function');
        source.emit('error', { data: 'connection failed' });

        await expect(read).rejects.toThrow('connection failed');
        expect(source.close).toHaveBeenCalledOnce();
        expect(mocks.predictions.cancel).toHaveBeenCalledWith('prediction');
    });

    it('waits for remote prediction cancellation before the stream read settles', async () => {
        let finishCancellation!: () => void;
        mocks.predictions.create.mockResolvedValue({
            id: 'prediction',
            status: 'processing',
            urls: { stream: 'https://replicate.test/stream' },
        });
        mocks.predictions.cancel.mockReturnValue(
            new Promise((resolve) => {
                finishCancellation = () => resolve({ status: 'canceled' });
            }),
        );
        const driver = new ReplicateDriver({ apiKey: 'key' });
        const controller = new AbortController();
        const stream = await driver.requestTextCompletionStream('prompt', options, controller.signal);
        let readSettled = false;
        const read = stream[Symbol.asyncIterator]()
            .next()
            .then((result) => {
                readSettled = true;
                return result;
            });

        controller.abort();
        await vi.waitFor(() => expect(mocks.predictions.cancel).toHaveBeenCalledWith('prediction'));
        expect(readSettled).toBe(false);

        finishCancellation();
        await expect(read).resolves.toEqual({ done: true, value: '' });
    });

    it('waits for cancellation when prediction creation races with abort', async () => {
        let finishCreation!: () => void;
        let finishCancellation!: () => void;
        mocks.predictions.create.mockReturnValue(
            new Promise((resolve) => {
                finishCreation = () =>
                    resolve({
                        id: 'prediction',
                        status: 'processing',
                        urls: { stream: 'https://replicate.test/stream' },
                    });
            }),
        );
        mocks.predictions.cancel.mockReturnValue(
            new Promise((resolve) => {
                finishCancellation = () => resolve({ status: 'canceled' });
            }),
        );
        const driver = new ReplicateDriver({ apiKey: 'key' });
        const controller = new AbortController();
        let creationSettled = false;
        const stream = driver.requestTextCompletionStream('prompt', options, controller.signal).then((result) => {
            creationSettled = true;
            return result;
        });

        controller.abort();
        finishCreation();
        await vi.waitFor(() => expect(mocks.predictions.cancel).toHaveBeenCalledWith('prediction'));
        expect(creationSettled).toBe(false);

        finishCancellation();
        await expect(stream).resolves.toBeDefined();
    });
});
