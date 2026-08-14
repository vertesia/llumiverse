import { describe, expect, it } from 'vitest';
import { EventStream } from './async.js';

describe('EventStream', () => {
    it('delivers failures that happen before iteration begins', async () => {
        const stream = new EventStream<string>();
        const failure = new Error('stream failed');

        stream.fail(failure);

        await expect(stream[Symbol.asyncIterator]().next()).rejects.toBe(failure);
    });

    it('rejects a pending read when the stream fails', async () => {
        const stream = new EventStream<string>();
        const read = stream[Symbol.asyncIterator]().next();
        const failure = new Error('stream failed');

        stream.fail(failure);

        await expect(read).rejects.toBe(failure);
    });
});
