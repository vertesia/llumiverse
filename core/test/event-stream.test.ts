import { describe, expect, it } from 'vitest';
import { EventStream } from '../src/async.js';

async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
    const items: T[] = [];
    for await (const item of iter) {
        items.push(item);
    }
    return items;
}

describe('EventStream', () => {
    it('yields events pushed before iteration starts', async () => {
        const stream = new EventStream<string>();
        stream.push('a');
        stream.push('b');
        stream.close();

        expect(await collect(stream)).toEqual(['a', 'b']);
    });

    it('yields events pushed during iteration', async () => {
        const stream = new EventStream<string>();

        const promise = collect(stream);

        stream.push('x');
        stream.push('y');
        stream.close();

        expect(await promise).toEqual(['x', 'y']);
    });

    it('returns empty for immediately closed stream', async () => {
        const stream = new EventStream<number>();
        stream.close();

        expect(await collect(stream)).toEqual([]);
    });

    it('throws when pushing to a closed stream', () => {
        const stream = new EventStream<string>();
        stream.close();

        expect(() => stream.push('late')).toThrow('Cannot push to a closed stream');
    });

    it('handles interleaved push and consume', async () => {
        const stream = new EventStream<number>();
        const results: number[] = [];

        const done = (async () => {
            for await (const n of stream) {
                results.push(n);
            }
        })();

        stream.push(1);
        // Give the consumer a tick to process
        await new Promise((r) => setTimeout(r, 0));
        stream.push(2);
        await new Promise((r) => setTimeout(r, 0));
        stream.push(3);
        stream.close();

        await done;
        expect(results).toEqual([1, 2, 3]);
    });

    it('supports early return from consumer', async () => {
        const stream = new EventStream<string>();
        stream.push('first');
        stream.push('second');
        stream.push('third');

        const results: string[] = [];
        for await (const item of stream) {
            results.push(item);
            if (item === 'second') break;
        }

        expect(results).toEqual(['first', 'second']);
    });

    it('works with typed payloads', async () => {
        interface Chunk {
            id: number;
            text: string;
        }

        const stream = new EventStream<Chunk>();
        stream.push({ id: 1, text: 'hello' });
        stream.push({ id: 2, text: 'world' });
        stream.close();

        const items = await collect(stream);
        expect(items).toEqual([
            { id: 1, text: 'hello' },
            { id: 2, text: 'world' },
        ]);
    });
});
