import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes, hashContentBytes } from './content-integrity.js';
import type { PagedRecordIndexStore, PagedRecordRef, PagedRecordValue } from './paged-record-index.js';
import {
    buildPagedRecordIndex,
    getPagedRecord,
    PagedRecordIndexPageSchema,
    putPagedRecord,
    scanPagedRecords,
} from './paged-record-index.js';

const hash = `sha256:${'0'.repeat(64)}`;

function memoryStore() {
    const bytes = new Map<string, Uint8Array>();
    let reads = 0;
    let writes = 0;
    let readBytes = 0;
    let writtenBytes = 0;
    const store: PagedRecordIndexStore = {
        async read(ref) {
            reads += 1;
            const retained = bytes.get(ref.content_hash);
            if (!retained) throw new Error('index page unavailable');
            readBytes += retained.byteLength;
            return Uint8Array.from(retained);
        },
        async write(value, ref) {
            writes += 1;
            writtenBytes += value.byteLength;
            const prior = bytes.get(ref.content_hash);
            if (
                prior &&
                (prior.byteLength !== value.byteLength || prior.some((byte, index) => byte !== value[index]))
            ) {
                throw new Error('immutable index page conflict');
            }
            bytes.set(ref.content_hash, Uint8Array.from(value));
        },
    };
    return {
        store,
        bytes,
        counters: () => ({ reads, writes, readBytes, writtenBytes }),
        reset: () => {
            reads = 0;
            writes = 0;
            readBytes = 0;
            writtenBytes = 0;
        },
    };
}

function value(id: string): PagedRecordValue {
    return { storage: 'record', kind: 'turn', id, content_hash: hash, size_bytes: 1 };
}

function records(count: number) {
    return Array.from({ length: count }, (_, index) => {
        const id = `turn:${index.toString().padStart(6, '0')}`;
        return { key: id, value: value(id) };
    });
}

describe('paged record index', () => {
    // A 100k cold import writes and verifies roughly 1,600 immutable pages before bounded-path checks.
    it.each([10_000, 100_000])(
        'reads and appends through bounded index paths with %i cold records',
        async (count) => {
            const memory = memoryStore();
            const oldRoot = await buildPagedRecordIndex(memory.store, records(count));
            expect(oldRoot).toBeDefined();
            memory.reset();
            expect(await getPagedRecord(memory.store, oldRoot, 'turn:000007')).toEqual(value('turn:000007'));
            expect(memory.counters()).toMatchObject({ writes: 0 });
            expect(memory.counters().reads).toBeLessThanOrEqual(5);

            memory.reset();
            const newId = `turn:${count.toString().padStart(6, '0')}`;
            const newRoot = await putPagedRecord(memory.store, oldRoot, newId, value(newId));
            expect(memory.counters().reads).toBeLessThanOrEqual(12);
            expect(memory.counters().writes).toBeLessThanOrEqual(6);
            expect(memory.counters().readBytes).toBeLessThan(1024 * 1024);
            expect(memory.counters().writtenBytes).toBeLessThan(1024 * 1024);
            expect(await getPagedRecord(memory.store, oldRoot, newId)).toBeUndefined();
            expect(await getPagedRecord(memory.store, newRoot, newId)).toEqual(value(newId));
        },
        15_000,
    );

    it('preserves ordinal keys, rejects duplicate identities, and retains the old root after conflict', async () => {
        const memory = memoryStore();
        const oldRoot = await buildPagedRecordIndex(memory.store, [
            { key: 'b:c', value: value('b:c') },
            { key: 'a:b:c', value: value('a:b:c') },
            { key: 'a', value: value('a') },
        ]);
        const newRoot = await putPagedRecord(memory.store, oldRoot, 'a:b', value('a:b'));
        const keys: string[] = [];
        for await (const entry of scanPagedRecords(memory.store, newRoot)) keys.push(entry.key);
        expect(keys).toEqual(['a', 'a:b', 'a:b:c', 'b:c']);
        await expect(putPagedRecord(memory.store, newRoot, 'a:b', value('other'))).rejects.toThrow(
            'identity already exists',
        );
        expect(await getPagedRecord(memory.store, oldRoot, 'a:b')).toBeUndefined();
    });

    it('rejects tampered index bytes before decoding a retained page', async () => {
        const memory = memoryStore();
        const root = (await buildPagedRecordIndex(memory.store, records(2))) as PagedRecordRef;
        const retained = memory.bytes.get(root.content_hash);
        expect(retained).toBeDefined();
        const changed = Uint8Array.from(retained as Uint8Array);
        changed[changed.length - 2] ^= 1;
        memory.bytes.set(root.content_hash, changed);
        await expect(getPagedRecord(memory.store, root, 'turn:000001')).rejects.toThrow('hash differs');
    });

    it('rejects valid-hash pages with a false child level or maximum key', async () => {
        const memory = memoryStore();
        const root = (await buildPagedRecordIndex(memory.store, records(65))) as PagedRecordRef;
        const original = PagedRecordIndexPageSchema.parse(
            JSON.parse(new TextDecoder().decode(memory.bytes.get(root.content_hash))),
        );
        if (original.kind !== 'branch') throw new Error('Expected a two-level index');
        const malformed = [
            { ...original, level: 2 },
            {
                ...original,
                children: [{ ...original.children[0], max_key: 'turn:000063x' }, original.children[1]],
            },
        ];
        for (const page of malformed) {
            const bytes = canonicalJsonContentBytes(page);
            const integrity = await hashContentBytes(bytes);
            const ref = { content_hash: integrity.content_hash, size_bytes: integrity.byte_length };
            memory.bytes.set(ref.content_hash, Uint8Array.from(bytes));
            await expect(getPagedRecord(memory.store, ref, 'turn:000007')).rejects.toThrow('child level or partition');
            await expect(putPagedRecord(memory.store, ref, 'turn:000007x', value('turn:000007x'))).rejects.toThrow(
                'child level or partition',
            );
            const iterate = async () => {
                for await (const _item of scanPagedRecords(memory.store, ref)) {
                    // Consume the authenticated traversal.
                }
            };
            await expect(iterate()).rejects.toThrow('child level or partition');
        }
    });
});
