import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import {
    buildPagedRecordIndex,
    type PagedRecordIndexStore,
    type PagedRecordRef,
    putPagedRecord,
    readPagedRecordRange,
    removePagedRecord,
} from '../src/paged-record-index.js';

function memoryStore() {
    const pages = new Map<string, Uint8Array>();
    const reads: PagedRecordRef[] = [];
    const store: PagedRecordIndexStore = {
        async read(ref) {
            reads.push(ref);
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Missing index page');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
    };
    return { store, pages, reads };
}
const key = (index: number) => String(index).padStart(6, '0');
const marker = (index: number) => ({ storage: 'marker' as const, kind: 'pending', id: key(index) });

describe('bounded immutable paged range and removal', () => {
    it('seeks a late pending page at 10k/100k without reading the cold prefix', async () => {
        const profiles: number[] = [];
        for (const count of [10_000, 100_000]) {
            const memory = memoryStore();
            const root = await buildPagedRecordIndex(
                memory.store,
                Array.from({ length: count }, (_, index) => ({ key: key(index), value: marker(index) })),
            );
            memory.reads.length = 0;
            const page = await readPagedRecordRange(memory.store, root, { after: key(count - 18), limit: 16 });
            expect(page.entries.map((entry) => entry.key)).toEqual(
                Array.from({ length: 16 }, (_, offset) => key(count - 17 + offset)),
            );
            expect(page.has_more).toBe(true);
            expect(page.next_cursor).toBe(key(count - 2));
            expect(page.page_reads).toBeLessThanOrEqual(5);
            expect(page.page_bytes).toBeLessThan(32 * 1024);
            profiles.push(page.page_reads);
            const last = await readPagedRecordRange(memory.store, root, { after: page.next_cursor, limit: 16 });
            expect(last.entries.map((entry) => entry.key)).toEqual([key(count - 1)]);
            expect(last.has_more).toBe(false);
            expect(last.next_cursor).toBeUndefined();
        }
        expect(Math.abs((profiles[1] ?? 0) - (profiles[0] ?? 0))).toBeLessThanOrEqual(1);
    });

    it('removes closed jobs physically, keeps prior roots immutable and permits later insertion', async () => {
        const memory = memoryStore();
        const initial = await buildPagedRecordIndex(
            memory.store,
            Array.from({ length: 130 }, (_, index) => ({ key: key(index), value: marker(index) })),
        );
        let root = initial;
        for (let index = 0; index < 129; index++) {
            const removed = await removePagedRecord(memory.store, root, key(index));
            expect(removed.applied).toBe(true);
            expect(removed.removed).toEqual(marker(index));
            root = removed.root;
        }
        const current = await readPagedRecordRange(memory.store, root);
        expect(current.entries).toEqual([{ key: key(129), value: marker(129) }]);
        expect(current.page_reads).toBe(1);
        const prior = await readPagedRecordRange(memory.store, initial, { limit: 16 });
        expect(prior.entries[0]?.key).toBe(key(0));
        expect(prior.has_more).toBe(true);
        const absent = await removePagedRecord(memory.store, root, 'missing');
        expect(absent).toEqual({ root, applied: false });
        const emptied = await removePagedRecord(memory.store, root, key(129));
        expect(emptied.root).toBeUndefined();
        expect((await readPagedRecordRange(memory.store, emptied.root)).entries).toEqual([]);
        const added = await putPagedRecord(memory.store, emptied.root, key(200), marker(200));
        expect((await readPagedRecordRange(memory.store, added)).entries).toEqual([
            { key: key(200), value: marker(200) },
        ]);
    });

    it('fails explicit byte/page overflow and rejects a hash-valid foreign child partition', async () => {
        const memory = memoryStore();
        const root = await buildPagedRecordIndex(
            memory.store,
            Array.from({ length: 65 }, (_, index) => ({ key: key(index), value: marker(index) })),
        );
        await expect(readPagedRecordRange(memory.store, root, { max_page_reads: 1 })).rejects.toThrow('bound');
        await expect(readPagedRecordRange(memory.store, root, { max_bytes: 1 })).rejects.toThrow('bound');
        const leaf = canonicalJsonContentBytes({ kind: 'leaf', entries: [{ key: 'a', value: marker(0) }] });
        const hash = await hashContentBytes(leaf);
        const child = { content_hash: hash.content_hash, size_bytes: hash.byte_length };
        memory.pages.set(child.content_hash, leaf);
        const branch = canonicalJsonContentBytes({
            kind: 'branch',
            level: 1,
            children: [
                { max_key: 'a', page: child },
                { max_key: 'b', page: child },
            ],
        });
        const parentHash = await hashContentBytes(branch);
        const parent = { content_hash: parentHash.content_hash, size_bytes: parentHash.byte_length };
        memory.pages.set(parent.content_hash, branch);
        await expect(readPagedRecordRange(memory.store, parent, { after: 'a' })).rejects.toThrow('partition');
        await expect(removePagedRecord(memory.store, parent, 'b')).rejects.toThrow('partition');
    });

    it('owns root/cursor before await and rejects accessors rather than coercing input', async () => {
        const memory = memoryStore();
        const root = await buildPagedRecordIndex(memory.store, [{ key: 'a', value: marker(0) }]);
        if (!root) throw new Error('Missing test root');
        const options = { limit: 1 };
        const mutableRoot = { ...root };
        const pending = readPagedRecordRange(memory.store, mutableRoot, options);
        options.limit = 2;
        mutableRoot.content_hash = `sha256:${'f'.repeat(64)}`;
        expect((await pending).entries).toHaveLength(1);
        const getter = () => {
            throw new Error('Getter invoked');
        };
        const accessor = {};
        Object.defineProperty(accessor, 'limit', { get: getter, enumerable: true });
        await expect(readPagedRecordRange(memory.store, root, accessor)).rejects.toThrow('bounded JSON');
    });
});
