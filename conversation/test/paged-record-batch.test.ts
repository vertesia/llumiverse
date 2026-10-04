import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes } from '../src/content-integrity.js';
import {
    buildPagedRecordIndex,
    getPagedRecord,
    insertPagedRecords,
    PAGED_RECORD_INDEX_MAX_BATCH_BYTES,
    PAGED_RECORD_INDEX_MAX_BATCH_KEYS,
    type PagedRecordIndexStore,
    type PagedRecordRef,
    putPagedRecord,
    readPagedRecordRange,
    scanPagedRecords,
} from '../src/paged-record-index.js';

function memory() {
    const pages = new Map<string, Uint8Array>();
    const reads: PagedRecordRef[] = [];
    const writes: PagedRecordRef[] = [];
    const store: PagedRecordIndexStore = {
        async read(ref) {
            reads.push(ref);
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Missing durable page');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            writes.push(ref);
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
    };
    return { store, pages, reads, writes };
}
const marker = (id: string) => ({ storage: 'marker' as const, kind: 'record', id });
const key = (index: number) => String(index).padStart(6, '0');

async function entries(store: PagedRecordIndexStore, root: PagedRecordRef | undefined) {
    const result = [];
    for await (const entry of scanPagedRecords(store, root)) result.push(entry);
    return result;
}

describe('bounded grouped immutable index insertion', () => {
    it('matches sequential inserts across splits and keeps the original root unchanged', async () => {
        const current = memory();
        const initial = await buildPagedRecordIndex(
            current.store,
            Array.from({ length: 130 }, (_, index) => ({ key: key(index * 2), value: marker(key(index * 2)) })),
        );
        const commands = Array.from({ length: 130 }, (_, index) => ({
            key: key(index * 2 + 1),
            value: marker(key(index * 2 + 1)),
        }));
        const grouped = await insertPagedRecords(current.store, initial, [...commands].reverse());
        let sequential = initial;
        for (const command of commands) {
            sequential = await putPagedRecord(current.store, sequential, command.key, command.value);
        }
        expect(await entries(current.store, grouped)).toEqual(await entries(current.store, sequential));
        expect((await entries(current.store, initial)).map((entry) => entry.key)).toEqual(
            Array.from({ length: 130 }, (_, index) => key(index * 2)),
        );
    });

    it('rejects duplicate commands and retained key collisions without publishing a root', async () => {
        const current = memory();
        const root = await buildPagedRecordIndex(current.store, [{ key: 'retained', value: marker('retained') }]);
        const original = await entries(current.store, root);
        await expect(
            insertPagedRecords(current.store, root, [
                { key: 'new', value: marker('first') },
                { key: 'new', value: marker('second') },
            ]),
        ).rejects.toThrow('strictly ordered');
        await expect(
            insertPagedRecords(current.store, root, [
                { key: 'new', value: marker('new') },
                { key: 'retained', value: marker('different') },
            ]),
        ).rejects.toThrow('identity already exists');
        expect(await entries(current.store, root)).toEqual(original);
    });

    it('bounds admitted command count and rejects over-limit input before writes', async () => {
        const current = memory();
        const commands = Array.from({ length: PAGED_RECORD_INDEX_MAX_BATCH_KEYS }, (_, index) => ({
            key: key(index),
            value: marker(key(index)),
        }));
        const root = await insertPagedRecords(current.store, undefined, commands);
        expect(await getPagedRecord(current.store, root, key(commands.length - 1))).toEqual(
            marker(key(commands.length - 1)),
        );
        const writes = current.writes.length;
        await expect(
            insertPagedRecords(current.store, undefined, [
                ...commands,
                { key: 'over-limit', value: marker('over-limit') },
            ]),
        ).rejects.toThrow();
        expect(current.writes).toHaveLength(writes);
    });

    it('accepts the exact command byte ceiling and rejects one additional byte before writes', async () => {
        const current = memory();
        const commands = Array.from({ length: 4000 }, (_, index) => ({
            key: `${'x'.repeat(900)}:${key(index)}`,
            value: marker('v'.repeat(1000)),
        }));
        let remaining =
            PAGED_RECORD_INDEX_MAX_BATCH_BYTES - canonicalJsonContentBytes({ records: commands }).byteLength;
        expect(remaining).toBeGreaterThan(0);
        let lastPadded = 0;
        for (let index = 0; index < commands.length && remaining > 0; index++) {
            // Leave one valid key character available for the exact byte-overflow negative.
            const added = Math.min(remaining, 2047 - commands[index].key.length);
            commands[index].key = `${'z'.repeat(added)}${commands[index].key}`;
            remaining -= added;
            lastPadded = index;
        }
        expect(remaining).toBe(0);
        expect(canonicalJsonContentBytes({ records: commands })).toHaveLength(PAGED_RECORD_INDEX_MAX_BATCH_BYTES);
        const root = await insertPagedRecords(current.store, undefined, commands);
        expect(await getPagedRecord(current.store, root, commands[0].key)).toEqual(commands[0].value);
        const writes = current.writes.length;
        commands[lastPadded].key = `z${commands[lastPadded].key}`;
        expect(canonicalJsonContentBytes({ records: commands })).toHaveLength(PAGED_RECORD_INDEX_MAX_BATCH_BYTES + 1);
        await expect(insertPagedRecords(current.store, undefined, commands)).rejects.toThrow('bounded owned JSON');
        expect(current.writes).toHaveLength(writes);
    });

    it('keeps cold reads bounded at 10k/100k and reads back every newly staged page', async () => {
        const profiles: number[] = [];
        for (const count of [10_000, 100_000]) {
            const current = memory();
            const root = await buildPagedRecordIndex(
                current.store,
                Array.from({ length: count }, (_, index) => ({ key: key(index), value: marker(key(index)) })),
            );
            current.reads.length = 0;
            current.writes.length = 0;
            const commands = Array.from({ length: 1024 }, (_, index) => ({
                key: `new:${key(index)}`,
                value: marker(`new:${key(index)}`),
            }));
            const updated = await insertPagedRecords(current.store, root, commands);
            const externalReads = new Set(current.reads.map((ref) => ref.content_hash));
            profiles.push(externalReads.size);
            expect(externalReads.size).toBeLessThan(32);
            expect(current.writes.every((ref) => externalReads.has(ref.content_hash))).toBe(true);
            expect((await readPagedRecordRange(current.store, updated, { after: 'new:000999' })).entries).toHaveLength(
                16,
            );
            expect(await getPagedRecord(current.store, root, commands[0].key)).toBeUndefined();
        }
        expect(Math.abs(profiles[1] - profiles[0])).toBeLessThanOrEqual(2);
    });

    it('owns commands/root before await, rejects getters, and refuses failed durable readback', async () => {
        const current = memory();
        const root = await buildPagedRecordIndex(current.store, [{ key: 'retained', value: marker('retained') }]);
        if (!root) throw new Error('Missing real root');
        const owned = { ...root };
        const commands = [{ key: 'new', value: marker('original') }];
        const pending = insertPagedRecords(current.store, owned, commands);
        owned.content_hash = `sha256:${'f'.repeat(64)}`;
        commands[0].value.id = 'late';
        const updated = await pending;
        expect(await getPagedRecord(current.store, updated, 'new')).toEqual(marker('original'));
        let reads = 0;
        const accessor = { key: 'nomination', value: marker('nomination') };
        Object.defineProperty(accessor, 'key', {
            enumerable: true,
            get() {
                reads++;
                return 'nomination';
            },
        });
        await expect(insertPagedRecords(current.store, root, [accessor])).rejects.toThrow('bounded owned JSON');
        expect(reads).toBe(0);
        const dropWrites: PagedRecordIndexStore = { read: current.store.read, async write() {} };
        await expect(
            insertPagedRecords(dropWrites, root, [{ key: 'uncommitted', value: marker('uncommitted') }]),
        ).rejects.toThrow('Missing durable page');
        expect(await getPagedRecord(current.store, root, 'uncommitted')).toBeUndefined();
    });
});
