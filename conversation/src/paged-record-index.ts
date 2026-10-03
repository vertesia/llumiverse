import { z } from 'zod';
import { canonicalJsonContentBytes, hashContentBytes } from './content-integrity.js';

/** Fixed fanout keeps index I/O independent of the number of cold conversation records. */
export const PAGED_RECORD_INDEX_FANOUT = 64;
export const PAGED_RECORD_INDEX_PAGE_MAX_BYTES = 256 * 1024;
const HashSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);

export const PagedRecordRefSchema = z.strictObject({
    content_hash: HashSchema,
    size_bytes: z.number().int().positive().max(PAGED_RECORD_INDEX_PAGE_MAX_BYTES),
});

const valueIdentity = { kind: z.string().min(1).max(64), id: z.string().min(1).max(1024) };
export const PagedRecordValueSchema = z.discriminatedUnion('storage', [
    z.strictObject({
        storage: z.literal('record'),
        ...valueIdentity,
        content_hash: HashSchema,
        size_bytes: z
            .number()
            .int()
            .nonnegative()
            .max(32 * 1024 * 1024),
    }),
    /** An index-only identity marker; never mistaken for a retrievable body. */
    z.strictObject({ storage: z.literal('marker'), ...valueIdentity }),
]);

const LeafEntrySchema = z.strictObject({ key: z.string().min(1).max(2048), value: PagedRecordValueSchema });
const BranchEntrySchema = z.strictObject({ max_key: z.string().min(1).max(2048), page: PagedRecordRefSchema });
export const PagedRecordIndexPageSchema = z.discriminatedUnion('kind', [
    z.strictObject({
        kind: z.literal('leaf'),
        entries: z.array(LeafEntrySchema).min(1).max(PAGED_RECORD_INDEX_FANOUT),
    }),
    z.strictObject({
        kind: z.literal('branch'),
        level: z.number().int().min(1).max(16),
        children: z.array(BranchEntrySchema).min(1).max(PAGED_RECORD_INDEX_FANOUT),
    }),
]);

export type PagedRecordRef = z.infer<typeof PagedRecordRefSchema>;
export type PagedRecordValue = z.infer<typeof PagedRecordValueSchema>;
export type PagedRecordIndexPage = z.infer<typeof PagedRecordIndexPageSchema>;

/** Storage keys are derived by the authenticated host from the hash, never accepted from the index. */
export interface PagedRecordIndexStore {
    read(ref: PagedRecordRef): Promise<Uint8Array>;
    /** A write must be immutable/create-only. The index verifies it by reading the returned ref. */
    write(bytes: Uint8Array, ref: PagedRecordRef): Promise<void>;
}

function compare(first: string, second: string): number {
    return first < second ? -1 : first > second ? 1 : 0;
}

function assertOrdered(keys: readonly string[]): void {
    for (let index = 1; index < keys.length; index += 1) {
        if (compare(keys[index - 1], keys[index]) >= 0)
            throw new Error('Paged record index keys are not strictly ordered');
    }
}

async function readPage(store: PagedRecordIndexStore, ref: PagedRecordRef): Promise<PagedRecordIndexPage> {
    const locator = PagedRecordRefSchema.parse(ref);
    const bytes = Uint8Array.from(await store.read(locator));
    if (bytes.byteLength !== locator.size_bytes || bytes.byteLength > PAGED_RECORD_INDEX_PAGE_MAX_BYTES) {
        throw new Error('Paged record index page size differs from its retained locator');
    }
    if ((await hashContentBytes(bytes)).content_hash !== locator.content_hash) {
        throw new Error('Paged record index page hash differs from its retained locator');
    }
    const page = PagedRecordIndexPageSchema.parse(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)));
    assertOrdered(
        page.kind === 'leaf' ? page.entries.map((entry) => entry.key) : page.children.map((child) => child.max_key),
    );
    return page;
}

async function writePage(store: PagedRecordIndexStore, page: PagedRecordIndexPage): Promise<PagedRecordRef> {
    const owned = PagedRecordIndexPageSchema.parse(page);
    assertOrdered(
        owned.kind === 'leaf' ? owned.entries.map((entry) => entry.key) : owned.children.map((child) => child.max_key),
    );
    const bytes = canonicalJsonContentBytes(owned);
    if (bytes.byteLength > PAGED_RECORD_INDEX_PAGE_MAX_BYTES)
        throw new Error('Paged record index page exceeds its bound');
    const integrity = await hashContentBytes(bytes);
    const ref = { content_hash: integrity.content_hash, size_bytes: integrity.byte_length };
    await store.write(Uint8Array.from(bytes), ref);
    await readPage(store, ref);
    return ref;
}

function maxKey(page: PagedRecordIndexPage): string {
    return page.kind === 'leaf'
        ? page.entries[page.entries.length - 1].key
        : page.children[page.children.length - 1].max_key;
}

function assertChildPage(page: PagedRecordIndexPage, parentLevel: number, declaredMaxKey: string): void {
    if (
        (page.kind === 'leaf' ? parentLevel !== 1 : page.level !== parentLevel - 1) ||
        maxKey(page) !== declaredMaxKey
    ) {
        throw new Error('Paged record index child level or partition differs from its parent');
    }
}

function branchIndex(page: Extract<PagedRecordIndexPage, { kind: 'branch' }>, key: string): number {
    const index = page.children.findIndex((child) => compare(key, child.max_key) <= 0);
    return index === -1 ? page.children.length - 1 : index;
}

interface WrittenPage {
    page: PagedRecordRef;
    max_key: string;
}

async function replacePage(
    store: PagedRecordIndexStore,
    ref: PagedRecordRef,
    key: string,
    value: PagedRecordValue,
    mode: 'insert' | 'replace',
    parentLevel?: number,
    declaredMaxKey?: string,
): Promise<WrittenPage[]> {
    const old = await readPage(store, ref);
    if (parentLevel !== undefined && declaredMaxKey !== undefined) {
        assertChildPage(old, parentLevel, declaredMaxKey);
    }
    if (old.kind === 'leaf') {
        const entries = [...old.entries];
        const index = entries.findIndex((entry) => compare(key, entry.key) <= 0);
        if (index >= 0 && entries[index].key === key) {
            if (mode === 'insert') throw new Error('Paged record index identity already exists');
            entries[index] = { key, value };
        } else {
            if (mode === 'replace') throw new Error('Paged record index identity is absent');
            entries.splice(index === -1 ? entries.length : index, 0, { key, value });
        }
        if (entries.length <= PAGED_RECORD_INDEX_FANOUT) {
            const page: PagedRecordIndexPage = { kind: 'leaf', entries };
            return [{ page: await writePage(store, page), max_key: maxKey(page) }];
        }
        const middle = Math.ceil(entries.length / 2);
        const left: PagedRecordIndexPage = { kind: 'leaf', entries: entries.slice(0, middle) };
        const right: PagedRecordIndexPage = { kind: 'leaf', entries: entries.slice(middle) };
        return [
            { page: await writePage(store, left), max_key: maxKey(left) },
            { page: await writePage(store, right), max_key: maxKey(right) },
        ];
    }
    const index = branchIndex(old, key);
    const replacement = await replacePage(
        store,
        old.children[index].page,
        key,
        value,
        mode,
        old.level,
        old.children[index].max_key,
    );
    const children = [...old.children];
    children.splice(index, 1, ...replacement);
    if (children.length <= PAGED_RECORD_INDEX_FANOUT) {
        const page: PagedRecordIndexPage = { kind: 'branch', level: old.level, children };
        return [{ page: await writePage(store, page), max_key: maxKey(page) }];
    }
    const middle = Math.ceil(children.length / 2);
    const left: PagedRecordIndexPage = { kind: 'branch', level: old.level, children: children.slice(0, middle) };
    const right: PagedRecordIndexPage = { kind: 'branch', level: old.level, children: children.slice(middle) };
    return [
        { page: await writePage(store, left), max_key: maxKey(left) },
        { page: await writePage(store, right), max_key: maxKey(right) },
    ];
}

/** Look up one immutable record without scanning any unrelated history page. */
export async function getPagedRecord(
    store: PagedRecordIndexStore,
    root: PagedRecordRef | undefined,
    key: string,
): Promise<PagedRecordValue | undefined> {
    if (root === undefined) return undefined;
    let page = await readPage(store, root);
    while (page.kind === 'branch') {
        const parent = page;
        const child = parent.children[branchIndex(parent, key)];
        page = await readPage(store, child.page);
        assertChildPage(page, parent.level, child.max_key);
    }
    return page.entries.find((entry) => entry.key === key)?.value;
}

/** Copy only the changed path; callers CAS the new root separately after all bytes are durable. */
export async function putPagedRecord(
    store: PagedRecordIndexStore,
    root: PagedRecordRef | undefined,
    key: string,
    valueInput: PagedRecordValue,
    mode: 'insert' | 'replace' = 'insert',
): Promise<PagedRecordRef> {
    if (!key || key.length > 2048) throw new Error('Paged record index key is invalid');
    const value = PagedRecordValueSchema.parse(valueInput);
    if (root === undefined) {
        if (mode === 'replace') throw new Error('Paged record index identity is absent');
        return writePage(store, { kind: 'leaf', entries: [{ key, value }] });
    }
    const previous = await readPage(store, root);
    const replaced = await replacePage(store, root, key, value, mode);
    if (replaced.length === 1) return replaced[0].page;
    return writePage(store, {
        kind: 'branch',
        level: previous.kind === 'leaf' ? 1 : previous.level + 1,
        children: [...replaced],
    });
}

/** One-time import of an already validated legacy snapshot; ordinary appends use putPagedRecord. */
export async function buildPagedRecordIndex(
    store: PagedRecordIndexStore,
    records: readonly { key: string; value: PagedRecordValue }[],
): Promise<PagedRecordRef | undefined> {
    if (records.length === 0) return undefined;
    const sorted = records.map((entry) => ({ key: entry.key, value: PagedRecordValueSchema.parse(entry.value) }));
    sorted.sort((left, right) => compare(left.key, right.key));
    assertOrdered(sorted.map((entry) => entry.key));
    let level = 0;
    let pages: WrittenPage[] = [];
    for (let offset = 0; offset < sorted.length; offset += PAGED_RECORD_INDEX_FANOUT) {
        const page: PagedRecordIndexPage = {
            kind: 'leaf',
            entries: sorted.slice(offset, offset + PAGED_RECORD_INDEX_FANOUT),
        };
        pages.push({ page: await writePage(store, page), max_key: maxKey(page) });
    }
    while (pages.length > 1) {
        level += 1;
        const parents: WrittenPage[] = [];
        for (let offset = 0; offset < pages.length; offset += PAGED_RECORD_INDEX_FANOUT) {
            const page: PagedRecordIndexPage = {
                kind: 'branch',
                level,
                children: pages.slice(offset, offset + PAGED_RECORD_INDEX_FANOUT),
            };
            parents.push({ page: await writePage(store, page), max_key: maxKey(page) });
        }
        pages = parents;
    }
    return pages[0].page;
}

async function* scanPage(
    store: PagedRecordIndexStore,
    ref: PagedRecordRef,
    parentLevel?: number,
    declaredMaxKey?: string,
    lowerBound?: string,
): AsyncGenerator<{ key: string; value: PagedRecordValue }> {
    const page = await readPage(store, ref);
    if (parentLevel !== undefined && declaredMaxKey !== undefined) {
        assertChildPage(page, parentLevel, declaredMaxKey);
    }
    if (page.kind === 'leaf') {
        for (const entry of page.entries) {
            if (
                (lowerBound !== undefined && compare(entry.key, lowerBound) <= 0) ||
                (declaredMaxKey !== undefined && compare(entry.key, declaredMaxKey) > 0)
            ) {
                throw new Error('Paged record index child key is outside its parent partition');
            }
            yield entry;
        }
    } else {
        let priorMax = lowerBound;
        for (const child of page.children) {
            yield* scanPage(store, child.page, page.level, child.max_key, priorMax);
            priorMax = child.max_key;
        }
    }
}

/** Iterate only one chosen index, such as the bounded active-context order index. */
export async function* scanPagedRecords(
    store: PagedRecordIndexStore,
    root: PagedRecordRef | undefined,
): AsyncGenerator<{ key: string; value: PagedRecordValue }> {
    if (root !== undefined) yield* scanPage(store, root);
}
