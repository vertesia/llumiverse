import { describe, expect, it } from 'vitest';
import { hashContentBytes } from '../src/content-integrity.js';
import {
    type IndexedConversationRecordStore,
    stageIndexedProcessingPhase,
    stageIndexedTextProcessingCompletion,
} from '../src/indexed-conversation.js';
import { createIndexedProcessingScratchStore } from '../src/indexed-processing-scratch-store.js';
import { buildIndexedTextExternalizationOutput } from '../src/indexed-processing-working-set.js';
import { indexedTextClaimFixture } from './indexed-processing-fixture.js';

function readOnlyStore(bodies: Map<string, Uint8Array>, reads: string[]): IndexedConversationRecordStore {
    return {
        async read(ref) {
            reads.push(ref.content_hash);
            const body = bodies.get(ref.content_hash);
            if (!body) throw new Error('Original immutable page missing');
            return Uint8Array.from(body);
        },
        async readRecord(ref) {
            reads.push(ref.content_hash);
            const body = bodies.get(ref.content_hash);
            if (!body) throw new Error('Original immutable record missing');
            return Uint8Array.from(body);
        },
        async write() {
            throw new Error('Admission wrote an underlying page');
        },
        async writeRecord() {
            throw new Error('Admission wrote an underlying record');
        },
    };
}

describe('bounded indexed processing scratch admission', () => {
    // Exact-capacity setup + full transition/retry is CPU work, not a production ACK deadline.
    it('plans actual 1024-block output/completion without publishing an underlying byte or changing the original root', async () => {
        const fixture = await indexedTextClaimFixture(0, 'text', 1024);
        const before = {
            pages: fixture.pages.size,
            records: fixture.records.size,
            root: structuredClone(fixture.staged.root),
        };
        const scratch = createIndexedProcessingScratchStore(fixture.store, async () => {
            throw new Error('Unexpected new archive');
        });
        const output = await buildIndexedTextExternalizationOutput(fixture.workspace);
        const staged = await stageIndexedProcessingPhase(scratch.store, fixture.staged.root, fixture.staged.locator, {
            phase: 'output',
            value: output,
        });
        const done = await stageIndexedTextProcessingCompletion(
            scratch.store,
            staged.root,
            staged.locator,
            fixture.workspace,
        );
        expect(done.completion.status).toBe('applied');
        expect(scratch.profile().page_count).toBeLessThan(4096);
        expect(scratch.profile().record_count).toBeLessThan(8192);
        expect(scratch.profile().total_bytes).toBeLessThan(64 * 1024 * 1024);
        expect(fixture.pages.size).toBe(before.pages);
        expect(fixture.records.size).toBe(before.records);
        expect(fixture.staged.root).toEqual(before.root);
        const retry = await stageIndexedTextProcessingCompletion(
            scratch.store,
            done.root,
            done.locator,
            fixture.workspace,
        );
        expect(retry.applied).toBe(false);
        expect(retry.completion).toEqual(done.completion);
    }, 30_000);

    it('charges one immutable descriptor once and returns separately owned cached bytes', async () => {
        const bytes = new TextEncoder().encode('exact original');
        const integrity = await hashContentBytes(bytes);
        const reads: string[] = [];
        const scratch = createIndexedProcessingScratchStore(
            readOnlyStore(new Map([[integrity.content_hash, bytes]]), reads),
            async () => undefined,
        );
        const ref = {
            storage: 'record' as const,
            kind: 'fixture',
            id: 'original',
            content_hash: integrity.content_hash,
            size_bytes: integrity.byte_length,
        };
        const first = await scratch.store.readRecord(ref);
        first[0] = 0;
        const second = await scratch.store.readRecord(ref);
        expect(second).toEqual(bytes);
        expect(reads).toHaveLength(1);
        expect(scratch.profile()).toEqual({ page_count: 0, record_count: 1, total_bytes: bytes.byteLength });
        await expect(scratch.store.readRecord({ ...ref, size_bytes: ref.size_bytes + 1 })).rejects.toThrow(
            'changed size',
        );
        expect(reads).toHaveLength(1);
    });

    it('owns planned write bytes before asynchronous hashing and retains them only in the scratch overlay', async () => {
        const bytes = new TextEncoder().encode('owned planned page');
        const original = Uint8Array.from(bytes);
        const integrity = await hashContentBytes(original);
        const reads: string[] = [];
        const scratch = createIndexedProcessingScratchStore(readOnlyStore(new Map(), reads), async () => undefined);
        const ref = { content_hash: integrity.content_hash, size_bytes: integrity.byte_length };
        const pending = scratch.store.write(bytes, ref);
        bytes.fill(0);
        await pending;
        expect(await scratch.store.read(ref)).toEqual(original);
        expect(reads).toHaveLength(0);
    });

    it('admits exactly 64 MiB unique record bytes and rejects the next descriptor before its underlying read', async () => {
        const first = new Uint8Array(32 * 1024 * 1024);
        const second = Uint8Array.from(first);
        second[0] = 1;
        const one = await hashContentBytes(first);
        const two = await hashContentBytes(second);
        const reads: string[] = [];
        const scratch = createIndexedProcessingScratchStore(
            readOnlyStore(
                new Map([
                    [one.content_hash, first],
                    [two.content_hash, second],
                ]),
                reads,
            ),
            async () => undefined,
        );
        for (const [id, value] of [
            ['first', one],
            ['second', two],
        ] as const) {
            await scratch.store.readRecord({
                storage: 'record',
                kind: 'fixture',
                id,
                content_hash: value.content_hash,
                size_bytes: value.byte_length,
            });
        }
        expect(scratch.profile().total_bytes).toBe(64 * 1024 * 1024);
        const extra = await hashContentBytes(new Uint8Array([2]));
        await expect(
            scratch.store.readRecord({
                storage: 'record',
                kind: 'fixture',
                id: 'extra',
                content_hash: extra.content_hash,
                size_bytes: 1,
            }),
        ).rejects.toThrow('bounded completion');
        expect(reads).toHaveLength(2);
    });

    it('rejects an oversized individual descriptor before reading or retaining its body', async () => {
        const reads: string[] = [];
        const scratch = createIndexedProcessingScratchStore(readOnlyStore(new Map(), reads), async () => undefined);
        await expect(
            scratch.store.read({ content_hash: `sha256:${'a'.repeat(64)}`, size_bytes: 256 * 1024 + 1 }),
        ).rejects.toThrow('descriptor is invalid');
        await expect(
            scratch.store.readRecord({
                storage: 'record',
                kind: 'fixture',
                id: 'huge',
                content_hash: `sha256:${'b'.repeat(64)}`,
                size_bytes: 32 * 1024 * 1024 + 1,
            }),
        ).rejects.toThrow('descriptor is invalid');
        expect(reads).toHaveLength(0);
        expect(scratch.profile()).toEqual({ page_count: 0, record_count: 0, total_bytes: 0 });
    });
});
