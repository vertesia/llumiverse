import { describe, expect, it, vi } from 'vitest';
import { type ResolveConversationAsset, readBoundedConversationAsset } from '../src/asset-resolution.js';
import { hashContentBytes } from '../src/content-integrity.js';
import type { Asset } from '../src/types.js';

const recordedAt = '2026-10-02T00:00:00.000Z';

async function asset(bytes: Uint8Array): Promise<Asset> {
    return {
        id: 'asset:image',
        kind: 'image',
        mime_type: 'image/png',
        storage: { type: 'external', resolver: 'url', locator: { url: 'gs://bucket/image.png' } },
        provenance: { type: 'received' },
        ...(await hashContentBytes(bytes)),
        created_at: recordedAt,
    };
}

describe('bounded conversation asset resolution', () => {
    it('keeps the declared integrity contract when the caller changes its asset during resolution', async () => {
        const input = await asset(new Uint8Array([1, 2, 3]));
        const wrongBytes = new Uint8Array([7, 8, 9]);
        const wrongHash = (await hashContentBytes(wrongBytes)).content_hash;
        const reading = readBoundedConversationAsset(
            input,
            async function* () {
                yield wrongBytes;
            },
            { max_bytes: 3, max_chunks: 1, require_integrity: true },
        );
        input.content_hash = wrongHash;

        await expect(reading).rejects.toThrow('content hash does not match resolved bytes');
    });

    it('gives the resolver its own descriptor without allowing it to change the integrity contract', async () => {
        const input = await asset(new Uint8Array([1, 2, 3]));
        const original = structuredClone(input);
        const wrongBytes = new Uint8Array([7, 8]);
        const wrongIntegrity = await hashContentBytes(wrongBytes);

        await expect(
            readBoundedConversationAsset(
                input,
                async function* (descriptor) {
                    Object.assign(descriptor, wrongIntegrity);
                    descriptor.storage = {
                        type: 'external',
                        resolver: 'url',
                        locator: { url: 'gs://bucket/other.png' },
                    };
                    yield wrongBytes;
                },
                { max_bytes: 3, max_chunks: 1, require_integrity: true },
            ),
        ).rejects.toThrow('byte length does not match resolved bytes');
        expect(input).toEqual(original);
    });

    it('owns reused chunks even when their slice method returns a borrowed view', async () => {
        const input = await asset(new Uint8Array([1, 2, 3, 4]));
        const reusable = new Uint8Array([1, 2]);
        // Node Buffer.slice has this borrowed-view behavior; the library also runs without Node types.
        vi.spyOn(reusable, 'slice').mockImplementation((start, end) => reusable.subarray(start, end));
        const resolve: ResolveConversationAsset = async function* () {
            yield reusable;
            reusable.set([3, 4]);
            yield reusable;
        };

        await expect(
            readBoundedConversationAsset(input, resolve, {
                max_bytes: 4,
                max_chunks: 2,
                require_integrity: true,
            }),
        ).resolves.toEqual(new Uint8Array([1, 2, 3, 4]));
    });

    it('owns chunks and validates declared length and hash', async () => {
        const first = new Uint8Array([1, 2]);
        const second = new Uint8Array([3]);
        const input = await asset(new Uint8Array([1, 2, 3]));
        const resolve: ResolveConversationAsset = async function* () {
            yield first;
            yield second;
        };

        const bytes = await readBoundedConversationAsset(input, resolve, {
            max_bytes: 3,
            max_chunks: 2,
            require_integrity: true,
        });
        first.fill(9);
        second.fill(9);

        expect(bytes).toEqual(new Uint8Array([1, 2, 3]));
    });

    it('rejects missing integrity, wrong bytes, excessive chunks and cancellation', async () => {
        const exact = new Uint8Array([1, 2, 3]);
        const missing = await asset(exact);
        delete missing.byte_length;
        delete missing.content_hash;
        const resolve = vi.fn<ResolveConversationAsset>(async function* () {
            yield exact;
        });
        await expect(
            readBoundedConversationAsset(missing, resolve, {
                max_bytes: 3,
                max_chunks: 2,
                require_integrity: true,
            }),
        ).rejects.toThrow('requires declared byte_length and content_hash');
        expect(resolve).not.toHaveBeenCalled();

        const wrong = await asset(exact);
        await expect(
            readBoundedConversationAsset(
                wrong,
                async function* () {
                    yield new Uint8Array([1, 2, 4]);
                },
                { max_bytes: 3, max_chunks: 2, require_integrity: true },
            ),
        ).rejects.toThrow('content hash does not match resolved bytes');

        await expect(
            readBoundedConversationAsset(
                wrong,
                async function* () {
                    yield new Uint8Array();
                    yield new Uint8Array();
                    yield exact;
                },
                { max_bytes: 3, max_chunks: 2 },
            ),
        ).rejects.toThrow('exceeds the hydration chunk limit');

        const controller = new AbortController();
        controller.abort(new Error('cancel asset'));
        await expect(
            readBoundedConversationAsset(wrong, resolve, {
                max_bytes: 3,
                max_chunks: 2,
                signal: controller.signal,
            }),
        ).rejects.toThrow('cancel asset');
    });
});
