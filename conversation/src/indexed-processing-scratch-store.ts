import { hashContentBytes } from './content-integrity.js';
import type { IndexedConversationRecordStore } from './indexed-conversation.js';
import type { Asset } from './types.js';

const MAX_PAGES = 4096;
const MAX_RECORDS = 8192;
const MAX_BYTES = 64 * 1024 * 1024;

/** An admission planner's isolated immutable overlay. No write reaches the underlying store and
 * no root is published. Bounds apply before storing a copy or reading a nominated descriptor;
 * cached repeated lookups cost once, while every new staged page/record is charged and read back.
 * This is data planning only: verifying a planned asset is not a file access/publication grant.
 */
export function createIndexedProcessingScratchStore(
    underlying: IndexedConversationRecordStore,
    verifyPlannedAsset: (asset: Asset) => Promise<void>,
) {
    const pageBodies = new Map<string, Uint8Array>();
    const recordBodies = new Map<string, Uint8Array>();
    const pageSizes = new Map<string, number>();
    const recordSizes = new Map<string, number>();
    let bytes = 0;
    const charge = (kind: 'page' | 'record', hash: string, size: number) => {
        if (
            !/^sha256:[a-f0-9]{64}$/.test(hash) ||
            !Number.isSafeInteger(size) ||
            size <= 0 ||
            size > (kind === 'page' ? 256 * 1024 : 32 * 1024 * 1024)
        )
            throw new TypeError('Indexed planning descriptor is invalid');
        const sizes = kind === 'page' ? pageSizes : recordSizes;
        const prior = sizes.get(hash);
        if (prior !== undefined) {
            if (prior !== size) throw new Error('Indexed planning immutable descriptor changed size');
            return;
        }
        if (sizes.size >= (kind === 'page' ? MAX_PAGES : MAX_RECORDS) || size > MAX_BYTES - bytes)
            throw new RangeError('Indexed processing planning exceeds its bounded completion I/O profile');
        sizes.set(hash, size);
        bytes += size;
    };
    const ownBody = async (input: Uint8Array, hash: string, size: number) => {
        if (input.byteLength !== size) throw new Error('Indexed planning record/page failed exact size');
        const owned = Uint8Array.from(input);
        if ((await hashContentBytes(owned)).content_hash !== hash)
            throw new Error('Indexed planning record/page failed exact integrity');
        return owned;
    };
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const descriptor = { ...ref };
            charge('page', descriptor.content_hash, descriptor.size_bytes);
            let body = pageBodies.get(descriptor.content_hash);
            if (!body) {
                body = await ownBody(await underlying.read(descriptor), descriptor.content_hash, descriptor.size_bytes);
                pageBodies.set(descriptor.content_hash, body);
            }
            return Uint8Array.from(body);
        },
        async write(input, ref) {
            const descriptor = { ...ref };
            charge('page', descriptor.content_hash, descriptor.size_bytes);
            const body = await ownBody(input, descriptor.content_hash, descriptor.size_bytes);
            pageBodies.set(descriptor.content_hash, body);
        },
        async readRecord(ref) {
            const descriptor = { ...ref };
            charge('record', descriptor.content_hash, descriptor.size_bytes);
            let body = recordBodies.get(descriptor.content_hash);
            if (!body) {
                body = await ownBody(
                    await underlying.readRecord(descriptor),
                    descriptor.content_hash,
                    descriptor.size_bytes,
                );
                recordBodies.set(descriptor.content_hash, body);
            }
            return Uint8Array.from(body);
        },
        async writeRecord(ref, input) {
            const descriptor = { ...ref };
            charge('record', descriptor.content_hash, descriptor.size_bytes);
            const body = await ownBody(input, descriptor.content_hash, descriptor.size_bytes);
            recordBodies.set(descriptor.content_hash, body);
        },
        async assertExternalAssetIntegrity(asset) {
            await verifyPlannedAsset(asset);
        },
    };
    return {
        store,
        profile: () => ({ page_count: pageSizes.size, record_count: recordSizes.size, total_bytes: bytes }),
    };
}
