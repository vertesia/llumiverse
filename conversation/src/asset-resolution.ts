import { hashContentBytes } from './content-integrity.js';
import type { Asset } from './types.js';

export type ResolveConversationAsset = (
    asset: Asset,
    signal?: AbortSignal,
) => AsyncIterable<Uint8Array> | Promise<AsyncIterable<Uint8Array>>;

export interface ReadBoundedConversationAssetOptions {
    max_bytes: number;
    max_chunks: number;
    signal?: AbortSignal;
    require_integrity?: boolean;
    label?: string;
}

function positiveSafeInteger(value: number, field: 'max_bytes' | 'max_chunks'): number {
    if (!Number.isSafeInteger(value) || value <= 0) {
        throw new RangeError(`${field} must be a positive safe integer`);
    }
    return value;
}

/**
 * Read one canonical asset through an in-process resolver under explicit byte and chunk bounds.
 *
 * The helper owns every returned chunk and can require the canonical asset's declared length and
 * content hash. Callers that resolve several assets remain responsible for an aggregate budget.
 */
export async function readBoundedConversationAsset(
    asset: Asset,
    resolveAsset: ResolveConversationAsset,
    options: ReadBoundedConversationAssetOptions,
): Promise<Uint8Array> {
    const maxBytes = positiveSafeInteger(options.max_bytes, 'max_bytes');
    const maxChunks = positiveSafeInteger(options.max_chunks, 'max_chunks');
    const label = options.label ?? 'Conversation asset';
    const signal = options.signal;
    const resolverAsset = structuredClone(asset);
    const { id, byte_length: expectedLength, content_hash: expectedHash } = resolverAsset;
    if (options.require_integrity === true && (expectedLength === undefined || expectedHash === undefined)) {
        throw new Error(`${label} ${id} requires declared byte_length and content_hash`);
    }
    if (expectedLength !== undefined && expectedLength > maxBytes) {
        throw new RangeError(`${label} ${id} exceeds max_bytes before resolution`);
    }

    signal?.throwIfAborted();
    const chunks: Uint8Array[] = [];
    let byteLength = 0;
    let chunkCount = 0;
    for await (const chunk of await resolveAsset(resolverAsset, signal)) {
        signal?.throwIfAborted();
        chunkCount += 1;
        if (chunkCount > maxChunks) {
            throw new RangeError(`${label} ${id} exceeds the hydration chunk limit`);
        }
        if (!(chunk instanceof Uint8Array)) throw new TypeError(`${label} ${id} yielded non-bytes`);
        if (chunk.byteLength === 0) continue;
        if (chunk.byteLength > maxBytes - byteLength) {
            throw new RangeError(`${label} ${id} exceeds max_bytes while resolving`);
        }
        byteLength += chunk.byteLength;
        chunks.push(new Uint8Array(chunk));
    }
    signal?.throwIfAborted();

    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
    }
    if (expectedLength !== undefined && expectedLength !== bytes.byteLength) {
        throw new Error(`${label} ${id} byte length does not match resolved bytes`);
    }
    if (expectedHash !== undefined && (await hashContentBytes(bytes)).content_hash !== expectedHash) {
        throw new Error(`${label} ${id} content hash does not match resolved bytes`);
    }
    return bytes;
}
