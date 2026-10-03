import {
    type Asset,
    type ConversationDocument,
    fingerprintJson,
    type ResolveConversationAsset,
    readBoundedConversationAsset,
} from '@llumiverse/conversation';
import { selectedCanonicalTurns } from './canonical-runtime.js';

const MAX_NATIVE_IMAGE_BYTES = 32 * 1024 * 1024;
const MAX_NATIVE_IMAGE_CHUNKS = 65_536;

type HydratedImage = { fingerprint: string; data: string };

export interface CanonicalHostImageOptions {
    document: ConversationDocument;
    label: string;
    selection: Parameters<typeof selectedCanonicalTurns>[1];
    resolve_asset?: ResolveConversationAsset;
    signal?: AbortSignal;
    hydrated: Map<string, HydratedImage>;
    native_external: (asset: Asset) => boolean;
    inline_asset?: (asset: Asset, data: string) => Asset;
}

function hasImageSignature(bytes: Uint8Array, mimeType: string): boolean {
    if (mimeType === 'image/png')
        return (
            bytes.length >= 8 &&
            Buffer.from(bytes.subarray(0, 8)).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
        );
    if (mimeType === 'image/jpeg') return bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
    if (mimeType === 'image/gif') {
        const header = Buffer.from(bytes.subarray(0, 6)).toString('ascii');
        return header === 'GIF87a' || header === 'GIF89a';
    }
    if (mimeType === 'image/webp') {
        return (
            bytes.length >= 12 &&
            Buffer.from(bytes.subarray(0, 4)).toString('ascii') === 'RIFF' &&
            Buffer.from(bytes.subarray(8, 12)).toString('ascii') === 'WEBP'
        );
    }
    return false;
}

/** Transient native projection; the returned copy never replaces the canonical document or its asset refs. */
export async function hydrateCanonicalHostImages(input: CanonicalHostImageOptions): Promise<ConversationDocument> {
    const selected = selectedCanonicalTurns(input.document, input.selection);
    const selectedIds = new Set<string>();
    for (const turn of selected) {
        for (const block of turn.blocks) {
            const content = block.type === 'tool_result' ? block.content : [block];
            for (const item of content) if (item.type === 'image') selectedIds.add(item.asset_id);
        }
    }
    const ids = [...selectedIds].filter((id) => {
        const asset = input.document.assets[id];
        if (asset === undefined) throw new TypeError(`${input.label} selected image asset ${id} is missing`);
        return asset.storage.type === 'external' && !input.native_external(asset);
    });
    if (ids.length === 0) return input.document;
    const nativeDocument = { ...input.document, assets: { ...input.document.assets } };
    let aggregateBytes = 0;
    for (const id of ids) {
        const asset = input.document.assets[id];
        if (asset === undefined || asset.kind !== 'image' || input.resolve_asset === undefined)
            throw new TypeError(`${input.label} external image asset ${id} has no host resolver`);
        if (!['image/png', 'image/jpeg', 'image/gif', 'image/webp'].includes(asset.mime_type))
            throw new TypeError(`${input.label} external image asset ${id} has unsupported MIME`);
        if (asset.byte_length === undefined || asset.byte_length > MAX_NATIVE_IMAGE_BYTES - aggregateBytes)
            throw new RangeError(`${input.label} external images exceed the native projection budget`);
        const fingerprint = await fingerprintJson(asset);
        input.signal?.throwIfAborted();
        let data = input.hydrated.get(id)?.data;
        if (data !== undefined && input.hydrated.get(id)?.fingerprint !== fingerprint)
            throw new TypeError(`${input.label} external image asset ${id} changed during native preparation`);
        if (data === undefined) {
            const bytes = await readBoundedConversationAsset(asset, input.resolve_asset, {
                max_bytes: MAX_NATIVE_IMAGE_BYTES,
                max_chunks: MAX_NATIVE_IMAGE_CHUNKS,
                signal: input.signal,
                require_integrity: true,
                label: `${input.label} external image`,
            });
            input.signal?.throwIfAborted();
            if (!hasImageSignature(bytes, asset.mime_type))
                throw new TypeError(`${input.label} external image asset ${id} has invalid media bytes`);
            data = Buffer.from(bytes).toString('base64');
            input.hydrated.set(id, { fingerprint, data });
        }
        aggregateBytes += asset.byte_length;
        nativeDocument.assets[id] = input.inline_asset?.(asset, data) ?? {
            ...asset,
            storage: { type: 'inline_base64', data },
        };
    }
    input.signal?.throwIfAborted();
    return nativeDocument;
}
