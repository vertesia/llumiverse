import {
    type Asset,
    type ConversationDocument,
    fingerprintJson,
    preflightJsonInput,
    type ResolveConversationAsset,
    readBoundedConversationAsset,
} from '@llumiverse/conversation';
import { AssetSchema } from '@llumiverse/conversation/schemas';
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
    const externalIds = [...selectedIds].filter((id) => {
        const asset = input.document.assets[id];
        if (asset === undefined) throw new TypeError(`${input.label} selected image asset ${id} is missing`);
        return asset.storage.type === 'external' && !input.native_external(asset);
    });
    if (externalIds.length === 0) return input.document;
    const assets = await hydrateCanonicalSelectedImageAssets({
        ...input,
        assets: input.document.assets,
        selected_ids: selectedIds,
    });
    return assets === input.document.assets ? input.document : { ...input.document, assets };
}

/** Explicit selected assets only; works for a bounded projection without manufacturing a full document. */
export async function hydrateCanonicalSelectedImageAssets(
    input: Omit<CanonicalHostImageOptions, 'document' | 'selection'> & {
        assets: Readonly<Record<string, Asset>>;
        selected_ids: ReadonlySet<string>;
    },
): Promise<Record<string, Asset>> {
    const label = input.label;
    const signal = input.signal;
    const resolver = input.resolve_asset;
    const nativeExternal = input.native_external;
    const inlineAsset = input.inline_asset;
    const hydrated = input.hydrated;
    const selected = [...input.selected_ids];
    const owned: Record<string, Asset> = Object.fromEntries(
        Object.entries(input.assets).map(([id, value]) => {
            if (!preflightJsonInput(value).success)
                throw new TypeError(`${label} selected asset is not owned bounded JSON`);
            return [id, AssetSchema.parse(value)];
        }),
    );
    const ids = selected.filter((id) => {
        const asset = owned[id];
        if (asset === undefined) throw new TypeError(`${label} selected image asset ${id} is missing`);
        return asset.storage.type === 'external' && !nativeExternal(asset);
    });
    if (ids.length === 0) return Object.fromEntries(Object.entries(owned));
    const nativeAssets = { ...owned };
    let aggregateBytes = 0;
    for (const id of ids) {
        const asset = owned[id];
        if (asset === undefined || asset.kind !== 'image' || resolver === undefined)
            throw new TypeError(`${label} external image asset ${id} has no host resolver`);
        if (!['image/png', 'image/jpeg', 'image/gif', 'image/webp'].includes(asset.mime_type))
            throw new TypeError(`${label} external image asset ${id} has unsupported MIME`);
        if (asset.byte_length === undefined || asset.byte_length > MAX_NATIVE_IMAGE_BYTES - aggregateBytes)
            throw new RangeError(`${label} external images exceed the native projection budget`);
        const fingerprint = await fingerprintJson(asset);
        signal?.throwIfAborted();
        let data = hydrated.get(id)?.data;
        if (data !== undefined && hydrated.get(id)?.fingerprint !== fingerprint)
            throw new TypeError(`${label} external image asset ${id} changed during native preparation`);
        if (data === undefined) {
            const bytes = await readBoundedConversationAsset(asset, resolver, {
                max_bytes: MAX_NATIVE_IMAGE_BYTES,
                max_chunks: MAX_NATIVE_IMAGE_CHUNKS,
                signal,
                require_integrity: true,
                label: `${label} external image`,
            });
            signal?.throwIfAborted();
            if (!hasImageSignature(bytes, asset.mime_type))
                throw new TypeError(`${label} external image asset ${id} has invalid media bytes`);
            data = Buffer.from(bytes).toString('base64');
            hydrated.set(id, { fingerprint, data });
        }
        aggregateBytes += asset.byte_length;
        nativeAssets[id] = inlineAsset?.(asset, data) ?? {
            ...asset,
            storage: { type: 'inline_base64', data },
        };
    }
    signal?.throwIfAborted();
    return nativeAssets;
}
