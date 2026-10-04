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
type NativeMediaKind = 'image' | 'document' | 'audio' | 'video';

function isNativeMediaKind(kind: Asset['kind']): kind is NativeMediaKind {
    return kind === 'image' || kind === 'document' || kind === 'audio' || kind === 'video';
}

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

export interface CanonicalHostMediaOptions extends CanonicalHostImageOptions {
    media_kinds: readonly NativeMediaKind[];
}

function hasExactAsciiBytes(bytes: Uint8Array, offset: number, text: string): boolean {
    if (bytes.length < offset + text.length) return false;
    for (let index = 0; index < text.length; index++) {
        if (bytes[offset + index] !== text.charCodeAt(index)) return false;
    }
    return true;
}

function hasImageSignature(bytes: Uint8Array, mimeType: string): boolean {
    if (mimeType === 'image/png')
        return (
            bytes.length >= 8 &&
            Buffer.from(bytes.subarray(0, 8)).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
        );
    if (mimeType === 'image/jpeg') return bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
    if (mimeType === 'image/gif') {
        return hasExactAsciiBytes(bytes, 0, 'GIF87a') || hasExactAsciiBytes(bytes, 0, 'GIF89a');
    }
    if (mimeType === 'image/webp') {
        return bytes.length >= 12 && hasExactAsciiBytes(bytes, 0, 'RIFF') && hasExactAsciiBytes(bytes, 8, 'WEBP');
    }
    return false;
}

function hasMediaSignature(bytes: Uint8Array, asset: Asset): boolean {
    if (asset.kind === 'image') return hasImageSignature(bytes, asset.mime_type);
    if (asset.kind === 'document' && asset.mime_type === 'application/pdf') {
        return hasExactAsciiBytes(bytes, 0, '%PDF-');
    }
    if (asset.kind === 'audio' && (asset.mime_type === 'audio/wav' || asset.mime_type === 'audio/x-wav')) {
        return hasExactAsciiBytes(bytes, 0, 'RIFF') && hasExactAsciiBytes(bytes, 8, 'WAVE');
    }
    if (asset.kind === 'audio' && (asset.mime_type === 'audio/mpeg' || asset.mime_type === 'audio/mp3')) {
        return (
            hasExactAsciiBytes(bytes, 0, 'ID3') || (bytes.length >= 2 && bytes[0] === 255 && (bytes[1] & 0xe0) === 0xe0)
        );
    }
    if (asset.kind === 'video' && asset.mime_type === 'video/mp4') {
        return hasExactAsciiBytes(bytes, 4, 'ftyp');
    }
    return false;
}

function supportedMediaMime(asset: Asset): boolean {
    if (asset.kind === 'image') return ['image/png', 'image/jpeg', 'image/gif', 'image/webp'].includes(asset.mime_type);
    if (asset.kind === 'document') return asset.mime_type === 'application/pdf';
    if (asset.kind === 'audio')
        return ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3'].includes(asset.mime_type);
    return asset.kind === 'video' && asset.mime_type === 'video/mp4';
}

/** Transient native projection; the returned copy never replaces the canonical document or its asset refs. */
export async function hydrateCanonicalHostImages(input: CanonicalHostImageOptions): Promise<ConversationDocument> {
    return hydrateCanonicalHostMedia({ ...input, media_kinds: ['image'] });
}

/** Selected typed media only; the canonical source retains its exact external references. */
export async function hydrateCanonicalHostMedia(input: CanonicalHostMediaOptions): Promise<ConversationDocument> {
    const selected = selectedCanonicalTurns(input.document, input.selection);
    const selectedIds = new Set<string>();
    for (const turn of selected) {
        for (const block of turn.blocks) {
            const content = block.type === 'tool_result' ? block.content : [block];
            for (const item of content) {
                if (
                    (item.type === 'image' ||
                        item.type === 'document' ||
                        item.type === 'audio' ||
                        item.type === 'video') &&
                    input.media_kinds.includes(item.type)
                )
                    selectedIds.add(item.asset_id);
            }
        }
    }
    const externalIds = [...selectedIds].filter((id) => {
        const asset = input.document.assets[id];
        if (asset === undefined) throw new TypeError(`${input.label} selected image asset ${id} is missing`);
        return asset.storage.type === 'external' && !input.native_external(asset);
    });
    if (externalIds.length === 0) return input.document;
    const assets = await hydrateCanonicalSelectedMediaAssets({
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
    return hydrateCanonicalSelectedMediaAssets({ ...input, media_kinds: ['image'] });
}

export async function hydrateCanonicalSelectedMediaAssets(
    input: Omit<CanonicalHostMediaOptions, 'document' | 'selection'> & {
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
    const mediaLabel = input.media_kinds.length === 1 && input.media_kinds[0] === 'image' ? 'image' : 'media';
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
        if (
            asset === undefined ||
            !isNativeMediaKind(asset.kind) ||
            !input.media_kinds.includes(asset.kind) ||
            resolver === undefined
        )
            throw new TypeError(`${label} external ${mediaLabel} asset ${id} has no host resolver`);
        if (!supportedMediaMime(asset))
            throw new TypeError(`${label} external ${mediaLabel} asset ${id} has unsupported MIME`);
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
            if (!hasMediaSignature(bytes, asset))
                throw new TypeError(`${label} external ${mediaLabel} asset ${id} has invalid media bytes`);
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
