import { type AssetStorage, hashContentBytes } from '@llumiverse/conversation';
import type { ExecutionOptions } from '@llumiverse/core';

export const MAX_GENERATED_IMAGE_BYTES = 50_000_000;
export const MAX_INLINE_GENERATED_IMAGE_BYTES = 8_000_000;
const MAX_CANONICAL_JSON_BYTES = 32 * 1024 * 1024;
const CANONICAL_RESPONSE_JSON_RESERVE_BYTES = 1024 * 1024;

export interface VerifiedGeneratedImage {
    bytes: Uint8Array;
    mime_type: string;
    content_hash: string;
}

function isBase64Alphabet(code: number): boolean {
    return (
        (code >= 0x41 && code <= 0x5a) ||
        (code >= 0x61 && code <= 0x7a) ||
        (code >= 0x30 && code <= 0x39) ||
        code === 0x2b ||
        code === 0x2f
    );
}

export function decodeStrictBase64(value: string, maximumBytes: number, label: string): Uint8Array {
    const maximumEncodedLength = Math.ceil(maximumBytes / 3) * 4;
    if (value.length === 0 || value.length > maximumEncodedLength || value.length % 4 !== 0) {
        throw new Error(`${label} image response contains malformed base64 data`);
    }
    let padding = 0;
    if (value.endsWith('==')) padding = 2;
    else if (value.endsWith('=')) padding = 1;
    for (let index = 0; index < value.length - padding; index += 1) {
        if (!isBase64Alphabet(value.charCodeAt(index))) {
            throw new Error(`${label} image response contains malformed base64 data`);
        }
    }
    for (let index = value.length - padding; index < value.length; index += 1) {
        if (value.charCodeAt(index) !== 0x3d) {
            throw new Error(`${label} image response contains malformed base64 data`);
        }
    }
    const bytes = new Uint8Array(Buffer.from(value, 'base64'));
    if (bytes.byteLength === 0 || bytes.byteLength > maximumBytes || Buffer.from(bytes).toString('base64') !== value) {
        throw new Error(`${label} image response contains malformed base64 data`);
    }
    return bytes;
}

export function detectGeneratedImageMimeType(bytes: Uint8Array, label: string): string {
    if (
        bytes.byteLength >= 8 &&
        bytes[0] === 0x89 &&
        bytes[1] === 0x50 &&
        bytes[2] === 0x4e &&
        bytes[3] === 0x47 &&
        bytes[4] === 0x0d &&
        bytes[5] === 0x0a &&
        bytes[6] === 0x1a &&
        bytes[7] === 0x0a
    ) {
        return 'image/png';
    }
    if (bytes.byteLength >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
        return 'image/jpeg';
    }
    if (
        bytes.byteLength >= 12 &&
        bytes[0] === 0x52 &&
        bytes[1] === 0x49 &&
        bytes[2] === 0x46 &&
        bytes[3] === 0x46 &&
        bytes[8] === 0x57 &&
        bytes[9] === 0x45 &&
        bytes[10] === 0x42 &&
        bytes[11] === 0x50
    ) {
        return 'image/webp';
    }
    throw new Error(`${label} generated image bytes have an unsupported format`);
}

export async function generatedImageContentHash(bytes: Uint8Array): Promise<string> {
    return (await hashContentBytes(bytes)).content_hash;
}

export async function verifiedBase64GeneratedImage(
    value: string,
    maximumBytes: number,
    label: string,
    declaredMimeType?: string,
): Promise<VerifiedGeneratedImage> {
    const bytes = decodeStrictBase64(value, maximumBytes, label);
    const mimeType = detectGeneratedImageMimeType(bytes, label);
    if (declaredMimeType !== undefined && declaredMimeType !== mimeType) {
        throw new Error(`${label} generated image MIME type ${declaredMimeType} does not match its bytes`);
    }
    return { bytes, mime_type: mimeType, content_hash: await generatedImageContentHash(bytes) };
}

export async function generatedImageStorage(
    image: VerifiedGeneratedImage,
    options: ExecutionOptions,
    label: string,
    signal?: AbortSignal,
): Promise<AssetStorage> {
    if (options.store_generated_asset === undefined) {
        if (image.bytes.byteLength > MAX_INLINE_GENERATED_IMAGE_BYTES) {
            throw new Error(
                `${label} generated image exceeds the ${MAX_INLINE_GENERATED_IMAGE_BYTES} byte inline asset limit`,
            );
        }
        return { type: 'inline_base64', data: Buffer.from(image.bytes).toString('base64') };
    }
    const source = new ReadableStream<Uint8Array>({
        start(controller) {
            controller.enqueue(image.bytes.slice());
            controller.close();
        },
    });
    const stored = await options.store_generated_asset(source, { kind: 'image', mime_type: image.mime_type }, signal);
    signal?.throwIfAborted();
    if (stored.byte_length !== image.bytes.byteLength || stored.content_hash !== image.content_hash) {
        throw new Error(`Generated asset storage did not preserve the exact ${label} image bytes`);
    }
    if (stored.storage.type !== 'external') {
        throw new Error('Generated asset storage must return external canonical storage');
    }
    return stored.storage;
}

export function maximumGeneratedImageOutputBytes(document: unknown, options: ExecutionOptions): number {
    if (options.store_generated_asset !== undefined) return MAX_GENERATED_IMAGE_BYTES;
    const currentBytes = new TextEncoder().encode(JSON.stringify(document)).byteLength;
    const remainingJsonBytes = MAX_CANONICAL_JSON_BYTES - currentBytes - CANONICAL_RESPONSE_JSON_RESERVE_BYTES;
    const remainingDecodedBytes = Math.floor((remainingJsonBytes * 3) / 4);
    const maximum = Math.min(MAX_INLINE_GENERATED_IMAGE_BYTES, remainingDecodedBytes);
    if (maximum <= 0) {
        throw new Error('Canonical conversation has no safe capacity for inline generated image output');
    }
    return maximum;
}
