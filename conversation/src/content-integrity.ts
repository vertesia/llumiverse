import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import { Base64Schema } from './schemas/primitives.js';
import type { AssetStorage, JsonValue } from './types.js';

export interface ContentIntegrity {
    content_hash: string;
    byte_length: number;
}

function stableJson(value: JsonValue): string {
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
    return `{${Object.keys(value)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`)
        .join(',')}}`;
}

/** Serialize JSON content with the same sorted-key representation used by canonical fingerprints. */
export function canonicalJsonContentString(value: unknown): string {
    const preflight = preflightJsonInput(value);
    if (!preflight.success) {
        throw new ConversationValidationError('Canonical JSON content failed preflight', preflight.diagnostics);
    }
    return stableJson(value as JsonValue);
}

/** Encode canonical sorted-key JSON content as UTF-8 bytes. */
export function canonicalJsonContentBytes(value: unknown): Uint8Array {
    return new TextEncoder().encode(canonicalJsonContentString(value));
}

/** Hash exact content bytes without incorporating MIME type or storage metadata. */
export async function hashContentBytes(bytes: Uint8Array): Promise<ContentIntegrity> {
    const owned = new Uint8Array(bytes.byteLength);
    owned.set(bytes);
    const digest = await globalThis.crypto.subtle.digest('SHA-256', owned.buffer);
    const hex = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
    return { content_hash: `sha256:${hex}`, byte_length: owned.byteLength };
}

export function utf8ContentBytes(content: string): Uint8Array {
    for (let index = 0; index < content.length; index += 1) {
        const codeUnit = content.charCodeAt(index);
        if (codeUnit < 0xd800 || codeUnit > 0xdfff) continue;
        if (codeUnit > 0xdbff) throw new TypeError('UTF-8 content must not contain an unpaired surrogate');
        const low = content.charCodeAt(index + 1);
        if (!Number.isFinite(low) || low < 0xdc00 || low > 0xdfff) {
            throw new TypeError('UTF-8 content must not contain an unpaired surrogate');
        }
        index += 1;
    }
    return new TextEncoder().encode(content);
}

export async function hashUtf8Content(content: string): Promise<ContentIntegrity> {
    return hashContentBytes(utf8ContentBytes(content));
}

export async function hashCanonicalJsonContent(value: unknown): Promise<ContentIntegrity> {
    return hashContentBytes(canonicalJsonContentBytes(value));
}

function decodeBase64Content(data: string): Uint8Array {
    const parsed = Base64Schema.safeParse(data);
    if (!parsed.success) throw new TypeError('Inline asset data must be valid base64');
    const binary = globalThis.atob(parsed.data);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return bytes;
}

/**
 * Return byte-verifiable integrity for inline storage. External locators intentionally have no byte claim.
 * Inline JSON uses canonical sorted-key JSON UTF-8 bytes because no original wire encoding is retained.
 */
export async function inlineAssetContentIntegrity(storage: AssetStorage): Promise<ContentIntegrity | undefined> {
    if (storage.type === 'external') return undefined;
    if (storage.type === 'inline_base64') return hashContentBytes(decodeBase64Content(storage.data));
    if (storage.type === 'inline_text') return hashUtf8Content(storage.text);
    return hashCanonicalJsonContent(storage.value);
}
