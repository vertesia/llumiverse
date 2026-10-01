import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import type { JsonValue } from './types.js';

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
