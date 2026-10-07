import { hashCanonicalJsonContent } from './content-integrity.js';
import { ConversationValidationError } from './diagnostics.js';

/** Produce a browser-safe SHA-256 fingerprint for JSON-safe canonical or native data. */
export async function fingerprintJson(value: unknown): Promise<string> {
    try {
        return (await hashCanonicalJsonContent(value)).content_hash;
    } catch (error: unknown) {
        if (error instanceof ConversationValidationError) {
            throw new ConversationValidationError('Fingerprint input failed JSON preflight', error.diagnostics);
        }
        throw error;
    }
}

/** Derive a compact deterministic entity ID from a stable request/import identity. */
export async function deriveConversationId(kind: string, ...identity: string[]): Promise<string> {
    const fingerprint = await fingerprintJson([kind, ...identity]);
    return `${kind}_${fingerprint.slice('sha256:'.length, 'sha256:'.length + 32)}`;
}
