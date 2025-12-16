/**
 * Utilities for cleaning up conversation objects before storage.
 *
 * These functions strip binary data (Uint8Array) and large base64 strings
 * from conversation objects to prevent JSON.stringify corruption and reduce
 * storage bloat.
 */

/**
 * Strip binary data (Uint8Array) from conversation to prevent JSON.stringify corruption.
 *
 * When Uint8Array is passed through JSON.stringify, it gets corrupted into an object
 * like { "0": 137, "1": 80, ... } instead of proper binary data. This breaks
 * subsequent API calls that expect binary data.
 *
 * Recursively walks the conversation object and replaces:
 * - Uint8Array → placeholder string
 *
 * @param obj The conversation object to strip binary data from
 * @returns A new object with all Uint8Array instances replaced with placeholder text
 */
export function stripBinaryFromConversation(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;

    // Strip Uint8Array (Bedrock's binary data for images, documents, videos)
    if (obj instanceof Uint8Array) {
        return '[Binary data stripped - use tool to fetch again]';
    }

    if (Array.isArray(obj)) {
        return obj.map(item => stripBinaryFromConversation(item));
    }

    if (typeof obj === 'object') {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = stripBinaryFromConversation(value);
        }
        return result;
    }

    return obj;
}

/**
 * Strip large base64 image data from conversation to reduce storage bloat.
 *
 * While base64 strings survive JSON.stringify (unlike Uint8Array), they can
 * significantly bloat conversation storage. This function strips:
 * - OpenAI: data:image/...;base64,... URLs
 * - Gemini: inlineData.data base64 strings (when > 1000 chars)
 *
 * @param obj The conversation object to strip base64 images from
 * @returns A new object with large base64 image data replaced with placeholder text
 */
export function stripBase64ImagesFromConversation(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;

    // Strip large base64 data URLs (OpenAI format: data:image/jpeg;base64,...)
    if (typeof obj === 'string' && obj.startsWith('data:image/') && obj.includes(';base64,')) {
        return '[Image data stripped - use tool to fetch again]';
    }

    if (Array.isArray(obj)) {
        return obj.map(item => stripBase64ImagesFromConversation(item));
    }

    if (typeof obj === 'object') {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
            // Strip Gemini inlineData.data (large base64 strings)
            if (key === 'inlineData' && typeof value === 'object' && value !== null) {
                const inlineData = value as Record<string, unknown>;
                if (inlineData.data && typeof inlineData.data === 'string' && (inlineData.data as string).length > 1000) {
                    result[key] = { ...inlineData, data: '[Image data stripped - use tool to fetch again]' };
                    continue;
                }
            }
            result[key] = stripBase64ImagesFromConversation(value);
        }
        return result;
    }

    return obj;
}
