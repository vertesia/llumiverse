/**
 * Utilities for cleaning up conversation objects before storage.
 *
 * These functions strip binary data (Uint8Array) and large base64 strings
 * from conversation objects to prevent JSON.stringify corruption and reduce
 * storage bloat.
 *
 * IMPORTANT: These functions replace entire image/document/video BLOCKS with
 * text placeholders, not just the data. This ensures the conversation remains
 * valid for subsequent API calls.
 */

const IMAGE_PLACEHOLDER = '[Image removed from conversation history]';
const DOCUMENT_PLACEHOLDER = '[Document removed from conversation history]';
const VIDEO_PLACEHOLDER = '[Video removed from conversation history]';

/**
 * Check if an object is a Bedrock image block: { image: { source: { bytes: Uint8Array } } }
 */
function isBedrockImageBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (!o.image || typeof o.image !== 'object') return false;
    const img = o.image as Record<string, unknown>;
    if (!img.source || typeof img.source !== 'object') return false;
    const src = img.source as Record<string, unknown>;
    return src.bytes instanceof Uint8Array;
}

/**
 * Check if an object is a Bedrock document block: { document: { source: { bytes: Uint8Array } } }
 */
function isBedrockDocumentBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (!o.document || typeof o.document !== 'object') return false;
    const doc = o.document as Record<string, unknown>;
    if (!doc.source || typeof doc.source !== 'object') return false;
    const src = doc.source as Record<string, unknown>;
    return src.bytes instanceof Uint8Array;
}

/**
 * Check if an object is a Bedrock video block: { video: { source: { bytes: Uint8Array } } }
 */
function isBedrockVideoBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (!o.video || typeof o.video !== 'object') return false;
    const vid = o.video as Record<string, unknown>;
    if (!vid.source || typeof vid.source !== 'object') return false;
    const src = vid.source as Record<string, unknown>;
    return src.bytes instanceof Uint8Array;
}

/**
 * Strip binary data (Uint8Array) from conversation to prevent JSON.stringify corruption.
 *
 * When Uint8Array is passed through JSON.stringify, it gets corrupted into an object
 * like { "0": 137, "1": 80, ... } instead of proper binary data. This breaks
 * subsequent API calls that expect binary data.
 *
 * This function replaces entire Bedrock image/document/video blocks with text blocks,
 * ensuring the conversation remains valid for subsequent API calls.
 *
 * @param obj The conversation object to strip binary data from
 * @returns A new object with binary content blocks replaced with text placeholders
 */
export function stripBinaryFromConversation(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;

    // Handle Uint8Array directly (shouldn't happen at top level, but be safe)
    if (obj instanceof Uint8Array) {
        return IMAGE_PLACEHOLDER;
    }

    if (Array.isArray(obj)) {
        return obj.map(item => {
            // Replace entire Bedrock image/document/video blocks with text blocks
            if (isBedrockImageBlock(item)) {
                return { text: IMAGE_PLACEHOLDER };
            }
            if (isBedrockDocumentBlock(item)) {
                return { text: DOCUMENT_PLACEHOLDER };
            }
            if (isBedrockVideoBlock(item)) {
                return { text: VIDEO_PLACEHOLDER };
            }
            return stripBinaryFromConversation(item);
        });
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
 * Check if an object is an OpenAI image_url block with base64 data
 */
function isOpenAIBase64ImageBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (o.type !== 'image_url') return false;
    if (!o.image_url || typeof o.image_url !== 'object') return false;
    const imgUrl = o.image_url as Record<string, unknown>;
    return typeof imgUrl.url === 'string' &&
        imgUrl.url.startsWith('data:image/') &&
        imgUrl.url.includes(';base64,');
}

/**
 * Check if an object is a Gemini inlineData block with large base64 data
 */
function isGeminiInlineDataBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (!o.inlineData || typeof o.inlineData !== 'object') return false;
    const inlineData = o.inlineData as Record<string, unknown>;
    return typeof inlineData.data === 'string' && (inlineData.data as string).length > 1000;
}

/**
 * Strip large base64 image data from conversation to reduce storage bloat.
 *
 * While base64 strings survive JSON.stringify (unlike Uint8Array), they can
 * significantly bloat conversation storage. This function replaces entire
 * image blocks with text placeholders:
 * - OpenAI: { type: "image_url", image_url: { url: "data:..." } } → { type: "text", text: "[placeholder]" }
 * - Gemini: { inlineData: { data: "...", mimeType: "..." } } → { text: "[placeholder]" }
 *
 * @param obj The conversation object to strip base64 images from
 * @returns A new object with image blocks replaced with text placeholders
 */
export function stripBase64ImagesFromConversation(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;

    // Handle base64 data URL string directly
    if (typeof obj === 'string' && obj.startsWith('data:image/') && obj.includes(';base64,')) {
        return IMAGE_PLACEHOLDER;
    }

    if (Array.isArray(obj)) {
        return obj.map(item => {
            // Replace entire OpenAI image_url blocks with text blocks
            if (isOpenAIBase64ImageBlock(item)) {
                return { type: 'text', text: IMAGE_PLACEHOLDER };
            }
            // Replace entire Gemini inlineData blocks with text blocks
            if (isGeminiInlineDataBlock(item)) {
                return { text: IMAGE_PLACEHOLDER };
            }
            return stripBase64ImagesFromConversation(item);
        });
    }

    if (typeof obj === 'object') {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = stripBase64ImagesFromConversation(value);
        }
        return result;
    }

    return obj;
}
