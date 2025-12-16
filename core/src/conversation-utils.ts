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

/** Metadata key used to store turn information in conversations */
const META_KEY = '_llumiverse_meta';

/**
 * Metadata stored in conversation objects to track turn numbers for deferred image stripping.
 */
export interface ConversationMeta {
    /** Current turn number (incremented each time a message is added) */
    turnNumber: number;
}

/**
 * Options for stripping functions
 */
export interface StripOptions {
    /**
     * Number of turns to keep images before stripping.
     * - 0 or undefined: Strip immediately (default)
     * - N > 0: Keep images for N turns, then strip
     */
    keepForTurns?: number;
    /**
     * Current turn number. Used with keepForTurns to determine when to strip.
     * If not provided, will be read from conversation metadata.
     */
    currentTurn?: number;
}

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
 * Check if an object is a serialized Bedrock image block (Uint8Array converted to base64)
 */
function isSerializedBedrockImageBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (!o.image || typeof o.image !== 'object') return false;
    const img = o.image as Record<string, unknown>;
    if (!img.source || typeof img.source !== 'object') return false;
    const src = img.source as Record<string, unknown>;
    // Check for our serialized format: bytes: { _base64: string }
    if (!src.bytes || typeof src.bytes !== 'object') return false;
    const bytes = src.bytes as Record<string, unknown>;
    return typeof bytes._base64 === 'string';
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
 * Check if an object is a serialized Bedrock document block
 */
function isSerializedBedrockDocumentBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (!o.document || typeof o.document !== 'object') return false;
    const doc = o.document as Record<string, unknown>;
    if (!doc.source || typeof doc.source !== 'object') return false;
    const src = doc.source as Record<string, unknown>;
    // Check for our serialized format: bytes: { _base64: string }
    if (!src.bytes || typeof src.bytes !== 'object') return false;
    const bytes = src.bytes as Record<string, unknown>;
    return typeof bytes._base64 === 'string';
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
 * Check if an object is a serialized Bedrock video block
 */
function isSerializedBedrockVideoBlock(obj: unknown): boolean {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as Record<string, unknown>;
    if (!o.video || typeof o.video !== 'object') return false;
    const vid = o.video as Record<string, unknown>;
    if (!vid.source || typeof vid.source !== 'object') return false;
    const src = vid.source as Record<string, unknown>;
    // Check for our serialized format: bytes: { _base64: string }
    if (!src.bytes || typeof src.bytes !== 'object') return false;
    const bytes = src.bytes as Record<string, unknown>;
    return typeof bytes._base64 === 'string';
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
 * Convert Uint8Array to base64 string for safe JSON serialization.
 */
function uint8ArrayToBase64(bytes: Uint8Array): string {
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

/**
 * Convert base64 string back to Uint8Array.
 */
function base64ToUint8Array(base64: string): Uint8Array {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

/**
 * Get metadata from a conversation object, or return defaults.
 */
export function getConversationMeta(conversation: unknown): ConversationMeta {
    if (typeof conversation === 'object' && conversation !== null) {
        const meta = (conversation as Record<string, unknown>)[META_KEY];
        if (meta && typeof meta === 'object') {
            return meta as ConversationMeta;
        }
    }
    return { turnNumber: 0 };
}

/**
 * Set metadata on a conversation object.
 */
export function setConversationMeta(conversation: unknown, meta: ConversationMeta): unknown {
    if (typeof conversation === 'object' && conversation !== null) {
        return { ...conversation as object, [META_KEY]: meta };
    }
    return conversation;
}

/**
 * Increment the turn number in a conversation and return the updated conversation.
 */
export function incrementConversationTurn(conversation: unknown): unknown {
    const meta = getConversationMeta(conversation);
    return setConversationMeta(conversation, { ...meta, turnNumber: meta.turnNumber + 1 });
}

/**
 * Strip binary data (Uint8Array) from conversation to prevent JSON.stringify corruption.
 *
 * When Uint8Array is passed through JSON.stringify, it gets corrupted into an object
 * like { "0": 137, "1": 80, ... } instead of proper binary data. This breaks
 * subsequent API calls that expect binary data.
 *
 * This function either:
 * - Strips images immediately (keepForTurns = 0, default)
 * - Serializes images to base64 for safe storage, then strips after N turns
 *
 * @param obj The conversation object to strip binary data from
 * @param options Optional settings for turn-based stripping
 * @returns A new object with binary content handled appropriately
 */
export function stripBinaryFromConversation(obj: unknown, options?: StripOptions): unknown {
    const { keepForTurns = 0 } = options ?? {};
    const currentTurn = options?.currentTurn ?? getConversationMeta(obj).turnNumber;

    // If we should keep images and haven't exceeded the turn threshold,
    // serialize Uint8Array to base64 for safe JSON storage
    if (keepForTurns > 0 && currentTurn < keepForTurns) {
        return serializeBinaryForStorage(obj);
    }

    // Strip all binary/serialized images
    return stripBinaryFromConversationInternal(obj);
}

/**
 * Serialize Uint8Array to base64 for safe JSON storage, preserving the image structure.
 */
function serializeBinaryForStorage(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;

    if (obj instanceof Uint8Array) {
        return { _base64: uint8ArrayToBase64(obj) };
    }

    if (Array.isArray(obj)) {
        return obj.map(item => serializeBinaryForStorage(item));
    }

    if (typeof obj === 'object') {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = serializeBinaryForStorage(value);
        }
        return result;
    }

    return obj;
}

/**
 * Restore Uint8Array from base64 serialization.
 * Call this before sending conversation to API if images were preserved.
 */
export function deserializeBinaryFromStorage(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;

    // Check for our serialized format
    if (typeof obj === 'object' && obj !== null) {
        const o = obj as Record<string, unknown>;
        if (typeof o._base64 === 'string' && Object.keys(o).length === 1) {
            return base64ToUint8Array(o._base64);
        }
    }

    if (Array.isArray(obj)) {
        return obj.map(item => deserializeBinaryFromStorage(item));
    }

    if (typeof obj === 'object') {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = deserializeBinaryFromStorage(value);
        }
        return result;
    }

    return obj;
}

function stripBinaryFromConversationInternal(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;

    // Handle Uint8Array directly
    if (obj instanceof Uint8Array) {
        return IMAGE_PLACEHOLDER;
    }

    // Handle our serialized format
    if (typeof obj === 'object' && obj !== null) {
        const o = obj as Record<string, unknown>;
        if (typeof o._base64 === 'string' && Object.keys(o).length === 1) {
            return IMAGE_PLACEHOLDER;
        }
    }

    if (Array.isArray(obj)) {
        return obj.map(item => {
            // Replace entire Bedrock image/document/video blocks with text blocks
            if (isBedrockImageBlock(item) || isSerializedBedrockImageBlock(item)) {
                return { text: IMAGE_PLACEHOLDER };
            }
            if (isBedrockDocumentBlock(item) || isSerializedBedrockDocumentBlock(item)) {
                return { text: DOCUMENT_PLACEHOLDER };
            }
            if (isBedrockVideoBlock(item) || isSerializedBedrockVideoBlock(item)) {
                return { text: VIDEO_PLACEHOLDER };
            }
            return stripBinaryFromConversationInternal(item);
        });
    }

    if (typeof obj === 'object') {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
            // Preserve metadata
            if (key === META_KEY) {
                result[key] = value;
            } else {
                result[key] = stripBinaryFromConversationInternal(value);
            }
        }
        return result;
    }

    return obj;
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
 * @param options Optional settings for turn-based stripping
 * @returns A new object with image blocks replaced with text placeholders
 */
export function stripBase64ImagesFromConversation(obj: unknown, options?: StripOptions): unknown {
    const { keepForTurns = 0 } = options ?? {};
    const currentTurn = options?.currentTurn ?? getConversationMeta(obj).turnNumber;

    // If we should keep images and haven't exceeded the turn threshold, don't strip
    // (base64 strings are already safe for JSON serialization)
    if (keepForTurns > 0 && currentTurn < keepForTurns) {
        return obj;
    }

    return stripBase64ImagesFromConversationInternal(obj);
}

function stripBase64ImagesFromConversationInternal(obj: unknown): unknown {
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
            return stripBase64ImagesFromConversationInternal(item);
        });
    }

    if (typeof obj === 'object') {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
            // Preserve metadata
            if (key === META_KEY) {
                result[key] = value;
            } else {
                result[key] = stripBase64ImagesFromConversationInternal(value);
            }
        }
        return result;
    }

    return obj;
}
