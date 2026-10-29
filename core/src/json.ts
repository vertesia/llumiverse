import type { JSONValue } from '@llumiverse/common';
import { jsonrepair } from 'jsonrepair';

function extractJsonFromText(text: string): string {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    return text.substring(start, end + 1);
}

export function extractAndParseJSON(text: string): JSONValue {
    return parseJSON(extractJsonFromText(text));
}

/**
 * Whether text is a complete JSON object, parsed strictly. Unlike {@link parseJSON} it never repairs the text, so
 * tool-call arguments that were cut off do not count as complete.
 */
export function isJSONObjectText(text: unknown): boolean {
    if (typeof text !== 'string') return false;
    try {
        const parsed: unknown = JSON.parse(text);
        return !!parsed && typeof parsed === 'object' && !Array.isArray(parsed);
    } catch {
        return false;
    }
}

export function parseJSON(text: string): JSONValue {
    text = text.trim();
    try {
        return JSON.parse(text);
    } catch (err: unknown) {
        // use a relaxed parser
        try {
            return JSON.parse(jsonrepair(text));
        } catch {
            // throw the original error
            throw err;
        }
    }
}
