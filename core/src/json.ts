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
