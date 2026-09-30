import type { JSONValue } from '@llumiverse/common';
import { jsonrepair } from 'jsonrepair';

function stripCodeFence(text: string): string {
    const match = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
    return match ? match[1].trim() : text;
}

function extractJsonFromText(text: string): { value: string; complete: boolean } | undefined {
    const objectStart = text.indexOf('{');
    const arrayStart = text.indexOf('[');
    const start = objectStart < 0 ? arrayStart : arrayStart < 0 ? objectStart : Math.min(objectStart, arrayStart);
    if (start < 0) return undefined;

    const delimiters: string[] = [];
    let quote: '"' | "'" | undefined;
    let escaped = false;
    for (let index = start; index < text.length; index++) {
        const character = text[index];
        if (quote) {
            if (escaped) {
                escaped = false;
            } else if (character === '\\') {
                escaped = true;
            } else if (character === quote) {
                quote = undefined;
            }
            continue;
        }
        if (character === '"' || character === "'") {
            quote = character;
        } else if (character === '{' || character === '[') {
            delimiters.push(character === '{' ? '}' : ']');
        } else if (character === '}' || character === ']') {
            if (delimiters.pop() !== character) {
                return { value: text.slice(start, index + 1), complete: false };
            }
            if (delimiters.length === 0) {
                return { value: text.slice(start, index + 1), complete: true };
            }
        }
    }
    return { value: text.slice(start), complete: false };
}

export function extractAndParseJSON(text: string, allowRepair = true): JSONValue {
    const normalized = stripCodeFence(text.trim());
    const extracted = extractJsonFromText(normalized);
    if (extracted) {
        if (!extracted.complete) {
            throw new SyntaxError('Unexpected end of JSON input');
        }
        return parseJSON(extracted.value, allowRepair);
    }

    return parseJSON(normalized, allowRepair);
}

export function parseJSON(text: string, allowRepair = true): JSONValue {
    text = text.trim();
    try {
        return JSON.parse(text);
    } catch (err: unknown) {
        if (!allowRepair) throw err;

        // use a relaxed parser
        try {
            return JSON.parse(jsonrepair(text));
        } catch {
            // throw the original error
            throw err;
        }
    }
}
