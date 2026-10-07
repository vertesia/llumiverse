import type { JSONOutputDiagnostic, JSONValue } from '@llumiverse/common';
import { jsonrepair } from 'jsonrepair';

export interface JSONOutputParseOptions {
    /** Disable syntax repair. Explicit JSON fences and a single value in prose may still be extracted. */
    allowRepair?: boolean;
    onDiagnostic?: (diagnostic: JSONOutputDiagnostic) => void;
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function repairJSON(text: string, parseError: unknown): JSONValue {
    try {
        const repaired = jsonrepair(text);
        return JSON.parse(repaired);
    } catch (repairError: unknown) {
        throw new SyntaxError(`JSON repair failed: ${errorMessage(repairError)}`, {
            cause: new AggregateError([parseError, repairError], 'JSON parsing and repair failed'),
        });
    }
}

/** Parse JSON, optionally using jsonrepair for malformed or incomplete input. */
export function parseJSON(text: string, allowRepair = true): JSONValue {
    try {
        return JSON.parse(text);
    } catch (parseError: unknown) {
        if (!allowRepair) throw parseError;
        return repairJSON(text, parseError);
    }
}

function extractCandidates(text: string): string[] {
    const fence = text.match(/^```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```$/i);
    if (fence) return [fence[1].trim()];

    const candidates: string[] = [];
    const start = text.search(/[[{]/);
    const end = Math.max(text.lastIndexOf('}'), text.lastIndexOf(']')) + 1;
    if (start >= 0 && end > start) candidates.push(text.slice(start, end));

    // Retain the old object extraction when surrounding prose contains brackets or other formatting.
    const objectStart = text.indexOf('{');
    const objectEnd = text.lastIndexOf('}') + 1;
    if (objectStart >= 0 && objectEnd > objectStart) candidates.push(text.slice(objectStart, objectEnd));
    return candidates;
}

/** Strict parsing first, then optional jsonrepair recovery. Callers validate the recovered value against their schema. */
export function parseJSONOutput(text: string, options: JSONOutputParseOptions = {}): JSONValue {
    let lastError: unknown;
    try {
        return JSON.parse(text);
    } catch (error: unknown) {
        lastError = error;
    }

    const normalized = text.trim();
    const candidates = extractCandidates(normalized);
    // Repair a root JSON value before extracting from it: braces inside unfinished strings are content.
    const sources = /^[{["']/.test(normalized) ? [normalized, ...candidates] : [...candidates, normalized];
    for (const source of new Set(sources)) {
        let value: JSONValue;
        let repaired = false;
        let parseError: unknown;
        try {
            value = JSON.parse(source);
        } catch (error: unknown) {
            parseError = error;
            lastError = error;
            if (options.allowRepair === false) continue;
            try {
                value = repairJSON(source, parseError);
                repaired = true;
            } catch (error: unknown) {
                lastError = error;
                continue;
            }
        }
        // A repaired wrapper must not turn an object answer plus trailing prose into an array of unrelated values.
        if (
            source === normalized &&
            normalized.startsWith('{') &&
            Array.isArray(value) &&
            candidates.some((c) => c !== normalized)
        ) {
            continue;
        }
        options.onDiagnostic?.({
            extracted: source !== normalized,
            repaired,
            original_text: text,
            ...(repaired ? { parse_error: errorMessage(parseError) } : {}),
        });
        return value;
    }
    throw lastError;
}

export function extractAndParseJSON(text: string, allowRepair = true): JSONValue {
    return parseJSONOutput(text, { allowRepair });
}
