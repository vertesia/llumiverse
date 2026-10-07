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

function extractCandidate(text: string): string {
    const fence = text.match(/^```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```$/i);
    if (fence) {
        if (/(?:^|\r?\n)```(?:json)?[ \t]*(?:\r?\n|$)/i.test(fence[1]))
            throw new SyntaxError('Ambiguous JSON output: multiple code blocks');
        return fence[1].trim();
    }

    // Preserve the historical prose extraction, extended to arrays. Do not search inside a root scalar or comment.
    if (/^["'\d/-]|^(?:true|false|null)\b/.test(text)) return text;
    const start = text.search(/[[{]/);
    const end = Math.max(text.lastIndexOf('}'), text.lastIndexOf(']')) + 1;
    if (start < 0 || end <= start) return text;
    const suffix = text
        .slice(end)
        .trim()
        .replace(/^```(?:\s|$)/, '')
        .trim();
    if (/[[{]/.test(suffix) || /^["'\d-]|^(?:true|false|null)\b/.test(suffix) || suffix.includes('```')) {
        return text;
    }
    return text.slice(start, end);
}

/** Strict parsing first, then optional jsonrepair recovery. Callers validate the recovered value against their schema. */
export function parseJSONOutput(text: string, options: JSONOutputParseOptions = {}): JSONValue {
    let parseError: unknown;
    try {
        return JSON.parse(text);
    } catch (error: unknown) {
        parseError = error;
    }

    const normalized = text.trim();
    const candidate = extractCandidate(normalized);
    const extracted = candidate !== normalized;
    let value: JSONValue | undefined;
    if (extracted) {
        try {
            value = JSON.parse(candidate);
        } catch (error: unknown) {
            parseError = error;
        }
        if (value !== undefined) {
            options.onDiagnostic?.({ extracted: true, repaired: false, original_text: text });
            return value;
        }
    }
    if (options.allowRepair === false) throw parseError;

    let repaired: JSONValue;
    let repairedExtraction = false;
    try {
        repaired = repairJSON(normalized, parseError);
    } catch (error: unknown) {
        if (!extracted) throw error;
        repaired = repairJSON(candidate, parseError);
        repairedExtraction = true;
    }
    options.onDiagnostic?.({
        extracted: repairedExtraction,
        repaired: true,
        original_text: text,
        parse_error: errorMessage(parseError),
    });
    return repaired;
}

export function extractAndParseJSON(text: string, allowRepair = true): JSONValue {
    return parseJSONOutput(text, { allowRepair });
}
