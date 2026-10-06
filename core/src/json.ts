import type { JSONOutputDiagnostic, JSONValue } from '@llumiverse/common';
import { parseExpressionAt, tokenizer, tokTypes } from 'acorn';
import { jsonrepair } from 'jsonrepair';

export interface JSONOutputParseOptions {
    /** Disable syntax repair. Explicit JSON fences and a single value in prose may still be extracted. */
    allowRepair?: boolean;
    onDiagnostic?: (diagnostic: JSONOutputDiagnostic) => void;
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function repairJSON(text: string, parseError: unknown): { text: string; value: JSONValue } {
    try {
        const repaired = jsonrepair(text);
        return { text: repaired, value: JSON.parse(repaired) };
    } catch (repairError: unknown) {
        throw new SyntaxError(`JSON repair failed: ${errorMessage(repairError)}`, {
            cause: new AggregateError([parseError, repairError], 'JSON parsing and repair failed'),
        });
    }
}

/** General-purpose, permissive repair utility. May complete truncated documents; use parseJSONOutput for model output. */
export function parseJSON(text: string, allowRepair = true): JSONValue {
    try {
        return JSON.parse(text);
    } catch (parseError: unknown) {
        if (!allowRepair) throw parseError;
        return repairJSON(text, parseError).value;
    }
}

const syntaxOptions = { ecmaVersion: 2020 } as const;

function extractCandidate(text: string): string {
    // Only an explicit, complete JSON fence is treated as a wrapper. Multiple fences are ambiguous.
    const fence = text.match(/^```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```$/i);
    if (fence && !fence[1].includes('```')) return fence[1].trim();

    // Retain the historical single object/array in prose fallback, but let a parser find its end.
    // Never search inside a root string, number, keyword, or comment.
    if (/^[{["'\d/-]|^(?:true|false|null)\b/.test(text)) return text;
    const start = text.search(/[[{]/);
    if (start < 0) return text;
    const expression = parseExpressionAt(text, start, syntaxOptions);
    if (expression.type !== 'ObjectExpression' && expression.type !== 'ArrayExpression') return text;
    const suffix = text.slice(expression.end).trim();
    // Prose must not hide a second value, a code fence, or an unmatched delimiter.
    if (!/^[\p{L}\p{M}\s.,!?;:-]*$/u.test(suffix) || /^(?:true|false|null)\b/.test(suffix)) {
        throw new SyntaxError('Ambiguous JSON output: unexpected content after the JSON value');
    }
    return text.slice(start, expression.end);
}

/**
 * Tokenize without evaluating JavaScript. Ignore only commas/colons; every value and container
 * boundary must survive repair. Acorn owns quote, escape, comment, and numeric lexing.
 * This deliberately rejects ambiguous strings and expressions rather than guessing their meaning.
 */
function contentTokens(text: string): string[] {
    const tokens: string[] = [];
    for (const token of tokenizer(text, syntaxOptions)) {
        const type = token.type;
        const value = 'value' in token ? token.value : undefined;
        if (type === tokTypes.comma || type === tokTypes.colon) continue;
        if (type === tokTypes.string || type === tokTypes.name) {
            tokens.push(`text:${value}`);
        } else if (type === tokTypes.num && typeof value === 'number') {
            tokens.push(`number:${value}`);
        } else if (
            type === tokTypes.braceL ||
            type === tokTypes.braceR ||
            type === tokTypes.bracketL ||
            type === tokTypes.bracketR ||
            type === tokTypes._true ||
            type === tokTypes._false ||
            type === tokTypes._null ||
            (type === tokTypes.plusMin && value === '-')
        ) {
            tokens.push(type === tokTypes.plusMin ? '-' : type.label);
        } else {
            throw new SyntaxError(`Unsupported token in JSON output at position ${token.start}`);
        }
    }
    return tokens;
}

/**
 * Strict parsing first; conservative syntax recovery second. Repair may normalize quotes, comments,
 * commas and colons, but cannot add/drop values or container boundaries. This is not a guarantee
 * of semantic completeness: only the provider/caller can know whether all intended data was generated.
 */
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

    const originalTokens = contentTokens(candidate);
    // A bare identifier is prose, not evidence of a JSON value (or a complete JSON keyword).
    if (originalTokens.length === 0 || /^[\p{L}_$]/u.test(candidate)) throw parseError;
    const repaired = repairJSON(candidate, parseError);
    const repairedTokens = contentTokens(repaired.text);
    if (
        originalTokens.length !== repairedTokens.length ||
        originalTokens.some((token, i) => token !== repairedTokens[i])
    ) {
        throw new SyntaxError('JSON repair would add, remove, or change values or container boundaries', {
            cause: parseError,
        });
    }
    options.onDiagnostic?.({ extracted, repaired: true, original_text: text, parse_error: errorMessage(parseError) });
    return repaired.value;
}

export function extractAndParseJSON(text: string, allowRepair = true): JSONValue {
    return parseJSONOutput(text, { allowRepair });
}
