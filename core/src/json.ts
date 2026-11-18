import type { JSONValue } from '@llumiverse/common';
import { jsonrepair } from 'jsonrepair';

/** Tags around the prompt copy of a result schema; drivers build the instruction from these. */
export const RESULT_SCHEMA_OPEN_TAG = '<response_schema>';
export const RESULT_SCHEMA_CLOSE_TAG = '</response_schema>';

/**
 * The answer in a reply to the prompt copy of a result schema, which asks for JSON "between the <response_schema>
 * tags". Models sometimes echo the schema block before the answer (`<response_schema>{schema}</response_schema>
 * {answer}`), so taking the first `{` to the last `}` spans both objects; the answer follows the last closing tag.
 * A reply that wraps its answer in the tags already parses without this.
 */
function textAfterEchoedResultSchema(text: string): string | undefined {
    const close = text.lastIndexOf(RESULT_SCHEMA_CLOSE_TAG);
    if (close === -1) return undefined;
    const after = text.slice(close + RESULT_SCHEMA_CLOSE_TAG.length);
    return after.includes('{') ? after : undefined;
}

function extractJsonFromText(text: string): string {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    return text.substring(start, end + 1);
}

export function extractAndParseJSON(text: string): JSONValue {
    try {
        return parseJSON(extractJsonFromText(text));
    } catch (error: unknown) {
        // Only a reply that fails to parse is read again past an echoed schema block, so every reply that parses
        // today is unchanged.
        const answer = textAfterEchoedResultSchema(text);
        if (answer === undefined) throw error;
        try {
            return parseJSON(extractJsonFromText(answer));
        } catch {
            // The original error describes the whole reply, which is what the caller sees.
            throw error;
        }
    }
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
