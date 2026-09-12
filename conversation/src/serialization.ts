import {
    boundConversationDiagnostic,
    ConversationValidationError,
    type ConversationValidationResult,
} from './diagnostics.js';
import { DEFAULT_JSON_INPUT_LIMITS } from './json-preflight.js';
import type { ConversationDocument } from './types.js';
import {
    type ConversationValidationOptions,
    parseConversationDocument,
    validateConversationDocument,
} from './validation.js';

function materializedByteLimit(options: ConversationValidationOptions): number {
    const limit = options.json_input_limits?.max_bytes ?? DEFAULT_JSON_INPUT_LIMITS.max_bytes;
    if (!Number.isSafeInteger(limit) || limit < 1) {
        throw new RangeError('max_bytes must be a positive safe integer');
    }
    return limit;
}

function jsonTextBytesWithinLimit(text: string, limit: number): number | undefined {
    // Every UTF-16 code unit occupies at least one UTF-8 byte. Avoid encoding a text that already
    // exceeds the limit by this cheap lower bound.
    if (text.length > limit) {
        return undefined;
    }
    const bytes = new TextEncoder().encode(text).byteLength;
    return bytes <= limit ? bytes : undefined;
}

export function parseConversationJson(
    text: string,
    options: ConversationValidationOptions = {},
): ConversationValidationResult<ConversationDocument> {
    const limit = materializedByteLimit(options);
    const bytes = jsonTextBytesWithinLimit(text, limit);
    if (bytes === undefined) {
        return {
            success: false,
            diagnostics: [
                boundConversationDiagnostic({
                    code: 'JSON_MAX_BYTES',
                    stage: 'preflight',
                    path: '',
                    message: 'Materialized conversation JSON exceeds the configured byte limit',
                    limit,
                    observed: Math.min(Number.MAX_SAFE_INTEGER, Math.max(limit + 1, text.length)),
                }),
            ],
        };
    }

    let input: unknown;
    try {
        input = JSON.parse(text);
    } catch {
        return {
            success: false,
            diagnostics: [
                boundConversationDiagnostic({
                    code: 'SCHEMA_INVALID',
                    stage: 'schema',
                    path: '',
                    message: 'Conversation input is not valid JSON text',
                }),
            ],
        };
    }
    return validateConversationDocument(input, options);
}

export function conversationDocumentFromJson(
    text: string,
    options: ConversationValidationOptions = {},
): ConversationDocument {
    const result = parseConversationJson(text, options);
    if (!result.success) {
        throw new ConversationValidationError('Conversation JSON validation failed', result.diagnostics);
    }
    return result.data;
}

export function conversationDocumentToJson(input: unknown, options: ConversationValidationOptions = {}): string {
    const document = parseConversationDocument(input, options);
    const text = JSON.stringify(document);
    const limit = materializedByteLimit(options);
    if (jsonTextBytesWithinLimit(text, limit) === undefined) {
        throw new ConversationValidationError('Serialized conversation exceeds the configured byte limit', [
            boundConversationDiagnostic({
                code: 'JSON_MAX_BYTES',
                stage: 'preflight',
                path: '',
                message: 'Serialized conversation exceeds the configured byte limit',
                limit,
                observed: Math.min(Number.MAX_SAFE_INTEGER, Math.max(limit + 1, text.length)),
            }),
        ]);
    }
    return text;
}
