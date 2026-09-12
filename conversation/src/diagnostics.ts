import {
    DIAGNOSTIC_MESSAGE_MAX_LENGTH,
    DIAGNOSTIC_PATH_MAX_LENGTH,
    DIAGNOSTIC_RECORD_ID_MAX_LENGTH,
    DIAGNOSTIC_RELATED_PATHS_MAX_LENGTH,
} from './schemas/diagnostics.js';
import type { ConversationDiagnostic, JsonPreflightDiagnostic, SemanticConversationDiagnostic } from './types.js';

const TRUNCATED_SUFFIX = '…';

function truncate(value: string, maximumLength: number): string {
    if (value.length <= maximumLength) {
        return value;
    }
    return `${value.slice(0, maximumLength - TRUNCATED_SUFFIX.length)}${TRUNCATED_SUFFIX}`;
}

/** Bounds untrusted values before interpolating them into a diagnostic message. */
export function diagnosticValue(value: string): string {
    return truncate(value, 128);
}

function appendPointerSegment(pointer: string, segment: string): { pointer: string; truncated: boolean } {
    let next = pointer;
    for (let index = 0; index < segment.length; index += 1) {
        const character = segment[index];
        const encoded = character === '~' ? '~0' : character === '/' ? '~1' : character;
        if (next.length + encoded.length > DIAGNOSTIC_PATH_MAX_LENGTH - TRUNCATED_SUFFIX.length) {
            return {
                pointer: `${next.slice(0, DIAGNOSTIC_PATH_MAX_LENGTH - TRUNCATED_SUFFIX.length)}${TRUNCATED_SUFFIX}`,
                truncated: true,
            };
        }
        next += encoded;
    }
    return { pointer: next, truncated: false };
}

/** Builds a bounded RFC 6901 pointer without first copying an untrusted full key. */
export function diagnosticPointer(segments: readonly (string | number)[]): string {
    let pointer = '';
    for (const segment of segments) {
        if (pointer.length + 1 > DIAGNOSTIC_PATH_MAX_LENGTH - TRUNCATED_SUFFIX.length) {
            return `${pointer.slice(0, DIAGNOSTIC_PATH_MAX_LENGTH - TRUNCATED_SUFFIX.length)}${TRUNCATED_SUFFIX}`;
        }
        pointer += '/';
        const appended = appendPointerSegment(pointer, String(segment));
        pointer = appended.pointer;
        if (appended.truncated) {
            return pointer;
        }
    }
    return truncate(pointer, DIAGNOSTIC_PATH_MAX_LENGTH);
}

/**
 * Applies the public diagnostic payload limits without modifying the underlying conversation value.
 * All validation stages use this final projection before returning diagnostics to a caller.
 */
export function boundConversationDiagnostic(diagnostic: JsonPreflightDiagnostic): JsonPreflightDiagnostic;
export function boundConversationDiagnostic(diagnostic: SemanticConversationDiagnostic): SemanticConversationDiagnostic;
export function boundConversationDiagnostic(diagnostic: ConversationDiagnostic): ConversationDiagnostic;
export function boundConversationDiagnostic(diagnostic: ConversationDiagnostic): ConversationDiagnostic {
    const relatedPaths = diagnostic.related_paths
        ?.slice(0, DIAGNOSTIC_RELATED_PATHS_MAX_LENGTH)
        .map((path) => truncate(path, DIAGNOSTIC_PATH_MAX_LENGTH));
    return {
        ...diagnostic,
        path: truncate(diagnostic.path, DIAGNOSTIC_PATH_MAX_LENGTH),
        message: truncate(diagnostic.message, DIAGNOSTIC_MESSAGE_MAX_LENGTH),
        record_id:
            diagnostic.record_id === undefined
                ? undefined
                : truncate(diagnostic.record_id, DIAGNOSTIC_RECORD_ID_MAX_LENGTH),
        related_paths: relatedPaths,
        limit: diagnostic.limit === undefined ? undefined : Math.min(diagnostic.limit, Number.MAX_SAFE_INTEGER),
        observed:
            diagnostic.observed === undefined ? undefined : Math.min(diagnostic.observed, Number.MAX_SAFE_INTEGER),
    };
}

export interface ConversationValidationSuccess<T> {
    success: true;
    data: T;
    diagnostics: [];
}

export interface ConversationValidationFailure {
    success: false;
    diagnostics: ConversationDiagnostic[];
}

export type ConversationValidationResult<T> = ConversationValidationSuccess<T> | ConversationValidationFailure;

export class ConversationValidationError extends Error {
    public readonly diagnostics: ConversationDiagnostic[];

    public constructor(message: string, diagnostics: ConversationDiagnostic[], options?: ErrorOptions) {
        super(message, options);
        this.name = 'ConversationValidationError';
        this.diagnostics = diagnostics;
    }
}
