import type { z } from 'zod';
import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import {
    NativeConversationImportOptionsSchema,
    NativeConversationImportReportSchema,
    NativeConversationImportResultSchema,
} from './schemas/native-import.js';
import type {
    NativeConversationImportOptions,
    NativeConversationImportReport,
    NativeConversationImportResult,
} from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

function parseImportValue<T>(schema: z.ZodType<T>, input: unknown): T {
    const preflight = preflightJsonInput(input);
    if (!preflight.success)
        throw new ConversationValidationError('Native import contract failed JSON preflight', preflight.diagnostics);
    const result = schema.safeParse(input);
    if (!result.success)
        throw new ConversationValidationError(
            'Native import contract failed schema validation',
            diagnosticsFromZodError(result.error),
        );
    // Preserve every own validated JSON key rather than relying on Zod record reconstruction.
    return structuredClone(input) as T;
}

export function parseNativeConversationImportOptions(input: unknown): NativeConversationImportOptions {
    return parseImportValue(NativeConversationImportOptionsSchema, input);
}

export function parseNativeConversationImportReport(input: unknown): NativeConversationImportReport {
    return parseImportValue(NativeConversationImportReportSchema, input);
}

export function parseNativeConversationImportResult(input: unknown): NativeConversationImportResult {
    const result = parseImportValue(NativeConversationImportResultSchema, input);
    return { document: parseConversationDocument(result.document), report: result.report };
}
