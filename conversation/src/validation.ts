import type { z } from 'zod';
import { boundConversationDiagnostic, ConversationValidationError, diagnosticPointer } from './diagnostics.js';
import { type JsonInputLimits, preflightJsonInput } from './json-preflight.js';
import { ConversationDocumentSchema } from './schemas/document.js';
import { validateConversationSemantics } from './semantic-validation.js';
import type { ConversationDiagnostic, ConversationDocument } from './types.js';

export interface ConversationValidationOptions {
    json_input_limits?: Partial<JsonInputLimits>;
}

const MAX_SCHEMA_DIAGNOSTICS = 32;

function schemaIssuePath(issue: z.core.$ZodIssue): string {
    return diagnosticPointer(
        issue.path.map((segment) => (typeof segment === 'symbol' ? (segment.description ?? 'symbol') : segment)),
    );
}

export function diagnosticsFromZodError(error: z.ZodError): ConversationDiagnostic[] {
    return error.issues.slice(0, MAX_SCHEMA_DIAGNOSTICS).map((issue) =>
        boundConversationDiagnostic({
            code: 'SCHEMA_INVALID',
            stage: 'schema',
            path: schemaIssuePath(issue),
            // Zod messages for unrecognized keys may embed every untrusted key. The public diagnostic
            // reports the authoritative issue code while the path identifies the affected value.
            message: `Conversation schema validation failed with ${issue.code}`,
        }),
    );
}

function cloneValidatedDocument(input: unknown): ConversationDocument {
    // Shape validation above proves this is a JSON-only document. Cloning the original, rather than
    // returning Zod's reconstructed record output, preserves every validated own JSON key and value.
    return structuredClone(input) as ConversationDocument;
}

export function validateConversationDocument(
    input: unknown,
    options: ConversationValidationOptions = {},
): import('./diagnostics.js').ConversationValidationResult<ConversationDocument> {
    const preflight = preflightJsonInput(input, options.json_input_limits);
    if (!preflight.success) {
        return { success: false, diagnostics: preflight.diagnostics };
    }

    const shape = ConversationDocumentSchema.safeParse(input);
    if (!shape.success) {
        return { success: false, diagnostics: diagnosticsFromZodError(shape.error) };
    }

    const document = cloneValidatedDocument(input);
    const diagnostics = validateConversationSemantics(document);
    if (diagnostics.length > 0) {
        return { success: false, diagnostics };
    }
    return { success: true, data: document, diagnostics: [] };
}

export function parseConversationDocument(
    input: unknown,
    options: ConversationValidationOptions = {},
): ConversationDocument {
    const result = validateConversationDocument(input, options);
    if (!result.success) {
        throw new ConversationValidationError('Conversation document validation failed', result.diagnostics);
    }
    return result.data;
}
