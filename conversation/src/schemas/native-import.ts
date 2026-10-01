import { z } from 'zod';
import { DIAGNOSTIC_MESSAGE_MAX_LENGTH } from '../runtime-constants.js';
import { ToolDefinitionSchema } from './content.js';
import { ConversationDocumentSchema } from './document.js';
import { IdentifierSchema, TimestampSchema } from './primitives.js';

export const NativeConversationImportCompletenessSchema = z
    .enum(['complete', 'fragment', 'unknown'])
    .meta({ id: 'NativeConversationImportCompleteness' });

/** Recorded archive provenance; a native protocol does not establish its hosting provider. */
export const NativeConversationImportOptionsSchema = z
    .strictObject({
        conversation_id: IdentifierSchema,
        recorded_at: TimestampSchema,
        provider: IdentifierSchema,
        model: IdentifierSchema.optional(),
        source_request_id: IdentifierSchema.optional(),
        tool_definitions: z.array(ToolDefinitionSchema).readonly().optional(),
        completeness: NativeConversationImportCompletenessSchema.optional(),
    })
    .meta({ id: 'NativeConversationImportOptions' });

export const NativeConversationImportDiagnosticCodeSchema = z
    .enum([
        'IMPORT_HISTORY_INCOMPLETE',
        'IMPORT_METADATA_MISSING',
        'IMPORT_PROTECTED_REPLAY_MODEL_UNKNOWN',
        'IMPORT_EXTERNAL_ASSET_UNRESOLVED',
        'IMPORT_TOOL_DEFINITION_MISSING',
        'IMPORT_CONTINUATION_NOT_VALIDATED',
        'IMPORT_BYTE_VIEW_PROPERTIES_EXCLUDED',
    ])
    .meta({ id: 'NativeConversationImportDiagnosticCode' });

export const NativeConversationImportDiagnosticSchema = z
    .strictObject({
        code: NativeConversationImportDiagnosticCodeSchema,
        message: z.string().min(1).max(DIAGNOSTIC_MESSAGE_MAX_LENGTH),
        entity_ids: z.array(IdentifierSchema).optional(),
    })
    .meta({ id: 'NativeConversationImportDiagnostic' });

/** Completeness is a source declaration; import alone never certifies target continuation readiness. */
export const NativeConversationImportReportSchema = z
    .strictObject({
        protocol: IdentifierSchema,
        adapter_version: IdentifierSchema,
        completeness: NativeConversationImportCompletenessSchema,
        readiness: z.literal('not_validated'),
        diagnostics: z.array(NativeConversationImportDiagnosticSchema),
    })
    .meta({ id: 'NativeConversationImportReport' });

export const NativeConversationImportResultSchema = z
    .strictObject({
        document: ConversationDocumentSchema,
        report: NativeConversationImportReportSchema,
    })
    .meta({ id: 'NativeConversationImportResult' });
