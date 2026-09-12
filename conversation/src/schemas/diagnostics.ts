import { z } from 'zod';
import { NonnegativeSafeIntegerSchema } from './primitives.js';

export const DIAGNOSTIC_PATH_MAX_LENGTH = 512 as const;
export const DIAGNOSTIC_MESSAGE_MAX_LENGTH = 1_024 as const;
export const DIAGNOSTIC_RECORD_ID_MAX_LENGTH = 512 as const;
export const DIAGNOSTIC_RELATED_PATHS_MAX_LENGTH = 8 as const;

export const JsonPreflightDiagnosticCodeSchema = z
    .enum([
        'JSON_ACCESSOR_PROPERTY',
        'JSON_ARRAY_PROPERTY',
        'JSON_CYCLE',
        'JSON_MAX_ARRAY_LENGTH',
        'JSON_MAX_BYTES',
        'JSON_MAX_DEPTH',
        'JSON_MAX_NODES',
        'JSON_MAX_OBJECT_PROPERTIES',
        'JSON_MAX_STRING_BYTES',
        'JSON_NEGATIVE_ZERO',
        'JSON_NON_ENUMERABLE_PROPERTY',
        'JSON_NON_FINITE_NUMBER',
        'JSON_NON_PLAIN_OBJECT',
        'JSON_RESERVED_PROPERTY_KEY',
        'JSON_SPARSE_ARRAY',
        'JSON_SYMBOL_KEY',
        'JSON_UNSUPPORTED_TYPE',
    ])
    .meta({ id: 'ConversationJsonPreflightDiagnosticCode' });

export const SemanticConversationDiagnosticCodeSchema = z
    .enum([
        'ACCOUNTING_PROVENANCE_MISMATCH',
        'ASSET_KIND_MISMATCH',
        'CONTEXT_BLOCK_ORDER_INVALID',
        'CONTEXT_DIRECT_REPLACEMENT_OVERLAP',
        'CONTEXT_REVISION_INVALID',
        'CONTEXT_SELECTION_OVERLAP',
        'DERIVED_AUTHORITY_MIXED',
        'DERIVED_AUTHORITY_PROMOTED',
        'DERIVED_PROVENANCE_CYCLE',
        'DERIVED_PROVENANCE_MISMATCH',
        'DUPLICATE_ID',
        'DUPLICATE_TERMINAL_RESULT',
        'GENERATION_REQUEST_MISMATCH',
        'GENERATION_SOURCE_INVALID',
        'GENERATION_USAGE_EMPTY',
        'INPUT_PARTITION_INVALID',
        'MAP_KEY_ID_MISMATCH',
        'PARENT_TURN_CYCLE',
        'REFERENCE_NOT_FOUND',
        'REQUEST_MAPPING_INVALID',
        'SELECTION_RANGE_INVALID',
        'SEMANTIC_DIAGNOSTIC_LIMIT',
        'SUPERSEDES_CYCLE',
        'TIMESTAMP_ORDER_INVALID',
        'TOOL_DEFINITION_MISMATCH',
        'TOOL_RECEIPT_MISMATCH',
        'TOOL_RESULT_UNRESOLVED',
        'USAGE_BREAKDOWN_INVALID',
        'USAGE_OVERFLOW',
        'USAGE_TOTAL_INVALID',
    ])
    .meta({ id: 'ConversationSemanticDiagnosticCode' });

export const ConversationDiagnosticStageSchema = z
    .enum(['preflight', 'schema', 'semantic'])
    .meta({ id: 'ConversationDiagnosticStage' });

export const ConversationDiagnosticCodeSchema = z
    .union([JsonPreflightDiagnosticCodeSchema, z.literal('SCHEMA_INVALID'), SemanticConversationDiagnosticCodeSchema])
    .meta({ id: 'ConversationDiagnosticCode' });

const diagnosticDetailShape = {
    path: z.string().max(DIAGNOSTIC_PATH_MAX_LENGTH),
    message: z.string().max(DIAGNOSTIC_MESSAGE_MAX_LENGTH),
    record_id: z.string().max(DIAGNOSTIC_RECORD_ID_MAX_LENGTH).optional(),
    related_paths: z
        .array(z.string().max(DIAGNOSTIC_PATH_MAX_LENGTH))
        .max(DIAGNOSTIC_RELATED_PATHS_MAX_LENGTH)
        .optional(),
    limit: NonnegativeSafeIntegerSchema.optional(),
    observed: NonnegativeSafeIntegerSchema.optional(),
};

export const JsonPreflightDiagnosticSchema = z
    .strictObject({
        code: JsonPreflightDiagnosticCodeSchema,
        stage: z.literal(ConversationDiagnosticStageSchema.enum.preflight),
        ...diagnosticDetailShape,
    })
    .meta({ id: 'ConversationJsonPreflightDiagnostic' });

export const ConversationSchemaDiagnosticSchema = z
    .strictObject({
        code: z.literal('SCHEMA_INVALID'),
        stage: z.literal(ConversationDiagnosticStageSchema.enum.schema),
        ...diagnosticDetailShape,
    })
    .meta({ id: 'ConversationSchemaDiagnostic' });

export const SemanticConversationDiagnosticSchema = z
    .strictObject({
        code: SemanticConversationDiagnosticCodeSchema,
        stage: z.literal(ConversationDiagnosticStageSchema.enum.semantic),
        ...diagnosticDetailShape,
    })
    .meta({ id: 'ConversationSemanticDiagnostic' });

export const ConversationDiagnosticSchema = z
    .discriminatedUnion('stage', [
        JsonPreflightDiagnosticSchema,
        ConversationSchemaDiagnosticSchema,
        SemanticConversationDiagnosticSchema,
    ])
    .meta({ id: 'ConversationDiagnostic' });
