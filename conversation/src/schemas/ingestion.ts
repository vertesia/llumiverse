import { z } from 'zod';
import { AssetSchema, ConversationTurnSchema, ToolDefinitionSchema } from './content.js';
import { ConversationDiagnosticSchema } from './diagnostics.js';
import { ContextEntrySchema, ConversationDocumentSchema } from './document.js';
import { ExecutedGenerationSchema, ExecutionReceiptSchema, GenerationSchema } from './execution.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema, TimestampSchema } from './primitives.js';

/** Concrete JSON additions; cross-record integrity is checked against the resulting document. */
export const ConversationRecordBatchSchema = z
    .strictObject({
        turns: z.array(ConversationTurnSchema).readonly().optional(),
        generations: z.array(GenerationSchema).readonly().optional(),
        assets: z.array(AssetSchema).readonly().optional(),
        tool_definitions: z.array(ToolDefinitionSchema).readonly().optional(),
        execution_receipts: z.array(ExecutionReceiptSchema).readonly().optional(),
        context_entries: z.array(ContextEntrySchema).readonly().optional(),
        active_tool_definition_ids: z.array(IdentifierSchema).readonly().optional(),
    })
    .meta({ id: 'ConversationRecordBatch' });

export const AppendConversationRecordsOptionsSchema = z
    .strictObject({
        expected_revision: NonnegativeSafeIntegerSchema,
        operation_id: IdentifierSchema,
        payload_fingerprint: ContentHashSchema,
        recorded_at: TimestampSchema,
    })
    .meta({ id: 'AppendConversationRecordsOptions' });

export const AppendConversationRecordsResultSchema = z
    .strictObject({
        document: ConversationDocumentSchema,
        applied: z.boolean(),
        accepted_turn_ids: z.array(IdentifierSchema),
        accepted_generation_ids: z.array(IdentifierSchema),
    })
    .meta({ id: 'AppendConversationRecordsResult' });

/** Completed decoder output, before the request/response identity and dependency checks. */
export const DecodedConversationResponseSchema = z
    .strictObject({
        turns: z.array(ConversationTurnSchema).readonly(),
        generation: ExecutedGenerationSchema,
        assets: z.array(AssetSchema).readonly().optional(),
        execution_receipts: z.array(ExecutionReceiptSchema).readonly().optional(),
        diagnostics: z.array(ConversationDiagnosticSchema),
        payload_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'DecodedConversationResponse' });
