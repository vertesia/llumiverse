import { z } from 'zod';
import { ProcessingBudgetSchema, ProcessorConfigurationSchema } from './processing-policy-foundation.js';

export { ProcessingBudgetSchema, ProcessorConfigurationSchema } from './processing-policy-foundation.js';

import { AssetSchema, ConversationTurnSchema, ToolDefinitionSchema } from './content.js';
import {
    CompactionStrategySchema,
    ContextEntrySchema,
    ContextRetrievalRequirementSchema,
} from './context-foundation.js';
import { ConversationDeletedTurnSchema } from './conversation-delete-operation.js';
import { ExecutionReceiptSchema, GenerationSchema, OperationReceiptSchema } from './execution.js';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    MetadataSchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';
import {
    ProcessingAttemptReceiptSchema,
    ProcessingCompletionReceiptSchema,
    ProcessingJobSchema,
    ProcessingOutputReceiptSchema,
    ProcessingReadinessCoverageSchema,
    ProcessingResolvedInputSchema,
    ProcessingSupersessionReceiptSchema,
} from './processing.js';

export {
    CompactionStrategySchema,
    ContextEntrySchema,
    ContextRetrievalRequirementSchema,
    ReplacementTurnContextEntrySchema,
    SourceTurnContextEntrySchema,
} from './context-foundation.js';

export const CacheIntentSchema = z
    .strictObject({
        namespace: IdentifierSchema,
        mode: z.enum(['auto', 'off', 'required']),
        stable_through_entry_id: IdentifierSchema.optional(),
        ttl_seconds: PositiveSafeIntegerSchema.optional(),
    })
    .meta({ id: 'ConversationCacheIntent' });

export const ConversationContextSchema = z
    .strictObject({
        revision: NonnegativeSafeIntegerSchema,
        entries: z.array(ContextEntrySchema),
        active_tool_definition_ids: z.array(IdentifierSchema),
        protected_entry_ids: z.array(IdentifierSchema),
        retrieval_requirements: z.array(ContextRetrievalRequirementSchema),
        cache_intent: CacheIntentSchema.optional(),
    })
    .meta({ id: 'ConversationContext' });

export const CompactionSourceSchema = z
    .strictObject({
        turn_ids: z.array(IdentifierSchema).min(1),
        block_ids: z.array(IdentifierSchema).min(1).optional(),
        source_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationCompactionSource' });

export const CompactionRecordSchema = z
    .strictObject({
        id: IdentifierSchema,
        operation_id: IdentifierSchema,
        strategy: CompactionStrategySchema,
        source: CompactionSourceSchema,
        replacement_turns: z.array(ConversationTurnSchema).min(1),
        /** Exact accepted pre-transform context; recovery must not replay later active selections. */
        original_context: ConversationContextSchema.optional(),
        fidelity: z.enum(['value_preserving', 'reversible_representation', 'heuristic', 'semantic', 'retrievable']),
        retained_asset_ids: z.array(IdentifierSchema),
        generation_ids: z.array(IdentifierSchema),
        derivation_generation: GenerationSchema.optional(),
        supersedes_compaction_id: IdentifierSchema.optional(),
        created_at: TimestampSchema,
        metadata: MetadataSchema.optional(),
    })
    .meta({ id: 'ConversationCompactionRecord' });

export const ProcessingStateSchema = z
    .strictObject({
        enabled: z.boolean(),
        policy_revision: NonnegativeSafeIntegerSchema,
        processors: z.array(ProcessorConfigurationSchema),
        budget: ProcessingBudgetSchema.optional(),
        jobs: z.record(IdentifierSchema, ProcessingJobSchema).optional(),
        resolved_inputs: z.record(IdentifierSchema, ProcessingResolvedInputSchema).optional(),
        attempts: z.record(IdentifierSchema, ProcessingAttemptReceiptSchema).optional(),
        outputs: z.record(IdentifierSchema, ProcessingOutputReceiptSchema).optional(),
        completions: z.record(IdentifierSchema, ProcessingCompletionReceiptSchema).optional(),
        supersessions: z.record(IdentifierSchema, ProcessingSupersessionReceiptSchema).optional(),
        coverage: ProcessingReadinessCoverageSchema.optional(),
        coverage_receipts: z.record(IdentifierSchema, ProcessingReadinessCoverageSchema).optional(),
    })
    .meta({ id: 'ConversationProcessingState' });

export const ConversationLineageParentSchema = z
    .strictObject({
        relation: z.enum(['fork', 'merge']),
        source: ConversationRefSchema,
    })
    .meta({ id: 'ConversationLineageParent' });

export const ConversationLineageSchema = z
    .strictObject({
        parents: z.array(ConversationLineageParentSchema).min(1),
    })
    .meta({ id: 'ConversationLineage' });

export const ConversationDocumentSchema = z
    .strictObject({
        format: z.literal(CONVERSATION_FORMAT),
        schema_version: z.literal(CONVERSATION_SCHEMA_VERSION),
        experimental_revision: z.literal(CONVERSATION_EXPERIMENTAL_REVISION),
        id: IdentifierSchema,
        revision: NonnegativeSafeIntegerSchema,
        created_at: TimestampSchema,
        updated_at: TimestampSchema,
        turns: z.array(ConversationTurnSchema),
        deleted_turns: z.record(IdentifierSchema, ConversationDeletedTurnSchema).optional(),
        generations: z.record(IdentifierSchema, GenerationSchema),
        operation_receipts: z.record(IdentifierSchema, OperationReceiptSchema),
        execution_receipts: z.record(IdentifierSchema, ExecutionReceiptSchema),
        assets: z.record(IdentifierSchema, AssetSchema),
        tool_definitions: z.record(IdentifierSchema, ToolDefinitionSchema),
        context: ConversationContextSchema,
        compactions: z.record(IdentifierSchema, CompactionRecordSchema),
        processing: ProcessingStateSchema,
        lineage: ConversationLineageSchema.optional(),
        metadata: MetadataSchema.optional(),
    })
    .meta({ id: 'ConversationDocumentV0' });

export const ProcessingRunResultSchema = z
    .strictObject({ status: z.enum(['completed', 'in_progress', 'superseded']), document: ConversationDocumentSchema })
    .meta({ id: 'ConversationProcessingRunResult' });
