import { z } from 'zod';
import { AssetSchema, ConversationTurnSchema, RetrievalCapabilitySchema, ToolDefinitionSchema } from './content.js';
import { ExecutionReceiptSchema, GenerationSchema, OperationReceiptSchema } from './execution.js';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    JsonObjectSchema,
    MetadataSchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';

export const SourceTurnContextEntrySchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('source_turn'),
        turn_id: IdentifierSchema,
        block_ids: z.array(IdentifierSchema).min(1).optional(),
    })
    .meta({ id: 'ConversationSourceTurnContextEntry' });

export const ReplacementTurnContextEntrySchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('replacement_turn'),
        compaction_id: IdentifierSchema,
        turn_id: IdentifierSchema,
        block_ids: z.array(IdentifierSchema).min(1).optional(),
    })
    .meta({ id: 'ConversationReplacementTurnContextEntry' });

export const ContextEntrySchema = z
    .discriminatedUnion('type', [SourceTurnContextEntrySchema, ReplacementTurnContextEntrySchema])
    .meta({ id: 'ConversationContextEntry' });

export const ContextRetrievalRequirementSchema = z
    .strictObject({
        id: IdentifierSchema,
        asset_id: IdentifierSchema,
        retrieval: RetrievalCapabilitySchema,
    })
    .meta({ id: 'ConversationContextRetrievalRequirement' });

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

export const CompactionStrategySchema = z
    .strictObject({
        id: IdentifierSchema,
        version: IdentifierSchema,
        configuration_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationCompactionStrategy' });

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
        fidelity: z.enum(['value_preserving', 'reversible_representation', 'heuristic', 'semantic', 'retrievable']),
        retained_asset_ids: z.array(IdentifierSchema),
        generation_ids: z.array(IdentifierSchema),
        supersedes_compaction_id: IdentifierSchema.optional(),
        created_at: TimestampSchema,
        metadata: MetadataSchema.optional(),
    })
    .meta({ id: 'ConversationCompactionRecord' });

export const ProcessorConfigurationSchema = z
    .strictObject({
        id: IdentifierSchema,
        version: IdentifierSchema,
        scope: z.enum(['on_append', 'on_budget', 'manual']),
        config: JsonObjectSchema,
        required: z.boolean(),
        failure_behavior: z.enum(['block', 'skip_with_diagnostic']),
    })
    .meta({ id: 'ConversationProcessorConfiguration' });

// This record is inert persisted configuration in the foundation revision. It deliberately has no
// readiness boolean, job runner, callback, or executable plugin instance.
export const ProcessingStateSchema = z
    .strictObject({
        enabled: z.boolean(),
        policy_revision: NonnegativeSafeIntegerSchema,
        processors: z.array(ProcessorConfigurationSchema),
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
