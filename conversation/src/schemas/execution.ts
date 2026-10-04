import { z } from 'zod';
import { CONVERSATION_USAGE_METRICS } from '../runtime-constants.js';
import { ContextEntrySchema } from './context-foundation.js';
import { ContextMeasurementSchema } from './context-measurement.js';
import { ConversationDeleteOperationSchema } from './conversation-delete-operation.js';
import { ConversationEditOperationSchema } from './conversation-edit-operation.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    JsonObjectSchema,
    JsonValueSchema,
    MetadataSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';
import { ProcessingOperationSchema } from './processing-operation.js';

/** Proof that the current canonical head already contains this request's complete model-visible input. */
export const ConversationMaterializedInputSchema = z
    .strictObject({
        operation_id: IdentifierSchema,
        result_revision: NonnegativeSafeIntegerSchema,
    })
    .meta({ id: 'ConversationMaterializedInput' });

export const ConversationRuntimeContextSchema = z
    .strictObject({
        conversation_id: IdentifierSchema.optional(),
        request_id: IdentifierSchema,
        attempt_id: IdentifierSchema,
        input_operation_id: IdentifierSchema,
        response_operation_id: IdentifierSchema,
        recorded_at: TimestampSchema,
        started_at: TimestampSchema.optional(),
        completed_at: TimestampSchema.optional(),
        purpose: IdentifierSchema.optional(),
        materialized_input: ConversationMaterializedInputSchema.optional(),
    })
    .meta({ id: 'ConversationRuntimeContext' });

export const UsageMetricSchema = z.enum(CONVERSATION_USAGE_METRICS).meta({ id: 'ConversationUsageMetric' });

export const AccountingProvenanceSchema = z
    .strictObject({
        method: z.enum(['reported', 'derived']),
        accounting_basis: IdentifierSchema,
        source: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationAccountingProvenance' });

export const UsageAccountingProvenanceSchema = z
    .strictObject({
        input_tokens: AccountingProvenanceSchema.optional(),
        output_tokens: AccountingProvenanceSchema.optional(),
        total_tokens: AccountingProvenanceSchema.optional(),
        reasoning_tokens: AccountingProvenanceSchema.optional(),
        cache_read_tokens: AccountingProvenanceSchema.optional(),
        cache_write_tokens: AccountingProvenanceSchema.optional(),
        input_new_tokens: AccountingProvenanceSchema.optional(),
    })
    .meta({ id: 'ConversationUsageAccountingProvenance' });

export const ReportedUsageSchema = z
    .strictObject({
        source: z.enum(['provider', 'legacy', 'host']),
        protocol: IdentifierSchema.optional(),
        version: IdentifierSchema.optional(),
        accounting_basis: IdentifierSchema.optional(),
        payload: JsonValueSchema,
    })
    .meta({ id: 'ConversationReportedUsage' });

export const CompleteInputPartitionSchema = z
    .strictObject({
        type: z.literal('complete_disjoint'),
        cache_write_bucket: z.enum(['included', 'inapplicable']),
    })
    .meta({ id: 'ConversationCompleteInputPartition' });

export const GenerationCostSchema = z
    .strictObject({
        amount: z.string().regex(/^(?:0|[1-9]\d*)(?:\.\d+)?$/),
        currency: z.string().regex(/^[A-Z]{3}$/),
        provenance: z.enum(['reported', 'estimated']),
        price_source: IdentifierSchema.optional(),
        price_version: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationGenerationCost' });

export const GenerationUsageSchema = z
    .strictObject({
        input_tokens: NonnegativeSafeIntegerSchema.optional(),
        output_tokens: NonnegativeSafeIntegerSchema.optional(),
        total_tokens: NonnegativeSafeIntegerSchema.optional(),
        reasoning_tokens: NonnegativeSafeIntegerSchema.optional(),
        cache_read_tokens: NonnegativeSafeIntegerSchema.optional(),
        cache_write_tokens: NonnegativeSafeIntegerSchema.optional(),
        input_new_tokens: NonnegativeSafeIntegerSchema.optional(),
        accounting_provenance: UsageAccountingProvenanceSchema.optional(),
        input_partition: CompleteInputPartitionSchema.optional(),
        reported_usage: z.array(ReportedUsageSchema).min(1).optional(),
        cost: GenerationCostSchema.optional(),
    })
    .meta({ id: 'ConversationGenerationUsage' });

export { ContextMeasurementSchema } from './context-measurement.js';

export const ModelTargetSchema = z
    .strictObject({
        provider: IdentifierSchema,
        protocol: IdentifierSchema,
        model: IdentifierSchema,
        adapter_version: IdentifierSchema,
        options: JsonObjectSchema.optional(),
    })
    .meta({ id: 'ConversationModelTarget' });

export const AssetVersionBindingSchema = z
    .strictObject({
        asset_id: IdentifierSchema,
        content_hash: ContentHashSchema,
    })
    .meta({ id: 'ConversationAssetVersionBinding' });

export const NativeItemMappingSchema = z
    .strictObject({
        canonical_id: IdentifierSchema,
        native_id: z.string().min(1),
        kind: z.enum(['turn', 'block', 'call']),
    })
    .meta({ id: 'ConversationNativeItemMapping' });

/** Host-owned immutable selected-content locator; the key is opaque, never a fetch URL. */
export const RequestSourceViewReferenceSchema = z
    .strictObject({
        version: z.literal(1),
        completeness: z.enum(['selected_execution', 'selected_content_unverified']),
        source: ConversationRefSchema,
        context_revision: NonnegativeSafeIntegerSchema,
        manifest_storage_key: z.string().min(1).max(512),
        manifest_content_hash: z.string().regex(/^sha256:[0-9a-f]{64}$/),
        manifest_size_bytes: z
            .number()
            .int()
            .positive()
            .max(256 * 1024),
        context_fingerprint: ContentHashSchema,
        request_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationRequestSourceViewReference' });

export const RequestReceiptSchema = z
    .strictObject({
        id: IdentifierSchema,
        request_id: IdentifierSchema,
        attempt_id: IdentifierSchema,
        source: ConversationRefSchema,
        source_tail_turn_id: IdentifierSchema.optional(),
        context_fingerprint: ContentHashSchema,
        tool_set_fingerprint: ContentHashSchema,
        request_fingerprint: ContentHashSchema,
        source_view: RequestSourceViewReferenceSchema.optional(),
        target: ModelTargetSchema,
        tool_definition_ids: z.array(IdentifierSchema),
        asset_versions: z.array(AssetVersionBindingSchema),
        item_mappings: z.array(NativeItemMappingSchema),
        measurement: ContextMeasurementSchema.optional(),
        recorded_at: TimestampSchema,
        metadata: MetadataSchema.optional(),
    })
    .meta({ id: 'ConversationRequestReceipt' });

export const GenerationTimestampsSchema = z
    .strictObject({
        recorded_at: TimestampSchema,
        started_at: TimestampSchema.optional(),
        completed_at: TimestampSchema.optional(),
        provider_duration_ms: z.number().nonnegative().optional(),
    })
    .meta({ id: 'ConversationGenerationTimestamps' });

export const GenerationStatusSchema = z
    .enum(['completed', 'failed', 'cancelled'])
    .meta({ id: 'ConversationGenerationStatus' });

const commonGenerationShape = {
    id: IdentifierSchema,
    request_id: IdentifierSchema,
    attempt_id: IdentifierSchema,
    provider_response_id: z.string().min(1).optional(),
    purpose: IdentifierSchema,
    requested_model: IdentifierSchema,
    resolved_model: IdentifierSchema.optional(),
    provider: IdentifierSchema,
    protocol: IdentifierSchema,
    model_options: JsonObjectSchema.optional(),
    adapter_version: IdentifierSchema,
    status: GenerationStatusSchema,
    finish_reason: z.string().optional(),
    timestamps: GenerationTimestampsSchema,
    source: ConversationRefSchema,
    context_fingerprint: ContentHashSchema.optional(),
    tool_set_fingerprint: ContentHashSchema.optional(),
    usage: GenerationUsageSchema.optional(),
    metadata: MetadataSchema.optional(),
};

export const ExecutedGenerationSchema = z
    .strictObject({
        ...commonGenerationShape,
        record_source: z.literal('executed'),
        request_receipt: RequestReceiptSchema,
    })
    .meta({ id: 'ConversationExecutedGeneration' });

export const ImportedGenerationSchema = z
    .strictObject({
        id: IdentifierSchema,
        record_source: z.literal('imported'),
        request_id: IdentifierSchema.optional(),
        attempt_id: IdentifierSchema.optional(),
        provider_response_id: z.string().min(1).optional(),
        purpose: IdentifierSchema.optional(),
        requested_model: IdentifierSchema.optional(),
        resolved_model: IdentifierSchema.optional(),
        provider: IdentifierSchema.optional(),
        protocol: IdentifierSchema.optional(),
        model_options: JsonObjectSchema.optional(),
        adapter_version: IdentifierSchema.optional(),
        status: GenerationStatusSchema,
        finish_reason: z.string().optional(),
        timestamps: GenerationTimestampsSchema,
        source: ConversationRefSchema,
        context_fingerprint: ContentHashSchema.optional(),
        tool_set_fingerprint: ContentHashSchema.optional(),
        usage: GenerationUsageSchema.optional(),
        metadata: MetadataSchema.optional(),
        request_receipt: RequestReceiptSchema.optional(),
        missing_metadata: z
            .array(
                z.enum([
                    'request_receipt',
                    'request_id',
                    'attempt_id',
                    'timestamps',
                    'usage',
                    'purpose',
                    'requested_model',
                    'resolved_model',
                    'provider',
                    'protocol',
                    'adapter_version',
                    'provider_response_id',
                    'finish_reason',
                ]),
            )
            .optional(),
    })
    .meta({ id: 'ConversationImportedGeneration' });

export const GenerationSchema = z
    .discriminatedUnion('record_source', [ExecutedGenerationSchema, ImportedGenerationSchema])
    .meta({ id: 'ConversationGeneration' });

export const ContextChangePlacementSchema = z
    .strictObject({
        mode: z.enum(['first_selected', 'per_selected_range']),
        causal_order: z.enum(['contiguous', 'explicit_disjoint_summary', 'preserved_disjoint_ranges']),
    })
    .meta({ id: 'ConversationContextChangePlacement' });

export const SelectedContextBlocksSchema = z
    .record(IdentifierSchema, z.array(IdentifierSchema).min(1))
    .meta({ id: 'ConversationSelectedContextBlocks' });

export const ContextChangeOperationSchema = z
    .strictObject({
        kind: z.enum(['exclude', 'replace_with_compaction']),
        removed_entry_ids: z.array(IdentifierSchema).min(1),
        inserted_entry_ids: z.array(IdentifierSchema),
        source_fingerprint: ContentHashSchema,
        placement: ContextChangePlacementSchema.optional(),
        selected_block_ids: SelectedContextBlocksSchema.optional(),
        remainder_entry_ids: z.array(IdentifierSchema).optional(),
        discarded_replay_block_ids: z.array(IdentifierSchema).min(1).max(4096).optional(),
    })
    .meta({ id: 'ConversationContextChangeOperation' });

export { ProcessingOperationSchema };
/** Accepted request intent, including omitted versus explicitly empty active-tool selection. */
export const AcceptedToolSelectionSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('unchanged') }),
        z.strictObject({ kind: z.literal('replace'), definition_ids: z.array(IdentifierSchema) }),
    ])
    .meta({ id: 'ConversationAcceptedToolSelection' });

export const OperationReceiptSchema = z
    .strictObject({
        id: IdentifierSchema,
        conversation_id: IdentifierSchema,
        payload_fingerprint: ContentHashSchema,
        base_revision: NonnegativeSafeIntegerSchema,
        result_revision: NonnegativeSafeIntegerSchema,
        recorded_at: TimestampSchema,
        accepted_turn_ids: z.array(IdentifierSchema).optional(),
        accepted_generation_ids: z.array(IdentifierSchema).optional(),
        accepted_asset_ids: z.array(IdentifierSchema).optional(),
        accepted_tool_definition_ids: z.array(IdentifierSchema).optional(),
        accepted_execution_receipt_ids: z.array(IdentifierSchema).optional(),
        accepted_context_entry_ids: z.array(IdentifierSchema).optional(),
        /** Immutable accepted references; active context may later remove or partition them. */
        accepted_context_entries: z.array(ContextEntrySchema).optional(),
        accepted_tool_selection: AcceptedToolSelectionSchema.optional(),
        /** Absent on append receipts, including historical ones. */
        operation_kind: z.enum(['context_change', 'conversation_edit', 'conversation_delete', 'processing']).optional(),
        context_change: ContextChangeOperationSchema.optional(),
        conversation_edit: ConversationEditOperationSchema.optional(),
        conversation_delete: ConversationDeleteOperationSchema.optional(),
        processing_operation: ProcessingOperationSchema.optional(),
    })
    .meta({ id: 'ConversationOperationReceipt' });

/** Immutable canonical identity of the application tool call authorized for execution. */
export const ToolCallSourceRefSchema = z
    .strictObject({
        conversation: ConversationRefSchema,
        turn_id: IdentifierSchema,
        block_id: IdentifierSchema,
        call_id: IdentifierSchema,
        call_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationToolCallSourceRef' });

export const ExecutionReceiptSchema = z
    .strictObject({
        id: IdentifierSchema,
        call_id: IdentifierSchema,
        executor: z.enum(['application', 'provider']),
        host_execution_id: IdentifierSchema.optional(),
        attempt_id: IdentifierSchema.optional(),
        status: z.enum(['success', 'error', 'cancelled', 'denied']),
        result_turn_id: IdentifierSchema.optional(),
        result_fingerprint: ContentHashSchema,
        recorded_at: TimestampSchema,
        call_source: ToolCallSourceRefSchema.optional(),
    })
    .meta({ id: 'ConversationExecutionReceipt' });
