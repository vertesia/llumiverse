import { z } from 'zod';
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
    })
    .meta({ id: 'ConversationRuntimeContext' });

export const UsageMetricSchema = z
    .enum([
        'input_tokens',
        'output_tokens',
        'total_tokens',
        'reasoning_tokens',
        'cache_read_tokens',
        'cache_write_tokens',
        'input_new_tokens',
    ])
    .meta({ id: 'ConversationUsageMetric' });

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

export const ContextMeasurementSchema = z
    .strictObject({
        input_tokens: NonnegativeSafeIntegerSchema,
        method: z.enum(['exact', 'estimated', 'provider_counted']),
        tokenizer: IdentifierSchema,
        tokenizer_version: IdentifierSchema.optional(),
        adapter: IdentifierSchema,
        adapter_version: IdentifierSchema,
        source_fingerprint: ContentHashSchema,
        target_model: IdentifierSchema,
        measured_at: TimestampSchema,
    })
    .meta({ id: 'ConversationContextMeasurement' });

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
    status: z.enum(['completed', 'failed', 'cancelled']),
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
        status: z.enum(['completed', 'failed', 'cancelled']),
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
    })
    .meta({ id: 'ConversationOperationReceipt' });

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
    })
    .meta({ id: 'ConversationExecutionReceipt' });
