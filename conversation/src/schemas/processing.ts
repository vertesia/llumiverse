import { z } from 'zod';
import { ContextChangeProposalSchema } from './context-change.js';
import { ContextEntrySchema } from './context-foundation.js';
import { GenerationUsageSchema } from './execution.js';
import {
    JsonMinificationMeasurementSchema,
    JsonMinificationNoOpReasonSchema,
    JsonMinificationProposalSchema,
} from './json-minification.js';
import {
    ContentHashSchema,
    IdentifierSchema,
    JsonObjectSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';

export { MAX_PROCESSING_OUTPUT_BYTES, MAX_PROCESSOR_CONFIGURATION_BYTES } from '../runtime-constants.js';

export const ProcessingJobSelectionSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({
            kind: z.literal('entries'),
            entry_ids: z.array(IdentifierSchema),
            selected_block_ids: z.record(IdentifierSchema, z.array(IdentifierSchema).min(1)).optional(),
            selected_entries: z.array(ContextEntrySchema).min(1).optional(),
        }),
        z.strictObject({ kind: z.literal('predecessor_output'), job_id: IdentifierSchema }),
    ])
    .meta({ id: 'ConversationProcessingJobSelection' });

export const ProcessingJobSchema = z
    .strictObject({
        id: IdentifierSchema,
        source_operation_id: IdentifierSchema,
        enqueue_revision: NonnegativeSafeIntegerSchema,
        policy_revision: NonnegativeSafeIntegerSchema,
        stage_index: NonnegativeSafeIntegerSchema,
        processor_index: NonnegativeSafeIntegerSchema,
        processor_id: IdentifierSchema,
        processor_version: IdentifierSchema,
        configuration_fingerprint: ContentHashSchema,
        configuration: JsonObjectSchema,
        scope: z.enum(['on_append', 'on_budget', 'manual']),
        required: z.boolean(),
        failure_behavior: z.enum(['block', 'skip_with_diagnostic']),
        selection: ProcessingJobSelectionSchema,
        selection_fingerprint: ContentHashSchema,
        target_fingerprint: ContentHashSchema.optional(),
    })
    .meta({ id: 'ConversationProcessingJob' });

export const ProcessingResolvedInputSchema = z
    .strictObject({
        job_id: IdentifierSchema,
        source_revision: NonnegativeSafeIntegerSchema,
        context_revision: NonnegativeSafeIntegerSchema,
        entry_ids: z.array(IdentifierSchema),
        selected_block_ids: z.record(IdentifierSchema, z.array(IdentifierSchema).min(1)).optional(),
        selected_entries: z.array(ContextEntrySchema).min(1).optional(),
        source_fingerprint: ContentHashSchema,
        context_fingerprint: ContentHashSchema,
        source_turn_ids: z.array(IdentifierSchema),
        target_fingerprint: ContentHashSchema.optional(),
        recorded_at: TimestampSchema,
    })
    .meta({ id: 'ConversationProcessingResolvedInput' });

export const ProcessingAttemptReceiptSchema = z
    .strictObject({
        job_id: IdentifierSchema,
        resolved_input_fingerprint: ContentHashSchema,
        attempt_token: IdentifierSchema,
        started_at: TimestampSchema,
    })
    .meta({ id: 'ConversationProcessingAttemptReceipt' });

export const ProcessingOutputReceiptSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({
            kind: z.literal('json_minification'),
            job_id: IdentifierSchema,
            resolved_input_fingerprint: ContentHashSchema,
            attempt_token: IdentifierSchema,
            output_fingerprint: ContentHashSchema,
            proposal: JsonMinificationProposalSchema,
            recorded_at: TimestampSchema,
        }),
        z.strictObject({
            kind: z.literal('json_minification_no_op'),
            job_id: IdentifierSchema,
            resolved_input_fingerprint: ContentHashSchema,
            attempt_token: IdentifierSchema,
            output_fingerprint: ContentHashSchema,
            reason: JsonMinificationNoOpReasonSchema,
            measurement: JsonMinificationMeasurementSchema.optional(),
            recorded_at: TimestampSchema,
        }),
        z.strictObject({
            kind: z.literal('proposal'),
            job_id: IdentifierSchema,
            resolved_input_fingerprint: ContentHashSchema,
            attempt_token: IdentifierSchema,
            output_fingerprint: ContentHashSchema,
            proposal: ContextChangeProposalSchema,
            usage: GenerationUsageSchema.optional(),
            recorded_at: TimestampSchema,
        }),
        z.strictObject({
            kind: z.literal('no_op'),
            job_id: IdentifierSchema,
            resolved_input_fingerprint: ContentHashSchema,
            attempt_token: IdentifierSchema.optional(),
            output_fingerprint: ContentHashSchema,
            reason: IdentifierSchema,
            usage: GenerationUsageSchema.optional(),
            recorded_at: TimestampSchema,
        }),
        z.strictObject({
            kind: z.enum(['failed', 'unknown_outcome']),
            job_id: IdentifierSchema,
            resolved_input_fingerprint: ContentHashSchema,
            attempt_token: IdentifierSchema,
            output_fingerprint: ContentHashSchema,
            diagnostic: z.string().min(1).max(8192),
            usage: GenerationUsageSchema.optional(),
            recovery_operation_id: IdentifierSchema.optional(),
            recorded_at: TimestampSchema,
        }),
    ])
    .meta({ id: 'ConversationProcessingOutputReceipt' });

export const ProcessingCompletionReceiptSchema = z
    .strictObject({
        job_id: IdentifierSchema,
        output_fingerprint: ContentHashSchema,
        status: z.enum(['applied', 'no_op', 'skipped', 'blocked']),
        result_revision: NonnegativeSafeIntegerSchema,
        inserted_entry_ids: z.array(IdentifierSchema),
        context_change_operation_id: IdentifierSchema.optional(),
        recorded_at: TimestampSchema,
    })
    .meta({ id: 'ConversationProcessingCompletionReceipt' });

export const ProcessingSupersessionReceiptSchema = z
    .strictObject({
        job_id: IdentifierSchema,
        policy_operation_id: IdentifierSchema,
        reason: IdentifierSchema,
        recorded_at: TimestampSchema,
    })
    .meta({ id: 'ConversationProcessingSupersessionReceipt' });

export const ProcessingReadinessCoverageSchema = z
    .strictObject({
        context_fingerprint: ContentHashSchema,
        policy_revision: NonnegativeSafeIntegerSchema,
        target_fingerprint: ContentHashSchema,
        measurement: z.strictObject({
            input_tokens: NonnegativeSafeIntegerSchema,
            tokenizer_id: IdentifierSchema,
            fingerprint: ContentHashSchema,
        }),
        required_job_ids: z.array(IdentifierSchema),
        status: z.enum(['ready', 'pending', 'blocked']),
        evaluated_at_revision: NonnegativeSafeIntegerSchema,
        recorded_at: TimestampSchema,
    })
    .meta({ id: 'ConversationProcessingReadinessCoverage' });

/** Bounded host acknowledgement: accepted content and readiness are separate durable outcomes. */
export const ProcessingAppendAcceptanceSchema = z
    .strictObject({
        operation_id: IdentifierSchema,
        conversation_id: IdentifierSchema,
        base_revision: NonnegativeSafeIntegerSchema,
        accepted_revision: NonnegativeSafeIntegerSchema,
        change_reference: z.strictObject({
            operation_id: IdentifierSchema,
            result_revision: NonnegativeSafeIntegerSchema,
        }),
        processing: z.strictObject({
            status: z.enum(['ready', 'pending', 'blocked']),
            job_ids: z.array(IdentifierSchema).max(16),
        }),
    })
    .meta({ id: 'ConversationProcessingAppendAcceptance' });
