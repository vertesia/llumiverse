import { z } from 'zod';
import { ContextMeasurementSchema } from './context-measurement.js';
import { ModelTargetSchema } from './execution.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';

export const ConversationModelSwitchRequestSchema = z
    .strictObject({
        source: ConversationRefSchema,
        expected_context_revision: NonnegativeSafeIntegerSchema,
        target: ModelTargetSchema,
        measurement_policy: z.enum(['exact_only', 'identified_estimate']).optional(),
        measurement_mode: z.enum(['local', 'provider']).optional(),
    })
    .meta({ id: 'ConversationModelSwitchRequest' });

export const ConversationModelSwitchBlockerSchema = z
    .strictObject({
        code: z.enum([
            'SOURCE_CHANGED',
            'OPEN_TOOL_CALL',
            'UNSETTLED_GENERATION',
            'TARGET_UNSUPPORTED',
            'MEASUREMENT_UNAVAILABLE',
            'MEASUREMENT_MISMATCH',
            'BUDGET_UNAVAILABLE',
            'BUDGET_UNSATISFIED',
            'PROCESSING_PENDING',
            'PROCESSING_BLOCKED',
        ]),
        message: z.string().min(1).max(1024),
    })
    .meta({ id: 'ConversationModelSwitchBlocker' });

export const ConversationModelSwitchBudgetAnalysisSchema = z
    .strictObject({
        context_limit_tokens: PositiveSafeIntegerSchema,
        output_reserve_tokens: NonnegativeSafeIntegerSchema,
        available_input_tokens: NonnegativeSafeIntegerSchema,
        measured_input_tokens: NonnegativeSafeIntegerSchema,
    })
    .meta({ id: 'ConversationModelSwitchBudgetAnalysis' });

/** A dry, revision-bound compatibility report; it is never execution authorization. */
export const ConversationModelSwitchPlanSchema = z
    .strictObject({
        version: z.literal(1),
        source: ConversationRefSchema,
        expected_context_revision: NonnegativeSafeIntegerSchema,
        source_fingerprint: ContentHashSchema,
        context_fingerprint: ContentHashSchema,
        target: ModelTargetSchema,
        target_fingerprint: ContentHashSchema,
        /** The requested policy; an existing source exact-only budget remains authoritative. */
        measurement_policy: z.enum(['exact_only', 'identified_estimate']).optional(),
        measurement_mode: z.enum(['local', 'provider']).optional(),
        options_fingerprint: ContentHashSchema,
        active_tool_definition_ids: z.array(IdentifierSchema),
        tool_set_fingerprint: ContentHashSchema,
        native_request_fingerprint: ContentHashSchema.optional(),
        measurement: ContextMeasurementSchema.optional(),
        budget: ConversationModelSwitchBudgetAnalysisSchema.optional(),
        compatibility: z.enum(['compatible', 'blocked']),
        blockers: z.array(ConversationModelSwitchBlockerSchema),
        media_transformations: z.array(IdentifierSchema),
        replay_exclusions: z.array(IdentifierSchema),
        required_context_changes: z.array(IdentifierSchema),
        cache_invalidations: z.array(z.enum(['native_request', 'provider_cache', 'provider_upload'])),
    })
    .meta({ id: 'ConversationModelSwitchPlan' });

/** The host commits this precondition atomically, then runs ordinary preparation again. */
export const ConversationModelSwitchNextRequestChangeSchema = z
    .strictObject({
        operation_id: IdentifierSchema,
        recorded_at: TimestampSchema,
        expected_source: ConversationRefSchema,
        expected_context_revision: NonnegativeSafeIntegerSchema,
        expected_source_fingerprint: ContentHashSchema,
        expected_context_fingerprint: ContentHashSchema,
        target: ModelTargetSchema,
        target_fingerprint: ContentHashSchema,
        measurement_policy: z.enum(['exact_only', 'identified_estimate']).optional(),
        measurement_mode: z.enum(['local', 'provider']).optional(),
        options_fingerprint: ContentHashSchema,
        active_tool_definition_ids: z.array(IdentifierSchema),
        tool_set_fingerprint: ContentHashSchema,
        native_request_fingerprint: ContentHashSchema,
        measurement: ContextMeasurementSchema,
        budget: ConversationModelSwitchBudgetAnalysisSchema,
        requires_quiescent_generation: z.literal(true),
        cache_invalidations: z.array(z.enum(['native_request', 'provider_cache', 'provider_upload'])),
    })
    .meta({ id: 'ConversationModelSwitchNextRequestChange' });
