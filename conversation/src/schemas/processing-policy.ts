import { z } from 'zod';
import {
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';
import { ProcessingBudgetSchema, ProcessorConfigurationSchema } from './processing-policy-foundation.js';

export const ProcessingPolicyCommandSchema = z
    .strictObject({
        operation_id: IdentifierSchema,
        expected_revision: NonnegativeSafeIntegerSchema,
        recorded_at: TimestampSchema,
        enabled: z.boolean(),
        processors: z.array(ProcessorConfigurationSchema).max(16),
        budget: ProcessingBudgetSchema.optional(),
        supersede_job_ids: z.array(IdentifierSchema).optional(),
        supersession_reason: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationProcessingPolicyCommand' });

/** Genuine startup policy bytes captured before the first accepted materialized policy transition. */
export const InitialProcessingPolicySchema = z
    .strictObject({
        enabled: z.boolean(),
        policy_revision: z.literal(0),
        processors: z.array(ProcessorConfigurationSchema).max(16),
        budget: ProcessingBudgetSchema.optional(),
    })
    .meta({ id: 'ConversationInitialProcessingPolicy' });
export const ProcessingPolicyGenesisSchema = z
    .strictObject({
        source: ConversationRefSchema,
        policy: InitialProcessingPolicySchema,
    })
    .meta({ id: 'ConversationProcessingPolicyGenesis' });
