import { z } from 'zod';
import {
    IdentifierSchema,
    JsonObjectSchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
} from './primitives.js';

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

export const ProcessingBudgetSchema = z
    .strictObject({
        max_input_tokens: PositiveSafeIntegerSchema,
        output_reserve_tokens: NonnegativeSafeIntegerSchema,
        measurement_policy: z.enum(['exact_only', 'identified_estimate']).optional(),
    })
    .meta({ id: 'ConversationProcessingBudget' });
