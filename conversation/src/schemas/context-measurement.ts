import { z } from 'zod';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema, TimestampSchema } from './primitives.js';

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
