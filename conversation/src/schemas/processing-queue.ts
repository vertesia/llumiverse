import { z } from 'zod';
import { ContextSelectionRequestSchema } from './context-selection-request.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema, TimestampSchema } from './primitives.js';

/** Shared materialized queue contract. Its acceptance hash covers this command and exact selector,
 * unlike the distinct native indexed current-entry command profile.
 */
export const ProcessingQueueCommandSchema = z
    .strictObject({
        operation_id: IdentifierSchema,
        expected_revision: NonnegativeSafeIntegerSchema,
        recorded_at: TimestampSchema,
        processor_id: IdentifierSchema,
        scope: z.enum(['manual', 'on_budget']),
        target_fingerprint: ContentHashSchema.optional(),
    })
    .meta({ id: 'ConversationProcessingQueueCommand' });
export const ProcessingQueueAcceptanceInputSchema = z
    .strictObject({
        command: ProcessingQueueCommandSchema,
        selection: ContextSelectionRequestSchema,
    })
    .meta({ id: 'ConversationProcessingQueueAcceptanceInput' });
export type ProcessingQueueAcceptanceInput = z.infer<typeof ProcessingQueueAcceptanceInputSchema>;
