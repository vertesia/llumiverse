import { z } from 'zod';
import { ContextEntrySchema } from './context-foundation.js';
import { JsonMinificationApplicationSchema } from './json-minification.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema } from './primitives.js';
import { ProcessingPolicyCommandSchema, ProcessingPolicyGenesisSchema } from './processing-policy.js';
import { ProcessingQueueAcceptanceInputSchema } from './processing-queue.js';

/** Receipt detail shared by durable processing mutations and the public change envelope. */
export const ProcessingOperationSchema = z
    .strictObject({
        phase: z.enum(['policy', 'queue', 'resolve', 'attempt', 'output', 'complete', 'coverage']),
        job_id: IdentifierSchema.optional(),
        policy_revision: NonnegativeSafeIntegerSchema,
        result_fingerprint: ContentHashSchema.optional(),
        transform_application: JsonMinificationApplicationSchema.optional(),
        superseded_job_ids: z.array(IdentifierSchema).optional(),
        /** Actual accepted materialized input, retained for cold migration and exact queue replay. */
        queue_command: ProcessingQueueAcceptanceInputSchema.optional(),
        /** Actual accepted policy input; its exact bytes are covered by the policy receipt hash. */
        policy_command: ProcessingPolicyCommandSchema.optional(),
        /** Actual predecessor source/policy at the first materialized transition; no job state or counts. */
        policy_genesis: ProcessingPolicyGenesisSchema.optional(),
        queue_selected_entries: z.array(ContextEntrySchema).max(4096).optional(),
    })
    .meta({ id: 'ConversationProcessingOperation' });

export const ProcessingChangeOperationSchema = z
    .strictObject({ kind: z.literal('processing'), detail: ProcessingOperationSchema })
    .meta({ id: 'ConversationProcessingChangeOperation' });
