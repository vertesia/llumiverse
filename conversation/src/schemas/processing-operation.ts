import { z } from 'zod';
import { JsonMinificationApplicationSchema } from './json-minification.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema } from './primitives.js';

/** Receipt detail shared by durable processing mutations and the public change envelope. */
export const ProcessingOperationSchema = z
    .strictObject({
        phase: z.enum(['policy', 'queue', 'resolve', 'attempt', 'output', 'complete', 'coverage']),
        job_id: IdentifierSchema.optional(),
        policy_revision: NonnegativeSafeIntegerSchema,
        result_fingerprint: ContentHashSchema.optional(),
        transform_application: JsonMinificationApplicationSchema.optional(),
        superseded_job_ids: z.array(IdentifierSchema).optional(),
    })
    .meta({ id: 'ConversationProcessingOperation' });

export const ProcessingChangeOperationSchema = z
    .strictObject({ kind: z.literal('processing'), detail: ProcessingOperationSchema })
    .meta({ id: 'ConversationProcessingChangeOperation' });
