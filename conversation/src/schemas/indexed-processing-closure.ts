import { z } from 'zod';
import { canonicalJsonContentBytes } from '../content-integrity.js';
import { PagedRecordRefSchema } from '../paged-record-index.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    JsonObjectSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';

/** Host-neutral bounded immutable binding. This is not a platform scheduler registration schema. */
const binding = JsonObjectSchema.superRefine((value, ctx) => {
    if (canonicalJsonContentBytes(value).byteLength > 32 * 1024)
        ctx.addIssue({ code: 'custom', message: 'Indexed closure binding exceeds its metadata bound' });
});
export const IndexedProcessingClosureCommandSchema = z.strictObject({
    // Absence preserves the original content-revision transition contract.
    publication: z.literal('retention').optional(),
    operation_id: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
    binding,
});
export const IndexedProcessingClosureWitnessSchema = z.strictObject({
    version: z.literal(1),
    // Absence preserves the original content-revision transition contract.
    publication: z.literal('retention').optional(),
    operation_id: IdentifierSchema,
    predecessor: z.strictObject({ source: ConversationRefSchema, root: PagedRecordRefSchema }),
    result_revision: NonnegativeSafeIntegerSchema,
    binding_fingerprint: ContentHashSchema,
    binding,
    recorded_at: TimestampSchema,
});
export type IndexedProcessingClosureWitness = z.infer<typeof IndexedProcessingClosureWitnessSchema>;
