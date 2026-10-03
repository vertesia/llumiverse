import {
    ContentHashSchema,
    ContextMeasurementSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
} from '@llumiverse/conversation/schemas';
import { z } from 'zod';

/** Ephemeral host count evidence; absent from serialized provider options and published HTTP schemas. */
export const CanonicalProjectedRequestMeasurementSchema = z.strictObject({
    measurement: ContextMeasurementSchema,
    counted_request_fingerprint: ContentHashSchema,
    readiness: z
        .strictObject({
            profile: IdentifierSchema,
            output_reserve_tokens: NonnegativeSafeIntegerSchema,
            context_limit: PositiveSafeIntegerSchema,
        })
        .optional(),
});
