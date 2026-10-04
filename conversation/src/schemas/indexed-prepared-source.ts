import { z } from 'zod';
import { PagedRecordRefSchema } from '../paged-record-index.js';
import { NonnegativeSafeIntegerSchema } from './primitives.js';

/** Validation of one selected text request against an immutable indexed root. */
export const INDEXED_TEXT_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-selected-text/2026-10-03.v1' as const;

export const INDEXED_DEPENDENCY_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-selected-dependencies/2026-10-04.v1' as const;

export const INDEXED_MEDIA_COMPACTION_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-selected-media-compaction/2026-10-04.v1' as const;

/** Private prepared evidence. The root is loaded by hash under an authenticated run prefix. */
export const IndexedPreparedSourceSchema = z.strictObject({
    version: z.literal(1),
    validator_profile: z.enum([
        INDEXED_TEXT_PREPARED_VALIDATOR_PROFILE,
        INDEXED_DEPENDENCY_PREPARED_VALIDATOR_PROFILE,
        INDEXED_MEDIA_COMPACTION_PREPARED_VALIDATOR_PROFILE,
    ]),
    root: PagedRecordRefSchema,
    context_revision: NonnegativeSafeIntegerSchema,
});

export type IndexedPreparedSource = z.infer<typeof IndexedPreparedSourceSchema>;
