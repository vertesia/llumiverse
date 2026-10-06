import { z } from 'zod';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
} from './primitives.js';

/** A body-free witness to content retained in an authenticated predecessor revision. */
export const ConversationDeletedTurnRefSchema = z
    .strictObject({
        id: IdentifierSchema,
        fingerprint: ContentHashSchema,
        block_ids: z.array(IdentifierSchema).min(1),
        call_ids: z.array(IdentifierSchema).min(1).optional(),
        accepted_operation_id: IdentifierSchema,
    })
    .meta({ id: 'ConversationDeletedTurnRef' });

/** Occupies historical IDs without exposing deleted content in the current document. */
export const ConversationDeletedTurnSchema = ConversationDeletedTurnRefSchema.extend({
    operation_id: IdentifierSchema,
    source_revision: NonnegativeSafeIntegerSchema,
}).meta({ id: 'ConversationDeletedTurn' });

export const ConversationDeleteOperationSchema = z
    .strictObject({
        version: z.literal(1),
        source: ConversationRefSchema,
        source_fingerprint: ContentHashSchema,
        dependency_policy: z.literal('reject'),
        /** Exact active entries removed atomically, only under explicit exclusion policy. */
        excluded_context_entry_ids: z.array(IdentifierSchema).min(1).max(4096).optional(),
        deleted_turns: z.array(ConversationDeletedTurnRefSchema).min(1).max(4096),
    })
    .meta({ id: 'ConversationDeleteOperation' });
