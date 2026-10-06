import { z } from 'zod';
import { INDEXED_CONVERSATION_UPGRADE_PROFILE } from '../indexed-upgrade-constants.js';
import { PagedRecordRefSchema } from '../paged-record-index.js';
import { IndexedConversationDirectoriesSchema } from './indexed-head.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';

/** Explicit paged audit; never a source projection or an admission/readiness token. */
export { INDEXED_CONVERSATION_UPGRADE_PROFILE } from '../indexed-upgrade-constants.js';

export const IndexedConversationUpgradeCommandSchema = z.strictObject({
    version: z.literal(1),
    profile: z.literal(INDEXED_CONVERSATION_UPGRADE_PROFILE),
    operation_id: IdentifierSchema,
    source: ConversationRefSchema,
    predecessor_root: PagedRecordRefSchema,
    recorded_at: TimestampSchema,
});
export type IndexedConversationUpgradeCommand = z.infer<typeof IndexedConversationUpgradeCommandSchema>;

/** Progress is host-retained and hash-addressed. A supplied cursor cannot authorize skipping work. */
export const IndexedConversationUpgradeProgressSchema = z.strictObject({
    version: z.literal(1),
    profile: z.literal(INDEXED_CONVERSATION_UPGRADE_PROFILE),
    command: IndexedConversationUpgradeCommandSchema,
    command_fingerprint: ContentHashSchema,
    predecessor_progress: PagedRecordRefSchema.optional(),
    step: NonnegativeSafeIntegerSchema,
    phase: z.enum([
        'receipts',
        'executions',
        'turns',
        'blocks',
        'turn_order',
        'processing',
        'audit',
        'active_window',
        'complete',
    ]),
    cursor: z.string().min(1).max(2048).optional(),
    item_cursor: NonnegativeSafeIntegerSchema.optional(),
    audit_family: NonnegativeSafeIntegerSchema.optional(),
    directories: IndexedConversationDirectoriesSchema,
    /** Temporary cross-reference indexes are private derivation facts, not canonical authority. */
    scratch: z.strictObject({
        call_results: PagedRecordRefSchema.optional(),
        request_generations: PagedRecordRefSchema.optional(),
        entry_turns: PagedRecordRefSchema.optional(),
        call_executions: PagedRecordRefSchema.optional(),
        audited_identifiers: PagedRecordRefSchema.optional(),
        ordered_turns: PagedRecordRefSchema.optional(),
        operation_jobs: PagedRecordRefSchema.optional(),
        execution_acceptances: PagedRecordRefSchema.optional(),
        response_revisions: PagedRecordRefSchema.optional(),
        input_revisions: PagedRecordRefSchema.optional(),
    }),
    counts: z.strictObject({
        live_turns: NonnegativeSafeIntegerSchema,
        jobs: NonnegativeSafeIntegerSchema,
        unresolved_jobs: NonnegativeSafeIntegerSchema,
        required_jobs: NonnegativeSafeIntegerSchema,
        required_unresolved_jobs: NonnegativeSafeIntegerSchema,
        required_blocked_jobs: NonnegativeSafeIntegerSchema,
        source_records: NonnegativeSafeIntegerSchema,
        audited_records: NonnegativeSafeIntegerSchema,
    }),
    previous_live_turn_id: IdentifierSchema.optional(),
    previous_live_ordinal: NonnegativeSafeIntegerSchema.optional(),
    restart_response: z
        .strictObject({ operation_id: IdentifierSchema, result_revision: NonnegativeSafeIntegerSchema })
        .optional(),
    restart_tool_input: z
        .strictObject({ operation_id: IdentifierSchema, result_revision: NonnegativeSafeIntegerSchema })
        .optional(),
});
export type IndexedConversationUpgradeProgress = z.infer<typeof IndexedConversationUpgradeProgressSchema>;
