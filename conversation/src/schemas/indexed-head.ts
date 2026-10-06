import { z } from 'zod';
import { PagedRecordRefSchema } from '../paged-record-index.js';
import { AssetSchema, ToolDefinitionSchema } from './content.js';
import { ContextEntrySchema } from './context-foundation.js';
import { ConversationDeletedTurnSchema } from './conversation-delete-operation.js';
import {
    CompactionRecordSchema,
    ConversationContextSchema,
    ConversationLineageSchema,
    ProcessingBudgetSchema,
    ProcessingStateSchema,
    ProcessorConfigurationSchema,
} from './document.js';
import { ExecutionReceiptSchema, GenerationSchema, OperationReceiptSchema } from './execution.js';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    MetadataSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';
import { ProjectedTurnHeaderSchema, RequestSourceProjectedTurnSchema } from './request-source-view.js';

/** Indexed storage is a distinct, partial materialization contract; it is never parsed as ConversationDocument. */
export const INDEXED_CONVERSATION_ROOT_MAX_BYTES = 256 * 1024;
export const INDEXED_CONVERSATION_ACTIVE_MAX_BYTES = 32 * 1024 * 1024;
/** Context metadata is an independently addressed segmented record, not an index page. Its
 * retrieval requirements remain subject to the existing active working-set/record byte ceiling. */
export const IndexedConversationContextHeaderRefSchema = z.strictObject({
    content_hash: ContentHashSchema,
    size_bytes: z.number().int().positive().max(INDEXED_CONVERSATION_ACTIVE_MAX_BYTES),
});

/** Initial processing profile bounds the entire active dependency closure, not lifetime turns.
 * These ceilings are checked before publishing an enabled snapshot or accepted append root. */
export const INDEXED_PROCESSING_SELECTED_MAX_BLOCKS = 4096;
export const INDEXED_PROCESSING_MAX_RECORD_READS = 8192;
export const INDEXED_PROCESSING_MAX_PAGE_READS = 4096;
export const INDEXED_PROCESSING_MAX_IO_BYTES = 64 * 1024 * 1024;
export const INDEXED_CONVERSATION_RECORD_SEGMENT_MAX_BYTES = 1024 * 1024;
export const INDEXED_CONVERSATION_INLINE_MAX_BYTES = 64 * 1024;
export const INDEXED_CONVERSATION_PROFILE = 'llumiverse.conversation/indexed/2026-10-02.v1' as const;
export const INDEXED_CONVERSATION_PROCESSING_PROFILE =
    'llumiverse.conversation/indexed-processing/2026-10-04.v1' as const;
export const INDEXED_CONVERSATION_DELETE_PROFILE = 'llumiverse.conversation/indexed-delete/2026-10-03.v1' as const;

export const INDEXED_CONVERSATION_DELETE_PROFILE_V2 = 'llumiverse.conversation/indexed-delete/2026-10-06.v2' as const;

export const INDEXED_CONVERSATION_RESTART_PROFILE = 'llumiverse.conversation/indexed-restart/2026-10-05.v1' as const;
const IndexedRestartOperationSchema = z.strictObject({
    operation_id: IdentifierSchema,
    result_revision: NonnegativeSafeIntegerSchema,
});

/** Every named family remains independently addressable without scanning lifetime history. */
export const IndexedConversationDirectoriesSchema = z.strictObject({
    identifiers: PagedRecordRefSchema.optional(),
    turns: PagedRecordRefSchema.optional(),
    blocks: PagedRecordRefSchema.optional(),
    generations: PagedRecordRefSchema.optional(),
    generation_acceptances: PagedRecordRefSchema.optional(),
    /** Accepted revision -> operation identity; content stays in the original canonical record families. */
    accepted_output_order: PagedRecordRefSchema.optional(),
    operation_receipts: PagedRecordRefSchema.optional(),
    execution_receipts: PagedRecordRefSchema.optional(),
    assets: PagedRecordRefSchema.optional(),
    tool_definitions: PagedRecordRefSchema.optional(),
    compactions: PagedRecordRefSchema.optional(),
    processing_records: PagedRecordRefSchema.optional(),
    /** Only unresolved unsuperseded jobs; completion removes entries without erasing history. */
    processing_pending: PagedRecordRefSchema.optional(),
    /** Persistent identity of all unsuperseded required jobs, including completed obligations. */
    processing_required: PagedRecordRefSchema.optional(),
    /** Complete immutable append/queue operation to exact accepted job-set witness. */
    processing_by_operation: PagedRecordRefSchema.optional(),
    /** Readiness identities have point lookups; no lifetime receipt scan. */
    processing_coverage: PagedRecordRefSchema.optional(),
    open_tool_calls: PagedRecordRefSchema.optional(),
    /** Complete call/result/terminal-receipt lookup for append validation. */
    tool_call_states: PagedRecordRefSchema.optional(),
    context_entries: PagedRecordRefSchema.optional(),
    active_context_order: PagedRecordRefSchema.optional(),
    turn_order: PagedRecordRefSchema.optional(),
    /** Complete point-lookup witnesses for dependency-closed logical deletion. */
    turn_acceptances: PagedRecordRefSchema.optional(),
    block_owners: PagedRecordRefSchema.optional(),
    deletion_blockers: PagedRecordRefSchema.optional(),
    /** v2 typed reverse dependencies, keyed by target hash and dependency identity. */
    deletion_dependencies: PagedRecordRefSchema.optional(),
    /** Reverse ordinal of LIVE completed display answers; deletion removes keys, not bodies. */
    display_answer_order: PagedRecordRefSchema.optional(),
    turn_links: PagedRecordRefSchema.optional(),
    deleted_turns: PagedRecordRefSchema.optional(),
});

export const IndexedConversationTurnLinkSchema = z.strictObject({
    id: IdentifierSchema,
    ordinal: NonnegativeSafeIntegerSchema,
    previous_turn_id: IdentifierSchema.optional(),
    next_turn_id: IdentifierSchema.optional(),
});

/** Body-free current-head witness with the exact immutable pre-delete root locator. */
export const IndexedConversationDeletedTurnSchema = z.strictObject({
    deleted_turn: ConversationDeletedTurnSchema,
    predecessor_root: PagedRecordRefSchema,
});

/** Internal indexed deletion command; the host pins expected_source_root to its authenticated CAS head. */
export const IndexedConversationDeleteCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    source: ConversationRefSchema,
    expected_source_root: PagedRecordRefSchema,
    recorded_at: TimestampSchema,
    dependency_policy: z.literal('reject'),
    context_policy: z.literal('exclude').optional(),
    turn_ids: z.array(IdentifierSchema).min(1).max(4096),
});
export type IndexedConversationDeleteCommand = z.infer<typeof IndexedConversationDeleteCommandSchema>;

/** The context header is bounded separately; its ordered entries live in active_context_order. */
export const IndexedConversationContextHeaderSchema = ConversationContextSchema.omit({ entries: true }).extend({
    active_entry_count: z.number().int().nonnegative().max(100_000),
    active_entry_bytes: z.number().int().nonnegative().max(INDEXED_CONVERSATION_ACTIVE_MAX_BYTES),
    context_fingerprint: z.string().regex(/^sha256:[0-9a-f]{64}$/),
});

export const IndexedConversationAcceptedResponseSchema = z.strictObject({
    operation_id: IdentifierSchema,
    generation_id: IdentifierSchema,
    turn_id: IdentifierSchema,
    accepted_revision: z.number().int().nonnegative(),
});

/** Original positions and hashes survive even when preparation loads one selected block. */
export const IndexedConversationTurnHeaderSchema = z.strictObject({
    turn: ProjectedTurnHeaderSchema,
    source: z.enum(['ordinary', 'replacement']),
    compaction_id: IdentifierSchema.optional(),
    block_ids: z.array(IdentifierSchema).max(100_000),
    block_ids_hash: z.string().regex(/^sha256:[0-9a-f]{64}$/),
});

export const IndexedConversationCompactionHeaderSchema = CompactionRecordSchema.omit({
    replacement_turns: true,
});

/** Content-hashed root artifact; the host CAS stores only its locator and exact revision. */
export const IndexedConversationRootSchema = z.strictObject({
    version: z.literal(1),
    validator_profile: z.literal(INDEXED_CONVERSATION_PROFILE),
    /** Missing on older roots; no deletion may rely on an incomplete reverse index. */
    delete_index_profile: z
        .enum([INDEXED_CONVERSATION_DELETE_PROFILE, INDEXED_CONVERSATION_DELETE_PROFILE_V2])
        .optional(),
    /** Absent on older v1 roots, whose tool-result closure cannot be proven by point lookup. */
    tool_call_state_complete: z.literal(true).optional(),
    /** Missing on older roots; enabled-policy transitions cannot infer outbox completeness. */
    processing_index_profile: z.literal(INDEXED_CONVERSATION_PROCESSING_PROFILE).optional(),
    format: z.literal(CONVERSATION_FORMAT),
    schema_version: z.literal(CONVERSATION_SCHEMA_VERSION),
    experimental_revision: z.literal(CONVERSATION_EXPERIMENTAL_REVISION),
    source: ConversationRefSchema,
    turn_count: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
    /** `turn_count` remains the monotonic ordinal; these are current live-turn witnesses. */
    live_turn_count: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).optional(),
    active_tail_turn_id: IdentifierSchema.nullable().optional(),
    created_at: TimestampSchema,
    updated_at: TimestampSchema,
    lineage: ConversationLineageSchema.optional(),
    metadata: MetadataSchema.optional(),
    context_header: IndexedConversationContextHeaderRefSchema,
    processing_header: PagedRecordRefSchema,
    directories: IndexedConversationDirectoriesSchema,
    accepted_response: IndexedConversationAcceptedResponseSchema.optional(),
    /** Migration validates complete receipt coverage; incremental append preserves it atomically.
     * Missing on older roots, which cannot infer restart completeness from absent pointers. */
    restart_index_profile: z.literal(INDEXED_CONVERSATION_RESTART_PROFILE).optional(),
    /** Newest RAW response acceptance, including imported or invalid tuples. Never rewind on rejection. */
    restart_response: IndexedRestartOperationSchema.optional(),
    /** Newest accepted tool input; its exact original receipt/executions remain in existing families. */
    restart_tool_input: IndexedRestartOperationSchema.optional(),
    /** Explicit completeness, including empty history. Older roots require authenticated snapshot upgrade. */
    accepted_output_index_complete: z.literal(true).optional(),
});

export type IndexedConversationRoot = z.infer<typeof IndexedConversationRootSchema>;
export type IndexedConversationDirectories = z.infer<typeof IndexedConversationDirectoriesSchema>;
export type IndexedConversationContextHeader = z.infer<typeof IndexedConversationContextHeaderSchema>;
export type IndexedConversationTurnHeader = z.infer<typeof IndexedConversationTurnHeaderSchema>;

/** Processing jobs live in their own directory; this header is not a readiness receipt. */
export const IndexedConversationProcessingHeaderSchema = ProcessingStateSchema.omit({
    jobs: true,
    resolved_inputs: true,
    attempts: true,
    outputs: true,
    completions: true,
    supersessions: true,
    coverage_receipts: true,
}).extend({
    /** Accepted private policy command whose registered selected processors were verified before CAS. */
    selected_policy_operation_id: IdentifierSchema.optional(),
    /** Validated migration witness; absent on older roots that cannot prove job drain by point lookup. */
    unresolved_job_count: NonnegativeSafeIntegerSchema.optional(),
    job_count: NonnegativeSafeIntegerSchema.optional(),
    required_unresolved_job_count: NonnegativeSafeIntegerSchema.optional(),
    required_job_count: NonnegativeSafeIntegerSchema.optional(),
    required_blocked_job_count: NonnegativeSafeIntegerSchema.optional(),
});

/** A bounded selected projection, never a complete ConversationDocument or permission to dispatch. */
export const IndexedConversationSelectedContextSchema = z.strictObject({
    completeness: z.enum([
        'selected_text_pending_admission',
        'selected_dependencies_pending_admission',
        'selected_media_compaction_pending_admission',
    ]),
    source: ConversationRefSchema,
    root: PagedRecordRefSchema,
    source_turn_count: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
    context: ConversationContextSchema,
    turns: z.array(RequestSourceProjectedTurnSchema),
    tool_definitions: z.record(IdentifierSchema, ToolDefinitionSchema),
    assets: z.record(IdentifierSchema, AssetSchema),
    generation_witnesses: z.record(
        IdentifierSchema,
        z.strictObject({
            generation: GenerationSchema,
            acceptance: OperationReceiptSchema,
        }),
    ),
    /** Exact selected-call terminal facts, absent on the original text-only profile. */
    execution_witnesses: z.record(IdentifierSchema, ExecutionReceiptSchema).optional(),
    /** Exact replacement projections and their accepted provenance, never retained original bodies. */
    replacement_turns: z
        .array(z.strictObject({ compaction_id: IdentifierSchema, projection: RequestSourceProjectedTurnSchema }))
        .optional(),
    compaction_witnesses: z
        .record(
            IdentifierSchema,
            z.strictObject({
                compaction: IndexedConversationCompactionHeaderSchema,
                acceptance: OperationReceiptSchema,
            }),
        )
        .optional(),
    operation_witnesses: z.record(IdentifierSchema, OperationReceiptSchema).optional(),
    source_tail_turn_id: IdentifierSchema.optional(),
});
export type IndexedConversationSelectedContext = z.infer<typeof IndexedConversationSelectedContextSchema>;

/** Internal selected processing data only; never accepted by a native prepared-request compiler. */
export const IndexedProcessingSelectedContextSchema = IndexedConversationSelectedContextSchema.omit({
    completeness: true,
}).extend({
    completeness: z.literal('active_processing_dependencies_verified'),
    /** Point-read accepted remainder entries from the original bounded sibling-job cohort. */
    lineage_entry_witnesses: z.record(IdentifierSchema, ContextEntrySchema).optional(),
    sibling_compaction_ids: z.array(IdentifierSchema).max(16).optional(),
});
export type IndexedProcessingSelectedContext = z.infer<typeof IndexedProcessingSelectedContextSchema>;

/** Indexed coverage commits a persistent obligation identity, never an invented empty materialized
 * required_job_ids list or a lifetime scan. The host still binds measured native bytes/concrete target.
 */
export const IndexedProcessingReadinessCoverageSchema = z.strictObject({
    version: z.literal(1),
    profile: z.literal(INDEXED_CONVERSATION_PROCESSING_PROFILE),
    context_fingerprint: ContentHashSchema,
    policy_revision: NonnegativeSafeIntegerSchema,
    target_fingerprint: ContentHashSchema,
    measurement: z.strictObject({
        input_tokens: NonnegativeSafeIntegerSchema,
        tokenizer_id: IdentifierSchema,
        fingerprint: ContentHashSchema,
    }),
    required_job_count: NonnegativeSafeIntegerSchema,
    required_jobs_root: PagedRecordRefSchema.optional(),
    status: z.enum(['ready', 'pending', 'blocked']),
    evaluated_at_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
});
export type IndexedProcessingReadinessCoverage = z.infer<typeof IndexedProcessingReadinessCoverageSchema>;

/** Exact data command shared by indexed coverage transition and retained prepared evidence. */
export const IndexedProcessingCoverageCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    target_fingerprint: ContentHashSchema,
    measured_input_tokens: NonnegativeSafeIntegerSchema,
    tokenizer_id: IdentifierSchema,
    measurement_fingerprint: ContentHashSchema,
    recorded_at: TimestampSchema,
});
export type IndexedProcessingCoverageCommand = z.infer<typeof IndexedProcessingCoverageCommandSchema>;

/** Internal data command. The host owns the current head and selected policy authority. */
export const IndexedProcessingPolicyCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
    enabled: z.boolean(),
    processors: z.array(ProcessorConfigurationSchema).max(16),
    budget: ProcessingBudgetSchema.optional(),
    supersede_job_ids: z.array(IdentifierSchema).max(256).optional(),
    supersession_reason: IdentifierSchema.optional(),
});
export type IndexedProcessingPolicyCommand = z.infer<typeof IndexedProcessingPolicyCommandSchema>;

/** Exact host-selected current entries, not a caller-supplied resolved plan or model proof. */
export const IndexedProcessingQueueCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    expected_context_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
    processor_id: IdentifierSchema,
    scope: z.enum(['manual', 'on_budget']),
    selected_entry_ids: z.array(IdentifierSchema).min(1).max(INDEXED_PROCESSING_SELECTED_MAX_BLOCKS),
    selected_block_ids: z.record(IdentifierSchema, z.array(IdentifierSchema).min(1)).optional(),
    target_fingerprint: ContentHashSchema.optional(),
});
export type IndexedProcessingQueueCommand = z.infer<typeof IndexedProcessingQueueCommandSchema>;
