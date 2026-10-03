import { CONVERSATION_EXPERIMENTAL_REVISION, CONVERSATION_SCHEMA_VERSION } from './schemas/primitives.js';

export const CONVERSATION_FOUNDATION_SCOPE = Object.freeze({
    schema_version: CONVERSATION_SCHEMA_VERSION,
    experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
    supported: Object.freeze([
        'materialized_document_schema',
        'bounded_json_preflight',
        'full_document_shape_and_semantic_validation',
        'materialized_json_round_trip',
        'basic_builders_and_inspection',
        'deterministic_json_schema',
        'idempotent_materialized_record_ingestion',
        'revisioned_active_context_selection_changes',
        'dependency_rejecting_whole_source_turn_logical_delete',
        'durable_processing_job_staging_and_injected_runner',
        'accepted_output_fragments',
        'bounded_safe_transcript_views',
        'prepared_request_records',
        'bounded_selected_request_archive',
        'experimental_indexed_head_and_text_selection',
        'bounded_stream_event_contracts_and_reconciliation',
    ]),
});

export const CONVERSATION_FOUNDATION_LIMITATIONS = Object.freeze([
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'segmented_storage',
        message:
            'Selected-request archives and an experimental paged run-head format have bounded manifests, segments and indexes. One-time 100k-turn migration has a finite full-input profile and pure append read-count evidence. Full indexed append/prepare registration, all consumers, compaction/tool/media witness closure, and authenticated host 10k/100k accepted-head benchmarks remain incomplete.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'general_fragments_and_working_sets',
        message:
            'Accepted-output fragments, safe transcript views and selected-content archives are supported. A selected archive is not a complete execution witness; general loadable history fragments and sparse mutation/revalidation remain incomplete.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'changes_and_editing',
        message:
            'Materialized active-context exclusion, compaction replacement, protect/unprotect, whole-block edits, bounded source-slice edits, and dependency-rejecting whole-source-turn logical deletion are supported. Partial/cascading deletion, indexed tombstone storage, merge, fork, rebase and full indexed mutation are not implemented.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'processing_runtime',
        message:
            'Bounded processor jobs, immutable stage inputs and outputs, injected plugin execution, and scoped readiness are implemented as a core engine. A production host adapter, complete ingestion coverage, prepare-time wait/CAN wiring, target measurement, strategy catalog, budget pass/cost limits, and segmented storage remain unimplemented.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'remaining_delivery_adapters_and_migrations',
        message:
            'Chat Completions, Responses, Claude Messages, Gemini GenerateContent, and Bedrock Converse have adapters; remaining modalities, provider typed-event production, durable delivery services, and migrations are not implemented.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'stable_publication',
        message: 'Schema version 1, core-preview acceptance, and npm publication gates have not been completed.',
    },
] as const);

export type ConversationFoundationLimitation = (typeof CONVERSATION_FOUNDATION_LIMITATIONS)[number];
