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
        'accepted_output_fragments',
        'prepared_request_records',
    ]),
});

export const CONVERSATION_FOUNDATION_LIMITATIONS = Object.freeze([
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'segmented_storage',
        message:
            'Manifest, segment, index-page, and bounded working-set contracts are not implemented in this revision.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'general_fragments_and_working_sets',
        message:
            'Accepted-output fragments are supported, but general loadable history fragments and bounded working-set validation are not implemented.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'changes_and_editing',
        message: 'Change, editing, conflict, merge, and fork contracts are not implemented in this revision.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'processing_runtime',
        message: 'Persisted processing and compaction data is inert and has no jobs, readiness, or execution.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'remaining_delivery_adapters_and_migrations',
        message:
            'Chat Completions, Responses, Claude Messages, Gemini GenerateContent, and Bedrock Converse have adapters; remaining modalities, delivery contracts, and migrations are not implemented.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'stable_publication',
        message: 'Schema version 1, core-preview acceptance, and npm publication gates have not been completed.',
    },
] as const);

export type ConversationFoundationLimitation = (typeof CONVERSATION_FOUNDATION_LIMITATIONS)[number];
