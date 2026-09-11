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
        feature: 'fragments',
        message: 'Fragment validation and completeness descriptors are not implemented in this revision.',
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
        feature: 'delivery_and_adapters',
        message: 'Delivery, adapter, and migration contracts are not implemented in this revision.',
    },
    {
        code: 'UNIMPLEMENTED_SCOPE',
        feature: 'stable_publication',
        message: 'Schema version 1, core-preview acceptance, and npm publication gates have not been completed.',
    },
] as const);

export type ConversationFoundationLimitation = (typeof CONVERSATION_FOUNDATION_LIMITATIONS)[number];
