/** Browser-safe canonical wire constants shared by schemas and runtime validators. */
export const CONVERSATION_FORMAT = 'llumiverse.conversation' as const;
export const CONVERSATION_SCHEMA_VERSION = 0 as const;
export const CONVERSATION_EXPERIMENTAL_REVISION = '2026-09-30.adoption.1' as const;

export const DIAGNOSTIC_PATH_MAX_LENGTH = 512 as const;
export const DIAGNOSTIC_MESSAGE_MAX_LENGTH = 1_024 as const;
export const DIAGNOSTIC_RECORD_ID_MAX_LENGTH = 512 as const;
export const DIAGNOSTIC_RELATED_PATHS_MAX_LENGTH = 8 as const;

export const CONVERSATION_USAGE_METRICS = [
    'input_tokens',
    'output_tokens',
    'total_tokens',
    'reasoning_tokens',
    'cache_read_tokens',
    'cache_write_tokens',
    'input_new_tokens',
] as const;

/** Browser-safe processing work bounds; schemas and semantic validation share these exact values. */
export const MAX_PROCESSING_OUTPUT_BYTES = 1024 * 1024;
export const MAX_PROCESSOR_CONFIGURATION_BYTES = 64 * 1024;
/** RFC 6901 grammar, shared by the public schema and schema-free runtime validation. */
export const JSON_POINTER_PATTERN_SOURCE = '^(?:/(?:[^~]|~[01])*)?$';
