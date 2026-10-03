import { z } from 'zod';
import {
    AssetSchema,
    ContentBlockSchema,
    DerivedAgentTurnSchema,
    GeneratedAgentTurnSchema,
    ImportedAgentTurnSchema,
    NongeneratedAgentTurnSchema,
    ProgramTurnSchema,
    ToolDefinitionSchema,
    ToolTurnSchema,
    UserTurnSchema,
} from './content.js';
import { ContextEntrySchema } from './context-foundation.js';
import { ConversationContextSchema } from './document.js';
import { RequestReceiptSchema } from './execution.js';
import { ContentHashSchema, ConversationRefSchema, IdentifierSchema } from './primitives.js';

export const SOURCE_VIEW_MANIFEST_MAX_BYTES = 256 * 1024;
export const SOURCE_VIEW_SEGMENT_MAX_BYTES = 1024 * 1024;
export const SOURCE_VIEW_INLINE_MAX_BYTES = 64 * 1024;
export const SOURCE_VIEW_WORKING_SET_MAX_BYTES = 32 * 1024 * 1024;
export const SOURCE_VIEW_MAX_SEGMENTS = 128;
export const SOURCE_VIEW_MAX_INDEX_PAGES = 128;

const BoundedStorageKeySchema = z.string().min(1).max(512);
const Sha256Schema = z.string().regex(/^sha256:[0-9a-f]{64}$/);

export const RequestSourceViewSegmentSchema = z.strictObject({
    storage_key: BoundedStorageKeySchema,
    content_hash: Sha256Schema,
    size_bytes: z.number().int().positive().max(SOURCE_VIEW_SEGMENT_MAX_BYTES),
});

export const RequestSourceViewRecordKindSchema = z.enum([
    'context',
    'source_turn',
    'replacement_turn',
    'asset',
    'tool_definition',
]);

export const RequestSourceViewRecordSchema = z.strictObject({
    kind: RequestSourceViewRecordKindSchema,
    id: IdentifierSchema,
    content_hash: Sha256Schema,
    size_bytes: z.number().int().positive().max(SOURCE_VIEW_WORKING_SET_MAX_BYTES),
    parts: z
        .array(
            z.strictObject({
                segment_index: z
                    .number()
                    .int()
                    .nonnegative()
                    .max(SOURCE_VIEW_MAX_SEGMENTS - 1),
                offset: z
                    .number()
                    .int()
                    .nonnegative()
                    .max(SOURCE_VIEW_SEGMENT_MAX_BYTES - 1),
                length: z.number().int().positive().max(SOURCE_VIEW_SEGMENT_MAX_BYTES),
            }),
        )
        .min(1)
        .max(SOURCE_VIEW_MAX_SEGMENTS),
    compaction_id: IdentifierSchema.optional(),
});

export const RequestSourceViewIndexPageSchema = z.strictObject({
    version: z.literal(1),
    records: z.array(RequestSourceViewRecordSchema).min(1),
});

/** This manifest names only selected records. It is not a sparse ConversationDocument. */
const requestSourceViewManifestFields = {
    completeness: z.literal('selected_content_unverified'),
    source: ConversationRefSchema,
    context_revision: z.number().int().nonnegative(),
    request_receipt_id: IdentifierSchema,
    request_fingerprint: ContentHashSchema,
    context_fingerprint: ContentHashSchema,
    selected_entry_ids_hash: Sha256Schema,
    active_tool_definition_ids_hash: Sha256Schema,
    record_count: z.number().int().positive().max(100_000),
    segments: z.array(RequestSourceViewSegmentSchema).min(1).max(SOURCE_VIEW_MAX_SEGMENTS),
    index_pages: z.array(RequestSourceViewSegmentSchema).min(1).max(SOURCE_VIEW_MAX_INDEX_PAGES),
};

/** Version 1 is the exact historical private manifest shape and remains parseable for inspection only. */
export const RequestSourceViewManifestSchema = z.discriminatedUnion('version', [
    z.strictObject({ ...requestSourceViewManifestFields, version: z.literal(1) }),
    z.strictObject({
        ...requestSourceViewManifestFields,
        version: z.literal(2),
        /** Hash of the complete accepted receipt except its self-referential source_view locator. */
        accepted_request_binding_hash: Sha256Schema,
        /** Covers runtime and derived generation/response identities, excluding only the self-referential locator. */
        accepted_prepared_record_binding_hash: Sha256Schema,
        /** Attested by the authenticated prepared-record CAS after full-document validation. */
        validated_document_hash: Sha256Schema,
        validator_profile: z.literal('llumiverse.conversation/full-materialized/2026-09-30.adoption.1'),
    }),
]);

export const ProjectedTurnHeaderSchema = z.union([
    UserTurnSchema.omit({ blocks: true }),
    GeneratedAgentTurnSchema.omit({ blocks: true }),
    ImportedAgentTurnSchema.omit({ blocks: true }),
    DerivedAgentTurnSchema.omit({ blocks: true }),
    NongeneratedAgentTurnSchema.omit({ blocks: true }),
    ToolTurnSchema.omit({ blocks: true }),
    ProgramTurnSchema.omit({ blocks: true }),
]);

/** The selected blocks are a projection, not an assertion that omitted blocks never existed. */
export const RequestSourceProjectedTurnSchema = z
    .strictObject({
        completeness: z.enum(['full_turn', 'selected_blocks']),
        header: ProjectedTurnHeaderSchema,
        selected_blocks: z.array(ContentBlockSchema),
        /** Original zero-based positions; omitted blocks retain their positions and are never renumbered. */
        selected_block_positions: z.array(z.number().int().nonnegative()),
        source_block_count: z.number().int().nonnegative(),
        source_block_ids_hash: Sha256Schema,
    })
    .superRefine((projection, context) => {
        if (projection.selected_block_positions.length !== projection.selected_blocks.length) {
            context.addIssue({ code: 'custom', message: 'Selected block positions differ from selected blocks' });
        }
        let previous = -1;
        for (const position of projection.selected_block_positions) {
            if (position <= previous || position >= projection.source_block_count) {
                context.addIssue({
                    code: 'custom',
                    message: 'Selected block positions are not strictly source ordered',
                });
            }
            previous = position;
        }
        if (
            projection.completeness === 'full_turn' &&
            (projection.source_block_count !== projection.selected_blocks.length ||
                projection.selected_block_positions.some((position, index) => position !== index))
        ) {
            context.addIssue({ code: 'custom', message: 'Full turn projection omits original block positions' });
        }
    });

/** A selected execution view is intentionally not a ConversationDocument or a general history fragment. */
export const RequestSourceWorkingSetSchema = z.strictObject({
    completeness: z.literal('selected_content_unverified'),
    /** Private archive format version. Only version 2 binds the complete accepted prepared record. */
    archive_version: z.union([z.literal(1), z.literal(2)]).optional(),
    accepted_prepared_record_binding_hash: Sha256Schema.optional(),
    source: ConversationRefSchema,
    context: ConversationContextSchema,
    request_receipt: RequestReceiptSchema,
    turns: z.array(RequestSourceProjectedTurnSchema),
    replacement_turns: z.array(
        z.strictObject({ compaction_id: IdentifierSchema, projection: RequestSourceProjectedTurnSchema }),
    ),
    assets: z.record(IdentifierSchema, AssetSchema),
    tool_definitions: z.record(IdentifierSchema, ToolDefinitionSchema),
    selected_entries: z.array(ContextEntrySchema),
});
