import { z } from 'zod';
import {
    AudioBlockSchema,
    DocumentBlockSchema,
    ImageBlockSchema,
    ImageRegionSchema,
    JsonBlockSchema,
    PageRangeSchema,
    TextBlockSchema,
    TimeRangeSchema,
    VideoBlockSchema,
} from './content.js';
import { JsonPointerSchema, TextCodePointRangeSchema } from './content-ranges.js';
import {
    ContextSelectionBlockTypeSchema,
    ContextSelectionRequestSchema,
    ContextSelectorSchema,
} from './context-selection.js';
import { ConversationDiagnosticSchema } from './diagnostics.js';
import { ContextEntrySchema } from './document.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
} from './primitives.js';

export { JsonPointerSchema, TextCodePointRangeSchema } from './content-ranges.js';

export const MediaSelectionRangeSchema = z
    .discriminatedUnion('type', [ImageRegionSchema, PageRangeSchema, TimeRangeSchema])
    .meta({ id: 'ConversationMediaSelectionRange' });
const sourceShape = {
    entry_id: IdentifierSchema,
    block_id: IdentifierSchema,
    expected_block_fingerprint: ContentHashSchema,
};
export const BlockSubselectionSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ ...sourceShape, kind: z.literal('text_range'), range: TextCodePointRangeSchema }),
        z.strictObject({ ...sourceShape, kind: z.literal('json_pointer'), pointer: JsonPointerSchema }),
        z.strictObject({
            ...sourceShape,
            kind: z.literal('media_range'),
            range: MediaSelectionRangeSchema,
            expected_asset_metadata_fingerprint: ContentHashSchema,
        }),
    ])
    .meta({ id: 'ConversationBlockSubselection' });
export const ConversationSelectorSchema = ContextSelectorSchema.extend({
    /** Refine already-selected blocks only. Unspecified selected blocks remain whole. */
    subselections: z.array(BlockSubselectionSchema).min(1).max(4096).optional(),
}).meta({ id: 'ConversationSelector' });
export const ConversationSelectionRequestSchema = ContextSelectionRequestSchema.extend({
    selector: ConversationSelectorSchema,
}).meta({ id: 'ConversationSelectionRequest' });

export const SelectionBinaryEvidenceSchema = z
    .discriminatedUnion('status', [
        z.strictObject({
            status: z.literal('unverified'),
            reason: z.enum(['external_content', 'unencodable_inline_text']),
            declared_content_hash: ContentHashSchema.optional(),
            declared_byte_length: NonnegativeSafeIntegerSchema.optional(),
        }),
        z.strictObject({
            status: z.literal('verified_inline'),
            basis: z.enum(['stored_base64_bytes', 'stored_text_utf8', 'canonical_json_utf8']),
            content_hash: ContentHashSchema,
            byte_length: NonnegativeSafeIntegerSchema,
        }),
        z.strictObject({
            status: z.literal('mismatch'),
            basis: z.enum(['stored_base64_bytes', 'stored_text_utf8', 'canonical_json_utf8']),
            computed_content_hash: ContentHashSchema,
            computed_byte_length: NonnegativeSafeIntegerSchema,
            declared_content_hash: ContentHashSchema.optional(),
            declared_byte_length: NonnegativeSafeIntegerSchema.optional(),
        }),
    ])
    .meta({ id: 'ConversationSelectionBinaryEvidence' });
export const SelectionMediaEvidenceSchema = z
    .strictObject({
        asset_id: IdentifierSchema,
        source_metadata_fingerprint: ContentHashSchema,
        /** Declared metadata bounds are not an independent decoder verification. */
        extent: z.enum(['declared', 'unknown']),
        content: SelectionBinaryEvidenceSchema,
        /** Read evidence does not perform mutation policy/readiness admission. */
        mutation_ready: z.literal(false),
    })
    .meta({ id: 'ConversationSelectionMediaEvidence' });
const blockShape = { block_id: IdentifierSchema, block_fingerprint: ContentHashSchema };
export const SelectedBlockSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ ...blockShape, kind: z.literal('whole'), block_type: ContextSelectionBlockTypeSchema }),
        z.strictObject({
            ...blockShape,
            kind: z.literal('text_range'),
            block_type: TextBlockSchema.shape.type,
            range: TextCodePointRangeSchema,
        }),
        z.strictObject({
            ...blockShape,
            kind: z.literal('json_pointer'),
            block_type: JsonBlockSchema.shape.type,
            pointer: JsonPointerSchema,
        }),
        z.discriminatedUnion('block_type', [
            z.strictObject({
                ...blockShape,
                kind: z.literal('media_range'),
                block_type: ImageBlockSchema.shape.type,
                range: ImageRegionSchema,
                evidence: SelectionMediaEvidenceSchema,
            }),
            z.strictObject({
                ...blockShape,
                kind: z.literal('media_range'),
                block_type: DocumentBlockSchema.shape.type,
                range: PageRangeSchema,
                evidence: SelectionMediaEvidenceSchema,
            }),
            z.strictObject({
                ...blockShape,
                kind: z.literal('media_range'),
                block_type: AudioBlockSchema.shape.type,
                range: TimeRangeSchema,
                evidence: SelectionMediaEvidenceSchema,
            }),
            z.strictObject({
                ...blockShape,
                kind: z.literal('media_range'),
                block_type: VideoBlockSchema.shape.type,
                range: TimeRangeSchema,
                evidence: SelectionMediaEvidenceSchema,
            }),
        ]),
    ])
    .meta({ id: 'ConversationSelectedBlock' });
export const SelectedContextEntrySchema = z
    .strictObject({
        /** Exact original entry reference. Only blocks below are selected, not every block of this entry. */
        entry: ContextEntrySchema,
        blocks: z.array(SelectedBlockSchema),
    })
    .meta({ id: 'ConversationSelectedContextEntry' });
export const ConversationSelectionSchema = z
    .strictObject({
        conversation: ConversationRefSchema,
        context_revision: NonnegativeSafeIntegerSchema,
        source_fingerprint: ContentHashSchema,
        access: z.literal('read_only'),
        entries: z.array(SelectedContextEntrySchema).min(1),
    })
    .meta({ id: 'ConversationSelection' });
const noMatchSchema = z.strictObject({
    kind: z.literal('no_match'),
    conversation: ConversationRefSchema,
    context_revision: NonnegativeSafeIntegerSchema,
    source_fingerprint: ContentHashSchema,
    diagnostics: z.array(ConversationDiagnosticSchema).length(0),
});
const rejectedSchema = z.strictObject({
    kind: z.literal('rejected'),
    diagnostics: z.array(ConversationDiagnosticSchema).min(1),
});
export const ConversationSelectionResultSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({
            kind: z.literal('selected'),
            selection: ConversationSelectionSchema,
            diagnostics: z.array(ConversationDiagnosticSchema).length(0),
        }),
        noMatchSchema,
        rejectedSchema,
    ])
    .meta({ id: 'ConversationSelectionResult' });
export const ConversationSliceSchema = z
    .strictObject({
        kind: z.literal('context_view'),
        selection: ConversationSelectionSchema,
    })
    .meta({ id: 'ConversationSlice' });
export const ConversationSliceResultSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({
            kind: z.literal('selected'),
            view: ConversationSliceSchema,
            diagnostics: z.array(ConversationDiagnosticSchema).length(0),
        }),
        noMatchSchema,
        rejectedSchema,
    ])
    .meta({ id: 'ConversationSliceResult' });
