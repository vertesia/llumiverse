import { z } from 'zod';
import {
    ImageRegionSchema,
    JsonPointerSchema,
    PageRangeSchema,
    TextCodePointRangeSchema,
    TimeRangeSchema,
} from './content-ranges.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
} from './primitives.js';

export const JsonSourceRegionSchema = z
    .strictObject({
        pointer: JsonPointerSchema,
        excluded_pointers: z.array(JsonPointerSchema).max(4096),
    })
    .meta({ id: 'ConversationJsonSourceRegion' });
export const SourceBlockSliceSchema = z
    .strictObject({
        source: ConversationRefSchema,
        turn_id: IdentifierSchema,
        block_id: IdentifierSchema,
        block_fingerprint: ContentHashSchema,
        selection: z.discriminatedUnion('kind', [
            z.strictObject({ kind: z.literal('whole') }),
            z.strictObject({ kind: z.literal('text_range'), range: TextCodePointRangeSchema }),
            z.strictObject({ kind: z.literal('json_region'), region: JsonSourceRegionSchema }),
            z.strictObject({
                kind: z.literal('media_range'),
                range: z.discriminatedUnion('type', [ImageRegionSchema, PageRangeSchema, TimeRangeSchema]),
                asset_id: IdentifierSchema,
                asset_metadata_fingerprint: ContentHashSchema,
                verified_content_hash: ContentHashSchema,
            }),
        ]),
    })
    .meta({ id: 'ConversationSourceBlockSlice' });
export const JsonInverseMappingSchema = z
    .strictObject({
        root_source_pointer: JsonPointerSchema,
        arrays: z
            .array(
                z.strictObject({
                    target_pointer: JsonPointerSchema,
                    source_pointer: JsonPointerSchema,
                    runs: z
                        .array(
                            z.strictObject({
                                target_start: NonnegativeSafeIntegerSchema,
                                source_start: NonnegativeSafeIntegerSchema,
                                length: PositiveSafeIntegerSchema,
                            }),
                        )
                        .max(4096),
                }),
            )
            .max(4096),
    })
    .meta({ id: 'ConversationJsonInverseMapping' });
const common = {
    target_block_ids: z.array(IdentifierSchema).min(1).max(4096),
    source_slices: z.array(SourceBlockSliceSchema).min(1).max(4096),
};
export const DerivedBlockLineageGroupSchema = z
    .discriminatedUnion('transform', [
        z.strictObject({
            target_block_ids: z.array(IdentifierSchema).length(1),
            source_slices: z.array(SourceBlockSliceSchema).length(1),
            transform: z.literal('json_minification'),
            parser: z.literal('rfc8259-lexical-v1'),
            fidelity: z.literal('value_preserving'),
        }),
        z.strictObject({ ...common, transform: z.literal('block_copy'), fidelity: z.literal('value_preserving') }),
        z.strictObject({ ...common, transform: z.literal('text_slice'), fidelity: z.literal('value_preserving') }),
        z.strictObject({
            ...common,
            transform: z.literal('json_projection'),
            fidelity: z.literal('value_preserving'),
            inverse: JsonInverseMappingSchema,
        }),
        z.strictObject({ ...common, transform: z.literal('media_reference'), fidelity: z.literal('value_preserving') }),
        z.strictObject({
            ...common,
            transform: z.literal('authored_replacement'),
            fidelity: z.enum(['heuristic', 'semantic']),
        }),
    ])
    .meta({ id: 'ConversationDerivedBlockLineageGroup' });
export const DerivedBlockLineageSchema = z
    .strictObject({
        version: z.literal(1),
        groups: z.array(DerivedBlockLineageGroupSchema).min(1).max(4096),
    })
    .meta({ id: 'ConversationDerivedBlockLineage' });

/** Explicit roots for a host-resolved bounded dependency view; absent scope verifies the whole snapshot. */
export const DerivedLineageVerificationScopeSchema = z
    .strictObject({ turn_ids: z.array(IdentifierSchema).min(1).max(4096) })
    .meta({ id: 'ConversationDerivedLineageVerificationScope' });
