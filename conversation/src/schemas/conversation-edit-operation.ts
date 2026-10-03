import { z } from 'zod';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
} from './primitives.js';
import { SourceBlockSliceSchema } from './source-slices.js';

/** Content-addressed canonical evidence; not another copy of historical conversation content. */
export const ConversationEditRecordRefSchema = z
    .strictObject({
        id: IdentifierSchema,
        fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationEditRecordRef' });
export const ConversationEditAnchorSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('before_entry'), entry_id: IdentifierSchema }),
        z.strictObject({ kind: z.literal('after_entry'), entry_id: IdentifierSchema }),
        z.strictObject({ kind: z.literal('head') }),
        z.strictObject({ kind: z.literal('tail') }),
    ])
    .meta({ id: 'ConversationEditAnchor' });
export const ConversationEditPlacementSchema = z
    .strictObject({
        mode: z.literal('first_selected'),
        causal_order: z.enum(['contiguous', 'explicit_disjoint']),
    })
    .meta({ id: 'ConversationEditPlacement' });
const operationShape = {
    version: z.literal(1),
    source: ConversationRefSchema,
    source_context_revision: NonnegativeSafeIntegerSchema,
    source_fingerprint: ContentHashSchema,
    selected_entries: z.array(ConversationEditRecordRefSchema),
    removed_entry_ids: z.array(IdentifierSchema),
    created_entries: z.array(ConversationEditRecordRefSchema),
    created_turns: z.array(ConversationEditRecordRefSchema),
    created_assets: z.array(ConversationEditRecordRefSchema),
    protected_entry_ids: z.array(IdentifierSchema),
    unprotected_entry_ids: z.array(IdentifierSchema),
};
/** Retain declared fidelity and actual causal placement as durable replacement evidence. */
export const ConversationEditOperationV1Schema = z
    .discriminatedUnion('kind', [
        z.strictObject({ ...operationShape, kind: z.literal('protect') }),
        z.strictObject({ ...operationShape, kind: z.literal('insert'), anchor: ConversationEditAnchorSchema }),
        z.strictObject({
            ...operationShape,
            kind: z.literal('replace'),
            fidelity: z.enum(['heuristic', 'semantic']),
            placement: ConversationEditPlacementSchema,
        }),
    ])
    .meta({ id: 'ConversationEditOperationV1' });

const sliceOperationShape = {
    ...operationShape,
    version: z.literal(2),
    source_slices: z.array(SourceBlockSliceSchema).min(1).max(4096),
    remainder_turn_ids: z.array(IdentifierSchema).max(4096),
    source_protected_entry_ids: z.array(IdentifierSchema).max(4096),
    source_entry_positions: z.array(NonnegativeSafeIntegerSchema).min(1).max(4096),
    source_topology_fingerprint: ContentHashSchema,
};
export const ConversationSliceEditOperationSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ ...sliceOperationShape, kind: z.literal('protect') }),
        z.strictObject({
            ...sliceOperationShape,
            kind: z.literal('replace'),
            fidelity: z.enum(['heuristic', 'semantic']),
            placement: ConversationEditPlacementSchema,
        }),
    ])
    .meta({ id: 'ConversationSliceEditOperation' });
/** Version 1 record values and fingerprinting are unchanged; precise cuts use explicit version 2. */
export const ConversationEditOperationSchema = z
    .union([ConversationEditOperationV1Schema, ConversationSliceEditOperationSchema])
    .meta({ id: 'ConversationEditOperation' });
