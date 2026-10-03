import { z } from 'zod';
import { AssetSchema } from './content.js';
import { ConversationInsertedTurnSchema } from './conversation-edit.js';
import { ConversationEditPlacementSchema } from './conversation-edit-operation.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';
import { ConversationSelectionSchema } from './selection.js';

/** Callers provide safe authored content, never fabricated derived lineage or target execution. */
export const ConversationSliceEditCommandSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('protect'), selection: ConversationSelectionSchema, protected: z.boolean() }),
        z.strictObject({
            kind: z.literal('replace'),
            selection: ConversationSelectionSchema,
            replacement_turn: ConversationInsertedTurnSchema,
            assets: z.array(AssetSchema).optional(),
            placement: ConversationEditPlacementSchema,
            fidelity: z.enum(['heuristic', 'semantic']),
        }),
    ])
    .meta({ id: 'ConversationSliceEditCommand' });
const shape = {
    version: z.literal(2),
    operation_id: IdentifierSchema,
    conversation: ConversationRefSchema,
    expected_context_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
    command: ConversationSliceEditCommandSchema,
};
export const ConversationSliceEditPlanInputSchema = z
    .strictObject(shape)
    .meta({ id: 'ConversationSliceEditPlanInput' });
export const ConversationSliceEditRequestSchema = z
    .strictObject({ ...shape, expected_source_fingerprint: ContentHashSchema })
    .meta({ id: 'ConversationSliceEditRequest' });
