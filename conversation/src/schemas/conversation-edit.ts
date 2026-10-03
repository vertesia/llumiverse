import { z } from 'zod';
import { ConversationEditChangeSchema } from './change.js';
import {
    AssetSchema,
    AudioBlockSchema,
    DerivedTurnProvenanceSchema,
    DocumentBlockSchema,
    ImageBlockSchema,
    InsertedTurnProvenanceSchema,
    JsonBlockSchema,
    NongeneratedAgentTurnSchema,
    ProgramTurnSchema,
    TextBlockSchema,
    UserTurnSchema,
    VideoBlockSchema,
} from './content.js';
import {
    ConversationEditAnchorSchema,
    ConversationEditOperationSchema,
    ConversationEditPlacementSchema,
} from './conversation-edit-operation.js';
import { ConversationDiagnosticSchema } from './diagnostics.js';
import { ConversationDocumentSchema } from './document.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './primitives.js';
import { ConversationSelectionSchema } from './selection.js';

/** No executable/provider/replay content or manufactured generation can enter through this edit API. */
export const ConversationEditableBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
    ])
    .meta({ id: 'ConversationEditableBlock' });
const insertedShape = {
    authority: z.literal('ordinary'),
    status: z.literal('completed'),
    blocks: z.array(ConversationEditableBlockSchema).min(1),
    provenance: InsertedTurnProvenanceSchema,
};
export const ConversationInsertedTurnSchema = z
    .discriminatedUnion('kind', [
        UserTurnSchema.omit({ execution_id: true, exchange_id: true }).extend(insertedShape),
        ProgramTurnSchema.omit({ execution_id: true, exchange_id: true }).extend(insertedShape),
        NongeneratedAgentTurnSchema.omit({ execution_id: true, exchange_id: true }).extend(insertedShape),
    ])
    .meta({ id: 'ConversationInsertedTurn' });
const replacementShape = { ...insertedShape, provenance: DerivedTurnProvenanceSchema };
export const ConversationReplacementTurnSchema = z
    .discriminatedUnion('kind', [
        UserTurnSchema.omit({ execution_id: true, exchange_id: true }).extend(replacementShape),
        ProgramTurnSchema.omit({ execution_id: true, exchange_id: true }).extend(replacementShape),
        NongeneratedAgentTurnSchema.omit({ execution_id: true, exchange_id: true }).extend(replacementShape),
    ])
    .meta({ id: 'ConversationReplacementTurn' });
export const ConversationEditCommandSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('protect'), selection: ConversationSelectionSchema, protected: z.boolean() }),
        z.strictObject({
            kind: z.literal('insert'),
            anchor: ConversationEditAnchorSchema,
            turns: z.array(ConversationInsertedTurnSchema).min(1),
            assets: z.array(AssetSchema).optional(),
        }),
        z.strictObject({
            kind: z.literal('replace'),
            selection: ConversationSelectionSchema,
            replacement_turn: ConversationReplacementTurnSchema,
            assets: z.array(AssetSchema).optional(),
            placement: ConversationEditPlacementSchema,
            fidelity: z.enum(['heuristic', 'semantic']),
        }),
    ])
    .meta({ id: 'ConversationEditCommand' });
const inputShape = {
    version: z.literal(1),
    operation_id: IdentifierSchema,
    conversation: ConversationRefSchema,
    expected_context_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
    command: ConversationEditCommandSchema,
};
export const ConversationEditPlanInputSchema = z.strictObject(inputShape).meta({ id: 'ConversationEditPlanInput' });
export const ConversationEditRequestSchema = z
    .strictObject({ ...inputShape, expected_source_fingerprint: ContentHashSchema })
    .meta({ id: 'ConversationEditRequest' });
export const ConversationEditPlanSchema = z
    .strictObject({
        operation: ConversationEditOperationSchema,
        diagnostics: z.array(ConversationDiagnosticSchema).length(0),
    })
    .meta({ id: 'ConversationEditPlan' });
export const ConversationEditResultSchema = z
    .strictObject({
        document: ConversationDocumentSchema,
        change: ConversationEditChangeSchema,
        applied: z.boolean(),
    })
    .meta({ id: 'ConversationEditResult' });
