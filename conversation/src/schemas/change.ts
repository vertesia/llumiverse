import { z } from 'zod';
import { ConversationDeleteOperationSchema } from './conversation-delete-operation.js';
import { ConversationEditOperationSchema } from './conversation-edit-operation.js';
import { ConversationDiagnosticSchema } from './diagnostics.js';
import { AcceptedToolSelectionSchema, ContextChangeOperationSchema } from './execution.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema } from './primitives.js';
import { ProcessingChangeOperationSchema } from './processing-operation.js';

const envelopeShape = {
    operation_id: IdentifierSchema,
    conversation_id: IdentifierSchema,
    base_revision: NonnegativeSafeIntegerSchema,
    result_revision: NonnegativeSafeIntegerSchema,
    diagnostics: z.array(ConversationDiagnosticSchema),
};
/** One canonical envelope; each current atomic mutation commits one ordered operation. */
export function createConversationChangeSchema<Operation extends z.ZodType>(operation: Operation, id: string) {
    return z.strictObject({ ...envelopeShape, operations: z.array(operation).length(1) }).meta({ id });
}
export const ConversationAppendOperationSchema = z
    .strictObject({
        kind: z.literal('append'),
        payload_fingerprint: ContentHashSchema,
        tool_selection: AcceptedToolSelectionSchema.optional(),
        accepted_turn_ids: z.array(IdentifierSchema),
        accepted_generation_ids: z.array(IdentifierSchema),
        accepted_asset_ids: z.array(IdentifierSchema),
        accepted_tool_definition_ids: z.array(IdentifierSchema),
        accepted_execution_receipt_ids: z.array(IdentifierSchema),
        accepted_context_entry_ids: z.array(IdentifierSchema),
    })
    .meta({ id: 'ConversationAppendOperation' });
export const ContextChangeSchema = createConversationChangeSchema(
    ContextChangeOperationSchema,
    'ConversationContextChange',
);
export const ConversationEditChangeSchema = createConversationChangeSchema(
    ConversationEditOperationSchema,
    'ConversationEditChange',
);
export const ConversationDeleteChangeSchema = createConversationChangeSchema(
    ConversationDeleteOperationSchema,
    'ConversationDeleteChange',
);
export const ConversationAppendChangeSchema = createConversationChangeSchema(
    ConversationAppendOperationSchema,
    'ConversationAppendChange',
);
export const ConversationChangeSchema = createConversationChangeSchema(
    z.union([
        ContextChangeOperationSchema,
        ConversationEditOperationSchema,
        ConversationDeleteOperationSchema,
        ConversationAppendOperationSchema,
        ProcessingChangeOperationSchema,
    ]),
    'ConversationChange',
);
