import { z } from 'zod';
import { ConversationDeleteChangeSchema } from './change.js';
import { ConversationDeleteOperationSchema } from './conversation-delete-operation.js';
import { ConversationDiagnosticSchema } from './diagnostics.js';
import { ConversationDocumentSchema } from './document.js';
import { ContentHashSchema, ConversationRefSchema, IdentifierSchema, TimestampSchema } from './primitives.js';

const inputShape = {
    version: z.literal(1),
    operation_id: IdentifierSchema,
    conversation: ConversationRefSchema,
    recorded_at: TimestampSchema,
    dependency_policy: z.literal('reject'),
    turn_ids: z.array(IdentifierSchema).min(1).max(4096),
};

export const ConversationDeletePlanInputSchema = z.strictObject(inputShape).meta({ id: 'ConversationDeletePlanInput' });
export const ConversationDeleteRequestSchema = z
    .strictObject({ ...inputShape, expected_source_fingerprint: ContentHashSchema })
    .meta({ id: 'ConversationDeleteRequest' });
export const ConversationDeletePlanSchema = z
    .strictObject({
        operation: ConversationDeleteOperationSchema,
        diagnostics: z.array(ConversationDiagnosticSchema).length(0),
    })
    .meta({ id: 'ConversationDeletePlan' });
export const ConversationDeleteResultSchema = z
    .strictObject({
        document: ConversationDocumentSchema,
        change: ConversationDeleteChangeSchema,
        applied: z.boolean(),
    })
    .meta({ id: 'ConversationDeleteResult' });
