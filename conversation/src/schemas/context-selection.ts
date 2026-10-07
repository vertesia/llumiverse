import { z } from 'zod';
import { ContextChangePlanSchema } from './context-change.js';
import { ConversationDiagnosticSchema } from './diagnostics.js';
import { ContentHashSchema, ConversationRefSchema, NonnegativeSafeIntegerSchema } from './primitives.js';

export * from './context-selection-request.js';

export const ContextSelectionResultSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({
            kind: z.literal('selected'),
            conversation: ConversationRefSchema,
            context_revision: NonnegativeSafeIntegerSchema,
            plan: ContextChangePlanSchema,
            diagnostics: z.array(ConversationDiagnosticSchema).length(0),
        }),
        z.strictObject({
            kind: z.literal('no_match'),
            conversation: ConversationRefSchema,
            context_revision: NonnegativeSafeIntegerSchema,
            source_fingerprint: ContentHashSchema,
            diagnostics: z.array(ConversationDiagnosticSchema).length(0),
        }),
        z.strictObject({ kind: z.literal('rejected'), diagnostics: z.array(ConversationDiagnosticSchema).min(1) }),
    ])
    .meta({ id: 'ConversationContextSelectionResult' });
