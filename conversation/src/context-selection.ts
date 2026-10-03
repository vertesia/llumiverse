import { planContextChange } from './context-change.js';
import { rejectContextSelection as reject, resolveActiveContextSelection } from './context-selection-resolution.js';
import { ConversationValidationError } from './diagnostics.js';
import { fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { ContextSelectionRequestSchema, ContextSelectionResultSchema } from './schemas/context-selection.js';
import type {
    ContextChangePlan,
    ContextSelectionRequest,
    ContextSelectionResult,
    ConversationDiagnostic,
    ConversationDocument,
} from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

/** Deterministic whole-block edit-eligible resolution over materialized active context; never expands dependency cuts. */
export async function resolveContextSelection(
    sourceInput: ConversationDocument,
    requestInput: ContextSelectionRequest,
): Promise<ContextSelectionResult> {
    try {
        const preflight = preflightJsonInput(requestInput);
        if (!preflight.success)
            throw new ConversationValidationError('Selection failed JSON preflight', preflight.diagnostics);
        const parsed = ContextSelectionRequestSchema.safeParse(requestInput);
        if (!parsed.success)
            throw new ConversationValidationError(
                'Selection failed schema validation',
                diagnosticsFromZodError(parsed.error),
            );
        const request = parsed.data;
        const document = parseConversationDocument(sourceInput);
        if (
            request.conversation.conversation_id !== document.id ||
            request.conversation.revision !== document.revision ||
            request.expected_context_revision !== document.context.revision
        )
            reject('Selection snapshot revision conflict');
        const { entryIds, selectedEntries, blockIds, partial } = resolveActiveContextSelection(
            document,
            request.selector,
        );
        const identity = { conversation_id: document.id, revision: document.revision };
        if (!entryIds.length)
            return ContextSelectionResultSchema.parse({
                kind: 'no_match',
                conversation: identity,
                context_revision: document.context.revision,
                source_fingerprint: await fingerprintJson({ document, selector: request.selector }),
                diagnostics: [],
            });
        let plan: ContextChangePlan;
        try {
            plan = await planContextChange(document, {
                expected_revision: document.revision,
                expected_context_revision: document.context.revision,
                entry_ids: entryIds,
                ...(partial ? { selected_block_ids: blockIds, selected_entries: selectedEntries } : {}),
            });
        } catch (error) {
            if (error instanceof ConversationValidationError) throw error;
            // The planner refuses cuts; it never repairs/expands the caller's selection.
            if (error instanceof Error) reject(error.message);
            throw error;
        }
        return ContextSelectionResultSchema.parse({
            kind: 'selected',
            conversation: identity,
            context_revision: document.context.revision,
            plan,
            diagnostics: [],
        });
    } catch (error) {
        if (error instanceof ConversationValidationError) {
            const diagnostics: ConversationDiagnostic[] = error.diagnostics;
            return ContextSelectionResultSchema.parse({ kind: 'rejected', diagnostics });
        }
        throw error;
    }
}
