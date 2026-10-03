import { z } from 'zod';
import { ConversationTurnSchema } from './content.js';
import { CompactionStrategySchema, ContextEntrySchema } from './context-foundation.js';
import { ContextChangePlacementSchema, GenerationSchema, SelectedContextBlocksSchema } from './execution.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema, TimestampSchema } from './primitives.js';

export const ContextChangeProposalSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('exclude') }),
        z.strictObject({
            kind: z.literal('replace_with_compaction'),
            compaction_id: IdentifierSchema,
            strategy: CompactionStrategySchema,
            replacement_turns: z.array(ConversationTurnSchema).min(1).max(4096),
            // Retrievable requires an already accepted exact original asset and a bound read capability.
            fidelity: z.enum(['heuristic', 'semantic', 'retrievable']),
            retained_asset_ids: z.array(IdentifierSchema),
            generation_ids: z.array(IdentifierSchema),
            derivation_generation: GenerationSchema.optional(),
            /** Optional producer input identity, included in the accepted context-edit payload hash. */
            accepted_input_fingerprint: ContentHashSchema.optional(),
            /** Accepted append receipt for a retrievable original; absent for other fidelity modes. */
            accepted_asset_operation_id: IdentifierSchema.optional(),
            placement: ContextChangePlacementSchema,
        }),
    ])
    .meta({ id: 'ConversationContextChangeProposal' });

const selectionShape = {
    expected_revision: NonnegativeSafeIntegerSchema,
    expected_context_revision: NonnegativeSafeIntegerSchema,
    entry_ids: z.array(IdentifierSchema).min(1).describe('Selected IDs in active context order'),
};
const wholeEntryShape = {
    selected_block_ids: z.never().optional(),
    selected_entries: z.never().optional(),
};
const partialEntryShape = {
    selected_block_ids: SelectedContextBlocksSchema,
    selected_entries: z.array(ContextEntrySchema).min(1),
};
export const ContextChangePlanInputSchema = z
    .union([
        z.strictObject({ ...selectionShape, ...wholeEntryShape }),
        z.strictObject({ ...selectionShape, ...partialEntryShape }),
    ])
    .meta({ id: 'ConversationContextChangePlanInput' });

const requestShape = {
    ...selectionShape,
    operation_id: IdentifierSchema,
    expected_source_fingerprint: ContentHashSchema,
    recorded_at: TimestampSchema,
    proposal: ContextChangeProposalSchema,
};
export const ContextChangeRequestSchema = z
    .union([
        z.strictObject({ ...requestShape, ...wholeEntryShape }),
        z.strictObject({ ...requestShape, ...partialEntryShape }),
    ])
    .meta({ id: 'ConversationContextChangeRequest' });

const planShape = {
    source_fingerprint: ContentHashSchema,
    entry_ids: z.array(IdentifierSchema).min(1),
    source_turn_ids: z.array(IdentifierSchema).min(1),
    source_block_ids: z.array(IdentifierSchema),
    selected_asset_ids: z.array(IdentifierSchema),
    disjoint_ranges: NonnegativeSafeIntegerSchema,
};
export const ContextChangePlanSchema = z
    .union([
        z.strictObject({ ...planShape, ...wholeEntryShape }),
        z.strictObject({ ...planShape, ...partialEntryShape }),
    ])
    .meta({ id: 'ConversationContextChangePlan' });

export { ConversationChangeSchema } from './change.js';
