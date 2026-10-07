import { z } from 'zod';
import { RetrievalCapabilitySchema } from './content.js';
import { ContentHashSchema, IdentifierSchema } from './primitives.js';

export const ContextRetrievalRequirementSchema = z
    .strictObject({
        id: IdentifierSchema,
        asset_id: IdentifierSchema,
        retrieval: RetrievalCapabilitySchema,
        /** Canonical append receipt that accepted the original asset before context replacement. */
        accepted_asset_operation_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationContextRetrievalRequirement' });

export const SourceTurnContextEntrySchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('source_turn'),
        turn_id: IdentifierSchema,
        block_ids: z.array(IdentifierSchema).min(1).optional(),
    })
    .meta({ id: 'ConversationSourceTurnContextEntry' });

export const ReplacementTurnContextEntrySchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('replacement_turn'),
        compaction_id: IdentifierSchema,
        turn_id: IdentifierSchema,
        block_ids: z.array(IdentifierSchema).min(1).optional(),
    })
    .meta({ id: 'ConversationReplacementTurnContextEntry' });

export const ContextEntrySchema = z
    .discriminatedUnion('type', [SourceTurnContextEntrySchema, ReplacementTurnContextEntrySchema])
    .meta({ id: 'ConversationContextEntry' });

export const CompactionStrategySchema = z
    .strictObject({
        id: IdentifierSchema,
        version: IdentifierSchema,
        configuration_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationCompactionStrategy' });
