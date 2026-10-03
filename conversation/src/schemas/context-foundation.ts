import { z } from 'zod';
import { ContentHashSchema, IdentifierSchema } from './primitives.js';

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
