import { z } from 'zod';
import { IdentifierSchema, NonnegativeSafeIntegerSchema } from './primitives.js';

export const TurnKindCountsSchema = z
    .strictObject({
        user: NonnegativeSafeIntegerSchema,
        agent: NonnegativeSafeIntegerSchema,
        tool: NonnegativeSafeIntegerSchema,
        program: NonnegativeSafeIntegerSchema,
    })
    .meta({ id: 'ConversationTurnKindCounts' });

export const ConversationInspectionSchema = z
    .strictObject({
        source_turn_count: NonnegativeSafeIntegerSchema,
        context_turn_count: NonnegativeSafeIntegerSchema,
        source_turns_by_kind: TurnKindCountsSchema,
        context_turns_by_kind: TurnKindCountsSchema,
        generation_count: NonnegativeSafeIntegerSchema,
        pending_tool_call_ids: z.array(IdentifierSchema),
    })
    .meta({ id: 'ConversationInspection' });
