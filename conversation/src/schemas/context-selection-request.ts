import { z } from 'zod';
import {
    ContentBlockSchema,
    GeneratedAgentTurnSchema,
    JsonPathSchema,
    ProgramTurnSchema,
    ToolTurnSchema,
    UserTurnSchema,
} from './content.js';
import {
    ConversationRefSchema,
    IdentifierSchema,
    JsonValueSchema,
    NonnegativeSafeIntegerSchema,
} from './primitives.js';

/** Whole top-level blocks only; nested JSON/media/text subranges are separate future selectors. */
export const ContextSelectionBlockTypeSchema = z
    .union(ContentBlockSchema.options.map((schema) => schema.shape.type))
    .meta({ id: 'ConversationContextSelectionBlockType' });
export const ContextSelectionActorKindSchema = z
    .union([
        UserTurnSchema.shape.kind,
        GeneratedAgentTurnSchema.shape.kind,
        ToolTurnSchema.shape.kind,
        ProgramTurnSchema.shape.kind,
    ])
    .meta({ id: 'ConversationContextSelectionActorKind' });
export const ContextSelectionAnchorSchema = z
    .discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('entry'), id: IdentifierSchema }),
        z.strictObject({ kind: z.literal('turn'), id: IdentifierSchema }),
    ])
    .meta({ id: 'ConversationContextSelectionAnchor' });
export const ContextSelectionRangeSchema = z
    .strictObject({
        from: ContextSelectionAnchorSchema,
        through: ContextSelectionAnchorSchema,
    })
    .meta({ id: 'ConversationContextSelectionRange' });
export const ContextMetadataPredicateSchema = z
    .discriminatedUnion('op', [
        z.strictObject({ op: z.literal('exists'), path: JsonPathSchema, exists: z.boolean() }),
        z.strictObject({ op: z.literal('equals'), path: JsonPathSchema, value: JsonValueSchema }),
        z.strictObject({ op: z.literal('in'), path: JsonPathSchema, values: z.array(JsonValueSchema).min(1) }),
    ])
    .meta({ id: 'ConversationContextMetadataPredicate' });
export const ContextSelectorSchema = z
    .strictObject({
        source: z.discriminatedUnion('kind', [
            z.strictObject({ kind: z.literal('all') }),
            z.strictObject({ kind: z.literal('turn_ids'), turn_ids: z.array(IdentifierSchema).min(1) }),
            z.strictObject({ kind: z.literal('range'), range: ContextSelectionRangeSchema }),
            z.strictObject({ kind: z.literal('ranges'), ranges: z.array(ContextSelectionRangeSchema).min(1) }),
        ]),
        filters: z
            .strictObject({
                actor_kinds: z.array(ContextSelectionActorKindSchema).min(1).optional(),
                actor_ids: z.array(IdentifierSchema).min(1).optional(),
                tool_names: z.array(IdentifierSchema).min(1).optional(),
                block_ids: z.array(IdentifierSchema).min(1).optional(),
                block_types: z.array(ContextSelectionBlockTypeSchema).min(1).optional(),
                /** All predicates apply to turn metadata; every dimension is AND, each value list is OR. */
                metadata: z.array(ContextMetadataPredicateSchema).min(1).optional(),
            })
            .optional(),
    })
    .meta({ id: 'ConversationContextSelector' });
export const ContextSelectionRequestSchema = z
    .strictObject({
        conversation: ConversationRefSchema,
        expected_context_revision: NonnegativeSafeIntegerSchema,
        selector: ContextSelectorSchema,
    })
    .meta({ id: 'ConversationContextSelectionRequest' });
