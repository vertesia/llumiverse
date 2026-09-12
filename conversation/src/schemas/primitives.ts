import { z } from 'zod';

export const CONVERSATION_FORMAT = 'llumiverse.conversation' as const;
export const CONVERSATION_SCHEMA_VERSION = 0 as const;
export const CONVERSATION_EXPERIMENTAL_REVISION = '2026-09-11.ingestion.1' as const;

export const IdentifierSchema = z.string().min(1).meta({ id: 'ConversationIdentifier' });

export const NonnegativeSafeIntegerSchema = z
    .number()
    .int()
    .nonnegative()
    .max(Number.MAX_SAFE_INTEGER)
    .meta({ id: 'ConversationNonnegativeSafeInteger' });

export const PositiveSafeIntegerSchema = z
    .number()
    .int()
    .positive()
    .max(Number.MAX_SAFE_INTEGER)
    .meta({ id: 'ConversationPositiveSafeInteger' });

export const TimestampSchema = z.iso.datetime({ offset: false }).meta({ id: 'ConversationTimestamp' });

export const ContentHashSchema = z.string().min(1).meta({ id: 'ConversationContentHash' });

export const Base64Schema = z
    .string()
    .regex(/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/)
    .meta({ id: 'ConversationBase64' });

// Zod 4's built-in recursive JSON schema provides both a real recursive runtime validator and a
// recursive draft-2020-12 schema. Public parsing runs the bounded preflight first; callers should
// use that boundary when they need preservation rather than Zod's reconstructed output value.
// Register the recursive instance itself: meta() clones it and leaves its recursion anonymously named.
export const JsonValueSchema = z.json().register(z.globalRegistry, { id: 'ConversationJsonValue' });

export const JsonObjectSchema = z.record(z.string(), JsonValueSchema).meta({ id: 'ConversationJsonObject' });

export const MetadataSchema = z.record(z.string().min(1), JsonValueSchema).meta({ id: 'ConversationMetadata' });

export const ModelVisibilitySchema = z.enum(['include', 'exclude']).meta({ id: 'ConversationModelVisibility' });

export const AuthoritySchema = z.enum(['system', 'developer', 'ordinary']).meta({ id: 'ConversationAuthority' });

export const TurnKindSchema = z.enum(['user', 'agent', 'tool', 'program']).meta({ id: 'ConversationTurnKind' });

export const TurnStatusSchema = z.enum(['completed', 'interrupted', 'failed']).meta({ id: 'ConversationTurnStatus' });

export const TurnTimestampsSchema = z
    .strictObject({
        recorded_at: TimestampSchema,
        started_at: TimestampSchema.optional(),
        completed_at: TimestampSchema.optional(),
    })
    .meta({ id: 'ConversationTurnTimestamps' });

export const ConversationRefSchema = z
    .strictObject({
        conversation_id: IdentifierSchema,
        revision: NonnegativeSafeIntegerSchema,
    })
    .meta({ id: 'ConversationRef' });
