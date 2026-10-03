import { z } from 'zod';
import { CompactionStrategySchema } from './context-foundation.js';
import { ContextMeasurementSchema } from './context-measurement.js';
import { ConversationEditRecordRefSchema } from './conversation-edit-operation.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
} from './primitives.js';
import { SourceBlockSliceSchema } from './source-slices.js';

export const JsonMinificationConfigurationSchema = z
    .strictObject({
        format: z.literal('raw_json_text'),
        minimum_token_reduction: PositiveSafeIntegerSchema,
        max_code_units: z.number().int().positive().max(1048576),
        max_depth: z.number().int().positive().max(128),
        max_lexical_tokens: z.number().int().positive().max(262144),
    })
    .meta({ id: 'ConversationJsonMinificationConfiguration' });

export const JsonMinificationTransformSchema = z
    .strictObject({
        entry_id: IdentifierSchema,
        source_slice: SourceBlockSliceSchema,
        replacement_text: z.string(),
        replacement_text_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationJsonMinificationTransform' });

export const JsonMinificationCandidateSchema = z
    .strictObject({
        kind: z.literal('json_minification_candidate'),
        parser: z.literal('rfc8259-lexical-v1'),
        strategy: CompactionStrategySchema,
        source_fingerprint: ContentHashSchema,
        transforms: z.array(JsonMinificationTransformSchema).min(1).max(256),
    })
    .meta({ id: 'ConversationJsonMinificationCandidate' });

export const JsonMinificationMeasuredProjectionSchema = ContextMeasurementSchema.extend({
    tokenizer_version: IdentifierSchema,
}).meta({ id: 'ConversationJsonMinificationMeasuredProjection' });
const measuredProjection = JsonMinificationMeasuredProjectionSchema;
export const JsonMinificationMeasurementIdentitySchema = measuredProjection
    .pick({
        tokenizer: true,
        tokenizer_version: true,
        adapter: true,
        adapter_version: true,
        target_model: true,
        method: true,
    })
    .meta({ id: 'ConversationJsonMinificationMeasurementIdentity' });

export const JsonMinificationMeasurementSchema = z
    .strictObject({
        target_fingerprint: ContentHashSchema,
        prospective_input_fingerprint: ContentHashSchema,
        original_projection_fingerprint: ContentHashSchema,
        replacement_projection_fingerprint: ContentHashSchema,
        original: measuredProjection,
        replacement: measuredProjection,
        minimum_token_reduction: PositiveSafeIntegerSchema,
    })
    .meta({ id: 'ConversationJsonMinificationMeasurement' });

export const JsonMinificationProposalSchema = JsonMinificationCandidateSchema.omit({ kind: true })
    .extend({
        kind: z.literal('validated_json_minification'),
        measurement: JsonMinificationMeasurementSchema,
    })
    .meta({ id: 'ConversationJsonMinificationProposal' });

export const JsonMinificationNoOpReasonSchema = z
    .enum(['measurement_unavailable', 'no_token_benefit', 'already_minified', 'no_eligible_blocks'])
    .meta({ id: 'ConversationJsonMinificationNoOpReason' });

/** Host-owned completed application evidence; never accepted as a raw context/edit command. */
export const JsonMinificationApplicationSchema = z
    .strictObject({
        kind: z.literal('json_minification'),
        compaction_id: IdentifierSchema,
        source: ConversationRefSchema,
        source_context_revision: NonnegativeSafeIntegerSchema,
        source_fingerprint: ContentHashSchema,
        output_fingerprint: ContentHashSchema,
        selected_entries: z.array(ConversationEditRecordRefSchema).min(1).max(256),
        source_slices: z.array(SourceBlockSliceSchema).min(1).max(256),
        source_entry_positions: z.array(NonnegativeSafeIntegerSchema).min(1).max(256),
        created_entries: z.array(ConversationEditRecordRefSchema).min(1).max(256),
        created_turns: z.array(ConversationEditRecordRefSchema).min(1).max(256),
    })
    .meta({ id: 'ConversationJsonMinificationApplication' });
