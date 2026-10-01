import { z } from 'zod';
import {
    AssetSchema,
    AudioBlockSchema,
    ConversationTurnSchema,
    DocumentBlockSchema,
    ImageBlockSchema,
    JsonBlockSchema,
    ReasoningBlockSchema,
    TextBlockSchema,
    VideoBlockSchema,
} from './content.js';
import { ExecutedGenerationSchema, GenerationSchema, ImportedGenerationSchema } from './execution.js';
import { ConversationOutputAssetSchema, ConversationOutputGenerationUsageSchema } from './output.js';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_SCHEMA_VERSION,
    ConversationRefSchema,
    IdentifierSchema,
    JsonValueSchema,
    TurnStatusSchema,
    TurnTimestampsSchema,
} from './primitives.js';

export const CONVERSATION_TRANSCRIPT_FORMAT = 'llumiverse.conversation-transcript' as const;
export const CONVERSATION_TRANSCRIPT_MAX_TURNS = 100 as const;
export const CONVERSATION_TRANSCRIPT_MAX_GENERATIONS = 100 as const;
export const CONVERSATION_TRANSCRIPT_MAX_ASSETS = 200 as const;
export const CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN = 1_000 as const;
export const CONVERSATION_TRANSCRIPT_MAX_OMISSIONS = 1_000 as const;

export const ConversationTranscriptSourceGapReasonSchema = z
    .enum(['compacted', 'not_retained'])
    .meta({ id: 'ConversationTranscriptSourceGapReason' });

export const ConversationTranscriptInputOmissionSchema = z
    .strictObject({
        turn_id: IdentifierSchema,
        reason: ConversationTranscriptSourceGapReasonSchema,
    })
    .meta({ id: 'ConversationTranscriptInputOmission' });

export const ConversationTranscriptGenerationInputOmissionSchema = z
    .strictObject({
        generation_id: IdentifierSchema,
        reason: ConversationTranscriptSourceGapReasonSchema,
    })
    .meta({ id: 'ConversationTranscriptGenerationInputOmission' });

export const ConversationTranscriptWindowSchema = z
    .strictObject({
        gap_before: z.boolean(),
        gap_after: z.boolean(),
        omitted_turns: z.array(ConversationTranscriptInputOmissionSchema).max(CONVERSATION_TRANSCRIPT_MAX_TURNS),
        omitted_generations: z
            .array(ConversationTranscriptGenerationInputOmissionSchema)
            .max(CONVERSATION_TRANSCRIPT_MAX_GENERATIONS),
    })
    .meta({ id: 'ConversationTranscriptWindow' });

/**
 * A bounded, preselected canonical window. The storage/API layer pins and selects the exact source
 * revision; this provider projection never scans a full document or operation history.
 */
export const ConversationTranscriptProjectionInputSchema = z
    .strictObject({
        source: ConversationRefSchema,
        turns: z.array(ConversationTurnSchema).max(CONVERSATION_TRANSCRIPT_MAX_TURNS),
        generations: z.array(GenerationSchema).max(CONVERSATION_TRANSCRIPT_MAX_GENERATIONS),
        assets: z.array(AssetSchema).max(CONVERSATION_TRANSCRIPT_MAX_ASSETS),
        window: ConversationTranscriptWindowSchema,
    })
    .meta({ id: 'ConversationTranscriptProjectionInput' });

export const ConversationTranscriptJsonToolArgumentsSchema = z
    .strictObject({
        type: z.literal('json'),
        value: JsonValueSchema,
    })
    .meta({ id: 'ConversationTranscriptJsonToolArguments' });

export const ConversationTranscriptInvalidToolArgumentsSchema = z
    .strictObject({
        type: z.literal('invalid'),
        raw: z.string(),
    })
    .meta({ id: 'ConversationTranscriptInvalidToolArguments' });

export const ConversationTranscriptToolArgumentsSchema = z
    .discriminatedUnion('type', [
        ConversationTranscriptJsonToolArgumentsSchema,
        ConversationTranscriptInvalidToolArgumentsSchema,
    ])
    .meta({ id: 'ConversationTranscriptToolArguments' });

export const ConversationTranscriptToolCallBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('tool_call'),
        call_id: IdentifierSchema,
        tool_name: IdentifierSchema,
        executor: z.enum(['application', 'provider']),
        arguments: ConversationTranscriptToolArgumentsSchema,
    })
    .meta({ id: 'ConversationTranscriptToolCallBlock' });

export const ConversationTranscriptRenderableBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ReasoningBlockSchema,
    ])
    .meta({ id: 'ConversationTranscriptRenderableBlock' });

export const ConversationTranscriptToolResultBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('tool_result'),
        call_id: IdentifierSchema,
        status: z.enum(['success', 'error', 'cancelled', 'denied', 'unknown']),
        content: z.array(ConversationTranscriptRenderableBlockSchema).max(CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN),
    })
    .meta({ id: 'ConversationTranscriptToolResultBlock' });

export const ConversationTranscriptUserBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
    ])
    .meta({ id: 'ConversationTranscriptUserBlock' });

export const ConversationTranscriptAgentBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ReasoningBlockSchema,
        ConversationTranscriptToolCallBlockSchema,
    ])
    .meta({ id: 'ConversationTranscriptAgentBlock' });

export const ConversationTranscriptProgramBlockSchema = ConversationTranscriptRenderableBlockSchema.meta({
    id: 'ConversationTranscriptProgramBlock',
});

const transcriptTurnShape = {
    id: IdentifierSchema,
    status: TurnStatusSchema,
    timestamps: TurnTimestampsSchema,
};

export const ConversationTranscriptUserTurnSchema = z
    .strictObject({
        ...transcriptTurnShape,
        kind: z.literal('user'),
        blocks: z.array(ConversationTranscriptUserBlockSchema).max(CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN),
    })
    .meta({ id: 'ConversationTranscriptUserTurn' });

export const ConversationTranscriptAgentTurnSchema = z
    .strictObject({
        ...transcriptTurnShape,
        kind: z.literal('agent'),
        generation_id: IdentifierSchema.optional(),
        blocks: z.array(ConversationTranscriptAgentBlockSchema).max(CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN),
    })
    .meta({ id: 'ConversationTranscriptAgentTurn' });

export const ConversationTranscriptToolTurnSchema = z
    .strictObject({
        ...transcriptTurnShape,
        kind: z.literal('tool'),
        blocks: z.array(ConversationTranscriptToolResultBlockSchema).length(1),
    })
    .meta({ id: 'ConversationTranscriptToolTurn' });

export const ConversationTranscriptProgramTurnSchema = z
    .strictObject({
        ...transcriptTurnShape,
        kind: z.literal('program'),
        blocks: z.array(ConversationTranscriptProgramBlockSchema).max(CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN),
    })
    .meta({ id: 'ConversationTranscriptProgramTurn' });

export const ConversationTranscriptTurnSchema = z
    .discriminatedUnion('kind', [
        ConversationTranscriptUserTurnSchema,
        ConversationTranscriptAgentTurnSchema,
        ConversationTranscriptToolTurnSchema,
        ConversationTranscriptProgramTurnSchema,
    ])
    .meta({ id: 'ConversationTranscriptTurn' });

// Reuse the authenticated output asset delivery contract, but never expose source provenance.
export const ConversationTranscriptAssetSchema = ConversationOutputAssetSchema.omit({ provenance: true }).meta({
    id: 'ConversationTranscriptAsset',
});

export const ConversationTranscriptExecutedGenerationSchema = ExecutedGenerationSchema.pick({
    id: true,
    record_source: true,
    purpose: true,
    requested_model: true,
    resolved_model: true,
    provider: true,
    protocol: true,
    status: true,
    finish_reason: true,
    timestamps: true,
    usage: true,
})
    .extend({ usage: ConversationOutputGenerationUsageSchema.optional() })
    .meta({
        id: 'ConversationTranscriptExecutedGeneration',
        description:
            'Safe recorded generation metadata. Usage is historical generation accounting, not context measurement.',
    });

export const ConversationTranscriptImportedGenerationSchema = ImportedGenerationSchema.pick({
    id: true,
    record_source: true,
    purpose: true,
    requested_model: true,
    resolved_model: true,
    provider: true,
    protocol: true,
    status: true,
    finish_reason: true,
    timestamps: true,
    usage: true,
})
    .extend({ usage: ConversationOutputGenerationUsageSchema.optional() })
    .meta({
        id: 'ConversationTranscriptImportedGeneration',
        description:
            'Safe recorded imported generation metadata. Recorded usage remains imported and is not billed execution.',
    });

export const ConversationTranscriptGenerationSchema = z
    .discriminatedUnion('record_source', [
        ConversationTranscriptExecutedGenerationSchema,
        ConversationTranscriptImportedGenerationSchema,
    ])
    .meta({ id: 'ConversationTranscriptGeneration' });

export const ConversationTranscriptGenerationMapSchema = z
    .record(IdentifierSchema, ConversationTranscriptGenerationSchema)
    .refine((value) => Object.keys(value).length <= CONVERSATION_TRANSCRIPT_MAX_GENERATIONS, {
        message: 'Transcript generation map exceeds the bounded generation limit',
    })
    .meta({
        id: 'ConversationTranscriptGenerationMap',
        maxProperties: CONVERSATION_TRANSCRIPT_MAX_GENERATIONS,
    });

export const ConversationTranscriptTurnOmissionReasonSchema = z
    .enum(['compacted', 'not_retained', 'internal_program'])
    .meta({ id: 'ConversationTranscriptTurnOmissionReason' });

export const ConversationTranscriptTurnOmissionSchema = z
    .strictObject({
        turn_id: IdentifierSchema,
        reason: ConversationTranscriptTurnOmissionReasonSchema,
    })
    .meta({ id: 'ConversationTranscriptTurnOmission' });

export const ConversationTranscriptBlockOmissionReasonSchema = z
    .enum([
        'native_replay',
        'unsupported_extension',
        'unsupported_external_reference',
        'model_visible_arguments_unavailable',
        'referenced_asset_unavailable',
        'referenced_asset_kind_mismatch',
    ])
    .meta({ id: 'ConversationTranscriptBlockOmissionReason' });

export const ConversationTranscriptBlockOmissionSchema = z
    .strictObject({
        turn_id: IdentifierSchema,
        block_id: IdentifierSchema,
        reason: ConversationTranscriptBlockOmissionReasonSchema,
    })
    .meta({ id: 'ConversationTranscriptBlockOmission' });

export const ConversationTranscriptAssetOmissionReasonSchema = z
    .enum(['not_supplied', 'kind_mismatch'])
    .meta({ id: 'ConversationTranscriptAssetOmissionReason' });

export const ConversationTranscriptAssetOmissionSchema = z
    .strictObject({
        asset_id: IdentifierSchema,
        reason: ConversationTranscriptAssetOmissionReasonSchema,
    })
    .meta({ id: 'ConversationTranscriptAssetOmission' });

export const ConversationTranscriptGenerationOmissionReasonSchema = z
    .enum(['compacted', 'not_retained', 'not_supplied'])
    .meta({ id: 'ConversationTranscriptGenerationOmissionReason' });

export const ConversationTranscriptGenerationOmissionSchema = z
    .strictObject({
        generation_id: IdentifierSchema,
        reason: ConversationTranscriptGenerationOmissionReasonSchema,
    })
    .meta({ id: 'ConversationTranscriptGenerationOmission' });

export const ConversationTranscriptCompletenessSchema = z
    .strictObject({
        gap_before: z.boolean(),
        gap_after: z.boolean(),
        semantic_content: z.enum(['complete', 'partial']),
        metadata: z.literal('omitted'),
        provenance: z.literal('omitted'),
        native_replay: z.literal('omitted'),
        omitted_turns: z.array(ConversationTranscriptTurnOmissionSchema).max(CONVERSATION_TRANSCRIPT_MAX_OMISSIONS),
        omitted_generations: z
            .array(ConversationTranscriptGenerationOmissionSchema)
            .max(CONVERSATION_TRANSCRIPT_MAX_OMISSIONS),
        omitted_blocks: z.array(ConversationTranscriptBlockOmissionSchema).max(CONVERSATION_TRANSCRIPT_MAX_OMISSIONS),
        omitted_assets: z.array(ConversationTranscriptAssetOmissionSchema).max(CONVERSATION_TRANSCRIPT_MAX_OMISSIONS),
    })
    .meta({ id: 'ConversationTranscriptCompleteness' });

export const ConversationTranscriptFragmentSchema = z
    .strictObject({
        format: z.literal(CONVERSATION_TRANSCRIPT_FORMAT),
        schema_version: z.literal(CONVERSATION_SCHEMA_VERSION),
        experimental_revision: z.literal(CONVERSATION_EXPERIMENTAL_REVISION),
        source: ConversationRefSchema,
        turns: z.array(ConversationTranscriptTurnSchema).max(CONVERSATION_TRANSCRIPT_MAX_TURNS),
        generations: ConversationTranscriptGenerationMapSchema,
        assets: z.record(IdentifierSchema, ConversationTranscriptAssetSchema),
        completeness: ConversationTranscriptCompletenessSchema,
    })
    .meta({ id: 'ConversationTranscriptFragment' });
