import { z } from 'zod';
import {
    AssetSchema,
    AudioBlockSchema,
    DocumentBlockSchema,
    GeneratedAgentTurnSchema,
    GeneratedAssetProvenanceSchema,
    ImageBlockSchema,
    JsonBlockSchema,
    ReasoningBlockSchema,
    TextBlockSchema,
    ToolCallBlockSchema,
    VideoBlockSchema,
} from './content.js';
import { ExecutedGenerationSchema, GenerationUsageSchema, OperationReceiptSchema } from './execution.js';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_SCHEMA_VERSION,
    ConversationRefSchema,
    IdentifierSchema,
} from './primitives.js';

export const CONVERSATION_ACCEPTED_OUTPUT_FORMAT = 'llumiverse.conversation-output' as const;

export const ConversationOutputToolCallBlockSchema = ToolCallBlockSchema.omit({
    native_id: true,
    definition_id: true,
}).meta({ id: 'ConversationOutputToolCallBlock' });

export const ConversationOutputBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ConversationOutputToolCallBlockSchema,
        ReasoningBlockSchema,
    ])
    .meta({ id: 'ConversationOutputBlock' });

export const ConversationOutputAssetSchema = AssetSchema.omit({ metadata: true, provenance: true })
    .extend({ provenance: GeneratedAssetProvenanceSchema })
    .meta({ id: 'ConversationOutputAsset' });

export const ConversationOutputGenerationUsageSchema = GenerationUsageSchema.omit({ reported_usage: true }).meta({
    id: 'ConversationOutputGenerationUsage',
});

export const ConversationOutputGenerationSchema = ExecutedGenerationSchema.pick({
    id: true,
    record_source: true,
    request_id: true,
    attempt_id: true,
    provider_response_id: true,
    purpose: true,
    requested_model: true,
    resolved_model: true,
    provider: true,
    protocol: true,
    adapter_version: true,
    status: true,
    finish_reason: true,
    timestamps: true,
    source: true,
    usage: true,
})
    .extend({ usage: ConversationOutputGenerationUsageSchema.optional() })
    .meta({ id: 'ConversationOutputGeneration' });

export const ConversationOutputReceiptSchema = OperationReceiptSchema.pick({
    id: true,
    conversation_id: true,
    base_revision: true,
    result_revision: true,
    recorded_at: true,
    accepted_turn_ids: true,
    accepted_generation_ids: true,
    accepted_asset_ids: true,
})
    .extend({
        accepted_turn_ids: z.array(IdentifierSchema).length(1),
        accepted_generation_ids: z.array(IdentifierSchema).length(1),
    })
    .meta({ id: 'ConversationOutputReceipt' });

export const ConversationOutputTurnSchema = GeneratedAgentTurnSchema.pick({
    id: true,
    status: true,
    timestamps: true,
    model_visibility: true,
    kind: true,
    authority: true,
    blocks: true,
    provenance: true,
    generation_id: true,
})
    .extend({ blocks: z.array(ConversationOutputBlockSchema) })
    .meta({ id: 'ConversationOutputTurn' });

export const ConversationOutputCompletenessSchema = z
    .strictObject({
        history: z.literal('omitted'),
        native_replay: z.literal('omitted'),
        metadata: z.literal('omitted'),
        semantic_content: z.enum(['complete', 'partial']),
        omitted_block_ids: z.array(IdentifierSchema),
        omitted_asset_ids: z.array(IdentifierSchema),
    })
    .meta({ id: 'ConversationOutputCompleteness' });

export const ConversationAcceptedOutputFragmentSchema = z
    .strictObject({
        format: z.literal(CONVERSATION_ACCEPTED_OUTPUT_FORMAT),
        schema_version: z.literal(CONVERSATION_SCHEMA_VERSION),
        experimental_revision: z.literal(CONVERSATION_EXPERIMENTAL_REVISION),
        source: ConversationRefSchema,
        receipt: ConversationOutputReceiptSchema,
        turn: ConversationOutputTurnSchema,
        generation: ConversationOutputGenerationSchema,
        assets: z.record(IdentifierSchema, ConversationOutputAssetSchema),
        completeness: ConversationOutputCompletenessSchema,
    })
    .meta({ id: 'ConversationAcceptedOutputFragment' });
