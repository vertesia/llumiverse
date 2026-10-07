import { z } from 'zod';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    DIAGNOSTIC_MESSAGE_MAX_LENGTH,
} from '../runtime-constants.js';
import { GenerationStatusSchema, GenerationUsageSchema } from './execution.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    JsonValueSchema,
    NonnegativeSafeIntegerSchema,
    TurnStatusSchema,
} from './primitives.js';

const streamIdentityShape = {
    stream_id: IdentifierSchema,
    request_id: IdentifierSchema,
    attempt_id: IdentifierSchema,
    response_operation_id: IdentifierSchema,
    generation_id: IdentifierSchema,
    draft_turn_id: IdentifierSchema,
};

export const ConversationStreamIdentitySchema = z
    .strictObject(streamIdentityShape)
    .meta({ id: 'ConversationStreamIdentity' });

export const ConversationStreamCursorSchema = z
    .strictObject({
        stream_id: IdentifierSchema,
        event_id: IdentifierSchema,
        sequence: NonnegativeSafeIntegerSchema,
    })
    .meta({ id: 'ConversationStreamCursor' });

const streamEnvelopeShape = {
    format: z.literal(CONVERSATION_FORMAT),
    schema_version: z.literal(CONVERSATION_SCHEMA_VERSION),
    experimental_revision: z.literal(CONVERSATION_EXPERIMENTAL_REVISION),
    ...streamIdentityShape,
    event_id: IdentifierSchema,
    sequence: NonnegativeSafeIntegerSchema,
};

export const NativeStreamPathSegmentSchema = z
    .union([z.string().min(1), NonnegativeSafeIntegerSchema])
    .meta({ id: 'ConversationNativeStreamPathSegment' });

export const NativeStreamPositionSchema = z
    .strictObject({
        protocol: IdentifierSchema,
        path: z.array(NativeStreamPathSegmentSchema).min(1),
        native_item_id: z.string().min(1).optional(),
    })
    .meta({ id: 'ConversationNativeStreamPosition' });

const draftBlockReferenceShape = {
    draft_block_id: IdentifierSchema,
    native_position: NativeStreamPositionSchema,
};

export const ConversationStreamDraftBlockSchema = z
    .discriminatedUnion('type', [
        z.strictObject({ type: z.literal('text') }),
        z.strictObject({ type: z.literal('reasoning'), visibility: z.literal('display') }),
        z.strictObject({
            type: z.literal('tool_call'),
            executor: z.enum(['application', 'provider']),
            call_id: z.string().min(1).optional(),
            tool_name: z.string().min(1).optional(),
        }),
        z.strictObject({
            type: z.enum(['image', 'audio', 'video', 'document']),
            mime_type: z.string().min(1).optional(),
        }),
    ])
    .meta({ id: 'ConversationStreamDraftBlock' });

export const ConversationStreamFailureDiagnosticSchema = z
    .strictObject({
        code: IdentifierSchema,
        message: z.string().max(DIAGNOSTIC_MESSAGE_MAX_LENGTH),
        retryable: z.boolean().optional(),
    })
    .meta({ id: 'ConversationStreamFailureDiagnostic' });

export const ConversationStreamReconciliationSchema = z
    .strictObject({
        draft_block_ids: z.array(IdentifierSchema).min(1),
        native_positions: z.array(NativeStreamPositionSchema).min(1),
        committed_block_ids: z.array(IdentifierSchema),
        disposition: z.enum(['direct', 'structured_output', 'omitted_invalid', 'replay_only']),
        transformation_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationStreamReconciliation' });

export const ConversationStreamTransformationProofSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('structured_output'),
        source_block_ids: z.array(IdentifierSchema).min(1),
        source_texts: z.array(z.string()).min(1),
        result_block_id: IdentifierSchema,
        source_fingerprint: ContentHashSchema,
        result_fingerprint: ContentHashSchema,
    })
    .meta({ id: 'ConversationStreamTransformationProof' });

export const ConversationStreamResponseMappingSchema = z
    .strictObject({
        canonical_id: IdentifierSchema,
        native_position: NativeStreamPositionSchema,
        kind: z.enum(['turn', 'block', 'call']),
    })
    .meta({ id: 'ConversationStreamResponseMapping' });

export const ConversationStreamDecodeEvidenceSchema = z
    .strictObject({
        item_mappings: z.array(ConversationStreamResponseMappingSchema),
        transformations: z.array(ConversationStreamTransformationProofSchema),
    })
    .meta({ id: 'ConversationStreamDecodeEvidence' });

const DraftStartedSchema = z.strictObject({
    ...streamEnvelopeShape,
    type: z.literal('draft_started'),
    origin: z.literal('live_transport'),
});

const DraftBlockStartedSchema = z.strictObject({
    ...streamEnvelopeShape,
    ...draftBlockReferenceShape,
    type: z.literal('draft_block_started'),
    block: ConversationStreamDraftBlockSchema,
});

const DraftTextDeltaSchema = z.strictObject({
    ...streamEnvelopeShape,
    ...draftBlockReferenceShape,
    type: z.literal('draft_text_delta'),
    text: z.string().min(1),
});

const DraftReasoningDeltaSchema = z.strictObject({
    ...streamEnvelopeShape,
    ...draftBlockReferenceShape,
    type: z.literal('draft_reasoning_delta'),
    text: z.string().min(1),
});

const DraftToolCallIdentitySchema = z.strictObject({
    ...streamEnvelopeShape,
    ...draftBlockReferenceShape,
    type: z.literal('draft_tool_call_identity'),
    call_id: z.string().min(1).optional(),
    tool_name: z.string().min(1).optional(),
});

const DraftToolArgumentsDeltaSchema = z.strictObject({
    ...streamEnvelopeShape,
    ...draftBlockReferenceShape,
    type: z.literal('draft_tool_arguments_delta'),
    arguments: z.discriminatedUnion('encoding', [
        z.strictObject({ encoding: z.literal('json_fragment'), fragment: z.string().min(1) }),
        z.strictObject({ encoding: z.literal('json_value_snapshot'), value: JsonValueSchema }),
    ]),
});

const DraftBlockFinishedSchema = z.strictObject({
    ...streamEnvelopeShape,
    ...draftBlockReferenceShape,
    type: z.literal('draft_block_finished'),
    outcome: z.enum(['native_complete', 'interrupted', 'malformed', 'failed']),
});

const UsageSnapshotSchema = z.strictObject({
    ...streamEnvelopeShape,
    type: z.literal('usage_snapshot'),
    usage: GenerationUsageSchema,
});

const DraftFinishedSchema = z.strictObject({
    ...streamEnvelopeShape,
    type: z.literal('draft_finished'),
    outcome: z.enum(['completed', 'interrupted', 'failed']),
    finish_reason: z.string().optional(),
    service_tier: z.string().min(1).optional(),
});

const ResponseAcceptedSchema = z.strictObject({
    ...streamEnvelopeShape,
    type: z.literal('response_accepted'),
    origin: z.enum(['live_transport', 'accepted_recovery']),
    conversation: ConversationRefSchema,
    operation_receipt_id: IdentifierSchema,
    committed_turn_id: IdentifierSchema,
    turn_status: TurnStatusSchema,
    generation_status: GenerationStatusSchema,
    committed_block_ids: z.array(IdentifierSchema),
    accepted_asset_ids: z.array(IdentifierSchema),
    reconciliations: z.array(ConversationStreamReconciliationSchema),
});

const StreamTerminatedSchema = z.strictObject({
    ...streamEnvelopeShape,
    type: z.literal('stream_terminated'),
    outcome: z.enum(['cancelled', 'failed']),
    diagnostic: ConversationStreamFailureDiagnosticSchema.optional(),
});

export const ConversationStreamEventSchema = z
    .discriminatedUnion('type', [
        DraftStartedSchema,
        DraftBlockStartedSchema,
        DraftTextDeltaSchema,
        DraftReasoningDeltaSchema,
        DraftToolCallIdentitySchema,
        DraftToolArgumentsDeltaSchema,
        DraftBlockFinishedSchema,
        UsageSnapshotSchema,
        DraftFinishedSchema,
        ResponseAcceptedSchema,
        StreamTerminatedSchema,
    ])
    .meta({ id: 'ConversationStreamEvent' });

export const ConversationStreamEventBatchSchema = z
    .strictObject({
        stream_id: IdentifierSchema,
        events: z.array(ConversationStreamEventSchema).min(1),
    })
    .meta({ id: 'ConversationStreamEventBatch' });
