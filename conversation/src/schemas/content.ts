import { z } from 'zod';
import {
    AuthoritySchema,
    Base64Schema,
    ContentHashSchema,
    IdentifierSchema,
    JsonObjectSchema,
    JsonValueSchema,
    MetadataSchema,
    ModelVisibilitySchema,
    NonnegativeSafeIntegerSchema,
    PositiveSafeIntegerSchema,
    TimestampSchema,
    TurnStatusSchema,
    TurnTimestampsSchema,
} from './primitives.js';

export const NativeIdentitySchema = z
    .strictObject({
        protocol: IdentifierSchema,
        scope: IdentifierSchema,
        value: z.string().min(1),
    })
    .meta({ id: 'ConversationNativeIdentity' });

export const ImageRegionSchema = z
    .strictObject({
        type: z.literal('image_region'),
        coordinate_space: z.enum(['pixels', 'normalized']),
        x: z.number().nonnegative(),
        y: z.number().nonnegative(),
        width: z.number().positive(),
        height: z.number().positive(),
    })
    .meta({ id: 'ConversationImageRegion' });

export const PageRangeSchema = z
    .strictObject({
        type: z.literal('page_range'),
        from_page: PositiveSafeIntegerSchema,
        through_page: PositiveSafeIntegerSchema,
    })
    .meta({ id: 'ConversationPageRange' });

export const TimeRangeSchema = z
    .strictObject({
        type: z.literal('time_range'),
        start_seconds: z.number().nonnegative(),
        end_seconds: z.number().positive(),
    })
    .meta({ id: 'ConversationTimeRange' });

export const AssetKindSchema = z
    .enum(['text', 'json', 'binary', 'image', 'document', 'audio', 'video', 'other'])
    .meta({ id: 'ConversationAssetKind' });

export const InlineTextAssetStorageSchema = z
    .strictObject({
        type: z.literal('inline_text'),
        text: z.string(),
    })
    .meta({ id: 'ConversationInlineTextAssetStorage' });

export const InlineJsonAssetStorageSchema = z
    .strictObject({
        type: z.literal('inline_json'),
        value: JsonValueSchema,
    })
    .meta({ id: 'ConversationInlineJsonAssetStorage' });

export const InlineBase64AssetStorageSchema = z
    .strictObject({
        type: z.literal('inline_base64'),
        data: Base64Schema,
    })
    .meta({ id: 'ConversationInlineBase64AssetStorage' });

export const ExternalAssetStorageSchema = z
    .strictObject({
        type: z.literal('external'),
        resolver: IdentifierSchema,
        locator: JsonObjectSchema,
    })
    .meta({ id: 'ConversationExternalAssetStorage' });

export const AssetStorageSchema = z
    .discriminatedUnion('type', [
        InlineTextAssetStorageSchema,
        InlineJsonAssetStorageSchema,
        InlineBase64AssetStorageSchema,
        ExternalAssetStorageSchema,
    ])
    .meta({ id: 'ConversationAssetStorage' });

export const ReceivedAssetProvenanceSchema = z
    .strictObject({
        type: z.literal('received'),
        source_turn_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationReceivedAssetProvenance' });

export const GeneratedAssetProvenanceSchema = z
    .strictObject({
        type: z.literal('generated'),
        generation_id: IdentifierSchema,
        source_turn_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationGeneratedAssetProvenance' });

export const ImportedAssetProvenanceSchema = z
    .strictObject({
        type: z.literal('imported'),
        source: IdentifierSchema,
        native_id: NativeIdentitySchema.optional(),
    })
    .meta({ id: 'ConversationImportedAssetProvenance' });

export const DerivedAssetProvenanceSchema = z
    .strictObject({
        type: z.literal('derived'),
        source_asset_id: IdentifierSchema,
        transform_id: IdentifierSchema,
        transform_version: IdentifierSchema,
        configuration: JsonObjectSchema.optional(),
    })
    .meta({ id: 'ConversationDerivedAssetProvenance' });

export const AssetProvenanceSchema = z
    .discriminatedUnion('type', [
        ReceivedAssetProvenanceSchema,
        GeneratedAssetProvenanceSchema,
        ImportedAssetProvenanceSchema,
        DerivedAssetProvenanceSchema,
    ])
    .meta({ id: 'ConversationAssetProvenance' });

export const AssetMediaMetadataSchema = z
    .strictObject({
        width: PositiveSafeIntegerSchema.optional(),
        height: PositiveSafeIntegerSchema.optional(),
        duration_seconds: z.number().positive().optional(),
        page_count: PositiveSafeIntegerSchema.optional(),
    })
    .meta({ id: 'ConversationAssetMediaMetadata' });

export const AssetSchema = z
    .strictObject({
        id: IdentifierSchema,
        kind: AssetKindSchema,
        mime_type: z.string().min(1),
        storage: AssetStorageSchema,
        provenance: AssetProvenanceSchema,
        byte_length: NonnegativeSafeIntegerSchema.optional(),
        content_hash: ContentHashSchema.optional(),
        media: AssetMediaMetadataSchema.optional(),
        created_at: TimestampSchema,
        metadata: MetadataSchema.optional(),
    })
    .meta({ id: 'ConversationAsset' });

export const ToolResultCapabilitySchema = z
    .enum(['text', 'json', 'image', 'document', 'audio', 'video'])
    .meta({ id: 'ConversationToolResultCapability' });

export const ToolInputSchemaSchema = z
    .union([JsonObjectSchema, z.boolean()])
    .meta({ id: 'ConversationToolInputSchema' });

export const ToolDefinitionSchema = z
    .strictObject({
        id: IdentifierSchema,
        name: IdentifierSchema,
        version: IdentifierSchema,
        description: z.string().optional(),
        input_schema: ToolInputSchemaSchema,
        result_capabilities: z.array(ToolResultCapabilitySchema).optional(),
        metadata: MetadataSchema.optional(),
    })
    .meta({ id: 'ConversationToolDefinition' });

export const TextBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('text'),
        text: z.string(),
        format: z.enum(['plain', 'markdown', 'code']),
        language: z.string().min(1).optional(),
    })
    .meta({ id: 'ConversationTextBlock' });

export const JsonBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('json'),
        value: JsonValueSchema,
    })
    .meta({ id: 'ConversationJsonBlock' });

export const ImageBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('image'),
        asset_id: IdentifierSchema,
        caption: z.string().optional(),
        selection: ImageRegionSchema.optional(),
    })
    .meta({ id: 'ConversationImageBlock' });

export const DocumentBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('document'),
        asset_id: IdentifierSchema,
        caption: z.string().optional(),
        selection: PageRangeSchema.optional(),
    })
    .meta({ id: 'ConversationDocumentBlock' });

export const AudioBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('audio'),
        asset_id: IdentifierSchema,
        caption: z.string().optional(),
        selection: TimeRangeSchema.optional(),
    })
    .meta({ id: 'ConversationAudioBlock' });

export const VideoBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('video'),
        asset_id: IdentifierSchema,
        caption: z.string().optional(),
        selection: TimeRangeSchema.optional(),
    })
    .meta({ id: 'ConversationVideoBlock' });

export const StructuredToolArgumentsSchema = z
    .strictObject({
        type: z.literal('json'),
        value: JsonValueSchema,
    })
    .meta({ id: 'ConversationStructuredToolArguments' });

export const InvalidToolArgumentsSchema = z
    .strictObject({
        type: z.literal('invalid'),
        raw: z.string(),
        error: z.string().optional(),
    })
    .meta({ id: 'ConversationInvalidToolArguments' });

export const ToolArgumentsSchema = z
    .discriminatedUnion('type', [StructuredToolArgumentsSchema, InvalidToolArgumentsSchema])
    .meta({ id: 'ConversationToolArguments' });

export const ToolCallBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('tool_call'),
        call_id: IdentifierSchema,
        tool_name: IdentifierSchema,
        definition_id: IdentifierSchema.optional(),
        executor: z.enum(['application', 'provider']),
        arguments: ToolArgumentsSchema,
        native_id: NativeIdentitySchema.optional(),
    })
    .meta({ id: 'ConversationToolCallBlock' });

export const RetrievalCapabilitySchema = z
    .strictObject({
        capability: IdentifierSchema,
        version: PositiveSafeIntegerSchema,
        arguments: JsonObjectSchema,
        tool_definition_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationRetrievalCapability' });

export const ExternalReferenceBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('external_reference'),
        asset_id: IdentifierSchema,
        original_type: z.enum([
            'text',
            'json',
            'image',
            'document',
            'audio',
            'video',
            'tool_call',
            'tool_result',
            'reasoning',
            'native_replay',
            'extension',
        ]),
        description: z.string().min(1),
        content_hash: ContentHashSchema.optional(),
        preview: z.string().optional(),
        retrieval: RetrievalCapabilitySchema,
    })
    .meta({ id: 'ConversationExternalReferenceBlock' });

export const ReasoningBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('reasoning'),
        text: z.string(),
        representation: z.enum(['text', 'summary']),
    })
    .meta({ id: 'ConversationReasoningBlock' });

export const ReplayCompatibilityScopeSchema = z
    .strictObject({
        provider: IdentifierSchema,
        protocol: IdentifierSchema,
        model: IdentifierSchema.optional(),
        adapter_version: IdentifierSchema,
    })
    .meta({ id: 'ConversationReplayCompatibilityScope' });

export const ReplayDependenciesSchema = z
    .strictObject({
        turn_ids: z.array(IdentifierSchema),
        block_ids: z.array(IdentifierSchema),
        call_ids: z.array(IdentifierSchema),
        request_ids: z.array(IdentifierSchema),
    })
    .meta({ id: 'ConversationReplayDependencies' });

export const NativeReplayBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('native_replay'),
        adapter: IdentifierSchema,
        protocol: IdentifierSchema,
        compatibility_scope: ReplayCompatibilityScopeSchema,
        payload: JsonValueSchema,
        dependencies: ReplayDependenciesSchema,
        content_hash: ContentHashSchema.optional(),
    })
    .meta({ id: 'ConversationNativeReplayBlock' });

export const ExtensionBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('extension'),
        namespace: IdentifierSchema,
        version: IdentifierSchema,
        payload: JsonValueSchema,
        model_projection: z.enum(['excluded', 'registered']).optional(),
    })
    .meta({ id: 'ConversationExtensionBlock' });

export const NestedToolResultContentBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ExternalReferenceBlockSchema,
        ReasoningBlockSchema,
        NativeReplayBlockSchema,
        ExtensionBlockSchema,
    ])
    .meta({ id: 'ConversationNestedToolResultContentBlock' });

export const ToolResultBlockSchema = z
    .strictObject({
        id: IdentifierSchema,
        type: z.literal('tool_result'),
        call_id: IdentifierSchema,
        // Some native histories and older hosts do not persist execution status. New application
        // input records the explicit terminal status supplied through PromptSegment.
        status: z.enum(['success', 'error', 'cancelled', 'denied', 'unknown']),
        content: z.array(NestedToolResultContentBlockSchema),
        native_id: NativeIdentitySchema.optional(),
    })
    .meta({ id: 'ConversationToolResultBlock' });

export const ContentBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ToolCallBlockSchema,
        ToolResultBlockSchema,
        ExternalReferenceBlockSchema,
        ReasoningBlockSchema,
        NativeReplayBlockSchema,
        ExtensionBlockSchema,
    ])
    .meta({ id: 'ConversationContentBlock' });

export const UserContentBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ExternalReferenceBlockSchema,
        ExtensionBlockSchema,
    ])
    .meta({ id: 'ConversationUserContentBlock' });

export const AgentContentBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ToolCallBlockSchema,
        ExternalReferenceBlockSchema,
        ReasoningBlockSchema,
        NativeReplayBlockSchema,
        ExtensionBlockSchema,
    ])
    .meta({ id: 'ConversationAgentContentBlock' });

export const ProgramContentBlockSchema = z
    .discriminatedUnion('type', [
        TextBlockSchema,
        JsonBlockSchema,
        ImageBlockSchema,
        DocumentBlockSchema,
        AudioBlockSchema,
        VideoBlockSchema,
        ExternalReferenceBlockSchema,
        ReasoningBlockSchema,
        NativeReplayBlockSchema,
        ExtensionBlockSchema,
    ])
    .meta({ id: 'ConversationProgramContentBlock' });

export const ReceivedTurnProvenanceSchema = z
    .strictObject({ type: z.literal('received') })
    .meta({ id: 'ConversationReceivedTurnProvenance' });

export const InsertedTurnProvenanceSchema = z
    .strictObject({
        type: z.literal('inserted'),
        operation_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationInsertedTurnProvenance' });

export const GeneratedTurnProvenanceSchema = z
    .strictObject({ type: z.literal('generated') })
    .meta({ id: 'ConversationGeneratedTurnProvenance' });

export const ImportedTurnProvenanceSchema = z
    .strictObject({
        type: z.literal('imported'),
        source: IdentifierSchema,
        native_id: NativeIdentitySchema.optional(),
        source_history_turn_number: NonnegativeSafeIntegerSchema.optional(),
        missing_metadata: z
            .array(
                z.enum(['actor_id', 'authority', 'generation', 'timestamps', 'usage', 'exchange', 'tool_definition']),
            )
            .optional(),
    })
    .meta({ id: 'ConversationImportedTurnProvenance' });

export const DerivedTurnProvenanceSchema = z
    .strictObject({
        type: z.literal('derived'),
        derivation_id: IdentifierSchema,
        source_turn_ids: z.array(IdentifierSchema).min(1),
        source_block_ids: z.array(IdentifierSchema).min(1).optional(),
        source_hash: ContentHashSchema,
    })
    .meta({ id: 'ConversationDerivedTurnProvenance' });

export const NonGeneratedTurnProvenanceSchema = z
    .discriminatedUnion('type', [
        ReceivedTurnProvenanceSchema,
        InsertedTurnProvenanceSchema,
        ImportedTurnProvenanceSchema,
        DerivedTurnProvenanceSchema,
    ])
    .meta({ id: 'ConversationNonGeneratedTurnProvenance' });

export const AgentTurnProvenanceSchema = z
    .discriminatedUnion('type', [
        ReceivedTurnProvenanceSchema,
        InsertedTurnProvenanceSchema,
        GeneratedTurnProvenanceSchema,
        ImportedTurnProvenanceSchema,
        DerivedTurnProvenanceSchema,
    ])
    .meta({ id: 'ConversationAgentTurnProvenance' });

const commonTurnShape = {
    id: IdentifierSchema,
    actor_id: IdentifierSchema.optional(),
    status: TurnStatusSchema,
    timestamps: TurnTimestampsSchema,
    execution_id: IdentifierSchema.optional(),
    exchange_id: IdentifierSchema.optional(),
    parent_turn_id: IdentifierSchema.optional(),
    model_visibility: ModelVisibilitySchema,
    metadata: MetadataSchema.optional(),
};

export const UserTurnSchema = z
    .strictObject({
        ...commonTurnShape,
        kind: z.literal('user'),
        authority: AuthoritySchema,
        blocks: z.array(UserContentBlockSchema),
        provenance: NonGeneratedTurnProvenanceSchema,
    })
    .meta({ id: 'ConversationUserTurn' });

const commonAgentTurnShape = {
    ...commonTurnShape,
    kind: z.literal('agent'),
    authority: AuthoritySchema,
    blocks: z.array(AgentContentBlockSchema),
};

export const GeneratedAgentTurnSchema = z
    .strictObject({
        ...commonAgentTurnShape,
        provenance: GeneratedTurnProvenanceSchema,
        generation_id: IdentifierSchema,
    })
    .meta({ id: 'ConversationGeneratedAgentTurn' });

export const ImportedAgentTurnSchema = z
    .strictObject({
        ...commonAgentTurnShape,
        provenance: ImportedTurnProvenanceSchema,
        generation_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationImportedAgentTurn' });

export const DerivedAgentTurnSchema = z
    .strictObject({
        ...commonAgentTurnShape,
        provenance: DerivedTurnProvenanceSchema,
        generation_id: IdentifierSchema.optional(),
    })
    .meta({ id: 'ConversationDerivedAgentTurn' });

export const NongeneratedAgentTurnSchema = z
    .strictObject({
        ...commonAgentTurnShape,
        provenance: z.discriminatedUnion('type', [ReceivedTurnProvenanceSchema, InsertedTurnProvenanceSchema]),
    })
    .meta({ id: 'ConversationNongeneratedAgentTurn' });

export const AgentTurnSchema = z
    .union([GeneratedAgentTurnSchema, ImportedAgentTurnSchema, DerivedAgentTurnSchema, NongeneratedAgentTurnSchema])
    .meta({ id: 'ConversationAgentTurn' });

export const ToolTurnSchema = z
    .strictObject({
        ...commonTurnShape,
        kind: z.literal('tool'),
        authority: z.literal('ordinary'),
        blocks: z.array(ToolResultBlockSchema).length(1),
        provenance: NonGeneratedTurnProvenanceSchema,
    })
    .meta({ id: 'ConversationToolTurn' });

export const ProgramTurnSchema = z
    .strictObject({
        ...commonTurnShape,
        kind: z.literal('program'),
        authority: AuthoritySchema,
        blocks: z.array(ProgramContentBlockSchema),
        provenance: NonGeneratedTurnProvenanceSchema,
    })
    .meta({ id: 'ConversationProgramTurn' });

export const ConversationTurnSchema = z
    .union([UserTurnSchema, AgentTurnSchema, ToolTurnSchema, ProgramTurnSchema])
    .meta({ id: 'ConversationTurn' });
