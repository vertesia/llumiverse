import type { z } from 'zod';
import type {
    AccountingProvenanceSchema,
    AgentContentBlockSchema,
    AgentTurnProvenanceSchema,
    AgentTurnSchema,
    AppendConversationRecordsOptionsSchema,
    AppendConversationRecordsResultSchema,
    AssetKindSchema,
    AssetMediaMetadataSchema,
    AssetProvenanceSchema,
    AssetSchema,
    AssetStorageSchema,
    AssetVersionBindingSchema,
    AudioBlockSchema,
    CacheIntentSchema,
    CompactionRecordSchema,
    CompactionSourceSchema,
    CompactionStrategySchema,
    CompleteInputPartitionSchema,
    ContentBlockSchema,
    ContextEntrySchema,
    ContextMeasurementSchema,
    ContextRetrievalRequirementSchema,
    ConversationContextSchema,
    ConversationDiagnosticCodeSchema,
    ConversationDiagnosticSchema,
    ConversationDiagnosticStageSchema,
    ConversationDocumentSchema,
    ConversationInspectionSchema,
    ConversationLineageParentSchema,
    ConversationLineageSchema,
    ConversationRecordBatchSchema,
    ConversationRefSchema,
    ConversationRuntimeContextSchema,
    ConversationTurnSchema,
    DecodedConversationResponseSchema,
    DerivedAgentTurnSchema,
    DerivedAssetProvenanceSchema,
    DerivedTurnProvenanceSchema,
    DocumentBlockSchema,
    ExecutedGenerationSchema,
    ExecutionReceiptSchema,
    ExtensionBlockSchema,
    ExternalAssetStorageSchema,
    ExternalReferenceBlockSchema,
    GeneratedAgentTurnSchema,
    GeneratedAssetProvenanceSchema,
    GeneratedTurnProvenanceSchema,
    GenerationCostSchema,
    GenerationSchema,
    GenerationTimestampsSchema,
    GenerationUsageSchema,
    ImageBlockSchema,
    ImageRegionSchema,
    ImportedAgentTurnSchema,
    ImportedAssetProvenanceSchema,
    ImportedGenerationSchema,
    ImportedTurnProvenanceSchema,
    InlineBase64AssetStorageSchema,
    InlineJsonAssetStorageSchema,
    InlineTextAssetStorageSchema,
    InsertedTurnProvenanceSchema,
    InvalidToolArgumentsSchema,
    JsonBlockSchema,
    JsonObjectSchema,
    JsonPreflightDiagnosticCodeSchema,
    JsonPreflightDiagnosticSchema,
    JsonValueSchema,
    MetadataSchema,
    ModelTargetSchema,
    NativeIdentitySchema,
    NativeItemMappingSchema,
    NativeReplayBlockSchema,
    NestedToolResultContentBlockSchema,
    NonGeneratedTurnProvenanceSchema,
    NongeneratedAgentTurnSchema,
    OperationReceiptSchema,
    PageRangeSchema,
    ProcessingStateSchema,
    ProcessorConfigurationSchema,
    ProgramContentBlockSchema,
    ProgramTurnSchema,
    ReasoningBlockSchema,
    ReceivedAssetProvenanceSchema,
    ReceivedTurnProvenanceSchema,
    ReplacementTurnContextEntrySchema,
    ReplayCompatibilityScopeSchema,
    ReplayDependenciesSchema,
    ReportedUsageSchema,
    RequestReceiptSchema,
    RetrievalCapabilitySchema,
    SemanticConversationDiagnosticCodeSchema,
    SemanticConversationDiagnosticSchema,
    SourceTurnContextEntrySchema,
    StructuredToolArgumentsSchema,
    TextBlockSchema,
    TimeRangeSchema,
    ToolArgumentsSchema,
    ToolCallBlockSchema,
    ToolDefinitionSchema,
    ToolInputSchemaSchema,
    ToolResultBlockSchema,
    ToolResultCapabilitySchema,
    ToolTurnSchema,
    TurnKindCountsSchema,
    UsageAccountingProvenanceSchema,
    UsageMetricSchema,
    UserContentBlockSchema,
    UserTurnSchema,
    VideoBlockSchema,
} from './schemas/index.js';

export type JsonValue = z.infer<typeof JsonValueSchema>;
export type JsonObject = z.infer<typeof JsonObjectSchema>;
export type ConversationMetadata = z.infer<typeof MetadataSchema>;
export type ConversationRef = z.infer<typeof ConversationRefSchema>;
export type JsonPreflightDiagnosticCode = z.infer<typeof JsonPreflightDiagnosticCodeSchema>;
export type JsonPreflightDiagnostic = z.infer<typeof JsonPreflightDiagnosticSchema>;
export type SemanticConversationDiagnosticCode = z.infer<typeof SemanticConversationDiagnosticCodeSchema>;
export type SemanticConversationDiagnostic = z.infer<typeof SemanticConversationDiagnosticSchema>;
export type ConversationDiagnosticStage = z.infer<typeof ConversationDiagnosticStageSchema>;
export type ConversationDiagnostic = z.infer<typeof ConversationDiagnosticSchema>;
export type ConversationDiagnosticCode = z.infer<typeof ConversationDiagnosticCodeSchema>;
export type TurnKindCounts = z.infer<typeof TurnKindCountsSchema>;
export type ConversationInspection = z.infer<typeof ConversationInspectionSchema>;
export type ConversationRecordBatch = z.infer<typeof ConversationRecordBatchSchema>;
export type AppendConversationRecordsOptions = z.infer<typeof AppendConversationRecordsOptionsSchema>;
export type AppendConversationRecordsResult = z.infer<typeof AppendConversationRecordsResultSchema>;
export type DecodedConversationResponse = z.infer<typeof DecodedConversationResponseSchema>;

export type NativeIdentity = z.infer<typeof NativeIdentitySchema>;
export type ImageRegion = z.infer<typeof ImageRegionSchema>;
export type PageRange = z.infer<typeof PageRangeSchema>;
export type TimeRange = z.infer<typeof TimeRangeSchema>;

export type AssetKind = z.infer<typeof AssetKindSchema>;
export type InlineTextAssetStorage = z.infer<typeof InlineTextAssetStorageSchema>;
export type InlineJsonAssetStorage = z.infer<typeof InlineJsonAssetStorageSchema>;
export type InlineBase64AssetStorage = z.infer<typeof InlineBase64AssetStorageSchema>;
export type ExternalAssetStorage = z.infer<typeof ExternalAssetStorageSchema>;
export type AssetStorage = z.infer<typeof AssetStorageSchema>;
export type ReceivedAssetProvenance = z.infer<typeof ReceivedAssetProvenanceSchema>;
export type GeneratedAssetProvenance = z.infer<typeof GeneratedAssetProvenanceSchema>;
export type ImportedAssetProvenance = z.infer<typeof ImportedAssetProvenanceSchema>;
export type DerivedAssetProvenance = z.infer<typeof DerivedAssetProvenanceSchema>;
export type AssetProvenance = z.infer<typeof AssetProvenanceSchema>;
export type AssetMediaMetadata = z.infer<typeof AssetMediaMetadataSchema>;
export type Asset = z.infer<typeof AssetSchema>;

export type ToolResultCapability = z.infer<typeof ToolResultCapabilitySchema>;
export type ToolInputSchema = z.infer<typeof ToolInputSchemaSchema>;
export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;
export type TextBlock = z.infer<typeof TextBlockSchema>;
export type JsonBlock = z.infer<typeof JsonBlockSchema>;
export type ImageBlock = z.infer<typeof ImageBlockSchema>;
export type DocumentBlock = z.infer<typeof DocumentBlockSchema>;
export type AudioBlock = z.infer<typeof AudioBlockSchema>;
export type VideoBlock = z.infer<typeof VideoBlockSchema>;
export type StructuredToolArguments = z.infer<typeof StructuredToolArgumentsSchema>;
export type InvalidToolArguments = z.infer<typeof InvalidToolArgumentsSchema>;
export type ToolArguments = z.infer<typeof ToolArgumentsSchema>;
export type ToolCallBlock = z.infer<typeof ToolCallBlockSchema>;
export type RetrievalCapability = z.infer<typeof RetrievalCapabilitySchema>;
export type ExternalReferenceBlock = z.infer<typeof ExternalReferenceBlockSchema>;
export type ReasoningBlock = z.infer<typeof ReasoningBlockSchema>;
export type ReplayCompatibilityScope = z.infer<typeof ReplayCompatibilityScopeSchema>;
export type ReplayDependencies = z.infer<typeof ReplayDependenciesSchema>;
export type NativeReplayBlock = z.infer<typeof NativeReplayBlockSchema>;
export type ExtensionBlock = z.infer<typeof ExtensionBlockSchema>;
export type NestedToolResultContentBlock = z.infer<typeof NestedToolResultContentBlockSchema>;
export type ToolResultBlock = z.infer<typeof ToolResultBlockSchema>;
export type ContentBlock = z.infer<typeof ContentBlockSchema>;
export type UserContentBlock = z.infer<typeof UserContentBlockSchema>;
export type AgentContentBlock = z.infer<typeof AgentContentBlockSchema>;
export type ProgramContentBlock = z.infer<typeof ProgramContentBlockSchema>;

export type ReceivedTurnProvenance = z.infer<typeof ReceivedTurnProvenanceSchema>;
export type InsertedTurnProvenance = z.infer<typeof InsertedTurnProvenanceSchema>;
export type GeneratedTurnProvenance = z.infer<typeof GeneratedTurnProvenanceSchema>;
export type ImportedTurnProvenance = z.infer<typeof ImportedTurnProvenanceSchema>;
export type DerivedTurnProvenance = z.infer<typeof DerivedTurnProvenanceSchema>;
export type NonGeneratedTurnProvenance = z.infer<typeof NonGeneratedTurnProvenanceSchema>;
export type AgentTurnProvenance = z.infer<typeof AgentTurnProvenanceSchema>;
export type UserTurn = z.infer<typeof UserTurnSchema>;
export type AgentTurn = z.infer<typeof AgentTurnSchema>;
export type GeneratedAgentTurn = z.infer<typeof GeneratedAgentTurnSchema>;
export type ImportedAgentTurn = z.infer<typeof ImportedAgentTurnSchema>;
export type DerivedAgentTurn = z.infer<typeof DerivedAgentTurnSchema>;
export type NongeneratedAgentTurn = z.infer<typeof NongeneratedAgentTurnSchema>;
export type ToolTurn = z.infer<typeof ToolTurnSchema>;
export type ProgramTurn = z.infer<typeof ProgramTurnSchema>;
export type ConversationTurn = z.infer<typeof ConversationTurnSchema>;

export type UsageMetric = z.infer<typeof UsageMetricSchema>;
export type AccountingProvenance = z.infer<typeof AccountingProvenanceSchema>;
export type UsageAccountingProvenance = z.infer<typeof UsageAccountingProvenanceSchema>;
export type ReportedUsage = z.infer<typeof ReportedUsageSchema>;
export type CompleteInputPartition = z.infer<typeof CompleteInputPartitionSchema>;
export type ConversationRuntimeContext = z.infer<typeof ConversationRuntimeContextSchema>;
export type GenerationCost = z.infer<typeof GenerationCostSchema>;
export type GenerationUsage = z.infer<typeof GenerationUsageSchema>;
export type ContextMeasurement = z.infer<typeof ContextMeasurementSchema>;
export type ModelTarget = z.infer<typeof ModelTargetSchema>;
export type AssetVersionBinding = z.infer<typeof AssetVersionBindingSchema>;
export type NativeItemMapping = z.infer<typeof NativeItemMappingSchema>;
export type RequestReceipt = z.infer<typeof RequestReceiptSchema>;
export type GenerationTimestamps = z.infer<typeof GenerationTimestampsSchema>;
export type ExecutedGeneration = z.infer<typeof ExecutedGenerationSchema>;
export type ImportedGeneration = z.infer<typeof ImportedGenerationSchema>;
export type Generation = z.infer<typeof GenerationSchema>;
export type OperationReceipt = z.infer<typeof OperationReceiptSchema>;
export type ExecutionReceipt = z.infer<typeof ExecutionReceiptSchema>;

export type SourceTurnContextEntry = z.infer<typeof SourceTurnContextEntrySchema>;
export type ReplacementTurnContextEntry = z.infer<typeof ReplacementTurnContextEntrySchema>;
export type ContextEntry = z.infer<typeof ContextEntrySchema>;
export type ContextRetrievalRequirement = z.infer<typeof ContextRetrievalRequirementSchema>;
export type CacheIntent = z.infer<typeof CacheIntentSchema>;
export type ConversationContext = z.infer<typeof ConversationContextSchema>;
export type CompactionStrategy = z.infer<typeof CompactionStrategySchema>;
export type CompactionSource = z.infer<typeof CompactionSourceSchema>;
export type CompactionRecord = z.infer<typeof CompactionRecordSchema>;
export type ProcessorConfiguration = z.infer<typeof ProcessorConfigurationSchema>;
export type ProcessingState = z.infer<typeof ProcessingStateSchema>;
export type ConversationLineageParent = z.infer<typeof ConversationLineageParentSchema>;
export type ConversationLineage = z.infer<typeof ConversationLineageSchema>;
export type ConversationDocument = z.infer<typeof ConversationDocumentSchema>;
