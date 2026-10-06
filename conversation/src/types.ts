import type { z } from 'zod';
import type {
    ContextChangeSchema,
    ConversationAppendChangeSchema,
    ConversationAppendOperationSchema,
    ConversationEditChangeSchema,
} from './schemas/change.js';
import type {
    ConversationEditableBlockSchema,
    ConversationEditCommandSchema,
    ConversationEditPlanInputSchema,
    ConversationEditPlanSchema,
    ConversationEditRequestSchema,
    ConversationEditResultSchema,
    ConversationInsertedTurnSchema,
    ConversationReplacementTurnSchema,
} from './schemas/conversation-edit.js';
import type {
    ConversationEditAnchorSchema,
    ConversationEditOperationSchema,
    ConversationEditPlacementSchema,
    ConversationEditRecordRefSchema,
} from './schemas/conversation-edit-operation.js';
import type {
    AcceptedToolSelectionSchema,
    AccountingProvenanceSchema,
    AgentContentBlockSchema,
    AgentTurnProvenanceSchema,
    AgentTurnSchema,
    AppendConversationRecordsOptionsSchema,
    AppendConversationRecordsResultSchema,
    AppendConversationRecordsWithProcessingResultSchema,
    ApplicationToolCallBlockSchema,
    ApplicationToolExecutionReceiptSchema,
    AssetKindSchema,
    AssetMediaMetadataSchema,
    AssetProvenanceSchema,
    AssetSchema,
    AssetStorageSchema,
    AssetVersionBindingSchema,
    AudioBlockSchema,
    BlockSubselectionSchema,
    CacheIntentSchema,
    CompactionRecordSchema,
    CompactionSourceSchema,
    CompactionStrategySchema,
    CompleteInputPartitionSchema,
    ContentBlockSchema,
    ContextChangeOperationSchema,
    ContextChangePlacementSchema,
    ContextChangePlanInputSchema,
    ContextChangePlanSchema,
    ContextChangeProposalSchema,
    ContextChangeRequestSchema,
    ContextEntrySchema,
    ContextMeasurementSchema,
    ContextMetadataPredicateSchema,
    ContextRetrievalRequirementSchema,
    ContextSelectionActorKindSchema,
    ContextSelectionAnchorSchema,
    ContextSelectionBlockTypeSchema,
    ContextSelectionRangeSchema,
    ContextSelectionRequestSchema,
    ContextSelectionResultSchema,
    ContextSelectorSchema,
    ConversationChangeSchema,
    ConversationContextSchema,
    ConversationDiagnosticCodeSchema,
    ConversationDiagnosticSchema,
    ConversationDiagnosticStageSchema,
    ConversationDocumentSchema,
    ConversationInspectionSchema,
    ConversationLineageParentSchema,
    ConversationLineageSchema,
    ConversationMaterializedInputSchema,
    ConversationPreparedRequestRecordSchema,
    ConversationPreparedRequestSchema,
    ConversationRecordBatchSchema,
    ConversationRefSchema,
    ConversationRuntimeContextSchema,
    ConversationSelectionRequestSchema,
    ConversationSelectionResultSchema,
    ConversationSelectionSchema,
    ConversationSelectorSchema,
    ConversationSliceResultSchema,
    ConversationSliceSchema,
    ConversationStreamCursorSchema,
    ConversationStreamDecodeEvidenceSchema,
    ConversationStreamDraftBlockSchema,
    ConversationStreamEventBatchSchema,
    ConversationStreamEventSchema,
    ConversationStreamFailureDiagnosticSchema,
    ConversationStreamIdentitySchema,
    ConversationStreamReconciliationSchema,
    ConversationStreamResponseMappingSchema,
    ConversationStreamTransformationProofSchema,
    ConversationToolExecutionRequestSchema,
    ConversationToolExecutionResultSchema,
    ConversationTurnSchema,
    DecodedConversationResponseSchema,
    DerivedAgentTurnSchema,
    DerivedAssetProvenanceSchema,
    DerivedTurnProvenanceSchema,
    DocumentBlockSchema,
    ExecutedGenerationSchema,
    ExecutedToolTurnSchema,
    ExecutionReceiptSchema,
    ExtensionBlockSchema,
    ExternalAssetStorageSchema,
    ExternalizedToolArgumentsSchema,
    ExternalReferenceBlockSchema,
    GeneratedAgentTurnSchema,
    GeneratedAssetProvenanceSchema,
    GeneratedTurnProvenanceSchema,
    GenerationCostSchema,
    GenerationSchema,
    GenerationStatusSchema,
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
    InvalidatedReplayArchiveSchema,
    InvalidToolArgumentsSchema,
    JsonBlockSchema,
    JsonObjectSchema,
    JsonPathSchema,
    JsonPathSegmentSchema,
    JsonPointerSchema,
    JsonPreflightDiagnosticCodeSchema,
    JsonPreflightDiagnosticSchema,
    JsonValueSchema,
    MediaSelectionRangeSchema,
    MetadataSchema,
    ModelTargetSchema,
    NativeConversationImportCompletenessSchema,
    NativeConversationImportDiagnosticCodeSchema,
    NativeConversationImportDiagnosticSchema,
    NativeConversationImportOptionsSchema,
    NativeConversationImportReportSchema,
    NativeConversationImportResultSchema,
    NativeIdentitySchema,
    NativeItemMappingSchema,
    NativeReplayBlockSchema,
    NativeStreamPathSegmentSchema,
    NativeStreamPositionSchema,
    NestedToolResultContentBlockSchema,
    NonGeneratedTurnProvenanceSchema,
    NongeneratedAgentTurnSchema,
    OperationReceiptSchema,
    PageRangeSchema,
    PendingApplicationToolCallSchema,
    ProcessingAppendAcceptanceSchema,
    ProcessingAttemptReceiptSchema,
    ProcessingBudgetSchema,
    ProcessingChangeOperationSchema,
    ProcessingChangeSchema,
    ProcessingCompletionReceiptSchema,
    ProcessingJobSchema,
    ProcessingJobSelectionSchema,
    ProcessingOperationSchema,
    ProcessingOutputReceiptSchema,
    ProcessingReadinessCoverageSchema,
    ProcessingResolvedInputSchema,
    ProcessingRunResultSchema,
    ProcessingStateSchema,
    ProcessingSupersessionReceiptSchema,
    ProcessorConfigurationSchema,
    ProgramContentBlockSchema,
    ProgramTurnPresentationSchema,
    ProgramTurnSchema,
    ReasoningBlockSchema,
    ReceivedAssetProvenanceSchema,
    ReceivedTurnProvenanceSchema,
    ReplacementTurnContextEntrySchema,
    ReplayCompatibilityScopeSchema,
    ReplayDependenciesSchema,
    ReportedUsageSchema,
    RequestReceiptSchema,
    RequestSourceViewReferenceSchema,
    ResolvedConversationRuntimeContextSchema,
    RetrievalCapabilitySchema,
    SelectedBlockSchema,
    SelectedContextBlocksSchema,
    SelectedContextEntrySchema,
    SelectionBinaryEvidenceSchema,
    SelectionMediaEvidenceSchema,
    SemanticConversationDiagnosticCodeSchema,
    SemanticConversationDiagnosticSchema,
    SourceTurnContextEntrySchema,
    StructuredToolArgumentsSchema,
    TextAssetToolArgumentHydrationSchema,
    TextBlockSchema,
    TextCodePointRangeSchema,
    TimeRangeSchema,
    ToolArgumentHydrationSchema,
    ToolArgumentsSchema,
    ToolCallBlockSchema,
    ToolCallSourceRefSchema,
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
import type {
    DerivedBlockLineageGroupSchema,
    DerivedBlockLineageSchema,
    JsonInverseMappingSchema,
    JsonSourceRegionSchema,
    SourceBlockSliceSchema,
} from './schemas/source-slices.js';

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
export type AppendConversationRecordsWithProcessingResult = z.infer<
    typeof AppendConversationRecordsWithProcessingResultSchema
>;
export type DecodedConversationResponse = z.infer<typeof DecodedConversationResponseSchema>;
export type NativeStreamPathSegment = z.infer<typeof NativeStreamPathSegmentSchema>;
export type NativeStreamPosition = z.infer<typeof NativeStreamPositionSchema>;
export type ConversationStreamDraftBlock = z.infer<typeof ConversationStreamDraftBlockSchema>;
export type ConversationStreamFailureDiagnostic = z.infer<typeof ConversationStreamFailureDiagnosticSchema>;
export type ConversationStreamReconciliation = z.infer<typeof ConversationStreamReconciliationSchema>;
export type ConversationStreamTransformationProof = z.infer<typeof ConversationStreamTransformationProofSchema>;
export type ConversationStreamResponseMapping = z.infer<typeof ConversationStreamResponseMappingSchema>;
export type ConversationStreamDecodeEvidence = z.infer<typeof ConversationStreamDecodeEvidenceSchema>;
export type ConversationStreamCursor = z.infer<typeof ConversationStreamCursorSchema>;
export type ConversationStreamIdentity = z.infer<typeof ConversationStreamIdentitySchema>;
export type ConversationStreamEvent = z.infer<typeof ConversationStreamEventSchema>;
export type ConversationStreamEventBatch = z.infer<typeof ConversationStreamEventBatchSchema>;

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
export type JsonPathSegment = z.infer<typeof JsonPathSegmentSchema>;
export type JsonPath = z.infer<typeof JsonPathSchema>;
export type TextAssetToolArgumentHydration = z.infer<typeof TextAssetToolArgumentHydrationSchema>;
export type ToolArgumentHydration = z.infer<typeof ToolArgumentHydrationSchema>;
export type InvalidatedReplayArchive = z.infer<typeof InvalidatedReplayArchiveSchema>;
export type ExternalizedToolArguments = z.infer<typeof ExternalizedToolArgumentsSchema>;
export type ToolArguments = z.infer<typeof ToolArgumentsSchema>;
export type ToolCallBlock = z.infer<typeof ToolCallBlockSchema>;
export type ToolCallSourceRef = z.infer<typeof ToolCallSourceRefSchema>;
export type ApplicationToolCallBlock = z.infer<typeof ApplicationToolCallBlockSchema>;
export type PendingApplicationToolCall = z.infer<typeof PendingApplicationToolCallSchema>;
export type ApplicationToolExecutionReceipt = z.infer<typeof ApplicationToolExecutionReceiptSchema>;
export type ExecutedToolTurn = z.infer<typeof ExecutedToolTurnSchema>;
export type ConversationToolExecutionRequest = z.infer<typeof ConversationToolExecutionRequestSchema>;
export type ConversationToolExecutionResult = z.infer<typeof ConversationToolExecutionResultSchema>;
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
export type ProgramTurnPresentation = z.infer<typeof ProgramTurnPresentationSchema>;

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
export type ConversationMaterializedInput = z.infer<typeof ConversationMaterializedInputSchema>;
export type ConversationRuntimeContext = z.infer<typeof ConversationRuntimeContextSchema>;
export type ResolvedConversationRuntimeContext = z.infer<typeof ResolvedConversationRuntimeContextSchema>;
export type ConversationPreparedRequestRecord = z.infer<typeof ConversationPreparedRequestRecordSchema>;
export type ConversationPreparedRequest = z.infer<typeof ConversationPreparedRequestSchema>;
export type GenerationCost = z.infer<typeof GenerationCostSchema>;
export type GenerationUsage = z.infer<typeof GenerationUsageSchema>;
export type GenerationStatus = z.infer<typeof GenerationStatusSchema>;
export type ContextMeasurement = z.infer<typeof ContextMeasurementSchema>;
export type ModelTarget = z.infer<typeof ModelTargetSchema>;
export type AssetVersionBinding = z.infer<typeof AssetVersionBindingSchema>;
export type NativeItemMapping = z.infer<typeof NativeItemMappingSchema>;
export type RequestReceipt = z.infer<typeof RequestReceiptSchema>;
export type RequestSourceViewReference = z.infer<typeof RequestSourceViewReferenceSchema>;
export type GenerationTimestamps = z.infer<typeof GenerationTimestampsSchema>;
export type ExecutedGeneration = z.infer<typeof ExecutedGenerationSchema>;
export type ImportedGeneration = z.infer<typeof ImportedGenerationSchema>;
export type Generation = z.infer<typeof GenerationSchema>;
export type OperationReceipt = z.infer<typeof OperationReceiptSchema>;
export type ProcessingOperation = z.infer<typeof ProcessingOperationSchema>;
export type ProcessingAppendAcceptance = z.infer<typeof ProcessingAppendAcceptanceSchema>;
export type ProcessingChangeOperation = z.infer<typeof ProcessingChangeOperationSchema>;
export type ProcessingChange = z.infer<typeof ProcessingChangeSchema>;
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
export type ProcessingBudget = z.infer<typeof ProcessingBudgetSchema>;
export type ProcessingJobSelection = z.infer<typeof ProcessingJobSelectionSchema>;
export type ProcessingJob = z.infer<typeof ProcessingJobSchema>;
export type ProcessingResolvedInput = z.infer<typeof ProcessingResolvedInputSchema>;
export type ProcessingAttemptReceipt = z.infer<typeof ProcessingAttemptReceiptSchema>;
export type ProcessingOutputReceipt = z.infer<typeof ProcessingOutputReceiptSchema>;
export type ProcessingCompletionReceipt = z.infer<typeof ProcessingCompletionReceiptSchema>;
export type ProcessingSupersessionReceipt = z.infer<typeof ProcessingSupersessionReceiptSchema>;
export type ProcessingReadinessCoverage = z.infer<typeof ProcessingReadinessCoverageSchema>;
export type ProcessingState = z.infer<typeof ProcessingStateSchema>;
export type ProcessingRunResult = z.infer<typeof ProcessingRunResultSchema>;
export type ConversationLineageParent = z.infer<typeof ConversationLineageParentSchema>;
export type ConversationLineage = z.infer<typeof ConversationLineageSchema>;
export type ConversationDocument = z.infer<typeof ConversationDocumentSchema>;
export type ContextChangePlan = z.infer<typeof ContextChangePlanSchema>;
export type ContextChangePlanInput = z.infer<typeof ContextChangePlanInputSchema>;
export type ContextChangePlacement = z.infer<typeof ContextChangePlacementSchema>;
export type ContextChangeOperation = z.infer<typeof ContextChangeOperationSchema>;
export type ContextChangeProposal = z.infer<typeof ContextChangeProposalSchema>;
export type ContextChangeRequest = z.infer<typeof ContextChangeRequestSchema>;
export type ConversationChange = z.infer<typeof ConversationChangeSchema>;

export type NativeConversationImportCompleteness = z.infer<typeof NativeConversationImportCompletenessSchema>;
export type NativeConversationImportOptions = z.infer<typeof NativeConversationImportOptionsSchema>;
export type NativeConversationImportDiagnosticCode = z.infer<typeof NativeConversationImportDiagnosticCodeSchema>;
export type NativeConversationImportDiagnostic = z.infer<typeof NativeConversationImportDiagnosticSchema>;
export type NativeConversationImportReport = z.infer<typeof NativeConversationImportReportSchema>;
export type NativeConversationImportResult = z.infer<typeof NativeConversationImportResultSchema>;

export type SelectedContextBlocks = z.infer<typeof SelectedContextBlocksSchema>;
export type ContextSelectionActorKind = z.infer<typeof ContextSelectionActorKindSchema>;
export type ContextSelectionBlockType = z.infer<typeof ContextSelectionBlockTypeSchema>;
export type ContextSelectionAnchor = z.infer<typeof ContextSelectionAnchorSchema>;
export type ContextSelectionRange = z.infer<typeof ContextSelectionRangeSchema>;
export type ContextMetadataPredicate = z.infer<typeof ContextMetadataPredicateSchema>;
export type ContextSelector = z.infer<typeof ContextSelectorSchema>;
export type ContextSelectionRequest = z.infer<typeof ContextSelectionRequestSchema>;
export type ContextSelectionResult = z.infer<typeof ContextSelectionResultSchema>;

export type TextCodePointRange = z.infer<typeof TextCodePointRangeSchema>;
export type JsonPointer = z.infer<typeof JsonPointerSchema>;
export type MediaSelectionRange = z.infer<typeof MediaSelectionRangeSchema>;
export type BlockSubselection = z.infer<typeof BlockSubselectionSchema>;
export type ConversationSelector = z.infer<typeof ConversationSelectorSchema>;
export type ConversationSelectionRequest = z.infer<typeof ConversationSelectionRequestSchema>;
export type SelectionBinaryEvidence = z.infer<typeof SelectionBinaryEvidenceSchema>;
export type SelectionMediaEvidence = z.infer<typeof SelectionMediaEvidenceSchema>;
export type SelectedBlock = z.infer<typeof SelectedBlockSchema>;
export type SelectedContextEntry = z.infer<typeof SelectedContextEntrySchema>;
export type ConversationSelection = z.infer<typeof ConversationSelectionSchema>;
export type ConversationSelectionResult = z.infer<typeof ConversationSelectionResultSchema>;
export type ConversationSlice = z.infer<typeof ConversationSliceSchema>;
export type ConversationSliceResult = z.infer<typeof ConversationSliceResultSchema>;

export type ConversationEditRecordRef = z.infer<typeof ConversationEditRecordRefSchema>;

export type ConversationEditAnchor = z.infer<typeof ConversationEditAnchorSchema>;

export type ConversationEditOperation = z.infer<typeof ConversationEditOperationSchema>;

export type ConversationEditableBlock = z.infer<typeof ConversationEditableBlockSchema>;

export type ConversationInsertedTurn = z.infer<typeof ConversationInsertedTurnSchema>;

export type ConversationReplacementTurn = z.infer<typeof ConversationReplacementTurnSchema>;

export type ConversationEditCommand = z.infer<typeof ConversationEditCommandSchema>;

export type ConversationEditPlanInput = z.infer<typeof ConversationEditPlanInputSchema>;

export type ConversationEditRequest = z.infer<typeof ConversationEditRequestSchema>;

export type ConversationEditPlan = z.infer<typeof ConversationEditPlanSchema>;

export type ConversationEditResult = z.infer<typeof ConversationEditResultSchema>;
export type ConversationDeletedTurnRef = z.infer<
    typeof import('./schemas/conversation-delete-operation.js').ConversationDeletedTurnRefSchema
>;
export type ConversationDeletedTurn = z.infer<
    typeof import('./schemas/conversation-delete-operation.js').ConversationDeletedTurnSchema
>;
export type ConversationDeleteOperation = z.infer<
    typeof import('./schemas/conversation-delete-operation.js').ConversationDeleteOperationSchema
>;
export type ConversationDeletePlanInput = z.infer<
    typeof import('./schemas/conversation-delete.js').ConversationDeletePlanInputSchema
>;
export type ConversationDeleteRequest = z.infer<
    typeof import('./schemas/conversation-delete.js').ConversationDeleteRequestSchema
>;
export type ConversationDeletePlan = z.infer<
    typeof import('./schemas/conversation-delete.js').ConversationDeletePlanSchema
>;
export type ConversationDeleteResult = z.infer<
    typeof import('./schemas/conversation-delete.js').ConversationDeleteResultSchema
>;

export type ContextChange = z.infer<typeof ContextChangeSchema>;

export type ConversationAppendOperation = z.infer<typeof ConversationAppendOperationSchema>;

export type ConversationAppendChange = z.infer<typeof ConversationAppendChangeSchema>;

export type ConversationEditChange = z.infer<typeof ConversationEditChangeSchema>;

export type AcceptedToolSelection = z.infer<typeof AcceptedToolSelectionSchema>;

export type ConversationEditPlacement = z.infer<typeof ConversationEditPlacementSchema>;

export type SourceBlockSlice = z.infer<typeof SourceBlockSliceSchema>;
export type DerivedBlockLineageGroup = z.infer<typeof DerivedBlockLineageGroupSchema>;
export type DerivedBlockLineage = z.infer<typeof DerivedBlockLineageSchema>;
export type JsonSourceRegion = z.infer<typeof JsonSourceRegionSchema>;
export type JsonInverseMapping = z.infer<typeof JsonInverseMappingSchema>;

export type ConversationSliceEditCommand = z.infer<
    typeof import('./schemas/conversation-slice-edit.js').ConversationSliceEditCommandSchema
>;
export type ConversationSliceEditPlanInput = z.infer<
    typeof import('./schemas/conversation-slice-edit.js').ConversationSliceEditPlanInputSchema
>;
export type ConversationSliceEditRequest = z.infer<
    typeof import('./schemas/conversation-slice-edit.js').ConversationSliceEditRequestSchema
>;
export type ConversationSliceEditOperation = z.infer<
    typeof import('./schemas/conversation-edit-operation.js').ConversationSliceEditOperationSchema
>;
export type ConversationEditOperationV1 = z.infer<
    typeof import('./schemas/conversation-edit-operation.js').ConversationEditOperationV1Schema
>;

export type DerivedLineageVerificationScope = z.infer<
    typeof import('./schemas/source-slices.js').DerivedLineageVerificationScopeSchema
>;

export type JsonMinificationConfiguration = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationConfigurationSchema
>;
export type JsonMinificationTransform = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationTransformSchema
>;
export type JsonMinificationCandidate = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationCandidateSchema
>;
export type JsonMinificationMeasurementIdentity = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationMeasurementIdentitySchema
>;
export type JsonMinificationMeasurement = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationMeasurementSchema
>;
export type JsonMinificationProposal = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationProposalSchema
>;
export type JsonMinificationNoOpReason = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationNoOpReasonSchema
>;
export type JsonMinificationApplication = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationApplicationSchema
>;

export type JsonMinificationMeasuredProjection = z.infer<
    typeof import('./schemas/json-minification.js').JsonMinificationMeasuredProjectionSchema
>;

export type ToolRetrievalExcerpt = z.infer<typeof import('./schemas/execution.js').ToolRetrievalExcerptSchema>;

export type ConversationIndexedUpgradeOperation = z.infer<
    typeof import('./schemas/execution.js').IndexedConversationUpgradeOperationSchema
>;
