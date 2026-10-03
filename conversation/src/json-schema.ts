import type { z } from 'zod';
import {
    AcceptedToolSelectionSchema,
    AppendConversationRecordsOptionsSchema,
    AppendConversationRecordsResultSchema,
    BlockSubselectionSchema,
    ContentBlockSchema,
    ContextChangeOperationSchema,
    ContextChangePlacementSchema,
    ContextChangePlanInputSchema,
    ContextChangePlanSchema,
    ContextChangeProposalSchema,
    ContextChangeRequestSchema,
    ContextChangeSchema,
    ContextSelectionRequestSchema,
    ContextSelectionResultSchema,
    ContextSelectorSchema,
    ConversationAcceptedOutputFragmentSchema,
    ConversationAppendChangeSchema,
    ConversationAppendOperationSchema,
    ConversationChangeSchema,
    ConversationDiagnosticSchema,
    ConversationDocumentSchema,
    ConversationEditAnchorSchema,
    ConversationEditableBlockSchema,
    ConversationEditChangeSchema,
    ConversationEditCommandSchema,
    ConversationEditOperationSchema,
    ConversationEditOperationV1Schema,
    ConversationEditPlacementSchema,
    ConversationEditPlanInputSchema,
    ConversationEditPlanSchema,
    ConversationEditRecordRefSchema,
    ConversationEditRequestSchema,
    ConversationEditResultSchema,
    ConversationInsertedTurnSchema,
    ConversationInspectionSchema,
    ConversationModelSwitchBlockerSchema,
    ConversationModelSwitchBudgetAnalysisSchema,
    ConversationModelSwitchNextRequestChangeSchema,
    ConversationModelSwitchPlanSchema,
    ConversationModelSwitchRequestSchema,
    ConversationRecordBatchSchema,
    ConversationReplacementTurnSchema,
    ConversationSelectionRequestSchema,
    ConversationSelectionResultSchema,
    ConversationSelectionSchema,
    ConversationSelectorSchema,
    ConversationSliceEditCommandSchema,
    ConversationSliceEditOperationSchema,
    ConversationSliceEditPlanInputSchema,
    ConversationSliceEditRequestSchema,
    ConversationSliceResultSchema,
    ConversationSliceSchema,
    ConversationStreamCursorSchema,
    ConversationStreamEventBatchSchema,
    ConversationStreamEventSchema,
    ConversationStreamIdentitySchema,
    ConversationToolExecutionRequestSchema,
    ConversationToolExecutionResultSchema,
    ConversationTranscriptFragmentSchema,
    ConversationTranscriptProjectionInputSchema,
    ConversationTurnSchema,
    DecodedConversationResponseSchema,
    DerivedBlockLineageGroupSchema,
    DerivedBlockLineageSchema,
    DerivedLineageVerificationScopeSchema,
    GenerationSchema,
    JsonInverseMappingSchema,
    JsonPointerSchema,
    JsonSourceRegionSchema,
    MediaSelectionRangeSchema,
    NativeConversationImportDiagnosticSchema,
    NativeConversationImportOptionsSchema,
    NativeConversationImportReportSchema,
    NativeConversationImportResultSchema,
    ProcessingJobSchema,
    ProcessingOutputReceiptSchema,
    ProcessingReadinessCoverageSchema,
    ProcessingRunResultSchema,
    SelectedBlockSchema,
    SelectedContextBlocksSchema,
    SelectedContextEntrySchema,
    SelectionBinaryEvidenceSchema,
    SelectionMediaEvidenceSchema,
    SourceBlockSliceSchema,
    TextCodePointRangeSchema,
} from './schemas/index.js';
import {
    JsonMinificationApplicationSchema,
    JsonMinificationCandidateSchema,
    JsonMinificationConfigurationSchema,
    JsonMinificationMeasuredProjectionSchema,
    JsonMinificationMeasurementIdentitySchema,
    JsonMinificationMeasurementSchema,
    JsonMinificationNoOpReasonSchema,
    JsonMinificationProposalSchema,
    JsonMinificationTransformSchema,
} from './schemas/json-minification.js';
import { CONVERSATION_EXPERIMENTAL_REVISION } from './schemas/primitives.js';

export type GeneratedConversationJsonSchema = Readonly<Record<string, unknown>>;

const JSON_SCHEMA_OPTIONS = {
    target: 'draft-2020-12',
    io: 'input',
    cycles: 'ref',
    reused: 'ref',
    unrepresentable: 'throw',
} as const;

function sortJsonValue<T>(value: T): T {
    if (Array.isArray(value)) {
        return value.map((item) => sortJsonValue(item)) as T;
    }
    if (value !== null && typeof value === 'object') {
        const sorted = Object.fromEntries(
            Object.entries(value)
                .sort(([first], [second]) => (first < second ? -1 : first > second ? 1 : 0))
                .map(([key, item]) => [key, sortJsonValue(item)]),
        );
        return sorted as T;
    }
    return value;
}

function deepFreeze<T>(value: T): Readonly<T> {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
        for (const child of Object.values(value)) {
            deepFreeze(child);
        }
        Object.freeze(value);
    }
    return value;
}

function emitJsonSchema(schema: z.ZodType, name: string): GeneratedConversationJsonSchema {
    const generated = schema.toJSONSchema(JSON_SCHEMA_OPTIONS);
    return deepFreeze(
        sortJsonValue({
            ...generated,
            $id: `urn:llumiverse:conversation:${CONVERSATION_EXPERIMENTAL_REVISION}:${name}`,
        }),
    );
}

export const ConversationDocumentJsonSchema = emitJsonSchema(ConversationDocumentSchema, 'document');
export const ConversationModelSwitchRequestJsonSchema = emitJsonSchema(
    ConversationModelSwitchRequestSchema,
    'model-switch-request',
);
export const ConversationModelSwitchBlockerJsonSchema = emitJsonSchema(
    ConversationModelSwitchBlockerSchema,
    'model-switch-blocker',
);
export const ConversationModelSwitchBudgetAnalysisJsonSchema = emitJsonSchema(
    ConversationModelSwitchBudgetAnalysisSchema,
    'model-switch-budget-analysis',
);
export const ConversationModelSwitchPlanJsonSchema = emitJsonSchema(
    ConversationModelSwitchPlanSchema,
    'model-switch-plan',
);
export const ConversationModelSwitchNextRequestChangeJsonSchema = emitJsonSchema(
    ConversationModelSwitchNextRequestChangeSchema,
    'model-switch-next-request-change',
);
export const ConversationTurnJsonSchema = emitJsonSchema(ConversationTurnSchema, 'turn');
export const ConversationContentBlockJsonSchema = emitJsonSchema(ContentBlockSchema, 'content-block');
export const ConversationGenerationJsonSchema = emitJsonSchema(GenerationSchema, 'generation');
export const ConversationDiagnosticJsonSchema = emitJsonSchema(ConversationDiagnosticSchema, 'diagnostic');
export const ConversationInspectionJsonSchema = emitJsonSchema(ConversationInspectionSchema, 'inspection');
export const ConversationRecordBatchJsonSchema = emitJsonSchema(ConversationRecordBatchSchema, 'record-batch');
export const ConversationAcceptedOutputFragmentJsonSchema = emitJsonSchema(
    ConversationAcceptedOutputFragmentSchema,
    'accepted-output-fragment',
);
export const ConversationStreamEventJsonSchema = emitJsonSchema(ConversationStreamEventSchema, 'stream-event');
export const ConversationStreamEventBatchJsonSchema = emitJsonSchema(
    ConversationStreamEventBatchSchema,
    'stream-event-batch',
);
export const ConversationStreamCursorJsonSchema = emitJsonSchema(ConversationStreamCursorSchema, 'stream-cursor');
export const ConversationStreamIdentityJsonSchema = emitJsonSchema(ConversationStreamIdentitySchema, 'stream-identity');
export const ConversationToolExecutionRequestJsonSchema = emitJsonSchema(
    ConversationToolExecutionRequestSchema,
    'tool-execution-request',
);
export const ConversationToolExecutionResultJsonSchema = emitJsonSchema(
    ConversationToolExecutionResultSchema,
    'tool-execution-result',
);
export const ConversationTranscriptFragmentJsonSchema = emitJsonSchema(
    ConversationTranscriptFragmentSchema,
    'transcript-fragment',
);
export const ConversationTranscriptProjectionInputJsonSchema = emitJsonSchema(
    ConversationTranscriptProjectionInputSchema,
    'transcript-projection-input',
);
export const AppendConversationRecordsOptionsJsonSchema = emitJsonSchema(
    AppendConversationRecordsOptionsSchema,
    'append-options',
);
export const AppendConversationRecordsResultJsonSchema = emitJsonSchema(
    AppendConversationRecordsResultSchema,
    'append-result',
);
export const DecodedConversationResponseJsonSchema = emitJsonSchema(
    DecodedConversationResponseSchema,
    'decoded-response',
);

export const NativeConversationImportOptionsJsonSchema = emitJsonSchema(
    NativeConversationImportOptionsSchema,
    'native-import-options',
);
export const NativeConversationImportDiagnosticJsonSchema = emitJsonSchema(
    NativeConversationImportDiagnosticSchema,
    'native-import-diagnostic',
);
export const NativeConversationImportReportJsonSchema = emitJsonSchema(
    NativeConversationImportReportSchema,
    'native-import-report',
);
export const NativeConversationImportResultJsonSchema = emitJsonSchema(
    NativeConversationImportResultSchema,
    'native-import-result',
);

export const ContextChangeRequestJsonSchema = emitJsonSchema(ContextChangeRequestSchema, 'context-change-request');
export const ContextChangeProposalJsonSchema = emitJsonSchema(ContextChangeProposalSchema, 'context-change-proposal');
export const ContextChangeOperationJsonSchema = emitJsonSchema(
    ContextChangeOperationSchema,
    'context-change-operation',
);
export const ContextChangePlacementJsonSchema = emitJsonSchema(
    ContextChangePlacementSchema,
    'context-change-placement',
);
export const ConversationChangeJsonSchema = emitJsonSchema(ConversationChangeSchema, 'change');

export const ContextSelectorJsonSchema = emitJsonSchema(ContextSelectorSchema, 'context-selector');
export const ContextSelectionRequestJsonSchema = emitJsonSchema(
    ContextSelectionRequestSchema,
    'context-selection-request',
);
export const ContextSelectionResultJsonSchema = emitJsonSchema(
    ContextSelectionResultSchema,
    'context-selection-result',
);
export const ContextChangePlanJsonSchema = emitJsonSchema(ContextChangePlanSchema, 'context-change-plan');
export const ContextChangePlanInputJsonSchema = emitJsonSchema(
    ContextChangePlanInputSchema,
    'context-change-plan-input',
);
export const SelectedContextBlocksJsonSchema = emitJsonSchema(SelectedContextBlocksSchema, 'selected-context-blocks');

export const ConversationSelectorJsonSchema = emitJsonSchema(ConversationSelectorSchema, 'conversation-selector');
export const ConversationSelectionRequestJsonSchema = emitJsonSchema(
    ConversationSelectionRequestSchema,
    'conversation-selection-request',
);
export const ConversationSelectionJsonSchema = emitJsonSchema(ConversationSelectionSchema, 'conversation-selection');
export const ConversationSelectionResultJsonSchema = emitJsonSchema(
    ConversationSelectionResultSchema,
    'conversation-selection-result',
);
export const ConversationSliceJsonSchema = emitJsonSchema(ConversationSliceSchema, 'conversation-slice');
export const ConversationSliceResultJsonSchema = emitJsonSchema(
    ConversationSliceResultSchema,
    'conversation-slice-result',
);
export const BlockSubselectionJsonSchema = emitJsonSchema(BlockSubselectionSchema, 'block-subselection');
export const SelectedBlockJsonSchema = emitJsonSchema(SelectedBlockSchema, 'selected-block');
export const SelectionMediaEvidenceJsonSchema = emitJsonSchema(
    SelectionMediaEvidenceSchema,
    'selection-media-evidence',
);

export const TextCodePointRangeJsonSchema = emitJsonSchema(TextCodePointRangeSchema, 'text-code-point-range');

export const JsonPointerJsonSchema = emitJsonSchema(JsonPointerSchema, 'json-pointer');

export const MediaSelectionRangeJsonSchema = emitJsonSchema(MediaSelectionRangeSchema, 'media-selection-range');

export const SelectionBinaryEvidenceJsonSchema = emitJsonSchema(
    SelectionBinaryEvidenceSchema,
    'selection-binary-evidence',
);

export const SelectedContextEntryJsonSchema = emitJsonSchema(SelectedContextEntrySchema, 'selected-context-entry');

export const ProcessingJobJsonSchema = emitJsonSchema(ProcessingJobSchema, 'processing-job');
export const ProcessingOutputReceiptJsonSchema = emitJsonSchema(
    ProcessingOutputReceiptSchema,
    'processing-output-receipt',
);
export const ProcessingReadinessCoverageJsonSchema = emitJsonSchema(
    ProcessingReadinessCoverageSchema,
    'processing-readiness-coverage',
);
export const ProcessingRunResultJsonSchema = emitJsonSchema(ProcessingRunResultSchema, 'processing-run-result');

export const ConversationEditRecordRefJsonSchema = emitJsonSchema(
    ConversationEditRecordRefSchema,
    'conversation-edit-record-ref',
);
export const ConversationEditAnchorJsonSchema = emitJsonSchema(
    ConversationEditAnchorSchema,
    'conversation-edit-anchor',
);
export const ConversationEditOperationJsonSchema = emitJsonSchema(
    ConversationEditOperationSchema,
    'conversation-edit-operation',
);
export const ConversationEditableBlockJsonSchema = emitJsonSchema(
    ConversationEditableBlockSchema,
    'conversation-editable-block',
);
export const ConversationInsertedTurnJsonSchema = emitJsonSchema(
    ConversationInsertedTurnSchema,
    'conversation-inserted-turn',
);
export const ConversationReplacementTurnJsonSchema = emitJsonSchema(
    ConversationReplacementTurnSchema,
    'conversation-replacement-turn',
);
export const ConversationEditCommandJsonSchema = emitJsonSchema(
    ConversationEditCommandSchema,
    'conversation-edit-command',
);
export const ConversationEditPlanInputJsonSchema = emitJsonSchema(
    ConversationEditPlanInputSchema,
    'conversation-edit-plan-input',
);
export const ConversationEditRequestJsonSchema = emitJsonSchema(
    ConversationEditRequestSchema,
    'conversation-edit-request',
);
export const ConversationEditPlanJsonSchema = emitJsonSchema(ConversationEditPlanSchema, 'conversation-edit-plan');
export const ConversationEditResultJsonSchema = emitJsonSchema(
    ConversationEditResultSchema,
    'conversation-edit-result',
);
export const ContextChangeJsonSchema = emitJsonSchema(ContextChangeSchema, 'context-change');
export const ConversationAppendOperationJsonSchema = emitJsonSchema(
    ConversationAppendOperationSchema,
    'conversation-append-operation',
);
export const ConversationAppendChangeJsonSchema = emitJsonSchema(
    ConversationAppendChangeSchema,
    'conversation-append-change',
);
export const ConversationEditChangeJsonSchema = emitJsonSchema(
    ConversationEditChangeSchema,
    'conversation-edit-change',
);

export const AcceptedToolSelectionJsonSchema = emitJsonSchema(AcceptedToolSelectionSchema, 'accepted-tool-selection');

export const ConversationEditPlacementJsonSchema = emitJsonSchema(
    ConversationEditPlacementSchema,
    'conversation-edit-placement',
);

export const ConversationEditOperationV1JsonSchema = emitJsonSchema(
    ConversationEditOperationV1Schema,
    'conversation-edit-operation-v1',
);
export const ConversationSliceEditOperationJsonSchema = emitJsonSchema(
    ConversationSliceEditOperationSchema,
    'conversation-slice-edit-operation',
);
export const ConversationSliceEditCommandJsonSchema = emitJsonSchema(
    ConversationSliceEditCommandSchema,
    'conversation-slice-edit-command',
);
export const ConversationSliceEditPlanInputJsonSchema = emitJsonSchema(
    ConversationSliceEditPlanInputSchema,
    'conversation-slice-edit-plan-input',
);
export const ConversationSliceEditRequestJsonSchema = emitJsonSchema(
    ConversationSliceEditRequestSchema,
    'conversation-slice-edit-request',
);
export const JsonSourceRegionJsonSchema = emitJsonSchema(JsonSourceRegionSchema, 'json-source-region');
export const SourceBlockSliceJsonSchema = emitJsonSchema(SourceBlockSliceSchema, 'source-block-slice');
export const JsonInverseMappingJsonSchema = emitJsonSchema(JsonInverseMappingSchema, 'json-inverse-mapping');
export const DerivedBlockLineageGroupJsonSchema = emitJsonSchema(
    DerivedBlockLineageGroupSchema,
    'derived-block-lineage-group',
);
export const DerivedBlockLineageJsonSchema = emitJsonSchema(DerivedBlockLineageSchema, 'derived-block-lineage');

export const DerivedLineageVerificationScopeJsonSchema = emitJsonSchema(
    DerivedLineageVerificationScopeSchema,
    'derived-lineage-verification-scope',
);

export const JsonMinificationConfigurationJsonSchema = emitJsonSchema(
    JsonMinificationConfigurationSchema,
    'json-minification-configuration',
);
export const JsonMinificationTransformJsonSchema = emitJsonSchema(
    JsonMinificationTransformSchema,
    'json-minification-transform',
);
export const JsonMinificationCandidateJsonSchema = emitJsonSchema(
    JsonMinificationCandidateSchema,
    'json-minification-candidate',
);
export const JsonMinificationMeasurementIdentityJsonSchema = emitJsonSchema(
    JsonMinificationMeasurementIdentitySchema,
    'json-minification-measurementidentity',
);
export const JsonMinificationMeasurementJsonSchema = emitJsonSchema(
    JsonMinificationMeasurementSchema,
    'json-minification-measurement',
);
export const JsonMinificationProposalJsonSchema = emitJsonSchema(
    JsonMinificationProposalSchema,
    'json-minification-proposal',
);
export const JsonMinificationNoOpReasonJsonSchema = emitJsonSchema(
    JsonMinificationNoOpReasonSchema,
    'json-minification-noopreason',
);
export const JsonMinificationApplicationJsonSchema = emitJsonSchema(
    JsonMinificationApplicationSchema,
    'json-minification-application',
);

export const JsonMinificationMeasuredProjectionJsonSchema = emitJsonSchema(
    JsonMinificationMeasuredProjectionSchema,
    'json-minification-measured-projection',
);

export const CONVERSATION_JSON_SCHEMAS = Object.freeze({
    conversation_model_switch_request: ConversationModelSwitchRequestJsonSchema,
    conversation_model_switch_blocker: ConversationModelSwitchBlockerJsonSchema,
    conversation_model_switch_budget_analysis: ConversationModelSwitchBudgetAnalysisJsonSchema,
    conversation_model_switch_plan: ConversationModelSwitchPlanJsonSchema,
    conversation_model_switch_next_request_change: ConversationModelSwitchNextRequestChangeJsonSchema,
    json_minification_configuration: JsonMinificationConfigurationJsonSchema,
    json_minification_transform: JsonMinificationTransformJsonSchema,
    json_minification_candidate: JsonMinificationCandidateJsonSchema,
    json_minification_measurement_identity: JsonMinificationMeasurementIdentityJsonSchema,
    json_minification_measured_projection: JsonMinificationMeasuredProjectionJsonSchema,
    json_minification_measurement: JsonMinificationMeasurementJsonSchema,
    json_minification_proposal: JsonMinificationProposalJsonSchema,
    json_minification_no_op_reason: JsonMinificationNoOpReasonJsonSchema,
    json_minification_application: JsonMinificationApplicationJsonSchema,

    conversation_edit_operation_v1: ConversationEditOperationV1JsonSchema,
    conversation_slice_edit_operation: ConversationSliceEditOperationJsonSchema,
    conversation_slice_edit_command: ConversationSliceEditCommandJsonSchema,
    conversation_slice_edit_plan_input: ConversationSliceEditPlanInputJsonSchema,
    conversation_slice_edit_request: ConversationSliceEditRequestJsonSchema,
    json_source_region: JsonSourceRegionJsonSchema,
    source_block_slice: SourceBlockSliceJsonSchema,
    json_inverse_mapping: JsonInverseMappingJsonSchema,
    derived_block_lineage_group: DerivedBlockLineageGroupJsonSchema,
    derived_block_lineage: DerivedBlockLineageJsonSchema,
    derived_lineage_verification_scope: DerivedLineageVerificationScopeJsonSchema,
    processing_job: ProcessingJobJsonSchema,
    processing_output_receipt: ProcessingOutputReceiptJsonSchema,
    processing_readiness_coverage: ProcessingReadinessCoverageJsonSchema,
    processing_run_result: ProcessingRunResultJsonSchema,
    conversation_edit_placement: ConversationEditPlacementJsonSchema,
    accepted_tool_selection: AcceptedToolSelectionJsonSchema,
    conversation_edit_record_ref: ConversationEditRecordRefJsonSchema,
    conversation_edit_anchor: ConversationEditAnchorJsonSchema,
    conversation_edit_operation: ConversationEditOperationJsonSchema,
    conversation_editable_block: ConversationEditableBlockJsonSchema,
    conversation_inserted_turn: ConversationInsertedTurnJsonSchema,
    conversation_replacement_turn: ConversationReplacementTurnJsonSchema,
    conversation_edit_command: ConversationEditCommandJsonSchema,
    conversation_edit_plan_input: ConversationEditPlanInputJsonSchema,
    conversation_edit_request: ConversationEditRequestJsonSchema,
    conversation_edit_plan: ConversationEditPlanJsonSchema,
    conversation_edit_result: ConversationEditResultJsonSchema,
    context_change: ContextChangeJsonSchema,
    conversation_append_operation: ConversationAppendOperationJsonSchema,
    conversation_append_change: ConversationAppendChangeJsonSchema,
    conversation_edit_change: ConversationEditChangeJsonSchema,

    context_selector: ContextSelectorJsonSchema,
    context_selection_request: ContextSelectionRequestJsonSchema,
    context_selection_result: ContextSelectionResultJsonSchema,
    context_change_plan: ContextChangePlanJsonSchema,
    context_change_plan_input: ContextChangePlanInputJsonSchema,
    selected_context_blocks: SelectedContextBlocksJsonSchema,
    text_code_point_range: TextCodePointRangeJsonSchema,
    json_pointer: JsonPointerJsonSchema,
    media_selection_range: MediaSelectionRangeJsonSchema,
    selection_binary_evidence: SelectionBinaryEvidenceJsonSchema,
    selected_context_entry: SelectedContextEntryJsonSchema,
    conversation_selector: ConversationSelectorJsonSchema,
    conversation_selection_request: ConversationSelectionRequestJsonSchema,
    conversation_selection: ConversationSelectionJsonSchema,
    conversation_selection_result: ConversationSelectionResultJsonSchema,
    conversation_slice: ConversationSliceJsonSchema,
    conversation_slice_result: ConversationSliceResultJsonSchema,
    block_subselection: BlockSubselectionJsonSchema,
    selected_block: SelectedBlockJsonSchema,
    selection_media_evidence: SelectionMediaEvidenceJsonSchema,
    native_import_options: NativeConversationImportOptionsJsonSchema,
    native_import_diagnostic: NativeConversationImportDiagnosticJsonSchema,
    native_import_report: NativeConversationImportReportJsonSchema,
    native_import_result: NativeConversationImportResultJsonSchema,
    accepted_output_fragment: ConversationAcceptedOutputFragmentJsonSchema,
    append_options: AppendConversationRecordsOptionsJsonSchema,
    append_result: AppendConversationRecordsResultJsonSchema,
    change: ConversationChangeJsonSchema,
    context_change_request: ContextChangeRequestJsonSchema,
    context_change_proposal: ContextChangeProposalJsonSchema,
    context_change_operation: ContextChangeOperationJsonSchema,
    context_change_placement: ContextChangePlacementJsonSchema,
    content_block: ConversationContentBlockJsonSchema,
    diagnostic: ConversationDiagnosticJsonSchema,
    decoded_response: DecodedConversationResponseJsonSchema,
    document: ConversationDocumentJsonSchema,
    generation: ConversationGenerationJsonSchema,
    inspection: ConversationInspectionJsonSchema,
    record_batch: ConversationRecordBatchJsonSchema,
    stream_cursor: ConversationStreamCursorJsonSchema,
    stream_event: ConversationStreamEventJsonSchema,
    stream_event_batch: ConversationStreamEventBatchJsonSchema,
    stream_identity: ConversationStreamIdentityJsonSchema,
    tool_execution_request: ConversationToolExecutionRequestJsonSchema,
    tool_execution_result: ConversationToolExecutionResultJsonSchema,
    transcript_fragment: ConversationTranscriptFragmentJsonSchema,
    transcript_projection_input: ConversationTranscriptProjectionInputJsonSchema,
    turn: ConversationTurnJsonSchema,
});
