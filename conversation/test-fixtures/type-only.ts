import type {
    ContextChangeOperation,
    ContextChangePlacement,
    ContextChangeProposal,
    ContextChangeRequest,
    ConversationChange,
    ConversationDocument,
    NativeConversationImportDiagnostic,
    NativeConversationImportOptions,
    NativeConversationImportReport,
    NativeConversationImportResult,
} from '@llumiverse/conversation';

export function getConversationId(document: ConversationDocument): string {
    return document.id;
}

export function importedConversationId(
    options: NativeConversationImportOptions,
    result: NativeConversationImportResult,
    report: NativeConversationImportReport,
    diagnostic: NativeConversationImportDiagnostic,
): string {
    return `${options.conversation_id}:${result.document.id}:${report.protocol}:${diagnostic.code}`;
}

export function contextChangeIdentity(
    request: ContextChangeRequest,
    change: ConversationChange,
    proposal: ContextChangeProposal,
    operation: ContextChangeOperation,
    placement: ContextChangePlacement,
): string {
    return `${request.operation_id}:${change.operation_id}:${proposal.kind}:${operation.kind}:${placement.mode}`;
}

export type {
    AcceptedToolSelection,
    BlockSubselection,
    ContextChange,
    ContextChangePlan,
    ContextChangePlanInput,
    ContextSelectionRequest,
    ContextSelectionResult,
    ContextSelector,
    ConversationAppendChange,
    ConversationAppendOperation,
    ConversationEditAnchor,
    ConversationEditableBlock,
    ConversationEditChange,
    ConversationEditCommand,
    ConversationEditOperation,
    ConversationEditOperationV1,
    ConversationEditPlacement,
    ConversationEditPlan,
    ConversationEditPlanInput,
    ConversationEditRecordRef,
    ConversationEditRequest,
    ConversationEditResult,
    ConversationInsertedTurn,
    ConversationReplacementTurn,
    ConversationSelection,
    ConversationSelectionRequest,
    ConversationSelectionResult,
    ConversationSelector,
    ConversationSlice,
    ConversationSliceEditCommand,
    ConversationSliceEditOperation,
    ConversationSliceEditPlanInput,
    ConversationSliceEditRequest,
    ConversationSliceResult,
    DerivedBlockLineage,
    DerivedBlockLineageGroup,
    DerivedLineageVerificationScope,
    JsonInverseMapping,
    JsonPointer,
    JsonSourceRegion,
    MediaSelectionRange,
    SelectedBlock,
    SelectedContextBlocks,
    SourceBlockSlice,
} from '@llumiverse/conversation';
