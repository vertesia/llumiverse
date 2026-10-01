import type {
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
