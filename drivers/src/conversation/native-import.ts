import {
    type ContentBlock,
    type ConversationDocument,
    ConversationValidationError,
    createConversationDocument,
    fingerprintJson,
    type JsonValue,
    type NativeConversationImportDiagnostic,
    type NativeConversationImportOptions,
    type NativeConversationImportResult,
    parseConversationDocument,
    parseNativeConversationImportOptions,
    parseNativeConversationImportResult,
    preflightNativeConversationImportInput,
    type ResolvedConversationRuntimeContext,
} from '@llumiverse/conversation';

export type NativeConversationImportFailureCode =
    | 'IMPORT_INVALID_OPTIONS'
    | 'IMPORT_INVALID_HISTORY'
    | 'IMPORT_PROTOCOL_REQUIRED';

/** Failed imports produce no document; the cause retains detailed schema/adapter diagnostics. */
export class NativeConversationImportError extends Error {
    constructor(
        public readonly code: NativeConversationImportFailureCode,
        message: string,
        cause?: unknown,
    ) {
        super(message, { cause });
        this.name = 'NativeConversationImportError';
    }
}

export async function guardNativeConversationImport<T>(importHistory: () => Promise<T>): Promise<T> {
    try {
        return await importHistory();
    } catch (cause: unknown) {
        if (cause instanceof NativeConversationImportError) throw cause;
        throw new NativeConversationImportError(
            'IMPORT_INVALID_HISTORY',
            cause instanceof Error ? cause.message.slice(0, 512) : 'Native history failed exact import validation',
            cause,
        );
    }
}

/** Use the shared bounded walker, including cycle/prototype/accessor/symbol policy. */
export function assertNativeImportInputBounds(input: unknown, byteViewPrototypes?: readonly object[]): void {
    const checked = preflightNativeConversationImportInput(input, {}, byteViewPrototypes);
    if (!checked.success)
        throw new NativeConversationImportError(
            'IMPORT_INVALID_HISTORY',
            'Native history failed bounded preflight',
            new ConversationValidationError('Native input validation failed', checked.diagnostics),
        );
}

export type {
    NativeConversationImportDiagnostic,
    NativeConversationImportDiagnosticCode,
    NativeConversationImportOptions,
    NativeConversationImportReport,
    NativeConversationImportResult,
} from '@llumiverse/conversation';

export type NativeConversationImportContext = Pick<
    ResolvedConversationRuntimeContext,
    'conversation_id' | 'request_id' | 'recorded_at'
>;

/** One owned, schema-validated options snapshot, captured synchronously before adapter awaits. */
export function snapshotNativeConversationImportOptions(
    options: NativeConversationImportOptions,
): NativeConversationImportOptions {
    try {
        return parseNativeConversationImportOptions(options);
    } catch (cause: unknown) {
        throw new NativeConversationImportError(
            'IMPORT_INVALID_OPTIONS',
            'Native import requires valid origin evidence',
            cause,
        );
    }
}

/** Internal context projection from the already validated owned options snapshot. */
export function nativeConversationImportContext(
    options: NativeConversationImportOptions,
): NativeConversationImportContext {
    return {
        conversation_id: options.conversation_id,
        request_id: options.source_request_id ?? `${options.conversation_id}:legacy`,
        recorded_at: options.recorded_at,
    };
}

export function newNativeImportDocument(options: NativeConversationImportOptions): ConversationDocument {
    const context = nativeConversationImportContext(options);
    return createConversationDocument({ id: context.conversation_id, created_at: context.recorded_at });
}

/** Validated source content is inspectable; it never establishes target continuation readiness. */
export function nativeConversationImportResult(
    documentInput: ConversationDocument,
    options: NativeConversationImportOptions,
    protocol: string,
    adapterVersion: string,
    additionalDiagnostics: NativeConversationImportDiagnostic[] = [],
): NativeConversationImportResult {
    const document = parseConversationDocument(documentInput);
    const completeness = options.completeness ?? 'unknown';
    if (completeness !== 'complete' && completeness !== 'fragment' && completeness !== 'unknown') {
        throw new TypeError('Native import completeness must be complete, fragment or unknown');
    }
    const diagnostics: NativeConversationImportDiagnostic[] = [
        ...additionalDiagnostics,
        {
            code: 'IMPORT_CONTINUATION_NOT_VALIDATED',
            message: 'Import does not validate a target, hydrate remote assets, or establish processing readiness.',
        },
    ];
    if (completeness !== 'complete')
        diagnostics.push({
            code: 'IMPORT_HISTORY_INCOMPLETE',
            message: 'The source is a fragment or has no declaration of complete history.',
        });
    if (
        document.turns.some((turn) => turn.provenance.type === 'imported' && turn.provenance.missing_metadata?.length)
    ) {
        diagnostics.push({
            code: 'IMPORT_METADATA_MISSING',
            message: 'Native history omits actor, timestamp or exchange evidence.',
        });
    }
    const blocks = document.turns.flatMap<ContentBlock>((turn) => turn.blocks);
    const protectedWithoutModel = blocks.filter(
        (block) =>
            block.type === 'native_replay' &&
            block.dependency_policy !== 'discard_on_dependency_change' &&
            block.compatibility_scope.model === undefined,
    );
    if (protectedWithoutModel.length)
        diagnostics.push({
            code: 'IMPORT_PROTECTED_REPLAY_MODEL_UNKNOWN',
            message: 'Protected replay is retained without known model provenance; compatibility is unresolved.',
            entity_ids: protectedWithoutModel.map((block) => block.id),
        });
    const remoteAssets = Object.values(document.assets).filter(
        (asset) =>
            asset.storage.type !== 'inline_base64' &&
            asset.storage.type !== 'inline_text' &&
            asset.storage.type !== 'inline_json',
    );
    if (remoteAssets.length)
        diagnostics.push({
            code: 'IMPORT_EXTERNAL_ASSET_UNRESOLVED',
            message: 'Import retains remote media references without resolving them.',
            entity_ids: remoteAssets.map((asset) => asset.id),
        });
    const missingDefinitions = blocks.filter(
        (block) => block.type === 'tool_call' && block.definition_id === undefined,
    );
    if (missingDefinitions.length)
        diagnostics.push({
            code: 'IMPORT_TOOL_DEFINITION_MISSING',
            message: 'Native calls have no recorded tool definition.',
            entity_ids: missingDefinitions.map((block) => block.id),
        });
    return parseNativeConversationImportResult({
        document,
        report: { protocol, adapter_version: adapterVersion, completeness, readiness: 'not_validated', diagnostics },
    });
}

/** The operation hash binds the owned native source and every declared import semantic. */
export function fingerprintNativeConversationImport(
    history: JsonValue,
    options: NativeConversationImportOptions,
    protocol: string,
    adapterVersion: string,
): Promise<string> {
    const effectiveOptions = {
        ...options,
        source_request_id: options.source_request_id ?? `${options.conversation_id}:legacy`,
        tool_definitions: [...(options.tool_definitions ?? [])],
        completeness: options.completeness ?? 'unknown',
    };
    // These options were JSON-preflighted and schema-cloned once before the first adapter await.
    return fingerprintJson({
        format: 'llumiverse.native-conversation-import/v1',
        protocol,
        adapter_version: adapterVersion,
        history,
        options: effectiveOptions as JsonValue,
    });
}
