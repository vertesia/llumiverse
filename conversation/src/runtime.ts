import { canonicalJsonContentString } from './content-integrity.js';
import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import { processingAppendAcceptance, stageProcessingAppend } from './processing.js';
import { ConversationAppendChangeSchema } from './schemas/change.js';
import {
    AppendConversationRecordsOptionsSchema,
    AppendConversationRecordsResultSchema,
    AppendConversationRecordsWithProcessingResultSchema,
    ConversationRecordBatchSchema,
    DecodedConversationResponseSchema,
} from './schemas/index.js';
import type {
    AppendConversationRecordsOptions,
    AppendConversationRecordsResult,
    AppendConversationRecordsWithProcessingResult,
    ConversationDiagnostic,
    ConversationDocument,
    ConversationPreparedRequestRecord,
    ConversationRecordBatch,
    DecodedConversationResponse,
    OperationReceipt,
    RequestReceipt,
} from './types.js';
import { parseConversationDocument } from './validation.js';

export interface PreparedConversationRequest<NativePayload> {
    document: ConversationDocument;
    payload: NativePayload;
    receipt: RequestReceipt;
    generation_id: string;
    response_turn_id: string;
    diagnostics: ConversationDiagnostic[];
}

export interface NativeConversationAdapter<
    NativeHistory,
    NativePayload,
    NativeResponse,
    PrepareOptions,
    DecodeOptions,
> {
    readonly protocol: string;
    readonly adapter_version: string;
    importConversation(history: NativeHistory, options: PrepareOptions): Promise<ConversationDocument>;
    prepare(
        conversation: ConversationDocument | NativeHistory | undefined | null,
        options: PrepareOptions,
    ): Promise<PreparedConversationRequest<NativePayload>>;
    decodeResponse(
        response: NativeResponse,
        prepared: PreparedConversationRequest<NativePayload>,
        options: DecodeOptions,
    ): Promise<DecodedConversationResponse>;
}

const stableJson = canonicalJsonContentString;

function appendChange(receipt: OperationReceipt, batch: ConversationRecordBatch) {
    return ConversationAppendChangeSchema.parse({
        operation_id: receipt.id,
        conversation_id: receipt.conversation_id,
        base_revision: receipt.base_revision,
        result_revision: receipt.result_revision,
        diagnostics: [],
        operations: [
            {
                kind: 'append',
                payload_fingerprint: receipt.payload_fingerprint,
                ...(receipt.accepted_tool_selection === undefined
                    ? {}
                    : { tool_selection: receipt.accepted_tool_selection }),
                accepted_turn_ids: receipt.accepted_turn_ids ?? acceptedIds(batch.turns),
                accepted_generation_ids: receipt.accepted_generation_ids ?? acceptedIds(batch.generations),
                accepted_asset_ids: receipt.accepted_asset_ids ?? acceptedIds(batch.assets),
                accepted_tool_definition_ids:
                    receipt.accepted_tool_definition_ids ?? acceptedIds(batch.tool_definitions),
                accepted_execution_receipt_ids:
                    receipt.accepted_execution_receipt_ids ?? acceptedIds(batch.execution_receipts),
                accepted_context_entry_ids: receipt.accepted_context_entry_ids ?? acceptedIds(batch.context_entries),
            },
        ],
    });
}

function checkedNextRevision(revision: number): number {
    const next = revision + 1;
    if (!Number.isSafeInteger(next)) {
        throw new RangeError('Conversation revision exceeds Number.MAX_SAFE_INTEGER');
    }
    return next;
}

function recordById<T extends { id: string }>(values: readonly T[] | undefined): Record<string, T> {
    const entries: Array<[string, T]> = [];
    const ids = new Set<string>();
    for (const value of values ?? []) {
        if (ids.has(value.id)) throw new Error(`Conversation batch contains duplicate record ID ${value.id}`);
        ids.add(value.id);
        entries.push([value.id, value]);
    }
    return Object.fromEntries(entries);
}

function acceptedIds<T extends { id: string }>(values: readonly T[] | undefined): string[] {
    return (values ?? []).map((value) => value.id);
}

function assertAcceptedIds(
    kind: string,
    incoming: readonly string[],
    accepted: readonly string[] | undefined,
    retained: (id: string) => boolean,
): string[] {
    const authoritative = accepted === undefined ? [...incoming] : [...accepted];
    if (stableJson(incoming) !== stableJson(authoritative)) {
        throw new Error(`Conversation operation retry does not identify its accepted ${kind}`);
    }
    const missing = authoritative.find((id) => !retained(id));
    if (missing !== undefined)
        throw new Error(`Conversation operation retry cannot resolve accepted ${kind} ${missing}`);
    return authoritative;
}

type RetryRecordKind =
    | 'turn'
    | 'generation'
    | 'asset'
    | 'tool definition'
    | 'execution receipt'
    | 'context entry'
    | 'retrieval requirement';

function retryComparableRecord(kind: RetryRecordKind, value: object): Record<string, unknown> {
    const comparable: Record<string, unknown> = { ...value };
    // Only schema-owned observation fields may change on retry. Keys with the same names in JSON
    // content, arguments, metadata, tool schemas, or asset locators are part of the accepted value.
    if (kind === 'turn' || kind === 'generation') delete comparable.timestamps;
    if (kind === 'asset') delete comparable.created_at;
    if (kind === 'execution receipt') delete comparable.recorded_at;
    if (kind === 'generation') {
        const receipt = comparable.request_receipt;
        if (receipt !== null && typeof receipt === 'object' && !Array.isArray(receipt)) {
            const requestReceipt: Record<string, unknown> = { ...receipt };
            delete requestReceipt.recorded_at;
            comparable.request_receipt = requestReceipt;
        }
    }
    return comparable;
}

function assertAcceptedRecords<T extends { id: string }>(
    kind: RetryRecordKind,
    incoming: readonly T[] | undefined,
    retained: (id: string) => T | undefined,
): void {
    for (const record of incoming ?? []) {
        const accepted = retained(record.id);
        if (
            accepted === undefined ||
            stableJson(retryComparableRecord(kind, record)) !== stableJson(retryComparableRecord(kind, accepted))
        ) {
            throw new Error(`Conversation operation retry changes accepted ${kind} ${record.id}`);
        }
    }
}

function assertExactRetry(
    document: ConversationDocument,
    batch: ConversationRecordBatch,
    receipt: OperationReceipt,
): { turn_ids: string[]; generation_ids: string[] } {
    if (batch.turns?.some((turn) => document.deleted_turns && Object.hasOwn(document.deleted_turns, turn.id))) {
        throw new Error('Accepted turn source was logically deleted; recover its authenticated predecessor revision');
    }
    const toolSelection =
        batch.active_tool_definition_ids === undefined
            ? { kind: 'unchanged' }
            : { kind: 'replace', definition_ids: [...batch.active_tool_definition_ids] };
    if (receipt.accepted_tool_selection !== undefined) {
        if (stableJson(toolSelection) !== stableJson(receipt.accepted_tool_selection)) {
            throw new Error('Conversation operation retry changes accepted active tool selection');
        }
    } else if (batch.active_tool_definition_ids !== undefined) {
        throw new Error('Historical append receipt cannot prove its accepted active tool selection');
    }
    const turnIds = assertAcceptedIds('turns', acceptedIds(batch.turns), receipt.accepted_turn_ids, (id) =>
        document.turns.some((turn) => turn.id === id),
    );
    const generationIds = assertAcceptedIds(
        'generations',
        acceptedIds(batch.generations),
        receipt.accepted_generation_ids,
        (id) => Object.hasOwn(document.generations, id),
    );
    assertAcceptedIds('assets', acceptedIds(batch.assets), receipt.accepted_asset_ids, (id) =>
        Object.hasOwn(document.assets, id),
    );
    assertAcceptedIds(
        'tool definitions',
        acceptedIds(batch.tool_definitions),
        receipt.accepted_tool_definition_ids,
        (id) => Object.hasOwn(document.tool_definitions, id),
    );
    assertAcceptedIds(
        'execution receipts',
        acceptedIds(batch.execution_receipts),
        receipt.accepted_execution_receipt_ids,
        (id) => Object.hasOwn(document.execution_receipts, id),
    );
    const acceptedEntries = receipt.accepted_context_entries ?? document.context.entries;
    assertAcceptedIds('context entries', acceptedIds(batch.context_entries), receipt.accepted_context_entry_ids, (id) =>
        acceptedEntries.some((entry) => entry.id === id),
    );
    assertAcceptedRecords('turn', batch.turns, (id) => document.turns.find((turn) => turn.id === id));
    assertAcceptedRecords('generation', batch.generations, (id) =>
        Object.hasOwn(document.generations, id) ? document.generations[id] : undefined,
    );
    assertAcceptedRecords('asset', batch.assets, (id) =>
        Object.hasOwn(document.assets, id) ? document.assets[id] : undefined,
    );
    assertAcceptedRecords('tool definition', batch.tool_definitions, (id) =>
        Object.hasOwn(document.tool_definitions, id) ? document.tool_definitions[id] : undefined,
    );
    assertAcceptedRecords('execution receipt', batch.execution_receipts, (id) =>
        Object.hasOwn(document.execution_receipts, id) ? document.execution_receipts[id] : undefined,
    );
    assertAcceptedRecords('context entry', batch.context_entries, (id) =>
        acceptedEntries.find((entry) => entry.id === id),
    );
    if (receipt.accepted_retrieval_requirements === undefined && (batch.retrieval_requirements?.length ?? 0) > 0) {
        throw new Error('Historical append receipt cannot prove its accepted retrieval requirements');
    }
    const acceptedRequirements = receipt.accepted_retrieval_requirements ?? [];
    assertAcceptedIds(
        'retrieval requirements',
        acceptedIds(batch.retrieval_requirements),
        acceptedIds(acceptedRequirements),
        (id) => acceptedRequirements.some((requirement) => requirement.id === id),
    );
    assertAcceptedRecords('retrieval requirement', batch.retrieval_requirements, (id) =>
        acceptedRequirements.find((requirement) => requirement.id === id),
    );
    return { turn_ids: turnIds, generation_ids: generationIds };
}

/** Internal shared assembly; caller owns publication, revision and operation-family receipt. */
export function assembleConversationRecordBatch(document: ConversationDocument, batch: ConversationRecordBatch) {
    const generationRecords = recordById(batch.generations);
    const assetRecords = recordById(batch.assets);
    const definitionRecords = recordById(batch.tool_definitions);
    const executionRecords = recordById(batch.execution_receipts);
    for (const id of Object.keys(generationRecords)) {
        if (Object.hasOwn(document.generations, id)) throw new Error(`Generation ${id} already exists`);
    }
    for (const id of Object.keys(assetRecords)) {
        if (Object.hasOwn(document.assets, id)) throw new Error(`Asset ${id} already exists`);
    }
    for (const id of Object.keys(executionRecords)) {
        if (Object.hasOwn(document.execution_receipts, id)) throw new Error(`Execution receipt ${id} already exists`);
    }
    for (const [id, definition] of Object.entries(definitionRecords)) {
        const retained = Object.hasOwn(document.tool_definitions, id) ? document.tool_definitions[id] : undefined;
        if (retained !== undefined && stableJson(retained) !== stableJson(definition)) {
            throw new Error(`Tool definition ${id} conflicts with the retained definition`);
        }
    }

    return {
        generationRecords,
        assetRecords,
        definitionRecords,
        executionRecords,
        turns: [...document.turns, ...(batch.turns ?? [])],
    };
}

/**
 * Append already-decoded canonical records to a materialized document.
 *
 * This is the narrow ingestion primitive used by native adapters. It does not edit, compact, or
 * process existing content. Operation receipts make exact retry delivery idempotent; conflicting
 * payloads under the same operation identity are rejected.
 */
function appendConversationRecordsUnchecked(
    input: ConversationDocument,
    batchInput: ConversationRecordBatch,
    optionsInput: AppendConversationRecordsOptions,
): AppendConversationRecordsResult {
    const batchPreflight = preflightJsonInput(batchInput);
    if (!batchPreflight.success) {
        throw new ConversationValidationError(
            'Conversation record batch failed JSON preflight',
            batchPreflight.diagnostics,
        );
    }
    const optionsPreflight = preflightJsonInput(optionsInput);
    if (!optionsPreflight.success) {
        throw new ConversationValidationError(
            'Conversation append options failed JSON preflight',
            optionsPreflight.diagnostics,
        );
    }
    const batch = ConversationRecordBatchSchema.parse(batchInput);
    const options = AppendConversationRecordsOptionsSchema.parse(optionsInput);
    const document = parseConversationDocument(input);
    const priorReceipt = Object.hasOwn(document.operation_receipts, options.operation_id)
        ? document.operation_receipts[options.operation_id]
        : undefined;
    if (priorReceipt !== undefined) {
        if (priorReceipt.operation_kind !== undefined) {
            throw new Error(
                `Conversation operation ${options.operation_id} belongs to ${priorReceipt.operation_kind === 'context_change' ? 'a context change' : 'a named mutation kind'}`,
            );
        }
        if (priorReceipt.payload_fingerprint !== options.payload_fingerprint) {
            throw new Error(`Conversation operation ${options.operation_id} was already used with a different payload`);
        }
        const accepted = assertExactRetry(document, batch, priorReceipt);
        return AppendConversationRecordsResultSchema.parse({
            document,
            applied: false,
            change: appendChange(priorReceipt, batch),
            accepted_turn_ids: accepted.turn_ids,
            accepted_generation_ids: accepted.generation_ids,
        });
    }
    if (options.expected_revision !== document.revision) {
        throw new Error(
            `Conversation revision conflict: expected ${options.expected_revision}, received ${document.revision}`,
        );
    }

    const records = assembleConversationRecordBatch(document, batch);
    const { generationRecords, assetRecords, definitionRecords, executionRecords } = records;

    const resultRevision = checkedNextRevision(document.revision);
    const operationReceipt: OperationReceipt = {
        id: options.operation_id,
        conversation_id: document.id,
        payload_fingerprint: options.payload_fingerprint,
        base_revision: document.revision,
        result_revision: resultRevision,
        recorded_at: options.recorded_at,
        accepted_turn_ids: acceptedIds(batch.turns),
        accepted_generation_ids: acceptedIds(batch.generations),
        accepted_asset_ids: acceptedIds(batch.assets),
        accepted_tool_definition_ids: acceptedIds(batch.tool_definitions),
        accepted_execution_receipt_ids: acceptedIds(batch.execution_receipts),
        accepted_context_entry_ids: acceptedIds(batch.context_entries),
        accepted_context_entries: [...(batch.context_entries ?? [])],
        ...((batch.retrieval_requirements?.length ?? 0) > 0
            ? { accepted_retrieval_requirements: [...(batch.retrieval_requirements ?? [])] }
            : {}),
        accepted_tool_selection:
            batch.active_tool_definition_ids === undefined
                ? { kind: 'unchanged' }
                : { kind: 'replace', definition_ids: [...batch.active_tool_definition_ids] },
    };

    const updated = {
        ...document,
        revision: resultRevision,
        updated_at: options.recorded_at,
        turns: [...document.turns, ...(batch.turns ?? [])],
        generations: { ...document.generations, ...generationRecords },
        operation_receipts: {
            ...document.operation_receipts,
            [operationReceipt.id]: operationReceipt,
        },
        execution_receipts: {
            ...document.execution_receipts,
            ...executionRecords,
        },
        assets: { ...document.assets, ...assetRecords },
        tool_definitions: { ...document.tool_definitions, ...definitionRecords },
        context: {
            ...document.context,
            revision: resultRevision,
            entries: [...document.context.entries, ...(batch.context_entries ?? [])],
            retrieval_requirements: [
                ...document.context.retrieval_requirements,
                ...(batch.retrieval_requirements ?? []),
            ],
            active_tool_definition_ids:
                batch.active_tool_definition_ids === undefined
                    ? document.context.active_tool_definition_ids
                    : [...batch.active_tool_definition_ids],
        },
    };

    return AppendConversationRecordsResultSchema.parse({
        document: parseConversationDocument(updated),
        applied: true,
        change: appendChange(operationReceipt, batch),
        accepted_turn_ids: acceptedIds(batch.turns),
        accepted_generation_ids: acceptedIds(batch.generations),
    });
}

/** Existing synchronous append remains valid only for documents without an enabled processing policy. */
export function appendConversationRecords(
    input: ConversationDocument,
    batchInput: ConversationRecordBatch,
    optionsInput: AppendConversationRecordsOptions,
): AppendConversationRecordsResult {
    if (parseConversationDocument(input).processing.enabled) {
        throw new Error('Enabled processing policy requires appendConversationRecordsWithProcessing');
    }
    return appendConversationRecordsUnchecked(input, batchInput, optionsInput);
}

/** Prepare an append and its on_append outbox jobs as one document for the caller's exact-head CAS. */
export async function appendConversationRecordsWithProcessing(
    input: ConversationDocument,
    batchInput: ConversationRecordBatch,
    optionsInput: AppendConversationRecordsOptions,
): Promise<AppendConversationRecordsWithProcessingResult> {
    const result = appendConversationRecordsUnchecked(input, batchInput, optionsInput);
    const operationId = optionsInput.operation_id;
    const acceptedEntryIds = batchInput.context_entries?.map((entry) => entry.id) ?? [];
    // An asset publication is an input to an existing processing job, not a new model-visible addition.
    const assetOnly = batchInput.assets !== undefined && Object.keys(batchInput).every((key) => key === 'assets');
    const document =
        !result.applied || !result.document.processing.enabled || assetOnly
            ? result.document
            : await stageProcessingAppend(result.document, operationId, acceptedEntryIds);
    return AppendConversationRecordsWithProcessingResultSchema.parse({
        ...result,
        document,
        acceptance: processingAppendAcceptance(document, operationId),
    });
}

/** Validate the response against a durable prepared record without materializing lifetime history. */
export function decodedResponseBatchFromAcceptedRecord(
    prepared: Pick<
        ConversationPreparedRequestRecord,
        'source' | 'request_receipt' | 'generation_id' | 'response_turn_id'
    >,
    decodedInput: DecodedConversationResponse,
    optionsInput: Omit<AppendConversationRecordsOptions, 'expected_revision' | 'payload_fingerprint'>,
): { batch: ConversationRecordBatch; options: AppendConversationRecordsOptions } {
    const optionsPreflight = preflightJsonInput(optionsInput);
    if (!optionsPreflight.success) {
        throw new ConversationValidationError(
            'Conversation response append options failed JSON preflight',
            optionsPreflight.diagnostics,
        );
    }
    const decodedPreflight = preflightJsonInput(decodedInput);
    if (!decodedPreflight.success) {
        throw new ConversationValidationError(
            'Decoded conversation response failed JSON preflight',
            decodedPreflight.diagnostics,
        );
    }
    const decoded = DecodedConversationResponseSchema.parse(decodedInput);
    const options = AppendConversationRecordsOptionsSchema.parse({
        ...optionsInput,
        expected_revision: prepared.source.revision,
        payload_fingerprint: decoded.payload_fingerprint,
    });
    if (
        prepared.request_receipt.source.conversation_id !== prepared.source.conversation_id ||
        prepared.request_receipt.source.revision !== prepared.source.revision ||
        decoded.generation.id !== prepared.generation_id ||
        decoded.generation.request_id !== prepared.request_receipt.request_id ||
        decoded.generation.attempt_id !== prepared.request_receipt.attempt_id ||
        decoded.generation.source.conversation_id !== prepared.request_receipt.source.conversation_id ||
        decoded.generation.source.revision !== prepared.request_receipt.source.revision ||
        decoded.generation.requested_model !== prepared.request_receipt.target.model ||
        decoded.generation.provider !== prepared.request_receipt.target.provider ||
        decoded.generation.protocol !== prepared.request_receipt.target.protocol ||
        decoded.generation.adapter_version !== prepared.request_receipt.target.adapter_version ||
        stableJson(decoded.generation.request_receipt) !== stableJson(prepared.request_receipt)
    ) {
        throw new Error('Decoded generation request receipt does not match the prepared request');
    }
    if (
        decoded.turns.length !== 1 ||
        decoded.turns[0].id !== prepared.response_turn_id ||
        decoded.turns.some(
            (turn) =>
                turn.kind !== 'agent' || !('generation_id' in turn) || turn.generation_id !== prepared.generation_id,
        )
    ) {
        throw new Error('Decoded response turns do not match the prepared response identity');
    }
    const batch: ConversationRecordBatch = {
        turns: decoded.turns,
        generations: [decoded.generation],
        context_entries: decoded.turns.map((turn, index) => ({
            id: `${options.operation_id}:context:${index}`,
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
        ...(decoded.assets === undefined ? {} : { assets: decoded.assets }),
        ...(decoded.execution_receipts === undefined ? {} : { execution_receipts: decoded.execution_receipts }),
    };
    return { batch, options };
}

function decodedResponseAppend<NativePayload>(
    prepared: PreparedConversationRequest<NativePayload>,
    decodedInput: DecodedConversationResponse,
    optionsInput: Omit<AppendConversationRecordsOptions, 'expected_revision' | 'payload_fingerprint'>,
): { batch: ConversationRecordBatch; options: AppendConversationRecordsOptions } {
    if (prepared.document.id !== prepared.receipt.source.conversation_id) {
        throw new Error('Materialized response document differs from its immutable prepared conversation');
    }
    return decodedResponseBatchFromAcceptedRecord(
        {
            source: prepared.receipt.source,
            request_receipt: prepared.receipt,
            generation_id: prepared.generation_id,
            response_turn_id: prepared.response_turn_id,
        },
        decodedInput,
        optionsInput,
    );
}

export function appendDecodedConversationResponse<NativePayload>(
    prepared: PreparedConversationRequest<NativePayload>,
    decodedInput: DecodedConversationResponse,
    optionsInput: Omit<AppendConversationRecordsOptions, 'expected_revision' | 'payload_fingerprint'>,
): AppendConversationRecordsResult {
    const { batch, options } = decodedResponseAppend(prepared, decodedInput, optionsInput);
    return appendConversationRecords(prepared.document, batch, options);
}

/** Stage an accepted native response and its on-append jobs in one exact-head document. */
export async function appendDecodedConversationResponseWithProcessing<NativePayload>(
    prepared: PreparedConversationRequest<NativePayload>,
    decodedInput: DecodedConversationResponse,
    optionsInput: Omit<AppendConversationRecordsOptions, 'expected_revision' | 'payload_fingerprint'>,
): Promise<AppendConversationRecordsWithProcessingResult> {
    const { batch, options } = decodedResponseAppend(prepared, decodedInput, optionsInput);
    return appendConversationRecordsWithProcessing(prepared.document, batch, options);
}

export { deriveConversationId, fingerprintJson } from './identity.js';
