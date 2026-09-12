import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import {
    AppendConversationRecordsOptionsSchema,
    AppendConversationRecordsResultSchema,
    ConversationRecordBatchSchema,
    DecodedConversationResponseSchema,
} from './schemas/index.js';
import type {
    AppendConversationRecordsOptions,
    AppendConversationRecordsResult,
    ConversationDiagnostic,
    ConversationDocument,
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

function retryComparableRecord(value: unknown): unknown {
    if (value === null || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map(retryComparableRecord);
    const comparable: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value)) {
        // Retrying an accepted input operation may allocate a fresh host attempt and timestamps.
        // Those observations are not semantic prompt content. All other persisted fields remain
        // byte-for-byte accountable to the accepted records.
        if (key === 'timestamps' || key === 'created_at' || key === 'recorded_at') continue;
        comparable[key] = retryComparableRecord(child);
    }
    return comparable;
}

function assertAcceptedRecords<T extends { id: string }>(
    kind: string,
    incoming: readonly T[] | undefined,
    retained: (id: string) => T | undefined,
): void {
    for (const record of incoming ?? []) {
        const accepted = retained(record.id);
        if (
            accepted === undefined ||
            stableJson(retryComparableRecord(record)) !== stableJson(retryComparableRecord(accepted))
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
    assertAcceptedIds('context entries', acceptedIds(batch.context_entries), receipt.accepted_context_entry_ids, (id) =>
        document.context.entries.some((entry) => entry.id === id),
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
        document.context.entries.find((entry) => entry.id === id),
    );
    return { turn_ids: turnIds, generation_ids: generationIds };
}

/**
 * Append already-decoded canonical records to a materialized document.
 *
 * This is the narrow ingestion primitive used by native adapters. It does not edit, compact, or
 * process existing content. Operation receipts make exact retry delivery idempotent; conflicting
 * payloads under the same operation identity are rejected.
 */
export function appendConversationRecords(
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
        if (priorReceipt.payload_fingerprint !== options.payload_fingerprint) {
            throw new Error(`Conversation operation ${options.operation_id} was already used with a different payload`);
        }
        const accepted = assertExactRetry(document, batch, priorReceipt);
        return AppendConversationRecordsResultSchema.parse({
            document,
            applied: false,
            accepted_turn_ids: accepted.turn_ids,
            accepted_generation_ids: accepted.generation_ids,
        });
    }
    if (options.expected_revision !== document.revision) {
        throw new Error(
            `Conversation revision conflict: expected ${options.expected_revision}, received ${document.revision}`,
        );
    }

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

    const resultRevision = checkedNextRevision(document.revision);
    const operationReceipt = {
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
    } as const;

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
            active_tool_definition_ids:
                batch.active_tool_definition_ids === undefined
                    ? document.context.active_tool_definition_ids
                    : [...batch.active_tool_definition_ids],
        },
    };

    return AppendConversationRecordsResultSchema.parse({
        document: parseConversationDocument(updated),
        applied: true,
        accepted_turn_ids: acceptedIds(batch.turns),
        accepted_generation_ids: acceptedIds(batch.generations),
    });
}

export function appendDecodedConversationResponse<NativePayload>(
    prepared: PreparedConversationRequest<NativePayload>,
    decodedInput: DecodedConversationResponse,
    optionsInput: Omit<AppendConversationRecordsOptions, 'expected_revision' | 'payload_fingerprint'>,
): AppendConversationRecordsResult {
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
        expected_revision: prepared.document.revision,
        payload_fingerprint: decoded.payload_fingerprint,
    });
    if (
        decoded.generation.id !== prepared.generation_id ||
        decoded.generation.request_id !== prepared.receipt.request_id ||
        decoded.generation.attempt_id !== prepared.receipt.attempt_id ||
        decoded.generation.source.conversation_id !== prepared.receipt.source.conversation_id ||
        decoded.generation.source.revision !== prepared.receipt.source.revision ||
        decoded.generation.requested_model !== prepared.receipt.target.model ||
        decoded.generation.provider !== prepared.receipt.target.provider ||
        decoded.generation.protocol !== prepared.receipt.target.protocol ||
        decoded.generation.adapter_version !== prepared.receipt.target.adapter_version ||
        stableJson(decoded.generation.request_receipt) !== stableJson(prepared.receipt)
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
    return appendConversationRecords(prepared.document, batch, options);
}

function stableJson(value: unknown): string {
    if (value === null || typeof value !== 'object') {
        return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
        return `[${value.map(stableJson).join(',')}]`;
    }
    return `{${Object.keys(value as object)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${stableJson((value as Record<string, unknown>)[key])}`)
        .join(',')}}`;
}

/** Produce a browser-safe SHA-256 fingerprint for JSON-safe canonical or native data. */
export async function fingerprintJson(value: unknown): Promise<string> {
    const preflight = preflightJsonInput(value);
    if (!preflight.success) {
        throw new ConversationValidationError('Fingerprint input failed JSON preflight', preflight.diagnostics);
    }
    const bytes = new TextEncoder().encode(stableJson(value));
    const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    const hex = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
    return `sha256:${hex}`;
}

/** Derive a compact deterministic entity ID from a stable request/import identity. */
export async function deriveConversationId(kind: string, ...identity: string[]): Promise<string> {
    const fingerprint = await fingerprintJson([kind, ...identity]);
    return `${kind}_${fingerprint.slice('sha256:'.length, 'sha256:'.length + 32)}`;
}
