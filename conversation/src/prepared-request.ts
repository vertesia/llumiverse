import { ConversationValidationError } from './diagnostics.js';
import { isGeneratedAgentTurn } from './guards.js';
import { preflightJsonInput } from './json-preflight.js';
import { deriveConversationId, fingerprintJson } from './runtime.js';
import { ConversationPreparedRequestSchema } from './schemas/prepared-request.js';
import type {
    ConversationDocument,
    ConversationPreparedRequest,
    ConversationPreparedRequestRecord,
    ConversationTurn,
    ExecutedGeneration,
    GeneratedAgentTurn,
} from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

export type ConversationPreparedRequestErrorCode =
    | 'asset_mismatch'
    | 'document_mismatch'
    | 'identity_mismatch'
    | 'indexed_source_mismatch'
    | 'mapping_mismatch'
    | 'response_mismatch'
    | 'source_view_mismatch'
    | 'tool_set_mismatch';

export class ConversationPreparedRequestError extends Error {
    constructor(
        readonly code: ConversationPreparedRequestErrorCode,
        message: string,
    ) {
        super(message);
        this.name = 'ConversationPreparedRequestError';
    }
}

function fail(code: ConversationPreparedRequestErrorCode, message: string): never {
    throw new ConversationPreparedRequestError(code, message);
}

function selectedTurns(
    document: ConversationDocument,
): Array<{ turn: ConversationTurn; selected_block_ids?: ReadonlySet<string> }> {
    const turnsById = new Map(document.turns.map((turn) => [turn.id, turn]));
    return document.context.entries.map((entry) => {
        const turn =
            entry.type === 'source_turn'
                ? turnsById.get(entry.turn_id)
                : document.compactions[entry.compaction_id]?.replacement_turns.find(
                      (candidate) => candidate.id === entry.turn_id,
                  );
        if (turn === undefined) fail('mapping_mismatch', `Selected conversation turn ${entry.turn_id} is unavailable`);
        return {
            turn,
            ...(entry.block_ids === undefined ? {} : { selected_block_ids: new Set(entry.block_ids) }),
        };
    });
}

function assertMappings(document: ConversationDocument, record: ConversationPreparedRequestRecord): void {
    const selections = selectedTurns(document);
    const turnIds = new Set(selections.map(({ turn }) => turn.id));
    const blockIds = new Set<string>();
    const callIds = new Set<string>();
    for (const { turn, selected_block_ids: selectedBlockIds } of selections) {
        for (const block of turn.blocks) {
            if (selectedBlockIds !== undefined && !selectedBlockIds.has(block.id)) continue;
            blockIds.add(block.id);
            if (block.type === 'tool_call') callIds.add(block.call_id);
            if (block.type === 'tool_result') {
                for (const nested of block.content) blockIds.add(nested.id);
            }
        }
    }
    for (const mapping of record.request_receipt.item_mappings) {
        const selectedIds = mapping.kind === 'turn' ? turnIds : mapping.kind === 'block' ? blockIds : callIds;
        if (!selectedIds.has(mapping.canonical_id)) {
            fail(
                'mapping_mismatch',
                `Prepared request mapping references unselected ${mapping.kind} ${mapping.canonical_id}`,
            );
        }
    }
}

async function assertPreparedRequestSemantics(
    document: ConversationDocument,
    record: ConversationPreparedRequestRecord,
): Promise<void> {
    const { request_receipt: receipt, runtime, source } = record;
    if (
        source.conversation_id !== document.id ||
        source.revision !== document.revision ||
        runtime.conversation_id !== document.id ||
        receipt.source.conversation_id !== source.conversation_id ||
        receipt.source.revision !== source.revision
    ) {
        fail('document_mismatch', 'Prepared request source does not match its canonical document head');
    }
    const sourceView = receipt.source_view;
    if (
        sourceView !== undefined &&
        (sourceView.source.conversation_id !== source.conversation_id ||
            sourceView.source.revision !== source.revision ||
            sourceView.context_revision !== document.context.revision ||
            sourceView.context_fingerprint !== receipt.context_fingerprint ||
            sourceView.request_fingerprint !== receipt.request_fingerprint)
    ) {
        fail('source_view_mismatch', 'Prepared request source view does not match its finalized request');
    }
    if (
        receipt.request_id !== runtime.request_id ||
        receipt.attempt_id !== runtime.attempt_id ||
        receipt.recorded_at !== runtime.recorded_at ||
        receipt.id !== (await deriveConversationId('request_receipt', runtime.request_id, runtime.attempt_id)) ||
        record.generation_id !== (await deriveConversationId('generation', runtime.request_id, runtime.attempt_id)) ||
        record.response_turn_id !== (await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'))
    ) {
        fail('identity_mismatch', 'Prepared request identities do not match the resolved execution runtime');
    }

    const sourceTail = document.turns.at(-1)?.id;
    if (receipt.source_tail_turn_id !== sourceTail) {
        fail('document_mismatch', 'Prepared request source tail does not match its canonical document');
    }
    if (
        receipt.tool_definition_ids.length !== document.context.active_tool_definition_ids.length ||
        receipt.tool_definition_ids.some((id, index) => id !== document.context.active_tool_definition_ids[index]) ||
        receipt.tool_definition_ids.some((id) => !Object.hasOwn(document.tool_definitions, id))
    ) {
        fail('tool_set_mismatch', 'Prepared request tool definitions do not match the active canonical tool set');
    }
    for (const binding of receipt.asset_versions) {
        const asset = Object.hasOwn(document.assets, binding.asset_id) ? document.assets[binding.asset_id] : undefined;
        if (asset?.content_hash !== binding.content_hash) {
            fail('asset_mismatch', `Prepared request asset ${binding.asset_id} does not match its content hash`);
        }
    }
    assertMappings(document, record);
}

/**
 * Parse host-persisted evidence produced after native request preparation and before provider transport.
 * This boundary validates both the complete canonical document and the request-specific cross-record invariants.
 */
export async function parseConversationPreparedRequest(input: unknown): Promise<ConversationPreparedRequest> {
    const preflight = preflightJsonInput(input);
    if (!preflight.success) {
        throw new ConversationValidationError(
            'Prepared conversation request failed JSON preflight',
            preflight.diagnostics,
        );
    }
    const shape = ConversationPreparedRequestSchema.safeParse(input);
    if (!shape.success) {
        throw new ConversationValidationError(
            'Prepared conversation request failed schema validation',
            diagnosticsFromZodError(shape.error),
        );
    }
    if (shape.data.record.indexed_source !== undefined) {
        fail('indexed_source_mismatch', 'Indexed prepared evidence cannot represent a full materialized document');
    }
    const document = parseConversationDocument(shape.data.document);
    const prepared = structuredClone(input) as ConversationPreparedRequest;
    prepared.document = document;
    await assertPreparedRequestSemantics(document, prepared.record);
    return prepared;
}

/** Parse the privacy-safe record retained after the full working document has been validated. */
export function parseConversationPreparedRequestRecord(input: unknown): ConversationPreparedRequestRecord {
    const preflight = preflightJsonInput(input);
    if (!preflight.success) {
        throw new ConversationValidationError('Prepared request record failed JSON preflight', preflight.diagnostics);
    }
    const shape = ConversationPreparedRequestSchema.shape.record.safeParse(input);
    if (!shape.success) {
        throw new ConversationValidationError(
            'Prepared request record failed schema validation',
            diagnosticsFromZodError(shape.error),
        );
    }
    return structuredClone(input) as ConversationPreparedRequestRecord;
}

/**
 * Adopt only the immutable source-view locator added by a durable host callback. All finalized
 * request fields, including an existing locator, must remain byte-equivalent to the original.
 */
export async function adoptConversationPreparedRequestRecord(
    originalInput: ConversationPreparedRequest,
    returnedInput: unknown,
): Promise<ConversationPreparedRequest> {
    const returned = parseConversationPreparedRequestRecord(returnedInput);
    const original = await parseConversationPreparedRequest(originalInput);
    const withoutSourceView = (record: ConversationPreparedRequestRecord) => {
        const { source_view: _sourceView, ...requestReceipt } = record.request_receipt;
        return { ...record, request_receipt: requestReceipt };
    };
    if (
        (await fingerprintJson(withoutSourceView(original.record))) !==
            (await fingerprintJson(withoutSourceView(returned))) ||
        (original.record.request_receipt.source_view !== undefined &&
            (returned.request_receipt.source_view === undefined ||
                (await fingerprintJson(original.record.request_receipt.source_view)) !==
                    (await fingerprintJson(returned.request_receipt.source_view))))
    ) {
        fail('source_view_mismatch', 'Durable host callback changed finalized prepared-request evidence');
    }
    return parseConversationPreparedRequest({ document: original.document, record: returned });
}

/** Verify that an accepted response was produced from an exact durably retained prepared-request record. */
export async function assertAcceptedResponseMatchesPreparedRecord(
    input: ConversationDocument,
    recordInput: ConversationPreparedRequestRecord,
): Promise<{ generation: ExecutedGeneration; turn: GeneratedAgentTurn }> {
    const document = parseConversationDocument(input);
    const record = parseConversationPreparedRequestRecord(recordInput);
    const operation = Object.hasOwn(document.operation_receipts, record.runtime.response_operation_id)
        ? document.operation_receipts[record.runtime.response_operation_id]
        : undefined;
    const generation = Object.hasOwn(document.generations, record.generation_id)
        ? document.generations[record.generation_id]
        : undefined;
    const turn = document.turns.find((candidate) => candidate.id === record.response_turn_id);
    if (
        operation?.accepted_generation_ids?.length !== 1 ||
        operation.accepted_generation_ids[0] !== record.generation_id ||
        operation.accepted_turn_ids?.length !== 1 ||
        operation.accepted_turn_ids[0] !== record.response_turn_id ||
        generation?.record_source !== 'executed' ||
        turn === undefined ||
        !isGeneratedAgentTurn(turn) ||
        turn.generation_id !== generation.id ||
        generation.request_id !== record.runtime.request_id ||
        generation.attempt_id !== record.runtime.attempt_id ||
        generation.source.conversation_id !== record.source.conversation_id ||
        generation.source.revision !== record.source.revision ||
        (await fingerprintJson(generation.request_receipt)) !== (await fingerprintJson(record.request_receipt))
    ) {
        fail('response_mismatch', 'Accepted response does not match its durably prepared request');
    }
    return { generation, turn };
}

/** Verify that an accepted response was produced from the exact durably prepared request. */
export async function assertAcceptedResponseMatchesPreparedRequest(
    input: ConversationDocument,
    preparedInput: ConversationPreparedRequest,
): Promise<{ generation: ExecutedGeneration; turn: GeneratedAgentTurn }> {
    const document = parseConversationDocument(input);
    const prepared = await parseConversationPreparedRequest(preparedInput);
    if (document.id !== prepared.document.id || document.revision < prepared.document.revision) {
        fail('response_mismatch', 'Accepted response is not a descendant of its prepared canonical document');
    }
    return assertAcceptedResponseMatchesPreparedRecord(document, prepared.record);
}
