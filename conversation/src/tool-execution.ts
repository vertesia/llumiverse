import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import { appendConversationRecordsWithProcessing, fingerprintJson } from './runtime.js';
import { AppendConversationRecordsOptionsSchema } from './schemas/ingestion.js';
import {
    ConversationToolExecutionRequestSchema,
    ConversationToolExecutionResultSchema,
} from './schemas/tool-execution.js';
import {
    type HydrateToolArgumentsOptions,
    hydrateToolCallArguments,
    type ResolveToolArgumentTextAsset,
} from './tool-arguments.js';
import { assertToolResultReceiptFingerprint } from './tool-result-integrity.js';
import { assertToolRetrievalExcerptBinding } from './tool-retrieval-excerpt.js';
import type {
    AppendConversationRecordsOptions,
    AppendConversationRecordsResult,
    ConversationDocument,
    ConversationToolExecutionRequest,
    ConversationToolExecutionResult,
    ToolCallBlock,
    ToolCallSourceRef,
} from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

function sourceCall(document: ConversationDocument, source: ToolCallSourceRef): ToolCallBlock {
    if (source.conversation.conversation_id !== document.id || source.conversation.revision > document.revision) {
        throw new Error(
            `Tool call ${source.call_id} source ${source.conversation.conversation_id} revision ` +
                `${source.conversation.revision} is not available in conversation ${document.id} revision ${document.revision}`,
        );
    }
    const turn = document.turns.find((candidate) => candidate.id === source.turn_id);
    const block = turn?.blocks.find((candidate) => candidate.id === source.block_id);
    if (block?.type !== 'tool_call' || block.call_id !== source.call_id) {
        throw new Error(`Tool call source ${source.turn_id}/${source.block_id}/${source.call_id} does not resolve`);
    }
    if (block.executor !== 'application') {
        throw new Error(`Tool call ${source.call_id} is not authorized for application execution`);
    }
    const acceptedAt = Object.values(document.operation_receipts).find((receipt) =>
        receipt.accepted_turn_ids?.includes(source.turn_id),
    )?.result_revision;
    if (acceptedAt !== undefined && acceptedAt > source.conversation.revision) {
        throw new Error(
            `Tool call ${source.call_id} was accepted at revision ${acceptedAt}, after source revision ` +
                `${source.conversation.revision}`,
        );
    }
    return block;
}

function hasTerminalResult(document: ConversationDocument, callId: string): boolean {
    if (Object.values(document.execution_receipts).some((receipt) => receipt.call_id === callId)) return true;
    return document.turns.some((turn) =>
        turn.blocks.some((block) => block.type === 'tool_result' && block.call_id === callId),
    );
}

async function assertCallFingerprint(call: ToolCallBlock, source: ToolCallSourceRef): Promise<void> {
    if ((await fingerprintJson(call)) !== source.call_fingerprint) {
        throw new Error(`Tool call ${source.call_id} no longer matches its authorized source fingerprint`);
    }
}

function parseToolCallSource(input: unknown): ToolCallSourceRef {
    const preflight = preflightJsonInput(input);
    if (!preflight.success) {
        throw new ConversationValidationError('Tool call source failed JSON preflight', preflight.diagnostics);
    }
    const parsed = ConversationToolExecutionRequestSchema.shape.source.safeParse(input);
    if (!parsed.success) {
        throw new ConversationValidationError(
            'Tool call source failed schema validation',
            diagnosticsFromZodError(parsed.error),
        );
    }
    return parsed.data;
}

/** Resolve one application-owned canonical call into the exact transient arguments supplied to its runner. */
export async function resolveToolExecutionRequest(
    input: ConversationDocument,
    sourceInput: ToolCallSourceRef,
    resolveAsset: ResolveToolArgumentTextAsset,
    options: HydrateToolArgumentsOptions = {},
): Promise<ConversationToolExecutionRequest> {
    const document = parseConversationDocument(input);
    const source = parseToolCallSource(sourceInput);
    if (source.conversation.revision !== document.revision) {
        throw new Error(
            `Tool call ${source.call_id} requires conversation revision ${source.conversation.revision}, ` +
                `received ${document.revision}`,
        );
    }
    const call = sourceCall(document, source);
    await assertCallFingerprint(call, source);
    if (hasTerminalResult(document, source.call_id)) {
        throw new Error(`Tool call ${source.call_id} already has a terminal result`);
    }
    const argumentsValue = await hydrateToolCallArguments(document, source.call_id, resolveAsset, options);
    return ConversationToolExecutionRequestSchema.parse({
        source,
        call,
        arguments: argumentsValue,
    });
}

export type AppendToolExecutionResultOptions = Omit<AppendConversationRecordsOptions, 'payload_fingerprint'>;

async function validatedToolExecutionRecords(
    input: ConversationDocument,
    resultInput: ConversationToolExecutionResult,
): Promise<{ document: ConversationDocument; result: ConversationToolExecutionResult }> {
    const inputPreflight = preflightJsonInput(resultInput);
    if (!inputPreflight.success) {
        throw new ConversationValidationError(
            'Tool execution result failed JSON preflight',
            inputPreflight.diagnostics,
        );
    }
    const result = ConversationToolExecutionResultSchema.parse(structuredClone(resultInput));
    const document = parseConversationDocument(input);
    const call = sourceCall(document, result.source);
    await assertCallFingerprint(call, result.source);
    const resultBlock = result.turn.blocks[0];
    const receipt = result.execution_receipt;
    if (
        result.source.call_id !== call.call_id ||
        receipt.call_id !== call.call_id ||
        receipt.call_source.conversation.conversation_id !== result.source.conversation.conversation_id ||
        receipt.call_source.conversation.revision !== result.source.conversation.revision ||
        receipt.call_source.turn_id !== result.source.turn_id ||
        receipt.call_source.block_id !== result.source.block_id ||
        receipt.call_source.call_id !== result.source.call_id ||
        receipt.call_source.call_fingerprint !== result.source.call_fingerprint ||
        resultBlock.call_id !== call.call_id ||
        receipt.result_turn_id !== result.turn.id ||
        result.turn.execution_id !== receipt.id ||
        resultBlock.status === 'unknown' ||
        receipt.status !== resultBlock.status
    ) {
        throw new Error(`Tool execution result does not match application call ${result.source.call_id}`);
    }
    await assertToolResultReceiptFingerprint(resultBlock, receipt);
    if (receipt.metadata?.retrieval_excerpt) {
        const retained = document.execution_receipts[receipt.id];
        if (retained) {
            // Exact committed retries reuse immutable provenance; later context need not expose the asset again.
            if ((await fingerprintJson(retained)) !== (await fingerprintJson(receipt)))
                throw new Error('Retained retrieval execution receipt changed');
        } else await assertToolRetrievalExcerptBinding(document, resultBlock, receipt);
    }
    return { document, result };
}

/**
 * Validate exact call and result evidence without changing the document. This also validates retries;
 * the append operation owns duplicate-terminal rejection and operation-receipt recovery.
 */
export async function validateToolExecutionResult(
    input: ConversationDocument,
    resultInput: ConversationToolExecutionResult,
): Promise<ConversationToolExecutionResult> {
    return (await validatedToolExecutionRecords(input, resultInput)).result;
}

/** Append one canonical terminal application result using the shared revision and operation-receipt contract. */
export async function appendToolExecutionResult(
    input: ConversationDocument,
    resultInput: ConversationToolExecutionResult,
    options: AppendToolExecutionResultOptions,
): Promise<AppendConversationRecordsResult> {
    const optionsPreflight = preflightJsonInput(options);
    if (!optionsPreflight.success) {
        throw new ConversationValidationError(
            'Tool append options failed JSON preflight',
            optionsPreflight.diagnostics,
        );
    }
    options = AppendConversationRecordsOptionsSchema.omit({ payload_fingerprint: true }).parse(
        structuredClone(options),
    );
    const { document, result } = await validatedToolExecutionRecords(input, resultInput);
    if (hasTerminalResult(document, result.source.call_id)) {
        const acceptedOperation = Object.hasOwn(document.operation_receipts, options.operation_id)
            ? document.operation_receipts[options.operation_id]
            : undefined;
        if (acceptedOperation === undefined) {
            throw new Error(`Tool call ${result.source.call_id} already has a terminal result`);
        }
    }
    const payloadFingerprint = await fingerprintJson(result);
    return appendConversationRecordsWithProcessing(
        document,
        {
            turns: [result.turn],
            ...(result.assets === undefined ? {} : { assets: result.assets }),
            execution_receipts: [result.execution_receipt],
            context_entries: [
                {
                    id: `${options.operation_id}:context`,
                    type: 'source_turn',
                    turn_id: result.turn.id,
                },
            ],
        },
        { ...options, payload_fingerprint: payloadFingerprint },
    );
}
