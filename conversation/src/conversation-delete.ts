import { canonicalJsonContentString } from './content-integrity.js';
import { nextRevision } from './conversation-edit-utils.js';
import { ConversationValidationError } from './diagnostics.js';
import { fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { ConversationDeleteChangeSchema } from './schemas/change.js';
import {
    ConversationDeletePlanInputSchema,
    ConversationDeletePlanSchema,
    ConversationDeleteRequestSchema,
    ConversationDeleteResultSchema,
} from './schemas/conversation-delete.js';
import { verifyDerivedBlockLineage } from './source-slice-lineage.js';
import type {
    ConversationDeletedTurnRef,
    ConversationDeleteOperation,
    ConversationDeletePlan,
    ConversationDeletePlanInput,
    ConversationDeleteResult,
    ConversationDocument,
    ConversationTurn,
} from './types.js';
import { parseConversationDocument } from './validation.js';

function preflight(value: unknown): void {
    const result = preflightJsonInput(value);
    if (!result.success)
        throw new ConversationValidationError('Conversation delete failed JSON preflight', result.diagnostics);
}

function same(first: unknown, second: unknown): boolean {
    return canonicalJsonContentString(first) === canonicalJsonContentString(second);
}

function assertDependencyClosure(document: ConversationDocument, selected: ReadonlySet<string>): void {
    const reject = (path: string): never => {
        throw new Error(`Conversation delete has a retained dependency at ${path}`);
    };
    const selectedBlocks = new Set(
        document.turns.filter((turn) => selected.has(turn.id)).flatMap((turn) => turn.blocks.map((block) => block.id)),
    );
    const selectedEntryIds = new Set<string>();
    const selectedAppendOperations = new Set<string>();
    for (const receipt of Object.values(document.operation_receipts)) {
        if (receipt.operation_kind !== undefined) continue;
        if (receipt.accepted_turn_ids?.some((id) => selected.has(id))) selectedAppendOperations.add(receipt.id);
        for (const entry of receipt.accepted_context_entries ?? []) {
            if (selected.has(entry.turn_id)) selectedEntryIds.add(entry.id);
        }
    }
    for (const [index, entry] of document.context.entries.entries()) {
        if (selected.has(entry.turn_id)) reject(`/context/entries/${index}`);
    }
    for (const [index, turn] of document.turns.entries()) {
        if (selected.has(turn.id)) {
            if (
                turn.blocks.some(
                    (block) =>
                        block.type === 'tool_call' || block.type === 'tool_result' || block.type === 'native_replay',
                )
            )
                reject(`/turns/${index}/blocks`);
            continue;
        }
        if (turn.parent_turn_id && selected.has(turn.parent_turn_id)) reject(`/turns/${index}/parent_turn_id`);
        if (turn.provenance.type === 'derived' && turn.provenance.source_turn_ids.some((id) => selected.has(id))) {
            reject(`/turns/${index}/provenance/source_turn_ids`);
        }
        for (const [blockIndex, block] of turn.blocks.entries()) {
            if (
                block.type === 'native_replay' &&
                (block.dependencies.turn_ids.some((id) => selected.has(id)) ||
                    block.dependencies.block_ids.some((id) => selectedBlocks.has(id)))
            )
                reject(`/turns/${index}/blocks/${blockIndex}/dependencies`);
        }
    }
    for (const [id, compaction] of Object.entries(document.compactions)) {
        if (
            compaction.source.turn_ids.some((turnId) => selected.has(turnId)) ||
            compaction.source.block_ids?.some((blockId) => selectedBlocks.has(blockId))
        )
            reject(`/compactions/${id}/source`);
    }
    for (const [id, asset] of Object.entries(document.assets)) {
        if (
            asset.provenance.type === 'received' &&
            asset.provenance.source_turn_id &&
            selected.has(asset.provenance.source_turn_id)
        )
            reject(`/assets/${id}/provenance/source_turn_id`);
    }
    for (const [id, generation] of Object.entries(document.generations)) {
        if (generation.record_source !== 'executed') continue;
        const receipt = generation.request_receipt;
        if (receipt.source_tail_turn_id && selected.has(receipt.source_tail_turn_id)) {
            reject(`/generations/${id}/request_receipt/source_tail_turn_id`);
        }
        if (
            receipt.item_mappings.some(
                (mapping) => selected.has(mapping.canonical_id) || selectedBlocks.has(mapping.canonical_id),
            )
        )
            reject(`/generations/${id}/request_receipt/item_mappings`);
    }
    for (const [id, receipt] of Object.entries(document.execution_receipts)) {
        if (receipt.result_turn_id && selected.has(receipt.result_turn_id))
            reject(`/execution_receipts/${id}/result_turn_id`);
        if (receipt.call_source && selected.has(receipt.call_source.turn_id)) {
            reject(`/execution_receipts/${id}/call_source/turn_id`);
        }
    }
    for (const [id, resolved] of Object.entries(document.processing.resolved_inputs ?? {})) {
        if (
            resolved.source_turn_ids.some((turnId) => selected.has(turnId)) ||
            resolved.selected_entries?.some((entry) => selected.has(entry.turn_id)) ||
            resolved.entry_ids.some((entryId) => selectedEntryIds.has(entryId))
        )
            reject(`/processing/resolved_inputs/${id}`);
    }
    for (const [id, job] of Object.entries(document.processing.jobs ?? {})) {
        if (
            selectedAppendOperations.has(job.source_operation_id) ||
            (job.selection.kind === 'entries' &&
                (job.selection.entry_ids.some((entryId) => selectedEntryIds.has(entryId)) ||
                    job.selection.selected_entries?.some((entry) => selected.has(entry.turn_id))))
        ) {
            reject(`/processing/jobs/${id}/selection`);
        }
    }
}

async function prepareDelete(
    document: ConversationDocument,
    input: ConversationDeletePlanInput,
): Promise<ConversationDeleteOperation> {
    if (input.conversation.conversation_id !== document.id || input.conversation.revision !== document.revision) {
        throw new Error('Conversation delete source revision conflict');
    }
    const selected = new Set(input.turn_ids);
    if (selected.size !== input.turn_ids.length) throw new Error('Conversation delete repeats a turn ID');
    const selectedTurns = document.turns.filter((turn) => selected.has(turn.id));
    if (
        selectedTurns.length !== input.turn_ids.length ||
        !same(
            selectedTurns.map((turn) => turn.id),
            input.turn_ids,
        )
    ) {
        throw new Error('Conversation delete selects unavailable or unordered source turns');
    }
    assertDependencyClosure(document, selected);
    const acceptedByTurn = new Map<string, string>();
    const duplicateAcceptedTurns = new Set<string>();
    for (const receipt of Object.values(document.operation_receipts)) {
        if (receipt.operation_kind !== undefined) continue;
        for (const id of receipt.accepted_turn_ids ?? []) {
            if (!selected.has(id)) continue;
            if (acceptedByTurn.has(id)) duplicateAcceptedTurns.add(id);
            else acceptedByTurn.set(id, receipt.id);
        }
    }
    const deletedTurns: ConversationDeletedTurnRef[] = [];
    for (const turn of selectedTurns) {
        const acceptedOperationId = acceptedByTurn.get(turn.id);
        if (!acceptedOperationId || duplicateAcceptedTurns.has(turn.id)) {
            throw new Error(`Conversation delete lacks one accepted append for ${turn.id}`);
        }
        deletedTurns.push({
            id: turn.id,
            fingerprint: await fingerprintJson(turn),
            block_ids: turn.blocks.map((block) => block.id),
            accepted_operation_id: acceptedOperationId,
        });
    }
    return {
        version: 1,
        source: input.conversation,
        source_fingerprint: await fingerprintJson({ document, turn_ids: input.turn_ids }),
        dependency_policy: 'reject',
        deleted_turns: deletedTurns,
    };
}

function deleteChange(
    operationId: string,
    conversationId: string,
    baseRevision: number,
    resultRevision: number,
    operation: ConversationDeleteOperation,
) {
    return ConversationDeleteChangeSchema.parse({
        operation_id: operationId,
        conversation_id: conversationId,
        base_revision: baseRevision,
        result_revision: resultRevision,
        operations: [operation],
        diagnostics: [],
    });
}

/** Plan a body-removing, dependency-closed history mutation without publishing storage. */
export async function planConversationDelete(
    sourceInput: ConversationDocument,
    input: unknown,
): Promise<ConversationDeletePlan> {
    preflight(input);
    const command = ConversationDeletePlanInputSchema.parse(input);
    const document = await verifyDerivedBlockLineage(parseConversationDocument(sourceInput));
    return ConversationDeletePlanSchema.parse({ operation: await prepareDelete(document, command), diagnostics: [] });
}

/** Old revision bytes and assets remain host-retained; this result never attests physical erasure. */
export async function applyConversationDelete(
    sourceInput: ConversationDocument,
    requestInput: unknown,
): Promise<ConversationDeleteResult> {
    preflight(requestInput);
    const request = ConversationDeleteRequestSchema.parse(requestInput);
    const document = await verifyDerivedBlockLineage(parseConversationDocument(sourceInput));
    const payloadFingerprint = await fingerprintJson({ domain: 'llumiverse.conversation.delete', version: 1, request });
    const prior = Object.hasOwn(document.operation_receipts, request.operation_id)
        ? document.operation_receipts[request.operation_id]
        : undefined;
    if (prior) {
        const detail = prior.conversation_delete;
        if (
            prior.operation_kind !== 'conversation_delete' ||
            !detail ||
            prior.payload_fingerprint !== payloadFingerprint ||
            prior.base_revision !== request.conversation.revision ||
            prior.recorded_at !== request.recorded_at ||
            detail.source_fingerprint !== request.expected_source_fingerprint ||
            detail.dependency_policy !== request.dependency_policy ||
            !same(detail.source, request.conversation) ||
            !same(
                detail.deleted_turns.map((turn) => turn.id),
                request.turn_ids,
            )
        )
            throw new Error('Conversation delete retry conflicts with its accepted receipt');
        for (const ref of detail.deleted_turns) {
            const witness =
                document.deleted_turns && Object.hasOwn(document.deleted_turns, ref.id)
                    ? document.deleted_turns[ref.id]
                    : undefined;
            if (
                !witness ||
                witness.operation_id !== prior.id ||
                witness.source_revision !== prior.base_revision ||
                witness.fingerprint !== ref.fingerprint ||
                witness.accepted_operation_id !== ref.accepted_operation_id ||
                !same(witness.block_ids, ref.block_ids) ||
                document.turns.some((turn) => turn.id === ref.id)
            )
                throw new Error('Conversation delete retry lacks its retained tombstone');
        }
        return ConversationDeleteResultSchema.parse({
            document,
            change: deleteChange(prior.id, prior.conversation_id, prior.base_revision, prior.result_revision, detail),
            applied: false,
        });
    }
    const { expected_source_fingerprint: expected, ...input } = request;
    const operation = await prepareDelete(document, input);
    if (operation.source_fingerprint !== expected) throw new Error('Conversation delete source fingerprint conflict');
    const revision = nextRevision(document.revision);
    const witnesses = operation.deleted_turns.map(
        (ref) =>
            [
                ref.id,
                {
                    ...ref,
                    operation_id: request.operation_id,
                    source_revision: document.revision,
                },
            ] as const,
    );
    const updated = parseConversationDocument({
        ...document,
        revision,
        updated_at: request.recorded_at,
        turns: document.turns.filter((turn: ConversationTurn) => !request.turn_ids.includes(turn.id)),
        deleted_turns: Object.fromEntries([...Object.entries(document.deleted_turns ?? {}), ...witnesses]),
        operation_receipts: {
            ...document.operation_receipts,
            [request.operation_id]: {
                id: request.operation_id,
                conversation_id: document.id,
                payload_fingerprint: payloadFingerprint,
                base_revision: document.revision,
                result_revision: revision,
                recorded_at: request.recorded_at,
                operation_kind: 'conversation_delete',
                conversation_delete: operation,
                accepted_turn_ids: [],
                accepted_generation_ids: [],
                accepted_asset_ids: [],
                accepted_tool_definition_ids: [],
                accepted_execution_receipt_ids: [],
                accepted_context_entry_ids: [],
            },
        },
    });
    return ConversationDeleteResultSchema.parse({
        document: updated,
        change: deleteChange(request.operation_id, document.id, document.revision, revision, operation),
        applied: true,
    });
}
