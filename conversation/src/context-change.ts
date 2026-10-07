import { z } from 'zod';
import { canonicalJsonContentString } from './content-integrity.js';
import { applyContextMutationWorkingSet } from './context-change-transition.js';
import {
    assertContextMutationDependencyClosure as assertWorkingSetDependencyClosure,
    materializedContextChangeWorkingSet,
    partitionSelection,
    planContextChangeWorkingSet,
    selectedRanges,
} from './context-change-working-set.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { ConversationValidationError } from './diagnostics.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { ContextChangeSchema } from './schemas/change.js';
import { ContextChangePlanInputSchema, ContextChangeRequestSchema } from './schemas/context-change.js';
import type {
    ContextChange,
    ContextChangePlan,
    ContextChangeRequest,
    ContextEntry,
    ConversationDocument,
} from './types.js';
import { parseConversationDocument } from './validation.js';

export interface AppliedContextChange {
    document: ConversationDocument;
    change: ContextChange;
    applied: boolean;
}

export function contextChangeSelectedRanges(document: ConversationDocument, input: unknown) {
    const workingSet = materializedContextChangeWorkingSet(document);
    return selectedRanges(workingSet, partitionSelection(workingSet, ContextChangePlanInputSchema.parse(input)));
}

async function remainderEntries(
    partition: ReturnType<typeof partitionSelection>,
    operationId: string,
): Promise<Map<number, ContextEntry>> {
    const result = new Map<number, ContextEntry>();
    let ordinal = 0;
    for (let index = 0; index < partition.segments.length; index += 1) {
        const segment = partition.segments[index];
        if (segment.selected || segment.block_ids === undefined || segment.block_ids.length === 0) continue;
        const id = await deriveConversationId('context_entry', operationId, 'remainder', String(ordinal++));
        result.set(index, { ...segment.entry, id, block_ids: segment.block_ids });
    }
    return result;
}

export function assertContextMutationDependencyClosure(
    document: ConversationDocument,
    removed: readonly ContextEntry[],
    retained: readonly ContextEntry[],
): void {
    assertWorkingSetDependencyClosure(materializedContextChangeWorkingSet(document), removed, retained);
}

/** Resolve a pinned selection, its dependencies and its source hash without changing history. */
export async function planContextChange(sourceInput: ConversationDocument, input: unknown): Promise<ContextChangePlan> {
    // Own the document and selector before fingerprintJson yields to the host event loop.
    const preflight = preflightJsonInput(input);
    if (!preflight.success)
        throw new ConversationValidationError('Context selection failed JSON preflight', preflight.diagnostics);
    // Named compatibility: existing callers may pass the complete context-edit request.
    // Both contracts validate the selection pair before projecting mutation-only fields away.
    const planInput = ContextChangePlanInputSchema.safeParse(input);
    const selection = planInput.success ? planInput.data : ContextChangeRequestSchema.parse(input);
    const ownedInput = ContextChangePlanInputSchema.parse({
        expected_revision: selection.expected_revision,
        expected_context_revision: selection.expected_context_revision,
        entry_ids: selection.entry_ids,
        ...(Object.hasOwn(selection, 'selected_block_ids') ? { selected_block_ids: selection.selected_block_ids } : {}),
        ...(Object.hasOwn(selection, 'selected_entries') ? { selected_entries: selection.selected_entries } : {}),
    });
    const document = parseConversationDocument(sourceInput);
    return planContextChangeWorkingSet(materializedContextChangeWorkingSet(document), ownedInput);
}

/**
 * Pure materialized context edit. Supply planContextChange's document-ordered entry_ids;
 * the host publishes the returned document and receipt with one exact-head CAS.
 */
export async function applyContextChange(
    sourceInput: ConversationDocument,
    requestInput: ContextChangeRequest,
): Promise<AppliedContextChange> {
    const preflight = preflightJsonInput(requestInput);
    if (!preflight.success)
        throw new ConversationValidationError('Context change failed JSON preflight', preflight.diagnostics);
    const parsedRequest = ContextChangeRequestSchema.safeParse(requestInput);
    if (!parsedRequest.success) {
        // Preserve useful pre-existing leaf paths after adding the whole/partial shape union.
        const leaves = (issues: readonly z.core.$ZodIssue[]): z.core.$ZodIssue[] =>
            issues.flatMap((issue) => (issue.code === 'invalid_union' ? issue.errors.flatMap(leaves) : [issue]));
        const error = new z.ZodError(leaves(parsedRequest.error.issues));
        Object.defineProperty(error, 'cause', { value: parsedRequest.error });
        throw error;
    }
    const request = parsedRequest.data;
    const document = parseConversationDocument(sourceInput);
    const payloadFingerprint = await fingerprintJson(request);
    const partitionInput = ContextChangePlanInputSchema.parse({
        expected_revision: request.expected_revision,
        expected_context_revision: request.expected_context_revision,
        entry_ids: request.entry_ids,
        ...(request.selected_block_ids
            ? { selected_block_ids: request.selected_block_ids, selected_entries: request.selected_entries }
            : {}),
    });
    const retryPartition = request.selected_entries
        ? partitionSelection(materializedContextChangeWorkingSet(document), partitionInput, request.selected_entries)
        : undefined;
    const retryRemainders = retryPartition
        ? await remainderEntries(retryPartition, request.operation_id)
        : new Map<number, ContextEntry>();
    const prior = Object.hasOwn(document.operation_receipts, request.operation_id)
        ? document.operation_receipts[request.operation_id]
        : undefined;
    if (prior) {
        if (prior.operation_kind !== 'context_change' || !prior.context_change) {
            throw new Error(`Operation ${request.operation_id} belongs to a different mutation kind`);
        }
        if (prior.payload_fingerprint !== payloadFingerprint || prior.base_revision !== request.expected_revision) {
            throw new Error(`Context change operation ${request.operation_id} conflicts with its accepted payload`);
        }
        const detail = prior.context_change;
        const retryProposal = request.proposal;
        const summaryIds =
            retryProposal.kind === 'replace_with_compaction'
                ? await Promise.all(
                      retryProposal.replacement_turns.map((turn) =>
                          deriveConversationId('context_entry', retryProposal.compaction_id, turn.id),
                      ),
                  )
                : [];
        const expectedInserted: string[] = [];
        let nextSummary = 0;
        let inSelectedRange = false;
        const rangeByBlock = new Map<string, number>();
        if (retryProposal.kind === 'replace_with_compaction' && summaryIds.length > 1) {
            for (const [rangeIndex, replacement] of retryProposal.replacement_turns.entries()) {
                if (replacement.provenance.type !== 'derived') {
                    throw new Error(
                        `Context change operation ${request.operation_id} has conflicting retained details`,
                    );
                }
                for (const blockId of replacement.provenance.source_block_ids ?? []) {
                    if (rangeByBlock.has(blockId)) {
                        throw new Error(
                            `Context change operation ${request.operation_id} has conflicting retained details`,
                        );
                    }
                    rangeByBlock.set(blockId, rangeIndex);
                }
            }
        }
        if (retryPartition) {
            const turns = createContextTurnIndex(document);
            for (const [index, segment] of retryPartition.segments.entries()) {
                const remainder = retryRemainders.get(index);
                if (remainder) expectedInserted.push(remainder.id);
                const blocks = segment.selected ? resolveContextEntry(turns, segment.entry).blocks : [];
                const segmentBlockIds = segment.block_ids ?? blocks.map((block) => block.id);
                const rangeIndex = rangeByBlock.size ? rangeByBlock.get(segmentBlockIds[0]) : undefined;
                if (
                    rangeByBlock.size &&
                    segment.selected &&
                    (rangeIndex === undefined ||
                        segmentBlockIds.some((blockId) => rangeByBlock.get(blockId) !== rangeIndex))
                ) {
                    throw new Error(
                        `Context change operation ${request.operation_id} has conflicting retained details`,
                    );
                }
                if (
                    segment.selected &&
                    (!inSelectedRange || (rangeIndex !== undefined && rangeIndex === nextSummary))
                ) {
                    const summaryId = summaryIds[nextSummary++];
                    if (summaryId) expectedInserted.push(summaryId);
                }
                inSelectedRange = segment.selected;
            }
        } else expectedInserted.push(...summaryIds);
        if (
            new Set(request.entry_ids).size !== request.entry_ids.length ||
            prior.conversation_id !== document.id ||
            prior.result_revision !== prior.base_revision + 1 ||
            prior.recorded_at !== request.recorded_at ||
            JSON.stringify(prior.accepted_turn_ids) !== '[]' ||
            JSON.stringify(prior.accepted_generation_ids) !== '[]' ||
            (prior.accepted_asset_ids?.length ?? 0) !== 0 ||
            (prior.accepted_tool_definition_ids?.length ?? 0) !== 0 ||
            (prior.accepted_execution_receipt_ids?.length ?? 0) !== 0 ||
            canonicalJsonContentString(detail.selected_block_ids ?? null) !==
                canonicalJsonContentString(request.selected_block_ids ?? null) ||
            canonicalJsonContentString(detail.remainder_entry_ids ?? null) !==
                canonicalJsonContentString(
                    request.selected_block_ids ? [...retryRemainders.values()].map((entry) => entry.id) : null,
                ) ||
            canonicalJsonContentString(detail.discarded_replay_block_ids ?? []) !==
                canonicalJsonContentString([...(retryPartition?.discardedReplayIds ?? [])]) ||
            detail.kind !== request.proposal.kind ||
            detail.source_fingerprint !== request.expected_source_fingerprint ||
            JSON.stringify(detail.removed_entry_ids) !== JSON.stringify(request.entry_ids) ||
            JSON.stringify(detail.inserted_entry_ids) !== JSON.stringify(expectedInserted) ||
            JSON.stringify(prior.accepted_context_entry_ids) !== JSON.stringify(expectedInserted) ||
            JSON.stringify(detail.placement) !==
                JSON.stringify(
                    request.proposal.kind === 'replace_with_compaction' ? request.proposal.placement : undefined,
                )
        ) {
            throw new Error(`Context change operation ${request.operation_id} has conflicting retained details`);
        }
        return {
            document,
            change: ContextChangeSchema.parse({
                operation_id: prior.id,
                conversation_id: prior.conversation_id,
                base_revision: prior.base_revision,
                result_revision: prior.result_revision,
                operations: [prior.context_change],
                diagnostics: [],
            }),
            applied: false,
        };
    }
    const mutation = await applyContextMutationWorkingSet(
        materializedContextChangeWorkingSet(document),
        {
            compactions: document.compactions,
            tool_definitions: document.tool_definitions,
            operation_receipts: document.operation_receipts,
        },
        request,
        payloadFingerprint,
    );
    const updated = parseConversationDocument({
        ...document,
        revision: mutation.receipt.result_revision,
        updated_at: request.recorded_at,
        compactions:
            mutation.compaction === undefined
                ? document.compactions
                : { ...document.compactions, [mutation.compaction.id]: mutation.compaction },
        context: mutation.context,
        operation_receipts: { ...document.operation_receipts, [request.operation_id]: mutation.receipt },
    });
    return { document: updated, change: mutation.change, applied: true };
}
