import { z } from 'zod';
import { canonicalJsonContentString, hashUtf8Content } from './content-integrity.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { ConversationValidationError } from './diagnostics.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { ContextChangeSchema } from './schemas/change.js';
import {
    ContextChangePlanInputSchema,
    ContextChangePlanSchema,
    ContextChangeRequestSchema,
} from './schemas/context-change.js';
import type {
    ContentBlock,
    ContextChange,
    ContextChangePlan,
    ContextChangePlanInput,
    ContextChangeRequest,
    ContextEntry,
    ContextRetrievalRequirement,
    ConversationDocument,
    ConversationTurn,
} from './types.js';
import { parseConversationDocument } from './validation.js';

export interface AppliedContextChange {
    document: ConversationDocument;
    change: ContextChange;
    applied: boolean;
}

function selectedBlocks(turn: ConversationTurn, entry: ContextEntry): ContentBlock[] {
    const ids = entry.block_ids === undefined ? undefined : new Set(entry.block_ids);
    return turn.blocks.filter((block) => ids === undefined || ids.has(block.id));
}

/** A partial selection partitions only the already-active top-level blocks, in original order. */
function partitionSelection(
    document: ConversationDocument,
    input: ContextChangePlanInput,
    entries = document.context.entries,
) {
    const turns = createContextTurnIndex(document);
    const ids = new Set(input.entry_ids);
    const selected = entries.filter((entry) => ids.has(entry.id));
    if (selected.length !== input.entry_ids.length) throw new Error('Context change selects an unavailable entry');
    if (
        input.selected_entries !== undefined &&
        canonicalJsonContentString(selected) !== canonicalJsonContentString(input.selected_entries)
    ) {
        throw new Error('Partial context selection source entries conflict');
    }
    const partial = input.selected_block_ids;
    if (partial && Object.keys(partial).some((id) => !ids.has(id)))
        throw new Error('Partial selection has an unavailable entry');
    const removed: ContextEntry[] = [];
    const retained: ContextEntry[] = [];
    const segments: { entry: ContextEntry; selected: boolean; block_ids?: string[] }[] = [];
    let ranges = 0;
    let inRange = false;
    for (const entry of entries) {
        if (!ids.has(entry.id)) {
            retained.push(entry);
            segments.push({ entry, selected: false });
            inRange = false;
            continue;
        }
        const blocks = resolveContextEntry(turns, entry).blocks;
        const blockIds = partial && Object.hasOwn(partial, entry.id) ? partial[entry.id] : undefined;
        if (blockIds === undefined) {
            removed.push(entry);
            segments.push({ entry, selected: true });
            if (!inRange) ranges += 1;
            inRange = true;
            continue;
        }
        if (entry.type === 'replacement_turn' && blockIds.length !== blocks.length) {
            throw new Error('Compaction replacement is an indivisible provenance unit');
        }
        const chosen = new Set(blockIds);
        if (
            chosen.size !== blockIds.length ||
            blocks.filter((block) => chosen.has(block.id)).length !== blockIds.length ||
            canonicalJsonContentString(blocks.filter((block) => chosen.has(block.id)).map((block) => block.id)) !==
                canonicalJsonContentString(blockIds)
        ) {
            throw new Error('Partial context selection blocks are unavailable, duplicated or out of order');
        }
        removed.push(blockIds.length === blocks.length ? entry : { ...entry, block_ids: blockIds });
        const remainder = blocks.filter((block) => !chosen.has(block.id)).map((block) => block.id);
        if (remainder.length) retained.push({ ...entry, block_ids: remainder });
        let run: { entry: ContextEntry; selected: boolean; block_ids: string[] } | undefined;
        for (const block of blocks) {
            const isSelected = chosen.has(block.id);
            if (!run || run.selected !== isSelected) {
                run = { entry, selected: isSelected, block_ids: [] };
                segments.push(run);
            }
            run.block_ids.push(block.id);
            if (isSelected && !inRange) ranges += 1;
            inRange = isSelected;
        }
    }
    return { removed, retained, segments, ranges };
}

/** Each replacement maps to one ordered selected range; intervening content remains in place. */
function selectedRanges(document: ConversationDocument, partition: ReturnType<typeof partitionSelection>) {
    const turns = createContextTurnIndex(document);
    const ranges: { entry_ids: string[]; turn_ids: string[]; block_ids: string[]; blocks: ContentBlock[] }[] = [];
    let open = false;
    for (const segment of partition.segments) {
        if (!segment.selected) {
            open = false;
            continue;
        }
        if (!open) {
            ranges.push({ entry_ids: [], turn_ids: [], block_ids: [], blocks: [] });
            open = true;
        }
        const range = ranges[ranges.length - 1];
        if (!range.entry_ids.includes(segment.entry.id)) range.entry_ids.push(segment.entry.id);
        if (!range.turn_ids.includes(segment.entry.turn_id)) range.turn_ids.push(segment.entry.turn_id);
        const blocks = resolveContextEntry(turns, segment.entry).blocks;
        const ids = segment.block_ids === undefined ? undefined : new Set(segment.block_ids);
        for (const block of blocks) {
            if (ids !== undefined && !ids.has(block.id)) continue;
            range.block_ids.push(block.id);
            range.blocks.push(block);
        }
    }
    return ranges;
}

export function contextChangeSelectedRanges(document: ConversationDocument, input: unknown) {
    return selectedRanges(document, partitionSelection(document, ContextChangePlanInputSchema.parse(input)));
}

async function remainderEntries(
    partition: ReturnType<typeof partitionSelection>,
    operationId: string,
): Promise<Map<number, ContextEntry>> {
    const result = new Map<number, ContextEntry>();
    let ordinal = 0;
    for (let index = 0; index < partition.segments.length; index += 1) {
        const segment = partition.segments[index];
        if (segment.selected || segment.block_ids === undefined) continue;
        const id = await deriveConversationId('context_entry', operationId, 'remainder', String(ordinal++));
        result.set(index, { ...segment.entry, id, block_ids: segment.block_ids });
    }
    return result;
}

function collectAssetIds(blocks: readonly ContentBlock[]): Set<string> {
    const ids = new Set<string>();
    const visit = (block: ContentBlock): void => {
        if ('asset_id' in block) ids.add(block.asset_id);
        if (block.type === 'tool_call' && block.arguments.type === 'externalized_json') {
            for (const item of block.arguments.hydration) ids.add(item.asset_id);
        }
        if (block.type === 'tool_result') for (const nested of block.content) visit(nested);
    };
    for (const block of blocks) visit(block);
    return ids;
}

function selectedIdentities(document: ConversationDocument, entries: readonly ContextEntry[]) {
    const turns = createContextTurnIndex(document);
    const turnIds = new Set<string>();
    const blockIds = new Set<string>();
    const callIds = new Set<string>();
    const calls = new Set<string>();
    const results = new Set<string>();
    const replay: Extract<ContentBlock, { type: 'native_replay' }>[] = [];
    const visit = (block: ContentBlock): void => {
        blockIds.add(block.id);
        if (block.type === 'tool_call') {
            callIds.add(block.call_id);
            if (block.executor === 'application') calls.add(block.call_id);
        }
        if (block.type === 'tool_result') {
            callIds.add(block.call_id);
            results.add(block.call_id);
            for (const nested of block.content) visit(nested);
        }
        if (block.type === 'native_replay') replay.push(block);
    };
    for (const entry of entries) {
        const turn = resolveContextEntry(turns, entry).turn;
        turnIds.add(turn.id);
        for (const block of selectedBlocks(turn, entry)) visit(block);
    }
    return { turnIds, blockIds, callIds, calls, results, replay };
}

export function assertContextMutationDependencyClosure(
    document: ConversationDocument,
    removed: readonly ContextEntry[],
    retained: readonly ContextEntry[],
): void {
    const removedIds = new Set(removed.map((entry) => entry.id));
    if (document.context.protected_entry_ids.some((id) => removedIds.has(id))) {
        throw new Error('Context change selects a protected entry');
    }
    const turns = createContextTurnIndex(document);
    for (const entry of removed) {
        const turn = resolveContextEntry(turns, entry).turn;
        if (turn.authority === 'system' || turn.authority === 'developer') {
            throw new Error(`Context change selects protected ${turn.authority} turn ${turn.id}`);
        }
    }
    const selected = selectedIdentities(document, removed);
    const before = selectedIdentities(document, document.context.entries);
    const after = selectedIdentities(document, retained);
    if (selected.replay.length > 0) throw new Error('Context change selects a protected native replay unit');
    for (const callId of selected.calls) {
        if (!before.results.has(callId)) throw new Error(`Context change selects pending tool call ${callId}`);
    }
    for (const callId of new Set([...before.calls, ...before.results])) {
        if (!before.calls.has(callId) || !before.results.has(callId)) continue;
        if (after.calls.has(callId) !== after.results.has(callId)) {
            throw new Error(`Context change would split application tool exchange ${callId}`);
        }
    }
    for (const block of after.replay) {
        const dependencies = block.dependencies;
        if (
            dependencies.turn_ids.some((id) => selected.turnIds.has(id)) ||
            dependencies.block_ids.some((id) => selected.blockIds.has(id)) ||
            dependencies.call_ids.some((id) => selected.callIds.has(id))
        ) {
            throw new Error(`Context change would orphan native replay dependency ${block.id}`);
        }
    }
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
    const entryIds = ownedInput.entry_ids;
    if (
        ownedInput.expected_revision !== document.revision ||
        ownedInput.expected_context_revision !== document.context.revision
    ) {
        throw new Error('Context change revision conflict');
    }
    if (entryIds.length === 0 || new Set(entryIds).size !== entryIds.length) {
        throw new Error('Context change requires unique selected entry IDs');
    }
    const { removed, retained, ranges } = partitionSelection(document, ownedInput);
    assertContextMutationDependencyClosure(document, removed, retained);
    const turns = createContextTurnIndex(document);
    const source = removed.map((entry) => {
        const turn = resolveContextEntry(turns, entry).turn;
        const { blocks: _allBlocks, ...turnIdentity } = turn;
        return { entry, turn: turnIdentity, blocks: selectedBlocks(turn, entry) };
    });
    const sourceTurnIds = [...new Set(removed.map((entry) => entry.turn_id))];
    const sourceBlockIds = removed.some((entry) => entry.block_ids !== undefined)
        ? [...new Set(source.flatMap(({ blocks }) => blocks.map((block) => block.id)))]
        : [];
    const assetIds = collectAssetIds(source.flatMap(({ blocks }) => blocks));
    const queue = [...assetIds];
    for (let index = 0; index < queue.length; index += 1) {
        const asset = document.assets[queue[index]];
        if (asset?.provenance.type === 'derived' && !assetIds.has(asset.provenance.source_asset_id)) {
            assetIds.add(asset.provenance.source_asset_id);
            queue.push(asset.provenance.source_asset_id);
        }
    }
    const assets = [...assetIds].sort().map((id) => ({ id, asset: document.assets[id] }));
    return ContextChangePlanSchema.parse({
        source_fingerprint: await fingerprintJson({
            conversation_id: document.id,
            revision: document.revision,
            context_revision: document.context.revision,
            selected: source,
            assets,
        }),
        entry_ids: removed.map((entry) => entry.id),
        ...(Object.hasOwn(ownedInput, 'selected_block_ids')
            ? { selected_block_ids: ownedInput.selected_block_ids }
            : {}),
        ...(Object.hasOwn(ownedInput, 'selected_entries') ? { selected_entries: ownedInput.selected_entries } : {}),
        source_turn_ids: sourceTurnIds,
        source_block_ids: sourceBlockIds,
        selected_asset_ids: [...assetIds].sort(),
        disjoint_ranges: ranges,
    });
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
        ? partitionSelection(document, partitionInput, request.selected_entries)
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
    const plan = await planContextChange(document, partitionInput);
    if (JSON.stringify(plan.entry_ids) !== JSON.stringify(request.entry_ids)) {
        throw new Error('Context change entry IDs must follow active context order');
    }
    if (plan.source_fingerprint !== request.expected_source_fingerprint) {
        throw new Error('Context change source fingerprint conflict');
    }
    if (document.context.cache_intent?.mode === 'required') {
        throw new Error('Context change would invalidate required cache intent');
    }
    const selected = new Set(plan.entry_ids);
    const proposal = request.proposal;
    const partition = partitionSelection(document, partitionInput);
    const ranges = selectedRanges(document, partition);
    const insertedEntryIds: string[] = [];
    const replacementEntries: ContextEntry[] = [];
    const retrievalRequirements: ContextRetrievalRequirement[] = [];
    let compactions = document.compactions;
    if (proposal.kind === 'replace_with_compaction') {
        if (proposal.fidelity !== 'retrievable' && proposal.accepted_asset_operation_id !== undefined) {
            throw new Error('Only retrievable compaction can bind an accepted original asset receipt');
        }
        const preserveRanges = proposal.fidelity === 'retrievable' && plan.disjoint_ranges > 1;
        const expectedCausalOrder = preserveRanges
            ? 'preserved_disjoint_ranges'
            : plan.disjoint_ranges > 1
              ? 'explicit_disjoint_summary'
              : 'contiguous';
        const expectedMode = preserveRanges ? 'per_selected_range' : 'first_selected';
        if (proposal.placement.causal_order !== expectedCausalOrder || proposal.placement.mode !== expectedMode) {
            throw new Error('Context change disjoint placement lacks its explicit causal-order policy');
        }
        if (Object.hasOwn(document.compactions, proposal.compaction_id)) {
            throw new Error(`Compaction ${proposal.compaction_id} already exists`);
        }
        if (
            proposal.fidelity === 'retrievable'
                ? proposal.replacement_turns.length !== ranges.length
                : proposal.replacement_turns.length !== 1
        ) {
            throw new Error('Compaction replacements do not map exactly to selected ranges');
        }
        if (
            proposal.replacement_turns.some(
                (replacement) =>
                    replacement.kind !== 'agent' ||
                    replacement.authority !== 'ordinary' ||
                    replacement.status !== 'completed' ||
                    replacement.model_visibility !== 'include' ||
                    replacement.blocks.some((block) =>
                        proposal.fidelity === 'retrievable'
                            ? block.type !== 'external_reference'
                            : block.type !== 'text',
                    ),
            )
        ) {
            throw new Error(
                proposal.fidelity === 'retrievable'
                    ? 'Retrievable compaction replacement must be completed ordinary external content'
                    : 'Compaction replacement must be completed ordinary text without executable content',
            );
        }
        if (proposal.fidelity === 'retrievable') {
            const publication = proposal.accepted_asset_operation_id
                ? document.operation_receipts[proposal.accepted_asset_operation_id]
                : undefined;
            const sources = ranges.flatMap((range) => range.blocks);
            const references = proposal.replacement_turns.reduce<ContentBlock[]>((blocks, replacement) => {
                blocks.push(...replacement.blocks);
                return blocks;
            }, []);
            if (
                sources.length === 0 ||
                sources.length !== references.length ||
                proposal.retained_asset_ids.length !== references.length ||
                new Set(proposal.retained_asset_ids).size !== references.length ||
                proposal.generation_ids.length !== 0 ||
                proposal.derivation_generation !== undefined ||
                proposal.accepted_asset_operation_id === undefined ||
                publication?.operation_kind !== undefined ||
                canonicalJsonContentString(publication?.accepted_asset_ids ?? []) !==
                    canonicalJsonContentString(proposal.retained_asset_ids)
            ) {
                throw new Error('Retrievable compaction requires exact ordered accepted text assets');
            }
            for (let index = 0; index < sources.length; index += 1) {
                const original = sources[index];
                const reference = references[index];
                const asset =
                    reference?.type === 'external_reference' ? document.assets[reference.asset_id] : undefined;
                const readDefinition =
                    reference?.type === 'external_reference'
                        ? document.tool_definitions[reference.retrieval.tool_definition_id ?? '']
                        : undefined;
                if (
                    original?.type !== 'text' ||
                    reference?.type !== 'external_reference' ||
                    reference.original_type !== 'text' ||
                    reference.preview === undefined ||
                    reference.preview.length > 512 ||
                    asset?.kind !== 'text' ||
                    asset.storage.type !== 'external' ||
                    !asset.content_hash ||
                    asset.byte_length === undefined ||
                    reference.content_hash !== asset.content_hash ||
                    proposal.retained_asset_ids[index] !== asset.id ||
                    !readDefinition ||
                    readDefinition.name !== reference.retrieval.capability ||
                    readDefinition.version !== String(reference.retrieval.version) ||
                    !document.context.active_tool_definition_ids.includes(readDefinition.id)
                ) {
                    throw new Error('Retrievable compaction requires exact text asset and active read tool per block');
                }
                const integrity = await hashUtf8Content(original.text);
                if (integrity.content_hash !== asset.content_hash || integrity.byte_length !== asset.byte_length) {
                    throw new Error('Retrievable compaction asset differs from the exact selected original');
                }
                const existingRequirements = document.context.retrieval_requirements.filter(
                    (candidate) =>
                        candidate.asset_id === asset.id &&
                        canonicalJsonContentString(candidate.retrieval) ===
                            canonicalJsonContentString(reference.retrieval),
                );
                if (
                    existingRequirements.length > 1 ||
                    (existingRequirements.length === 1 &&
                        existingRequirements[0].accepted_asset_operation_id !== proposal.accepted_asset_operation_id)
                ) {
                    throw new Error('Retrievable compaction conflicts with an accepted original asset requirement');
                }
                if (existingRequirements.length === 0) {
                    retrievalRequirements.push({
                        id: await deriveConversationId('retrieval_requirement', request.operation_id, asset.id),
                        asset_id: asset.id,
                        retrieval: reference.retrieval,
                        accepted_asset_operation_id: proposal.accepted_asset_operation_id,
                    });
                }
            }
        }
        for (const [index, replacement] of proposal.replacement_turns.entries()) {
            const range = ranges[index];
            const sourceTurnIds = proposal.fidelity === 'retrievable' ? range.turn_ids : plan.source_turn_ids;
            const sourceBlockIds =
                proposal.fidelity === 'retrievable' && (ranges.length > 1 || range.blocks.length > 1)
                    ? range.block_ids
                    : plan.source_block_ids;
            if (
                (proposal.fidelity === 'retrievable' && replacement.blocks.length !== range.blocks.length) ||
                replacement.provenance.type !== 'derived' ||
                replacement.provenance.derivation_id !== proposal.compaction_id ||
                replacement.provenance.source_hash !== plan.source_fingerprint ||
                JSON.stringify(replacement.provenance.source_turn_ids) !== JSON.stringify(sourceTurnIds) ||
                JSON.stringify(replacement.provenance.source_block_ids ?? []) !== JSON.stringify(sourceBlockIds)
            ) {
                throw new Error('Compaction replacement provenance does not map to its exact selected range');
            }
            const entryId = await deriveConversationId('context_entry', proposal.compaction_id, replacement.id);
            replacementEntries.push({
                id: entryId,
                type: 'replacement_turn',
                compaction_id: proposal.compaction_id,
                turn_id: replacement.id,
            });
        }

        const consumedCompactions = Object.values(document.compactions).filter((record) =>
            document.context.entries.some(
                (entry) =>
                    selected.has(entry.id) && entry.type === 'replacement_turn' && entry.compaction_id === record.id,
            ),
        );
        const superseded = consumedCompactions.length === 1 ? consumedCompactions[0] : undefined;
        compactions = {
            ...document.compactions,
            [proposal.compaction_id]: {
                id: proposal.compaction_id,
                operation_id: request.operation_id,
                strategy: proposal.strategy,
                source: {
                    turn_ids: plan.source_turn_ids,
                    ...(plan.source_block_ids.length ? { block_ids: plan.source_block_ids } : {}),
                    source_fingerprint: plan.source_fingerprint,
                },
                replacement_turns: proposal.replacement_turns,
                fidelity: proposal.fidelity,
                retained_asset_ids: proposal.retained_asset_ids,
                generation_ids: proposal.generation_ids,
                ...(proposal.derivation_generation ? { derivation_generation: proposal.derivation_generation } : {}),
                ...(superseded ? { supersedes_compaction_id: superseded.id } : {}),
                created_at: request.recorded_at,
                metadata: {
                    applied_revision: document.revision + 1,
                    source_context_revision: document.context.revision,
                    payload_fingerprint: payloadFingerprint,
                    ...(proposal.accepted_input_fingerprint
                        ? { accepted_input_fingerprint: proposal.accepted_input_fingerprint }
                        : {}),
                },
            },
        };
    }
    const remainders = await remainderEntries(partition, request.operation_id);
    const entries: ContextEntry[] = [];
    let nextReplacement = 0;
    let inSelectedRange = false;
    for (const [index, segment] of partition.segments.entries()) {
        if (!segment.selected) {
            const remainder = remainders.get(index);
            entries.push(remainder ?? segment.entry);
            if (remainder) insertedEntryIds.push(remainder.id);
        } else if (!inSelectedRange) {
            const replacement = replacementEntries[nextReplacement++];
            if (replacement) {
                entries.push(replacement);
                insertedEntryIds.push(replacement.id);
            }
        }
        inSelectedRange = segment.selected;
    }
    const nextRevision = document.revision + 1;
    if (!Number.isSafeInteger(nextRevision)) throw new RangeError('Conversation revision exceeds safe integer range');
    const cacheIntent = document.context.cache_intent;
    const clearedCacheIntent =
        cacheIntent?.mode === 'auto' ||
        (cacheIntent?.mode === 'off' &&
            cacheIntent.stable_through_entry_id !== undefined &&
            selected.has(cacheIntent.stable_through_entry_id))
            ? Object.fromEntries(Object.entries(cacheIntent).filter(([key]) => key !== 'stable_through_entry_id'))
            : cacheIntent;
    const change = ContextChangeSchema.parse({
        operation_id: request.operation_id,
        conversation_id: document.id,
        base_revision: document.revision,
        result_revision: nextRevision,
        operations: [
            {
                kind: proposal.kind,
                removed_entry_ids: plan.entry_ids,
                inserted_entry_ids: insertedEntryIds,
                source_fingerprint: plan.source_fingerprint,
                ...(request.selected_block_ids
                    ? {
                          selected_block_ids: request.selected_block_ids,
                          remainder_entry_ids: [...remainders.values()].map((entry) => entry.id),
                      }
                    : {}),
                ...(proposal.kind === 'replace_with_compaction' ? { placement: proposal.placement } : {}),
            },
        ],
        diagnostics: [],
    });
    const updated = parseConversationDocument({
        ...document,
        revision: nextRevision,
        updated_at: request.recorded_at,
        compactions,
        context: {
            ...document.context,
            revision: document.context.revision + 1,
            entries,
            retrieval_requirements: retrievalRequirements.length
                ? [...document.context.retrieval_requirements, ...retrievalRequirements]
                : document.context.retrieval_requirements,
            protected_entry_ids: document.context.protected_entry_ids.filter((id) => !selected.has(id)),
            ...(clearedCacheIntent ? { cache_intent: clearedCacheIntent } : {}),
        },
        operation_receipts: {
            ...document.operation_receipts,
            [request.operation_id]: {
                id: request.operation_id,
                conversation_id: document.id,
                payload_fingerprint: payloadFingerprint,
                base_revision: document.revision,
                result_revision: nextRevision,
                recorded_at: request.recorded_at,
                accepted_turn_ids: [],
                accepted_generation_ids: [],
                accepted_context_entry_ids: insertedEntryIds,
                operation_kind: 'context_change',
                context_change: {
                    ...change.operations[0],
                },
            },
        },
    });
    return { document: updated, change, applied: true };
}
