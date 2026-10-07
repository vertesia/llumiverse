import {
    canonicalJsonContentBytes,
    canonicalJsonContentString,
    hashContentBytes,
    hashUtf8Content,
} from './content-integrity.js';
import {
    type ContextChangeWorkingSet,
    partitionSelection,
    planContextChangeWorkingSet,
    selectedRanges,
} from './context-change-working-set.js';
import { cacheAfterContextRemoval } from './conversation-edit-utils.js';
import { deriveConversationId } from './identity.js';
import { ContextChangeSchema } from './schemas/change.js';
import { ContextChangePlanInputSchema } from './schemas/context-change.js';
import { ConversationContextSchema } from './schemas/document.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import type {
    CompactionRecord,
    ContentBlock,
    ContextChange,
    ContextChangeRequest,
    ContextEntry,
    ContextRetrievalRequirement,
    ConversationContext,
    OperationReceipt,
    ToolDefinition,
} from './types.js';

/** Immutable selected witnesses, not a replacement history representation. The indexed adapter
 * proves global compaction-ID uniqueness with its identifier index before calling this helper.
 * Inputs are already owned by the materialized or indexed schema adapter, before any await.
 */
export interface ContextMutationEvidence {
    compactions: Readonly<Record<string, Pick<CompactionRecord, 'id'>>>;
    tool_definitions: Record<string, ToolDefinition>;
    operation_receipts: Record<string, OperationReceipt>;
    /** Exact selected external originals, independently read and owned by the host before mutation. */
    original_archive_bytes?: ReadonlyMap<string, Uint8Array>;
}

export async function contextMutationRemainderEntries(
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

export interface ContextMutationResult {
    context: ConversationContext;
    compaction?: CompactionRecord;
    change: ContextChange;
    receipt: OperationReceipt;
}

export async function applyContextMutationWorkingSet(
    frame: ContextChangeWorkingSet,
    evidence: ContextMutationEvidence,
    request: ContextChangeRequest,
    payloadFingerprint: string,
): Promise<ContextMutationResult> {
    const originalArchives = new Map<string, Uint8Array>();
    let archiveBytes = 0;
    for (const [id, bytes] of evidence.original_archive_bytes ?? []) {
        if (!(bytes instanceof Uint8Array)) throw new TypeError('Retrievable source archive is not bytes');
        archiveBytes += bytes.byteLength;
        if (archiveBytes > 32 * 1024 * 1024 || originalArchives.size >= 4096)
            throw new RangeError('Retrievable source archives exceed the bounded evidence profile');
        originalArchives.set(id, Uint8Array.from(bytes));
    }
    const partitionInput = ContextChangePlanInputSchema.parse({
        expected_revision: request.expected_revision,
        expected_context_revision: request.expected_context_revision,
        entry_ids: request.entry_ids,
        ...(request.selected_block_ids
            ? { selected_block_ids: request.selected_block_ids, selected_entries: request.selected_entries }
            : {}),
    });
    const plan = await planContextChangeWorkingSet(frame, partitionInput);
    if (JSON.stringify(plan.entry_ids) !== JSON.stringify(request.entry_ids)) {
        throw new Error('Context change entry IDs must follow active context order');
    }
    if (plan.source_fingerprint !== request.expected_source_fingerprint) {
        throw new Error('Context change source fingerprint conflict');
    }
    if (frame.context.cache_intent?.mode === 'required') {
        throw new Error('Context change would invalidate required cache intent');
    }
    const selected = new Set(plan.entry_ids);
    const proposal = request.proposal;
    const partition = partitionSelection(frame, partitionInput);
    const ranges = selectedRanges(frame, partition);
    const insertedEntryIds: string[] = [];
    const replacementEntries: ContextEntry[] = [];
    const retrievalRequirements: ContextRetrievalRequirement[] = [];
    let compaction: CompactionRecord | undefined;
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
        if (Object.hasOwn(evidence.compactions, proposal.compaction_id)) {
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
                ? evidence.operation_receipts[proposal.accepted_asset_operation_id]
                : undefined;
            const sources = ranges.flatMap((range) => range.blocks);
            const explicitlySelectedBlockIds = request.entry_ids.flatMap(
                (id) => request.selected_block_ids?.[id] ?? [],
            );
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
            const selectedExchange = sources.some((source) => source.type !== 'text');
            if (selectedExchange) {
                const call = sources.find((source) => source.type === 'tool_call');
                const result = sources.find((source) => source.type === 'tool_result');
                if (
                    !request.selected_block_ids ||
                    explicitlySelectedBlockIds.length !== sources.length ||
                    explicitlySelectedBlockIds.some((id, index) => id !== sources[index]?.id) ||
                    sources.length !== 2 ||
                    sources[0]?.type !== 'tool_call' ||
                    sources[1]?.type !== 'tool_result' ||
                    call?.type !== 'tool_call' ||
                    call.executor !== 'application' ||
                    result?.type !== 'tool_result' ||
                    result.call_id !== call.call_id ||
                    result.status === 'unknown' ||
                    result.content.length !== 1 ||
                    result.content[0]?.type !== 'external_reference' ||
                    result.content[0].original_type !== 'text' ||
                    originalArchives.size !== 1
                ) {
                    throw new Error(
                        'Retrievable exchange must select one whole completed call and exact archived result',
                    );
                }
                if (
                    frame.context.entries.length > 100_000 ||
                    archiveBytes +
                        canonicalJsonContentBytes(frame.context.entries).byteLength +
                        canonicalJsonContentBytes(request).byteLength >
                        32 * 1024 * 1024
                ) {
                    throw new RangeError('Retrievable exchange exceeds its aggregate selected-byte or entry bound');
                }
            } else if (originalArchives.size !== 0) {
                throw new Error('Retrievable text edit has unexpected external archive evidence');
            }
            for (let index = 0; index < sources.length; index += 1) {
                const original = sources[index];
                const reference = references[index];
                const asset = reference?.type === 'external_reference' ? frame.assets[reference.asset_id] : undefined;
                const readDefinition =
                    reference?.type === 'external_reference'
                        ? evidence.tool_definitions[reference.retrieval.tool_definition_id ?? '']
                        : undefined;
                if (
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
                    // Capability ABI is independent of the accepted definition content version.
                    reference.retrieval.version !== 1 ||
                    !frame.context.active_tool_definition_ids.includes(readDefinition.id)
                ) {
                    throw new Error('Retrievable compaction requires exact text asset and active read tool per block');
                }
                let integrity: { content_hash: string; byte_length: number };
                if (original?.type === 'text') {
                    integrity = await hashUtf8Content(original.text);
                } else if (original?.type === 'tool_call') {
                    const sourceTurn = [...frame.turns.values()].find((turn) =>
                        turn.active_blocks.some((block) => block.id === original.id),
                    );
                    if (
                        !selectedExchange ||
                        asset.provenance.type !== 'received' ||
                        asset.provenance.source_turn_id !== sourceTurn?.header.id
                    ) {
                        throw new Error('Retrievable call archive lacks its exact selected source turn');
                    }
                    integrity = await hashContentBytes(canonicalJsonContentBytes(original));
                } else if (original?.type === 'tool_result') {
                    const nested = original.content[0];
                    const originalAsset =
                        nested?.type === 'external_reference' ? frame.assets[nested.asset_id] : undefined;
                    const originalRequirement = frame.context.retrieval_requirements.filter(
                        (item) =>
                            item.asset_id === originalAsset?.id &&
                            nested?.type === 'external_reference' &&
                            canonicalJsonContentString(item.retrieval) === canonicalJsonContentString(nested.retrieval),
                    );
                    const acceptedOriginalRequirement =
                        originalRequirement.length === 1 ? originalRequirement[0] : undefined;
                    const originalPublication = acceptedOriginalRequirement?.accepted_asset_operation_id
                        ? evidence.operation_receipts[acceptedOriginalRequirement.accepted_asset_operation_id]
                        : undefined;
                    const originalDefinition =
                        nested?.type === 'external_reference'
                            ? evidence.tool_definitions[nested.retrieval.tool_definition_id ?? '']
                            : undefined;
                    const bytes = originalAsset ? originalArchives.get(originalAsset.id) : undefined;
                    if (
                        !selectedExchange ||
                        nested?.type !== 'external_reference' ||
                        originalAsset?.kind !== 'text' ||
                        originalAsset.storage.type !== 'external' ||
                        originalAsset.content_hash !== nested.content_hash ||
                        originalAsset.byte_length === undefined ||
                        !originalPublication ||
                        originalPublication.id !== originalRequirement[0]?.accepted_asset_operation_id ||
                        originalPublication.conversation_id !== frame.source.conversation_id ||
                        originalPublication.result_revision > frame.source.revision ||
                        originalPublication.operation_kind !== undefined ||
                        originalPublication.accepted_asset_ids?.filter((id) => id === originalAsset.id).length !== 1 ||
                        originalPublication.accepted_retrieval_requirements?.filter(
                            (item) =>
                                item.id === originalRequirement[0]?.id &&
                                item.asset_id === originalAsset.id &&
                                canonicalJsonContentString(item.retrieval) ===
                                    canonicalJsonContentString(nested.retrieval),
                        ).length !== 1 ||
                        originalDefinition?.name !== nested.retrieval.capability ||
                        !frame.context.active_tool_definition_ids.includes(originalDefinition.id) ||
                        nested.retrieval.version !== 1 ||
                        !bytes ||
                        asset.provenance.type !== 'derived' ||
                        asset.provenance.source_asset_id !== originalAsset.id ||
                        asset.provenance.transform_id !== 'conversation.archive_rehome' ||
                        asset.provenance.transform_version !== '1'
                    ) {
                        throw new Error('Retrievable result archive lacks its exact original and publication');
                    }
                    integrity = await hashContentBytes(bytes);
                    if (
                        integrity.content_hash !== originalAsset.content_hash ||
                        integrity.byte_length !== originalAsset.byte_length
                    ) {
                        throw new Error('Retrievable source archive bytes differ from their accepted original');
                    }
                } else {
                    throw new Error('Retrievable compaction selects unsupported source content');
                }
                if (integrity.content_hash !== asset.content_hash || integrity.byte_length !== asset.byte_length) {
                    throw new Error('Retrievable compaction asset differs from the exact selected original');
                }
                const existingRequirements = frame.context.retrieval_requirements.filter(
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

        const consumedCompactions = Object.values(evidence.compactions).filter((record) =>
            frame.context.entries.some(
                (entry) =>
                    selected.has(entry.id) && entry.type === 'replacement_turn' && entry.compaction_id === record.id,
            ),
        );
        const superseded = consumedCompactions.length === 1 ? consumedCompactions[0] : undefined;
        compaction = {
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
                applied_revision: frame.source.revision + 1,
                source_context_revision: frame.context.revision,
                payload_fingerprint: payloadFingerprint,
                ...(proposal.accepted_input_fingerprint
                    ? { accepted_input_fingerprint: proposal.accepted_input_fingerprint }
                    : {}),
            },
        };
    }
    const remainders = await contextMutationRemainderEntries(partition, request.operation_id);
    const entries: ContextEntry[] = [];
    let nextReplacement = 0;
    let inSelectedRange = false;
    for (const [index, segment] of partition.segments.entries()) {
        if (!segment.selected) {
            if (segment.block_ids?.length === 0) {
                inSelectedRange = false;
                continue;
            }
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
    const nextRevision = frame.source.revision + 1;
    if (!Number.isSafeInteger(nextRevision)) throw new RangeError('Conversation revision exceeds safe integer range');
    const clearedCacheIntent = cacheAfterContextRemoval(frame.context, selected);
    const change = ContextChangeSchema.parse({
        operation_id: request.operation_id,
        conversation_id: frame.source.conversation_id,
        base_revision: frame.source.revision,
        result_revision: nextRevision,
        operations: [
            {
                kind: proposal.kind,
                removed_entry_ids: plan.entry_ids,
                inserted_entry_ids: insertedEntryIds,
                source_fingerprint: plan.source_fingerprint,
                ...(plan.discarded_replay_block_ids
                    ? { discarded_replay_block_ids: plan.discarded_replay_block_ids }
                    : {}),
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
    return {
        context: ConversationContextSchema.parse({
            ...frame.context,
            revision: frame.context.revision + 1,
            entries,
            retrieval_requirements: retrievalRequirements.length
                ? [...frame.context.retrieval_requirements, ...retrievalRequirements]
                : frame.context.retrieval_requirements,
            protected_entry_ids: frame.context.protected_entry_ids.filter((id) => !selected.has(id)),
            ...(clearedCacheIntent ? { cache_intent: clearedCacheIntent } : {}),
        }),
        ...(compaction === undefined ? {} : { compaction }),
        change,
        receipt: OperationReceiptSchema.parse({
            id: request.operation_id,
            conversation_id: frame.source.conversation_id,
            payload_fingerprint: payloadFingerprint,
            base_revision: frame.source.revision,
            result_revision: nextRevision,
            recorded_at: request.recorded_at,
            accepted_turn_ids: [],
            accepted_generation_ids: [],
            accepted_context_entry_ids: insertedEntryIds,
            operation_kind: 'context_change',
            context_change: { ...change.operations[0] },
        }),
    };
}
