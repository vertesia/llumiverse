import { canonicalJsonContentString } from './content-integrity.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { preflightJsonInput } from './json-preflight.js';
import { IndexedConversationSelectedContextSchema } from './schemas/indexed-head.js';
import type { Asset, ContentBlock, ExternalReferenceBlock, ToolDefinition } from './types.js';
import { parseConversationDocument } from './validation.js';

export interface ActiveTextExternalReference {
    asset: Asset;
    block: ExternalReferenceBlock;
    tool_definition: ToolDefinition;
    accepted_asset_operation_id: string;
}

/**
 * Resolve an active text/JSON canonical reference and its accepted original, independent of any host storage protocol.
 * The host remains responsible for interpreting the opaque locator, tool arguments, and retrieval capability.
 */
export function resolveActiveTextExternalReference(
    input: unknown,
    assetId: string,
    blockId?: string,
): ActiveTextExternalReference {
    const document = parseConversationDocument(input);
    const asset = Object.hasOwn(document.assets, assetId) ? document.assets[assetId] : undefined;
    if (
        (asset?.kind !== 'text' && asset?.kind !== 'json') ||
        asset.storage.type !== 'external' ||
        asset.content_hash === undefined ||
        asset.byte_length === undefined
    ) {
        throw new Error('Canonical text retrieval has no verified external asset');
    }
    const turns = createContextTurnIndex(document);
    for (const entry of document.context.entries) {
        const resolved = resolveContextEntry(turns, entry);
        if (resolved.turn.model_visibility !== 'include') continue;
        for (const candidate of resolved.blocks.flatMap((block) =>
            block.type === 'tool_result' ? block.content : [block],
        )) {
            if (
                candidate.type !== 'external_reference' ||
                candidate.asset_id !== assetId ||
                (blockId !== undefined && candidate.id !== blockId) ||
                candidate.original_type !== asset.kind ||
                candidate.content_hash !== asset.content_hash
            )
                continue;
            const definitionId = candidate.retrieval.tool_definition_id;
            const definition = definitionId ? document.tool_definitions[definitionId] : undefined;
            if (
                !definition ||
                !document.context.active_tool_definition_ids.includes(definition.id) ||
                definition.name !== candidate.retrieval.capability ||
                // Capability ABI is independent of the accepted definition content version.
                candidate.retrieval.version !== 1
            )
                continue;
            const requirements = document.context.retrieval_requirements.filter(
                (requirement) =>
                    requirement.asset_id === assetId &&
                    canonicalJsonContentString(requirement.retrieval) ===
                        canonicalJsonContentString(candidate.retrieval),
            );
            if (requirements.length !== 1) continue;
            const acceptedId = requirements[0]?.accepted_asset_operation_id;
            const accepted = acceptedId ? document.operation_receipts[acceptedId] : undefined;
            if (
                !acceptedId ||
                accepted?.operation_kind !== undefined ||
                accepted?.accepted_asset_ids?.filter((id) => id === assetId).length !== 1
            )
                continue;
            return {
                asset: structuredClone(asset),
                block: structuredClone(candidate),
                tool_definition: structuredClone(definition),
                accepted_asset_operation_id: acceptedId,
            };
        }
    }
    throw new Error('Canonical text retrieval is not required by the active context');
}

/** Selected indexed witness only; this does not establish whole-history or processing completeness. */
export function resolveIndexedTextExternalReference(
    input: unknown,
    assetId: string,
    blockId: string,
): ActiveTextExternalReference {
    if (!preflightJsonInput(input).success) throw new TypeError('Indexed retrieval evidence is not bounded JSON');
    const selected = IndexedConversationSelectedContextSchema.parse(structuredClone(input));
    if (selected.completeness !== 'selected_media_compaction_pending_admission')
        throw new Error('Indexed retrieval requires the media/compaction witness profile');
    const asset = selected.assets[assetId];
    if (
        (asset?.kind !== 'text' && asset?.kind !== 'json') ||
        asset.storage.type !== 'external' ||
        asset.content_hash === undefined ||
        asset.byte_length === undefined
    )
        throw new Error('Indexed retrieval has no exact external asset');
    const candidates: ExternalReferenceBlock[] = [];
    for (const turn of [...selected.turns, ...(selected.replacement_turns ?? []).map((value) => value.projection)]) {
        if (turn.header.model_visibility !== 'include') continue;
        for (const block of turn.selected_blocks) {
            const contents: readonly ContentBlock[] = block.type === 'tool_result' ? block.content : [block];
            for (const candidate of contents) {
                if (
                    candidate.type === 'external_reference' &&
                    candidate.id === blockId &&
                    candidate.asset_id === assetId
                )
                    candidates.push(candidate);
            }
        }
    }
    if (candidates.length !== 1) throw new Error('Indexed retrieval is not one exact selected reference');
    const candidate = candidates[0];
    if (
        candidate?.type !== 'external_reference' ||
        candidate.original_type !== asset.kind ||
        candidate.content_hash !== asset.content_hash ||
        candidate.retrieval.version !== 1
    )
        throw new Error('Indexed retrieval reference differs from its asset/ABI');
    const replacement = selected.replacement_turns?.find((value) =>
        value.projection.selected_blocks.some((block) => block.id === candidate.id),
    );
    if (replacement) {
        const witness = selected.compaction_witnesses?.[replacement.compaction_id];
        if (
            !witness ||
            witness.compaction.id !== replacement.compaction_id ||
            witness.acceptance.id !== witness.compaction.operation_id ||
            witness.acceptance.conversation_id !== selected.source.conversation_id ||
            witness.acceptance.operation_kind !== 'context_change' ||
            witness.acceptance.context_change?.kind !== 'replace_with_compaction' ||
            witness.acceptance.context_change.source_fingerprint !== witness.compaction.source.source_fingerprint ||
            witness.acceptance.result_revision !== witness.acceptance.base_revision + 1 ||
            witness.compaction.created_at !== witness.acceptance.recorded_at ||
            witness.acceptance.result_revision > selected.source.revision ||
            witness.compaction.metadata?.applied_revision !== witness.acceptance.result_revision ||
            witness.compaction.metadata?.payload_fingerprint !== witness.acceptance.payload_fingerprint ||
            !witness.compaction.retained_asset_ids.includes(assetId)
        )
            throw new Error('Indexed retrieval lacks exact accepted compaction/asset evidence');
    }
    const definitionId = candidate.retrieval.tool_definition_id;
    const definition = definitionId ? selected.tool_definitions[definitionId] : undefined;
    const requirements = selected.context.retrieval_requirements.filter(
        (value) =>
            value.asset_id === assetId &&
            canonicalJsonContentString(value.retrieval) === canonicalJsonContentString(candidate.retrieval),
    );
    const acceptedId = requirements.length === 1 ? requirements[0]?.accepted_asset_operation_id : undefined;
    const accepted = acceptedId ? selected.operation_witnesses?.[acceptedId] : undefined;
    if (
        !definition ||
        !selected.context.active_tool_definition_ids.includes(definition.id) ||
        definition.name !== candidate.retrieval.capability ||
        !acceptedId ||
        !accepted ||
        accepted.id !== acceptedId ||
        accepted.conversation_id !== selected.source.conversation_id ||
        accepted.result_revision > selected.source.revision ||
        accepted.operation_kind !== undefined ||
        accepted.accepted_asset_ids?.filter((id) => id === assetId).length !== 1
    )
        throw new Error('Indexed retrieval lacks exact active tool and accepted-asset evidence');
    return { asset, block: candidate, tool_definition: definition, accepted_asset_operation_id: acceptedId };
}
