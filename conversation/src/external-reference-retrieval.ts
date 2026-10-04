import { canonicalJsonContentString } from './content-integrity.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import type { Asset, ExternalReferenceBlock, ToolDefinition } from './types.js';
import { parseConversationDocument } from './validation.js';

export interface ActiveTextExternalReference {
    asset: Asset;
    block: ExternalReferenceBlock;
    tool_definition: ToolDefinition;
    accepted_asset_operation_id: string;
}

/**
 * Resolve an active canonical reference and its accepted original, independent of any host storage protocol.
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
        asset?.kind !== 'text' ||
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
        for (const candidate of resolved.blocks) {
            if (
                candidate.type !== 'external_reference' ||
                candidate.asset_id !== assetId ||
                (blockId !== undefined && candidate.id !== blockId) ||
                candidate.original_type !== 'text' ||
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
