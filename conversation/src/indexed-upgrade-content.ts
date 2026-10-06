import type { z } from 'zod';
import { inlineAssetContentIntegrity } from './content-integrity.js';
import { fingerprintJson } from './identity.js';
import { type IndexedConversationRecordStore, loadRecord } from './indexed-conversation.js';
import { IndexedConversationUpgradeEvidenceError } from './indexed-upgrade-progress.js';
import { getPagedRecord } from './paged-record-index.js';
import { AssetSchema, type ContentBlockSchema, ToolDefinitionSchema } from './schemas/content.js';
import type { IndexedConversationRoot } from './schemas/indexed-head.js';

/** Original media stays an authenticated canonical reference; migration performs no retrieval
 * or provider preparation, but does verify every retained identity and inline byte claim. */
export async function auditIndexedUpgradeContent(
    store: IndexedConversationRecordStore,
    root: IndexedConversationRoot,
    block: z.infer<typeof ContentBlockSchema>,
): Promise<void> {
    if ('asset_id' in block) {
        const asset = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.assets, block.asset_id),
            AssetSchema,
        );
        if (
            asset.id !== block.asset_id ||
            (['image', 'document', 'audio', 'video'].includes(block.type) && asset.kind !== block.type) ||
            (block.type === 'external_reference' &&
                block.content_hash !== undefined &&
                block.content_hash !== asset.content_hash)
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade content references different original media',
            );
        const integrity = await inlineAssetContentIntegrity(asset.storage);
        if (
            integrity &&
            ((asset.content_hash !== undefined && asset.content_hash !== integrity.content_hash) ||
                (asset.byte_length !== undefined && asset.byte_length !== integrity.byte_length))
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade inline original bytes differ from retained integrity',
            );
        if (
            block.type === 'document' &&
            block.selection &&
            asset.media?.page_count !== undefined &&
            block.selection.through_page > asset.media.page_count
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade original document selection exceeds media',
            );
        if (
            (block.type === 'audio' || block.type === 'video') &&
            block.selection &&
            asset.media?.duration_seconds !== undefined &&
            block.selection.end_seconds > asset.media.duration_seconds
        )
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade original media selection exceeds duration',
            );
    }
    const definitionId =
        block.type === 'tool_call'
            ? block.definition_id
            : block.type === 'external_reference'
              ? block.retrieval.tool_definition_id
              : undefined;
    if (definitionId !== undefined) {
        const definition = await loadRecord(
            store,
            await getPagedRecord(store, root.directories.tool_definitions, definitionId),
            ToolDefinitionSchema,
        );
        if (definition.id !== definitionId || (block.type === 'tool_call' && definition.name !== block.tool_name))
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade content lacks original matching tool definition',
            );
    }
    if (block.type === 'tool_call' && block.arguments.type === 'externalized_json') {
        if ((await fingerprintJson(block.arguments.value)) !== block.arguments.exact_arguments_hash)
            throw new IndexedConversationUpgradeEvidenceError(
                'Indexed upgrade original external arguments hash differs',
            );
        for (const hydration of block.arguments.hydration) {
            const asset = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.assets, hydration.asset_id),
                AssetSchema,
            );
            if (
                asset.id !== hydration.asset_id ||
                asset.content_hash !== hydration.content_hash ||
                asset.kind !== 'text'
            )
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade original argument hydration differs',
                );
        }
        for (const archive of block.arguments.invalidated_replay_archives ?? []) {
            const asset = await loadRecord(
                store,
                await getPagedRecord(store, root.directories.assets, archive.asset_id),
                AssetSchema,
            );
            if (asset.id !== archive.asset_id || asset.content_hash !== archive.content_hash)
                throw new IndexedConversationUpgradeEvidenceError(
                    'Indexed upgrade original invalidated replay archive differs',
                );
        }
    }
}
