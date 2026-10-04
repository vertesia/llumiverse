import { canonicalJsonContentString, hashUtf8Content } from './content-integrity.js';
import {
    type ContextChangeWorkingSet,
    partitionSelection,
    planContextChangeWorkingSet,
    selectedRanges,
} from './context-change-working-set.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import type { ProcessorResult } from './processing.js';
import { RetrievalCapabilitySchema } from './schemas/content.js';
import { ContextChangePlanInputSchema } from './schemas/context-change.js';
import {
    MAX_TEXT_EXTERNALIZATION_BLOCKS,
    MAX_TEXT_EXTERNALIZATION_BYTES,
    TEXT_EXTERNALIZATION_PROCESSOR_ID,
    TEXT_EXTERNALIZATION_PROCESSOR_VERSION,
} from './text-externalization-constants.js';
import type { TextExternalizationSourceBlock } from './text-externalization-processor.js';
import type {
    Asset,
    ExternalReferenceBlock,
    OperationReceipt,
    ProcessingJob,
    ProcessingResolvedInput,
    ProcessorConfiguration,
    RetrievalCapability,
    ToolDefinition,
} from './types.js';

export function selectedTextWorkingSet(
    document: ContextChangeWorkingSet,
    resolution: Pick<ProcessingResolvedInput, 'entry_ids' | 'selected_block_ids'>,
): TextExternalizationSourceBlock[] {
    const chosen = new Set(resolution.entry_ids);
    if (chosen.size !== resolution.entry_ids.length) throw new Error('Text externalization duplicates a source entry');
    const entries = document.context.entries.filter((entry) => chosen.has(entry.id));
    if (entries.length !== chosen.size || entries.some((entry, index) => entry.id !== resolution.entry_ids[index]))
        throw new Error('Text externalization source entry is unavailable or out of order');
    const texts: TextExternalizationSourceBlock[] = [];
    for (const entry of entries) {
        const turn = document.turns.get(entry.turn_id);
        if (!turn || entry.type !== 'source_turn') throw new Error('Text externalization source entry is unavailable');
        const activeIds = entry.block_ids === undefined ? undefined : new Set(entry.block_ids);
        const blocks = turn.active_blocks.filter((block) => activeIds === undefined || activeIds.has(block.id));
        const selectedIds = resolution.selected_block_ids?.[entry.id];
        const selected = selectedIds === undefined ? blocks : blocks.filter((block) => selectedIds.includes(block.id));
        if (
            selectedIds !== undefined &&
            (selectedIds.length !== selected.length || selected.some((block, index) => block.id !== selectedIds[index]))
        )
            throw new Error('Text externalization selected blocks are unavailable or out of order');
        for (const block of selected) {
            if (block.type !== 'text') throw new Error('Text externalization requires ordinary selected text');
            texts.push({ entry_id: entry.id, turn_id: entry.turn_id, block_id: block.id, text: block.text });
        }
    }
    return texts;
}

function preview(text: string): string {
    let result = '';
    for (const scalar of text) {
        if (result.length + scalar.length > 160) break;
        result += scalar;
    }
    return result || '[archived empty text]';
}

/** Shared pure builder; the owning adapter has already verified exact archive custody and retrieval
 * authorization. This function neither grants access, invokes inference nor creates an archive.
 */
export async function buildTextWorkingSetProposal(
    document: ContextChangeWorkingSet,
    toolDefinitions: Record<string, ToolDefinition>,
    updatedAt: string,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    configuration: ProcessorConfiguration,
    assets: readonly Asset[],
    receipt: OperationReceipt,
    retrievals: readonly RetrievalCapability[],
): Promise<Extract<ProcessorResult, { kind: 'proposal' }>> {
    const texts = selectedTextWorkingSet(document, resolution);
    if (
        job.processor_id !== TEXT_EXTERNALIZATION_PROCESSOR_ID ||
        job.processor_version !== TEXT_EXTERNALIZATION_PROCESSOR_VERSION ||
        Object.keys(configuration.config).length !== 0 ||
        texts.length === 0 ||
        texts.length > MAX_TEXT_EXTERNALIZATION_BLOCKS ||
        retrievals.length !== texts.length ||
        assets.length !== texts.length ||
        receipt.id !== `processing:archive:${job.id}` ||
        receipt.operation_kind !== undefined ||
        receipt.conversation_id !== document.source.conversation_id ||
        receipt.result_revision > document.source.revision ||
        canonicalJsonContentString(receipt.accepted_asset_ids ?? []) !==
            canonicalJsonContentString(assets.map((a) => a.id))
    )
        throw new Error('Text externalization archive/configuration binding is unavailable');
    const integrities = await Promise.all(texts.map((item) => hashUtf8Content(item.text)));
    if (integrities.reduce((total, item) => total + item.byte_length, 0) > MAX_TEXT_EXTERNALIZATION_BYTES)
        throw new RangeError('Text externalization exceeds the verified asset bound');
    for (const [index, asset] of assets.entries()) {
        if (
            asset.kind !== 'text' ||
            asset.storage.type !== 'external' ||
            asset.content_hash !== integrities[index].content_hash ||
            asset.byte_length !== integrities[index].byte_length
        )
            throw new Error('Accepted archive is not the exact ordered selected originals');
    }
    const selection = ContextChangePlanInputSchema.parse({
        expected_revision: document.source.revision,
        expected_context_revision: document.context.revision,
        entry_ids: resolution.entry_ids,
        ...(resolution.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: resolution.selected_block_ids,
                  selected_entries: resolution.selected_entries,
              }),
    });
    const plan = await planContextChangeWorkingSet(document, selection);
    const compactionId = await deriveConversationId('text-externalization', job.id);
    const ranges = selectedRanges(document, partitionSelection(document, selection));
    let ordinal = 0;
    const replacementTurns = [];
    for (const [index, range] of ranges.entries()) {
        const blocks: ExternalReferenceBlock[] = [];
        for (const source of range.blocks) {
            const selectedBlock = texts[ordinal];
            const integrity = integrities[ordinal];
            const asset = assets[ordinal];
            if (
                source.id !== selectedBlock?.block_id ||
                asset.content_hash !== integrity.content_hash ||
                asset.byte_length !== integrity.byte_length
            ) {
                throw new Error('Accepted archive is not the exact ordered selected originals');
            }
            const retrieval = RetrievalCapabilitySchema.parse(retrievals[ordinal]);
            const definitionId = retrieval.tool_definition_id;
            const definition = definitionId ? toolDefinitions[definitionId] : undefined;
            if (
                !definition ||
                !document.context.active_tool_definition_ids.includes(definition.id) ||
                definition.name !== retrieval.capability ||
                // The capability ABI is not the tool-definition content version. Exact ID/name
                // plus the trusted host binder retain the original accepted tool/schema binding.
                retrieval.version !== 1
            )
                throw new Error('Text externalization requires a host-bound active retrieval tool');
            blocks.push({
                id:
                    texts.length === 1
                        ? await deriveConversationId('text-externalization-block', job.id)
                        : await deriveConversationId('text-externalization-block', job.id, source.id),
                type: 'external_reference',
                asset_id: asset.id,
                original_type: 'text',
                description: 'Exact original text is available on demand',
                content_hash: asset.content_hash,
                preview: preview(selectedBlock.text),
                retrieval: structuredClone(retrieval),
            });
            ordinal += 1;
        }
        replacementTurns.push({
            id:
                ranges.length === 1
                    ? await deriveConversationId('text-externalization-turn', job.id)
                    : await deriveConversationId('text-externalization-turn', job.id, String(index)),
            kind: 'agent' as const,
            authority: 'ordinary' as const,
            status: 'completed' as const,
            model_visibility: 'include' as const,
            timestamps: { recorded_at: updatedAt },
            provenance: {
                type: 'derived' as const,
                derivation_id: compactionId,
                source_turn_ids: ranges.length === 1 ? resolution.source_turn_ids : range.turn_ids,
                ...(ranges.length > 1 || range.blocks.length > 1
                    ? { source_block_ids: range.block_ids }
                    : plan.source_block_ids.length
                      ? { source_block_ids: plan.source_block_ids }
                      : {}),
                source_hash: resolution.source_fingerprint,
            },
            blocks,
        });
    }
    if (ordinal !== texts.length) throw new Error('Text archive does not cover every selected source block');
    return {
        kind: 'proposal',
        proposal: {
            kind: 'replace_with_compaction',
            compaction_id: compactionId,
            strategy: {
                id: TEXT_EXTERNALIZATION_PROCESSOR_ID,
                version: TEXT_EXTERNALIZATION_PROCESSOR_VERSION,
                configuration_fingerprint: await fingerprintJson(configuration.config),
            },
            replacement_turns: replacementTurns,
            fidelity: 'retrievable',
            accepted_asset_operation_id: receipt.id,
            retained_asset_ids: assets.map((asset) => asset.id),
            generation_ids: [],
            placement: {
                mode: ranges.length > 1 ? 'per_selected_range' : 'first_selected',
                causal_order: ranges.length > 1 ? 'preserved_disjoint_ranges' : 'contiguous',
            },
        },
    };
}
