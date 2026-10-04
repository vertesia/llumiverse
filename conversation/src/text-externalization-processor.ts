import { hashUtf8Content } from './content-integrity.js';
import { contextChangeSelectedRanges, planContextChange } from './context-change.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import type { ConversationProcessor, ProcessorResult } from './processing.js';
import { RetrievalCapabilitySchema } from './schemas/content.js';
import type {
    Asset,
    ConversationDocument,
    ExternalReferenceBlock,
    OperationReceipt,
    ProcessingJob,
    ProcessingResolvedInput,
    ProcessorConfiguration,
    RetrievalCapability,
} from './types.js';

export const TEXT_EXTERNALIZATION_PROCESSOR_ID = 'externalize-text';
export const TEXT_EXTERNALIZATION_PROCESSOR_VERSION = '1';
export const MAX_TEXT_EXTERNALIZATION_BYTES = 32 * 1024 * 1024;
export const MAX_TEXT_EXTERNALIZATION_BLOCKS = 4096;

export function textExternalizationAssetOperationId(jobId: string): string {
    return `processing:archive:${jobId}`;
}

export async function textExternalizationBlockOperationId(
    jobId: string,
    blockId: string,
    count: number,
): Promise<string> {
    return count === 1
        ? textExternalizationAssetOperationId(jobId)
        : deriveConversationId('text-externalization-asset', jobId, blockId);
}

export type TextExternalizationJobSelection =
    | { kind: 'eligible'; texts: TextExternalizationSourceBlock[] }
    | { kind: 'no_eligible_blocks' }
    | { kind: 'unsupported'; reason: 'non_entry_selection' | 'non_text_block' };

export interface TextExternalizationSourceBlock {
    entry_id: string;
    turn_id: string;
    block_id: string;
    text: string;
}

function inspectSelectedOriginal(
    document: ConversationDocument,
    resolution: Pick<ProcessingResolvedInput, 'entry_ids' | 'selected_block_ids'>,
): TextExternalizationJobSelection {
    if (resolution.entry_ids.length === 0) return { kind: 'no_eligible_blocks' };
    const turns = createContextTurnIndex(document);
    const chosen = new Set(resolution.entry_ids);
    if (chosen.size !== resolution.entry_ids.length) throw new Error('Text externalization duplicates a source entry');
    const entries = document.context.entries.filter((entry) => chosen.has(entry.id));
    if (
        entries.length !== resolution.entry_ids.length ||
        entries.some((entry, index) => entry.id !== resolution.entry_ids[index])
    ) {
        throw new Error('Text externalization source entry is unavailable or out of order');
    }
    const texts: TextExternalizationSourceBlock[] = [];
    for (const entry of entries) {
        if (entry.type !== 'source_turn') throw new Error('Text externalization source entry is unavailable');
        const { blocks } = resolveContextEntry(turns, entry);
        const selectedIds = resolution.selected_block_ids?.[entry.id];
        const selected = selectedIds ? blocks.filter((block) => selectedIds.includes(block.id)) : blocks;
        if (
            selectedIds &&
            (selectedIds.length !== selected.length || selected.some((block, index) => block.id !== selectedIds[index]))
        ) {
            throw new Error('Text externalization selected blocks are unavailable or out of order');
        }
        for (const block of selected) {
            if (block.type !== 'text') return { kind: 'unsupported', reason: 'non_text_block' };
            texts.push({ entry_id: entry.id, turn_id: entry.turn_id, block_id: block.id, text: block.text });
        }
    }
    if (texts.length === 0) return { kind: 'no_eligible_blocks' };
    return { kind: 'eligible', texts };
}

function selectedOriginal(
    document: ConversationDocument,
    resolution: Pick<ProcessingResolvedInput, 'entry_ids' | 'selected_block_ids'>,
): string {
    const selected = inspectSelectedOriginal(document, resolution);
    if (selected.kind === 'eligible' && selected.texts.length === 1) return selected.texts[0].text;
    if (selected.kind === 'eligible') throw new Error('Text externalization selects more than one text block');
    if (selected.kind === 'no_eligible_blocks') {
        throw new Error('Text externalization selects exactly one context entry');
    }
    throw new Error('Text externalization requires one ordinary text block');
}

function archivedAssets(
    document: ConversationDocument,
    job: ProcessingJob,
    count: number,
): { assets: Asset[]; receipt: OperationReceipt } {
    const receipt = document.operation_receipts[textExternalizationAssetOperationId(job.id)];
    const assetIds = receipt?.accepted_asset_ids ?? [];
    const assets = assetIds.map((id) => document.assets[id]);
    if (
        !receipt ||
        receipt.operation_kind !== undefined ||
        assetIds.length !== count ||
        new Set(assetIds).size !== count ||
        assets.some(
            (asset) =>
                asset?.kind !== 'text' ||
                asset.storage.type !== 'external' ||
                asset.byte_length === undefined ||
                asset.content_hash === undefined,
        )
    ) {
        throw new Error('Text externalization has no durably accepted verified asset');
    }
    return { assets, receipt };
}

function preview(text: string): string {
    let result = '';
    for (const scalar of text) {
        if (result.length + scalar.length > 160) break;
        result += scalar;
    }
    return result || '[archived empty text]';
}

export type TextExternalizationRetrievalBinder = (input: {
    asset: Asset;
    receipt: OperationReceipt;
    document: ConversationDocument;
    job: ProcessingJob;
}) => RetrievalCapability;

async function validatedTextExternalizationArchives(
    document: ConversationDocument,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    configuration: ProcessorConfiguration,
) {
    if (
        job.processor_id !== TEXT_EXTERNALIZATION_PROCESSOR_ID ||
        job.processor_version !== TEXT_EXTERNALIZATION_PROCESSOR_VERSION ||
        Object.keys(configuration.config).length !== 0
    ) {
        throw new Error('Text externalization processor configuration is unavailable');
    }
    const selected = inspectSelectedOriginal(document, resolution);
    if (selected.kind !== 'eligible') throw new Error('Text externalization has no supported selected text');
    const texts = selected.texts;
    if (texts.length > MAX_TEXT_EXTERNALIZATION_BLOCKS) {
        throw new RangeError('Text externalization exceeds the selected block bound');
    }
    const integrities = await Promise.all(texts.map((item) => hashUtf8Content(item.text)));
    if (integrities.reduce((total, item) => total + item.byte_length, 0) > MAX_TEXT_EXTERNALIZATION_BYTES) {
        throw new RangeError('Text externalization exceeds the verified asset bound');
    }
    const { assets, receipt } = archivedAssets(document, job, texts.length);
    for (const [index, asset] of assets.entries()) {
        if (
            asset.content_hash !== integrities[index].content_hash ||
            asset.byte_length !== integrities[index].byte_length
        ) {
            throw new Error('Accepted archive is not the exact ordered selected originals');
        }
    }
    return { texts, integrities, assets, receipt };
}

/** Pure proposal replay from retained capability data. It does not grant access or invoke a binder/plugin. */
export async function buildTextExternalizationProposal(
    document: ConversationDocument,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    configuration: ProcessorConfiguration,
    retrievals: readonly RetrievalCapability[],
): Promise<Extract<ProcessorResult, { kind: 'proposal' }>> {
    const { texts, integrities, assets, receipt } = await validatedTextExternalizationArchives(
        document,
        job,
        resolution,
        configuration,
    );
    if (retrievals.length !== texts.length) throw new Error('Text externalization retrieval bindings are incomplete');
    const plan = await planContextChange(document, {
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        entry_ids: resolution.entry_ids,
        ...(resolution.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: resolution.selected_block_ids,
                  selected_entries: resolution.selected_entries,
              }),
    });
    const compactionId = await deriveConversationId('text-externalization', job.id);
    const ranges = contextChangeSelectedRanges(document, {
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        entry_ids: resolution.entry_ids,
        ...(resolution.selected_block_ids === undefined
            ? {}
            : {
                  selected_block_ids: resolution.selected_block_ids,
                  selected_entries: resolution.selected_entries,
              }),
    });
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
            const definition = definitionId ? document.tool_definitions[definitionId] : undefined;
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
            timestamps: { recorded_at: document.updated_at },
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

/** The binder is host-injected at runtime; only a prior accepted asset append can become a proposal. */
export function createTextExternalizationProcessor(
    bindRetrieval: TextExternalizationRetrievalBinder,
): ConversationProcessor {
    return {
        async run({ document, job, resolved_input: resolution, configuration }) {
            const { assets, receipt } = await validatedTextExternalizationArchives(
                document,
                job,
                resolution,
                configuration,
            );
            const retrievals = assets.map((asset) =>
                RetrievalCapabilitySchema.parse(bindRetrieval({ asset, receipt, document, job })),
            );
            return buildTextExternalizationProposal(document, job, resolution, configuration, retrievals);
        },
    };
}

/** Source used by the host archive stage, before the processor starts or resolves a job. */
export function textForExternalizationJob(document: ConversationDocument, job: ProcessingJob): string {
    if (job.selection.kind !== 'entries') throw new Error('Text externalization requires an explicit source selection');
    return selectedOriginal(document, job.selection);
}

export function textBlocksForExternalizationJob(
    document: ConversationDocument,
    job: ProcessingJob,
): TextExternalizationSourceBlock[] {
    if (job.selection.kind !== 'entries') throw new Error('Text externalization requires an explicit source selection');
    const selected = inspectSelectedOriginal(document, job.selection);
    if (selected.kind !== 'eligible') throw new Error('Text externalization requires selected ordinary text blocks');
    return selected.texts;
}

/** The singleton fingerprint remains its historical text hash; batches bind ordered source identities. */
export async function textExternalizationArchiveInputs(document: ConversationDocument, job: ProcessingJob) {
    const texts = textBlocksForExternalizationJob(document, job);
    if (texts.length > MAX_TEXT_EXTERNALIZATION_BLOCKS) {
        throw new RangeError('Text externalization exceeds the selected block bound');
    }
    const integrities = await Promise.all(texts.map((item) => hashUtf8Content(item.text)));
    if (integrities.reduce((total, item) => total + item.byte_length, 0) > MAX_TEXT_EXTERNALIZATION_BYTES) {
        throw new RangeError('Text externalization exceeds the verified asset bound');
    }
    return {
        texts,
        integrities,
        payload_fingerprint:
            texts.length === 1
                ? integrities[0].content_hash
                : await fingerprintJson({
                      kind: 'text_externalization_batch',
                      blocks: texts.map((item, index) => ({
                          entry_id: item.entry_id,
                          turn_id: item.turn_id,
                          block_id: item.block_id,
                          ...integrities[index],
                      })),
                  }),
    };
}

/** An operational host decision; unsupported v1 selections never authorize archive I/O. */
export function inspectTextExternalizationJobSelection(
    document: ConversationDocument,
    job: ProcessingJob,
): TextExternalizationJobSelection {
    if (job.selection.kind !== 'entries') return { kind: 'unsupported', reason: 'non_entry_selection' };
    return inspectSelectedOriginal(document, job.selection);
}
