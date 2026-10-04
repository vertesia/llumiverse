import {
    MAX_TEXT_EXTERNALIZATION_BLOCKS,
    MAX_TEXT_EXTERNALIZATION_BYTES,
    TEXT_EXTERNALIZATION_PROCESSOR_ID,
    TEXT_EXTERNALIZATION_PROCESSOR_VERSION,
} from './text-externalization-constants.js';

export {
    MAX_TEXT_EXTERNALIZATION_BLOCKS,
    MAX_TEXT_EXTERNALIZATION_BYTES,
    TEXT_EXTERNALIZATION_PROCESSOR_ID,
    TEXT_EXTERNALIZATION_PROCESSOR_VERSION,
} from './text-externalization-constants.js';

import { hashUtf8Content } from './content-integrity.js';
import { materializedContextChangeWorkingSet } from './context-change-working-set.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import type { ConversationProcessor, ProcessorResult } from './processing.js';
import { RetrievalCapabilitySchema } from './schemas/content.js';
import { buildTextWorkingSetProposal } from './text-externalization-working-set.js';
import type {
    Asset,
    ConversationDocument,
    OperationReceipt,
    ProcessingJob,
    ProcessingResolvedInput,
    ProcessorConfiguration,
    RetrievalCapability,
} from './types.js';

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
    const { assets, receipt } = await validatedTextExternalizationArchives(document, job, resolution, configuration);
    return buildTextWorkingSetProposal(
        materializedContextChangeWorkingSet(document),
        document.tool_definitions,
        document.updated_at,
        job,
        resolution,
        configuration,
        assets,
        receipt,
        retrievals,
    );
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
