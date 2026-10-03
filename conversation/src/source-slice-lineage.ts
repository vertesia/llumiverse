import { fingerprintAssetSelectionMetadata } from './asset-selection-integrity.js';
import { canonicalJsonContentString, inlineAssetContentIntegrity } from './content-integrity.js';
import { createContextTurnIndex } from './context-entry-resolution.js';
import { assertSliceEditTopology, fingerprintSliceEditTopology } from './conversation-slice-topology.js';
import { fingerprintJson } from './identity.js';
import { assertJsonMinification } from './json-minification.js';
import { assertRetainedJsonMinificationOutput } from './json-minification-evidence.js';
import { pointerPrefix, pointerTokens, resolveJsonPointer } from './json-pointer.js';
import { assertDerivedBlockLineageStructure } from './source-slice-structure.js';

export { assertDerivedBlockLineageStructure } from './source-slice-structure.js';

import { preflightJsonInput } from './json-preflight.js';
import { DerivedLineageVerificationScopeSchema } from './schemas/source-slices.js';
import { projectOwnedJsonSourceRegion } from './source-slice-projection.js';
import { rejectSourceSlice, type SourceSliceWork } from './source-slice-work.js';
import type {
    Asset,
    ContentBlock,
    ConversationDocument,
    ConversationTurn,
    DerivedLineageVerificationScope,
} from './types.js';
import { parseConversationDocument } from './validation.js';

const same = (first: unknown, second: unknown) =>
    canonicalJsonContentString(first) === canonicalJsonContentString(second);

async function retainedEditReceipt(document: ConversationDocument, turn: ConversationTurn) {
    if (turn.provenance.type !== 'derived') rejectSourceSlice('Expected derived provenance');
    const id = turn.provenance.derivation_id;
    const compaction = Object.hasOwn(document.compactions, id) ? document.compactions[id] : undefined;
    const receiptId = compaction?.operation_id ?? id;
    const receipt = Object.hasOwn(document.operation_receipts, receiptId)
        ? document.operation_receipts[receiptId]
        : undefined;
    const operation = receipt?.processing_operation;
    const application = operation?.transform_application;
    if (receipt?.operation_kind === 'processing' && operation?.phase === 'complete' && application) {
        const jobId = operation.job_id;
        const job = jobId === undefined ? undefined : document.processing.jobs?.[jobId];
        const output = jobId === undefined ? undefined : document.processing.outputs?.[jobId];
        const resolution = jobId === undefined ? undefined : document.processing.resolved_inputs?.[jobId];
        const completion = jobId === undefined ? undefined : document.processing.completions?.[jobId];
        if (
            !compaction ||
            compaction.id !== application.compaction_id ||
            compaction.fidelity !== 'value_preserving' ||
            !job ||
            output?.kind !== 'json_minification' ||
            !resolution ||
            !completion ||
            application.source.conversation_id !== document.id ||
            application.source.revision !== receipt.base_revision ||
            application.output_fingerprint !== output.output_fingerprint ||
            completion.output_fingerprint !== output.output_fingerprint ||
            completion.status !== 'applied' ||
            completion.result_revision !== receipt.result_revision ||
            operation.result_fingerprint !== (await fingerprintJson(completion)) ||
            receipt.payload_fingerprint !== (await fingerprintJson({ completion, application }))
        )
            rejectSourceSlice('JSON minification lineage lacks its exact durable processing authority');
        await assertRetainedJsonMinificationOutput(document, job, resolution, output);
        const retainedSlices = compaction.replacement_turns.flatMap((replacement) =>
            replacement.provenance.type === 'derived'
                ? (replacement.provenance.block_lineage?.groups.flatMap((group) => group.source_slices) ?? [])
                : [],
        );
        if (
            !same(retainedSlices, application.source_slices) ||
            application.created_turns.length !== compaction.replacement_turns.length ||
            !same(
                completion.inserted_entry_ids,
                application.created_entries.map((entry) => entry.id),
            ) ||
            !same(receipt.accepted_context_entry_ids, completion.inserted_entry_ids)
        )
            rejectSourceSlice('JSON minification application coverage conflicts');
        for (const replacement of compaction.replacement_turns) {
            if (
                application.created_turns.find((ref) => ref.id === replacement.id)?.fingerprint !==
                (await fingerprintJson(replacement))
            )
                rejectSourceSlice('JSON minification application target coverage conflicts');
        }

        return { receipt, created_turns: application.created_turns };
    }
    if (receipt?.operation_kind !== 'conversation_edit' || receipt.conversation_edit === undefined) {
        rejectSourceSlice('Precise derived lineage requires its retained edit operation receipt');
    }
    const detail = receipt.conversation_edit;
    if (
        detail.source.conversation_id !== document.id ||
        detail.source.revision !== receipt.base_revision ||
        !detail.created_turns.some((item) => item.id === turn.id)
    ) {
        rejectSourceSlice('Derived lineage is not bound to the retained edit source and target');
    }
    return { receipt, created_turns: detail.created_turns };
}

/**
 * Own and validate before the first await, then verify every retained source/target content proof.
 * Callers must use the returned snapshot. Schema parsing alone intentionally does not verify hashes.
 */
export async function verifyDerivedBlockLineage(
    input: ConversationDocument,
    scopeInput?: DerivedLineageVerificationScope,
): Promise<ConversationDocument> {
    if (scopeInput !== undefined && !preflightJsonInput(scopeInput).success)
        rejectSourceSlice('Invalid bounded lineage verification scope');
    const scope = scopeInput === undefined ? undefined : DerivedLineageVerificationScopeSchema.parse(scopeInput);
    const document = parseConversationDocument(input),
        work: SourceSliceWork = { nodes: 0 };
    assertDerivedBlockLineageStructure(document, work);
    if (scope === undefined) {
        for (const receipt of Object.values(document.operation_receipts)) {
            const operation = receipt.conversation_edit;
            if (receipt.operation_kind === 'conversation_edit' && operation?.version === 2) {
                assertSliceEditTopology(operation);
                if (operation.source_topology_fingerprint !== (await fingerprintSliceEditTopology(operation)))
                    rejectSourceSlice('Retained slice topology fingerprint conflicts');
            }
        }
    }
    const turns = createContextTurnIndex(document),
        hashes = new Map<string, Promise<string>>(),
        assetMetadataHashes = new Map<string, ReturnType<typeof fingerprintAssetSelectionMetadata>>(),
        assetIntegrity = new Map<string, ReturnType<typeof inlineAssetContentIntegrity>>();
    const hashBlock = (block: ContentBlock) => {
        let hash = hashes.get(block.id);
        if (hash === undefined) {
            hash = fingerprintJson(block);
            hashes.set(block.id, hash);
        }
        return hash;
    };
    const metadataHash = (asset: Asset) => {
        let pending = assetMetadataHashes.get(asset.id);
        if (pending === undefined) {
            pending = fingerprintAssetSelectionMetadata(asset);
            assetMetadataHashes.set(asset.id, pending);
        }
        return pending;
    };
    const contentIntegrity = (asset: Asset) => {
        let pending = assetIntegrity.get(asset.id);
        if (pending === undefined) {
            pending = inlineAssetContentIntegrity(asset.storage);
            assetIntegrity.set(asset.id, pending);
        }
        return pending;
    };
    const required = new Set<string>();
    const collect = (id: string, path: ReadonlySet<string>): void => {
        if (path.size > 128 || path.has(id)) rejectSourceSlice('Lineage dependency cycle or depth limit');
        if (required.has(id)) return;
        const turn = turns.get(id);
        if (turn === undefined) rejectSourceSlice('Required lineage dependency record is unavailable');
        required.add(id);
        if (turn.provenance.type === 'derived' && turn.provenance.block_lineage !== undefined) {
            const next = new Set(path);
            next.add(id);
            for (const group of turn.provenance.block_lineage.groups)
                for (const slice of group.source_slices) collect(slice.turn_id, next);
        }
    };
    for (const id of scope?.turn_ids ?? [...turns.keys()]) collect(id, new Set());
    for (const id of required) {
        const turn = turns.get(id);
        if (turn === undefined) rejectSourceSlice('Required lineage dependency record is unavailable');
        if (turn.provenance.type !== 'derived' || turn.provenance.block_lineage === undefined) continue;
        const authority = await retainedEditReceipt(document, turn),
            receipt = authority.receipt,
            detail = receipt.conversation_edit;
        if (detail?.version === 2) {
            assertSliceEditTopology(detail);
            if (detail.source_topology_fingerprint !== (await fingerprintSliceEditTopology(detail)))
                rejectSourceSlice('Retained slice topology fingerprint conflicts');
        }
        const target = authority.created_turns.find((item) => item.id === turn.id);
        if (target === undefined || target.fingerprint !== (await fingerprintJson(turn))) {
            rejectSourceSlice('Derived target content conflicts with the accepted edit receipt');
        }
        for (const group of turn.provenance.block_lineage.groups) {
            for (const slice of group.source_slices) {
                const source = turns.get(slice.turn_id),
                    block = source?.blocks.find((item) => item.id === slice.block_id);
                if (
                    block === undefined ||
                    slice.source.revision !== receipt.base_revision ||
                    slice.block_fingerprint !== (await hashBlock(block))
                ) {
                    rejectSourceSlice('Derived source revision or block content hash conflicts with its accepted edit');
                }
                if (slice.selection.kind === 'media_range') {
                    const assetId = slice.selection.asset_id;
                    const asset = Object.hasOwn(document.assets, assetId) ? document.assets[assetId] : undefined;
                    if (
                        asset === undefined ||
                        slice.selection.asset_metadata_fingerprint !== (await metadataHash(asset))
                    ) {
                        rejectSourceSlice('Media lineage source metadata does not match');
                    }
                    const integrity = await contentIntegrity(asset);
                    if (
                        integrity === undefined ||
                        integrity.content_hash !== slice.selection.verified_content_hash ||
                        (asset.content_hash !== undefined && asset.content_hash !== integrity.content_hash) ||
                        (asset.byte_length !== undefined && asset.byte_length !== integrity.byte_length)
                    ) {
                        rejectSourceSlice('Media lineage requires verified original bytes');
                    }
                }
            }
            if (group.transform === 'authored_replacement') continue;
            const slice = group.source_slices[0],
                targetId = group.target_block_ids[0];
            const sourceBlock = turns.get(slice.turn_id)?.blocks.find((item) => item.id === slice.block_id);
            const targetBlock = turn.blocks.find((item) => item.id === targetId);
            if (sourceBlock === undefined || targetBlock === undefined)
                rejectSourceSlice('Derived projection block is missing');
            const selection = slice.selection;
            if (group.transform === 'block_copy') {
                if (selection.kind !== 'whole' || !same({ ...sourceBlock, id: targetBlock.id }, targetBlock)) {
                    rejectSourceSlice('Derived block copy differs from its original source');
                }
            } else if (group.transform === 'json_minification') {
                if (
                    receipt.processing_operation?.transform_application === undefined ||
                    sourceBlock.type !== 'text' ||
                    targetBlock.type !== 'text' ||
                    sourceBlock.format !== 'plain' ||
                    selection.kind !== 'whole'
                )
                    rejectSourceSlice('JSON minification requires processing authority and whole raw text');
                assertJsonMinification(sourceBlock.text, targetBlock.text);
                const output = document.processing.outputs?.[receipt.processing_operation.job_id ?? ''];
                if (
                    output?.kind !== 'json_minification' ||
                    !output.proposal.transforms.some(
                        (transform) =>
                            transform.source_slice.turn_id === slice.turn_id &&
                            transform.source_slice.block_id === slice.block_id &&
                            transform.source_slice.block_fingerprint === slice.block_fingerprint &&
                            transform.replacement_text === targetBlock.text &&
                            transform.source_slice.source.revision <= receipt.base_revision,
                    )
                )
                    rejectSourceSlice('JSON minification target differs from its retained measured output');

                if (!same({ ...sourceBlock, id: targetBlock.id, text: targetBlock.text }, targetBlock))
                    rejectSourceSlice('JSON minification changed source metadata');
            } else if (group.transform === 'text_slice') {
                if (
                    sourceBlock.type !== 'text' ||
                    targetBlock.type !== 'text' ||
                    selection.kind !== 'text_range' ||
                    !same(
                        {
                            ...sourceBlock,
                            id: targetBlock.id,
                            text: Array.from(sourceBlock.text)
                                .slice(selection.range.start_code_point, selection.range.end_code_point)
                                .join(''),
                        },
                        targetBlock,
                    )
                ) {
                    rejectSourceSlice('Derived text content differs from its exact source code-point slice');
                }
            } else if (group.transform === 'json_projection') {
                if (sourceBlock.type !== 'json' || targetBlock.type !== 'json' || selection.kind !== 'json_region')
                    rejectSourceSlice('JSON projection type conflicts');
                const projected = projectOwnedJsonSourceRegion(sourceBlock.value, selection.region, work);
                if (!same(projected.value, targetBlock.value) || !same(projected.inverse, group.inverse))
                    rejectSourceSlice('JSON projection content or inverse source positions conflict');
                for (const array of group.inverse.arrays) {
                    const sourceValue = resolveJsonPointer(sourceBlock.value, array.source_pointer).value;
                    if (
                        !Array.isArray(sourceValue) ||
                        !pointerPrefix(pointerTokens(selection.region.pointer), pointerTokens(array.source_pointer))
                    ) {
                        rejectSourceSlice('JSON inverse array mapping is outside the source region');
                    }
                }
            } else if (group.transform === 'media_reference') {
                if (
                    selection.kind !== 'media_range' ||
                    !same({ ...sourceBlock, id: targetBlock.id, selection: selection.range }, targetBlock)
                ) {
                    rejectSourceSlice(
                        'Derived media reference does not preserve original asset bytes and exact selection',
                    );
                }
            }
        }
    }
    return document;
}
