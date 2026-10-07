import { canonicalJsonContentString } from './content-integrity.js';
import { createContextTurnIndex } from './context-entry-resolution.js';
import { resolveSourceSliceCoverage, sourceSlicesOverlap } from './source-slice-coverage.js';
import { projectOwnedJsonSourceRegion } from './source-slice-projection.js';
import {
    rejectSourceSlice,
    rejectUnsupportedSliceFormat,
    type SourceSliceWork,
    spendSourceSliceWork,
} from './source-slice-work.js';
import type { ContentBlock, ConversationDocument, SourceBlockSlice } from './types.js';

const same = (first: unknown, second: unknown) =>
    canonicalJsonContentString(first) === canonicalJsonContentString(second);

function assertSliceBounds(block: ContentBlock, slice: SourceBlockSlice, work: SourceSliceWork): void {
    const selection = slice.selection;
    if (selection.kind === 'whole') return;
    if (selection.kind === 'text_range') {
        if (block.type !== 'text') rejectSourceSlice('Text lineage requires a text source block');
        if (block.format !== 'plain') rejectUnsupportedSliceFormat();
        let pointCount = 0;
        for (const _point of block.text) {
            pointCount++;
            spendSourceSliceWork(work);
        }
        if (
            selection.range.start_code_point >= selection.range.end_code_point ||
            selection.range.end_code_point > pointCount
        ) {
            rejectSourceSlice('Text lineage range is shifted or outside the source');
        }
        return;
    }
    if (selection.kind === 'json_region') {
        if (block.type !== 'json') rejectSourceSlice('JSON lineage requires a JSON source block');
        projectOwnedJsonSourceRegion(block.value, selection.region, work);
        return;
    }
    if (block.type !== 'image' && block.type !== 'document' && block.type !== 'audio' && block.type !== 'video') {
        rejectSourceSlice('Media lineage requires an original media source block');
    }
    if (block.asset_id !== selection.asset_id || block.selection === undefined) {
        rejectSourceSlice('Media reference mutation requires an already bounded original selection');
    }
    const outer = block.selection,
        inner = selection.range;
    for (const range of [outer, inner]) {
        if (
            range.type === 'image_region' &&
            (!Number.isFinite(range.x + range.width) || !Number.isFinite(range.y + range.height))
        )
            rejectSourceSlice('Media reference bounds overflow');
    }
    const contained =
        outer.type === 'image_region' && inner.type === 'image_region'
            ? outer.coordinate_space === inner.coordinate_space &&
              inner.x >= outer.x &&
              inner.y >= outer.y &&
              inner.x + inner.width <= outer.x + outer.width &&
              inner.y + inner.height <= outer.y + outer.height
            : outer.type === 'page_range' && inner.type === 'page_range'
              ? inner.from_page >= outer.from_page && inner.through_page <= outer.through_page
              : outer.type === 'time_range' &&
                inner.type === 'time_range' &&
                inner.start_seconds >= outer.start_seconds &&
                inner.end_seconds <= outer.end_seconds;
    const ordered =
        inner.type === 'time_range'
            ? inner.start_seconds < inner.end_seconds
            : inner.type === 'page_range'
              ? inner.from_page <= inner.through_page
              : true;
    if (!contained || !ordered) rejectSourceSlice('Media lineage range is shifted or outside the bounded source');
}

/** Structural only. This never claims that a retained SHA-256 string matches source content. */
export function assertDerivedBlockLineageStructure(
    document: ConversationDocument,
    work: SourceSliceWork = { nodes: 0 },
): void {
    const turns = createContextTurnIndex(document);
    for (const turn of turns.values()) {
        if (turn.provenance.type !== 'derived' || turn.provenance.block_lineage === undefined) continue;
        const lineage = turn.provenance.block_lineage;
        const targetIds = new Set(turn.blocks.map((block) => block.id));
        const assigned = new Set<string>(),
            direct: SourceBlockSlice[] = [];
        for (const group of lineage.groups) {
            spendSourceSliceWork(work);
            for (const id of group.target_block_ids) {
                if (!targetIds.has(id) || assigned.has(id))
                    rejectSourceSlice('Every derived target block requires exactly one lineage group');
                assigned.add(id);
            }
            if (
                group.transform !== 'authored_replacement' &&
                (group.target_block_ids.length !== 1 || group.source_slices.length !== 1)
            ) {
                rejectSourceSlice('Exact projection requires one target and one source slice');
            }
            for (const slice of group.source_slices) {
                spendSourceSliceWork(work);
                const source = turns.get(slice.turn_id),
                    block = source?.blocks.find((item) => item.id === slice.block_id);
                if (source === undefined || block === undefined)
                    rejectSourceSlice('Lineage source block is not retained');
                if (slice.source.conversation_id !== document.id || slice.source.revision > document.revision) {
                    rejectSourceSlice('Lineage source snapshot is not retained');
                }
                if (
                    group.transform === 'json_minification' &&
                    (block.type !== 'text' || block.format !== 'plain' || slice.selection.kind !== 'whole')
                )
                    rejectSourceSlice('JSON minification requires a whole plain text source');
                assertSliceBounds(block, slice, work);
                direct.push(slice);
            }
        }
        if (assigned.size !== targetIds.size) rejectSourceSlice('Derived target block has no lineage group');
        const sourceTurns = [...new Set(direct.map((slice) => slice.turn_id))];
        const sourceBlocks = [...new Set(direct.map((slice) => slice.block_id))];
        if (
            !same(sourceTurns, turn.provenance.source_turn_ids) ||
            !same(sourceBlocks, turn.provenance.source_block_ids)
        ) {
            rejectSourceSlice('Precise lineage differs from declared derived source identities');
        }
        const coverage = resolveSourceSliceCoverage(document, direct, work);
        for (let index = 0; index < coverage.length; index++) {
            for (let prior = 0; prior < index; prior++) {
                spendSourceSliceWork(work);
                if (sourceSlicesOverlap(coverage[index], coverage[prior]))
                    rejectSourceSlice('Derived lineage source slices overlap');
            }
        }
    }
}
