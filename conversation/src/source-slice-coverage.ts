import { createContextTurnIndex } from './context-entry-resolution.js';
import { pointerPrefix, pointerTokens } from './json-pointer.js';
import { rejectSourceSlice, type SourceSliceWork, spendSourceSliceWork } from './source-slice-work.js';

export { type SourceSliceWork, spendSourceSliceWork } from './source-slice-work.js';

import { inverseJsonPointer } from './json-pointer-inverse.js';
import type {
    ContentBlock,
    ConversationDocument,
    DerivedBlockLineageGroup,
    JsonSourceRegion,
    SourceBlockSlice,
} from './types.js';

function contained(outer: SourceBlockSlice['selection'], inner: SourceBlockSlice['selection']): boolean {
    if (outer.kind === 'whole') return true;
    if (outer.kind !== inner.kind) return false;
    if (outer.kind === 'text_range' && inner.kind === 'text_range')
        return (
            inner.range.start_code_point >= outer.range.start_code_point &&
            inner.range.end_code_point <= outer.range.end_code_point
        );
    if (outer.kind === 'json_region' && inner.kind === 'json_region') {
        const root = pointerTokens(inner.region.pointer);
        return (
            pointerPrefix(pointerTokens(outer.region.pointer), root) &&
            !outer.region.excluded_pointers.some((pointer) => pointerPrefix(pointerTokens(pointer), root))
        );
    }
    if (outer.kind === 'media_range' && inner.kind === 'media_range') {
        const a = outer.range,
            b = inner.range;
        if (a.type === 'time_range' && b.type === 'time_range')
            return b.start_seconds >= a.start_seconds && b.end_seconds <= a.end_seconds;
        if (a.type === 'page_range' && b.type === 'page_range')
            return b.from_page >= a.from_page && b.through_page <= a.through_page;
        if (a.type === 'image_region' && b.type === 'image_region')
            return (
                a.coordinate_space === b.coordinate_space &&
                b.x >= a.x &&
                b.y >= a.y &&
                b.x + b.width <= a.x + a.width &&
                b.y + b.height <= a.y + a.height
            );
    }
    return false;
}
export function sourceSlicesOverlap(a: SourceBlockSlice, b: SourceBlockSlice): boolean {
    if (a.turn_id !== b.turn_id || a.block_id !== b.block_id) return false;
    const x = a.selection,
        y = b.selection;
    if (x.kind === 'whole' || y.kind === 'whole') return true;
    if (x.kind === 'text_range' && y.kind === 'text_range')
        return x.range.start_code_point < y.range.end_code_point && y.range.start_code_point < x.range.end_code_point;
    if (x.kind === 'json_region' && y.kind === 'json_region') {
        const p = pointerTokens(x.region.pointer),
            q = pointerTokens(y.region.pointer);
        if (!pointerPrefix(p, q) && !pointerPrefix(q, p)) return false;
        const root = p.length > q.length ? p : q;
        return ![...x.region.excluded_pointers, ...y.region.excluded_pointers].some((pointer) =>
            pointerPrefix(pointerTokens(pointer), root),
        );
    }
    if (x.kind === 'media_range' && y.kind === 'media_range') {
        const p = x.range,
            q = y.range;
        if (p.type === 'time_range' && q.type === 'time_range')
            return p.start_seconds < q.end_seconds && q.start_seconds < p.end_seconds;
        if (p.type === 'page_range' && q.type === 'page_range')
            return p.from_page <= q.through_page && q.from_page <= p.through_page;
        if (p.type === 'image_region' && q.type === 'image_region') {
            if (p.coordinate_space !== q.coordinate_space)
                throw new Error('Source lineage cannot compare unproved mixed coordinate spaces');
            return p.x < q.x + q.width && q.x < p.x + p.width && p.y < q.y + q.height && q.y < p.y + p.height;
        }
    }
    throw new Error('Source-slice kinds do not match their original block');
}
function lift(group: DerivedBlockLineageGroup, slice: SourceBlockSlice, work: SourceSliceWork): SourceBlockSlice[] {
    if (slice.selection.kind === 'whole') return group.source_slices;
    if (
        group.target_block_ids.length !== 1 ||
        group.source_slices.length !== 1 ||
        group.transform === 'authored_replacement'
    )
        throw new Error('Source lineage group is indivisible');
    const source = group.source_slices[0],
        inner = slice.selection;
    if (group.transform === 'block_copy' && source.selection.kind === 'whole') return [{ ...source, selection: inner }];
    if (group.transform === 'text_slice' && source.selection.kind === 'text_range' && inner.kind === 'text_range') {
        const range = {
            start_code_point: source.selection.range.start_code_point + inner.range.start_code_point,
            end_code_point: source.selection.range.start_code_point + inner.range.end_code_point,
        };
        const selected: SourceBlockSlice = { ...source, selection: { kind: 'text_range', range } };
        if (!contained(source.selection, selected.selection))
            throw new Error('Source lineage text range shifted outside its source');
        return [selected];
    }
    if (
        group.transform === 'json_projection' &&
        source.selection.kind === 'json_region' &&
        inner.kind === 'json_region'
    ) {
        const pointer = inverseJsonPointer(group.inverse, inner.region.pointer, work),
            tokens = pointerTokens(pointer);
        const inherited = source.selection.region.excluded_pointers.filter((excluded) =>
            pointerPrefix(tokens, pointerTokens(excluded)),
        );
        const excluded = [
            ...inherited,
            ...inner.region.excluded_pointers.map((excluded) => inverseJsonPointer(group.inverse, excluded, work)),
        ];
        const region: JsonSourceRegion = { pointer, excluded_pointers: excluded };
        const selected: SourceBlockSlice = { ...source, selection: { kind: 'json_region', region } };
        if (!contained(source.selection, selected.selection))
            throw new Error('Source lineage JSON range shifted outside its source');
        return [selected];
    }
    if (
        group.transform === 'media_reference' &&
        source.selection.kind === 'media_range' &&
        inner.kind === 'media_range'
    ) {
        if (!contained(source.selection, inner))
            throw new Error('Source lineage media range shifted outside its source');
        return [
            {
                ...source,
                selection: {
                    ...inner,
                    asset_id: source.selection.asset_id,
                    asset_metadata_fingerprint: source.selection.asset_metadata_fingerprint,
                    verified_content_hash: source.selection.verified_content_hash,
                },
            },
        ];
    }
    throw new Error('Source lineage transform does not support this refinement');
}
/** Canonical ephemeral source coverage, never a second persisted conversation representation. */
export function resolveSourceSliceCoverage(
    document: ConversationDocument,
    input: readonly SourceBlockSlice[],
    work: SourceSliceWork = { nodes: 0 },
): SourceBlockSlice[] {
    const turns = createContextTurnIndex(document),
        result: SourceBlockSlice[] = [];
    const visitBatch = (slices: readonly SourceBlockSlice[], ancestors: ReadonlySet<string>): void => {
        if (ancestors.size > 128) rejectSourceSlice('Source lineage exceeds bounded depth');
        const expanded = new Set<string>();
        for (const slice of slices) {
            spendSourceSliceWork(work);
            if (slice.source.conversation_id !== document.id || slice.source.revision > document.revision) {
                rejectSourceSlice('Source lineage snapshot is outside the conversation');
            }
            const turn = turns.get(slice.turn_id),
                block = turn?.blocks.find((candidate) => candidate.id === slice.block_id);
            if (!turn || !block) rejectSourceSlice('Source lineage references an unknown block');
            const provenance = turn.provenance;
            if (provenance.type !== 'derived') {
                result.push(slice);
                continue;
            }
            const lineage = provenance.block_lineage;
            const groups = lineage?.groups.filter((group) => group.target_block_ids.includes(block.id));
            if (groups !== undefined && groups.length !== 1)
                rejectSourceSlice('Every derived target block requires exactly one lineage group');
            const group = groups?.[0];
            const indivisible =
                group === undefined ||
                group.transform === 'authored_replacement' ||
                group.target_block_ids.length !== 1;
            const key = `${turn.id}\u0000${group === undefined ? 'legacy' : lineage?.groups.indexOf(group)}`;
            if (ancestors.has(key)) rejectSourceSlice('Source lineage contains a cycle');
            if (indivisible) {
                if (expanded.has(key)) continue;
                const targets = group?.target_block_ids ?? turn.blocks.map((item) => item.id);
                if (
                    !targets.every((id) =>
                        slices.some(
                            (item) =>
                                item.turn_id === turn.id && item.block_id === id && item.selection.kind === 'whole',
                        ),
                    )
                ) {
                    rejectSourceSlice('Derived source lineage group is indivisible');
                }
                expanded.add(key);
            }
            const path = new Set(ancestors);
            path.add(key);
            if (group !== undefined) {
                visitBatch(lift(group, slice, work), path);
                continue;
            }
            const sources: SourceBlockSlice[] = [];
            for (const sourceId of provenance.source_turn_ids) {
                const source = turns.get(sourceId);
                if (!source) rejectSourceSlice('Legacy derived source is not retained');
                for (const sourceBlock of source.blocks) {
                    spendSourceSliceWork(work);
                    if (
                        provenance.source_block_ids !== undefined &&
                        !provenance.source_block_ids.includes(sourceBlock.id)
                    )
                        continue;
                    sources.push({
                        source: slice.source,
                        turn_id: sourceId,
                        block_id: sourceBlock.id,
                        block_fingerprint: 'legacy:unverified',
                        selection: { kind: 'whole' },
                    });
                }
            }
            visitBatch(sources, path);
        }
    };
    visitBatch(input, new Set());
    return result;
}
export function sourceSliceForBlock(
    document: ConversationDocument,
    turn_id: string,
    block: ContentBlock,
    block_fingerprint = 'structural:unverified',
): SourceBlockSlice {
    return {
        source: { conversation_id: document.id, revision: document.revision },
        turn_id,
        block_id: block.id,
        block_fingerprint,
        selection: { kind: 'whole' },
    };
}
