import { pointerFromTokens, pointerPrefix, pointerTokens, resolveJsonPointer } from './json-pointer.js';
import { projectOwnedJsonSourceRegion } from './source-slice-projection.js';
import {
    rejectSourceSlice,
    rejectUnsupportedSliceFormat,
    type SourceSliceWork,
    spendSourceSliceWork,
} from './source-slice-work.js';
import type {
    ContentBlock,
    DerivedBlockLineageGroup,
    MediaSelectionRange,
    SelectedBlock,
    SourceBlockSlice,
} from './types.js';

/** Ephemeral partition of one owned canonical block, consumed by the canonical edit publisher. */
export interface SourceSliceSegment {
    selected: boolean;
    source: SourceBlockSlice;
    block: ContentBlock;
    transform: Exclude<DerivedBlockLineageGroup['transform'], 'json_minification'>;
    json_source_interval?: { start: number; end: number; total: number };
    inverse?: Extract<DerivedBlockLineageGroup, { transform: 'json_projection' }>['inverse'];
}
function assertMediaOrder(first: MediaSelectionRange, second: MediaSelectionRange): void {
    if (first.type !== second.type) rejectSourceSlice('Media subranges use different coordinate kinds');
    if (first.type === 'time_range' && second.type === 'time_range' && first.end_seconds > second.start_seconds)
        rejectSourceSlice('Media subranges overlap or are unordered');
    if (first.type === 'page_range' && second.type === 'page_range' && first.through_page >= second.from_page)
        rejectSourceSlice('Media subranges overlap or are unordered');
    if (first.type === 'image_region' && second.type === 'image_region') {
        if (first.coordinate_space !== second.coordinate_space)
            rejectSourceSlice('Media subranges require one proved coordinate space');
        if (first.y > second.y || (first.y === second.y && first.x >= second.x))
            rejectSourceSlice('Image subranges are unordered');
        if (
            first.x < second.x + second.width &&
            second.x < first.x + first.width &&
            first.y < second.y + second.height &&
            second.y < first.y + first.height
        )
            rejectSourceSlice('Image subranges overlap');
    }
}
/** The source is already bounded; this computes typed reference complements, never decoded/cropped bytes. */
function mediaRemainders(
    outer: MediaSelectionRange,
    chosen: readonly MediaSelectionRange[],
    work: SourceSliceWork,
): MediaSelectionRange[] {
    if (outer.type === 'time_range') {
        let cursor = outer.start_seconds;
        const remaining: MediaSelectionRange[] = [];
        for (const range of chosen) {
            if (
                range.type !== 'time_range' ||
                range.start_seconds < cursor ||
                range.end_seconds > outer.end_seconds ||
                range.start_seconds >= range.end_seconds
            )
                rejectSourceSlice('Time subrange is outside its bounded original source');
            if (cursor < range.start_seconds)
                remaining.push({ ...outer, start_seconds: cursor, end_seconds: range.start_seconds });
            cursor = range.end_seconds;
        }
        if (cursor < outer.end_seconds) remaining.push({ ...outer, start_seconds: cursor });
        return remaining;
    }
    if (outer.type === 'page_range') {
        let cursor = outer.from_page;
        const remaining: MediaSelectionRange[] = [];
        for (const range of chosen) {
            if (
                range.type !== 'page_range' ||
                range.from_page < cursor ||
                range.through_page > outer.through_page ||
                range.from_page > range.through_page
            )
                rejectSourceSlice('Page subrange is outside its bounded original source');
            if (cursor < range.from_page)
                remaining.push({ ...outer, from_page: cursor, through_page: range.from_page - 1 });
            cursor = range.through_page === Number.MAX_SAFE_INTEGER ? Number.POSITIVE_INFINITY : range.through_page + 1;
        }
        if (cursor <= outer.through_page) remaining.push({ ...outer, from_page: cursor });
        return remaining;
    }
    if (!Number.isFinite(outer.x + outer.width) || !Number.isFinite(outer.y + outer.height))
        rejectSourceSlice('Original media reference bounds overflow');
    let remaining = [outer];
    for (const range of chosen) {
        spendSourceSliceWork(work, remaining.length);
        if (
            range.type !== 'image_region' ||
            range.coordinate_space !== outer.coordinate_space ||
            range.x < outer.x ||
            range.y < outer.y ||
            !Number.isFinite(range.x + range.width) ||
            !Number.isFinite(range.y + range.height) ||
            range.x + range.width > outer.x + outer.width ||
            range.y + range.height > outer.y + outer.height
        )
            rejectSourceSlice('Image subrange is outside its bounded original source');
        const next: typeof remaining = [];
        for (const tile of remaining) {
            const x = Math.max(tile.x, range.x),
                y = Math.max(tile.y, range.y);
            const right = Math.min(tile.x + tile.width, range.x + range.width),
                bottom = Math.min(tile.y + tile.height, range.y + range.height);
            if (x >= right || y >= bottom) {
                next.push(tile);
                continue;
            }
            if (tile.y < y) next.push({ ...tile, height: y - tile.y });
            if (bottom < tile.y + tile.height) next.push({ ...tile, y: bottom, height: tile.y + tile.height - bottom });
            if (tile.x < x) next.push({ ...tile, y, width: x - tile.x, height: bottom - y });
            if (right < tile.x + tile.width)
                next.push({ ...tile, x: right, y, width: tile.x + tile.width - right, height: bottom - y });
        }
        if (next.length > 4096) rejectSourceSlice('Media reference partition exceeds bounded record limits');
        remaining = next;
    }
    return remaining;
}
export function segmentOwnedSourceBlock(
    block: ContentBlock,
    base: Omit<SourceBlockSlice, 'selection'>,
    selected: readonly SelectedBlock[],
    work: SourceSliceWork,
): SourceSliceSegment[] {
    if (!selected.length || selected.length > 256)
        rejectSourceSlice('Source block requires a bounded nonempty subrange selection');
    if (
        selected.some(
            (item) =>
                item.block_id !== block.id ||
                item.block_fingerprint !== base.block_fingerprint ||
                item.block_type !== block.type,
        )
    )
        rejectSourceSlice('Subrange source identity or fingerprint conflicts');
    if (selected.some((item) => item.kind === 'whole')) {
        if (selected.length !== 1) rejectSourceSlice('Whole source block cannot overlap a subrange');
        return [{ selected: true, source: { ...base, selection: { kind: 'whole' } }, block, transform: 'block_copy' }];
    }
    if (block.type === 'text') {
        if (block.format !== 'plain') rejectUnsupportedSliceFormat();
        let count = 0;
        for (const _point of block.text) {
            count++;
            spendSourceSliceWork(work);
        }
        const points = Array.from(block.text),
            parts: SourceSliceSegment[] = [];
        const emit = (start: number, end: number, chosen: boolean) => {
            if (start === end) return;
            parts.push({
                selected: chosen,
                source: {
                    ...base,
                    selection: { kind: 'text_range', range: { start_code_point: start, end_code_point: end } },
                },
                block: { ...block, text: points.slice(start, end).join('') },
                transform: 'text_slice',
            });
        };
        let cursor = 0;
        for (const item of selected) {
            if (
                item.kind !== 'text_range' ||
                item.range.start_code_point < cursor ||
                item.range.start_code_point >= item.range.end_code_point ||
                item.range.end_code_point > count
            )
                rejectSourceSlice('Text subranges overlap, shift or exceed their source');
            emit(cursor, item.range.start_code_point, false);
            emit(item.range.start_code_point, item.range.end_code_point, true);
            cursor = item.range.end_code_point;
        }
        emit(cursor, count, false);
        return parts;
    }
    if (block.type === 'json') {
        // Subtree positions precede projection/packing. Scalars and empty containers are atomic;
        // nonempty containers span their original ordered descendants.
        const intervals = new Map<string, { start: number; end: number }>();
        let cursor = 0;
        const visit = (value: import('./types.js').JsonValue, path: string[]): void => {
            spendSourceSliceWork(work);
            const start = cursor;
            if (value === null || typeof value !== 'object') cursor++;
            else if (Array.isArray(value)) {
                if (!value.length) cursor++;
                for (const [index, child] of value.entries()) visit(child, [...path, String(index)]);
            } else {
                const keys = Object.keys(value).sort();
                if (!keys.length) cursor++;
                for (const key of keys) visit(value[key], [...path, key]);
            }
            intervals.set(pointerFromTokens(path), { start, end: cursor });
        };
        visit(block.value, []);

        const pointers: string[] = [],
            parts: SourceSliceSegment[] = [];
        const order = new WeakMap<object, ReadonlyMap<string, number>>();
        let previous: number[] | undefined;
        for (const item of selected) {
            if (item.kind !== 'json_pointer') rejectSourceSlice('JSON subrange has another content kind');
            const current = resolveJsonPointer(block.value, item.pointer, order);
            spendSourceSliceWork(work, pointers.length + 1);
            for (const prior of pointers)
                if (
                    pointerPrefix(pointerTokens(prior), current.segments) ||
                    pointerPrefix(current.segments, pointerTokens(prior))
                )
                    rejectSourceSlice('JSON subtrees overlap');
            if (previous !== undefined) {
                let comparison = 0;
                for (
                    let index = 0;
                    index < Math.min(previous.length, current.order.length) && comparison === 0;
                    index++
                )
                    comparison = previous[index] - current.order[index];
                if (comparison === 0) comparison = previous.length - current.order.length;
                if (comparison >= 0) rejectSourceSlice('JSON subtrees must follow original source order');
            }
            previous = current.order;
            const interval = intervals.get(item.pointer);
            if (interval === undefined) rejectSourceSlice('JSON source topology is unavailable');
            pointers.push(item.pointer);
            const region = { pointer: item.pointer, excluded_pointers: [] };
            const projected = projectOwnedJsonSourceRegion(block.value, region, work);
            parts.push({
                selected: true,
                json_source_interval: { ...interval, total: cursor },
                source: { ...base, selection: { kind: 'json_region', region } },
                block: { ...block, value: projected.value },
                transform: 'json_projection',
                inverse: projected.inverse,
            });
        }
        if (!pointers.includes('')) {
            const region = { pointer: '', excluded_pointers: pointers };
            const projected = projectOwnedJsonSourceRegion(block.value, region, work);
            parts.push({
                selected: false,
                source: { ...base, selection: { kind: 'json_region', region } },
                block: { ...block, value: projected.value },
                transform: 'json_projection',
                inverse: projected.inverse,
            });
        }
        return parts;
    }
    if (block.type !== 'image' && block.type !== 'document' && block.type !== 'audio' && block.type !== 'video')
        rejectSourceSlice('This source block is not eligible for partial mutation');
    if (block.selection === undefined)
        rejectSourceSlice('Media mutation requires an already bounded original source selection');
    const chosen: MediaSelectionRange[] = [],
        parts: SourceSliceSegment[] = [];
    for (const item of selected) {
        if (item.kind !== 'media_range' || item.evidence.content.status !== 'verified_inline')
            rejectSourceSlice('Media mutation requires verified original bytes');
        for (const prior of chosen) {
            spendSourceSliceWork(work);
            assertMediaOrder(prior, item.range);
        }
        const firstEvidence = selected[0];
        if (
            firstEvidence.kind !== 'media_range' ||
            firstEvidence.evidence.content.status !== 'verified_inline' ||
            item.evidence.asset_id !== block.asset_id ||
            item.evidence.source_metadata_fingerprint !== firstEvidence.evidence.source_metadata_fingerprint ||
            item.evidence.content.content_hash !== firstEvidence.evidence.content.content_hash
        )
            rejectSourceSlice('Same-block media source evidence conflicts');
        chosen.push(item.range);
    }
    const remainder = mediaRemainders(block.selection, chosen, work);
    const first = selected[0];
    if (first.kind !== 'media_range' || first.evidence.content.status !== 'verified_inline')
        rejectSourceSlice('Missing verified original media evidence');
    for (const [range, isSelected] of [
        ...chosen.map((range) => [range, true] as const),
        ...remainder.map((range) => [range, false] as const),
    ]) {
        const selection: SourceBlockSlice['selection'] = {
            kind: 'media_range',
            range,
            asset_id: block.asset_id,
            asset_metadata_fingerprint: first.evidence.source_metadata_fingerprint,
            verified_content_hash: first.evidence.content.content_hash,
        };
        let target: ContentBlock;
        if (block.type === 'image' && range.type === 'image_region') target = { ...block, selection: range };
        else if (block.type === 'document' && range.type === 'page_range') target = { ...block, selection: range };
        else if ((block.type === 'audio' || block.type === 'video') && range.type === 'time_range')
            target = { ...block, selection: range };
        else rejectSourceSlice('Media reference type and range conflict');
        parts.push({
            selected: isSelected,
            source: { ...base, selection },
            block: target,
            transform: 'media_reference',
        });
    }
    return parts.sort((a, b) => {
        if (a.source.selection.kind !== 'media_range' || b.source.selection.kind !== 'media_range') return 0;
        const x = a.source.selection.range,
            y = b.source.selection.range;
        if (x.type === 'time_range' && y.type === 'time_range') return x.start_seconds - y.start_seconds;
        if (x.type === 'page_range' && y.type === 'page_range') return x.from_page - y.from_page;
        return x.type === 'image_region' && y.type === 'image_region' ? x.y - y.y || x.x - y.x : 0;
    });
}
