import { fingerprintAssetSelectionMetadata } from './asset-selection-integrity.js';

export { fingerprintAssetSelectionMetadata } from './asset-selection-integrity.js';

import {
    type ContentIntegrity,
    hashContentBytes,
    inlineAssetContentIntegrity,
    utf8ContentBytes,
} from './content-integrity.js';
import { rejectContextSelection as reject, resolveActiveContextSelection } from './context-selection-resolution.js';
import { ConversationValidationError } from './diagnostics.js';
import { resolveJsonPointer as jsonPointerLocation, pointerPrefix as prefix } from './json-pointer.js';
import { preflightJsonInput } from './json-preflight.js';
import { fingerprintJson } from './runtime.js';
import {
    ConversationSelectionRequestSchema,
    ConversationSelectionResultSchema,
    ConversationSliceResultSchema,
    SelectedBlockSchema,
} from './schemas/selection.js';
import { verifyDerivedBlockLineage } from './source-slice-lineage.js';
import type {
    Asset,
    BlockSubselection,
    ContentBlock,
    ConversationDocument,
    ConversationSelectionResult,
    ConversationSliceResult,
    ImageRegion,
    MediaSelectionRange,
    SelectedBlock,
    SelectedContextEntry,
    SelectionMediaEvidence,
} from './types.js';
import { diagnosticsFromZodError } from './validation.js';

function validateTextRange(length: number, start: number, end: number): void {
    if (start >= end) reject('Text range must be nonempty and ordered');
    if (end > length) reject('Text range exceeds source code-point length');
}
function compareOrder(first: readonly number[], second: readonly number[]): number {
    for (let index = 0; index < Math.min(first.length, second.length); index += 1)
        if (first[index] !== second[index]) return first[index] - second[index];
    return first.length - second.length;
}
function imageInPixels(region: ImageRegion, asset: Asset): ImageRegion {
    if (region.coordinate_space === 'pixels') return region;
    if (asset.media?.width === undefined || asset.media.height === undefined)
        reject('Cross-coordinate image refinement requires retained dimensions');
    return {
        ...region,
        coordinate_space: 'pixels',
        x: region.x * asset.media.width,
        y: region.y * asset.media.height,
        width: region.width * asset.media.width,
        height: region.height * asset.media.height,
    };
}
function imageContains(outer: ImageRegion, inner: ImageRegion): boolean {
    return (
        inner.x >= outer.x &&
        inner.y >= outer.y &&
        inner.x + inner.width <= outer.x + outer.width &&
        inner.y + inner.height <= outer.y + outer.height
    );
}
function validateMediaRange(block: ContentBlock, range: MediaSelectionRange, asset: Asset): 'declared' | 'unknown' {
    if (range.type === 'image_region') {
        if (block.type !== 'image') reject('Image range requires an image block');
        if (!Number.isFinite(range.x + range.width) || !Number.isFinite(range.y + range.height))
            reject('Image extent overflows finite geometry');
        if (range.coordinate_space === 'normalized' && (range.x + range.width > 1 || range.y + range.height > 1))
            reject('Normalized image range exceeds unit bounds');
        if (
            range.coordinate_space === 'pixels' &&
            ((asset.media?.width !== undefined && range.x + range.width > asset.media.width) ||
                (asset.media?.height !== undefined && range.y + range.height > asset.media.height))
        )
            reject('Image range exceeds retained dimensions');
        if (block.selection) {
            const sameSpace = range.coordinate_space === block.selection.coordinate_space;
            const outer = sameSpace ? block.selection : imageInPixels(block.selection, asset);
            const inner = sameSpace ? range : imageInPixels(range, asset);
            if (!imageContains(outer, inner)) reject('Image refinement exceeds the selected source region');
        }
        return asset.media?.width !== undefined && asset.media.height !== undefined ? 'declared' : 'unknown';
    }
    if (range.type === 'page_range') {
        if (block.type !== 'document') reject('Page range requires a document block');
        if (range.from_page > range.through_page) reject('Page range is reversed');
        if (asset.media?.page_count !== undefined && range.through_page > asset.media.page_count)
            reject('Page range exceeds retained page count');
        if (
            block.selection &&
            (range.from_page < block.selection.from_page || range.through_page > block.selection.through_page)
        )
            reject('Page refinement exceeds the selected source range');
        return asset.media?.page_count === undefined ? 'unknown' : 'declared';
    }
    if (block.type !== 'audio' && block.type !== 'video') reject('Time range requires an audio or video block');
    if (range.start_seconds >= range.end_seconds) reject('Time range must be nonempty and ordered');
    if (asset.media?.duration_seconds !== undefined && range.end_seconds > asset.media.duration_seconds)
        reject('Time range exceeds retained duration');
    if (
        block.selection &&
        (range.start_seconds < block.selection.start_seconds || range.end_seconds > block.selection.end_seconds)
    )
        reject('Time refinement exceeds the selected source range');
    return asset.media?.duration_seconds === undefined ? 'unknown' : 'declared';
}
function mediaOverlap(first: MediaSelectionRange, second: MediaSelectionRange, asset: Asset): boolean {
    if (first.type === 'image_region' && second.type === 'image_region') {
        const sameSpace = first.coordinate_space === second.coordinate_space;
        const a = sameSpace ? first : imageInPixels(first, asset),
            b = sameSpace ? second : imageInPixels(second, asset);
        return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
    }
    if (first.type === 'page_range' && second.type === 'page_range')
        return first.from_page <= second.through_page && second.from_page <= first.through_page;
    if (first.type === 'time_range' && second.type === 'time_range')
        return first.start_seconds < second.end_seconds && second.start_seconds < first.end_seconds;
    reject('Same-block media range kinds conflict');
}
function mediaOrder(range: MediaSelectionRange, asset: Asset, mixedImageSpace: boolean): number[] {
    if (range.type === 'page_range') return [range.from_page, range.through_page];
    if (range.type === 'time_range') return [range.start_seconds, range.end_seconds];
    const image = mixedImageSpace ? imageInPixels(range, asset) : range;
    return [image.y, image.x, image.height, image.width];
}
async function mediaEvidence(asset: Asset, extent: 'declared' | 'unknown'): Promise<SelectionMediaEvidence> {
    const sourceMetadataFingerprint = await fingerprintAssetSelectionMetadata(asset);
    let integrity: ContentIntegrity | undefined;
    if (asset.storage.type === 'inline_text') {
        let bytes: Uint8Array | undefined;
        try {
            bytes = utf8ContentBytes(asset.storage.text);
        } catch (error) {
            // Only encoding failure is unverified; actual hash/crypto failures must propagate.
            if (!(error instanceof TypeError)) throw error;
        }
        if (bytes !== undefined) integrity = await hashContentBytes(bytes);
    } else integrity = await inlineAssetContentIntegrity(asset.storage);
    const basis =
        asset.storage.type === 'inline_base64'
            ? ('stored_base64_bytes' as const)
            : asset.storage.type === 'inline_json'
              ? ('canonical_json_utf8' as const)
              : ('stored_text_utf8' as const);
    const declared = {
        ...(asset.content_hash === undefined ? {} : { declared_content_hash: asset.content_hash }),
        ...(asset.byte_length === undefined ? {} : { declared_byte_length: asset.byte_length }),
    };
    const content =
        integrity === undefined
            ? {
                  status: 'unverified' as const,
                  reason:
                      asset.storage.type === 'external'
                          ? ('external_content' as const)
                          : ('unencodable_inline_text' as const),
                  ...declared,
              }
            : (asset.content_hash !== undefined && asset.content_hash !== integrity.content_hash) ||
                (asset.byte_length !== undefined && asset.byte_length !== integrity.byte_length)
              ? {
                    status: 'mismatch' as const,
                    basis,
                    computed_content_hash: integrity.content_hash,
                    computed_byte_length: integrity.byte_length,
                    ...declared,
                }
              : { status: 'verified_inline' as const, basis, ...integrity };
    return {
        asset_id: asset.id,
        source_metadata_fingerprint: sourceMetadataFingerprint,
        extent,
        content,
        mutation_ready: false,
    };
}

/** Read-only whole/subrange selection. It does not expand cuts, mutate the document or infer readiness. */
export async function resolveConversationSelection(
    sourceInput: ConversationDocument,
    requestInput: unknown,
): Promise<ConversationSelectionResult> {
    try {
        const preflight = preflightJsonInput(requestInput);
        if (!preflight.success)
            throw new ConversationValidationError('Selection failed JSON preflight', preflight.diagnostics);
        const parsed = ConversationSelectionRequestSchema.safeParse(requestInput);
        if (!parsed.success)
            throw new ConversationValidationError(
                'Selection failed schema validation',
                diagnosticsFromZodError(parsed.error),
            );
        const request = parsed.data,
            document = await verifyDerivedBlockLineage(sourceInput);
        if (
            request.conversation.conversation_id !== document.id ||
            request.conversation.revision !== document.revision ||
            request.expected_context_revision !== document.context.revision
        )
            reject('Selection snapshot revision conflict');
        const { matchedEntries } = resolveActiveContextSelection(document, request.selector);
        const refinements = new Map<string, Map<string, BlockSubselection[]>>();
        const selectedBlocks = new Map(
            matchedEntries.map(({ entry, blocks }) => [entry.id, new Set(blocks.map((block) => block.id))]),
        );
        for (const refinement of request.selector.subselections ?? []) {
            if (!selectedBlocks.get(refinement.entry_id)?.has(refinement.block_id))
                reject('Subselection target was not selected by source and filters', 'REFERENCE_NOT_FOUND');
            const byBlock = refinements.get(refinement.entry_id) ?? new Map<string, BlockSubselection[]>();
            const group = byBlock.get(refinement.block_id) ?? [];
            if (group.length >= 256) reject('A block permits at most 256 disjoint refinements');
            group.push(refinement);
            byBlock.set(refinement.block_id, group);
            refinements.set(refinement.entry_id, byBlock);
        }
        const identity = { conversation_id: document.id, revision: document.revision };
        if (!matchedEntries.length)
            return ConversationSelectionResultSchema.parse({
                kind: 'no_match',
                conversation: identity,
                context_revision: document.context.revision,
                source_fingerprint: await fingerprintJson({ document, selector: request.selector }),
                diagnostics: [],
            });
        const entries: SelectedContextEntry[] = [];
        const jsonOrder = new WeakMap<object, ReadonlyMap<string, number>>();
        const assetEvidence = new Map<string, Promise<SelectionMediaEvidence>>();
        for (const { entry, blocks } of matchedEntries) {
            const selected: SelectedBlock[] = [];
            for (const block of blocks) {
                const blockFingerprint = await fingerprintJson(block);
                const group = refinements.get(entry.id)?.get(block.id);
                if (!group) {
                    selected.push({
                        kind: 'whole',
                        block_id: block.id,
                        block_type: block.type,
                        block_fingerprint: blockFingerprint,
                    });
                    continue;
                }
                const normalized: {
                    result: SelectedBlock;
                    order: number[];
                    pointer?: string[];
                    media?: MediaSelectionRange;
                }[] = [];
                const common = { block_id: block.id, block_fingerprint: blockFingerprint };
                let textLength = 0;
                if (block.type === 'text') for (const _point of block.text) textLength += 1;
                const imageSpaces = new Set(
                    group.flatMap((item) =>
                        item.kind === 'media_range' && item.range.type === 'image_region'
                            ? [item.range.coordinate_space]
                            : [],
                    ),
                );
                const mixedSpace = imageSpaces.size > 1;
                for (const refinement of group) {
                    if (refinement.expected_block_fingerprint !== blockFingerprint)
                        reject('Subselection block fingerprint conflict');
                    if (refinement.kind === 'text_range') {
                        if (block.type !== 'text') reject('Text range requires a text block');
                        validateTextRange(
                            textLength,
                            refinement.range.start_code_point,
                            refinement.range.end_code_point,
                        );
                        normalized.push({
                            result: { ...common, kind: 'text_range', block_type: 'text', range: refinement.range },
                            order: [refinement.range.start_code_point, refinement.range.end_code_point],
                        });
                    } else if (refinement.kind === 'json_pointer') {
                        if (block.type !== 'json')
                            reject('JSON Pointer requires a JSON block; opaque replay cannot be decoded');
                        const location = jsonPointerLocation(block.value, refinement.pointer, jsonOrder);
                        normalized.push({
                            result: {
                                ...common,
                                kind: 'json_pointer',
                                block_type: 'json',
                                pointer: refinement.pointer,
                            },
                            ...location,
                            pointer: location.segments,
                        });
                    } else {
                        if (
                            block.type !== 'image' &&
                            block.type !== 'document' &&
                            block.type !== 'audio' &&
                            block.type !== 'video'
                        )
                            reject('Media range requires a media block');
                        const asset = document.assets[block.asset_id];
                        const extent = validateMediaRange(block, refinement.range, asset);
                        let evidencePromise = assetEvidence.get(asset.id);
                        if (!evidencePromise) {
                            evidencePromise = mediaEvidence(asset, extent);
                            assetEvidence.set(asset.id, evidencePromise);
                        }
                        const evidence = { ...(await evidencePromise), extent };
                        if (evidence.source_metadata_fingerprint !== refinement.expected_asset_metadata_fingerprint)
                            reject('Subselection asset metadata fingerprint conflict');
                        normalized.push({
                            result: SelectedBlockSchema.parse({
                                ...common,
                                kind: 'media_range',
                                block_type: block.type,
                                range: refinement.range,
                                evidence,
                            }),
                            order: mediaOrder(refinement.range, asset, mixedSpace),
                            media: refinement.range,
                        });
                    }
                }
                normalized.sort((first, second) => compareOrder(first.order, second.order));
                for (let index = 0; index < normalized.length; index += 1)
                    for (let prior = 0; prior < index; prior += 1) {
                        const a = normalized[prior],
                            b = normalized[index];
                        if (a.pointer && b.pointer) {
                            if (prefix(a.pointer, b.pointer) || prefix(b.pointer, a.pointer))
                                reject('JSON Pointer selections overlap', 'CONTEXT_SELECTION_OVERLAP');
                        } else if (a.media && b.media && 'asset_id' in block) {
                            if (mediaOverlap(a.media, b.media, document.assets[block.asset_id]))
                                reject('Media selections overlap', 'CONTEXT_SELECTION_OVERLAP');
                        } else if (
                            a.result.kind === 'text_range' &&
                            b.result.kind === 'text_range' &&
                            a.result.range.end_code_point > b.result.range.start_code_point
                        )
                            reject('Text ranges overlap', 'CONTEXT_SELECTION_OVERLAP');
                    }
                selected.push(...normalized.map((item) => item.result));
            }
            entries.push({ entry, blocks: selected });
        }
        const selection = {
            conversation: identity,
            context_revision: document.context.revision,
            access: 'read_only',
            entries,
        };
        return ConversationSelectionResultSchema.parse({
            kind: 'selected',
            selection: { ...selection, source_fingerprint: await fingerprintJson({ document, selection }) },
            diagnostics: [],
        });
    } catch (error) {
        if (error instanceof ConversationValidationError)
            return ConversationSelectionResultSchema.parse({ kind: 'rejected', diagnostics: error.diagnostics });
        throw error;
    }
}

/** A pinned reference view over the same resolver; it is not a rewritten or provider-ready document. */
export async function sliceConversation(
    source: ConversationDocument,
    request: unknown,
): Promise<ConversationSliceResult> {
    const result = await resolveConversationSelection(source, request);
    return ConversationSliceResultSchema.parse(
        result.kind === 'selected'
            ? { kind: 'selected', view: { kind: 'context_view', selection: result.selection }, diagnostics: [] }
            : result,
    );
}
