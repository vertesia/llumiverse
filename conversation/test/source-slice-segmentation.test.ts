import { describe, expect, it } from 'vitest';
import { fingerprintAssetSelectionMetadata } from '../src/asset-selection-integrity.js';
import { ConversationDocumentSchema, fingerprintJson, resolveConversationSelection } from '../src/index.js';
import { segmentOwnedSourceBlock } from '../src/source-slice-segmentation.js';
import type { ContentBlock, ConversationDocument, SourceBlockSlice } from '../src/types.js';
import { emptyDocument, RECORDED_AT, textBlock, userTurn } from './fixtures.js';

function documentFor(block: ContentBlock): ConversationDocument {
    const document = emptyDocument();
    const turns = [{ ...userTurn('source'), blocks: [block] }];
    document.context.entries = [{ id: 'entry', type: 'source_turn', turn_id: 'source' }];
    return ConversationDocumentSchema.parse({ ...document, turns });
}
async function base(document: ConversationDocument, block: ContentBlock): Promise<Omit<SourceBlockSlice, 'selection'>> {
    return {
        source: { conversation_id: document.id, revision: document.revision },
        turn_id: 'source',
        block_id: block.id,
        block_fingerprint: await fingerprintJson(block),
    };
}

describe('bounded source slice segmentation', () => {
    it('keeps unmatched plain text and emoji in source order with disjoint code-point lineage', async () => {
        const block = textBlock('text', 'A😀BC😎D'),
            document = documentFor(block),
            source = await base(document, block);
        const segments = segmentOwnedSourceBlock(
            block,
            source,
            [
                {
                    kind: 'text_range',
                    block_type: 'text',
                    block_id: block.id,
                    block_fingerprint: source.block_fingerprint,
                    range: { start_code_point: 1, end_code_point: 2 },
                },
                {
                    kind: 'text_range',
                    block_type: 'text',
                    block_id: block.id,
                    block_fingerprint: source.block_fingerprint,
                    range: { start_code_point: 4, end_code_point: 5 },
                },
            ],
            { nodes: 0 },
        );
        expect(segments.map((segment) => (segment.block.type === 'text' ? segment.block.text : ''))).toEqual([
            'A',
            '😀',
            'BC',
            '😎',
            'D',
        ]);
        expect(segments.map((segment) => segment.selected)).toEqual([false, true, false, true, false]);
        expect(segments.map((segment) => segment.source.selection)).toEqual([
            { kind: 'text_range', range: { start_code_point: 0, end_code_point: 1 } },
            { kind: 'text_range', range: { start_code_point: 1, end_code_point: 2 } },
            { kind: 'text_range', range: { start_code_point: 2, end_code_point: 4 } },
            { kind: 'text_range', range: { start_code_point: 4, end_code_point: 5 } },
            { kind: 'text_range', range: { start_code_point: 5, end_code_point: 6 } },
        ]);
    });
    it('rejects markdown partial mutation without inventing a safe-looking boundary validator', async () => {
        const block = { ...textBlock('markdown', '**bold**'), format: 'markdown' as const },
            document = documentFor(block),
            source = await base(document, block);
        const selected = await resolveConversationSelection(document, {
            conversation: source.source,
            expected_context_revision: 0,
            selector: {
                source: { kind: 'all' },
                subselections: [
                    {
                        entry_id: 'entry',
                        block_id: block.id,
                        expected_block_fingerprint: source.block_fingerprint,
                        kind: 'text_range',
                        range: { start_code_point: 2, end_code_point: 6 },
                    },
                ],
            },
        });
        if (selected.kind !== 'selected') throw new Error('Read-only markdown inspection must remain valid');
        expect(() =>
            segmentOwnedSourceBlock(block, source, selected.selection.entries[0].blocks, { nodes: 0 }),
        ).toThrow('registered format boundary validator');
    });
    it('retains JSON array positions and escaped keys in the explicit complement mapping', async () => {
        const block: ContentBlock = { id: 'json', type: 'json', value: [{ 'a/b': null }, [], { '~': 4 }] };
        const document = documentFor(block),
            source = await base(document, block);
        const selected = await resolveConversationSelection(document, {
            conversation: source.source,
            expected_context_revision: 0,
            selector: {
                source: { kind: 'all' },
                subselections: [
                    {
                        entry_id: 'entry',
                        block_id: block.id,
                        expected_block_fingerprint: source.block_fingerprint,
                        kind: 'json_pointer',
                        pointer: '/0/a~1b',
                    },
                ],
            },
        });
        if (selected.kind !== 'selected') throw new Error('Fixture selection failed');
        const segments = segmentOwnedSourceBlock(block, source, selected.selection.entries[0].blocks, { nodes: 0 });
        expect(segments.map((segment) => (segment.block.type === 'json' ? segment.block.value : undefined))).toEqual([
            null,
            [{}, [], { '~': 4 }],
        ]);
        expect(segments[1].source.selection).toEqual({
            kind: 'json_region',
            region: { pointer: '', excluded_pointers: ['/0/a~1b'] },
        });
        expect(segments[1].inverse?.arrays).toEqual(
            expect.arrayContaining([
                { target_pointer: '', source_pointer: '', runs: [{ target_start: 0, source_start: 0, length: 3 }] },
            ]),
        );
    });
    it('forms media references using verified original bytes and a bounded source, without cropping claims', async () => {
        const block: ContentBlock = {
            id: 'image',
            type: 'image',
            asset_id: 'asset',
            selection: { type: 'image_region', coordinate_space: 'normalized', x: 0, y: 0, width: 1, height: 1 },
        };
        const document = documentFor(block),
            source = await base(document, block);
        document.assets.asset = {
            id: 'asset',
            kind: 'image',
            mime_type: 'image/png',
            storage: { type: 'inline_base64', data: 'YWJj' },
            provenance: { type: 'received', source_turn_id: 'source' },
            created_at: RECORDED_AT,
        };
        const selected = await resolveConversationSelection(document, {
            conversation: source.source,
            expected_context_revision: 0,
            selector: {
                source: { kind: 'all' },
                subselections: [
                    {
                        entry_id: 'entry',
                        block_id: block.id,
                        expected_block_fingerprint: source.block_fingerprint,
                        expected_asset_metadata_fingerprint: await fingerprintAssetSelectionMetadata(
                            document.assets.asset,
                        ),
                        kind: 'media_range',
                        range: {
                            type: 'image_region',
                            coordinate_space: 'normalized',
                            x: 0.25,
                            y: 0.25,
                            width: 0.5,
                            height: 0.5,
                        },
                    },
                ],
            },
        });
        if (selected.kind !== 'selected') throw new Error('Fixture selection failed');
        const evidence = selected.selection.entries[0].blocks[0];
        if (evidence.kind !== 'media_range') throw new Error('Missing media evidence');
        expect(evidence.evidence.extent).toBe('unknown');
        expect(evidence.evidence.mutation_ready).toBe(false);
        const segments = segmentOwnedSourceBlock(block, source, [evidence], { nodes: 0 });
        expect(segments).toHaveLength(5);
        expect(segments.filter((segment) => segment.selected)).toHaveLength(1);
        expect(
            segments.every(
                (segment) =>
                    segment.transform === 'media_reference' &&
                    'asset_id' in segment.block &&
                    segment.block.asset_id === 'asset',
            ),
        ).toBe(true);
        expect(document.assets.asset.storage).toEqual({ type: 'inline_base64', data: 'YWJj' });
        expect(() =>
            segmentOwnedSourceBlock({ ...block, selection: undefined }, source, [evidence], { nodes: 0 }),
        ).toThrow('bounded original');
    });
});
