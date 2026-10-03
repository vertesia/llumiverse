import { describe, expect, it } from 'vitest';
import {
    applyConversationEdit,
    fingerprintJson,
    planConversationEdit,
    resolveConversationSelection,
} from '../src/index.js';
import { resolveSourceSliceCoverage, sourceSlicesOverlap } from '../src/source-slice-coverage.js';
import { verifyDerivedBlockLineage } from '../src/source-slice-lineage.js';
import { inverseJsonPointer, projectJsonSourceRegion } from '../src/source-slice-projection.js';
import type {
    ConversationDocument,
    ConversationEditPlanInput,
    ConversationReplacementTurn,
    SourceBlockSlice,
} from '../src/types.js';
import { emptyDocument, textBlock, userTurn } from './fixtures.js';

async function acceptedReplacement(): Promise<ConversationDocument> {
    const document = emptyDocument();
    document.turns = [
        { ...userTurn('source', 'source:block'), blocks: [textBlock('source:block'), textBlock('unselected:block')] },
    ];
    document.context.entries = [{ id: 'source:entry', type: 'source_turn', turn_id: 'source' }];
    const selected = await resolveConversationSelection(document, {
        conversation: { conversation_id: document.id, revision: document.revision },
        expected_context_revision: document.context.revision,
        selector: { source: { kind: 'all' }, filters: { block_ids: ['source:block'] } },
    });
    if (selected.kind !== 'selected') throw new Error('Fixture selection failed');
    const slice: SourceBlockSlice = {
        source: selected.selection.conversation,
        turn_id: 'source',
        block_id: 'source:block',
        block_fingerprint: await fingerprintJson(document.turns[0].blocks[0]),
        selection: { kind: 'whole' },
    };
    const replacement: ConversationReplacementTurn = {
        ...userTurn('derived', 'derived:block'),
        authority: 'ordinary',
        status: 'completed',
        blocks: [textBlock('derived:block')],
        timestamps: { recorded_at: '2026-10-02T00:00:00.000Z' },
        provenance: {
            type: 'derived',
            derivation_id: 'edit:derived',
            source_turn_ids: ['source'],
            source_block_ids: ['source:block'],
            source_hash: selected.selection.source_fingerprint,
            block_lineage: {
                version: 1,
                groups: [
                    {
                        transform: 'authored_replacement',
                        fidelity: 'semantic',
                        target_block_ids: ['derived:block'],
                        source_slices: [slice],
                    },
                ],
            },
        },
    };
    const input: ConversationEditPlanInput = {
        version: 1,
        operation_id: 'edit:derived',
        conversation: selected.selection.conversation,
        expected_context_revision: document.context.revision,
        recorded_at: '2026-10-02T00:00:00.000Z',
        command: {
            kind: 'replace',
            selection: selected.selection,
            replacement_turn: replacement,
            fidelity: 'semantic',
            placement: { mode: 'first_selected', causal_order: 'contiguous' },
        },
    };
    const plan = await planConversationEdit(document, input);
    return (
        await applyConversationEdit(document, {
            ...input,
            expected_source_fingerprint: plan.operation.source_fingerprint,
        })
    ).document;
}

describe('precise source slice lineage groundwork', () => {
    it('preserves explicit null and empty containers, removes array positions without null placeholders', () => {
        const source = ['removed', null, { 'a/b': [1, 2, 3], '~': {} }, []];
        const projected = projectJsonSourceRegion(source, { pointer: '', excluded_pointers: ['/0', '/2/a~1b/1'] });
        expect(projected.value).toEqual([null, { 'a/b': [1, 3], '~': {} }, []]);
        expect(inverseJsonPointer(projected.inverse, '/1/a~1b/1')).toBe('/2/a~1b/2');
        expect(inverseJsonPointer(projected.inverse, '/1/~0')).toBe('/2/~0');
        const second = projectJsonSourceRegion(projected.value, { pointer: '', excluded_pointers: ['/0'] });
        const intermediate = inverseJsonPointer(second.inverse, '/0/a~1b/1');
        expect(inverseJsonPointer(projected.inverse, intermediate)).toBe('/2/a~1b/2');
        expect(second.value).toEqual([{ 'a/b': [1, 3], '~': {} }, []]);
    });
    it('distinguishes absent keys, rejects ancestor/descendant exclusion overlap and canonical array shifts', () => {
        expect(
            projectJsonSourceRegion({ first: null, second: [] }, { pointer: '', excluded_pointers: ['/first'] }).value,
        ).toEqual({ second: [] });
        expect(() =>
            projectJsonSourceRegion(
                { first: { second: null } },
                { pointer: '', excluded_pointers: ['/first', '/first/second'] },
            ),
        ).toThrow('overlap');
        expect(() => projectJsonSourceRegion([null], { pointer: '/01', excluded_pointers: [] })).toThrow('unavailable');
        expect(() =>
            projectJsonSourceRegion({ first: null }, { pointer: '', excluded_pointers: ['/missing'] }),
        ).toThrow('unavailable');
    });
    it('rejects inherited metadata and accessors before projection', () => {
        let calls = 0;
        const input = {
            get value() {
                calls++;
                return 'secret';
            },
        };
        expect(() => projectJsonSourceRegion(input, { pointer: '', excluded_pointers: [] })).toThrow('preflight');
        expect(calls).toBe(0);
        expect(() =>
            projectJsonSourceRegion(Object.create({ inherited: 'hidden' }), { pointer: '', excluded_pointers: [] }),
        ).toThrow('preflight');
    });
    it('verifies retained source hashes and target receipt fingerprints after JSON restoration', async () => {
        const document = await acceptedReplacement();
        const restored = JSON.parse(JSON.stringify(document));
        expect(await verifyDerivedBlockLineage(restored)).toEqual(document);
        const source = document.turns.find((turn) => turn.id === 'source');
        if (source === undefined) throw new Error('Missing source');
        source.blocks[0] = textBlock('source:block', 'forged source');
        await expect(verifyDerivedBlockLineage(document)).rejects.toThrow('content hash');
    });
    it('rejects accepted target mutations, shifted source revisions, missing groups', async () => {
        const original = await acceptedReplacement();
        const target = structuredClone(original);
        const derived = target.turns.find((turn) => turn.id === 'derived');
        if (derived?.provenance.type !== 'derived' || derived.provenance.block_lineage === undefined)
            throw new Error('Missing lineage');
        derived.blocks = [textBlock('derived:block', 'different target')];
        await expect(verifyDerivedBlockLineage(target)).rejects.toThrow('target content');
        derived.blocks = original.turns.find((turn) => turn.id === 'derived')?.blocks ?? [];
        derived.provenance.block_lineage.groups[0].source_slices[0].source.revision++;
        await expect(verifyDerivedBlockLineage(target)).rejects.toThrow();
        const missing = structuredClone(original),
            omitted = missing.turns.find((turn) => turn.id === 'derived');
        if (omitted?.provenance.type !== 'derived' || omitted.provenance.block_lineage === undefined)
            throw new Error('Missing lineage');
        omitted.provenance.block_lineage.groups[0].target_block_ids = ['unknown'];
        await expect(verifyDerivedBlockLineage(missing)).rejects.toMatchObject({
            diagnostics: expect.arrayContaining([
                expect.objectContaining({
                    code: 'DERIVED_PROVENANCE_MISMATCH',
                    message: expect.stringContaining('lineage group'),
                }),
            ]),
        });
    });
    it('owns the exact document before an asynchronous hash and returns that verified snapshot', async () => {
        const original = await acceptedReplacement();
        const input = structuredClone(original);
        const pending = verifyDerivedBlockLineage(input);
        const source = input.turns.find((turn) => turn.id === 'source');
        if (source === undefined) throw new Error('Missing source');
        source.blocks = [textBlock('source:block', 'caller mutation after first await')];
        expect(await pending).toEqual(original);
    });
    it('expands a legacy multi-block group once and rejects an unsupported partial cut', () => {
        const document = emptyDocument();
        const original = userTurn('original');
        original.blocks = [textBlock('a'), textBlock('b')];
        const derived = {
            ...userTurn('legacy'),
            blocks: [textBlock('x'), textBlock('y')],
            provenance: {
                type: 'derived' as const,
                derivation_id: 'historical',
                source_turn_ids: ['original'],
                source_block_ids: ['a', 'b'],
                source_hash: 'legacy:hash',
            },
        };
        document.turns = [original, derived];
        const slice = (block_id: string): SourceBlockSlice => ({
            source: { conversation_id: document.id, revision: 0 },
            turn_id: 'legacy',
            block_id,
            block_fingerprint: 'unverified',
            selection: { kind: 'whole' },
        });
        const coverage = resolveSourceSliceCoverage(document, [slice('x'), slice('y')]);
        expect(coverage.map((item) => item.block_id)).toEqual(['a', 'b']);
        expect(() => resolveSourceSliceCoverage(document, [slice('x')])).toThrow('indivisible');
    });
    it('does not treat adjacent text ranges or JSON ancestor complements as overlap', () => {
        const base: SourceBlockSlice = {
            source: { conversation_id: 'conversation', revision: 0 },
            turn_id: 'source',
            block_id: 'block',
            block_fingerprint: 'content:hash',
            selection: { kind: 'whole' },
        };
        expect(
            sourceSlicesOverlap(
                { ...base, selection: { kind: 'text_range', range: { start_code_point: 0, end_code_point: 1 } } },
                { ...base, selection: { kind: 'text_range', range: { start_code_point: 1, end_code_point: 2 } } },
            ),
        ).toBe(false);
        expect(
            sourceSlicesOverlap(
                { ...base, selection: { kind: 'json_region', region: { pointer: '', excluded_pointers: ['/a'] } } },
                { ...base, selection: { kind: 'json_region', region: { pointer: '/a', excluded_pointers: [] } } },
            ),
        ).toBe(false);
    });
});
