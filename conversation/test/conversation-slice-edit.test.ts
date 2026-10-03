import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    applyConversationEdit,
    applyConversationSliceEdit,
    type ContentBlock,
    type ConversationDocument,
    ConversationInsertedTurnSchema,
    type ConversationSelection,
    type ConversationSliceEditCommand,
    type ConversationSliceEditRequest,
    ConversationSliceEditRequestSchema,
    fingerprintJson,
    parseConversationDocument,
    planConversationSliceEdit,
    resolveConversationSelection,
    verifyDerivedBlockLineage,
} from '../src/index.js';
import { emptyDocument, textBlock, userTurn } from './fixtures.js';

const at = '2026-10-02T00:00:00.000Z';
function source(text = 'a😀bcdef'): ConversationDocument {
    const doc = emptyDocument();
    doc.turns = [{ ...userTurn('original'), blocks: [textBlock('text', text), textBlock('unmatched', 'keep')] }];
    doc.context.entries = [{ id: 'original-entry', type: 'source_turn', turn_id: 'original' }];
    return parseConversationDocument(doc);
}
async function select(
    doc: ConversationDocument,
    blockId: string,
    start: number,
    end: number,
): Promise<ConversationSelection> {
    const block = doc.turns.flatMap<ContentBlock>((turn) => turn.blocks).find((block) => block.id === blockId);
    if (!block) throw new Error('fixture block');
    const result = await resolveConversationSelection(doc, {
        conversation: { conversation_id: doc.id, revision: doc.revision },
        expected_context_revision: doc.context.revision,
        selector: {
            source: { kind: 'all' },
            filters: { block_ids: [blockId] },
            subselections: [
                {
                    kind: 'text_range',
                    entry_id: doc.context.entries.find((entry) =>
                        doc.turns
                            .find((turn) => turn.id === entry.turn_id)
                            ?.blocks.some((block) => block.id === blockId),
                    )?.id,
                    block_id: blockId,
                    expected_block_fingerprint: await fingerprintJson(block),
                    range: { start_code_point: start, end_code_point: end },
                },
            ],
        },
    });
    if (result.kind !== 'selected') throw new Error(JSON.stringify(result));
    return result.selection;
}
async function request(
    doc: ConversationDocument,
    command: ConversationSliceEditCommand,
    id = 'slice',
): Promise<ConversationSliceEditRequest> {
    const input = {
        version: 2,
        operation_id: id,
        conversation: { conversation_id: doc.id, revision: doc.revision },
        expected_context_revision: doc.context.revision,
        recorded_at: at,
        command,
    };
    const plan = await planConversationSliceEdit(doc, input);
    return ConversationSliceEditRequestSchema.parse({
        ...input,
        expected_source_fingerprint: plan.operation.source_fingerprint,
    });
}
function replacement(id = 'replacement', operation_id = 'slice') {
    return ConversationInsertedTurnSchema.parse({
        ...userTurn(id),
        blocks: [textBlock(`${id}:block`, 'summary')],
        timestamps: { recorded_at: at },
        provenance: { type: 'inserted', operation_id },
    });
}
function activeText(doc: ConversationDocument): string[] {
    return doc.context.entries.flatMap((entry) => {
        const turn = doc.turns.find((turn) => turn.id === entry.turn_id);
        return (
            turn?.blocks
                .filter((block) => !entry.block_ids || entry.block_ids.includes(block.id))
                .map((block) => (block.type === 'text' ? block.text : block.type)) ?? []
        );
    });
}

describe('version2 precise source-slice edits', () => {
    it('replaces code-point slices, retains all unmatched bytes/order and exposes exact lineage in shared change', async () => {
        const doc = source(),
            selection = await select(doc, 'text', 1, 3);
        const input = await request(doc, {
            kind: 'replace',
            selection,
            replacement_turn: replacement(),
            fidelity: 'semantic',
            placement: { mode: 'first_selected', causal_order: 'contiguous' },
        });
        const result = await applyConversationEdit(doc, input);
        expect(activeText(result.document)).toEqual(['a', 'summary', 'cdef', 'keep']);
        expect(result.document.turns[0]).toEqual(doc.turns[0]);
        expect(result.change.operations[0]).toMatchObject({
            version: 2,
            kind: 'replace',
            source_slices: [
                {
                    turn_id: 'original',
                    block_id: 'text',
                    selection: { kind: 'text_range', range: { start_code_point: 1, end_code_point: 3 } },
                },
            ],
            fidelity: 'semantic',
        });
        const retry = await applyConversationSliceEdit(JSON.parse(JSON.stringify(result.document)), input);
        expect(retry.applied).toBe(false);
        expect(retry.change).toEqual(result.change);
        expect(retry.document).toEqual(result.document);
    });
    it('binds retry to original accepted effects after a later slice/edit/append and rejects changed ranges', async () => {
        const doc = source('abcdef'),
            input = await request(doc, {
                kind: 'protect',
                selection: await select(doc, 'text', 1, 3),
                protected: false,
            });
        const first = await applyConversationSliceEdit(doc, input);
        const tail = first.document.turns.find((turn) =>
            turn.blocks.some((block) => block.type === 'text' && block.text === 'def'),
        );
        if (!tail) throw new Error('fixture');
        const next = await request(
            first.document,
            { kind: 'protect', selection: await select(first.document, tail.blocks[0].id, 0, 1), protected: true },
            'next',
        );
        const later = await applyConversationSliceEdit(first.document, next);
        const appended = appendConversationRecords(
            later.document,
            { turns: [userTurn('later')] },
            {
                operation_id: 'append',
                payload_fingerprint: 'payload:append',
                recorded_at: at,
                expected_revision: later.document.revision,
            },
        ).document;
        const retry = await applyConversationSliceEdit(JSON.parse(JSON.stringify(appended)), input);
        expect(retry.change).toEqual(first.change);
        expect(retry.document.context).toEqual(appended.context);
        const conflicting = structuredClone(input);
        const selected = conflicting.command.selection.entries[0].blocks[0];
        if (selected.kind !== 'text_range') throw new Error('fixture');
        selected.range.end_code_point++;
        await expect(applyConversationSliceEdit(appended, conflicting)).rejects.toThrow('conflicts');
    });
    it('separates contextual unprotect from intrinsic instructions and preserves inherited remainder pins', async () => {
        const doc = source('abcdef');
        doc.context.protected_entry_ids = ['original-entry'];
        const input = await request(doc, {
            kind: 'protect',
            selection: await select(doc, 'text', 1, 3),
            protected: false,
        });
        const result = await applyConversationSliceEdit(doc, input);
        expect(activeText(result.document)).toEqual(['a', 'bc', 'def', 'keep']);
        expect(result.document.context.protected_entry_ids).toEqual([
            result.document.context.entries[0].id,
            result.document.context.entries[2].id,
            result.document.context.entries[3].id,
        ]);
        const elevated = source();
        elevated.turns[0].authority = 'developer';
        const selection = await select(elevated, 'text', 1, 3);
        await expect(
            planConversationSliceEdit(elevated, {
                ...input,
                expected_source_fingerprint: undefined,
                command: { kind: 'protect', selection, protected: false },
            }),
        ).rejects.toThrow();
    });
    it('rejects unsupported markdown mutation, forged source hashes and required-cache invalidation before publication', async () => {
        const doc = source();
        const input = await request(doc, {
            kind: 'protect',
            selection: await select(doc, 'text', 1, 3),
            protected: true,
        });
        doc.context.cache_intent = { mode: 'required', stable_through_entry_id: 'original-entry', namespace: 'cache' };
        await expect(applyConversationSliceEdit(doc, input)).rejects.toThrow('fingerprint');
        const cacheInput = await request(doc, {
            kind: 'protect',
            selection: await select(doc, 'text', 1, 3),
            protected: true,
        });
        await expect(applyConversationSliceEdit(doc, cacheInput)).rejects.toThrow('required cache');
        const markdown = source();
        const block = markdown.turns[0].blocks[0];
        if (block.type !== 'text') throw new Error('fixture');
        block.format = 'markdown';
        const selection = await select(markdown, 'text', 1, 3);
        await expect(request(markdown, { kind: 'protect', selection, protected: true })).rejects.toThrow(
            'registered format',
        );
        const good = await applyConversationSliceEdit(source(), input);
        const forged = structuredClone(good.document);
        const derived = forged.turns.find((turn) => turn.provenance.type === 'derived');
        if (derived?.provenance.type !== 'derived' || !derived.provenance.block_lineage) throw new Error('fixture');
        derived.provenance.block_lineage.groups[0].source_slices[0].block_fingerprint = 'forged';
        await expect(verifyDerivedBlockLineage(forged)).rejects.toThrow();
    });
    it('owns request and source before hash awaits and rejects cross-family operation reuse', async () => {
        const doc = source(),
            input = await request(doc, {
                kind: 'replace',
                selection: await select(doc, 'text', 1, 3),
                replacement_turn: replacement(),
                fidelity: 'heuristic',
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            });
        const pending = applyConversationSliceEdit(doc, input);
        input.command.selection.entries[0].blocks[0].block_fingerprint = 'mutated';
        doc.turns[0].blocks = [textBlock('text', 'mutated')];
        const result = await pending;
        expect(activeText(result.document)).toEqual(['a', 'summary', 'cdef', 'keep']);
        expect(() =>
            appendConversationRecords(
                result.document,
                { turns: [] },
                {
                    operation_id: 'slice',
                    payload_fingerprint: result.document.operation_receipts.slice.payload_fingerprint,
                    recorded_at: at,
                    expected_revision: result.document.revision,
                },
            ),
        ).toThrow();
    });
});

describe('restored JSON projection mutations', () => {
    async function jsonSelection(doc: ConversationDocument, blockId: string, pointer: string) {
        const entry = doc.context.entries.find((entry) =>
            doc.turns.find((turn) => turn.id === entry.turn_id)?.blocks.some((block) => block.id === blockId),
        );
        const block = doc.turns.flatMap<ContentBlock>((turn) => turn.blocks).find((block) => block.id === blockId);
        if (!entry || !block) throw new Error('fixture');
        const selected = await resolveConversationSelection(doc, {
            conversation: { conversation_id: doc.id, revision: doc.revision },
            expected_context_revision: doc.context.revision,
            selector: {
                source: { kind: 'all' },
                filters: { block_ids: [blockId] },
                subselections: [
                    {
                        kind: 'json_pointer',
                        entry_id: entry.id,
                        block_id: blockId,
                        expected_block_fingerprint: await fingerprintJson(block),
                        pointer,
                    },
                ],
            },
        });
        if (selected.kind !== 'selected') throw new Error(JSON.stringify(selected));
        return selected.selection;
    }
    it('retains null/empty containers and escaped-key/array inverse mappings across repeated projection and retry', async () => {
        const doc = source();
        doc.turns[0].blocks = [
            { id: 'json', type: 'json', value: { 'a/b': [{ k: null }, { k: {} }, { k: [] }], keep: null } },
        ];
        const firstInput = await request(doc, {
            kind: 'protect',
            selection: await jsonSelection(doc, 'json', '/a~1b/0'),
            protected: false,
        });
        const first = await applyConversationSliceEdit(doc, firstInput);
        const remainder = first.document.turns.find(
            (turn) =>
                turn.provenance.type === 'derived' &&
                turn.blocks[0].type === 'json' &&
                typeof turn.blocks[0].value === 'object' &&
                turn.blocks[0].value !== null &&
                'keep' in turn.blocks[0].value,
        );
        if (!remainder) throw new Error('fixture complement');
        const secondInput = await request(
            first.document,
            {
                kind: 'protect',
                selection: await jsonSelection(first.document, remainder.blocks[0].id, '/a~1b/0/k'),
                protected: false,
            },
            'second',
        );
        const second = await applyConversationSliceEdit(JSON.parse(JSON.stringify(first.document)), secondInput);
        const values = second.document.context.entries
            .map((entry) => second.document.turns.find((turn) => turn.id === entry.turn_id)?.blocks[0])
            .map((block) => (block?.type === 'json' ? block.value : undefined));
        expect(values).toEqual([{ k: null }, {}, { 'a/b': [{}, { k: [] }], keep: null }]);
        const verified = await verifyDerivedBlockLineage(JSON.parse(JSON.stringify(second.document)));
        expect(verified).toEqual(second.document);
        expect((await applyConversationSliceEdit(verified, firstInput)).change).toEqual(first.change);
        expect((await applyConversationSliceEdit(verified, secondInput)).change).toEqual(second.change);
        // Reintroducing the original full span after any precise projection still overlaps active coverage.
        expect(() =>
            appendConversationRecords(
                second.document,
                { context_entries: [{ id: 'reintroduced', type: 'source_turn', turn_id: 'original' }] },
                {
                    operation_id: 'reintroduce',
                    payload_fingerprint: 'source',
                    recorded_at: at,
                    expected_revision: second.document.revision,
                },
            ),
        ).toThrow();
    });
    it('rejects lineage cycles, missing dependency records and forged inverse array positions after JSON reload', async () => {
        const doc = source();
        doc.turns[0].blocks = [{ id: 'json', type: 'json', value: [null, {}, []] }];
        const input = await request(doc, {
            kind: 'protect',
            selection: await jsonSelection(doc, 'json', '/1'),
            protected: false,
        });
        const result = await applyConversationSliceEdit(doc, input);
        const forged = structuredClone(result.document),
            target = forged.turns.find(
                (turn) =>
                    turn.provenance.type === 'derived' &&
                    turn.provenance.block_lineage?.groups[0].transform === 'json_projection' &&
                    turn.provenance.block_lineage.groups[0].inverse.arrays.some((array) => array.runs.length > 0),
            );
        if (target?.provenance.type !== 'derived' || !target.provenance.block_lineage) throw new Error('fixture');
        const group = target.provenance.block_lineage.groups[0];
        if (group.transform !== 'json_projection') throw new Error('fixture');
        const array = group.inverse.arrays.find((array) => array.runs.length);
        if (!array) throw new Error('fixture');
        array.runs[0].source_start++;
        await expect(verifyDerivedBlockLineage(JSON.parse(JSON.stringify(forged)))).rejects.toThrow();
        const missing = structuredClone(result.document);
        missing.turns = missing.turns.filter((turn) => turn.id !== 'original');
        await expect(verifyDerivedBlockLineage(missing, { turn_ids: [target.id] })).rejects.toThrow();
        const cycle = structuredClone(result.document),
            cyclic = cycle.turns.find((turn) => turn.id === target.id);
        if (cyclic?.provenance.type !== 'derived' || !cyclic.provenance.block_lineage) throw new Error('fixture');
        const slice = cyclic.provenance.block_lineage.groups[0].source_slices[0];
        slice.turn_id = cyclic.id;
        slice.block_id = cyclic.blocks[0].id;
        cyclic.provenance.source_turn_ids = [cyclic.id];
        cyclic.provenance.source_block_ids = [cyclic.blocks[0].id];
        await expect(verifyDerivedBlockLineage(cycle)).rejects.toThrow();
    });
});

describe('typed original-media reference edits', () => {
    it('partitions an already bounded image reference with verified bytes, retains the same original asset and refuses unbounded/forged evidence', async () => {
        const doc = source();
        doc.turns[0].blocks = [
            {
                id: 'image',
                type: 'image',
                asset_id: 'original-asset',
                selection: { type: 'image_region', coordinate_space: 'normalized', x: 0, y: 0, width: 1, height: 1 },
            },
        ];
        doc.assets['original-asset'] = {
            id: 'original-asset',
            kind: 'image',
            mime_type: 'image/png',
            storage: { type: 'inline_base64', data: 'YWJj' },
            created_at: at,
            provenance: { type: 'received', source_turn_id: 'original' },
        };
        const assetMetadata = await import('../src/asset-selection-integrity.js');
        const selected = await resolveConversationSelection(doc, {
            conversation: { conversation_id: doc.id, revision: 0 },
            expected_context_revision: 0,
            selector: {
                source: { kind: 'all' },
                subselections: [
                    {
                        kind: 'media_range',
                        entry_id: 'original-entry',
                        block_id: 'image',
                        expected_block_fingerprint: await fingerprintJson(doc.turns[0].blocks[0]),
                        expected_asset_metadata_fingerprint: await assetMetadata.fingerprintAssetSelectionMetadata(
                            doc.assets['original-asset'],
                        ),
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
        if (selected.kind !== 'selected') throw new Error(JSON.stringify(selected));
        const evidence = selected.selection.entries[0].blocks[0];
        if (evidence.kind !== 'media_range') throw new Error('fixture');
        expect(evidence.evidence.extent).toBe('unknown');
        expect(evidence.evidence.mutation_ready).toBe(false);
        const input = await request(doc, { kind: 'protect', selection: selected.selection, protected: true });
        const result = await applyConversationSliceEdit(doc, input);
        expect(result.document.context.entries).toHaveLength(5);
        expect(result.document.assets).toEqual(doc.assets);
        expect(
            result.document.turns
                .slice(1)
                .every((turn) => turn.blocks[0].type === 'image' && turn.blocks[0].asset_id === 'original-asset'),
        ).toBe(true);
        const retry = await applyConversationSliceEdit(JSON.parse(JSON.stringify(result.document)), input);
        expect(retry.change).toEqual(result.change);
        const forged = structuredClone(result.document);
        forged.assets['original-asset'].storage = { type: 'inline_base64', data: 'YWJk' };
        await expect(verifyDerivedBlockLineage(forged)).rejects.toThrow('bytes');
        const unbounded = structuredClone(doc),
            original = unbounded.turns[0].blocks[0];
        if (original.type !== 'image') throw new Error('fixture');
        delete original.selection;
        const unboundedSelection = structuredClone(selected.selection);
        unboundedSelection.entries[0].blocks[0].block_fingerprint = await fingerprintJson(original);
        const { source_fingerprint: _fingerprint, ...rest } = unboundedSelection;
        unboundedSelection.source_fingerprint = await fingerprintJson({ document: unbounded, selection: rest });
        await expect(
            request(unbounded, { kind: 'protect', selection: unboundedSelection, protected: true }),
        ).rejects.toThrow('bounded original');
    });
});

describe('partial dependency and cache boundaries', () => {
    it('permits read-only inspection but rejects cutting a retained replay dependency through the mutation API', async () => {
        const doc = source();
        doc.turns.push({
            ...userTurn('replay'),
            kind: 'agent',
            provenance: { type: 'received' },
            blocks: [
                {
                    id: 'replay-block',
                    type: 'native_replay',
                    adapter: 'adapter',
                    protocol: 'protocol',
                    compatibility_scope: { provider: 'provider', protocol: 'protocol', adapter_version: '1' },
                    payload: {},
                    dependencies: { turn_ids: [], block_ids: ['text'], call_ids: [], request_ids: [] },
                },
            ],
        });
        doc.context.entries.push({ id: 'replay-entry', type: 'source_turn', turn_id: 'replay' });
        const selection = await select(doc, 'text', 1, 3);
        expect(selection.access).toBe('read_only');
        await expect(
            request(doc, {
                kind: 'replace',
                selection,
                replacement_turn: replacement(),
                fidelity: 'heuristic',
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            }),
        ).rejects.toThrow('replay dependency');
    });
    it('retains auto/off namespace while dropping only invalidated boundaries after actual partial mutation', async () => {
        for (const mode of ['auto', 'off'] as const) {
            const doc = source();
            doc.context.cache_intent = {
                mode,
                namespace: 'pinned:namespace',
                stable_through_entry_id: 'original-entry',
            };
            const input = await request(doc, {
                kind: 'protect',
                selection: await select(doc, 'text', 1, 3),
                protected: true,
            });
            const result = await applyConversationSliceEdit(doc, input);
            expect(result.document.context.cache_intent).toEqual({ mode, namespace: 'pinned:namespace' });
            expect(parseConversationDocument(JSON.parse(JSON.stringify(result.document))).context.cache_intent).toEqual(
                { mode, namespace: 'pinned:namespace' },
            );
        }
    });
});

describe('bounded selected lineage verification', () => {
    it('hash-verifies only declared roots and their exact transitive dependencies, never blessing unrelated cold records', async () => {
        const doc = source();
        const first = await applyConversationSliceEdit(
            doc,
            await request(doc, { kind: 'protect', selection: await select(doc, 'text', 1, 3), protected: false }),
        );
        const second = await applyConversationSliceEdit(
            first.document,
            await request(
                first.document,
                { kind: 'protect', selection: await select(first.document, 'unmatched', 0, 1), protected: false },
                'cold-edit',
            ),
        );
        const root = first.document.turns[1],
            damaged = structuredClone(second.document),
            cold = damaged.turns.find(
                (turn) => turn.provenance.type === 'derived' && turn.provenance.derivation_id === 'cold-edit',
            );
        if (cold?.provenance.type !== 'derived' || !cold.provenance.block_lineage) throw new Error('fixture');
        cold.provenance.block_lineage.groups[0].source_slices[0].block_fingerprint = 'unverified:cold';
        const scoped = await verifyDerivedBlockLineage(damaged, { turn_ids: [root.id] });
        expect(scoped).toEqual(damaged); // Only the explicit roots are authenticated by this call.
        await expect(verifyDerivedBlockLineage(damaged)).rejects.toThrow();
        await expect(verifyDerivedBlockLineage(damaged, { turn_ids: ['missing-root'] })).rejects.toThrow('unavailable');
        const sourceForgery = structuredClone(damaged),
            original = sourceForgery.turns[0].blocks[0];
        if (original.type !== 'text') throw new Error('fixture');
        original.text = 'forged';
        await expect(verifyDerivedBlockLineage(sourceForgery, { turn_ids: [root.id] })).rejects.toThrow();
    });
});
