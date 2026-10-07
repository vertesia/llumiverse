import { describe, expect, it } from 'vitest';
import {
    applyConversationSliceEdit,
    type ConversationDocument,
    ConversationInsertedTurnSchema,
    type ConversationSelection,
    ConversationSliceEditRequestSchema,
    fingerprintJson,
    planConversationSliceEdit,
    resolveConversationSelection,
} from '../src/index.js';
import { emptyDocument, textBlock, userTurn } from './fixtures.js';

const at = '2026-10-02T00:00:00.000Z';
async function replace(
    doc: ConversationDocument,
    selection: ConversationSelection,
    causal_order: 'contiguous' | 'explicit_disjoint',
    operation_id = 'replace',
) {
    const command = {
        kind: 'replace',
        selection,
        replacement_turn: ConversationInsertedTurnSchema.parse({
            ...userTurn(`replacement:${operation_id}`),
            blocks: [textBlock(`block:${operation_id}`)],
            provenance: { type: 'inserted', operation_id },
            timestamps: { recorded_at: at },
        }),
        fidelity: 'semantic',
        placement: { mode: 'first_selected', causal_order },
    };
    const input = {
        version: 2,
        operation_id,
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
async function select(
    doc: ConversationDocument,
    block_ids: string[],
    pointers?: string[],
): Promise<ConversationSelection> {
    const result = await resolveConversationSelection(doc, {
        conversation: { conversation_id: doc.id, revision: doc.revision },
        expected_context_revision: doc.context.revision,
        selector: {
            source: { kind: 'all' },
            filters: { block_ids },
            ...(pointers
                ? {
                      subselections: await Promise.all(
                          pointers.map(async (pointer) => ({
                              kind: 'json_pointer',
                              entry_id: doc.context.entries[0].id,
                              block_id: 'json',
                              expected_block_fingerprint: await fingerprintJson(doc.turns[0].blocks[0]),
                              pointer,
                          })),
                      ),
                  }
                : {}),
        },
    });
    if (result.kind !== 'selected') throw new Error(JSON.stringify(result));
    return result.selection;
}
function source(): ConversationDocument {
    const doc = emptyDocument();
    doc.turns = [userTurn('first', 'a'), userTurn('middle', 'b'), userTurn('third', 'c')];
    doc.context.entries = doc.turns.map((turn) => ({ id: `entry:${turn.id}`, type: 'source_turn', turn_id: turn.id }));
    return doc;
}
describe('retained source topology for exact slice retries', () => {
    it.each([
        [['a', 'c'], 'explicit_disjoint', [0, 2]],
        [['a', 'b'], 'contiguous', [0, 1]],
    ] as const)('retains original whole-entry gaps for %j placement %s', async (blocks, placement, positions) => {
        const doc = source(),
            selection = await select(doc, [...blocks]),
            input = await replace(doc, selection, placement);
        const first = await applyConversationSliceEdit(doc, input);
        expect(first.change.operations[0]).toMatchObject({ version: 2, source_entry_positions: positions });
        expect((await applyConversationSliceEdit(JSON.parse(JSON.stringify(first.document)), input)).change).toEqual(
            first.change,
        );
        const laterInput = await replace(
            first.document,
            await select(first.document, ['block:replace']),
            'contiguous',
            'later',
        );
        const later = await applyConversationSliceEdit(first.document, laterInput);
        expect((await applyConversationSliceEdit(JSON.parse(JSON.stringify(later.document)), input)).change).toEqual(
            first.change,
        );
        const corrupted = structuredClone(later.document),
            detail = corrupted.operation_receipts.replace.conversation_edit;
        if (detail?.version !== 2) throw new Error('fixture');
        detail.source_entry_positions[0]++;
        await expect(applyConversationSliceEdit(corrupted, input)).rejects.toThrow('topology');
    });
    it('rejects false contiguous fresh placement and preserves JSON subtree gaps instead of packed order', async () => {
        const doc = source(),
            selection = await select(doc, ['a', 'c']);
        await expect(replace(doc, selection, 'contiguous')).rejects.toThrow('disjoint');
        const json = emptyDocument();
        json.turns = [{ ...userTurn('json-turn'), blocks: [{ id: 'json', type: 'json', value: ['a', 'b', 'c'] }] }];
        json.context.entries = [{ id: 'entry', type: 'source_turn', turn_id: 'json-turn' }];
        const gap = await select(json, ['json'], ['/0', '/2']);
        await expect(replace(json, gap, 'contiguous')).rejects.toThrow('disjoint');
        const input = await replace(json, gap, 'explicit_disjoint');
        const applied = await applyConversationSliceEdit(json, input);
        expect(applied.document.turns[0]).toEqual(json.turns[0]);
        expect(
            applied.document.turns.some((turn) =>
                turn.blocks.some((block) => block.type === 'json' && JSON.stringify(block.value) === '["b"]'),
            ),
        ).toBe(true);
        expect((await applyConversationSliceEdit(JSON.parse(JSON.stringify(applied.document)), input)).change).toEqual(
            applied.change,
        );
        const adjacent = await select(json, ['json'], ['/0', '/1']);
        const contiguous = await replace(json, adjacent, 'contiguous');
        const next = await applyConversationSliceEdit(json, contiguous);
        expect(
            (await applyConversationSliceEdit(JSON.parse(JSON.stringify(next.document)), contiguous)).change,
        ).toEqual(next.change);
        await expect(replace(json, adjacent, 'explicit_disjoint')).rejects.toThrow('disjoint');
    });
});
