import { describe, expect, it } from 'vitest';
import {
    CheckpointSummaryForkCapacityError,
    createCheckpointSummaryFork,
    createCheckpointSummaryForkFromIndexedSelection,
} from '../src/checkpoint-summary-fork.js';
import { hashContentBytes } from '../src/content-integrity.js';
import {
    type IndexedConversationRecordStore,
    IndexedPresentationNominationConflict,
    loadIndexedCheckpointSummarySelectedContext,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingPhase,
    stageIndexedTextProcessingCompletion,
} from '../src/indexed-conversation.js';
import { buildIndexedTextExternalizationOutput } from '../src/indexed-processing-working-set.js';
import { appendConversationRecords } from '../src/runtime.js';
import { emptyDocument, RECORDED_AT, textBlock, userTurn } from './fixtures.js';
import { indexedTextClaimFixture } from './indexed-processing-fixture.js';

function memoryStore(): IndexedConversationRecordStore {
    const values = new Map<string, Uint8Array>();
    return {
        async read(ref) {
            const value = values.get(ref.content_hash);
            if (!value) throw new Error('Missing page');
            return value;
        },
        async write(bytes, ref) {
            values.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            const value = values.get(ref.content_hash);
            if (!value) throw new Error('Missing record');
            return value;
        },
        async writeRecord(ref, bytes) {
            expect((await hashContentBytes(bytes)).content_hash).toBe(ref.content_hash);
            values.set(ref.content_hash, Uint8Array.from(bytes));
        },
    };
}

describe('separate selected checkpoint summary fork', () => {
    it('requires genuine settled processing and renders selected replacements plus preserved structured remainder', async () => {
        const f = await indexedTextClaimFixture(0, 'mixed');
        await expect(
            loadIndexedCheckpointSummarySelectedContext(f.store, f.staged.root, f.staged.locator),
        ).rejects.toThrow(IndexedPresentationNominationConflict);
        const output = await buildIndexedTextExternalizationOutput(f.workspace);
        const persisted = await stageIndexedProcessingPhase(f.store, f.staged.root, f.staged.locator, {
            phase: 'output',
            value: output,
        });
        const completed = await stageIndexedTextProcessingCompletion(
            f.store,
            persisted.root,
            persisted.locator,
            f.workspace,
        );
        expect(completed.completion.status).toBe('applied');
        const selected = await loadIndexedCheckpointSummarySelectedContext(f.store, completed.root, completed.locator);
        expect(selected.completeness).toBe('active_processing_dependencies_verified');
        expect(selected.replacement_turns).toHaveLength(1);
        expect(Object.keys(selected.compaction_witnesses ?? {})).toHaveLength(1);
        const fork = await createCheckpointSummaryForkFromIndexedSelection(
            completed.root,
            selected,
            'summary:replacement',
        );
        const text = fork.turns[0].blocks.find((block) => block.type === 'text');
        if (text?.type !== 'text') throw new Error('Expected genuine fork transcript');
        expect(text.text).toContain('"preserve":[null,3,false]');
        expect(text.text).toContain('[external reference asset');
        expect(text.text).not.toContain('\noriginal\n');
        expect(fork.lineage).toEqual({ parents: [{ relation: 'fork', source: completed.root.source }] });
        const { processing_index_profile: _profile, ...unavailable } = completed.root;
        await expect(
            loadIndexedCheckpointSummarySelectedContext(f.store, unavailable, completed.locator),
        ).rejects.toThrow(IndexedPresentationNominationConflict);
    });

    it('constructs an empty separate fork without inventing an accepted response or content history', async () => {
        const source = emptyDocument('no-output-source');
        const store = memoryStore();
        const staged = await stageIndexedConversationSnapshot(source, undefined, store);
        const selected = await loadIndexedCheckpointSummarySelectedContext(store, staged.root, staged.locator);
        const fork = await createCheckpointSummaryForkFromIndexedSelection(staged.root, selected, 'summary:empty');
        expect(fork).toEqual(await createCheckpointSummaryFork(source, 'summary:empty'));
        expect(fork.generations).toEqual({});
        expect(fork.assets).toEqual({});
    });

    it('preserves exact context block selection, source revision, lineage and rendering without lifetime history', async () => {
        const turn = userTurn('selected');
        turn.blocks = [
            textBlock('omitted', 'Do not render this block'),
            textBlock('chosen', 'Chosen text'),
            { id: 'json', type: 'json', value: { greeting: 'hello' } },
        ];
        const cold = userTurn('cold');
        cold.model_visibility = 'exclude';
        const document = appendConversationRecords(
            emptyDocument('fork-source'),
            {
                turns: [cold, turn],
                context_entries: [
                    { id: 'selected-entry', type: 'source_turn', turn_id: turn.id, block_ids: ['chosen', 'json'] },
                ],
            },
            {
                expected_revision: 0,
                operation_id: 'accepted:source',
                payload_fingerprint: 'source',
                recorded_at: RECORDED_AT,
            },
        ).document;
        const store = memoryStore();
        const staged = await stageIndexedConversationSnapshot(document, undefined, store);
        const selection = await loadIndexedCheckpointSummarySelectedContext(store, staged.root, staged.locator);
        const actual = await createCheckpointSummaryForkFromIndexedSelection(staged.root, selection, 'summary:1');
        expect(actual).toEqual(await createCheckpointSummaryFork(document, 'summary:1'));
        expect(actual.id).not.toBe(document.id);
        expect(actual.lineage).toEqual({ parents: [{ relation: 'fork', source: staged.root.source }] });
        expect(actual.turns).toHaveLength(1);
        expect(JSON.stringify(actual.turns)).toContain('Chosen text');
        expect(JSON.stringify(actual.turns)).not.toContain('Do not render');
        expect(JSON.stringify(actual.turns)).not.toContain('cold-text');
        expect(actual.assets).toEqual({});
        expect(actual.generations).toEqual({});
        expect(actual.processing).toEqual(emptyDocument().processing);
        await expect(
            createCheckpointSummaryForkFromIndexedSelection(
                staged.root,
                { ...selection, source: { ...selection.source, revision: selection.source.revision + 1 } },
                'summary:1',
            ),
        ).rejects.toThrow('exact indexed source');
    });

    it('bounds encoded JSON escaping before constructing an oversized transcript or fork', async () => {
        const turn = userTurn('escaped');
        turn.blocks = [{ id: 'escaped-json', type: 'json', value: '\u0000'.repeat(2.5 * 1024 * 1024) }];
        const document = appendConversationRecords(
            emptyDocument('escaped-fork'),
            {
                turns: [turn],
                context_entries: [{ id: 'entry', type: 'source_turn', turn_id: turn.id }],
            },
            {
                expected_revision: 0,
                operation_id: 'escaped:source',
                payload_fingerprint: 'source',
                recorded_at: RECORDED_AT,
            },
        ).document;
        await expect(createCheckpointSummaryFork(document, 'summary:escaped')).rejects.toThrow(
            CheckpointSummaryForkCapacityError,
        );
    });
});
