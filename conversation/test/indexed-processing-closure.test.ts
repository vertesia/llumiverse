import { describe, expect, it } from 'vitest';
import { hashContentBytes } from '../src/content-integrity.js';
import { fingerprintJson } from '../src/identity.js';
import {
    type IndexedConversationRecordStore,
    loadIndexedProcessingClosureWitness,
    recoverIndexedProcessingClosureRoot,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingClosureWitness,
    stageIndexedRecordBatch,
} from '../src/indexed-conversation.js';
import { IndexedProcessingClosureCommandSchema } from '../src/schemas/indexed-processing-closure.js';
import { emptyDocument, userTurn } from './fixtures.js';
import { indexedTextClaimFixture } from './indexed-processing-fixture.js';

const at = '2026-09-11T00:00:00.000Z';

function closureStore() {
    const records = new Map<string, Uint8Array>();
    const pages = new Map<string, Uint8Array>();
    const reads = { pages: 0, records: 0, coldTurns: 0 };
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            reads.pages++;
            const body = pages.get(ref.content_hash);
            if (!body) throw new Error('Missing closure page');
            return Uint8Array.from(body);
        },
        async write(body, ref) {
            pages.set(ref.content_hash, Uint8Array.from(body));
        },
        async readRecord(ref) {
            reads.records++;
            if (ref.kind === 'turns') reads.coldTurns++;
            const body = records.get(ref.content_hash);
            if (!body) throw new Error('Missing closure record');
            return Uint8Array.from(body);
        },
        async writeRecord(ref, body) {
            if ((await hashContentBytes(body)).content_hash !== ref.content_hash)
                throw new Error('Invalid closure record write');
            records.set(ref.content_hash, Uint8Array.from(body));
        },
    };
    return { store, pages, records, reads };
}

function command(operationId: string, revision: number) {
    return {
        operation_id: operationId,
        expected_revision: revision,
        recorded_at: at,
        binding: { kind: 'test-closure', activation: operationId, terminal: 'accepted' },
    };
}

describe('generic indexed processing closure witnesses', () => {
    it('recovers an old closing root after later epochs without writes, current-head substitution or cold turns', async () => {
        const fixture = closureStore();
        let current = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:epochs'),
            undefined,
            fixture.store,
        );
        const first = await stageIndexedProcessingClosureWitness(
            fixture.store,
            current.root,
            current.locator,
            command('closure:0', current.root.source.revision),
        );
        current = first;
        for (let index = 1; index < 128; index++) {
            current = await stageIndexedProcessingClosureWitness(
                fixture.store,
                current.root,
                current.locator,
                command(`closure:${index}`, current.root.source.revision),
            );
        }
        const counts = { pages: fixture.pages.size, records: fixture.records.size };
        fixture.reads.pages = fixture.reads.records = 0;
        const recovered = await recoverIndexedProcessingClosureRoot(fixture.store, current.root, 'closure:0');
        expect(recovered?.root).toEqual(first.root);
        expect(recovered?.locator).toEqual(first.locator);
        expect(recovered?.witness).toEqual(first.witness);
        expect(recovered?.root.source).not.toEqual(current.root.source);
        expect(fixture.reads.pages).toBeLessThan(64);
        expect(fixture.reads.records).toBeLessThan(8);
        expect(fixture.reads.coldTurns).toBe(0);
        expect({ pages: fixture.pages.size, records: fixture.records.size }).toEqual(counts);
        const retry = await stageIndexedProcessingClosureWitness(
            fixture.store,
            current.root,
            current.locator,
            command('closure:0', 0),
        );
        expect(retry.applied).toBe(false);
        expect(retry.witness).toEqual(first.witness);
        expect(retry.root).toEqual(current.root);
        await expect(
            stageIndexedProcessingClosureWitness(fixture.store, current.root, current.locator, {
                ...command('closure:0', 0),
                binding: { terminal: 'foreign' },
            }),
        ).rejects.toThrow('original binding');
    });

    it('does not publish a closure while a genuine required job remains unresolved', async () => {
        const fixture = await indexedTextClaimFixture();
        const count = fixture.records.size;
        await expect(
            stageIndexedProcessingClosureWitness(
                fixture.store,
                fixture.staged.root,
                fixture.staged.locator,
                command('closure:pending', fixture.staged.root.source.revision),
            ),
        ).rejects.toThrow('unresolved');
        expect(fixture.records.size).toBe(count);
        expect(
            await loadIndexedProcessingClosureWitness(fixture.store, fixture.staged.root, 'closure:pending'),
        ).toBeUndefined();
    });

    it('rejects identifier collisions, foreign predecessor locators and stale competing publication', async () => {
        const fixture = closureStore();
        const document = emptyDocument('conversation:collision');
        document.turns.push(userTurn('turn:existing', 'original'));
        const initial = await stageIndexedConversationSnapshot(document, undefined, fixture.store);
        await expect(
            stageIndexedProcessingClosureWitness(
                fixture.store,
                initial.root,
                initial.locator,
                command('turn:existing', initial.root.source.revision),
            ),
        ).rejects.toThrow();
        expect(await loadIndexedProcessingClosureWitness(fixture.store, initial.root, 'turn:existing')).toBeUndefined();
        const closed = await stageIndexedProcessingClosureWitness(
            fixture.store,
            initial.root,
            initial.locator,
            command('closure:winner', initial.root.source.revision),
        );
        await expect(
            stageIndexedProcessingClosureWitness(
                fixture.store,
                closed.root,
                initial.locator,
                command('closure:winner', initial.root.source.revision),
            ),
        ).rejects.toThrow('predecessor locator');
        await expect(
            stageIndexedProcessingClosureWitness(
                fixture.store,
                closed.root,
                closed.locator,
                command('closure:foreign', initial.root.source.revision),
            ),
        ).rejects.toThrow('exact current source');
        const foreign = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:foreign'),
            undefined,
            fixture.store,
        );
        await expect(
            stageIndexedProcessingClosureWitness(
                fixture.store,
                initial.root,
                foreign.locator,
                command('closure:foreign', initial.root.source.revision),
            ),
        ).rejects.toThrow('predecessor locator');
    });

    it('retains abandoned ACK metadata without changing logical content, and rejects stale equal-revision root publication', async () => {
        const fixture = closureStore();
        const initial = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:retention'),
            undefined,
            fixture.store,
        );
        const originalBytes = Uint8Array.from(fixture.records.get(initial.locator.content_hash) ?? []);
        const draft = {
            ...command('closure:abandoned:0', initial.root.source.revision),
            publication: 'retention' as const,
            binding: {
                kind: 'test-closure',
                terminal: 'abandoned',
                original_source: initial.root.source,
                original_root: initial.locator,
            },
        };
        const first = await stageIndexedProcessingClosureWitness(fixture.store, initial.root, initial.locator, draft);
        expect(first.root.source).toEqual(initial.root.source);
        expect(first.root.updated_at).toBe(initial.root.updated_at);
        expect(first.locator).not.toEqual(initial.locator);
        const { processing_records: _records, identifiers: _identifiers, ...unchanged } = first.root.directories;
        const { processing_records: _oldRecords, identifiers: _oldIdentifiers, ...original } = initial.root.directories;
        expect(unchanged).toEqual(original);
        expect({ ...first.root, directories: initial.root.directories }).toEqual(initial.root);
        let current = first;
        for (let index = 1; index < 128; index++) {
            current = await stageIndexedProcessingClosureWitness(fixture.store, current.root, current.locator, {
                ...command(`closure:abandoned:${index}`, initial.root.source.revision),
                publication: 'retention',
                binding: { terminal: 'abandoned', epoch: index },
            });
        }
        expect(current.root.source).toEqual(initial.root.source);
        fixture.reads.pages = fixture.reads.records = 0;
        const count = { pages: fixture.pages.size, records: fixture.records.size };
        const historical = await recoverIndexedProcessingClosureRoot(fixture.store, current.root, draft.operation_id);
        expect(historical?.root).toEqual(first.root);
        expect(historical?.witness.predecessor).toEqual({ source: initial.root.source, root: initial.locator });
        expect(historical?.witness.result_revision).toBe(initial.root.source.revision);
        expect(fixture.reads.pages).toBeLessThan(64);
        expect(fixture.reads.records).toBeLessThan(8);
        expect({ pages: fixture.pages.size, records: fixture.records.size }).toEqual(count);
        expect(fixture.records.get(initial.locator.content_hash)).toEqual(originalBytes);
        await expect(
            stageIndexedProcessingClosureWitness(fixture.store, current.root, initial.locator, draft),
        ).rejects.toThrow('predecessor locator');
        const { publication: _publication, ...wrongPhase } = draft;
        await expect(
            stageIndexedProcessingClosureWitness(fixture.store, current.root, current.locator, wrongPhase),
        ).rejects.toThrow('original binding');
        const turn = userTurn('turn:after-abandon', 'Accepted after abandonment');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:after-abandon', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const append = await stageIndexedRecordBatch(
            current.root,
            {
                conversation_id: current.root.source.conversation_id,
                batch,
                options: {
                    operation_id: 'append:after-abandon',
                    expected_revision: current.root.source.revision,
                    recorded_at: at,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            fixture.store,
        );
        expect(append.root.source.revision).toBe(initial.root.source.revision + 1);
        expect(await loadIndexedProcessingClosureWitness(fixture.store, append.root, draft.operation_id)).toEqual(
            first.witness,
        );
        const retry = await stageIndexedRecordBatch(
            append.root,
            {
                conversation_id: current.root.source.conversation_id,
                batch,
                options: {
                    operation_id: 'append:after-abandon',
                    expected_revision: current.root.source.revision,
                    recorded_at: at,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            fixture.store,
        );
        expect(retry.applied).toBe(false);
        expect(retry.receipt).toEqual(append.receipt);
    });

    it('enforces its explicit bounded opaque metadata before accepting a host binding', () => {
        const draft = command('closure:metadata', 0);
        const binding = { padding: '' };
        // ASCII JSON byte length is deterministic; boundaries include the object envelope.
        binding.padding = 'x'.repeat(32 * 1024 - JSON.stringify(binding).length);
        expect(IndexedProcessingClosureCommandSchema.parse({ ...draft, binding }).binding).toEqual(binding);
        expect(() =>
            IndexedProcessingClosureCommandSchema.parse({ ...draft, binding: { padding: `${binding.padding}x` } }),
        ).toThrow('metadata bound');
    });
});
