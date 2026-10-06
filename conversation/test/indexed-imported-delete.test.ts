import { readFile } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import {
    appendConversationRecords,
    canonicalJsonContentBytes,
    createConversationDocument,
    createUserTurn,
    fingerprintJson,
    getPagedRecord,
    hashContentBytes,
    type IndexedConversationRecordStore,
    loadIndexedProjectedTurn,
    loadIndexedToolCallSelection,
    putPagedRecord,
    stageIndexedConversationDelete,
    stageIndexedConversationSnapshot,
    stageIndexedRecordBatch,
} from '../src/index.js';
import { PagedRecordRefSchema } from '../src/paged-record-index.js';
import { ConversationTurnSchema } from '../src/schemas/content.js';
import { ExecutionReceiptSchema, ImportedGenerationSchema, OperationReceiptSchema } from '../src/schemas/execution.js';
import { IndexedConversationRootSchema, IndexedConversationTurnHeaderSchema } from '../src/schemas/indexed-head.js';

async function fixture(options: { executor?: 'application' | 'provider'; imported_generation?: boolean } = {}) {
    const captured = z
        .strictObject({
            root: IndexedConversationRootSchema,
            locator: PagedRecordRefSchema,
            pages: z.record(z.string(), z.string()),
            records: z.record(z.string(), z.string()),
            provenance: z.strictObject({
                capture_sha256: z.literal('f05ba731ce7dd797548e5628330c03c9a53e42c464c2cebd4254a430e258aa3e'),
                historical_provider_commit: z.literal('fffdaef'),
                case: z.literal('imported_closed'),
            }),
        })
        .parse(
            JSON.parse(
                await readFile(new URL('./fixtures/indexed-imported-closed-fffdaef.json', import.meta.url), 'utf8'),
            ),
        );
    const pages = new Map<string, Uint8Array>(
        Object.entries(captured.pages).map(([key, value]) => [key, Uint8Array.from(Buffer.from(value, 'base64'))]),
    );
    const records = new Map<string, Uint8Array>(
        Object.entries(captured.records).map(([key, value]) => [key, Uint8Array.from(Buffer.from(value, 'base64'))]),
    );
    let writes = 0;
    const reads: string[] = [];
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Captured original page missing');
            reads.push(ref.content_hash);
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            expect((await hashContentBytes(bytes)).content_hash).toBe(ref.content_hash);
            pages.set(ref.content_hash, Uint8Array.from(bytes));
            writes++;
        },
        async readRecord(ref) {
            const bytes = records.get(`${ref.kind}:${ref.content_hash}`);
            if (!bytes) throw new Error('Captured original record missing');
            reads.push(`${ref.kind}:${ref.id}`);
            return Uint8Array.from(bytes);
        },
        async writeRecord(ref, bytes) {
            expect((await hashContentBytes(bytes)).content_hash).toBe(ref.content_hash);
            records.set(`${ref.kind}:${ref.content_hash}`, Uint8Array.from(bytes));
            writes++;
        },
    };
    const call = await loadIndexedProjectedTurn(store, captured.root, 'imported:call:turn');
    const result = await loadIndexedProjectedTurn(store, captured.root, 'imported:result:turn');
    const operationRef = await getPagedRecord(
        store,
        captured.root.directories.operation_receipts,
        'append:historical:imported-closed',
    );
    const receiptRef = await getPagedRecord(store, captured.root.directories.execution_receipts, 'imported:execution');
    if (operationRef?.storage !== 'record' || receiptRef?.storage !== 'record')
        throw new Error('Original accepted records missing');
    const originalOperation = OperationReceiptSchema.parse(
        JSON.parse(new TextDecoder().decode(await store.readRecord(operationRef))),
    );
    let receipt = ExecutionReceiptSchema.parse(
        JSON.parse(new TextDecoder().decode(await store.readRecord(receiptRef))),
    );
    let turns = [call, result].map((turn) =>
        ConversationTurnSchema.parse({ ...turn.header, blocks: turn.selected_blocks }),
    );
    expect(await fingerprintJson({ turns, execution_receipts: [receipt] })).toBe(originalOperation.payload_fingerprint);
    if (options.executor) {
        receipt = ExecutionReceiptSchema.parse({ ...receipt, executor: options.executor });
        turns = turns.map((turn) =>
            ConversationTurnSchema.parse({
                ...turn,
                blocks: turn.blocks.map((block) =>
                    block.type === 'tool_call' ? { ...block, executor: options.executor } : block,
                ),
            }),
        );
    }
    const generation = options.imported_generation
        ? ImportedGenerationSchema.parse({
              id: 'imported:call:generation',
              record_source: 'imported',
              status: 'completed',
              timestamps: { recorded_at: originalOperation.recorded_at },
              source: {
                  conversation_id: captured.root.source.conversation_id,
                  revision: originalOperation.base_revision,
              },
          })
        : undefined;
    if (generation) {
        const first = turns[0];
        if (first.kind !== 'agent' || first.provenance.type !== 'imported')
            throw new Error('Imported call turn absent');
        turns[0] = ConversationTurnSchema.parse({
            ...first,
            generation_id: generation.id,
            provenance: { type: 'imported', source: first.provenance.source },
        });
    }
    const batch = {
        turns,
        execution_receipts: [receipt],
        ...(generation === undefined ? {} : { generations: [generation] }),
    };
    const variant = options.executor !== undefined || options.imported_generation === true;
    const operationId = variant
        ? `append:imported:${options.executor ?? 'provider'}:${generation ? 'generation' : 'no-generation'}`
        : originalOperation.id;
    const materialized = appendConversationRecords(
        createConversationDocument({ id: captured.root.source.conversation_id, created_at: captured.root.created_at }),
        batch,
        {
            operation_id: operationId,
            expected_revision: originalOperation.base_revision,
            recorded_at: originalOperation.recorded_at,
            payload_fingerprint: variant ? await fingerprintJson(batch) : originalOperation.payload_fingerprint,
        },
    ).document;
    const operation = materialized.operation_receipts[operationId];
    if (!operation) throw new Error('Actual imported append omitted its original receipt');
    if (!variant) expect(operation).toEqual(originalOperation);
    const snapshot = await stageIndexedConversationSnapshot(materialized, undefined, store);
    const originalCall = turns[0].blocks[0];
    if (originalCall?.type !== 'tool_call') throw new Error('Original imported call absent');
    const source = {
        conversation: snapshot.root.source,
        turn_id: call.header.id,
        block_id: originalCall.id,
        call_id: originalCall.call_id,
        call_fingerprint: await fingerprintJson(originalCall),
    };
    reads.length = 0;
    return {
        captured,
        materialized,
        generation,
        snapshot,
        store,
        operation,
        receipt,
        batch,
        source,
        reads,
        records,
        getWrites: () => writes,
    };
}

async function replaceRetainedRecord(
    f: Awaited<ReturnType<typeof fixture>>,
    family: 'generations' | 'turns' | 'generation_acceptances',
    id: string,
    value: unknown,
    marker = false,
) {
    const bytes = canonicalJsonContentBytes(value);
    const integrity = await hashContentBytes(bytes);
    const ref = {
        storage: 'record' as const,
        kind: family,
        id,
        content_hash: integrity.content_hash,
        size_bytes: integrity.byte_length,
    };
    if (!marker) await f.store.writeRecord(ref, bytes);
    const directory = await putPagedRecord(
        f.store,
        f.snapshot.root.directories[family],
        id,
        marker ? { storage: 'marker', kind: 'generation_acceptance', id: 'append:foreign' } : ref,
        'replace',
    );
    const root = IndexedConversationRootSchema.parse({
        ...f.snapshot.root,
        directories: { ...f.snapshot.root.directories, [family]: directory },
    });
    const rootBytes = canonicalJsonContentBytes(root);
    const hash = await hashContentBytes(rootBytes);
    const locator = { content_hash: hash.content_hash, size_bytes: hash.byte_length };
    await f.store.writeRecord(
        { storage: 'record', kind: 'root', id: root.source.conversation_id, ...locator },
        rootBytes,
    );
    return { root, locator };
}

describe('indexed imported terminal history deletion', () => {
    it.each(['together', 'result_then_call'] as const)(
        'removes genuine imported closed bodies %s without creating execution authority',
        async (order) => {
            const f = await fixture();
            expect(f.receipt.call_source).toBeUndefined();
            expect(f.batch.turns[0].provenance.type).toBe('imported');
            expect(f.batch.turns[1]).not.toHaveProperty('execution_id');
            await expect(loadIndexedToolCallSelection(f.store, f.snapshot.root, f.source)).rejects.toThrow(
                'accepted generated content',
            );
            const del = async (
                root: typeof f.snapshot.root,
                locator: typeof f.snapshot.locator,
                ids: string[],
                operationId: string,
            ) =>
                stageIndexedConversationDelete(
                    root,
                    {
                        operation_id: operationId,
                        source: root.source,
                        expected_source_root: locator,
                        recorded_at: root.updated_at,
                        dependency_policy: 'reject',
                        turn_ids: ids,
                    },
                    f.store,
                );
            const before = f.getWrites();
            await expect(
                del(f.snapshot.root, f.snapshot.locator, ['imported:call:turn'], 'delete:live-dependent'),
            ).rejects.toThrow('live dependent turn');
            expect(f.getWrites()).toBe(before);
            const first = await del(
                f.snapshot.root,
                f.snapshot.locator,
                order === 'together' ? ['imported:call:turn', 'imported:result:turn'] : ['imported:result:turn'],
                'delete:archival:first',
            );
            if (!first.locator) throw new Error('Fresh accepted delete lacks actual root locator');
            const completed =
                order === 'together'
                    ? first
                    : await del(first.root, first.locator, ['imported:call:turn'], 'delete:archival:call');
            expect(completed.root.live_turn_count).toBe(0);
            const operationRef = await getPagedRecord(
                f.store,
                completed.root.directories.operation_receipts,
                f.operation.id,
            );
            const executionRef = await getPagedRecord(
                f.store,
                completed.root.directories.execution_receipts,
                f.receipt.id,
            );
            if (operationRef?.storage !== 'record' || executionRef?.storage !== 'record')
                throw new Error('Retained original receipt missing');
            expect(
                OperationReceiptSchema.parse(
                    JSON.parse(new TextDecoder().decode(await f.store.readRecord(operationRef))),
                ),
            ).toEqual(f.operation);
            expect(
                ExecutionReceiptSchema.parse(
                    JSON.parse(new TextDecoder().decode(await f.store.readRecord(executionRef))),
                ),
            ).toEqual(f.receipt);
            expect(
                await getPagedRecord(f.store, completed.root.directories.tool_call_states, f.source.call_id),
            ).toEqual(await getPagedRecord(f.store, f.snapshot.root.directories.tool_call_states, f.source.call_id));
            expect(
                await getPagedRecord(f.store, completed.root.directories.identifiers, 'imported:result:nested'),
            ).toBeDefined();
            expect(
                await getPagedRecord(f.store, completed.root.directories.identifiers, f.source.call_id),
            ).toBeDefined();
            await expect(loadIndexedToolCallSelection(f.store, completed.root, f.source)).rejects.toThrow();
            expect(
                (await loadIndexedProjectedTurn(f.store, f.snapshot.root, f.source.turn_id)).selected_blocks,
            ).toEqual(f.batch.turns[0].blocks);
            const saved = f.getWrites();
            await expect(
                stageIndexedRecordBatch(
                    completed.root,
                    {
                        conversation_id: completed.root.source.conversation_id,
                        batch: f.batch,
                        options: {
                            operation_id: 'append:resurrect-imported',
                            expected_revision: completed.root.source.revision,
                            recorded_at: completed.root.updated_at,
                            payload_fingerprint: await fingerprintJson(f.batch),
                        },
                    },
                    f.store,
                ),
            ).rejects.toThrow();
            expect(f.getWrites()).toBe(saved);
            const reusedNestedId = createUserTurn({
                id: 'turn:new-received',
                authority: 'ordinary',
                status: 'completed',
                model_visibility: 'include',
                provenance: { type: 'received' },
                timestamps: { recorded_at: completed.root.updated_at },
                blocks: [
                    {
                        id: 'imported:result:nested',
                        type: 'text',
                        text: 'Must not reuse the deleted nested block ID.',
                        format: 'plain',
                    },
                ],
            });
            const freshBatch = { turns: [reusedNestedId] };
            await expect(
                stageIndexedRecordBatch(
                    completed.root,
                    {
                        conversation_id: completed.root.source.conversation_id,
                        batch: freshBatch,
                        options: {
                            operation_id: 'append:reuse-nested',
                            expected_revision: completed.root.source.revision,
                            recorded_at: completed.root.updated_at,
                            payload_fingerprint: await fingerprintJson(freshBatch),
                        },
                    },
                    f.store,
                ),
            ).rejects.toMatchObject({ code: 'record_conflict' });
            expect(f.getWrites()).toBe(saved);
            expect(f.reads.length).toBeLessThan(1024);
        },
    );

    it('rejects corrupted original execution bytes before deletion publication', async () => {
        const f = await fixture();
        const ref = await getPagedRecord(f.store, f.snapshot.root.directories.execution_receipts, f.receipt.id);
        if (ref?.storage !== 'record') throw new Error('Original receipt descriptor absent');
        f.records.set(
            `${ref.kind}:${ref.content_hash}`,
            canonicalJsonContentBytes({ ...f.receipt, result_fingerprint: 'sha256:tampered' }),
        );
        const writes = f.getWrites();
        await expect(
            stageIndexedConversationDelete(
                f.snapshot.root,
                {
                    operation_id: 'delete:tampered',
                    source: f.snapshot.root.source,
                    expected_source_root: f.snapshot.locator,
                    recorded_at: f.snapshot.root.updated_at,
                    dependency_policy: 'reject',
                    turn_ids: ['imported:call:turn', 'imported:result:turn'],
                },
                f.store,
            ),
        ).rejects.toThrow();
        expect(f.getWrites()).toBe(writes);
    });
    it.each([false, true])(
        'deletes accepted imported application facts with optional source absent and generation present=%s',
        async (withGeneration) => {
            const f = await fixture({ executor: 'application', imported_generation: withGeneration });
            expect(f.receipt.executor).toBe('application');
            expect(f.receipt.call_source).toBeUndefined();
            expect(f.batch.turns[0].provenance.type).toBe('imported');
            if (withGeneration) {
                expect(f.generation?.record_source).toBe('imported');
                expect(f.generation?.request_receipt).toBeUndefined();
                expect(f.operation.accepted_generation_ids).toEqual([f.generation?.id]);
            }
            await expect(loadIndexedToolCallSelection(f.store, f.snapshot.root, f.source)).rejects.toThrow(
                'accepted generated content',
            );
            const deleted = await stageIndexedConversationDelete(
                f.snapshot.root,
                {
                    operation_id: 'delete:imported-application',
                    source: f.snapshot.root.source,
                    expected_source_root: f.snapshot.locator,
                    recorded_at: f.snapshot.root.updated_at,
                    dependency_policy: 'reject',
                    turn_ids: ['imported:call:turn', 'imported:result:turn'],
                },
                f.store,
            );
            expect(deleted.applied).toBe(true);
            expect(deleted.root.live_turn_count).toBe(0);
            const ref = await getPagedRecord(f.store, deleted.root.directories.execution_receipts, f.receipt.id);
            if (ref?.storage !== 'record') throw new Error('Original optional-source receipt lost');
            expect(
                ExecutionReceiptSchema.parse(JSON.parse(new TextDecoder().decode(await f.store.readRecord(ref)))),
            ).toEqual(f.receipt);
        },
    );

    it.each(['source', 'acceptance'] as const)(
        'rejects imported generation %s corruption with authenticated bytes before deletion writes',
        async (kind) => {
            const f = await fixture({ imported_generation: true });
            if (!f.generation) throw new Error('Actual accepted imported generation absent');
            const corrupted =
                kind === 'source'
                    ? await replaceRetainedRecord(f, 'generations', f.generation.id, {
                          ...f.generation,
                          source: f.snapshot.root.source,
                      })
                    : await replaceRetainedRecord(f, 'generation_acceptances', f.generation.id, f.generation, true);
            const writes = f.getWrites();
            await expect(
                stageIndexedConversationDelete(
                    corrupted.root,
                    {
                        operation_id: 'delete:corrupt-imported-generation',
                        source: corrupted.root.source,
                        expected_source_root: corrupted.locator,
                        recorded_at: corrupted.root.updated_at,
                        dependency_policy: 'reject',
                        turn_ids: ['imported:call:turn', 'imported:result:turn'],
                    },
                    f.store,
                ),
            ).rejects.toThrow('accepted generation/request chain');
            expect(f.getWrites()).toBe(writes);
        },
    );

    it('rejects a conflicting present result execution ID at import and before indexed deletion', async () => {
        const f = await fixture();
        const result = f.batch.turns[1];
        if (result.kind !== 'tool') throw new Error('Actual imported result turn absent');
        const corruptBatch = {
            ...f.batch,
            turns: [f.batch.turns[0], { ...result, execution_id: 'execution:foreign' }],
        };
        expect(() =>
            appendConversationRecords(
                createConversationDocument({ id: f.materialized.id, created_at: f.materialized.created_at }),
                corruptBatch,
                {
                    operation_id: 'append:invalid-import',
                    expected_revision: 0,
                    recorded_at: f.operation.recorded_at,
                    payload_fingerprint: 'import:invalid-conflicting-execution',
                },
            ),
        ).toThrow();
        const projected = await loadIndexedProjectedTurn(f.store, f.snapshot.root, result.id);
        const descriptor = await getPagedRecord(f.store, f.snapshot.root.directories.turns, result.id);
        if (descriptor?.storage !== 'record') throw new Error('Actual result header descriptor absent');
        const header = JSON.parse(new TextDecoder().decode(await f.store.readRecord(descriptor)));
        const validated = IndexedConversationTurnHeaderSchema.parse(header);
        expect(validated.turn).toEqual(projected.header);
        const corrupted = await replaceRetainedRecord(f, 'turns', result.id, {
            ...validated,
            turn: { ...validated.turn, execution_id: 'execution:foreign' },
        });
        const writes = f.getWrites();
        await expect(
            stageIndexedConversationDelete(
                corrupted.root,
                {
                    operation_id: 'delete:conflicting-execution',
                    source: corrupted.root.source,
                    expected_source_root: corrupted.locator,
                    recorded_at: corrupted.root.updated_at,
                    dependency_policy: 'reject',
                    turn_ids: ['imported:call:turn', 'imported:result:turn'],
                },
                f.store,
            ),
        ).rejects.toThrow('another execution identity');
        expect(f.getWrites()).toBe(writes);
    });
});
