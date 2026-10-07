import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { hashContentBytes } from '../src/content-integrity.js';
import { fingerprintJson } from '../src/identity.js';
import {
    type IndexedConversationRecordStore,
    IndexedRecordAppendConflict,
    indexedOrderedKey,
    loadIndexedAcceptedOutputHistoryPage,
    loadIndexedActiveContext,
    loadIndexedPendingProcessingJobs,
    loadRecord,
    stageIndexedRecordBatch,
    stageRecord,
} from '../src/indexed-conversation.js';
import { loadIndexedUpgradeAcceptedSource } from '../src/indexed-upgrade-accepted.js';
import { INDEXED_CONVERSATION_UPGRADE_PROFILE } from '../src/indexed-upgrade-constants.js';
import { finishIndexedConversationUpgrade } from '../src/indexed-upgrade-finish.js';
import {
    createIndexedUpgradeStepStore,
    INDEXED_UPGRADE_ACTIVE_LIMITS,
    INDEXED_UPGRADE_STEP_LIMITS,
} from '../src/indexed-upgrade-io.js';
import { beginIndexedConversationUpgrade, readIndexedUpgradeProgress } from '../src/indexed-upgrade-progress.js';
import { advanceIndexedConversationUpgrade } from '../src/indexed-upgrade-step.js';
import { getPagedRecord, PagedRecordRefSchema, putPagedRecord, removePagedRecord } from '../src/paged-record-index.js';
import { ContentBlockSchema } from '../src/schemas/content.js';
import { ExecutionReceiptSchema, GenerationSchema } from '../src/schemas/execution.js';
import {
    IndexedConversationProcessingHeaderSchema,
    IndexedConversationRootSchema,
    IndexedConversationTurnLinkSchema,
} from '../src/schemas/indexed-head.js';
import type { IndexedConversationUpgradeCommand } from '../src/schemas/indexed-upgrade.js';

const artifact = z
    .object({
        cases: z.record(z.string(), z.object({ root: IndexedConversationRootSchema, locator: PagedRecordRefSchema })),
        pages: z.record(z.string(), z.string()),
        records: z.record(z.string(), z.string()),
    })
    .parse(
        JSON.parse(
            readFileSync(
                new URL('./fixtures/indexed-upgrade-fffdaef/historical-fffdaef-artifacts.json', import.meta.url),
                'utf8',
            ),
        ),
    );
const at = '2026-10-06T00:00:00.000Z';
// Decode the pinned immutable byte cohort once; each fixture still owns fresh byte arrays and
// mutable maps, so corruption/recovery tests cannot change another case's authentic source.
const originalPages = Object.entries(artifact.pages).map(
    ([key, value]) => [key, Uint8Array.from(Buffer.from(value, 'base64'))] as const,
);
const originalRecords = Object.entries(artifact.records).map(
    ([key, value]) => [key, Uint8Array.from(Buffer.from(value, 'base64'))] as const,
);

function fixture(name: string) {
    const original = artifact.cases[name];
    if (!original) throw new Error('Historical case missing');
    const pages = new Map(originalPages.map(([key, bytes]) => [key, Uint8Array.from(bytes)]));
    const records = new Map(originalRecords.map(([key, bytes]) => [key, Uint8Array.from(bytes)]));
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Historical page missing');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            const bytes = records.get(`${ref.kind}:${ref.content_hash}`);
            if (!bytes) throw new Error('Historical record missing');
            return Uint8Array.from(bytes);
        },
        async writeRecord(ref, bytes) {
            records.set(`${ref.kind}:${ref.content_hash}`, Uint8Array.from(bytes));
        },
    };
    const command: IndexedConversationUpgradeCommand = {
        version: 1,
        profile: INDEXED_CONVERSATION_UPGRADE_PROFILE,
        operation_id: `upgrade:${name}`,
        source: original.root.source,
        predecessor_root: original.locator,
        recorded_at: at,
    };
    return { ...original, store, pages, records, command };
}

async function complete(f: ReturnType<typeof fixture>) {
    let state = await beginIndexedConversationUpgrade(f.store, f.command);
    let steps = 0;
    let activeReads = 0;
    while (state.progress.phase !== 'complete') {
        if (++steps > 10000) throw new Error('Historical upgrade failed to terminate');
        const previous = state;
        const advanced = await advanceIndexedConversationUpgrade(f.store, f.command, state.locator);
        state = advanced;
        const limits =
            previous.progress.phase === 'active_window' ? INDEXED_UPGRADE_ACTIVE_LIMITS : INDEXED_UPGRADE_STEP_LIMITS;
        const usage = advanced.usage;
        expect(usage.record_reads).toBeLessThanOrEqual(limits.record_reads);
        expect(usage.page_reads).toBeLessThanOrEqual(limits.page_reads);
        expect(usage.bytes).toBeLessThanOrEqual(limits.bytes);
        expect(usage.writes).toBeLessThanOrEqual(limits.writes);
        if (previous.progress.phase === 'active_window') activeReads = usage.record_reads;
        // Every next advance authenticates this immutable progress locator itself. Explicit
        // roundtrips cover the first step, each phase/audit-family boundary and completion;
        // the dedicated interrupted-step test separately proves exact predecessor replay.
        if (
            steps === 1 ||
            state.progress.phase !== previous.progress.phase ||
            state.progress.audit_family !== previous.progress.audit_family ||
            state.progress.phase === 'complete'
        )
            expect(await readIndexedUpgradeProgress(f.store, f.command, state.locator)).toEqual(state.progress);
    }
    return { ...state, steps, activeReads };
}

describe('bounded historical indexed upgrade', () => {
    it('point-authenticates rebuilt original witnesses without staging or changing historical completeness', async () => {
        const f = fixture('pending');
        const completed = await complete(f);
        const accepted = await finishIndexedConversationUpgrade(
            f.store,
            f.root,
            f.locator,
            f.command,
            completed.locator,
        );
        const original = structuredClone(f.root);
        const pages = f.pages.size;
        const records = f.records.size;
        const observed = await loadIndexedUpgradeAcceptedSource(
            f.store,
            accepted.root,
            accepted.locator,
            f.command.operation_id,
        );
        expect(observed.receipt).toEqual(accepted.receipt);
        expect(observed.original).toEqual(original);
        expect(observed.original_locator).toEqual(f.locator);
        expect(observed.original.processing_index_profile).toBeUndefined();
        expect(observed.progress.phase).toBe('complete');
        expect(f.pages.size).toBe(pages);
        expect(f.records.size).toBe(records);
        await expect(
            loadIndexedUpgradeAcceptedSource(
                f.store,
                { ...accepted.root, context_header: fixture('received').root.context_header },
                accepted.locator,
                f.command.operation_id,
            ),
        ).rejects.toThrow('original content, context or source facts');
        await expect(
            loadIndexedUpgradeAcceptedSource(
                f.store,
                {
                    ...accepted.root,
                    directories: { ...accepted.root.directories, turns: fixture('received').root.directories.turns },
                },
                accepted.locator,
                f.command.operation_id,
            ),
        ).rejects.toThrow('audited original directory');
        await expect(
            loadIndexedUpgradeAcceptedSource(f.store, accepted.root, accepted.locator, 'upgrade:unrelated'),
        ).rejects.toThrow('exact accepted receipt');
        expect(f.root).toEqual(original);
        expect(f.pages.size).toBe(pages);
        expect(f.records.size).toBe(records);
    });

    it.each(['operation_receipts', 'identifiers'] as const)(
        'rejects omission of original %s while the accepted upgrade marker remains present',
        async (family) => {
            const f = fixture('pending');
            const completed = await complete(f);
            const accepted = await finishIndexedConversationUpgrade(
                f.store,
                f.root,
                f.locator,
                f.command,
                completed.locator,
            );
            const value = await getPagedRecord(f.store, accepted.root.directories[family], f.command.operation_id);
            if (!value) throw new Error('Genuine accepted upgrade directory marker missing');
            const reduced = await putPagedRecord(f.store, undefined, f.command.operation_id, value);
            expect(reduced).not.toEqual(accepted.root.directories[family]);
            const pages = f.pages.size;
            const records = f.records.size;
            await expect(
                loadIndexedUpgradeAcceptedSource(
                    f.store,
                    {
                        ...accepted.root,
                        directories: { ...accepted.root.directories, [family]: reduced },
                    },
                    accepted.locator,
                    f.command.operation_id,
                ),
            ).rejects.toThrow('audited receipt or identifier directories');
            expect(f.pages.size).toBe(pages);
            expect(f.records.size).toBe(records);
        },
    );

    it('uses independently reproduced pinned original fixture bytes', async () => {
        const bytes = readFileSync(
            new URL('./fixtures/indexed-upgrade-fffdaef/historical-fffdaef-artifacts.json', import.meta.url),
        );
        expect((await hashContentBytes(bytes)).content_hash).toBe(
            'sha256:0a0c3ed13999e5dee8b6925d0216257eae9169900dc0797cb8d0ad3fd1fc8311',
        );
    });

    it.each([
        'received',
        'deleted',
        'pending',
        'imported_missing_generation',
        'imported_closed',
        'imported_generation_no_request',
        'executed',
    ])('upgrades genuine pinned old %s bytes through recovered phases', async (name) => {
        const f = fixture(name);
        const originalBytes = f.records.get(`root:${f.locator.content_hash}`);
        const originalContext = await loadIndexedActiveContext(f.store, f.root);
        const completed = await complete(f);
        const accepted = await finishIndexedConversationUpgrade(
            f.store,
            f.root,
            f.locator,
            f.command,
            completed.locator,
        );
        expect(accepted.applied).toBe(true);
        expect(accepted.root.source.revision).toBe(f.root.source.revision + 1);
        expect(accepted.receipt.operation_kind).toBe('indexed_upgrade');
        expect(accepted.root.accepted_output_index_complete).toBe(true);
        expect(await loadIndexedActiveContext(f.store, accepted.root)).toEqual(originalContext);
        expect(f.records.get(`root:${f.locator.content_hash}`)).toEqual(originalBytes);
        const retry = await finishIndexedConversationUpgrade(
            f.store,
            accepted.root,
            accepted.locator,
            f.command,
            completed.locator,
        );
        expect(retry.applied).toBe(false);
        expect(retry.receipt).toEqual(accepted.receipt);
        for (const source of [
            { ...accepted.root.source, conversation_id: 'conversation:foreign-upgrade-retry' },
            { ...accepted.root.source, revision: f.root.source.revision },
        ]) {
            await expect(
                finishIndexedConversationUpgrade(
                    f.store,
                    { ...accepted.root, source },
                    accepted.locator,
                    f.command,
                    completed.locator,
                ),
            ).rejects.toThrow('exact retained acceptance');
        }

        if (name === 'received') {
            expect(completed.steps).toBeGreaterThan(256);
            expect(completed.activeReads).toBeGreaterThan(64);
        }
        if (name === 'imported_generation_no_request') {
            const generation = await loadRecord(
                f.store,
                await getPagedRecord(f.store, accepted.root.directories.generations, 'imported:optional-generation'),
                GenerationSchema,
            );
            expect(generation.record_source).toBe('imported');
            expect(generation.request_id).toBeUndefined();
            expect(generation.request_receipt).toBeUndefined();
            expect(await getPagedRecord(f.store, accepted.root.directories.generations, generation.id)).toEqual(
                await getPagedRecord(f.store, f.root.directories.generations, generation.id),
            );
        }
        if (name === 'pending') {
            const header = await loadRecord(
                f.store,
                {
                    storage: 'record',
                    kind: 'processing_header',
                    id: accepted.root.source.conversation_id,
                    ...accepted.root.processing_header,
                },
                IndexedConversationProcessingHeaderSchema,
            );
            expect(header.unresolved_job_count).toBe(1);
            expect((await loadIndexedPendingProcessingJobs(f.store, accepted.root, { limit: 1 })).jobs).toHaveLength(1);
        }
        const history = await loadIndexedAcceptedOutputHistoryPage(f.store, accepted.root, {
            snapshot_revision: accepted.root.source.revision,
            limit: 10,
        });
        expect(history.references).toHaveLength(name === 'executed' ? 1 : 0);
        if (name === 'deleted') {
            for (const family of ['deleted_turns', 'operation_receipts'] as const) {
                const id = family === 'deleted_turns' ? 'turn:129' : 'delete:historical:one';
                expect(await getPagedRecord(f.store, accepted.root.directories[family], id)).toEqual(
                    await getPagedRecord(f.store, f.root.directories[family], id),
                );
            }
        }
    });

    it('reserves original nested content and call identities without converting imported facts to generated authority', async () => {
        const f = fixture('imported_closed');
        const completed = await complete(f);
        const accepted = await finishIndexedConversationUpgrade(
            f.store,
            f.root,
            f.locator,
            f.command,
            completed.locator,
        );
        expect(await getPagedRecord(f.store, accepted.root.directories.identifiers, 'imported:result:nested')).toEqual({
            storage: 'marker',
            kind: 'block',
            id: 'imported:result:nested',
        });
        const command = {
            conversation_id: accepted.root.source.conversation_id,
            options: {
                operation_id: 'append:reuse-nested',
                expected_revision: accepted.root.source.revision,
                payload_fingerprint: await fingerprintJson({ reuse: 'nested' }),
                recorded_at: at,
            },
            batch: {
                turns: [
                    {
                        id: 'reuse:turn',
                        kind: 'user' as const,
                        authority: 'ordinary' as const,
                        status: 'completed' as const,
                        timestamps: { recorded_at: at },
                        model_visibility: 'include' as const,
                        provenance: { type: 'received' as const },
                        blocks: [
                            {
                                id: 'imported:result:nested',
                                type: 'text' as const,
                                format: 'plain' as const,
                                text: 'New body cannot reuse archived nested identity',
                            },
                        ],
                    },
                ],
            },
        };
        command.options.payload_fingerprint = await fingerprintJson(command.batch);
        const beforeCollision = { pages: f.pages.size, records: f.records.size };
        await expect(stageIndexedRecordBatch(accepted.root, command, f.store)).rejects.toMatchObject({
            name: 'IndexedRecordAppendConflict',
            code: 'record_conflict',
            message: 'Indexed append identity imported:result:nested already exists',
        });
        expect({ pages: f.pages.size, records: f.records.size }).toEqual(beforeCollision);
        const callCommand = {
            ...command,
            options: { ...command.options, operation_id: 'append:reuse-call' },
            batch: {
                turns: [
                    {
                        id: 'reuse:agent',
                        kind: 'user' as const,
                        authority: 'ordinary' as const,
                        status: 'completed' as const,
                        timestamps: { recorded_at: at },
                        model_visibility: 'include' as const,
                        provenance: { type: 'received' as const },
                        blocks: [
                            {
                                id: 'imported:call',
                                type: 'text' as const,
                                format: 'plain' as const,
                                text: 'A fresh valid user block cannot reuse an original call identity.',
                            },
                        ],
                    },
                ],
            },
        };
        callCommand.options.payload_fingerprint = await fingerprintJson(callCommand.batch);
        await expect(stageIndexedRecordBatch(accepted.root, callCommand, f.store)).rejects.toBeInstanceOf(
            IndexedRecordAppendConflict,
        );
        await expect(stageIndexedRecordBatch(accepted.root, callCommand, f.store)).rejects.toMatchObject({
            code: 'record_conflict',
            message: 'Indexed append identity imported:call already exists',
        });
        expect({ pages: f.pages.size, records: f.records.size }).toEqual(beforeCollision);
        const validBatch = structuredClone(callCommand.batch);
        const validBlock = validBatch.turns[0]?.blocks[0];
        if (!validBlock) throw new Error('Actual valid user block fixture missing');
        validBlock.id = 'fresh:valid:user:block';
        const valid = await stageIndexedRecordBatch(
            accepted.root,
            {
                ...callCommand,
                batch: validBatch,
                options: {
                    ...callCommand.options,
                    operation_id: 'append:valid-original-namespace-control',
                    payload_fingerprint: await fingerprintJson(validBatch),
                },
            },
            f.store,
        );
        expect(valid.applied).toBe(true);
        expect(valid.receipt.accepted_turn_ids).toEqual(['reuse:agent']);
        const original = await getPagedRecord(f.store, f.root.directories.turns, 'imported:call:turn');
        expect(await getPagedRecord(f.store, accepted.root.directories.turns, 'imported:call:turn')).toEqual(original);
    });

    it('replays one interrupted step from the same immutable predecessor without changing its result', async () => {
        const f = fixture('received');
        const start = await beginIndexedConversationUpgrade(f.store, f.command);
        const first = await advanceIndexedConversationUpgrade(f.store, f.command, start.locator);
        const replay = await advanceIndexedConversationUpgrade(f.store, f.command, start.locator);
        expect(replay.locator).toEqual(first.locator);
        expect(replay.progress).toEqual(first.progress);
        await expect(
            finishIndexedConversationUpgrade(f.store, f.root, f.locator, f.command, first.locator),
        ).rejects.toThrow('incomplete evidence');
    });

    it('does not certify missing or corrupted immutable original bytes', async () => {
        const f = fixture('received');
        const start = await beginIndexedConversationUpgrade(f.store, f.command);
        const receipt = await getPagedRecord(
            f.store,
            f.root.directories.operation_receipts,
            'append:historical:received',
        );
        if (receipt?.storage !== 'record') throw new Error('Original receipt missing');
        const key = `${receipt.kind}:${receipt.content_hash}`;
        const bytes = f.records.get(key);
        if (!bytes) throw new Error('Original receipt bytes missing');
        f.records.delete(key);
        await expect(advanceIndexedConversationUpgrade(f.store, f.command, start.locator)).rejects.toThrow('missing');
        f.records.set(key, Uint8Array.from(bytes));
        f.records.get(key)?.fill(0);
        await expect(advanceIndexedConversationUpgrade(f.store, f.command, start.locator)).rejects.toThrow();
        expect(await readIndexedUpgradeProgress(f.store, f.command, start.locator)).toEqual(start.progress);
    });

    it('rejects hashed original nested identity duplication even within the same turn', async () => {
        const f = fixture('imported_closed');
        const block = await loadRecord(
            f.store,
            await getPagedRecord(f.store, f.root.directories.blocks, 'imported:result:block'),
            ContentBlockSchema,
        );
        if (block.type !== 'tool_result' || !block.content[0]) throw new Error('Actual captured nested result missing');
        const repeated = { ...block, content: [...block.content, structuredClone(block.content[0])] };
        const blockRef = await stageRecord(f.store, 'blocks', repeated.id, repeated);
        const receipt = await loadRecord(
            f.store,
            await getPagedRecord(f.store, f.root.directories.execution_receipts, 'imported:execution'),
            ExecutionReceiptSchema,
        );
        const changedReceipt = { ...receipt, result_fingerprint: await fingerprintJson(repeated) };
        const receiptRef = await stageRecord(f.store, 'execution_receipts', receipt.id, changedReceipt);
        const root = {
            ...f.root,
            directories: {
                ...f.root.directories,
                blocks: await putPagedRecord(f.store, f.root.directories.blocks, repeated.id, blockRef, 'replace'),
                execution_receipts: await putPagedRecord(
                    f.store,
                    f.root.directories.execution_receipts,
                    receipt.id,
                    receiptRef,
                    'replace',
                ),
            },
        };
        const stored = await stageRecord(f.store, 'root', root.source.conversation_id, root);
        const locator = { content_hash: stored.content_hash, size_bytes: stored.size_bytes };
        const command = { ...f.command, predecessor_root: locator };
        await expect(complete({ ...f, root, locator, command })).rejects.toThrow('multiple original occurrences');
        expect((await loadRecord(f.store, blockRef, ContentBlockSchema)).id).toBe(block.id);
    });

    it('rejects duplicate original deleted-turn nominations at separate lifetime ordinals', async () => {
        const f = fixture('deleted');
        const root = {
            ...f.root,
            directories: {
                ...f.root.directories,
                turn_order: await putPagedRecord(
                    f.store,
                    f.root.directories.turn_order,
                    indexedOrderedKey(128),
                    { storage: 'marker', kind: 'deleted_turn_order', id: 'turn:129' },
                    'replace',
                ),
            },
        };
        const stored = await stageRecord(f.store, 'root', root.source.conversation_id, root);
        const locator = { content_hash: stored.content_hash, size_bytes: stored.size_bytes };
        const command = { ...f.command, predecessor_root: locator };
        await expect(complete({ ...f, root, locator, command })).rejects.toThrow('repeats an original turn');
    });

    it('rejects an original turn omitted from order despite coherently reduced forward counters and tail', async () => {
        const f = fixture('received');
        const removed = await removePagedRecord(f.store, f.root.directories.turn_order, indexedOrderedKey(129));
        if (!removed.applied) throw new Error('Actual original order nomination missing');
        const link = await loadRecord(
            f.store,
            await getPagedRecord(f.store, f.root.directories.turn_links, 'turn:128'),
            IndexedConversationTurnLinkSchema,
        );
        const { next_turn_id: _next, ...tail } = link;
        const linkRef = await stageRecord(f.store, 'turn_links', tail.id, tail);
        const root = {
            ...f.root,
            turn_count: 129,
            live_turn_count: 129,
            active_tail_turn_id: 'turn:128',
            directories: {
                ...f.root.directories,
                turn_order: removed.root,
                turn_links: await putPagedRecord(f.store, f.root.directories.turn_links, tail.id, linkRef, 'replace'),
            },
        };
        const stored = await stageRecord(f.store, 'root', root.source.conversation_id, root);
        const locator = { content_hash: stored.content_hash, size_bytes: stored.size_bytes };
        const command = { ...f.command, predecessor_root: locator };
        await expect(complete({ ...f, root, locator, command })).rejects.toThrow('differs from complete order');
        expect(await getPagedRecord(f.store, root.directories.turns, 'turn:129')).toEqual(
            await getPagedRecord(f.store, f.root.directories.turns, 'turn:129'),
        );
    });

    it('rejects a changed head before publication and preserves historical root identities', async () => {
        const f = fixture('executed');
        const completed = await complete(f);
        const other = fixture('received');
        await expect(
            finishIndexedConversationUpgrade(f.store, other.root, other.locator, f.command, completed.locator),
        ).rejects.toThrow('original predecessor');
        expect(await fingerprintJson(f.root)).toEqual(await fingerprintJson(artifact.cases.executed.root));
    });

    it('counts pre-phase and nested active reads and rejects complete byte overflow before access', async () => {
        const f = fixture('received');
        const initial = createIndexedUpgradeStepStore(f.store);
        await initial.store.readRecord({
            storage: 'record',
            kind: 'root',
            id: f.root.source.conversation_id,
            ...f.locator,
        });
        const active = createIndexedUpgradeStepStore(f.store, INDEXED_UPGRADE_ACTIVE_LIMITS, initial.usage);
        await loadIndexedActiveContext(active.store, f.root);
        expect(active.usage.record_reads).toBeGreaterThan(initial.usage.record_reads);
        expect(active.usage.page_reads).toBeGreaterThan(0);
        const before = { ...active.usage };
        await expect(
            active.store.readRecord({
                storage: 'record',
                kind: 'root',
                id: f.root.source.conversation_id,
                content_hash: f.locator.content_hash,
                size_bytes: INDEXED_UPGRADE_ACTIVE_LIMITS.bytes + 1,
            }),
        ).rejects.toThrow('IO budget');
        expect(active.usage).toEqual(before);
    });
});
