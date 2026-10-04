import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { fingerprintJson } from '../src/identity.js';
import {
    type IndexedConversationRecordStore,
    loadIndexedPendingProcessingJobs,
    loadIndexedProcessingAppendAcceptance,
    loadIndexedProcessingSelectedContext,
    loadIndexedSelectedTextContext,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingNoOpCompletion,
    stageIndexedProcessingPhase,
    stageIndexedRecordBatch,
} from '../src/indexed-conversation.js';
import { resolveIndexedProcessingTextInput } from '../src/indexed-processing-working-set.js';
import { setProcessingPolicy } from '../src/processing.js';
import { appendConversationRecordsWithProcessing } from '../src/runtime.js';
import { INDEXED_PROCESSING_SELECTED_MAX_BLOCKS } from '../src/schemas/indexed-head.js';
import { emptyDocument, RECORDED_AT, userTurn } from './fixtures.js';

// Exhaustive CPU capacity fixtures include real immutable setup, dependency validation and retry;
// this is not an ACK/transport latency assertion. Smaller behavior tests retain Vitest's default.
const WORKING_SET_CAPACITY_TEST_TIMEOUT_MS = 30_000;

function storage() {
    const pages = new Map<string, Uint8Array>();
    const records = new Map<string, Uint8Array>();
    const reads: string[] = [];
    const pageReads: string[] = [];
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            pageReads.push(ref.content_hash);
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Indexed page absent');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            reads.push(`${ref.kind}:${ref.id}`);
            const bytes = records.get(ref.content_hash);
            if (!bytes) throw new Error('Indexed record absent');
            return Uint8Array.from(bytes);
        },
        async writeRecord(ref, bytes) {
            const actualHash = (await hashContentBytes(bytes)).content_hash;
            if (actualHash !== ref.content_hash)
                throw new Error(
                    `Indexed fixture ${ref.kind}:${ref.id} write hash ${actualHash} differs from ${ref.content_hash}`,
                );
            records.set(ref.content_hash, Uint8Array.from(bytes));
        },
    };
    return { store, reads, pageReads, pages, records };
}

async function configured() {
    return (
        await setProcessingPolicy(emptyDocument('conversation:indexed-processing'), {
            operation_id: 'operation:policy',
            expected_revision: 0,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                {
                    id: 'externalize-text',
                    version: '1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        })
    ).document;
}

describe('indexed processing append index', () => {
    it('rejects unsupported mixed trigger policy before indexed activation rather than queueing an irrecoverable policy-index job', async () => {
        const mixed = (
            await setProcessingPolicy(emptyDocument('conversation:mixed-policy'), {
                operation_id: 'operation:mixed-policy',
                expected_revision: 0,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors: [
                    {
                        id: 'future-budget',
                        version: '1',
                        scope: 'on_budget',
                        config: {},
                        required: true,
                        failure_behavior: 'block',
                    },
                    {
                        id: 'externalize-text',
                        version: '1',
                        scope: 'on_append',
                        config: {},
                        required: true,
                        failure_behavior: 'block',
                    },
                ],
                budget: { max_input_tokens: 200, output_reserve_tokens: 0, measurement_policy: 'exact_only' },
            })
        ).document;
        const { store } = storage();
        await expect(stageIndexedConversationSnapshot(mixed, undefined, store)).rejects.toThrow(
            'only registered on-append ordinary text',
        );
    });

    it(
        'validates the complete active block bound before acceptance and caches repeated immutable lookups per operation',
        async () => {
            const document = await configured();
            const memory = storage();
            const turn = userTurn('turn:at-limit', 'block:at-limit');
            turn.blocks = Array.from({ length: INDEXED_PROCESSING_SELECTED_MAX_BLOCKS }, (_, index) => ({
                id: `block:limit:${index}`,
                type: 'json',
                value: { index },
            }));
            const batch = {
                turns: [turn],
                context_entries: [{ id: 'entry:at-limit', type: 'source_turn' as const, turn_id: turn.id }],
            };
            const options = {
                operation_id: 'operation:at-limit',
                expected_revision: document.revision,
                payload_fingerprint: await fingerprintJson(batch),
                recorded_at: RECORDED_AT,
            };
            const full = await appendConversationRecordsWithProcessing(document, batch, options);
            const appended = await stageIndexedConversationSnapshot(full.document, undefined, memory.store);
            if (!appended.locator) throw new Error('At-limit accepted append lacks its immutable root');
            memory.reads.length = 0;
            memory.pageReads.length = 0;
            const selected = await loadIndexedProcessingSelectedContext(memory.store, appended.root, appended.locator);
            expect(selected.turns[0].selected_blocks).toHaveLength(INDEXED_PROCESSING_SELECTED_MAX_BLOCKS);
            expect(memory.reads.length).toBeGreaterThan(128);
            expect(new Set(memory.reads).size).toBe(memory.reads.length);
            expect(new Set(memory.pageReads).size).toBe(memory.pageReads.length);
            const { job } = (await loadIndexedPendingProcessingJobs(memory.store, appended.root)).jobs[0];
            expect((await resolveIndexedProcessingTextInput(selected, job, RECORDED_AT)).entry_ids).toEqual([]);
            const nextTurn = userTurn('turn:overflow', 'block:overflow');
            const nextBatch = {
                turns: [nextTurn],
                context_entries: [{ id: 'entry:overflow', type: 'source_turn' as const, turn_id: nextTurn.id }],
            };
            await expect(
                stageIndexedRecordBatch(
                    appended.root,
                    {
                        conversation_id: document.id,
                        batch: nextBatch,
                        options: {
                            operation_id: 'operation:overflow',
                            expected_revision: appended.root.source.revision,
                            payload_fingerprint: await fingerprintJson(nextBatch),
                            recorded_at: RECORDED_AT,
                        },
                    },
                    memory.store,
                ),
            ).rejects.toThrow('selected-block bound');
            // Staged writes cannot publish a rejected root; the original receipt/jobs remain exact.
            expect((await loadIndexedPendingProcessingJobs(memory.store, appended.root)).job_count).toBe(1);
            const overflowing = await appendConversationRecordsWithProcessing(full.document, nextBatch, {
                operation_id: 'operation:overflow',
                expected_revision: full.document.revision,
                payload_fingerprint: await fingerprintJson(nextBatch),
                recorded_at: RECORDED_AT,
            });
            await expect(
                stageIndexedConversationSnapshot(overflowing.document, undefined, memory.store),
            ).rejects.toThrow('selected-block bound');
        },
        WORKING_SET_CAPACITY_TEST_TIMEOUT_MS,
    );

    it(
        'rejects text closure without completion headroom before enqueue, while the same small shape remains supported',
        async () => {
            const document = await configured();
            const memory = storage();
            const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
            const turn = userTurn('turn:no-headroom', 'block:no-headroom');
            turn.blocks = Array.from({ length: 3000 }, (_, index) => ({
                id: `block:headroom:${index}`,
                type: 'text',
                text: 'x',
                format: 'plain',
            }));
            const batch = {
                turns: [turn],
                context_entries: [{ id: 'entry:no-headroom', type: 'source_turn' as const, turn_id: turn.id }],
            };
            const options = {
                operation_id: 'operation:no-headroom',
                expected_revision: document.revision,
                payload_fingerprint: await fingerprintJson(batch),
                recorded_at: RECORDED_AT,
            };
            const full = await appendConversationRecordsWithProcessing(document, batch, options);
            await expect(stageIndexedConversationSnapshot(full.document, undefined, memory.store)).rejects.toThrow(
                'completion headroom',
            );
            expect((await loadIndexedPendingProcessingJobs(memory.store, initial.root)).job_count).toBe(0);
            turn.blocks = turn.blocks.slice(0, 256);
            const small = await stageIndexedRecordBatch(
                initial.root,
                {
                    conversation_id: document.id,
                    batch,
                    options: { ...options, payload_fingerprint: await fingerprintJson(batch) },
                },
                memory.store,
            );
            if (!small.locator) throw new Error('Supported small append has no root');
            const selected = await loadIndexedProcessingSelectedContext(memory.store, small.root, small.locator);
            const { job } = (await loadIndexedPendingProcessingJobs(memory.store, small.root)).jobs[0];
            expect((await resolveIndexedProcessingTextInput(selected, job, RECORDED_AT)).entry_ids).toEqual([
                'entry:no-headroom',
            ]);
        },
        WORKING_SET_CAPACITY_TEST_TIMEOUT_MS,
    );

    it('preserves pure JSON and completes its real empty text-stage job without an attempt or archive', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const turn = userTurn('turn:json', 'block:json');
        turn.blocks = [{ id: 'block:json', type: 'json', value: { preserve: [null, false, 3] } }];
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:json', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const options = {
            operation_id: 'operation:json',
            expected_revision: document.revision,
            payload_fingerprint: await fingerprintJson(batch),
            recorded_at: RECORDED_AT,
        };
        const materialized = await appendConversationRecordsWithProcessing(document, batch, options);
        const appended = await stageIndexedRecordBatch(
            initial.root,
            { conversation_id: document.id, batch, options },
            memory.store,
        );
        if (!appended.locator) throw new Error('Accepted JSON append lacks its immutable root');
        const [job] = Object.values(materialized.document.processing.jobs ?? {});
        if (!job) throw new Error('JSON append lost its actual text-stage job');
        expect(job.selection).toEqual({ kind: 'entries', entry_ids: [] });
        expect((await loadIndexedPendingProcessingJobs(memory.store, appended.root)).jobs[0].job).toEqual(job);
        const selected = await loadIndexedProcessingSelectedContext(memory.store, appended.root, appended.locator);
        const resolution = await resolveIndexedProcessingTextInput(selected, job, RECORDED_AT);
        let head = await stageIndexedProcessingPhase(memory.store, appended.root, appended.locator, {
            phase: 'resolve',
            value: resolution,
        });
        const payload = {
            job_id: job.id,
            resolved_input_fingerprint: await fingerprintJson(resolution),
            kind: 'no_op' as const,
            reason: 'no_eligible_blocks',
            recorded_at: RECORDED_AT,
        };
        head = await stageIndexedProcessingPhase(memory.store, head.root, head.locator, {
            phase: 'output',
            value: { ...payload, output_fingerprint: await fingerprintJson(payload) },
        });
        const completed = await stageIndexedProcessingNoOpCompletion(memory.store, head.root, head.locator, job.id);
        expect(completed.completion.status).toBe('no_op');
        const retained = await loadIndexedProcessingSelectedContext(memory.store, completed.root, completed.locator);
        expect(retained.turns[0].selected_blocks).toEqual(turn.blocks);
        expect(retained.context.entries).toEqual(selected.context.entries);
        expect(
            (await stageIndexedProcessingNoOpCompletion(memory.store, completed.root, completed.locator, job.id))
                .applied,
        ).toBe(false);
    });

    it('shares exact job construction with materialized append and never reports ACK as readiness', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const turn = userTurn('turn:input', 'block:input');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:input', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const options = {
            expected_revision: document.revision,
            operation_id: 'operation:input',
            recorded_at: RECORDED_AT,
            payload_fingerprint: await fingerprintJson(batch),
        };
        const materialized = await appendConversationRecordsWithProcessing(document, batch, options);
        const staged = await stageIndexedRecordBatch(
            initial.root,
            { conversation_id: document.id, batch, options },
            memory.store,
        );
        expect(staged.applied).toBe(true);
        const ack = await loadIndexedProcessingAppendAcceptance(memory.store, staged.root, options.operation_id);
        expect(ack.receipt).toEqual(materialized.document.operation_receipts[options.operation_id]);
        expect(ack.jobs).toEqual(Object.values(materialized.document.processing.jobs ?? {}));
        expect(ack.processing.status).toBe('pending');
        const pending = await loadIndexedPendingProcessingJobs(memory.store, staged.root);
        expect(pending.jobs.map((entry) => entry.job)).toEqual(ack.jobs);
        expect(pending.unresolved_job_count).toBe(1);
        expect(pending.required_unresolved_job_count).toBe(1);
        expect(pending.required_blocked_job_count).toBe(0);
        expect(pending.has_more).toBe(false);
        if (!staged.locator) throw new Error('Staged root locator absent');
        await expect(loadIndexedSelectedTextContext(memory.store, staged.root, staged.locator)).rejects.toThrow();
        const bytesBefore = memory.records.size;
        const retry = await stageIndexedRecordBatch(
            staged.root,
            { conversation_id: document.id, batch, options },
            memory.store,
        );
        expect(retry.applied).toBe(false);
        expect(retry.root).toEqual(staged.root);
        expect(retry.receipt).toEqual(ack.receipt);
        expect(memory.records.size).toBe(bytesBefore);
        expect(await loadIndexedPendingProcessingJobs(memory.store, retry.root)).toEqual(pending);
        expect(canonicalJsonContentBytes(initial.root)).not.toEqual(canonicalJsonContentBytes(staged.root));
    });

    it('reads only current pending pages and receipt associations at 10k and 100k cold turns', async () => {
        const profiles: number[] = [];
        for (const count of [10_000, 100_000]) {
            const document = await configured();
            const template = userTurn('turn:cold-template', 'block:cold-template');
            const cold = {
                ...document,
                turns: Array.from({ length: count }, (_, index) => ({
                    ...template,
                    id: `turn:cold:${index}`,
                    blocks: [{ ...template.blocks[0], id: `block:cold:${index}` }],
                })),
            };
            const memory = storage();
            let root = (await stageIndexedConversationSnapshot(cold, undefined, memory.store)).root;
            for (const index of [0, 1]) {
                const turn = userTurn(`turn:input:${index}`, `block:input:${index}`);
                const batch = {
                    turns: [turn],
                    context_entries: [{ id: `entry:input:${index}`, type: 'source_turn' as const, turn_id: turn.id }],
                };
                root = (
                    await stageIndexedRecordBatch(
                        root,
                        {
                            conversation_id: document.id,
                            batch,
                            options: {
                                expected_revision: root.source.revision,
                                operation_id: `operation:input:${index}`,
                                payload_fingerprint: await fingerprintJson(batch),
                                recorded_at: RECORDED_AT,
                            },
                        },
                        memory.store,
                    )
                ).root;
            }
            memory.reads.length = 0;
            const page = await loadIndexedPendingProcessingJobs(memory.store, root, { limit: 1 });
            expect(page.jobs).toHaveLength(1);
            expect(page.has_more).toBe(true);
            if (!page.next_cursor) throw new Error('Pending page has no exclusive next cursor');
            const next = await loadIndexedPendingProcessingJobs(memory.store, root, {
                cursor: page.next_cursor,
                limit: 1,
            });
            expect(next.jobs).toHaveLength(1);
            expect(next.has_more).toBe(false);
            expect(new Set([...page.jobs, ...next.jobs].map((entry) => entry.job.id)).size).toBe(2);
            const originalAck = await loadIndexedProcessingAppendAcceptance(memory.store, root, 'operation:input:0');
            expect(originalAck.jobs).toHaveLength(1);
            expect(originalAck.receipt.result_revision).toBeLessThan(root.source.revision);
            expect(originalAck.processing.status).toBe('pending');
            expect(memory.reads.some((id) => id.startsWith('turns:') || id.startsWith('blocks:'))).toBe(false);
            expect(memory.reads.length).toBeLessThan(32);
            profiles.push(memory.reads.length);
        }
        expect(profiles[1]).toBe(profiles[0]);
    }, 120_000);

    it('rejects incomplete pending or receipt association indexes without scanning historical records', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const turn = userTurn('turn:input', 'block:input');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:input', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const staged = await stageIndexedRecordBatch(
            initial.root,
            {
                conversation_id: document.id,
                batch,
                options: {
                    expected_revision: document.revision,
                    operation_id: 'operation:input',
                    payload_fingerprint: await fingerprintJson(batch),
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        const missingPending = structuredClone(staged.root);
        delete missingPending.directories.processing_pending;
        await expect(loadIndexedPendingProcessingJobs(memory.store, missingPending)).rejects.toThrow(
            'complete header count disagree',
        );
        const missingAssociation = structuredClone(staged.root);
        delete missingAssociation.directories.processing_by_operation;
        await expect(loadIndexedPendingProcessingJobs(memory.store, missingAssociation)).rejects.toThrow();
        const readsBefore = memory.reads.length;
        let getterReads = 0;
        const options = Object.defineProperty({}, 'limit', {
            enumerable: true,
            get() {
                getterReads++;
                return 1;
            },
        });
        await expect(loadIndexedPendingProcessingJobs(memory.store, staged.root, options)).rejects.toThrow(
            'not bounded JSON',
        );
        expect(getterReads).toBe(0);
        expect(memory.reads).toHaveLength(readsBefore);
    });
});
