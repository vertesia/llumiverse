import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { fingerprintJson } from '../src/identity.js';
import {
    assertIndexedCurrentPolicy,
    type IndexedConversationRecordStore,
    loadIndexedPendingProcessingJobs,
    loadIndexedProcessingAppendAcceptance,
    loadIndexedProcessingJobState,
    loadIndexedProcessingPredecessorEvidence,
    loadIndexedProcessingQueuedJobAcceptance,
    loadIndexedProcessingSelectedContext,
    loadIndexedReadySelectedContext,
    loadIndexedSelectedTextContext,
    loadIndexedSettledProcessingSelectedContext,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingCoverage,
    stageIndexedProcessingNoOpCompletion,
    stageIndexedProcessingPhase,
    stageIndexedProcessingPolicy,
    stageIndexedProcessingQueue,
    stageIndexedRecordBatch,
    supportsIndexedInheritedProcessingPolicy,
    supportsIndexedRegisteredProcessingPolicy,
} from '../src/indexed-conversation.js';
import {
    indexedTextExternalizationOriginals,
    resolveIndexedProcessingTextInput,
} from '../src/indexed-processing-working-set.js';
import { getPagedRecord } from '../src/paged-record-index.js';
import {
    type ProcessingStore,
    resolveProcessingJobInput,
    runProcessingJob,
    setProcessingPolicy,
} from '../src/processing.js';
import { appendConversationRecordsWithProcessing } from '../src/runtime.js';
import {
    INDEXED_PROCESSING_SELECTED_MAX_BLOCKS,
    IndexedConversationProcessingHeaderSchema,
} from '../src/schemas/indexed-head.js';
import { ProcessingResolvedInputSchema } from '../src/schemas/processing.js';
import { parseConversationDocument } from '../src/validation.js';
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
    it('prepares default-disabled indexed input without scheduling work and binds readiness to the exact target/count', async () => {
        const document = emptyDocument('conversation:disabled-native');
        const memory = storage();
        const snapshot = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const turn = userTurn('turn:disabled-native');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:disabled-native', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const accepted = await stageIndexedRecordBatch(
            snapshot.root,
            {
                conversation_id: document.id,
                batch,
                options: {
                    operation_id: 'append:disabled-native',
                    expected_revision: snapshot.root.source.revision,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            memory.store,
        );
        if (!accepted.locator) throw new Error('Disabled accepted input lacks its actual root');
        const pending = await loadIndexedPendingProcessingJobs(memory.store, accepted.root);
        expect(pending.enabled).toBe(false);
        expect(pending.job_count).toBe(0);
        expect(pending.jobs).toEqual([]);
        const selected = await loadIndexedSettledProcessingSelectedContext(
            memory.store,
            accepted.root,
            accepted.locator,
        );
        expect(selected.turns[0].selected_blocks).toEqual(turn.blocks);
        expect(supportsIndexedInheritedProcessingPolicy(document.processing)).toBe(true);
        expect(supportsIndexedRegisteredProcessingPolicy(document.processing)).toBe(false);
        const binding = {
            target_fingerprint: `sha256:${'a'.repeat(64)}`,
            measured_input_tokens: 8,
            tokenizer_id: 'tokenizer:disabled',
            measurement_fingerprint: `sha256:${'b'.repeat(64)}`,
        };
        const covered = await stageIndexedProcessingCoverage(memory.store, accepted.root, accepted.locator, {
            ...binding,
            operation_id: 'coverage:disabled-native',
            expected_revision: accepted.root.source.revision,
            recorded_at: RECORDED_AT,
        });
        expect(covered.coverage.status).toBe('ready');
        expect(
            (await loadIndexedReadySelectedContext(memory.store, covered.root, covered.locator, binding)).coverage,
        ).toEqual(covered.coverage);
        for (const changed of [
            { ...binding, target_fingerprint: `sha256:${'c'.repeat(64)}` },
            { ...binding, measured_input_tokens: binding.measured_input_tokens + 1 },
            { ...binding, measurement_fingerprint: `sha256:${'d'.repeat(64)}` },
        ]) {
            await expect(
                loadIndexedReadySelectedContext(memory.store, covered.root, covered.locator, changed),
            ).rejects.toThrow('exact context/target/count');
        }
        const writes = [memory.pages.size, memory.records.size];
        await expect(
            stageIndexedProcessingQueue(memory.store, covered.root, covered.locator, {
                operation_id: 'queue:disabled-native',
                expected_revision: covered.root.source.revision,
                expected_context_revision: selected.context.revision,
                recorded_at: RECORDED_AT,
                processor_id: 'externalize-text',
                scope: 'manual',
                selected_entry_ids: ['entry:disabled-native'],
            }),
        ).rejects.toThrow('no enabled bounded policy capacity');
        expect([memory.pages.size, memory.records.size]).toEqual(writes);
    });

    it('retains the genuine disable command through preparation and rejects missing or tampered accepted policy evidence', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const command = {
            operation_id: 'policy:disable-native',
            expected_revision: initial.root.source.revision,
            recorded_at: RECORDED_AT,
            enabled: false,
            processors: document.processing.processors,
        };
        const disabled = await stageIndexedProcessingPolicy(
            memory.store,
            initial.root,
            initial.locator,
            command,
            async () => {
                throw new Error('Disabling must not invoke processor execution capability');
            },
        );
        const headerBytes = memory.records.get(disabled.root.processing_header.content_hash);
        if (!headerBytes) throw new Error('Actual disabled header is absent');
        const header = IndexedConversationProcessingHeaderSchema.parse(
            JSON.parse(new TextDecoder().decode(headerBytes)),
        );
        expect(header.enabled).toBe(false);
        expect(header.selected_policy_operation_id).toBe(command.operation_id);
        await expect(
            loadIndexedSettledProcessingSelectedContext(memory.store, disabled.root, disabled.locator),
        ).resolves.toMatchObject({ source: disabled.root.source });
        expect(
            (
                await stageIndexedProcessingPolicy(
                    memory.store,
                    disabled.root,
                    disabled.locator,
                    command,
                    async () => undefined,
                )
            ).receipt,
        ).toEqual(disabled.receipt);
        for (const descriptor of [
            await getPagedRecord(memory.store, disabled.root.directories.operation_receipts, command.operation_id),
            await getPagedRecord(
                memory.store,
                disabled.root.directories.processing_records,
                JSON.stringify(['selected_policy_commands', command.operation_id]),
            ),
        ]) {
            if (descriptor?.storage !== 'record') throw new Error('Accepted disabled policy point witness is absent');
            const original = memory.records.get(descriptor.content_hash);
            if (!original) throw new Error('Accepted disabled policy bytes are absent');
            memory.records.delete(descriptor.content_hash);
            await expect(
                loadIndexedSettledProcessingSelectedContext(memory.store, disabled.root, disabled.locator),
            ).rejects.toThrow('Indexed record absent');
            memory.records.set(descriptor.content_hash, canonicalJsonContentBytes({ ...command, enabled: true }));
            await expect(
                loadIndexedSettledProcessingSelectedContext(memory.store, disabled.root, disabled.locator),
            ).rejects.toThrow();
            memory.records.set(descriptor.content_hash, original);
        }
        await expect(
            assertIndexedCurrentPolicy(memory.store, disabled.root, { ...header, enabled: true }),
        ).rejects.toThrow('immutable root descriptor');
        await expect(
            loadIndexedSettledProcessingSelectedContext(memory.store, disabled.root, disabled.locator),
        ).resolves.toMatchObject({ source: disabled.root.source });
    });

    it('does not erase accepted required work when disabling processing without explicit supersession', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const turn = userTurn('turn:disable-pending');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:disable-pending', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const accepted = await stageIndexedRecordBatch(
            initial.root,
            {
                conversation_id: document.id,
                batch,
                options: {
                    operation_id: 'append:disable-pending',
                    expected_revision: initial.root.source.revision,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            memory.store,
        );
        if (!accepted.locator) throw new Error('Actual pending source lacks its root');
        const pending = await loadIndexedPendingProcessingJobs(memory.store, accepted.root);
        expect(pending.required_unresolved_job_count).toBe(1);
        const writes = [memory.pages.size, memory.records.size];
        await expect(
            stageIndexedProcessingPolicy(
                memory.store,
                accepted.root,
                accepted.locator,
                {
                    operation_id: 'policy:disable-pending',
                    expected_revision: accepted.root.source.revision,
                    recorded_at: RECORDED_AT,
                    enabled: false,
                    processors: document.processing.processors,
                },
                async () => undefined,
            ),
        ).rejects.toThrow('explicit supersession of every unresolved job');
        expect([memory.pages.size, memory.records.size]).toEqual(writes);
        expect(await loadIndexedPendingProcessingJobs(memory.store, accepted.root)).toEqual(pending);
        await expect(
            loadIndexedSettledProcessingSelectedContext(memory.store, accepted.root, accepted.locator),
        ).rejects.toThrow('unresolved processing obligations');
    });

    it('retains registered ordered automatic and budget policy through a real materialized snapshot', async () => {
        const initial = await configured();
        const configuration = initial.processing.processors[0];
        const policy = await setProcessingPolicy(initial, {
            operation_id: 'policy:inherited-ordered',
            expected_revision: initial.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [configuration, { ...configuration }, { ...configuration, scope: 'on_budget' }],
            budget: { max_input_tokens: 128, output_reserve_tokens: 16, measurement_policy: 'exact_only' },
        });
        const turn = userTurn('turn:inherited-ordered');
        const accepted = await appendConversationRecordsWithProcessing(
            policy.document,
            {
                turns: [turn],
                context_entries: [{ id: 'entry:inherited-ordered', type: 'source_turn', turn_id: turn.id }],
            },
            {
                operation_id: 'append:inherited-ordered',
                expected_revision: policy.document.revision,
                payload_fingerprint: 'sha256:inherited-ordered',
                recorded_at: RECORDED_AT,
            },
        );
        const memory = storage();
        const staged = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        const bytes = memory.records.get(staged.root.processing_header.content_hash);
        if (!bytes) throw new Error('Inherited policy lacks its actual immutable header');
        const header = IndexedConversationProcessingHeaderSchema.parse(JSON.parse(new TextDecoder().decode(bytes)));
        expect(header.selected_policy_operation_id).toBe(policy.change.operation_id);
        expect(header.selected_policy_origin).toBe('materialized');
        expect(header.processors).toEqual(policy.document.processing.processors);
        await expect(assertIndexedCurrentPolicy(memory.store, staged.root, header)).resolves.toBeUndefined();
        const pending = await loadIndexedPendingProcessingJobs(memory.store, staged.root);
        expect(pending.jobs.map(({ job }) => job.processor_index).sort((left, right) => left - right)).toEqual([0, 1]);
        await expect(
            assertIndexedCurrentPolicy(memory.store, staged.root, {
                ...header,
                processors: [{ ...configuration, id: 'unregistered' }],
            }),
        ).rejects.toThrow('immutable root descriptor');
        const unsupported = await setProcessingPolicy(initial, {
            operation_id: 'policy:unregistered-inherited',
            expected_revision: initial.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [{ ...configuration, id: 'unregistered' }],
        });
        const unsupportedSnapshot = await stageIndexedConversationSnapshot(
            unsupported.document,
            undefined,
            memory.store,
        );
        const unsupportedBytes = memory.records.get(unsupportedSnapshot.root.processing_header.content_hash);
        if (!unsupportedBytes) throw new Error('Unsupported policy lacks its actual snapshot header');
        await expect(
            assertIndexedCurrentPolicy(
                memory.store,
                unsupportedSnapshot.root,
                IndexedConversationProcessingHeaderSchema.parse(JSON.parse(new TextDecoder().decode(unsupportedBytes))),
            ),
        ).rejects.toThrow('registered bounded');
    });
    it.each(['resolve', 'attempt', 'output'] as const)(
        'rejects real fault-paused materialized %s before any indexed write',
        async (phase) => {
            const policy = await configured();
            const turn = userTurn(`turn:paused-${phase}`);
            const batch = {
                turns: [turn],
                context_entries: [{ id: `entry:paused-${phase}`, type: 'source_turn' as const, turn_id: turn.id }],
            };
            const accepted = await appendConversationRecordsWithProcessing(policy, batch, {
                operation_id: `append:paused-${phase}`,
                expected_revision: policy.revision,
                recorded_at: RECORDED_AT,
                payload_fingerprint: await fingerprintJson(batch),
            });
            let current = accepted.document;
            const job = Object.values(current.processing.jobs ?? {})[0];
            if (!job) throw new Error('Actual accepted append did not enqueue processing');
            const unstarted = storage();
            const initial = await stageIndexedConversationSnapshot(current, undefined, unstarted.store);
            expect(
                (await loadIndexedPendingProcessingJobs(unstarted.store, initial.root)).jobs.map(({ job }) => job.id),
            ).toEqual([job.id]);
            let armed = true;
            let calls = 0;
            const materialized: ProcessingStore = {
                load: async () => structuredClone(current),
                async commit(expected, document) {
                    if (current.revision !== expected) return false;
                    current = parseConversationDocument(document);
                    if (armed && current.operation_receipts[`processing:${phase}:${job.id}`]) {
                        armed = false;
                        throw new Error('Actual phase committed but process lost its ACK');
                    }
                    return true;
                },
            };
            const registry = {
                resolve: () => ({
                    run: async () => {
                        calls += 1;
                        return { kind: 'no_op' as const, reason: 'Genuine materialized processor output' };
                    },
                }),
            };
            await expect(
                runProcessingJob(materialized, registry, job.id, `attempt:paused-${phase}`, () => RECORDED_AT),
            ).rejects.toThrow('Actual phase committed but process lost its ACK');
            const retained = structuredClone(current);
            expect(retained.processing.resolved_inputs?.[job.id]).toBeDefined();
            expect(retained.operation_receipts[`processing:${phase}:${job.id}`]).toBeDefined();
            const writes = storage();
            await expect(stageIndexedConversationSnapshot(current, undefined, writes.store)).rejects.toThrow(
                'unresolved materialized phases to drain or be superseded',
            );
            expect(writes.pages.size).toBe(0);
            expect(writes.records.size).toBe(0);
            expect(current).toEqual(retained);
            expect(calls).toBe(phase === 'output' ? 1 : 0);
            if (phase === 'attempt') {
                expect(
                    (await runProcessingJob(materialized, registry, job.id, 'attempt:retry', () => RECORDED_AT)).status,
                ).toBe('in_progress');
                expect(calls).toBe(0); // An uncertain materialized attempt remains owned by its original recovery protocol.
            } else {
                await runProcessingJob(materialized, registry, job.id, 'attempt:retry', () => RECORDED_AT);
                expect(calls).toBe(1);
                const finished = storage();
                const migrated = await stageIndexedConversationSnapshot(current, undefined, finished.store);
                expect((await loadIndexedPendingProcessingJobs(finished.store, migrated.root)).jobs).toEqual([]);
                expect((await loadIndexedProcessingJobState(finished.store, migrated.root, job.id)).completion).toEqual(
                    current.processing.completions?.[job.id],
                );
            }
            if (phase === 'resolve') {
                const superseded = await setProcessingPolicy(retained, {
                    operation_id: 'policy:supersede-paused',
                    expected_revision: retained.revision,
                    recorded_at: RECORDED_AT,
                    enabled: true,
                    processors: retained.processing.processors,
                    supersede_job_ids: [job.id],
                    supersession_reason: 'Explicit original-policy supersession',
                });
                const memory = storage();
                const migrated = await stageIndexedConversationSnapshot(superseded.document, undefined, memory.store);
                expect((await loadIndexedPendingProcessingJobs(memory.store, migrated.root)).jobs).toEqual([]);
                const descriptor = await getPagedRecord(
                    memory.store,
                    migrated.root.directories.processing_records,
                    JSON.stringify(['resolved_inputs', job.id]),
                );
                if (
                    descriptor?.storage !== 'record' ||
                    descriptor.kind !== 'processing_records' ||
                    descriptor.id !== job.id
                )
                    throw new Error('Exact superseded resolution was not retained');
                const bytes = await memory.store.readRecord(descriptor);
                expect((await hashContentBytes(bytes)).content_hash).toBe(descriptor.content_hash);
                expect(bytes.byteLength).toBe(descriptor.size_bytes);
                expect(ProcessingResolvedInputSchema.parse(JSON.parse(new TextDecoder().decode(bytes)))).toEqual(
                    retained.processing.resolved_inputs?.[job.id],
                );
                // Historical inspection authenticates the real old policy without permitting new work.
                const historical = await loadIndexedProcessingJobState(memory.store, migrated.root, job.id);
                expect(historical.job).toEqual(job);
                expect(historical.resolution).toEqual(retained.processing.resolved_inputs?.[job.id]);
                expect(historical.supersession).toEqual(superseded.document.processing.supersessions?.[job.id]);
                const recordsBefore = memory.records.size,
                    pagesBefore = memory.pages.size;
                await expect(
                    stageIndexedProcessingPhase(memory.store, migrated.root, migrated.locator, {
                        phase: 'attempt',
                        value: {
                            job_id: job.id,
                            resolved_input_fingerprint: await fingerprintJson(historical.resolution),
                            attempt_token: 'attempt:cannot-restart-superseded',
                            started_at: RECORDED_AT,
                        },
                    }),
                ).rejects.toThrow('completed or superseded job');
                expect(memory.records.size).toBe(recordsBefore);
                expect(memory.pages.size).toBe(pagesBefore);
            }
        },
    );

    it.each([false, true])(
        'retains genuine materialized processing phase bytes through indexed activation (duplicate hash present=%s)',
        async (duplicateHash) => {
            const configuration = (await configured()).processing.processors[0];
            const policy = await setProcessingPolicy(emptyDocument('conversation:migrated-processing'), {
                operation_id: 'policy:migrated',
                expected_revision: 0,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors: [configuration],
            });
            const turn = userTurn('turn:migrated-processing');
            const batch = {
                turns: [turn],
                context_entries: [{ id: 'entry:migrated-processing', type: 'source_turn' as const, turn_id: turn.id }],
            };
            const accepted = await appendConversationRecordsWithProcessing(policy.document, batch, {
                operation_id: 'append:migrated-processing',
                expected_revision: policy.document.revision,
                recorded_at: RECORDED_AT,
                payload_fingerprint: await fingerprintJson(batch),
            });
            let current = accepted.document;
            const materialized: ProcessingStore = {
                load: async () => structuredClone(current),
                async commit(expected, document) {
                    if (current.revision !== expected) return false;
                    current = parseConversationDocument(document);
                    return true;
                },
            };
            const jobs = Object.values(current.processing.jobs ?? {}).sort((a, b) => a.stage_index - b.stage_index);
            expect(jobs).toHaveLength(1);
            let calls = 0;
            const registry = {
                resolve: () => ({
                    run: async () => {
                        calls += 1;
                        return { kind: 'no_op' as const, reason: 'Actual materialized processing result' };
                    },
                }),
            };
            for (const job of jobs)
                await runProcessingJob(materialized, registry, job.id, `attempt:${job.id}`, () => RECORDED_AT);
            expect(calls).toBe(1);
            for (const job of jobs) {
                const resolution = current.processing.resolved_inputs?.[job.id];
                const completion = current.processing.completions?.[job.id];
                if (!resolution || !completion) throw new Error('Real materialized job did not complete');
                expect(completion.status).toBe('no_op');
                for (const [phase, value] of [
                    ['resolve', resolution],
                    ['complete', completion],
                ] as const) {
                    const receipt = current.operation_receipts[`processing:${phase}:${job.id}`];
                    if (!receipt.processing_operation) throw new Error('Actual phase has no typed operation');
                    expect(receipt.processing_operation.result_fingerprint).toBeUndefined();
                    expect(receipt.payload_fingerprint).toBe(await fingerprintJson(value));
                    if (duplicateHash) receipt.processing_operation.result_fingerprint = receipt.payload_fingerprint;
                }
            }
            const memory = storage();
            const migrated = await stageIndexedConversationSnapshot(current, undefined, memory.store);
            if (!migrated.locator) throw new Error('Real imported processing fixture lacks root');
            for (const job of jobs) {
                const state = await loadIndexedProcessingJobState(memory.store, migrated.root, job.id);
                expect(state.job).toEqual(job);
                expect(state.resolution).toEqual(current.processing.resolved_inputs?.[job.id]);
                expect(state.resolution_receipt).toEqual(current.operation_receipts[`processing:resolve:${job.id}`]);
                expect(state.completion).toEqual(current.processing.completions?.[job.id]);
            }
            const selected = await loadIndexedProcessingSelectedContext(memory.store, migrated.root, migrated.locator);
            const first = await loadIndexedProcessingJobState(memory.store, migrated.root, jobs[0].id);
            if (!first.resolution || !first.resolution_receipt) throw new Error('Imported resolution missing');
            // Completed imported jobs are read as retained evidence, not re-executed using
            // the indexed executor's independently versioned selected-context fingerprint.
            expect(await resolveProcessingJobInput(accepted.document, first.job, RECORDED_AT)).toEqual(
                first.resolution,
            );
            const pending = await loadIndexedPendingProcessingJobs(memory.store, migrated.root);
            expect(pending.jobs).toEqual([]);
            expect(pending.unresolved_job_count).toBe(0);
            const retainedTurn = selected.turns.find((projection) => projection.header.id === turn.id);
            if (!retainedTurn) throw new Error('Bounded selected original turn missing');
            expect(retainedTurn.selected_blocks).toEqual(turn.blocks);
            expect(retainedTurn.selected_blocks).toEqual([
                { id: turn.blocks[0].id, type: 'text', text: 'turn:migrated-processing-text', format: 'plain' },
            ]);
            const indexedResolution = await resolveIndexedProcessingTextInput(
                { ...selected, source: { ...selected.source, revision: first.resolution.source_revision } },
                first.job,
                first.resolution.recorded_at,
            );
            expect(indexedResolution.source_fingerprint).toBe(first.resolution.source_fingerprint);
            expect(indexedResolution.context_fingerprint).not.toBe(first.resolution.context_fingerprint);
            await expect(
                indexedTextExternalizationOriginals(selected, first.job, first.resolution, first.resolution_receipt),
            ).rejects.toThrow('Indexed originals differ from their exact retained resolved selection');
            const corrupted = structuredClone(current);
            const receipt = corrupted.operation_receipts[`processing:resolve:${jobs[0].id}`];
            if (!receipt.processing_operation) throw new Error('Corruption fixture lost actual phase');
            receipt.processing_operation.result_fingerprint = `sha256:${'0'.repeat(64)}`;
            const badMemory = storage();
            const bad = await stageIndexedConversationSnapshot(corrupted, undefined, badMemory.store);
            await expect(loadIndexedProcessingJobState(badMemory.store, bad.root, jobs[0].id)).rejects.toThrow(
                'exact immutable phase receipt',
            );
        },
    );

    it('reads exact no-op predecessor receipts from a genuine multi-stage indexed policy', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        if (!initial.locator) throw new Error('Indexed predecessor fixture lacks root');
        const policy = await stageIndexedProcessingPolicy(
            memory.store,
            initial.root,
            initial.locator,
            {
                operation_id: 'policy:indexed-predecessor',
                expected_revision: initial.root.source.revision,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors: [document.processing.processors[0], document.processing.processors[0]],
            },
            async () => undefined,
        );
        const turn = {
            ...userTurn('turn:indexed-predecessor'),
            blocks: [{ id: 'block:indexed-predecessor', type: 'json' as const, value: { actual: true } }],
        };
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:indexed-predecessor', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const appended = await stageIndexedRecordBatch(
            policy.root,
            {
                conversation_id: document.id,
                batch,
                options: {
                    operation_id: 'append:indexed-predecessor',
                    expected_revision: policy.root.source.revision,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            memory.store,
        );
        if (!appended.locator) throw new Error('Indexed predecessor append lacks root');
        const jobs = (await loadIndexedPendingProcessingJobs(memory.store, appended.root)).jobs
            .map(({ job }) => job)
            .sort((a, b) => a.stage_index - b.stage_index);
        expect(jobs).toHaveLength(2);
        const selected = await loadIndexedProcessingSelectedContext(memory.store, appended.root, appended.locator);
        const resolution = await resolveIndexedProcessingTextInput(selected, jobs[0], RECORDED_AT);
        const resolved = await stageIndexedProcessingPhase(memory.store, appended.root, appended.locator, {
            phase: 'resolve',
            value: resolution,
        });
        const payload = {
            kind: 'no_op' as const,
            job_id: jobs[0].id,
            reason: 'no_eligible_blocks',
            resolved_input_fingerprint: await fingerprintJson(resolution),
            recorded_at: RECORDED_AT,
        };
        const output = await stageIndexedProcessingPhase(memory.store, resolved.root, resolved.locator, {
            phase: 'output',
            value: { ...payload, output_fingerprint: await fingerprintJson(payload) },
        });
        const completed = await stageIndexedProcessingNoOpCompletion(
            memory.store,
            output.root,
            output.locator,
            jobs[0].id,
        );
        const first = await loadIndexedProcessingJobState(memory.store, completed.root, jobs[0].id);
        const predecessor = await loadIndexedProcessingPredecessorEvidence(memory.store, completed.root, jobs[1].id);
        if (!first.completion) throw new Error('Actual predecessor completion missing');
        expect(first.completion.status).toBe('no_op');
        expect(predecessor?.job).toEqual(jobs[0]);
        expect(predecessor?.receipt.payload_fingerprint).toBe(await fingerprintJson(first.completion));
        expect(predecessor?.receipt.processing_operation?.result_fingerprint).toBe(
            predecessor?.receipt.payload_fingerprint,
        );
        const retained = await getPagedRecord(
            memory.store,
            completed.root.directories.operation_receipts,
            `processing:complete:${jobs[0].id}`,
        );
        if (retained?.storage !== 'record' || !predecessor) throw new Error('Predecessor exact receipt missing');
        memory.records.set(
            retained.content_hash,
            canonicalJsonContentBytes({
                ...predecessor.receipt,
                processing_operation: {
                    ...predecessor.receipt.processing_operation,
                    result_fingerprint: `sha256:${'0'.repeat(64)}`,
                },
            }),
        );
        await expect(
            loadIndexedProcessingPredecessorEvidence(memory.store, completed.root, jobs[1].id),
        ).rejects.toThrow();
    });

    it('requires explicit supersession of every pending obligation for an enabled policy transition and exact retry', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        if (!initial.locator) throw new Error('Pending policy fixture lacks its initial root');
        const processors = [
            document.processing.processors[0],
            { ...document.processing.processors[0], required: false },
        ];
        const policy = await stageIndexedProcessingPolicy(
            memory.store,
            initial.root,
            initial.locator,
            {
                operation_id: 'policy:two-obligations',
                expected_revision: initial.root.source.revision,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors,
            },
            async () => undefined,
        );
        const turn = userTurn('turn:policy-obligations');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:policy-obligations', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const appended = await stageIndexedRecordBatch(
            policy.root,
            {
                conversation_id: document.id,
                batch,
                options: {
                    operation_id: 'append:policy-obligations',
                    expected_revision: policy.root.source.revision,
                    payload_fingerprint: await fingerprintJson(batch),
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        if (!appended.locator) throw new Error('Pending policy fixture lacks its accepted root');
        const pending = await loadIndexedPendingProcessingJobs(memory.store, appended.root);
        expect(pending.jobs).toHaveLength(2);
        expect(pending.required_unresolved_job_count).toBe(1);
        const command = {
            operation_id: 'policy:replacement',
            expected_revision: appended.root.source.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [document.processing.processors[0]],
        };
        await expect(
            stageIndexedProcessingPolicy(memory.store, appended.root, appended.locator, command, async () => undefined),
        ).rejects.toThrow('explicit supersession of every unresolved job');
        await expect(
            stageIndexedProcessingPolicy(
                memory.store,
                appended.root,
                appended.locator,
                { ...command, supersede_job_ids: [pending.jobs[0].job.id], supersession_reason: 'reconfigured' },
                async () => undefined,
            ),
        ).rejects.toThrow('explicit supersession of every unresolved job');
        expect(
            (await loadIndexedPendingProcessingJobs(memory.store, appended.root)).jobs.map(({ job }) => job.id),
        ).toEqual(pending.jobs.map(({ job }) => job.id));
        const acceptedCommand = {
            ...command,
            supersede_job_ids: pending.jobs.map(({ job }) => job.id),
            supersession_reason: 'reconfigured',
        };
        const changed = await stageIndexedProcessingPolicy(
            memory.store,
            appended.root,
            appended.locator,
            acceptedCommand,
            async () => undefined,
        );
        expect((await loadIndexedPendingProcessingJobs(memory.store, changed.root)).unresolved_job_count).toBe(0);
        expect(changed.receipt.processing_operation?.superseded_job_ids).toEqual(acceptedCommand.supersede_job_ids);
        const retry = await stageIndexedProcessingPolicy(
            memory.store,
            changed.root,
            changed.locator,
            acceptedCommand,
            async () => undefined,
        );
        expect(retry.applied).toBe(false);
        expect(retry.receipt).toEqual(changed.receipt);
        expect(retry.locator).toEqual(changed.locator);
        await expect(
            stageIndexedProcessingPolicy(
                memory.store,
                changed.root,
                changed.locator,
                { ...acceptedCommand, supersession_reason: 'different' },
                async () => undefined,
            ),
        ).rejects.toThrow();
    });
    it(
        'queues a new manual job after more than 256 actual completed indexed jobs without scanning cold job records',
        async () => {
            const document = await configured();
            const memory = storage();
            const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
            if (!initial.locator) throw new Error('Historical queue fixture lacks its initial root');
            const manual = { ...document.processing.processors[0], scope: 'manual' as const };
            const policy = await stageIndexedProcessingPolicy(
                memory.store,
                initial.root,
                initial.locator,
                {
                    operation_id: 'operation:history-policy',
                    expected_revision: initial.root.source.revision,
                    recorded_at: RECORDED_AT,
                    enabled: true,
                    processors: [...document.processing.processors, manual],
                },
                async () => undefined,
            );
            let head = { root: policy.root, locator: policy.locator };
            for (let index = 0; index < 257; index++) {
                const turn = {
                    ...userTurn(`turn:history:${index}`),
                    blocks: [{ id: `block:history:${index}`, type: 'json' as const, value: { index } }],
                };
                const batch = {
                    turns: [turn],
                    context_entries:
                        index === 0
                            ? [
                                  {
                                      id: `entry:history:${index}`,
                                      type: 'source_turn' as const,
                                      turn_id: turn.id,
                                  },
                              ]
                            : [],
                };
                const appended = await stageIndexedRecordBatch(
                    head.root,
                    {
                        conversation_id: document.id,
                        batch,
                        options: {
                            operation_id: `operation:history:${index}`,
                            expected_revision: head.root.source.revision,
                            payload_fingerprint: await fingerprintJson(batch),
                            recorded_at: RECORDED_AT,
                        },
                    },
                    memory.store,
                );
                if (!appended.locator) throw new Error('Historical append lacks its root');
                const job = (await loadIndexedPendingProcessingJobs(memory.store, appended.root)).jobs[0]?.job;
                if (!job) throw new Error('Historical append lacks its actual job');
                const selected = await loadIndexedProcessingSelectedContext(
                    memory.store,
                    appended.root,
                    appended.locator,
                );
                const resolution = await resolveIndexedProcessingTextInput(selected, job, RECORDED_AT);
                const resolved = await stageIndexedProcessingPhase(memory.store, appended.root, appended.locator, {
                    phase: 'resolve',
                    value: resolution,
                });
                const payload = {
                    kind: 'no_op' as const,
                    job_id: job.id,
                    reason: 'no_eligible_blocks',
                    resolved_input_fingerprint: await fingerprintJson(resolution),
                    recorded_at: RECORDED_AT,
                };
                const output = await stageIndexedProcessingPhase(memory.store, resolved.root, resolved.locator, {
                    phase: 'output',
                    value: { ...payload, output_fingerprint: await fingerprintJson(payload) },
                });
                const completed = await stageIndexedProcessingNoOpCompletion(
                    memory.store,
                    output.root,
                    output.locator,
                    job.id,
                );
                head = { root: completed.root, locator: completed.locator };
            }
            const settled = await loadIndexedPendingProcessingJobs(memory.store, head.root);
            expect(settled.job_count).toBe(257);
            expect(settled.unresolved_job_count).toBe(0);
            const selected = await loadIndexedProcessingSelectedContext(memory.store, head.root, head.locator);
            const command = {
                operation_id: 'operation:history-manual',
                expected_revision: head.root.source.revision,
                expected_context_revision: selected.context.revision,
                recorded_at: RECORDED_AT,
                processor_id: 'externalize-text',
                scope: 'manual' as const,
                selected_entry_ids: ['entry:history:0'],
            };
            memory.reads.length = 0;
            const queued = await stageIndexedProcessingQueue(memory.store, head.root, head.locator, command);
            expect(queued.applied).toBe(true);
            const pending = await loadIndexedPendingProcessingJobs(memory.store, queued.root);
            expect(pending.job_count).toBe(258);
            expect(pending.unresolved_job_count).toBe(1);
            expect(memory.reads.filter((read) => read.startsWith('processing_records:')).length).toBeLessThanOrEqual(8);
            expect(
                (await stageIndexedProcessingQueue(memory.store, queued.root, queued.locator, command)).applied,
            ).toBe(false);
        },
        WORKING_SET_CAPACITY_TEST_TIMEOUT_MS,
    );

    it('binds a selected on-budget job, its retry and measured target coverage to one accepted indexed policy', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        if (!initial.locator) throw new Error('Indexed policy fixture lacks a root');
        const policyCommand = {
            operation_id: 'operation:selected-policy',
            expected_revision: initial.root.source.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                ...document.processing.processors,
                {
                    id: 'semantic-summary',
                    version: '1',
                    scope: 'on_budget' as const,
                    config: {},
                    required: true,
                    failure_behavior: 'block' as const,
                },
            ],
            budget: { max_input_tokens: 1, output_reserve_tokens: 0, measurement_policy: 'exact_only' as const },
        };
        const supported: string[] = [];
        const assertSupported = async (configuration: { id: string }) => {
            supported.push(configuration.id);
        };
        const policy = await stageIndexedProcessingPolicy(
            memory.store,
            initial.root,
            initial.locator,
            policyCommand,
            assertSupported,
        );
        if (!policy.locator) throw new Error('Indexed policy was not staged');
        expect(supported).toEqual(['externalize-text', 'semantic-summary']);
        const selectedPolicyBytes = memory.records.get(policy.root.processing_header.content_hash);
        if (!selectedPolicyBytes) throw new Error('Actual selected custom policy lacks its immutable header');
        const selectedPolicy = IndexedConversationProcessingHeaderSchema.parse(
            JSON.parse(new TextDecoder().decode(selectedPolicyBytes)),
        );
        expect(selectedPolicy.selected_policy_operation_id).toBe(policyCommand.operation_id);
        expect(selectedPolicy.selected_policy_origin).toBe('native_registry');
        // The portable core retains a host-validated registered extension. This is not native
        // processor execution capability: the native host predicate still refuses this profile.
        expect(supportsIndexedRegisteredProcessingPolicy(selectedPolicy)).toBe(false);
        await expect(assertIndexedCurrentPolicy(memory.store, policy.root, selectedPolicy)).resolves.toBeUndefined();
        const selectedPolicyReceipt = await getPagedRecord(
            memory.store,
            policy.root.directories.operation_receipts,
            policyCommand.operation_id,
        );
        if (selectedPolicyReceipt?.storage !== 'record')
            throw new Error('Actual selected custom policy receipt is absent');
        const receiptBytes = memory.records.get(selectedPolicyReceipt.content_hash);
        if (!receiptBytes) throw new Error('Actual selected custom policy receipt bytes are absent');
        memory.records.delete(selectedPolicyReceipt.content_hash);
        await expect(assertIndexedCurrentPolicy(memory.store, policy.root, selectedPolicy)).rejects.toThrow(
            'Indexed record absent',
        );
        memory.records.set(selectedPolicyReceipt.content_hash, receiptBytes);

        expect(
            (
                await stageIndexedProcessingPolicy(
                    memory.store,
                    policy.root,
                    policy.locator,
                    policyCommand,
                    assertSupported,
                )
            ).applied,
        ).toBe(false);
        expect(supported).toHaveLength(2);
        const turn = userTurn('turn:budget', 'block:budget');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:budget', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const appended = await stageIndexedRecordBatch(
            policy.root,
            {
                conversation_id: document.id,
                batch,
                options: {
                    operation_id: 'operation:budget-source',
                    expected_revision: policy.root.source.revision,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            memory.store,
        );
        if (!appended.locator) throw new Error('Indexed budget source was not accepted');
        const selected = await loadIndexedProcessingSelectedContext(memory.store, appended.root, appended.locator);
        const target = `sha256:${'a'.repeat(64)}`;
        const queueCommand = {
            operation_id: 'operation:budget-queue',
            expected_revision: appended.root.source.revision,
            expected_context_revision: selected.context.revision,
            recorded_at: RECORDED_AT,
            processor_id: 'semantic-summary',
            scope: 'on_budget' as const,
            selected_entry_ids: ['entry:budget'],
            target_fingerprint: target,
        };
        const queued = await stageIndexedProcessingQueue(memory.store, appended.root, appended.locator, queueCommand);
        if (!queued.locator) throw new Error('Indexed budget job was not accepted');
        expect(queued.job).toMatchObject({ scope: 'on_budget', target_fingerprint: target, required: true });
        expect((await loadIndexedPendingProcessingJobs(memory.store, queued.root)).unresolved_job_count).toBe(2);
        expect(
            (await stageIndexedProcessingQueue(memory.store, queued.root, queued.locator, queueCommand)).job,
        ).toEqual(queued.job);
        await expect(
            stageIndexedProcessingQueue(memory.store, queued.root, queued.locator, {
                ...queueCommand,
                target_fingerprint: `sha256:${'b'.repeat(64)}`,
            }),
        ).rejects.toThrow('retry conflicts');
        const coverageCommand = {
            operation_id: 'operation:budget-coverage',
            expected_revision: queued.root.source.revision,
            target_fingerprint: target,
            measured_input_tokens: 2,
            tokenizer_id: 'tokenizer:exact',
            measurement_fingerprint: `sha256:${'c'.repeat(64)}`,
            recorded_at: RECORDED_AT,
        };
        const coverage = await stageIndexedProcessingCoverage(
            memory.store,
            queued.root,
            queued.locator,
            coverageCommand,
        );
        expect(coverage.coverage.status).toBe('pending');
        const foreignTarget = await stageIndexedProcessingCoverage(memory.store, queued.root, queued.locator, {
            ...coverageCommand,
            operation_id: 'operation:foreign-target',
            target_fingerprint: `sha256:${'d'.repeat(64)}`,
        });
        expect(foreignTarget.coverage.status).toBe('blocked');
    });

    it('derives a manual text job from one accepted current turn and replays its exact source', async () => {
        const document = await configured();
        const memory = storage();
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        if (!initial.locator) throw new Error('Indexed manual policy fixture lacks a root');
        const policy = await stageIndexedProcessingPolicy(
            memory.store,
            initial.root,
            initial.locator,
            {
                operation_id: 'operation:manual-policy',
                expected_revision: initial.root.source.revision,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors: [
                    {
                        id: 'externalize-text',
                        version: '1',
                        scope: 'manual',
                        config: {},
                        required: true,
                        failure_behavior: 'block',
                    },
                ],
            },
            async () => {},
        );
        if (!policy.locator) throw new Error('Indexed manual policy lacks an accepted locator');
        const turn = userTurn('turn:manual', 'block:manual');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:manual', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const appended = await stageIndexedRecordBatch(
            policy.root,
            {
                conversation_id: document.id,
                batch,
                options: {
                    operation_id: 'operation:manual-source',
                    expected_revision: policy.root.source.revision,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            memory.store,
        );
        if (!appended.locator) throw new Error('Indexed manual source lacks an accepted locator');
        expect(
            (await loadIndexedProcessingAppendAcceptance(memory.store, appended.root, 'operation:manual-source')).jobs,
        ).toHaveLength(0);
        const selected = await loadIndexedProcessingSelectedContext(memory.store, appended.root, appended.locator);
        const command = {
            operation_id: 'operation:manual-queue',
            expected_revision: appended.root.source.revision,
            expected_context_revision: selected.context.revision,
            recorded_at: RECORDED_AT,
            processor_id: 'externalize-text',
            scope: 'manual' as const,
            selected_entry_ids: ['entry:manual'],
        };
        const queued = await stageIndexedProcessingQueue(memory.store, appended.root, appended.locator, command);
        if (!queued.locator) throw new Error('Indexed manual queue lacks an accepted locator');
        expect(queued.job.scope).toBe('manual');
        expect(
            (await loadIndexedProcessingQueuedJobAcceptance(memory.store, queued.root, queued.job.id)).command,
        ).toEqual(command);
        const current = await loadIndexedProcessingSelectedContext(memory.store, queued.root, queued.locator);
        const resolution = await resolveIndexedProcessingTextInput(current, queued.job, RECORDED_AT);
        expect(resolution.entry_ids).toEqual(['entry:manual']);
        expect((await stageIndexedProcessingQueue(memory.store, queued.root, queued.locator, command)).job).toEqual(
            queued.job,
        );
        await expect(
            stageIndexedProcessingQueue(memory.store, queued.root, queued.locator, {
                ...command,
                selected_entry_ids: ['entry:other'],
            }),
        ).rejects.toThrow('retry conflicts');
        const retainedCommand = await getPagedRecord(
            memory.store,
            queued.root.directories.processing_records,
            JSON.stringify(['selected_queue_commands', command.operation_id]),
        );
        if (retainedCommand?.storage !== 'record') throw new Error('Retained queue command is absent');
        memory.records.set(
            retainedCommand.content_hash,
            canonicalJsonContentBytes({
                ...command,
                selected_entry_ids: ['entry:forged'],
            }),
        );
        await expect(
            loadIndexedProcessingQueuedJobAcceptance(memory.store, queued.root, queued.job.id),
        ).rejects.toThrow();
    });

    it('retains mixed trigger policy data but rejects unsupported native preparation and queue before publication', async () => {
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
        const turn = userTurn('turn:mixed-policy');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:mixed-policy', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const accepted = await appendConversationRecordsWithProcessing(mixed, batch, {
            operation_id: 'append:mixed-policy',
            expected_revision: mixed.revision,
            payload_fingerprint: await fingerprintJson(batch),
            recorded_at: RECORDED_AT,
        });
        const memory = storage();
        const staged = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        const bytes = await memory.store.readRecord({
            storage: 'record',
            kind: 'processing_header',
            id: staged.root.source.conversation_id,
            ...staged.root.processing_header,
        });
        const header = IndexedConversationProcessingHeaderSchema.parse(JSON.parse(new TextDecoder().decode(bytes)));
        expect(header.processors).toEqual(mixed.processing.processors);
        expect(header.budget).toEqual(mixed.processing.budget);
        const pendingBefore = await loadIndexedPendingProcessingJobs(memory.store, staged.root);
        expect(pendingBefore.jobs).toHaveLength(1);
        const recordsBefore = memory.records.size;
        const pagesBefore = memory.pages.size;
        await expect(assertIndexedCurrentPolicy(memory.store, staged.root, header)).rejects.toThrow(
            'registered bounded ordered text or whole-exchange policy',
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(memory.store, staged.root, staged.locator),
        ).rejects.toThrow('registered bounded ordered text or whole-exchange policy');
        await expect(
            stageIndexedProcessingQueue(memory.store, staged.root, staged.locator, {
                operation_id: 'queue:unsupported-budget',
                expected_revision: staged.root.source.revision,
                expected_context_revision: accepted.document.context.revision,
                processor_id: 'future-budget',
                scope: 'on_budget',
                selected_entry_ids: ['entry:mixed-policy'],
                target_fingerprint: 'sha256:measured-target',
                recorded_at: RECORDED_AT,
            }),
        ).rejects.toThrow('registered bounded ordered text or whole-exchange policy');
        expect(memory.records.size).toBe(recordsBefore);
        expect(memory.pages.size).toBe(pagesBefore);
        expect(await loadIndexedPendingProcessingJobs(memory.store, staged.root)).toEqual(pendingBefore);
        expect(
            await getPagedRecord(memory.store, staged.root.directories.operation_receipts, 'queue:unsupported-budget'),
        ).toBeUndefined();
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
