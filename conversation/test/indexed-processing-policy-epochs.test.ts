import { Ajv2020 } from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { fingerprintJson } from '../src/identity.js';
import {
    assertIndexedCurrentPolicy,
    auditIndexedProcessingJobEvidence,
    auditIndexedProcessingPolicyEpoch,
    authenticateIndexedCurrentPolicy,
    type IndexedConversationRecordStore,
    IndexedProcessingPolicyEpochSchema,
    loadIndexedPendingProcessingJobs,
    loadIndexedProcessingJobState,
    loadIndexedSettledProcessingSelectedContext,
    loadRecord,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingPolicy,
    stageIndexedRecordBatch,
    stageRecord,
} from '../src/indexed-conversation.js';
import { INDEXED_CONVERSATION_UPGRADE_PROFILE } from '../src/indexed-upgrade-constants.js';
import { finishIndexedConversationUpgrade } from '../src/indexed-upgrade-finish.js';
import { INDEXED_UPGRADE_ACTIVE_LIMITS, INDEXED_UPGRADE_STEP_LIMITS } from '../src/indexed-upgrade-io.js';
import { beginIndexedConversationUpgrade } from '../src/indexed-upgrade-progress.js';
import { advanceIndexedConversationUpgrade } from '../src/indexed-upgrade-step.js';
import { ProcessingPolicyCommandJsonSchema, ProcessingPolicyGenesisJsonSchema } from '../src/json-schema.js';
import { getPagedRecord, putPagedRecord, removePagedRecord } from '../src/paged-record-index.js';
import {
    type ProcessingPolicyCommand,
    type ProcessingStore,
    runProcessingJob,
    setProcessingPolicy,
} from '../src/processing.js';
import { appendConversationRecordsWithProcessing } from '../src/runtime.js';
import {
    IndexedConversationProcessingHeaderSchema,
    type IndexedConversationRoot,
} from '../src/schemas/indexed-head.js';
import { ProcessingPolicyCommandSchema, ProcessingPolicyGenesisSchema } from '../src/schemas/processing-policy.js';
import { conversationDocumentFromJson, conversationDocumentToJson } from '../src/serialization.js';
import { parseConversationDocument } from '../src/validation.js';
import { emptyDocument, RECORDED_AT, userTurn } from './fixtures.js';

const processorA = {
    id: 'externalize-text',
    version: '1',
    scope: 'on_append',
    config: {},
    required: true,
    failure_behavior: 'block',
} satisfies ProcessingPolicyCommand['processors'][number];
const processorB = { ...processorA, required: false };

function memoryStore() {
    const pages = new Map<string, Uint8Array>();
    const records = new Map<string, Uint8Array>();
    const reads: string[] = [];
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Owned page missing');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            reads.push(`${ref.kind}:${ref.id}`);
            const bytes = records.get(ref.content_hash);
            if (!bytes) throw new Error('Owned record missing');
            return Uint8Array.from(bytes);
        },
        async writeRecord(ref, bytes) {
            expect((await hashContentBytes(bytes)).content_hash).toBe(ref.content_hash);
            records.set(ref.content_hash, Uint8Array.from(bytes));
        },
    };
    return { store, records, reads, pages };
}

async function appendNative(memory: ReturnType<typeof memoryStore>, root: IndexedConversationRoot) {
    const turn = userTurn('turn:epoch');
    const batch = {
        turns: [turn],
        context_entries: [{ id: 'entry:epoch', type: 'source_turn' as const, turn_id: turn.id }],
    };
    const appended = await stageIndexedRecordBatch(
        root,
        {
            conversation_id: root.source.conversation_id,
            batch,
            options: {
                operation_id: 'append:epoch',
                expected_revision: root.source.revision,
                payload_fingerprint: await fingerprintJson(batch),
                recorded_at: RECORDED_AT,
            },
        },
        memory.store,
    );
    if (!appended.locator) throw new Error('Actual accepted append root missing');
    const pending = await loadIndexedPendingProcessingJobs(memory.store, appended.root);
    const job = pending.jobs[0]?.job;
    if (!job || pending.jobs.length !== 1) throw new Error('Actual accepted epoch job absent');
    return { appended: { ...appended, locator: appended.locator }, job };
}

async function nativeTransition(genesis = false) {
    const memory = memoryStore();
    const initial = emptyDocument('conversation:policy-epochs');
    const source = genesis
        ? parseConversationDocument({
              ...initial,
              processing: { enabled: true, policy_revision: 0, processors: [processorA] },
          })
        : (
              await setProcessingPolicy(initial, {
                  operation_id: 'policy:A',
                  expected_revision: initial.revision,
                  recorded_at: RECORDED_AT,
                  enabled: true,
                  processors: [processorA],
              })
          ).document;
    const snapshot = await stageIndexedConversationSnapshot(source, undefined, memory.store);
    const { appended, job } = await appendNative(memory, snapshot.root);
    const transitioned = await stageIndexedProcessingPolicy(
        memory.store,
        appended.root,
        appended.locator,
        {
            operation_id: 'policy:B',
            expected_revision: appended.root.source.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [processorB],
            supersede_job_ids: [job.id],
            supersession_reason: 'genuine-policy-change',
        },
        async () => undefined,
    );
    return { ...memory, source, job, transitioned };
}

async function replaceEpoch(f: Awaited<ReturnType<typeof nativeTransition>>, value: unknown) {
    const id = String(f.job.policy_revision);
    const ref = await stageRecord(f.store, 'processing_records', id, IndexedProcessingPolicyEpochSchema.parse(value));
    const processing = await putPagedRecord(
        f.store,
        f.transitioned.root.directories.processing_records,
        JSON.stringify(['policy_epochs', id]),
        ref,
        'replace',
    );
    return {
        ...f.transitioned.root,
        directories: { ...f.transitioned.root.directories, processing_records: processing },
    };
}

async function materializedHistory(genesis = false) {
    const initial = emptyDocument('conversation:materialized-epochs');
    let current = genesis
        ? parseConversationDocument({
              ...initial,
              processing: { enabled: true, policy_revision: 0, processors: [processorA] },
          })
        : (
              await setProcessingPolicy(emptyDocument('conversation:materialized-epochs'), {
                  operation_id: 'policy:A',
                  expected_revision: 0,
                  recorded_at: RECORDED_AT,
                  enabled: true,
                  processors: [processorA],
              })
          ).document;
    const sourceA = structuredClone(current);
    const turn = userTurn('turn:materialized-epoch');
    const batch = {
        turns: [turn],
        context_entries: [{ id: 'entry:materialized-epoch', type: 'source_turn' as const, turn_id: turn.id }],
    };
    current = (
        await appendConversationRecordsWithProcessing(current, batch, {
            operation_id: 'append:materialized-epoch',
            expected_revision: current.revision,
            recorded_at: RECORDED_AT,
            payload_fingerprint: await fingerprintJson(batch),
        })
    ).document;
    const job = Object.values(current.processing.jobs ?? {})[0];
    if (!job) throw new Error('Actual accepted materialized job absent');
    const store: ProcessingStore = {
        async load() {
            return structuredClone(current);
        },
        async commit(expected, next) {
            if (expected !== current.revision) return false;
            current = next;
            return true;
        },
    };
    await runProcessingJob(
        store,
        { resolve: () => ({ run: async () => ({ kind: 'no_op', reason: 'actual-no-op' }) }) },
        job.id,
        'attempt:epoch',
        () => RECORDED_AT,
    );
    const predecessor = structuredClone(current);
    current = (
        await setProcessingPolicy(current, {
            operation_id: 'policy:B',
            expected_revision: current.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [processorB],
        })
    ).document;
    return { document: current, sourceA, predecessor, job };
}

describe('authenticated historical processing policy epochs', () => {
    it('audits an old native job against its accepted command while current counters stay current', async () => {
        const f = await nativeTransition();
        f.reads.length = 0;
        const evidence = await loadIndexedProcessingJobState(f.store, f.transitioned.root, f.job.id);
        expect(evidence.configuration).toEqual(processorA);
        expect(evidence.job.policy_revision).toBe(1);
        expect(evidence.header.policy_revision).toBe(2);
        expect(evidence.header.processors).toEqual([processorB]);
        expect(evidence.header.required_job_count).toBe(0);
        expect(evidence.header.unresolved_job_count).toBe(0);
        expect(evidence.supersession?.policy_operation_id).toBe('policy:B');
        expect(f.reads).toContain('operation_receipts:policy:A');
        expect(f.reads.length).toBeLessThan(20);
    });

    it('rejects tampered epoch mapping, command hash, future acceptance and processor stage', async () => {
        const f = await nativeTransition();
        const epochDescriptor = await getPagedRecord(
            f.store,
            f.transitioned.root.directories.processing_records,
            JSON.stringify(['policy_epochs', '1']),
        );
        const epoch = await loadRecord(f.store, epochDescriptor, IndexedProcessingPolicyEpochSchema);
        if (epoch.kind !== 'accepted_command') throw new Error('Expected actual accepted command epoch');
        const wrong = await replaceEpoch(f, {
            ...epoch,
            operation_id: 'policy:B',
            receipt_fingerprint: await fingerprintJson(f.transitioned.receipt),
        });
        await expect(loadIndexedProcessingJobState(f.store, wrong, f.job.id)).rejects.toThrow('exact accepted command');
        const command = await getPagedRecord(
            f.store,
            f.transitioned.root.directories.processing_records,
            JSON.stringify(['selected_policy_commands', 'policy:A']),
        );
        if (command?.storage !== 'record') throw new Error('Actual accepted A command absent');
        const bytes = f.records.get(command.content_hash);
        if (!bytes) throw new Error('Actual accepted A bytes absent');
        f.records.set(command.content_hash, canonicalJsonContentBytes({ changed: true }));
        await expect(loadIndexedProcessingJobState(f.store, f.transitioned.root, f.job.id)).rejects.toThrow();
        f.records.set(command.content_hash, bytes);
        const header = await loadRecord(
            f.store,
            { storage: 'record', kind: 'processing_header', id: f.source.id, ...f.transitioned.root.processing_header },
            IndexedConversationProcessingHeaderSchema,
        );
        await expect(
            auditIndexedProcessingJobEvidence(f.store, f.transitioned.root, { ...f.job, enqueue_revision: 0 }, header),
        ).rejects.toThrow('retained policy stage');
        await expect(
            auditIndexedProcessingJobEvidence(f.store, f.transitioned.root, { ...f.job, processor_index: 1 }, header),
        ).rejects.toThrow('retained policy stage');
    });

    it('retains actual materialized policy commands and old completed jobs after policy A to B', async () => {
        const original = await materializedHistory();
        const receipt = original.document.operation_receipts['policy:A'];
        const command = receipt.processing_operation?.policy_command;
        if (!command) throw new Error('Actual accepted materialized command absent');
        expect(await fingerprintJson(command)).toBe(receipt.payload_fingerprint);
        const f = memoryStore();
        const snapshot = await stageIndexedConversationSnapshot(original.document, undefined, f.store);
        const old = await loadIndexedProcessingJobState(f.store, snapshot.root, original.job.id);
        expect(old.configuration).toEqual(processorA);
        expect(old.completion?.status).toBe('no_op');
        expect(old.header.processors).toEqual([processorB]);
        expect(old.header.policy_revision).toBe(2);
    });

    it('recovers a legacy old epoch only from the exact authenticated original policy source', async () => {
        const original = await materializedHistory();
        const receipt = original.document.operation_receipts['policy:A'];
        if (!receipt.processing_operation) throw new Error('Actual policy receipt missing');
        const { policy_command: _captured, ...detail } = receipt.processing_operation;
        const legacyReceipt = { ...receipt, processing_operation: detail };
        const document = parseConversationDocument({
            ...original.document,
            operation_receipts: { ...original.document.operation_receipts, 'policy:A': legacyReceipt },
        });
        const source = parseConversationDocument({
            ...original.sourceA,
            operation_receipts: { ...original.sourceA.operation_receipts, 'policy:A': legacyReceipt },
        });
        const missing = memoryStore();
        await expect(stageIndexedConversationSnapshot(document, undefined, missing.store)).rejects.toThrow(
            'authenticated original policy evidence',
        );
        expect(missing.records.size).toBe(0);
        expect(missing.pages.size).toBe(0);
        const f = memoryStore();
        const requests: { conversation_id: string; revision: number }[] = [];
        const snapshot = await stageIndexedConversationSnapshot(document, undefined, f.store, async (requested) => {
            requests.push(requested);
            return source;
        });
        expect(requests).toEqual([{ conversation_id: document.id, revision: receipt.result_revision }]);
        expect((await loadIndexedProcessingJobState(f.store, snapshot.root, original.job.id)).configuration).toEqual(
            processorA,
        );
        for (const returned of [
            { ...source, revision: source.revision + 1 },
            { ...source, processing: { ...source.processing, policy_revision: source.processing.policy_revision + 1 } },
            { ...source, processing: { ...source.processing, processors: [processorB] } },
            { ...source, id: 'conversation:foreign' },
        ]) {
            const rejected = memoryStore();
            await expect(
                stageIndexedConversationSnapshot(document, undefined, rejected.store, async () => returned),
            ).rejects.toThrow();
            expect(rejected.records.size).toBe(0);
            expect(rejected.pages.size).toBe(0);
        }
    });

    it('reconstructs retained materialized commands through bounded explicit upgrade receipt steps', async () => {
        const original = await materializedHistory();
        const f = memoryStore();
        const snapshot = await stageIndexedConversationSnapshot(original.document, undefined, f.store);
        let records = snapshot.root.directories.processing_records;
        for (const [family, id] of [
            ['selected_policy_commands', 'policy:A'],
            ['selected_policy_commands', 'policy:B'],
            ['policy_epochs', '1'],
            ['policy_epochs', '2'],
        ]) {
            records = (await removePagedRecord(f.store, records, JSON.stringify([family, id]))).root;
        }
        const predecessor = {
            ...snapshot.root,
            directories: { ...snapshot.root.directories, processing_records: records },
        };
        const ref = await stageRecord(f.store, 'root', predecessor.source.conversation_id, predecessor);
        const command = {
            version: 1 as const,
            profile: INDEXED_CONVERSATION_UPGRADE_PROFILE,
            operation_id: 'upgrade:policy-history',
            source: predecessor.source,
            predecessor_root: { content_hash: ref.content_hash, size_bytes: ref.size_bytes },
            recorded_at: RECORDED_AT,
        };
        let state = await beginIndexedConversationUpgrade(f.store, command);
        let steps = 0;
        while (state.progress.phase !== 'complete') {
            if (++steps > 200) throw new Error('Bounded upgrade failed to terminate');
            const limits =
                state.progress.phase === 'active_window' ? INDEXED_UPGRADE_ACTIVE_LIMITS : INDEXED_UPGRADE_STEP_LIMITS;
            const next = await advanceIndexedConversationUpgrade(f.store, command, state.locator);
            expect(next.usage.record_reads).toBeLessThanOrEqual(limits.record_reads);
            expect(next.usage.page_reads).toBeLessThanOrEqual(limits.page_reads);
            expect(next.usage.bytes).toBeLessThanOrEqual(limits.bytes);
            state = next;
        }
        const accepted = await finishIndexedConversationUpgrade(
            f.store,
            predecessor,
            command.predecessor_root,
            command,
            state.locator,
        );
        const old = await loadIndexedProcessingJobState(f.store, accepted.root, original.job.id);
        expect(old.configuration).toEqual(processorA);
        expect(old.header.policy_revision).toBe(2);
        expect(await auditIndexedProcessingPolicyEpoch(f.store, accepted.root, 2)).toMatchObject({
            processors: [processorB],
        });
    });

    it('retains authentic unsupported pluggable policy through bounded storage upgrade without granting native readiness', async () => {
        const policy = await setProcessingPolicy(emptyDocument('conversation:portable-upgrade'), {
            operation_id: 'policy:portable-extension',
            expected_revision: 0,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                {
                    id: 'portable-custom-processor',
                    version: '7',
                    scope: 'manual',
                    config: { instruction: 'Retain portable configuration' },
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const f = memoryStore();
        const snapshot = await stageIndexedConversationSnapshot(policy.document, undefined, f.store);
        const header = await loadRecord(
            f.store,
            {
                storage: 'record',
                kind: 'processing_header',
                id: snapshot.root.source.conversation_id,
                ...snapshot.root.processing_header,
            },
            IndexedConversationProcessingHeaderSchema,
        );
        expect(header.selected_policy_origin).toBe('materialized');
        await expect(authenticateIndexedCurrentPolicy(f.store, snapshot.root, header)).resolves.toBe('materialized');
        await expect(
            authenticateIndexedCurrentPolicy(f.store, snapshot.root, {
                ...header,
                selected_policy_origin: 'native_registry',
            }),
        ).rejects.toThrow('immutable root descriptor');
        await expect(assertIndexedCurrentPolicy(f.store, snapshot.root, header)).rejects.toThrow('registered bounded');
        const command = {
            version: 1 as const,
            profile: INDEXED_CONVERSATION_UPGRADE_PROFILE,
            operation_id: 'upgrade:portable-extension',
            source: snapshot.root.source,
            predecessor_root: snapshot.locator,
            recorded_at: RECORDED_AT,
        };
        let state = await beginIndexedConversationUpgrade(f.store, command);
        let steps = 0;
        while (state.progress.phase !== 'complete') {
            if (++steps > 200) throw new Error('Portable policy upgrade failed to terminate');
            const limits =
                state.progress.phase === 'active_window' ? INDEXED_UPGRADE_ACTIVE_LIMITS : INDEXED_UPGRADE_STEP_LIMITS;
            const next = await advanceIndexedConversationUpgrade(f.store, command, state.locator);
            expect(next.usage.record_reads).toBeLessThanOrEqual(limits.record_reads);
            expect(next.usage.page_reads).toBeLessThanOrEqual(limits.page_reads);
            expect(next.usage.bytes).toBeLessThanOrEqual(limits.bytes);
            state = next;
        }
        const accepted = await finishIndexedConversationUpgrade(
            f.store,
            snapshot.root,
            snapshot.locator,
            command,
            state.locator,
        );
        expect(accepted.root.source.revision).toBe(policy.document.revision + 1);
        expect(await auditIndexedProcessingPolicyEpoch(f.store, accepted.root, 1)).toMatchObject({
            enabled: true,
            processors: policy.document.processing.processors,
        });
        const finalHeader = await loadRecord(
            f.store,
            {
                storage: 'record',
                kind: 'processing_header',
                id: accepted.root.source.conversation_id,
                ...accepted.root.processing_header,
            },
            IndexedConversationProcessingHeaderSchema,
        );
        await expect(authenticateIndexedCurrentPolicy(f.store, accepted.root, finalHeader)).resolves.toBe(
            'materialized',
        );
        const writes = { records: f.records.size, pages: f.pages.size };
        await expect(
            loadIndexedSettledProcessingSelectedContext(f.store, accepted.root, accepted.locator),
        ).rejects.toThrow('registered bounded');
        expect({ records: f.records.size, pages: f.pages.size }).toEqual(writes);
    });

    it('pins genuine genesis and rejects foreign or tampered initial-header witnesses', async () => {
        const f = await nativeTransition(true);
        expect((await loadIndexedProcessingJobState(f.store, f.transitioned.root, f.job.id)).configuration).toEqual(
            processorA,
        );
        const descriptor = await getPagedRecord(
            f.store,
            f.transitioned.root.directories.processing_records,
            JSON.stringify(['policy_epochs', '0']),
        );
        const epoch = await loadRecord(f.store, descriptor, IndexedProcessingPolicyEpochSchema);
        if (epoch.kind !== 'genesis') throw new Error('Genuine genesis witness absent');
        const foreign = await replaceEpoch(f, {
            ...epoch,
            source: { ...epoch.source, conversation_id: 'conversation:foreign' },
        });
        await expect(loadIndexedProcessingJobState(f.store, foreign, f.job.id)).rejects.toThrow(
            'different accepted source',
        );
        const bytes = f.records.get(epoch.processing_header.content_hash);
        if (!bytes) throw new Error('Genuine initial header absent');
        f.records.set(epoch.processing_header.content_hash, canonicalJsonContentBytes({ invented: true }));
        await expect(loadIndexedProcessingJobState(f.store, f.transitioned.root, f.job.id)).rejects.toThrow();
        f.records.set(epoch.processing_header.content_hash, bytes);
        const removed = await removePagedRecord(
            f.store,
            f.transitioned.root.directories.processing_records,
            JSON.stringify(['policy_epochs', '0']),
        );
        await expect(
            loadIndexedProcessingJobState(
                f.store,
                {
                    ...f.transitioned.root,
                    directories: { ...f.transitioned.root.directories, processing_records: removed.root },
                },
                f.job.id,
            ),
        ).rejects.toThrow('authenticated original policy evidence');
    });

    it('retains real startup epoch0 across first change, JSON persistence and snapshot without an external resolver', async () => {
        const original = await materializedHistory(true);
        const receipt = original.document.operation_receipts['policy:B'];
        const captured = receipt.processing_operation?.policy_genesis;
        if (!receipt.processing_operation || !captured) throw new Error('Genuine startup policy capture absent');
        expect(captured).toEqual({
            source: { conversation_id: original.document.id, revision: receipt.base_revision },
            policy: { enabled: true, policy_revision: 0, processors: [processorA] },
        });
        expect(captured?.policy).not.toHaveProperty('jobs');
        expect(captured?.policy).not.toHaveProperty('job_count');
        const reloaded = conversationDocumentFromJson(conversationDocumentToJson(original.document));
        const f = memoryStore();
        const snapshot = await stageIndexedConversationSnapshot(reloaded, undefined, f.store);
        const old = await loadIndexedProcessingJobState(f.store, snapshot.root, original.job.id);
        expect(old.job.policy_revision).toBe(0);
        expect(old.configuration).toEqual(processorA);
        expect(old.completion?.status).toBe('no_op');
        expect(old.header.policy_revision).toBe(1);
        expect(old.header.processors).toEqual([processorB]);
        expect(await auditIndexedProcessingPolicyEpoch(f.store, snapshot.root, 0)).toMatchObject({
            policy_revision: 0,
            processors: [processorA],
            first_transition_revision: receipt.base_revision,
        });
        for (const changed of [
            { ...captured, source: { conversation_id: 'conversation:foreign', revision: receipt.base_revision } },
            { ...captured, source: { conversation_id: original.document.id, revision: receipt.base_revision + 1 } },
            { ...captured, policy: { enabled: true, policy_revision: 0, processors: [processorB] } },
        ]) {
            const rejected = memoryStore();
            await expect(
                stageIndexedConversationSnapshot(
                    {
                        ...reloaded,
                        operation_receipts: {
                            ...reloaded.operation_receipts,
                            'policy:B': {
                                ...receipt,
                                processing_operation: {
                                    ...receipt.processing_operation,
                                    policy_genesis: ProcessingPolicyGenesisSchema.parse(changed),
                                },
                            },
                        },
                    },
                    undefined,
                    rejected.store,
                ),
            ).rejects.toThrow();
            expect(rejected.records.size).toBe(0);
            expect(rejected.pages.size).toBe(0);
        }
    });

    it('recovers old startup epoch0 only from the authenticated first-transition predecessor', async () => {
        const original = await materializedHistory(true);
        const receipt = original.document.operation_receipts['policy:B'];
        if (!receipt.processing_operation) throw new Error('Genuine first accepted policy receipt missing');
        const { policy_genesis: _capture, ...detail } = receipt.processing_operation;
        const legacy = parseConversationDocument({
            ...original.document,
            operation_receipts: {
                ...original.document.operation_receipts,
                'policy:B': { ...receipt, processing_operation: detail },
            },
        });
        const missing = memoryStore();
        await expect(stageIndexedConversationSnapshot(legacy, undefined, missing.store)).rejects.toThrow(
            'authenticated original predecessor policy',
        );
        expect(missing.records.size).toBe(0);
        expect(missing.pages.size).toBe(0);
        const requests: { conversation_id: string; revision: number }[] = [];
        const f = memoryStore();
        const snapshot = await stageIndexedConversationSnapshot(legacy, undefined, f.store, async (source) => {
            requests.push(source);
            return original.predecessor;
        });
        expect(requests).toEqual([{ conversation_id: legacy.id, revision: receipt.base_revision }]);
        expect((await loadIndexedProcessingJobState(f.store, snapshot.root, original.job.id)).configuration).toEqual(
            processorA,
        );
        for (const changed of [
            { ...original.predecessor, revision: original.predecessor.revision + 1 },
            { ...original.predecessor, id: 'conversation:foreign' },
            { ...original.predecessor, processing: { ...original.predecessor.processing, policy_revision: 1 } },
            { ...original.predecessor, processing: { ...original.predecessor.processing, processors: [processorB] } },
        ]) {
            const rejected = memoryStore();
            await expect(
                stageIndexedConversationSnapshot(legacy, undefined, rejected.store, async () => changed),
            ).rejects.toThrow();
            expect(rejected.records.size).toBe(0);
            expect(rejected.pages.size).toBe(0);
        }
    });

    it('keeps captured genesis closed and identical in Zod and JSON Schema', () => {
        const source = { conversation_id: 'conversation:startup', revision: 4 };
        const value = { source, policy: { enabled: true, policy_revision: 0, processors: [processorA] } };
        const validate = new Ajv2020({ strict: false, validateFormats: false }).compile(
            ProcessingPolicyGenesisJsonSchema,
        );
        for (const candidate of [
            value,
            { ...value, caller_override: true },
            { ...value, policy: { ...value.policy, job_count: 1 } },
            { ...value, policy: { ...value.policy, policy_revision: 1 } },
        ])
            expect(validate(candidate)).toBe(ProcessingPolicyGenesisSchema.safeParse(candidate).success);
        expect(validate(value)).toBe(true);
        expect(
            ProcessingPolicyGenesisSchema.safeParse({ ...value, policy: { ...value.policy, job_count: 1 } }).success,
        ).toBe(false);
    });

    it('publishes the strict captured policy command shape consistently with JSON Schema', () => {
        const command = {
            operation_id: 'policy:A',
            expected_revision: 0,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [processorA],
        };
        const validate = new Ajv2020({ strict: false, validateFormats: false }).compile(
            ProcessingPolicyCommandJsonSchema,
        );
        for (const value of [
            command,
            { ...command, caller_override: true },
            { ...command, processors: [{ ...processorA, invented: true }] },
        ])
            expect(validate(value)).toBe(ProcessingPolicyCommandSchema.safeParse(value).success);
        expect(validate(command)).toBe(true);
        expect(ProcessingPolicyCommandSchema.safeParse({ ...command, caller_override: true }).success).toBe(false);
    });
});
