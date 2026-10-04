import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { applyContextChange, planContextChange } from '../src/context-change.js';
import { fingerprintJson } from '../src/identity.js';
import {
    type IndexedConversationRecordStore,
    loadIndexedPendingProcessingJobs,
    loadIndexedProcessingJobState,
    loadIndexedProcessingSelectedContext,
    loadIndexedReadySelectedContext,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingCoverage,
    stageIndexedProcessingNoOpCompletion,
    stageIndexedProcessingPhase,
    stageIndexedRecordBatch,
    stageIndexedTextProcessingCompletion,
} from '../src/indexed-conversation.js';
import {
    applyIndexedTextExternalizationOutput,
    buildIndexedTextExternalizationOutput,
    buildIndexedTextExternalizationProposal,
    indexedTextExternalizationOriginals,
    resolveIndexedProcessingTextInput,
} from '../src/indexed-processing-working-set.js';
import { PAGED_RECORD_INDEX_PAGE_MAX_BYTES } from '../src/paged-record-index.js';
import { buildProcessingPhaseDocument, resolveProcessingJobInput } from '../src/processing.js';
import { ContextChangeRequestSchema } from '../src/schemas/context-change.js';
import {
    INDEXED_CONVERSATION_ACTIVE_MAX_BYTES,
    IndexedConversationContextHeaderRefSchema,
} from '../src/schemas/indexed-head.js';
import {
    INDEXED_PROCESSING_ARCHIVE_ASSET_MAX_BYTES,
    INDEXED_PROCESSING_RETRIEVAL_MAX_BYTES,
    IndexedProcessingArchiveAssetSchema,
    IndexedProcessingRetrievalSchema,
} from '../src/schemas/indexed-processing.js';
import { buildTextExternalizationProposal } from '../src/text-externalization-processor.js';
import { emptyDocument, userTurn } from './fixtures.js';
import { indexedTextClaimFixture } from './indexed-processing-fixture.js';

// Exhaustive CPU capacity fixtures include real immutable setup, dependency validation and retry;
// this is not an ACK/transport latency assertion. Smaller behavior tests retain Vitest's default.
const WORKING_SET_CAPACITY_TEST_TIMEOUT_MS = 30_000;

const at = '2026-09-11T00:00:00.000Z';

describe('indexed selected text processing transitions', () => {
    it('replays original text at the exact retained resolve phase after later phase commits', async () => {
        const { store, staged, workspace } = await indexedTextClaimFixture();
        const state = await loadIndexedProcessingJobState(store, staged.root, workspace.job.id);
        const receipt = state.resolution_receipt;
        if (!receipt?.processing_operation) throw new Error('Actual indexed resolve phase has no receipt');
        const phase = receipt.processing_operation;
        expect(workspace.selected.source.revision).toBeGreaterThan(workspace.resolution.source_revision);
        const original = await indexedTextExternalizationOriginals(
            workspace.selected,
            workspace.job,
            workspace.resolution,
            receipt,
        );
        expect(original.map((item) => item.text)).toEqual(['original']);
        await expect(
            indexedTextExternalizationOriginals(workspace.selected, workspace.job, workspace.resolution),
        ).rejects.toThrow('exact retained resolution phase');
        await expect(
            indexedTextExternalizationOriginals(
                workspace.selected,
                workspace.job,
                { ...workspace.resolution, source_revision: workspace.resolution.source_revision + 1 },
                receipt,
            ),
        ).rejects.toThrow('exact retained resolution phase');
        await expect(
            indexedTextExternalizationOriginals(workspace.selected, workspace.job, workspace.resolution, {
                ...receipt,
                processing_operation: { ...phase, job_id: 'job:foreign' },
            }),
        ).rejects.toThrow('exact retained resolution phase');
        const changed = structuredClone(workspace.selected);
        const text = changed.turns[0]?.selected_blocks[0];
        if (text?.type !== 'text') throw new Error('Actual indexed selection lost its original text');
        text.text = 'different original';
        await expect(
            indexedTextExternalizationOriginals(changed, workspace.job, workspace.resolution, receipt),
        ).rejects.toThrow('exact retained resolved selection');
        await expect(
            indexedTextExternalizationOriginals(
                {
                    ...workspace.selected,
                    context: { ...workspace.selected.context, revision: workspace.selected.context.revision + 1 },
                },
                workspace.job,
                workspace.resolution,
                receipt,
            ),
        ).rejects.toThrow('exact retained resolution phase');
        await expect(
            indexedTextExternalizationOriginals(
                { ...workspace.selected, context: { ...workspace.selected.context, active_tool_definition_ids: [] } },
                workspace.job,
                workspace.resolution,
                receipt,
            ),
        ).rejects.toThrow('exact retained resolved selection');
        const selection = { kind: 'entries' as const, entry_ids: [] };
        await expect(
            indexedTextExternalizationOriginals(
                workspace.selected,
                { ...workspace.job, selection, selection_fingerprint: await fingerprintJson(selection) },
                workspace.resolution,
                receipt,
            ),
        ).rejects.toThrow('exact retained resolved selection');
    });

    it(
        'completes a real 1024-block admitted job with grouped indexes and exact retry',
        async () => {
            const { store, staged, workspace, document, pages } = await indexedTextClaimFixture(0, 'text', 1024);
            const sourceBytes = new Map([...pages].map(([hash, bytes]) => [hash, Uint8Array.from(bytes)]));
            const output = await buildIndexedTextExternalizationOutput(workspace);
            const retained = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
                phase: 'output',
                value: output,
            });
            const readPages = new Set<string>();
            const measuredStore: IndexedConversationRecordStore = {
                ...store,
                async read(ref) {
                    readPages.add(ref.content_hash);
                    return store.read(ref);
                },
            };
            const completed = await stageIndexedTextProcessingCompletion(
                measuredStore,
                retained.root,
                retained.locator,
                workspace,
            );
            expect(completed.completion.status).toBe('applied');
            expect(completed.root.context_header.size_bytes).toBeGreaterThan(PAGED_RECORD_INDEX_PAGE_MAX_BYTES);
            expect(completed.root.context_header.size_bytes).toBeLessThan(INDEXED_CONVERSATION_ACTIVE_MAX_BYTES);
            expect(readPages.size).toBeLessThan(512);
            const selected = await loadIndexedProcessingSelectedContext(store, completed.root, completed.locator);
            expect((selected.replacement_turns ?? []).flatMap((item) => item.projection.selected_blocks)).toHaveLength(
                1024,
            );
            expect((await loadIndexedPendingProcessingJobs(store, completed.root)).unresolved_job_count).toBe(0);
            const retry = await stageIndexedTextProcessingCompletion(
                store,
                completed.root,
                completed.locator,
                workspace,
            );
            expect(retry.applied).toBe(false);
            expect(retry.receipt).toEqual(completed.receipt);
            for (const [hash, bytes] of sourceBytes) expect(pages.get(hash)).toEqual(bytes);
            expect(document.turns[0].blocks).toHaveLength(1024);
        },
        WORKING_SET_CAPACITY_TEST_TIMEOUT_MS,
    );

    it('keeps the existing context-record ceiling and rejects impossible headers before download', async () => {
        const { store, staged, reads } = await indexedTextClaimFixture();
        const atLimit = { ...staged.root.context_header, size_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES };
        expect(IndexedConversationContextHeaderRefSchema.parse(atLimit)).toEqual(atLimit);
        expect(() =>
            IndexedConversationContextHeaderRefSchema.parse({
                ...atLimit,
                size_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES + 1,
            }),
        ).toThrow();
        reads.length = 0;
        await expect(
            loadIndexedProcessingSelectedContext(store, { ...staged.root, context_header: atLimit }, staged.locator),
        ).rejects.toThrow('headers exceed working-set bound before record read');
        expect(reads).toEqual([]);
        const empty = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:empty-header'),
            undefined,
            store,
        );
        reads.length = 0;
        await expect(
            loadIndexedProcessingSelectedContext(
                store,
                {
                    ...empty.root,
                    context_header: { ...empty.root.context_header, size_bytes: INDEXED_CONVERSATION_ACTIVE_MAX_BYTES },
                },
                empty.locator,
            ),
        ).rejects.toThrow('headers exceed working-set bound before record read');
        expect(reads).toEqual([]);
    });

    it('enforces exact archive/retrieval metadata boundaries before output or archive publication', async () => {
        const { store, staged, workspace } = await indexedTextClaimFixture();
        const asset = { ...workspace.archives.assets[0], id: 'asset:bounded', metadata: { pad: '' } };
        asset.metadata.pad = 'x'.repeat(
            INDEXED_PROCESSING_ARCHIVE_ASSET_MAX_BYTES - canonicalJsonContentBytes(asset).byteLength,
        );
        expect(canonicalJsonContentBytes(asset)).toHaveLength(INDEXED_PROCESSING_ARCHIVE_ASSET_MAX_BYTES);
        expect(IndexedProcessingArchiveAssetSchema.parse(asset)).toEqual(asset);
        const tooLarge = { ...asset, metadata: { pad: `${asset.metadata.pad}x` } };
        expect(() => IndexedProcessingArchiveAssetSchema.parse(tooLarge)).toThrow('metadata profile');
        const badArchive = {
            conversation_id: staged.root.source.conversation_id,
            batch: { assets: [tooLarge] },
            options: {
                operation_id: 'processing:archive:oversized-archive',
                expected_revision: staged.root.source.revision,
                payload_fingerprint: await fingerprintJson(tooLarge),
                recorded_at: at,
            },
        };
        await expect(stageIndexedRecordBatch(staged.root, badArchive, store)).rejects.toThrow('metadata profile');
        expect((await loadIndexedPendingProcessingJobs(store, staged.root)).unresolved_job_count).toBe(1);
        const retrieval = { ...workspace.archives.retrievals[0], arguments: { path: '' } };
        retrieval.arguments.path = 'x'.repeat(
            INDEXED_PROCESSING_RETRIEVAL_MAX_BYTES - canonicalJsonContentBytes(retrieval).byteLength,
        );
        expect(canonicalJsonContentBytes(retrieval)).toHaveLength(INDEXED_PROCESSING_RETRIEVAL_MAX_BYTES);
        expect(IndexedProcessingRetrievalSchema.parse(retrieval)).toEqual(retrieval);
        const oversized = { ...retrieval, arguments: { path: `${retrieval.arguments.path}x` } };
        expect(() => IndexedProcessingRetrievalSchema.parse(oversized)).toThrow('metadata profile');
        await expect(
            buildIndexedTextExternalizationOutput({
                ...workspace,
                archives: { ...workspace.archives, retrievals: [oversized] },
            }),
        ).rejects.toThrow('metadata profile');
    });

    it('reuses materialized planning/proposal math and owns nested worker input before awaiting', async () => {
        const { workspace, document } = await indexedTextClaimFixture();
        const fullResolution = await resolveProcessingJobInput(document, workspace.job, at);
        expect(workspace.resolution.source_fingerprint).toBe(fullResolution.source_fingerprint);
        expect(workspace.resolution.entry_ids).toEqual(fullResolution.entry_ids);
        expect(workspace.resolution.source_turn_ids).toEqual(fullResolution.source_turn_ids);
        // Context identity is explicitly selected-only, not a claim about a complete history.
        expect(workspace.resolution.context_fingerprint).not.toBe(fullResolution.context_fingerprint);
        const fullProposal = await buildTextExternalizationProposal(
            document,
            workspace.job,
            workspace.resolution,
            workspace.configuration,
            workspace.archives.retrievals,
        );
        expect(await buildIndexedTextExternalizationProposal(workspace)).toEqual(fullProposal);
        const owned = structuredClone(workspace);
        const expected = await buildIndexedTextExternalizationOutput(owned);
        const pending = buildIndexedTextExternalizationOutput(owned);
        owned.archives.retrievals[0].arguments.path = 'late-mutation';
        owned.selected.turns[0].header.timestamps.recorded_at = '2026-09-12T00:00:00.000Z';
        owned.archives.assets[0].storage = { type: 'inline_text', text: 'late-mutation' };
        expect(await pending).toEqual(expected);
        let getterReads = 0;
        Object.defineProperty(owned, 'snapshot_at', {
            enumerable: true,
            get() {
                getterReads++;
                return at;
            },
        });
        await expect(buildIndexedTextExternalizationOutput(owned)).rejects.toThrow('not bounded JSON');
        expect(getterReads).toBe(0);
    });

    it('commits exact phases and selected delta while retaining original receipt/body and retries', async () => {
        const { store, staged, workspace, document, reads, records } = await indexedTextClaimFixture();
        const beforeBytes = new Map([...records].map(([hash, bytes]) => [hash, Uint8Array.from(bytes)]));
        const output = await buildIndexedTextExternalizationOutput(workspace);
        const persisted = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
            phase: 'output',
            value: output,
        });
        const retry = await stageIndexedProcessingPhase(store, persisted.root, persisted.locator, {
            phase: 'output',
            value: output,
        });
        expect(retry.applied).toBe(false);
        expect(retry.receipt).toEqual(persisted.receipt);
        const currentSelected = await loadIndexedProcessingSelectedContext(store, persisted.root, persisted.locator);
        const mutation = await applyIndexedTextExternalizationOutput(
            { ...workspace, selected: currentSelected },
            output,
        );
        let full = await buildProcessingPhaseDocument(document, 'resolve', workspace.job, workspace.resolution, at, {
            ...document.processing,
            resolved_inputs: { ...document.processing.resolved_inputs, [workspace.job.id]: workspace.resolution },
        });
        full = await buildProcessingPhaseDocument(full, 'attempt', workspace.job, workspace.attempt, at, {
            ...full.processing,
            attempts: { ...full.processing.attempts, [workspace.job.id]: workspace.attempt },
        });
        full = await buildProcessingPhaseDocument(full, 'output', workspace.job, output, at, {
            ...full.processing,
            outputs: { ...full.processing.outputs, [workspace.job.id]: output },
        });
        if (output.kind !== 'proposal' || output.proposal.kind !== 'replace_with_compaction')
            throw new Error('Fixture lost actual pure text output');
        const plan = await planContextChange(full, {
            expected_revision: full.revision,
            expected_context_revision: full.context.revision,
            entry_ids: workspace.resolution.entry_ids,
        });
        const request = ContextChangeRequestSchema.parse({
            operation_id: mutation.receipt.id,
            expected_revision: full.revision,
            expected_context_revision: full.context.revision,
            expected_source_fingerprint: plan.source_fingerprint,
            entry_ids: plan.entry_ids,
            recorded_at: at,
            proposal: {
                ...output.proposal,
                replacement_turns: output.proposal.replacement_turns.map((turn) => {
                    if (turn.provenance.type !== 'derived') throw new Error('Fixture lost exact derived provenance');
                    return { ...turn, provenance: { ...turn.provenance, source_hash: plan.source_fingerprint } };
                }),
            },
        });
        const fullMutation = await applyContextChange(full, request);
        expect(mutation.context).toEqual(fullMutation.document.context);
        expect(mutation.receipt).toEqual(fullMutation.document.operation_receipts[request.operation_id]);
        const committed = await stageIndexedTextProcessingCompletion(
            store,
            persisted.root,
            persisted.locator,
            workspace,
        );
        expect(committed.applied).toBe(true);
        expect(committed.receipt).toEqual(mutation.receipt);
        expect(committed.completion.status).toBe('applied');
        expect(
            (await loadIndexedPendingProcessingJobs(store, committed.root)).jobs.map((item) => item.job.id),
        ).not.toContain(workspace.job.id);
        const retained = await stageIndexedTextProcessingCompletion(
            store,
            committed.root,
            committed.locator,
            workspace,
        );
        expect(retained.applied).toBe(false);
        expect(retained.receipt).toEqual(committed.receipt);
        for (const [hash, bytes] of beforeBytes) expect(records.get(hash)).toEqual(bytes);
        expect(reads).not.toContain('turns:turn:cold-template');
        const changed = structuredClone(output);
        changed.output_fingerprint = await fingerprintJson({ forged: true });
        await expect(
            stageIndexedProcessingPhase(store, committed.root, committed.locator, { phase: 'output', value: changed }),
        ).rejects.toThrow('immutable phase evidence');
    });

    it('settles real empty JSON jobs and reaches a finite target-specific coverage fixed point', async () => {
        const { store, staged, workspace, reads } = await indexedTextClaimFixture();
        const output = await buildIndexedTextExternalizationOutput(workspace);
        let head = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
            phase: 'output',
            value: output,
        });
        head = await stageIndexedTextProcessingCompletion(store, head.root, head.locator, workspace);
        expect((await loadIndexedPendingProcessingJobs(store, head.root)).jobs).toEqual([]);
        const json = userTurn('turn:coverage-json', 'block:coverage-json');
        json.blocks = [{ id: 'block:coverage-json', type: 'json', value: { untouched: [null, true] } }];
        const jsonBatch = {
            turns: [json],
            context_entries: [{ id: 'entry:coverage-json', type: 'source_turn' as const, turn_id: json.id }],
        };
        const jsonAccepted = await stageIndexedRecordBatch(
            head.root,
            {
                conversation_id: head.root.source.conversation_id,
                batch: jsonBatch,
                options: {
                    operation_id: 'operation:coverage-json',
                    expected_revision: head.root.source.revision,
                    payload_fingerprint: await fingerprintJson(jsonBatch),
                    recorded_at: at,
                },
            },
            store,
        );
        if (!jsonAccepted.locator) throw new Error('Real empty JSON job has no accepted root');
        head = {
            root: jsonAccepted.root,
            locator: jsonAccepted.locator,
            receipt: jsonAccepted.receipt,
            applied: jsonAccepted.applied,
        };
        const pending = await loadIndexedPendingProcessingJobs(store, head.root);
        expect(pending.jobs).toHaveLength(1);
        const emptyJob = pending.jobs[0].job;
        const emptySelection = await loadIndexedProcessingSelectedContext(store, head.root, head.locator);
        const resolution = await resolveIndexedProcessingTextInput(emptySelection, emptyJob, at);
        expect(resolution.entry_ids).toEqual([]);
        head = await stageIndexedProcessingPhase(store, head.root, head.locator, {
            phase: 'resolve',
            value: resolution,
        });
        const payload = {
            job_id: emptyJob.id,
            resolved_input_fingerprint: await fingerprintJson(resolution),
            kind: 'no_op' as const,
            reason: 'no_eligible_blocks',
            recorded_at: at,
        };
        head = await stageIndexedProcessingPhase(store, head.root, head.locator, {
            phase: 'output',
            value: { ...payload, output_fingerprint: await fingerprintJson(payload) },
        });
        const settled = await stageIndexedProcessingNoOpCompletion(store, head.root, head.locator, emptyJob.id);
        expect(settled.completion.status).toBe('no_op');
        expect(
            (await stageIndexedProcessingNoOpCompletion(store, settled.root, settled.locator, emptyJob.id)).applied,
        ).toBe(false);
        expect((await loadIndexedPendingProcessingJobs(store, settled.root)).unresolved_job_count).toBe(0);
        const binding = {
            target_fingerprint: await fingerprintJson({ target: 'A' }),
            measured_input_tokens: 9,
            tokenizer_id: 'fixture:actual-native',
            measurement_fingerprint: await fingerprintJson({ native: 'A', tokens: 9 }),
        };
        reads.length = 0;
        const coveredA = await stageIndexedProcessingCoverage(store, settled.root, settled.locator, {
            ...binding,
            operation_id: 'coverage:arbitrary-a',
            expected_revision: settled.root.source.revision,
            recorded_at: at,
        });
        expect(coveredA.coverage.status).toBe('ready');
        expect(coveredA.coverage.required_job_count).toBe(2);
        expect(coveredA.coverage.required_jobs_root).toEqual(settled.root.directories.processing_required);
        expect(reads.filter((id) => id.includes('processing_records:') && id.includes('processing:'))).toEqual([]);
        const coveredB = await stageIndexedProcessingCoverage(store, coveredA.root, coveredA.locator, {
            ...binding,
            target_fingerprint: await fingerprintJson({ target: 'B' }),
            operation_id: 'coverage:arbitrary-b',
            expected_revision: coveredA.root.source.revision,
            recorded_at: at,
        });
        const retainedA = await loadIndexedReadySelectedContext(store, coveredB.root, coveredB.locator, binding);
        expect(retainedA.coverage).toEqual(coveredA.coverage);
        expect(retainedA.selection.source).toEqual(coveredB.root.source);
        // Reading the same A measurement does not publish another coverage/head revision.
        expect(
            (await loadIndexedReadySelectedContext(store, coveredB.root, coveredB.locator, binding)).coverage,
        ).toEqual(coveredA.coverage);
        const turn = userTurn('turn:next', 'block:next');
        const batch = {
            turns: [turn],
            context_entries: [{ id: 'entry:next', type: 'source_turn' as const, turn_id: turn.id }],
        };
        const queued = await stageIndexedRecordBatch(
            coveredB.root,
            {
                conversation_id: coveredB.root.source.conversation_id,
                batch,
                options: {
                    operation_id: 'operation:next',
                    expected_revision: coveredB.root.source.revision,
                    payload_fingerprint: await fingerprintJson(batch),
                    recorded_at: at,
                },
            },
            store,
        );
        if (!queued.locator) throw new Error('Real queued append lacks root');
        await expect(loadIndexedReadySelectedContext(store, queued.root, queued.locator, binding)).rejects.toThrow(
            'unresolved',
        );
        const pendingCoverage = await stageIndexedProcessingCoverage(store, queued.root, queued.locator, {
            ...binding,
            operation_id: 'coverage:pending',
            expected_revision: queued.root.source.revision,
            recorded_at: at,
        });
        expect(pendingCoverage.coverage.status).toBe('pending');
        expect(pendingCoverage.coverage.required_job_count).toBe(3);
        expect(pendingCoverage.coverage.required_jobs_root).not.toEqual(coveredA.coverage.required_jobs_root);
        const { processing_index_profile: _missingProfile, ...withoutProcessingProfile } = coveredB.root;
        await expect(
            loadIndexedReadySelectedContext(store, withoutProcessingProfile, coveredB.locator, binding),
        ).rejects.toThrow('complete processing indexes');
    });

    it('externalizes only text in mixed text/JSON and preserves exact JSON through real partial completion', async () => {
        const { store, staged, workspace, document } = await indexedTextClaimFixture(0, 'mixed');
        const [json] = document.turns[0].blocks.filter((block) => block.type === 'json');
        if (!json) throw new Error('Mixed fixture lost its real structured block');
        expect(workspace.job.selection).toMatchObject({
            kind: 'entries',
            entry_ids: ['entry:input'],
            selected_block_ids: { 'entry:input': ['block:input'] },
        });
        const fullResolution = await resolveProcessingJobInput(document, workspace.job, at);
        expect(workspace.resolution.source_fingerprint).toBe(fullResolution.source_fingerprint);
        const materializedProposal = await buildTextExternalizationProposal(
            document,
            workspace.job,
            workspace.resolution,
            workspace.configuration,
            workspace.archives.retrievals,
        );
        expect(await buildIndexedTextExternalizationProposal(workspace)).toEqual(materializedProposal);
        const output = await buildIndexedTextExternalizationOutput(workspace);
        const persisted = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
            phase: 'output',
            value: output,
        });
        const completed = await stageIndexedTextProcessingCompletion(
            store,
            persisted.root,
            persisted.locator,
            workspace,
        );
        expect(completed.completion.status).toBe('applied');
        const selected = await loadIndexedProcessingSelectedContext(store, completed.root, completed.locator);
        expect(selected.turns.flatMap((turn) => turn.selected_blocks)).toContainEqual(json);
        expect(selected.turns.flatMap((turn) => turn.selected_blocks).some((block) => block.id === 'block:input')).toBe(
            false,
        );
        expect(selected.replacement_turns).toHaveLength(1);
        expect(
            (selected.replacement_turns ?? [])
                .flatMap((turn) => turn.projection.selected_blocks)
                .some((block) => block.type === 'external_reference'),
        ).toBe(true);
        expect(
            (await stageIndexedTextProcessingCompletion(store, completed.root, completed.locator, workspace)).applied,
        ).toBe(false);
    });

    it('rejects a real turn/global-ID collision before first phase publication while retained phase retry stays exact', async () => {
        const { store, staged, workspace, document } = await indexedTextClaimFixture();
        const colliding = userTurn(`processing:resolve:${workspace.job.id}`, 'block:phase-collision');
        const conflicted = await stageIndexedConversationSnapshot(
            { ...document, turns: [colliding, ...document.turns] },
            undefined,
            store,
        );
        await expect(
            stageIndexedProcessingPhase(store, conflicted.root, conflicted.locator, {
                phase: 'resolve',
                value: workspace.resolution,
            }),
        ).rejects.toThrow('already exists');
        expect(
            (
                await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
                    phase: 'resolve',
                    value: workspace.resolution,
                })
            ).applied,
        ).toBe(false);
    });

    it('accepts record stores whose prototype methods require their class receiver', async () => {
        const fixture = await indexedTextClaimFixture();
        class ClassStore implements IndexedConversationRecordStore {
            constructor(private readonly delegate: IndexedConversationRecordStore) {}
            read(...args: Parameters<IndexedConversationRecordStore['read']>) {
                return this.delegate.read(...args);
            }
            write(...args: Parameters<IndexedConversationRecordStore['write']>) {
                return this.delegate.write(...args);
            }
            readRecord(...args: Parameters<IndexedConversationRecordStore['readRecord']>) {
                return this.delegate.readRecord(...args);
            }
            writeRecord(...args: Parameters<IndexedConversationRecordStore['writeRecord']>) {
                return this.delegate.writeRecord(...args);
            }
        }
        const store = new ClassStore(fixture.store);
        const output = await buildIndexedTextExternalizationOutput(fixture.workspace);
        const phase = await stageIndexedProcessingPhase(store, fixture.staged.root, fixture.staged.locator, {
            phase: 'output',
            value: output,
        });
        const completed = await stageIndexedTextProcessingCompletion(
            store,
            phase.root,
            phase.locator,
            fixture.workspace,
        );
        expect(completed.completion.status).toBe('applied');
    });

    it('rejects missing selected dependencies, foreign archives, and altered output before completion', async () => {
        const { workspace } = await indexedTextClaimFixture();
        const missing = structuredClone(workspace);
        missing.selected.turns[0].selected_blocks = [];
        await expect(buildIndexedTextExternalizationOutput(missing)).rejects.toThrow();
        const foreign = structuredClone(workspace);
        foreign.archives.acceptance.conversation_id = 'foreign';
        await expect(buildIndexedTextExternalizationOutput(foreign)).rejects.toThrow('archive/configuration');
        const output = await buildIndexedTextExternalizationOutput(workspace);
        if (output.kind !== 'proposal' || output.proposal.kind !== 'replace_with_compaction')
            throw new Error('Fixture lost exact externalization proposal');
        output.proposal.replacement_turns[0].blocks = [
            { id: 'forged:text', type: 'text', text: 'fabricated', format: 'plain' },
        ];
        await expect(applyIndexedTextExternalizationOutput(workspace, output)).rejects.toThrow(
            'exact deterministic archived output',
        );
    });

    it('uses the same selected transition/read profile with 10k and 100k unrelated cold turns', async () => {
        const profiles: number[] = [];
        for (const count of [10_000, 100_000]) {
            const { store, staged, workspace, reads } = await indexedTextClaimFixture(count);
            reads.length = 0;
            const output = await buildIndexedTextExternalizationOutput(workspace);
            const bytes = canonicalJsonContentBytes(output);
            expect((await hashContentBytes(bytes)).byte_length).toBe(bytes.byteLength);
            const persisted = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
                phase: 'output',
                value: output,
            });
            const committed = await stageIndexedTextProcessingCompletion(
                store,
                persisted.root,
                persisted.locator,
                workspace,
            );
            const page = await loadIndexedPendingProcessingJobs(store, committed.root);
            expect(page.jobs.every((entry) => entry.job.id !== workspace.job.id)).toBe(true);
            expect(reads.some((id) => id.startsWith('turns:turn:cold:') || id.startsWith('blocks:block:cold:'))).toBe(
                false,
            );
            expect(reads.length).toBeLessThan(96);
            profiles.push(reads.length);
        }
        expect(profiles[1]).toBe(profiles[0]);
    }, 120_000);
});
