import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it, vi } from 'vitest';
import {
    abandonProcessingAttempt,
    appendConversationRecords,
    appendConversationRecordsWithProcessing,
    applyContextChange,
    assertProcessingReady,
    assessProcessingReadiness,
    type ConversationDocument,
    createConversationDocument,
    createTextBlock,
    createTextExternalizationProcessor,
    createUserTurn,
    hashUtf8Content,
    inspectTextExternalizationJobSelection,
    ProcessingJobSchema,
    ProcessingKnownFailure,
    ProcessingOutputReceiptSchema,
    type ProcessingStore,
    ProcessingUnknownFailure,
    parseConversationDocument,
    planContextChange,
    processingAppendAcceptance,
    queueProcessingForExisting,
    recordProcessingCoverage,
    runProcessingJob,
    setProcessingPolicy,
    type TextExternalizationRetrievalBinder,
    textExternalizationArchiveInputs,
    textForExternalizationJob,
} from '../src/index.js';
import { ProcessingJobJsonSchema, ProcessingOutputReceiptJsonSchema } from '../src/json-schema.js';
import { generatedAgentTurn, importedGeneration, toolCallBlock, toolResultTurn } from './fixtures.js';

const at = '2026-09-11T00:01:00.000Z';
const base = () =>
    createConversationDocument({ id: 'processing-conversation', created_at: '2026-09-11T00:00:00.000Z' });
const processor = (id: string, required = true, scope: 'on_append' | 'on_budget' | 'manual' = 'on_append') => ({
    id,
    version: 'v1',
    scope,
    config: {},
    required,
    failure_behavior: required ? ('block' as const) : ('skip_with_diagnostic' as const),
});
const turn = createUserTurn({
    id: 'received-turn',
    authority: 'ordinary',
    status: 'completed',
    timestamps: { recorded_at: at },
    model_visibility: 'include',
    provenance: { type: 'received' },
    blocks: [createTextBlock({ id: 'received-block', text: 'original', format: 'plain' })],
});
const batch = {
    turns: [turn],
    context_entries: [{ id: 'received-entry', type: 'source_turn' as const, turn_id: turn.id }],
};
const appendOptions = {
    operation_id: 'append:received',
    expected_revision: 1,
    payload_fingerprint: 'sha256:received',
    recorded_at: at,
};
async function enabled(processors = [processor('noop')]) {
    return (
        await setProcessingPolicy(base(), {
            operation_id: 'policy:1',
            expected_revision: 0,
            recorded_at: at,
            enabled: true,
            processors,
        })
    ).document;
}
class MemoryStore implements ProcessingStore {
    constructor(
        public current: ConversationDocument,
        public refuseNext = false,
    ) {}
    async load() {
        return structuredClone(this.current);
    }
    async commit(expectedRevision: number, document: ConversationDocument) {
        if (this.refuseNext && document.context.revision > this.current.context.revision) {
            this.refuseNext = false;
            return false;
        }
        if (this.current.revision !== expectedRevision) return false;
        this.current = parseConversationDocument(structuredClone(document));
        return true;
    }
}

function firstJob(document: ConversationDocument) {
    return Object.values(document.processing.jobs ?? {})[0];
}

// The host supplies a clock and an atomic exact-head store; no plugin runs inside append or its acknowledgement.
describe('durable processing jobs', () => {
    it('archives ordered selected text around an unselected image and preserves both ranges', async () => {
        const source = await setProcessingPolicy(base(), {
            operation_id: 'policy:multi-text',
            expected_revision: 0,
            recorded_at: at,
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
        });
        const accepted = await appendConversationRecordsWithProcessing(
            source.document,
            {
                turns: [
                    {
                        ...turn,
                        id: 'turn:multi',
                        blocks: [
                            createTextBlock({ id: 'block:first', text: 'first', format: 'plain' }),
                            createTextBlock({ id: 'block:second', text: 'second', format: 'plain' }),
                            { id: 'block:image', type: 'image', asset_id: 'asset:image' },
                            createTextBlock({ id: 'block:third', text: 'third', format: 'plain' }),
                        ],
                    },
                ],
                assets: [
                    {
                        id: 'asset:image',
                        kind: 'image',
                        mime_type: 'image/png',
                        storage: { type: 'inline_base64', data: 'AA==' },
                        provenance: { type: 'received' },
                        created_at: at,
                    },
                ],
                context_entries: [{ id: 'entry:multi', type: 'source_turn', turn_id: 'turn:multi' }],
                tool_definitions: [
                    { id: 'definition:read-multi', name: 'read_blob', version: '1', input_schema: true },
                ],
                active_tool_definition_ids: ['definition:read-multi'],
            },
            { ...appendOptions, operation_id: 'append:multi-text', expected_revision: source.document.revision },
        );
        const job = firstJob(accepted.document);
        const archiveInput = await textExternalizationArchiveInputs(accepted.document, job);
        expect(archiveInput.texts.map((item) => item.block_id)).toEqual(['block:first', 'block:second', 'block:third']);
        const assets = archiveInput.integrities.map((integrity, index) => ({
            id: `asset:text:${index}`,
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: { type: 'external' as const, resolver: 'test.blob', locator: { key: `text:${index}` } },
            provenance: { type: 'received' as const },
            content_hash: integrity.content_hash,
            byte_length: integrity.byte_length,
            created_at: at,
        }));
        const archived = await appendConversationRecordsWithProcessing(
            accepted.document,
            { assets },
            {
                operation_id: `processing:archive:${job.id}`,
                expected_revision: accepted.document.revision,
                payload_fingerprint: archiveInput.payload_fingerprint,
                recorded_at: at,
            },
        );
        const store = new MemoryStore(archived.document);
        const processor = createTextExternalizationProcessor(({ asset }) => ({
            capability: 'read_blob',
            version: 1,
            arguments: { asset_id: asset.id },
            tool_definition_id: 'definition:read-multi',
        }));
        const result = await runProcessingJob(store, { resolve: () => processor }, job.id, 'attempt:multi', () => at);
        expect(result.status).toBe('completed');
        expect(store.current.context.entries.map((entry) => entry.type)).toEqual([
            'replacement_turn',
            'source_turn',
            'replacement_turn',
        ]);
        expect(store.current.context.entries[1]?.block_ids).toEqual(['block:image']);
        expect(store.current.context.retrieval_requirements.map((requirement) => requirement.asset_id)).toEqual(
            assets.map((asset) => asset.id),
        );
        const replacementTurns = store.current.compactions[Object.keys(store.current.compactions)[0]].replacement_turns;
        expect(
            replacementTurns.map((replacement) =>
                replacement.blocks.map((block) =>
                    block.type === 'external_reference' ? block.asset_id : 'unexpected-non-reference',
                ),
            ),
        ).toEqual([['asset:text:0', 'asset:text:1'], ['asset:text:2']]);
        const applyId = `processing:apply:${job.id}`;
        const applyReceipt = store.current.operation_receipts[applyId];
        const detail = applyReceipt?.context_change;
        const resolved = store.current.processing.resolved_inputs?.[job.id];
        const output = store.current.processing.outputs?.[job.id];
        const compaction = store.current.compactions[Object.keys(store.current.compactions)[0]];
        const sourceContextRevision = compaction?.metadata?.source_context_revision;
        if (
            !applyReceipt ||
            !detail ||
            !resolved?.selected_entries ||
            !resolved.selected_block_ids ||
            output?.kind !== 'proposal' ||
            output.proposal.kind !== 'replace_with_compaction' ||
            !compaction ||
            typeof sourceContextRevision !== 'number' ||
            !Number.isSafeInteger(sourceContextRevision)
        )
            throw new Error('Fixture lacks the accepted partial compaction evidence');
        const retry = await applyContextChange(store.current, {
            operation_id: applyId,
            expected_revision: applyReceipt.base_revision,
            expected_context_revision: sourceContextRevision,
            expected_source_fingerprint: detail.source_fingerprint,
            recorded_at: applyReceipt.recorded_at,
            entry_ids: resolved.entry_ids,
            selected_entries: resolved.selected_entries,
            selected_block_ids: resolved.selected_block_ids,
            proposal: { ...output.proposal, replacement_turns: compaction.replacement_turns },
        });
        expect(retry.applied).toBe(false);
        expect(retry.document).toEqual(store.current);
        expect(
            (await runProcessingJob(store, { resolve: () => processor }, job.id, 'attempt:multi', () => at)).status,
        ).toBe('completed');
    });

    it('classifies a text archive selection before the host performs asset I/O', async () => {
        const source = await enabled([processor('externalize-text')]);
        const accepted = await appendConversationRecordsWithProcessing(source, batch, {
            ...appendOptions,
            expected_revision: source.revision,
        });
        const job = firstJob(accepted.document);
        if (job?.selection.kind !== 'entries') throw new Error('Fixture needs an entries job');
        expect(inspectTextExternalizationJobSelection(accepted.document, job)).toEqual({
            kind: 'eligible',
            texts: [
                { entry_id: 'received-entry', turn_id: 'received-turn', block_id: 'received-block', text: 'original' },
            ],
        });
        expect(
            inspectTextExternalizationJobSelection(accepted.document, {
                ...job,
                selection: { ...job.selection, entry_ids: [] },
            }),
        ).toEqual({ kind: 'no_eligible_blocks' });
        const multipleEntryJob = {
            ...job,
            selection: { ...job.selection, entry_ids: ['received-entry', 'another-entry'] },
        };
        expect(() => inspectTextExternalizationJobSelection(accepted.document, multipleEntryJob)).toThrow(
            'source entry is unavailable',
        );
        expect(
            inspectTextExternalizationJobSelection(accepted.document, {
                ...job,
                selection: { kind: 'predecessor_output', job_id: 'previous-job' },
            }),
        ).toEqual({ kind: 'unsupported', reason: 'non_entry_selection' });
        const multipleBlocks = structuredClone(accepted.document);
        const original = multipleBlocks.turns.find((candidate) => candidate.id === turn.id);
        if (original?.kind !== 'user') throw new Error('Fixture lost its user source turn');
        original.blocks.push(createTextBlock({ id: 'second-block', text: 'second', format: 'plain' }));
        expect(inspectTextExternalizationJobSelection(multipleBlocks, job)).toEqual({
            kind: 'eligible',
            texts: [
                { entry_id: 'received-entry', turn_id: 'received-turn', block_id: 'received-block', text: 'original' },
                { entry_id: 'received-entry', turn_id: 'received-turn', block_id: 'second-block', text: 'second' },
            ],
        });
        expect(() => textForExternalizationJob(multipleBlocks, job)).toThrow('more than one text block');
        const mixed = await appendConversationRecordsWithProcessing(
            source,
            {
                turns: [
                    createUserTurn({
                        ...turn,
                        id: 'mixed-turn',
                        blocks: [
                            createTextBlock({ id: 'mixed-text', text: 'selected text', format: 'plain' }),
                            { id: 'mixed-image', type: 'image', asset_id: 'mixed-image-asset' },
                        ],
                    }),
                ],
                assets: [
                    {
                        id: 'mixed-image-asset',
                        kind: 'image',
                        mime_type: 'image/png',
                        storage: { type: 'inline_base64', data: 'AA==' },
                        provenance: { type: 'received' },
                        created_at: at,
                    },
                ],
                context_entries: [{ id: 'mixed-entry', type: 'source_turn', turn_id: 'mixed-turn' }],
            },
            { ...appendOptions, operation_id: 'append:mixed', expected_revision: source.revision },
        );
        const mixedJob = firstJob(mixed.document);
        if (mixedJob?.selection.kind !== 'entries') throw new Error('Fixture lost mixed selection');
        expect(mixedJob.selection.selected_block_ids?.['mixed-entry']).toEqual(['mixed-text']);
        expect(inspectTextExternalizationJobSelection(mixed.document, mixedJob)).toEqual({
            kind: 'eligible',
            texts: [{ entry_id: 'mixed-entry', turn_id: 'mixed-turn', block_id: 'mixed-text', text: 'selected text' }],
        });
        const missing = structuredClone(accepted.document);
        missing.context.entries = [];
        expect(() => inspectTextExternalizationJobSelection(missing, job)).toThrow('source entry is unavailable');
    });

    it('recovers only an interrupted built-in text attempt from its exact durable archive', async () => {
        const source = await setProcessingPolicy(base(), {
            operation_id: 'policy:text-recovery',
            expected_revision: 0,
            recorded_at: at,
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
        });
        const accepted = await appendConversationRecordsWithProcessing(
            source.document,
            {
                ...batch,
                tool_definitions: [
                    {
                        id: 'definition:read',
                        name: 'read_artifact',
                        version: '1',
                        input_schema: {
                            type: 'object',
                            properties: { path: { type: 'string' }, asset_id: { type: 'string' } },
                            required: ['path'],
                            additionalProperties: false,
                        },
                    },
                ],
                active_tool_definition_ids: ['definition:read'],
            },
            { ...appendOptions, expected_revision: source.document.revision },
        );
        const job = firstJob(accepted.document);
        const integrity = await hashUtf8Content('original');
        const archived = await appendConversationRecordsWithProcessing(
            accepted.document,
            {
                assets: [
                    {
                        id: 'asset:original',
                        kind: 'text',
                        mime_type: 'text/plain',
                        storage: {
                            type: 'external',
                            resolver: 'test.archive',
                            locator: { key: 'original' },
                        },
                        provenance: { type: 'received' },
                        content_hash: integrity.content_hash,
                        byte_length: integrity.byte_length,
                        created_at: at,
                    },
                ],
            },
            {
                operation_id: `processing:archive:${job.id}`,
                expected_revision: accepted.document.revision,
                payload_fingerprint: integrity.content_hash,
                recorded_at: at,
            },
        );
        const store = new MemoryStore(archived.document);
        const binder: TextExternalizationRetrievalBinder = () => ({
            capability: 'read_artifact',
            version: 1,
            arguments: { asset_id: 'asset:original', path: 'original' },
            tool_definition_id: 'definition:read',
        });
        const interrupted = new AbortController();
        let pluginRuns = 0;
        await expect(
            runProcessingJob(
                store,
                {
                    resolve: () => ({
                        run: async () => {
                            pluginRuns += 1;
                            interrupted.abort(new Error('worker stopped after durable attempt'));
                            throw interrupted.signal.reason;
                        },
                    }),
                },
                job.id,
                'attempt:first-task',
                () => at,
                interrupted.signal,
            ),
        ).rejects.toThrow('worker stopped');
        expect(store.current.processing.attempts?.[job.id]?.attempt_token).toBe('attempt:first-task');
        expect(store.current.processing.outputs?.[job.id]).toBeUndefined();
        const attempted = store.current;
        await expect(
            setProcessingPolicy(attempted, {
                operation_id: 'policy:cannot-supersede-active-attempt',
                expected_revision: attempted.revision,
                recorded_at: at,
                enabled: false,
                processors: [],
                supersede_job_ids: [job.id],
                supersession_reason: 'operator_cancelled',
            }),
        ).rejects.toThrow('unresolved external attempt');
        expect(
            (await runProcessingJob(store, { resolve: () => undefined }, job.id, 'attempt:second-task', () => at))
                .status,
        ).toBe('in_progress');
        const advanced = await appendConversationRecordsWithProcessing(
            store.current,
            {
                assets: [
                    {
                        id: 'asset:unrelated',
                        kind: 'text',
                        mime_type: 'text/plain',
                        storage: { type: 'external', resolver: 'test.archive', locator: { key: 'unrelated' } },
                        provenance: { type: 'received' },
                        content_hash: (await hashUtf8Content('unrelated')).content_hash,
                        byte_length: 9,
                        created_at: at,
                    },
                ],
            },
            {
                operation_id: 'append:unrelated-after-attempt',
                expected_revision: store.current.revision,
                payload_fingerprint: 'sha256:unrelated-after-attempt',
                recorded_at: at,
            },
        );
        const conflicted = new MemoryStore(advanced.document);
        await expect(
            runProcessingJob(
                conflicted,
                {
                    resolve: () => {
                        throw new Error('Registry must not run');
                    },
                },
                job.id,
                'attempt:second-task',
                () => at,
                undefined,
                { text_externalization_recovery: binder },
            ),
        ).rejects.toThrow('lost its exact attempt and source');
        expect(conflicted.current.processing.outputs?.[job.id]).toBeUndefined();
        let racedHead = store.current;
        let raced = false;
        const racingStore: ProcessingStore = {
            load: async () => structuredClone(racedHead),
            commit: async (expected, document) => {
                if (!raced && document.processing.outputs?.[job.id]) {
                    raced = true;
                    racedHead = advanced.document;
                    return false;
                }
                if (racedHead.revision !== expected) return false;
                racedHead = parseConversationDocument(document);
                return true;
            },
        };
        await expect(
            runProcessingJob(
                racingStore,
                {
                    resolve: () => {
                        throw new Error('Registry must not run');
                    },
                },
                job.id,
                'attempt:second-task',
                () => at,
                undefined,
                { text_externalization_recovery: binder },
            ),
        ).rejects.toThrow('lost its exact attempt and source');
        expect(raced).toBe(true);
        expect(racedHead.processing.outputs?.[job.id]).toBeUndefined();
        const completedElsewhere = new MemoryStore(attempted);
        let settled = false;
        const completionRace: ProcessingStore = {
            load: async () => completedElsewhere.load(),
            commit: async (expected, document) => {
                if (!settled && document.processing.outputs?.[job.id]) {
                    settled = true;
                    await abandonProcessingAttempt(completedElsewhere, {
                        operation_id: 'abandon:competing-host',
                        job_id: job.id,
                        attempt_token: 'attempt:first-task',
                        expected_revision: completedElsewhere.current.revision,
                        recorded_at: at,
                    });
                    return false;
                }
                return completedElsewhere.commit(expected, document);
            },
        };
        await expect(
            runProcessingJob(
                completionRace,
                {
                    resolve: () => {
                        throw new Error('Registry must not run');
                    },
                },
                job.id,
                'attempt:second-task',
                () => at,
                undefined,
                { text_externalization_recovery: binder },
            ),
        ).rejects.toThrow('lost its exact attempt and source');
        expect(settled).toBe(true);
        expect(completedElsewhere.current.processing.outputs?.[job.id]?.kind).toBe('unknown_outcome');
        expect(completedElsewhere.current.processing.completions?.[job.id]?.status).toBe('blocked');
        const resumedAt = new Date(Date.parse(at) + 60_000).toISOString();
        const resumed = await runProcessingJob(
            store,
            {
                resolve: () => {
                    throw new Error('Generic registry must not rerun');
                },
            },
            job.id,
            'attempt:second-task',
            () => resumedAt,
            undefined,
            { text_externalization_recovery: binder },
        );
        expect(resumed.status).toBe('completed');
        expect(pluginRuns).toBe(1);
        expect(store.current.processing.outputs?.[job.id]?.attempt_token).toBe('attempt:first-task');
        const recoveredOutput = store.current.processing.outputs?.[job.id];
        if (recoveredOutput?.kind !== 'proposal' || recoveredOutput.proposal.kind !== 'replace_with_compaction') {
            throw new Error('Recovered output lost its deterministic proposal');
        }
        expect(recoveredOutput.proposal.replacement_turns[0].timestamps.recorded_at).toBe(attempted.updated_at);
        expect(recoveredOutput.recorded_at).toBe(resumedAt);
        expect(store.current.operation_receipts[`processing:output:${job.id}`].recorded_at).toBe(resumedAt);
        expect(recoveredOutput.proposal.replacement_turns[0].provenance).toMatchObject({
            type: 'derived',
            source_hash: attempted.processing.resolved_inputs?.[job.id]?.source_fingerprint,
        });
        expect(store.current.processing.completions?.[job.id]?.status).toBe('applied');
        expect(store.current.context.entries[0]?.type).toBe('replacement_turn');
    });
    it('rejects enabled synchronous append, stages on_append atomically, and retries without a second job', async () => {
        const source = await enabled();
        expect(() => appendConversationRecords(source, batch, appendOptions)).toThrow(
            'requires appendConversationRecordsWithProcessing',
        );
        const accepted = await appendConversationRecordsWithProcessing(source, batch, appendOptions);
        expect(accepted.applied).toBe(true);
        expect(firstJob(accepted.document).selection).toMatchObject({ kind: 'entries', entry_ids: ['received-entry'] });
        expect(accepted.document.operation_receipts[appendOptions.operation_id].accepted_context_entry_ids).toEqual([
            'received-entry',
        ]);
        expect(accepted.acceptance).toMatchObject({
            operation_id: appendOptions.operation_id,
            conversation_id: source.id,
            base_revision: source.revision,
            accepted_revision: accepted.document.revision,
            change_reference: {
                operation_id: appendOptions.operation_id,
                result_revision: accepted.document.revision,
            },
            processing: { status: 'pending', job_ids: [firstJob(accepted.document).id] },
        });
        const retry = await appendConversationRecordsWithProcessing(accepted.document, batch, appendOptions);
        expect(retry.applied).toBe(false);
        expect(retry.document.processing.jobs).toEqual(accepted.document.processing.jobs);
        expect(retry.acceptance).toEqual(accepted.acceptance);
    });

    it('acknowledges disabled policy as ready and an existing required failure as blocked', async () => {
        const noPolicy = await appendConversationRecordsWithProcessing(base(), batch, {
            ...appendOptions,
            expected_revision: 0,
        });
        expect(noPolicy.acceptance.processing).toEqual({ status: 'ready', job_ids: [] });
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        await runProcessingJob(
            store,
            {
                resolve: () => ({
                    run: async () => {
                        throw new ProcessingKnownFailure('rejected');
                    },
                }),
            },
            job.id,
            'attempt:ack-blocked',
            () => at,
        );
        expect(processingAppendAcceptance(store.current, appendOptions.operation_id).processing.status).toBe('blocked');
    });

    it('owns the accepted operation identity before asynchronous job staging', async () => {
        const options = { ...appendOptions };
        const pending = appendConversationRecordsWithProcessing(await enabled(), batch, options);
        options.operation_id = 'append:mutated-after-invocation';
        const accepted = await pending;
        expect(accepted.acceptance.operation_id).toBe(appendOptions.operation_id);
        expect(firstJob(accepted.document).source_operation_id).toBe(appendOptions.operation_id);
        expect(accepted.document.operation_receipts[options.operation_id]).toBeUndefined();
    });

    it('stores a no-eligible-block result without calling a processor', async () => {
        const source = await enabled();
        const accepted = await appendConversationRecordsWithProcessing(
            source,
            {},
            { ...appendOptions, payload_fingerprint: 'sha256:empty' },
        );
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        const run = vi.fn();
        await runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:no-op', () => at);
        expect(run).not.toHaveBeenCalled();
        expect(store.current.processing.outputs?.[job.id]?.kind).toBe('no_op');
        expect(store.current.processing.completions?.[job.id]?.status).toBe('no_op');
    });

    it('reuses durable output after a failed context CAS and never reruns its processor', async () => {
        const accepted = await appendConversationRecordsWithProcessing(
            await enabled([processor('exclude')]),
            batch,
            appendOptions,
        );
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document, true);
        const run = vi.fn(async () => ({ kind: 'proposal' as const, proposal: { kind: 'exclude' as const } }));
        await runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:exclude', () => at);
        expect(run).toHaveBeenCalledTimes(1);
        expect(store.current.processing.completions?.[job.id]?.status).toBe('applied');
        expect(store.current.context.entries).toEqual([]);
        expect(store.current.processing.outputs?.[job.id]?.kind).toBe('proposal');
        const retained = parseConversationDocument(JSON.parse(JSON.stringify(store.current)));
        const replay = await appendConversationRecordsWithProcessing(retained, batch, appendOptions);
        expect(replay.applied).toBe(false);
        expect(replay.acceptance.operation_id).toBe(appendOptions.operation_id);
        expect(replay.document.revision).toBe(retained.revision);
        expect(replay.document.processing.jobs).toEqual(retained.processing.jobs);
        expect(replay.document.operation_receipts[appendOptions.operation_id]).toEqual(
            accepted.document.operation_receipts[appendOptions.operation_id],
        );
    });

    it('rejects a changed recovered proposal after output persistence without rerunning inference', async () => {
        const accepted = await appendConversationRecordsWithProcessing(
            await enabled([processor('summarize')]),
            batch,
            appendOptions,
        );
        const job = firstJob(accepted.document);
        class CrashAfterOutputStore extends MemoryStore {
            override async commit(expectedRevision: number, document: ConversationDocument) {
                if (document.context.revision > this.current.context.revision)
                    throw new Error('crash after durable processor output');
                return super.commit(expectedRevision, document);
            }
        }
        const store = new CrashAfterOutputStore(accepted.document);
        const run = vi.fn(
            async ({
                resolved_input,
            }: {
                resolved_input: NonNullable<ConversationDocument['processing']['resolved_inputs']>[string];
            }) => ({
                kind: 'proposal' as const,
                proposal: {
                    kind: 'replace_with_compaction' as const,
                    compaction_id: 'summary:tamper',
                    strategy: { id: 'summarize', version: 'v1', configuration_fingerprint: 'sha256:config' },
                    replacement_turns: [
                        {
                            ...turn,
                            id: 'summary:turn',
                            kind: 'agent' as const,
                            provenance: {
                                type: 'derived' as const,
                                derivation_id: 'summary:tamper',
                                source_turn_ids: resolved_input.source_turn_ids,
                                source_hash: resolved_input.source_fingerprint,
                            },
                            blocks: [createTextBlock({ id: 'summary:block', text: 'short', format: 'plain' })],
                        },
                    ],
                    fidelity: 'semantic' as const,
                    retained_asset_ids: [],
                    generation_ids: [],
                    placement: { mode: 'first_selected' as const, causal_order: 'contiguous' as const },
                },
            }),
        );
        const registry = { resolve: () => ({ run }) };
        await expect(runProcessingJob(store, registry, job.id, 'attempt:tamper', () => at)).rejects.toThrow(
            'crash after durable processor output',
        );
        const output = store.current.processing.outputs?.[job.id];
        expect(output?.kind).toBe('proposal');
        if (output?.kind !== 'proposal' || output.proposal.kind !== 'replace_with_compaction')
            throw new Error('Expected retained proposal');
        const block = output.proposal.replacement_turns[0].blocks[0];
        if (block.type !== 'text') throw new Error('Expected summary text');
        block.text = 'changed after persistence';
        await expect(runProcessingJob(store, registry, job.id, 'attempt:retry', () => at)).rejects.toThrow(
            'output fingerprint changed',
        );
        expect(run).toHaveBeenCalledTimes(1);
        expect(store.current.context.entries.map((entry) => entry.id)).toEqual(['received-entry']);
    });

    it('resolves stage two from the accepted replacement of stage one', async () => {
        const accepted = await appendConversationRecordsWithProcessing(
            await enabled([processor('summarize'), processor('inspect')]),
            batch,
            appendOptions,
        );
        const jobs = Object.values(accepted.document.processing.jobs ?? {});
        const store = new MemoryStore(accepted.document);
        const observed: string[][] = [];
        const registry = {
            resolve: (id: string) => ({
                run: async ({
                    resolved_input,
                }: {
                    resolved_input: NonNullable<ConversationDocument['processing']['resolved_inputs']>[string];
                }) => {
                    observed.push(resolved_input.entry_ids);
                    if (id === 'inspect') return { kind: 'no_op' as const, reason: 'inspected' };
                    return {
                        kind: 'proposal' as const,
                        proposal: {
                            kind: 'replace_with_compaction' as const,
                            compaction_id: 'summary:1',
                            strategy: { id: 'summarize', version: 'v1', configuration_fingerprint: 'sha256:config' },
                            replacement_turns: [
                                {
                                    ...turn,
                                    id: 'summary-turn',
                                    kind: 'agent' as const,
                                    provenance: {
                                        type: 'derived' as const,
                                        derivation_id: 'summary:1',
                                        source_turn_ids: resolved_input.source_turn_ids,
                                        source_hash: resolved_input.source_fingerprint,
                                    },
                                    blocks: [createTextBlock({ id: 'summary-block', text: 'short', format: 'plain' })],
                                },
                            ],
                            fidelity: 'semantic' as const,
                            retained_asset_ids: [],
                            generation_ids: [],
                            placement: { mode: 'first_selected' as const, causal_order: 'contiguous' as const },
                        },
                    };
                },
            }),
        };
        await runProcessingJob(store, registry, jobs[0].id, 'attempt:summary', () => at);
        await runProcessingJob(store, registry, jobs[1].id, 'attempt:inspect', () => at);
        expect(observed[0]).toEqual(['received-entry']);
        expect(observed[1]).toEqual(store.current.processing.completions?.[jobs[0].id]?.inserted_entry_ids);
        expect(observed[1]).toHaveLength(1);
        expect(store.current.processing.completions?.[jobs[1].id]?.status).toBe('no_op');
    });

    it('classifies an orphan started attempt as unknown rather than reinvoking an external processor', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const source = structuredClone(accepted.document);
        source.processing.resolved_inputs = {
            [job.id]: {
                job_id: job.id,
                source_revision: source.revision,
                context_revision: source.context.revision,
                entry_ids: ['received-entry'],
                source_fingerprint: 'sha256:retained',
                context_fingerprint: 'sha256:retained-context',
                source_turn_ids: ['received-turn'],
                recorded_at: at,
            },
        };
        const { fingerprintJson } = await import('../src/index.js');
        source.processing.attempts = {
            [job.id]: {
                job_id: job.id,
                resolved_input_fingerprint: await fingerprintJson(source.processing.resolved_inputs[job.id]),
                attempt_token: 'attempt:orphan',
                started_at: at,
            },
        };
        const store = new MemoryStore(parseConversationDocument(source));
        const run = vi.fn();
        const observed = await runProcessingJob(
            store,
            { resolve: () => ({ run }) },
            job.id,
            'attempt:observer',
            () => at,
        );
        expect(run).not.toHaveBeenCalled();
        expect(observed.status).toBe('in_progress');
        expect(store.current.processing.outputs?.[job.id]).toBeUndefined();
        await abandonProcessingAttempt(store, {
            operation_id: 'recover:orphan',
            job_id: job.id,
            attempt_token: 'attempt:orphan',
            expected_revision: store.current.revision,
            recorded_at: at,
        });
        expect(store.current.processing.outputs?.[job.id]?.kind).toBe('unknown_outcome');
        expect(store.current.processing.completions?.[job.id]?.status).toBe('blocked');
    });

    it('keeps a second observer pending while the first fenced attempt is running', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        let enter: (() => void) | undefined;
        let release: (() => void) | undefined;
        const entered = new Promise<void>((resolve) => {
            enter = resolve;
        });
        const gate = new Promise<void>((resolve) => {
            release = resolve;
        });
        const run = vi.fn(async () => {
            enter?.();
            await gate;
            return { kind: 'no_op' as const, reason: 'complete' };
        });
        const first = runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:owner', () => at);
        await entered;
        const observer = await runProcessingJob(
            store,
            { resolve: () => ({ run }) },
            job.id,
            'attempt:observer',
            () => at,
        );
        expect(observer.status).toBe('in_progress');
        expect(store.current.processing.outputs?.[job.id]).toBeUndefined();
        expect(run).toHaveBeenCalledTimes(1);
        release?.();
        expect((await first).status).toBe('completed');
        expect(store.current.processing.outputs?.[job.id]?.kind).toBe('no_op');
        expect(run).toHaveBeenCalledTimes(1);
    });

    it('allows only an explicit matching-token abandon to win a race with a late owner', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        let enter: (() => void) | undefined;
        let release: (() => void) | undefined;
        const entered = new Promise<void>((resolve) => {
            enter = resolve;
        });
        const gate = new Promise<void>((resolve) => {
            release = resolve;
        });
        const run = vi.fn(async () => {
            enter?.();
            await gate;
            return { kind: 'no_op' as const, reason: 'late' };
        });
        const owner = runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:owner', () => at);
        await entered;
        await expect(
            abandonProcessingAttempt(store, {
                operation_id: 'recover:wrong',
                job_id: job.id,
                attempt_token: 'attempt:wrong',
                expected_revision: store.current.revision,
                recorded_at: at,
            }),
        ).rejects.toThrow('fenced attempt');
        const abandoned = await abandonProcessingAttempt(store, {
            operation_id: 'recover:owner',
            job_id: job.id,
            attempt_token: 'attempt:owner',
            expected_revision: store.current.revision,
            recorded_at: at,
        });
        expect(abandoned.status).toBe('completed');
        expect(store.current.processing.outputs?.[job.id]).toMatchObject({
            kind: 'unknown_outcome',
            recovery_operation_id: 'recover:owner',
            attempt_token: 'attempt:owner',
        });
        release?.();
        await expect(owner).rejects.toThrow('conflicts with its durable result');
        expect(store.current.processing.outputs?.[job.id]?.kind).toBe('unknown_outcome');
        expect(run).toHaveBeenCalledTimes(1);
    });

    it('retains an exact processor result but rejects application after selected context changes', async () => {
        const accepted = await appendConversationRecordsWithProcessing(
            await enabled([processor('exclude')]),
            batch,
            appendOptions,
        );
        const job = firstJob(accepted.document);
        class StaleStore extends MemoryStore {
            private shifted = false;
            override async commit(expectedRevision: number, document: ConversationDocument) {
                const acceptedCommit = await super.commit(expectedRevision, document);
                if (acceptedCommit && !this.shifted && this.current.processing.outputs?.[job.id]) {
                    this.shifted = true;
                    const plan = await planContextChange(this.current, {
                        expected_revision: this.current.revision,
                        expected_context_revision: this.current.context.revision,
                        entry_ids: ['received-entry'],
                    });
                    this.current = (
                        await applyContextChange(this.current, {
                            operation_id: 'concurrent:edit',
                            expected_revision: this.current.revision,
                            expected_context_revision: this.current.context.revision,
                            expected_source_fingerprint: plan.source_fingerprint,
                            entry_ids: plan.entry_ids,
                            recorded_at: at,
                            proposal: { kind: 'exclude' },
                        })
                    ).document;
                }
                return acceptedCommit;
            }
        }
        const store = new StaleStore(accepted.document);
        const run = vi.fn(async () => ({ kind: 'proposal' as const, proposal: { kind: 'exclude' as const } }));
        await expect(
            runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:stale', () => at),
        ).rejects.toThrow('source context changed');
        expect(run).toHaveBeenCalledTimes(1);
        expect(store.current.processing.outputs?.[job.id]?.kind).toBe('proposal');
        expect(store.current.processing.completions?.[job.id]).toBeUndefined();
    });

    it('rejects oversized processor config before accepting policy or retained job snapshots', async () => {
        const oversized = processor('oversized');
        const config = { instructions: 'x'.repeat(70 * 1024) };
        await expect(
            setProcessingPolicy(base(), {
                operation_id: 'policy:oversized',
                expected_revision: 0,
                recorded_at: at,
                enabled: true,
                processors: [{ ...oversized, config }],
            }),
        ).rejects.toThrow('configuration exceeds durable bound');
        expect(() =>
            parseConversationDocument({
                ...base(),
                processing: {
                    ...base().processing,
                    processors: [{ ...oversized, config }],
                },
            }),
        ).toThrow();
    });

    it('preflights policy and queue objects before cloning accessors', async () => {
        let reads = 0;
        const command = Object.defineProperty(
            { operation_id: 'policy:getter', expected_revision: 0, recorded_at: at, enabled: true, processors: [] },
            'enabled',
            {
                enumerable: true,
                get() {
                    reads += 1;
                    return true;
                },
            },
        );
        await expect(setProcessingPolicy(base(), command)).rejects.toThrow('bounded JSON');
        const source = await enabled([processor('manual', true, 'manual')]);
        const accepted = await appendConversationRecordsWithProcessing(source, batch, appendOptions);
        const selection = {
            conversation: { conversation_id: accepted.document.id, revision: accepted.document.revision },
            expected_context_revision: accepted.document.context.revision,
            selector: { source: { kind: 'all' as const } },
        };
        const queue = Object.defineProperty(
            {
                operation_id: 'queue:getter',
                expected_revision: accepted.document.revision,
                recorded_at: at,
                processor_id: 'manual',
                scope: 'manual' as const,
            },
            'processor_id',
            {
                enumerable: true,
                get() {
                    reads += 1;
                    return 'manual';
                },
            },
        );
        await expect(queueProcessingForExisting(accepted.document, selection, queue)).rejects.toThrow('bounded JSON');
        expect(reads).toBe(0);
    });

    it('requires explicit current coverage, and target changes invalidate readiness', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        expect((await assessProcessingReadiness(accepted.document, 'sha256:target-a', 'sha256:measure-a')).status).toBe(
            'pending',
        );
        await expect(
            assertProcessingReady(accepted.document, 'sha256:target-a', 'sha256:measure-a'),
        ).rejects.toMatchObject({
            code: 'PROCESSING_PENDING',
        });
        const checked = await recordProcessingCoverage(accepted.document, {
            operation_id: 'coverage:1',
            expected_revision: accepted.document.revision,
            target_fingerprint: 'sha256:target-a',
            measured_input_tokens: 10,
            tokenizer_id: 'tokenizer:v1',
            measurement_fingerprint: 'sha256:measure-a',
            recorded_at: at,
        });
        expect(checked.coverage.status).toBe('pending');
        expect((await assessProcessingReadiness(checked.document, 'sha256:target-b', 'sha256:measure-a')).status).toBe(
            'pending',
        );
        const job = firstJob(checked.document);
        const store = new MemoryStore(checked.document);
        await runProcessingJob(
            store,
            { resolve: () => ({ run: async () => ({ kind: 'no_op', reason: 'nothing_to_change' }) }) },
            job.id,
            'attempt:coverage',
            () => at,
        );
        expect((await assessProcessingReadiness(store.current, 'sha256:target-a', 'sha256:measure-a')).status).toBe(
            'pending',
        );
        const ready = await recordProcessingCoverage(store.current, {
            operation_id: 'coverage:2',
            expected_revision: store.current.revision,
            target_fingerprint: 'sha256:target-a',
            measured_input_tokens: 10,
            tokenizer_id: 'tokenizer:v1',
            measurement_fingerprint: 'sha256:measure-a',
            recorded_at: at,
        });
        expect((await assessProcessingReadiness(ready.document, 'sha256:target-a', 'sha256:measure-a')).status).toBe(
            'ready',
        );
        expect((await assessProcessingReadiness(ready.document, 'sha256:target-b', 'sha256:measure-a')).status).toBe(
            'pending',
        );
        expect((await assessProcessingReadiness(ready.document, 'sha256:target-a', 'sha256:new-measure')).status).toBe(
            'pending',
        );
        await expect(
            assertProcessingReady(ready.document, 'sha256:target-a', 'sha256:measure-a'),
        ).resolves.toMatchObject({
            target_fingerprint: 'sha256:target-a',
            status: 'ready',
        });
    });

    it('distinguishes required known failure from an optional skip', async () => {
        for (const required of [true, false]) {
            const accepted = await appendConversationRecordsWithProcessing(
                await enabled([processor(`failing-${required}`, required)]),
                batch,
                appendOptions,
            );
            const job = firstJob(accepted.document);
            const store = new MemoryStore(accepted.document);
            await runProcessingJob(
                store,
                {
                    resolve: () => ({
                        run: async () => {
                            throw new ProcessingKnownFailure('deterministic rejection', { usage: { input_tokens: 2 } });
                        },
                    }),
                },
                job.id,
                `attempt:${required}`,
                () => at,
            );
            expect(store.current.processing.outputs?.[job.id]?.kind).toBe('failed');
            expect(store.current.processing.outputs?.[job.id]).toMatchObject({ usage: { input_tokens: 2 } });
            expect(store.current.processing.completions?.[job.id]?.status).toBe(required ? 'blocked' : 'skipped');
            const coverage = await recordProcessingCoverage(store.current, {
                operation_id: `coverage:${required}`,
                expected_revision: store.current.revision,
                target_fingerprint: 'sha256:target',
                measured_input_tokens: 1,
                tokenizer_id: 'tokenizer:v1',
                measurement_fingerprint: 'sha256:measure-b',
                recorded_at: at,
            });
            expect(coverage.coverage.status).toBe(required ? 'blocked' : 'ready');
        }
    });

    it('retains reported usage on an owner-observed unknown external failure', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        await runProcessingJob(
            store,
            {
                resolve: () => ({
                    run: async () => {
                        throw new ProcessingUnknownFailure('provider acknowledgement lost', {
                            usage: { output_tokens: 4 },
                        });
                    },
                }),
            },
            job.id,
            'attempt:unknown',
            () => at,
        );
        expect(store.current.processing.outputs?.[job.id]).toMatchObject({
            kind: 'unknown_outcome',
            usage: { output_tokens: 4 },
        });
    });

    it('preserves independently valid usage when returned output exceeds the durable bound', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        const run = vi.fn(async () => ({
            kind: 'no_op' as const,
            reason: 'x'.repeat(1024 * 1024),
            usage: { input_tokens: 7 },
        }));
        await runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:oversized', () => at);
        expect(run).toHaveBeenCalledTimes(1);
        expect(store.current.processing.outputs?.[job.id]).toMatchObject({
            kind: 'unknown_outcome',
            usage: { input_tokens: 7 },
        });
        expect(store.current.processing.completions?.[job.id]?.status).toBe('blocked');
    });

    it('preserves valid usage when a returned processor result is malformed', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        await runProcessingJob(
            store,
            { resolve: () => ({ run: async () => ({ kind: 'no_op', reason: '', usage: { output_tokens: 4 } }) }) },
            job.id,
            'attempt:malformed',
            () => at,
        );
        expect(store.current.processing.outputs?.[job.id]).toMatchObject({
            kind: 'unknown_outcome',
            usage: { output_tokens: 4 },
        });
    });

    it('never invokes an accessor on a returned processor result', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        const reasonGetter = vi.fn(() => 'unsafe');
        const run = vi.fn(async () => ({
            kind: 'no_op' as const,
            get reason() {
                return reasonGetter();
            },
            usage: { output_tokens: 3 },
        }));
        await runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:accessor', () => at);
        expect(run).toHaveBeenCalledTimes(1);
        expect(reasonGetter).not.toHaveBeenCalled();
        expect(store.current.processing.outputs?.[job.id]).toMatchObject({
            kind: 'unknown_outcome',
            usage: { output_tokens: 3 },
        });
    });

    it('returns the original accepted coverage on retry after a newer measurement', async () => {
        const source = await enabled([]);
        const command = {
            operation_id: 'coverage:original',
            expected_revision: source.revision,
            target_fingerprint: 'sha256:target-a',
            measured_input_tokens: 10,
            tokenizer_id: 'tokenizer:v1',
            measurement_fingerprint: 'sha256:measure-original',
            recorded_at: at,
        };
        const original = await recordProcessingCoverage(source, command);
        const newer = await recordProcessingCoverage(original.document, {
            ...command,
            operation_id: 'coverage:newer',
            expected_revision: original.document.revision,
            measured_input_tokens: 20,
            measurement_fingerprint: 'sha256:measure-newer',
        });
        const retry = await recordProcessingCoverage(newer.document, command);
        expect(retry.applied).toBe(false);
        expect(retry.coverage).toEqual(original.coverage);
        expect(retry.change).toEqual(original.change);
        expect(retry.document.processing.coverage).toEqual(newer.coverage);
        const forged = parseConversationDocument({
            ...newer.document,
            processing: {
                ...newer.document.processing,
                coverage_receipts: {
                    ...newer.document.processing.coverage_receipts,
                    [command.operation_id]: {
                        ...original.coverage,
                        measurement: { ...original.coverage.measurement, input_tokens: 999 },
                    },
                },
            },
        });
        await expect(recordProcessingCoverage(forged, command)).rejects.toThrow('accepted target');
    });

    it('keeps strict persisted job and output JSON schemas in Zod/AJV parity', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const store = new MemoryStore(accepted.document);
        await runProcessingJob(
            store,
            { resolve: () => ({ run: async () => ({ kind: 'no_op', reason: 'unchanged' }) }) },
            job.id,
            'attempt:parity',
            () => at,
        );
        const output = store.current.processing.outputs?.[job.id];
        const ajv = new Ajv2020({ allErrors: true, strict: true });
        formatsPlugin.default(ajv);
        const validateJob = ajv.compile(ProcessingJobJsonSchema);
        const validateOutput = ajv.compile(ProcessingOutputReceiptJsonSchema);
        for (const fixture of [
            job,
            {
                ...job,
                selection: {
                    kind: 'entries',
                    entry_ids: ['received-entry'],
                    selected_entries: [
                        { id: 'received-entry', type: 'source_turn', turn_id: 'received-turn', extra: true },
                    ],
                },
            },
        ]) {
            expect(validateJob(fixture), JSON.stringify(validateJob.errors)).toBe(
                ProcessingJobSchema.safeParse(fixture).success,
            );
        }
        for (const fixture of [output, { ...output, usage: { input_tokens: 1, extra: true } }]) {
            expect(validateOutput(fixture), JSON.stringify(validateOutput.errors)).toBe(
                ProcessingOutputReceiptSchema.safeParse(fixture).success,
            );
        }
        expect(validateJob(job)).toBe(true);
        expect(validateOutput(output)).toBe(true);
        expect(
            validateJob({
                ...job,
                selection: {
                    kind: 'entries',
                    entry_ids: ['received-entry'],
                    selected_entries: [
                        { id: 'received-entry', type: 'source_turn', turn_id: 'received-turn', extra: true },
                    ],
                },
            }),
        ).toBe(false);
        expect(validateOutput({ ...output, usage: { input_tokens: 1, extra: true } })).toBe(false);
    });

    it('accepts exact policy and existing-selection retries, then rejects conflicting reuse', async () => {
        const command = {
            operation_id: 'policy:manual',
            expected_revision: 0,
            recorded_at: at,
            enabled: true,
            processors: [processor('manual', true, 'manual')],
        };
        const enabledSource = await setProcessingPolicy(base(), command);
        expect((await setProcessingPolicy(enabledSource.document, command)).applied).toBe(false);
        await expect(setProcessingPolicy(enabledSource.document, { ...command, enabled: false })).rejects.toThrow(
            'conflicts',
        );
        const accepted = await appendConversationRecordsWithProcessing(enabledSource.document, batch, appendOptions);
        const selection = {
            conversation: { conversation_id: accepted.document.id, revision: accepted.document.revision },
            expected_context_revision: accepted.document.context.revision,
            selector: { source: { kind: 'all' as const } },
        };
        const queue = {
            operation_id: 'queue:manual',
            expected_revision: accepted.document.revision,
            recorded_at: at,
            processor_id: 'manual',
            scope: 'manual' as const,
        };
        const first = await queueProcessingForExisting(accepted.document, selection, queue);
        const retried = await queueProcessingForExisting(first.document, selection, queue);
        expect(retried.applied).toBe(false);
        expect(retried.job_id).toBe(first.job_id);
        expect(retried.document.processing.jobs).toEqual(first.document.processing.jobs);
    });

    it('keeps original processor indices and accepted configs across later policy updates', async () => {
        const source = await enabled([processor('manual', true, 'manual'), processor('append', true, 'on_append')]);
        const accepted = await appendConversationRecordsWithProcessing(source, batch, appendOptions);
        const job = firstJob(accepted.document);
        expect(job.processor_index).toBe(1);
        expect(job.stage_index).toBe(0);
        const changed = await setProcessingPolicy(accepted.document, {
            operation_id: 'policy:2',
            expected_revision: accepted.document.revision,
            recorded_at: at,
            enabled: true,
            processors: [processor('different', true, 'manual')],
        });
        const store = new MemoryStore(changed.document);
        const run = vi.fn(async () => ({ kind: 'no_op' as const, reason: 'old_snapshot' }));
        await runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:old-policy', () => at);
        expect(run).toHaveBeenCalledTimes(1);
        expect(store.current.processing.completions?.[job.id]?.status).toBe('no_op');
        const coverage = await recordProcessingCoverage(store.current, {
            operation_id: 'coverage:old',
            expected_revision: store.current.revision,
            target_fingerprint: 'sha256:old-target',
            measured_input_tokens: 3,
            tokenizer_id: 'tokenizer:v1',
            measurement_fingerprint: 'sha256:old-measure',
            recorded_at: at,
        });
        expect(coverage.coverage.required_job_ids).toContain(job.id);
    });

    it('requires an explicit receipt-linked supersession before disabling accepted pending jobs', async () => {
        const accepted = await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions);
        const job = firstJob(accepted.document);
        const command = {
            operation_id: 'policy:disable',
            expected_revision: accepted.document.revision,
            recorded_at: at,
            enabled: false,
            processors: [],
        };
        await expect(setProcessingPolicy(accepted.document, command)).rejects.toThrow(
            'explicit pending-job supersession',
        );
        const disabled = await setProcessingPolicy(accepted.document, {
            ...command,
            supersede_job_ids: [job.id],
            supersession_reason: 'operator_cancelled',
        });
        expect(disabled.document.processing.supersessions?.[job.id]).toMatchObject({
            policy_operation_id: command.operation_id,
            reason: 'operator_cancelled',
        });
        expect(
            disabled.document.operation_receipts[command.operation_id].processing_operation?.superseded_job_ids,
        ).toEqual([job.id]);
        const store = new MemoryStore(disabled.document);
        const run = vi.fn();
        expect(
            (await runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:disabled', () => at)).status,
        ).toBe('superseded');
        expect(run).not.toHaveBeenCalled();
        expect(
            (
                await setProcessingPolicy(disabled.document, {
                    ...command,
                    supersede_job_ids: [job.id],
                    supersession_reason: 'operator_cancelled',
                })
            ).applied,
        ).toBe(false);
    });

    it('compares the declared input cap directly and leaves context-window reserve to provider preparation', async () => {
        const source = (
            await setProcessingPolicy(base(), {
                operation_id: 'policy:budget',
                expected_revision: 0,
                recorded_at: at,
                enabled: true,
                processors: [],
                budget: { max_input_tokens: 100, output_reserve_tokens: 40 },
            })
        ).document;
        const within = await recordProcessingCoverage(source, {
            operation_id: 'coverage:within',
            expected_revision: source.revision,
            recorded_at: at,
            target_fingerprint: 'sha256:target',
            measured_input_tokens: 90,
            tokenizer_id: 'tokenizer:v1',
            measurement_fingerprint: 'sha256:90',
        });
        expect(within.coverage.status).toBe('ready');
        const over = await recordProcessingCoverage(within.document, {
            operation_id: 'coverage:over',
            expected_revision: within.document.revision,
            recorded_at: at,
            target_fingerprint: 'sha256:target',
            measured_input_tokens: 101,
            tokenizer_id: 'tokenizer:v1',
            measurement_fingerprint: 'sha256:101',
        });
        expect(over.coverage.status).toBe('blocked');
    });

    it('selects only eligible text from a mixed pending-tool turn and no-ops on a completed tool result', async () => {
        const source = await enabled();
        const generation = {
            ...importedGeneration('generation:mixed'),
            source: { conversation_id: source.id, revision: source.revision },
        };
        const agent = generatedAgentTurn('agent:mixed', generation.id, [
            createTextBlock({ id: 'agent:text', text: 'useful', format: 'plain' }),
            toolCallBlock('agent:call', 'call:mixed'),
        ]);
        const first = await appendConversationRecordsWithProcessing(
            source,
            {
                turns: [agent],
                generations: [generation],
                context_entries: [{ id: 'entry:agent', type: 'source_turn', turn_id: agent.id }],
            },
            { ...appendOptions, operation_id: 'append:mixed', payload_fingerprint: 'sha256:mixed' },
        );
        const job = firstJob(first.document);
        expect(job.selection).toMatchObject({
            kind: 'entries',
            entry_ids: ['entry:agent'],
            selected_block_ids: { 'entry:agent': ['agent:text'] },
        });
        const store = new MemoryStore(first.document);
        const run = vi.fn(async () => ({ kind: 'no_op' as const, reason: 'preserve_tool_call' }));
        await runProcessingJob(store, { resolve: () => ({ run }) }, job.id, 'attempt:mixed', () => at);
        expect(store.current.processing.resolved_inputs?.[job.id]?.selected_block_ids).toEqual({
            'entry:agent': ['agent:text'],
        });
        const result = toolResultTurn('tool:result', 'call:mixed');
        const second = await appendConversationRecordsWithProcessing(
            store.current,
            {
                turns: [result],
                context_entries: [{ id: 'entry:tool', type: 'source_turn', turn_id: result.id }],
            },
            {
                operation_id: 'append:tool',
                expected_revision: store.current.revision,
                payload_fingerprint: 'sha256:tool',
                recorded_at: at,
            },
        );
        const resultJob = Object.values(second.document.processing.jobs ?? {}).find(
            (item) => item.source_operation_id === 'append:tool',
        );
        expect(resultJob?.selection).toMatchObject({ kind: 'entries', entry_ids: [] });
        const resultStore = new MemoryStore(second.document);
        await runProcessingJob(
            resultStore,
            { resolve: () => ({ run }) },
            resultJob?.id ?? '',
            'attempt:tool',
            () => at,
        );
        expect(resultStore.current.processing.outputs?.[resultJob?.id ?? '']?.kind).toBe('no_op');
        expect(run).toHaveBeenCalledTimes(1);
    });

    it('preserves a partial selection through a no-op predecessor stage', async () => {
        const source = await enabled([processor('first'), processor('second')]);
        const generation = {
            ...importedGeneration('generation:partial'),
            source: { conversation_id: source.id, revision: source.revision },
        };
        const agent = generatedAgentTurn('agent:partial', generation.id, [
            createTextBlock({ id: 'partial:text', text: 'useful', format: 'plain' }),
            toolCallBlock('partial:call', 'call:partial'),
        ]);
        const accepted = await appendConversationRecordsWithProcessing(
            source,
            {
                turns: [agent],
                generations: [generation],
                context_entries: [{ id: 'entry:partial', type: 'source_turn', turn_id: agent.id }],
            },
            { ...appendOptions, operation_id: 'append:partial', payload_fingerprint: 'sha256:partial' },
        );
        const jobs = Object.values(accepted.document.processing.jobs ?? {});
        const store = new MemoryStore(accepted.document);
        const seen: Array<Record<string, string[]> | undefined> = [];
        const registry = {
            resolve: () => ({
                run: async ({
                    resolved_input,
                }: {
                    resolved_input: NonNullable<ConversationDocument['processing']['resolved_inputs']>[string];
                }) => {
                    seen.push(resolved_input.selected_block_ids);
                    return { kind: 'no_op' as const, reason: 'retain' };
                },
            }),
        };
        await runProcessingJob(store, registry, jobs[0].id, 'attempt:partial-1', () => at);
        await runProcessingJob(store, registry, jobs[1].id, 'attempt:partial-2', () => at);
        expect(seen).toEqual([{ 'entry:partial': ['partial:text'] }, { 'entry:partial': ['partial:text'] }]);
    });
});

describe('trusted target capture at processing resolution', () => {
    async function acceptedSource() {
        return (await appendConversationRecordsWithProcessing(await enabled(), batch, appendOptions)).document;
    }
    const registry = {
        resolve: () => ({ run: async () => ({ kind: 'no_op' as const, reason: 'unchanged' }) }),
    };

    it('captures target before await, retains it across JSON reload and rejects changed target after completion', async () => {
        const store = new MemoryStore(await acceptedSource());
        const job = firstJob(store.current);
        const acceptedJob = structuredClone(job);
        const capability = { target_fingerprint: 'sha256:target-a' };
        const pending = runProcessingJob(store, registry, job.id, 'attempt:target', () => at, undefined, capability);
        capability.target_fingerprint = 'sha256:target-mutated';
        expect((await pending).status).toBe('completed');
        expect(store.current.processing.resolved_inputs?.[job.id]?.target_fingerprint).toBe('sha256:target-a');
        expect(store.current.processing.jobs?.[job.id]).toEqual(acceptedJob);
        store.current = parseConversationDocument(JSON.parse(JSON.stringify(store.current)));
        const receiptCount = Object.keys(store.current.operation_receipts).length;
        await expect(
            runProcessingJob(store, registry, job.id, 'attempt:retry', () => at, undefined, {
                target_fingerprint: 'sha256:target-a',
            }),
        ).resolves.toMatchObject({ status: 'completed' });
        await expect(
            runProcessingJob(store, registry, job.id, 'attempt:conflict', () => at, undefined, {
                target_fingerprint: 'sha256:target-b',
            }),
        ).rejects.toThrow('resolution target evidence');
        expect(Object.keys(store.current.operation_receipts)).toHaveLength(receiptCount);
    });

    it('resolves exactly one target under competing CAS and never invokes the losing processor', async () => {
        const store = new MemoryStore(await acceptedSource());
        const job = firstJob(store.current);
        const calls: string[] = [];
        const forTarget = (target: string) => ({
            resolve: () => ({
                run: async () => {
                    calls.push(target);
                    return { kind: 'no_op' as const, reason: 'unchanged' };
                },
            }),
        });
        const results = await Promise.allSettled(
            ['sha256:target-a', 'sha256:target-b'].map((target, index) =>
                runProcessingJob(store, forTarget(target), job.id, `attempt:race-${index}`, () => at, undefined, {
                    target_fingerprint: target,
                }),
            ),
        );
        expect(results.filter((result) => result.status === 'rejected')).toHaveLength(1);
        const target = store.current.processing.resolved_inputs?.[job.id]?.target_fingerprint;
        expect(calls).toEqual([target]);
        expect(
            Object.values(store.current.operation_receipts).filter(
                (receipt) => receipt.processing_operation?.phase === 'resolve',
            ),
        ).toHaveLength(1);
    });

    it('rejects a conflicting already-bound job before resolution or processor I/O', async () => {
        const source = await enabled([processor('noop', true, 'manual')]);
        const appended = await appendConversationRecordsWithProcessing(source, batch, appendOptions);
        const queued = await queueProcessingForExisting(
            appended.document,
            {
                conversation: { conversation_id: appended.document.id, revision: appended.document.revision },
                expected_context_revision: appended.document.context.revision,
                selector: { source: { kind: 'all' } },
            },
            {
                operation_id: 'queue:bound',
                expected_revision: appended.document.revision,
                recorded_at: at,
                processor_id: 'noop',
                scope: 'manual',
                target_fingerprint: 'sha256:target-a',
            },
        );
        const store = new MemoryStore(queued.document);
        const resolve = vi.fn(registry.resolve);
        await expect(
            runProcessingJob(store, { resolve }, firstJob(store.current).id, 'attempt:bound', () => at, undefined, {
                target_fingerprint: 'sha256:target-b',
            }),
        ).rejects.toThrow('job target conflicts');
        expect(resolve).not.toHaveBeenCalled();
        expect(store.current).toEqual(queued.document);
    });

    it('preserves legacy unbound resolutions rather than silently upgrading their target evidence', async () => {
        const store = new MemoryStore(await acceptedSource());
        const job = firstJob(store.current);
        await runProcessingJob(store, registry, job.id, 'attempt:legacy', () => at);
        expect(store.current.processing.resolved_inputs?.[job.id]?.target_fingerprint).toBeUndefined();
        const retained = structuredClone(store.current);
        await expect(
            runProcessingJob(store, registry, job.id, 'attempt:legacy-upgrade', () => at, undefined, {
                target_fingerprint: 'sha256:target-a',
            }),
        ).rejects.toThrow('resolution target evidence');
        expect(store.current).toEqual(retained);
    });
});
