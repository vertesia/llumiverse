import { describe, expect, it } from 'vitest';
import {
    type Asset,
    appendConversationRecordsWithProcessing,
    appendToolExecutionResult,
    applyContextChange,
    applyToolResultTextExternalizationOutput,
    type ConversationDocument,
    type ConversationRecordBatch,
    ConversationToolExecutionResultSchema,
    conversationDocumentFromJson,
    conversationDocumentToJson,
    createConversationDocument,
    createTextExternalizationProcessor,
    createToolResultTextExternalizationProcessor,
    fingerprintJson,
    getPagedRecord,
    hashContentBytes,
    hashUtf8Content,
    isToolResultTextProcessor,
    type NativeReplayBlock,
    type ProcessingStore,
    parseConversationDocument,
    planContextChange,
    putPagedRecord,
    readPagedRecordRange,
    removePagedRecord,
    resolveActiveTextExternalReference,
    runProcessingJob,
    setProcessingPolicy,
    type ToolDefinition,
    type ToolResultBlock,
    type ToolResultTextStrategy,
    toolResultExternalizationArchiveInputs,
    toolResultTextSelection,
} from '../src/index.js';
import {
    type IndexedConversationRecordStore,
    IndexedToolResultOriginalSourceSchema,
    loadIndexedProcessingToolResultSelectedContext,
    loadIndexedSettledProcessingSelectedContext,
    loadIndexedToolCallTerminalResult,
    loadRecord,
    recoverIndexedToolResultValidations,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingPhase,
    stageIndexedRecordBatch,
    stageIndexedTextProcessingCompletion,
    stageRecord,
} from '../src/indexed-conversation.js';
import {
    applyIndexedTextExternalizationOutput,
    buildIndexedTextExternalizationOutput,
    indexedCompletedJobEntrySelection,
    resolveIndexedProcessingTextInput,
} from '../src/indexed-processing-working-set.js';
import { finishIndexedConversationUpgrade } from '../src/indexed-upgrade-finish.js';
import { beginIndexedConversationUpgrade } from '../src/indexed-upgrade-progress.js';
import { advanceIndexedConversationUpgrade } from '../src/indexed-upgrade-step.js';
import { ExecutionReceiptSchema, OperationReceiptSchema } from '../src/schemas/execution.js';
import { IndexedConversationTurnHeaderSchema } from '../src/schemas/indexed-head.js';
import { IndexedProcessingClaimWorkspaceSchema } from '../src/schemas/indexed-processing.js';
import { INDEXED_CONVERSATION_UPGRADE_PROFILE } from '../src/schemas/indexed-upgrade.js';
import {
    ProcessingAttemptReceiptSchema,
    ProcessingCompletionReceiptSchema,
    ProcessingJobSchema,
} from '../src/schemas/processing.js';

const at = '2026-10-04T00:00:00.000Z';
const originalText = `Exact result α\n${'long result '.repeat(1000)}`;

async function ownedImage(id: string): Promise<Asset> {
    const data = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lL0AAAAASUVORK5CYII=';
    const bytes = Uint8Array.from(atob(data), (character) => character.charCodeAt(0));
    return {
        id,
        kind: 'image',
        mime_type: 'image/png',
        storage: { type: 'inline_base64', data },
        provenance: { type: 'received', source_turn_id: 'turn:result' },
        created_at: at,
        ...(await hashContentBytes(bytes)),
    };
}

function callReplay(): NativeReplayBlock {
    // The ordinary Responses function_call replay remains tied to the unchanged executable call.
    return {
        id: 'replay:call',
        type: 'native_replay',
        adapter: 'openai-responses@1',
        protocol: 'openai.responses',
        compatibility_scope: {
            provider: 'openai',
            protocol: 'openai.responses',
            adapter_version: 'openai-responses@1',
        },
        payload: {
            type: 'openai_responses_items',
            items: [
                {
                    type: 'function_call',
                    call_id: 'call:one',
                    name: 'write_artifact',
                    arguments: '{"path":"report.txt","content":"execution bytes"}',
                },
            ],
            semantic_entries: [{ kind: 'tool_call', block_id: 'block:call', call_id: 'call:one', item_index: 0 }],
            block_offset: 0,
            item_order: 0,
        },
        dependencies: { turn_ids: [], block_ids: ['block:call'], call_ids: ['call:one'], request_ids: [] },
        dependency_policy: 'discard_on_dependency_change',
    };
}

async function completedResult(
    withTextPredecessor = false,
    withCallReplay = false,
    partial?: {
        strategy: ToolResultTextStrategy;
        content: ToolResultBlock['content'];
        assets?: Asset[];
        definition?: ToolDefinition;
        retrieval_requirements?: ConversationRecordBatch['retrieval_requirements'];
    },
) {
    let initial: ConversationDocument = createConversationDocument({ id: 'tool-result-text', created_at: at });
    initial.turns.push({
        id: 'turn:call',
        kind: 'agent',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'imported', source: 'test' },
        model_visibility: 'include',
        blocks: [
            {
                id: 'block:call',
                type: 'tool_call',
                call_id: 'call:one',
                tool_name: 'write_artifact',
                executor: 'application',
                arguments: { type: 'json', value: { path: 'report.txt', content: 'execution bytes' } },
            },
        ],
    });
    const callTurn = initial.turns[0];
    if (callTurn.kind !== 'agent') throw new Error('Fixture must retain an agent call turn');
    if (withCallReplay) callTurn.blocks.push(callReplay());
    initial.context.entries.push({ id: 'entry:call', type: 'source_turn', turn_id: 'turn:call' });
    if (partial) {
        // The v2 fixture accepts a real executed generation, so its original call can also
        // be migrated to the strict indexed call/generation-acceptance reader.
        const target = { provider: 'test', protocol: 'test.generate', model: 'model', adapter_version: '1' };
        const originalCall = callTurn.blocks.map((block) =>
            block.type === 'tool_call' ? { ...block, definition_id: 'definition:writer' } : block,
        );
        initial.turns = [];
        initial.context.entries = [];
        const writerDefinition = {
            id: 'definition:writer',
            name: 'write_artifact',
            version: '1',
            input_schema: { type: 'object' },
        };
        const advertisedDefinitions = [writerDefinition, ...(partial.definition ? [partial.definition] : [])];
        const definitionBatch = {
            tool_definitions: advertisedDefinitions,
            active_tool_definition_ids: advertisedDefinitions.map((definition) => definition.id),
        } satisfies ConversationRecordBatch;
        initial = (
            await appendConversationRecordsWithProcessing(initial, definitionBatch, {
                operation_id: 'append:writer-definition',
                expected_revision: initial.revision,
                recorded_at: at,
                payload_fingerprint: await fingerprintJson(definitionBatch),
            })
        ).document;
        const source = { conversation_id: initial.id, revision: initial.revision };
        const callBatch = {
            turns: [
                {
                    ...callTurn,
                    blocks: originalCall,
                    generation_id: 'generation:call',
                    provenance: { type: 'generated' },
                },
            ],
            context_entries: [{ id: 'entry:call', type: 'source_turn', turn_id: callTurn.id }],
            generations: [
                {
                    id: 'generation:call',
                    record_source: 'executed',
                    request_id: 'request:call',
                    attempt_id: 'attempt:call',
                    purpose: 'conversation',
                    requested_model: target.model,
                    provider: target.provider,
                    protocol: target.protocol,
                    adapter_version: target.adapter_version,
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    source,
                    request_receipt: {
                        id: 'receipt:call',
                        request_id: 'request:call',
                        attempt_id: 'attempt:call',
                        source,
                        context_fingerprint: await fingerprintJson(initial.context),
                        tool_set_fingerprint: await fingerprintJson(advertisedDefinitions),
                        request_fingerprint: await fingerprintJson({ target, source }),
                        target,
                        tool_definition_ids: advertisedDefinitions.map((definition) => definition.id),
                        asset_versions: [],
                        item_mappings: [],
                        recorded_at: at,
                    },
                },
            ],
        } satisfies ConversationRecordBatch;
        initial = (
            await appendConversationRecordsWithProcessing(initial, callBatch, {
                operation_id: 'append:call',
                expected_revision: initial.revision,
                recorded_at: at,
                payload_fingerprint: await fingerprintJson(callBatch),
            })
        ).document;
    }
    const configured = (
        await setProcessingPolicy(initial, {
            operation_id: 'policy:tool-results',
            expected_revision: initial.revision,
            recorded_at: at,
            enabled: true,
            processors: [
                ...(withTextPredecessor
                    ? [
                          {
                              id: 'externalize-text',
                              version: '1',
                              scope: 'on_append' as const,
                              config: {},
                              required: true,
                              failure_behavior: 'block' as const,
                          },
                      ]
                    : []),
                {
                    id: partial?.strategy.processor_id ?? 'externalize-tool-result-text',
                    version: partial?.strategy.processor_version ?? '1',
                    scope: 'on_append',
                    config: partial?.strategy.configuration ?? {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        })
    ).document;
    const source = {
        conversation: { conversation_id: initial.id, revision: configured.revision },
        turn_id: 'turn:call',
        block_id: 'block:call',
        call_id: 'call:one',
        call_fingerprint: await fingerprintJson(initial.turns[0].blocks[0]),
    };
    const block = {
        id: 'block:result',
        type: 'tool_result' as const,
        call_id: source.call_id,
        status: 'success' as const,
        content: partial?.content ?? [
            { id: 'block:text', type: 'text' as const, format: 'plain' as const, text: originalText },
        ],
    };
    const result = ConversationToolExecutionResultSchema.parse({
        source,
        ...(partial?.assets ? { assets: partial.assets } : {}),
        turn: {
            id: 'turn:result',
            kind: 'tool' as const,
            authority: 'ordinary' as const,
            status: 'completed' as const,
            timestamps: { recorded_at: at },
            provenance: { type: 'received' as const },
            model_visibility: 'include' as const,
            blocks: [block],
            execution_id: 'execution:one',
        },
        execution_receipt: {
            id: 'execution:one',
            call_id: source.call_id,
            executor: 'application' as const,
            status: 'success' as const,
            result_turn_id: 'turn:result',
            result_fingerprint: await fingerprintJson(block),
            recorded_at: at,
            call_source: source,
        },
    });
    const options = { expected_revision: configured.revision, operation_id: 'append:result', recorded_at: at };
    const retrievalBatch = partial?.retrieval_requirements
        ? ({
              turns: [result.turn],
              execution_receipts: [result.execution_receipt],
              ...(result.assets ? { assets: result.assets } : {}),
              context_entries: [{ id: 'entry:result', type: 'source_turn' as const, turn_id: result.turn.id }],
              retrieval_requirements: partial.retrieval_requirements,
          } satisfies ConversationRecordBatch)
        : undefined;
    const accepted = retrievalBatch
        ? await appendConversationRecordsWithProcessing(configured, retrievalBatch, {
              ...options,
              payload_fingerprint: await fingerprintJson(retrievalBatch),
          })
        : await appendToolExecutionResult(configured, result, options);
    const document = accepted.document;
    const job = Object.values(document.processing.jobs ?? {}).find(isToolResultTextProcessor);
    if (job?.selection.kind !== 'entries' || job.selection.entry_ids.length !== 1)
        throw new Error('Fixture must enqueue one exact completed tool result');
    return { document, job, entryId: job.selection.entry_ids[0] };
}

class MemoryStore implements ProcessingStore {
    constructor(public current: ConversationDocument) {}
    async load() {
        return structuredClone(this.current);
    }
    async commit(revision: number, document: ConversationDocument) {
        if (this.current.revision !== revision) return false;
        this.current = parseConversationDocument(structuredClone(document));
        return true;
    }
}

describe('completed tool-result text processing', () => {
    it('commits real indexed partial phases and reads the derived result through original terminal authority', async () => {
        const retainedJson = '{"retained":true}';
        const retainedIntegrity = await hashUtf8Content(retainedJson);
        const priorRetrieval = {
            capability: 'read_existing',
            version: 1 as const,
            tool_definition_id: 'definition:existing-reader',
            arguments: { path: 'existing.json' },
        };
        const priorAsset: Asset = {
            id: 'asset:existing-json',
            kind: 'json',
            mime_type: 'application/json',
            storage: { type: 'external', resolver: 'test.blob', locator: { key: 'existing.json' } },
            provenance: { type: 'received', source_turn_id: 'turn:result' },
            created_at: at,
            ...retainedIntegrity,
        };
        const f = await completedResult(false, false, {
            strategy: {
                processor_id: 'externalize-tool-result-text',
                processor_version: '2',
                configuration: {
                    selector: {
                        kind: 'exact_blocks',
                        blocks: [{ turn_id: 'turn:result', result_block_id: 'block:result', block_id: 'text:chosen' }],
                    },
                },
            },
            content: [
                { id: 'text:chosen', type: 'text', format: 'plain', text: originalText },
                { id: 'text:other', type: 'text', format: 'plain', text: `${originalText} preserved` },
                { id: 'json:other', type: 'json', value: { preserved: true } },
                { id: 'image:indexed', type: 'image', asset_id: 'asset:indexed-image' },
                {
                    id: 'reference:existing-json',
                    type: 'external_reference',
                    original_type: 'json',
                    asset_id: priorAsset.id,
                    content_hash: retainedIntegrity.content_hash,
                    description: 'Existing accepted JSON archive',
                    retrieval: priorRetrieval,
                },
            ],
            definition: {
                id: priorRetrieval.tool_definition_id,
                name: priorRetrieval.capability,
                version: '1',
                input_schema: true,
            },
            retrieval_requirements: [
                {
                    id: 'requirement:existing-json',
                    asset_id: priorAsset.id,
                    retrieval: priorRetrieval,
                    accepted_asset_operation_id: 'append:result',
                },
            ],
            assets: [priorAsset, await ownedImage('asset:indexed-image')],
        });
        const originals = await toolResultExternalizationArchiveInputs(f.document, f.job);
        const asset = {
            id: 'asset:indexed-original',
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: { type: 'external' as const, resolver: 'test.blob', locator: { key: 'indexed-original' } },
            provenance: { type: 'received' as const },
            created_at: at,
            ...originals.integrities[0],
        };
        const materializedArchive = await appendConversationRecordsWithProcessing(
            f.document,
            {
                assets: [asset],
                tool_definitions: [{ id: 'definition:read', name: 'read_artifact', version: '1', input_schema: true }],
                active_tool_definition_ids: ['definition:read', ...f.document.context.active_tool_definition_ids],
            },
            {
                expected_revision: f.document.revision,
                operation_id: `processing:archive:${f.job.id}`,
                payload_fingerprint: originals.payload_fingerprint,
                recorded_at: at,
            },
        );
        const materialized = new MemoryStore(materializedArchive.document);
        const processor = createToolResultTextExternalizationProcessor(({ asset }) => ({
            capability: 'read_artifact',
            version: 1,
            tool_definition_id: 'definition:read',
            arguments: { asset_id: asset.id },
        }));
        await runProcessingJob(materialized, { resolve: () => processor }, f.job.id, 'attempt:materialized', () => at);
        expect(materialized.current.processing.completions?.[f.job.id].status).toBe('applied');
        expect(
            materialized.current.context.retrieval_requirements.filter(
                (requirement) => requirement.asset_id === priorAsset.id,
            ),
        ).toEqual(
            f.document.context.retrieval_requirements.filter((requirement) => requirement.asset_id === priorAsset.id),
        );
        expect(Object.values(materialized.current.compactions)[0].retained_asset_ids).toEqual([
            asset.id,
            'asset:indexed-image',
            priorAsset.id,
        ]);
        expect(materialized.current.execution_receipts).toEqual(f.document.execution_receipts);
        const bytes = new Map<string, Uint8Array>();
        const preparationReads: { id: string; kind: string; size_bytes: number }[] = [];
        let forbidColdResults = false;
        let coldOutputHash: string | undefined;
        const store: IndexedConversationRecordStore = {
            async read(ref) {
                const value = bytes.get(ref.content_hash);
                if (!value) throw new Error('Missing owned page');
                return Uint8Array.from(value);
            },
            async write(value, ref) {
                bytes.set(ref.content_hash, Uint8Array.from(value));
            },
            async readRecord(ref) {
                preparationReads.push({ id: ref.id, kind: ref.kind, size_bytes: ref.size_bytes });
                if (
                    forbidColdResults &&
                    ((ref.kind === 'blocks' && ref.id === 'block:result') || ref.content_hash === coldOutputHash)
                )
                    throw new Error('Cold original body is deliberately unavailable during prepare');
                const value = bytes.get(ref.content_hash);
                if (!value) throw new Error('Missing owned record');
                return Uint8Array.from(value);
            },
            async writeRecord(ref, value) {
                bytes.set(ref.content_hash, Uint8Array.from(value));
            },
            async assertExternalAssetIntegrity(value) {
                const expected = value.id === priorAsset.id ? priorAsset : asset;
                expect(value.content_hash).toBe(expected.content_hash);
                expect(value.byte_length).toBe(expected.byte_length);
            },
        };
        let staged = await stageIndexedConversationSnapshot(f.document, undefined, store);
        const archived = await stageIndexedRecordBatch(
            staged.root,
            {
                conversation_id: f.document.id,
                batch: {
                    assets: [asset],
                    tool_definitions: [
                        { id: 'definition:read', name: 'read_artifact', version: '1', input_schema: true },
                    ],
                    active_tool_definition_ids: [
                        'definition:read',
                        ...(f.job.processor_version === '2' ? f.document.context.active_tool_definition_ids : []),
                    ],
                },
                options: {
                    expected_revision: staged.root.source.revision,
                    operation_id: `processing:archive:${f.job.id}`,
                    payload_fingerprint: await fingerprintJson([asset]),
                    recorded_at: at,
                },
            },
            store,
        );
        if (!archived.locator) throw new Error('Actual archive append has no locator');
        staged = { root: archived.root, locator: archived.locator };
        const selected = await loadIndexedProcessingToolResultSelectedContext(store, staged.root, staged.locator);
        const resolution = await resolveIndexedProcessingTextInput(selected, f.job, at);
        staged = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
            phase: 'resolve',
            value: resolution,
        });
        const attempt = {
            job_id: f.job.id,
            resolved_input_fingerprint: await fingerprintJson(resolution),
            attempt_token: 'attempt:indexed',
            started_at: at,
        };
        staged = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
            phase: 'attempt',
            value: attempt,
        });
        const workspace = IndexedProcessingClaimWorkspaceSchema.parse({
            version: 1,
            selected: await loadIndexedProcessingToolResultSelectedContext(store, staged.root, staged.locator),
            job: f.job,
            resolution,
            attempt,
            configuration: {
                id: f.job.processor_id,
                version: f.job.processor_version,
                scope: f.job.scope,
                config: f.job.configuration,
                required: f.job.required,
                failure_behavior: f.job.failure_behavior,
            },
            snapshot_at: at,
            archives: {
                assets: [asset],
                acceptance: archived.receipt,
                retrievals: [
                    {
                        capability: 'read_artifact',
                        version: 1,
                        tool_definition_id: 'definition:read',
                        arguments: { asset_id: asset.id },
                    },
                ],
            },
        });
        const output = await buildIndexedTextExternalizationOutput(workspace);
        if (output.kind !== 'proposal' || output.proposal.kind !== 'replace_with_compaction')
            throw new Error('Actual partial processor did not create a compaction');
        const corrupt = structuredClone(output);
        if (corrupt.kind !== 'proposal' || corrupt.proposal.kind !== 'replace_with_compaction')
            throw new Error('Copied genuine compaction has an unexpected kind');
        const corruptResult = corrupt.proposal.replacement_turns[0].blocks[0];
        if (corruptResult.type !== 'tool_result') throw new Error('Actual proposal lost its result');
        const unselected = corruptResult.content[1];
        if (unselected.type !== 'text') throw new Error('Actual proposal lost unselected text');
        unselected.text = 'foreign replacement bytes';
        const { output_fingerprint: _oldFingerprint, ...corruptPayload } = corrupt;
        corrupt.output_fingerprint = await fingerprintJson(corruptPayload);
        const originalRecords = new Map(bytes);
        await expect(applyIndexedTextExternalizationOutput(workspace, corrupt)).rejects.toThrow(
            'differs from its exact deterministic archived output',
        );
        expect(bytes).toEqual(originalRecords);
        staged = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
            phase: 'output',
            value: output,
        });
        const completed = await stageIndexedTextProcessingCompletion(store, staged.root, staged.locator, workspace);
        expect(completed.completion.status).toBe('applied');
        const completionEvidence = {
            job: f.job,
            resolution,
            attempt,
            output,
            completion: completed.completion,
            receipt: completed.receipt,
            resolution_receipt: await loadRecord(
                store,
                await getPagedRecord(
                    store,
                    completed.root.directories.operation_receipts,
                    `processing:resolve:${f.job.id}`,
                ),
                OperationReceiptSchema,
            ),
        };
        expect(await indexedCompletedJobEntrySelection(completionEvidence)).toEqual({
            entry_ids: completed.completion.inserted_entry_ids,
        });
        const ordinaryApplyHash = await fingerprintJson({
            operation_id: completed.receipt.id,
            recorded_at: completed.receipt.recorded_at,
            expected_revision: completed.receipt.base_revision,
            expected_context_revision: resolution.context_revision,
            expected_source_fingerprint: resolution.source_fingerprint,
            entry_ids: resolution.entry_ids,
            proposal: output.proposal,
        });
        await expect(
            indexedCompletedJobEntrySelection({
                ...completionEvidence,
                receipt: { ...completed.receipt, payload_fingerprint: ordinaryApplyHash },
            }),
        ).rejects.toThrow('tool-result predecessor apply receipt');
        if (!completed.receipt.context_change) throw new Error('Actual compaction lost its context change receipt');
        await expect(
            indexedCompletedJobEntrySelection({
                ...completionEvidence,
                receipt: {
                    ...completed.receipt,
                    context_change: {
                        ...completed.receipt.context_change,
                        source_fingerprint: await fingerprintJson({ foreign: 'source selection' }),
                    },
                },
            }),
        ).rejects.toThrow('tool-result predecessor apply receipt');
        await expect(
            indexedCompletedJobEntrySelection({
                ...completionEvidence,
                job: {
                    ...f.job,
                    processor_id: 'externalize-text',
                    processor_version: '1',
                    configuration: {},
                    configuration_fingerprint: await fingerprintJson({}),
                },
            }),
        ).rejects.toThrow('predecessor apply receipt differs');
        const coldOutputDescriptor = await getPagedRecord(
            store,
            completed.root.directories.processing_records,
            JSON.stringify(['outputs', f.job.id]),
        );
        if (coldOutputDescriptor?.storage !== 'record') throw new Error('Actual commit lost its output descriptor');
        coldOutputHash = coldOutputDescriptor.content_hash;
        forbidColdResults = true;
        preparationReads.length = 0;
        // This is a strict settled archive readback, not a provider-ready count/coverage token.
        const loaded = await loadIndexedSettledProcessingSelectedContext(store, completed.root, completed.locator);
        const projection = loaded.replacement_turns?.[0].projection;
        const result = projection?.selected_blocks[0];
        if (result?.type !== 'tool_result') throw new Error('Expected accepted derived result');
        expect(result.content[0]).toMatchObject({ type: 'external_reference', asset_id: asset.id });
        expect(result.content[1]).toMatchObject({ type: 'text', text: `${originalText} preserved` });
        expect(result.content[2]).toMatchObject({ type: 'json', value: { preserved: true } });
        expect(result.content[3]).toMatchObject({ type: 'image', asset_id: 'asset:indexed-image' });
        expect(loaded.assets['asset:indexed-image']).toEqual(f.document.assets['asset:indexed-image']);
        expect(result.content[4]).toMatchObject({
            type: 'external_reference',
            asset_id: priorAsset.id,
            original_type: 'json',
            retrieval: priorRetrieval,
        });
        expect(loaded.assets[priorAsset.id]).toEqual(priorAsset);
        expect(loaded.execution_witnesses?.['execution:one']).toEqual(f.document.execution_receipts['execution:one']);
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);
        expect(preparationReads.some((read) => read.kind === 'context_entries' && read.id === f.entryId)).toBe(false);
        const terminalSource = f.document.execution_receipts['execution:one'].call_source;
        if (!terminalSource) throw new Error('Genuine completed result lacks its original accepted call source');
        await expect(loadIndexedToolCallTerminalResult(store, completed.root, terminalSource)).resolves.toBe(true);
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);
        preparationReads.length = 0;
        await loadIndexedSettledProcessingSelectedContext(store, completed.root, completed.locator);
        const firstPreparationReads = [...preparationReads];
        preparationReads.length = 0;
        await loadIndexedSettledProcessingSelectedContext(store, completed.root, completed.locator);
        expect(preparationReads).toEqual(firstPreparationReads);

        const changedRoot = async (
            family: keyof typeof completed.root.directories,
            key: string,
            descriptor: NonNullable<Awaited<ReturnType<typeof getPagedRecord>>>,
        ) => {
            const root = structuredClone(completed.root);
            root.directories[family] = await putPagedRecord(
                store,
                root.directories[family],
                key,
                descriptor,
                'replace',
            );
            const record = await stageRecord(store, 'root', root.source.conversation_id, root);
            return { root, locator: { content_hash: record.content_hash, size_bytes: record.size_bytes } };
        };
        const changedOutput = await changedRoot('processing_records', JSON.stringify(['outputs', f.job.id]), {
            ...coldOutputDescriptor,
            content_hash: await fingerprintJson({ foreign: 'processing output' }),
        });
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedOutput.root, changedOutput.locator),
        ).rejects.toThrow('immutable output/completion/original terminal evidence');
        await expect(loadIndexedToolCallTerminalResult(store, changedOutput.root, terminalSource)).rejects.toThrow(
            'committed processing witness',
        );
        const changedProjection = await changedRoot(
            'blocks',
            result.id,
            await stageRecord(store, 'blocks', result.id, {
                ...result,
                content: result.content.map((content) =>
                    content.type === 'text' ? { ...content, text: `${content.text} foreign projection` } : content,
                ),
            }),
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedProjection.root, changedProjection.locator),
        ).rejects.toThrow('exact accepted context projection');

        const validationKey = JSON.stringify(['tool_result_validations', f.job.id]);
        const validationDescriptor = await getPagedRecord(
            store,
            completed.root.directories.processing_records,
            validationKey,
        );
        if (validationDescriptor?.storage !== 'record') throw new Error('Genuine commit lacks its private validation');
        const absentRoot = structuredClone(completed.root);
        const removedValidation = await removePagedRecord(
            store,
            absentRoot.directories.processing_records,
            validationKey,
        );
        if (!removedValidation.root) throw new Error('Removing validation unexpectedly erased processing history');
        absentRoot.directories.processing_records = removedValidation.root;
        const absentRecord = await stageRecord(store, 'root', absentRoot.source.conversation_id, absentRoot);
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, absentRoot, {
                content_hash: absentRecord.content_hash,
                size_bytes: absentRecord.size_bytes,
            }),
        ).rejects.toThrow('record is unavailable');

        const completionKey = JSON.stringify(['completions', f.job.id]);
        const completionDescriptor = await getPagedRecord(
            store,
            completed.root.directories.processing_records,
            completionKey,
        );
        const completionRecord = await loadRecord(store, completionDescriptor, ProcessingCompletionReceiptSchema);
        const changedCompletion = await changedRoot(
            'processing_records',
            completionKey,
            await stageRecord(store, 'processing_records', f.job.id, {
                ...completionRecord,
                recorded_at: '2026-10-04T00:00:01.000Z',
            }),
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedCompletion.root, changedCompletion.locator),
        ).rejects.toThrow('committed processing lineage');

        const jobKey = JSON.stringify(['jobs', f.job.id]);
        const jobRecord = await loadRecord(
            store,
            await getPagedRecord(store, completed.root.directories.processing_records, jobKey),
            ProcessingJobSchema,
        );
        const changedJob = await changedRoot(
            'processing_records',
            jobKey,
            await stageRecord(store, 'processing_records', f.job.id, { ...jobRecord, required: !jobRecord.required }),
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedJob.root, changedJob.locator),
        ).rejects.toThrow('committed processing lineage');
        const attemptKey = JSON.stringify(['attempts', f.job.id]);
        const attemptRecord = await loadRecord(
            store,
            await getPagedRecord(store, completed.root.directories.processing_records, attemptKey),
            ProcessingAttemptReceiptSchema,
        );
        const changedAttempt = await changedRoot(
            'processing_records',
            attemptKey,
            await stageRecord(store, 'processing_records', f.job.id, {
                ...attemptRecord,
                started_at: '2026-10-04T00:00:01.000Z',
            }),
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedAttempt.root, changedAttempt.locator),
        ).rejects.toThrow('committed processing lineage');

        const originalBlock = await getPagedRecord(store, completed.root.directories.blocks, 'block:result');
        if (originalBlock?.storage !== 'record') throw new Error('Genuine commit lost its cold original descriptor');
        const changedOriginal = await changedRoot('blocks', 'block:result', {
            ...originalBlock,
            content_hash: await fingerprintJson({ foreign: 'original result bytes' }),
        });
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedOriginal.root, changedOriginal.locator),
        ).rejects.toThrow('original record descriptor');

        await expect(loadIndexedToolCallTerminalResult(store, changedOriginal.root, terminalSource)).rejects.toThrow(
            'immutable original record descriptor',
        );

        const headerDescriptor = await getPagedRecord(store, completed.root.directories.turns, 'turn:result');
        const originalHeader = await loadRecord(store, headerDescriptor, IndexedConversationTurnHeaderSchema);
        const changedProvenance = await changedRoot(
            'turns',
            'turn:result',
            await stageRecord(store, 'turns', 'turn:result', {
                ...originalHeader,
                turn: {
                    ...originalHeader.turn,
                    provenance: {
                        type: 'derived',
                        derivation_id: 'foreign:compaction',
                        source_turn_ids: ['foreign:turn'],
                        source_hash: await fingerprintJson({ foreign: 'provenance' }),
                    },
                },
            }),
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedProvenance.root, changedProvenance.locator),
        ).rejects.toThrow('original record descriptor');

        const terminalDescriptor = await getPagedRecord(
            store,
            completed.root.directories.execution_receipts,
            'execution:one',
        );
        const terminal = await loadRecord(store, terminalDescriptor, ExecutionReceiptSchema);
        const changedTerminal = await changedRoot(
            'execution_receipts',
            'execution:one',
            await stageRecord(store, 'execution_receipts', 'execution:one', {
                ...terminal,
                result_fingerprint: await fingerprintJson({ foreign: 'terminal result' }),
            }),
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, changedTerminal.root, changedTerminal.locator),
        ).rejects.toThrow('original record descriptor');

        await expect(loadIndexedToolCallTerminalResult(store, changedTerminal.root, terminalSource)).rejects.toThrow(
            'immutable original record descriptor',
        );

        const validationBytes = bytes.get(validationDescriptor.content_hash);
        if (!validationBytes) throw new Error('Genuine validation bytes are absent');
        bytes.set(
            validationDescriptor.content_hash,
            Uint8Array.from(validationBytes, (byte, i) => (i === 0 ? 32 : byte)),
        );
        await expect(
            loadIndexedSettledProcessingSelectedContext(store, completed.root, completed.locator),
        ).rejects.toThrow('differs from its authenticated index');
        bytes.set(validationDescriptor.content_hash, validationBytes);

        // One-time materialized migration rebuilds from originals and produces the same bounded
        // prepare witness; materialized archive fingerprints retain their own canonical profile.
        forbidColdResults = false;
        const migrated = await stageIndexedConversationSnapshot(materialized.current, undefined, store);
        forbidColdResults = true;
        preparationReads.length = 0;
        const migratedSelection = await loadIndexedSettledProcessingSelectedContext(
            store,
            migrated.root,
            migrated.locator,
        );
        expect(migratedSelection.replacement_turns?.[0].projection.selected_blocks).toEqual(
            loaded.replacement_turns?.[0].projection.selected_blocks,
        );
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);

        const materializedCompaction = Object.values(materialized.current.compactions)[0];
        expect(materializedCompaction.original_context).toEqual(materializedArchive.document.context);
        expect(conversationDocumentFromJson(conversationDocumentToJson(materialized.current))).toEqual(
            materialized.current,
        );
        expect(Object.values(migratedSelection.compaction_witnesses ?? {})[0].compaction).not.toHaveProperty(
            'original_context',
        );

        const legacyImmediate = structuredClone(materialized.current);
        delete legacyImmediate.compactions[materializedCompaction.id].original_context;
        forbidColdResults = false;
        const immediatelyMigrated = await stageIndexedConversationSnapshot(legacyImmediate, undefined, store);
        forbidColdResults = true;
        preparationReads.length = 0;
        const immediateSelection = await loadIndexedSettledProcessingSelectedContext(
            store,
            immediatelyMigrated.root,
            immediatelyMigrated.locator,
        );
        expect(immediateSelection.replacement_turns?.[0].projection.selected_blocks).toEqual(
            loaded.replacement_turns?.[0].projection.selected_blocks,
        );
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);

        const followupBatch = {
            turns: [
                {
                    id: 'turn:after-program',
                    kind: 'program',
                    authority: 'system',
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    provenance: { type: 'inserted', operation_id: 'append:after-compaction' },
                    model_visibility: 'include',
                    blocks: [
                        { id: 'block:after-program', type: 'text', format: 'plain', text: 'Later program context' },
                    ],
                },
                {
                    id: 'turn:after-user',
                    kind: 'user',
                    authority: 'ordinary',
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    provenance: { type: 'received' },
                    model_visibility: 'include',
                    blocks: [{ id: 'block:after-user', type: 'text', format: 'plain', text: 'Later user context' }],
                },
            ],
            context_entries: [
                { id: 'entry:after-program', type: 'source_turn', turn_id: 'turn:after-program' },
                { id: 'entry:after-user', type: 'source_turn', turn_id: 'turn:after-user' },
            ],
            active_tool_definition_ids: materialized.current.context.active_tool_definition_ids.filter(
                (id) => id !== 'definition:writer',
            ),
        } satisfies ConversationRecordBatch;
        const laterMaterialized = await appendConversationRecordsWithProcessing(materialized.current, followupBatch, {
            expected_revision: materialized.current.revision,
            operation_id: 'append:after-compaction',
            recorded_at: at,
            payload_fingerprint: await fingerprintJson(followupBatch),
        });
        forbidColdResults = false;
        const laterMigration = await stageIndexedConversationSnapshot(laterMaterialized.document, undefined, store);
        forbidColdResults = true;
        preparationReads.length = 0;
        const laterMigratedSelection = await loadIndexedSettledProcessingSelectedContext(
            store,
            laterMigration.root,
            laterMigration.locator,
        );
        expect(laterMigratedSelection.replacement_turns?.[0].projection.selected_blocks).toEqual(
            loaded.replacement_turns?.[0].projection.selected_blocks,
        );
        expect(laterMigratedSelection.context.entries.slice(-2).map((entry) => entry.id)).toEqual([
            'entry:after-program',
            'entry:after-user',
        ]);
        expect(laterMigratedSelection.context.active_tool_definition_ids).not.toContain('definition:writer');
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);

        const migratedMissingValidation = structuredClone(laterMigration.root);
        const migratedValidationRemoval = await removePagedRecord(
            store,
            migratedMissingValidation.directories.processing_records,
            JSON.stringify(['tool_result_validations', f.job.id]),
        );
        if (!migratedValidationRemoval.root) throw new Error('Materialized migration lost processing history');
        migratedMissingValidation.directories.processing_records = migratedValidationRemoval.root;
        forbidColdResults = false;
        const migratedRecoveredDirectories = await recoverIndexedToolResultValidations(
            store,
            migratedMissingValidation,
        );
        const migratedRecoveredRoot = { ...migratedMissingValidation, directories: migratedRecoveredDirectories };
        const migratedRecoveredRecord = await stageRecord(
            store,
            'root',
            migratedRecoveredRoot.source.conversation_id,
            migratedRecoveredRoot,
        );
        forbidColdResults = true;
        preparationReads.length = 0;
        const migratedRecoveredSelection = await loadIndexedSettledProcessingSelectedContext(
            store,
            migratedRecoveredRoot,
            {
                content_hash: migratedRecoveredRecord.content_hash,
                size_bytes: migratedRecoveredRecord.size_bytes,
            },
        );
        expect(migratedRecoveredSelection.replacement_turns?.[0].projection.selected_blocks).toEqual(
            loaded.replacement_turns?.[0].projection.selected_blocks,
        );
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);

        // Old snapshots lacking an original closure require a real authenticated historical document.
        const legacyLater = structuredClone(laterMaterialized.document);
        delete legacyLater.compactions[materializedCompaction.id].original_context;
        forbidColdResults = false;
        await expect(stageIndexedConversationSnapshot(legacyLater, undefined, store)).rejects.toThrow(
            'requires its authenticated original context',
        );
        await expect(
            stageIndexedConversationSnapshot(legacyLater, undefined, store, async () => laterMaterialized.document),
        ).rejects.toThrow('foreign original context');
        const resolvedMigration = await stageIndexedConversationSnapshot(
            legacyLater,
            undefined,
            store,
            async () => materializedArchive.document,
        );
        forbidColdResults = true;
        preparationReads.length = 0;
        const resolvedSelection = await loadIndexedSettledProcessingSelectedContext(
            store,
            resolvedMigration.root,
            resolvedMigration.locator,
        );
        expect(resolvedSelection.replacement_turns?.[0].projection.selected_blocks).toEqual(
            loaded.replacement_turns?.[0].projection.selected_blocks,
        );
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);

        const afterIndexed = await stageIndexedRecordBatch(
            completed.root,
            {
                conversation_id: completed.root.source.conversation_id,
                batch: followupBatch,
                options: {
                    expected_revision: completed.root.source.revision,
                    operation_id: 'append:after-compaction',
                    recorded_at: at,
                    payload_fingerprint: await fingerprintJson(followupBatch),
                },
            },
            store,
        );
        if (!afterIndexed.locator) throw new Error('Post-compaction append lost its genuine locator');
        const missingLaterValidation = structuredClone(afterIndexed.root);
        const laterValidationRemoval = await removePagedRecord(
            store,
            missingLaterValidation.directories.processing_records,
            JSON.stringify(['tool_result_validations', f.job.id]),
        );
        if (!laterValidationRemoval.root) throw new Error('Post-compaction processing history disappeared');
        missingLaterValidation.directories.processing_records = laterValidationRemoval.root;
        forbidColdResults = false;
        const laterRecoveredDirectories = await recoverIndexedToolResultValidations(store, missingLaterValidation);
        const laterRecoveredRoot = { ...missingLaterValidation, directories: laterRecoveredDirectories };
        const laterRecoveredRecord = await stageRecord(
            store,
            'root',
            laterRecoveredRoot.source.conversation_id,
            laterRecoveredRoot,
        );
        forbidColdResults = true;
        preparationReads.length = 0;
        const laterRecoveredSelection = await loadIndexedSettledProcessingSelectedContext(store, laterRecoveredRoot, {
            content_hash: laterRecoveredRecord.content_hash,
            size_bytes: laterRecoveredRecord.size_bytes,
        });
        expect(laterRecoveredSelection.replacement_turns?.[0].projection.selected_blocks).toEqual(
            loaded.replacement_turns?.[0].projection.selected_blocks,
        );
        expect(laterRecoveredSelection.context.entries.slice(-2).map((entry) => entry.id)).toEqual([
            'entry:after-program',
            'entry:after-user',
        ]);
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);

        forbidColdResults = false;
        const sourceKey = JSON.stringify(['tool_result_sources', f.job.id]);
        const originalSourceDescriptor = await getPagedRecord(
            store,
            missingLaterValidation.directories.processing_records,
            sourceKey,
        );
        if (originalSourceDescriptor?.storage !== 'record') throw new Error('Actual commit lost original source');
        const originalSource = await loadRecord(store, originalSourceDescriptor, IndexedToolResultOriginalSourceSchema);
        const foreignSourceRoot = structuredClone(missingLaterValidation);
        foreignSourceRoot.directories.processing_records = await putPagedRecord(
            store,
            foreignSourceRoot.directories.processing_records,
            sourceKey,
            await stageRecord(store, 'processing_records', f.job.id, {
                ...originalSource,
                source: { kind: 'indexed_root', root: afterIndexed.locator },
            }),
            'replace',
        );
        await expect(recoverIndexedToolResultValidations(store, foreignSourceRoot)).rejects.toThrow(
            'foreign to its accepted job',
        );
        const foreignOriginalClosureRoot = structuredClone(missingLaterValidation);
        foreignOriginalClosureRoot.directories.processing_records = await putPagedRecord(
            store,
            foreignOriginalClosureRoot.directories.processing_records,
            sourceKey,
            await stageRecord(store, 'processing_records', f.job.id, {
                ...originalSource,
                source: {
                    kind: 'materialized_context',
                    context: { ...materializedArchive.document.context, protected_entry_ids: ['entry:call'] },
                },
            }),
            'replace',
        );
        await expect(recoverIndexedToolResultValidations(store, foreignOriginalClosureRoot)).rejects.toThrow(
            'changed its accepted original context closure',
        );
        const unavailableSourceRoot = structuredClone(missingLaterValidation);
        const sourceRemoval = await removePagedRecord(
            store,
            unavailableSourceRoot.directories.processing_records,
            sourceKey,
        );
        if (!sourceRemoval.root) throw new Error('Source removal erased processing history');
        unavailableSourceRoot.directories.processing_records = sourceRemoval.root;
        await expect(recoverIndexedToolResultValidations(store, unavailableSourceRoot)).rejects.toThrow(
            'requires its authenticated original context',
        );
        const exactArchiveLocator = archived.locator;
        // A retained terminal relation still authenticates the original source descriptor,
        // so an equivalent context at another root cannot replace that immutable witness.
        await expect(
            recoverIndexedToolResultValidations(store, unavailableSourceRoot, async () => exactArchiveLocator),
        ).rejects.toThrow('terminal validation relation');
        if (originalSource.source.kind !== 'indexed_root')
            throw new Error('Actual indexed commit lost its original root');
        const exactCommittedLocator = originalSource.source.root;
        await recoverIndexedToolResultValidations(store, unavailableSourceRoot, async () => exactCommittedLocator);
        // An older artifact lacking the entire private attestation family has no such retained
        // binding. Its authenticated historical closure can produce a new verified attestation.
        const legacyWithoutAttestation = structuredClone(unavailableSourceRoot);
        const removedTerminal = await removePagedRecord(
            store,
            legacyWithoutAttestation.directories.processing_records,
            JSON.stringify(['tool_result_validation_by_terminal', 'execution:one']),
        );
        if (!removedTerminal.root) throw new Error('Legacy omission erased processing history');
        legacyWithoutAttestation.directories.processing_records = removedTerminal.root;
        await recoverIndexedToolResultValidations(store, legacyWithoutAttestation, async () => exactArchiveLocator);

        // Older indexed snapshots omitted removed entries and the validation namespace.
        // Recovery may load those originals once; later prepare must retain the cold-body bound.
        forbidColdResults = false;
        const recoveryInput = structuredClone(absentRoot);
        const removedEntry = await removePagedRecord(store, recoveryInput.directories.context_entries, f.entryId);
        if (removedEntry.root) recoveryInput.directories.context_entries = removedEntry.root;
        else delete recoveryInput.directories.context_entries;
        const recoveredDirectories = await recoverIndexedToolResultValidations(store, recoveryInput);
        const recoveredRoot = { ...recoveryInput, directories: recoveredDirectories };
        const recoveredRecord = await stageRecord(store, 'root', recoveredRoot.source.conversation_id, recoveredRoot);
        forbidColdResults = true;
        preparationReads.length = 0;
        const recovered = await loadIndexedSettledProcessingSelectedContext(store, recoveredRoot, {
            content_hash: recoveredRecord.content_hash,
            size_bytes: recoveredRecord.size_bytes,
        });
        expect(recovered.replacement_turns).toEqual(loaded.replacement_turns);
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);

        // The explicit paged upgrade recognizes and audits committed private validations.
        // Separate bounded validation and terminal-relation phases recover historical proof before publication.
        forbidColdResults = false;
        const upgradeCommand = {
            version: 1 as const,
            profile: INDEXED_CONVERSATION_UPGRADE_PROFILE,
            operation_id: 'upgrade:partial-validation',
            source: recoveryInput.source,
            predecessor_root: { content_hash: absentRecord.content_hash, size_bytes: absentRecord.size_bytes },
            recorded_at: at,
        };
        // Pin the exact root supplied to the audit, including its missing original entry.
        const recoveryRecord = await stageRecord(store, 'root', recoveryInput.source.conversation_id, recoveryInput);
        upgradeCommand.predecessor_root = {
            content_hash: recoveryRecord.content_hash,
            size_bytes: recoveryRecord.size_bytes,
        };
        let progress = await beginIndexedConversationUpgrade(store, upgradeCommand);
        let steps = 0;
        let sawDeferredValidation = false;
        while (progress.progress.phase !== 'complete') {
            if (++steps > 1000) throw new Error('Partial-result indexed recovery failed to terminate');
            const previous = progress;
            progress = await advanceIndexedConversationUpgrade(store, upgradeCommand, previous.locator);
            if (previous.progress.phase === 'active_window') {
                expect(progress.progress.phase).toBe('tool_result_validations');
                expect(progress.progress.cursor).toBeUndefined();
                expect(
                    await getPagedRecord(
                        store,
                        progress.progress.directories.processing_records,
                        JSON.stringify(['tool_result_validations', f.job.id]),
                    ),
                ).toBeUndefined();
            }
            if (previous.progress.phase === 'tool_result_validations' && previous.progress.cursor === undefined) {
                expect(progress.progress.cursor?.endsWith(`:${f.job.id}`)).toBe(true);
                expect(
                    await getPagedRecord(
                        store,
                        progress.progress.directories.processing_records,
                        JSON.stringify(['tool_result_validations', f.job.id]),
                    ),
                ).toBeDefined();
            }
            if (progress.progress.scratch.missing_tool_result_validations !== undefined) {
                sawDeferredValidation = true;
                expect(
                    (
                        await readPagedRecordRange(store, progress.progress.scratch.missing_tool_result_validations, {
                            limit: 1,
                        })
                    ).entries[0]?.value,
                ).toEqual({
                    storage: 'marker',
                    kind: 'upgrade_missing_tool_result_validation',
                    id: f.job.id,
                });
            }
        }
        expect(sawDeferredValidation).toBe(true);
        expect(progress.progress.scratch.missing_tool_result_validations).toBeUndefined();
        const upgraded = await finishIndexedConversationUpgrade(
            store,
            recoveryInput,
            upgradeCommand.predecessor_root,
            upgradeCommand,
            progress.locator,
        );
        expect(await getPagedRecord(store, upgraded.root.directories.tool_call_states, 'call:one')).toEqual(
            await getPagedRecord(store, recoveryInput.directories.tool_call_states, 'call:one'),
        );
        expect(await getPagedRecord(store, upgraded.root.directories.execution_receipts, 'execution:one')).toEqual(
            await getPagedRecord(store, recoveryInput.directories.execution_receipts, 'execution:one'),
        );
        forbidColdResults = true;
        preparationReads.length = 0;
        const upgradedSelection = await loadIndexedSettledProcessingSelectedContext(
            store,
            upgraded.root,
            upgraded.locator,
        );
        expect(upgradedSelection.replacement_turns).toEqual(loaded.replacement_turns);
        expect(preparationReads.some((read) => read.kind === 'blocks' && read.id === 'block:result')).toBe(false);
        // A genuine later context exclusion retires the entire completed call/result exchange.
        // Historical missing-witness obligations remain mandatory and recover one per retained step.
        forbidColdResults = false;
        const inactiveSelection = {
            expected_revision: laterMaterialized.document.revision,
            expected_context_revision: laterMaterialized.document.context.revision,
            entry_ids: laterMaterialized.document.context.entries
                .filter(
                    (entry) =>
                        entry.turn_id === 'turn:call' ||
                        (entry.type === 'replacement_turn' && entry.compaction_id === materializedCompaction.id),
                )
                .map((entry) => entry.id),
        };
        const inactivePlan = await planContextChange(laterMaterialized.document, inactiveSelection);
        const inactiveDocument = await applyContextChange(laterMaterialized.document, {
            ...inactiveSelection,
            operation_id: 'exclude:completed-exchange',
            expected_source_fingerprint: inactivePlan.source_fingerprint,
            recorded_at: at,
            proposal: { kind: 'exclude' },
        });
        const inactiveSnapshot = await stageIndexedConversationSnapshot(inactiveDocument.document, undefined, store);
        const inactiveSourceDescriptor = await getPagedRecord(
            store,
            inactiveSnapshot.root.directories.processing_records,
            JSON.stringify(['tool_result_sources', f.job.id]),
        );
        if (inactiveSourceDescriptor?.storage !== 'record')
            throw new Error('Inactive migration lost its genuine original source');
        const inactiveSource = await loadRecord(store, inactiveSourceDescriptor, IndexedToolResultOriginalSourceSchema);
        expect(inactiveSource.source).toEqual({
            kind: 'materialized_context',
            context: materializedCompaction.original_context,
        });
        expect(
            await getPagedRecord(
                store,
                inactiveSnapshot.root.directories.processing_records,
                JSON.stringify(['tool_result_validations', f.job.id]),
            ),
        ).toBeDefined();
        const inactiveRoot = structuredClone(inactiveSnapshot.root);
        const inactiveValidationRemoval = await removePagedRecord(
            store,
            inactiveRoot.directories.processing_records,
            JSON.stringify(['tool_result_validations', f.job.id]),
        );
        if (!inactiveValidationRemoval.root) throw new Error('Inactive fixture lost its processing history');
        inactiveRoot.directories.processing_records = inactiveValidationRemoval.root;
        const inactiveDescriptor = await stageRecord(store, 'root', inactiveRoot.source.conversation_id, inactiveRoot);
        const inactiveCommand = {
            ...upgradeCommand,
            operation_id: 'upgrade:inactive-partial-validation',
            source: inactiveRoot.source,
            predecessor_root: {
                content_hash: inactiveDescriptor.content_hash,
                size_bytes: inactiveDescriptor.size_bytes,
            },
        };
        let inactiveProgress = await beginIndexedConversationUpgrade(store, inactiveCommand);
        let inactiveSteps = 0;
        let sawIncrementalRecovery = false;
        while (inactiveProgress.progress.phase !== 'complete') {
            if (++inactiveSteps > 1000) throw new Error('Inactive-result indexed recovery failed to terminate');
            const previous = inactiveProgress;
            inactiveProgress = await advanceIndexedConversationUpgrade(store, inactiveCommand, previous.locator);
            if (previous.progress.phase === 'active_window') {
                expect(inactiveProgress.progress.phase).toBe('tool_result_validations');
                expect(inactiveProgress.progress.cursor).toBeUndefined();
                expect(
                    await getPagedRecord(
                        store,
                        inactiveProgress.progress.directories.processing_records,
                        JSON.stringify(['tool_result_validations', f.job.id]),
                    ),
                ).toBeUndefined();
            }
            if (previous.progress.phase === 'tool_result_validations' && previous.progress.cursor === undefined) {
                sawIncrementalRecovery = true;
                expect(inactiveProgress.progress.phase).toBe('tool_result_validations');
                expect(inactiveProgress.progress.cursor?.endsWith(`:${f.job.id}`)).toBe(true);
                expect(
                    await getPagedRecord(
                        store,
                        inactiveProgress.progress.directories.processing_records,
                        JSON.stringify(['tool_result_validations', f.job.id]),
                    ),
                ).toBeDefined();
                const repeat = await advanceIndexedConversationUpgrade(store, inactiveCommand, previous.locator);
                expect(repeat.locator).toEqual(inactiveProgress.locator);
            }
        }
        expect(sawIncrementalRecovery).toBe(true);
        expect(inactiveProgress.progress.scratch.missing_tool_result_validations).toBeUndefined();
        const inactiveUpgrade = await finishIndexedConversationUpgrade(
            store,
            inactiveRoot,
            inactiveCommand.predecessor_root,
            inactiveCommand,
            inactiveProgress.locator,
        );
        forbidColdResults = true;
        const inactivePrepared = await loadIndexedSettledProcessingSelectedContext(
            store,
            inactiveUpgrade.root,
            inactiveUpgrade.locator,
        );
        expect(inactivePrepared.context.entries.map((entry) => entry.id)).toEqual([
            'entry:after-program',
            'entry:after-user',
        ]);
        expect(inactivePrepared.replacement_turns ?? []).toHaveLength(0);
        forbidColdResults = false;
        const retry = await stageIndexedTextProcessingCompletion(store, completed.root, completed.locator, workspace);
        expect(retry.locator).toEqual(completed.locator);
        expect(retry.completion).toEqual(completed.completion);
    });

    it('externalizes only the explicitly selected text sibling and preserves other text and JSON after reload', async () => {
        const content: ToolResultBlock['content'] = [
            { id: 'text:chosen', type: 'text', format: 'plain', text: originalText },
            { id: 'text:unchosen', type: 'text', format: 'plain', text: `${originalText} untouched` },
            { id: 'json:unchanged', type: 'json', value: { answer: 42, nested: ['retained'] } },
            { id: 'image:unchanged', type: 'image', asset_id: 'asset:original-image' },
        ];
        const f = await completedResult(false, false, {
            strategy: {
                processor_id: 'externalize-tool-result-text',
                processor_version: '2',
                configuration: {
                    selector: {
                        kind: 'exact_blocks',
                        blocks: [{ turn_id: 'turn:result', result_block_id: 'block:result', block_id: 'text:chosen' }],
                    },
                },
            },
            content,
            assets: [await ownedImage('asset:original-image')],
        });
        const archive = await toolResultExternalizationArchiveInputs(f.document, f.job);
        expect(archive.texts.map((item) => item.block_id)).toEqual(['text:chosen']);
        const asset = {
            id: 'asset:chosen',
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: { type: 'external' as const, resolver: 'test.blob', locator: { key: 'chosen' } },
            provenance: { type: 'received' as const },
            created_at: at,
            ...archive.integrities[0],
        };
        const archived = await appendConversationRecordsWithProcessing(
            f.document,
            {
                assets: [asset],
                tool_definitions: [{ id: 'definition:read', name: 'read_artifact', version: '1', input_schema: true }],
                active_tool_definition_ids: [
                    'definition:read',
                    ...(f.job.processor_version === '2' ? f.document.context.active_tool_definition_ids : []),
                ],
            },
            {
                expected_revision: f.document.revision,
                operation_id: `processing:archive:${f.job.id}`,
                payload_fingerprint: archive.payload_fingerprint,
                recorded_at: at,
            },
        );
        const store = new MemoryStore(parseConversationDocument(JSON.parse(JSON.stringify(archived.document))));
        const processor = createToolResultTextExternalizationProcessor(({ asset }) => ({
            capability: 'read_artifact',
            version: 1,
            tool_definition_id: 'definition:read',
            arguments: { asset_id: asset.id },
        }));
        await runProcessingJob(store, { resolve: () => processor }, f.job.id, 'attempt:partial', () => at);
        expect(store.current.processing.completions?.[f.job.id].status).toBe('applied');
        const projected = Object.values(store.current.compactions)[0].replacement_turns[0].blocks[0];
        if (projected.type !== 'tool_result') throw new Error('Expected the genuine derived tool-result projection');
        expect(projected.content[0]).toMatchObject({ type: 'external_reference', asset_id: asset.id });
        expect(projected.content[1]).toMatchObject({ type: 'text', text: `${originalText} untouched` });
        expect(projected.content[2]).toMatchObject({ type: 'json', value: { answer: 42, nested: ['retained'] } });
        expect(projected.content[3]).toMatchObject({ type: 'image', asset_id: 'asset:original-image' });
        expect(store.current.assets['asset:original-image']).toEqual(f.document.assets['asset:original-image']);
        expect(store.current.turns).toEqual(f.document.turns);
        expect(store.current.execution_receipts).toEqual(f.document.execution_receipts);
        const completed = JSON.stringify(store.current);
        await runProcessingJob(store, { resolve: () => processor }, f.job.id, 'attempt:retry', () => at);
        expect(JSON.stringify(store.current)).toBe(completed);
    });

    it('publishes exact durable assets before replacing only result text and preserves executed bytes on retry', async () => {
        const f = await completedResult(false, true);
        const originalTurns = structuredClone(f.document.turns);
        const originalReceipts = structuredClone(f.document.execution_receipts);
        const archive = await toolResultExternalizationArchiveInputs(f.document, f.job);
        expect(archive.texts.map((item) => item.text)).toEqual([originalText]);
        expect(archive.texts.some((item) => item.text.includes('execution bytes'))).toBe(false);
        const asset = {
            id: 'asset:result',
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: { type: 'external' as const, resolver: 'test.blob', locator: { key: 'result' } },
            provenance: { type: 'received' as const },
            created_at: at,
            ...archive.integrities[0],
        };
        const archived = await appendConversationRecordsWithProcessing(
            f.document,
            {
                assets: [asset],
                tool_definitions: [{ id: 'definition:read', name: 'read_artifact', version: '1', input_schema: true }],
                active_tool_definition_ids: [
                    'definition:read',
                    ...(f.job.processor_version === '2' ? f.document.context.active_tool_definition_ids : []),
                ],
            },
            {
                expected_revision: f.document.revision,
                operation_id: `processing:archive:${f.job.id}`,
                payload_fingerprint: archive.payload_fingerprint,
                recorded_at: at,
            },
        );
        const store = new MemoryStore(archived.document);
        const processor = createToolResultTextExternalizationProcessor(({ asset }) => ({
            capability: 'read_artifact',
            version: 1,
            tool_definition_id: 'definition:read',
            arguments: { asset_id: asset.id },
        }));
        await runProcessingJob(store, { resolve: () => processor }, f.job.id, 'attempt:one', () => at);
        expect(store.current.processing.completions?.[f.job.id].status).toBe('applied');
        expect(store.current.turns).toEqual(originalTurns);
        expect(store.current.execution_receipts).toEqual(originalReceipts);
        expect(store.current.context.entries.find((entry) => entry.id === 'entry:call')).toEqual(
            f.document.context.entries[0],
        );
        expect(store.current.context.entries.some((entry) => entry.id === f.entryId)).toBe(false);
        const retained = resolveActiveTextExternalReference(store.current, asset.id);
        expect(retained.accepted_asset_operation_id).toBe(`processing:archive:${f.job.id}`);
        expect(retained.asset.content_hash).toBe(archive.integrities[0].content_hash);
        expect(retained.block.preview).toBe(originalText.slice(0, 512));
        expect(retained.block.preview?.length).toBeLessThanOrEqual(512);
        const projection = Object.values(store.current.compactions)[0].replacement_turns[0];
        expect(projection.execution_id).toBe('execution:one');
        expect(projection.provenance).toMatchObject({
            type: 'derived',
            source_turn_ids: ['turn:result'],
            source_block_ids: ['block:result'],
        });
        expect(store.current.execution_receipts['execution:one'].result_turn_id).toBe('turn:result');
        expect(await fingerprintJson(projection.blocks[0])).not.toBe(
            originalReceipts['execution:one'].result_fingerprint,
        );
        expect(Object.values(store.current.compactions)[0].source.block_ids).toEqual(['block:result']);
        const compaction = Object.values(store.current.compactions)[0];
        const applied = store.current.operation_receipts[compaction.operation_id];
        expect(compaction.metadata).toMatchObject({
            applied_revision: applied.result_revision,
            payload_fingerprint: applied.payload_fingerprint,
        });
        expect(store.current.turns[0].blocks.at(-1)).toEqual(callReplay());
        const resolution = store.current.processing.resolved_inputs?.[f.job.id];
        const output = store.current.processing.outputs?.[f.job.id];
        if (!resolution || output?.kind !== 'proposal' || output.proposal.kind !== 'replace_with_compaction')
            throw new Error('Fixture requires an exact retained deterministic projection');
        for (const field of ['execution', 'call', 'source'] as const) {
            const tampered = structuredClone(output);
            if (tampered.proposal.kind !== 'replace_with_compaction') throw new Error('Fixture requires compaction');
            const turn = tampered.proposal.replacement_turns[0];
            if (field === 'execution') turn.execution_id = 'execution:other';
            else if (field === 'call') {
                const result = turn.blocks[0];
                if (result.type !== 'tool_result') throw new Error('Fixture requires a result projection');
                result.call_id = 'call:other';
            } else {
                if (turn.provenance.type !== 'derived') throw new Error('Fixture requires derived provenance');
                turn.provenance.source_turn_ids = ['turn:call'];
            }
            await expect(
                applyToolResultTextExternalizationOutput(archived.document, f.job, resolution, tampered, at),
            ).rejects.toThrow(/exact original\/dependency/);
        }
        const beforeRetry = structuredClone(store.current);
        await runProcessingJob(store, { resolve: () => processor }, f.job.id, 'attempt:retry', () => at);
        expect(store.current).toEqual(beforeRetry);
    });

    it('waits for the real preceding stage before publishing a final tool-result archive', async () => {
        const f = await completedResult(true);
        await expect(toolResultExternalizationArchiveInputs(f.document, f.job)).rejects.toThrow(
            /preceding processing stage/,
        );
        const previous = Object.values(f.document.processing.jobs ?? {}).find((job) => job.stage_index === 0);
        if (!previous) throw new Error('Fixture must enqueue its ordinary text predecessor');
        const store = new MemoryStore(f.document);
        const processor = createTextExternalizationProcessor(() => {
            throw new Error('Empty ordinary text must not need retrieval');
        });
        await runProcessingJob(store, { resolve: () => processor }, previous.id, 'attempt:previous', () => at);
        expect(store.current.processing.completions?.[previous.id].status).toBe('no_op');
        await expect(toolResultExternalizationArchiveInputs(store.current, f.job)).resolves.toMatchObject({
            texts: [{ text: originalText }],
        });
    });

    it('does not queue tool-result processing for an ordinary user append', async () => {
        const f = await completedResult();
        const before = Object.keys(f.document.processing.jobs ?? {});
        const batch: Parameters<typeof appendConversationRecordsWithProcessing>[1] = {
            turns: [
                {
                    id: 'user:next',
                    kind: 'user',
                    authority: 'ordinary',
                    status: 'completed',
                    model_visibility: 'include',
                    provenance: { type: 'received' },
                    timestamps: { recorded_at: at },
                    blocks: [{ id: 'user:text', type: 'text', format: 'plain', text: 'user input stays inline' }],
                },
            ],
            context_entries: [{ id: 'user:entry', type: 'source_turn', turn_id: 'user:next' }],
        };
        const appended = await appendConversationRecordsWithProcessing(f.document, batch, {
            operation_id: 'append:user',
            expected_revision: f.document.revision,
            recorded_at: at,
            payload_fingerprint: await fingerprintJson(batch),
        });
        expect(Object.keys(appended.document.processing.jobs ?? {})).toEqual(before);
    });

    it.each(['call', 'result'] as const)('protects both sides of a protected %s dependency', async (side) => {
        const f = await completedResult();
        f.document.context.protected_entry_ids = [side === 'call' ? 'entry:call' : f.entryId];
        await expect(toolResultTextSelection(f.document, [f.entryId])).rejects.toThrow(/protected/);
    });

    it('rejects protected call replay without changing its original call or result', async () => {
        const f = await completedResult(false, true);
        const replay = f.document.turns[0].blocks.find((block) => block.type === 'native_replay');
        if (replay?.type !== 'native_replay') throw new Error('Fixture needs ordinary call replay');
        delete replay.dependency_policy;
        await expect(toolResultTextSelection(f.document, [f.entryId])).rejects.toThrow(/protected/);
    });

    it.each(['turn', 'result', 'text'] as const)(
        'rejects unchanged active replay whose %s dependency would lose original result bytes',
        async (dependency) => {
            const f = await completedResult(false, true);
            const replay = callReplay();
            replay.id = 'replay:dependent';
            replay.dependencies = {
                turn_ids: dependency === 'turn' ? ['turn:result'] : [],
                block_ids: dependency === 'result' ? ['block:result'] : dependency === 'text' ? ['block:text'] : [],
                call_ids: [],
                request_ids: [],
            };
            f.document.turns.push({
                id: 'turn:dependent',
                kind: 'agent',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: at },
                provenance: { type: 'imported', source: 'test' },
                model_visibility: 'include',
                blocks: [replay],
            });
            f.document.context.entries.push({ id: 'entry:dependent', type: 'source_turn', turn_id: 'turn:dependent' });
            await expect(toolResultTextSelection(f.document, [f.entryId])).rejects.toThrow(/native replay dependency/);
        },
    );

    it('cannot externalize an unresolved result or changed executed argument/result receipt', async () => {
        const f = await completedResult();
        const changedCall = structuredClone(f.document);
        const call = changedCall.turns[0].blocks[0];
        if (
            call.type !== 'tool_call' ||
            call.arguments.type !== 'json' ||
            call.arguments.value === null ||
            typeof call.arguments.value !== 'object' ||
            Array.isArray(call.arguments.value)
        )
            throw new Error('Fixture needs an exact object call');
        call.arguments.value.content = 'changed executable bytes';
        await expect(toolResultTextSelection(changedCall, [f.entryId])).rejects.toThrow(/executed call/);
        const changedReceipt = structuredClone(f.document);
        changedReceipt.execution_receipts['execution:one'].result_fingerprint = `sha256:${'0'.repeat(64)}`;
        await expect(toolResultTextSelection(changedReceipt, [f.entryId])).rejects.toThrow(/fingerprint/);
        const unresolved = structuredClone(f.document);
        const result = unresolved.turns.at(-1)?.blocks[0];
        if (result?.type !== 'tool_result') throw new Error('Fixture needs an exact result');
        result.status = 'unknown';
        await expect(toolResultTextSelection(unresolved, [f.entryId])).rejects.toThrow(/terminal/);
    });
});
