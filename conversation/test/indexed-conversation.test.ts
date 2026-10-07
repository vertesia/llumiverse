import { describe, expect, it } from 'vitest';
import { createProgramToolCall, createProgramTurn, createToolTurn, createUserTurn } from '../src/builders.js';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { applyContextChange, planContextChange } from '../src/context-change.js';
import { resolveIndexedTextExternalReference } from '../src/external-reference-retrieval.js';
import { fingerprintJson } from '../src/identity.js';
import {
    assertIndexedCurrentPolicy,
    assertIndexedFreshReceivedTextInput,
    assertIndexedFreshTextInput,
    IndexedAcceptedOutputHistoryUpgradeRequired,
    type IndexedConversationRecordStore,
    IndexedPresentationCapacityError,
    IndexedPresentationNominationConflict,
    type IndexedRecordBatchCommand,
    IndexedRestartSourceUnavailable,
    loadIndexedAcceptedOutputHistoryPage,
    loadIndexedAcceptedOutputPresentation,
    loadIndexedActiveContext,
    loadIndexedActiveToolDefinitions,
    loadIndexedPendingProcessingJobs,
    loadIndexedProcessingJobState,
    loadIndexedProcessingSelectedContext,
    loadIndexedProgramToolCallSelection,
    loadIndexedProjectedTurn,
    loadIndexedReadySelectedContext,
    loadIndexedRestartEvidence,
    loadIndexedRetainedAcceptedOutputPresentation,
    loadIndexedSelectedDependencyContext,
    loadIndexedSelectedMediaCompactionContext,
    loadIndexedSelectedTextContext,
    loadIndexedSettledProcessingSelectedContext,
    loadIndexedSettledRetrievalSelectedContext,
    loadIndexedTerminalProgramPresentation,
    loadIndexedToolCallSelection,
    loadIndexedToolCallTerminalResult,
    renderIndexedConversationRecentMessages,
    renderIndexedConversationSearchText,
    renderIndexedConversationTopicText,
    stageIndexedConversationDelete,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingPhase,
    stageIndexedProgramAppend,
    stageIndexedProgramToolCall,
    stageIndexedRecordBatch,
    stageIndexedTextProcessingCompletion,
    supportsIndexedInheritedProcessingPolicy,
} from '../src/indexed-conversation.js';
import { buildIndexedExchangeOutput } from '../src/indexed-exchange-processing.js';
import { resolveIndexedProcessingTextInput } from '../src/indexed-processing-working-set.js';
import { preflightJsonInput } from '../src/json-preflight.js';
import { getPagedRecord, putPagedRecord } from '../src/paged-record-index.js';
import {
    assertProcessingReady,
    ProcessingKnownFailure,
    runProcessingJob,
    setProcessingPolicy,
} from '../src/processing.js';
import { renderConversationText } from '../src/rendering.js';
import { appendConversationRecords, appendConversationRecordsWithProcessing } from '../src/runtime.js';
import { AgentTurnSchema, ProgramTurnSchema, ToolTurnSchema } from '../src/schemas/content.js';
import { GenerationSchema } from '../src/schemas/execution.js';
import { IndexedConversationProcessingHeaderSchema } from '../src/schemas/indexed-head.js';
import { IndexedRecordBatchCommandSchema } from '../src/schemas/ingestion.js';
import { validateToolExecutionResult } from '../src/tool-execution.js';
import type { Asset, ConversationDocument } from '../src/types.js';
import { parseConversationDocument } from '../src/validation.js';
import {
    emptyDocument,
    importedGeneration,
    RECORDED_AT,
    textBlock,
    toolCallBlock,
    toolResultTurn,
    userTurn,
} from './fixtures.js';

function memoryStore() {
    const pages = new Map<string, Uint8Array>();
    const records = new Map<string, Uint8Array>();
    const recordReads: string[] = [];
    const pageReads: string[] = [];
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            pageReads.push(ref.content_hash);
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('page unavailable');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(value) {
            recordReads.push(`${value.kind}:${value.id}`);
            const bytes = records.get(`${value.kind}:${value.content_hash}`);
            if (!bytes) throw new Error('record unavailable');
            return Uint8Array.from(bytes);
        },
        async writeRecord(value, bytes) {
            expect((await hashContentBytes(bytes)).content_hash).toBe(value.content_hash);
            records.set(`${value.kind}:${value.content_hash}`, Uint8Array.from(bytes));
        },
    };
    return { store, recordReads, pageReads, records, pages };
}

async function toolMediaCommands(source: { conversation_id: string; revision: number }) {
    const call = {
        ...toolCallBlock('block:dependency-call', 'call:dependency'),
        definition_id: 'definition:dependency',
    };
    const agent = {
        id: 'turn:dependency-agent',
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks: [call],
        status: 'completed' as const,
        timestamps: { recorded_at: RECORDED_AT },
        generation_id: 'generation:dependency',
        provenance: { type: 'generated' as const },
        model_visibility: 'include' as const,
    };
    const target = { provider: 'test', protocol: 'test.generate', model: 'test-model', adapter_version: '1' };
    const definition = { id: call.definition_id, name: call.tool_name, version: '1', input_schema: { type: 'object' } };
    const agentBatch: IndexedRecordBatchCommand['batch'] = {
        turns: [agent],
        tool_definitions: [definition],
        active_tool_definition_ids: [definition.id],
        generations: [
            {
                id: agent.generation_id,
                record_source: 'executed',
                request_id: 'request:dependency',
                attempt_id: 'attempt:dependency',
                purpose: 'conversation',
                requested_model: target.model,
                provider: target.provider,
                protocol: target.protocol,
                adapter_version: target.adapter_version,
                status: 'completed',
                timestamps: { recorded_at: RECORDED_AT },
                source,
                request_receipt: {
                    id: 'receipt:dependency-request',
                    request_id: 'request:dependency',
                    attempt_id: 'attempt:dependency',
                    source,
                    context_fingerprint: 'sha256:context',
                    tool_set_fingerprint: 'sha256:tools',
                    request_fingerprint: 'sha256:request',
                    target,
                    tool_definition_ids: [],
                    asset_versions: [],
                    item_mappings: [],
                    recorded_at: RECORDED_AT,
                },
            },
        ],
        context_entries: [{ id: 'entry:dependency-agent', type: 'source_turn', turn_id: agent.id }],
    };
    const bytes = new TextEncoder().encode('integrity-bound inline media');
    const integrity = await hashContentBytes(bytes);
    const asset = {
        id: 'asset:dependency-image',
        kind: 'image' as const,
        mime_type: 'image/png',
        storage: { type: 'inline_base64' as const, data: btoa(new TextDecoder().decode(bytes)) },
        provenance: { type: 'received' as const, source_turn_id: 'turn:dependency-result' },
        created_at: RECORDED_AT,
        ...integrity,
    };
    const result = {
        ...toolResultTurn('turn:dependency-result', call.call_id),
        execution_id: 'execution:dependency',
        blocks: [
            {
                id: 'block:dependency-result',
                type: 'tool_result' as const,
                call_id: call.call_id,
                status: 'success' as const,
                content: [
                    textBlock('block:dependency-text', 'Owned tool result.'),
                    { id: 'block:dependency-image', type: 'image' as const, asset_id: asset.id },
                ],
            },
        ],
    };
    const execution = {
        id: result.execution_id,
        call_id: call.call_id,
        executor: 'application' as const,
        status: 'success' as const,
        result_turn_id: result.id,
        result_fingerprint: await fingerprintJson(result.blocks[0]),
        recorded_at: RECORDED_AT,
        call_source: {
            conversation: { ...source, revision: source.revision + 1 },
            turn_id: agent.id,
            block_id: call.id,
            call_id: call.call_id,
            call_fingerprint: await fingerprintJson(call),
        },
    };
    const resultBatch: IndexedRecordBatchCommand['batch'] = {
        turns: [result],
        assets: [asset],
        execution_receipts: [execution],
        context_entries: [{ id: 'entry:dependency-result', type: 'source_turn', turn_id: result.id }],
    };
    return {
        call,
        agent,
        result,
        asset,
        execution,
        agentCommand: {
            conversation_id: source.conversation_id,
            batch: agentBatch,
            options: {
                expected_revision: source.revision,
                operation_id: 'operation:dependency-agent',
                payload_fingerprint: await fingerprintJson(agentBatch),
                recorded_at: RECORDED_AT,
            },
        },
        resultCommand: {
            conversation_id: source.conversation_id,
            batch: resultBatch,
            options: {
                expected_revision: source.revision + 1,
                operation_id: 'operation:dependency-result',
                payload_fingerprint: await fingerprintJson(resultBatch),
                recorded_at: RECORDED_AT,
            },
        },
    };
}

describe('indexed conversation snapshot', () => {
    it.each([
        { id: 'externalize-text', scope: 'on_append' as const, version: '1', config: {}, supported: true },
        { id: 'externalize-whole-exchange', scope: 'on_append' as const, version: '1', config: {}, supported: true },
        { id: 'externalize-text', scope: 'manual' as const, version: '1', config: {}, supported: true },
        { id: 'externalize-text', scope: 'on_budget' as const, version: '1', config: {}, supported: true },
        { id: 'externalize-text', scope: 'on_append' as const, version: '2', config: {}, supported: false },
    ])(
        'retains inherited policy while independently proving native capability for $id/$scope/$version',
        async (profile) => {
            const policy = await setProcessingPolicy(emptyDocument('conversation:inherited-capability'), {
                operation_id: 'policy:inherited-capability',
                expected_revision: 0,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors: [
                    {
                        id: profile.id,
                        scope: profile.scope,
                        version: profile.version,
                        config: profile.config,
                        required: true,
                        failure_behavior: 'block',
                    },
                ],
            });
            expect(supportsIndexedInheritedProcessingPolicy(policy.document.processing)).toBe(profile.supported);
            const memory = memoryStore();
            const staged = await stageIndexedConversationSnapshot(policy.document, undefined, memory.store);
            expect(staged.root.source).toEqual({
                conversation_id: policy.document.id,
                revision: policy.document.revision,
            });
            const bytes = await memory.store.readRecord({
                storage: 'record',
                kind: 'processing_header',
                id: staged.root.source.conversation_id,
                ...staged.root.processing_header,
            });
            const header = IndexedConversationProcessingHeaderSchema.parse(JSON.parse(new TextDecoder().decode(bytes)));
            expect(header.processors).toEqual(policy.document.processing.processors);
            if (profile.supported)
                await expect(assertIndexedCurrentPolicy(memory.store, staged.root, header)).resolves.toBeUndefined();
            else
                await expect(assertIndexedCurrentPolicy(memory.store, staged.root, header)).rejects.toThrow(
                    'Indexed readiness requires a registered bounded ordered text or whole-exchange policy',
                );
        },
    );

    it('rejects an unbound retrieval requirement before indexed record publication', async () => {
        const memory = memoryStore();
        const staged = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:indexed-retrieval-rejection'),
            undefined,
            memory.store,
        );
        const recordsBefore = memory.records.size;
        await expect(
            stageIndexedRecordBatch(
                staged.root,
                {
                    conversation_id: staged.root.source.conversation_id,
                    batch: {
                        retrieval_requirements: [
                            {
                                id: 'requirement:one',
                                asset_id: 'asset:one',
                                retrieval: {
                                    capability: 'read_artifact',
                                    version: 1,
                                    arguments: { path: 'archive.json' },
                                    tool_definition_id: 'definition:read',
                                },
                                accepted_asset_operation_id: 'append:one',
                            },
                        ],
                    },
                    options: {
                        expected_revision: staged.root.source.revision,
                        operation_id: 'append:one',
                        payload_fingerprint: 'sha256:append-one',
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('exact selected definition, asset and source block');
        expect(memory.records.size).toBe(recordsBefore);
    });
    it('accepts an exact selected nested reference, preserves its receipt across later turns, and rejects altered retries', async () => {
        const memory = memoryStore();
        let verifiedAssets = 0;
        memory.store.assertExternalAssetIntegrity = async () => {
            verifiedAssets += 1;
        };
        const initial = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:indexed-retrieval-append'),
            undefined,
            memory.store,
        );
        const commands = await toolMediaCommands(initial.root.source);
        const called = await stageIndexedRecordBatch(initial.root, commands.agentCommand, memory.store);
        const bytes = new TextEncoder().encode('Accepted archive bytes');
        const integrity = await hashContentBytes(bytes);
        const asset = {
            id: 'asset:retrieval-append',
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: { type: 'external' as const, resolver: 'test.blob', locator: { key: 'accepted.txt' } },
            provenance: { type: 'received' as const, source_turn_id: commands.result.id },
            created_at: RECORDED_AT,
            ...integrity,
        };
        const retrieval = {
            capability: 'read',
            version: 1 as const,
            tool_definition_id: 'definition:dependency',
            arguments: { path: 'accepted.txt' },
        };
        const reference = {
            id: 'block:retrieval-append',
            type: 'external_reference' as const,
            original_type: 'text' as const,
            asset_id: asset.id,
            description: 'Accepted archive',
            preview: 'Accepted archive preview',
            content_hash: asset.content_hash,
            retrieval,
        };
        const originalResult = commands.result.blocks[0];
        if (originalResult?.type !== 'tool_result') throw new Error('Tool result fixture is absent');
        const result = {
            ...commands.result,
            blocks: [{ ...originalResult, content: [reference] }],
        };
        const execution = {
            ...commands.execution,
            result_fingerprint: await fingerprintJson(result.blocks[0]),
        };
        const requirement = {
            id: 'requirement:retrieval-append',
            asset_id: asset.id,
            retrieval,
            accepted_asset_operation_id: commands.resultCommand.options.operation_id,
        };
        const batch: IndexedRecordBatchCommand['batch'] = {
            turns: [result],
            assets: [asset],
            execution_receipts: [execution],
            context_entries: commands.resultCommand.batch.context_entries,
            retrieval_requirements: [requirement],
        };
        const command: IndexedRecordBatchCommand = {
            ...commands.resultCommand,
            batch,
            options: {
                ...commands.resultCommand.options,
                payload_fingerprint: await fingerprintJson(batch),
            },
        };
        const accepted = await stageIndexedRecordBatch(called.root, command, memory.store);
        expect(accepted.receipt.accepted_retrieval_requirements).toEqual([requirement]);
        expect(verifiedAssets).toBe(1);
        if (!accepted.locator) throw new Error('Accepted root locator absent');
        const selected = await loadIndexedSelectedMediaCompactionContext(memory.store, accepted.root, accepted.locator);
        expect(resolveIndexedTextExternalReference(selected, asset.id, reference.id)).toMatchObject({
            asset,
            block: reference,
            accepted_asset_operation_id: command.options.operation_id,
        });
        const laterTurn = userTurn('turn:retrieval-later', 'block:retrieval-later');
        const laterBatch: IndexedRecordBatchCommand['batch'] = {
            turns: [laterTurn],
            context_entries: [{ id: 'entry:retrieval-later', type: 'source_turn', turn_id: laterTurn.id }],
            active_tool_definition_ids: [],
        };
        const later = await stageIndexedRecordBatch(
            accepted.root,
            {
                conversation_id: accepted.root.source.conversation_id,
                batch: laterBatch,
                options: {
                    expected_revision: accepted.root.source.revision,
                    operation_id: 'operation:retrieval-later',
                    payload_fingerprint: await fingerprintJson(laterBatch),
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        expect((await stageIndexedRecordBatch(later.root, command, memory.store)).applied).toBe(false);
        expect(verifiedAssets).toBe(1);
        for (const changed of [
            { ...batch, retrieval_requirements: [] },
            { ...batch, retrieval_requirements: [{ ...requirement, id: 'requirement:other' }] },
            {
                ...batch,
                retrieval_requirements: [
                    { ...requirement, retrieval: { ...retrieval, arguments: { path: 'forged.txt' } } },
                ],
            },
        ]) {
            await expect(
                stageIndexedRecordBatch(later.root, { ...command, batch: changed }, memory.store),
            ).rejects.toThrow('accepted retrieval requirements');
        }
        const missingSelectionBatch: IndexedRecordBatchCommand['batch'] = {
            ...batch,
            context_entries: [],
            retrieval_requirements: [{ ...requirement, accepted_asset_operation_id: 'operation:unselected-reference' }],
        };
        const missingSelection = { ...command, batch: missingSelectionBatch };
        missingSelection.options = {
            ...command.options,
            operation_id: 'operation:unselected-reference',
            expected_revision: called.root.source.revision,
            payload_fingerprint: await fingerprintJson(missingSelectionBatch),
        };
        await expect(stageIndexedRecordBatch(called.root, missingSelection, memory.store)).rejects.toThrow(
            'exact selected definition, asset and source block',
        );
        const wrongContent = structuredClone(command);
        const changedResult = wrongContent.batch.turns?.[0]?.blocks[0];
        if (changedResult?.type !== 'tool_result') throw new Error('Tool result fixture is absent');
        const changedReference = changedResult.content[0];
        if (changedReference?.type !== 'external_reference') throw new Error('Reference fixture is absent');
        changedReference.content_hash = `sha256:${'f'.repeat(64)}`;
        const changedExecution = wrongContent.batch.execution_receipts?.[0];
        if (!changedExecution) throw new Error('Execution fixture is absent');
        changedExecution.result_fingerprint = await fingerprintJson(changedResult);
        const changedRequirement = wrongContent.batch.retrieval_requirements?.[0];
        if (!changedRequirement) throw new Error('Retrieval fixture is absent');
        changedRequirement.accepted_asset_operation_id = 'operation:wrong-retrieval-content';
        wrongContent.options = {
            ...command.options,
            operation_id: 'operation:wrong-retrieval-content',
            expected_revision: called.root.source.revision,
            payload_fingerprint: await fingerprintJson(wrongContent.batch),
        };
        await expect(stageIndexedRecordBatch(called.root, wrongContent, memory.store)).rejects.toThrow(
            'exact selected definition, asset and source block',
        );
    });

    it('proves incoming replay dependencies without relying on call block order and rejects foreign identities', async () => {
        const source = emptyDocument('conversation:indexed-replay-dependencies');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const commands = await toolMediaCommands(initial.root.source);
        const replay = {
            id: 'block:dependency-replay',
            type: 'native_replay' as const,
            adapter: 'test',
            protocol: 'test.generate',
            compatibility_scope: {
                provider: 'test',
                protocol: 'test.generate',
                model: 'test-model',
                adapter_version: '1',
            },
            payload: { retained: 'exact native state' },
            dependencies: {
                turn_ids: [commands.agent.id],
                block_ids: [commands.call.id],
                call_ids: [commands.call.call_id],
                request_ids: ['request:dependency'],
            },
        };
        const command = {
            ...commands.agentCommand,
            batch: { ...commands.agentCommand.batch, turns: [{ ...commands.agent, blocks: [replay, commands.call] }] },
        };
        const recordsBefore = memory.records.size;
        await expect(
            stageIndexedRecordBatch(
                initial.root,
                {
                    ...command,
                    batch: {
                        ...command.batch,
                        turns: [
                            {
                                ...commands.agent,
                                blocks: [
                                    {
                                        ...replay,
                                        dependencies: { ...replay.dependencies, block_ids: ['block:foreign'] },
                                    },
                                    commands.call,
                                ],
                            },
                        ],
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('exact executed generation/dependency');
        expect(memory.records.size).toBe(recordsBefore);
        const called = await stageIndexedRecordBatch(initial.root, command, memory.store);
        const completed = await stageIndexedRecordBatch(called.root, commands.resultCommand, memory.store);
        if (!completed.locator) throw new Error('Replay dependency root is absent');
        const projection = await loadIndexedSelectedDependencyContext(memory.store, completed.root, completed.locator);
        expect(projection.turns[0]?.selected_blocks[0]).toEqual(replay);
        expect(projection.generation_witnesses[commands.agent.generation_id]?.generation.request_id).toBe(
            'request:dependency',
        );
    });

    it('requires first-adoption custody even for an unattached external asset and recovers its exact receipt', async () => {
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:unattached'),
            undefined,
            memory.store,
        );
        const commands = await toolMediaCommands(initial.root.source);
        const asset = {
            ...commands.asset,
            provenance: { type: 'received' as const },
            storage: {
                type: 'external' as const,
                resolver: 'url',
                locator: { url: 'gs://project/runs/other/media/image.png' },
            },
        };
        const batch = { assets: [asset] };
        const command: IndexedRecordBatchCommand = {
            conversation_id: initial.root.source.conversation_id,
            batch,
            options: {
                expected_revision: initial.root.source.revision,
                operation_id: 'operation:unattached',
                payload_fingerprint: await fingerprintJson(batch),
                recorded_at: RECORDED_AT,
            },
        };
        const before = memory.records.size;
        await expect(stageIndexedRecordBatch(initial.root, command, memory.store)).rejects.toThrow('host custody');
        expect(memory.records.size).toBe(before);
        let checks = 0;
        memory.store.assertExternalAssetIntegrity = async (captured) => {
            checks++;
            expect(captured).toEqual(asset);
        };
        const accepted = await stageIndexedRecordBatch(initial.root, command, memory.store);
        expect(accepted.receipt.accepted_asset_ids).toEqual([asset.id]);
        expect(checks).toBe(1);
        delete memory.store.assertExternalAssetIntegrity;
        const recovered = await stageIndexedRecordBatch(accepted.root, command, memory.store);
        expect(recovered.applied).toBe(false);
        expect(recovered.receipt).toEqual(accepted.receipt);
        expect(checks).toBe(1);
    });

    it('keeps external tool-result publication, retry and selected closure bounded at 10k and 100k cold turns', async () => {
        const phaseProfiles: Record<string, number>[] = [];
        for (const coldCount of [10_000, 100_000]) {
            const memory = memoryStore();
            const source = emptyDocument('conversation:external-result');
            const cold = userTurn('turn:external-cold-template', 'block:external-cold-template');
            source.turns = Array.from({ length: coldCount }, (_, index) => ({
                ...cold,
                id: `turn:external-cold:${index}`,
                blocks: [{ ...cold.blocks[0], id: `block:external-cold:${index}` }],
            }));
            const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
            memory.recordReads.length = 0;
            memory.pageReads.length = 0;
            const profile: Record<string, number> = {};
            const observePhase = (phase: string) => {
                expect(memory.recordReads.some((id) => id.startsWith('blocks:block:external-cold:'))).toBe(false);
                expect(memory.recordReads.length).toBeLessThan(64);
                expect(memory.pageReads.length).toBeLessThan(512);
                profile[phase] = memory.recordReads.length;
                memory.recordReads.length = 0;
                memory.pageReads.length = 0;
            };
            const commands = await toolMediaCommands(initial.root.source);
            const called = await stageIndexedRecordBatch(initial.root, commands.agentCommand, memory.store);
            observePhase('accepted_call');
            const external = {
                ...commands.asset,
                storage: {
                    type: 'external' as const,
                    resolver: 'url',
                    locator: { url: 'gs://project/runs/run/media/image.png' },
                },
            };
            const batch = { ...commands.resultCommand.batch, assets: [external] };
            const command = {
                ...commands.resultCommand,
                batch,
                options: { ...commands.resultCommand.options, payload_fingerprint: await fingerprintJson(batch) },
            };
            const before = memory.records.size;
            await expect(stageIndexedRecordBatch(called.root, command, memory.store)).rejects.toThrow('host custody');
            expect(memory.records.size).toBe(before);
            observePhase('rejected_custody');
            let checks = 0;
            memory.store.assertExternalAssetIntegrity = async (asset) => {
                checks++;
                expect(asset).toEqual(external);
            };
            const completed = await stageIndexedRecordBatch(called.root, command, memory.store);
            observePhase('accepted_result');
            if (!completed.locator) throw new Error('External result root missing');
            const retry = await stageIndexedRecordBatch(completed.root, command, memory.store);
            expect(retry.applied).toBe(false);
            expect(retry.receipt).toEqual(completed.receipt);
            expect(checks).toBe(1);
            observePhase('exact_retry');
            const selected = await loadIndexedSelectedMediaCompactionContext(
                memory.store,
                completed.root,
                completed.locator,
            );
            expect(selected.assets[external.id]).toEqual(external);
            expect(selected.execution_witnesses?.[commands.execution.id]).toEqual(commands.execution);
            observePhase('selected_closure');
            await expect(
                loadIndexedSelectedDependencyContext(memory.store, completed.root, completed.locator),
            ).rejects.toThrow();
            observePhase('unsupported_profile');
            phaseProfiles.push(profile);
        }
        expect(phaseProfiles[1]).toEqual(phaseProfiles[0]);
    }, 120_000);

    it('loads accepted compaction/retrieval witnesses without reading cold originals at 100k turns', async () => {
        const coldCount = 100_000;
        let source = emptyDocument('conversation:compaction-scale');
        const original = userTurn('turn:retired-original', 'block:retired-original');
        source = appendConversationRecords(
            source,
            {
                turns: [original],
                context_entries: [{ id: 'entry:retired-original', type: 'source_turn', turn_id: original.id }],
            },
            {
                expected_revision: source.revision,
                operation_id: 'operation:original',
                payload_fingerprint: 'sha256:original',
                recorded_at: RECORDED_AT,
            },
        ).document;
        const originalBlock = original.blocks[0];
        if (originalBlock?.type !== 'text') throw new Error('Original must be text');
        const integrity = await hashContentBytes(new TextEncoder().encode(originalBlock.text));
        const asset = {
            id: 'asset:archived-original',
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: { type: 'external' as const, resolver: 'test.blob', locator: { key: 'original.txt' } },
            provenance: { type: 'received' as const, source_turn_id: original.id },
            created_at: RECORDED_AT,
            ...integrity,
        };
        const tool = {
            id: 'definition:read-original',
            name: 'read_original',
            version: `sha256:${'a'.repeat(64)}`,
            input_schema: true,
        };
        source = appendConversationRecords(
            source,
            { assets: [asset], tool_definitions: [tool], active_tool_definition_ids: [tool.id] },
            {
                expected_revision: source.revision,
                operation_id: 'operation:archive',
                payload_fingerprint: 'sha256:archive',
                recorded_at: RECORDED_AT,
            },
        ).document;
        const selectedEntry = source.context.entries[0];
        if (!selectedEntry) throw new Error('Original selected entry missing');
        const selection = {
            expected_revision: source.revision,
            expected_context_revision: source.context.revision,
            entry_ids: [selectedEntry.id],
            selected_entries: [selectedEntry],
            selected_block_ids: { [selectedEntry.id]: [originalBlock.id] },
        };
        const plan = await planContextChange(source, selection);
        expect(plan.source_turn_ids).toEqual([original.id]);
        expect(plan.source_block_ids).toEqual([]);
        const reference = {
            id: 'block:retrieval',
            type: 'external_reference' as const,
            original_type: 'text' as const,
            asset_id: asset.id,
            description: 'Retained original',
            preview: originalBlock.text,
            content_hash: asset.content_hash,
            retrieval: {
                capability: tool.name,
                version: 1,
                tool_definition_id: tool.id,
                arguments: { asset_id: asset.id },
            },
        };
        const replacement = {
            ...userTurn('turn:replacement'),
            kind: 'agent' as const,
            blocks: [reference],
            provenance: {
                type: 'derived' as const,
                derivation_id: 'compaction:original',
                source_turn_ids: [...plan.source_turn_ids],
                ...(plan.source_block_ids.length ? { source_block_ids: [...plan.source_block_ids] } : {}),
                source_hash: plan.source_fingerprint,
            },
        };
        const compacted = await applyContextChange(source, {
            ...selection,
            operation_id: 'operation:compact',
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: RECORDED_AT,
            proposal: {
                kind: 'replace_with_compaction',
                compaction_id: 'compaction:original',
                strategy: { id: 'externalize-text', version: '1', configuration_fingerprint: 'sha256:config' },
                replacement_turns: [replacement],
                fidelity: 'retrievable',
                accepted_asset_operation_id: 'operation:archive',
                retained_asset_ids: [asset.id],
                generation_ids: [],
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            },
        });
        const cold = userTurn('turn:cold-template', 'block:cold-template');
        const large = {
            ...compacted.document,
            turns: [
                ...compacted.document.turns,
                ...Array.from({ length: coldCount - compacted.document.turns.length }, (_, index) => ({
                    ...cold,
                    id: `turn:cold-compaction:${index}`,
                    blocks: [{ ...cold.blocks[0], id: `block:cold-compaction:${index}` }],
                })),
            ],
        };
        const memory = memoryStore();
        const staged = await stageIndexedConversationSnapshot(large, undefined, memory.store);
        memory.recordReads.length = 0;
        memory.pageReads.length = 0;
        const loaded = await loadIndexedSelectedMediaCompactionContext(memory.store, staged.root, staged.locator);
        expect(loaded.turns).toEqual([]);
        expect(loaded.replacement_turns?.[0]?.projection.selected_blocks).toEqual([reference]);
        expect(loaded.compaction_witnesses?.['compaction:original']?.acceptance).toEqual(
            compacted.document.operation_receipts['operation:compact'],
        );
        expect(resolveIndexedTextExternalReference(loaded, asset.id, reference.id).tool_definition).toEqual(tool);
        expect(
            memory.recordReads.some(
                (id) => id.startsWith('turns:turn:cold-compaction:') || id === `turns:${original.id}`,
            ),
        ).toBe(false);
        expect(memory.recordReads.length).toBeLessThan(24);
        expect(memory.pageReads.length).toBeLessThan(128);
        expect(await getPagedRecord(memory.store, staged.root.directories.block_owners, reference.id)).toEqual({
            storage: 'marker',
            kind: 'block_owner',
            id: replacement.id,
        });
        const commands = await toolMediaCommands(staged.root.source);
        const responseCommand: IndexedRecordBatchCommand = {
            ...commands.agentCommand,
            batch: {
                ...commands.agentCommand.batch,
                generations: commands.agentCommand.batch.generations?.map((generation) => {
                    if (generation.record_source !== 'executed') throw new Error('Executed fixture generation absent');
                    return {
                        ...generation,
                        request_receipt: {
                            ...generation.request_receipt,
                            item_mappings: [
                                { canonical_id: replacement.id, native_id: 'selected:turn', kind: 'turn' },
                                { canonical_id: reference.id, native_id: 'selected:block', kind: 'block' },
                            ],
                        },
                    };
                }),
            },
        };
        responseCommand.options.payload_fingerprint = await fingerprintJson(responseCommand.batch);
        memory.recordReads.length = 0;
        memory.pageReads.length = 0;
        const response = await stageIndexedRecordBatch(staged.root, responseCommand, memory.store);
        expect(response.applied).toBe(true);
        expect(await getPagedRecord(memory.store, response.root.directories.deletion_blockers, replacement.id)).toEqual(
            {
                storage: 'marker',
                kind: 'delete_blocker',
                id: replacement.id,
            },
        );
        expect(await getPagedRecord(memory.store, response.root.directories.deletion_blockers, original.id)).toEqual({
            storage: 'marker',
            kind: 'delete_blocker',
            id: original.id,
        });
        expect(
            memory.recordReads.some(
                (id) => id.startsWith('turns:turn:cold-compaction:') || id === `turns:${original.id}`,
            ),
        ).toBe(false);
        expect(memory.recordReads.length).toBeLessThan(64);
        expect(memory.pageReads.length).toBeLessThan(512);
        expect((await stageIndexedRecordBatch(response.root, responseCommand, memory.store)).applied).toBe(false);
        if (!response.locator) throw new Error('Response root locator absent');
        await expect(
            stageIndexedConversationDelete(
                response.root,
                {
                    operation_id: 'operation:delete-compaction-original',
                    source: response.root.source,
                    expected_source_root: response.locator,
                    recorded_at: RECORDED_AT,
                    dependency_policy: 'reject',
                    turn_ids: [original.id],
                },
                memory.store,
            ),
        ).rejects.toThrow('retained asset or derivation lineage');
        await expect(
            stageIndexedConversationDelete(
                response.root,
                {
                    operation_id: 'operation:delete-compaction-replacement',
                    source: response.root.source,
                    expected_source_root: response.locator,
                    recorded_at: RECORDED_AT,
                    dependency_policy: 'reject',
                    turn_ids: [replacement.id],
                },
                memory.store,
            ),
        ).rejects.toThrow('prior active-context exclusion');
        const foreignCompaction = structuredClone(loaded);
        const witness = foreignCompaction.compaction_witnesses?.['compaction:original'];
        if (!witness) throw new Error('Compaction witness missing');
        witness.compaction.retained_asset_ids = ['asset:foreign'];
        expect(() => resolveIndexedTextExternalReference(foreignCompaction, asset.id, reference.id)).toThrow(
            'compaction/asset evidence',
        );
        const corrupt = structuredClone(loaded);
        delete corrupt.operation_witnesses?.['operation:archive'];
        expect(() => resolveIndexedTextExternalReference(corrupt, asset.id, reference.id)).toThrow(
            'accepted-asset evidence',
        );
        corrupt.operation_witnesses = loaded.operation_witnesses;
        const requirement = corrupt.context.retrieval_requirements[0];
        if (!requirement) throw new Error('Retrieval requirement missing');
        requirement.retrieval.arguments = { asset_id: 'asset:foreign' };
        expect(() => resolveIndexedTextExternalReference(corrupt, asset.id, reference.id)).toThrow();
    }, 120_000);

    it('requires exact accepted call/source/result and inline media bytes before dependency preparation', async () => {
        const source = emptyDocument('conversation:dependency-negatives');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const commands = await toolMediaCommands(initial.root.source);
        const called = await stageIndexedRecordBatch(initial.root, commands.agentCommand, memory.store);
        const recordCount = memory.records.size;
        await expect(
            stageIndexedRecordBatch(
                called.root,
                {
                    ...commands.resultCommand,
                    batch: {
                        ...commands.resultCommand.batch,
                        execution_receipts: [
                            {
                                ...commands.execution,
                                call_source: {
                                    ...commands.execution.call_source,
                                    conversation: { ...called.root.source, revision: 0 },
                                },
                            },
                        ],
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('source differs');
        await expect(
            stageIndexedRecordBatch(
                called.root,
                {
                    ...commands.resultCommand,
                    batch: {
                        ...commands.resultCommand.batch,
                        assets: [{ ...commands.asset, content_hash: `sha256:${'a'.repeat(64)}` }],
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('inline custody');
        expect(memory.records.size).toBe(recordCount);
        const completed = await stageIndexedRecordBatch(called.root, commands.resultCommand, memory.store);
        if (!completed.locator) throw new Error('Indexed dependency root is absent');
        const selected = await loadIndexedSelectedDependencyContext(memory.store, completed.root, completed.locator);
        expect(selected.turns.map((turn) => turn.header.id)).toEqual([commands.agent.id, commands.result.id]);
        expect(Object.keys(selected.execution_witnesses ?? {})).toEqual([commands.execution.id]);
        expect((await stageIndexedRecordBatch(completed.root, commands.resultCommand, memory.store)).applied).toBe(
            false,
        );
        await expect(loadIndexedSelectedTextContext(memory.store, completed.root, completed.locator)).rejects.toThrow(
            'unsupported selected',
        );
    });

    it('stages independent cold turns with bounded concurrent read-back before publishing a root', async () => {
        const source = emptyDocument('conversation:indexed-bounded-migration');
        const turns = Array.from({ length: 96 }, (_, index) => userTurn(`turn:cold:${index}`, `block:cold:${index}`));
        const memory = memoryStore();
        let inFlight = 0;
        let peak = 0;
        const store: IndexedConversationRecordStore = {
            ...memory.store,
            async writeRecord(value, bytes) {
                inFlight += 1;
                peak = Math.max(peak, inFlight);
                try {
                    await new Promise((resolve) => setTimeout(resolve, 1));
                    await memory.store.writeRecord(value, bytes);
                } finally {
                    inFlight -= 1;
                }
            },
        };
        const staged = await stageIndexedConversationSnapshot(
            {
                ...source,
                turns,
                context: {
                    ...source.context,
                    entries: [{ id: 'entry:active', type: 'source_turn', turn_id: turns[95].id }],
                },
            },
            undefined,
            store,
        );
        expect(staged.root.turn_count).toBe(turns.length);
        expect(peak).toBeGreaterThan(1);
        expect(peak).toBeLessThanOrEqual(32);
        expect((await loadIndexedActiveContext(store, staged.root)).entries).toEqual([
            { id: 'entry:active', type: 'source_turn', turn_id: turns[95].id },
        ]);
    });
    it('drains a failed cold import window before rejecting without publishing a root', async () => {
        const source = emptyDocument('conversation:indexed-failed-migration');
        const turns = Array.from({ length: 64 }, (_, index) => userTurn(`turn:cold:${index}`, `block:cold:${index}`));
        const memory = memoryStore();
        let inFlight = 0;
        const store: IndexedConversationRecordStore = {
            ...memory.store,
            async writeRecord(value, bytes) {
                inFlight += 1;
                try {
                    await new Promise((resolve) => setTimeout(resolve, 1));
                    if (value.kind === 'turns' && value.id === turns[0].id) throw new Error('failed immutable write');
                    await memory.store.writeRecord(value, bytes);
                } finally {
                    inFlight -= 1;
                }
            },
        };
        await expect(stageIndexedConversationSnapshot({ ...source, turns }, undefined, store)).rejects.toThrow(
            'failed immutable write',
        );
        expect(inFlight).toBe(0);
        expect([...memory.records.keys()].some((key) => key.startsWith('root:'))).toBe(false);
    });

    it('binds one received text input to its exact indexed append receipt before preparation', async () => {
        const source = emptyDocument('conversation:indexed-received-input');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const turn = userTurn('turn:received', 'block:received');
        const accepted = await stageIndexedRecordBatch(
            initial.root,
            {
                conversation_id: source.id,
                batch: {
                    turns: [turn],
                    context_entries: [{ id: 'entry:received', type: 'source_turn', turn_id: turn.id }],
                },
                options: {
                    expected_revision: source.revision,
                    operation_id: 'input:received',
                    payload_fingerprint: 'sha256:received',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        const identity = {
            operation_id: 'input:received',
            result_revision: accepted.root.source.revision,
            response_operation_id: 'response:received',
        };
        expect(await assertIndexedFreshReceivedTextInput(memory.store, accepted.root, identity)).toEqual(
            accepted.receipt,
        );
        expect(await assertIndexedFreshTextInput(memory.store, accepted.root, identity)).toEqual(accepted.receipt);
        await expect(
            assertIndexedFreshReceivedTextInput(memory.store, accepted.root, {
                ...identity,
                result_revision: source.revision,
            }),
        ).rejects.toThrow('one received input');
        await expect(
            assertIndexedFreshReceivedTextInput(memory.store, accepted.root, {
                ...identity,
                response_operation_id: identity.operation_id,
            }),
        ).rejects.toThrow('already accepted');
        const noContext = await stageIndexedRecordBatch(
            initial.root,
            {
                conversation_id: source.id,
                batch: { turns: [turn] },
                options: {
                    expected_revision: source.revision,
                    operation_id: 'input:unselected',
                    payload_fingerprint: 'sha256:unselected',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        await expect(
            assertIndexedFreshReceivedTextInput(memory.store, noContext.root, {
                operation_id: 'input:unselected',
                result_revision: noContext.root.source.revision,
                response_operation_id: 'response:unselected',
            }),
        ).rejects.toThrow('one received input');
    });
    it('maintains live links through consecutive, disjoint and final deletions before fresh selected preparation', async () => {
        const source = emptyDocument('conversation:indexed-delete-links');
        const memory = memoryStore();
        const migrated = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const turns = ['a', 'b', 'c', 'd', 'e'].map((name) => userTurn(`turn:${name}`, `block:${name}`));
        const accepted = await stageIndexedRecordBatch(
            migrated.root,
            {
                conversation_id: source.id,
                batch: { turns },
                options: {
                    expected_revision: 0,
                    operation_id: 'operation:five',
                    payload_fingerprint: 'sha256:five',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        if (!accepted.locator) throw new Error('Indexed batch lacks its staged root');
        const deleteTurns = async (
            root: typeof accepted.root,
            locator: typeof accepted.locator,
            ids: string[],
            id: string,
        ) =>
            stageIndexedConversationDelete(
                root,
                {
                    operation_id: id,
                    source: root.source,
                    expected_source_root: locator,
                    recorded_at: RECORDED_AT,
                    dependency_policy: 'reject',
                    turn_ids: ids,
                },
                memory.store,
            );
        const consecutive = await deleteTurns(accepted.root, accepted.locator, ['turn:a', 'turn:b'], 'delete:ab');
        expect(consecutive.root.live_turn_count).toBe(3);
        expect(consecutive.root.active_tail_turn_id).toBe('turn:e');
        if (!consecutive.locator) throw new Error('Indexed consecutive delete lacks its staged root');
        const disjoint = await deleteTurns(consecutive.root, consecutive.locator, ['turn:c', 'turn:e'], 'delete:ce');
        expect(disjoint.root.live_turn_count).toBe(1);
        expect(disjoint.root.active_tail_turn_id).toBe('turn:d');
        if (!disjoint.locator) throw new Error('Indexed disjoint delete lacks its staged root');
        const final = await deleteTurns(disjoint.root, disjoint.locator, ['turn:d'], 'delete:d');
        expect(final.root.live_turn_count).toBe(0);
        expect(final.root.active_tail_turn_id).toBeNull();
        const fresh = userTurn('turn:fresh', 'block:fresh');
        const appended = await stageIndexedRecordBatch(
            final.root,
            {
                conversation_id: source.id,
                batch: {
                    turns: [fresh],
                    context_entries: [{ id: 'entry:fresh', type: 'source_turn', turn_id: fresh.id }],
                },
                options: {
                    expected_revision: final.root.source.revision,
                    operation_id: 'operation:fresh',
                    payload_fingerprint: 'sha256:fresh',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        expect(appended.root.live_turn_count).toBe(1);
        expect(appended.root.active_tail_turn_id).toBe(fresh.id);
        if (!appended.locator) throw new Error('Indexed fresh append lacks its staged root');
        const selected = await loadIndexedSelectedTextContext(memory.store, appended.root, appended.locator);
        expect(selected.turns.map((turn) => turn.header.id)).toEqual([fresh.id]);
        expect(selected.context.entries.map((entry) => entry.turn_id)).toEqual([fresh.id]);
    });

    it('maintains received-asset source blockers from later indexed appends', async () => {
        const source = emptyDocument('conversation:indexed-delete-asset');
        const memory = memoryStore();
        const migrated = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const old = userTurn('turn:old', 'block:old');
        const first = await stageIndexedRecordBatch(
            migrated.root,
            {
                conversation_id: source.id,
                batch: { turns: [old] },
                options: {
                    expected_revision: 0,
                    operation_id: 'operation:old',
                    payload_fingerprint: 'sha256:old',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        const asset = {
            id: 'asset:later',
            kind: 'image' as const,
            mime_type: 'image/png',
            storage: { type: 'inline_base64' as const, data: 'AQID' },
            provenance: { type: 'received' as const, source_turn_id: old.id },
            created_at: RECORDED_AT,
        };
        const second = await stageIndexedRecordBatch(
            first.root,
            {
                conversation_id: source.id,
                batch: { assets: [asset] },
                options: {
                    expected_revision: first.root.source.revision,
                    operation_id: 'operation:asset',
                    payload_fingerprint: 'sha256:asset',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        expect(await getPagedRecord(memory.store, second.root.directories.deletion_blockers, old.id)).toEqual({
            storage: 'marker',
            kind: 'delete_blocker',
            id: old.id,
        });
        if (!second.locator) throw new Error('Indexed asset append lacks its staged root');
        await expect(
            stageIndexedConversationDelete(
                second.root,
                {
                    operation_id: 'operation:delete-old',
                    source: second.root.source,
                    expected_source_root: second.locator,
                    recorded_at: RECORDED_AT,
                    dependency_policy: 'reject',
                    turn_ids: [old.id],
                },
                memory.store,
            ),
        ).rejects.toThrow('retained asset or derivation lineage');
    });

    it('retains complete point-lookup delete witnesses across later indexed appends', async () => {
        const source = emptyDocument('conversation:indexed-delete-index');
        const old = userTurn('turn:old', 'block:old');
        const accepted = appendConversationRecords(
            source,
            { turns: [old] },
            {
                expected_revision: 0,
                operation_id: 'operation:old',
                payload_fingerprint: 'sha256:old',
                recorded_at: RECORDED_AT,
            },
        );
        const memory = memoryStore();
        const migrated = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        expect(migrated.root.delete_index_profile).toBeDefined();
        expect(migrated.root.live_turn_count).toBe(1);
        expect(migrated.root.active_tail_turn_id).toBe(old.id);
        expect(await getPagedRecord(memory.store, migrated.root.directories.turn_acceptances, old.id)).toEqual({
            storage: 'marker',
            kind: 'turn_acceptance',
            id: 'operation:old',
        });
        expect(await getPagedRecord(memory.store, migrated.root.directories.block_owners, old.blocks[0].id)).toEqual({
            storage: 'marker',
            kind: 'block_owner',
            id: old.id,
        });
        const child = { ...userTurn('turn:child', 'block:child'), parent_turn_id: old.id };
        const next = await stageIndexedRecordBatch(
            migrated.root,
            {
                conversation_id: source.id,
                batch: { turns: [child] },
                options: {
                    expected_revision: 1,
                    operation_id: 'operation:child',
                    payload_fingerprint: 'sha256:child',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        expect(next.root.live_turn_count).toBe(2);
        expect(next.root.active_tail_turn_id).toBe(child.id);
        expect(await getPagedRecord(memory.store, next.root.directories.deletion_blockers, old.id)).toEqual({
            storage: 'marker',
            kind: 'delete_blocker',
            id: old.id,
        });
        expect(await getPagedRecord(memory.store, next.root.directories.turn_acceptances, child.id)).toEqual({
            storage: 'marker',
            kind: 'turn_acceptance',
            id: 'operation:child',
        });
        if (!next.locator) throw new Error('Indexed child append lacks its staged root');
        await expect(
            stageIndexedConversationDelete(
                next.root,
                {
                    operation_id: 'operation:delete-parent',
                    source: next.root.source,
                    expected_source_root: next.locator,
                    recorded_at: RECORDED_AT,
                    dependency_policy: 'reject',
                    turn_ids: [old.id],
                },
                memory.store,
            ),
        ).rejects.toThrow('live dependent turn');
    });

    it('deletes an older excluded turn and recovers its accepted append from the pinned predecessor', async () => {
        const source = emptyDocument('conversation:indexed-delete-retry');
        const old = userTurn('turn:old', 'block:old');
        const later = userTurn('turn:later', 'block:later');
        const firstBatch = { turns: [old] };
        const firstOptions = {
            expected_revision: 0,
            operation_id: 'operation:old',
            payload_fingerprint: 'sha256:old',
            recorded_at: RECORDED_AT,
        };
        const secondOptions = {
            expected_revision: 1,
            operation_id: 'operation:later',
            payload_fingerprint: 'sha256:later',
            recorded_at: RECORDED_AT,
        };
        const first = appendConversationRecords(source, firstBatch, firstOptions);
        const second = appendConversationRecords(first.document, { turns: [later] }, secondOptions);
        const memory = memoryStore();
        const migrated = await stageIndexedConversationSnapshot(second.document, undefined, memory.store);
        const command = {
            operation_id: 'operation:delete',
            source: migrated.root.source,
            expected_source_root: migrated.locator,
            recorded_at: RECORDED_AT,
            dependency_policy: 'reject' as const,
            turn_ids: [old.id],
        };
        const deleted = await stageIndexedConversationDelete(migrated.root, command, memory.store);
        expect(deleted.applied).toBe(true);
        expect(deleted.root.turn_count).toBe(2);
        expect(deleted.root.live_turn_count).toBe(1);
        expect(deleted.root.active_tail_turn_id).toBe(later.id);
        expect(await getPagedRecord(memory.store, deleted.root.directories.turns, old.id)).toEqual({
            storage: 'marker',
            kind: 'deleted_turn',
            id: old.id,
        });
        expect(await getPagedRecord(memory.store, deleted.root.directories.blocks, old.blocks[0].id)).toEqual({
            storage: 'marker',
            kind: 'deleted_block',
            id: old.blocks[0].id,
        });
        const advanced = await stageIndexedRecordBatch(
            deleted.root,
            {
                conversation_id: source.id,
                batch: { turns: [userTurn('turn:newer', 'block:newer')] },
                options: {
                    expected_revision: deleted.root.source.revision,
                    operation_id: 'operation:newer',
                    payload_fingerprint: 'sha256:newer',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        const recordsBeforeRetry = memory.records.size;
        const replay = await stageIndexedRecordBatch(
            advanced.root,
            { conversation_id: source.id, batch: firstBatch, options: firstOptions },
            memory.store,
        );
        expect(replay).toMatchObject({
            applied: false,
            receipt: first.document.operation_receipts[firstOptions.operation_id],
        });
        expect(memory.records.size).toBe(recordsBeforeRetry);
        expect((await stageIndexedConversationDelete(advanced.root, command, memory.store)).applied).toBe(false);
        await expect(
            stageIndexedRecordBatch(
                advanced.root,
                {
                    conversation_id: source.id,
                    batch: { turns: [{ ...old, blocks: [textBlock('block:old', 'changed')] }] },
                    options: firstOptions,
                },
                memory.store,
            ),
        ).rejects.toThrow();
        await expect(
            stageIndexedRecordBatch(
                advanced.root,
                {
                    conversation_id: source.id,
                    batch: { turns: [old] },
                    options: {
                        expected_revision: advanced.root.source.revision,
                        operation_id: 'operation:new',
                        payload_fingerprint: 'sha256:new',
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('already exists');
        memory.records.delete(`root:${migrated.locator.content_hash}`);
        await expect(
            stageIndexedRecordBatch(
                advanced.root,
                { conversation_id: source.id, batch: firstBatch, options: firstOptions },
                memory.store,
            ),
        ).rejects.toThrow('record unavailable');
    });

    it('appends canonical user/media, generated tool call and terminal tool result with materialized parity', async () => {
        const source = emptyDocument('conversation:indexed-batch');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const media = {
            id: 'asset:image',
            kind: 'image' as const,
            mime_type: 'image/png',
            storage: { type: 'inline_base64' as const, data: 'AQID' },
            provenance: { type: 'received' as const, source_turn_id: 'turn:user' },
            created_at: RECORDED_AT,
        };
        const user = {
            ...userTurn('turn:user', 'block:user'),
            blocks: [
                textBlock('block:user', 'look'),
                { id: 'block:image', type: 'image' as const, asset_id: media.id },
            ],
        };
        const definition = { id: 'definition:read', name: 'read', version: '1', input_schema: { type: 'object' } };
        const userBatch = {
            turns: [user],
            assets: [media],
            tool_definitions: [definition],
            active_tool_definition_ids: [definition.id],
            context_entries: [{ id: 'entry:user', type: 'source_turn' as const, turn_id: user.id }],
        };
        const userOptions = {
            expected_revision: 0,
            operation_id: 'operation:user',
            payload_fingerprint: 'sha256:user-batch',
            recorded_at: RECORDED_AT,
        };
        const materializedUser = appendConversationRecords(source, userBatch, userOptions);
        const stagedUser = await stageIndexedRecordBatch(
            initial.root,
            { conversation_id: source.id, batch: userBatch, options: userOptions },
            memory.store,
        );
        expect(stagedUser.receipt).toEqual(materializedUser.document.operation_receipts[userOptions.operation_id]);
        expect((await loadIndexedActiveContext(memory.store, stagedUser.root)).entries).toEqual(
            materializedUser.document.context.entries,
        );
        const retry = await stageIndexedRecordBatch(
            stagedUser.root,
            { conversation_id: source.id, batch: userBatch, options: userOptions },
            memory.store,
        );
        expect(retry.applied).toBe(false);
        const retryAt = '2026-09-11T00:01:00.000Z';
        const userTimestampRetry = {
            ...userBatch,
            turns: [{ ...user, timestamps: { recorded_at: retryAt } }],
            assets: [{ ...media, created_at: retryAt }],
        };
        expect(
            appendConversationRecords(materializedUser.document, userTimestampRetry, {
                ...userOptions,
                recorded_at: retryAt,
            }).applied,
        ).toBe(false);
        expect(
            (
                await stageIndexedRecordBatch(
                    stagedUser.root,
                    {
                        conversation_id: source.id,
                        batch: userTimestampRetry,
                        options: { ...userOptions, recorded_at: retryAt },
                    },
                    memory.store,
                )
            ).applied,
        ).toBe(false);
        await expect(
            stageIndexedRecordBatch(
                stagedUser.root,
                {
                    conversation_id: source.id,
                    batch: { ...userBatch, turns: [{ ...user, blocks: [textBlock('block:user', 'changed')] }] },
                    options: userOptions,
                },
                memory.store,
            ),
        ).rejects.toThrow();

        const requestReceipt = {
            id: 'receipt:request',
            request_id: 'request:1',
            attempt_id: 'attempt:1',
            source: { conversation_id: source.id, revision: 1 },
            source_tail_turn_id: user.id,
            context_fingerprint: 'sha256:context',
            tool_set_fingerprint: 'sha256:tools',
            request_fingerprint: 'sha256:request',
            target: { provider: 'test', protocol: 'test.generate', model: 'test-model', adapter_version: '1' },
            tool_definition_ids: [definition.id],
            asset_versions: [],
            item_mappings: [],
            recorded_at: RECORDED_AT,
        };
        const generation = {
            id: 'generation:1',
            record_source: 'executed' as const,
            request_id: requestReceipt.request_id,
            attempt_id: requestReceipt.attempt_id,
            purpose: 'conversation',
            requested_model: 'test-model',
            provider: 'test',
            protocol: 'test.generate',
            adapter_version: '1',
            status: 'completed' as const,
            timestamps: { recorded_at: RECORDED_AT },
            source: requestReceipt.source,
            request_receipt: requestReceipt,
            usage: {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                accounting_provenance: {
                    input_tokens: { method: 'reported' as const, accounting_basis: 'provider' },
                    output_tokens: { method: 'reported' as const, accounting_basis: 'provider' },
                    total_tokens: { method: 'derived' as const, accounting_basis: 'provider' },
                },
                reported_usage: [{ source: 'provider' as const, payload: { opaque: 'provider-payload' } }],
            },
        };
        const call = {
            ...toolCallBlock('block:call', 'call:1'),
            executor: 'application' as const,
            definition_id: definition.id,
        };
        const agent = {
            id: 'turn:agent',
            kind: 'agent' as const,
            authority: 'ordinary' as const,
            blocks: [textBlock('block:agent', 'working'), call],
            status: 'completed' as const,
            timestamps: { recorded_at: RECORDED_AT },
            generation_id: generation.id,
            provenance: { type: 'generated' as const },
            model_visibility: 'include' as const,
        };
        const agentBatch = {
            turns: [agent],
            generations: [generation],
            context_entries: [{ id: 'entry:agent', type: 'source_turn' as const, turn_id: agent.id }],
        };
        const agentOptions = {
            expected_revision: 1,
            operation_id: 'operation:agent',
            payload_fingerprint: 'sha256:agent-batch',
            recorded_at: RECORDED_AT,
        };
        const invalidAgentBatch = {
            ...agentBatch,
            generations: [{ ...generation, usage: { input_tokens: 10 } }],
        };
        expect(() => appendConversationRecords(materializedUser.document, invalidAgentBatch, agentOptions)).toThrow();
        await expect(
            stageIndexedRecordBatch(
                stagedUser.root,
                { conversation_id: source.id, batch: invalidAgentBatch, options: agentOptions },
                memory.store,
            ),
        ).rejects.toThrow(/ACCOUNTING_PROVENANCE_MISMATCH/);
        expect(
            await getPagedRecord(memory.store, stagedUser.root.directories.generations, generation.id),
        ).toBeUndefined();
        const materializedAgent = appendConversationRecords(materializedUser.document, agentBatch, agentOptions);
        const appendedMappingStore = memoryStore();
        const beforeMapping = await stageIndexedConversationSnapshot(
            materializedUser.document,
            undefined,
            appendedMappingStore.store,
        );
        const appendedHistoricalTurnId = 'turn:indexed-mapping-absent';
        const appendedHistoricalBlockId = 'block:indexed-mapping-absent';
        const mappedGeneration = {
            ...generation,
            request_receipt: {
                ...generation.request_receipt,
                item_mappings: [
                    { canonical_id: appendedHistoricalTurnId, native_id: 'native:old-turn', kind: 'turn' as const },
                    { canonical_id: appendedHistoricalBlockId, native_id: 'native:old-block', kind: 'block' as const },
                ],
            },
        };
        const appendedMapping = await stageIndexedRecordBatch(
            beforeMapping.root,
            {
                conversation_id: source.id,
                batch: { ...agentBatch, generations: [mappedGeneration] },
                options: agentOptions,
            },
            appendedMappingStore.store,
        );
        const afterMapping = await stageIndexedRecordBatch(
            appendedMapping.root,
            {
                conversation_id: source.id,
                batch: { turns: [userTurn('turn:mapping-unrelated', 'block:mapping-unrelated')] },
                options: {
                    expected_revision: appendedMapping.root.source.revision,
                    operation_id: 'operation:mapping-unrelated',
                    payload_fingerprint: 'sha256:mapping-unrelated',
                    recorded_at: RECORDED_AT,
                },
            },
            appendedMappingStore.store,
        );
        for (const id of [appendedHistoricalTurnId, appendedHistoricalBlockId]) {
            expect(
                await getPagedRecord(appendedMappingStore.store, afterMapping.root.directories.identifiers, id),
            ).toEqual({
                storage: 'marker',
                kind: 'historical_reference',
                id,
            });
        }
        for (const turn of [
            userTurn(appendedHistoricalTurnId, 'block:other'),
            userTurn('turn:other', appendedHistoricalBlockId),
        ]) {
            await expect(
                stageIndexedRecordBatch(
                    afterMapping.root,
                    {
                        conversation_id: source.id,
                        batch: { turns: [turn] },
                        options: {
                            expected_revision: afterMapping.root.source.revision,
                            operation_id: `operation:${turn.id}`,
                            payload_fingerprint: `sha256:${turn.id}`,
                            recorded_at: RECORDED_AT,
                        },
                    },
                    appendedMappingStore.store,
                ),
            ).rejects.toThrow('already exists');
        }
        const historicalTurnId = 'turn:historical-absent';
        const historicalBlockId = 'block:historical-absent';
        const historicalDocument = parseConversationDocument({
            ...materializedAgent.document,
            generations: {
                ...materializedAgent.document.generations,
                [generation.id]: {
                    ...materializedAgent.document.generations[generation.id],
                    request_receipt: {
                        ...generation.request_receipt,
                        item_mappings: [
                            { canonical_id: historicalTurnId, native_id: 'native:turn', kind: 'turn' },
                            { canonical_id: historicalBlockId, native_id: 'native:block', kind: 'block' },
                        ],
                    },
                },
            },
        });
        const historicalStore = memoryStore();
        const historical = await stageIndexedConversationSnapshot(historicalDocument, undefined, historicalStore.store);
        for (const id of [historicalTurnId, historicalBlockId]) {
            expect(await getPagedRecord(historicalStore.store, historical.root.directories.identifiers, id)).toEqual({
                storage: 'marker',
                kind: 'historical_reference',
                id,
            });
        }
        const unrelated = await stageIndexedRecordBatch(
            historical.root,
            {
                conversation_id: source.id,
                batch: { turns: [userTurn('turn:unrelated', 'block:unrelated')] },
                options: {
                    expected_revision: historical.root.source.revision,
                    operation_id: 'operation:unrelated',
                    payload_fingerprint: 'sha256:unrelated',
                    recorded_at: RECORDED_AT,
                },
            },
            historicalStore.store,
        );
        for (const turn of [userTurn(historicalTurnId, 'block:later'), userTurn('turn:later', historicalBlockId)]) {
            await expect(
                stageIndexedRecordBatch(
                    unrelated.root,
                    {
                        conversation_id: source.id,
                        batch: { turns: [turn] },
                        options: {
                            expected_revision: unrelated.root.source.revision,
                            operation_id: `operation:${turn.id}`,
                            payload_fingerprint: `sha256:${turn.id}`,
                            recorded_at: RECORDED_AT,
                        },
                    },
                    historicalStore.store,
                ),
            ).rejects.toThrow('already exists');
        }
        if (!unrelated.locator) throw new Error('Indexed unrelated append lacks its staged root');
        const removedUnrelated = await stageIndexedConversationDelete(
            unrelated.root,
            {
                operation_id: 'operation:delete-unrelated',
                source: unrelated.root.source,
                expected_source_root: unrelated.locator,
                recorded_at: RECORDED_AT,
                dependency_policy: 'reject',
                turn_ids: ['turn:unrelated'],
            },
            historicalStore.store,
        );
        expect(removedUnrelated.applied).toBe(true);
        expect(
            await getPagedRecord(
                historicalStore.store,
                removedUnrelated.root.directories.identifiers,
                historicalBlockId,
            ),
        ).toEqual({ storage: 'marker', kind: 'historical_reference', id: historicalBlockId });
        const stagedAgent = await stageIndexedRecordBatch(
            stagedUser.root,
            { conversation_id: source.id, batch: agentBatch, options: agentOptions },
            memory.store,
        );
        expect(stagedAgent.receipt).toEqual(materializedAgent.document.operation_receipts[agentOptions.operation_id]);
        const generationRef = await getPagedRecord(
            memory.store,
            stagedAgent.root.directories.generations,
            generation.id,
        );
        if (generationRef?.storage !== 'record') throw new Error('Indexed generation record was not retained');
        const generationBytes = memory.records.get(`generations:${generationRef.content_hash}`);
        if (!generationBytes) throw new Error('Indexed generation bytes were not retained');
        expect(GenerationSchema.parse(JSON.parse(new TextDecoder().decode(generationBytes))).usage).toEqual(
            generation.usage,
        );
        expect(stagedAgent.root.accepted_response).toMatchObject({ generation_id: generation.id, turn_id: agent.id });
        expect(
            await getPagedRecord(memory.store, stagedAgent.root.directories.open_tool_calls, call.call_id),
        ).toMatchObject({ storage: 'record', kind: 'open_tool_calls', id: call.call_id });
        const agentTimestampRetry = {
            ...agentBatch,
            turns: [{ ...agent, timestamps: { recorded_at: retryAt } }],
            generations: [
                {
                    ...generation,
                    timestamps: { recorded_at: retryAt },
                    request_receipt: { ...requestReceipt, recorded_at: retryAt },
                },
            ],
        };
        expect(
            appendConversationRecords(materializedAgent.document, agentTimestampRetry, {
                ...agentOptions,
                recorded_at: retryAt,
            }).applied,
        ).toBe(false);
        expect(
            (
                await stageIndexedRecordBatch(
                    stagedAgent.root,
                    {
                        conversation_id: source.id,
                        batch: agentTimestampRetry,
                        options: { ...agentOptions, recorded_at: retryAt },
                    },
                    memory.store,
                )
            ).applied,
        ).toBe(false);
        const result = { ...toolResultTurn('turn:tool', call.call_id), execution_id: 'execution:1' };
        const receipt = {
            id: 'execution:1',
            call_id: call.call_id,
            executor: call.executor,
            status: 'success' as const,
            result_turn_id: result.id,
            result_fingerprint: await fingerprintJson(result.blocks[0]),
            recorded_at: RECORDED_AT,
            call_source: {
                conversation: { conversation_id: source.id, revision: 2 },
                turn_id: agent.id,
                block_id: call.id,
                call_id: call.call_id,
                call_fingerprint: (await hashContentBytes(canonicalJsonContentBytes(call))).content_hash,
            },
        };
        const mismatchedReceipt = { ...receipt, result_fingerprint: 'sha256:not-the-result' };
        await expect(
            validateToolExecutionResult(materializedAgent.document, {
                source: mismatchedReceipt.call_source,
                turn: result,
                execution_receipt: mismatchedReceipt,
            }),
        ).rejects.toThrow('result fingerprint does not match');
        const recordCountBeforeInvalidResult = memory.records.size;
        await expect(
            stageIndexedRecordBatch(
                stagedAgent.root,
                {
                    conversation_id: source.id,
                    batch: {
                        turns: [result],
                        execution_receipts: [mismatchedReceipt],
                        context_entries: [{ id: 'entry:tool', type: 'source_turn' as const, turn_id: result.id }],
                    },
                    options: {
                        expected_revision: 2,
                        operation_id: 'operation:tool',
                        payload_fingerprint: 'sha256:tool-batch',
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('result fingerprint does not match');
        expect(memory.records.size).toBe(recordCountBeforeInvalidResult);
        expect(stagedAgent.root.source.revision).toBe(2);
        expect(
            await getPagedRecord(memory.store, stagedAgent.root.directories.operation_receipts, 'operation:tool'),
        ).toBeUndefined();
        await expect(
            validateToolExecutionResult(materializedAgent.document, {
                source: receipt.call_source,
                turn: result,
                execution_receipt: receipt,
            }),
        ).resolves.toBeDefined();
        const resultBatch = {
            turns: [result],
            execution_receipts: [receipt],
            context_entries: [{ id: 'entry:tool', type: 'source_turn' as const, turn_id: result.id }],
        };
        const resultOptions = {
            expected_revision: 2,
            operation_id: 'operation:tool',
            payload_fingerprint: 'sha256:tool-batch',
            recorded_at: RECORDED_AT,
        };
        const materializedResult = appendConversationRecords(materializedAgent.document, resultBatch, resultOptions);
        const stagedResult = await stageIndexedRecordBatch(
            stagedAgent.root,
            { conversation_id: source.id, batch: resultBatch, options: resultOptions },
            memory.store,
        );
        expect(stagedResult.receipt).toEqual(
            materializedResult.document.operation_receipts[resultOptions.operation_id],
        );
        // Complete restart profiles retain only active nominations; the independent terminal
        // call/result index remains authoritative after this pending nomination is removed.
        expect(stagedResult.root.restart_index_profile).toBe(stagedAgent.root.restart_index_profile);
        expect(stagedResult.root.restart_index_profile).toBeDefined();
        expect(
            await getPagedRecord(memory.store, stagedResult.root.directories.open_tool_calls, call.call_id),
        ).toBeUndefined();
        expect(await loadIndexedToolCallTerminalResult(memory.store, stagedResult.root, receipt.call_source)).toBe(
            true,
        );
        const repeatBatch: IndexedRecordBatchCommand['batch'] = {
            turns: [
                {
                    ...agent,
                    id: 'turn:repeat-call',
                    generation_id: 'generation:repeat-call',
                    blocks: [{ ...call, id: 'block:repeat-call' }],
                },
            ],
            generations: [
                {
                    ...generation,
                    id: 'generation:repeat-call',
                    request_id: 'request:repeat-call',
                    source: stagedResult.root.source,
                    request_receipt: {
                        ...requestReceipt,
                        id: 'receipt:repeat-call',
                        request_id: 'request:repeat-call',
                        source: stagedResult.root.source,
                    },
                },
            ],
        };
        const recordsBeforeRepeat = memory.records.size;
        await expect(
            stageIndexedRecordBatch(
                stagedResult.root,
                {
                    conversation_id: source.id,
                    batch: repeatBatch,
                    options: {
                        expected_revision: stagedResult.root.source.revision,
                        operation_id: 'operation:repeat-call',
                        payload_fingerprint: await fingerprintJson(repeatBatch),
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow(`Indexed append identity ${call.call_id} already exists`);
        expect(memory.records.size).toBe(recordsBeforeRepeat);
        // Older roots still preserve their historical closed-marker representation.
        const { restart_index_profile: _restartProfile, ...oldMarkerProfile } = stagedAgent.root;
        const oldResult = await stageIndexedRecordBatch(
            oldMarkerProfile,
            {
                conversation_id: source.id,
                batch: resultBatch,
                options: resultOptions,
            },
            memory.store,
        );
        expect(
            await getPagedRecord(memory.store, oldResult.root.directories.open_tool_calls, call.call_id),
        ).toMatchObject({ storage: 'marker', kind: 'closed_tool_call', id: call.call_id });
        expect(await loadIndexedToolCallTerminalResult(memory.store, oldResult.root, receipt.call_source)).toBe(true);
        const resultTimestampRetry = {
            ...resultBatch,
            turns: [{ ...result, timestamps: { recorded_at: retryAt } }],
            execution_receipts: [{ ...receipt, recorded_at: retryAt }],
        };
        expect(
            appendConversationRecords(materializedResult.document, resultTimestampRetry, {
                ...resultOptions,
                recorded_at: retryAt,
            }).applied,
        ).toBe(false);
        expect(
            (
                await stageIndexedRecordBatch(
                    stagedResult.root,
                    {
                        conversation_id: source.id,
                        batch: resultTimestampRetry,
                        options: { ...resultOptions, recorded_at: retryAt },
                    },
                    memory.store,
                )
            ).applied,
        ).toBe(false);
        await expect(
            stageIndexedRecordBatch(
                stagedResult.root,
                {
                    conversation_id: source.id,
                    batch: { turns: [toolResultTurn('turn:second-result', call.call_id)] },
                    options: { ...resultOptions, operation_id: 'operation:second-result', expected_revision: 3 },
                },
                memory.store,
            ),
        ).rejects.toThrow('no open retained call');
        await expect(
            stageIndexedRecordBatch(
                stagedResult.root,
                {
                    conversation_id: source.id,
                    batch: { execution_receipts: [{ ...receipt, id: 'execution:duplicate' }] },
                    options: { ...resultOptions, operation_id: 'operation:duplicate-receipt', expected_revision: 3 },
                },
                memory.store,
            ),
        ).rejects.toThrow('no unterminated call');
        await expect(
            stageIndexedRecordBatch(
                stagedResult.root,
                {
                    conversation_id: source.id,
                    batch: resultBatch,
                    options: { ...resultOptions, operation_id: 'operation:duplicate', expected_revision: 3 },
                },
                memory.store,
            ),
        ).rejects.toThrow();
    });

    it('omits empty fresh directories and preserves retained refs on assets-only appends', async () => {
        const source = emptyDocument('conversation:empty-index-groups');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const asset: Asset = {
            id: 'asset:empty-index-groups',
            kind: 'image',
            mime_type: 'image/png',
            storage: { type: 'inline_base64', data: 'AQID' },
            provenance: { type: 'received' },
            created_at: RECORDED_AT,
        };
        const batch = { assets: [asset] };
        const accepted = await stageIndexedRecordBatch(
            initial.root,
            {
                conversation_id: source.id,
                batch,
                options: {
                    operation_id: 'operation:empty-index-groups',
                    expected_revision: 0,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            memory.store,
        );
        for (const family of [
            'turn_order',
            'active_context_order',
            'deletion_blockers',
            'turn_acceptances',
            'block_owners',
            'turn_links',
        ] as const) {
            expect(Object.hasOwn(initial.root.directories, family)).toBe(false);
            expect(Object.hasOwn(accepted.root.directories, family)).toBe(false);
        }
        expect(preflightJsonInput(accepted.root).success).toBe(true);
        const turnBatch = { turns: [userTurn('turn:empty-index-groups', 'block:empty-index-groups')] };
        const withTurn = await stageIndexedRecordBatch(
            accepted.root,
            {
                conversation_id: source.id,
                batch: turnBatch,
                options: {
                    operation_id: 'operation:empty-index-groups:turn',
                    expected_revision: 1,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(turnBatch),
                },
            },
            memory.store,
        );
        const laterBatch = { assets: [{ ...asset, id: 'asset:empty-index-groups:later' }] };
        const later = await stageIndexedRecordBatch(
            withTurn.root,
            {
                conversation_id: source.id,
                batch: laterBatch,
                options: {
                    operation_id: 'operation:empty-index-groups:later',
                    expected_revision: 2,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(laterBatch),
                },
            },
            memory.store,
        );
        for (const family of ['turn_order', 'turn_acceptances', 'block_owners', 'turn_links'] as const) {
            expect(withTurn.root.directories[family]).toBeDefined();
            expect(later.root.directories[family]).toEqual(withTurn.root.directories[family]);
        }
        expect(preflightJsonInput(later.root).success).toBe(true);
    });

    it('shares retained identity ancestors during a fresh 1,000-turn append and rechecks corrupted storage', async () => {
        const memory = memoryStore();
        const source = emptyDocument('conversation:bulk-identities');
        source.turns = Array.from({ length: 1024 }, (_, index) => userTurn(`cold:${index}`, `cold-block:${index}`));
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const identities = initial.root.directories.identifiers;
        if (!identities) throw new Error('Expected real existing identity index');
        const batch = {
            turns: Array.from({ length: 1000 }, (_, index) => userTurn(`new:${index}`, `new-block:${index}`)),
        };
        const command = {
            conversation_id: source.id,
            batch,
            options: {
                operation_id: 'operation:bulk-identities',
                expected_revision: source.revision,
                recorded_at: RECORDED_AT,
                payload_fingerprint: await fingerprintJson(batch),
            },
        };
        memory.pageReads.length = 0;
        const accepted = await stageIndexedRecordBatch(initial.root, command, memory.store);
        expect(accepted.root.turn_count).toBe(2024);
        // One bulk lookup + the existing immutable batch-insert/readback path; no per-ID ancestor decode.
        expect(memory.pageReads.filter((hash) => hash === identities.content_hash).length).toBeLessThanOrEqual(4);
        expect(await getPagedRecord(memory.store, accepted.root.directories.identifiers, 'new:999')).toEqual({
            storage: 'marker',
            kind: 'turn',
            id: 'new:999',
        });
        const original = memory.pages.get(identities.content_hash);
        if (!original) throw new Error('Expected retained identity bytes');
        memory.pages.set(identities.content_hash, Uint8Array.from([...original.slice(0, -1), 0]));
        await expect(stageIndexedRecordBatch(initial.root, command, memory.store)).rejects.toThrow('hash differs');
        expect(initial.root.turn_count).toBe(1024);
    });

    it('groups a genuine 1,000-turn fresh append while retaining collision and original historical witnesses', async () => {
        const source = emptyDocument('conversation:grouped-native-append');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const batch = {
            turns: Array.from({ length: 1000 }, (_, index) =>
                userTurn(`turn:grouped:${index}`, `block:grouped:${index}`),
            ),
        };
        const options = {
            operation_id: 'operation:grouped',
            expected_revision: 0,
            recorded_at: RECORDED_AT,
            payload_fingerprint: await fingerprintJson(batch),
        };
        const pagesBefore = memory.pages.size;
        const accepted = await stageIndexedRecordBatch(
            initial.root,
            { conversation_id: source.id, batch, options },
            memory.store,
        );
        expect(accepted.root.turn_count).toBe(1000);
        expect(Object.hasOwn(accepted.root.directories, 'deletion_blockers')).toBe(false);
        expect(Object.hasOwn(accepted.root.directories, 'active_context_order')).toBe(false);
        expect(preflightJsonInput(accepted.root).success).toBe(true);
        expect(memory.pages.size - pagesBefore).toBeLessThan(256);
        expect(
            (await loadIndexedProjectedTurn(memory.store, accepted.root, 'turn:grouped:999')).selected_blocks,
        ).toEqual(batch.turns[999].blocks);
        expect(initial.root.turn_count).toBe(0);
        const retry = await stageIndexedRecordBatch(
            accepted.root,
            { conversation_id: source.id, batch, options },
            memory.store,
        );
        expect(retry.applied).toBe(false);
        expect(retry.root).toEqual(accepted.root);
        const collided = { turns: [userTurn('turn:grouped:0', 'block:foreign')] };
        await expect(
            stageIndexedRecordBatch(
                accepted.root,
                {
                    conversation_id: source.id,
                    batch: collided,
                    options: {
                        ...options,
                        operation_id: 'operation:collision',
                        expected_revision: 1,
                        payload_fingerprint: await fingerprintJson(collided),
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('already exists');
        const duplicate = {
            turns: [userTurn('turn:duplicate', 'block:duplicate'), userTurn('turn:duplicate', 'block:other')],
        };
        await expect(
            stageIndexedRecordBatch(
                accepted.root,
                {
                    conversation_id: source.id,
                    batch: duplicate,
                    options: {
                        ...options,
                        operation_id: 'operation:duplicate',
                        expected_revision: 1,
                        payload_fingerprint: await fingerprintJson(duplicate),
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow();
        expect((await loadIndexedProjectedTurn(memory.store, accepted.root, 'turn:grouped:0')).selected_blocks).toEqual(
            batch.turns[0].blocks,
        );
        expect(accepted.root.turn_count).toBe(1000);
    });

    it('rejects duplicate identities, missing media and dangling tool results before publishing an indexed root', async () => {
        const source = emptyDocument('conversation:indexed-negative');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const options = {
            expected_revision: 0,
            operation_id: 'operation:negative',
            payload_fingerprint: 'sha256:negative',
            recorded_at: RECORDED_AT,
        };
        const cases = [
            {
                turns: [
                    {
                        ...userTurn('turn:first', 'block:duplicate'),
                        blocks: [textBlock('block:duplicate'), textBlock('block:duplicate')],
                    },
                ],
            },
            {
                turns: [
                    {
                        ...userTurn('turn:media', 'block:media'),
                        blocks: [{ id: 'block:media', type: 'image' as const, asset_id: 'missing' }],
                    },
                ],
            },
            { turns: [toolResultTurn('turn:tool', 'call:missing')] },
            {
                turns: [userTurn('turn:overlap', 'block:overlap')],
                context_entries: [
                    { id: 'entry:one', type: 'source_turn' as const, turn_id: 'turn:overlap' },
                    { id: 'entry:two', type: 'source_turn' as const, turn_id: 'turn:overlap' },
                ],
            },
        ];
        for (const batch of cases) {
            expect(() => appendConversationRecords(source, batch, options)).toThrow();
            await expect(
                stageIndexedRecordBatch(
                    initial.root,
                    {
                        conversation_id: source.id,
                        batch,
                        options,
                    },
                    memory.store,
                ),
            ).rejects.toThrow();
        }
    });

    it('recovers an old accepted indexed batch after a later head without restoring its active tool selection', async () => {
        const source = emptyDocument('conversation:indexed-historical');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const tool = { id: 'definition:old', name: 'lookup', version: '1', input_schema: { type: 'object' } };
        const firstBatch = { tool_definitions: [tool], active_tool_definition_ids: [tool.id] };
        const firstOptions = {
            expected_revision: 0,
            operation_id: 'operation:first',
            payload_fingerprint: 'sha256:first',
            recorded_at: RECORDED_AT,
        };
        const first = await stageIndexedRecordBatch(
            initial.root,
            { conversation_id: source.id, batch: firstBatch, options: firstOptions },
            memory.store,
        );
        const second = await stageIndexedRecordBatch(
            first.root,
            {
                conversation_id: source.id,
                batch: { active_tool_definition_ids: [] },
                options: {
                    expected_revision: 1,
                    operation_id: 'operation:second',
                    payload_fingerprint: 'sha256:second',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        const retry = await stageIndexedRecordBatch(
            second.root,
            { conversation_id: source.id, batch: firstBatch, options: firstOptions },
            memory.store,
        );
        expect(retry).toMatchObject({ applied: false, receipt: first.receipt });
        expect((await loadIndexedActiveContext(memory.store, retry.root)).active_tool_definition_ids).toEqual([]);
        await expect(
            stageIndexedRecordBatch(
                second.root,
                {
                    conversation_id: source.id,
                    batch: { ...firstBatch, active_tool_definition_ids: [] },
                    options: firstOptions,
                },
                memory.store,
            ),
        ).rejects.toThrow('accepted operation');
    });

    // One-time cold migration validates and reads back every immutable record. Keep both sizes
    // here so the bounded append/delete read-count comparison exercises the same store profile.
    it('migrates valid 10k and 100k cold turns, then appends with fixed active context and bounded reads', async () => {
        const pageReadCounts: number[] = [];
        const deleteReadCounts: number[] = [];
        const dependencyReadCounts: number[] = [];
        for (const coldCount of [10_000, 100_000]) {
            const source = emptyDocument(`conversation:indexed-scale:${coldCount}`);
            const template = userTurn('turn:template', 'block:template');
            const turns = Array.from({ length: coldCount }, (_, index) => ({
                ...template,
                id: `turn:cold:${index}`,
                blocks: [{ ...template.blocks[0], id: `block:cold:${index}` }],
            }));
            const document = {
                ...source,
                turns,
                context: {
                    ...source.context,
                    entries: [{ id: 'entry:active', type: 'source_turn' as const, turn_id: turns.at(-1)?.id ?? '' }],
                },
            };
            expect(preflightJsonInput(document).success).toBe(coldCount === 10_000);
            const memory = memoryStore();
            const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
            expect(initial.root.turn_count).toBe(coldCount);
            expect((await loadIndexedActiveContext(memory.store, initial.root)).entries).toEqual(
                document.context.entries,
            );
            if (coldCount === 100_000) {
                const retainedRecordCount = memory.records.size;
                await expect(
                    stageIndexedConversationSnapshot(
                        { ...document, turns: [...turns, userTurn('turn:over-limit')] },
                        undefined,
                        memory.store,
                    ),
                ).rejects.toMatchObject({ diagnostics: [{ code: 'JSON_MAX_ARRAY_LENGTH' }] });
                expect(memory.records.size).toBe(retainedRecordCount);
            }
            memory.pageReads.length = 0;
            memory.recordReads.length = 0;
            const staged = await stageIndexedRecordBatch(
                initial.root,
                {
                    conversation_id: source.id,
                    batch: { turns: [userTurn('turn:new', 'block:new')] },
                    options: {
                        expected_revision: 0,
                        operation_id: 'operation:new',
                        payload_fingerprint: 'sha256:new',
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            );
            expect(staged.applied).toBe(true);
            expect(staged.root.turn_count).toBe(coldCount + 1);
            // The complete delete index adds three copy-on-write lookup paths, still independent of cold history.
            expect(memory.pageReads.length).toBeLessThan(128);
            expect(memory.recordReads.length).toBeLessThan(16);
            pageReadCounts.push(memory.pageReads.length);
            const later = await stageIndexedRecordBatch(
                staged.root,
                {
                    conversation_id: source.id,
                    batch: { turns: [userTurn('turn:after', 'block:after')] },
                    options: {
                        expected_revision: 1,
                        operation_id: 'operation:after',
                        payload_fingerprint: 'sha256:after',
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            );
            if (!later.locator) throw new Error('Indexed later append lacks its staged root');
            memory.pageReads.length = 0;
            memory.recordReads.length = 0;
            const deleted = await stageIndexedConversationDelete(
                later.root,
                {
                    operation_id: 'operation:delete',
                    source: later.root.source,
                    expected_source_root: later.locator,
                    recorded_at: RECORDED_AT,
                    dependency_policy: 'reject',
                    turn_ids: ['turn:new'],
                },
                memory.store,
            );
            expect(deleted.applied).toBe(true);
            expect(deleted.root.live_turn_count).toBe(coldCount + 1);
            expect(deleted.root.active_tail_turn_id).toBe('turn:after');
            expect(memory.pageReads.length).toBeLessThan(256);
            expect(memory.recordReads.length).toBeLessThan(24);
            deleteReadCounts.push(memory.pageReads.length);
            const commands = await toolMediaCommands(deleted.root.source);
            const called = await stageIndexedRecordBatch(deleted.root, commands.agentCommand, memory.store);
            const completed = await stageIndexedRecordBatch(called.root, commands.resultCommand, memory.store);
            if (!completed.locator) throw new Error('Indexed tool/media append lacks its immutable root');
            memory.pageReads.length = 0;
            memory.recordReads.length = 0;
            const selected = await loadIndexedSelectedDependencyContext(
                memory.store,
                completed.root,
                completed.locator,
            );
            expect(selected.completeness).toBe('selected_dependencies_pending_admission');
            expect(selected.assets[commands.asset.id]).toEqual(commands.asset);
            expect(selected.execution_witnesses?.[commands.execution.id]).toEqual(commands.execution);
            expect(selected.generation_witnesses[commands.agent.generation_id]?.generation.record_source).toBe(
                'executed',
            );
            expect(memory.recordReads.filter((id) => id.startsWith('turns:turn:cold:'))).toEqual([
                `turns:turn:cold:${coldCount - 1}`,
            ]);
            expect(memory.recordReads.length).toBeLessThan(32);
            expect(memory.pageReads.length).toBeLessThan(512);
            dependencyReadCounts.push(memory.recordReads.length);
        }
        expect(pageReadCounts[1]).toBeLessThan(pageReadCounts[0] + 20);
        expect(dependencyReadCounts[0]).toBe(dependencyReadCounts[1]);
        expect(deleteReadCounts[1]).toBeLessThan(deleteReadCounts[0] + 40);
    }, 120_000);
    it('resolves one selected small block without reading a large unselected body', async () => {
        const initial = emptyDocument('conversation:indexed');
        const turn = createUserTurn({
            id: 'turn:user',
            authority: 'ordinary',
            blocks: [textBlock('block:small', 'selected'), textBlock('block:large', 'x'.repeat(4 * 1024 * 1024))],
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT },
            provenance: { type: 'received' },
            model_visibility: 'include',
        });
        const accepted = appendConversationRecords(
            initial,
            {
                turns: [turn],
                context_entries: [
                    { id: 'entry:small', type: 'source_turn', turn_id: turn.id, block_ids: ['block:small'] },
                ],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:user',
                payload_fingerprint: 'sha256:accepted-user-input',
                recorded_at: RECORDED_AT,
            },
        );
        const memory = memoryStore();
        const staged = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        expect(staged.root.source.revision).toBe(1);
        memory.recordReads.length = 0;
        const context = await loadIndexedActiveContext(memory.store, staged.root);
        expect(context.entries).toEqual(accepted.document.context.entries);
        const projection = await loadIndexedProjectedTurn(memory.store, staged.root, turn.id, ['block:small']);
        expect(projection).toMatchObject({
            completeness: 'selected_blocks',
            selected_block_positions: [0],
            source_block_count: 2,
            selected_blocks: [{ id: 'block:small', text: 'selected' }],
        });
        expect(memory.recordReads).not.toContain('blocks:block:large');
        expect(memory.records.size).toBeGreaterThan(4);
    });

    it('refuses a later selected body from its descriptor before reading past the shared byte budget', async () => {
        const source = emptyDocument('conversation:indexed-budget');
        const turn = createUserTurn({
            id: 'turn:large-selected',
            authority: 'ordinary',
            blocks: [textBlock('block:first', 'a'.repeat(700_000)), textBlock('block:second', 'b'.repeat(700_000))],
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT },
            provenance: { type: 'received' },
            model_visibility: 'include',
        });
        const accepted = appendConversationRecords(
            source,
            { turns: [turn], context_entries: [{ id: 'entry:selected', type: 'source_turn', turn_id: turn.id }] },
            {
                expected_revision: source.revision,
                operation_id: 'operation:selected',
                payload_fingerprint: 'sha256:selected',
                recorded_at: RECORDED_AT,
            },
        );
        const memory = memoryStore();
        const staged = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        memory.recordReads.length = 0;
        await expect(
            loadIndexedSelectedTextContext(memory.store, staged.root, staged.locator, 900_000),
        ).rejects.toThrow('before record read');
        expect(memory.recordReads).toContain('blocks:block:first');
        expect(memory.recordReads).not.toContain('blocks:block:second');
    });

    it('permits read-only settled retrieval and preparation for disabled processing without inventing readiness', async () => {
        const memory = memoryStore();
        const document = emptyDocument('conversation:disabled-retrieval');
        const staged = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const count = memory.records.size;
        const selected = await loadIndexedSettledRetrievalSelectedContext(memory.store, staged.root, staged.locator);
        expect(selected.source).toEqual(staged.root.source);
        expect(selected.completeness).toBe('selected_media_compaction_pending_admission');
        expect(selected.turns).toEqual([]);
        expect(selected.execution_witnesses).toEqual({});
        const preparation = await loadIndexedSettledProcessingSelectedContext(
            memory.store,
            staged.root,
            staged.locator,
        );
        expect(preparation).toEqual(selected);
        expect(preparation.completeness).toBe('selected_media_compaction_pending_admission');
        await expect(
            loadIndexedReadySelectedContext(memory.store, staged.root, staged.locator, {
                target_fingerprint: `sha256:${'a'.repeat(64)}`,
                measured_input_tokens: 1,
                tokenizer_id: 'exact:test',
                measurement_fingerprint: `sha256:${'b'.repeat(64)}`,
            }),
        ).rejects.toThrow('exact context/target/count is unavailable');
        expect(memory.records.size).toBe(count);
    });

    it('preserves materialized job-drain obligations across genuine policy changes and rejects disabled-header tampering', async () => {
        const initial = emptyDocument('conversation:indexed-processing');
        const enabled = await setProcessingPolicy(initial, {
            operation_id: 'policy:enable',
            expected_revision: initial.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                {
                    id: 'externalize-text',
                    version: '1',
                    config: {},
                    scope: 'on_append',
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const turn = userTurn('turn:processing');
        const accepted = await appendConversationRecordsWithProcessing(
            enabled.document,
            { turns: [turn], context_entries: [{ id: 'entry:processing', type: 'source_turn', turn_id: turn.id }] },
            {
                expected_revision: enabled.document.revision,
                operation_id: 'operation:processing-input',
                payload_fingerprint: 'sha256:processing-input',
                recorded_at: RECORDED_AT,
            },
        );
        const job = Object.values(accepted.document.processing.jobs ?? {})[0];
        if (!job) throw new Error('Accepted processing append did not enqueue its job');
        const disabledWithoutSupersession = (source: ConversationDocument) => {
            const altered = structuredClone(source);
            altered.processing.enabled = false;
            return parseConversationDocument(altered);
        };
        const retainWithoutAutomaticProcessing = async (source: ConversationDocument) =>
            (
                await setProcessingPolicy(source, {
                    operation_id: 'policy:manual-only',
                    expected_revision: source.revision,
                    recorded_at: RECORDED_AT,
                    enabled: true,
                    processors: source.processing.processors.map((processor) => ({
                        ...processor,
                        scope: 'manual' as const,
                    })),
                })
            ).document;
        const assertIndexedBlocked = async (source: ConversationDocument, expectedCount: number) => {
            await expect(assertProcessingReady(source, '', '')).rejects.toThrow(
                'Current context, policy and target need processing evaluation',
            );
            const memory = memoryStore();
            const staged = await stageIndexedConversationSnapshot(source, undefined, memory.store);
            const headerBytes = memory.records.get(`processing_header:${staged.root.processing_header.content_hash}`);
            if (!headerBytes) throw new Error('Indexed processing header was not retained');
            expect(JSON.parse(new TextDecoder().decode(headerBytes)).unresolved_job_count).toBe(expectedCount);
            const recordCount = memory.records.size;
            await expect(loadIndexedSelectedTextContext(memory.store, staged.root, staged.locator)).rejects.toThrow(
                'Indexed selected preparation requires processing readiness',
            );
            await expect(
                loadIndexedSettledRetrievalSelectedContext(memory.store, staged.root, staged.locator),
            ).rejects.toThrow('unresolved processing obligations');
            expect(memory.records.size).toBe(recordCount);
            return { memory, staged };
        };
        // The real materialized transition cannot disable outstanding jobs without supersession.
        await expect(
            setProcessingPolicy(accepted.document, {
                operation_id: 'policy:invalid-disable',
                expected_revision: accepted.document.revision,
                recorded_at: RECORDED_AT,
                enabled: false,
                processors: [],
            }),
        ).rejects.toThrow('Disabling processing requires explicit pending-job supersession');
        const forgedDisabled = disabledWithoutSupersession(accepted.document);
        await expect(assertProcessingReady(forgedDisabled, '', '')).rejects.toThrow(
            'Accepted processing jobs remain outstanding',
        );
        const rejected = memoryStore();
        await expect(stageIndexedConversationSnapshot(forgedDisabled, undefined, rejected.store)).rejects.toThrow(
            'current policy differs from its retained accepted command',
        );
        expect(rejected.pages.size).toBe(0);
        expect(rejected.records.size).toBe(0);
        const pending = await retainWithoutAutomaticProcessing(accepted.document);
        expect(pending.processing.jobs?.[job.id]).toEqual(job);
        const { memory, staged } = await assertIndexedBlocked(pending, 1);
        const headerKey = `processing_header:${staged.root.processing_header.content_hash}`;
        const currentHeaderBytes = memory.records.get(headerKey);
        if (!currentHeaderBytes) throw new Error('Missing current processing header');
        const legacyHeader = JSON.parse(new TextDecoder().decode(currentHeaderBytes));
        delete legacyHeader.unresolved_job_count;
        const legacyBytes = canonicalJsonContentBytes(legacyHeader);
        const legacyHash = (await hashContentBytes(legacyBytes)).content_hash;
        memory.records.set(`processing_header:${legacyHash}`, legacyBytes);
        const legacyRoot = {
            ...staged.root,
            processing_header: { content_hash: legacyHash, size_bytes: legacyBytes.byteLength },
        };
        const legacyRootBytes = canonicalJsonContentBytes(legacyRoot);
        const legacyRootHash = (await hashContentBytes(legacyRootBytes)).content_hash;
        memory.records.set(`root:${legacyRootHash}`, legacyRootBytes);
        const legacyLocator = { content_hash: legacyRootHash, size_bytes: legacyRootBytes.byteLength };
        await expect(loadIndexedSelectedTextContext(memory.store, legacyRoot, legacyLocator)).rejects.toThrow(
            'Indexed selected preparation requires processing readiness',
        );

        const ordinary = emptyDocument('conversation:indexed-legacy-no-processing');
        const ordinaryMemory = memoryStore();
        const ordinaryStaged = await stageIndexedConversationSnapshot(ordinary, undefined, ordinaryMemory.store);
        const ordinaryHeaderKey = `processing_header:${ordinaryStaged.root.processing_header.content_hash}`;
        const ordinaryHeaderBytes = ordinaryMemory.records.get(ordinaryHeaderKey);
        if (!ordinaryHeaderBytes) throw new Error('Missing ordinary processing header');
        const ordinaryLegacyHeader = JSON.parse(new TextDecoder().decode(ordinaryHeaderBytes));
        delete ordinaryLegacyHeader.unresolved_job_count;
        const ordinaryLegacyBytes = canonicalJsonContentBytes(ordinaryLegacyHeader);
        const ordinaryLegacyHash = (await hashContentBytes(ordinaryLegacyBytes)).content_hash;
        ordinaryMemory.records.set(`processing_header:${ordinaryLegacyHash}`, ordinaryLegacyBytes);
        const ordinaryLegacyRoot = {
            ...ordinaryStaged.root,
            processing_header: { content_hash: ordinaryLegacyHash, size_bytes: ordinaryLegacyBytes.byteLength },
        };
        const ordinaryLegacyRootBytes = canonicalJsonContentBytes(ordinaryLegacyRoot);
        const ordinaryLegacyRootHash = (await hashContentBytes(ordinaryLegacyRootBytes)).content_hash;
        ordinaryMemory.records.set(`root:${ordinaryLegacyRootHash}`, ordinaryLegacyRootBytes);
        await expect(
            loadIndexedSelectedTextContext(ordinaryMemory.store, ordinaryLegacyRoot, {
                content_hash: ordinaryLegacyRootHash,
                size_bytes: ordinaryLegacyRootBytes.byteLength,
            }),
        ).resolves.toMatchObject({ turns: [] });

        let current = accepted.document;
        const store = {
            async load() {
                return structuredClone(current);
            },
            async commit(expectedRevision: number, next: ConversationDocument) {
                if (current.revision !== expectedRevision) return false;
                current = parseConversationDocument(next);
                return true;
            },
        };
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
            'attempt:blocked',
            () => RECORDED_AT,
        );
        expect(current.processing.completions?.[job.id]?.status).toBe('blocked');
        const blocked = await retainWithoutAutomaticProcessing(current);
        expect(blocked.processing.jobs?.[job.id]).toEqual(job);
        await expect(assertProcessingReady(blocked, '', '')).rejects.toThrow(
            'Current context, policy and target need processing evaluation',
        );
        const blockedMemory = memoryStore();
        const blockedBefore = structuredClone(blocked);
        await expect(stageIndexedConversationSnapshot(blocked, undefined, blockedMemory.store)).rejects.toThrow(
            'unresolved materialized phases to drain or be superseded',
        );
        expect(blockedMemory.pages.size).toBe(0);
        expect(blockedMemory.records.size).toBe(0);
        expect(blocked).toEqual(blockedBefore);

        const superseded = await setProcessingPolicy(accepted.document, {
            operation_id: 'policy:supersede',
            expected_revision: accepted.document.revision,
            recorded_at: RECORDED_AT,
            enabled: false,
            processors: [],
            supersede_job_ids: [job.id],
            supersession_reason: 'operator_cancelled',
        });
        await expect(assertProcessingReady(superseded.document, '', '')).resolves.toBeUndefined();
        const supersededMemory = memoryStore();
        const supersededRoot = await stageIndexedConversationSnapshot(
            superseded.document,
            undefined,
            supersededMemory.store,
        );
        await expect(
            loadIndexedSelectedTextContext(supersededMemory.store, supersededRoot.root, supersededRoot.locator),
        ).resolves.toMatchObject({ turns: [{ header: { id: turn.id } }] });

        current = accepted.document;
        await runProcessingJob(
            store,
            { resolve: () => ({ run: async () => ({ kind: 'no_op' as const, reason: 'done' }) }) },
            job.id,
            'attempt:completed',
            () => RECORDED_AT,
        );
        expect(current.processing.completions?.[job.id]?.status).toBe('no_op');
        const completed = await setProcessingPolicy(current, {
            operation_id: 'policy:complete',
            expected_revision: current.revision,
            recorded_at: RECORDED_AT,
            enabled: false,
            processors: [],
        });
        await expect(assertProcessingReady(completed.document, '', '')).resolves.toBeUndefined();
        const completedMemory = memoryStore();
        const completedRoot = await stageIndexedConversationSnapshot(
            completed.document,
            undefined,
            completedMemory.store,
        );
        await expect(
            loadIndexedSelectedTextContext(completedMemory.store, completedRoot.root, completedRoot.locator),
        ).resolves.toMatchObject({ turns: [{ header: { id: turn.id } }] });
    });

    it('appends one program result with exact durable receipt and no second write on retry', async () => {
        const source = emptyDocument('conversation:indexed-program');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const turn = {
            id: 'turn:program',
            kind: 'program' as const,
            authority: 'ordinary' as const,
            status: 'completed' as const,
            timestamps: { recorded_at: RECORDED_AT },
            provenance: { type: 'inserted' as const, operation_id: 'operation:program' },
            model_visibility: 'include' as const,
            blocks: [{ id: 'block:program', type: 'text' as const, text: 'Continue.', format: 'plain' as const }],
        };
        const entry = { id: 'entry:program', type: 'source_turn' as const, turn_id: turn.id };
        const command = {
            conversation_id: source.id,
            expected_revision: source.revision,
            operation_id: 'operation:program',
            recorded_at: RECORDED_AT,
            turn,
            entry,
            payload_fingerprint: await fingerprintJson({ turns: [turn], context_entries: [entry] }),
        };
        const expected = appendConversationRecords(
            source,
            { turns: [turn], context_entries: [entry] },
            {
                expected_revision: source.revision,
                operation_id: command.operation_id,
                payload_fingerprint: command.payload_fingerprint,
                recorded_at: RECORDED_AT,
            },
        );
        const staged = await stageIndexedProgramAppend(initial.root, command, memory.store);
        expect(staged.applied).toBe(true);
        expect(staged.root.source.revision).toBe(1);
        expect(Object.hasOwn(staged.root.directories, 'deletion_blockers')).toBe(false);
        expect(preflightJsonInput(staged.root).success).toBe(true);
        expect(staged.receipt).toEqual(expected.document.operation_receipts[command.operation_id]);
        expect((await loadIndexedActiveContext(memory.store, staged.root)).entries).toEqual(
            expected.document.context.entries,
        );
        const writes = memory.records.size;
        const retry = await stageIndexedProgramAppend(staged.root, command, memory.store);
        expect(retry).toMatchObject({ applied: false, receipt: staged.receipt });
        expect(memory.records.size).toBe(writes);
        await expect(
            stageIndexedProgramAppend(
                staged.root,
                { ...command, turn: { ...turn, blocks: [{ ...turn.blocks[0], text: 'changed' }] } },
                memory.store,
            ),
        ).rejects.toThrow('fingerprint differs');
    });

    it('rejects program records whose unresolved dependencies or timestamps fail full append validation', async () => {
        const source = emptyDocument('conversation:indexed-invalid-program');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const base = {
            id: 'turn:invalid',
            kind: 'program',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT },
            provenance: { type: 'inserted', operation_id: 'operation:invalid' },
            model_visibility: 'include',
            blocks: [{ id: 'block:invalid', type: 'text', text: 'Continue.', format: 'plain' }],
        };
        const entry = { id: 'entry:invalid', type: 'source_turn' as const, turn_id: base.id };
        const malformed = [
            { ...base, parent_turn_id: 'missing-parent' },
            { ...base, execution_id: 'missing-execution' },
            { ...base, blocks: [{ id: 'block:invalid', type: 'image', asset_id: 'missing-asset' }] },
            {
                ...base,
                timestamps: {
                    recorded_at: RECORDED_AT,
                    started_at: '2026-10-02T00:01:00.000Z',
                    completed_at: '2026-10-02T00:00:00.000Z',
                },
            },
            { ...base, timestamps: { recorded_at: '2026-01-01T00:00:00.000Z' } },
        ];
        for (const input of malformed) {
            const turn = ProgramTurnSchema.parse(input);
            const recordedAt = turn.timestamps.recorded_at;
            const command = {
                conversation_id: source.id,
                expected_revision: source.revision,
                operation_id: 'operation:invalid',
                recorded_at: recordedAt,
                turn,
                entry,
                payload_fingerprint: await fingerprintJson({ turns: [turn], context_entries: [entry] }),
            };
            expect(() =>
                appendConversationRecords(
                    source,
                    { turns: [turn], context_entries: [entry] },
                    {
                        expected_revision: source.revision,
                        operation_id: command.operation_id,
                        payload_fingerprint: command.payload_fingerprint,
                        recorded_at: recordedAt,
                    },
                ),
            ).toThrow();
            const writes = memory.records.size;
            await expect(stageIndexedProgramAppend(initial.root, command, memory.store)).rejects.toThrow();
            expect(memory.records.size).toBe(writes);
        }
    });
});

describe('indexed scheduled whole-exchange processing', () => {
    it('keeps two completed calls and results as two bounded accepted jobs in one native append', async () => {
        const memory = memoryStore();
        memory.store.assertExternalAssetIntegrity = async () => undefined;
        const source = (
            await setProcessingPolicy(emptyDocument('conversation:indexed-two-exchanges'), {
                operation_id: 'operation:two-policy',
                expected_revision: 0,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors: [
                    {
                        id: 'externalize-whole-exchange',
                        version: '1',
                        scope: 'on_append',
                        config: {},
                        required: true,
                        failure_behavior: 'block',
                    },
                ],
            })
        ).document;
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const first = await toolMediaCommands(initial.root.source);
        const secondRaw: unknown = JSON.parse(JSON.stringify(first).replaceAll('dependency', 'second'));
        if (
            !secondRaw ||
            typeof secondRaw !== 'object' ||
            !('agentCommand' in secondRaw) ||
            !('resultCommand' in secondRaw)
        ) {
            throw new Error('Second accepted exchange fixture is invalid');
        }
        const secondAgent = IndexedRecordBatchCommandSchema.parse(secondRaw.agentCommand);
        const secondCall = secondAgent.batch.turns?.[0]?.blocks[0];
        if (secondCall?.type !== 'tool_call') throw new Error('Second accepted call fixture is missing');
        const secondCallFingerprint = await fingerprintJson(secondCall);
        const secondResultRaw = IndexedRecordBatchCommandSchema.parse(secondRaw.resultCommand);
        const secondResult: IndexedRecordBatchCommand = {
            ...secondResultRaw,
            batch: {
                ...secondResultRaw.batch,
                execution_receipts: secondResultRaw.batch.execution_receipts?.map((receipt) => ({
                    ...receipt,
                    ...(receipt.call_source
                        ? {
                              call_source: {
                                  ...receipt.call_source,
                                  call_fingerprint: secondCallFingerprint,
                              },
                          }
                        : {}),
                })),
            },
        };
        const reader = {
            id: 'definition:two-originals',
            name: 'read_artifact',
            version: '1',
            input_schema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
        };
        const agentBatch: IndexedRecordBatchCommand['batch'] = {
            turns: [...(first.agentCommand.batch.turns ?? []), ...(secondAgent.batch.turns ?? [])],
            generations: [...(first.agentCommand.batch.generations ?? []), ...(secondAgent.batch.generations ?? [])],
            tool_definitions: [
                ...(first.agentCommand.batch.tool_definitions ?? []),
                ...(secondAgent.batch.tool_definitions ?? []),
                reader,
            ],
            active_tool_definition_ids: [first.call.definition_id, 'definition:second', reader.id],
            context_entries: [
                ...(first.agentCommand.batch.context_entries ?? []),
                ...(secondAgent.batch.context_entries ?? []),
            ],
        };
        const called = await stageIndexedRecordBatch(
            initial.root,
            {
                conversation_id: initial.root.source.conversation_id,
                batch: agentBatch,
                options: {
                    operation_id: 'operation:two-calls',
                    expected_revision: initial.root.source.revision,
                    payload_fingerprint: await fingerprintJson(agentBatch),
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        const original = [first.resultCommand, secondResult];
        const results = await Promise.all(
            original.map(async (command, index) => {
                const turn = command.batch.turns?.[0];
                const block = turn?.blocks[0];
                const execution = command.batch.execution_receipts?.[0];
                if (!turn || block?.type !== 'tool_result' || !execution)
                    throw new Error('Fixture result is incomplete');
                const path = `canonical-tool-results/v1/two/${index}/result.json`;
                const integrity = await hashContentBytes(new TextEncoder().encode(`accepted original ${index}`));
                const asset: Asset = {
                    id: `asset:two-original-${index}`,
                    kind: 'text',
                    mime_type: 'application/json',
                    storage: {
                        type: 'external',
                        resolver: 'vertesia.agent_artifact',
                        locator: { storage_id: 'agent:root', artifact_path: path },
                    },
                    provenance: { type: 'received', source_turn_id: turn.id },
                    created_at: RECORDED_AT,
                    ...integrity,
                };
                const retrieval = {
                    capability: 'read_artifact',
                    version: 1 as const,
                    tool_definition_id: reader.id,
                    arguments: { path },
                };
                const resultBlock = {
                    ...block,
                    content: [
                        {
                            id: `block:two-reference-${index}`,
                            type: 'external_reference' as const,
                            original_type: 'text' as const,
                            asset_id: asset.id,
                            description: `Accepted original ${index}`,
                            preview: `Original ${index}`,
                            content_hash: integrity.content_hash,
                            retrieval,
                        },
                    ],
                };
                return {
                    turn: ToolTurnSchema.parse({ ...turn, blocks: [resultBlock] }),
                    asset,
                    execution: { ...execution, result_fingerprint: await fingerprintJson(resultBlock) },
                    entry: command.batch.context_entries?.[0],
                    requirement: {
                        id: `requirement:two-original-${index}`,
                        asset_id: asset.id,
                        retrieval,
                        accepted_asset_operation_id: 'operation:two-results',
                    },
                };
            }),
        );
        const resultBatch: IndexedRecordBatchCommand['batch'] = {
            turns: results.map((result) => result.turn),
            assets: results.map((result) => result.asset),
            execution_receipts: results.map((result) => result.execution),
            context_entries: results.flatMap((result) => (result.entry === undefined ? [] : [result.entry])),
            retrieval_requirements: results.map((result) => result.requirement),
        };
        const appended = await stageIndexedRecordBatch(
            called.root,
            {
                conversation_id: called.root.source.conversation_id,
                batch: resultBatch,
                options: {
                    operation_id: 'operation:two-results',
                    expected_revision: called.root.source.revision,
                    payload_fingerprint: await fingerprintJson(resultBatch),
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        const pending = await loadIndexedPendingProcessingJobs(memory.store, appended.root);
        expect(pending.unresolved_job_count).toBe(2);
        expect(pending.jobs.map((item) => item.job.selection)).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    kind: 'entries',
                    entry_ids: ['entry:dependency-agent', 'entry:dependency-result'],
                }),
                expect.objectContaining({ kind: 'entries', entry_ids: ['entry:second-agent', 'entry:second-result'] }),
            ]),
        );
        expect(new Set(pending.jobs.map((item) => item.job.id)).size).toBe(2);
    });

    it('carries an idle exchange policy through ordinary materialized input while refusing a selected result', async () => {
        const policy = await setProcessingPolicy(emptyDocument('conversation:materialized-exchange-source'), {
            operation_id: 'operation:exchange-policy',
            expected_revision: 0,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                {
                    id: 'externalize-whole-exchange',
                    version: '1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const input = userTurn('turn:exchange-input');
        const inputBatch = {
            turns: [input],
            context_entries: [{ id: 'entry:exchange-input', type: 'source_turn' as const, turn_id: input.id }],
        };
        const inputOptions = {
            operation_id: 'operation:exchange-input',
            expected_revision: policy.document.revision,
            payload_fingerprint: await fingerprintJson(inputBatch),
            recorded_at: RECORDED_AT,
        };
        const accepted = await appendConversationRecordsWithProcessing(policy.document, inputBatch, inputOptions);
        expect(accepted.applied).toBe(true);
        expect(accepted.acceptance.processing).toEqual({ status: 'pending', job_ids: [] });
        expect(Object.values(accepted.document.processing.jobs ?? {})).toEqual([]);
        const retry = await appendConversationRecordsWithProcessing(accepted.document, inputBatch, inputOptions);
        expect(retry.applied).toBe(false);
        expect(retry.acceptance).toEqual(accepted.acceptance);

        const commands = await toolMediaCommands({
            conversation_id: accepted.document.id,
            revision: accepted.document.revision,
        });
        const originalReader = {
            id: 'definition:materialized-original',
            name: 'read_artifact',
            version: '1',
            input_schema: {
                type: 'object',
                properties: { path: { type: 'string' } },
                required: ['path'],
                additionalProperties: false,
            },
        };
        const callBatch: IndexedRecordBatchCommand['batch'] = {
            ...commands.agentCommand.batch,
            tool_definitions: [...(commands.agentCommand.batch.tool_definitions ?? []), originalReader],
            active_tool_definition_ids: [commands.call.definition_id, originalReader.id],
        };
        const called = await appendConversationRecordsWithProcessing(accepted.document, callBatch, {
            ...commands.agentCommand.options,
            payload_fingerprint: await fingerprintJson(callBatch),
        });
        expect(called.acceptance.processing.job_ids).toEqual([]);
        await expect(
            appendConversationRecordsWithProcessing(called.document, commands.resultCommand.batch, {
                ...commands.resultCommand.options,
                expected_revision: called.document.revision,
                payload_fingerprint: await fingerprintJson(commands.resultCommand.batch),
            }),
        ).rejects.toThrow('Materialized append cannot process an accepted whole exchange');
        const originalBytes = new TextEncoder().encode('{"accepted":"original"}');
        const integrity = await hashContentBytes(originalBytes);
        const originalAsset: Asset = {
            id: 'asset:materialized-original',
            kind: 'text',
            mime_type: 'application/json',
            storage: {
                type: 'external',
                resolver: 'vertesia.agent_artifact',
                locator: { storage_id: 'agent:root', artifact_path: 'canonical-tool-results/v1/tool/result.json' },
            },
            provenance: { type: 'received', source_turn_id: commands.result.id },
            created_at: RECORDED_AT,
            ...integrity,
        };
        const retrieval = {
            capability: 'read_artifact',
            version: 1 as const,
            tool_definition_id: originalReader.id,
            arguments: { path: 'canonical-tool-results/v1/tool/result.json' },
        };
        const originalBlock = commands.result.blocks[0];
        if (originalBlock?.type !== 'tool_result') throw new Error('Fixture has no accepted result block');
        const result = {
            ...commands.result,
            blocks: [
                {
                    ...originalBlock,
                    content: [
                        {
                            id: 'block:materialized-original',
                            type: 'external_reference' as const,
                            original_type: 'text' as const,
                            asset_id: originalAsset.id,
                            description: 'Original accepted tool result',
                            preview: 'Original accepted tool result preview',
                            content_hash: integrity.content_hash,
                            retrieval,
                        },
                    ],
                },
            ],
        };
        const resultBatch: IndexedRecordBatchCommand['batch'] = {
            turns: [result],
            assets: [originalAsset],
            execution_receipts: [
                { ...commands.execution, result_fingerprint: await fingerprintJson(result.blocks[0]) },
            ],
            context_entries: commands.resultCommand.batch.context_entries,
            retrieval_requirements: [
                {
                    id: 'requirement:materialized-original',
                    asset_id: originalAsset.id,
                    retrieval,
                    accepted_asset_operation_id: commands.resultCommand.options.operation_id,
                },
            ],
        };
        await expect(
            appendConversationRecordsWithProcessing(called.document, resultBatch, {
                ...commands.resultCommand.options,
                expected_revision: called.document.revision,
                payload_fingerprint: await fingerprintJson(resultBatch),
            }),
        ).rejects.toThrow('Materialized append cannot process an accepted whole exchange');
        // The public processing append above is the only supported acceptance path for this
        // enabled policy. It rejects the eligible result before a receipt or processing job exists.
    });

    it('enqueues the exact accepted call and result, then settles one registered copied archive through the existing phase chain', async () => {
        const memory = memoryStore();
        memory.store.assertExternalAssetIntegrity = async () => undefined;
        const configuration = {
            id: 'externalize-whole-exchange',
            version: '1',
            scope: 'on_append' as const,
            config: {},
            required: true,
            failure_behavior: 'block' as const,
        };
        const source = (
            await setProcessingPolicy(emptyDocument('conversation:indexed-exchange-processing'), {
                operation_id: 'operation:policy',
                expected_revision: 0,
                recorded_at: RECORDED_AT,
                enabled: true,
                processors: [configuration],
            })
        ).document;
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const commands = await toolMediaCommands(initial.root.source);
        const originalReader = {
            id: 'definition:original-archive',
            name: 'read_artifact',
            version: '1',
            input_schema: {
                type: 'object',
                properties: { path: { type: 'string' } },
                required: ['path'],
                additionalProperties: false,
            },
        };
        const runReader = {
            id: 'definition:run-archive',
            name: 'read_run_artifact',
            version: '1',
            input_schema: {
                type: 'object',
                properties: {
                    indexed_namespace_id: { type: 'string' },
                    asset_id: { type: 'string' },
                },
                required: ['indexed_namespace_id', 'asset_id'],
                additionalProperties: false,
            },
        };
        const agentBatch: IndexedRecordBatchCommand['batch'] = {
            ...commands.agentCommand.batch,
            tool_definitions: [...(commands.agentCommand.batch.tool_definitions ?? []), originalReader, runReader],
            active_tool_definition_ids: [commands.call.definition_id, originalReader.id, runReader.id],
        };
        const agentCommand = {
            ...commands.agentCommand,
            batch: agentBatch,
            options: { ...commands.agentCommand.options, payload_fingerprint: await fingerprintJson(agentBatch) },
        };
        const called = await stageIndexedRecordBatch(initial.root, agentCommand, memory.store);
        if (!called.locator) throw new Error('Accepted call has no root');
        expect((await loadIndexedPendingProcessingJobs(memory.store, called.root)).unresolved_job_count).toBe(0);
        expect((await loadIndexedProcessingSelectedContext(memory.store, called.root, called.locator)).source).toEqual(
            called.root.source,
        );
        const ordinary = await stageIndexedRecordBatch(
            called.root,
            {
                ...commands.resultCommand,
                options: {
                    ...commands.resultCommand.options,
                    expected_revision: called.root.source.revision,
                    payload_fingerprint: await fingerprintJson(commands.resultCommand.batch),
                },
            },
            memory.store,
        );
        expect(ordinary.applied).toBe(true);
        expect((await loadIndexedPendingProcessingJobs(memory.store, ordinary.root)).unresolved_job_count).toBe(0);
        const originalBytes = new TextEncoder().encode('Exact accepted external tool output');
        const originalIntegrity = await hashContentBytes(originalBytes);
        const originalAsset: Asset = {
            id: 'asset:original-archive',
            kind: 'text',
            mime_type: 'application/json',
            storage: {
                type: 'external',
                resolver: 'vertesia.agent_artifact',
                locator: {
                    storage_id: 'agent:root',
                    artifact_path: 'canonical-tool-results/v1/tool/call/result.json',
                },
            },
            provenance: { type: 'received', source_turn_id: commands.result.id },
            created_at: RECORDED_AT,
            ...originalIntegrity,
        };
        const originalRetrieval = {
            capability: 'read_artifact',
            version: 1 as const,
            tool_definition_id: originalReader.id,
            arguments: { path: 'canonical-tool-results/v1/tool/call/result.json' },
        };
        const existingResult = commands.result.blocks[0];
        if (existingResult?.type !== 'tool_result') throw new Error('Fixture result missing');
        const result = {
            ...commands.result,
            blocks: [
                {
                    ...existingResult,
                    content: [
                        {
                            id: 'block:original-reference',
                            type: 'external_reference' as const,
                            original_type: 'text' as const,
                            asset_id: originalAsset.id,
                            description: 'Exact original archive',
                            preview: 'Original archive preview',
                            content_hash: originalIntegrity.content_hash,
                            retrieval: originalRetrieval,
                        },
                    ],
                },
            ],
        };
        const execution = { ...commands.execution, result_fingerprint: await fingerprintJson(result.blocks[0]) };
        const requirement = {
            id: 'requirement:original-archive',
            asset_id: originalAsset.id,
            retrieval: originalRetrieval,
            accepted_asset_operation_id: commands.resultCommand.options.operation_id,
        };
        const resultBatch: IndexedRecordBatchCommand['batch'] = {
            turns: [result],
            assets: [originalAsset],
            execution_receipts: [execution],
            context_entries: commands.resultCommand.batch.context_entries,
            retrieval_requirements: [requirement],
        };
        const siblingBlock = textBlock('block:unarchived-sibling', 'Unarchived content');
        const mixedResult = {
            ...result,
            blocks: [
                {
                    ...result.blocks[0],
                    content: [...result.blocks[0].content, siblingBlock],
                },
            ],
        };
        const mixedBatch: IndexedRecordBatchCommand['batch'] = {
            ...resultBatch,
            turns: [mixedResult],
            execution_receipts: [{ ...execution, result_fingerprint: await fingerprintJson(mixedResult.blocks[0]) }],
        };
        await expect(
            stageIndexedRecordBatch(
                called.root,
                {
                    conversation_id: called.root.source.conversation_id,
                    batch: mixedBatch,
                    options: {
                        operation_id: 'operation:mixed-original',
                        expected_revision: called.root.source.revision,
                        payload_fingerprint: await fingerprintJson(mixedBatch),
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('Indexed whole-exchange result requires one exact original archive reference');
        const appended = await stageIndexedRecordBatch(
            called.root,
            {
                ...commands.resultCommand,
                batch: resultBatch,
                options: { ...commands.resultCommand.options, payload_fingerprint: await fingerprintJson(resultBatch) },
            },
            memory.store,
        );
        if (!appended.locator) throw new Error('Accepted result has no root');
        const pending = await loadIndexedPendingProcessingJobs(memory.store, appended.root);
        expect(pending.unresolved_job_count).toBe(1);
        const job = pending.jobs[0]?.job;
        if (job?.selection.kind !== 'entries') throw new Error('Actual exchange job was not accepted');
        expect(job.selection.entry_ids).toEqual(['entry:dependency-agent', 'entry:dependency-result']);
        expect(job.selection.selected_block_ids).toEqual({
            'entry:dependency-agent': [commands.call.id],
            'entry:dependency-result': [result.blocks[0].id],
        });
        const callIntegrity = await hashContentBytes(canonicalJsonContentBytes(commands.call));
        const copiedAssets: Asset[] = [
            {
                id: 'asset:copied-call',
                kind: 'text',
                mime_type: 'application/json',
                storage: {
                    type: 'external',
                    resolver: 'vertesia.run_text_archive',
                    locator: {
                        indexed_namespace_id: 'namespace:owned',
                        artifact_path: `archive/${callIntegrity.content_hash}`,
                    },
                },
                provenance: { type: 'received', source_turn_id: commands.agent.id },
                created_at: RECORDED_AT,
                ...callIntegrity,
            },
            {
                id: 'asset:copied-result',
                kind: 'text',
                mime_type: 'application/json',
                storage: {
                    type: 'external',
                    resolver: 'vertesia.run_text_archive',
                    locator: {
                        indexed_namespace_id: 'namespace:owned',
                        artifact_path: `archive/${originalIntegrity.content_hash}`,
                    },
                },
                provenance: {
                    type: 'derived',
                    source_asset_id: originalAsset.id,
                    transform_id: 'conversation.archive_rehome',
                    transform_version: '1',
                },
                created_at: RECORDED_AT,
                ...originalIntegrity,
            },
        ];
        const derivedCopy = copiedAssets[1];
        const callCopy = copiedAssets[0];
        if (!derivedCopy || !callCopy) throw new Error('Copied exchange fixture lacks both assets');
        const wrongSourceOperation = 'operation:wrong-source-archive';
        const wrongSourceBatch: IndexedRecordBatchCommand['batch'] = {
            ...resultBatch,
            assets: [{ ...originalAsset, provenance: { type: 'received', source_turn_id: commands.agent.id } }],
            retrieval_requirements: [{ ...requirement, accepted_asset_operation_id: wrongSourceOperation }],
        };
        const wrongSource = await stageIndexedRecordBatch(
            called.root,
            {
                conversation_id: called.root.source.conversation_id,
                batch: wrongSourceBatch,
                options: {
                    operation_id: wrongSourceOperation,
                    expected_revision: called.root.source.revision,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(wrongSourceBatch),
                },
            },
            memory.store,
        );
        const wrongJob = (await loadIndexedPendingProcessingJobs(memory.store, wrongSource.root)).jobs[0]?.job;
        if (!wrongJob) throw new Error('Wrong-source fixture did not accept a processing job');
        await expect(
            stageIndexedRecordBatch(
                wrongSource.root,
                {
                    conversation_id: wrongSource.root.source.conversation_id,
                    batch: { assets: copiedAssets },
                    options: {
                        operation_id: `processing:archive:${wrongJob.id}`,
                        expected_revision: wrongSource.root.source.revision,
                        recorded_at: RECORDED_AT,
                        payload_fingerprint: await fingerprintJson(copiedAssets),
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('Indexed derived archive lacks its exact selected original and immutable bytes');
        for (const altered of [
            {
                ...derivedCopy,
                provenance: {
                    type: 'derived' as const,
                    source_asset_id: 'asset:foreign-original',
                    transform_id: 'conversation.archive_rehome',
                    transform_version: '1',
                },
            },
            { ...derivedCopy, content_hash: `sha256:${'f'.repeat(64)}` },
            {
                ...derivedCopy,
                provenance: {
                    type: 'derived' as const,
                    source_asset_id: originalAsset.id,
                    transform_id: 'conversation.other_transform',
                    transform_version: '1',
                },
            },
        ]) {
            const invalidAssets: Asset[] = [callCopy, altered];
            await expect(
                stageIndexedRecordBatch(
                    appended.root,
                    {
                        conversation_id: appended.root.source.conversation_id,
                        batch: { assets: invalidAssets },
                        options: {
                            expected_revision: appended.root.source.revision,
                            operation_id: `processing:archive:${job.id}`,
                            recorded_at: RECORDED_AT,
                            payload_fingerprint: await fingerprintJson(invalidAssets),
                        },
                    },
                    memory.store,
                ),
            ).rejects.toThrow('Indexed derived archive lacks its exact selected original and immutable bytes');
        }
        const copied = await stageIndexedRecordBatch(
            appended.root,
            {
                conversation_id: appended.root.source.conversation_id,
                batch: { assets: copiedAssets },
                options: {
                    expected_revision: appended.root.source.revision,
                    operation_id: `processing:archive:${job.id}`,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(copiedAssets),
                },
            },
            memory.store,
        );
        if (!copied.locator) throw new Error('Copied archive has no root');
        let staged = { root: copied.root, locator: copied.locator };
        const resolution = await resolveIndexedProcessingTextInput(
            await loadIndexedProcessingSelectedContext(memory.store, staged.root, staged.locator),
            job,
            RECORDED_AT,
        );
        staged = await stageIndexedProcessingPhase(memory.store, staged.root, staged.locator, {
            phase: 'resolve',
            value: resolution,
        });
        const attempt = {
            job_id: job.id,
            resolved_input_fingerprint: await fingerprintJson(resolution),
            attempt_token: 'attempt:exchange',
            started_at: RECORDED_AT,
        };
        staged = await stageIndexedProcessingPhase(memory.store, staged.root, staged.locator, {
            phase: 'attempt',
            value: attempt,
        });
        const selected = await loadIndexedProcessingSelectedContext(memory.store, staged.root, staged.locator);
        const retrievals = copiedAssets.map((asset) => ({
            capability: 'read_run_artifact',
            version: 1 as const,
            tool_definition_id: runReader.id,
            arguments: { indexed_namespace_id: 'namespace:owned', asset_id: asset.id },
        }));
        const workspace = {
            version: 1 as const,
            selected,
            job,
            configuration,
            resolution,
            attempt,
            snapshot_at: RECORDED_AT,
            archives: { assets: copiedAssets, acceptance: copied.receipt, retrievals },
        };
        const output = await buildIndexedExchangeOutput(workspace);
        const firstCopiedAsset = copiedAssets[0];
        if (!firstCopiedAsset) throw new Error('Accepted call copy is missing');
        await expect(
            buildIndexedExchangeOutput({
                ...workspace,
                archives: {
                    ...workspace.archives,
                    acceptance: { ...copied.receipt, accepted_asset_ids: [firstCopiedAsset.id] },
                },
            }),
        ).rejects.toThrow('exact accepted job/archive binding');
        const originalPublication = selected.operation_witnesses?.[requirement.accepted_asset_operation_id];
        if (!originalPublication) throw new Error('Accepted original publication is missing from selected proof');
        await expect(
            buildIndexedExchangeOutput({
                ...workspace,
                selected: {
                    ...selected,
                    operation_witnesses: {
                        ...selected.operation_witnesses,
                        [requirement.accepted_asset_operation_id]: {
                            ...originalPublication,
                            accepted_retrieval_requirements: [],
                        },
                    },
                },
            }),
        ).rejects.toThrow('one completed call and its exact archived result');
        await expect(
            buildIndexedExchangeOutput({
                ...workspace,
                selected: {
                    ...selected,
                    context: {
                        ...selected.context,
                        active_tool_definition_ids: selected.context.active_tool_definition_ids.filter(
                            (id) => id !== runReader.id,
                        ),
                    },
                },
            }),
        ).rejects.toThrow('exact resolved selection');
        staged = await stageIndexedProcessingPhase(memory.store, staged.root, staged.locator, {
            phase: 'output',
            value: output,
        });
        const retained = await loadIndexedProcessingJobState(memory.store, staged.root, job.id);
        expect(retained.output).toEqual(output);
        const beforeRejectedCompletion = memory.records.size;
        await expect(
            stageIndexedTextProcessingCompletion(
                memory.store,
                staged.root,
                staged.locator,
                workspace,
                new Map([[originalAsset.id, new TextEncoder().encode('changed original')]]),
            ),
        ).rejects.toThrow('source archive bytes differ');
        expect(memory.records.size).toBe(beforeRejectedCompletion);
        const completed = await stageIndexedTextProcessingCompletion(
            memory.store,
            staged.root,
            staged.locator,
            workspace,
            new Map([[originalAsset.id, originalBytes]]),
        );
        expect(completed.completion.status).toBe('applied');
        expect((await loadIndexedPendingProcessingJobs(memory.store, completed.root)).unresolved_job_count).toBe(0);
        const active = await loadIndexedProcessingSelectedContext(memory.store, completed.root, completed.locator);
        expect(active.context.retrieval_requirements).toHaveLength(3);
        expect(active.replacement_turns?.flatMap((turn) => turn.projection.selected_blocks)).toHaveLength(2);
        const retry = await stageIndexedTextProcessingCompletion(
            memory.store,
            completed.root,
            completed.locator,
            workspace,
            new Map(),
        );
        expect(retry).toMatchObject({ applied: false, receipt: completed.receipt });
    });
});

describe('processing-only pending call selection', () => {
    it('resolves a real accepted text job around an unresolved call without selecting the call for compaction', async () => {
        const memory = memoryStore();
        const policy = await setProcessingPolicy(emptyDocument('conversation:pending-text-job'), {
            operation_id: 'operation:text-policy',
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
        });
        const initial = await stageIndexedConversationSnapshot(policy.document, undefined, memory.store);
        const commands = await toolMediaCommands(initial.root.source);
        const agent = {
            ...commands.agent,
            blocks: [textBlock('block:pending-text', 'Archive this original text.'), commands.call],
        };
        const batch = { ...commands.agentCommand.batch, turns: [agent] };
        const called = await stageIndexedRecordBatch(
            initial.root,
            {
                ...commands.agentCommand,
                batch,
                options: { ...commands.agentCommand.options, payload_fingerprint: await fingerprintJson(batch) },
            },
            memory.store,
        );
        if (!called.locator) throw new Error('Accepted pending text call root missing');
        const pending = await loadIndexedPendingProcessingJobs(memory.store, called.root);
        expect(pending.unresolved_job_count).toBe(1);
        const job = pending.jobs[0]?.job;
        if (!job) throw new Error('Real text append job missing');
        const selected = await loadIndexedProcessingSelectedContext(memory.store, called.root, called.locator);
        const resolution = await resolveIndexedProcessingTextInput(selected, job, RECORDED_AT);
        expect(resolution.selected_block_ids).toEqual({ 'entry:dependency-agent': ['block:pending-text'] });
        expect(selected.execution_witnesses).toEqual({});
        await expect(
            loadIndexedSettledProcessingSelectedContext(memory.store, called.root, called.locator),
        ).rejects.toThrow('unresolved processing obligations');
    });

    it('retains exact accepted pending calls for archival input without granting provider preparation or inventing results', async () => {
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:pending-processing'),
            undefined,
            memory.store,
        );
        const commands = await toolMediaCommands(initial.root.source);
        const called = await stageIndexedRecordBatch(initial.root, commands.agentCommand, memory.store);
        if (!called.locator) throw new Error('Accepted pending call root missing');
        const selected = await loadIndexedProcessingSelectedContext(memory.store, called.root, called.locator);
        expect(selected.completeness).toBe('active_processing_dependencies_verified');
        expect(selected.turns[0]?.selected_blocks).toEqual([commands.call]);
        expect(selected.execution_witnesses).toEqual({});
        await expect(loadIndexedSelectedDependencyContext(memory.store, called.root, called.locator)).rejects.toThrow(
            'exact selected terminal result',
        );
        // Enable a real registered policy after accepting the call: enabling does not retroactively enqueue work.
        const accepted = appendConversationRecords(
            emptyDocument('conversation:pending-processing'),
            commands.agentCommand.batch,
            commands.agentCommand.options,
        );
        const policy = await setProcessingPolicy(accepted.document, {
            operation_id: 'operation:enable-after-call',
            expected_revision: accepted.document.revision,
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
        });
        const strict = await stageIndexedConversationSnapshot(policy.document, undefined, memory.store);
        expect((await loadIndexedPendingProcessingJobs(memory.store, strict.root)).unresolved_job_count).toBe(0);
        await expect(
            loadIndexedSettledProcessingSelectedContext(memory.store, strict.root, strict.locator),
        ).rejects.toThrow('exact selected terminal result');
        const descriptor = await getPagedRecord(
            memory.store,
            called.root.directories.tool_call_states,
            commands.call.call_id,
        );
        if (descriptor?.storage !== 'record') throw new Error('Actual accepted call state missing');
        const original = JSON.parse(new TextDecoder().decode(await memory.store.readRecord(descriptor)));
        for (const mutation of [
            { call_fingerprint: `sha256:${'0'.repeat(64)}` },
            { terminal_receipt_id: 'receipt:invented' },
            { result_block_id: 'block:invented' },
        ]) {
            const bytes = new TextEncoder().encode(JSON.stringify({ ...original, ...mutation }));
            const integrity = await hashContentBytes(bytes);
            const value = { ...descriptor, content_hash: integrity.content_hash, size_bytes: bytes.byteLength };
            await memory.store.writeRecord(value, bytes);
            const index = await putPagedRecord(
                memory.store,
                called.root.directories.tool_call_states,
                commands.call.call_id,
                value,
                'replace',
            );
            const corrupted = { ...called.root, directories: { ...called.root.directories, tool_call_states: index } };
            await expect(
                loadIndexedProcessingSelectedContext(memory.store, corrupted, called.locator),
            ).rejects.toThrow();
        }
        const completed = await stageIndexedRecordBatch(called.root, commands.resultCommand, memory.store);
        if (!completed.locator) throw new Error('Accepted terminal result root missing');
        const retainedResult = await loadIndexedProcessingSelectedContext(
            memory.store,
            completed.root,
            completed.locator,
        );
        expect(retainedResult.execution_witnesses).toHaveProperty(commands.execution.id, commands.execution);
        expect(
            (await loadIndexedSelectedDependencyContext(memory.store, completed.root, completed.locator)).source,
        ).toEqual(completed.root.source);
    });
});

describe('bounded indexed historical presentation', () => {
    it('binds one selected terminal program block to its actual accepted append receipt', async () => {
        const source = emptyDocument('conversation:indexed-terminal-presentation');
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const turn = ProgramTurnSchema.parse({
            id: 'turn:indexed-terminal',
            kind: 'program',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT },
            provenance: { type: 'inserted', operation_id: 'operation:indexed-terminal' },
            model_visibility: 'exclude',
            presentation: 'transcript',
            blocks: [{ id: 'block:indexed-terminal', type: 'json', value: { done: true } }],
        });
        const entry = { id: 'entry:indexed-terminal', type: 'source_turn' as const, turn_id: turn.id };
        const staged = await stageIndexedProgramAppend(
            initial.root,
            {
                conversation_id: source.id,
                expected_revision: source.revision,
                operation_id: 'operation:indexed-terminal',
                recorded_at: RECORDED_AT,
                turn,
                entry,
                payload_fingerprint: await fingerprintJson({ turns: [turn], context_entries: [entry] }),
            },
            memory.store,
        );

        const selected = await loadIndexedTerminalProgramPresentation(
            memory.store,
            staged.root,
            staged.receipt,
            turn.id,
        );
        expect(selected).toEqual({ receipt: staged.receipt, turn });
        await expect(
            loadIndexedTerminalProgramPresentation(memory.store, staged.root, staged.receipt, 'turn:foreign'),
        ).rejects.toThrow(IndexedPresentationNominationConflict);
    });

    it('selects exact accepted generation/turn records and rejects a changed receipt', async () => {
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:indexed-presentation'),
            undefined,
            memory.store,
        );
        const commands = await toolMediaCommands(initial.root.source);
        const accepted = await stageIndexedRecordBatch(initial.root, commands.agentCommand, memory.store);
        const receipt = accepted.receipt;
        const nomination = {
            id: receipt.id,
            conversation_id: receipt.conversation_id,
            base_revision: receipt.base_revision,
            result_revision: receipt.result_revision,
            recorded_at: receipt.recorded_at,
            accepted_turn_ids: receipt.accepted_turn_ids,
            accepted_generation_ids: receipt.accepted_generation_ids,
            ...(receipt.accepted_asset_ids === undefined ? {} : { accepted_asset_ids: receipt.accepted_asset_ids }),
        };
        memory.recordReads.length = 0;
        const selected = await loadIndexedAcceptedOutputPresentation(memory.store, accepted.root, nomination);
        expect(selected.fragment.receipt).toEqual(nomination);
        expect(selected.fragment.turn.blocks).toMatchObject([{ type: 'tool_call', call_id: commands.call.call_id }]);
        expect(selected.fragment.generation.record_source).toBe('executed');
        expect(selected.include_reasoning).toBe(false);
        expect(memory.recordReads).toContain('generations:generation:dependency');
        expect(memory.recordReads.length).toBeLessThan(32);
        await expect(
            loadIndexedAcceptedOutputPresentation(memory.store, accepted.root, {
                ...nomination,
                recorded_at: '2026-10-01T00:00:00.000Z',
            }),
        ).rejects.toThrow(IndexedPresentationNominationConflict);
    });

    async function historyResponseCommand(
        source: { conversation_id: string; revision: number },
        index: number,
        imported = false,
    ) {
        const template = (await toolMediaCommands(source)).agentCommand;
        const command = IndexedRecordBatchCommandSchema.parse(
            JSON.parse(JSON.stringify(template).replaceAll(':dependency', `:history:${index}`)),
        );
        const turn = command.batch.turns?.[0];
        const generation = command.batch.generations?.[0];
        if (turn?.kind !== 'agent' || !generation) throw new Error('Expected genuine generated history fixture');
        turn.blocks = [textBlock(`block:history:${index}`, `Accepted response ${index}`)];
        if (imported) command.batch.generations = [{ ...importedGeneration(generation.id), source }];
        command.options.payload_fingerprint = await fingerprintJson(command.batch);
        return command;
    }

    it('publishes ordered accepted history in fresh roots and seeks pinned pages without reading output blocks', async () => {
        const memory = memoryStore();
        let staged = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:history-order'),
            undefined,
            memory.store,
        );
        let lastCommand: IndexedRecordBatchCommand | undefined;
        for (let index = 0; index < 150; index += 1) {
            lastCommand = await historyResponseCommand(staged.root.source, index);
            const next = await stageIndexedRecordBatch(staged.root, lastCommand, memory.store);
            if (!next.locator) throw new Error('Expected fresh staged response root');
            staged = { root: next.root, locator: next.locator };
        }
        if (!lastCommand) throw new Error('Expected retained last accepted command');
        expect((await stageIndexedRecordBatch(staged.root, lastCommand, memory.store)).applied).toBe(false);
        const { accepted_output_order: _order, ...missingOrder } = staged.root.directories;
        await expect(
            stageIndexedRecordBatch({ ...staged.root, directories: missingOrder }, lastCommand, memory.store),
        ).rejects.toThrow('retry lacks its original ordered nomination');
        memory.pageReads.length = 0;
        memory.recordReads.length = 0;
        const first = await loadIndexedAcceptedOutputHistoryPage(memory.store, staged.root, {
            snapshot_revision: 120,
            limit: 100,
        });
        expect(first.references.map((reference) => reference.source.revision)).toEqual(
            Array.from({ length: 100 }, (_, index) => index + 1),
        );
        expect(first.next_after_revision).toBe(100);
        expect(first.omissions).toEqual([]);
        expect(memory.recordReads.some((id) => id.startsWith('blocks:'))).toBe(false);
        memory.pageReads.length = 0;
        memory.recordReads.length = 0;
        const second = await loadIndexedAcceptedOutputHistoryPage(memory.store, staged.root, {
            snapshot_revision: 120,
            after_revision: 100,
            limit: 100,
        });
        expect(second.references.map((reference) => reference.source.revision)).toEqual(
            Array.from({ length: 20 }, (_, index) => index + 101),
        );
        expect(second.next_after_revision).toBeUndefined();
        expect(memory.recordReads.length).toBeLessThan(100);
        expect(memory.pageReads.length).toBeLessThan(512);
        expect(
            (
                await loadIndexedAcceptedOutputHistoryPage(memory.store, staged.root, {
                    snapshot_revision: 120,
                    after_revision: 120,
                    limit: 10,
                })
            ).references,
        ).toEqual([]);
        await expect(
            loadIndexedAcceptedOutputHistoryPage(memory.store, staged.root, {
                snapshot_revision: 120,
                after_revision: 121,
                limit: 10,
            }),
        ).rejects.toThrow('cursor is invalid');
        const { accepted_output_index_complete: _complete, ...oldRoot } = staged.root;
        memory.recordReads.length = 0;
        memory.pageReads.length = 0;
        await expect(
            loadIndexedAcceptedOutputHistoryPage(memory.store, oldRoot, { snapshot_revision: 120, limit: 10 }),
        ).rejects.toThrow(IndexedAcceptedOutputHistoryUpgradeRequired);
        expect(memory.recordReads).toEqual([]);
        expect(memory.pageReads).toEqual([]);
    }, 30_000);

    it('imports original history once and consumes imported omissions without scanning to fill', async () => {
        let document = emptyDocument('conversation:history-import');
        for (let index = 0; index < 3; index += 1) {
            const command = await historyResponseCommand(
                { conversation_id: document.id, revision: document.revision },
                index,
                index === 1,
            );
            document = appendConversationRecords(document, command.batch, command.options).document;
        }
        const memory = memoryStore();
        const staged = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const first = await loadIndexedAcceptedOutputHistoryPage(memory.store, staged.root, {
            snapshot_revision: 3,
            limit: 1,
        });
        expect(first.references[0]?.source.revision).toBe(1);
        expect(first.next_after_revision).toBe(1);
        const omitted = await loadIndexedAcceptedOutputHistoryPage(memory.store, staged.root, {
            snapshot_revision: 3,
            after_revision: 1,
            limit: 1,
        });
        expect(omitted.references).toEqual([]);
        expect(omitted.omissions).toEqual([
            { operation_id: 'operation:history:1-agent', revision: 2, reason: 'imported' },
        ]);
        expect(omitted.next_after_revision).toBe(2);
        const last = await loadIndexedAcceptedOutputHistoryPage(memory.store, staged.root, {
            snapshot_revision: 3,
            after_revision: 2,
            limit: 1,
        });
        expect(last.references[0]?.source.revision).toBe(3);
        expect(last.next_after_revision).toBeUndefined();
    });

    it('projects an older accepted source from actual retained points without resurrecting a deleted turn', async () => {
        const memory = memoryStore();
        const source = emptyDocument('conversation:retained-stream-output');
        const commands = await toolMediaCommands({ conversation_id: source.id, revision: source.revision });
        const batch = structuredClone(commands.agentCommand.batch);
        const agent = batch.turns?.[0];
        if (agent?.kind !== 'agent') throw new Error('Expected genuine generated output fixture');
        agent.blocks = [textBlock('block:retained-answer', 'Actual accepted answer')];
        const accepted = appendConversationRecords(source, batch, commands.agentCommand.options);
        const later = appendConversationRecords(
            accepted.document,
            { turns: [userTurn('turn:later', 'block:later')] },
            {
                expected_revision: accepted.document.revision,
                operation_id: 'operation:later',
                recorded_at: RECORDED_AT,
                payload_fingerprint: 'sha256:later',
            },
        );
        // One-time import is anchored at N=2. There is deliberately no indexed physical root for M=1.
        const migrated = await stageIndexedConversationSnapshot(later.document, undefined, memory.store);
        const receipt = accepted.document.operation_receipts[accepted.change.operation_id];
        if (!receipt) throw new Error('Expected genuine accepted operation');
        const nomination = {
            id: receipt.id,
            conversation_id: receipt.conversation_id,
            base_revision: receipt.base_revision,
            result_revision: receipt.result_revision,
            recorded_at: receipt.recorded_at,
            accepted_turn_ids: receipt.accepted_turn_ids,
            accepted_generation_ids: receipt.accepted_generation_ids,
            ...(receipt.accepted_asset_ids === undefined ? {} : { accepted_asset_ids: receipt.accepted_asset_ids }),
        };
        memory.recordReads.length = 0;
        const selected = await loadIndexedRetainedAcceptedOutputPresentation(memory.store, migrated.root, nomination);
        expect(selected.fragment.source).toEqual({ conversation_id: source.id, revision: 1 });
        expect(selected.fragment.receipt).toEqual(nomination);
        expect(selected.fragment.turn.blocks).toEqual(agent.blocks);
        expect(memory.recordReads.length).toBeLessThan(32);
        await expect(loadIndexedAcceptedOutputPresentation(memory.store, migrated.root, nomination)).rejects.toThrow(
            IndexedPresentationNominationConflict,
        );
        await expect(
            loadIndexedRetainedAcceptedOutputPresentation(memory.store, migrated.root, {
                ...nomination,
                conversation_id: 'conversation:foreign',
            }),
        ).rejects.toThrow(IndexedPresentationNominationConflict);
        const selectedEntry = later.document.context.entries.find((entry) => entry.turn_id === agent.id);
        if (!selectedEntry) throw new Error('Expected active accepted answer entry');
        const exclusionSelection = {
            expected_revision: later.document.revision,
            expected_context_revision: later.document.context.revision,
            entry_ids: [selectedEntry.id],
        };
        const exclusionPlan = await planContextChange(later.document, exclusionSelection);
        const excluded = await applyContextChange(later.document, {
            ...exclusionSelection,
            operation_id: 'operation:exclude-old-answer',
            expected_source_fingerprint: exclusionPlan.source_fingerprint,
            recorded_at: RECORDED_AT,
            proposal: { kind: 'exclude' },
        });
        const exclusionRoot = await stageIndexedConversationSnapshot(excluded.document, undefined, memory.store);
        expect(exclusionRoot.root.source.revision).toBe(migrated.root.source.revision + 1);
        expect(await loadIndexedActiveContext(memory.store, exclusionRoot.root)).toMatchObject({
            entries: expect.not.arrayContaining([expect.objectContaining({ turn_id: agent.id })]),
        });
        expect(
            await loadIndexedRetainedAcceptedOutputPresentation(memory.store, exclusionRoot.root, nomination),
        ).toEqual(selected);
        const deleted = await stageIndexedConversationDelete(
            exclusionRoot.root,
            {
                operation_id: 'operation:delete-old-answer',
                source: exclusionRoot.root.source,
                expected_source_root: exclusionRoot.locator,
                recorded_at: RECORDED_AT,
                dependency_policy: 'reject',
                turn_ids: [agent.id],
            },
            memory.store,
        );
        await expect(
            loadIndexedRetainedAcceptedOutputPresentation(memory.store, deleted.root, nomination),
        ).rejects.toThrow('logically deleted');
        await expect(loadIndexedRestartEvidence(memory.store, deleted.root)).rejects.toMatchObject({
            reason: 'logically_deleted',
        });
        expect((await loadIndexedRestartEvidence(memory.store, migrated.root)).kind).toBe('accepted_output');
        const deletedHistory = await loadIndexedAcceptedOutputHistoryPage(memory.store, deleted.root, {
            snapshot_revision: migrated.root.source.revision,
            limit: 1,
        });
        expect(deletedHistory.references).toEqual([]);
        expect(deletedHistory.omissions).toEqual([
            { operation_id: receipt.id, revision: receipt.result_revision, reason: 'logically_deleted' },
        ]);
        expect(deletedHistory.next_after_revision).toBeUndefined();
        const forgedTombstone = await putPagedRecord(
            memory.store,
            deleted.root.directories.turns,
            agent.id,
            { storage: 'marker', kind: 'deleted_turn', id: 'turn:foreign-tombstone' },
            'replace',
        );
        await expect(
            loadIndexedAcceptedOutputHistoryPage(
                memory.store,
                { ...deleted.root, directories: { ...deleted.root.directories, turns: forgedTombstone } },
                { snapshot_revision: migrated.root.source.revision, limit: 1 },
            ),
        ).rejects.toThrow('exact tombstone');
        expect(
            (
                await loadIndexedAcceptedOutputHistoryPage(memory.store, migrated.root, {
                    snapshot_revision: migrated.root.source.revision,
                    limit: 1,
                })
            ).references,
        ).toEqual([{ source: selected.fragment.source, receipt: nomination }]);
        expect(await loadIndexedRetainedAcceptedOutputPresentation(memory.store, migrated.root, nomination)).toEqual(
            selected,
        );
    });

    it('selects the exact visible tail of a history larger than the topic profile without cold record reads', async () => {
        const memory = memoryStore();
        const document = emptyDocument('conversation:recent-tail');
        const turns = Array.from({ length: 4100 }, (_, index) => userTurn(`turn:recent:${index}`));
        const hidden = ProgramTurnSchema.parse({
            ...turns[0],
            id: 'program:recent:hidden',
            kind: 'program',
            presentation: 'transcript',
            blocks: [textBlock('block:recent:hidden', 'program-only')],
        });
        const accepted = appendConversationRecords(
            document,
            { turns: [...turns, hidden] },
            {
                expected_revision: 0,
                operation_id: 'operation:recent',
                payload_fingerprint: 'sha256:recent',
                recorded_at: RECORDED_AT,
            },
        );
        const staged = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        memory.recordReads.length = 0;
        memory.pageReads.length = 0;
        expect(await renderIndexedConversationRecentMessages(memory.store, staged.root, 2)).toEqual([
            { role: 'user', content: 'turn:recent:4098-text' },
            { role: 'user', content: 'turn:recent:4099-text' },
        ]);
        expect(memory.recordReads.length).toBeLessThan(16);
        expect(memory.pageReads.length).toBeLessThan(100);
        expect(memory.recordReads).not.toContain('blocks:block:recent:hidden');
        expect(memory.recordReads).not.toContain('turns:turn:recent:0');
        await expect(renderIndexedConversationRecentMessages(memory.store, staged.root, 101)).rejects.toThrow(
            IndexedPresentationCapacityError,
        );
        memory.recordReads.length = 0;
        expect(await renderIndexedConversationRecentMessages(memory.store, staged.root, 0)).toEqual([]);
        expect(memory.recordReads).toEqual([]);
    });

    it('filters private blocks and tool-only turns before selecting visible recent messages', async () => {
        const memory = memoryStore();
        const base = emptyDocument('conversation:recent-visibility');
        const call = toolCallBlock('block:mail:call', 'call:mail');
        call.arguments = { type: 'json', value: { password: 'private-argument' } };
        const toolOnly = AgentTurnSchema.parse({ ...userTurn('turn:mail:tool-only'), kind: 'agent', blocks: [call] });
        const answer = AgentTurnSchema.parse({
            ...userTurn('turn:mail:answer'),
            kind: 'agent',
            blocks: [
                textBlock('block:mail:answer', 'Visible answer'),
                { id: 'block:mail:reasoning', type: 'reasoning', text: 'private reasoning', representation: 'text' },
                { id: 'block:mail:json', type: 'json', value: { done: true } },
            ],
        });
        const accepted = appendConversationRecords(
            base,
            { turns: [userTurn('turn:mail:user'), toolOnly, answer] },
            {
                expected_revision: 0,
                operation_id: 'operation:mail:visible',
                payload_fingerprint: 'sha256:mail',
                recorded_at: RECORDED_AT,
            },
        );
        const staged = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        expect(await renderIndexedConversationRecentMessages(memory.store, staged.root, 2)).toEqual([
            { role: 'user', content: 'turn:mail:user-text' },
            { role: 'assistant', content: 'Visible answer\n{"done":true}' },
        ]);
        expect(await renderIndexedConversationRecentMessages(memory.store, staged.root, 1)).toEqual([
            { role: 'assistant', content: 'Visible answer\n{"done":true}' },
        ]);
    });

    it('renders bounded exact search content without private reasoning or non-transcript programs', async () => {
        const document = emptyDocument('conversation:indexed-lessons');
        const memory = memoryStore();
        const turn = userTurn('turn:search');
        const program = ProgramTurnSchema.parse({
            id: 'program:instruction',
            kind: 'program',
            authority: 'system',
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT },
            provenance: { type: 'received' },
            model_visibility: 'include',
            presentation: 'internal',
            blocks: [{ id: 'block:instruction', type: 'text', text: 'system instructions', format: 'plain' }],
        });
        const transcript = ProgramTurnSchema.parse({
            ...program,
            id: 'program:transcript',
            authority: 'ordinary',
            presentation: 'transcript',
            blocks: [
                { id: 'block:transcript', type: 'json', value: { done: true } },
                { id: 'block:private-reasoning', type: 'reasoning', text: 'private thought', representation: 'text' },
            ],
        });
        const accepted = appendConversationRecords(
            document,
            { turns: [turn, program, transcript] },
            {
                expected_revision: 0,
                operation_id: 'operation:search',
                payload_fingerprint: 'sha256:search',
                recorded_at: RECORDED_AT,
            },
        );
        const staged = await stageIndexedConversationSnapshot(accepted.document, undefined, memory.store);
        const originalPages = memory.pages.size;
        const originalRecords = memory.records.size;
        expect(await renderIndexedConversationSearchText(memory.store, staged.root)).toBe(
            '[USER]: turn:search-text\n\n[PROGRAM]: {"done":true}',
        );
        expect(await renderIndexedConversationTopicText(memory.store, staged.root)).toBe(
            renderConversationText(accepted.document),
        );
        expect(memory.pages.size).toBe(originalPages);
        expect(memory.records.size).toBe(originalRecords);
        await expect(
            renderIndexedConversationSearchText(memory.store, { ...staged.root, live_turn_count: 4097 }),
        ).rejects.toThrow(IndexedPresentationCapacityError);
        const record = [...memory.records.entries()].find(([key]) => key.startsWith('turns:'));
        if (!record) throw new Error('Exact original turn record absent');
        memory.records.set(record[0], new TextEncoder().encode('{}'));
        await expect(renderIndexedConversationSearchText(memory.store, staged.root)).rejects.toThrow();
    });

    it('renders exact historical live turns, including nested tool content, and rejects an unbounded profile', async () => {
        const memory = memoryStore();
        const original = emptyDocument('conversation:indexed-topic');
        const first = appendConversationRecords(
            original,
            { turns: [userTurn('turn:topic-user', 'block:topic-user')] },
            {
                expected_revision: original.revision,
                operation_id: 'operation:topic-user',
                payload_fingerprint: 'sha256:topic-user',
                recorded_at: RECORDED_AT,
            },
        );
        const staged = await stageIndexedConversationSnapshot(first.document, undefined, memory.store);
        expect(await renderIndexedConversationTopicText(memory.store, staged.root)).toBe(
            renderConversationText(first.document),
        );
        await expect(
            renderIndexedConversationTopicText(memory.store, {
                ...staged.root,
                live_turn_count: 4097,
            }),
        ).rejects.toThrow(IndexedPresentationCapacityError);
    });
});

describe('bounded indexed application call selection', () => {
    it('loads the exact accepted call/catalog and independently fences a later terminal result by point reads', async () => {
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:selected-call'),
            undefined,
            memory.store,
        );
        const commands = await toolMediaCommands(initial.root.source);
        const accepted = await stageIndexedRecordBatch(initial.root, commands.agentCommand, memory.store);
        const source = commands.execution.call_source;
        memory.recordReads.length = 0;
        const selected = await loadIndexedToolCallSelection(memory.store, accepted.root, source);
        expect(selected.source).toEqual(source);
        expect(selected.call).toEqual(commands.call);
        expect(selected.definition).toMatchObject({ id: commands.call.definition_id, name: commands.call.tool_name });
        expect(selected.assets).toEqual({});
        expect(memory.recordReads.length).toBeLessThan(16);
        expect(await loadIndexedToolCallTerminalResult(memory.store, accepted.root, source)).toBe(false);
        const terminal = await stageIndexedRecordBatch(accepted.root, commands.resultCommand, memory.store);
        expect(await loadIndexedToolCallTerminalResult(memory.store, terminal.root, source)).toBe(true);
        expect((await loadIndexedActiveToolDefinitions(memory.store, terminal.root)).map((entry) => entry.id)).toEqual([
            commands.call.definition_id,
        ]);
        await expect(loadIndexedToolCallSelection(memory.store, terminal.root, source)).rejects.toThrow(
            'exact complete accepted source',
        );
        for (const forged of [
            { ...source, call_fingerprint: `sha256:${'0'.repeat(64)}` },
            { ...source, turn_id: 'foreign:turn' },
            { ...source, conversation: { ...source.conversation, conversation_id: 'foreign:conversation' } },
        ]) {
            await expect(loadIndexedToolCallSelection(memory.store, accepted.root, forged)).rejects.toThrow();
            await expect(loadIndexedToolCallTerminalResult(memory.store, terminal.root, forged)).rejects.toThrow();
        }
        await expect(
            loadIndexedToolCallSelection(
                memory.store,
                {
                    ...accepted.root,
                    directories: {
                        ...accepted.root.directories,
                        generation_acceptances: initial.root.directories.generation_acceptances,
                    },
                },
                source,
            ),
        ).rejects.toThrow('exact accepted generation');
        await expect(
            loadIndexedToolCallSelection(
                memory.store,
                {
                    ...accepted.root,
                    context_header: initial.root.context_header,
                },
                source,
            ),
        ).rejects.toThrow('active catalog');
    });
});

describe('bounded indexed restart evidence', () => {
    it('retains original executed output and application source while incremental result acceptance clears the pending index', async () => {
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:restart'),
            undefined,
            memory.store,
        );
        expect(await loadIndexedRestartEvidence(memory.store, initial.root)).toEqual({
            kind: 'no_output',
            source: initial.root.source,
        });
        const commands = await toolMediaCommands(initial.root.source);
        const called = await stageIndexedRecordBatch(initial.root, commands.agentCommand, memory.store);
        memory.recordReads.length = 0;
        memory.pageReads.length = 0;
        const pending = await loadIndexedRestartEvidence(memory.store, called.root);
        if (pending.kind !== 'accepted_output') throw new Error('Expected authentic accepted restart output');
        expect(pending.accepted.source).toEqual(commands.execution.call_source.conversation);
        expect(pending.pending).toEqual([
            {
                source: commands.execution.call_source,
                call: {
                    call_id: commands.call.call_id,
                    tool_name: commands.call.tool_name,
                    definition_id: commands.call.definition_id,
                    executor: 'application',
                },
            },
        ]);
        expect(pending.materialized_input).toBeUndefined();
        expect(memory.recordReads.length).toBeLessThan(32);
        const completed = await stageIndexedRecordBatch(called.root, commands.resultCommand, memory.store);
        const resumed = await loadIndexedRestartEvidence(memory.store, completed.root);
        if (resumed.kind !== 'accepted_output') throw new Error('Expected authentic output after result');
        expect(resumed.accepted).toEqual(pending.accepted);
        expect(resumed.pending).toEqual([]);
        expect(resumed.materialized_input).toEqual({
            operation_id: commands.resultCommand.options.operation_id,
            result_revision: completed.root.source.revision,
        });
        expect(completed.root.directories.open_tool_calls).toBeUndefined();
        expect((await loadIndexedRestartEvidence(memory.store, called.root)).kind).toBe('accepted_output');
        const { restart_index_profile: _profile, ...oldProfile } = completed.root;
        await expect(loadIndexedRestartEvidence(memory.store, oldProfile)).rejects.toMatchObject({
            reason: 'upgrade_required',
        });
    });

    it.each(['imported', 'imported_turn', 'imported_without_generation', 'executed', 'tool'] as const)(
        'refuses mixed migration with an old valid output and a newer unnominated %s turn',
        async (kind) => {
            const document = emptyDocument(`conversation:restart-mixed:${kind}`);
            const commands = await toolMediaCommands({ conversation_id: document.id, revision: document.revision });
            const called = appendConversationRecords(
                document,
                commands.agentCommand.batch,
                commands.agentCommand.options,
            ).document;
            let mixed: ConversationDocument;
            let operationId: string;
            if (kind === 'tool') {
                mixed = appendConversationRecords(
                    called,
                    commands.resultCommand.batch,
                    commands.resultCommand.options,
                ).document;
                operationId = commands.resultCommand.options.operation_id;
            } else {
                const original = commands.agentCommand.batch.generations?.[0];
                if (original?.record_source !== 'executed') throw new Error('Expected real executed fixture');
                const source = { conversation_id: called.id, revision: called.revision };
                const generation =
                    kind !== 'executed'
                        ? { ...importedGeneration('generation:mixed'), source }
                        : {
                              ...original,
                              id: 'generation:mixed',
                              request_id: 'request:mixed',
                              attempt_id: 'attempt:mixed',
                              source,
                              request_receipt: {
                                  ...original.request_receipt,
                                  id: 'prepared:mixed',
                                  request_id: 'request:mixed',
                                  attempt_id: 'attempt:mixed',
                                  source,
                              },
                          };
                const { generation_id: _priorGenerationId, ...originalTurn } = commands.agent;
                const turn = AgentTurnSchema.parse({
                    ...originalTurn,
                    id: 'turn:mixed',
                    ...(kind === 'imported_without_generation' ? {} : { generation_id: generation.id }),
                    provenance:
                        kind === 'imported_turn' || kind === 'imported_without_generation'
                            ? { type: 'imported' as const, source: 'recorded-history' }
                            : commands.agent.provenance,
                    blocks: [textBlock('block:mixed', 'Newest raw generated response')],
                });
                const batch = {
                    ...(kind === 'imported_without_generation' ? {} : { generations: [generation] }),
                    turns: [turn],
                };
                operationId = 'operation:mixed';
                mixed = appendConversationRecords(called, batch, {
                    expected_revision: called.revision,
                    operation_id: operationId,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                }).document;
            }
            delete mixed.operation_receipts[operationId];
            // This is a valid older canonical snapshot, not corruption manufactured past parsing.
            const validated = parseConversationDocument(mixed);
            expect(validated.operation_receipts[commands.agentCommand.options.operation_id]).toBeDefined();
            const memory = memoryStore();
            const migrated = await stageIndexedConversationSnapshot(validated, undefined, memory.store);
            expect(migrated.root.restart_index_profile).toBeUndefined();
            expect(migrated.root.restart_response).toBeUndefined();
            await expect(loadIndexedRestartEvidence(memory.store, migrated.root)).rejects.toMatchObject({
                reason: 'upgrade_required',
            });
        },
    );

    it('authentically migrates later tool input and refuses to rewind over a newest imported acceptance', async () => {
        const memory = memoryStore();
        const document = emptyDocument('conversation:restart-migration');
        const commands = await toolMediaCommands({ conversation_id: document.id, revision: document.revision });
        const called = appendConversationRecords(
            document,
            commands.agentCommand.batch,
            commands.agentCommand.options,
        ).document;
        const completed = appendConversationRecords(
            called,
            commands.resultCommand.batch,
            commands.resultCommand.options,
        ).document;
        const migrated = await stageIndexedConversationSnapshot(completed, undefined, memory.store);
        const evidence = await loadIndexedRestartEvidence(memory.store, migrated.root);
        if (evidence.kind !== 'accepted_output') throw new Error('Expected migrated executed acceptance');
        expect(evidence.pending).toEqual([]);
        expect(evidence.materialized_input).toEqual({
            operation_id: commands.resultCommand.options.operation_id,
            result_revision: completed.revision,
        });
        const generation = { ...importedGeneration('generation:restart-imported'), source: migrated.root.source };
        const turn = {
            ...commands.agent,
            id: 'turn:restart-imported',
            generation_id: generation.id,
            blocks: [textBlock('block:restart-imported', 'Imported newest output')],
        };
        const batch = { generations: [generation], turns: [turn] };
        const recordsBeforeImportedAppend = memory.records.size;
        await expect(
            stageIndexedRecordBatch(
                migrated.root,
                {
                    conversation_id: document.id,
                    batch,
                    options: {
                        expected_revision: migrated.root.source.revision,
                        operation_id: 'operation:restart-imported',
                        recorded_at: RECORDED_AT,
                        payload_fingerprint: await fingerprintJson(batch),
                    },
                },
                memory.store,
            ),
        ).rejects.toThrow('Indexed append requires an executed generation');
        expect(memory.records.size).toBe(recordsBeforeImportedAppend);
        const importedDocument = appendConversationRecords(completed, batch, {
            expected_revision: completed.revision,
            operation_id: 'operation:restart-imported',
            recorded_at: RECORDED_AT,
            payload_fingerprint: await fingerprintJson(batch),
        }).document;
        const importedMigration = await stageIndexedConversationSnapshot(importedDocument, undefined, memory.store);
        expect(importedMigration.root.restart_index_profile).toBeDefined();
        expect(importedMigration.root.restart_response?.operation_id).toBe('operation:restart-imported');
        await expect(loadIndexedRestartEvidence(memory.store, importedMigration.root)).rejects.toMatchObject({
            reason: 'imported',
        });
        await expect(
            loadIndexedRestartEvidence(memory.store, {
                ...importedMigration.root,
                restart_response: {
                    operation_id: 'operation:missing',
                    result_revision: importedMigration.root.source.revision,
                },
            }),
        ).rejects.toThrow();
        const { generation_id: _importedGeneration, ...rawImportedTurn } = turn;
        const rawBatch = {
            turns: [
                {
                    ...rawImportedTurn,
                    id: 'turn:restart-imported-no-generation',
                    provenance: { type: 'imported' as const, source: 'recorded-history' },
                    blocks: [textBlock('block:restart-imported-no-generation', 'Unspecified imported newest response')],
                },
            ],
        };
        const rawImportedDocument = appendConversationRecords(importedDocument, rawBatch, {
            expected_revision: importedDocument.revision,
            operation_id: 'operation:restart-imported-no-generation',
            recorded_at: RECORDED_AT,
            payload_fingerprint: await fingerprintJson(rawBatch),
        }).document;
        const rawImported = await stageIndexedConversationSnapshot(rawImportedDocument, undefined, memory.store);
        // A legacy acceptance without its generation cannot establish complete restart evidence.
        expect(rawImported.root.restart_index_profile).toBeUndefined();
        expect(rawImported.root.restart_response).toBeUndefined();
        await expect(loadIndexedRestartEvidence(memory.store, rawImported.root)).rejects.toMatchObject({
            reason: 'upgrade_required',
        });
        const { restart_index_profile: _profile, ...incomplete } = importedMigration.root;
        await expect(loadIndexedRestartEvidence(memory.store, incomplete)).rejects.toThrow(
            IndexedRestartSourceUnavailable,
        );
    });
});

describe('indexed host program application calls', () => {
    const definition = { id: 'definition:program', name: 'read', version: '1', input_schema: { type: 'object' } };

    async function prepare() {
        const memory = memoryStore();
        const document = emptyDocument('process:canonical');
        document.tool_definitions[definition.id] = definition;
        document.context.active_tool_definition_ids = [definition.id];
        const initial = await stageIndexedConversationSnapshot(document, undefined, memory.store);
        const call = createProgramToolCall({
            id: 'block:program-call',
            call_id: 'call:program',
            tool_name: definition.name,
            definition_id: definition.id,
            arguments: { path: '/tmp/process-node' },
        });
        const turn = createProgramTurn({
            id: 'turn:program-call',
            authority: 'ordinary',
            blocks: [call],
            status: 'completed',
            presentation: 'internal',
            model_visibility: 'exclude',
            provenance: { type: 'inserted', operation_id: 'operation:program-call' },
            timestamps: { recorded_at: RECORDED_AT },
        });
        const batch: IndexedRecordBatchCommand['batch'] = {
            turns: [turn],
            context_entries: [{ id: 'entry:program-call', type: 'source_turn', turn_id: turn.id }],
        };
        const command: IndexedRecordBatchCommand = {
            conversation_id: document.id,
            batch,
            options: {
                operation_id: 'operation:program-call',
                expected_revision: 0,
                recorded_at: RECORDED_AT,
                payload_fingerprint: await fingerprintJson(batch),
            },
        };
        return { memory, initial, call, turn, command };
    }

    it('should accept a genuine inserted program operation with zero generated records', async () => {
        const { memory, initial, call, turn, command } = await prepare();
        await expect(stageIndexedRecordBatch(initial.root, command, memory.store)).rejects.toThrow(
            'accepted generated agent turn',
        );
        const accepted = await stageIndexedProgramToolCall(initial.root, command, memory.store);
        expect(accepted.receipt.accepted_generation_ids).toEqual([]);
        expect(accepted.root.directories.generations).toBeUndefined();
        const source = {
            conversation: accepted.root.source,
            turn_id: turn.id,
            block_id: call.id,
            call_id: call.call_id,
            call_fingerprint: await fingerprintJson(call),
        };
        memory.recordReads.length = 0;
        memory.pageReads.length = 0;
        const selected = await loadIndexedProgramToolCallSelection(memory.store, accepted.root, source);
        expect(selected.call).toEqual(call);
        expect(selected.definition).toEqual(definition);
        expect(selected.operation_receipt).toEqual(accepted.receipt);
        expect(memory.recordReads.some((read) => read.startsWith('generations:'))).toBe(false);
        expect(memory.recordReads.length).toBeLessThan(12);
        expect(memory.pageReads.length).toBeLessThan(100);
        await expect(loadIndexedToolCallSelection(memory.store, accepted.root, source)).rejects.toThrow(
            'generated content',
        );
        expect((await stageIndexedProgramToolCall(accepted.root, command, memory.store)).applied).toBe(false);
        expect(ProgramTurnSchema.safeParse({ ...turn, blocks: [{ ...call, executor: 'provider' }] }).success).toBe(
            false,
        );
        expect(ProgramTurnSchema.safeParse({ ...turn, generation_id: 'generation:invented' }).success).toBe(false);
    });

    it('should reject stale, foreign, revoked and forged program nominations', async () => {
        const { memory, initial, call, turn, command } = await prepare();
        const accepted = await stageIndexedProgramToolCall(initial.root, command, memory.store);
        const source = {
            conversation: accepted.root.source,
            turn_id: turn.id,
            block_id: call.id,
            call_id: call.call_id,
            call_fingerprint: await fingerprintJson(call),
        };
        for (const conversation of [
            { ...source.conversation, revision: 0 },
            { ...source.conversation, conversation_id: 'foreign' },
        ]) {
            await expect(
                loadIndexedProgramToolCallSelection(memory.store, accepted.root, { ...source, conversation }),
            ).rejects.toThrow('exact current source');
        }
        const batch = { active_tool_definition_ids: [] };
        const revoked = await stageIndexedRecordBatch(
            accepted.root,
            {
                conversation_id: accepted.root.source.conversation_id,
                batch,
                options: {
                    operation_id: 'operation:revoke-program',
                    expected_revision: 1,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            },
            memory.store,
        );
        await expect(
            loadIndexedProgramToolCallSelection(memory.store, revoked.root, {
                ...source,
                conversation: revoked.root.source,
            }),
        ).rejects.toThrow('active definition');
        const turnAcceptances = await putPagedRecord(
            memory.store,
            accepted.root.directories.turn_acceptances,
            turn.id,
            { storage: 'marker', kind: 'turn_acceptance', id: 'operation:revoke-program' },
            'replace',
        );
        await expect(
            loadIndexedProgramToolCallSelection(
                memory.store,
                { ...revoked.root, directories: { ...revoked.root.directories, turn_acceptances: turnAcceptances } },
                { ...source, conversation: revoked.root.source },
            ),
        ).rejects.toThrow('inserted operation proof');
        await expect(
            stageIndexedProgramToolCall(
                initial.root,
                { ...command, options: { ...command.options, payload_fingerprint: `sha256:${'0'.repeat(64)}` } },
                memory.store,
            ),
        ).rejects.toThrow('fingerprint differs');
    });

    it('should reject retained open-call proof when its original context entry is no longer active', async () => {
        const { memory, initial, call, turn, command } = await prepare();
        const accepted = await stageIndexedProgramToolCall(initial.root, command, memory.store);
        const source = {
            conversation: accepted.root.source,
            turn_id: turn.id,
            block_id: call.id,
            call_id: call.call_id,
            call_fingerprint: await fingerprintJson(call),
        };
        // Change only active order/header custody. Retained source entry, call state, acceptance,
        // original operation and definition remain present, so none can stand in for membership.
        const excluded = {
            ...accepted.root,
            context_header: initial.root.context_header,
            directories: {
                ...accepted.root.directories,
                active_context_order: initial.root.directories.active_context_order,
            },
        };
        await expect(loadIndexedProgramToolCallSelection(memory.store, excluded, source)).rejects.toThrow(
            'no longer active in context',
        );
    });

    it('should reject a legitimate attempt to exclude the unresolved program call dependency', async () => {
        const { memory, initial, turn, command } = await prepare();
        const accepted = await stageIndexedProgramToolCall(initial.root, command, memory.store);
        const document = emptyDocument('process:canonical');
        document.revision = 1;
        document.context.revision = 1;
        document.tool_definitions[definition.id] = definition;
        document.context.active_tool_definition_ids = [definition.id];
        document.turns = [turn];
        document.context.entries = [...(command.batch.context_entries ?? [])];
        document.operation_receipts[accepted.receipt.id] = accepted.receipt;
        await expect(
            planContextChange(document, {
                expected_revision: 1,
                expected_context_revision: 1,
                entry_ids: ['entry:program-call'],
            }),
        ).rejects.toThrow('pending tool call');
    });

    it('should reject a receipt whose empty accepted families or unchanged catalog are missing', async () => {
        const { memory, initial, call, turn, command } = await prepare();
        const accepted = await stageIndexedProgramToolCall(initial.root, command, memory.store);
        const source = {
            conversation: accepted.root.source,
            turn_id: turn.id,
            block_id: call.id,
            call_id: call.call_id,
            call_fingerprint: await fingerprintJson(call),
        };
        const descriptor = await getPagedRecord(
            memory.store,
            accepted.root.directories.operation_receipts,
            accepted.receipt.id,
        );
        if (descriptor?.storage !== 'record') throw new Error('Original program operation is missing');
        for (const field of [
            'accepted_generation_ids',
            'accepted_asset_ids',
            'accepted_execution_receipt_ids',
            'accepted_tool_definition_ids',
            'accepted_tool_selection',
        ] as const) {
            const receipt = { ...accepted.receipt };
            delete receipt[field];
            const bytes = canonicalJsonContentBytes(receipt);
            const integrity = await hashContentBytes(bytes);
            const forged = { ...descriptor, content_hash: integrity.content_hash, size_bytes: bytes.byteLength };
            await memory.store.writeRecord(forged, bytes);
            const operations = await putPagedRecord(
                memory.store,
                accepted.root.directories.operation_receipts,
                accepted.receipt.id,
                forged,
                'replace',
            );
            await expect(
                loadIndexedProgramToolCallSelection(
                    memory.store,
                    { ...accepted.root, directories: { ...accepted.root.directories, operation_receipts: operations } },
                    source,
                ),
            ).rejects.toThrow('inserted operation proof');
        }
    });

    it.each(['success', 'cancelled'] as const)(
        'should retain one %s terminal result and exact lost-ACK retry',
        async (status) => {
            const { memory, initial, call, turn, command } = await prepare();
            const accepted = await stageIndexedProgramToolCall(initial.root, command, memory.store);
            const source = {
                conversation: accepted.root.source,
                turn_id: turn.id,
                block_id: call.id,
                call_id: call.call_id,
                call_fingerprint: await fingerprintJson(call),
            };
            const resultBlock = {
                id: 'block:program-result',
                type: 'tool_result' as const,
                call_id: call.call_id,
                status,
                content: [
                    { id: 'block:program-result-text', type: 'text' as const, format: 'plain' as const, text: status },
                ],
            };
            const result = createToolTurn({
                id: 'turn:program-result',
                authority: 'ordinary',
                blocks: [resultBlock],
                status: 'completed',
                timestamps: { recorded_at: RECORDED_AT },
                execution_id: 'execution:program',
                model_visibility: 'exclude',
                provenance: { type: 'inserted', operation_id: 'execution:program' },
            });
            const batch: IndexedRecordBatchCommand['batch'] = {
                turns: [result],
                execution_receipts: [
                    {
                        id: 'execution:program',
                        call_id: call.call_id,
                        executor: 'application',
                        call_source: source,
                        result_turn_id: result.id,
                        status,
                        recorded_at: RECORDED_AT,
                        result_fingerprint: await fingerprintJson(resultBlock),
                    },
                ],
                context_entries: [{ id: 'entry:program-result', type: 'source_turn', turn_id: result.id }],
            };
            const completion: IndexedRecordBatchCommand = {
                conversation_id: source.conversation.conversation_id,
                batch,
                options: {
                    operation_id: 'operation:program-result',
                    expected_revision: 1,
                    recorded_at: RECORDED_AT,
                    payload_fingerprint: await fingerprintJson(batch),
                },
            };
            const completed = await stageIndexedRecordBatch(accepted.root, completion, memory.store);
            expect(await loadIndexedToolCallTerminalResult(memory.store, completed.root, source)).toBe(true);
            expect((await stageIndexedRecordBatch(completed.root, completion, memory.store)).applied).toBe(false);
            expect((await stageIndexedProgramToolCall(completed.root, command, memory.store)).applied).toBe(false);
            await expect(
                loadIndexedProgramToolCallSelection(memory.store, completed.root, {
                    ...source,
                    conversation: completed.root.source,
                }),
            ).rejects.toThrow('open-call index');
            expect(completed.root.directories.generations).toBeUndefined();
        },
    );
});
