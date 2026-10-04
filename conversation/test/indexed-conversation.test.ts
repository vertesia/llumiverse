import { describe, expect, it } from 'vitest';
import { createUserTurn } from '../src/builders.js';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { applyContextChange, planContextChange } from '../src/context-change.js';
import { resolveIndexedTextExternalReference } from '../src/external-reference-retrieval.js';
import { fingerprintJson } from '../src/identity.js';
import {
    assertIndexedFreshReceivedTextInput,
    assertIndexedFreshTextInput,
    type IndexedConversationRecordStore,
    type IndexedRecordBatchCommand,
    loadIndexedActiveContext,
    loadIndexedProjectedTurn,
    loadIndexedSelectedDependencyContext,
    loadIndexedSelectedMediaCompactionContext,
    loadIndexedSelectedTextContext,
    stageIndexedConversationDelete,
    stageIndexedConversationSnapshot,
    stageIndexedProgramAppend,
    stageIndexedRecordBatch,
} from '../src/indexed-conversation.js';
import { preflightJsonInput } from '../src/json-preflight.js';
import { getPagedRecord } from '../src/paged-record-index.js';
import {
    assertProcessingReady,
    ProcessingKnownFailure,
    runProcessingJob,
    setProcessingPolicy,
} from '../src/processing.js';
import { appendConversationRecords, appendConversationRecordsWithProcessing } from '../src/runtime.js';
import { ProgramTurnSchema } from '../src/schemas/content.js';
import { GenerationSchema } from '../src/schemas/execution.js';
import { validateToolExecutionResult } from '../src/tool-execution.js';
import type { ConversationDocument } from '../src/types.js';
import { parseConversationDocument } from '../src/validation.js';
import { emptyDocument, RECORDED_AT, textBlock, toolCallBlock, toolResultTurn, userTurn } from './fixtures.js';

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
    return { store, recordReads, pageReads, records };
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
    it('rejects unsupported retrieval requirements before indexed record publication', async () => {
        const memory = memoryStore();
        const staged = await stageIndexedConversationSnapshot(
            emptyDocument('conversation:indexed-retrieval-rejection'),
            undefined,
            memory.store,
        );
        const readsBefore = memory.recordReads.length;
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
        ).rejects.toThrow('does not yet accept new retrieval requirements');
        expect(memory.recordReads).toHaveLength(readsBefore);
        expect(memory.records.size).toBe(recordsBefore);
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
        ).rejects.toThrow('retained dependent record');
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
        ).rejects.toThrow('dependent record');
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
        ).rejects.toThrow('dependent record');
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
        expect(
            await getPagedRecord(memory.store, stagedResult.root.directories.open_tool_calls, call.call_id),
        ).toMatchObject({ storage: 'marker', kind: 'closed_tool_call', id: call.call_id });
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

    it('retains materialized processing job-drain parity when disabled policy hides pending or blocked work', async () => {
        const initial = emptyDocument('conversation:indexed-processing');
        const enabled = await setProcessingPolicy(initial, {
            operation_id: 'policy:enable',
            expected_revision: initial.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                {
                    id: 'processor:required',
                    version: 'v1',
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
        const assertIndexedBlocked = async (source: ConversationDocument, expectedCount: number) => {
            await expect(assertProcessingReady(source, '', '')).rejects.toThrow(
                'Accepted processing jobs remain outstanding',
            );
            const memory = memoryStore();
            const staged = await stageIndexedConversationSnapshot(source, undefined, memory.store);
            const headerBytes = memory.records.get(`processing_header:${staged.root.processing_header.content_hash}`);
            if (!headerBytes) throw new Error('Indexed processing header was not retained');
            expect(JSON.parse(new TextDecoder().decode(headerBytes)).unresolved_job_count).toBe(expectedCount);
            const recordCount = memory.records.size;
            await expect(loadIndexedSelectedTextContext(memory.store, staged.root, staged.locator)).rejects.toThrow(
                'accepted processing jobs outstanding',
            );
            expect(memory.records.size).toBe(recordCount);
            return { memory, staged };
        };
        const pending = disabledWithoutSupersession(accepted.document);
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
            'no accepted processing job-drain witness',
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
        await assertIndexedBlocked(disabledWithoutSupersession(current), 1);

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
