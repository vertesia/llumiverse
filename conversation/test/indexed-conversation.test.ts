import { describe, expect, it } from 'vitest';
import { createUserTurn } from '../src/builders.js';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { fingerprintJson } from '../src/identity.js';
import {
    type IndexedConversationRecordStore,
    loadIndexedActiveContext,
    loadIndexedProjectedTurn,
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

describe('indexed conversation snapshot', () => {
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

    it('migrates valid 10k and 100k cold turns, then appends with fixed active context and bounded reads', async () => {
        const pageReadCounts: number[] = [];
        const deleteReadCounts: number[] = [];
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
        }
        expect(pageReadCounts[1]).toBeLessThan(pageReadCounts[0] + 20);
        expect(deleteReadCounts[1]).toBeLessThan(deleteReadCounts[0] + 40);
    }, 60_000);
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
