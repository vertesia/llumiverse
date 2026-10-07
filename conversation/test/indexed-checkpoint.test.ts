import { describe, expect, it } from 'vitest';
import {
    buildSelectedCheckpointRequest,
    type IndexedCheckpointSummaryCommand,
} from '../src/checkpoint-context-change.js';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { materializedContextChangeWorkingSet, planContextChangeWorkingSet } from '../src/context-change-working-set.js';
import { fingerprintJson } from '../src/identity.js';
import {
    type IndexedConversationRecordStore,
    loadIndexedActiveContext,
    loadIndexedPendingProcessingJobs,
    loadIndexedProjectedTurn,
    stageIndexedCheckpointSummary,
    stageIndexedConversationSnapshot,
    stageIndexedRecordBatch,
} from '../src/indexed-conversation.js';
import { getPagedRecord } from '../src/paged-record-index.js';
import { appendConversationRecords } from '../src/runtime.js';
import type { Asset, ConversationDocument, ConversationRecordBatch, NativeReplayBlock } from '../src/types.js';
import { parseConversationDocument } from '../src/validation.js';
import { emptyDocument, generatedAgentTurn, RECORDED_AT, toolCallBlock, toolResultTurn, userTurn } from './fixtures.js';

function memoryStore() {
    const pages = new Map<string, Uint8Array>();
    const records = new Map<string, Uint8Array>();
    const reads: string[] = [];
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const value = pages.get(ref.content_hash);
            if (!value) throw new Error('Immutable page unavailable');
            return Uint8Array.from(value);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            reads.push(`${ref.kind}:${ref.id}`);
            const value = records.get(`${ref.kind}:${ref.content_hash}`);
            if (!value) throw new Error('Immutable record unavailable');
            return Uint8Array.from(value);
        },
        async writeRecord(ref, bytes) {
            expect((await hashContentBytes(bytes)).content_hash).toBe(ref.content_hash);
            records.set(`${ref.kind}:${ref.content_hash}`, Uint8Array.from(bytes));
        },
    };
    return { store, reads, records, pages };
}
async function sourceFixture(coldCount = 0): Promise<ConversationDocument> {
    const source = emptyDocument('conversation:checkpoint');
    const active = userTurn('turn:task');
    active.blocks.push({ id: 'block:active-image', type: 'image', asset_id: 'asset:active' });
    const protectedTurn = { ...userTurn('turn:system'), authority: 'system' as const };
    const call = toolCallBlock('block:pending', 'call:pending');
    const pending = generatedAgentTurn('turn:pending', 'generation:pending', [call]);
    const integrity = await hashContentBytes(new Uint8Array([1, 2, 3]));
    const asset: Asset = {
        id: 'asset:active',
        kind: 'image',
        mime_type: 'image/png',
        storage: { type: 'inline_base64', data: 'AQID' },
        created_at: RECORDED_AT,
        ...integrity,
        provenance: { type: 'received', source_turn_id: active.id },
    };
    source.turns = [protectedTurn, active];
    source.assets[asset.id] = asset;
    source.context.entries = source.turns.map((turn) => ({
        id: `entry:${turn.id}`,
        type: 'source_turn',
        turn_id: turn.id,
    }));
    source.context.protected_entry_ids = ['entry:turn:system'];
    for (let i = 0; i < coldCount; i++) {
        const cold = userTurn(`turn:cold:${i}`);
        source.turns.push(cold);
        const coldAsset = {
            ...asset,
            id: `asset:cold:${i}`,
            provenance: { type: 'received' as const, source_turn_id: cold.id },
        };
        source.assets[coldAsset.id] = coldAsset;
    }
    const input = parseConversationDocument(source);
    const acceptedSource = { conversation_id: input.id, revision: input.revision };
    const target = { provider: 'test', protocol: 'test.generate', model: 'test-model', adapter_version: '1' };
    const batch: ConversationRecordBatch = {
        turns: [pending],
        generations: [
            {
                id: pending.generation_id,
                record_source: 'executed',
                request_id: 'request:pending',
                attempt_id: 'attempt:pending',
                purpose: 'conversation',
                requested_model: target.model,
                provider: target.provider,
                protocol: target.protocol,
                adapter_version: target.adapter_version,
                status: 'completed',
                timestamps: { recorded_at: RECORDED_AT },
                source: acceptedSource,
                request_receipt: {
                    id: 'receipt:pending-request',
                    request_id: 'request:pending',
                    attempt_id: 'attempt:pending',
                    source: acceptedSource,
                    context_fingerprint: await fingerprintJson(input.context),
                    tool_set_fingerprint: await fingerprintJson([]),
                    request_fingerprint: await fingerprintJson({ source: acceptedSource, target }),
                    target,
                    tool_definition_ids: [],
                    asset_versions: [],
                    item_mappings: [],
                    recorded_at: RECORDED_AT,
                },
            },
        ],
        context_entries: [{ id: 'entry:turn:pending', type: 'source_turn', turn_id: pending.id }],
    };
    return appendConversationRecords(input, batch, {
        expected_revision: input.revision,
        operation_id: 'append:accepted-pending-call',
        payload_fingerprint: await fingerprintJson(batch),
        recorded_at: RECORDED_AT,
    }).document;
}
function command(source: ConversationDocument): IndexedCheckpointSummaryCommand {
    return {
        source: { conversation_id: source.id, revision: source.revision },
        operation_id: 'checkpoint:ordinary',
        summary: 'Retain the current task and pending original call.',
        recorded_at: RECORDED_AT,
    };
}

describe('bounded ordinary indexed checkpoints', () => {
    it('preserves protected/pending calls and raw history while nominating only selected assets under version two', async () => {
        const source = await sourceFixture(256);
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        memory.reads.length = 0;
        const staged = await stageIndexedCheckpointSummary(
            memory.store,
            initial.root,
            initial.locator,
            command(source),
        );
        expect(staged.applied).toBe(true);
        expect(staged.receipt.operation_kind).toBe('context_change');
        expect(staged.receipt.context_change?.removed_entry_ids).toEqual(['entry:turn:task']);
        expect(staged.compaction.strategy.version).toBe('2');
        expect(staged.compaction.retained_asset_ids).toEqual(['asset:active']);
        expect(staged.root.turn_count).toBe(initial.root.turn_count);
        expect(staged.root.processing_header).toEqual(initial.root.processing_header);
        expect(staged.root.directories.assets).toEqual(initial.root.directories.assets);
        expect([...new Set(memory.reads.filter((value) => value.startsWith('assets:')))]).toEqual([
            'assets:asset:active',
        ]);
        expect(memory.reads.some((value) => value.includes(':cold:'))).toBe(false);
        expect(memory.reads.length).toBeLessThan(128);
        const initialProcessing = await loadIndexedPendingProcessingJobs(memory.store, initial.root);
        const stagedProcessing = await loadIndexedPendingProcessingJobs(memory.store, staged.root);
        expect(initialProcessing).toMatchObject({ required_job_count: 0, unresolved_job_count: 0, jobs: [] });
        expect(stagedProcessing).toEqual({ ...initialProcessing, source: staged.root.source });
        const current = await loadIndexedActiveContext(memory.store, staged.root);
        expect(current.entries.map((entry) => entry.turn_id)).toEqual([
            'turn:system',
            staged.compaction.replacement_turns[0].id,
            'turn:pending',
        ]);
        expect(await loadIndexedProjectedTurn(memory.store, staged.root, 'turn:task')).toMatchObject({
            header: { id: 'turn:task' },
            selected_blocks: source.turns[1].blocks,
        });
        expect(await getPagedRecord(memory.store, staged.root.directories.assets, 'asset:cold:255')).toBeDefined();
    });

    it.each([false, true])(
        'preserves replay authority and retries exact partial checkpoint selection (protected=%s)',
        async (protectedReplay) => {
            const original = await sourceFixture();
            const text = {
                id: 'block:replay-text',
                type: 'text' as const,
                text: 'Actual accepted answer',
                format: 'plain' as const,
            };
            const replay: NativeReplayBlock = {
                id: 'block:replay-wire',
                type: 'native_replay',
                adapter: 'test',
                protocol: 'test.generate',
                compatibility_scope: { provider: 'test', protocol: 'test.generate', adapter_version: '1' },
                payload: { original: 'opaque' },
                dependencies: {
                    turn_ids: [],
                    block_ids: [protectedReplay ? 'turn:task-text' : text.id],
                    call_ids: [],
                    request_ids: [],
                },
                ...(protectedReplay ? {} : { dependency_policy: 'discard_on_dependency_change' as const }),
            };
            const answer = {
                ...userTurn('turn:replay-answer'),
                kind: 'agent' as const,
                provenance: { type: 'received' as const },
                blocks: [text, replay],
            };
            const extra = userTurn('turn:eligible-extra');
            const batch: ConversationRecordBatch = {
                turns: [answer, extra],
                context_entries: [answer, extra].map((turn) => ({
                    id: `entry:${turn.id}`,
                    type: 'source_turn',
                    turn_id: turn.id,
                })),
            };
            const source = appendConversationRecords(original, batch, {
                expected_revision: original.revision,
                operation_id: 'append:replay-answer',
                payload_fingerprint: await fingerprintJson(batch),
                recorded_at: RECORDED_AT,
            }).document;
            const memory = memoryStore();
            const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
            const input = command(source);
            const frame = materializedContextChangeWorkingSet(source);
            await expect(
                planContextChangeWorkingSet(frame, {
                    expected_revision: source.revision,
                    expected_context_revision: source.context.revision,
                    entry_ids: ['entry:turn:replay-answer'],
                }),
            ).rejects.toThrow('protected native replay unit');
            const request = await buildSelectedCheckpointRequest(frame, input);
            const staged = await stageIndexedCheckpointSummary(memory.store, initial.root, initial.locator, input);
            expect(staged.receipt.payload_fingerprint).toBe(await fingerprintJson(request));
            expect(staged.receipt.context_change?.removed_entry_ids).toEqual(
                protectedReplay
                    ? ['entry:turn:eligible-extra']
                    : ['entry:turn:task', 'entry:turn:replay-answer', 'entry:turn:eligible-extra'],
            );
            expect(staged.receipt.context_change?.selected_block_ids).toEqual(
                protectedReplay ? undefined : { 'entry:turn:replay-answer': [text.id] },
            );
            expect(staged.receipt.context_change?.discarded_replay_block_ids).toEqual(
                protectedReplay ? undefined : [replay.id],
            );
            const selected = await loadIndexedActiveContext(memory.store, staged.root);
            expect(selected.entries.some((entry) => entry.turn_id === answer.id)).toBe(protectedReplay);
            expect(await loadIndexedProjectedTurn(memory.store, staged.root, answer.id)).toMatchObject({
                selected_blocks: [text, replay],
            });
            const laterTurn = userTurn('turn:after-replay-checkpoint');
            const laterBatch: ConversationRecordBatch = { turns: [laterTurn] };
            const later = await stageIndexedRecordBatch(
                staged.root,
                {
                    conversation_id: staged.root.source.conversation_id,
                    batch: laterBatch,
                    options: {
                        expected_revision: staged.root.source.revision,
                        operation_id: 'append:after-replay-checkpoint',
                        payload_fingerprint: await fingerprintJson(laterBatch),
                        recorded_at: RECORDED_AT,
                    },
                },
                memory.store,
            );
            if (!later.locator) throw new Error('Expected actual later append locator');
            const retry = await stageIndexedCheckpointSummary(memory.store, later.root, later.locator, input);
            expect(retry.applied).toBe(false);
            expect(retry.receipt).toEqual(staged.receipt);
            expect(retry.root).toEqual(later.root);
            await expect(
                stageIndexedCheckpointSummary(memory.store, later.root, later.locator, {
                    ...input,
                    summary: 'Changed',
                }),
            ).rejects.toThrow('exact accepted selected mutation');
        },
    );

    it('preserves the closed original exchange when protected replay depends on nested result content', async () => {
        const original = await sourceFixture();
        const call = toolCallBlock('block:pending', 'call:pending');
        const result = {
            ...toolResultTurn('turn:nested-result', call.call_id),
            execution_id: 'execution:nested-result',
        };
        const replay: NativeReplayBlock = {
            id: 'block:nested-dependent-replay',
            type: 'native_replay',
            adapter: 'test',
            protocol: 'test.generate',
            compatibility_scope: { provider: 'test', protocol: 'test.generate', adapter_version: '1' },
            payload: { original: 'protected' },
            dependencies: { turn_ids: [], block_ids: ['turn:nested-result-content'], call_ids: [], request_ids: [] },
        };
        const replayTurn = {
            ...userTurn('turn:nested-replay'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [replay],
        };
        const batch: ConversationRecordBatch = {
            turns: [result, replayTurn],
            execution_receipts: [
                {
                    id: result.execution_id,
                    call_id: call.call_id,
                    executor: 'application',
                    status: 'success',
                    result_turn_id: result.id,
                    result_fingerprint: await fingerprintJson(result.blocks[0]),
                    recorded_at: RECORDED_AT,
                    call_source: {
                        conversation: { conversation_id: original.id, revision: original.revision },
                        turn_id: 'turn:pending',
                        block_id: call.id,
                        call_id: call.call_id,
                        call_fingerprint: await fingerprintJson(call),
                    },
                },
            ],
            context_entries: [result, replayTurn].map((turn) => ({
                id: `entry:${turn.id}`,
                type: 'source_turn',
                turn_id: turn.id,
            })),
        };
        const source = appendConversationRecords(original, batch, {
            expected_revision: original.revision,
            operation_id: 'append:nested-replay-result',
            payload_fingerprint: await fingerprintJson(batch),
            recorded_at: RECORDED_AT,
        }).document;
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const staged = await stageIndexedCheckpointSummary(
            memory.store,
            initial.root,
            initial.locator,
            command(source),
        );
        expect(staged.receipt.context_change?.removed_entry_ids).toEqual(['entry:turn:task']);
        const selected = await loadIndexedActiveContext(memory.store, staged.root);
        expect(selected.entries.map((entry) => entry.turn_id)).toContain('turn:pending');
        expect(selected.entries.map((entry) => entry.turn_id)).toContain(result.id);
        expect(selected.entries.map((entry) => entry.turn_id)).toContain(replayTurn.id);
        expect(staged.receipt.context_change?.discarded_replay_block_ids).toBeUndefined();
    });

    it('replays the exact original receipt after later head changes and rejects changed summary/time/source', async () => {
        const source = await sourceFixture();
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        const input = command(source);
        const first = await stageIndexedCheckpointSummary(memory.store, initial.root, initial.locator, input);
        const later = await stageIndexedRecordBatch(
            first.root,
            {
                conversation_id: source.id,
                batch: { turns: [userTurn('turn:later')] },
                options: {
                    expected_revision: first.root.source.revision,
                    operation_id: 'append:later',
                    payload_fingerprint: 'hash:later',
                    recorded_at: RECORDED_AT,
                },
            },
            memory.store,
        );
        if (!later.applied || !later.locator) throw new Error('Expected a freshly staged later append locator');
        const laterLocator = later.locator;
        const retry = await stageIndexedCheckpointSummary(memory.store, later.root, laterLocator, input);
        expect(retry.applied).toBe(false);
        expect(retry.root).toEqual(later.root);
        expect(retry.receipt).toEqual(first.receipt);
        expect(retry.compaction).toEqual(first.compaction);
        for (const changed of [
            { ...input, summary: 'Changed paid summary' },
            { ...input, recorded_at: '2026-10-06T00:00:00.000Z' },
            { ...input, source: { ...input.source, revision: 7 } },
        ])
            await expect(
                stageIndexedCheckpointSummary(memory.store, later.root, laterLocator, changed),
            ).rejects.toThrow();
    });

    it('fails on foreign/stale roots, old incomplete profiles and global identifier collisions before root publication', async () => {
        const source = await sourceFixture();
        const memory = memoryStore();
        const initial = await stageIndexedConversationSnapshot(source, undefined, memory.store);
        await expect(
            stageIndexedCheckpointSummary(memory.store, initial.root, initial.locator, {
                ...command(source),
                source: { conversation_id: 'foreign', revision: 0 },
            }),
        ).rejects.toThrow();
        await expect(
            stageIndexedCheckpointSummary(memory.store, initial.root, initial.locator, {
                ...command(source),
                source: { conversation_id: source.id, revision: 9 },
            }),
        ).rejects.toThrow();
        const { processing_index_profile: _profile, ...oldRoot } = initial.root;
        await expect(
            stageIndexedCheckpointSummary(memory.store, oldRoot, initial.locator, command(source)),
        ).rejects.toThrow(/complete processing/);
        const request = await buildSelectedCheckpointRequest(
            materializedContextChangeWorkingSet(source),
            command(source),
        );
        if (request.proposal.kind !== 'replace_with_compaction') throw new Error('Expected checkpoint proposal');
        const collision = structuredClone(source);
        collision.turns.push(userTurn(request.proposal.replacement_turns[0].id));
        const collided = await stageIndexedConversationSnapshot(
            parseConversationDocument(collision),
            undefined,
            memory.store,
        );
        await expect(
            stageIndexedCheckpointSummary(memory.store, collided.root, collided.locator, command(collision)),
        ).rejects.toThrow(/identity already belongs/);
        expect(initial.root.source.revision).toBe(source.revision);
    });

    it('owns selected inputs before awaiting hashes', async () => {
        const source = await sourceFixture();
        const frame = materializedContextChangeWorkingSet(source);
        const input = command(source);
        const operation = buildSelectedCheckpointRequest(frame, input);
        input.summary = 'mutated';
        source.context.entries.length = 0;
        source.assets['asset:active'].storage = { type: 'inline_base64', data: 'AAAA' };
        const request = await operation;
        expect(request.entry_ids).toEqual(['entry:turn:task']);
        if (request.proposal.kind !== 'replace_with_compaction') throw new Error('Expected checkpoint');
        expect(request.proposal.replacement_turns[0].blocks[0]).toMatchObject({
            text: 'Retain the current task and pending original call.',
        });
        expect(canonicalJsonContentBytes(request).byteLength).toBeLessThan(16 * 1024);
    });
});
