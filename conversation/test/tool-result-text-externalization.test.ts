import { describe, expect, it } from 'vitest';
import {
    appendConversationRecordsWithProcessing,
    appendToolExecutionResult,
    applyToolResultTextExternalizationOutput,
    type ConversationDocument,
    ConversationToolExecutionResultSchema,
    createConversationDocument,
    createTextExternalizationProcessor,
    createToolResultTextExternalizationProcessor,
    fingerprintJson,
    isToolResultTextProcessor,
    type NativeReplayBlock,
    type ProcessingStore,
    parseConversationDocument,
    resolveActiveTextExternalReference,
    runProcessingJob,
    setProcessingPolicy,
    toolResultExternalizationArchiveInputs,
    toolResultTextSelection,
} from '../src/index.js';

const at = '2026-10-04T00:00:00.000Z';
const originalText = `Exact result α\n${'long result '.repeat(1000)}`;

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

async function completedResult(withTextPredecessor = false, withCallReplay = false) {
    const initial: ConversationDocument = createConversationDocument({ id: 'tool-result-text', created_at: at });
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
    const configured = (
        await setProcessingPolicy(initial, {
            operation_id: 'policy:tool-results',
            expected_revision: 0,
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
                    id: 'externalize-tool-result-text',
                    version: '1',
                    scope: 'on_append',
                    config: {},
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
        content: [{ id: 'block:text', type: 'text' as const, format: 'plain' as const, text: originalText }],
    };
    const result = ConversationToolExecutionResultSchema.parse({
        source,
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
    const accepted = await appendToolExecutionResult(configured, result, {
        expected_revision: configured.revision,
        operation_id: 'append:result',
        recorded_at: at,
    });
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
                active_tool_definition_ids: ['definition:read'],
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
