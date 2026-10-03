import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    appendConversationRecordsWithProcessing,
    applyContextChange,
    applyConversationDelete,
    type ConversationDeletePlanInput,
    type ConversationDeleteRequest,
    fingerprintJson,
    type IndexedConversationRecordStore,
    parseConversationDocument,
    planContextChange,
    planConversationDelete,
    setProcessingPolicy,
    stageIndexedConversationSnapshot,
} from '../src/index.js';
import { emptyDocument, importedGeneration, textBlock, toolCallBlock, userTurn } from './fixtures.js';

const at = '2026-10-03T00:00:00.000Z';
const later = '2026-10-03T00:01:00.000Z';

function appendSource(active = true) {
    const turn = userTurn('source');
    const batch = {
        turns: [turn],
        ...(active
            ? { context_entries: [{ id: 'entry:source', type: 'source_turn' as const, turn_id: turn.id }] }
            : {}),
    };
    const options = {
        operation_id: 'append:source',
        expected_revision: 0,
        payload_fingerprint: 'sha256:source-append',
        recorded_at: at,
    };
    return { turn, batch, options, document: appendConversationRecords(emptyDocument(), batch, options).document };
}

async function excludeSource(document: ReturnType<typeof appendSource>['document']) {
    const plan = await planContextChange(document, {
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        entry_ids: ['entry:source'],
    });
    return (
        await applyContextChange(document, {
            operation_id: 'exclude:source',
            expected_revision: document.revision,
            expected_context_revision: document.context.revision,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: at,
            entry_ids: ['entry:source'],
            proposal: { kind: 'exclude' },
        })
    ).document;
}

async function deleteRequest(
    document: ReturnType<typeof appendSource>['document'],
): Promise<ConversationDeleteRequest> {
    const input: ConversationDeletePlanInput = {
        version: 1,
        operation_id: 'delete:source',
        conversation: { conversation_id: document.id, revision: document.revision },
        recorded_at: later,
        dependency_policy: 'reject',
        turn_ids: ['source'],
    };
    const plan = await planConversationDelete(document, input);
    return { ...input, expected_source_fingerprint: plan.operation.source_fingerprint };
}

describe('logical source-turn deletion', () => {
    it('removes the body after exclusion, preserves accepted receipts and usage, and retries after a later revision', async () => {
        const source = appendSource();
        const excluded = await excludeSource(source.document);
        const imported = importedGeneration('history:usage');
        imported.source = { conversation_id: excluded.id, revision: excluded.revision };
        imported.usage = {
            input_tokens: 7,
            accounting_provenance: { input_tokens: { method: 'reported', accounting_basis: 'provider' } },
        };
        const withUsage = parseConversationDocument({
            ...excluded,
            generations: { ...excluded.generations, [imported.id]: imported },
        });
        const request = await deleteRequest(withUsage);
        const result = await applyConversationDelete(withUsage, request);
        expect(result.applied).toBe(true);
        expect(result.document.turns).toEqual([]);
        expect(result.document.context.entries).toEqual([]);
        expect(result.document.deleted_turns?.source).toMatchObject({
            id: 'source',
            operation_id: 'delete:source',
            accepted_operation_id: 'append:source',
            block_ids: ['source-text'],
        });
        expect(result.document.operation_receipts['append:source']).toEqual(
            withUsage.operation_receipts['append:source'],
        );
        expect(result.document.generations[imported.id]?.usage).toEqual(imported.usage);
        expect(result.change).toMatchObject({
            operation_id: 'delete:source',
            base_revision: withUsage.revision,
            result_revision: withUsage.revision + 1,
            operations: [{ dependency_policy: 'reject', deleted_turns: [{ id: 'source' }] }],
        });
        const reloaded = parseConversationDocument(JSON.parse(JSON.stringify(result.document)));
        const advanced = appendConversationRecords(
            reloaded,
            { turns: [userTurn('later')] },
            {
                operation_id: 'append:later',
                expected_revision: reloaded.revision,
                payload_fingerprint: 'sha256:later-append',
                recorded_at: later,
            },
        ).document;
        const retry = await applyConversationDelete(advanced, request);
        expect(retry.applied).toBe(false);
        expect(retry.change).toEqual(result.change);
        expect(retry.document.turns.map((turn) => turn.id)).toEqual(['later']);
        expect(() => appendConversationRecords(advanced, source.batch, source.options)).toThrow(
            'authenticated predecessor',
        );
    });

    it('rejects active, parent, and tool dependencies without deleting anything', async () => {
        const source = appendSource();
        await expect(deleteRequest(source.document)).rejects.toThrow('/context/entries/0');
        const excluded = await excludeSource(source.document);
        const child = userTurn('child');
        child.parent_turn_id = 'source';
        const parentDependent = parseConversationDocument({ ...excluded, turns: [...excluded.turns, child] });
        await expect(deleteRequest(parentDependent)).rejects.toThrow('/turns/1/parent_turn_id');
        const tool = structuredClone(excluded);
        tool.turns = [
            {
                ...userTurn('source'),
                kind: 'agent',
                provenance: { type: 'received' },
                blocks: [toolCallBlock('call:block', 'call:id')],
            },
        ];
        await expect(deleteRequest(parseConversationDocument(tool))).rejects.toThrow('/turns/0/blocks');
        expect(excluded.turns[0].id).toBe('source');
    });

    it('rejects changed requests, tampered tombstones, and reuse of deleted IDs', async () => {
        const excluded = await excludeSource(appendSource().document);
        const request = await deleteRequest(excluded);
        const result = await applyConversationDelete(excluded, request);
        await expect(applyConversationDelete(result.document, { ...request, turn_ids: ['different'] })).rejects.toThrow(
            'conflicts',
        );
        const tampered = structuredClone(result.document);
        if (!tampered.deleted_turns?.source) throw new Error('Expected tombstone');
        tampered.deleted_turns.source.fingerprint = await fingerprintJson({ counterfeit: true });
        expect(() => parseConversationDocument(tampered)).toThrow();
        const wrongRevision = structuredClone(result.document);
        if (!wrongRevision.deleted_turns?.source) throw new Error('Expected tombstone');
        wrongRevision.deleted_turns.source.source_revision += 1;
        expect(() => parseConversationDocument(wrongRevision)).toThrow();
        const fakeArchive = structuredClone(result.document);
        fakeArchive.operation_receipts['delete:source'].accepted_context_entries = [
            { id: 'entry:source', type: 'source_turn', turn_id: 'source' },
        ];
        expect(() => parseConversationDocument(fakeArchive)).toThrow();
        const fakeToolSelection = structuredClone(result.document);
        fakeToolSelection.operation_receipts['delete:source'].accepted_tool_selection = {
            kind: 'replace',
            definition_ids: [],
        };
        expect(() => parseConversationDocument(fakeToolSelection)).toThrow();
        expect(() =>
            appendConversationRecords(
                result.document,
                { turns: [userTurn('source')] },
                {
                    operation_id: 'append:duplicate',
                    expected_revision: result.document.revision,
                    payload_fingerprint: 'sha256:duplicate',
                    recorded_at: later,
                },
            ),
        ).toThrow();
        expect(() =>
            appendConversationRecords(
                result.document,
                { turns: [{ ...userTurn('other'), blocks: [textBlock('source-text')] }] },
                {
                    operation_id: 'append:duplicate-block',
                    expected_revision: result.document.revision,
                    payload_fingerprint: 'sha256:duplicate-block',
                    recorded_at: later,
                },
            ),
        ).toThrow();
    });

    it('requires exactly one prior accepted append witness for each removed turn', async () => {
        const excluded = await excludeSource(appendSource().document);
        const missing = structuredClone(excluded);
        delete missing.operation_receipts['append:source'];
        await expect(deleteRequest(parseConversationDocument(missing))).rejects.toThrow('one accepted append');
        const duplicated = structuredClone(excluded);
        duplicated.operation_receipts['append:other'] = {
            ...duplicated.operation_receipts['append:source'],
            id: 'append:other',
        };
        await expect(deleteRequest(parseConversationDocument(duplicated))).rejects.toThrow('one accepted append');
    });

    it('rejects a queued job that retains only archived entry IDs for the deleted source', async () => {
        const policy = (
            await setProcessingPolicy(emptyDocument(), {
                operation_id: 'policy:delete-test',
                expected_revision: 0,
                recorded_at: at,
                enabled: true,
                processors: [
                    {
                        id: 'processor:test',
                        version: 'v1',
                        scope: 'on_append',
                        config: {},
                        required: true,
                        failure_behavior: 'block',
                    },
                ],
            })
        ).document;
        const source = appendSource();
        const accepted = await appendConversationRecordsWithProcessing(policy, source.batch, {
            ...source.options,
            expected_revision: policy.revision,
        });
        const excluded = await excludeSource(accepted.document);
        const pending = structuredClone(excluded);
        const jobId = Object.keys(pending.processing.jobs ?? {})[0];
        const job = pending.processing.jobs?.[jobId];
        if (job?.selection.kind !== 'entries') throw new Error('Expected staged entry job');
        job.selection = { kind: 'entries', entry_ids: ['entry:source'] };
        job.selection_fingerprint = await fingerprintJson(job.selection);
        await expect(deleteRequest(parseConversationDocument(pending))).rejects.toThrow(
            `/processing/jobs/${jobId}/selection`,
        );
    });

    it('refuses indexed migration before writing because v1 index lacks tombstone witnesses', async () => {
        const excluded = await excludeSource(appendSource().document);
        const deleted = (await applyConversationDelete(excluded, await deleteRequest(excluded))).document;
        let writes = 0;
        const store: IndexedConversationRecordStore = {
            async read() {
                throw new Error('unexpected read');
            },
            async write() {
                writes += 1;
            },
            async readRecord() {
                throw new Error('unexpected record read');
            },
            async writeRecord() {
                writes += 1;
            },
        };
        await expect(stageIndexedConversationSnapshot(deleted, undefined, store)).rejects.toThrow(
            'tombstone witnesses',
        );
        expect(writes).toBe(0);
    });
});
