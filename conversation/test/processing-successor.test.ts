import { describe, expect, it, vi } from 'vitest';
import {
    appendConversationRecordsWithProcessing,
    createConversationDocument,
    createTextExternalizationProcessor,
    fingerprintJson,
    MAX_PROCESSING_SUCCESSOR_BYTES,
    type ProcessingStore,
    parseConversationDocument,
    runProcessingJob,
    setProcessingPolicy,
    textExternalizationArchiveInputs,
    verifyProcessingSuccessor,
} from '../src/index.js';
import { buildProcessingPhaseDocument, resolveProcessingJobInput } from '../src/processing.js';
import { generatedAgentTurn, importedGeneration, userTurn } from './fixtures.js';

const at = '2026-09-11T00:01:00.000Z';

async function accepted(processorId = 'externalize-text', selectInput = true) {
    const retained = await appendConversationRecordsWithProcessing(
        createConversationDocument({ id: 'lineage', created_at: '2026-09-11T00:00:00Z' }),
        {
            turns: [generatedAgentTurn('history-turn', 'history-generation')],
            generations: [
                {
                    ...importedGeneration('history-generation'),
                    source: { conversation_id: 'lineage', revision: 0 },
                    model_options: { temperature: 0.2 },
                },
            ],
        },
        {
            operation_id: 'history-append',
            expected_revision: 0,
            payload_fingerprint: 'sha256:history',
            recorded_at: at,
        },
    );
    const policy = await setProcessingPolicy(retained.document, {
        operation_id: 'policy',
        expected_revision: retained.document.revision,
        recorded_at: at,
        enabled: true,
        processors: [
            {
                id: processorId,
                version: '1',
                scope: 'on_append',
                config: {},
                required: true,
                failure_behavior: 'block',
            },
        ],
    });
    const result = await appendConversationRecordsWithProcessing(
        policy.document,
        {
            turns: [{ ...userTurn('input'), timestamps: { recorded_at: at } }],
            context_entries: selectInput ? [{ id: 'entry', type: 'source_turn', turn_id: 'input' }] : [],
            tool_definitions: [{ id: 'read-definition', name: 'read_blob', version: '1', input_schema: true }],
            active_tool_definition_ids: ['read-definition'],
        },
        {
            operation_id: 'input-append',
            expected_revision: policy.document.revision,
            payload_fingerprint: 'sha256:input',
            recorded_at: at,
        },
    );
    return result.document;
}

async function lineage() {
    const source = await accepted();
    const job = Object.values(source.processing.jobs ?? {})[0];
    const archive = await textExternalizationArchiveInputs(source, job);
    const archived = await appendConversationRecordsWithProcessing(
        source,
        {
            assets: [
                {
                    id: 'archive',
                    kind: 'text',
                    mime_type: 'text/plain',
                    storage: { type: 'external', resolver: 'blob', locator: { key: 'original' } },
                    provenance: { type: 'received' },
                    created_at: at,
                    ...archive.integrities[0],
                },
            ],
        },
        {
            operation_id: `processing:archive:${job.id}`,
            expected_revision: source.revision,
            payload_fingerprint: archive.payload_fingerprint,
            recorded_at: at,
        },
    );
    const successors = [archived.document];
    let current = archived.document;
    const store: ProcessingStore = {
        async load() {
            return structuredClone(current);
        },
        async commit(revision, document) {
            if (revision !== current.revision) return false;
            current = parseConversationDocument(document);
            successors.push(structuredClone(current));
            return true;
        },
    };
    const processor = createTextExternalizationProcessor(({ asset }) => ({
        capability: 'read_blob',
        version: 1,
        arguments: { asset_id: asset.id },
        tool_definition_id: 'read-definition',
    }));
    const resolve = vi.fn(() => processor);
    await runProcessingJob(store, { resolve }, job.id, 'actual-attempt', () => at);
    return {
        input: {
            accepted_document: source,
            anchor: { kind: 'accepted_append' as const, receipt: source.operation_receipts['input-append'] },
            successors,
        },
        resolve,
        current,
    };
}

describe('pure processing successor lineage', () => {
    it('replays archive, resolve, attempt, output, and application without registry/plugin calls', async () => {
        const { input, resolve, current } = await lineage();
        const calls = resolve.mock.calls.length;
        const result = await verifyProcessingSuccessor(input);
        expect(result).toEqual({
            source: { conversation_id: current.id, revision: current.revision },
            operation_ids: input.successors.map((document, index) =>
                Object.keys(document.operation_receipts).find(
                    (id) =>
                        !Object.hasOwn(
                            index
                                ? input.successors[index - 1].operation_receipts
                                : input.accepted_document.operation_receipts,
                            id,
                        ),
                ),
            ),
        });
        expect(resolve).toHaveBeenCalledTimes(calls);
        expect(result).not.toHaveProperty('ready');
        expect(result).not.toHaveProperty('admission');
    });

    it.each([false, true])(
        'retains no-op/failure completion facts without claiming readiness (selected=%s)',
        async (selectInput) => {
            const source = await accepted('externalize-text', selectInput);
            let current = source;
            const successors: (typeof source)[] = [];
            const store: ProcessingStore = {
                async load() {
                    return structuredClone(current);
                },
                async commit(revision, document) {
                    if (revision !== current.revision) return false;
                    current = parseConversationDocument(document);
                    successors.push(structuredClone(current));
                    return true;
                },
            };
            const job = Object.values(source.processing.jobs ?? {})[0];
            // Actual builtin fails without its prior archive; empty selection never resolves a processor.
            const resolve = vi.fn(() =>
                createTextExternalizationProcessor(() => {
                    throw new Error('No retained archive');
                }),
            );
            await runProcessingJob(store, { resolve }, job.id, 'attempt-failure', () => at);
            const result = await verifyProcessingSuccessor({
                accepted_document: source,
                anchor: { kind: 'accepted_append', receipt: source.operation_receipts['input-append'] },
                successors,
            });
            expect(result.source.revision).toBe(current.revision);
            expect(current.processing.completions?.[job.id].status).toBe(selectInput ? 'blocked' : 'no_op');
            expect(result).not.toHaveProperty('ready');
            if (!selectInput) expect(resolve).not.toHaveBeenCalled();
        },
    );

    it('accepts an exact initialized document anchor without inventing an append receipt', async () => {
        const source = createConversationDocument({ id: 'empty-initial', created_at: at });
        expect(
            await verifyProcessingSuccessor({
                accepted_document: source,
                anchor: { kind: 'initialized_source', document_fingerprint: await fingerprintJson(source) },
                successors: [],
            }),
        ).toEqual({ source: { conversation_id: source.id, revision: 0 }, operation_ids: [] });
    });

    it('requires every immutable revision and the exact original receipt', async () => {
        const { input } = await lineage();
        await expect(
            verifyProcessingSuccessor({ ...input, successors: input.successors.slice(1) }),
        ).rejects.toMatchObject({ code: 'MISSING_EVIDENCE' });
        await expect(
            verifyProcessingSuccessor({
                ...input,
                anchor: { ...input.anchor, receipt: { ...input.anchor.receipt, payload_fingerprint: 'changed' } },
            }),
        ).rejects.toMatchObject({ code: 'CONFLICT' });
    });

    it.each([
        'receipt',
        'unrelated_receipt',
        'body',
        'configuration',
        'metadata',
        'generation',
        'policy',
        'job',
        'output',
        'catalog',
    ] as const)('rejects an intervening %s mutation even with unchanged revision/queue', async (field) => {
        const { input } = await lineage();
        const changed = structuredClone(input);
        const snapshot = changed.successors[changed.successors.length - 1];
        if (field === 'receipt') snapshot.operation_receipts['input-append'].payload_fingerprint = 'changed';
        if (field === 'body')
            snapshot.turns[0].blocks[0] = { id: 'input-text', type: 'text', text: 'changed', format: 'plain' };
        if (field === 'configuration') snapshot.generations['history-generation'].model_options = { temperature: 0.8 };
        if (field === 'metadata') snapshot.metadata = { authoring: 'changed' };
        if (field === 'generation') snapshot.generations['history-generation'].metadata = { changed: true };
        if (field === 'unrelated_receipt')
            snapshot.operation_receipts['history-append'].payload_fingerprint = 'changed';
        if (field === 'policy') snapshot.processing.enabled = false;
        const id = Object.keys(snapshot.processing.jobs ?? {})[0];
        if (field === 'job' && snapshot.processing.jobs) snapshot.processing.jobs[id].configuration = { changed: true };
        if (field === 'output' && snapshot.processing.outputs)
            snapshot.processing.outputs[id].output_fingerprint = 'changed';
        if (field === 'catalog') snapshot.tool_definitions['read-definition'].description = 'changed';
        await expect(verifyProcessingSuccessor(changed)).rejects.toThrow();
    });

    it('does not accept a new authoring append or policy operation as processing lineage', async () => {
        const source = await accepted();
        const changed = await appendConversationRecordsWithProcessing(
            source,
            { turns: [userTurn('second')] },
            {
                operation_id: 'new-authoring',
                expected_revision: source.revision,
                payload_fingerprint: 'sha256:second',
                recorded_at: at,
            },
        );
        await expect(
            verifyProcessingSuccessor({
                accepted_document: source,
                anchor: { kind: 'accepted_append', receipt: source.operation_receipts['input-append'] },
                successors: [changed.document],
            }),
        ).rejects.toThrow();
    });

    it('blocks unsupported processors explicitly without resolving them', async () => {
        const source = await accepted('foreign-plugin');
        const job = Object.values(source.processing.jobs ?? {})[0];
        const resolution = await resolveProcessingJobInput(source, job, at);
        const next = await buildProcessingPhaseDocument(source, 'resolve', job, resolution, at, {
            ...source.processing,
            resolved_inputs: { [job.id]: resolution },
        });
        await expect(
            verifyProcessingSuccessor({
                accepted_document: source,
                anchor: { kind: 'accepted_append', receipt: source.operation_receipts['input-append'] },
                successors: [next],
            }),
        ).rejects.toMatchObject({ code: 'UNSUPPORTED' });
    });

    it('owns evidence before awaits and owns its returned source', async () => {
        const { input, current } = await lineage();
        const pending = verifyProcessingSuccessor(input);
        input.successors[0].assets.archive.content_hash = 'caller-mutation';
        input.anchor.receipt.payload_fingerprint = 'caller-mutation';
        expect((await pending).source).toEqual({ conversation_id: current.id, revision: current.revision });
    });

    it('fails complete snapshot/byte/depth bounds rather than blessing a partial prefix', async () => {
        const source = await accepted();
        const input = {
            accepted_document: source,
            anchor: { kind: 'accepted_append', receipt: source.operation_receipts['input-append'] },
            successors: Array.from({ length: 65 }, () => source),
        };
        await expect(verifyProcessingSuccessor(input)).rejects.toMatchObject({ code: 'RESOURCE_LIMIT' });
        await expect(
            verifyProcessingSuccessor({
                ...input,
                successors: [],
                overflow: 'x'.repeat(MAX_PROCESSING_SUCCESSOR_BYTES),
            }),
        ).rejects.toMatchObject({ code: 'RESOURCE_LIMIT' });
        let nested: unknown = null;
        for (let depth = 0; depth < 66; depth += 1) nested = { child: nested };
        await expect(verifyProcessingSuccessor({ ...input, successors: [], nested })).rejects.toMatchObject({
            code: 'RESOURCE_LIMIT',
        });
        const getter = vi.fn(() => []);
        await expect(
            verifyProcessingSuccessor({
                accepted_document: source,
                anchor: input.anchor,
                get successors() {
                    return getter();
                },
            }),
        ).rejects.toThrow();
        expect(getter).not.toHaveBeenCalled();
    });
});
