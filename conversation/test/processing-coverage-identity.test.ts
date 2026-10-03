import { describe, expect, it } from 'vitest';
import {
    appendConversationRecordsWithProcessing,
    assessProcessingReadiness,
    type ConversationDocument,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    ProcessingKnownFailure,
    type ProcessingStore,
    parseConversationDocument,
    processingContextFingerprint,
    queueProcessingForExisting,
    recordProcessingCoverage,
    runProcessingJob,
    setProcessingPolicy,
} from '../src/index.js';

const at = '2026-10-03T00:00:00Z';
async function base() {
    return (
        await setProcessingPolicy(createConversationDocument({ id: 'multi-child', created_at: at }), {
            operation_id: 'policy',
            expected_revision: 0,
            recorded_at: at,
            enabled: true,
            processors: [],
        })
    ).document;
}
async function coverage(document: Awaited<ReturnType<typeof base>>, child: string, operationId = `coverage:${child}`) {
    const measurementFingerprint = await fingerprintJson({ child, native_body: { messages: [] }, input_tokens: 12 });
    const command = {
        operation_id: operationId,
        expected_revision: document.revision,
        target_fingerprint: await fingerprintJson({ provider: 'actual-provider', model: child }),
        measured_input_tokens: 12,
        tokenizer_id: 'identified-tokenizer',
        measurement_fingerprint: measurementFingerprint,
        recorded_at: at,
    };
    return { command, result: await recordProcessingCoverage(document, command) };
}

describe('retained target-specific coverage for unchanged context', () => {
    it('reaches a finite fixed point: one commit then exact retry without another revision', async () => {
        const source = await base();
        const first = await coverage(source, 'a');
        const retried = await recordProcessingCoverage(first.result.document, first.command);
        expect(retried.applied).toBe(false);
        expect(retried.document).toEqual(first.result.document);
        expect(await processingContextFingerprint(retried.document)).toBe(await processingContextFingerprint(source));
        expect(
            (
                await assessProcessingReadiness(
                    retried.document,
                    first.command.target_fingerprint,
                    first.command.measurement_fingerprint,
                )
            ).status,
        ).toBe('ready');
    });
    it('retains both virtual child evaluations after a second child publishes coverage', async () => {
        const source = await base();
        const a = await coverage(source, 'a');
        const b = await coverage(a.result.document, 'b');
        expect(b.result.document.revision).toBe(source.revision + 2);
        expect(await processingContextFingerprint(b.result.document)).toBe(await processingContextFingerprint(source));
        for (const child of [a, b]) {
            const ready = await assessProcessingReadiness(
                b.result.document,
                child.command.target_fingerprint,
                child.command.measurement_fingerprint,
            );
            expect(ready.status).toBe('ready');
            const retried = await recordProcessingCoverage(b.result.document, child.command);
            expect(retried.applied).toBe(false);
            expect(retried.document.revision).toBe(b.result.document.revision);
        }
    });
    it('rejects missing receipt and changed retained evaluation instead of trusting the map', async () => {
        const a = await coverage(await base(), 'a');
        const b = await coverage(a.result.document, 'b');
        const missing = structuredClone(b.result.document);
        delete missing.operation_receipts[a.command.operation_id];
        const changed = structuredClone(b.result.document);
        const evidence = changed.processing.coverage_receipts?.[a.command.operation_id];
        if (!evidence) throw new Error('Test requires actual retained coverage');
        evidence.measurement.input_tokens++;
        await expect(
            assessProcessingReadiness(missing, a.command.target_fingerprint, a.command.measurement_fingerprint),
        ).rejects.toThrow('validation');
        expect(
            (await assessProcessingReadiness(changed, a.command.target_fingerprint, a.command.measurement_fingerprint))
                .status,
        ).toBe('pending');
    });
    it('does not use retained coverage after a required job is queued or blocked on unchanged context', async () => {
        const configured = await setProcessingPolicy(
            createConversationDocument({ id: 'required-job', created_at: at }),
            {
                operation_id: 'policy',
                expected_revision: 0,
                recorded_at: at,
                enabled: true,
                processors: [
                    {
                        id: 'required-test',
                        version: '1',
                        scope: 'manual',
                        config: {},
                        required: true,
                        failure_behavior: 'block',
                    },
                ],
            },
        );
        const input = await appendConversationRecordsWithProcessing(
            configured.document,
            {
                turns: [
                    createUserTurn({
                        id: 'input',
                        authority: 'ordinary',
                        model_visibility: 'include',
                        status: 'completed',
                        timestamps: { recorded_at: at },
                        provenance: { type: 'received' },
                        blocks: [createTextBlock({ id: 'text', text: 'retained input', format: 'plain' })],
                    }),
                ],
                context_entries: [{ id: 'entry', type: 'source_turn', turn_id: 'input' }],
            },
            {
                operation_id: 'accepted-input',
                expected_revision: configured.document.revision,
                recorded_at: at,
                payload_fingerprint: 'sha256:input',
            },
        );
        const a = await coverage(input.document, 'a');
        const b = await coverage(a.result.document, 'b');
        const queued = await queueProcessingForExisting(
            b.result.document,
            {
                conversation: { conversation_id: b.result.document.id, revision: b.result.document.revision },
                expected_context_revision: b.result.document.context.revision,
                selector: { source: { kind: 'all' } },
            },
            {
                operation_id: 'queue',
                expected_revision: b.result.document.revision,
                recorded_at: at,
                processor_id: 'required-test',
                scope: 'manual',
            },
        );
        expect(await processingContextFingerprint(queued.document)).toBe(
            await processingContextFingerprint(input.document),
        );
        expect(
            (
                await assessProcessingReadiness(
                    queued.document,
                    a.command.target_fingerprint,
                    a.command.measurement_fingerprint,
                )
            ).status,
        ).toBe('pending');
        let current = queued.document;
        const store: ProcessingStore = {
            load: async () => parseConversationDocument(current),
            commit: async (expectedRevision: number, document: ConversationDocument) => {
                if (current.revision !== expectedRevision) return false;
                current = parseConversationDocument(document);
                return true;
            },
        };
        await runProcessingJob(
            store,
            {
                resolve: () => ({
                    run: async () => {
                        throw new ProcessingKnownFailure('required processing rejected');
                    },
                }),
            },
            queued.job_id,
            'actual-test-attempt',
            () => at,
        );
        expect(current.processing.completions?.[queued.job_id]?.status).toBe('blocked');
        expect(await processingContextFingerprint(current)).toBe(await processingContextFingerprint(input.document));
        expect(
            (await assessProcessingReadiness(current, a.command.target_fingerprint, a.command.measurement_fingerprint))
                .status,
        ).not.toBe('ready');
        const blockedA = await coverage(current, 'a', 'coverage:a-after-block');
        const blockedB = await coverage(blockedA.result.document, 'b', 'coverage:b-after-block');
        expect(
            (
                await assessProcessingReadiness(
                    blockedB.result.document,
                    blockedA.command.target_fingerprint,
                    blockedA.command.measurement_fingerprint,
                )
            ).status,
        ).toBe('blocked');
    });
    it('re-evaluates the SAME target/count after a required no-op job without context change', async () => {
        const configured = await setProcessingPolicy(createConversationDocument({ id: 'no-op-job', created_at: at }), {
            operation_id: 'policy',
            expected_revision: 0,
            recorded_at: at,
            enabled: true,
            processors: [
                {
                    id: 'required-noop',
                    version: '1',
                    scope: 'manual',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const source = await appendConversationRecordsWithProcessing(
            configured.document,
            {
                turns: [
                    createUserTurn({
                        id: 'input',
                        authority: 'ordinary',
                        model_visibility: 'include',
                        status: 'completed',
                        timestamps: { recorded_at: at },
                        provenance: { type: 'received' },
                        blocks: [createTextBlock({ id: 'text', text: 'same input', format: 'plain' })],
                    }),
                ],
                context_entries: [{ id: 'entry', type: 'source_turn', turn_id: 'input' }],
            },
            {
                operation_id: 'accepted-input',
                expected_revision: configured.document.revision,
                recorded_at: at,
                payload_fingerprint: 'sha256:input',
            },
        );
        const a = await coverage(source.document, 'a');
        const b = await coverage(a.result.document, 'b');
        const queued = await queueProcessingForExisting(
            b.result.document,
            {
                conversation: { conversation_id: b.result.document.id, revision: b.result.document.revision },
                expected_context_revision: b.result.document.context.revision,
                selector: { source: { kind: 'all' } },
            },
            {
                operation_id: 'queue',
                expected_revision: b.result.document.revision,
                recorded_at: at,
                processor_id: 'required-noop',
                scope: 'manual',
            },
        );
        let current = queued.document;
        const store: ProcessingStore = {
            load: async () => parseConversationDocument(current),
            commit: async (expectedRevision: number, document: ConversationDocument) => {
                if (current.revision !== expectedRevision) return false;
                current = parseConversationDocument(document);
                return true;
            },
        };
        await runProcessingJob(
            store,
            {
                resolve: () => ({
                    run: async () => ({
                        kind: 'no_op',
                        reason: 'deterministic required evaluation complete',
                    }),
                }),
            },
            queued.job_id,
            'noop-attempt',
            () => at,
        );
        expect(current.processing.completions?.[queued.job_id]?.status).toBe('no_op');
        expect(await processingContextFingerprint(current)).toBe(await processingContextFingerprint(source.document));
        expect(
            (await assessProcessingReadiness(current, a.command.target_fingerprint, a.command.measurement_fingerprint))
                .status,
        ).toBe('pending');
        const refreshedA = await coverage(current, 'a', 'coverage:a-after-noop');
        const refreshedB = await coverage(refreshedA.result.document, 'b', 'coverage:b-after-noop');
        expect(refreshedA.command.target_fingerprint).toBe(a.command.target_fingerprint);
        expect(refreshedA.command.measurement_fingerprint).toBe(a.command.measurement_fingerprint);
        for (const child of [refreshedA, refreshedB]) {
            expect(
                (
                    await assessProcessingReadiness(
                        refreshedB.result.document,
                        child.command.target_fingerprint,
                        child.command.measurement_fingerprint,
                    )
                ).status,
            ).toBe('ready');
            expect((await recordProcessingCoverage(refreshedB.result.document, child.command)).applied).toBe(false);
        }
    });
    it('rejects changed target, count identity, policy and context', async () => {
        const a = await coverage(await base(), 'a');
        const b = await coverage(a.result.document, 'b');
        expect(
            (await assessProcessingReadiness(b.result.document, 'sha256:other', a.command.measurement_fingerprint))
                .status,
        ).toBe('pending');
        expect(
            (await assessProcessingReadiness(b.result.document, a.command.target_fingerprint, 'sha256:other')).status,
        ).toBe('pending');
        const policy = await setProcessingPolicy(b.result.document, {
            operation_id: 'changed-policy',
            expected_revision: b.result.document.revision,
            recorded_at: at,
            enabled: true,
            processors: [],
        });
        expect(
            (
                await assessProcessingReadiness(
                    policy.document,
                    a.command.target_fingerprint,
                    a.command.measurement_fingerprint,
                )
            ).status,
        ).toBe('pending');
        const context = await appendConversationRecordsWithProcessing(
            b.result.document,
            {
                turns: [
                    createUserTurn({
                        id: 'new-input',
                        authority: 'ordinary',
                        model_visibility: 'include',
                        status: 'completed',
                        timestamps: { recorded_at: at },
                        provenance: { type: 'received' },
                        blocks: [createTextBlock({ id: 'new-text', text: 'changed input', format: 'plain' })],
                    }),
                ],
                context_entries: [{ id: 'new-entry', type: 'source_turn', turn_id: 'new-input' }],
            },
            {
                operation_id: 'changed-context',
                expected_revision: b.result.document.revision,
                recorded_at: at,
                payload_fingerprint: 'sha256:changed',
            },
        );
        expect(
            (
                await assessProcessingReadiness(
                    context.document,
                    a.command.target_fingerprint,
                    a.command.measurement_fingerprint,
                )
            ).status,
        ).toBe('pending');
    });
});
