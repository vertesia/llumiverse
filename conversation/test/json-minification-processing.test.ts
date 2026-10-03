import { describe, expect, it, vi } from 'vitest';
import {
    appendConversationRecords,
    type ConversationDocument,
    type ConversationTurn,
    createConversationDocument,
    createGeneratedAgentTurn,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    type Generation,
    JSON_MINIFICATION_PROCESSOR_ID,
    type JsonMinificationHostCapability,
    jsonMinificationProcessor,
    type ProcessingStore,
    parseConversationDocument,
    queueProcessingForExisting,
    runProcessingJob,
    setProcessingPolicy,
    verifyDerivedBlockLineage,
} from '../src/index.js';

const at = '2026-10-02T00:00:00.000Z';
const target = 'sha256:target';
const config = {
    format: 'raw_json_text',
    minimum_token_reduction: 2,
    max_code_units: 1048576,
    max_depth: 128,
    max_lexical_tokens: 262144,
};
class Store implements ProcessingStore {
    constructor(public current: ConversationDocument) {}
    async load() {
        return structuredClone(this.current);
    }
    async commit(revision: number, next: ConversationDocument) {
        if (revision !== this.current.revision) return false;
        this.current = parseConversationDocument(next);
        return true;
    }
}
async function fixture(
    texts = [' { "n": 900719925474099312345, "n": -0 } ', ' [ 1e+999, "\\u0061" ] '],
    supplied?: { turns: ConversationTurn[]; generations?: Generation[] },
) {
    const turns =
        supplied?.turns ??
        texts.map((text, i) =>
            createUserTurn({
                id: `turn:${i}`,
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: at },
                model_visibility: 'include',
                provenance: { type: 'received' },
                blocks: [createTextBlock({ id: `block:${i}`, text, format: 'plain' })],
            }),
        );
    const base = appendConversationRecords(
        createConversationDocument({ id: 'json-processing', created_at: at }),
        {
            turns,
            generations: supplied?.generations ?? [],
            context_entries: turns.map((turn, i) => ({ id: `entry:${i}`, type: 'source_turn', turn_id: turn.id })),
        },
        { operation_id: 'append', expected_revision: 0, recorded_at: at, payload_fingerprint: 'sha256:append' },
    ).document;
    const enabled = (
        await setProcessingPolicy(base, {
            operation_id: 'policy',
            expected_revision: base.revision,
            recorded_at: at,
            enabled: true,
            processors: [
                {
                    id: JSON_MINIFICATION_PROCESSOR_ID,
                    version: '1',
                    scope: 'manual',
                    config,
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        })
    ).document;
    const queued = await queueProcessingForExisting(
        enabled,
        {
            conversation: { conversation_id: enabled.id, revision: enabled.revision },
            expected_context_revision: enabled.context.revision,
            selector: { source: { kind: 'all' } },
        },
        {
            operation_id: 'queue',
            expected_revision: enabled.revision,
            recorded_at: at,
            processor_id: JSON_MINIFICATION_PROCESSOR_ID,
            scope: 'manual',
            target_fingerprint: target,
        },
    );
    return { store: new Store(queued.document), jobId: queued.job_id, turns };
}
function capability(delta = 8): JsonMinificationHostCapability {
    const identity = {
        tokenizer: 'test-projected-tokenizer',
        tokenizer_version: '1',
        adapter: 'test-native-projection',
        adapter_version: '1',
        target_model: 'test-model',
        method: 'exact' as const,
    };
    return {
        target_fingerprint: target,
        identity: structuredClone(identity),
        measureProspective: vi.fn(async (input) => {
            // A trusted host capability fixture, not a production tokenizer or inferred char count.
            expect(parseConversationDocument(input.source)).toEqual(input.source);
            expect(Object.keys(input.source.compactions)).toHaveLength(0);
            const retainedResolution = input.source.processing.resolved_inputs?.[input.processing_job_id];
            expect(retainedResolution?.job_id).toBe(input.processing_job_id);
            expect(input.resolved_input_fingerprint).toBe(await fingerprintJson(retainedResolution));
            expect(input.input_fingerprint).toBe(
                await fingerprintJson({
                    processing_job_id: input.processing_job_id,
                    resolved_input_fingerprint: input.resolved_input_fingerprint,
                    context_fingerprint: input.context_fingerprint,
                    candidate: input.candidate,
                    target_fingerprint: input.target_fingerprint,
                }),
            );
            return {
                status: 'measured' as const,
                measurement: {
                    target_fingerprint: target,
                    prospective_input_fingerprint: input.input_fingerprint,
                    original_projection_fingerprint: await fingerprintJson({
                        request: 'original',
                        source: input.context_fingerprint,
                    }),
                    replacement_projection_fingerprint: await fingerprintJson({
                        request: 'prospective',
                        plan: input.candidate,
                    }),
                    original: {
                        ...identity,
                        input_tokens: 100,
                        source_fingerprint: input.context_fingerprint,
                        measured_at: at,
                    },
                    replacement: {
                        ...identity,
                        input_tokens: 100 - delta,
                        source_fingerprint: input.input_fingerprint,
                        measured_at: at,
                    },
                },
            };
        }),
    };
}
const registry = { resolve: () => jsonMinificationProcessor };
async function run(store: Store, jobId: string, host?: JsonMinificationHostCapability) {
    return runProcessingJob(store, registry, jobId, 'attempt', () => at, undefined, { json_minification: host });
}

describe('validated JSON processing application', () => {
    it('applies independent blocks in place, retains original lexemes/records, and verifies precise lineage', async () => {
        const { store, jobId, turns } = await fixture();
        const host = capability();
        const result = await run(store, jobId, host);
        expect(result.status).toBe('completed');
        expect(store.current.turns).toEqual(turns);
        expect(store.current.processing.completions?.[jobId]?.status).toBe('applied');
        const derived = Object.values(store.current.compactions)[0].replacement_turns;
        expect(derived.map((turn) => turn.kind)).toEqual(['user', 'user']);
        expect(derived.map((turn) => turn.blocks[0])).toMatchObject([
            { text: '{"n":900719925474099312345,"n":-0}' },
            { text: '[1e+999,"\\u0061"]' },
        ]);
        expect(store.current.context.entries.map((entry) => entry.turn_id)).toEqual(derived.map((turn) => turn.id));
        expect(await verifyDerivedBlockLineage(store.current)).toEqual(store.current);
        const prior = structuredClone(store.current);
        await run(store, jobId, host);
        expect(host.measureProspective).toHaveBeenCalledTimes(1);
        expect(store.current).toEqual(prior);
        expect(store.current.generations).toEqual({});
    });
    it.each([
        ['unavailable', undefined, 'measurement_unavailable'],
        ['no benefit', capability(1), 'no_token_benefit'],
    ] as const)(
        'durably records %s without publishing a transformation or budget readiness',
        async (_label, host, reason) => {
            const { store, jobId } = await fixture();
            await run(store, jobId, host);
            expect(store.current.processing.outputs?.[jobId]).toMatchObject({
                kind: 'json_minification_no_op',
                reason,
            });
            expect(store.current.processing.completions?.[jobId]?.status).toBe('no_op');
            expect(store.current.compactions).toEqual({});
            expect(store.current.processing.coverage).toBeUndefined();
        },
    );
    it('records already minified input as an explicit durable no-op without counting', async () => {
        const { store, jobId } = await fixture(['{"x":1}']);
        const host = capability();
        await run(store, jobId, host);
        expect(store.current.processing.outputs?.[jobId]).toMatchObject({
            kind: 'json_minification_no_op',
            reason: 'already_minified',
        });
        expect(host.measureProspective).not.toHaveBeenCalled();
    });
    it('rejects forged processor replacement and host measurement binding before application', async () => {
        const { store, jobId } = await fixture();
        const fake = {
            resolve: () => ({
                run: async (input: Parameters<typeof jsonMinificationProcessor.run>[0]) => {
                    const result = await jsonMinificationProcessor.run(input);
                    if (result.kind === 'json_minification_candidate')
                        result.transforms[0].replacement_text = '{"n":1}';
                    return result;
                },
            }),
        };
        await runProcessingJob(store, fake, jobId, 'attempt', () => at, undefined, { json_minification: capability() });
        expect(store.current.processing.outputs?.[jobId]?.kind).toBe('unknown_outcome');
        expect(store.current.compactions).toEqual({});
        const second = await fixture();
        const host = capability();
        const original = host.measureProspective;
        host.measureProspective = async (input, signal) => {
            const result = await original(input, signal);
            if (result.status === 'measured') result.measurement.prospective_input_fingerprint = 'sha256:forged';
            return result;
        };
        await run(second.store, second.jobId, host);
        expect(second.store.current.compactions).toEqual({});
        expect(second.store.current.processing.completions?.[second.jobId]?.status).toBe('blocked');
    });
    it('rejects forged accepted source/output in retained lineage', async () => {
        const { store, jobId } = await fixture();
        await run(store, jobId, capability());
        const changedSource = structuredClone(store.current);
        const block = changedSource.turns[0].blocks[0];
        if (block.type === 'text') block.text = '{"different":1}';
        await expect(verifyDerivedBlockLineage(changedSource)).rejects.toThrow();
        const changedOutput = structuredClone(store.current);
        const output = changedOutput.processing.outputs?.[jobId];
        if (output?.kind === 'json_minification') output.proposal.measurement.replacement.input_tokens = 0;
        await expect(verifyDerivedBlockLineage(changedOutput)).rejects.toThrow();
    });
    it('recovers a persisted output after lost completion CAS without rerunning processor/count', async () => {
        const { store, jobId } = await fixture();
        const host = capability();
        const commit = store.commit.bind(store);
        let refused = false;
        store.commit = async (revision, next) => {
            if (!refused && next.processing.completions?.[jobId]) {
                refused = true;
                return false;
            }
            return commit(revision, next);
        };
        await run(store, jobId, host);
        expect(refused).toBe(true);
        expect(host.measureProspective).toHaveBeenCalledTimes(1);
        expect(Object.keys(store.current.compactions)).toHaveLength(1);
        expect(await verifyDerivedBlockLineage(store.current)).toEqual(store.current);
    });
    it('rejects stale source after output persistence and retains original durable output', async () => {
        const { store, jobId } = await fixture();
        const commit = store.commit.bind(store);
        store.commit = async (revision, next) => {
            const ok = await commit(revision, next);
            if (ok && next.processing.outputs?.[jobId] && !next.processing.completions?.[jobId]) {
                const block = store.current.turns[0].blocks[0];
                if (block.type === 'text') block.text = '{"changed":1}';
            }
            return ok;
        };
        await expect(run(store, jobId, capability())).rejects.toThrow('source context changed');
        expect(store.current.processing.outputs?.[jobId]?.kind).toBe('json_minification');
        expect(store.current.compactions).toEqual({});
    });
    it('cancels asynchronous count without output, transform, or fabricated readiness', async () => {
        const { store, jobId } = await fixture();
        const controller = new AbortController();
        const host = capability();
        host.measureProspective = async () => {
            controller.abort(new Error('cancelled'));
            return new Promise(() => {});
        };
        await expect(
            runProcessingJob(store, registry, jobId, 'attempt', () => at, controller.signal, {
                json_minification: host,
            }),
        ).rejects.toThrow('cancelled');
        expect(store.current.processing.outputs?.[jobId]).toBeUndefined();
        expect(store.current.processing.attempts?.[jobId]).toBeDefined();
        expect(store.current.compactions).toEqual({});
    });
    it('bounds an unavailable count and records a durable typed reason', async () => {
        const { store, jobId } = await fixture();
        const host = capability();
        let entered: (() => void) | undefined;
        const started = new Promise<void>((resolve) => {
            entered = resolve;
        });
        host.measureProspective = async () => {
            entered?.();
            return new Promise(() => {});
        };
        vi.useFakeTimers();
        try {
            const pending = run(store, jobId, host);
            await started;
            await vi.runAllTimersAsync();
            await pending;
        } finally {
            vi.useRealTimers();
        }
        expect(store.current.processing.outputs?.[jobId]).toMatchObject({
            kind: 'json_minification_no_op',
            reason: 'measurement_unavailable',
        });
    });
    it('rejects grammar/prose, source bounds, and caller-authored validated result kinds', async () => {
        const bad = await fixture(['This is prose, not JSON.']);
        await run(bad.store, bad.jobId, capability());
        expect(bad.store.current.processing.completions?.[bad.jobId]?.status).toBe('blocked');
        expect(bad.store.current.compactions).toEqual({});
        const forged = await fixture();
        const fake = { resolve: () => ({ run: async () => JSON.parse('{"kind":"json_minification","proposal":{}}') }) };
        await runProcessingJob(forged.store, fake, forged.jobId, 'attempt', () => at, undefined, {
            json_minification: capability(),
        });
        expect(forged.store.current.processing.outputs?.[forged.jobId]?.kind).toBe('failed');
        expect(forged.store.current.compactions).toEqual({});
    });

    it('captures target/tokenizer/callback before the first store await', async () => {
        const { store, jobId } = await fixture();
        const host = capability();
        const originalCount = host.measureProspective;
        const load = store.load.bind(store);
        let release: (() => void) | undefined;
        const barrier = new Promise<void>((resolve) => {
            release = resolve;
        });
        let entered: (() => void) | undefined;
        const started = new Promise<void>((resolve) => {
            entered = resolve;
        });
        let first = true;
        store.load = async () => {
            if (first) {
                first = false;
                entered?.();
                await barrier;
            }
            return load();
        };
        const pending = run(store, jobId, host);
        await started;
        host.target_fingerprint = 'sha256:changed';
        host.identity.tokenizer = 'changed';
        host.measureProspective = async () => {
            throw new Error('changed callback must not run');
        };
        release?.();
        await pending;
        expect(originalCount).toHaveBeenCalledTimes(1);
        expect(store.current.processing.outputs?.[jobId]).toMatchObject({
            kind: 'json_minification',
            proposal: {
                measurement: { target_fingerprint: target, original: { tokenizer: 'test-projected-tokenizer' } },
            },
        });
    });
    it('preserves generated agent usage/identity and program role/presentation in an independent batch', async () => {
        const generation: Generation = {
            id: 'generation',
            record_source: 'executed',
            request_id: 'request',
            attempt_id: 'generation-attempt',
            purpose: 'conversation',
            requested_model: 'model',
            provider: 'provider',
            protocol: 'protocol',
            adapter_version: '1',
            status: 'completed',
            source: { conversation_id: 'json-processing', revision: 0 },
            timestamps: { recorded_at: at },
            usage: {
                input_tokens: 7,
                output_tokens: 3,
                total_tokens: 10,
                accounting_provenance: {
                    input_tokens: { method: 'reported', accounting_basis: 'provider-test' },
                    output_tokens: { method: 'reported', accounting_basis: 'provider-test' },
                    total_tokens: { method: 'derived', accounting_basis: 'provider-test' },
                },
            },
            request_receipt: {
                id: 'request-receipt',
                request_id: 'request',
                attempt_id: 'generation-attempt',
                source: { conversation_id: 'json-processing', revision: 0 },
                context_fingerprint: 'sha256:ctx',
                tool_set_fingerprint: 'sha256:tools',
                request_fingerprint: 'sha256:request',
                target: { provider: 'provider', protocol: 'protocol', model: 'model', adapter_version: '1' },
                tool_definition_ids: [],
                asset_versions: [],
                item_mappings: [],
                recorded_at: at,
            },
        };
        const agent = createGeneratedAgentTurn({
            id: 'agent',
            generation_id: generation.id,
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at, completed_at: at },
            metadata: { preserved: true },
            provenance: { type: 'generated' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'agent-json', text: ' { "a": 1 } ', format: 'plain' })],
        });
        const program = createProgramTurn({
            id: 'program',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            presentation: 'transcript',
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'program-json', text: ' [ 2 ] ', format: 'plain' })],
        });
        const { store, jobId } = await fixture([], { turns: [agent, program], generations: [generation] });
        await run(store, jobId, capability());
        const replacements = Object.values(store.current.compactions)[0].replacement_turns;
        expect(replacements[0]).toMatchObject({
            kind: 'agent',
            generation_id: 'generation',
            timestamps: agent.timestamps,
            metadata: agent.metadata,
        });
        expect(replacements[1]).toMatchObject({ kind: 'program', authority: 'ordinary', presentation: 'transcript' });
        expect(store.current.generations.generation).toEqual(generation);
        expect(store.current.turns).toEqual([agent, program]);
        expect(await verifyDerivedBlockLineage(store.current)).toEqual(store.current);
    });
    it('bounds full lineage with unchanged blocks before host count/output publication', async () => {
        const item = createUserTurn({
            id: 'many-blocks',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: Array.from({ length: 257 }, (_, i) =>
                createTextBlock({ id: `many:${i}`, text: i === 0 ? ' [ 1 ] ' : '[1]', format: 'plain' }),
            ),
        });
        const { store, jobId } = await fixture([], { turns: [item] });
        const host = capability();
        await run(store, jobId, host);
        expect(host.measureProspective).not.toHaveBeenCalled();
        expect(store.current.processing.completions?.[jobId]?.status).toBe('blocked');
        expect(store.current.compactions).toEqual({});
    });
    it('rejects protected and incomplete selected turns without counting or transforming', async () => {
        const item = createProgramTurn({
            id: 'protected',
            authority: 'developer',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'protected-json', text: ' [ 1 ] ', format: 'plain' })],
        });
        await expect(fixture([], { turns: [item] })).rejects.toThrow('selection was rejected');
        const { store, jobId } = await fixture([], {
            turns: [{ ...item, authority: 'ordinary', status: 'interrupted' }],
        });
        const host = capability();
        await run(store, jobId, host);
        expect(host.measureProspective).not.toHaveBeenCalled();
        expect(store.current.compactions).toEqual({});
        expect(store.current.processing.completions?.[jobId]?.status).toBe('blocked');
    });

    it('recovers a lost output acknowledgement without a second processor or count', async () => {
        const { store, jobId } = await fixture();
        const host = capability();
        const commit = store.commit.bind(store);
        let lost = false;
        store.commit = async (revision, next) => {
            const ok = await commit(revision, next);
            if (ok && !lost && next.processing.outputs?.[jobId]) {
                lost = true;
                throw new Error('lost output acknowledgement');
            }
            return ok;
        };
        await expect(run(store, jobId, host)).rejects.toThrow('lost output acknowledgement');
        expect(store.current.processing.outputs?.[jobId]?.kind).toBe('json_minification');
        await run(store, jobId, host);
        expect(host.measureProspective).toHaveBeenCalledTimes(1);
        expect(Object.keys(store.current.compactions)).toHaveLength(1);
    });
    it('classifies its deadline as unavailable when the real host rejects on abort', async () => {
        const { store, jobId } = await fixture();
        const host = capability();
        let entered: (() => void) | undefined;
        const started = new Promise<void>((resolve) => {
            entered = resolve;
        });
        host.measureProspective = (_input, signal) => {
            entered?.();
            return new Promise((_resolve, reject) => {
                signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
            });
        };
        vi.useFakeTimers();
        try {
            const pending = run(store, jobId, host);
            await started;
            await vi.runAllTimersAsync();
            await pending;
        } finally {
            vi.useRealTimers();
        }
        expect(store.current.processing.outputs?.[jobId]).toMatchObject({
            kind: 'json_minification_no_op',
            reason: 'measurement_unavailable',
        });
        expect(store.current.processing.completions?.[jobId]?.status).toBe('no_op');
        expect(store.current.compactions).toEqual({});
    });
});
