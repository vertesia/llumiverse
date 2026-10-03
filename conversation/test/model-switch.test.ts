import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import {
    appendConversationRecordsWithProcessing,
    applyModelSwitch,
    ConversationModelSwitchNextRequestChangeSchema,
    ConversationModelSwitchPlanSchema,
    type ConversationModelSwitchRequest,
    ConversationModelSwitchRequestSchema,
    type ConversationModelSwitchRuntime,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    prepareModelSwitch,
    processingContextFingerprint,
    recordProcessingCoverage,
    setProcessingPolicy,
} from '../src/index.js';
import {
    CONVERSATION_JSON_SCHEMAS,
    ConversationModelSwitchBlockerJsonSchema,
    ConversationModelSwitchBudgetAnalysisJsonSchema,
    ConversationModelSwitchNextRequestChangeJsonSchema,
    ConversationModelSwitchPlanJsonSchema,
    ConversationModelSwitchRequestJsonSchema,
} from '../src/json-schema.js';
import { generatedAgentTurn, importedGeneration, toolCallBlock } from './fixtures.js';

const at = '2026-10-03T00:00:00.000Z';
const source = () => createConversationDocument({ id: 'switch:conversation', created_at: at });
const target = {
    provider: 'openai',
    protocol: 'openai.chat.completions',
    model: 'new-model',
    adapter_version: 'adapter:v1',
    options: { temperature: 0.2 },
};

async function runtimeFor(document = source()): Promise<ConversationModelSwitchRuntime> {
    const sourceFingerprint = await processingContextFingerprint(document);
    const nativeRequest = { model: target.model, messages: [], stream: false };
    const countedRequestFingerprint = await fingerprintJson(nativeRequest);
    return {
        hasUnsettledGeneration: async () => false,
        project: async () => ({
            status: 'compiled',
            native_request: nativeRequest,
            counted_request_fingerprint: countedRequestFingerprint,
            measurement: {
                input_tokens: 12,
                method: 'exact',
                tokenizer: 'native-count',
                adapter: target.protocol,
                adapter_version: target.adapter_version,
                source_fingerprint: sourceFingerprint,
                target_model: target.model,
                measured_at: at,
            },
            context_limit_tokens: 100,
            output_reserve_tokens: 20,
        }),
    };
}

function request(document = source()) {
    return {
        source: { conversation_id: document.id, revision: document.revision },
        expected_context_revision: document.context.revision,
        target: structuredClone(target),
    };
}

describe('revision-bound model switch proposal', () => {
    it('binds provider-count opt-in to plan, apply and JSON contract without weakening source checks', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const plan = await prepareModelSwitch(
            document,
            { ...request(document), measurement_mode: 'provider' },
            runtime,
        );
        expect(plan.measurement_mode).toBe('provider');
        const change = await applyModelSwitch(
            document,
            plan,
            { operation_id: 'switch:provider-count', recorded_at: at },
            runtime,
        );
        expect(change.measurement_mode).toBe('provider');
        const changedSource = { ...document, id: 'switch:changed-source' };
        await expect(
            applyModelSwitch(
                changedSource,
                plan,
                { operation_id: 'switch:stale-provider-count', recorded_at: at },
                runtime,
            ),
        ).rejects.toThrow('changed before apply');
        expect(
            ConversationModelSwitchPlanSchema.safeParse({ ...plan, measurement_mode: 'remote-inference' }).success,
        ).toBe(false);
    });
    it('exports the frozen JSON lookup and preserves Zod/JSON Schema round trips', async () => {
        const named = {
            conversation_model_switch_request: ConversationModelSwitchRequestJsonSchema,
            conversation_model_switch_blocker: ConversationModelSwitchBlockerJsonSchema,
            conversation_model_switch_budget_analysis: ConversationModelSwitchBudgetAnalysisJsonSchema,
            conversation_model_switch_plan: ConversationModelSwitchPlanJsonSchema,
            conversation_model_switch_next_request_change: ConversationModelSwitchNextRequestChangeJsonSchema,
        };
        for (const [key, schema] of Object.entries(named)) {
            expect(Reflect.get(CONVERSATION_JSON_SCHEMAS, key)).toBe(schema);
            expect(Object.isFrozen(schema)).toBe(true);
        }
        const document = source();
        const plan = await prepareModelSwitch(document, request(document), await runtimeFor(document));
        const change = await applyModelSwitch(
            document,
            plan,
            { operation_id: 'switch:json', recorded_at: at },
            await runtimeFor(document),
        );
        const ajv = new Ajv2020({ allErrors: true, strict: true });
        formatsPlugin.default(ajv);
        for (const [zodSchema, jsonSchema, value] of [
            [ConversationModelSwitchPlanSchema, ConversationModelSwitchPlanJsonSchema, plan],
            [
                ConversationModelSwitchNextRequestChangeSchema,
                ConversationModelSwitchNextRequestChangeJsonSchema,
                change,
            ],
        ] as const) {
            const roundTrip: unknown = JSON.parse(JSON.stringify(value));
            expect(zodSchema.safeParse(roundTrip).success).toBe(true);
            const validate = ajv.compile(jsonSchema);
            expect(validate(roundTrip), JSON.stringify(validate.errors)).toBe(true);
            const invalid = { ...value, active_tool_definition_ids: [42] };
            expect(zodSchema.safeParse(invalid).success).toBe(false);
            expect(validate(invalid)).toBe(false);
        }
    });
    it('plans compatible native request and returns only a CAS-bound next-request change', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const plan = await prepareModelSwitch(document, request(document), runtime);
        expect(plan.compatibility).toBe('compatible');
        expect(plan.blockers).toEqual([]);
        expect(plan.budget).toMatchObject({ available_input_tokens: 80, measured_input_tokens: 12 });
        expect(plan.media_transformations).toEqual([]);
        expect(plan.replay_exclusions).toEqual([]);
        const change = await applyModelSwitch(
            document,
            plan,
            { operation_id: 'switch:next', recorded_at: at },
            runtime,
        );
        expect(change.expected_source).toEqual({ conversation_id: document.id, revision: document.revision });
        expect(change.expected_context_revision).toBe(document.context.revision);
        expect(change.target).toEqual(target);
        expect(change.requires_quiescent_generation).toBe(true);
        expect(change.cache_invalidations).toEqual(['native_request', 'provider_cache', 'provider_upload']);
    });

    it('blocks unknown budget, unsupported target and a live generation without issuing an apply change', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const noBudget = await prepareModelSwitch(document, request(document), {
            ...runtime,
            project: async () => ({
                status: 'compiled',
                native_request: { model: target.model },
            }),
        });
        expect(noBudget.blockers.map((blocker) => blocker.code)).toEqual([
            'MEASUREMENT_UNAVAILABLE',
            'BUDGET_UNAVAILABLE',
        ]);
        await expect(
            applyModelSwitch(document, noBudget, { operation_id: 'switch:blocked', recorded_at: at }, runtime),
        ).rejects.toThrow('Blocked model switch');
        const unsupported = await prepareModelSwitch(document, request(document), {
            ...runtime,
            project: async () => ({ status: 'unsupported', reason: 'Protected replay cannot move to this model' }),
        });
        expect(unsupported.blockers).toContainEqual({
            code: 'TARGET_UNSUPPORTED',
            message: 'Protected replay cannot move to this model',
        });
        const unsettled = await prepareModelSwitch(document, request(document), {
            ...runtime,
            hasUnsettledGeneration: async () => true,
        });
        expect(unsettled.blockers.some((blocker) => blocker.code === 'UNSETTLED_GENERATION')).toBe(true);
    });

    it('rechecks source, native request and quiescence at apply', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const plan = await prepareModelSwitch(document, request(document), runtime);
        const mutated = { ...document, revision: document.revision + 1 };
        await expect(
            applyModelSwitch(mutated, plan, { operation_id: 'switch:stale', recorded_at: at }, runtime),
        ).rejects.toThrow();
        await expect(
            applyModelSwitch(
                document,
                plan,
                { operation_id: 'switch:new-payload', recorded_at: at },
                {
                    ...runtime,
                    project: async () => ({
                        ...(await runtime.project(document, target)),
                        status: 'compiled',
                        native_request: { model: target.model, messages: [{ role: 'user', content: 'changed' }] },
                    }),
                },
            ),
        ).rejects.toThrow('native request or budget changed');
        await expect(
            applyModelSwitch(
                document,
                plan,
                { operation_id: 'switch:owed', recorded_at: at },
                {
                    ...runtime,
                    hasUnsettledGeneration: async () => true,
                },
            ),
        ).rejects.toThrow('native request or budget changed');
    });

    it('blocks a retained open provider tool dependency before changing target', async () => {
        const document = source();
        document.turns.push(
            generatedAgentTurn('agent:pending', 'generation:pending', [
                toolCallBlock('block:pending', 'call:pending', 'provider'),
            ]),
        );
        document.generations['generation:pending'] = {
            ...importedGeneration('generation:pending'),
            source: { conversation_id: document.id, revision: document.revision },
        };
        const plan = await prepareModelSwitch(document, request(document), await runtimeFor(document));
        expect(plan.blockers.some((blocker) => blocker.code === 'OPEN_TOOL_CALL')).toBe(true);
    });

    it('does not treat disabled policy with an outstanding accepted job as ready', async () => {
        const enabled = await setProcessingPolicy(source(), {
            operation_id: 'policy:enabled',
            expected_revision: 0,
            recorded_at: at,
            enabled: true,
            processors: [
                {
                    id: 'processor:text',
                    version: '1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const turn = createUserTurn({
            id: 'turn:pending',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'block:pending', text: 'pending', format: 'plain' })],
        });
        const accepted = await appendConversationRecordsWithProcessing(
            enabled.document,
            { turns: [turn], context_entries: [{ id: 'entry:pending', type: 'source_turn', turn_id: turn.id }] },
            {
                operation_id: 'append:pending',
                expected_revision: 1,
                payload_fingerprint: 'sha256:pending',
                recorded_at: at,
            },
        );
        const disabled = { ...accepted.document, processing: { ...accepted.document.processing, enabled: false } };
        const plan = await prepareModelSwitch(disabled, request(disabled), await runtimeFor(disabled));
        expect(plan.blockers.some((blocker) => blocker.code === 'PROCESSING_PENDING')).toBe(true);
    });

    it('distinguishes pending coverage from a blocked accepted budget evaluation', async () => {
        const enabled = await setProcessingPolicy(source(), {
            operation_id: 'policy:budget',
            expected_revision: 0,
            recorded_at: at,
            enabled: true,
            processors: [],
            budget: { max_input_tokens: 10, output_reserve_tokens: 2 },
        });
        const runtime = await runtimeFor(enabled.document);
        const projection = await runtime.project(enabled.document, target);
        if (projection.status !== 'compiled') throw new Error('Expected compiled fixture');
        const measured: ConversationModelSwitchRuntime = {
            ...runtime,
            project: async () => ({ ...projection, processing_measurement_fingerprint: 'sha256:measure' }),
        };
        const pending = await prepareModelSwitch(enabled.document, request(enabled.document), measured);
        expect(pending.blockers.map((blocker) => blocker.code)).toContain('PROCESSING_PENDING');
        const coverage = await recordProcessingCoverage(enabled.document, {
            operation_id: 'coverage:blocked',
            expected_revision: enabled.document.revision,
            target_fingerprint: await fingerprintJson(target),
            measured_input_tokens: 12,
            tokenizer_id: 'native-count',
            measurement_fingerprint: 'sha256:measure',
            recorded_at: at,
        });
        expect(coverage.coverage.status).toBe('blocked');
        const fresh = await runtimeFor(coverage.document);
        const freshProjection = await fresh.project(coverage.document, target);
        if (freshProjection.status !== 'compiled') throw new Error('Expected compiled fixture');
        const blocked = await prepareModelSwitch(coverage.document, request(coverage.document), {
            ...fresh,
            project: async () => ({ ...freshProjection, processing_measurement_fingerprint: 'sha256:measure' }),
        });
        expect(blocked.blockers.map((blocker) => blocker.code)).toContain('PROCESSING_BLOCKED');
    });

    it('binds counting to the exact native body and enforces both reserve and measured budget', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const projection = await runtime.project(document, target);
        if (projection.status !== 'compiled' || !projection.measurement) throw new Error('Expected compiled fixture');
        const measurement = projection.measurement;
        const mismatched = await prepareModelSwitch(document, request(document), {
            ...runtime,
            project: async () => ({ ...projection, counted_request_fingerprint: 'sha256:other' }),
        });
        expect(mismatched.blockers.map((blocker) => blocker.code)).toContain('MEASUREMENT_MISMATCH');
        const overReserve = await prepareModelSwitch(document, request(document), {
            ...runtime,
            project: async () => ({ ...projection, output_reserve_tokens: 100 }),
        });
        expect(overReserve.blockers.map((blocker) => blocker.code)).toContain('BUDGET_UNSATISFIED');
        const overInput = await prepareModelSwitch(document, request(document), {
            ...runtime,
            project: async () => ({ ...projection, measurement: { ...measurement, input_tokens: 81 } }),
        });
        expect(overInput.blockers.map((blocker) => blocker.code)).toContain('BUDGET_UNSATISFIED');
    });

    it('requires tokenizer identity and an explicit identified-estimate policy', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const projection = await runtime.project(document, target);
        if (projection.status !== 'compiled' || !projection.measurement) throw new Error('Expected compiled fixture');
        for (const measurement of [
            { ...projection.measurement, method: 'estimated' as const },
            { ...projection.measurement, method: 'estimated' as const, tokenizer_version: 'v1' },
        ]) {
            const plan = await prepareModelSwitch(document, request(document), {
                ...runtime,
                project: async () => ({ ...projection, measurement }),
            });
            expect(plan.blockers.map((blocker) => blocker.code)).toContain('MEASUREMENT_MISMATCH');
        }
    });

    it('binds an ordinary estimate opt-in without weakening a retained exact-only source policy', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const projection = await runtime.project(document, target);
        if (projection.status !== 'compiled' || !projection.measurement) throw new Error('Expected compiled fixture');
        const baselineMeasurement = projection.measurement;
        const estimated: ConversationModelSwitchRuntime = {
            ...runtime,
            project: async () => ({
                ...projection,
                measurement: {
                    ...baselineMeasurement,
                    method: 'estimated',
                    tokenizer_version: 'identified:v1',
                },
            }),
        };
        const baseline = await prepareModelSwitch(document, request(document), estimated);
        expect(baseline.blockers.map((blocker) => blocker.code)).toContain('MEASUREMENT_MISMATCH');
        const opted = await prepareModelSwitch(
            document,
            { ...request(document), measurement_policy: 'identified_estimate' },
            estimated,
        );
        expect(opted.compatibility).toBe('compatible');
        expect(opted.measurement_policy).toBe('identified_estimate');
        const change = await applyModelSwitch(
            document,
            opted,
            { operation_id: 'switch:identified-estimate', recorded_at: at },
            estimated,
        );
        expect(change.measurement_policy).toBe('identified_estimate');
        const ajv = new Ajv2020({ allErrors: true, strict: true });
        formatsPlugin.default(ajv);
        for (const [schema, jsonSchema, value] of [
            [
                ConversationModelSwitchRequestSchema,
                ConversationModelSwitchRequestJsonSchema,
                {
                    ...request(document),
                    measurement_policy: 'identified_estimate',
                },
            ],
            [ConversationModelSwitchPlanSchema, ConversationModelSwitchPlanJsonSchema, opted],
            [
                ConversationModelSwitchNextRequestChangeSchema,
                ConversationModelSwitchNextRequestChangeJsonSchema,
                change,
            ],
        ] as const) {
            const roundTrip: unknown = JSON.parse(JSON.stringify(value));
            expect(schema.safeParse(roundTrip).success).toBe(true);
            const validate = ajv.compile(jsonSchema);
            expect(validate(roundTrip), JSON.stringify(validate.errors)).toBe(true);
            const invalid = { ...value, measurement_policy: 'unidentified' };
            expect(schema.safeParse(invalid).success).toBe(false);
            expect(validate(invalid)).toBe(false);
        }
        await expect(
            applyModelSwitch(
                document,
                { ...opted, measurement_policy: 'exact_only' },
                { operation_id: 'switch:tampered-policy', recorded_at: at },
                estimated,
            ),
        ).rejects.toThrow('Model switch source');

        const exact = await setProcessingPolicy(source(), {
            operation_id: 'policy:exact-model-switch',
            expected_revision: 0,
            recorded_at: at,
            enabled: false,
            processors: [],
            budget: { max_input_tokens: 100, output_reserve_tokens: 20, measurement_policy: 'exact_only' },
        });
        const exactRuntime = await runtimeFor(exact.document);
        const exactProjection = await exactRuntime.project(exact.document, target);
        if (exactProjection.status !== 'compiled' || !exactProjection.measurement)
            throw new Error('Expected compiled exact-source fixture');
        const exactMeasurement = exactProjection.measurement;
        const exactPlan = await prepareModelSwitch(
            exact.document,
            { ...request(exact.document), measurement_policy: 'identified_estimate' },
            {
                ...exactRuntime,
                project: async () => ({
                    ...exactProjection,
                    measurement: {
                        ...exactMeasurement,
                        method: 'estimated',
                        tokenizer_version: 'identified:v1',
                    },
                }),
            },
        );
        expect(exactPlan.blockers.map((blocker) => blocker.code)).toContain('MEASUREMENT_MISMATCH');

        const identified = await setProcessingPolicy(source(), {
            operation_id: 'policy:identified-model-switch',
            expected_revision: 0,
            recorded_at: at,
            enabled: false,
            processors: [],
            budget: { max_input_tokens: 100, output_reserve_tokens: 20, measurement_policy: 'identified_estimate' },
        });
        const identifiedRuntime = await runtimeFor(identified.document);
        const identifiedProjection = await identifiedRuntime.project(identified.document, target);
        if (identifiedProjection.status !== 'compiled' || !identifiedProjection.measurement)
            throw new Error('Expected compiled identified-source fixture');
        const identifiedMeasurement = identifiedProjection.measurement;
        const stricter = await prepareModelSwitch(
            identified.document,
            { ...request(identified.document), measurement_policy: 'exact_only' },
            {
                ...identifiedRuntime,
                project: async () => ({
                    ...identifiedProjection,
                    measurement: {
                        ...identifiedMeasurement,
                        method: 'estimated',
                        tokenizer_version: 'identified:v1',
                    },
                }),
            },
        );
        expect(stricter.blockers.map((blocker) => blocker.code)).toContain('MEASUREMENT_MISMATCH');
    });

    it('rejects apply after target, options or native request changes', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        const plan = await prepareModelSwitch(document, request(document), runtime);
        for (const changed of [
            { ...plan, target: { ...plan.target, model: 'other-model' } },
            { ...plan, target: { ...plan.target, options: { temperature: 0.9 } } },
            { ...plan, active_tool_definition_ids: ['tool:unaccepted'] },
        ]) {
            await expect(
                applyModelSwitch(document, changed, { operation_id: 'switch:drift', recorded_at: at }, runtime),
            ).rejects.toThrow();
        }
    });

    it('bounds malformed or oversized native projections', async () => {
        const document = source();
        const runtime = await runtimeFor(document);
        for (const nativeRequest of [{ value: () => 'not JSON' }, { text: 'x'.repeat(769 * 1024) }]) {
            await expect(
                prepareModelSwitch(document, request(document), {
                    ...runtime,
                    project: async () => ({ status: 'compiled', native_request: nativeRequest }),
                }),
            ).rejects.toThrow();
        }
    });

    it('owns source, request and callbacks before the first await', async () => {
        const document = source();
        const requested: ConversationModelSwitchRequest = { ...request(document), measurement_policy: 'exact_only' };
        const original = await runtimeFor(document);
        let release!: (value: boolean) => void;
        const observation = new Promise<boolean>((resolve) => {
            release = resolve;
        });
        const runtime: ConversationModelSwitchRuntime = {
            hasUnsettledGeneration: () => observation,
            project: original.project,
        };
        const pending = prepareModelSwitch(document, requested, runtime);
        document.revision = 99;
        requested.target.model = 'mutated';
        requested.measurement_policy = 'identified_estimate';
        runtime.project = async () => ({ status: 'unsupported', reason: 'late replacement' });
        release(false);
        const plan = await pending;
        expect(plan.source.revision).toBe(0);
        expect(plan.target.model).toBe('new-model');
        expect(plan.measurement_policy).toBe('exact_only');
        expect(plan.compatibility).toBe('compatible');
    });
});
