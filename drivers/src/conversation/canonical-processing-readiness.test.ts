import type { CanonicalProjectedRequestMeasurement, ExecutionOptions } from '@llumiverse/common';
import { deriveCanonicalProjectedMeasurementIdentity } from '@llumiverse/common/schemas';
import {
    assertProcessingReady,
    createConversationDocument,
    deriveConversationId,
    fingerprintJson,
    type ModelTarget,
    parseConversationDocument,
    processingContextFingerprint,
    recordProcessingCoverage,
    setProcessingPolicy,
} from '@llumiverse/conversation';
import { PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIChatCompletionsDriver } from '../openai/openai_chat_completions.js';
import {
    appendCanonicalPrompt,
    type CanonicalPreparedStateBase,
    createRequestReceipt,
    publishCanonicalPreparedRequest,
} from './canonical-runtime.js';

const at = '2026-10-02T00:00:00.000Z';
const runtime = {
    conversation_id: 'readiness:source',
    request_id: 'readiness:request',
    attempt_id: 'readiness:attempt',
    input_operation_id: 'readiness:input',
    response_operation_id: 'readiness:response',
    recorded_at: at,
    started_at: at,
    purpose: 'interaction' as const,
};
const target: ModelTarget = {
    provider: 'openai_compatible',
    protocol: 'openai.chat.completions',
    model: 'gpt-4o-2024-08-06',
    adapter_version: 'readiness:v1',
};
const payload = { model: target.model, stream: false, messages: [{ role: 'user', content: 'owned input' }] };

async function fixture(kind: 'count_only' | 'absent' | 'stale' | 'pending' | 'ready' | 'over_limit') {
    const policy = await setProcessingPolicy(
        createConversationDocument({ id: runtime.conversation_id, created_at: at }),
        {
            operation_id: 'readiness:policy',
            expected_revision: 0,
            recorded_at: at,
            enabled: true,
            budget: { max_input_tokens: 100, output_reserve_tokens: 10, measurement_policy: 'identified_estimate' },
            processors:
                kind === 'pending'
                    ? [
                          {
                              id: 'required:noop',
                              version: 'v1',
                              config: {},
                              scope: 'on_append',
                              required: true,
                              failure_behavior: 'block',
                          },
                      ]
                    : [],
        },
    );
    let document = (
        await appendCanonicalPrompt(
            policy.document,
            {
                turns: [
                    {
                        id: 'readiness:turn',
                        kind: 'user',
                        authority: 'ordinary',
                        model_visibility: 'include',
                        status: 'completed',
                        timestamps: { recorded_at: at },
                        provenance: { type: 'received' },
                        blocks: [{ id: 'readiness:block', type: 'text', format: 'plain', text: 'owned input' }],
                    },
                ],
                assets: [],
                item_mappings: [],
                context_entries: [{ id: 'readiness:entry', type: 'source_turn', turn_id: 'readiness:turn' }],
            },
            runtime,
            undefined,
            payload,
        )
    ).document;
    const projection: CanonicalProjectedRequestMeasurement = {
        measurement: {
            input_tokens: 7,
            method: 'estimated',
            tokenizer: 'named:bpe',
            tokenizer_version: 'v1',
            adapter: target.protocol,
            adapter_version: target.adapter_version,
            target_model: target.model,
            source_fingerprint: await processingContextFingerprint(document),
            measured_at: at,
        },
        counted_request_fingerprint: await fingerprintJson(payload),
        ...(kind === 'count_only'
            ? {}
            : {
                  readiness: {
                      profile: 'full-native-json-bpe-v1',
                      output_reserve_tokens: 10,
                      context_limit: kind === 'over_limit' ? 16 : 128000,
                  },
              }),
    };
    if (kind === 'stale' || kind === 'pending' || kind === 'ready' || kind === 'over_limit') {
        document = (
            await recordProcessingCoverage(document, {
                operation_id: 'readiness:coverage',
                expected_revision: document.revision,
                target_fingerprint: await fingerprintJson(target),
                measured_input_tokens: 7,
                tokenizer_id: 'named:bpe',
                measurement_fingerprint:
                    kind === 'stale'
                        ? 'sha256:stale-counted-body'
                        : await deriveCanonicalProjectedMeasurementIdentity(projection, target),
                recorded_at: at,
            })
        ).document;
    }
    const receipt = await createRequestReceipt(document, runtime, target, payload, [], []);
    const state: CanonicalPreparedStateBase = {
        document,
        runtime,
        receipt: { ...receipt, measurement: projection.measurement },
        tool_definitions: [],
        generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
        response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
    };
    return { state, projection };
}

describe('shared fresh processing readiness publication barrier', () => {
    it.each(['count_only', 'absent', 'stale', 'pending', 'over_limit'] as const)(
        'holds %s coverage before host publication or transport',
        async (kind) => {
            const { state, projection } = await fixture(kind);
            const publish = vi.fn(async () => undefined);
            const sdk = vi.fn();
            await expect(
                publishCanonicalPreparedRequest(
                    state,
                    { model: target.model, on_canonical_request_prepared: publish },
                    projection,
                ).then(() => sdk()),
            ).rejects.toThrow(/processing|Processing/);
            expect(publish).not.toHaveBeenCalled();
            expect(sdk).not.toHaveBeenCalled();
        },
    );
    it('accepts genuinely ready coverage bound to freshly derived source, target and count identity', async () => {
        const { state, projection } = await fixture('ready');
        const publish = vi.fn(async () => undefined);
        expect(
            await publishCanonicalPreparedRequest(
                state,
                { model: target.model, on_canonical_request_prepared: publish },
                projection,
            ),
        ).toBeDefined();
        expect(publish).toHaveBeenCalledTimes(1);
    });
    it('owns readiness before asynchronous parsing and forwards the verified original projection', async () => {
        const { state, projection } = await fixture('ready');
        const original = structuredClone(projection);
        const publish = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(async () => undefined);
        const result = publishCanonicalPreparedRequest(
            state,
            { model: target.model, on_canonical_request_prepared: publish },
            projection,
        );
        if (projection.readiness === undefined) throw new Error('Fixture requires readiness');
        projection.readiness.context_limit = 1;
        projection.measurement.input_tokens = 999;
        await expect(result).resolves.toBeDefined();
        expect(publish.mock.calls[0]?.[1]).toEqual(original);
    });
});

it.each([false, true])(
    'holds actual Chat with a legitimate count callback and pending required job (readiness fields=%s)',
    async (withReadiness) => {
        const policy = await setProcessingPolicy(
            createConversationDocument({ id: runtime.conversation_id, created_at: at }),
            {
                operation_id: 'chat:policy',
                expected_revision: 0,
                recorded_at: at,
                enabled: true,
                budget: { max_input_tokens: 100, output_reserve_tokens: 10, measurement_policy: 'identified_estimate' },
                processors: [
                    {
                        id: 'required:noop',
                        version: 'v1',
                        scope: 'on_append',
                        config: {},
                        required: true,
                        failure_behavior: 'block',
                    },
                ],
            },
        );
        const driver = new OpenAIChatCompletionsDriver({
            apiKey: 'offline-test-only',
            endpoint: 'http://unused.invalid',
        });
        const sdk = vi
            .spyOn(driver.service.chat.completions, 'create')
            .mockRejectedValue(new Error('Unready input must not dispatch Chat'));
        const publish = vi.fn(async () => undefined);
        const count = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_projected']>>(async (projection) => ({
            counted_request_fingerprint: await fingerprintJson(projection.native_request),
            measurement: {
                input_tokens: 7,
                method: 'estimated',
                tokenizer: 'named:bpe',
                tokenizer_version: 'v1',
                adapter: projection.target.protocol,
                adapter_version: projection.target.adapter_version,
                target_model: projection.target.model,
                source_fingerprint: await processingContextFingerprint(projection.document),
                measured_at: at,
            },
            ...(withReadiness
                ? {
                      readiness: {
                          profile: 'full-native-json-bpe-v1',
                          output_reserve_tokens: 10,
                          context_limit: 128000,
                      },
                  }
                : {}),
        }));
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'owned input' }], {
                model: target.model,
                conversation: policy.document,
                conversation_runtime: runtime,
                on_canonical_request_projected: count,
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow(/processing|Processing/);
        expect(count).toHaveBeenCalledTimes(1);
        expect(publish).not.toHaveBeenCalled();
        expect(sdk).not.toHaveBeenCalled();
    },
);

describe('disabled policy imported outstanding job drain', () => {
    it.each([false, true])(
        'holds unsuperseded jobs before prepared publication (count-only evidence=%s)',
        async (withCount) => {
            const { state, projection } = await fixture('pending');
            const persisted = JSON.parse(JSON.stringify(state.document));
            persisted.processing.enabled = false;
            state.document = parseConversationDocument(persisted);
            expect(state.document.processing.enabled).toBe(false);
            await expect(assertProcessingReady(state.document, '', '')).rejects.toThrow('remain outstanding');
            const { readiness: _unusedReadiness, ...countOnly } = projection;
            const publish = vi.fn(async () => undefined);
            const sdk = vi.fn();
            await expect(
                publishCanonicalPreparedRequest(
                    state,
                    { model: target.model, on_canonical_request_prepared: publish },
                    withCount ? countOnly : undefined,
                ).then(() => sdk()),
            ).rejects.toThrow('remain outstanding');
            expect(publish).not.toHaveBeenCalled();
            expect(sdk).not.toHaveBeenCalled();
        },
    );
    it.each([false, true])('preserves ordinary disabled input (count-only evidence=%s)', async (withCount) => {
        const { state, projection } = await fixture('absent');
        const persisted = JSON.parse(JSON.stringify(state.document));
        persisted.processing.enabled = false;
        state.document = parseConversationDocument(persisted);
        const { readiness: _unusedReadiness, ...countOnly } = projection;
        const publish = vi.fn(async () => undefined);
        await expect(
            publishCanonicalPreparedRequest(
                state,
                { model: target.model, on_canonical_request_prepared: publish },
                withCount ? countOnly : undefined,
            ),
        ).resolves.toBeDefined();
        expect(publish).toHaveBeenCalledOnce();
    });
});
