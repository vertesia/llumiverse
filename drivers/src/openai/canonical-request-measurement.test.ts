import type { CanonicalRequestProjection, CanonicalRequestProjectionCompiler } from '@llumiverse/common';
import { createConversationDocument, fingerprintJson, processingContextFingerprint } from '@llumiverse/conversation';
import { describe, expect, it, vi } from 'vitest';
import {
    boundedCanonicalProjectionOperation,
    projectCanonicalRequestMeasurement,
} from './canonical-request-measurement.js';

const at = '2026-10-02T00:00:00.000Z';
function projection(): CanonicalRequestProjection {
    return {
        document: createConversationDocument({ id: 'projection:1', created_at: at }),
        runtime: {
            conversation_id: 'projection:1',
            request_id: 'request:1',
            attempt_id: 'attempt:1',
            input_operation_id: 'input:1',
            response_operation_id: 'response:1',
            recorded_at: at,
            started_at: at,
            purpose: 'interaction',
        },
        target: {
            provider: 'openai',
            protocol: 'openai.chat.completions',
            model: 'gpt-4o-2024-08-06',
            adapter_version: 'v1',
        },
        native_request: {
            model: 'gpt-4o-2024-08-06',
            stream: true,
            stream_options: { include_usage: true },
            messages: [{ role: 'user', content: 'owned' }],
        },
    };
}
const compiler: CanonicalRequestProjectionCompiler = {
    compileDocument: vi.fn(async () => projection().native_request),
    compileProspective: vi.fn(async () => ({ status: 'unavailable' as const })),
};
async function measurement(input: CanonicalRequestProjection) {
    return {
        counted_request_fingerprint: await fingerprintJson(input.native_request),
        measurement: {
            input_tokens: 7,
            method: 'estimated' as const,
            tokenizer: 'named:estimate',
            tokenizer_version: 'version:1',
            adapter: input.target.protocol,
            adapter_version: input.target.adapter_version,
            source_fingerprint: await processingContextFingerprint(input.document),
            target_model: input.target.model,
            measured_at: at,
        },
    };
}

describe('actual counted request projection barrier', () => {
    it('binds actual SDK transport flags without changing source context fingerprint semantics', async () => {
        const input = projection();
        const expected = await measurement(input);
        await expect(projectCanonicalRequestMeasurement(input, compiler, async () => expected)).resolves.toEqual(
            expected,
        );
        const changed = structuredClone(input);
        changed.native_request = {
            model: 'gpt-4o-2024-08-06',
            stream: true,
            stream_options: { include_usage: false },
            messages: [{ role: 'user', content: 'owned' }],
        };
        await expect(projectCanonicalRequestMeasurement(changed, compiler, async () => expected)).rejects.toThrow(
            'actual request',
        );
    });

    it('owns source/runtime/target/request before awaits and denies cross-source/target/adapter evidence', async () => {
        const input = projection();
        const expected = await measurement(input);
        const pending = projectCanonicalRequestMeasurement(input, compiler, async (owned) => {
            expect(owned.native_request).toEqual(projection().native_request);
            return expected;
        });
        input.native_request = { model: 'mutated' };
        input.target.model = 'mutated';
        input.runtime.request_id = 'mutated';
        expect(await pending).toEqual(expected);
        for (const changed of [
            { target_model: 'other-model' },
            { adapter_version: 'other-version' },
            { adapter: 'other-protocol' },
            { source_fingerprint: 'sha256:other' },
        ]) {
            await expect(
                projectCanonicalRequestMeasurement(projection(), compiler, async () => ({
                    ...expected,
                    measurement: { ...expected.measurement, ...changed },
                })),
            ).rejects.toThrow('source/target');
        }
    });

    it('does not accept missing or malformed count evidence for a required budget', async () => {
        const input = projection();
        input.document.processing.budget = { max_input_tokens: 100, output_reserve_tokens: 0 };
        input.document.processing.enabled = true;
        await expect(projectCanonicalRequestMeasurement(input, compiler, undefined)).rejects.toThrow('unavailable');
        await expect(projectCanonicalRequestMeasurement(input, compiler, async () => undefined)).rejects.toThrow(
            'unavailable',
        );
        const expected = await measurement(input);
        await expect(
            projectCanonicalRequestMeasurement(input, compiler, async () => ({
                ...expected,
                counted_request_fingerprint: 'sha256:wrong',
            })),
        ).rejects.toThrow('actual request');
        expect(await projectCanonicalRequestMeasurement(projection(), compiler, undefined)).toBeUndefined();
    });

    it('rejects a synchronous late resolve even when the event loop prevents the timer from firing', async () => {
        const prepared = vi.fn();
        const sdk = vi.fn();
        const guarded = async () => {
            await boundedCanonicalProjectionOperation(
                async () => {
                    // Test-only synchronous blocking reproduces the timer/microtask race without a busy loop.
                    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 15);
                    return 'late ACK';
                },
                undefined,
                2,
            );
            prepared();
            sdk();
        };
        await expect(guarded()).rejects.toThrow('deadline');
        expect(prepared).not.toHaveBeenCalled();
        expect(sdk).not.toHaveBeenCalled();
    });

    it('bounds compiler work and cancellation even when a callback ignores the signal', async () => {
        await expect(
            boundedCanonicalProjectionOperation(async () => new Promise(() => {}), undefined, 5),
        ).rejects.toThrow('deadline');
        const controller = new AbortController();
        const pending = boundedCanonicalProjectionOperation(async () => new Promise(() => {}), controller.signal);
        controller.abort(new Error('cancelled'));
        await expect(pending).rejects.toThrow('cancelled');
    });
});
