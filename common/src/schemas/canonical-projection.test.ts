import { fingerprintJson } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { deriveCanonicalProjectedMeasurementIdentity } from '../canonical-projection.js';
import * as CommonRoot from '../index.js';
import type { CanonicalProjectedRequestMeasurement, ExecutionOptions } from '../types.js';
import {
    CanonicalProjectedRequestMeasurementSchema,
    deriveCanonicalProjectedMeasurementIdentity as publicValidationHelper,
    StatelessExecutionOptionsSchema,
} from './index.js';

const evidence: CanonicalProjectedRequestMeasurement = {
    counted_request_fingerprint: 'sha256:actual-sdk-json',
    measurement: {
        input_tokens: 12,
        method: 'estimated',
        tokenizer: 'full-native-json-bpe-v1',
        tokenizer_version: '1.0.22',
        adapter: 'openai.chat.completions',
        adapter_version: 'v1',
        source_fingerprint: 'sha256:source-context',
        target_model: 'gpt-4o-2024-08-06',
        measured_at: '2026-10-02T00:00:00.000Z',
    },
};

describe('runtime-only canonical projection contract', () => {
    it('roundtrips typed count evidence with distinct source and actual request fingerprints', () => {
        expect(CanonicalProjectedRequestMeasurementSchema.parse(JSON.parse(JSON.stringify(evidence)))).toEqual(
            evidence,
        );
        expect(evidence.counted_request_fingerprint).not.toBe(evidence.measurement.source_fingerprint);
    });

    it('rejects missing, unknown and malformed outer or nested evidence', () => {
        for (const invalid of [
            {},
            { ...evidence, counted_request_fingerprint: null },
            { ...evidence, counted_request_fingerprint: 17 },
            { ...evidence, authority: true },
            { ...evidence, measurement: { ...evidence.measurement, input_tokens: -1 } },
            { ...evidence, measurement: { ...evidence.measurement, input_tokens: Number.POSITIVE_INFINITY } },
            { ...evidence, measurement: { ...evidence.measurement, method: 'guaranteed_fit' } },
            { ...evidence, measurement: { ...evidence.measurement, provider_authority: true } },
        ]) {
            expect(CanonicalProjectedRequestMeasurementSchema.safeParse(invalid).success).toBe(false);
        }
    });

    it('keeps the callback typed at runtime and absent from serialized wire options', () => {
        const callback: NonNullable<ExecutionOptions['on_canonical_request_projected']> = async () => evidence;
        const options: ExecutionOptions = { model: 'gpt-4o-2024-08-06', on_canonical_request_projected: callback };
        expect(JSON.stringify(options)).toBe('{"model":"gpt-4o-2024-08-06"}');
        expect(StatelessExecutionOptionsSchema.safeParse(options).success).toBe(false);
        const wireSchema = z.toJSONSchema(StatelessExecutionOptionsSchema);
        expect(StatelessExecutionOptionsSchema.shape).not.toHaveProperty('on_canonical_request_projected');
        expect(JSON.stringify(wireSchema)).not.toContain('on_canonical_request_projected');
        expect(JSON.stringify(wireSchema)).not.toContain('on_canonical_request_prepared');
    });
});

it('keeps prepared hook count metadata optional and inferred from the same strict schema', () => {
    type PreparedProjection = Parameters<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>[1];
    const forwarded: PreparedProjection = evidence;
    const absent: PreparedProjection = undefined;
    expect(CanonicalProjectedRequestMeasurementSchema.parse(forwarded)).toEqual(evidence);
    expect(absent).toBeUndefined();
});

const readiness = { profile: 'full-native-json-bpe-v1', output_reserve_tokens: 100, context_limit: 128000 };
const target = {
    provider: 'openai_compatible',
    protocol: 'openai.chat.completions',
    adapter_version: 'v1',
    model: 'gpt-4o-2024-08-06',
};
it('derives the same historical coverage identity while owning inputs and excluding only event time', async () => {
    const input = { ...structuredClone(evidence), readiness: { ...readiness } };
    const { measured_at: _measuredAt, ...measurement } = evidence.measurement;
    const expected = await fingerprintJson({
        measurement,
        counted_request_fingerprint: evidence.counted_request_fingerprint,
        target_fingerprint: await fingerprintJson(target),
        ...readiness,
    });
    const ownedTargetInput = { ...target };
    const pending = deriveCanonicalProjectedMeasurementIdentity(input, ownedTargetInput);
    ownedTargetInput.model = 'changed after capture';
    input.readiness.output_reserve_tokens = 999;
    expect(await pending).toBe(expected);
    expect(
        await deriveCanonicalProjectedMeasurementIdentity(
            {
                ...evidence,
                readiness,
                measurement: { ...evidence.measurement, measured_at: '2026-10-03T00:00:00.000Z' },
            },
            target,
        ),
    ).toBe(expected);
    await expect(
        deriveCanonicalProjectedMeasurementIdentity(
            { ...evidence, readiness: { ...readiness, profile: 'x'.repeat(70 * 1024) } },
            target,
        ),
    ).rejects.toThrow('bounded envelope');
    await expect(deriveCanonicalProjectedMeasurementIdentity(evidence, target)).rejects.toThrow('unavailable');
    await expect(
        deriveCanonicalProjectedMeasurementIdentity({ ...evidence, readiness }, { ...target, model: 'changed' }),
    ).rejects.toThrow('target conflicts');
});
it('strictly bounds readiness without making count-only evidence a serialized wire option', () => {
    expect(CanonicalProjectedRequestMeasurementSchema.parse({ ...evidence, readiness }).readiness).toEqual(readiness);
    for (const invalid of [
        null,
        { ...readiness, authority: true },
        { ...readiness, context_limit: 0 },
        { ...readiness, profile: '' },
        { ...readiness, output_reserve_tokens: -1 },
        { ...readiness, context_limit: Number.POSITIVE_INFINITY },
    ])
        expect(CanonicalProjectedRequestMeasurementSchema.safeParse({ ...evidence, readiness: invalid }).success).toBe(
            false,
        );
    expect(CanonicalProjectedRequestMeasurementSchema.safeParse(evidence).success).toBe(true);
    expect(JSON.stringify(z.toJSONSchema(StatelessExecutionOptionsSchema))).not.toContain('readiness');
});

it('exposes the runtime validator only through the schemas subpath', () => {
    expect('deriveCanonicalProjectedMeasurementIdentity' in CommonRoot).toBe(false);
    expect(publicValidationHelper).toBe(deriveCanonicalProjectedMeasurementIdentity);
});
