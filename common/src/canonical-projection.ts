import { fingerprintJson, type ModelTarget, preflightJsonInput } from '@llumiverse/conversation';
import { ModelTargetSchema } from '@llumiverse/conversation/schemas';
import { CanonicalProjectedRequestMeasurementSchema } from './schemas/canonical-projection.js';
import type { CanonicalProjectedRequestMeasurement } from './types.js';

/** Runtime-only identity shared by processing coverage and the actual prepared-request barrier. */
export async function deriveCanonicalProjectedMeasurementIdentity(
    input: CanonicalProjectedRequestMeasurement,
    targetInput: ModelTarget,
): Promise<string> {
    if (
        !preflightJsonInput(input, { max_bytes: 64 * 1024 }).success ||
        !preflightJsonInput(targetInput, { max_bytes: 768 * 1024 }).success
    )
        throw new RangeError('Canonical processing measurement identity exceeds its bounded envelope');
    const owned = CanonicalProjectedRequestMeasurementSchema.parse(structuredClone(input));
    const target = ModelTargetSchema.parse(structuredClone(targetInput));
    if (owned.readiness === undefined) throw new Error('Canonical processing readiness identity is unavailable');
    if (
        owned.measurement.target_model !== target.model ||
        owned.measurement.adapter !== target.protocol ||
        owned.measurement.adapter_version !== target.adapter_version
    )
        throw new Error('Canonical processing measurement target conflicts');
    const { measured_at: _measuredAt, ...measurement } = owned.measurement;
    return fingerprintJson({
        measurement,
        counted_request_fingerprint: owned.counted_request_fingerprint,
        target_fingerprint: await fingerprintJson(target),
        output_reserve_tokens: owned.readiness.output_reserve_tokens,
        context_limit: owned.readiness.context_limit,
        profile: owned.readiness.profile,
    });
}
