import type {
    CanonicalProjectedRequestMeasurement,
    CanonicalRequestProjection,
    CanonicalRequestProjectionCompiler,
    ExecutionOptions,
} from '@llumiverse/common';
import { CanonicalProjectedRequestMeasurementSchema } from '@llumiverse/common/schemas';
import {
    fingerprintJson,
    parseConversationDocument,
    preflightJsonInput,
    processingContextFingerprint,
} from '@llumiverse/conversation';
import {
    JsonValueSchema,
    ModelTargetSchema,
    ResolvedConversationRuntimeContextSchema,
} from '@llumiverse/conversation/schemas';
import { markCanonicalHostCallbackFailure } from '@llumiverse/core';

/** Own the whole projection before the first asynchronous callback or hash. */
export function ownCanonicalRequestProjection(input: CanonicalRequestProjection): CanonicalRequestProjection {
    if (!preflightJsonInput(input, { max_bytes: 16 * 1024 * 1024 }).success)
        throw new RangeError('Canonical request projection exceeds its materialized envelope');
    if (!preflightJsonInput(input.native_request, { max_bytes: 768 * 1024 }).success)
        throw new RangeError('Canonical native request exceeds its measurement envelope');
    const owned = structuredClone(input);
    return {
        document: parseConversationDocument(owned.document),
        runtime: ResolvedConversationRuntimeContextSchema.parse(owned.runtime),
        target: ModelTargetSchema.parse(owned.target),
        native_request: JsonValueSchema.parse(owned.native_request),
    };
}

/** The adapter validates count identity without changing its historical requestBinding fingerprint. */
export async function projectCanonicalRequestMeasurement(
    input: CanonicalRequestProjection,
    compiler: CanonicalRequestProjectionCompiler,
    callback: ExecutionOptions['on_canonical_request_projected'],
    signal?: AbortSignal,
): Promise<CanonicalProjectedRequestMeasurement | undefined> {
    const projection = ownCanonicalRequestProjection(input);
    const required = projection.document.processing.enabled && projection.document.processing.budget !== undefined;
    if (!callback) {
        if (required) throw new Error('Required processing measurement is unavailable');
        return undefined;
    }
    signal?.throwIfAborted();
    const countedFingerprint = await fingerprintJson(projection.native_request);
    const sourceFingerprint = await processingContextFingerprint(projection.document);
    signal?.throwIfAborted();
    let result: CanonicalProjectedRequestMeasurement | undefined;
    try {
        result = await boundedCanonicalProjectionOperation(
            () => callback(structuredClone(projection), compiler),
            signal,
            30000,
        );
    } catch (error: unknown) {
        throw markCanonicalHostCallbackFailure(error);
    }
    signal?.throwIfAborted();
    if (result === undefined) {
        if (required) throw new Error('Required processing measurement is unavailable');
        return undefined;
    }
    if (!preflightJsonInput(result, { max_bytes: 64 * 1024 }).success)
        throw new RangeError('Canonical projected measurement exceeds its bounded envelope');
    const accepted = CanonicalProjectedRequestMeasurementSchema.parse(structuredClone(result));
    if (
        accepted.counted_request_fingerprint !== countedFingerprint ||
        accepted.measurement.source_fingerprint !== sourceFingerprint ||
        accepted.measurement.adapter !== projection.target.protocol ||
        accepted.measurement.adapter_version !== projection.target.adapter_version ||
        accepted.measurement.target_model !== projection.target.model
    )
        throw new Error('Canonical projected measurement conflicts with the actual request/source/target');
    return accepted;
}

/** Abort-aware deadline for pure compiler work and the host source-ACK callback. */
export async function boundedCanonicalProjectionOperation<T>(
    operation: (signal: AbortSignal) => Promise<T>,
    signal?: AbortSignal,
    timeoutMs = 10000,
): Promise<T> {
    signal?.throwIfAborted();
    const controller = new AbortController();
    const deadline = performance.now() + timeoutMs;
    const assertDeadline = () => {
        controller.signal.throwIfAborted();
        if (performance.now() >= deadline) throw new Error('Canonical projection deadline exceeded');
    };
    const forwardAbort = () => controller.abort(signal?.reason);
    signal?.addEventListener('abort', forwardAbort, { once: true });
    const timer = setTimeout(() => controller.abort(new Error('Canonical projection deadline exceeded')), timeoutMs);
    let rejectAbort: ((cause: unknown) => void) | undefined;
    const aborted = new Promise<never>((_resolve, reject) => {
        rejectAbort = reject;
    });
    const rejectOnAbort = () => rejectAbort?.(controller.signal.reason);
    controller.signal.addEventListener('abort', rejectOnAbort, { once: true });
    try {
        assertDeadline();
        const result = await Promise.race([operation(controller.signal), aborted]);
        // A blocked event loop can resolve before the timer callback runs. Elapsed monotonic
        // time and the owned abort state remain authoritative before any transport may resume.
        assertDeadline();
        return result;
    } finally {
        clearTimeout(timer);
        signal?.removeEventListener('abort', forwardAbort);
        controller.signal.removeEventListener('abort', rejectOnAbort);
    }
}
