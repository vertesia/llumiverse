import {
    type ConversationPreparedRequestRecord,
    fingerprintJson,
    INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
    INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE,
    type IndexedConversationSelectedContext,
    indexedProcessingContextFingerprint,
    type RequestReceipt,
} from '@llumiverse/conversation';

/** The host authenticates the epoch and publishes readiness. The native adapter independently
 * checks the exact retained measurement/coverage/content binding before its durable dispatch fence.
 * Old profiles retain whole-receipt equality; this does not accept an unbound measurement upgrade.
 */
export async function indexedPreparedReceiptMatches(
    record: ConversationPreparedRequestRecord,
    selection: IndexedConversationSelectedContext,
    compiled: RequestReceipt,
): Promise<boolean> {
    if (
        record.source.conversation_id !== selection.source.conversation_id ||
        record.source.revision !== selection.source.revision
    )
        return false;
    const indexed = record.indexed_source;
    const witness =
        indexed?.validator_profile === INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE
            ? indexed.processing_input
            : indexed?.validator_profile === INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE
              ? indexed.native_measurement
              : undefined;
    if (
        indexed?.validator_profile !== INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE &&
        indexed?.validator_profile !== INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE
    )
        return (await fingerprintJson(record.request_receipt)) === (await fingerprintJson(compiled));
    if (indexed?.validator_profile === INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE) {
        const accepted = indexed.native_measurement;
        if (
            !accepted ||
            accepted.runtime_input_operation_id !== record.runtime.input_operation_id ||
            accepted.accepted_operation_id !== record.runtime.materialized_input?.operation_id ||
            accepted.accepted_source.revision !== record.runtime.materialized_input.result_revision ||
            accepted.accepted_source.conversation_id !== record.source.conversation_id
        )
            return false;
    }
    const measurement = record.request_receipt.measurement;
    if (
        !witness ||
        !measurement ||
        witness.settled_source.conversation_id !== record.source.conversation_id ||
        witness.settled_source.revision + 1 !== record.source.revision ||
        witness.coverage.expected_revision !== witness.settled_source.revision ||
        witness.coverage.recorded_at !== record.runtime.recorded_at ||
        measurement.measured_at !== record.runtime.recorded_at ||
        measurement.method !== 'estimated' ||
        !measurement.tokenizer_version ||
        measurement.adapter !== compiled.target.protocol ||
        measurement.adapter_version !== compiled.target.adapter_version ||
        measurement.target_model !== compiled.target.model ||
        measurement.input_tokens !== witness.coverage.measured_input_tokens ||
        `${measurement.tokenizer}:${measurement.tokenizer_version}` !== witness.coverage.tokenizer_id ||
        (await fingerprintJson(measurement)) !== witness.coverage.measurement_fingerprint ||
        (await fingerprintJson(compiled.target)) !== witness.coverage.target_fingerprint ||
        measurement.source_fingerprint !==
            (await indexedProcessingContextFingerprint({
                ...selection,
                completeness: 'active_processing_dependencies_verified',
            }))
    )
        return false;
    return (await fingerprintJson(record.request_receipt)) === (await fingerprintJson({ ...compiled, measurement }));
}
