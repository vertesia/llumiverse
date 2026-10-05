import {
    type ConversationPreparedRequestRecord,
    fingerprintJson,
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
    if (record.indexed_source?.validator_profile !== INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE)
        return (await fingerprintJson(record.request_receipt)) === (await fingerprintJson(compiled));
    const witness = record.indexed_source.processing_input;
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
