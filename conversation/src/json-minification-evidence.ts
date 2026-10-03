import { canonicalJsonContentString } from './content-integrity.js';
import { fingerprintJson } from './identity.js';
import { JsonMinificationConfigurationSchema } from './schemas/json-minification.js';
import type { ConversationDocument, ProcessingJob, ProcessingOutputReceipt, ProcessingResolvedInput } from './types.js';

/** Validates retained evidence from the host-owned store; hashes are not caller authentication. */
export async function assertRetainedJsonMinificationOutput(
    document: ConversationDocument,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    output: Extract<ProcessingOutputReceipt, { kind: 'json_minification' }>,
): Promise<void> {
    const config = JsonMinificationConfigurationSchema.parse(job.configuration);
    const attempt = document.processing.attempts?.[job.id];
    const receipt = document.operation_receipts[`processing:output:${job.id}`];
    const { output_fingerprint, ...payload } = output;
    const { measurement, kind: _kind, ...plan } = output.proposal;
    const candidate = { ...plan, kind: 'json_minification_candidate' };
    const identity = (projection: typeof measurement.original) => {
        const { input_tokens: _tokens, source_fingerprint: _source, measured_at: _at, ...value } = projection;
        return value;
    };
    if (
        output.job_id !== job.id ||
        output.output_fingerprint !== (await fingerprintJson(payload)) ||
        receipt?.operation_kind !== 'processing' ||
        receipt.processing_operation?.phase !== 'output' ||
        receipt.processing_operation.job_id !== job.id ||
        receipt.payload_fingerprint !== (await fingerprintJson(output)) ||
        receipt.recorded_at !== output.recorded_at ||
        attempt?.attempt_token !== output.attempt_token ||
        attempt.resolved_input_fingerprint !== output.resolved_input_fingerprint ||
        output.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
        job.configuration_fingerprint !== (await fingerprintJson(job.configuration)) ||
        output.proposal.strategy.id !== job.processor_id ||
        output.proposal.strategy.version !== job.processor_version ||
        output.proposal.strategy.configuration_fingerprint !== job.configuration_fingerprint ||
        output.proposal.source_fingerprint !== resolution.source_fingerprint ||
        measurement.minimum_token_reduction !== config.minimum_token_reduction ||
        measurement.target_fingerprint !== resolution.target_fingerprint ||
        measurement.original.source_fingerprint !== resolution.context_fingerprint ||
        measurement.replacement.source_fingerprint !== measurement.prospective_input_fingerprint ||
        measurement.prospective_input_fingerprint !==
            (await fingerprintJson({
                processing_job_id: job.id,
                resolved_input_fingerprint: output.resolved_input_fingerprint,
                context_fingerprint: resolution.context_fingerprint,
                candidate,
                target_fingerprint: resolution.target_fingerprint,
            })) ||
        canonicalJsonContentString(identity(measurement.original)) !==
            canonicalJsonContentString(identity(measurement.replacement)) ||
        measurement.original.input_tokens - measurement.replacement.input_tokens < config.minimum_token_reduction
    )
        throw new Error('JSON minification output lost its exact retained source/attempt/target/count evidence');
}
