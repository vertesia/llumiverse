import { fingerprintJson } from './identity.js';
import type { ProcessingPolicyCommand } from './processing.js';
import { ProcessingPolicyCommandSchema } from './schemas/processing-policy.js';
import type { OperationReceipt, ProcessingBudget, ProcessorConfiguration } from './types.js';

/** Cold migration/explicit upgrade only. The caller supplies an authenticated actual policy header
 * at this receipt's accepted epoch; a job configuration is never an original policy source.
 * Omitted vs explicit empty supersession arrays have distinct hashes and are checked separately. */
export async function recoverAcceptedProcessingPolicyCommand(
    receipt: OperationReceipt,
    policy: {
        enabled: boolean;
        policy_revision: number;
        processors: ProcessorConfiguration[];
        budget?: ProcessingBudget;
    },
    supersessionReason?: string,
): Promise<ProcessingPolicyCommand | undefined> {
    if (
        receipt.operation_kind !== 'processing' ||
        receipt.processing_operation?.phase !== 'policy' ||
        receipt.processing_operation.policy_revision + 1 !== policy.policy_revision
    )
        return undefined;
    const ids = receipt.processing_operation.superseded_job_ids ?? [];
    const base = {
        operation_id: receipt.id,
        expected_revision: receipt.base_revision,
        recorded_at: receipt.recorded_at,
        enabled: policy.enabled,
        processors: policy.processors,
        ...(policy.budget === undefined ? {} : { budget: policy.budget }),
        ...(supersessionReason === undefined ? {} : { supersession_reason: supersessionReason }),
    };
    const candidates =
        ids.length === 0 ? [base, { ...base, supersede_job_ids: ids }] : [{ ...base, supersede_job_ids: ids }];
    for (const candidate of candidates) {
        const parsed = ProcessingPolicyCommandSchema.safeParse(candidate);
        if (parsed.success && (await fingerprintJson(parsed.data)) === receipt.payload_fingerprint) return parsed.data;
    }
    return undefined;
}
