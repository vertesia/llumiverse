import { fingerprintJson } from './identity.js';
import type { ExecutionReceipt, ToolResultBlock } from './types.js';

/** Both materialized and indexed acceptance attest to the same exact result block. */
export async function assertToolResultReceiptFingerprint(
    resultBlock: ToolResultBlock,
    receipt: Pick<ExecutionReceipt, 'id' | 'result_fingerprint'>,
): Promise<void> {
    if (receipt.result_fingerprint !== (await fingerprintJson(resultBlock))) {
        throw new Error(`Tool execution receipt ${receipt.id} result fingerprint does not match its result block`);
    }
}
