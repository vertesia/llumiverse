import { z } from 'zod';
import { preflightJsonInput } from './json-preflight.js';
import { OperationReceiptSchema, ProcessingOperationSchema } from './schemas/execution.js';
import { ContentHashSchema, ConversationRefSchema, IdentifierSchema, TimestampSchema } from './schemas/primitives.js';

const inputSchema = z.strictObject({
    source: ConversationRefSchema,
    operation_id: IdentifierSchema,
    payload_fingerprint: ContentHashSchema,
    recorded_at: TimestampSchema,
    processing_operation: ProcessingOperationSchema,
});
export type ProcessingTransitionReceiptInput = z.infer<typeof inputSchema>;

/** Shared receipt construction only. The materialized/indexed adapters still verify source,
 * mutation evidence and retry before publishing with their existing guarded head CAS.
 */
export function createProcessingTransitionReceipt(input: ProcessingTransitionReceiptInput) {
    if (!preflightJsonInput(input).success) throw new TypeError('Processing receipt input is not bounded JSON');
    const owned = inputSchema.parse(structuredClone(input));
    const revision = owned.source.revision + 1;
    if (!Number.isSafeInteger(revision)) throw new RangeError('Conversation revision exceeds the safe integer range');
    return OperationReceiptSchema.parse({
        id: owned.operation_id,
        conversation_id: owned.source.conversation_id,
        payload_fingerprint: owned.payload_fingerprint,
        base_revision: owned.source.revision,
        result_revision: revision,
        recorded_at: owned.recorded_at,
        accepted_turn_ids: [],
        accepted_generation_ids: [],
        accepted_asset_ids: [],
        accepted_tool_definition_ids: [],
        accepted_execution_receipt_ids: [],
        accepted_context_entry_ids: [],
        operation_kind: 'processing',
        processing_operation: owned.processing_operation,
    });
}
