import { z } from 'zod';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { ContextEntrySchema } from './schemas/context-foundation.js';
import { ProcessorConfigurationSchema } from './schemas/document.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema } from './schemas/primitives.js';
import { ProcessingJobSchema } from './schemas/processing.js';
import {
    TOOL_RESULT_TEXT_PROCESSOR_ID,
    TOOL_RESULT_TEXT_PROCESSOR_VERSION,
} from './tool-result-text-externalization.js';
import type { ProcessingJob } from './types.js';

export const MAX_PROCESSING_STAGES_PER_OPERATION = 16;

const JobConstructionSchema = z.strictObject({
    conversation_id: IdentifierSchema,
    revision: NonnegativeSafeIntegerSchema,
    source_operation_id: IdentifierSchema,
    policy_revision: NonnegativeSafeIntegerSchema,
    processors: z.array(ProcessorConfigurationSchema),
    processor_indices: z.array(NonnegativeSafeIntegerSchema).max(MAX_PROCESSING_STAGES_PER_OPERATION),
    entry_ids: z.array(IdentifierSchema),
    selected_block_ids: z.record(IdentifierSchema, z.array(IdentifierSchema).min(1)).optional(),
    selected_entries: z.array(ContextEntrySchema).optional(),
    target_fingerprint: ContentHashSchema.optional(),
    tool_result_entry_ids: z.array(IdentifierSchema).optional(),
});
export type ProcessingJobConstruction = z.infer<typeof JobConstructionSchema>;

/** Internal deterministic builder shared by materialized and indexed append/queue adapters.
 * Receipts and authenticated policy/source publication remain the caller's responsibility.
 */
export async function constructProcessingJobs(input: ProcessingJobConstruction): Promise<ProcessingJob[]> {
    if (!preflightJsonInput(input).success) throw new TypeError('Processing job construction is not bounded JSON');
    const owned = JobConstructionSchema.parse(structuredClone(input));
    if (new Set(owned.processor_indices).size !== owned.processor_indices.length)
        throw new Error('Processing stages repeat an accepted policy index');
    const jobs: ProcessingJob[] = [];
    for (const [index, policyIndex] of owned.processor_indices.entries()) {
        const processor = owned.processors[policyIndex];
        if (!processor) throw new Error('Processing configuration is not in its accepted policy');
        const id = await deriveConversationId(
            'processing_job',
            owned.conversation_id,
            owned.source_operation_id,
            String(index),
        );
        const toolResultStage =
            processor.id === TOOL_RESULT_TEXT_PROCESSOR_ID && processor.version === TOOL_RESULT_TEXT_PROCESSOR_VERSION;
        if (toolResultStage && processor.scope !== 'on_append')
            throw new Error('Tool-result text externalization supports only accepted on-append results');
        if (toolResultStage && index !== owned.processor_indices.length - 1)
            throw new Error('Tool-result text externalization must be the final on-append stage');
        if (toolResultStage && owned.tool_result_entry_ids?.length === 0) continue;
        const predecessor = jobs.at(-1);
        if (index > 0 && !toolResultStage && !predecessor)
            throw new Error('Processing stage has no accepted predecessor');
        const selection = toolResultStage
            ? { kind: 'entries', entry_ids: [...(owned.tool_result_entry_ids ?? owned.entry_ids)] }
            : index === 0
              ? {
                    kind: 'entries',
                    entry_ids: [...owned.entry_ids],
                    ...(owned.selected_block_ids === undefined ? {} : { selected_block_ids: owned.selected_block_ids }),
                    ...(owned.selected_entries === undefined ? {} : { selected_entries: owned.selected_entries }),
                }
              : { kind: 'predecessor_output', job_id: predecessor?.id };
        jobs.push(
            ProcessingJobSchema.parse({
                id,
                source_operation_id: owned.source_operation_id,
                enqueue_revision: owned.revision,
                policy_revision: owned.policy_revision,
                stage_index: index,
                processor_index: policyIndex,
                processor_id: processor.id,
                processor_version: processor.version,
                configuration_fingerprint: await fingerprintJson(processor.config),
                configuration: processor.config,
                scope: processor.scope,
                required: processor.required,
                failure_behavior: processor.failure_behavior,
                selection,
                selection_fingerprint: await fingerprintJson(selection),
                ...(owned.target_fingerprint === undefined ? {} : { target_fingerprint: owned.target_fingerprint }),
            }),
        );
    }
    return jobs;
}
