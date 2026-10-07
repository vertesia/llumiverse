import { z } from 'zod';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { INDEXED_EXCHANGE_PROCESSOR_ID, INDEXED_EXCHANGE_PROCESSOR_VERSION } from './indexed-exchange-constants.js';
import { preflightJsonInput } from './json-preflight.js';
import { ContextEntrySchema } from './schemas/context-foundation.js';
import { ProcessorConfigurationSchema } from './schemas/document.js';
import { ContentHashSchema, IdentifierSchema, NonnegativeSafeIntegerSchema } from './schemas/primitives.js';
import { ProcessingJobSchema } from './schemas/processing.js';
import { isToolResultTextProcessor } from './tool-result-text-externalization.js';
import { parseToolResultTextStrategy, supportsToolResultTextProcessingScope } from './tool-result-text-strategy.js';
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
    exchange_selections: z
        .array(
            z.strictObject({
                entry_ids: z.array(IdentifierSchema).length(2),
                selected_block_ids: z.record(IdentifierSchema, z.array(IdentifierSchema).length(1)),
                selected_entries: z.array(ContextEntrySchema).length(2),
            }),
        )
        .max(MAX_PROCESSING_STAGES_PER_OPERATION)
        .optional(),
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
        const toolResultStage = isToolResultTextProcessor({
            processor_id: processor.id,
            processor_version: processor.version,
        });
        if (toolResultStage)
            parseToolResultTextStrategy({
                processor_id: processor.id,
                processor_version: processor.version,
                configuration: processor.config,
            });
        if (
            toolResultStage &&
            !supportsToolResultTextProcessingScope({
                processor_id: processor.id,
                processor_version: processor.version,
                scope: processor.scope,
            })
        )
            throw new Error('Tool-result text externalization supports append jobs or explicit manual v2 selections');
        if (
            toolResultStage &&
            processor.scope === 'manual' &&
            (index !== 0 ||
                owned.processor_indices.length !== 1 ||
                owned.selected_block_ids !== undefined ||
                owned.target_fingerprint !== undefined)
        )
            throw new Error('Manual tool-result processing requires one complete result-entry selection');
        if (toolResultStage && processor.scope === 'on_append' && index !== owned.processor_indices.length - 1)
            throw new Error('Tool-result text externalization must be the final on-append stage');
        if (toolResultStage && owned.tool_result_entry_ids?.length === 0) continue;
        const exchangeStage =
            processor.id === INDEXED_EXCHANGE_PROCESSOR_ID && processor.version === INDEXED_EXCHANGE_PROCESSOR_VERSION;
        if (
            exchangeStage &&
            (processor.scope !== 'on_append' ||
                index !== 0 ||
                owned.processor_indices.length !== 1 ||
                Object.keys(processor.config).length !== 0)
        )
            throw new Error('Whole-exchange processing requires one registered on-append stage');
        if (exchangeStage && !owned.exchange_selections?.length) continue;
        const predecessor = jobs.at(-1);
        if (index > 0 && !toolResultStage && !predecessor)
            throw new Error('Processing stage has no accepted predecessor');
        const selection = exchangeStage
            ? { kind: 'entries', ...owned.exchange_selections?.[0] }
            : toolResultStage
              ? { kind: 'entries', entry_ids: [...(owned.tool_result_entry_ids ?? owned.entry_ids)] }
              : index === 0
                ? {
                      kind: 'entries',
                      entry_ids: [...owned.entry_ids],
                      ...(owned.selected_block_ids === undefined
                          ? {}
                          : { selected_block_ids: owned.selected_block_ids }),
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
        if (exchangeStage) {
            for (const [exchangeIndex, exchangeSelection] of (owned.exchange_selections ?? []).entries()) {
                if (exchangeIndex === 0) continue;
                const nextSelection = { kind: 'entries', ...exchangeSelection };
                jobs.push(
                    ProcessingJobSchema.parse({
                        ...jobs[jobs.length - 1],
                        id: await deriveConversationId(
                            'processing_job',
                            owned.conversation_id,
                            owned.source_operation_id,
                            String(index),
                            String(exchangeIndex),
                        ),
                        selection: nextSelection,
                        selection_fingerprint: await fingerprintJson(nextSelection),
                    }),
                );
            }
        }
    }
    return jobs;
}
