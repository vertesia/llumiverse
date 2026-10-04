import { z } from 'zod';
import { canonicalJsonContentString } from './content-integrity.js';
import { fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import {
    buildProcessingCompletionDocument,
    buildProcessingPhaseDocument,
    resolveProcessingJobInput,
} from './processing.js';
import { appendConversationRecordsWithProcessing } from './runtime.js';
import { ConversationDocumentSchema, ProcessorConfigurationSchema } from './schemas/document.js';
import { OperationReceiptSchema } from './schemas/execution.js';
import { ContentHashSchema, ConversationRefSchema, IdentifierSchema } from './schemas/primitives.js';
import {
    buildTextExternalizationProposal,
    TEXT_EXTERNALIZATION_PROCESSOR_ID,
    TEXT_EXTERNALIZATION_PROCESSOR_VERSION,
    textExternalizationAssetOperationId,
} from './text-externalization-processor.js';
import {
    buildToolResultTextExternalizationProposal,
    canonicalTextExternalizationArchiveInputs,
    isToolResultTextProcessor,
} from './tool-result-text-externalization.js';
import type { ConversationDocument, OperationReceipt, ProcessingJob, ProcessingOutputReceipt } from './types.js';
import { parseConversationDocument } from './validation.js';

export const MAX_PROCESSING_SUCCESSOR_SNAPSHOTS = 64;
export const MAX_PROCESSING_SUCCESSOR_BYTES = 32 * 1024 * 1024;
export const MAX_PROCESSING_SUCCESSOR_DEPTH = 64;

/** Ordinary immutable snapshots, loaded and authenticated by the host; not a new durable journal. */
export const ProcessingSuccessorInputSchema = z.strictObject({
    accepted_document: ConversationDocumentSchema,
    anchor: z.discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('accepted_append'), receipt: OperationReceiptSchema }),
        z.strictObject({ kind: z.literal('initialized_source'), document_fingerprint: ContentHashSchema }),
    ]),
    successors: z.array(ConversationDocumentSchema).max(MAX_PROCESSING_SUCCESSOR_SNAPSHOTS),
});
export type ProcessingSuccessorInput = z.infer<typeof ProcessingSuccessorInputSchema>;

export const ProcessingSuccessorEvidenceSchema = z.strictObject({
    source: ConversationRefSchema,
    operation_ids: z.array(IdentifierSchema).max(MAX_PROCESSING_SUCCESSOR_SNAPSHOTS),
});
export type ProcessingSuccessorEvidence = z.infer<typeof ProcessingSuccessorEvidenceSchema>;

export class ProcessingSuccessorError extends Error {
    constructor(
        readonly code: 'RESOURCE_LIMIT' | 'MISSING_EVIDENCE' | 'CONFLICT' | 'UNSUPPORTED',
        message: string,
    ) {
        super(message);
        this.name = 'ProcessingSuccessorError';
    }
}

function equal(left: unknown, right: unknown): boolean {
    return left === undefined || right === undefined
        ? left === right
        : canonicalJsonContentString(left) === canonicalJsonContentString(right);
}
function reject(message: string): never {
    throw new ProcessingSuccessorError('CONFLICT', message);
}
function builtinJob(job: ProcessingJob | undefined): ProcessingJob {
    if (!job)
        throw new ProcessingSuccessorError('MISSING_EVIDENCE', 'Processing job is not retained at input acceptance');
    if (
        (isToolResultTextProcessor(job) && job.scope !== 'on_append') ||
        (!isToolResultTextProcessor(job) &&
            (job.processor_id !== TEXT_EXTERNALIZATION_PROCESSOR_ID ||
                job.processor_version !== TEXT_EXTERNALIZATION_PROCESSOR_VERSION))
    ) {
        throw new ProcessingSuccessorError(
            'UNSUPPORTED',
            'Processing successor requires a supported deterministic processor',
        );
    }
    return job;
}

async function replayArchive(
    previous: ConversationDocument,
    next: ConversationDocument,
    receipt: OperationReceipt,
    job: ProcessingJob,
): Promise<ConversationDocument> {
    const archive = await canonicalTextExternalizationArchiveInputs(previous, job);
    if (
        receipt.id !== textExternalizationAssetOperationId(job.id) ||
        receipt.payload_fingerprint !== archive.payload_fingerprint
    ) {
        reject('Processing archive is not the exact selected original');
    }
    const ids = receipt.accepted_asset_ids ?? [];
    if (ids.length !== archive.integrities.length || new Set(ids).size !== ids.length)
        reject('Processing archive is incomplete');
    const assets = ids.map((id, index) => {
        const asset = next.assets[id];
        const integrity = archive.integrities[index];
        if (
            asset?.kind !== 'text' ||
            asset.storage.type !== 'external' ||
            asset.content_hash !== integrity.content_hash ||
            asset.byte_length !== integrity.byte_length
        ) {
            reject('Processing archive changed the exact original bytes');
        }
        return asset;
    });
    return (
        await appendConversationRecordsWithProcessing(
            previous,
            { assets },
            {
                operation_id: receipt.id,
                expected_revision: previous.revision,
                payload_fingerprint: archive.payload_fingerprint,
                recorded_at: receipt.recorded_at,
            },
        )
    ).document;
}

async function assertOutput(
    previous: ConversationDocument,
    job: ProcessingJob,
    output: ProcessingOutputReceipt,
): Promise<void> {
    const resolution = previous.processing.resolved_inputs?.[job.id];
    const attempt = previous.processing.attempts?.[job.id];
    if (
        !resolution ||
        output.job_id !== job.id ||
        output.resolved_input_fingerprint !== (await fingerprintJson(resolution))
    )
        reject('Processing output lost its exact resolved input');
    if (output.attempt_token === undefined ? attempt !== undefined : attempt?.attempt_token !== output.attempt_token)
        reject('Processing output lost its fenced attempt');
    const { output_fingerprint, ...payload } = output;
    if ((await fingerprintJson(payload)) !== output_fingerprint) reject('Processing output fingerprint changed');
    if (output.kind === 'proposal') {
        if (output.usage !== undefined) reject('Pure text archive proposal cannot invent provider usage');
        if (output.proposal.kind !== 'replace_with_compaction')
            reject('Processing output is not a text archive replacement');
        const retainedAssetIds = output.proposal.retained_asset_ids;
        const replay = isToolResultTextProcessor(job)
            ? await buildToolResultTextExternalizationProposal(
                  previous,
                  job,
                  resolution,
                  output.proposal.replacement_turns.flatMap((turn) =>
                      turn.blocks.flatMap((block) => {
                          if (block.type !== 'tool_result')
                              reject('Tool-result processing proposal changed its result block');
                          return block.content.flatMap((item) =>
                              item.type === 'external_reference' && retainedAssetIds.includes(item.asset_id)
                                  ? [item.retrieval]
                                  : [],
                          );
                      }),
                  ),
              )
            : await buildTextExternalizationProposal(
                  previous,
                  job,
                  resolution,
                  ProcessorConfigurationSchema.parse({
                      id: job.processor_id,
                      version: job.processor_version,
                      scope: job.scope,
                      config: job.configuration,
                      required: job.required,
                      failure_behavior: job.failure_behavior,
                  }),
                  output.proposal.replacement_turns.flatMap((turn) =>
                      turn.blocks.map((block) => {
                          if (block.type !== 'external_reference')
                              reject('Processing proposal contains a non-archive replacement');
                          return block.retrieval;
                      }),
                  ),
              );
        if (!equal(replay.proposal, output.proposal))
            reject('Processing proposal is not the deterministic selected archive replacement');
    } else if (output.kind === 'no_op') {
        if (output.usage !== undefined) reject('Pure empty-selection no-op cannot invent provider usage');
        if (
            resolution.entry_ids.length !== 0 ||
            output.reason !== 'no_eligible_blocks' ||
            output.attempt_token !== undefined
        )
            reject('Processing no-op is not an empty resolved selection');
    } else if (output.kind !== 'failed' && output.kind !== 'unknown_outcome') {
        throw new ProcessingSuccessorError('UNSUPPORTED', 'Processing successor output family is not supported');
    }
}

async function replayStep(
    previous: ConversationDocument,
    next: ConversationDocument,
    receipt: OperationReceipt,
    accepted: ConversationDocument,
): Promise<ConversationDocument> {
    const phase = receipt.processing_operation?.phase;
    const jobId =
        receipt.processing_operation?.job_id ??
        Object.keys(accepted.processing.jobs ?? {}).find(
            (id) => receipt.id === textExternalizationAssetOperationId(id) || receipt.id === `processing:apply:${id}`,
        );
    const job = builtinJob(jobId === undefined ? undefined : accepted.processing.jobs?.[jobId]);
    if (previous.processing.completions?.[job.id] || previous.processing.supersessions?.[job.id])
        reject('Processing successor changes an already settled job');
    if (
        (await fingerprintJson(job.selection)) !== job.selection_fingerprint ||
        (await fingerprintJson(job.configuration)) !== job.configuration_fingerprint
    )
        reject('Processing job identity is invalid');
    if (receipt.operation_kind === undefined) {
        if (
            previous.processing.resolved_inputs?.[job.id] ||
            previous.processing.attempts?.[job.id] ||
            previous.processing.outputs?.[job.id]
        )
            reject('Processing archive follows an already pinned resolution');
        return replayArchive(previous, next, receipt, job);
    }
    if (receipt.operation_kind === 'context_change' || phase === 'complete') {
        const output = previous.processing.outputs?.[job.id];
        if (!output)
            throw new ProcessingSuccessorError('MISSING_EVIDENCE', 'Processing completion has no retained output');
        if (!['proposal', 'no_op', 'failed', 'unknown_outcome'].includes(output.kind))
            throw new ProcessingSuccessorError('UNSUPPORTED', 'Retained processing output family is not supported');
        return buildProcessingCompletionDocument(previous, job, output, receipt.recorded_at);
    }
    if (receipt.operation_kind !== 'processing') reject('Processing successor contains an authoring operation');
    if (phase === 'resolve') {
        if (previous.processing.resolved_inputs?.[job.id]) reject('Processing resolution was already retained');
        const value = await resolveProcessingJobInput(previous, job, receipt.recorded_at);
        return buildProcessingPhaseDocument(previous, phase, job, value, receipt.recorded_at, {
            ...previous.processing,
            resolved_inputs: { ...previous.processing.resolved_inputs, [job.id]: value },
        });
    }
    if (phase === 'attempt') {
        const value = next.processing.attempts?.[job.id];
        const resolution = previous.processing.resolved_inputs?.[job.id];
        if (
            !value ||
            !resolution ||
            previous.processing.attempts?.[job.id] ||
            previous.processing.outputs?.[job.id] ||
            value.resolved_input_fingerprint !== (await fingerprintJson(resolution)) ||
            value.started_at !== receipt.recorded_at
        )
            reject('Processing attempt lacks its exact unresolved fence');
        return buildProcessingPhaseDocument(previous, phase, job, value, receipt.recorded_at, {
            ...previous.processing,
            attempts: { ...previous.processing.attempts, [job.id]: value },
        });
    }
    if (phase === 'output') {
        const value = next.processing.outputs?.[job.id];
        if (!value || previous.processing.outputs?.[job.id] || value.recorded_at !== receipt.recorded_at)
            reject('Processing output is missing or changed');
        await assertOutput(previous, job, value);
        return buildProcessingPhaseDocument(previous, phase, job, value, receipt.recorded_at, {
            ...previous.processing,
            outputs: { ...previous.processing.outputs, [job.id]: value },
        });
    }
    reject('Processing successor changed policy, queue, coverage, or unsupported evidence');
}

/** Validates only lineage. It never reports readiness, grants authority, resolves a registry, or invokes a plugin.
 * Preconditions: the host authenticated every complete immutable snapshot and bound the final one to its exact
 * selected head. Archive publication guards must have independently reconstructed the trusted retrieval binder
 * (including capability arguments/locator to asset binding). Retained proposal data is not retrieval authority.
 * Live membership/current-execution fences and target-bound readiness remain separate host obligations.
 */
export async function verifyProcessingSuccessor(input: unknown): Promise<ProcessingSuccessorEvidence> {
    const checked = preflightJsonInput(input, {
        max_bytes: MAX_PROCESSING_SUCCESSOR_BYTES,
        max_depth: MAX_PROCESSING_SUCCESSOR_DEPTH,
        max_nodes: 250_000,
        max_array_length: 100_000,
    });
    if (!checked.success)
        throw new ProcessingSuccessorError(
            'RESOURCE_LIMIT',
            'Processing successor input is malformed or exceeds its complete bounded working set',
        );
    // Own every snapshot before the first await; caller mutation cannot replace retained evidence.
    const owned = structuredClone(input);
    if (
        owned !== null &&
        typeof owned === 'object' &&
        'successors' in owned &&
        Array.isArray(owned.successors) &&
        owned.successors.length > MAX_PROCESSING_SUCCESSOR_SNAPSHOTS
    ) {
        throw new ProcessingSuccessorError(
            'RESOURCE_LIMIT',
            'Processing successor snapshot count exceeds its complete bound',
        );
    }
    const parsed = ProcessingSuccessorInputSchema.safeParse(owned);
    if (!parsed.success)
        throw new ProcessingSuccessorError(
            'MISSING_EVIDENCE',
            'Processing successor snapshots or exact accepted input are unavailable',
        );
    const { accepted_document, anchor, successors } = parsed.data;
    let previous = parseConversationDocument(accepted_document);
    if (anchor.kind === 'accepted_append') {
        const receipt = anchor.receipt;
        if (
            receipt.operation_kind !== undefined ||
            receipt.conversation_id !== previous.id ||
            receipt.result_revision !== previous.revision ||
            !equal(previous.operation_receipts[receipt.id], receipt)
        )
            reject('Processing successor lost the exact accepted original append');
    } else if ((await fingerprintJson(previous)) !== anchor.document_fingerprint) {
        reject('Processing successor lost the exact initialized source document');
    }
    const operation_ids: string[] = [];
    for (const snapshot of successors) {
        const next = parseConversationDocument(snapshot);
        if (next.id !== previous.id || next.revision !== previous.revision + 1)
            throw new ProcessingSuccessorError(
                'MISSING_EVIDENCE',
                'Processing successor revisions are missing or out of order',
            );
        const added = Object.values(next.operation_receipts).filter(
            (receipt) => !Object.hasOwn(previous.operation_receipts, receipt.id),
        );
        if (
            added.length !== 1 ||
            added[0].base_revision !== previous.revision ||
            added[0].result_revision !== next.revision
        )
            reject('Processing successor must retain exactly one next operation');
        const expected = await replayStep(previous, next, added[0], accepted_document);
        if (!equal(expected, next))
            reject('Processing successor changed evidence outside its exact replayed transition');
        operation_ids.push(added[0].id);
        previous = next;
    }
    return ProcessingSuccessorEvidenceSchema.parse({
        source: { conversation_id: previous.id, revision: previous.revision },
        operation_ids,
    });
}
