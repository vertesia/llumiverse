import { z } from 'zod';
import { canonicalJsonContentString } from './content-integrity.js';
import { applyContextChange, contextChangeSelectedRanges, planContextChange } from './context-change.js';
import { createContextTurnIndex } from './context-entry-resolution.js';
import { resolveContextSelection } from './context-selection.js';
import { fingerprintJson } from './identity.js';
import { applyJsonMinificationOutput } from './json-minification-application.js';
import {
    captureJsonMinificationHostCapability,
    type JsonMinificationHostCapability,
    jsonMinificationProcessor,
    validateJsonMinificationCandidate,
} from './json-minification-processor.js';
import { preflightJsonInput } from './json-preflight.js';
import { eligibleProcessingAppendRecords } from './processing-append-selection.js';
import { constructProcessingJobs, MAX_PROCESSING_STAGES_PER_OPERATION } from './processing-job-construction.js';
import { countUnresolvedProcessingJobs } from './processing-job-status.js';
import { createProcessingTransitionReceipt } from './processing-transition-receipt.js';
import {
    ContextChangePlanInputSchema,
    ContextChangeProposalSchema,
    ContextChangeRequestSchema,
} from './schemas/context-change.js';
import { ContextSelectionRequestSchema } from './schemas/context-selection.js';
import { ProcessingBudgetSchema, ProcessingRunResultSchema, ProcessorConfigurationSchema } from './schemas/document.js';
import { GenerationUsageSchema } from './schemas/execution.js';
import type { JsonMinificationCandidateSchema, JsonMinificationNoOpReasonSchema } from './schemas/json-minification.js';
import {
    ContentHashSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
    TimestampSchema,
} from './schemas/primitives.js';
import {
    MAX_PROCESSING_OUTPUT_BYTES,
    MAX_PROCESSOR_CONFIGURATION_BYTES,
    ProcessingAppendAcceptanceSchema,
    ProcessingOutputReceiptSchema,
    ProcessingReadinessCoverageSchema,
    ProcessingResolvedInputSchema,
} from './schemas/processing.js';
import { ProcessingChangeSchema } from './schemas/processing-change.js';
import { verifyDerivedBlockLineage } from './source-slice-lineage.js';
import {
    createTextExternalizationProcessor,
    TEXT_EXTERNALIZATION_PROCESSOR_ID,
    TEXT_EXTERNALIZATION_PROCESSOR_VERSION,
    type TextExternalizationRetrievalBinder,
} from './text-externalization-processor.js';
import {
    applyToolResultTextExternalizationOutput,
    createToolResultTextExternalizationProcessor,
    eligibleToolResultTextEntries,
    isToolResultTextProcessor,
    TOOL_RESULT_TEXT_PROCESSOR_ID,
    TOOL_RESULT_TEXT_PROCESSOR_VERSION,
    toolResultTextSelection,
} from './tool-result-text-externalization.js';
import type {
    ContextChangeProposal,
    ContextEntry,
    ContextSelectionRequest,
    ConversationDocument,
    GenerationUsage,
    OperationReceipt,
    ProcessingAppendAcceptance,
    ProcessingChange,
    ProcessingJob,
    ProcessingOutputReceipt,
    ProcessingReadinessCoverage,
    ProcessingResolvedInput,
    ProcessingRunResult,
    ProcessorConfiguration,
} from './types.js';
import { parseConversationDocument } from './validation.js';

export { MAX_PROCESSING_OUTPUT_BYTES, MAX_PROCESSOR_CONFIGURATION_BYTES } from './schemas/processing.js';

const MAX_PROCESSING_JOBS = 256;

const ProcessingPolicyCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
    enabled: z.boolean(),
    processors: z.array(ProcessorConfigurationSchema).max(MAX_PROCESSING_STAGES_PER_OPERATION),
    budget: ProcessingBudgetSchema.optional(),
    supersede_job_ids: z.array(IdentifierSchema).optional(),
    supersession_reason: IdentifierSchema.optional(),
});

export type ProcessingPolicyCommand = z.infer<typeof ProcessingPolicyCommandSchema>;

const ProcessingQueueCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
    processor_id: IdentifierSchema,
    scope: z.enum(['manual', 'on_budget']),
    target_fingerprint: ContentHashSchema.optional(),
});

export type ProcessingQueueCommand = z.infer<typeof ProcessingQueueCommandSchema>;

export type ProcessingReadiness =
    | { status: 'ready'; coverage?: ProcessingReadinessCoverage }
    | { status: 'pending'; code: 'PROCESSING_PENDING'; reason: string }
    | { status: 'blocked'; code: 'PROCESSING_BLOCKED'; reason: string };

export class ProcessingReadinessError extends Error {
    constructor(
        readonly code: 'PROCESSING_PENDING' | 'PROCESSING_BLOCKED',
        message: string,
    ) {
        super(message);
        this.name = 'ProcessingReadinessError';
    }
}

export interface ProcessingStore {
    load(): Promise<ConversationDocument>;
    commit(expectedRevision: number, document: ConversationDocument): Promise<boolean>;
}

export type ProcessorResult =
    | z.infer<typeof JsonMinificationCandidateSchema>
    | { kind: 'json_minification_no_op'; reason: z.infer<typeof JsonMinificationNoOpReasonSchema> }
    | { kind: 'proposal'; proposal: ContextChangeProposal; usage?: GenerationUsage }
    | { kind: 'no_op'; reason: string; usage?: GenerationUsage };

export interface ConversationProcessor {
    run(input: {
        document: ConversationDocument;
        job: ProcessingJob;
        resolved_input: ProcessingResolvedInput;
        configuration: ProcessorConfiguration;
        signal?: AbortSignal;
    }): Promise<ProcessorResult>;
}

export interface ProcessorRegistry {
    resolve(id: string, version: string): ConversationProcessor | undefined;
}

/** A trusted processor may identify a known failure before any uncertain external effect. */
export class ProcessingKnownFailure extends Error {
    readonly usage?: GenerationUsage;
    constructor(message: string, options?: ErrorOptions & { usage?: GenerationUsage }) {
        super(message, options);
        this.name = 'ProcessingKnownFailure';
        this.usage = options?.usage;
    }
}

/** A processor can report measured usage even when the external outcome is unknown. */
export class ProcessingUnknownFailure extends Error {
    readonly usage?: GenerationUsage;
    constructor(message: string, options?: ErrorOptions & { usage?: GenerationUsage }) {
        super(message, options);
        this.name = 'ProcessingUnknownFailure';
        this.usage = options?.usage;
    }
}

function nextRevision(revision: number): number {
    if (!Number.isSafeInteger(revision + 1))
        throw new RangeError('Conversation revision exceeds the safe integer range');
    return revision + 1;
}

function ownJson<T>(input: T): T {
    const preflight = preflightJsonInput(input);
    if (!preflight.success) throw new TypeError('Processing input is not valid bounded JSON');
    return structuredClone(input);
}

function assertExpected(document: ConversationDocument, expected: number): void {
    if (document.revision !== expected) {
        throw new Error(
            `Conversation processing revision conflict: expected ${expected}, received ${document.revision}`,
        );
    }
}

function processingReceipt(
    document: ConversationDocument,
    id: string,
    payloadFingerprint: string,
    recordedAt: string,
    phase: NonNullable<OperationReceipt['processing_operation']>['phase'],
    jobId?: string,
    supersededJobIds?: readonly string[],
): OperationReceipt {
    return createProcessingTransitionReceipt({
        source: { conversation_id: document.id, revision: document.revision },
        operation_id: id,
        payload_fingerprint: payloadFingerprint,
        recorded_at: recordedAt,
        processing_operation: {
            phase,
            policy_revision: document.processing.policy_revision,
            ...(jobId === undefined ? {} : { job_id: jobId }),
            ...(supersededJobIds === undefined ? {} : { superseded_job_ids: [...supersededJobIds] }),
        },
    });
}

/** Reconstruct the exact atomic change from an accepted processing receipt, including on retry. */
function processingChangeFromReceipt(receipt: OperationReceipt): ProcessingChange {
    if (receipt.operation_kind !== 'processing' || !receipt.processing_operation)
        throw new Error('Accepted receipt is not a processing mutation');
    return ProcessingChangeSchema.parse({
        operation_id: receipt.id,
        conversation_id: receipt.conversation_id,
        base_revision: receipt.base_revision,
        result_revision: receipt.result_revision,
        operations: [{ kind: 'processing', detail: receipt.processing_operation }],
        diagnostics: [],
    });
}

function advanceProcessing(
    document: ConversationDocument,
    receipt: OperationReceipt,
    processing: ConversationDocument['processing'],
): ConversationDocument {
    if (Object.hasOwn(document.operation_receipts, receipt.id)) {
        throw new Error(`Processing operation ${receipt.id} was already recorded`);
    }
    return parseConversationDocument({
        ...document,
        revision: receipt.result_revision,
        updated_at: receipt.recorded_at,
        operation_receipts: { ...document.operation_receipts, [receipt.id]: receipt },
        processing,
    });
}

function exactRetry(
    document: ConversationDocument,
    operationId: string,
    fingerprint: string,
    phase: NonNullable<OperationReceipt['processing_operation']>['phase'],
    expectedRevision: number,
    recordedAt: string,
): boolean {
    const prior = document.operation_receipts[operationId];
    if (!prior) return false;
    if (
        prior.operation_kind !== 'processing' ||
        prior.processing_operation?.phase !== phase ||
        prior.payload_fingerprint !== fingerprint ||
        prior.base_revision !== expectedRevision ||
        prior.result_revision !== expectedRevision + 1 ||
        prior.recorded_at !== recordedAt ||
        (prior.accepted_turn_ids?.length ?? 0) !== 0 ||
        (prior.accepted_generation_ids?.length ?? 0) !== 0 ||
        (prior.accepted_asset_ids?.length ?? 0) !== 0 ||
        (prior.accepted_tool_definition_ids?.length ?? 0) !== 0 ||
        (prior.accepted_execution_receipt_ids?.length ?? 0) !== 0 ||
        (prior.accepted_context_entry_ids?.length ?? 0) !== 0
    ) {
        throw new Error(`Processing operation ${operationId} conflicts with its accepted input`);
    }
    return true;
}

export async function setProcessingPolicy(
    sourceInput: ConversationDocument,
    commandInput: ProcessingPolicyCommand,
): Promise<{ document: ConversationDocument; change: ProcessingChange; applied: boolean }> {
    const source = parseConversationDocument(sourceInput);
    const command = ProcessingPolicyCommandSchema.parse(ownJson(commandInput));
    for (const processor of command.processors) {
        if (
            new TextEncoder().encode(canonicalJsonContentString(processor.config)).byteLength >
            MAX_PROCESSOR_CONFIGURATION_BYTES
        )
            throw new RangeError(`Processor ${processor.id} configuration exceeds durable bound`);
    }
    const fingerprint = await fingerprintJson(command);
    if (
        exactRetry(source, command.operation_id, fingerprint, 'policy', command.expected_revision, command.recorded_at)
    ) {
        if (
            canonicalJsonContentString(
                source.operation_receipts[command.operation_id].processing_operation?.superseded_job_ids ?? [],
            ) !== canonicalJsonContentString(command.supersede_job_ids ?? [])
        )
            throw new Error('Processing policy retry has conflicting supersession details');
        return {
            document: source,
            change: processingChangeFromReceipt(source.operation_receipts[command.operation_id]),
            applied: false,
        };
    }
    assertExpected(source, command.expected_revision);
    const supersede = new Set(command.supersede_job_ids ?? []);
    if (supersede.size !== (command.supersede_job_ids?.length ?? 0))
        throw new Error('Processing supersession lists duplicate jobs');
    if (supersede.size && command.supersession_reason === undefined)
        throw new Error('Processing supersession requires a reason');
    for (const jobId of supersede) {
        const completion = source.processing.completions?.[jobId];
        if (
            !source.processing.jobs?.[jobId] ||
            source.processing.supersessions?.[jobId] ||
            (completion && completion.status !== 'blocked')
        )
            throw new Error(`Processing job ${jobId} cannot be superseded`);
        if (source.processing.attempts?.[jobId] && !source.processing.outputs?.[jobId])
            throw new Error(`Processing job ${jobId} has an unresolved external attempt`);
    }
    if (!command.enabled) {
        const outstanding = Object.values(source.processing.jobs ?? {}).filter(
            (job) =>
                (!source.processing.completions?.[job.id] ||
                    source.processing.completions[job.id].status === 'blocked') &&
                !source.processing.supersessions?.[job.id] &&
                !supersede.has(job.id),
        );
        if (outstanding.length) throw new Error('Disabling processing requires explicit pending-job supersession');
    }
    const policyRevision = nextRevision(source.processing.policy_revision);
    const supersessionReason = command.supersession_reason;
    const receipt = processingReceipt(
        source,
        command.operation_id,
        fingerprint,
        command.recorded_at,
        'policy',
        undefined,
        [...supersede],
    );
    const { budget: _budget, coverage: _coverage, ...processing } = source.processing;
    const document = advanceProcessing(source, receipt, {
        ...processing,
        enabled: command.enabled,
        policy_revision: policyRevision,
        processors: structuredClone(command.processors),
        ...(command.budget === undefined ? {} : { budget: structuredClone(command.budget) }),
        supersessions: {
            ...source.processing.supersessions,
            ...(supersessionReason === undefined
                ? {}
                : Object.fromEntries(
                      [...supersede].map((jobId) => [
                          jobId,
                          {
                              job_id: jobId,
                              policy_operation_id: command.operation_id,
                              reason: supersessionReason,
                              recorded_at: command.recorded_at,
                          },
                      ]),
                  )),
        },
    });
    return { document, change: processingChangeFromReceipt(receipt), applied: true };
}

async function createJobs(
    document: ConversationDocument,
    sourceOperationId: string,
    entryIds: string[],
    selectedBlockIds: Record<string, string[]> | undefined,
    processors: readonly ProcessorConfiguration[],
    targetFingerprint?: string,
    selectedEntries?: readonly ContextEntry[],
    toolResultEntryIds?: readonly string[],
): Promise<ProcessingJob[]> {
    if (
        (document.processing.jobs ? Object.keys(document.processing.jobs).length : 0) + processors.length >
        MAX_PROCESSING_JOBS
    )
        throw new RangeError('Materialized processing job limit exceeded');
    if (processors.length > MAX_PROCESSING_STAGES_PER_OPERATION)
        throw new RangeError('Processing stage limit exceeded');
    return constructProcessingJobs({
        conversation_id: document.id,
        revision: document.revision,
        source_operation_id: sourceOperationId,
        policy_revision: document.processing.policy_revision,
        processors: document.processing.processors,
        processor_indices: processors.map((processor) => {
            const policyIndex = document.processing.processors.findIndex(
                (candidate) =>
                    candidate.id === processor.id &&
                    candidate.version === processor.version &&
                    candidate.scope === processor.scope,
            );
            if (policyIndex < 0) throw new Error('Processing configuration is not in its accepted policy');
            return policyIndex;
        }),
        entry_ids: entryIds,
        ...(selectedBlockIds === undefined ? {} : { selected_block_ids: selectedBlockIds }),
        ...(selectedEntries === undefined ? {} : { selected_entries: [...selectedEntries] }),
        ...(targetFingerprint === undefined ? {} : { target_fingerprint: targetFingerprint }),
        ...(toolResultEntryIds === undefined ? {} : { tool_result_entry_ids: [...toolResultEntryIds] }),
    });
}

function eligibleAppendSelection(document: ConversationDocument, acceptedEntryIds: readonly string[]) {
    const processor = document.processing.processors.find((processor) => processor.scope === 'on_append');
    return eligibleProcessingAppendRecords(
        document.context.entries,
        createContextTurnIndex(document),
        acceptedEntryIds,
        document.context.protected_entry_ids,
        processor?.id === TEXT_EXTERNALIZATION_PROCESSOR_ID &&
            processor.version === TEXT_EXTERNALIZATION_PROCESSOR_VERSION,
    );
}

/** Stage configured on_append jobs in the same in-memory snapshot as the accepted append. */
export async function stageProcessingAppend(
    acceptedInput: ConversationDocument,
    sourceOperationId: string,
    acceptedEntryIds: readonly string[],
): Promise<ConversationDocument> {
    const accepted = parseConversationDocument(acceptedInput);
    if (!accepted.processing.enabled) return accepted;
    const processors = accepted.processing.processors.filter((processor) => processor.scope === 'on_append');
    const selection = eligibleAppendSelection(accepted, acceptedEntryIds);
    const jobs = await createJobs(
        accepted,
        sourceOperationId,
        selection.entryIds,
        selection.selectedBlockIds,
        processors,
        undefined,
        selection.selectedEntries,
        processors.some(
            (processor) =>
                processor.id === TOOL_RESULT_TEXT_PROCESSOR_ID &&
                processor.version === TOOL_RESULT_TEXT_PROCESSOR_VERSION,
        )
            ? await eligibleToolResultTextEntries(accepted, acceptedEntryIds)
            : undefined,
    );
    const existing = accepted.processing.jobs ?? {};
    for (const job of jobs) if (Object.hasOwn(existing, job.id)) throw new Error(`Processing job ${job.id} collides`);
    const { coverage: _coverage, ...processing } = accepted.processing;
    return parseConversationDocument({
        ...accepted,
        processing: {
            ...processing,
            jobs: { ...existing, ...Object.fromEntries(jobs.map((job) => [job.id, job])) },
        },
    });
}

/** Build a bounded acknowledgement from the accepted append receipt; never execute a plugin here. */
export function processingAppendAcceptance(
    documentInput: ConversationDocument,
    operationId: string,
): ProcessingAppendAcceptance {
    const document = parseConversationDocument(documentInput);
    const receipt = document.operation_receipts[operationId];
    if (!receipt || receipt.operation_kind !== undefined)
        throw new Error(`Conversation operation ${operationId} is not an accepted append`);
    const jobs = Object.values(document.processing.jobs ?? {}).filter((job) => job.source_operation_id === operationId);
    if (jobs.length > MAX_PROCESSING_STAGES_PER_OPERATION)
        throw new RangeError('Accepted append has more processing jobs than the stage bound');
    const blocked = Object.values(document.processing.jobs ?? {}).some(
        (job) =>
            job.required &&
            !document.processing.supersessions?.[job.id] &&
            document.processing.completions?.[job.id]?.status === 'blocked',
    );
    return ProcessingAppendAcceptanceSchema.parse({
        operation_id: receipt.id,
        conversation_id: receipt.conversation_id,
        base_revision: receipt.base_revision,
        accepted_revision: receipt.result_revision,
        change_reference: { operation_id: receipt.id, result_revision: receipt.result_revision },
        processing: {
            status: document.processing.enabled ? (blocked ? 'blocked' : 'pending') : 'ready',
            job_ids: jobs.map((job) => job.id),
        },
    });
}

export async function queueProcessingForExisting(
    sourceInput: ConversationDocument,
    selectionInput: ContextSelectionRequest,
    commandInput: ProcessingQueueCommand,
): Promise<{ document: ConversationDocument; change: ProcessingChange; applied: boolean; job_id: string }> {
    const source = parseConversationDocument(sourceInput);
    const command = ProcessingQueueCommandSchema.parse(ownJson(commandInput));
    if (command.scope === 'on_budget' && command.target_fingerprint === undefined)
        throw new Error('On-budget processing requires an exact target fingerprint');
    const selectionRequest = ContextSelectionRequestSchema.parse(ownJson(selectionInput));
    const fingerprint = await fingerprintJson({ command, selection: selectionRequest });
    const priorJob = Object.values(source.processing.jobs ?? {}).find(
        (job) => job.source_operation_id === command.operation_id,
    );
    if (
        exactRetry(source, command.operation_id, fingerprint, 'queue', command.expected_revision, command.recorded_at)
    ) {
        if (!priorJob) throw new Error('Accepted processing queue operation has no retained job');
        return {
            document: source,
            change: processingChangeFromReceipt(source.operation_receipts[command.operation_id]),
            applied: false,
            job_id: priorJob.id,
        };
    }
    assertExpected(source, command.expected_revision);
    const selection = await resolveContextSelection(source, selectionRequest);
    if (selection.kind === 'rejected') throw new Error('Processing selection was rejected');
    if (!source.processing.enabled) throw new Error('Processing policy is disabled');
    const processor = source.processing.processors.find(
        (item) => item.id === command.processor_id && item.scope === command.scope,
    );
    if (!processor) throw new Error('Configured processor is unavailable for this scope');
    const receipt = processingReceipt(source, command.operation_id, fingerprint, command.recorded_at, 'queue');
    const staged = { ...source, revision: receipt.result_revision };
    const jobs = await createJobs(
        staged,
        command.operation_id,
        selection.kind === 'selected' ? selection.plan.entry_ids : [],
        selection.kind === 'selected' ? selection.plan.selected_block_ids : undefined,
        [processor],
        command.target_fingerprint,
        selection.kind === 'selected' ? selection.plan.selected_entries : undefined,
    );
    const job = jobs[0];
    if (Object.hasOwn(source.processing.jobs ?? {}, job.id)) throw new Error(`Processing job ${job.id} collides`);
    const { coverage: _coverage, ...processing } = source.processing;
    const document = advanceProcessing(source, receipt, {
        ...processing,
        jobs: { ...source.processing.jobs, [job.id]: job },
    });
    return { document, change: processingChangeFromReceipt(receipt), applied: true, job_id: job.id };
}

export async function processingContextFingerprint(documentInput: ConversationDocument): Promise<string> {
    const document = parseConversationDocument(documentInput);
    const retainedTurns = new Map(document.turns.map((turn) => [turn.id, turn]));
    const entries = document.context.entries.map((entry) => ({
        entry,
        turn:
            entry.type === 'source_turn'
                ? retainedTurns.get(entry.turn_id)
                : document.compactions[entry.compaction_id]?.replacement_turns.find(
                      (turn) => turn.id === entry.turn_id,
                  ),
    }));
    return fingerprintJson({ context: document.context, entries });
}

export async function assessProcessingReadiness(
    documentInput: ConversationDocument,
    targetFingerprint: string,
    measurementFingerprint: string,
): Promise<ProcessingReadiness> {
    const document = parseConversationDocument(documentInput);
    const outstanding = countUnresolvedProcessingJobs(document.processing) > 0;
    if (!document.processing.enabled)
        return outstanding
            ? { status: 'pending', code: 'PROCESSING_PENDING', reason: 'Accepted processing jobs remain outstanding' }
            : { status: 'ready' };
    const contextFingerprint = await processingContextFingerprint(document);
    const matches = (candidate: ProcessingReadinessCoverage) =>
        candidate.context_fingerprint === contextFingerprint &&
        candidate.policy_revision === document.processing.policy_revision &&
        candidate.target_fingerprint === targetFingerprint &&
        candidate.measurement.fingerprint === measurementFingerprint;
    let coverage = document.processing.coverage;
    if (!coverage || !matches(coverage)) {
        // Each virtual child has an independently identified measurement. Publishing another child's
        // coverage cannot erase this child's exact retained evaluation for the unchanged context/policy.
        // Public coverage operation IDs are caller-selected; do not impose a private host naming rule.
        // The newest matching complete retained evaluation wins, then the unchanged job gates below apply.
        let retained: { id: string; coverage: ProcessingReadinessCoverage } | undefined;
        const receipts = document.processing.coverage_receipts ?? {};
        for (const id in receipts) {
            if (!Object.hasOwn(receipts, id)) continue;
            const candidate = receipts[id];
            if (
                matches(candidate) &&
                (!retained ||
                    candidate.evaluated_at_revision > retained.coverage.evaluated_at_revision ||
                    (candidate.evaluated_at_revision === retained.coverage.evaluated_at_revision && id > retained.id))
            )
                retained = { id, coverage: candidate };
        }
        const receipt = retained && document.operation_receipts[retained.id];
        if (
            retained &&
            receipt?.operation_kind === 'processing' &&
            receipt.processing_operation?.phase === 'coverage' &&
            receipt.result_revision === retained.coverage.evaluated_at_revision &&
            receipt.result_revision <= document.revision &&
            receipt.processing_operation.result_fingerprint === (await fingerprintJson(retained.coverage))
        )
            coverage = retained.coverage;
        else coverage = undefined;
    }
    if (!coverage || !matches(coverage)) {
        return {
            status: 'pending',
            code: 'PROCESSING_PENDING',
            reason: 'Current context, policy and target need processing evaluation',
        };
    }
    const requiredNow = Object.values(document.processing.jobs ?? {})
        .filter(
            (job) =>
                job.required &&
                !document.processing.supersessions?.[job.id] &&
                (job.target_fingerprint === undefined || job.target_fingerprint === targetFingerprint),
        )
        .map((job) => job.id);
    if (canonicalJsonContentString(requiredNow) !== canonicalJsonContentString(coverage.required_job_ids))
        return { status: 'pending', code: 'PROCESSING_PENDING', reason: 'Processing job coverage has changed' };
    if (coverage.status === 'blocked')
        return {
            status: 'blocked',
            code: 'PROCESSING_BLOCKED',
            reason: 'Required processing or budget evaluation is blocked',
        };
    if (coverage.status === 'pending')
        return { status: 'pending', code: 'PROCESSING_PENDING', reason: 'Required processing is pending' };
    for (const id of coverage.required_job_ids) {
        const completion = document.processing.completions?.[id];
        if (!completion)
            return { status: 'pending', code: 'PROCESSING_PENDING', reason: `Processing job ${id} is pending` };
        if (completion.status === 'blocked')
            return { status: 'blocked', code: 'PROCESSING_BLOCKED', reason: `Processing job ${id} is blocked` };
    }
    return { status: 'ready', coverage };
}

/** Host preflight for a fresh request; callers still run the ordinary budget, media and replay validators. */
export async function assertProcessingReady(
    document: ConversationDocument,
    targetFingerprint: string,
    measurementFingerprint: string,
): Promise<ProcessingReadinessCoverage | undefined> {
    const readiness = await assessProcessingReadiness(document, targetFingerprint, measurementFingerprint);
    if (readiness.status !== 'ready') throw new ProcessingReadinessError(readiness.code, readiness.reason);
    return readiness.coverage;
}

const ProcessingCoverageCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    target_fingerprint: ContentHashSchema,
    measured_input_tokens: NonnegativeSafeIntegerSchema,
    tokenizer_id: IdentifierSchema,
    measurement_fingerprint: ContentHashSchema,
    recorded_at: TimestampSchema,
});

export async function recordProcessingCoverage(
    sourceInput: ConversationDocument,
    input: z.infer<typeof ProcessingCoverageCommandSchema>,
): Promise<{
    document: ConversationDocument;
    change: ProcessingChange;
    applied: boolean;
    coverage: ProcessingReadinessCoverage;
}> {
    const source = parseConversationDocument(sourceInput);
    const request = ProcessingCoverageCommandSchema.parse(ownJson(input));
    if (!source.processing.enabled) throw new Error('Processing policy is disabled');
    const fingerprint = await fingerprintJson(request);
    if (
        exactRetry(
            source,
            request.operation_id,
            fingerprint,
            'coverage',
            request.expected_revision,
            request.recorded_at,
        )
    ) {
        const retained = source.processing.coverage_receipts?.[request.operation_id];
        const prior = source.operation_receipts[request.operation_id];
        if (
            !retained ||
            retained.target_fingerprint !== request.target_fingerprint ||
            retained.measurement.input_tokens !== request.measured_input_tokens ||
            retained.measurement.tokenizer_id !== request.tokenizer_id ||
            retained.measurement.fingerprint !== request.measurement_fingerprint ||
            prior.processing_operation?.result_fingerprint !== (await fingerprintJson(retained))
        )
            throw new Error('Processing coverage retry cannot resolve its accepted target');
        return {
            document: source,
            change: processingChangeFromReceipt(source.operation_receipts[request.operation_id]),
            applied: false,
            coverage: retained,
        };
    }
    assertExpected(source, request.expected_revision);
    const requiredJobs = Object.values(source.processing.jobs ?? {}).filter(
        (job) =>
            job.required &&
            !source.processing.supersessions?.[job.id] &&
            (job.target_fingerprint === undefined || job.target_fingerprint === request.target_fingerprint),
    );
    const blocked = requiredJobs.some((job) => source.processing.completions?.[job.id]?.status === 'blocked');
    const pending = requiredJobs.some((job) => source.processing.completions?.[job.id] === undefined);
    const budget = source.processing.budget;
    // The reserve is enforced against the model context limit at provider preparation, not subtracted twice here.
    const overBudget = budget !== undefined && request.measured_input_tokens > budget.max_input_tokens;
    const hasBudgetJob = requiredJobs.some(
        (job) => job.scope === 'on_budget' && source.processing.completions?.[job.id] === undefined,
    );
    const status = blocked || (overBudget && !hasBudgetJob) ? 'blocked' : pending || overBudget ? 'pending' : 'ready';
    const draftReceipt = processingReceipt(source, request.operation_id, fingerprint, request.recorded_at, 'coverage');
    const coverage = ProcessingReadinessCoverageSchema.parse({
        context_fingerprint: await processingContextFingerprint(source),
        policy_revision: source.processing.policy_revision,
        target_fingerprint: request.target_fingerprint,
        measurement: {
            input_tokens: request.measured_input_tokens,
            tokenizer_id: request.tokenizer_id,
            fingerprint: request.measurement_fingerprint,
        },
        required_job_ids: requiredJobs.map((job) => job.id),
        status,
        evaluated_at_revision: draftReceipt.result_revision,
        recorded_at: request.recorded_at,
    });
    if (!draftReceipt.processing_operation) throw new Error('Processing coverage receipt has no operation detail');
    const receipt: OperationReceipt = {
        ...draftReceipt,
        processing_operation: {
            ...draftReceipt.processing_operation,
            result_fingerprint: await fingerprintJson(coverage),
        },
    };
    const document = advanceProcessing(source, receipt, {
        ...source.processing,
        coverage,
        coverage_receipts: { ...source.processing.coverage_receipts, [receipt.id]: coverage },
    });
    return { document, change: processingChangeFromReceipt(receipt), applied: true, coverage };
}

function jobConfiguration(job: ProcessingJob): ProcessorConfiguration {
    return ProcessorConfigurationSchema.parse({
        id: job.processor_id,
        version: job.processor_version,
        scope: job.scope,
        config: job.configuration,
        required: job.required,
        failure_behavior: job.failure_behavior,
    });
}

export async function resolveProcessingJobInput(
    document: ConversationDocument,
    job: ProcessingJob,
    recordedAt: string,
    capturedTarget?: string,
): Promise<ProcessingResolvedInput> {
    const targetFingerprint = job.target_fingerprint ?? capturedTarget;
    let entryIds: string[];
    let selectedBlockIds: Record<string, string[]> | undefined;
    let selectedEntries: ProcessingResolvedInput['selected_entries'];
    if (job.selection.kind === 'entries') {
        entryIds = [...job.selection.entry_ids];
        selectedBlockIds = job.selection.selected_block_ids;
        selectedEntries = job.selection.selected_entries;
    } else {
        const predecessor = document.processing.completions?.[job.selection.job_id];
        if (!predecessor) throw new Error('Predecessor processing stage is not complete');
        if (predecessor.status === 'blocked') throw new Error('Predecessor processing stage is blocked');
        entryIds =
            predecessor.status === 'applied'
                ? predecessor.inserted_entry_ids
                : [...(document.processing.resolved_inputs?.[job.selection.job_id]?.entry_ids ?? [])];
        if (predecessor.status !== 'applied') {
            const previous = document.processing.resolved_inputs?.[job.selection.job_id];
            selectedBlockIds = previous?.selected_block_ids;
            selectedEntries = previous?.selected_entries;
        }
    }
    if (entryIds.length === 0) {
        return {
            job_id: job.id,
            source_revision: document.revision,
            context_revision: document.context.revision,
            entry_ids: [],
            source_fingerprint: await fingerprintJson({ entry_ids: [] }),
            context_fingerprint: await processingContextFingerprint(document),
            source_turn_ids: [],
            ...(targetFingerprint === undefined ? {} : { target_fingerprint: targetFingerprint }),
            recorded_at: recordedAt,
        } satisfies ProcessingResolvedInput;
    }
    if (isToolResultTextProcessor(job)) {
        if (selectedBlockIds !== undefined)
            throw new Error('Tool-result text processing cannot select partial executable blocks');
        const selected = await toolResultTextSelection(document, entryIds);
        return ProcessingResolvedInputSchema.parse({
            job_id: job.id,
            source_revision: document.revision,
            context_revision: document.context.revision,
            entry_ids: entryIds,
            source_fingerprint: selected.source_fingerprint,
            context_fingerprint: await processingContextFingerprint(document),
            source_turn_ids: selected.records.map((item) => item.turn.id),
            ...(targetFingerprint === undefined ? {} : { target_fingerprint: targetFingerprint }),
            recorded_at: recordedAt,
        });
    }
    const plan = await planContextChange(
        document,
        ContextChangePlanInputSchema.parse({
            expected_revision: document.revision,
            expected_context_revision: document.context.revision,
            entry_ids: entryIds,
            ...(selectedBlockIds === undefined ? {} : { selected_block_ids: selectedBlockIds }),
            ...(selectedEntries === undefined ? {} : { selected_entries: selectedEntries }),
        }),
    );
    return ProcessingResolvedInputSchema.parse({
        job_id: job.id,
        source_revision: document.revision,
        context_revision: document.context.revision,
        entry_ids: plan.entry_ids,
        ...(selectedBlockIds === undefined ? {} : { selected_block_ids: selectedBlockIds }),
        ...(selectedEntries === undefined ? {} : { selected_entries: selectedEntries }),
        source_fingerprint: plan.source_fingerprint,
        context_fingerprint: await processingContextFingerprint(document),
        source_turn_ids: plan.source_turn_ids,
        ...(targetFingerprint === undefined ? {} : { target_fingerprint: targetFingerprint }),
        recorded_at: recordedAt,
    });
}

/** Deterministic receipt/document construction only; performs no store, registry, or processor work. */
export async function buildProcessingPhaseDocument(
    source: ConversationDocument,
    phase: NonNullable<OperationReceipt['processing_operation']>['phase'],
    job: ProcessingJob,
    value: unknown,
    recordedAt: string,
    processing: ConversationDocument['processing'],
): Promise<ConversationDocument> {
    const id = `processing:${phase}:${job.id}`;
    const receipt = processingReceipt(source, id, await fingerprintJson(value), recordedAt, phase, job.id);
    return advanceProcessing(source, receipt, processing);
}

async function commitPhase(
    store: ProcessingStore,
    source: ConversationDocument,
    phase: NonNullable<OperationReceipt['processing_operation']>['phase'],
    job: ProcessingJob,
    value: unknown,
    recordedAt: string,
    processing: ConversationDocument['processing'],
): Promise<boolean> {
    return store.commit(
        source.revision,
        await buildProcessingPhaseDocument(source, phase, job, value, recordedAt, processing),
    );
}

async function persistOutput(
    store: ProcessingStore,
    job: ProcessingJob,
    output: ProcessingOutputReceipt,
    exactAttemptRevision?: number,
): Promise<ConversationDocument> {
    for (let attempt = 0; attempt < 8; attempt += 1) {
        const document = parseConversationDocument(await store.load());
        if (exactAttemptRevision !== undefined && document.revision !== exactAttemptRevision)
            throw new Error('Deterministic processing recovery lost its exact attempt and source');
        const prior = document.processing.outputs?.[job.id];
        if (prior) {
            if (canonicalJsonContentString(prior) !== canonicalJsonContentString(output))
                throw new Error('Processing output conflicts with its durable result');
            return document;
        }
        const resolution = document.processing.resolved_inputs?.[job.id];
        if (!resolution || (await fingerprintJson(resolution)) !== output.resolved_input_fingerprint)
            throw new Error('Processing output no longer matches its resolved input');
        const attempt = document.processing.attempts?.[job.id];
        if (
            output.attempt_token === undefined ? attempt !== undefined : attempt?.attempt_token !== output.attempt_token
        )
            throw new Error('Processing output does not match the fenced attempt');
        if (
            await commitPhase(store, document, 'output', job, output, output.recorded_at, {
                ...document.processing,
                outputs: { ...document.processing.outputs, [job.id]: output },
            })
        )
            return parseConversationDocument(await store.load());
    }
    throw new Error('Processing output could not be persisted after concurrent updates');
}

type UnfingerprintedOutput = ProcessingOutputReceipt extends infer T
    ? T extends ProcessingOutputReceipt
        ? Omit<T, 'output_fingerprint'>
        : never
    : never;

function boundedOutput(value: UnfingerprintedOutput): Promise<ProcessingOutputReceipt> {
    const snapshot = ownJson(value);
    const bytes = new TextEncoder().encode(canonicalJsonContentString(snapshot));
    if (bytes.byteLength > MAX_PROCESSING_OUTPUT_BYTES)
        throw new RangeError('Processing result exceeds durable output bound');
    return fingerprintJson(snapshot).then((output_fingerprint) =>
        ProcessingOutputReceiptSchema.parse({ ...snapshot, output_fingerprint }),
    );
}

async function assertOutputIntegrity(output: ProcessingOutputReceipt): Promise<void> {
    const { output_fingerprint: fingerprint, ...payload } = output;
    if ((await fingerprintJson(payload)) !== fingerprint)
        throw new Error('Durable processing output fingerprint changed');
}

/** Capture independently reported usage without reading a getter on a malformed processor result. */
function ownReportedUsage(result: unknown): GenerationUsage | undefined {
    if (result === null || typeof result !== 'object') return undefined;
    const descriptor = Object.getOwnPropertyDescriptor(result, 'usage');
    if (!descriptor || !Object.hasOwn(descriptor, 'value') || descriptor.value === undefined) return undefined;
    const preflight = preflightJsonInput(descriptor.value, { max_bytes: 64 * 1024 });
    if (!preflight.success) return undefined;
    const parsed = GenerationUsageSchema.safeParse(structuredClone(descriptor.value));
    return parsed.success ? parsed.data : undefined;
}

function assertProcessingTarget(
    job: ProcessingJob,
    resolution: ProcessingResolvedInput | undefined,
    capturedTarget: string | undefined,
): void {
    if (capturedTarget && job.target_fingerprint && job.target_fingerprint !== capturedTarget)
        throw new Error('Processing job target conflicts with captured target');
    const expectedTarget = job.target_fingerprint ?? capturedTarget;
    if (resolution && expectedTarget && resolution.target_fingerprint !== expectedTarget)
        throw new Error('Processing resolution target evidence is unavailable or conflicts with captured target');
}

/** Settle a retained output deterministically. Never resolves or invokes a processor. */
export async function buildProcessingCompletionDocument(
    document: ConversationDocument,
    job: ProcessingJob,
    output: ProcessingOutputReceipt,
    recordedAt: string,
): Promise<ConversationDocument> {
    const jobId = job.id;
    const retainedResolution = document.processing.resolved_inputs?.[jobId];
    await assertOutputIntegrity(output);
    if (!retainedResolution || (await fingerprintJson(retainedResolution)) !== output.resolved_input_fingerprint)
        throw new Error('Durable processing result lost its exact source selection');
    if (output.kind === 'json_minification') {
        if ((await processingContextFingerprint(document)) !== retainedResolution.context_fingerprint)
            throw new Error('Processing source context changed before applying its durable result');
        const completed = await applyJsonMinificationOutput(document, job, retainedResolution, output, recordedAt);
        return completed;
    }
    if (isToolResultTextProcessor(job) && output.kind === 'proposal') {
        if ((await processingContextFingerprint(document)) !== retainedResolution.context_fingerprint)
            throw new Error('Tool-result text context changed before applying its durable output');
        return applyToolResultTextExternalizationOutput(document, job, retainedResolution, output, recordedAt);
    }
    if (output.kind === 'proposal') {
        const proposal = ContextChangeProposalSchema.parse(output.proposal);
        if ((await processingContextFingerprint(document)) !== retainedResolution.context_fingerprint)
            throw new Error('Processing source context changed before applying its durable result');
        const selection = ContextChangePlanInputSchema.parse({
            expected_revision: document.revision,
            expected_context_revision: document.context.revision,
            entry_ids: retainedResolution.entry_ids,
            ...(retainedResolution.selected_block_ids === undefined
                ? {}
                : {
                      selected_block_ids: retainedResolution.selected_block_ids,
                      selected_entries: retainedResolution.selected_entries,
                  }),
        });
        const currentPlan = await planContextChange(document, selection);
        const ranges =
            proposal.kind === 'replace_with_compaction' && proposal.fidelity === 'retrievable'
                ? contextChangeSelectedRanges(document, selection)
                : undefined;
        const appliedProposal =
            proposal.kind === 'replace_with_compaction'
                ? {
                      ...proposal,
                      replacement_turns: proposal.replacement_turns.map((turn, index) => {
                          const sourceTurnIds = ranges ? ranges[index]?.turn_ids : retainedResolution.source_turn_ids;
                          if (
                              turn.provenance.type !== 'derived' ||
                              turn.provenance.source_hash !== retainedResolution.source_fingerprint ||
                              !sourceTurnIds ||
                              canonicalJsonContentString(turn.provenance.source_turn_ids) !==
                                  canonicalJsonContentString(sourceTurnIds)
                          )
                              throw new Error('Processing proposal is not derived from its resolved source');
                          return {
                              ...turn,
                              provenance: { ...turn.provenance, source_hash: currentPlan.source_fingerprint },
                          };
                      }),
                  }
                : proposal;
        const operationId = `processing:apply:${jobId}`;
        const applied = await applyContextChange(
            document,
            ContextChangeRequestSchema.parse({
                operation_id: operationId,
                expected_revision: document.revision,
                expected_context_revision: document.context.revision,
                expected_source_fingerprint: currentPlan.source_fingerprint,
                recorded_at: recordedAt,
                entry_ids: retainedResolution.entry_ids,
                ...(retainedResolution.selected_block_ids === undefined
                    ? {}
                    : {
                          selected_block_ids: retainedResolution.selected_block_ids,
                          selected_entries: retainedResolution.selected_entries,
                      }),
                proposal: appliedProposal,
            }),
        );
        const inserted = applied.change.operations[0].inserted_entry_ids;
        const { coverage: _coverage, ...processing } = applied.document.processing;
        const completed = parseConversationDocument({
            ...applied.document,
            processing: {
                ...processing,
                completions: {
                    ...applied.document.processing.completions,
                    [jobId]: {
                        job_id: jobId,
                        output_fingerprint: output.output_fingerprint,
                        status: 'applied',
                        result_revision: applied.document.revision,
                        inserted_entry_ids: inserted,
                        context_change_operation_id: operationId,
                        recorded_at: applied.document.updated_at,
                    },
                },
            },
        });
        return completed;
    }
    const status =
        output.kind === 'no_op' || output.kind === 'json_minification_no_op'
            ? 'no_op'
            : job.required || job.failure_behavior === 'block'
              ? 'blocked'
              : 'skipped';
    const at = recordedAt;
    const completion: NonNullable<ConversationDocument['processing']['completions']>[string] = {
        job_id: jobId,
        output_fingerprint: output.output_fingerprint,
        status,
        result_revision: nextRevision(document.revision),
        inserted_entry_ids: [],
        recorded_at: at,
    };
    const { coverage: _coverage, ...processing } = document.processing;
    const next = advanceProcessing(
        document,
        processingReceipt(
            document,
            `processing:complete:${jobId}`,
            await fingerprintJson(completion),
            at,
            'complete',
            jobId,
        ),
        {
            ...processing,
            completions: { ...document.processing.completions, [jobId]: completion },
        },
    );
    return next;
}

/** Drive one durable stage. The registry and atomic store are injected; no callback is serialized. */
export async function runProcessingJob(
    store: ProcessingStore,
    registry: ProcessorRegistry,
    jobId: string,
    attemptToken: string,
    recordedAt: () => string,
    signal?: AbortSignal,
    capabilities: {
        json_minification?: JsonMinificationHostCapability;
        /** Exact built-in pure processor may reconstruct an interrupted attempt; never authorizes inference retry. */
        text_externalization_recovery?: TextExternalizationRetrievalBinder;
        /** Trusted host target captured at durable resolution, never inferred from mutable policy. */
        target_fingerprint?: string;
    } = {},
): Promise<ProcessingRunResult> {
    const jsonCapability = captureJsonMinificationHostCapability(capabilities.json_minification);
    const textRecoveryBinder = capabilities.text_externalization_recovery;
    const capturedTarget = ContentHashSchema.optional().parse(capabilities.target_fingerprint);
    if (capturedTarget && jsonCapability && jsonCapability.target_fingerprint !== capturedTarget)
        throw new Error('Processing host capability target conflicts with captured target');
    IdentifierSchema.parse(attemptToken);
    for (let retries = 0; retries < 16; retries += 1) {
        signal?.throwIfAborted();
        let document = parseConversationDocument(await store.load());
        const job = document.processing.jobs?.[jobId];
        if (!job) throw new Error(`Processing job ${jobId} does not exist`);
        if (document.processing.supersessions?.[jobId])
            return ProcessingRunResultSchema.parse({ status: 'superseded', document });
        if ((await fingerprintJson(job.selection)) !== job.selection_fingerprint)
            throw new Error('Processing job selection fingerprint changed');
        assertProcessingTarget(job, document.processing.resolved_inputs?.[jobId], capturedTarget);
        const boundTarget = job.target_fingerprint ?? document.processing.resolved_inputs?.[jobId]?.target_fingerprint;
        if (jsonCapability && boundTarget && jsonCapability.target_fingerprint !== boundTarget)
            throw new Error('Processing host capability target conflicts with durable target');
        if (document.processing.completions?.[jobId]) {
            const completedOutput = document.processing.outputs?.[jobId];
            if (completedOutput) await assertOutputIntegrity(completedOutput);
            if (completedOutput?.kind === 'json_minification') document = await verifyDerivedBlockLineage(document);
            return ProcessingRunResultSchema.parse({ status: 'completed', document });
        }
        const configuration = jobConfiguration(job);
        if ((await fingerprintJson(configuration.config)) !== job.configuration_fingerprint)
            throw new Error('Processing stage configuration fingerprint changed');
        let resolution = document.processing.resolved_inputs?.[jobId];
        if (!resolution) {
            const prepared = await resolveProcessingJobInput(document, job, recordedAt(), capturedTarget);
            if (
                !(await commitPhase(store, document, 'resolve', job, prepared, prepared.recorded_at, {
                    ...document.processing,
                    resolved_inputs: { ...document.processing.resolved_inputs, [jobId]: prepared },
                }))
            )
                continue;
            document = parseConversationDocument(await store.load());
            resolution = document.processing.resolved_inputs?.[jobId];
            if (!resolution) throw new Error('Committed processing resolution is unavailable');
        }
        assertProcessingTarget(job, resolution, capturedTarget);
        const resolvedFingerprint = await fingerprintJson(resolution);
        let output = document.processing.outputs?.[jobId];
        let recoveryAttemptRevision: number | undefined;
        if (!output) {
            if (!resolution.entry_ids.length) {
                output = await boundedOutput({
                    kind: 'no_op',
                    job_id: jobId,
                    resolved_input_fingerprint: resolvedFingerprint,
                    reason: 'no_eligible_blocks',
                    recorded_at: recordedAt(),
                });
            } else if (document.processing.attempts?.[jobId]) {
                const priorAttempt = document.processing.attempts[jobId];
                if (
                    (!isToolResultTextProcessor(job) &&
                        (job.processor_id !== TEXT_EXTERNALIZATION_PROCESSOR_ID ||
                            job.processor_version !== TEXT_EXTERNALIZATION_PROCESSOR_VERSION)) ||
                    !textRecoveryBinder
                ) {
                    return ProcessingRunResultSchema.parse({ status: 'in_progress', document });
                }
                const attemptReceipt = document.operation_receipts[`processing:attempt:${jobId}`];
                if (
                    attemptReceipt?.operation_kind !== 'processing' ||
                    attemptReceipt.processing_operation?.phase !== 'attempt' ||
                    attemptReceipt.processing_operation.job_id !== jobId ||
                    attemptReceipt.result_revision !== document.revision ||
                    priorAttempt.resolved_input_fingerprint !== resolvedFingerprint ||
                    (await processingContextFingerprint(document)) !== resolution.context_fingerprint
                ) {
                    throw new Error('Deterministic processing recovery lost its exact attempt and source');
                }
                recoveryAttemptRevision = document.revision;
                signal?.throwIfAborted();
                const reconstructed = await (isToolResultTextProcessor(job)
                    ? createToolResultTextExternalizationProcessor(textRecoveryBinder)
                    : createTextExternalizationProcessor(textRecoveryBinder)
                ).run({
                    document: structuredClone(document),
                    job: structuredClone(job),
                    resolved_input: structuredClone(resolution),
                    configuration: structuredClone(configuration),
                    signal,
                });
                if (reconstructed.kind !== 'proposal') {
                    throw new Error('Deterministic text recovery did not reconstruct the expected proposal');
                }
                output = await boundedOutput({
                    ...reconstructed,
                    job_id: jobId,
                    resolved_input_fingerprint: resolvedFingerprint,
                    attempt_token: priorAttempt.attempt_token,
                    recorded_at: recordedAt(),
                });
            } else {
                const startedAt = recordedAt();
                const attemptReceipt = {
                    job_id: jobId,
                    resolved_input_fingerprint: resolvedFingerprint,
                    attempt_token: attemptToken,
                    started_at: startedAt,
                };
                if (
                    !(await commitPhase(store, document, 'attempt', job, attemptReceipt, startedAt, {
                        ...document.processing,
                        attempts: { ...document.processing.attempts, [jobId]: attemptReceipt },
                    }))
                )
                    continue;
                document = parseConversationDocument(await store.load());
                const processor = registry.resolve(job.processor_id, job.processor_version);
                if (!processor) {
                    output = await boundedOutput({
                        kind: 'failed',
                        job_id: jobId,
                        resolved_input_fingerprint: resolvedFingerprint,
                        attempt_token: attemptToken,
                        diagnostic: `Processor ${job.processor_id}@${job.processor_version} is unavailable`,
                        recorded_at: recordedAt(),
                    });
                } else {
                    let reportedUsage: GenerationUsage | undefined;
                    try {
                        signal?.throwIfAborted();
                        const result = await processor.run({
                            document: structuredClone(document),
                            job: structuredClone(job),
                            resolved_input: structuredClone(resolution),
                            configuration: structuredClone(configuration),
                            signal,
                        });
                        reportedUsage = ownReportedUsage(result);
                        const snapshot = ownJson(result);
                        let verifiedResult:
                            | ProcessorResult
                            | Awaited<ReturnType<typeof validateJsonMinificationCandidate>> = snapshot;
                        if (snapshot.kind === 'json_minification_candidate') {
                            verifiedResult = await validateJsonMinificationCandidate(
                                document,
                                job,
                                resolution,
                                snapshot,
                                jsonCapability,
                                signal,
                            );
                        } else if (snapshot.kind === 'json_minification_no_op') {
                            const expected = await jsonMinificationProcessor.run({
                                document: structuredClone(document),
                                job: structuredClone(job),
                                resolved_input: structuredClone(resolution),
                                configuration: structuredClone(configuration),
                                signal,
                            });
                            if (canonicalJsonContentString(expected) !== canonicalJsonContentString(snapshot))
                                throw new ProcessingKnownFailure(
                                    'JSON minification no-op conflicts with verified source',
                                );
                        } else if (snapshot.kind !== 'proposal' && snapshot.kind !== 'no_op') {
                            throw new ProcessingKnownFailure(
                                'Processor returned an unsupported or unverified result kind',
                            );
                        }
                        output = await boundedOutput({
                            ...verifiedResult,
                            job_id: jobId,
                            resolved_input_fingerprint: resolvedFingerprint,
                            attempt_token: attemptToken,
                            recorded_at: recordedAt(),
                        } as UnfingerprintedOutput);
                    } catch (cause: unknown) {
                        // Cancellation leaves the durable attempt unresolved for explicit fenced recovery.
                        if (signal?.aborted) throw cause;
                        output = await boundedOutput({
                            kind: cause instanceof ProcessingKnownFailure ? 'failed' : 'unknown_outcome',
                            job_id: jobId,
                            resolved_input_fingerprint: resolvedFingerprint,
                            attempt_token: attemptToken,
                            diagnostic:
                                cause instanceof Error ? cause.message.slice(0, 8192) : 'Unknown processor failure',
                            ...(cause instanceof ProcessingKnownFailure || cause instanceof ProcessingUnknownFailure
                                ? cause.usage === undefined
                                    ? {}
                                    : { usage: cause.usage }
                                : reportedUsage === undefined
                                  ? {}
                                  : { usage: reportedUsage }),
                            recorded_at: recordedAt(),
                        });
                    }
                }
            }
            document = await persistOutput(store, job, output, recoveryAttemptRevision);
        }
        const completed = await buildProcessingCompletionDocument(document, job, output, recordedAt());
        signal?.throwIfAborted();
        if (await store.commit(document.revision, completed))
            return ProcessingRunResultSchema.parse({ status: 'completed', document: completed });
    }
    throw new Error(`Processing job ${jobId} could not commit after concurrent updates`);
}

const ProcessingAbandonCommandSchema = z.strictObject({
    operation_id: IdentifierSchema,
    job_id: IdentifierSchema,
    attempt_token: IdentifierSchema,
    expected_revision: NonnegativeSafeIntegerSchema,
    recorded_at: TimestampSchema,
});

export type ProcessingAbandonCommand = z.infer<typeof ProcessingAbandonCommandSchema>;

/** Host-owned recovery decision. An observed attempt alone never authorizes unknown-outcome publication. */
export async function abandonProcessingAttempt(
    store: ProcessingStore,
    commandInput: ProcessingAbandonCommand,
): Promise<ProcessingRunResult> {
    const command = ProcessingAbandonCommandSchema.parse(ownJson(commandInput));
    const document = parseConversationDocument(await store.load());
    const job = document.processing.jobs?.[command.job_id];
    const attempt = document.processing.attempts?.[command.job_id];
    if (!job || !attempt || attempt.attempt_token !== command.attempt_token)
        throw new Error('Processing abandon command does not match a fenced attempt');
    const prior = document.processing.outputs?.[command.job_id];
    if (prior) {
        if (
            prior.kind !== 'unknown_outcome' ||
            prior.recovery_operation_id !== command.operation_id ||
            prior.attempt_token !== command.attempt_token ||
            prior.recorded_at !== command.recorded_at
        )
            throw new Error('Processing attempt already has a different durable output');
    } else {
        assertExpected(document, command.expected_revision);
        const output = await boundedOutput({
            kind: 'unknown_outcome',
            job_id: command.job_id,
            resolved_input_fingerprint: attempt.resolved_input_fingerprint,
            attempt_token: command.attempt_token,
            recovery_operation_id: command.operation_id,
            diagnostic: 'Host explicitly abandoned an attempt without a durable processor result',
            recorded_at: command.recorded_at,
        });
        const committed = await commitPhase(store, document, 'output', job, output, command.recorded_at, {
            ...document.processing,
            outputs: { ...document.processing.outputs, [job.id]: output },
        });
        if (!committed) throw new Error('Processing abandon command lost its exact-head CAS');
    }
    return runProcessingJob(
        store,
        { resolve: () => undefined },
        job.id,
        command.attempt_token,
        () => command.recorded_at,
    );
}
