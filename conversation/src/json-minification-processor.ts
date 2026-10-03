import type { z } from 'zod';
import { canonicalJsonContentString } from './content-integrity.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { fingerprintJson } from './identity.js';
import { minifyJsonLexically } from './json-minification.js';
import { preflightJsonInput } from './json-preflight.js';
import type { ConversationProcessor } from './processing.js';
import {
    JsonMinificationCandidateSchema,
    JsonMinificationConfigurationSchema,
    JsonMinificationMeasurementIdentitySchema,
    JsonMinificationMeasurementSchema,
    JsonMinificationProposalSchema,
} from './schemas/json-minification.js';
import type { ConversationDocument, ProcessingJob, ProcessingResolvedInput } from './types.js';

export const JSON_MINIFICATION_PROCESSOR_ID = 'builtin.json_minification';
export const JSON_MINIFICATION_PROCESSOR_VERSION = '1';
type Candidate = z.infer<typeof JsonMinificationCandidateSchema>;
type Proposal = z.infer<typeof JsonMinificationProposalSchema>;
type Measurement = z.infer<typeof JsonMinificationMeasurementSchema>;

/** Ephemeral prospective projection, never an accepted document or transport input. */
export interface JsonMinificationProspectiveInput {
    source: ConversationDocument;
    processing_job_id: string;
    resolved_input_fingerprint: string;
    candidate: Candidate;
    context_fingerprint: string;
    target_fingerprint: string;
    input_fingerprint: string;
}
export interface JsonMinificationHostCapability {
    target_fingerprint: string;
    identity: z.infer<typeof JsonMinificationMeasurementIdentitySchema>;
    measureProspective(
        input: JsonMinificationProspectiveInput,
        signal?: AbortSignal,
    ): Promise<
        { status: 'unavailable' } | { status: 'measured'; measurement: Omit<Measurement, 'minimum_token_reduction'> }
    >;
}

export const jsonMinificationProcessor: ConversationProcessor = {
    async run({ document, job, resolved_input, configuration, signal }) {
        const config = JsonMinificationConfigurationSchema.parse(configuration.config);
        if (
            configuration.id !== JSON_MINIFICATION_PROCESSOR_ID ||
            configuration.version !== JSON_MINIFICATION_PROCESSOR_VERSION
        )
            throw new Error('JSON minification processor configuration identity conflicts');
        const turns = createContextTurnIndex(document);
        const transforms: Candidate['transforms'] = [];
        let codeUnits = 0;
        let eligible = 0;
        let lineageBlocks = 0;
        const prospectiveRecords: unknown[] = [];
        for (const entryId of resolved_input.entry_ids) {
            const entry = document.context.entries.find((item) => item.id === entryId);
            if (!entry) throw new Error('JSON minification entry is unavailable');
            const { turn, blocks } = resolveContextEntry(turns, entry);
            if (turn.kind !== 'user' && turn.kind !== 'agent' && turn.kind !== 'program')
                throw new Error('JSON minification does not transform executable exchanges');
            if (
                turn.status !== 'completed' ||
                turn.execution_id !== undefined ||
                turn.exchange_id !== undefined ||
                blocks.some((block) => block.type !== 'text' && block.type !== 'json')
            )
                throw new Error('JSON minification does not rewrite incomplete/executable or unsupported content');
            const before = transforms.length;
            const selected = resolved_input.selected_block_ids?.[entryId];
            for (const block of blocks) {
                signal?.throwIfAborted();
                if ((selected && !selected.includes(block.id)) || block.type !== 'text') continue;
                if (block.format !== 'plain')
                    throw new Error('JSON minification requires explicitly selected plain raw JSON text');
                eligible++;
                codeUnits += block.text.length;
                if (codeUnits > config.max_code_units)
                    throw new RangeError('JSON minification batch exceeds source bound');
                const replacement = minifyJsonLexically(block.text, {
                    limits: {
                        max_code_units: config.max_code_units,
                        max_depth: config.max_depth,
                        max_lexical_tokens: config.max_lexical_tokens,
                    },
                    signal,
                });
                if (replacement === block.text) continue;
                if (transforms.length >= 256) throw new RangeError('JSON minification batch exceeds transform bound');
                transforms.push({
                    entry_id: entryId,
                    source_slice: {
                        source: { conversation_id: document.id, revision: resolved_input.source_revision },
                        turn_id: turn.id,
                        block_id: block.id,
                        block_fingerprint: await fingerprintJson(block),
                        selection: { kind: 'whole' },
                    },
                    replacement_text: replacement,
                    replacement_text_fingerprint: await fingerprintJson(replacement),
                });
            }
            if (transforms.length !== before) {
                lineageBlocks += blocks.length;
                if (lineageBlocks > 256)
                    throw new RangeError('JSON minification batch exceeds complete lineage block bound');
                prospectiveRecords.push({ entry, turn, active_blocks: blocks });
            }
        }
        if (
            new TextEncoder().encode(canonicalJsonContentString({ transforms, prospectiveRecords })).byteLength >
            768 * 1024
        )
            throw new RangeError('JSON minification prospective records exceed 768KiB bound');
        signal?.throwIfAborted();
        if (!transforms.length)
            return { kind: 'json_minification_no_op', reason: eligible ? 'already_minified' : 'no_eligible_blocks' };
        return JsonMinificationCandidateSchema.parse({
            kind: 'json_minification_candidate',
            parser: 'rfc8259-lexical-v1',
            strategy: {
                id: configuration.id,
                version: configuration.version,
                configuration_fingerprint: job.configuration_fingerprint,
            },
            source_fingerprint: resolved_input.source_fingerprint,
            transforms,
        });
    },
};

/** Reconstruct and verify source/output before asking a separately configured host to count a dry projection. */
export async function validateJsonMinificationCandidate(
    document: ConversationDocument,
    jobInput: ProcessingJob,
    resolutionInput: ProcessingResolvedInput,
    candidateInput: Candidate,
    capability?: JsonMinificationHostCapability,
    signal?: AbortSignal,
): Promise<
    | { kind: 'json_minification'; proposal: Proposal }
    | {
          kind: 'json_minification_no_op';
          reason: 'measurement_unavailable' | 'no_token_benefit';
          measurement?: Measurement;
      }
> {
    const host = captureJsonMinificationHostCapability(capability);
    const source = structuredClone(document);
    const job = structuredClone(jobInput);
    const resolution = structuredClone(resolutionInput);
    const candidate = JsonMinificationCandidateSchema.parse(structuredClone(candidateInput));
    const config = JsonMinificationConfigurationSchema.parse(job.configuration);
    const expected = await jsonMinificationProcessor.run({
        document: source,
        job,
        resolved_input: resolution,
        configuration: {
            id: job.processor_id,
            version: job.processor_version,
            config: job.configuration,
            scope: job.scope,
            required: job.required,
            failure_behavior: job.failure_behavior,
        },
        signal,
    });
    if (canonicalJsonContentString(expected) !== canonicalJsonContentString(candidate))
        throw new Error('JSON minification candidate differs from its verified selected source/output');
    if ((await fingerprintJson(job.configuration)) !== job.configuration_fingerprint)
        throw new Error('JSON minification configuration fingerprint conflicts');
    if (!host || !resolution.target_fingerprint || host.target_fingerprint !== resolution.target_fingerprint)
        return { kind: 'json_minification_no_op', reason: 'measurement_unavailable' };
    const identity = JsonMinificationMeasurementIdentitySchema.parse(structuredClone(host.identity));
    const targetFingerprint = host.target_fingerprint;
    const measure = host.measureProspective;
    const resolvedInputFingerprint = await fingerprintJson(resolution);
    const inputFingerprint = await fingerprintJson({
        processing_job_id: job.id,
        resolved_input_fingerprint: resolvedInputFingerprint,
        context_fingerprint: resolution.context_fingerprint,
        candidate,
        target_fingerprint: targetFingerprint,
    });
    signal?.throwIfAborted();
    const result = await boundedMeasurement(
        measure,
        {
            source: structuredClone(source),
            processing_job_id: job.id,
            resolved_input_fingerprint: resolvedInputFingerprint,
            candidate: structuredClone(candidate),
            context_fingerprint: resolution.context_fingerprint,
            target_fingerprint: targetFingerprint,
            input_fingerprint: inputFingerprint,
        },
        signal,
    );
    signal?.throwIfAborted();
    if (!preflightJsonInput(result, { max_bytes: 64 * 1024 }).success)
        throw new Error('JSON minification measurement is not bounded owned JSON');
    if (result.status === 'unavailable') return { kind: 'json_minification_no_op', reason: 'measurement_unavailable' };
    const measurement = JsonMinificationMeasurementSchema.parse({
        ...structuredClone(result.measurement),
        minimum_token_reduction: config.minimum_token_reduction,
    });
    if (
        measurement.target_fingerprint !== targetFingerprint ||
        measurement.prospective_input_fingerprint !== inputFingerprint ||
        measurement.original.source_fingerprint !== resolution.context_fingerprint ||
        measurement.replacement.source_fingerprint !== inputFingerprint ||
        canonicalJsonContentString(measurementIdentity(measurement.original)) !==
            canonicalJsonContentString(identity) ||
        canonicalJsonContentString(measurementIdentity(measurement.replacement)) !==
            canonicalJsonContentString(identity)
    )
        throw new Error('JSON minification host measurement binding conflicts');
    if (measurement.original.input_tokens - measurement.replacement.input_tokens < config.minimum_token_reduction)
        return { kind: 'json_minification_no_op', reason: 'no_token_benefit', measurement };
    return {
        kind: 'json_minification',
        proposal: JsonMinificationProposalSchema.parse({
            ...candidate,
            kind: 'validated_json_minification',
            measurement,
        }),
    };
}

function measurementIdentity(value: Measurement['original']) {
    const { tokenizer, tokenizer_version, adapter, adapter_version, target_model, method } = value;
    return { tokenizer, tokenizer_version, adapter, adapter_version, target_model, method };
}

/** The host must dry compile/count only. No provider transport or accepted readiness is authorized. */
async function boundedMeasurement(
    measure: JsonMinificationHostCapability['measureProspective'],
    input: JsonMinificationProspectiveInput,
    signal?: AbortSignal,
): ReturnType<JsonMinificationHostCapability['measureProspective']> {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let rejectAbort: ((reason: unknown) => void) | undefined;
    const aborted = new Promise<never>((_resolve, reject) => {
        rejectAbort = reject;
    });
    const cancel = () => {
        controller.abort(signal?.reason);
        rejectAbort?.(signal?.reason);
    };
    signal?.throwIfAborted();
    signal?.addEventListener('abort', cancel, { once: true });
    const timeout = new Promise<{ status: 'unavailable' }>((resolve) => {
        timer = setTimeout(() => {
            // Settle the runner's typed deadline before notifying an abort-aware host.
            resolve({ status: 'unavailable' });
            controller.abort(new Error('JSON projection count exceeded 10000ms bound'));
        }, 10000);
    });
    try {
        return await Promise.race([measure(input, controller.signal), timeout, aborted]);
    } finally {
        if (timer !== undefined) clearTimeout(timer);
        signal?.removeEventListener('abort', cancel);
    }
}

/** Capture host authority synchronously, before store, processor, hash, or count awaits. */
export function captureJsonMinificationHostCapability(
    capability?: JsonMinificationHostCapability,
): JsonMinificationHostCapability | undefined {
    if (capability === undefined) return undefined;
    const target = Object.getOwnPropertyDescriptor(capability, 'target_fingerprint');
    const identity = Object.getOwnPropertyDescriptor(capability, 'identity');
    const callback = Object.getOwnPropertyDescriptor(capability, 'measureProspective');
    if (
        !target ||
        !identity ||
        !callback ||
        !Object.hasOwn(target, 'value') ||
        !Object.hasOwn(identity, 'value') ||
        !Object.hasOwn(callback, 'value') ||
        typeof target.value !== 'string' ||
        !target.value ||
        typeof callback.value !== 'function' ||
        !preflightJsonInput(identity.value, { max_bytes: 8192 }).success
    )
        throw new TypeError('JSON minification host capability requires plain own authority data/function');
    const capturedIdentity = JsonMinificationMeasurementIdentitySchema.parse(structuredClone(identity.value));
    const capturedCallback = callback.value;
    return {
        target_fingerprint: target.value,
        identity: capturedIdentity,
        measureProspective: (input, signal) => capturedCallback(input, signal),
    };
}
