import type { z } from 'zod';
import { fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { assessProcessingReadiness, processingContextFingerprint } from './processing.js';
import { ContextMeasurementSchema } from './schemas/context-measurement.js';
import { ModelTargetSchema } from './schemas/execution.js';
import {
    type ConversationModelSwitchBlockerSchema,
    type ConversationModelSwitchBudgetAnalysisSchema,
    ConversationModelSwitchNextRequestChangeSchema,
    ConversationModelSwitchPlanSchema,
    ConversationModelSwitchRequestSchema,
} from './schemas/model-switch.js';
import { JsonValueSchema, NonnegativeSafeIntegerSchema, PositiveSafeIntegerSchema } from './schemas/primitives.js';
import type { ContextMeasurement, ConversationDocument, ModelTarget } from './types.js';
import { parseConversationDocument } from './validation.js';

export type ConversationModelSwitchRequest = z.infer<typeof ConversationModelSwitchRequestSchema>;
export type ConversationModelSwitchBlocker = z.infer<typeof ConversationModelSwitchBlockerSchema>;
export type ConversationModelSwitchBudgetAnalysis = z.infer<typeof ConversationModelSwitchBudgetAnalysisSchema>;
export type ConversationModelSwitchPlan = z.infer<typeof ConversationModelSwitchPlanSchema>;
export type ConversationModelSwitchNextRequestChange = z.infer<typeof ConversationModelSwitchNextRequestChangeSchema>;

export type ConversationModelSwitchProjection =
    | { status: 'unsupported'; reason: string }
    | {
          status: 'compiled';
          /** The actual provider SDK request body after adapter conversion, not an intermediate receipt payload. */
          native_request: unknown;
          measurement?: ContextMeasurement;
          /** Hash of the exact request body counted by the host's tokenizer. */
          counted_request_fingerprint?: string;
          context_limit_tokens?: number;
          output_reserve_tokens?: number;
          /** Existing processing coverage identity, when an enabled policy is being evaluated. */
          processing_measurement_fingerprint?: string;
      };

/** Host-supplied callbacks are evaluated again at apply and at ordinary execution preparation. */
export interface ConversationModelSwitchRuntime {
    project(document: ConversationDocument, target: ModelTarget): Promise<ConversationModelSwitchProjection>;
    hasUnsettledGeneration(document: ConversationDocument): Promise<boolean>;
}

type Blocker = ConversationModelSwitchPlan['blockers'][number];

function block(blockers: Blocker[], code: Blocker['code'], message: string): void {
    blockers.push({ code, message });
}

function hasOpenToolCall(document: ConversationDocument): boolean {
    const terminal = new Set(Object.values(document.execution_receipts).map((receipt) => receipt.call_id));
    for (const turn of document.turns) {
        for (const item of turn.blocks) {
            if (item.type === 'tool_result') terminal.add(item.call_id);
        }
    }
    for (const turn of document.turns) {
        for (const item of turn.blocks) {
            if (item.type === 'tool_call' && !terminal.has(item.call_id)) return true;
        }
    }
    return false;
}

/** Dry compatible-only plan. It cannot authorize transport or carry provider options from a prior request. */
export async function prepareModelSwitch(
    sourceInput: ConversationDocument,
    requestInput: ConversationModelSwitchRequest,
    runtime: ConversationModelSwitchRuntime,
): Promise<ConversationModelSwitchPlan> {
    const source = parseConversationDocument(sourceInput);
    const request = ConversationModelSwitchRequestSchema.parse(structuredClone(requestInput));
    const project = runtime.project;
    const hasUnsettledGeneration = runtime.hasUnsettledGeneration;
    if (typeof project !== 'function' || typeof hasUnsettledGeneration !== 'function')
        throw new TypeError('Model switch requires host projection and live-generation observation');
    const target = ModelTargetSchema.parse(structuredClone(request.target));
    const toolIds = [...source.context.active_tool_definition_ids];
    const tools = toolIds.map((id) => {
        const definition = source.tool_definitions[id];
        if (!definition) throw new Error(`Active tool definition ${id} is unavailable`);
        return definition;
    });
    const blockers: Blocker[] = [];
    const sourceRef = { conversation_id: source.id, revision: source.revision };
    if (
        request.source.conversation_id !== source.id ||
        request.source.revision !== source.revision ||
        request.expected_context_revision !== source.context.revision
    ) {
        block(blockers, 'SOURCE_CHANGED', 'Conversation source or active context revision changed');
    }
    if (hasOpenToolCall(source)) block(blockers, 'OPEN_TOOL_CALL', 'A tool call has no terminal result');
    const unsettled = await hasUnsettledGeneration(structuredClone(source));
    if (typeof unsettled !== 'boolean') throw new TypeError('Host generation observation must return a boolean');
    if (unsettled)
        block(blockers, 'UNSETTLED_GENERATION', 'A generation is still owed, streaming or awaiting settlement');

    const sourceFingerprint = await fingerprintJson(source);
    const contextFingerprint = await processingContextFingerprint(source);
    const targetFingerprint = await fingerprintJson(target);
    const optionsFingerprint = await fingerprintJson(target.options ?? {});
    const toolSetFingerprint = await fingerprintJson(tools);
    const sourceMeasurementPolicy = source.processing.budget?.measurement_policy;
    const estimateAllowed =
        sourceMeasurementPolicy !== 'exact_only' &&
        request.measurement_policy !== 'exact_only' &&
        (sourceMeasurementPolicy === 'identified_estimate' || request.measurement_policy === 'identified_estimate');
    const projection = structuredClone(await project(structuredClone(source), structuredClone(target)));
    let nativeRequestFingerprint: string | undefined;
    let measurement: ContextMeasurement | undefined;
    let budget: ConversationModelSwitchPlan['budget'];
    if (projection.status === 'unsupported') {
        block(
            blockers,
            'TARGET_UNSUPPORTED',
            projection.reason.slice(0, 1024) || 'Target cannot compile the selected context',
        );
    } else {
        if (!preflightJsonInput(projection.native_request, { max_bytes: 768 * 1024 }).success)
            throw new RangeError('Model switch native request exceeds its bounded projection envelope');
        const nativeRequest = JsonValueSchema.parse(structuredClone(projection.native_request));
        nativeRequestFingerprint = await fingerprintJson(nativeRequest);
        if (projection.measurement === undefined || projection.counted_request_fingerprint === undefined) {
            block(blockers, 'MEASUREMENT_UNAVAILABLE', 'Target has no identified native-request measurement');
        } else {
            measurement = ContextMeasurementSchema.parse(structuredClone(projection.measurement));
            if (
                projection.counted_request_fingerprint !== nativeRequestFingerprint ||
                measurement.source_fingerprint !== contextFingerprint ||
                measurement.target_model !== target.model ||
                measurement.adapter !== target.protocol ||
                measurement.adapter_version !== target.adapter_version ||
                (measurement.method === 'estimated' &&
                    (measurement.tokenizer_version === undefined || !estimateAllowed))
            ) {
                block(blockers, 'MEASUREMENT_MISMATCH', 'Measurement does not identify the selected source and target');
            }
        }
        if (projection.context_limit_tokens === undefined || projection.output_reserve_tokens === undefined) {
            block(blockers, 'BUDGET_UNAVAILABLE', 'Target context limit or output reserve is unknown');
        } else {
            const limit = PositiveSafeIntegerSchema.parse(projection.context_limit_tokens);
            const reserve = NonnegativeSafeIntegerSchema.parse(projection.output_reserve_tokens);
            if (reserve >= limit) {
                block(blockers, 'BUDGET_UNSATISFIED', 'Output reserve consumes the target context window');
            } else if (measurement !== undefined) {
                const available = limit - reserve;
                budget = {
                    context_limit_tokens: limit,
                    output_reserve_tokens: reserve,
                    available_input_tokens: available,
                    measured_input_tokens: measurement.input_tokens,
                };
                if (measurement.input_tokens > available)
                    block(blockers, 'BUDGET_UNSATISFIED', 'Measured request exceeds the target context window');
            }
        }
        if (source.processing.enabled) {
            if (projection.processing_measurement_fingerprint === undefined) {
                block(blockers, 'PROCESSING_PENDING', 'Target-specific processing coverage is unavailable');
            } else {
                const readiness = await assessProcessingReadiness(
                    source,
                    targetFingerprint,
                    projection.processing_measurement_fingerprint,
                );
                if (readiness.status !== 'ready')
                    block(
                        blockers,
                        readiness.status === 'blocked' ? 'PROCESSING_BLOCKED' : 'PROCESSING_PENDING',
                        readiness.reason,
                    );
            }
        } else {
            const readiness = await assessProcessingReadiness(source, targetFingerprint, 'sha256:disabled');
            if (readiness.status !== 'ready') block(blockers, 'PROCESSING_PENDING', readiness.reason);
        }
    }
    return ConversationModelSwitchPlanSchema.parse({
        version: 1,
        source: sourceRef,
        expected_context_revision: source.context.revision,
        source_fingerprint: sourceFingerprint,
        context_fingerprint: contextFingerprint,
        target,
        target_fingerprint: targetFingerprint,
        ...(request.measurement_policy === undefined ? {} : { measurement_policy: request.measurement_policy }),
        options_fingerprint: optionsFingerprint,
        active_tool_definition_ids: toolIds,
        tool_set_fingerprint: toolSetFingerprint,
        ...(nativeRequestFingerprint === undefined ? {} : { native_request_fingerprint: nativeRequestFingerprint }),
        ...(measurement === undefined ? {} : { measurement }),
        ...(budget === undefined ? {} : { budget }),
        compatibility: blockers.length === 0 ? 'compatible' : 'blocked',
        blockers,
        media_transformations: [],
        replay_exclusions: [],
        required_context_changes: [],
        cache_invalidations: ['native_request', 'provider_cache', 'provider_upload'],
    });
}

/** Reproject before issuing a CAS-bound next-request change; the host then prepares normally. */
export async function applyModelSwitch(
    sourceInput: ConversationDocument,
    planInput: ConversationModelSwitchPlan,
    command: { operation_id: string; recorded_at: string },
    runtime: ConversationModelSwitchRuntime,
): Promise<ConversationModelSwitchNextRequestChange> {
    const source = parseConversationDocument(sourceInput);
    const plan = ConversationModelSwitchPlanSchema.parse(structuredClone(planInput));
    const ownedCommand = structuredClone(command);
    if (plan.compatibility !== 'compatible' || plan.blockers.length !== 0)
        throw new Error('Blocked model switch cannot be applied');
    const refreshed = await prepareModelSwitch(
        source,
        {
            source: plan.source,
            expected_context_revision: plan.expected_context_revision,
            target: plan.target,
            ...(plan.measurement_policy === undefined ? {} : { measurement_policy: plan.measurement_policy }),
        },
        runtime,
    );
    if (
        refreshed.compatibility !== 'compatible' ||
        refreshed.source_fingerprint !== plan.source_fingerprint ||
        refreshed.context_fingerprint !== plan.context_fingerprint ||
        refreshed.target_fingerprint !== plan.target_fingerprint ||
        refreshed.measurement_policy !== plan.measurement_policy ||
        refreshed.options_fingerprint !== plan.options_fingerprint ||
        refreshed.active_tool_definition_ids.length !== plan.active_tool_definition_ids.length ||
        refreshed.active_tool_definition_ids.some((id, index) => id !== plan.active_tool_definition_ids[index]) ||
        refreshed.tool_set_fingerprint !== plan.tool_set_fingerprint ||
        refreshed.native_request_fingerprint !== plan.native_request_fingerprint ||
        refreshed.expected_context_revision !== plan.expected_context_revision ||
        refreshed.budget?.context_limit_tokens !== plan.budget?.context_limit_tokens ||
        refreshed.budget?.output_reserve_tokens !== plan.budget?.output_reserve_tokens ||
        refreshed.budget?.measured_input_tokens !== plan.budget?.measured_input_tokens ||
        refreshed.measurement?.method !== plan.measurement?.method ||
        refreshed.measurement?.tokenizer !== plan.measurement?.tokenizer ||
        refreshed.measurement?.tokenizer_version !== plan.measurement?.tokenizer_version
    ) {
        throw new Error('Model switch source, target, native request or budget changed before apply');
    }
    if (!refreshed.native_request_fingerprint || !refreshed.measurement || !refreshed.budget)
        throw new Error('Model switch lacks complete measured request evidence');
    return ConversationModelSwitchNextRequestChangeSchema.parse({
        ...ownedCommand,
        expected_source: refreshed.source,
        expected_context_revision: refreshed.expected_context_revision,
        expected_source_fingerprint: refreshed.source_fingerprint,
        expected_context_fingerprint: refreshed.context_fingerprint,
        target: refreshed.target,
        target_fingerprint: refreshed.target_fingerprint,
        ...(refreshed.measurement_policy === undefined ? {} : { measurement_policy: refreshed.measurement_policy }),
        options_fingerprint: refreshed.options_fingerprint,
        active_tool_definition_ids: refreshed.active_tool_definition_ids,
        tool_set_fingerprint: refreshed.tool_set_fingerprint,
        native_request_fingerprint: refreshed.native_request_fingerprint,
        measurement: refreshed.measurement,
        budget: refreshed.budget,
        requires_quiescent_generation: true,
        cache_invalidations: refreshed.cache_invalidations,
    });
}
