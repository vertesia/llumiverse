import type {
    AudioResult,
    Completion,
    CompletionResult,
    ExecutionOptions,
    ExecutionTokenUsage,
    HttpTimeoutOptions,
    JSONSchema,
    JsonResult,
    ModelOptions,
    PromptCacheDiagnostic,
    PromptCacheMode,
    ResultValidationError,
    ToolUse,
} from '@llumiverse/common';
import {
    type Asset,
    type AssetKind,
    type AssetMediaMetadata,
    type AssetStorage,
    type ConversationAcceptedOutputFragment,
    type ConversationDocument,
    type ConversationOutputBlock,
    type ConversationPreparedRequest,
    type ConversationPreparedRequestRecord,
    type ConversationRuntimeContext,
    ConversationRuntimeContextSchema,
    createAcceptedOutputFragment,
    createConversationDocument,
    parseAcceptedOutputFragment,
    parseConversationDocument,
    type ResolvedConversationRuntimeContext,
    ResolvedConversationRuntimeContextSchema,
    toolArgumentsForModel,
} from '@llumiverse/conversation';
import { MalformedStreamingToolArgumentsError } from './stream-errors.js';

/** Runtime-only provenance. Symbol keys survive internal object spreads but are never serialized on the wire. */
export const CANONICAL_ACCEPTED_RECOVERY = Symbol('llumiverse.canonical-accepted-recovery');

/** Direct canonical provider result. The complete document is authoritative; accepted_output is its safe projection. */
export interface CanonicalExecutionResponse {
    readonly [CANONICAL_ACCEPTED_RECOVERY]?: true;
    conversation: ConversationDocument;
    accepted_output: ConversationAcceptedOutputFragment;
    execution_time?: number;
    chunks?: number;
    service_tier?: string;
    prompt_cache_diagnostic?: PromptCacheDiagnostic;
    original_response?: unknown;
}

/**
 * Provider-facing options for current canonical execution.
 *
 * `ExecutionOptions` remains the supported legacy Driver boundary and therefore keeps its opaque
 * native `conversation` field. Canonical execution snapshots that boundary synchronously into this
 * shape before prompt preparation or provider I/O. From that point on the complete document and
 * resolved request identity are the only conversation authority.
 */
export type CanonicalExecutionOptions = Omit<ExecutionOptions, 'conversation' | 'conversation_runtime'> & {
    conversation: ConversationDocument;
    conversation_runtime: ResolvedConversationRuntimeContext;
};

/** Caller-facing canonical input before defaults and the empty document have been resolved. */
export type CanonicalExecutionInputOptions = Omit<ExecutionOptions, 'conversation' | 'conversation_runtime'> & {
    conversation?: ConversationDocument | null;
    conversation_runtime: ConversationRuntimeContext;
};

/**
 * Transport and response policy for current canonical execution.
 *
 * This is intentionally declared independently from legacy `ExecutionOptions`: prompt formatters,
 * native conversation values, and legacy tool DTOs are authoring/import concerns and cannot enter
 * an already materialized canonical context. Projection-retention fields remain temporarily because
 * adopted providers still apply those policies while compiling a canonical document to native wire.
 */
export interface CanonicalExecutionTransportOptions {
    model: string;
    result_schema?: JSONSchema;
    prompt_cache_schema_suffix?: boolean;
    include_original_response?: boolean;
    model_options?: ModelOptions;
    prompt_cache_key?: string;
    prompt_cache_mode?: PromptCacheMode;
    prompt_cache_ttl_seconds?: number;
    httpTimeout?: HttpTimeoutOptions;
    output_storage_uri?: string;
    store_audio?: (
        stream: ReadableStream<Uint8Array>,
        metadata: Omit<AudioResult, 'type' | 'value'>,
        signal?: AbortSignal,
    ) => Promise<string>;
    store_generated_asset?: (
        stream: ReadableStream<Uint8Array>,
        metadata: { kind: AssetKind; mime_type: string; media?: AssetMediaMetadata },
        signal?: AbortSignal,
    ) => Promise<{ storage: AssetStorage; byte_length: number; content_hash: string }>;
    load_recovered_canonical_output?: (identity: {
        conversation_id: string;
        response_operation_id: string;
        prepared_request?: ConversationPreparedRequestRecord;
    }) => Promise<
        | ConversationAcceptedOutputFragment
        | { accepted_output: ConversationAcceptedOutputFragment; conversation?: ConversationDocument }
        | undefined
    >;
    on_canonical_request_prepared?: (prepared: ConversationPreparedRequest) => Promise<void>;
    labels?: Record<string, string>;
    stripImagesAfterTurns?: number;
    stripTextMaxTokens?: number;
    stripHeartbeatsAfterTurns?: number;
}

/** Caller-facing, already materialized canonical context. */
export interface CanonicalExecutionContextInputOptions extends CanonicalExecutionTransportOptions {
    conversation: ConversationDocument;
    conversation_runtime: ConversationRuntimeContext;
}

/** Validated and owned canonical context used by provider adapters. */
export interface CanonicalExecutionContextOptions extends CanonicalExecutionTransportOptions {
    conversation: ConversationDocument;
    conversation_runtime: ResolvedConversationRuntimeContext;
}

function cloneOptionValue<T>(value: T | undefined): T | undefined {
    return value === undefined ? undefined : structuredClone(value);
}

/**
 * Validate and own one canonical request snapshot at the public Driver boundary.
 *
 * An absent document starts an empty canonical conversation only when the caller supplied its
 * durable id. Opaque native histories are intentionally rejected; their one-time conversion belongs
 * to the named provider import APIs rather than the current execution path.
 */
export function resolveCanonicalExecutionOptions(options: ExecutionOptions): CanonicalExecutionOptions {
    const suppliedRuntime = ConversationRuntimeContextSchema.parse(options.conversation_runtime);
    const conversation =
        options.conversation === undefined || options.conversation === null
            ? (() => {
                  if (suppliedRuntime.conversation_id === undefined) {
                      throw new TypeError(
                          'Canonical execution without a document requires conversation_runtime.conversation_id',
                      );
                  }
                  return createConversationDocument({
                      id: suppliedRuntime.conversation_id,
                      created_at: suppliedRuntime.recorded_at,
                  });
              })()
            : parseConversationDocument(options.conversation);
    if (suppliedRuntime.conversation_id !== undefined && suppliedRuntime.conversation_id !== conversation.id) {
        throw new TypeError('conversation_runtime.conversation_id does not match the canonical document');
    }
    const runtime = ResolvedConversationRuntimeContextSchema.parse({
        ...suppliedRuntime,
        conversation_id: conversation.id,
        purpose: suppliedRuntime.purpose ?? 'conversation',
    });
    const {
        conversation: _conversation,
        conversation_runtime: _conversationRuntime,
        httpTimeout,
        labels,
        model_options,
        result_schema,
        tools,
        ...rest
    } = options;
    return {
        ...rest,
        conversation,
        conversation_runtime: runtime,
        ...(httpTimeout === undefined ? {} : { httpTimeout: cloneOptionValue(httpTimeout) }),
        ...(labels === undefined ? {} : { labels: cloneOptionValue(labels) }),
        ...(model_options === undefined ? {} : { model_options: cloneOptionValue(model_options) }),
        ...(result_schema === undefined ? {} : { result_schema: cloneOptionValue(result_schema) }),
        ...(tools === undefined ? {} : { tools: cloneOptionValue(tools) }),
    };
}

/**
 * Validate and own an already materialized canonical context without accepting authoring inputs.
 * The returned snapshot contains no legacy tool catalog or custom prompt formatter.
 */
export function resolveCanonicalExecutionContextOptions(
    options: CanonicalExecutionContextInputOptions,
): CanonicalExecutionContextOptions {
    const unsafe = options as CanonicalExecutionContextInputOptions & {
        format?: unknown;
        output_modality?: unknown;
        tools?: unknown;
    };
    if (unsafe.format !== undefined)
        throw new TypeError('Canonical context execution does not accept a prompt formatter');
    if (unsafe.tools !== undefined) {
        throw new TypeError('Canonical context execution uses the document active tool definitions');
    }
    if (unsafe.output_modality !== undefined) {
        throw new TypeError('Canonical context execution does not accept legacy output modality policy');
    }
    const conversation = parseConversationDocument(options.conversation);
    const suppliedRuntime = ConversationRuntimeContextSchema.parse(options.conversation_runtime);
    if (suppliedRuntime.conversation_id !== undefined && suppliedRuntime.conversation_id !== conversation.id) {
        throw new TypeError('conversation_runtime.conversation_id does not match the canonical document');
    }
    const conversationRuntime = ResolvedConversationRuntimeContextSchema.parse({
        ...suppliedRuntime,
        conversation_id: conversation.id,
        purpose: suppliedRuntime.purpose ?? 'conversation',
    });
    const {
        conversation: _conversation,
        conversation_runtime: _conversationRuntime,
        httpTimeout,
        labels,
        model_options,
        result_schema,
        ...rest
    } = options;
    return {
        ...rest,
        conversation,
        conversation_runtime: conversationRuntime,
        ...(httpTimeout === undefined ? {} : { httpTimeout: cloneOptionValue(httpTimeout) }),
        ...(labels === undefined ? {} : { labels: cloneOptionValue(labels) }),
        ...(model_options === undefined ? {} : { model_options: cloneOptionValue(model_options) }),
        ...(result_schema === undefined ? {} : { result_schema: cloneOptionValue(result_schema) }),
    };
}

export function markCanonicalAcceptedRecovery<T extends object>(value: T): T & { [CANONICAL_ACCEPTED_RECOVERY]: true } {
    Object.defineProperty(value, CANONICAL_ACCEPTED_RECOVERY, {
        configurable: false,
        enumerable: true,
        value: true,
        writable: false,
    });
    return value as T & { [CANONICAL_ACCEPTED_RECOVERY]: true };
}

export function isCanonicalAcceptedRecovery(value: unknown): boolean {
    return typeof value === 'object' && value !== null && CANONICAL_ACCEPTED_RECOVERY in value;
}

/**
 * Internal success control flow for a host-verified accepted output found after request preparation.
 *
 * Output-only retention is deliberately not a CanonicalExecutionResponse: it has no resumable history or
 * provider-native replay. Hosts catch this signal at their execution boundary and project the accepted fragment
 * without treating it as a provider failure.
 */
export class CanonicalAcceptedOutputRecovered extends Error {
    readonly accepted_output: ConversationAcceptedOutputFragment;
    readonly conversation?: ConversationDocument;

    constructor(input: {
        accepted_output: ConversationAcceptedOutputFragment;
        conversation?: ConversationDocument;
    }) {
        super('A durably accepted canonical output was recovered before provider transport');
        this.name = 'CanonicalAcceptedOutputRecovered';
        if (input.conversation === undefined) {
            this.accepted_output = parseAcceptedOutputFragment(input.accepted_output);
            return;
        }
        const response = createCanonicalExecutionResponse(
            input.conversation,
            input.accepted_output.receipt.id,
            {},
            input.accepted_output,
        );
        this.accepted_output = response.accepted_output;
        this.conversation = response.conversation;
    }

    static is(error: unknown): error is CanonicalAcceptedOutputRecovered {
        return error instanceof CanonicalAcceptedOutputRecovered;
    }
}

/** Live text is only a preview; completion carries the authoritative canonical records. */
export interface CanonicalExecutionStream extends AsyncIterable<string> {
    completion: CanonicalExecutionResponse | undefined;
    /** Host-verified accepted output recovered while a finite fallback attempted to start transport. */
    readonly accepted_recovery?: CanonicalAcceptedOutputRecovered;
    cancel(): Promise<void>;
}

function sameIds(first: readonly string[] | undefined, second: readonly string[] | undefined): boolean {
    const left = first ?? [];
    const right = second ?? [];
    return left.length === right.length && left.every((id, index) => id === right[index]);
}

function recoveredAcceptedOutput(
    conversation: ConversationDocument,
    responseOperationId: string,
    input: ConversationAcceptedOutputFragment,
): ConversationAcceptedOutputFragment {
    const fragment = parseAcceptedOutputFragment(input);
    const receipt = Object.hasOwn(conversation.operation_receipts, responseOperationId)
        ? conversation.operation_receipts[responseOperationId]
        : undefined;
    const generation = Object.hasOwn(conversation.generations, fragment.generation.id)
        ? conversation.generations[fragment.generation.id]
        : undefined;
    const turn = conversation.turns.find((candidate) => candidate.id === fragment.turn.id);
    if (
        fragment.source.conversation_id !== conversation.id ||
        fragment.source.revision > conversation.revision ||
        fragment.receipt.id !== responseOperationId ||
        receipt === undefined ||
        receipt.id !== fragment.receipt.id ||
        receipt.conversation_id !== fragment.receipt.conversation_id ||
        receipt.base_revision !== fragment.receipt.base_revision ||
        receipt.result_revision !== fragment.receipt.result_revision ||
        receipt.recorded_at !== fragment.receipt.recorded_at ||
        !sameIds(receipt.accepted_turn_ids, fragment.receipt.accepted_turn_ids) ||
        !sameIds(receipt.accepted_generation_ids, fragment.receipt.accepted_generation_ids) ||
        !sameIds(receipt.accepted_asset_ids, fragment.receipt.accepted_asset_ids) ||
        generation?.record_source !== 'executed' ||
        generation.id !== fragment.generation.id ||
        generation.request_id !== fragment.generation.request_id ||
        generation.attempt_id !== fragment.generation.attempt_id ||
        generation.source.conversation_id !== fragment.generation.source.conversation_id ||
        generation.source.revision !== fragment.generation.source.revision ||
        generation.provider !== fragment.generation.provider ||
        generation.protocol !== fragment.generation.protocol ||
        generation.requested_model !== fragment.generation.requested_model ||
        generation.adapter_version !== fragment.generation.adapter_version ||
        turn?.kind !== 'agent' ||
        !('generation_id' in turn) ||
        turn.generation_id !== fragment.generation.id ||
        turn.id !== fragment.turn.id
    ) {
        throw new Error('Recovered canonical output does not match the retained response receipt');
    }
    return fragment;
}

export function canonicalExecutionPreview(
    response: CanonicalExecutionResponse,
    includeReasoning: boolean,
    omittedCommittedBlockIds?: ReadonlySet<string>,
): string {
    return response.accepted_output.turn.blocks
        .flatMap((block): string[] => {
            if (omittedCommittedBlockIds?.has(block.id)) return [];
            switch (block.type) {
                case 'text':
                    return [block.text];
                case 'reasoning':
                    return includeReasoning ? [block.text] : [];
                case 'json':
                    return [JSON.stringify(block.value)];
                case 'image':
                    return ['[Image]'];
                case 'audio':
                    return ['[Audio]'];
                case 'video':
                    return ['[Video]'];
                case 'document':
                case 'tool_call':
                    return [];
                default: {
                    const _exhaustive: never = block;
                    return [String(_exhaustive)];
                }
            }
        })
        .join('');
}

/** Canonical sync fallback for model paths whose transport cannot stream. */
export class FallbackCanonicalExecutionStream implements CanonicalExecutionStream {
    completion: CanonicalExecutionResponse | undefined;
    accepted_recovery: CanonicalAcceptedOutputRecovered | undefined;
    private readonly abortController = new AbortController();
    private iteratorCreated = false;

    constructor(
        private readonly execute: (signal: AbortSignal) => Promise<CanonicalExecutionResponse>,
        private readonly includeReasoning = false,
    ) {}

    async cancel(): Promise<void> {
        this.abortController.abort();
    }

    [Symbol.asyncIterator](): AsyncIterator<string> {
        if (this.iteratorCreated) throw new Error('Canonical execution stream can only be consumed once');
        this.iteratorCreated = true;
        const self = this;
        return (async function* () {
            if (self.abortController.signal.aborted) return;
            let completion: CanonicalExecutionResponse;
            try {
                completion = await self.execute(self.abortController.signal);
            } catch (error) {
                if (!CanonicalAcceptedOutputRecovered.is(error)) throw error;
                self.accepted_recovery = error;
                return;
            }
            if (self.abortController.signal.aborted) return;
            self.completion = completion;
            const preview = canonicalExecutionPreview(completion, self.includeReasoning);
            if (preview.length > 0) yield preview;
        })();
    }
}

export function createCanonicalExecutionResponse(
    documentInput: ConversationDocument,
    responseOperationId: string,
    metadata: Pick<
        CanonicalExecutionResponse,
        'execution_time' | 'chunks' | 'service_tier' | 'prompt_cache_diagnostic' | 'original_response'
    > = {},
    recoveredOutput?: ConversationAcceptedOutputFragment,
): CanonicalExecutionResponse {
    const conversation = parseConversationDocument(documentInput);
    return {
        conversation,
        accepted_output:
            recoveredOutput === undefined
                ? createAcceptedOutputFragment(conversation, responseOperationId)
                : recoveredAcceptedOutput(conversation, responseOperationId, recoveredOutput),
        ...(metadata.execution_time === undefined ? {} : { execution_time: metadata.execution_time }),
        ...(metadata.chunks === undefined ? {} : { chunks: metadata.chunks }),
        ...(metadata.service_tier === undefined ? {} : { service_tier: metadata.service_tier }),
        ...(metadata.prompt_cache_diagnostic === undefined
            ? {}
            : { prompt_cache_diagnostic: metadata.prompt_cache_diagnostic }),
        ...(metadata.original_response === undefined ? {} : { original_response: metadata.original_response }),
    };
}

function legacyAssetValue(asset: Asset): string {
    if (asset.storage.type === 'inline_base64') return `data:${asset.mime_type};base64,${asset.storage.data}`;
    if (asset.storage.type === 'external') {
        const uri = asset.storage.locator.uri;
        if (typeof uri === 'string' && uri.length > 0) return uri;
        const url = asset.storage.locator.url;
        if (typeof url === 'string' && url.length > 0) return url;
    }
    throw new Error(`Canonical asset ${asset.id} has no legacy media value`);
}

function legacyAudioResult(asset: Asset): AudioResult {
    return {
        type: 'audio',
        value: legacyAssetValue(asset),
        mime_type: asset.mime_type,
        ...(asset.media?.container === undefined ? {} : { container: asset.media.container }),
        ...(asset.media?.codec === undefined ? {} : { codec: asset.media.codec }),
        ...(asset.media?.sample_rate === undefined ? {} : { sample_rate: asset.media.sample_rate }),
        ...(asset.media?.channels === undefined ? {} : { channels: asset.media.channels }),
        ...(asset.media?.sample_encoding === undefined ? {} : { sample_encoding: asset.media.sample_encoding }),
        ...(asset.media?.byte_order === undefined ? {} : { byte_order: asset.media.byte_order }),
    };
}

function legacyResult(
    block: ConversationOutputBlock,
    assets: ConversationAcceptedOutputFragment['assets'],
    includeReasoning: boolean,
): CompletionResult | undefined {
    switch (block.type) {
        case 'text':
            return { type: 'text', value: block.text };
        case 'json':
            return { type: 'json', value: block.value } satisfies JsonResult;
        case 'reasoning':
            return includeReasoning ? { type: 'thoughts', value: block.text } : undefined;
        case 'image': {
            const asset = Object.hasOwn(assets, block.asset_id) ? assets[block.asset_id] : undefined;
            if (asset === undefined) throw new Error(`Canonical output is missing image asset ${block.asset_id}`);
            return { type: 'image', value: legacyAssetValue(asset) };
        }
        case 'audio': {
            const asset = Object.hasOwn(assets, block.asset_id) ? assets[block.asset_id] : undefined;
            if (asset === undefined) throw new Error(`Canonical output is missing audio asset ${block.asset_id}`);
            return legacyAudioResult(asset);
        }
        case 'video': {
            const asset = Object.hasOwn(assets, block.asset_id) ? assets[block.asset_id] : undefined;
            if (asset === undefined) throw new Error(`Canonical output is missing video asset ${block.asset_id}`);
            return { type: 'video', value: legacyAssetValue(asset) };
        }
        case 'document':
        case 'tool_call':
            return undefined;
    }
}

function ownRecord(value: unknown): Record<string, unknown> | undefined {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : undefined;
}

function nonnegativeSafeInteger(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function bedrockOneHourCacheWriteTokens(response: CanonicalExecutionResponse): number | undefined {
    const generationId = response.accepted_output.generation.id;
    const generation = Object.hasOwn(response.conversation.generations, generationId)
        ? response.conversation.generations[generationId]
        : undefined;
    if (generation?.protocol !== 'aws.bedrock.converse') return undefined;
    const reported = generation.usage?.reported_usage?.find(
        (candidate) => candidate.source === 'provider' && candidate.protocol === 'aws.bedrock.converse',
    );
    const cacheDetails = ownRecord(reported?.payload)?.cacheDetails;
    if (!Array.isArray(cacheDetails)) return undefined;
    let total = 0;
    let found = false;
    for (const detail of cacheDetails) {
        const record = ownRecord(detail);
        if (record?.ttl !== '1h') continue;
        const tokens = nonnegativeSafeInteger(record.inputTokens);
        if (tokens === undefined || total > Number.MAX_SAFE_INTEGER - tokens) return undefined;
        total += tokens;
        found = true;
    }
    return found ? total : undefined;
}

function legacyUsage(
    fragment: ConversationAcceptedOutputFragment,
    response?: CanonicalExecutionResponse,
): ExecutionTokenUsage | undefined {
    const usage = fragment.generation.usage;
    if (usage === undefined) return undefined;
    const providerCostUsd = usage.cost?.currency === 'USD' ? Number(usage.cost.amount) : undefined;
    const promptCacheWrite1h = response === undefined ? undefined : bedrockOneHourCacheWriteTokens(response);
    const isOpenAIImage = fragment.generation.protocol === 'openai.images.generate';
    const resultImage = isOpenAIImage ? usage.output_tokens : undefined;
    const promptCached = isOpenAIImage ? undefined : usage.cache_read_tokens;
    const promptCacheWrite = isOpenAIImage ? undefined : usage.cache_write_tokens;
    return {
        prompt: usage.input_tokens,
        prompt_new: usage.input_new_tokens,
        result: usage.output_tokens,
        ...(resultImage === undefined ? {} : { result_image: resultImage }),
        total: usage.total_tokens,
        ...(promptCached === undefined ? {} : { prompt_cached: promptCached }),
        ...(promptCacheWrite === undefined ? {} : { prompt_cache_write: promptCacheWrite }),
        ...(promptCacheWrite1h === undefined ? {} : { prompt_cache_write_1h: promptCacheWrite1h }),
        ...(providerCostUsd === undefined || !Number.isFinite(providerCostUsd)
            ? {}
            : { provider_cost_usd: providerCostUsd }),
    };
}

function legacyFinishReason(fragment: ConversationAcceptedOutputFragment, hasTools: boolean): string | undefined {
    switch (fragment.generation.finish_reason) {
        case 'end_turn':
        case 'stop':
        case 'tool_use':
        case undefined:
            return hasTools
                ? 'tool_use'
                : fragment.generation.finish_reason === 'end_turn'
                  ? 'stop'
                  : fragment.generation.finish_reason;
        case 'max_tokens':
        case 'model_context_window_exceeded':
            return 'length';
        default:
            return fragment.generation.finish_reason;
    }
}

/** Scalar accounting for canonical hosts; semantic blocks are never projected to Completion. */
export function canonicalExecutionAccounting(
    response: CanonicalExecutionResponse,
): Pick<Completion, 'token_usage' | 'finish_reason'> {
    const fragment = response.accepted_output;
    const tokenUsage = legacyUsage(fragment, response);
    const hasTools = fragment.turn.blocks.some(
        (block) => block.type === 'tool_call' && block.executor === 'application',
    );
    return {
        ...(tokenUsage === undefined ? {} : { token_usage: tokenUsage }),
        finish_reason: legacyFinishReason(fragment, hasTools),
    };
}

function ownString(value: unknown, key: string): string | undefined {
    const record = ownRecord(value);
    return typeof record?.[key] === 'string' ? record[key] : undefined;
}

/** Read an accepted canonical failure without relying on the lossy output-fragment metadata projection. */
export function canonicalExecutionFailure(response: CanonicalExecutionResponse): ResultValidationError | undefined {
    const generationId = response.accepted_output.generation.id;
    const generation = Object.hasOwn(response.conversation.generations, generationId)
        ? response.conversation.generations[generationId]
        : undefined;
    const structuredOutput = ownRecord(generation?.metadata?.structured_output);
    if (structuredOutput?.status === 'invalid') {
        const code = ownString(structuredOutput, 'code');
        return {
            code: code === 'json_error' ? 'json_error' : 'validation_error',
            message: ownString(structuredOutput, 'message') ?? 'Canonical structured output validation failed',
        };
    }
    if (response.accepted_output.generation.status === 'failed' || response.accepted_output.turn.status === 'failed') {
        return {
            code: 'validation_error',
            message: generation?.finish_reason
                ? `Canonical generation failed: ${generation.finish_reason}`
                : 'Canonical generation failed',
        };
    }
    return undefined;
}

function acceptedOutputFailure(fragment: ConversationAcceptedOutputFragment): ResultValidationError | undefined {
    if (fragment.generation.status !== 'failed' && fragment.turn.status !== 'failed') return undefined;
    return {
        code: 'validation_error',
        message: fragment.generation.finish_reason
            ? `Canonical generation failed: ${fragment.generation.finish_reason}`
            : 'Canonical generation failed; detailed metadata was not retained',
    };
}

/**
 * Project a verified output-only fragment to the old Completion API without inventing conversation history.
 * The returned value intentionally omits `conversation`, native replay, and metadata-only diagnostics.
 */
function projectAcceptedOutput(
    fragment: ConversationAcceptedOutputFragment,
    options: { include_reasoning?: boolean } = {},
): Completion {
    const providerFinishReason = legacyFinishReason(fragment, false);
    const toolUse = fragment.turn.blocks.flatMap((block): ToolUse<unknown>[] => {
        if (block.type !== 'tool_call' || block.executor !== 'application') return [];
        if (block.arguments.type === 'invalid') {
            if (providerFinishReason === 'length') return [];
            const tool = {
                id: block.call_id,
                tool_name: block.tool_name,
                tool_input: block.arguments.raw,
            };
            let parseError: unknown;
            try {
                JSON.parse(block.arguments.raw);
                parseError = new TypeError('Canonical tool arguments are not an executable JSON object');
            } catch (error: unknown) {
                parseError = error;
            }
            throw new MalformedStreamingToolArgumentsError(
                tool,
                providerFinishReason,
                { provider: fragment.generation.provider, model: fragment.generation.requested_model },
                parseError,
            );
        }
        if (block.arguments.type === 'externalized_json') {
            throw new Error(
                `Canonical tool call ${block.call_id} requires lossless argument hydration before legacy projection`,
            );
        }
        return [
            {
                id: block.call_id,
                tool_name: block.tool_name,
                tool_input: toolArgumentsForModel(block.arguments),
            },
        ];
    });
    const result = fragment.turn.blocks.flatMap((block) => {
        const projected = legacyResult(block, fragment.assets, options.include_reasoning === true);
        return projected === undefined ? [] : [projected];
    });
    const tokenUsage = legacyUsage(fragment);
    const error = acceptedOutputFailure(fragment);
    return {
        result,
        ...(toolUse.length === 0 ? {} : { tool_use: toolUse }),
        ...(tokenUsage === undefined ? {} : { token_usage: tokenUsage }),
        finish_reason: legacyFinishReason(fragment, toolUse.length > 0),
        ...(error === undefined ? {} : { error }),
    };
}

export function legacyCompletionFromAcceptedOutput(
    fragmentInput: ConversationAcceptedOutputFragment,
    options: { include_reasoning?: boolean } = {},
): Completion {
    return projectAcceptedOutput(parseAcceptedOutputFragment(fragmentInput), options);
}

/** Explicit compatibility projection used only by the old Completion/ToolUse API boundary. */
export function legacyCompletionFromCanonicalExecution(
    response: CanonicalExecutionResponse,
    options: { include_reasoning?: boolean } = {},
): Completion {
    const projected = projectAcceptedOutput(response.accepted_output, options);
    const { token_usage: _projectedTokenUsage, error: _projectedError, ...base } = projected;
    const tokenUsage = legacyUsage(response.accepted_output, response);
    const error = canonicalExecutionFailure(response);
    const completion: Completion = {
        ...base,
        ...(tokenUsage === undefined ? {} : { token_usage: tokenUsage }),
        ...(error === undefined ? {} : { error }),
        ...(response.execution_time === undefined ? {} : { execution_time: response.execution_time }),
        ...(response.chunks === undefined ? {} : { chunks: response.chunks }),
        ...(response.service_tier === undefined ? {} : { service_tier: response.service_tier }),
        ...(response.prompt_cache_diagnostic === undefined
            ? {}
            : { prompt_cache_diagnostic: response.prompt_cache_diagnostic }),
        conversation: response.conversation,
        ...(response.original_response === undefined ? {} : { original_response: response.original_response }),
    };
    return isCanonicalAcceptedRecovery(response) ? markCanonicalAcceptedRecovery(completion) : completion;
}
