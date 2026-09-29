import type {
    AudioResult,
    Completion,
    CompletionResult,
    ExecutionTokenUsage,
    JsonResult,
    PromptCacheDiagnostic,
    ResultValidationError,
    ToolUse,
} from '@llumiverse/common';
import {
    type Asset,
    type ConversationAcceptedOutputFragment,
    type ConversationDocument,
    type ConversationOutputBlock,
    createAcceptedOutputFragment,
    parseAcceptedOutputFragment,
    parseConversationDocument,
    toolArgumentsForModel,
} from '@llumiverse/conversation';
import { MalformedStreamingToolArgumentsError } from './stream-errors.js';

/** Direct canonical provider result. The complete document is authoritative; accepted_output is its safe projection. */
export interface CanonicalExecutionResponse {
    conversation: ConversationDocument;
    accepted_output: ConversationAcceptedOutputFragment;
    execution_time?: number;
    chunks?: number;
    service_tier?: string;
    prompt_cache_diagnostic?: PromptCacheDiagnostic;
    original_response?: unknown;
}

/** Live text is only a preview; completion carries the authoritative canonical records. */
export interface CanonicalExecutionStream extends AsyncIterable<string> {
    completion: CanonicalExecutionResponse | undefined;
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

function canonicalPreview(response: CanonicalExecutionResponse, includeReasoning: boolean): string {
    return response.accepted_output.turn.blocks
        .flatMap((block): string[] => {
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
            const completion = await self.execute(self.abortController.signal);
            if (self.abortController.signal.aborted) return;
            self.completion = completion;
            const preview = canonicalPreview(completion, self.includeReasoning);
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

function legacyUsage(response: CanonicalExecutionResponse): ExecutionTokenUsage | undefined {
    const fragment = response.accepted_output;
    const usage = fragment.generation.usage;
    if (usage === undefined) return undefined;
    const generation = Object.hasOwn(response.conversation.generations, fragment.generation.id)
        ? response.conversation.generations[fragment.generation.id]
        : undefined;
    const providerCostUsd =
        generation?.usage?.cost?.currency === 'USD' ? Number(generation.usage.cost.amount) : undefined;
    const promptCacheWrite1h = bedrockOneHourCacheWriteTokens(response);
    const isOpenAIImage = generation?.protocol === 'openai.images.generate';
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

/** Explicit compatibility projection used only by the old Completion/ToolUse API boundary. */
export function legacyCompletionFromCanonicalExecution(
    response: CanonicalExecutionResponse,
    options: { include_reasoning?: boolean } = {},
): Completion {
    const fragment = response.accepted_output;
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
    const tokenUsage = legacyUsage(response);
    const error = canonicalExecutionFailure(response);
    return {
        result,
        ...(toolUse.length === 0 ? {} : { tool_use: toolUse }),
        ...(tokenUsage === undefined ? {} : { token_usage: tokenUsage }),
        finish_reason: legacyFinishReason(fragment, toolUse.length > 0),
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
}
