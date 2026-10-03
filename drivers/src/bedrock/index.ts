import {
    Bedrock,
    CreateModelCustomizationJobCommand,
    type FoundationModelSummary,
    GetModelCustomizationJobCommand,
    type GetModelCustomizationJobCommandOutput,
    ModelCustomizationJobStatus,
    StopModelCustomizationJobCommand,
} from '@aws-sdk/client-bedrock';
import {
    BedrockRuntime,
    type ContentBlock,
    type ConverseRequest,
    type ConverseResponse,
    type ConverseStreamCommandOutput,
    type ConverseStreamOutput,
    type InferenceConfiguration,
    type InvokeModelCommandOutput,
    type Message,
    type ServiceTierType,
    type TokenUsage,
    type Tool,
    type ToolResultContentBlock,
} from '@aws-sdk/client-bedrock-runtime';
import { S3Client } from '@aws-sdk/client-s3';
import {
    type ToolDefinition as CanonicalToolDefinition,
    CONVERSATION_STREAM_MAX_TOTAL_BYTES,
    canonicalJsonContentString,
    createStructuredOutputTransformationProof,
    type DecodedConversationResponse,
    isConversationDocumentFormat,
    type JsonValue,
    type NativeStreamPosition,
    toolArgumentsForModel,
} from '@llumiverse/conversation';
import {
    type AIModel,
    type BedrockClaudeOptions,
    type BedrockGptOssOptions,
    type BedrockPalmyraOptions,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    createCanonicalExecutionResponse,
    type DataSource,
    type DriverCompletionStream,
    deserializeBinaryFromStorage,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    FallbackCanonicalExecutionEventStream,
    getConversationMeta,
    getMaxTokensLimitBedrock,
    type HttpTimeoutOptions,
    incrementConversationTurn,
    isEmbeddingModel,
    type JSONObject,
    type ToolDefinition as LegacyToolDefinition,
    LlumiverseError,
    type LlumiverseErrorContext,
    legacyCompletionFromCanonicalExecution,
    type ModelOptions,
    markCanonicalAcceptedRecovery,
    type NovaCanvasOptions,
    type PromptSegment,
    Providers,
    parseClaudeVersion,
    type StatelessExecutionOptions,
    stripBinaryFromConversation,
    stripHeartbeatsFromConversation,
    type TextFallbackOptions,
    type ToolUse,
    type TrainingJob,
    TrainingJobStatus,
    type TrainingOptions,
    truncateLargeTextInConversation,
} from '@llumiverse/core';
import { transformAsyncIterator } from '@llumiverse/core/async';
import { AbstractDriver } from '@llumiverse/core/driver';
import { formatNovaPrompt, type NovaMessagesPrompt } from '@llumiverse/core/formatters';
import { mergeDriverHttpTimeoutOptions, resolveDriverHttpTimeouts } from '@llumiverse/core/http-agent';
import { LRUCache } from 'lru-cache';
import { canonicalNativeExecutionEventStream } from '../conversation/canonical-execution-event-stream.js';
import {
    assertAcceptedCanonicalRequest,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
} from '../conversation/canonical-runtime.js';
import {
    normalizeDecodedStructuredOutputForSchema,
    rejectDecodedStructuredOutput,
} from '../conversation/structured-output.js';
import type { BedrockDriverOptions } from '../driver-options.js';
import { logClaudeTruncation } from '../shared/claude-stop-reason.js';
import { resolveClaudeThinking } from '../shared/claude-thinking.js';
import { truncateBinaryForDebug, uint8ArrayToBase64ForDebug } from '../shared/debug-prompt.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';
import { createToolChoiceConfigurationError } from '../shared/tool-choice-error.js';
import {
    appendBedrockConverseCanonicalResponseWithProcessing,
    bedrockConverseJsonValue,
    decodeBedrockConverseCanonicalResponse,
    finalizeBedrockConversePreparedRequest,
    type PreparedBedrockConverseConversation,
    prepareBedrockConverseCanonicalContext,
    prepareBedrockConverseCanonicalState,
} from './bedrock-converse-conversation-adapter.js';
import {
    converseConcatMessages,
    converseJSONprefill,
    converseSystemToMessages,
    formatConversePrompt,
    projectConverseContextResultSchema,
    relocateConverseToolImages,
    shouldIncludeSchemaInConversePrompt,
    supportsConverseOutputConfig,
} from './converse.js';
import { generateBedrockEmbeddings } from './embeddings.js';
import {
    executeNovaCanvasCanonical,
    type NovaCanvasPayload,
    validateNovaCanvasCanonicalInput,
} from './nova-image-canonical.js';
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from './nova-image-payload.js';
import { forceUploadFile } from './s3.js';
import {
    formatTwelvelabsPegasusPrompt,
    type TwelvelabsPegasusCanonicalPrompt,
    type TwelvelabsPegasusRequest,
    validateTwelvelabsPegasusCanonicalInput,
} from './twelvelabs.js';
import {
    executeTwelvelabsPegasusCanonical,
    executeTwelvelabsPegasusCanonicalContext,
    streamTwelvelabsPegasusCanonicalContextEvents,
    streamTwelvelabsPegasusCanonicalEvents,
    type TwelvelabsPegasusInvokeRequest,
    TwelvelabsPegasusNativeStreamAccumulator,
    type TwelvelabsPegasusStreamEvent,
    type TwelvelabsPegasusTransport,
} from './twelvelabs-canonical.js';

export type { BedrockDriverOptions } from '../driver-options.js';
export type {
    BedrockConverseConversation,
    BedrockConverseFamilyCapabilities,
    CompiledBedrockConversation,
    ImportBedrockConverseConversationOptions,
    PreparedBedrockConverseConversation,
} from './bedrock-converse-conversation-adapter.js';
export {
    appendBedrockConverseCanonicalResponse,
    bedrockConverseFamilyCapabilities,
    bedrockConverseGenerationUsage,
    compileBedrockConverseConversation,
    decodeBedrockConverseCanonicalResponse,
    exportLegacyBedrockConverseConversation,
    finalizeBedrockConversePreparedRequest,
    importBedrockConverseConversation,
    isBedrockConverseHistory,
    prepareBedrockConverseCanonicalState,
} from './bedrock-converse-conversation-adapter.js';

const supportStreamingCache = new LRUCache<string, boolean>({ max: 4096 });
const TWELVELABS_PEGASUS_CANONICAL_FORMAT = Symbol('twelvelabs.pegasus.canonical_format');
type BedrockToolDefinition =
    | Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'>
    | LegacyToolDefinition;

const TWELVELABS_PEGASUS_MAX_INLINE_VIDEO_BYTES = 25 * 1024 * 1024;
type PegasusCanonicalExecutionOptions = CanonicalExecutionInputOptions & {
    [TWELVELABS_PEGASUS_CANONICAL_FORMAT]?: true;
};

type AwsSdkError = {
    name?: string;
    message?: string;
    $metadata?: {
        httpStatusCode?: number;
        requestId?: string;
    };
    $fault?: string;
};
type ReasoningBlockStart = {
    reasoningContent?: {
        redactedContent?: Uint8Array;
    };
};
type BedrockRuntimeExecutorScope = {
    executor: BedrockRuntime;
    close(): void;
};

type BedrockServiceTierOptions = {
    service_tier?: string;
};

function getBedrockServiceTier(modelOptions?: ModelOptions): ServiceTierType | undefined {
    // The public option deliberately accepts future provider values that may predate the installed SDK union.
    return (modelOptions as BedrockServiceTierOptions | undefined)?.service_tier as ServiceTierType | undefined;
}

enum BedrockModelType {
    FoundationModel = 'foundation-model',
    InferenceProfile = 'inference-profile',
    CustomModel = 'custom-model',
    Unknown = 'unknown',
}

function bedrockFoundationModelLookupId(model: string): string {
    const emptyRegionArnMarker = ':bedrock:::foundation-model/';
    const markerIndex = model.indexOf(emptyRegionArnMarker);
    return markerIndex < 0 ? model : model.slice(markerIndex + emptyRegionArnMarker.length);
}

/** Of the cache-write tokens, those written with a one-hour lifetime (the rest used the five-minute default). */
function oneHourCacheWriteTokens(usage: TokenUsage | undefined): number | undefined {
    const tokens = usage?.cacheDetails
        ?.filter((detail) => detail.ttl === '1h')
        .reduce((sum, detail) => sum + (detail.inputTokens ?? 0), 0);
    return tokens || undefined;
}

/**
 * Converse usage as token usage. `inputTokens` already excludes cache reads and writes, so it is the new prompt
 * tokens; `prompt` is the total, cache reads and writes included, consistent with the Vertex Claude driver.
 */
function converseTokenUsage(usage: TokenUsage | undefined): ExecutionTokenUsage | undefined {
    if (!usage) return undefined;
    const rawInputTokens = usage.inputTokens;
    const inputNew = Number.isSafeInteger(rawInputTokens) && (rawInputTokens ?? -1) >= 0 ? rawInputTokens : undefined;
    const reportedCacheRead =
        Number.isSafeInteger(usage.cacheReadInputTokens) && (usage.cacheReadInputTokens ?? -1) >= 0
            ? usage.cacheReadInputTokens
            : undefined;
    const reportedCacheWrite =
        Number.isSafeInteger(usage.cacheWriteInputTokens) && (usage.cacheWriteInputTokens ?? -1) >= 0
            ? usage.cacheWriteInputTokens
            : undefined;
    const cacheRead = inputNew === undefined ? reportedCacheRead : (reportedCacheRead ?? 0);
    const cacheWrite = inputNew === undefined ? reportedCacheWrite : (reportedCacheWrite ?? 0);
    const input = inputNew === undefined ? undefined : inputNew + (cacheRead ?? 0) + (cacheWrite ?? 0);
    const oneHourWrite = oneHourCacheWriteTokens(usage);
    return {
        ...(inputNew === undefined ? {} : { prompt_new: inputNew }),
        ...(input === undefined ? {} : { prompt: input }),
        ...(usage.outputTokens === undefined ? {} : { result: usage.outputTokens }),
        ...(usage.totalTokens === undefined ? {} : { total: usage.totalTokens }),
        ...(cacheRead === undefined ? {} : { prompt_cached: cacheRead }),
        ...(cacheWrite === undefined ? {} : { prompt_cache_write: cacheWrite }),
        ...(oneHourWrite === undefined ? {} : { prompt_cache_write_1h: oneHourWrite }),
    };
}

function converseFinishReason(reason: string | undefined) {
    //Possible values:
    //end_turn | tool_use | max_tokens | stop_sequence | guardrail_intervened | content_filtered
    if (!reason) return undefined;
    switch (reason) {
        case 'end_turn':
            return 'stop';
        case 'max_tokens':
        case 'model_context_window_exceeded':
            return 'length';
        default:
            return reason;
    }
}

function recoveredBedrockStream(completion: Completion): DriverCompletionStream {
    const stream = (async function* (): AsyncIterable<CompletionChunkObject> {
        yield {
            result: completion.result,
            tool_use: completion.tool_use,
            token_usage: completion.token_usage,
            finish_reason: completion.finish_reason,
            service_tier: completion.service_tier,
        };
    })();
    return markCanonicalAcceptedRecovery(
        Object.assign(stream, {
            finalizeConversation: () => completion.conversation,
        }),
    );
}

export function excludesBedrockReasoningReplay(model: string): boolean {
    const modelId = model.toLowerCase().split('/').pop() ?? '';
    return /^(?:(?:us|eu|apac)\.)?deepseek\.r1-v1(?::\d+)?$/.test(modelId);
}

function appendBytes(left: Uint8Array | undefined, right: Uint8Array): Uint8Array {
    if (!left?.length) return new Uint8Array(right);
    const combined = new Uint8Array(left.length + right.length);
    combined.set(left);
    combined.set(right, left.length);
    return combined;
}

function collectBedrockNativeStreamBlock(blocks: Map<number, ContentBlock>, event: ConverseStreamOutput): void {
    const start = event.contentBlockStart;
    if (start?.start?.toolUse) {
        blocks.set(start.contentBlockIndex ?? -1, {
            toolUse: {
                toolUseId: start.start.toolUse.toolUseId,
                name: start.start.toolUse.name,
                input: '' as unknown as JSONObject,
                ...(start.start.toolUse.type === undefined ? {} : { type: start.start.toolUse.type }),
            },
        });
    } else if (start?.start) {
        const reasoningStart = start.start as unknown as {
            reasoningContent?: { redactedContent?: Uint8Array };
        };
        if (reasoningStart.reasoningContent?.redactedContent) {
            blocks.set(start.contentBlockIndex ?? -1, {
                reasoningContent: { redactedContent: new Uint8Array(reasoningStart.reasoningContent.redactedContent) },
            });
        }
    }

    const blockDelta = event.contentBlockDelta;
    const delta = blockDelta?.delta;
    if (!delta) return;
    const index = blockDelta.contentBlockIndex ?? -1;
    if (delta.text !== undefined) {
        const current = blocks.get(index);
        const existingText = current && 'text' in current ? (current.text ?? '') : '';
        blocks.set(index, { text: existingText + delta.text });
    } else if (delta.toolUse?.input !== undefined) {
        const current = blocks.get(index);
        if (current && 'toolUse' in current && current.toolUse) {
            current.toolUse.input =
                `${String(current.toolUse.input ?? '')}${delta.toolUse.input}` as unknown as JSONObject;
        }
    } else if (delta.reasoningContent) {
        const reasoning = delta.reasoningContent;
        const current = blocks.get(index);
        const existing =
            current && 'reasoningContent' in current
                ? (current.reasoningContent as {
                      reasoningText?: { text: string; signature?: string };
                      redactedContent?: Uint8Array;
                  })
                : undefined;
        if (reasoning.redactedContent) {
            blocks.set(index, {
                reasoningContent: {
                    redactedContent: appendBytes(existing?.redactedContent, reasoning.redactedContent),
                },
            });
        } else {
            const reasoningText = existing?.reasoningText ?? { text: '' };
            if (reasoning.text) reasoningText.text += reasoning.text;
            if (reasoning.signature) reasoningText.signature = (reasoningText.signature ?? '') + reasoning.signature;
            blocks.set(index, { reasoningContent: { reasoningText } });
        }
    }
}

function finalizeBedrockNativeBlockEntries(blocks: Map<number, ContentBlock>): Array<[number, ContentBlock]> {
    return [...blocks.entries()]
        .sort(([left], [right]) => left - right)
        .flatMap(([index, block]): Array<[number, ContentBlock]> => {
            if ('toolUse' in block && block.toolUse && typeof block.toolUse.input === 'string') {
                try {
                    return [
                        [
                            index,
                            { toolUse: { ...block.toolUse, input: JSON.parse(block.toolUse.input) as JSONObject } },
                        ],
                    ];
                } catch {
                    // Invalid streamed JSON is not a complete tool call. Do not put it in native
                    // replay, where it would require a tool_result the workflow cannot produce.
                    return [];
                }
            }
            return [[index, block]];
        });
}

function finalizeBedrockNativeBlocks(blocks: Map<number, ContentBlock>): ContentBlock[] {
    return finalizeBedrockNativeBlockEntries(blocks).map(([, block]) => block);
}

const BEDROCK_STREAM_EXCEPTION_KEYS = [
    'modelStreamErrorException',
    'internalServerException',
    'validationException',
    'throttlingException',
    'serviceUnavailableException',
] as const;

function assertBedrockConverseStreamEvent(event: ConverseStreamOutput, messageStopped: boolean): void {
    const exception = BEDROCK_STREAM_EXCEPTION_KEYS.find((key) => event[key] !== undefined);
    if (exception !== undefined) {
        throw new Error(`Bedrock Converse stream emitted ${exception}`);
    }
    if (event.$unknown !== undefined) {
        throw new Error('Bedrock Converse stream emitted an unsupported event');
    }
    if (
        messageStopped &&
        (event.messageStart !== undefined ||
            event.contentBlockStart !== undefined ||
            event.contentBlockDelta !== undefined ||
            event.contentBlockStop !== undefined ||
            event.messageStop !== undefined)
    ) {
        throw new Error('Bedrock Converse stream emitted content after its terminal message stop');
    }
}

function bedrockConverseStreamEventBytes(event: ConverseStreamOutput): number {
    const serialized = canonicalJsonContentString(bedrockConverseJsonValue(event));
    return new TextEncoder().encode(serialized).byteLength;
}

interface BedrockCanonicalDraft {
    draft_block_id: string;
    native_position: NativeStreamPosition;
    kind: 'text' | 'reasoning' | 'tool_call';
    text: string;
    tool_argument_fragments: string[];
}

function bedrockStreamPosition(index: number, nativeItemId?: string): NativeStreamPosition {
    return {
        protocol: 'aws.bedrock.converse',
        path: ['output', 'message', 'content', index],
        ...(nativeItemId === undefined ? {} : { native_item_id: nativeItemId }),
    };
}

function bedrockSemanticBlocks(decoded: DecodedConversationResponse, turnId: string) {
    const turn = decoded.turns.find((candidate) => candidate.id === turnId);
    if (turn?.kind !== 'agent') throw new Error('Bedrock Converse stream decode has no generated agent turn');
    return turn.blocks.filter((block) => block.type !== 'native_replay');
}

function bedrockEntryHasSemanticBlock(block: ContentBlock): boolean {
    if (block.reasoningContent?.redactedContent !== undefined) return false;
    return (
        block.text !== undefined ||
        block.toolUse !== undefined ||
        block.reasoningContent?.reasoningText !== undefined ||
        block.image !== undefined ||
        block.audio !== undefined ||
        block.video !== undefined ||
        block.document !== undefined
    );
}

function assertBedrockToolDraftArguments(
    draft: BedrockCanonicalDraft,
    block: ReturnType<typeof bedrockSemanticBlocks>[number],
): void {
    if (draft.kind !== 'tool_call' || block.type !== 'tool_call' || block.arguments.type === 'invalid') return;
    let streamed: JsonValue;
    try {
        streamed = bedrockConverseJsonValue(JSON.parse(draft.tool_argument_fragments.join('')));
    } catch {
        throw new Error('Bedrock Converse tool argument fragments do not form valid JSON');
    }
    if (canonicalJsonContentString(streamed) !== canonicalJsonContentString(toolArgumentsForModel(block.arguments))) {
        throw new Error('Bedrock Converse tool argument fragments differ from the terminal tool input');
    }
}

function withBedrockRuntimeScope<T>(iterable: AsyncIterable<T>, scope: BedrockRuntimeExecutorScope): AsyncIterable<T> {
    return {
        async *[Symbol.asyncIterator]() {
            try {
                yield* iterable;
            } finally {
                scope.close();
            }
        },
    };
}

export interface BedrockModelCapabilities {
    name: string;
    canStream: boolean;
}

//Used to get a max_token value when not specified in the model options. Claude requires it to be set.
function maxTokenFallbackClaude(option: StatelessExecutionOptions): number {
    const modelOptions = option.model_options as BedrockClaudeOptions | undefined;
    if (modelOptions && typeof modelOptions.max_tokens === 'number') {
        // Clamp stored/user-provided values to the model's output limit: configs
        // written for a larger-output model (or provider) otherwise pass through
        // verbatim and Bedrock rejects the whole request with a ValidationException.
        const limit = getMaxTokensLimitBedrock(option.model);
        return limit ? Math.min(modelOptions.max_tokens, limit) : modelOptions.max_tokens;
    } else {
        let maxSupportedTokens = getMaxTokensLimitBedrock(option.model) ?? 8192; // Should always return a number for claude, 8192 is to satisfy the TypeScript type checker;
        // Fallback to the default max tokens limit for the model
        if (option.model.includes('claude-3-7-sonnet') && (modelOptions?.thinking_budget_tokens ?? 0) < 48000) {
            maxSupportedTokens = 64000; // Claude 3.7 can go up to 128k with a beta header, but when no max tokens is specified, we default to 64k.
        }
        return maxSupportedTokens;
    }
}

export type BedrockPrompt = NovaMessagesPrompt | ConverseRequest | TwelvelabsPegasusRequest;

type BedrockSystemBlock = NonNullable<ConverseRequest['system']>[number];
type BedrockToolEntry = NonNullable<NonNullable<ConverseRequest['toolConfig']>['tools']>[number];

function formatBedrockBytes(bytes: Uint8Array | string | undefined): string | undefined {
    if (bytes instanceof Uint8Array) {
        return truncateBinaryForDebug(uint8ArrayToBase64ForDebug(bytes));
    }
    if (typeof bytes === 'string') {
        return truncateBinaryForDebug(bytes);
    }
    return bytes;
}

function formatBedrockBytesForDebug(bytes: Uint8Array | string | undefined): Uint8Array {
    // AWS SDK prompt types require Uint8Array here, but the debug prompt returned to
    // Studio must be JSON-safe. Keep the mismatch contained to this conversion.
    return formatBedrockBytes(bytes) as unknown as Uint8Array;
}

function formatBedrockContentBlockForDebug(block: ContentBlock): ContentBlock {
    if (block.image?.source?.bytes) {
        return {
            ...block,
            image: {
                ...block.image,
                source: {
                    ...block.image.source,
                    bytes: formatBedrockBytesForDebug(block.image.source.bytes),
                },
            },
        };
    }
    if (block.document?.source?.bytes) {
        return {
            ...block,
            document: {
                ...block.document,
                source: {
                    ...block.document.source,
                    bytes: formatBedrockBytesForDebug(block.document.source.bytes),
                },
            },
        };
    }
    if (block.video?.source?.bytes) {
        return {
            ...block,
            video: {
                ...block.video,
                source: {
                    ...block.video.source,
                    bytes: formatBedrockBytesForDebug(block.video.source.bytes),
                },
            },
        };
    }
    if (block.toolResult?.content) {
        return {
            ...block,
            toolResult: {
                ...block.toolResult,
                content: block.toolResult.content.map(formatBedrockToolResultContentBlockForDebug),
            },
        };
    }
    return block;
}

function formatBedrockToolResultContentBlockForDebug(block: ToolResultContentBlock): ToolResultContentBlock {
    if (block.image?.source?.bytes) {
        return {
            ...block,
            image: {
                ...block.image,
                source: {
                    ...block.image.source,
                    bytes: formatBedrockBytesForDebug(block.image.source.bytes),
                },
            },
        };
    }
    if (block.document?.source?.bytes) {
        return {
            ...block,
            document: {
                ...block.document,
                source: {
                    ...block.document.source,
                    bytes: formatBedrockBytesForDebug(block.document.source.bytes),
                },
            },
        };
    }
    if (block.video?.source?.bytes) {
        return {
            ...block,
            video: {
                ...block.video,
                source: {
                    ...block.video.source,
                    bytes: formatBedrockBytesForDebug(block.video.source.bytes),
                },
            },
        };
    }
    return block;
}

function formatConversePromptForDebug(prompt: ConverseRequest): ConverseRequest {
    return {
        ...prompt,
        messages: prompt.messages?.map((message) => ({
            ...message,
            content: message.content?.map(formatBedrockContentBlockForDebug),
        })),
    };
}

function formatNovaPromptForDebug(prompt: NovaMessagesPrompt): NovaMessagesPrompt {
    return {
        ...prompt,
        messages: prompt.messages.map((message) => ({
            ...message,
            content: message.content.map((part) => {
                if (!part.image?.source.bytes && !part.video?.source.bytes) {
                    return part;
                }
                return {
                    ...part,
                    image: part.image
                        ? {
                              ...part.image,
                              source: {
                                  ...part.image.source,
                                  bytes: truncateBinaryForDebug(part.image.source.bytes),
                              },
                          }
                        : undefined,
                    video: part.video
                        ? {
                              ...part.video,
                              source: {
                                  ...part.video.source,
                                  bytes: part.video.source.bytes
                                      ? truncateBinaryForDebug(part.video.source.bytes)
                                      : undefined,
                              },
                          }
                        : undefined,
                };
            }),
        })),
    };
}

function formatTwelvelabsPromptForDebug(prompt: TwelvelabsPegasusRequest): TwelvelabsPegasusRequest {
    if (!prompt.mediaSource.base64String) {
        return prompt;
    }
    return {
        ...prompt,
        mediaSource: {
            ...prompt.mediaSource,
            base64String: truncateBinaryForDebug(prompt.mediaSource.base64String),
        },
    };
}

export class BedrockDriver extends AbstractDriver<BedrockDriverOptions, BedrockPrompt> {
    static readonly PROVIDER = Providers.bedrock;

    provider = BedrockDriver.PROVIDER;

    private _executor?: BedrockRuntime;
    private _service?: Bedrock;
    private _service_region?: string;
    constructor(options: BedrockDriverOptions) {
        super(options);
        if (!options.region) {
            throw new Error("No region found. Set the region in the environment's endpoint URL.");
        }
    }

    protected override supportsCanonicalConversation(options: ExecutionOptions): boolean {
        if (this.isImageModel(options.model)) return this.supportsCanonicalImageGeneration(options);
        return true;
    }

    protected override supportsCanonicalContextConversation(_options: CanonicalExecutionContextOptions): boolean {
        return true;
    }

    override async executeCanonical(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        if (options.model.includes('twelvelabs.pegasus')) {
            validateTwelvelabsPegasusCanonicalInput(segments, options);
            return super.executeCanonical(
                segments,
                { ...options, [TWELVELABS_PEGASUS_CANONICAL_FORMAT]: true } as PegasusCanonicalExecutionOptions,
                signal,
            );
        }
        return super.executeCanonical(segments, options, signal);
    }

    override async streamCanonicalEvents(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        if (options.model.includes('twelvelabs.pegasus')) {
            validateTwelvelabsPegasusCanonicalInput(segments, options);
            return super.streamCanonicalEvents(
                segments,
                { ...options, [TWELVELABS_PEGASUS_CANONICAL_FORMAT]: true } as PegasusCanonicalExecutionOptions,
                signal,
                open,
            );
        }
        return super.streamCanonicalEvents(segments, options, signal, open);
    }

    protected override supportsCanonicalImageGeneration(options: ExecutionOptions): boolean {
        return this.isImageModel(options.model);
    }

    protected override validateCanonicalImageInput(segments: PromptSegment[], options: ExecutionOptions): void {
        validateNovaCanvasCanonicalInput(segments, options);
    }

    /**
     * Build a Smithy `requestHandler` config from the driver's
     * `httpTimeout` so AWS SDK calls have a bounded safety timeout instead of
     * using the AWS default (no request timeout).
     * Returns a partial config the SDK merges into its default handler.
     */
    private getBedrockRequestHandlerConfig(httpTimeout?: HttpTimeoutOptions) {
        const timeouts = resolveDriverHttpTimeouts(
            mergeDriverHttpTimeoutOptions(this.options.httpTimeout, httpTimeout),
        );
        return {
            requestTimeout: timeouts.headersTimeout,
            throwOnRequestTimeout: true,
            connectionTimeout: timeouts.connectTimeout,
            socketTimeout: timeouts.bodyTimeout,
        };
    }

    private createExecutor(httpTimeout?: HttpTimeoutOptions) {
        return new BedrockRuntime({
            region: this.options.region,
            credentials: this.options.credentials,
            requestHandler: this.getBedrockRequestHandlerConfig(httpTimeout),
        });
    }

    getExecutor(httpTimeout?: HttpTimeoutOptions) {
        if (httpTimeout) {
            return this.createExecutor(httpTimeout);
        }
        if (!this._executor) {
            this._executor = this.createExecutor();
        }
        return this._executor;
    }

    private getScopedExecutor(options: Pick<ExecutionOptions, 'httpTimeout'>): BedrockRuntimeExecutorScope {
        if (!options.httpTimeout) {
            return {
                executor: this.getExecutor(),
                close: () => undefined,
            };
        }

        const executor = this.getExecutor(options.httpTimeout);
        return {
            executor,
            close: () => executor.destroy(),
        };
    }

    private twelvelabsPegasusTransport(options: ExecutionOptions): TwelvelabsPegasusTransport {
        return {
            invoke: async (request: TwelvelabsPegasusInvokeRequest, signal?: AbortSignal) => {
                const scope = this.getScopedExecutor(options);
                try {
                    return signal
                        ? await scope.executor.invokeModel(request, { abortSignal: signal })
                        : await scope.executor.invokeModel(request);
                } finally {
                    scope.close();
                }
            },
            stream: async (request: TwelvelabsPegasusInvokeRequest, signal: AbortSignal) => {
                const scope = this.getScopedExecutor(options);
                try {
                    const response = await scope.executor.invokeModelWithResponseStream(request, {
                        abortSignal: signal,
                    });
                    if (response.body === undefined) throw new Error('[Bedrock] Stream not found in response');
                    return {
                        body: withBedrockRuntimeScope(
                            response.body as AsyncIterable<TwelvelabsPegasusStreamEvent>,
                            scope,
                        ),
                        ...(response.$metadata.requestId === undefined
                            ? {}
                            : { provider_response_id: response.$metadata.requestId }),
                        ...(response.serviceTier === undefined ? {} : { service_tier: response.serviceTier }),
                    };
                } catch (error: unknown) {
                    scope.close();
                    throw error;
                }
            },
        };
    }

    getService(region: string = this.options.region) {
        if (!this._service || this._service_region !== region) {
            this._service = new Bedrock({
                region: region,
                credentials: this.options.credentials,
                requestHandler: this.getBedrockRequestHandlerConfig(),
            });
            this._service_region = region;
        }
        return this._service;
    }

    protected async formatPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<BedrockPrompt> {
        if (opts.model.includes('canvas')) {
            return await formatNovaPrompt(segments, opts.result_schema);
        }
        if (opts.model.includes('twelvelabs.pegasus')) {
            return await formatTwelvelabsPegasusPrompt(
                segments,
                opts,
                (opts as PegasusCanonicalExecutionOptions)[TWELVELABS_PEGASUS_CANONICAL_FORMAT] === true
                    ? { max_video_bytes: TWELVELABS_PEGASUS_MAX_INLINE_VIDEO_BYTES }
                    : {},
            );
        }
        return await formatConversePrompt(segments, opts);
    }

    public formatDebugPrompt(prompt: BedrockPrompt): BedrockPrompt {
        if ('mediaSource' in prompt) {
            return formatTwelvelabsPromptForDebug(prompt);
        }
        if ('modelId' in prompt) {
            return formatConversePromptForDebug(prompt);
        }
        return formatNovaPromptForDebug(prompt);
    }

    /**
     * Format AWS Bedrock errors into LlumiverseError with proper status codes and retryability.
     *
     * AWS SDK errors provide:
     * - error.name: The exception type (e.g., "ThrottlingException")
     * - error.$metadata.httpStatusCode: The HTTP status code
     * - error.$metadata.requestId: The AWS request ID for tracking
     * - error.$fault: "client" or "server" indicating error category
     *
     * @param error - The AWS SDK error
     * @param context - Context about where the error occurred
     * @returns A standardized LlumiverseError
     */
    public formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        // Check if it's an AWS SDK error with $metadata
        const awsError = error as AwsSdkError;
        const hasMetadata = awsError?.$metadata !== undefined;

        if (!hasMetadata) {
            // Not an AWS SDK error, use default handling
            return super.formatLlumiverseError(error, context);
        }

        // Extract AWS-specific fields
        const errorName = awsError.name || 'UnknownError';
        const httpStatusCode = awsError.$metadata?.httpStatusCode;
        const requestId = awsError.$metadata?.requestId;
        const fault = awsError.$fault; // "client" or "server"

        // Extract error message - handle both Error instances and plain objects
        let message: string;
        if (error instanceof Error) {
            message = error.message;
        } else if (typeof awsError.message === 'string') {
            message = awsError.message;
        } else {
            message = String(error);
        }

        // Build user-facing message with error name and status code
        let userMessage = message;

        // Include status code in message if available (for end-user visibility)
        if (httpStatusCode) {
            userMessage = `[${httpStatusCode}] ${userMessage}`;
        }

        // Prefix with error name if it's meaningful (not just "Error")
        if (errorName && errorName !== 'Error' && errorName !== 'UnknownError') {
            userMessage = `${errorName}: ${userMessage}`;
        }

        // Add request ID if available (useful for AWS support)
        if (requestId) {
            userMessage += ` (Request ID: ${requestId})`;
        }

        // Determine retryability based on AWS error types
        const retryable = this.isBedrockErrorRetryable(errorName, httpStatusCode, fault);

        return new LlumiverseError(
            `[${this.provider}] ${userMessage}`,
            retryable,
            context,
            error,
            httpStatusCode, // Only set code if we have numeric status code
            errorName, // Preserve AWS error name
        );
    }

    /**
     * Determine if a Bedrock error is retryable based on error type and status.
     *
     * Retryable errors:
     * - ThrottlingException: Rate limit exceeded, retry with backoff
     * - ServiceUnavailableException: Service temporarily down
     * - InternalServerException: Server-side error
     * - ServiceQuotaExceededException: Quota exhausted, may recover
     * - 5xx status codes: Server errors
     * - 429, 408 status codes: Rate limit, timeout
     *
     * Non-retryable errors:
     * - ValidationException: Invalid request parameters
     * - AccessDeniedException: Authentication/authorization failure
     * - ResourceNotFoundException: Resource doesn't exist
     * - ConflictException: Resource state conflict
     * - ResourceInUseException: Resource locked by another operation
     * - 4xx status codes (except 429, 408): Client errors
     *
     * @param errorName - The AWS error name (e.g., "ThrottlingException")
     * @param httpStatusCode - The HTTP status code if available
     * @param fault - The fault type ("client" or "server")
     * @returns True if retryable, false if not retryable, undefined if unknown
     */
    private isBedrockErrorRetryable(
        errorName: string,
        httpStatusCode: number | undefined,
        fault: string | undefined,
    ): boolean | undefined {
        // Check specific AWS error types first
        switch (errorName) {
            // Retryable errors
            case 'ThrottlingException':
            case 'ServiceUnavailableException':
            case 'InternalServerException':
            case 'ServiceQuotaExceededException':
                return true;

            // Non-retryable errors
            case 'ValidationException':
            case 'AccessDeniedException':
            case 'ResourceNotFoundException':
            case 'ConflictException':
            case 'ResourceInUseException':
            case 'TooManyTagsException':
                return false;
        }

        // If we have HTTP status code, use it
        if (httpStatusCode !== undefined) {
            if (httpStatusCode === 429 || httpStatusCode === 408) return true; // Rate limit, timeout
            if (httpStatusCode === 529) return true; // Overloaded
            if (httpStatusCode >= 500 && httpStatusCode < 600) return true; // Server errors
            if (httpStatusCode >= 400 && httpStatusCode < 500) return false; // Client errors
        }

        // Fall back to fault type
        if (fault === 'server') return true;
        if (fault === 'client') return false;

        // Unknown error type - let consumer decide retry strategy
        return undefined;
    }

    getExtractedExecution(
        result: ConverseResponse,
        _prompt?: BedrockPrompt,
        options?: ExecutionOptions,
    ): CompletionChunkObject {
        let resultText = '';
        let reasoning = '';

        if (options?.model.toLowerCase().includes('claude')) {
            logClaudeTruncation(this.logger, result.stopReason, { provider: this.provider, model: options.model });
        }

        if (result.output?.message?.content) {
            for (const content of result.output.message.content) {
                // Get text output
                if (content.text) {
                    resultText += content.text;
                } else if (content.reasoningContent) {
                    // Extract reasoning content if include_thoughts is true, or if it's a
                    // reasoning-only model (e.g. DeepSeek R1) that returns no text blocks
                    const claudeOptions = options?.model_options as BedrockClaudeOptions;
                    const isReasoningModel = options?.model?.includes('deepseek') && options?.model?.includes('r1');
                    if (claudeOptions?.include_thoughts || isReasoningModel) {
                        if (content.reasoningContent.reasoningText) {
                            reasoning += content.reasoningContent.reasoningText.text;
                        } else if (content.reasoningContent.redactedContent) {
                            // Handle redacted thinking content
                            const redactedData = new TextDecoder().decode(content.reasoningContent.redactedContent);
                            reasoning += `[Redacted thinking: ${redactedData}]`;
                        }
                    } else {
                        this.logger.info('[Bedrock] Not outputting reasoning content as include_thoughts is false');
                    }
                } else {
                    // Get content block type
                    const type = Object.keys(content).find(
                        (key) => key !== '$unknown' && content[key as keyof typeof content] !== undefined,
                    );
                    this.logger.info({ type }, '[Bedrock] Unsupported content response type:');
                }
            }

            // Add spacing if we have reasoning content
            if (reasoning) {
                reasoning += '\n\n';
            }
        }

        const completionResult: CompletionChunkObject = {
            result: reasoning + resultText ? [{ type: 'text', value: reasoning + resultText }] : [],
            token_usage: converseTokenUsage(result.usage) ?? {},
            service_tier: result.serviceTier?.type,
            finish_reason: converseFinishReason(result.stopReason),
        };

        return completionResult;
    }

    getExtractedStream(
        result: ConverseStreamOutput,
        _prompt?: BedrockPrompt,
        options?: ExecutionOptions,
        streamingToolBlocks?: Map<number, { id: string; name: string }>,
    ): CompletionChunkObject {
        let output: string = '';
        let reasoning: string = '';
        let stop_reason = '';
        let token_usage: ExecutionTokenUsage | undefined;
        let tool_use: ToolUse<unknown>[] | undefined;

        // Check if we should include thoughts (always true for reasoning-only models like DeepSeek R1)
        const isReasoningModel = options?.model?.includes('deepseek') && options?.model?.includes('r1');
        const shouldIncludeThoughts =
            isReasoningModel || (options && (options.model_options as BedrockClaudeOptions)?.include_thoughts);

        // Handle content block start events (for reasoning blocks and tool use)
        if (result.contentBlockStart) {
            if (
                result.contentBlockStart.start &&
                'toolUse' in result.contentBlockStart.start &&
                result.contentBlockStart.start.toolUse
            ) {
                // Register new tool call block and emit an initial chunk so the accumulator can track it by id
                const toolUseStart = result.contentBlockStart.start.toolUse;
                const blockIndex = result.contentBlockStart.contentBlockIndex ?? -1;
                const id = toolUseStart.toolUseId ?? '';
                const name = toolUseStart.name ?? '';
                if (toolUseStart.type !== 'server_tool_use') {
                    streamingToolBlocks?.set(blockIndex, { id, name });
                    tool_use = [{ id, tool_name: name, tool_input: '' }];
                }
            } else if (
                result.contentBlockStart.start &&
                'reasoningContent' in result.contentBlockStart.start &&
                shouldIncludeThoughts
            ) {
                // Handle redacted content at block start
                const reasoningStart = result.contentBlockStart.start as ReasoningBlockStart;
                if (reasoningStart.reasoningContent?.redactedContent) {
                    const redactedData = new TextDecoder().decode(reasoningStart.reasoningContent.redactedContent);
                    reasoning = `[Redacted thinking: ${redactedData}]`;
                }
            }
        }

        // Handle content block deltas (text, reasoning, and tool use)
        if (result.contentBlockDelta) {
            const delta = result.contentBlockDelta.delta;
            if (delta?.toolUse) {
                // Emit tool input chunk; the accumulator in DefaultCompletionStream concatenates these strings
                const blockIndex = result.contentBlockDelta.contentBlockIndex ?? -1;
                const toolBlock = streamingToolBlocks?.get(blockIndex);
                if (toolBlock && delta.toolUse.input !== undefined) {
                    tool_use = [{ id: toolBlock.id, tool_name: '', tool_input: delta.toolUse.input }];
                }
            } else if (delta?.text) {
                output = delta.text;
            } else if (delta?.reasoningContent && shouldIncludeThoughts) {
                if (delta.reasoningContent.text) {
                    reasoning = delta.reasoningContent.text;
                } else if (delta.reasoningContent.redactedContent) {
                    const redactedData = new TextDecoder().decode(delta.reasoningContent.redactedContent);
                    reasoning = `[Redacted thinking: ${redactedData}]`;
                } else if (delta.reasoningContent.signature) {
                    // Handle signature updates for reasoning content - end of thinking
                    reasoning = '\n\n';
                    // Putting logging here so it only triggers once.
                    this.logger.info('[Bedrock] Not outputting reasoning content as include_thoughts is false');
                }
            } else if (delta) {
                // Get content block type
                const type = Object.keys(delta).find(
                    (key) => key !== '$unknown' && (delta as unknown as Record<string, unknown>)[key] !== undefined,
                );
                this.logger.info({ type }, '[Bedrock] Unsupported content response type:');
            }
        }

        // Handle content block stop events
        if (result.contentBlockStop) {
            // Clean up tool block tracking entry
            const blockIndex = result.contentBlockStop.contentBlockIndex ?? -1;
            streamingToolBlocks?.delete(blockIndex);
            // Add minimal spacing for reasoning blocks if not already present
            if (reasoning && !reasoning.endsWith('\n\n') && shouldIncludeThoughts) {
                reasoning += '\n\n';
            }
        }

        if (result.messageStop) {
            stop_reason = result.messageStop.stopReason ?? '';
            if (options?.model.toLowerCase().includes('claude')) {
                logClaudeTruncation(this.logger, stop_reason, { provider: this.provider, model: options.model });
            }
        }

        if (result.metadata) {
            token_usage = converseTokenUsage(result.metadata.usage) ?? {};
        }

        const completionResult: CompletionChunkObject = {
            result: reasoning + output ? [{ type: 'text', value: reasoning + output }] : [],
            token_usage: token_usage,
            service_tier: result.metadata?.serviceTier?.type,
            finish_reason: converseFinishReason(stop_reason),
            tool_use,
        };

        return completionResult;
    }

    extractRegion(modelString: string, defaultRegion: string): string {
        // Match region in full ARN pattern
        const arnMatch = modelString.match(/arn:aws[^:]*:bedrock:([^:]+):/);
        if (arnMatch) {
            return arnMatch[1];
        }

        // Match common AWS regions directly in string
        const regionMatch = modelString.match(
            /(?:us|eu|ap|sa|ca|me|af)[-](east|west|central|south|north|southeast|southwest|northeast|northwest)[-][1-9]/,
        );
        if (regionMatch) {
            return regionMatch[0];
        }

        return defaultRegion;
    }

    private async getCanStream(model: string, type: BedrockModelType, signal?: AbortSignal): Promise<boolean> {
        let canStream: boolean = false;
        let error: unknown = null;
        const region = this.extractRegion(model, this.options.region);
        const requestOptions = signal ? { abortSignal: signal } : undefined;
        if (type === BedrockModelType.FoundationModel || type === BedrockModelType.Unknown) {
            try {
                const response = await this.getService(region).getFoundationModel(
                    { modelIdentifier: model },
                    requestOptions,
                );
                canStream = response.modelDetails?.responseStreamingSupported ?? false;
                return canStream;
            } catch (e) {
                signal?.throwIfAborted();
                error = e;
            }
        }
        if (type === BedrockModelType.InferenceProfile || type === BedrockModelType.Unknown) {
            try {
                const response = await this.getService(region).getInferenceProfile(
                    { inferenceProfileIdentifier: model },
                    requestOptions,
                );
                canStream = await this.getCanStream(
                    bedrockFoundationModelLookupId(response.models?.[0].modelArn ?? ''),
                    BedrockModelType.FoundationModel,
                    signal,
                );
                return canStream;
            } catch (e) {
                signal?.throwIfAborted();
                error = e;
            }
        }
        if (type === BedrockModelType.CustomModel || type === BedrockModelType.Unknown) {
            try {
                const response = await this.getService(region).getCustomModel(
                    { modelIdentifier: model },
                    requestOptions,
                );
                canStream = await this.getCanStream(
                    response.baseModelArn ?? '',
                    BedrockModelType.FoundationModel,
                    signal,
                );
                return canStream;
            } catch (e) {
                signal?.throwIfAborted();
                error = e;
            }
        }
        if (error) {
            console.warn(`Error on canStream check for model: ${model} region detected: ${region}`, error);
        }
        return canStream;
    }

    protected async canStream(options: ExecutionOptions, signal?: AbortSignal): Promise<boolean> {
        // Pegasus supports InvokeModelWithResponseStream for both foundation-model and inference-profile selectors.
        if (options.model.includes('twelvelabs.pegasus')) return true;

        let canStream = supportStreamingCache.get(options.model);
        if (canStream == null) {
            let type = BedrockModelType.Unknown;
            if (options.model.includes('foundation-model')) {
                type = BedrockModelType.FoundationModel;
            } else if (options.model.includes('inference-profile')) {
                type = BedrockModelType.InferenceProfile;
            } else if (options.model.includes('custom-model')) {
                type = BedrockModelType.CustomModel;
            }
            canStream = await this.getCanStream(options.model, type, signal);
            signal?.throwIfAborted();
            supportStreamingCache.set(options.model, canStream);
        }
        return canStream;
    }

    /**
     * Build conversation context after streaming completion.
     * Reconstructs the assistant message from accumulated results and applies stripping.
     */
    buildStreamingConversation(
        prompt: BedrockPrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): ConverseRequest | undefined {
        // Only handle ConverseRequest prompts (not NovaMessagesPrompt or TwelvelabsPegasusRequest)
        if (options.model.includes('canvas') || options.model.includes('twelvelabs.pegasus')) {
            return undefined;
        }

        const conversePrompt = prompt as ConverseRequest;
        const completionResults = result as CompletionResult[];

        // Convert accumulated results to text content for assistant message
        const textContent = completionResults
            .map((r) => {
                switch (r.type) {
                    case 'text':
                        return r.value;
                    case 'thoughts':
                        return '';
                    case 'json':
                        return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                    case 'image':
                        // Skip images in conversation - they're in the result
                        return '';
                    case 'audio':
                    case 'video':
                        return '';
                    default: {
                        const _exhaustive: never = r;
                        return String(_exhaustive);
                    }
                }
            })
            .join('');

        // Deserialize any base64-encoded binary data back to Uint8Array
        const incomingConversation = deserializeBinaryFromStorage(options.conversation) as ConverseRequest;

        // Start with the conversation from options combined with the prompt
        let conversation = updateConversation(incomingConversation, conversePrompt);

        // Build assistant message content
        const messageContent: ContentBlock[] = [];
        if (textContent) {
            messageContent.push({ text: textContent });
        }
        // Add tool use blocks if present
        if (toolUse && toolUse.length > 0) {
            for (const tool of toolUse as ToolUse[]) {
                messageContent.push({
                    toolUse: {
                        toolUseId: tool.id,
                        name: tool.tool_name,
                        input: tool.tool_input,
                    },
                });
            }
        }

        // Add assistant message
        const assistantMessage: ConverseRequest = {
            messages: [
                {
                    content: messageContent.length > 0 ? messageContent : [{ text: '' }],
                    role: 'assistant',
                },
            ],
            modelId: conversePrompt.modelId,
        };
        conversation = updateConversation(conversation, assistantMessage);

        // Increment turn counter
        conversation = incrementConversationTurn(conversation) as ConverseRequest;

        // Apply stripping based on options
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens,
        };
        let processedConversation = stripBinaryFromConversation(conversation, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        return processedConversation as ConverseRequest;
    }

    async requestTextCompletion(
        prompt: BedrockPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        if (options.model.includes('twelvelabs.pegasus')) {
            return this.requestTwelvelabsPegasusCompletion(prompt as TwelvelabsPegasusRequest, options, signal);
        }
        const response = await this.requestCanonicalTextCompletion(prompt, options, signal);
        const isReasoningModel = options.model.includes('deepseek') && options.model.includes('r1');
        return legacyCompletionFromCanonicalExecution(response, {
            include_reasoning:
                isReasoningModel ||
                (options.model_options as BedrockClaudeOptions | undefined)?.include_thoughts === true,
        });
    }

    async requestCanonicalTextCompletion(
        prompt: BedrockPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        if (options.model.includes('twelvelabs.pegasus')) {
            return executeTwelvelabsPegasusCanonical({
                provider: this.provider,
                region: this.options.region,
                prompt: prompt as TwelvelabsPegasusCanonicalPrompt,
                options,
                signal,
                transport: this.twelvelabsPegasusTransport(options),
            });
        }

        const conversePrompt = prompt as ConverseRequest;
        const canonicalState = await prepareBedrockConverseCanonicalState({
            conversation: isConversationDocumentFormat(options.conversation)
                ? options.conversation
                : deserializeBinaryFromStorage(options.conversation),
            prompt: conversePrompt,
            options,
            provider: this.provider,
        });
        return this.requestPreparedCanonicalTextCompletion(canonicalState, options, signal);
    }

    async requestCanonicalContextCompletion(
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        if (options.model.includes('twelvelabs.pegasus')) {
            return executeTwelvelabsPegasusCanonicalContext({
                provider: this.provider,
                region: this.options.region,
                options,
                signal,
                transport: this.twelvelabsPegasusTransport(options),
            });
        }
        const canonicalState = await prepareBedrockConverseCanonicalContext({
            options,
            provider: this.provider,
        });
        return this.requestPreparedCanonicalTextCompletion(canonicalState, options, signal, true);
    }

    private async requestPreparedCanonicalTextCompletion(
        canonicalState: Omit<PreparedBedrockConverseConversation, 'payload' | 'receipt' | 'diagnostics'>,
        options: ExecutionOptions,
        signal?: AbortSignal,
        contextOnly = false,
    ): Promise<CanonicalExecutionResponse> {
        const baseConversation: ConverseRequest = {
            ...canonicalState.native_conversation,
            modelId: options.model,
        };
        const conversation = contextOnly
            ? projectConverseContextResultSchema(baseConversation, options, canonicalState.tool_definitions.length > 0)
            : baseConversation;
        const payload = this.preparePayload(conversation, options, canonicalState.tool_definitions);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            { provider: this.provider, protocol: 'aws.bedrock.converse', model: options.model },
            bedrockConverseJsonValue(payload),
        );
        if (canonicalState.accepted_response !== undefined) {
            if (options.include_original_response) {
                throw new Error(
                    'An idempotently recovered Bedrock Converse response cannot reconstruct original_response',
                );
            }
            return recoverCanonicalExecutionResponse(canonicalState, options);
        }
        const prepared = await finalizeBedrockConversePreparedRequest(
            { ...canonicalState, native_conversation: conversation },
            payload,
        );
        await publishCanonicalPreparedRequest(prepared, options);
        const executorScope = this.getScopedExecutor(options);

        let res: ConverseResponse;
        try {
            res = signal
                ? await executorScope.executor.converse({ ...payload }, { abortSignal: signal })
                : await executorScope.executor.converse({ ...payload });
        } finally {
            executorScope.close();
        }

        let tool_use: ToolUse<unknown>[] | undefined;
        //Get tool requests, we check tool use regardless of finish reason, as you can hit length and still get a valid response.
        tool_use = res.output?.message?.content?.reduce((tools: ToolUse<unknown>[], c) => {
            if (c.toolUse && c.toolUse.type !== 'server_tool_use') {
                tools.push({
                    tool_name: c.toolUse.name ?? '',
                    tool_input: c.toolUse.input,
                    id: c.toolUse.toolUseId ?? '',
                } satisfies ToolUse<unknown>);
            }
            return tools;
        }, []);
        //If no tools were used, set to undefined
        if (tool_use && tool_use.length === 0) {
            tool_use = undefined;
        }

        const rawDecoded = await decodeBedrockConverseCanonicalResponse(res, prepared);
        const normalized =
            tool_use === undefined && options.result_schema
                ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
                : undefined;
        let decoded =
            normalized?.status === 'valid'
                ? await decodeBedrockConverseCanonicalResponse(res, prepared, normalized.structured_output)
                : rawDecoded;
        if (normalized?.status === 'invalid') decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
        const processedConversation = await appendBedrockConverseCanonicalResponseWithProcessing(prepared, decoded);

        return createCanonicalExecutionResponse(processedConversation, prepared.runtime.response_operation_id, {
            ...(res.serviceTier?.type === undefined ? {} : { service_tier: res.serviceTier.type }),
            ...(options.include_original_response ? { original_response: res } : {}),
        });
    }

    private async requestTwelvelabsPegasusCompletion(
        prompt: TwelvelabsPegasusRequest,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const executorScope = this.getScopedExecutor(options);

        let res: InvokeModelCommandOutput;
        try {
            const request = {
                modelId: options.model,
                contentType: 'application/json',
                accept: 'application/json',
                body: JSON.stringify(prompt),
                serviceTier: getBedrockServiceTier(options.model_options),
            };
            res = signal
                ? await executorScope.executor.invokeModel(request, { abortSignal: signal })
                : await executorScope.executor.invokeModel(request);
        } finally {
            executorScope.close();
        }

        const decoder = new TextDecoder();
        const body = decoder.decode(res.body);
        const result = JSON.parse(body);

        // Extract the response according to TwelveLabs Pegasus format
        let finishReason: string | undefined;
        switch (result.finishReason) {
            case 'stop':
                finishReason = 'stop';
                break;
            case 'length':
                finishReason = 'length';
                break;
            default:
                finishReason = result.finishReason;
        }

        return {
            result: result.message ? [{ type: 'text' as const, value: result.message }] : [],
            service_tier: res.serviceTier,
            finish_reason: finishReason,
            original_response: options.include_original_response ? result : undefined,
        };
    }

    private async requestTwelvelabsPegasusCompletionStream(
        prompt: TwelvelabsPegasusRequest,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        const executorScope = this.getScopedExecutor(options);
        try {
            const request = {
                modelId: options.model,
                contentType: 'application/json',
                accept: 'application/json',
                body: JSON.stringify(prompt),
                serviceTier: getBedrockServiceTier(options.model_options),
            };
            const res = signal
                ? await executorScope.executor.invokeModelWithResponseStream(request, {
                      abortSignal: signal,
                  })
                : await executorScope.executor.invokeModelWithResponseStream(request);

            if (!res.body) {
                throw new Error('[Bedrock] Stream not found in response');
            }
            const accumulator = new TwelvelabsPegasusNativeStreamAccumulator();
            const stream = (async function* (): AsyncIterable<CompletionChunkObject> {
                for await (const event of res.body as AsyncIterable<TwelvelabsPegasusStreamEvent>) {
                    const chunk = accumulator.accept(event);
                    yield {
                        result: chunk.fragment.length === 0 ? [] : [{ type: 'text', value: chunk.fragment }],
                        finish_reason: chunk.finish_reason,
                        service_tier: res.serviceTier,
                    };
                }
                accumulator.response({
                    provider_response_id: res.$metadata?.requestId,
                    service_tier: res.serviceTier,
                });
            })();
            return withBedrockRuntimeScope(stream, executorScope);
        } catch (err) {
            executorScope.close();
            throw err;
        }
    }

    async requestTextCompletionStream(
        prompt: BedrockPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        // Handle Twelvelabs Pegasus models
        if (options.model.includes('twelvelabs.pegasus')) {
            return this.requestTwelvelabsPegasusCompletionStream(prompt as TwelvelabsPegasusRequest, options, signal);
        }

        // Handle other Bedrock models that use Converse API
        const conversePrompt = prompt as ConverseRequest;
        const canonicalState = await prepareBedrockConverseCanonicalState({
            conversation: isConversationDocumentFormat(options.conversation)
                ? options.conversation
                : deserializeBinaryFromStorage(options.conversation),
            prompt: conversePrompt,
            options,
            provider: this.provider,
        });
        const conversation: ConverseRequest = {
            ...canonicalState.native_conversation,
            modelId: options.model,
        };
        const payload = this.preparePayload(conversation, options);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            { provider: this.provider, protocol: 'aws.bedrock.converse', model: options.model },
            bedrockConverseJsonValue(payload),
        );
        if (canonicalState.accepted_response !== undefined) {
            const canonical = await recoverCanonicalExecutionResponse(canonicalState, options);
            return recoveredBedrockStream(
                legacyCompletionFromCanonicalExecution(canonical, {
                    include_reasoning:
                        (options.model.includes('deepseek') && options.model.includes('r1')) ||
                        (options.model_options as BedrockClaudeOptions | undefined)?.include_thoughts === true,
                }),
            );
        }
        const prepared = await finalizeBedrockConversePreparedRequest(
            { ...canonicalState, native_conversation: conversation },
            payload,
        );
        await publishCanonicalPreparedRequest(prepared, options);
        const executorScope = this.getScopedExecutor(options);
        const response = signal
            ? executorScope.executor.converseStream({ ...payload }, { abortSignal: signal })
            : executorScope.executor.converseStream({ ...payload });
        return response
            .then((res) => {
                const stream = res.stream;

                if (!stream) {
                    throw new Error('[Bedrock] Stream not found in response');
                }

                const streamingToolBlocks = new Map<number, { id: string; name: string }>();
                const nativeBlocks = new Map<number, ContentBlock>();
                let stopReason: ConverseResponse['stopReason'];
                let usage: ConverseResponse['usage'];
                let metrics: ConverseResponse['metrics'];
                let additionalModelResponseFields: ConverseResponse['additionalModelResponseFields'];
                let trace: ConverseResponse['trace'];
                let performanceConfig: ConverseResponse['performanceConfig'];
                let serviceTier: ConverseResponse['serviceTier'];
                const transformedStream = transformAsyncIterator(stream, (streamSegment: ConverseStreamOutput) => {
                    collectBedrockNativeStreamBlock(nativeBlocks, streamSegment);
                    if (streamSegment.messageStop !== undefined) {
                        stopReason = streamSegment.messageStop.stopReason;
                        additionalModelResponseFields = streamSegment.messageStop.additionalModelResponseFields;
                    }
                    if (streamSegment.metadata !== undefined) {
                        usage = streamSegment.metadata.usage;
                        metrics = streamSegment.metadata.metrics;
                        trace = streamSegment.metadata.trace;
                        performanceConfig = streamSegment.metadata.performanceConfig;
                        serviceTier = streamSegment.metadata.serviceTier;
                    }
                    return this.getExtractedStream(streamSegment, conversePrompt, options, streamingToolBlocks);
                });
                const scoped = withBedrockRuntimeScope(transformedStream, executorScope);
                let finalizedConversation: Promise<unknown> | undefined;
                const finalizeConversation = () => {
                    finalizedConversation ??= (async () => {
                        if (stopReason === undefined) {
                            throw new Error('Bedrock Converse stream ended without a terminal stop reason');
                        }
                        const blocks = finalizeBedrockNativeBlocks(nativeBlocks);
                        const terminalResponse = {
                            output: {
                                message: {
                                    role: 'assistant' as const,
                                    content: blocks.length === 0 ? [{ text: '' }] : blocks,
                                },
                            },
                            stopReason,
                            ...(usage === undefined ? {} : { usage }),
                            ...(metrics === undefined ? {} : { metrics }),
                            ...(additionalModelResponseFields === undefined ? {} : { additionalModelResponseFields }),
                            ...(trace === undefined ? {} : { trace }),
                            ...(performanceConfig === undefined ? {} : { performanceConfig }),
                            ...(serviceTier === undefined ? {} : { serviceTier }),
                            $metadata: res.$metadata,
                        } as unknown as ConverseResponse;
                        const rawDecoded = await decodeBedrockConverseCanonicalResponse(terminalResponse, prepared);
                        const normalized =
                            !blocks.some(
                                (block) => block.toolUse !== undefined && block.toolUse.type !== 'server_tool_use',
                            ) && options.result_schema
                                ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
                                : undefined;
                        let decoded =
                            normalized?.status === 'valid'
                                ? await decodeBedrockConverseCanonicalResponse(
                                      terminalResponse,
                                      prepared,
                                      normalized.structured_output,
                                  )
                                : rawDecoded;
                        if (normalized?.status === 'invalid') {
                            decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
                        }
                        return await appendBedrockConverseCanonicalResponseWithProcessing(prepared, decoded);
                    })();
                    return finalizedConversation;
                };
                return Object.assign(scoped, { finalizeConversation });
            })
            .catch((err) => {
                executorScope.close();
                this.logger.error({ error: err }, '[Bedrock] Failed to stream');
                throw err;
            });
    }

    async requestCanonicalTextCompletionEventStream(
        prompt: BedrockPrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        if (options.model.includes('twelvelabs.pegasus')) {
            return streamTwelvelabsPegasusCanonicalEvents({
                provider: this.provider,
                region: this.options.region,
                prompt: prompt as TwelvelabsPegasusCanonicalPrompt,
                options,
                signal,
                open,
                transport: this.twelvelabsPegasusTransport(options),
            });
        }
        const conversePrompt = prompt as ConverseRequest;
        const canonicalState = await prepareBedrockConverseCanonicalState({
            conversation: isConversationDocumentFormat(options.conversation)
                ? options.conversation
                : deserializeBinaryFromStorage(options.conversation),
            prompt: conversePrompt,
            options,
            provider: this.provider,
        });
        return this.requestPreparedCanonicalTextCompletionEventStream(canonicalState, options, signal, open);
    }

    async requestCanonicalContextCompletionEventStream(
        options: CanonicalExecutionContextOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        if (options.model.includes('twelvelabs.pegasus')) {
            return streamTwelvelabsPegasusCanonicalContextEvents({
                provider: this.provider,
                region: this.options.region,
                options,
                signal,
                open,
                transport: this.twelvelabsPegasusTransport(options),
            });
        }
        const canonicalState = await prepareBedrockConverseCanonicalContext({
            options,
            provider: this.provider,
        });
        return this.requestPreparedCanonicalTextCompletionEventStream(canonicalState, options, signal, open, true);
    }

    private async requestPreparedCanonicalTextCompletionEventStream(
        canonicalState: Omit<PreparedBedrockConverseConversation, 'payload' | 'receipt' | 'diagnostics'>,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
        contextOnly = false,
    ): Promise<CanonicalExecutionEventStream> {
        const baseConversation: ConverseRequest = { ...canonicalState.native_conversation, modelId: options.model };
        const conversation = contextOnly
            ? projectConverseContextResultSchema(baseConversation, options, canonicalState.tool_definitions.length > 0)
            : baseConversation;
        const payload = this.preparePayload(conversation, options, canonicalState.tool_definitions);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            { provider: this.provider, protocol: 'aws.bedrock.converse', model: options.model },
            bedrockConverseJsonValue(payload),
        );
        const accepted = canonicalState.accepted_response;
        const identity = {
            request_id: accepted?.generation.request_id ?? canonicalState.runtime.request_id,
            attempt_id: accepted?.generation.attempt_id ?? canonicalState.runtime.attempt_id,
            response_operation_id: canonicalState.runtime.response_operation_id,
            generation_id: accepted?.generation.id ?? canonicalState.generation_id,
            draft_turn_id: accepted?.turn.id ?? canonicalState.response_turn_id,
        };
        if (accepted !== undefined) {
            if (options.include_original_response) {
                throw new Error(
                    'An idempotently recovered Bedrock Converse response cannot reconstruct original_response',
                );
            }
            return new FallbackCanonicalExecutionEventStream(
                identity,
                () => recoverCanonicalExecutionResponse(canonicalState, options),
                { ...open, origin: 'accepted_recovery' },
            );
        }

        const prepared = await finalizeBedrockConversePreparedRequest(
            { ...canonicalState, native_conversation: conversation },
            payload,
        );
        const abortController = new AbortController();
        const forwardAbort = () => abortController.abort(signal?.reason);
        const nativeBlocks = new Map<number, ContentBlock>();
        const drafts = new Map<number, BedrockCanonicalDraft>();
        let outputRole: Message['role'] | undefined;
        let stopReason: ConverseResponse['stopReason'];
        let usage: ConverseResponse['usage'];
        let metrics: ConverseResponse['metrics'];
        let additionalModelResponseFields: ConverseResponse['additionalModelResponseFields'];
        let trace: ConverseResponse['trace'];
        let performanceConfig: ConverseResponse['performanceConfig'];
        let serviceTier: ConverseResponse['serviceTier'];
        let streamResponse: ConverseStreamCommandOutput | undefined;
        let executorScope: BedrockRuntimeExecutorScope | undefined;
        let messageStopped = false;
        let nativeContentBytes = 0;
        let previewChunks = 0;
        const maxNativeContentBytes = open.max_total_bytes ?? CONVERSATION_STREAM_MAX_TOTAL_BYTES;

        const eventStream = canonicalNativeExecutionEventStream({
            identity,
            open,
            openSource: async () => {
                const scope = this.getScopedExecutor(options);
                executorScope = scope;
                try {
                    streamResponse = await scope.executor.converseStream(
                        { ...payload },
                        { abortSignal: abortController.signal },
                    );
                } catch (error: unknown) {
                    scope.close();
                    executorScope = undefined;
                    throw error;
                }
                if (streamResponse.stream === undefined) {
                    throw new Error('[Bedrock] Stream not found in response');
                }
                return streamResponse.stream;
            },
            map: async (event, writer) => {
                assertBedrockConverseStreamEvent(event, messageStopped);
                const eventBytes = bedrockConverseStreamEventBytes(event);
                if (eventBytes > maxNativeContentBytes - nativeContentBytes) {
                    throw new Error('Bedrock Converse native stream content exceeds max_total_bytes');
                }
                nativeContentBytes += eventBytes;
                collectBedrockNativeStreamBlock(nativeBlocks, event);
                if (event.messageStart !== undefined) outputRole = event.messageStart.role;
                if (event.contentBlockStart !== undefined) {
                    const index = event.contentBlockStart.contentBlockIndex ?? -1;
                    const start = event.contentBlockStart.start;
                    if (start?.toolUse !== undefined) {
                        if (drafts.has(index)) throw new Error(`Bedrock Converse duplicate content block ${index}`);
                        const executor = start.toolUse.type === 'server_tool_use' ? 'provider' : 'application';
                        const position = bedrockStreamPosition(index, start.toolUse.toolUseId);
                        const draft: BedrockCanonicalDraft = {
                            draft_block_id: `${prepared.response_turn_id}:bedrock:${index}`,
                            native_position: position,
                            kind: 'tool_call',
                            text: '',
                            tool_argument_fragments: [],
                        };
                        drafts.set(index, draft);
                        await writer.startBlock({
                            draft_block_id: draft.draft_block_id,
                            native_position: position,
                            block: {
                                type: 'tool_call',
                                executor,
                                call_id: start.toolUse.toolUseId,
                                tool_name: start.toolUse.name,
                            },
                        });
                    } else if (start !== undefined) {
                        const startType = Object.keys(start).find(
                            (key) =>
                                key !== '$unknown' && (start as unknown as Record<string, unknown>)[key] !== undefined,
                        );
                        if (startType !== 'reasoningContent') {
                            throw new Error(
                                `Bedrock Converse typed stream does not support ${startType ?? 'unknown'} blocks`,
                            );
                        }
                    }
                }
                if (event.contentBlockDelta !== undefined) {
                    const index = event.contentBlockDelta.contentBlockIndex ?? -1;
                    const delta = event.contentBlockDelta.delta;
                    if (delta?.text !== undefined) {
                        let draft = drafts.get(index);
                        if (draft === undefined) {
                            const position = bedrockStreamPosition(index);
                            draft = {
                                draft_block_id: `${prepared.response_turn_id}:bedrock:${index}`,
                                native_position: position,
                                kind: 'text',
                                text: '',
                                tool_argument_fragments: [],
                            };
                            drafts.set(index, draft);
                            await writer.startBlock({
                                draft_block_id: draft.draft_block_id,
                                native_position: position,
                                block: { type: 'text' },
                            });
                        }
                        if (draft.kind !== 'text')
                            throw new Error(`Bedrock Converse content block ${index} changed kind`);
                        draft.text += delta.text;
                        if (delta.text.length > 0) previewChunks += 1;
                        await writer.text({
                            draft_block_id: draft.draft_block_id,
                            native_position: draft.native_position,
                            text: delta.text,
                        });
                    } else if (delta?.toolUse?.input !== undefined) {
                        const draft = drafts.get(index);
                        if (draft?.kind !== 'tool_call') {
                            throw new Error(`Bedrock Converse tool delta has no draft at content index ${index}`);
                        }
                        draft.tool_argument_fragments.push(delta.toolUse.input);
                        await writer.toolArgumentsFragment({
                            draft_block_id: draft.draft_block_id,
                            native_position: draft.native_position,
                            fragment: delta.toolUse.input,
                        });
                    } else if (delta?.reasoningContent !== undefined) {
                        if (delta.reasoningContent.text !== undefined) {
                            let draft = drafts.get(index);
                            if (draft === undefined) {
                                const position = bedrockStreamPosition(index);
                                draft = {
                                    draft_block_id: `${prepared.response_turn_id}:bedrock:${index}`,
                                    native_position: position,
                                    kind: 'reasoning',
                                    text: '',
                                    tool_argument_fragments: [],
                                };
                                drafts.set(index, draft);
                                await writer.startBlock({
                                    draft_block_id: draft.draft_block_id,
                                    native_position: position,
                                    block: { type: 'reasoning', visibility: 'display' },
                                });
                            }
                            if (draft.kind !== 'reasoning') {
                                throw new Error(`Bedrock Converse content block ${index} changed kind`);
                            }
                            draft.text += delta.reasoningContent.text;
                            if (delta.reasoningContent.text.length > 0) previewChunks += 1;
                            await writer.reasoning({
                                draft_block_id: draft.draft_block_id,
                                native_position: draft.native_position,
                                text: delta.reasoningContent.text,
                            });
                        }
                    } else if (delta !== undefined) {
                        const deltaType = Object.keys(delta).find(
                            (key) =>
                                key !== '$unknown' && (delta as unknown as Record<string, unknown>)[key] !== undefined,
                        );
                        throw new Error(
                            `Bedrock Converse typed stream does not support ${deltaType ?? 'unknown'} deltas`,
                        );
                    }
                }
                if (event.messageStop !== undefined) {
                    stopReason = event.messageStop.stopReason;
                    additionalModelResponseFields = event.messageStop.additionalModelResponseFields;
                    messageStopped = true;
                }
                if (event.metadata !== undefined) {
                    usage = event.metadata.usage;
                    metrics = event.metadata.metrics;
                    trace = event.metadata.trace;
                    performanceConfig = event.metadata.performanceConfig;
                    serviceTier = event.metadata.serviceTier;
                }
            },
            finalize: async () => {
                if (stopReason === undefined) {
                    throw new Error('Bedrock Converse stream ended without a terminal stop reason');
                }
                const entries = finalizeBedrockNativeBlockEntries(nativeBlocks);
                if (entries.length === 0) {
                    throw new Error('Bedrock Converse stream ended without decodable content');
                }
                const blocks = entries.map(([, block]) => block);
                const terminalResponse = {
                    output: { message: { role: outputRole ?? ('assistant' as const), content: blocks } },
                    stopReason,
                    ...(usage === undefined ? {} : { usage }),
                    ...(metrics === undefined ? {} : { metrics }),
                    ...(additionalModelResponseFields === undefined ? {} : { additionalModelResponseFields }),
                    ...(trace === undefined ? {} : { trace }),
                    ...(performanceConfig === undefined ? {} : { performanceConfig }),
                    ...(serviceTier === undefined ? {} : { serviceTier }),
                    ...(streamResponse?.$metadata === undefined ? {} : { $metadata: streamResponse.$metadata }),
                } as ConverseResponse;
                const hasApplicationTool = blocks.some(
                    (block) => block.toolUse !== undefined && block.toolUse.type !== 'server_tool_use',
                );
                const rawDecoded = await decodeBedrockConverseCanonicalResponse(terminalResponse, prepared);
                const normalized =
                    !hasApplicationTool && options.result_schema
                        ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
                        : undefined;
                let decoded =
                    normalized?.status === 'valid'
                        ? await decodeBedrockConverseCanonicalResponse(
                              terminalResponse,
                              prepared,
                              normalized.structured_output,
                          )
                        : rawDecoded;
                if (normalized?.status === 'invalid') {
                    decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
                }
                const document = await appendBedrockConverseCanonicalResponseWithProcessing(prepared, decoded);
                const response = createCanonicalExecutionResponse(document, prepared.runtime.response_operation_id, {
                    ...(serviceTier?.type === undefined ? {} : { service_tier: serviceTier.type }),
                    chunks: previewChunks,
                    ...(options.include_original_response ? { original_response: terminalResponse } : {}),
                });
                return {
                    decoded,
                    response,
                    prepare_reconciliation: async () => {
                        const semanticEntries = entries.filter(([, block]) => bedrockEntryHasSemanticBlock(block));
                        const positions = semanticEntries.map(([index, block]) =>
                            bedrockStreamPosition(index, block.toolUse?.toolUseId),
                        );
                        const rawBlocks = bedrockSemanticBlocks(rawDecoded, prepared.response_turn_id);
                        if (rawBlocks.length !== positions.length) {
                            throw new Error(
                                'Bedrock Converse stream decode does not match terminal native content positions',
                            );
                        }
                        const orderedDrafts = semanticEntries.map(([index]) => drafts.get(index));
                        const matchedDraftIds = new Set<string>();
                        for (const [index, block] of rawBlocks.entries()) {
                            const draft = orderedDrafts[index];
                            if (draft === undefined) continue;
                            matchedDraftIds.add(draft.draft_block_id);
                            assertBedrockToolDraftArguments(draft, block);
                        }
                        const itemMappings = rawBlocks.flatMap((block, index) => {
                            const position = positions[index];
                            if (position === undefined) return [];
                            return [
                                { canonical_id: block.id, native_position: position, kind: 'block' as const },
                                ...(block.type === 'tool_call'
                                    ? [
                                          {
                                              canonical_id: block.call_id,
                                              native_position: position,
                                              kind: 'call' as const,
                                          },
                                      ]
                                    : []),
                            ];
                        });
                        const transformations = [];
                        const reconciliations = [];
                        if (normalized?.status === 'valid') {
                            const sources = rawBlocks.filter((block) => block.type === 'text');
                            const result = bedrockSemanticBlocks(decoded, prepared.response_turn_id).find(
                                (block) => block.type === 'json',
                            );
                            if (sources.length === 0 || result?.type !== 'json') {
                                throw new Error('Bedrock structured stream is missing source or result blocks');
                            }
                            const proof = await createStructuredOutputTransformationProof({
                                id: `${prepared.generation_id}:structured-output`,
                                source_blocks: sources,
                                result_block: result,
                            });
                            transformations.push(proof);
                            const sourceDrafts = rawBlocks.flatMap((block, index) =>
                                block.type === 'text' && orderedDrafts[index] !== undefined
                                    ? [orderedDrafts[index]]
                                    : [],
                            );
                            reconciliations.push({
                                draft_block_ids: sourceDrafts.map((draft) => draft.draft_block_id),
                                native_positions: sourceDrafts.map((draft) => draft.native_position),
                                committed_block_ids: [result.id],
                                disposition: 'structured_output' as const,
                                transformation_id: proof.id,
                            });
                        }
                        for (const [index, block] of rawBlocks.entries()) {
                            if (normalized?.status === 'valid' && block.type === 'text') continue;
                            const draft = orderedDrafts[index];
                            if (draft === undefined) continue;
                            reconciliations.push({
                                draft_block_ids: [draft.draft_block_id],
                                native_positions: [draft.native_position],
                                committed_block_ids: [block.id],
                                disposition: 'direct' as const,
                            });
                        }
                        for (const draft of drafts.values()) {
                            if (matchedDraftIds.has(draft.draft_block_id)) continue;
                            if (draft.kind !== 'tool_call') {
                                throw new Error(
                                    `Bedrock terminal response omitted non-tool draft ${draft.draft_block_id}`,
                                );
                            }
                            reconciliations.push({
                                draft_block_ids: [draft.draft_block_id],
                                native_positions: [draft.native_position],
                                committed_block_ids: [],
                                disposition: 'omitted_invalid' as const,
                            });
                        }
                        const decodedWithEvidence = {
                            ...decoded,
                            stream_evidence: { item_mappings: itemMappings, transformations },
                        };
                        return {
                            decoded: decodedWithEvidence,
                            reconciliations,
                            deliver_final_events: async (writer) => {
                                if (decodedWithEvidence.generation.usage !== undefined) {
                                    await writer.usage(decodedWithEvidence.generation.usage);
                                }
                                for (const draft of drafts.values()) {
                                    const rawBlock = rawBlocks.find(
                                        (_block, index) =>
                                            orderedDrafts[index]?.draft_block_id === draft.draft_block_id,
                                    );
                                    await writer.finishBlock({
                                        draft_block_id: draft.draft_block_id,
                                        native_position: draft.native_position,
                                        outcome:
                                            rawBlock?.type === 'tool_call' && rawBlock.arguments.type === 'invalid'
                                                ? 'malformed'
                                                : rawBlock === undefined
                                                  ? 'malformed'
                                                  : decodedWithEvidence.generation.status === 'cancelled'
                                                    ? 'interrupted'
                                                    : decodedWithEvidence.generation.status === 'failed'
                                                      ? 'failed'
                                                      : 'native_complete',
                                    });
                                }
                                await writer.finish({
                                    outcome:
                                        decodedWithEvidence.generation.status === 'cancelled'
                                            ? 'interrupted'
                                            : decodedWithEvidence.generation.status === 'failed'
                                              ? 'failed'
                                              : 'completed',
                                    finish_reason: decodedWithEvidence.generation.finish_reason,
                                    ...(serviceTier?.type === undefined ? {} : { service_tier: serviceTier.type }),
                                });
                            },
                        };
                    },
                    ...(normalized?.status === 'valid' && options.result_schema !== undefined
                        ? { result_schema: options.result_schema }
                        : {}),
                };
            },
            abort: () => abortController.abort(),
            close: () => {
                signal?.removeEventListener('abort', forwardAbort);
                executorScope?.close();
                executorScope = undefined;
            },
        });
        await publishCanonicalPreparedRequest(prepared, options);
        if (signal?.aborted) forwardAbort();
        else signal?.addEventListener('abort', forwardAbort, { once: true });
        return eventStream;
    }

    preparePayload(
        prompt: ConverseRequest,
        options: ExecutionOptions,
        toolDefinitions: readonly BedrockToolDefinition[] | undefined = options.tools,
    ) {
        const model_options: TextFallbackOptions = (options.model_options as TextFallbackOptions) ?? {
            _option_id: 'text-fallback',
        };
        const privateToolOptions = model_options as typeof model_options & {
            tool_choice?: 'auto' | 'none' | 'any' | 'required';
            required_tool_name?: string;
        };
        const forcedToolRequested =
            privateToolOptions.required_tool_name !== undefined ||
            privateToolOptions.tool_choice === 'required' ||
            privateToolOptions.tool_choice === 'any';
        const tool_defs = getToolDefinitions(toolDefinitions);
        if (forcedToolRequested && !tool_defs?.length) {
            throw createToolChoiceConfigurationError(
                'A forced Bedrock tool turn requires at least one tool definition.',
                { provider: this.provider, model: options.model, operation: 'stream' },
            );
        }

        let additionalField: Record<string, unknown> = {};
        let supportsJSONPrefill = false;

        // Resolve thinking, effort, and sampling restrictions using shared Claude helper
        const claudeThinking = resolveClaudeThinking(
            options.model,
            options.model_options as BedrockClaudeOptions | undefined,
        );
        const claudeVersion = parseClaudeVersion(options.model);
        const onlySupportsAdaptiveThinking = claudeVersion?.variant === 'fable' || claudeVersion?.variant === 'mythos';
        const useAutomaticToolChoiceForAdaptiveClaude =
            options.model.includes('claude') && forcedToolRequested && onlySupportsAdaptiveThinking;
        // Bedrock only permits auto/none tool choice while thinking is active. Disable thinking
        // for the constrained turn on models that support the disabled form. Adaptive-only
        // families keep thinking and use automatic tool choice; the caller's required-tool
        // validator remains authoritative and triggers the bounded recovery path if the model
        // does not call the requested tool.
        const disableClaudeThinkingForForcedTool =
            options.model.includes('claude') &&
            forcedToolRequested &&
            claudeThinking.supportsThinking &&
            !useAutomaticToolChoiceForAdaptiveClaude;
        const hasSamplingRestriction = claudeThinking.hasSamplingRestriction;

        if (options.model.includes('amazon')) {
            supportsJSONPrefill = true;
            //Titan models also exists but does not support any additional options
            if (options.model.includes('nova')) {
                additionalField = { inferenceConfig: { topK: model_options.top_k } };
            }
        } else if (options.model.includes('claude')) {
            const claude_options = model_options as ModelOptions as BedrockClaudeOptions;
            if (disableClaudeThinkingForForcedTool) {
                additionalField = { ...additionalField, reasoning_config: { type: 'disabled' } };
            }
            // Claude never uses JSON prefill: newer models (4.6+) reject assistant
            // message prefill outright, and every supported Claude follows the
            // schema instruction injected into the prompt — so the model-version
            // gate this used to need is gone. Titan/Nova (above) keep prefill as
            // they have no native JSON adherence.

            // Claude 3.7+ supports thinking — use shared helper for reasoning_config
            if (claudeThinking.supportsThinking && !disableClaudeThinkingForForcedTool) {
                if (claudeThinking.thinking) {
                    additionalField = {
                        ...additionalField,
                        reasoning_config: claudeThinking.thinking,
                    };
                }
                // For Claude 3.7 with extended thinking + high output, add beta header
                if (
                    claudeThinking.thinking?.type === 'enabled' &&
                    options.model.includes('claude-3-7-sonnet') &&
                    ((claude_options.max_tokens ?? 0) > 64000 || (claude_options.thinking_budget_tokens ?? 0) > 64000)
                ) {
                    additionalField = {
                        ...additionalField,
                        anthropic_beta: ['output-128k-2025-02-19'],
                    };
                }
            }
            // Add effort parameter via output_config (Opus 4.5+, Sonnet 4.6+, all 4.7+)
            if (claudeThinking.outputConfig && !disableClaudeThinkingForForcedTool) {
                additionalField = {
                    ...additionalField,
                    output_config: claudeThinking.outputConfig,
                };
            }
            // Needs max_tokens to be set — and caller-provided values clamped to
            // the model's output limit (both handled by maxTokenFallbackClaude).
            model_options.max_tokens = maxTokenFallbackClaude(options);
            // Only models without sampling restrictions support top_k
            if (!hasSamplingRestriction) {
                additionalField = { ...additionalField, top_k: model_options.top_k };
            }
        } else if (options.model.includes('meta')) {
            //LLaMA models support no additional options
        } else if (options.model.includes('mistral')) {
            //7B instruct and 8x7B instruct
            if (options.model.includes('7b')) {
                additionalField = { top_k: model_options.top_k };
                //Does not support system messages
                if (prompt.system && prompt.system?.length !== 0) {
                    prompt.messages?.push(converseSystemToMessages(prompt.system));
                    prompt.system = undefined;
                    prompt.messages = converseConcatMessages(prompt.messages);
                }
            } else {
                //Other models such as Mistral Small,Large and Large 2
                //Support no additional fields.
            }
        } else if (options.model.includes('ai21')) {
            // Jurassic uses nested penalty scales; Jamba accepts the numeric fields directly.
            if (options.model.includes('j2')) {
                additionalField = {
                    presencePenalty: { scale: model_options.presence_penalty },
                    frequencyPenalty: { scale: model_options.frequency_penalty },
                };
                //Does not support system messages
                if (prompt.system && prompt.system?.length !== 0) {
                    prompt.messages?.push(converseSystemToMessages(prompt.system));
                    prompt.system = undefined;
                    prompt.messages = converseConcatMessages(prompt.messages);
                }
            } else if (options.model.includes('jamba')) {
                additionalField = {
                    presence_penalty: model_options.presence_penalty,
                    frequency_penalty: model_options.frequency_penalty,
                };
            }
        } else if (options.model.includes('cohere.command')) {
            // If last message is "```json", remove it.
            //Command R and R plus
            if (options.model.includes('cohere.command-r')) {
                additionalField = {
                    k: model_options.top_k,
                    frequency_penalty: model_options.frequency_penalty,
                    presence_penalty: model_options.presence_penalty,
                };
            } else {
                // Command non-R
                additionalField = { k: model_options.top_k };
                //Does not support system messages
                if (prompt.system && prompt.system?.length !== 0) {
                    prompt.messages?.push(converseSystemToMessages(prompt.system));
                    prompt.system = undefined;
                    prompt.messages = converseConcatMessages(prompt.messages);
                }
            }
        } else if (options.model.includes('palmyra')) {
            const palmyraOptions = model_options as ModelOptions as BedrockPalmyraOptions;
            additionalField = {
                seed: palmyraOptions?.seed,
                presence_penalty: palmyraOptions?.presence_penalty,
                frequency_penalty: palmyraOptions?.frequency_penalty,
                min_tokens: palmyraOptions?.min_tokens,
            };
        } else if (options.model.includes('deepseek')) {
            // DeepSeek models: no additional options, no stopSequences, only one of temperature/top_p
            model_options.stop_sequence = undefined;
            model_options.top_p = undefined;
        } else if (options.model.includes('gpt-oss')) {
            const gptOssOptions = model_options as ModelOptions as BedrockGptOssOptions;
            additionalField = {
                reasoning_effort: gptOssOptions?.reasoning_effort,
            };
        }

        //If last message is "```json", add corresponding ``` as a stop sequence.
        if (prompt.messages && prompt.messages.length > 0) {
            if (prompt.messages[prompt.messages.length - 1].content?.[0]?.text === '```json') {
                const stopSeq = model_options.stop_sequence;
                if (!stopSeq) {
                    model_options.stop_sequence = ['```'];
                } else if (!stopSeq.includes('```')) {
                    stopSeq.push('```');
                    model_options.stop_sequence = stopSeq;
                }
            }
        }

        // Use prefill when there is a schema and tools are not being used
        if (
            supportsJSONPrefill &&
            options.result_schema &&
            !tool_defs &&
            shouldIncludeSchemaInConversePrompt(options.model)
        ) {
            prompt.messages = converseJSONprefill(prompt.messages);
        }

        // Clean undefined values from additionalField since AWS Bedrock requires valid JSON
        // and will throw an exception for unrecognized parameters
        const cleanedAdditionalFields = removeUndefinedValues(additionalField);
        // Models with sampling parameter restrictions don't support temperature/top_p - exclude them from inference config
        const cleanedModelOptions = removeUndefinedValues({
            maxTokens: model_options.max_tokens,
            ...(hasSamplingRestriction
                ? {}
                : {
                      temperature: model_options.temperature,
                      topP: model_options.temperature != null ? undefined : model_options.top_p,
                  }),
            stopSequences: model_options.stop_sequence,
        } satisfies InferenceConfiguration);

        //Construct the final request payload
        // We only add fields that are defined to avoid AWS errors
        const request: ConverseRequest = {
            modelId: options.model,
        };

        const serviceTier = getBedrockServiceTier(options.model_options);
        if (serviceTier) {
            request.serviceTier = { type: serviceTier };
        }

        if (prompt.messages) {
            request.messages = relocateConverseToolImages(prompt.messages, options.model);
        }

        if (prompt.system) {
            request.system = prompt.system;
        }

        if (Object.keys(cleanedModelOptions).length > 0) {
            request.inferenceConfig = cleanedModelOptions;
        }

        if (Object.keys(cleanedAdditionalFields).length > 0) {
            request.additionalModelRequestFields =
                cleanedAdditionalFields as unknown as ConverseRequest['additionalModelRequestFields'];
        }

        if (options.result_schema && supportsConverseOutputConfig(options.model)) {
            request.outputConfig = {
                textFormat: {
                    type: 'json_schema',
                    structure: {
                        jsonSchema: {
                            name: 'output',
                            schema: JSON.stringify(options.result_schema),
                        },
                    },
                },
            };
        }

        const supportsForcedToolChoice = options.model.includes('claude') || options.model.includes('amazon.nova');
        if (forcedToolRequested && !supportsForcedToolChoice) {
            throw createToolChoiceConfigurationError(
                `Bedrock model ${options.model} cannot enforce the requested tool choice; use a tool-choice-capable model.`,
                { provider: this.provider, model: options.model, operation: 'stream' },
            );
        }
        if (tool_defs?.length) {
            request.toolConfig = {
                tools: tool_defs,
                ...(supportsForcedToolChoice &&
                !useAutomaticToolChoiceForAdaptiveClaude &&
                privateToolOptions.required_tool_name
                    ? { toolChoice: { tool: { name: privateToolOptions.required_tool_name } } }
                    : supportsForcedToolChoice &&
                        !useAutomaticToolChoiceForAdaptiveClaude &&
                        (privateToolOptions.tool_choice === 'required' || privateToolOptions.tool_choice === 'any')
                      ? { toolChoice: { any: {} } }
                      : {}),
            };
        } else if (request.messages && messagesContainToolBlocks(request.messages)) {
            // Bedrock requires toolConfig when conversation contains toolUse/toolResult blocks.
            // When no tools are provided (e.g. checkpoint summary calls), convert tool blocks
            // to text representations so the conversation data is preserved while satisfying
            // Bedrock's API requirements without making tools callable.
            request.messages = convertToolBlocksToText(request.messages);
        }

        // Prompt caching: use three breakpoints so stable system blocks, tool definitions,
        // and the conversation history prefix can all be reused across Claude turns.
        if (options.model.includes('claude')) {
            // Always strip stale markers from prior turns
            if (request.messages) {
                request.messages = stripClaudeCachePoints(request.messages);
            }
            request.system = stripClaudeCachePointsFromSystem(request.system);
            if (request.toolConfig?.tools) {
                request.toolConfig = {
                    ...request.toolConfig,
                    tools: stripClaudeCachePointsFromTools(request.toolConfig.tools),
                };
            }

            const claudeOptions = model_options as unknown as BedrockClaudeOptions;
            const cacheEnabled = options.prompt_cache_key !== undefined || claudeOptions?.cache_enabled === true;
            if (cacheEnabled) {
                const cacheTtl = claudeOptions?.cache_ttl;
                const cachePointBlock = { type: 'default' as const, ...(cacheTtl && { ttl: cacheTtl }) };

                if (request.system && request.system.length > 0) {
                    request.system = [...request.system, { cachePoint: cachePointBlock } satisfies BedrockSystemBlock];
                }

                if (request.toolConfig?.tools && request.toolConfig.tools.length > 0) {
                    request.toolConfig.tools = [
                        ...request.toolConfig.tools,
                        { cachePoint: cachePointBlock } satisfies BedrockToolEntry,
                    ];
                }

                if (options.prompt_cache_key !== undefined && request.messages && request.messages.length > 0) {
                    const lastMessage = request.messages[request.messages.length - 1];
                    if (lastMessage.content && lastMessage.content.length >= 2) {
                        lastMessage.content = [
                            ...lastMessage.content.slice(0, -1),
                            { cachePoint: cachePointBlock },
                            lastMessage.content[lastMessage.content.length - 1],
                        ];
                    }
                } else if (request.messages && request.messages.length >= 4) {
                    const pivotMsg = request.messages[request.messages.length - 2];
                    if (pivotMsg.content && Array.isArray(pivotMsg.content) && pivotMsg.content.length > 0) {
                        pivotMsg.content = [...pivotMsg.content, { cachePoint: cachePointBlock }];
                    }
                }
            }
        }

        return request;
    }

    protected isImageModel(model: string): boolean {
        // This execution path serializes the Nova Canvas wire schema. Other image families need their own request path.
        return model.includes('nova-canvas');
    }

    private async invokeNovaCanvas(
        payload: NovaCanvasPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<InvokeModelCommandOutput> {
        const executorScope = this.getScopedExecutor(options);
        try {
            const requestTimeout = this.getDriverRequestTimeoutMs(options.httpTimeout);
            return await executorScope.executor.invokeModel(
                {
                    modelId: options.model,
                    contentType: 'application/json',
                    accept: 'application/json',
                    body: JSON.stringify(payload),
                },
                {
                    abortSignal: signal,
                    requestTimeout,
                },
            );
        } finally {
            executorScope.close();
        }
    }

    override async requestCanonicalImageGeneration(
        prompt: NovaMessagesPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        return executeNovaCanvasCanonical({
            provider: this.provider,
            region: this.options.region,
            prompt,
            options,
            signal,
            invoke: (payload, invokeOptions, invokeSignal) =>
                this.invokeNovaCanvas(payload, invokeOptions, invokeSignal),
        });
    }

    async requestImageGeneration(
        prompt: NovaMessagesPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        if (
            options.model_options?._option_id !== undefined &&
            options.model_options?._option_id !== 'bedrock-nova-canvas'
        ) {
            this.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        const model_options = options.model_options as NovaCanvasOptions | undefined;

        const taskType = model_options?.taskType ?? NovaImageGenerationTaskType.TEXT_IMAGE;

        this.logger.info(`Task type: ${taskType}`);

        if (typeof prompt === 'string') {
            throw new Error('Bad prompt format');
        }

        const payload = await formatNovaImageGenerationPayload(taskType, prompt, options);
        const res = await this.invokeNovaCanvas(payload, options, signal);

        const decoder = new TextDecoder();
        const body = decoder.decode(res.body);
        const bedrockResult = JSON.parse(body);

        return {
            error: bedrockResult.error,
            result: bedrockResult.images.map((image: string) => ({
                type: 'image' as const,
                value: image,
            })),
        };
    }

    async startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob> {
        //convert options.params to Record<string, string>
        const params: Record<string, string> = {};
        for (const [key, value] of Object.entries(options.params || {})) {
            params[key] = String(value);
        }

        if (!this.options.training_bucket) {
            throw new Error(
                "Training cannot nbe used since the 'training_bucket' property was not specified in driver options",
            );
        }

        const s3 = new S3Client({ region: this.options.region, credentials: this.options.credentials });
        const stream = await dataset.getStream();
        const upload = await forceUploadFile(s3, stream, this.options.training_bucket, dataset.name);

        const service = this.getService();
        const response = await service.send(
            new CreateModelCustomizationJobCommand({
                jobName: `${options.name}-job`,
                customModelName: options.name,
                roleArn: this.options.training_role_arn || undefined,
                baseModelIdentifier: options.model,
                clientRequestToken: `llumiverse-${Date.now()}`,
                trainingDataConfig: {
                    s3Uri: `s3://${upload.Bucket}/${upload.Key}`,
                },
                outputDataConfig: undefined,
                hyperParameters: params,
                //TODO not supported?
                //customizationType: "FINE_TUNING",
            }),
        );

        const job = await service.send(
            new GetModelCustomizationJobCommand({
                jobIdentifier: response.jobArn,
            }),
        );

        // biome-ignore lint/style/noNonNullAssertion: jobArn is always returned by a successful CreateModelCustomizationJob response; AWS SDK types it optional
        return jobInfo(job, response.jobArn!);
    }

    async cancelTraining(jobId: string): Promise<TrainingJob> {
        const service = this.getService();
        await service.send(
            new StopModelCustomizationJobCommand({
                jobIdentifier: jobId,
            }),
        );
        const job = await service.send(
            new GetModelCustomizationJobCommand({
                jobIdentifier: jobId,
            }),
        );

        return jobInfo(job, jobId);
    }

    async getTrainingJob(jobId: string): Promise<TrainingJob> {
        const service = this.getService();
        const job = await service.send(
            new GetModelCustomizationJobCommand({
                jobIdentifier: jobId,
            }),
        );

        return jobInfo(job, jobId);
    }

    // ===================== management API ==================

    async validateConnection(): Promise<boolean> {
        const service = this.getService();
        this.logger.debug('[Bedrock] validating connection', service.config.credentials.name);
        //return true as if the client has been initialized, it means the connection is valid
        return true;
    }

    async listTrainableModels(): Promise<AIModel[]> {
        this.logger.debug('[Bedrock] listing trainable models');
        return this._listModels((m) =>
            m.customizationsSupported ? m.customizationsSupported.includes('FINE_TUNING') : false,
        );
    }

    async listModels(): Promise<AIModel[]> {
        this.logger.debug('[Bedrock] listing models');
        // exclude trainable models since they are not executable
        // exclude embedding models, not to be used for typical completions.
        const filter = (m: FoundationModelSummary) =>
            (m.inferenceTypesSupported?.includes('ON_DEMAND') && !m.outputModalities?.includes('EMBEDDING')) ?? false;
        return this._listModels(filter);
    }

    async _listModels(foundationFilter?: (m: FoundationModelSummary) => boolean): Promise<AIModel[]> {
        const service = this.getService();
        const [foundationModelsList, customModelsList, inferenceProfilesList] = await Promise.all([
            service.listFoundationModels({}).catch(() => {
                this.logger.warn(
                    "[Bedrock] Can't list foundation models. Check if the user has the right permissions.",
                );
                return undefined;
            }),
            service.listCustomModels({}).catch(() => {
                this.logger.warn("[Bedrock] Can't list custom models. Check if the user has the right permissions.");
                return undefined;
            }),
            service.listInferenceProfiles({}).catch(() => {
                this.logger.warn(
                    "[Bedrock] Can't list inference profiles. Check if the user has the right permissions.",
                );
                return undefined;
            }),
        ]);

        if (!foundationModelsList?.modelSummaries) {
            throw new Error('Foundation models not found');
        }

        let foundationModels = foundationModelsList.modelSummaries || [];
        if (foundationFilter) {
            foundationModels = foundationModels.filter(foundationFilter);
        }

        // Intentional allow-list: Bedrock spans several incompatible invocation schemas. Future versions from these
        // known Converse-compatible publishers remain visible, but do not add a new publisher until its request path
        // is verified. Per-model exclusions below are deterministic endpoint/schema incompatibilities, not guesses.
        const supportedPublishers = [
            'amazon',
            'anthropic',
            'cohere',
            'ai21',
            'mistral',
            'meta',
            'deepseek',
            'writer',
            'openai',
            'twelvelabs',
            'qwen',
            'google',
            'minimax',
            'moonshot',
            'moonshotai',
            'nvidia',
            'zai',
        ];
        const unsupportedModelsByPublisher = {
            amazon: ['nova-reel', 'nova-sonic', 'nova-2-sonic', 'titan-image-generator', 'rerank'],
            anthropic: [],
            cohere: ['rerank', 'embed'],
            ai21: [],
            mistral: [],
            meta: [],
            deepseek: [],
            writer: [],
            openai: [],
            twelvelabs: ['marengo'],
            qwen: [],
            google: [],
            minimax: [],
            moonshot: [],
            moonshotai: [],
            nvidia: [],
            zai: [],
        };

        // Helper function to check if model should be filtered out
        const shouldIncludeModel = (modelId?: string, providerName?: string): boolean => {
            if (!modelId || !providerName) return false;

            // Normalize punctuation so publisher display names such as "Z.AI" match the stable "zai" key.
            const normalizedProvider = providerName.toLowerCase().replace(/[^a-z0-9]/g, '');

            // Check if provider is supported
            const isProviderSupported = supportedPublishers.some((provider) => normalizedProvider.includes(provider));

            if (!isProviderSupported) return false;

            // Check if model is in the unsupported list for its provider
            for (const provider of supportedPublishers) {
                if (normalizedProvider.includes(provider)) {
                    const unsupportedModels =
                        unsupportedModelsByPublisher[provider as keyof typeof unsupportedModelsByPublisher] || [];
                    return !unsupportedModels.some((unsupported) => modelId.toLowerCase().includes(unsupported));
                }
            }

            return true;
        };

        foundationModels = foundationModels.filter((m) => shouldIncludeModel(m.modelId, m.providerName));

        const aiModels: AIModel[] = foundationModels.map((m) => {
            if (!m.modelId) {
                throw new Error('modelId not found');
            }

            const modelMetadata = resolveModelListingMetadata(m.modelArn ?? m.modelId, this.provider, {
                input_modalities: m.inputModalities,
                output_modalities: m.outputModalities,
            });

            const model: AIModel = {
                id: m.modelArn ?? m.modelId,
                name: m.modelName ?? m.modelId,
                provider: this.provider,
                owner: m.providerName,
                can_stream: m.responseStreamingSupported ?? false,
                ...modelMetadata,
            };

            return model;
        });

        //add custom models
        if (customModelsList?.modelSummaries) {
            customModelsList.modelSummaries.forEach((m) => {
                if (!m.modelArn) {
                    throw new Error('Model ID not found');
                }

                const capabilityModelId = m.baseModelName ?? m.modelArn;
                if (isEmbeddingModel({ id: capabilityModelId }, this.provider)) return;
                const modelMetadata = resolveModelListingMetadata(capabilityModelId, this.provider);

                const model: AIModel = {
                    id: m.modelArn,
                    name: m.modelName ?? m.modelArn,
                    provider: this.provider,
                    owner: 'custom',
                    description: `Custom model from ${m.baseModelName}`,
                    is_custom: true,
                    ...modelMetadata,
                };

                aiModels.push(model);
            });
        }

        //add inference profiles
        if (inferenceProfilesList?.inferenceProfileSummaries) {
            inferenceProfilesList.inferenceProfileSummaries.forEach((p) => {
                if (!p.inferenceProfileArn) {
                    throw new Error('Profile ARN not found');
                }

                // Apply the same filtering logic to inference profiles based on their name
                const profileId = p.inferenceProfileId || '';
                const profileName = p.inferenceProfileName || '';

                // Extract provider name from profile name or ID
                let providerName = '';
                for (const provider of supportedPublishers) {
                    if (profileName.toLowerCase().includes(provider) || profileId.toLowerCase().includes(provider)) {
                        providerName = provider;
                        break;
                    }
                }

                const modelMetadata = resolveModelListingMetadata(
                    p.inferenceProfileArn ?? p.inferenceProfileId,
                    this.provider,
                );

                if (
                    providerName &&
                    shouldIncludeModel(profileId, providerName) &&
                    !isEmbeddingModel(
                        {
                            id: p.inferenceProfileArn ?? p.inferenceProfileId,
                            input_modalities: modelMetadata.input_modalities,
                            output_modalities: modelMetadata.output_modalities,
                        },
                        this.provider,
                    )
                ) {
                    const model: AIModel = {
                        id: p.inferenceProfileArn ?? p.inferenceProfileId,
                        name: p.inferenceProfileName ?? p.inferenceProfileArn,
                        provider: this.provider,
                        owner: providerName,
                        ...modelMetadata,
                    };

                    aiModels.push(model);
                }
            });
        }

        return aiModels;
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return generateBedrockEmbeddings(this, options);
    }

    /**
     * Cleanup AWS SDK clients after the evicted driver has no active executions.
     */
    protected override destroyProviderResources(): void {
        this._executor?.destroy();
        this._service?.destroy();
        this._executor = undefined;
        this._service = undefined;
    }
}

function jobInfo(job: GetModelCustomizationJobCommandOutput, jobId: string): TrainingJob {
    const jobStatus = job.status;
    let status = TrainingJobStatus.running;
    let details: string | undefined;
    if (jobStatus === ModelCustomizationJobStatus.COMPLETED) {
        status = TrainingJobStatus.succeeded;
    } else if (jobStatus === ModelCustomizationJobStatus.FAILED) {
        status = TrainingJobStatus.failed;
        details = job.failureMessage || 'error';
    } else if (jobStatus === ModelCustomizationJobStatus.STOPPED) {
        status = TrainingJobStatus.cancelled;
    } else {
        status = TrainingJobStatus.running;
        details = jobStatus;
    }
    return {
        id: jobId,
        model: job.outputModelArn,
        status,
        details,
    };
}

function getToolDefinitions(tools?: readonly BedrockToolDefinition[]): Tool[] | undefined {
    return tools ? tools.map(getToolDefinition) : undefined;
}

function getToolDefinition(tool: BedrockToolDefinition): Tool.ToolSpecMember {
    return {
        toolSpec: {
            name: tool.name,
            description: tool.description,
            inputSchema: {
                json: tool.input_schema,
            } as NonNullable<NonNullable<Tool.ToolSpecMember['toolSpec']>['inputSchema']>,
        },
    };
}

/**
 * Checks whether any message contains toolUse or toolResult content blocks.
 */
export function messagesContainToolBlocks(messages: Message[]): boolean {
    for (const msg of messages) {
        if (!msg.content) continue;
        for (const block of msg.content) {
            if ((block as ContentBlock.ToolUseMember).toolUse || (block as ContentBlock.ToolResultMember).toolResult) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Converts toolUse and toolResult content blocks to text representations.
 * This preserves the tool call information in the conversation while removing
 * the structured tool blocks that require Bedrock's toolConfig to be set.
 *
 * Used when no tools are provided (e.g. checkpoint summary calls) but the
 * conversation history contains tool interactions from prior turns.
 */
export function convertToolBlocksToText(messages: Message[]): Message[] {
    return messages.map((msg) => {
        if (!msg.content) return msg;
        let hasToolBlocks = false;
        for (const block of msg.content) {
            if ((block as ContentBlock.ToolUseMember).toolUse || (block as ContentBlock.ToolResultMember).toolResult) {
                hasToolBlocks = true;
                break;
            }
        }
        if (!hasToolBlocks) return msg;

        const newContent: ContentBlock[] = [];
        for (const block of msg.content) {
            const toolUse = (block as ContentBlock.ToolUseMember).toolUse;
            const toolResult = (block as ContentBlock.ToolResultMember).toolResult;
            if (toolUse) {
                const inputStr = toolUse.input ? JSON.stringify(toolUse.input) : '';
                const truncatedInput = inputStr.length > 500 ? `${inputStr.substring(0, 500)}...` : inputStr;
                newContent.push({
                    text: `[Tool call: ${toolUse.name}(${truncatedInput})]`,
                } as ContentBlock.TextMember);
            } else if (toolResult) {
                const resultTexts: string[] = [];
                if (toolResult.content) {
                    for (const c of toolResult.content) {
                        if ('text' in c && typeof c.text === 'string') {
                            const text = c.text;
                            resultTexts.push(text.length > 500 ? `${text.substring(0, 500)}...` : text);
                        }
                    }
                }
                const resultStr = resultTexts.length > 0 ? resultTexts.join('\n') : 'No text content';
                newContent.push({
                    text: `[Tool result: ${resultStr}]`,
                } as ContentBlock.TextMember);
            } else {
                newContent.push(block);
            }
        }
        return { ...msg, content: newContent };
    });
}

/**
 * Recursively removes undefined values from an object.
 * AWS Bedrock's additionalModelRequestFields must be valid JSON, and undefined is not valid JSON.
 * Any unrecognized parameters will cause an exception.
 */
function removeUndefinedValues(obj: Record<string, unknown>): Record<string, unknown> {
    if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
        return obj;
    }

    const cleaned: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
        if (value !== undefined) {
            if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
                const cleanedNested = removeUndefinedValues(value as Record<string, unknown>);
                // Only include nested objects if they have properties after cleaning
                if (Object.keys(cleanedNested).length > 0) {
                    cleaned[key] = cleanedNested;
                }
            } else {
                cleaned[key] = value;
            }
        }
    }
    return cleaned;
}

/**
 * Update the conversation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: ConverseRequest, prompt: ConverseRequest): ConverseRequest {
    const combinedMessages = [...(conversation?.messages || []), ...(prompt.messages || [])];
    const combinedSystem = prompt.system || conversation?.system;

    // Fix both orphan directions before returning: a toolUse with no result
    // (interrupted run) gets a synthetic result; a toolResult with no matching
    // toolUse in the previous message (e.g. compaction-trimmed) is dropped. Either
    // would otherwise trip the Converse API's toolUse/toolResult pairing check.
    const fixedMessages = fixOrphanedToolResults(fixOrphanedToolUse(combinedMessages));

    return {
        modelId: prompt?.modelId || conversation?.modelId,
        messages: fixedMessages.length > 0 ? fixedMessages : [],
        system: combinedSystem && combinedSystem.length > 0 ? combinedSystem : undefined,
    };
}

function stripClaudeCachePoints(messages: Message[]): Message[] {
    return messages.map((message) => ({
        ...message,
        content: message.content?.filter((block) => !('cachePoint' in block)),
    }));
}

function stripClaudeCachePointsFromSystem(system?: ConverseRequest['system']): ConverseRequest['system'] | undefined {
    return (system?.filter((block) => !('cachePoint' in (block as object))) ?? undefined) as
        | ConverseRequest['system']
        | undefined;
}

function stripClaudeCachePointsFromTools(
    tools?: NonNullable<NonNullable<ConverseRequest['toolConfig']>['tools']>,
): NonNullable<NonNullable<ConverseRequest['toolConfig']>['tools']> | undefined {
    return (tools?.filter((tool) => !('cachePoint' in (tool as object))) ?? undefined) as
        | NonNullable<NonNullable<ConverseRequest['toolConfig']>['tools']>
        | undefined;
}

/**
 * Fix orphaned toolUse blocks in the conversation.
 *
 * When an agent is stopped mid-tool-execution, the assistant message contains toolUse blocks
 * but no corresponding toolResult was added. The AWS Converse API requires that every toolUse
 * must be followed by a toolResult in the next user message.
 *
 * This function detects such cases and injects synthetic toolResult blocks indicating
 * the tools were interrupted, allowing the conversation to continue.
 */
export function fixOrphanedToolUse(messages: Message[]): Message[] {
    if (messages.length < 2) return messages;

    const result: Message[] = [];

    for (let i = 0; i < messages.length; i++) {
        const current = messages[i];
        result.push(current);

        // Check if this is an assistant message with toolUse blocks
        if (current.role === 'assistant' && current.content) {
            // Extract toolUse blocks using simple property check (same pattern as existing Bedrock code)
            const toolUseBlocks: Array<{ toolUseId: string; name: string }> = [];
            for (const block of current.content) {
                if (block.toolUse?.toolUseId) {
                    toolUseBlocks.push({
                        toolUseId: block.toolUse.toolUseId,
                        name: block.toolUse.name ?? 'unknown',
                    });
                }
            }

            if (toolUseBlocks.length > 0) {
                // Check if the next message is a user message with matching toolResults
                const nextMessage = messages[i + 1];

                if (nextMessage && nextMessage.role === 'user' && nextMessage.content) {
                    // Get toolResult IDs from the next message using simple property check
                    const toolResultIds = new Set<string>();
                    for (const block of nextMessage.content) {
                        if (block.toolResult?.toolUseId) {
                            toolResultIds.add(block.toolResult.toolUseId);
                        }
                    }

                    // Find orphaned toolUse blocks (no matching toolResult)
                    const orphanedToolUse = toolUseBlocks.filter((tu) => !toolResultIds.has(tu.toolUseId));

                    if (orphanedToolUse.length > 0) {
                        // Inject synthetic toolResults for orphaned toolUse
                        const syntheticResults: ContentBlock[] = orphanedToolUse.map((tu) => ({
                            toolResult: {
                                toolUseId: tu.toolUseId,
                                content: [
                                    {
                                        text: `[Tool interrupted: The user stopped the operation before "${tu.name}" could execute.]`,
                                    },
                                ],
                            },
                        }));

                        // Prepend synthetic results to the next user message
                        const updatedNextMessage: Message = {
                            ...nextMessage,
                            content: [...syntheticResults, ...nextMessage.content],
                        };

                        // Replace the next message in our iteration
                        messages[i + 1] = updatedNextMessage;
                    }
                } else if (nextMessage && nextMessage.role === 'user' && !nextMessage.content) {
                    // Next message is a user message but has no content
                    // We need to add toolResults
                    const syntheticResults: ContentBlock[] = toolUseBlocks.map((tu) => ({
                        toolResult: {
                            toolUseId: tu.toolUseId,
                            content: [
                                {
                                    text: `[Tool interrupted: The user stopped the operation before "${tu.name}" could execute.]`,
                                },
                            ],
                        },
                    }));

                    const updatedNextMessage: Message = {
                        role: 'user',
                        content: syntheticResults,
                    };

                    messages[i + 1] = updatedNextMessage;
                }
                // Note: If there's no nextMessage, we leave the conversation as-is.
                // The toolUse blocks are expected to be there - the next turn will provide toolResults.
            }
        }
    }

    return result;
}

/**
 * Drop toolResult blocks whose toolUseId has no matching toolUse in the
 * immediately-preceding assistant message. Mirror of {@link fixOrphanedToolUse}:
 * that function synthesizes results for an unanswered toolUse (e.g. a cancelled
 * run); this one removes results left dangling after their toolUse was dropped
 * (e.g. by conversation compaction/trimming).
 *
 * Without this, the AWS Converse API rejects the request because every
 * toolResult must correspond to a toolUse in the previous message. Bedrock
 * conversations are already role-alternating, so a compaction that drops an
 * assistant toolUse turn leaves an orphaned toolResult (and a user/user
 * adjacency); dropping the orphan — and the now-empty message — repairs both.
 */
export function fixOrphanedToolResults(messages: Message[]): Message[] {
    if (messages.length === 0) return messages;
    const result: Message[] = [];
    for (let i = 0; i < messages.length; i++) {
        const message = messages[i];
        if (message.role !== 'user' || !message.content) {
            result.push(message);
            continue;
        }
        const hasToolResult = message.content.some((block) => block.toolResult);
        if (!hasToolResult) {
            result.push(message);
            continue;
        }
        const prev = messages[i - 1];
        const allowedIds = new Set<string>();
        if (prev && prev.role === 'assistant' && prev.content) {
            for (const block of prev.content) {
                if (block.toolUse?.toolUseId) allowedIds.add(block.toolUse.toolUseId);
            }
        }
        const filtered = message.content.filter((block) =>
            block.toolResult ? allowedIds.has(block.toolResult.toolUseId ?? '') : true,
        );
        // Drop the message if every block was an orphaned toolResult.
        if (filtered.length === 0) continue;
        result.push(filtered.length === message.content.length ? message : { ...message, content: filtered });
    }
    return result;
}
