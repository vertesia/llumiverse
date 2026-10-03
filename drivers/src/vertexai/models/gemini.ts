import type { ApiError } from '@google/genai';
import {
    type Content,
    FinishReason,
    FunctionCallingConfigMode,
    type FunctionDeclaration,
    type FunctionResponsePart,
    type GenerateContentConfig,
    type GenerateContentParameters,
    type GenerateContentResponse,
    type GenerateContentResponseUsageMetadata,
    HarmBlockThreshold,
    HarmCategory,
    MediaModality,
    Modality,
    type Part,
    ProminentPeople,
    type SafetyRating,
    type SafetySetting,
    type ThinkingConfig,
    ThinkingLevel,
    type Tool,
} from '@google/genai';
import {
    type ToolDefinition as CanonicalToolDefinition,
    canonicalJsonContentString,
    createStructuredOutputTransformationProof,
    type DecodedConversationResponse,
    fingerprintJson,
    isConversationDocumentFormat,
    type JsonValue,
    type NativeStreamPosition,
    parseConversationDocument,
    toolArgumentsForModel,
} from '@llumiverse/conversation';
import {
    type AIModel,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalHostCapabilities,
    type CanonicalStreamOpenOptions,
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    createCanonicalExecutionResponse,
    type DataSource,
    type DriverCompletionStream,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    FallbackCanonicalExecutionEventStream,
    isGeminiModelVersionGte,
    LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptOptions,
    PromptRole,
    type PromptSegment,
    readStreamAsBase64,
    type StatelessExecutionOptions,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    type ToolDefinition,
    type ToolUse,
    truncateLargeTextInConversation,
    type VertexAIGeminiOptions,
} from '@llumiverse/core';
import { asyncMap } from '@llumiverse/core/async';
import { canonicalNativeExecutionEventStream } from '../../conversation/canonical-execution-event-stream.js';
import {
    acceptedCanonicalResponse,
    assertAcceptedCanonicalRequest,
    canonicalConversationTurnNumber,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
    resolveConversationRuntime,
} from '../../conversation/canonical-runtime.js';
import {
    normalizeDecodedStructuredOutputForSchema,
    rejectDecodedStructuredOutput,
} from '../../conversation/structured-output.js';
import { boundedAudioStream, canonicalAudioAssetStorage, storeAudioResult } from '../../shared/audio.js';
import { truncateBinaryForDebug } from '../../shared/debug-prompt.js';
import { createToolChoiceConfigurationError } from '../../shared/tool-choice-error.js';
import type { GenerateContentPrompt, VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';
import { type GeminiContextCacheExecution, generateWithGeminiContextCache } from './gemini-context-cache.js';
import {
    appendGeminiCanonicalResponseWithProcessing,
    cleanGeminiPromptPart,
    compileGeminiConversation,
    decodeGeminiCanonicalResponse,
    finalizeGeminiPreparedRequest,
    formatGeminiFunctionResponse,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    geminiToolUsesFromContent,
    type PreparedGeminiConversation,
    prepareGeminiCanonicalContext,
    prepareGeminiCanonicalState,
} from './gemini-conversation-adapter.js';

type GoogleApiErrorLike = Pick<ApiError, 'status' | 'message'>;
type GeminiFinishReasonHandling = { message: string; retryable: boolean };

const geminiFinishReasonHandling: Partial<Record<FinishReason, GeminiFinishReasonHandling>> = {
    [FinishReason.SAFETY]: {
        message: 'Gemini blocked the response because it may violate safety policies.',
        retryable: false,
    },
    [FinishReason.RECITATION]: {
        message: 'Gemini blocked the response because it may reproduce protected source material.',
        retryable: false,
    },
    [FinishReason.LANGUAGE]: {
        message: 'Gemini stopped because the response used an unsupported language.',
        retryable: false,
    },
    [FinishReason.BLOCKLIST]: {
        message: 'Gemini blocked the response because it contains a forbidden term.',
        retryable: false,
    },
    [FinishReason.PROHIBITED_CONTENT]: {
        message: 'Gemini blocked the response because it may contain prohibited content.',
        retryable: false,
    },
    [FinishReason.SPII]: {
        message: 'Gemini blocked the response because it may contain sensitive personal information.',
        retryable: false,
    },
    [FinishReason.IMAGE_SAFETY]: {
        message: 'Gemini blocked the generated image because it may violate safety policies.',
        retryable: false,
    },
    [FinishReason.IMAGE_PROHIBITED_CONTENT]: {
        message: 'Gemini blocked the generated image because it may contain prohibited content.',
        retryable: false,
    },
    [FinishReason.IMAGE_RECITATION]: {
        message: 'Gemini blocked the generated image because it may reproduce protected source material.',
        retryable: false,
    },
    [FinishReason.MALFORMED_FUNCTION_CALL]: {
        message: 'Gemini generated an invalid function call.',
        retryable: true,
    },
    [FinishReason.NO_IMAGE]: { message: 'Gemini did not generate the requested image.', retryable: true },
    [FinishReason.IMAGE_OTHER]: {
        message: 'Gemini stopped image generation for an unspecified reason.',
        retryable: true,
    },
    [FinishReason.OTHER]: { message: 'Gemini stopped generation for an unspecified reason.', retryable: true },
};

class GeminiFinishReasonError extends Error {
    readonly retryable: boolean | undefined;

    constructor(
        readonly finishReason: FinishReason,
        readonly finishMessage?: string,
        readonly safetyRatings?: SafetyRating[],
    ) {
        super(formatGeminiFinishReasonErrorMessage(finishReason, finishMessage, safetyRatings));
        this.name = 'GeminiFinishReasonError';
        this.retryable = geminiFinishReasonHandling[finishReason]?.retryable;
    }
}

function formatGeminiFinishReasonErrorMessage(
    finishReason: FinishReason,
    finishMessage?: string,
    safetyRatings?: SafetyRating[],
): string {
    const summary =
        geminiFinishReasonHandling[finishReason]?.message ??
        'Gemini stopped generation with an unsupported finish reason.';

    const details = [`${summary} Finish reason: ${finishReason}.`];
    if (finishMessage) details.push(`Finish message: ${finishMessage}.`);
    if (safetyRatings?.length) details.push(`Safety ratings: ${JSON.stringify(safetyRatings)}.`);
    return details.join(' ');
}

function supportsStructuredOutput(options: PromptOptions): boolean {
    // Gemini 1.0 Ultra does not support JSON output, 1.0 Pro does.
    return !!options.result_schema && !options.model.includes('ultra');
}

export function resolveVertexAIServiceTier(modelOptions?: VertexAIGeminiOptions): string | undefined {
    return modelOptions?.service_tier ?? (modelOptions?.flex ? 'flex' : undefined);
}

export function normalizeVertexAIResolvedServiceTier(trafficType?: string): string | undefined {
    switch (trafficType) {
        case 'ON_DEMAND':
            return 'default';
        case 'ON_DEMAND_PRIORITY':
            return 'priority';
        case 'ON_DEMAND_FLEX':
            return 'flex';
        case 'PROVISIONED_THROUGHPUT':
            return 'provisioned';
        default:
            return undefined;
    }
}

const geminiSafetySettings: SafetySetting[] = [
    {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
];

function formatGeminiContentForDebug(content: Content): Content {
    return {
        ...content,
        parts: content.parts?.map((part) => {
            const cleaned = cleanGeminiPromptPart(part);
            if (!cleaned.inlineData?.data) {
                return cleaned;
            }
            return {
                ...cleaned,
                inlineData: {
                    ...cleaned.inlineData,
                    data: truncateBinaryForDebug(cleaned.inlineData.data),
                },
            } satisfies Part;
        }),
    };
}

export function formatGeminiDebugPrompt(prompt: GenerateContentPrompt): GenerateContentPrompt {
    return {
        ...prompt,
        contents: prompt.contents.map(formatGeminiContentForDebug),
        system: prompt.system ? formatGeminiContentForDebug(prompt.system) : undefined,
    };
}

// We do the mapping here rather than in common to avoid bringing the SDK into the common package.
function getProminentPeopleOption(
    prominentPeople?: 'PROMINENT_PEOPLE_UNSPECIFIED' | 'ALLOW_PROMINENT_PEOPLE' | 'BLOCK_PROMINENT_PEOPLE',
) {
    switch (prominentPeople) {
        case 'ALLOW_PROMINENT_PEOPLE':
            return ProminentPeople.ALLOW_PROMINENT_PEOPLE;
        case 'BLOCK_PROMINENT_PEOPLE':
            return ProminentPeople.BLOCK_PROMINENT_PEOPLE;
        case 'PROMINENT_PEOPLE_UNSPECIFIED':
            return ProminentPeople.PROMINENT_PEOPLE_UNSPECIFIED;
        default:
            return undefined;
    }
}

type GeminiToolDefinition = Pick<CanonicalToolDefinition, 'name' | 'description' | 'input_schema'> | ToolDefinition;

export function getGeminiPayload(
    options: ExecutionOptions,
    prompt: GenerateContentPrompt,
    operation: LlumiverseErrorContext['operation'] = 'execute',
    toolDefinitions: readonly GeminiToolDefinition[] | undefined = options.tools,
): GenerateContentParameters {
    const model_options = options.model_options as
        | (VertexAIGeminiOptions & {
              tool_choice?: 'auto' | 'none' | 'any' | 'required';
              required_tool_name?: string;
          })
        | undefined;
    const tools = getToolDefinitions(toolDefinitions);
    const forcedToolRequested =
        model_options?.required_tool_name !== undefined ||
        model_options?.tool_choice === 'required' ||
        model_options?.tool_choice === 'any';
    if (forcedToolRequested && !tools) {
        throw createToolChoiceConfigurationError(
            '[Vertex AI Gemini] A required tool choice was requested, but no tools are available.',
            { provider: 'vertexai', model: options.model, operation },
        );
    }

    // When no tools are provided but conversation contains functionCall/functionResponse parts
    // (e.g. checkpoint summary calls), convert them to text to avoid API errors.
    // Use a local variable to avoid mutating the caller's conversation object.
    let payloadContents = mergeFunctionResponseContents(
        (prompt.contents ?? []).map((content) => ({
            ...content,
            parts: content.parts?.map((part) => cleanGeminiPromptPart(part)),
        })),
    );
    if (!tools && payloadContents) {
        const hasToolParts = payloadContents.some((c) => c.parts?.some((p) => p.functionCall || p.functionResponse));
        if (hasToolParts) {
            payloadContents = convertGeminiFunctionPartsToText(payloadContents);
        }
    }
    // Drop functionResponse parts whose functionCall was lost (e.g. compaction),
    // which would otherwise trip Gemini's functionCall/functionResponse pairing.
    if (payloadContents) {
        payloadContents = fixOrphanedToolResults(payloadContents);
    }

    const useStructuredOutput = supportsStructuredOutput(options) && !tools;

    const configNanoBanana: GenerateContentConfig = {
        systemInstruction: prompt.system,
        safetySettings: geminiSafetySettings,
        responseModalities: [Modality.TEXT, Modality.IMAGE], // This is an error if only Text, and Only Image just gets blank responses.
        candidateCount: 1,
        //Model options
        temperature: model_options?.temperature,
        topP: model_options?.top_p,
        maxOutputTokens: model_options?.max_tokens,
        stopSequences: model_options?.stop_sequence,
        thinkingConfig: geminiThinkingConfig(options),
        labels: options.labels,
        imageConfig: {
            imageSize: model_options?.image_size,
            aspectRatio: model_options?.image_aspect_ratio,
            personGeneration: model_options?.person_generation,
            prominentPeople: getProminentPeopleOption(model_options?.prominent_people),
            outputMimeType: model_options?.output_mime_type,
            outputCompressionQuality: model_options?.output_compression_quality,
        },
    };

    const config: GenerateContentConfig = {
        systemInstruction: prompt.system,
        safetySettings: geminiSafetySettings,
        tools: tools ? [tools] : undefined,
        toolConfig: tools
            ? {
                  functionCallingConfig: {
                      mode:
                          model_options?.tool_choice === 'none'
                              ? FunctionCallingConfigMode.NONE
                              : model_options?.tool_choice === 'required' || model_options?.tool_choice === 'any'
                                ? FunctionCallingConfigMode.ANY
                                : FunctionCallingConfigMode.AUTO,
                      ...(model_options?.required_tool_name
                          ? { allowedFunctionNames: [model_options.required_tool_name] }
                          : {}),
                  },
              }
            : undefined,
        candidateCount: 1,
        //JSON/Structured output
        responseMimeType: useStructuredOutput ? 'application/json' : undefined,
        responseJsonSchema: useStructuredOutput ? options.result_schema : undefined,
        //Model options
        temperature: model_options?.temperature,
        topP: model_options?.top_p,
        topK: model_options?.top_k,
        maxOutputTokens: model_options?.max_tokens,
        stopSequences: model_options?.stop_sequence,
        presencePenalty: model_options?.presence_penalty,
        frequencyPenalty: model_options?.frequency_penalty,
        seed: model_options?.seed,
        thinkingConfig: geminiThinkingConfig(options),
        labels: options.labels,
    };

    return {
        model: options.model,
        contents: payloadContents,
        config: options.model.toLowerCase().includes('image') ? configNanoBanana : config,
    };
}

/**
 * Collect all parts (text and images) from content in order.
 * This preserves the original ordering of text and image parts.
 */
function extractCompletionResults(content: Content, includeThoughts = true): CompletionResult[] {
    const results: CompletionResult[] = [];
    const parts = content.parts;
    if (parts) {
        for (const part of parts) {
            if (part.text) {
                if (part.thought) {
                    if (includeThoughts) results.push({ type: 'thoughts', value: part.text });
                } else {
                    results.push({ type: 'text', value: part.text });
                }
            } else if (part.inlineData) {
                if (part.inlineData.mimeType?.startsWith('audio/'))
                    throw new Error('Audio output requires a file speech model');
                const base64ImageBytes: string = part.inlineData.data ?? '';
                const mimeType = part.inlineData.mimeType ?? 'image/png';
                const imageUrl = `data:${mimeType};base64,${base64ImageBytes}`;
                results.push({
                    type: 'image',
                    value: imageUrl,
                });
            }
        }
    }
    return results;
}

function preserveGeminiSignedSubtree(value: unknown): boolean {
    if (!value || typeof value !== 'object') return false;
    const thoughtSignature = (value as { thoughtSignature?: unknown }).thoughtSignature;
    return typeof thoughtSignature === 'string' && thoughtSignature.length > 0;
}

function projectGeminiHistoryContent(content: Content, options: ExecutionOptions, currentTurn: number): Content {
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: options.stripTextMaxTokens,
        preserveSubtree: preserveGeminiSignedSubtree,
    };
    let projected = stripBase64ImagesFromConversation(content, stripOptions);
    projected = truncateLargeTextInConversation(projected, stripOptions);
    projected = stripHeartbeatsFromConversation(projected, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
        preserveSubtree: preserveGeminiSignedSubtree,
    });
    return projected as Content;
}

function prepareCanonicalGeminiProjection(
    prepared: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
    options: ExecutionOptions,
    contextOnly = false,
): GenerateContentPrompt {
    const currentIndexes = new Set(prepared.current_native_content_indexes);
    const currentTurn = canonicalConversationTurnNumber(prepared.document);
    const projected: GenerateContentPrompt = {
        contents: prepared.native_conversation.contents.map((content, index) =>
            currentIndexes.has(index) ? content : projectGeminiHistoryContent(content, options, currentTurn),
        ),
        ...(prepared.native_conversation.system === undefined ? {} : { system: prepared.native_conversation.system }),
    };
    if (!contextOnly || options.result_schema === undefined) return projected;

    const hasTools = prepared.tool_definitions.length > 0;
    const instruction =
        supportsStructuredOutput(options) && !hasTools
            ? 'Fill all appropriate fields in the JSON output.'
            : hasTools
              ? `When not calling tools, the output must be a JSON object using the following JSON Schema:\n${JSON.stringify(options.result_schema)}`
              : `The output must be a JSON object using the following JSON Schema:\n${JSON.stringify(options.result_schema)}`;
    return {
        ...projected,
        system: {
            ...(projected.system ?? { role: 'user' }),
            parts: [...(projected.system?.parts ?? []), { text: instruction }],
        },
    };
}

function reportedGeminiUsage(value: unknown): GenerateContentResponseUsageMetadata | undefined {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined;
    const record = value as Record<string, unknown>;
    for (const key of [
        'cachedContentTokenCount',
        'candidatesTokenCount',
        'promptTokenCount',
        'thoughtsTokenCount',
        'toolUsePromptTokenCount',
        'totalTokenCount',
    ]) {
        const candidate = record[key];
        if (candidate !== undefined && (typeof candidate !== 'number' || !Number.isSafeInteger(candidate))) {
            return undefined;
        }
    }
    if (record.trafficType !== undefined && typeof record.trafficType !== 'string') return undefined;
    return record as GenerateContentResponseUsageMetadata;
}

function canonicalGeminiUsage(
    prepared: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
    definition: GeminiModelDefinition,
    driver: VertexAIDriver,
): ExecutionTokenUsage | undefined {
    const usage = prepared.accepted_response?.generation.usage;
    if (usage === undefined) return undefined;
    const reported = usage.reported_usage?.find(
        (candidate) => candidate.source === 'provider' && candidate.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL,
    );
    const native = reportedGeminiUsage(reported?.payload);
    if (native !== undefined) return definition.usageMetadataToTokenUsage(driver, native);
    return {
        ...(usage.input_tokens === undefined ? {} : { prompt: usage.input_tokens }),
        ...(usage.output_tokens === undefined ? {} : { result: usage.output_tokens }),
        ...(usage.total_tokens === undefined ? {} : { total: usage.total_tokens }),
        ...(usage.cache_read_tokens === undefined ? {} : { prompt_cached: usage.cache_read_tokens }),
        ...(usage.input_new_tokens === undefined ? {} : { prompt_new: usage.input_new_tokens }),
    };
}

function canonicalGeminiServiceTier(
    prepared: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
): string | undefined {
    const usage = prepared.accepted_response?.generation.usage;
    const reported = usage?.reported_usage?.find(
        (candidate) => candidate.source === 'provider' && candidate.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL,
    );
    return normalizeVertexAIResolvedServiceTier(reportedGeminiUsage(reported?.payload)?.trafficType);
}

function acceptedGeminiContent(
    prepared: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
): Content {
    const accepted = prepared.accepted_response;
    if (accepted === undefined) throw new Error('No accepted Gemini response is available');
    const compiled = compileGeminiConversation(prepared.document, {
        provider: prepared.provider,
        model: prepared.requested_model,
    });
    const mapping = compiled.mappings.find(
        (candidate) => candidate.kind === 'turn' && candidate.canonical_id === accepted.turn.id,
    );
    const match = mapping === undefined ? undefined : /^contents\/(\d+)$/.exec(mapping.native_id);
    const content =
        match === undefined || match === null ? undefined : compiled.conversation.contents[Number(match[1])];
    if (content?.role !== 'model') {
        throw new Error(`Accepted Gemini turn ${accepted.turn.id} has no native model projection`);
    }
    return content;
}

async function recoverGeminiCompletion(
    prepared: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
    definition: GeminiModelDefinition,
    driver: VertexAIDriver,
    options: ExecutionOptions,
    includeThoughts: boolean,
): Promise<Completion> {
    const accepted = prepared.accepted_response;
    if (accepted === undefined) throw new Error('No accepted Gemini response is available');
    if (options.include_original_response) {
        throw new Error('An idempotently recovered Gemini response cannot reconstruct original_response');
    }
    const content = acceptedGeminiContent(prepared);
    const toolUse = await geminiToolUsesFromContent(content, prepared.runtime.response_operation_id);
    return {
        result: extractCompletionResults(content, includeThoughts),
        ...(toolUse === undefined ? {} : { tool_use: toolUse }),
        token_usage: canonicalGeminiUsage(prepared, definition, driver),
        service_tier: canonicalGeminiServiceTier(prepared),
        finish_reason: toolUse === undefined ? accepted.generation.finish_reason : 'tool_use',
        conversation: prepared.document,
    };
}

function recoveredGeminiStream(completion: Completion): DriverCompletionStream {
    const stream = (async function* (): AsyncIterable<CompletionChunkObject> {
        yield {
            result: completion.result,
            tool_use: completion.tool_use,
            token_usage: completion.token_usage,
            service_tier: completion.service_tier,
            finish_reason: completion.finish_reason,
        };
    })();
    return Object.assign(stream, { finalizeConversation: () => completion.conversation });
}

function appendGeminiStreamPart(target: Part[], part: Part): number {
    const previous = target.at(-1);
    const canMergeText =
        typeof part.text === 'string' &&
        part.text.length > 0 &&
        typeof previous?.text === 'string' &&
        !previous.thoughtSignature &&
        !part.thoughtSignature &&
        !!previous.thought === !!part.thought;
    if (canMergeText && previous) {
        previous.text = (previous.text ?? '') + part.text;
        return target.length - 1;
    }
    target.push(structuredClone(part));
    return target.length - 1;
}

function appendGeminiStreamParts(target: Part[], incoming: Part[]): void {
    for (const part of incoming) appendGeminiStreamPart(target, part);
}

interface GeminiCanonicalDraft {
    draft_block_id: string;
    native_position: NativeStreamPosition;
    kind: 'text' | 'reasoning' | 'tool_call' | 'image' | 'audio' | 'video' | 'document';
    text: string;
    tool_arguments?: JsonValue;
}

function geminiStreamPosition(partIndex: number, nativeItemId?: string): NativeStreamPosition {
    return {
        protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
        path: ['candidates', 0, 'content', 'parts', partIndex],
        ...(nativeItemId === undefined ? {} : { native_item_id: nativeItemId }),
    };
}

function geminiPromptFeedbackPosition(): NativeStreamPosition {
    return {
        protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
        path: ['promptFeedback', 'blockReasonMessage'],
    };
}

function geminiSemanticBlocks(decoded: DecodedConversationResponse, turnId: string) {
    const turn = decoded.turns.find((candidate) => candidate.id === turnId);
    if (turn?.kind !== 'agent') throw new Error('Gemini stream decode has no generated agent turn');
    return turn.blocks.filter(
        (block) => block.type !== 'native_replay' && block.type !== 'extension' && block.type !== 'external_reference',
    );
}

function geminiSemanticPositions(content: Content): NativeStreamPosition[] {
    return (content.parts ?? []).flatMap((part, partIndex) => {
        if (typeof part.text === 'string') {
            return part.text.length === 0 ? [] : [geminiStreamPosition(partIndex)];
        }
        if (part.functionCall !== undefined) {
            return [geminiStreamPosition(partIndex, part.functionCall.id)];
        }
        if (part.inlineData !== undefined || part.fileData !== undefined) return [geminiStreamPosition(partIndex)];
        return [];
    });
}

function geminiDraftMediaKind(
    mimeType: string | undefined,
): Extract<GeminiCanonicalDraft['kind'], 'image' | 'audio' | 'video' | 'document'> {
    if (mimeType?.startsWith('image/')) return 'image';
    if (mimeType?.startsWith('audio/')) return 'audio';
    if (mimeType?.startsWith('video/')) return 'video';
    return 'document';
}

function assertGeminiToolDraftArguments(
    draft: GeminiCanonicalDraft,
    block: ReturnType<typeof geminiSemanticBlocks>[number],
): void {
    if (draft.kind !== 'tool_call' || block.type !== 'tool_call' || block.arguments.type === 'invalid') return;
    if (draft.tool_arguments === undefined) return;
    if (
        canonicalJsonContentString(draft.tool_arguments) !==
        canonicalJsonContentString(toolArgumentsForModel(block.arguments))
    ) {
        throw new Error('Gemini tool argument snapshot differs from its terminal function call');
    }
}

/** True when `content` is a user turn holding nothing but functionResponse parts. */
function isFunctionResponseOnlyContent(content: Content): boolean {
    return content.role === 'user' && !!content.parts?.length && content.parts.every((part) => part.functionResponse);
}

/**
 * Recombine runs of consecutive user contents that hold nothing but functionResponse parts into a
 * single user turn. The prompt builder emits one content per tool-result segment, but Gemini
 * requires every response to a model function-call turn to arrive in ONE user turn whose
 * functionResponse count equals the call count — split parallel results are rejected with 400
 * INVALID_ARGUMENT ("Please ensure that the number of function response parts is equal to the
 * number of function call parts of the function call turn"). Only function-response contents are
 * merged: text segments keep their boundaries, which are explicit-cache breakpoints
 * (see gemini-context-cache.ts) — and a cached prefix only ever holds static text parts, so this
 * merge can never move the prefix boundary.
 */
export function mergeFunctionResponseContents(contents: Content[]): Content[] {
    const result: Content[] = [];
    for (const content of contents) {
        const previous = result.at(-1);
        if (previous && isFunctionResponseOnlyContent(previous) && isFunctionResponseOnlyContent(content)) {
            result[result.length - 1] = {
                ...previous,
                parts: [...(previous.parts ?? []), ...(content.parts ?? [])],
            };
        } else {
            result.push(content);
        }
    }
    return result;
}

/**
 * Drop functionResponse parts whose name has no matching functionCall in the
 * immediately-preceding `model` content. Gemini pairs a functionResponse to its
 * functionCall by name; a response left dangling after its call was dropped
 * (e.g. by conversation compaction/trimming, or an unmergeable parallel batch)
 * causes the API to reject the request. Mirrors the same guard added to the
 * Claude, Bedrock, and OpenAI drivers.
 *
 * The matching model call set remains active across a run of user function-response contents, so
 * split parallel tool results are not mistaken for orphans even when this runs on contents that
 * have not been through mergeFunctionResponseContents.
 */
export function fixOrphanedToolResults(contents: Content[]): Content[] {
    if (contents.length === 0) return contents;
    const result: Content[] = [];
    let allowedNames = new Set<string>();
    for (const content of contents) {
        if (content.role === 'model') {
            allowedNames = new Set(
                (content.parts ?? []).flatMap((part) => (part.functionCall?.name ? [part.functionCall.name] : [])),
            );
            result.push(content);
            continue;
        }
        if (content.role !== 'user' || !content.parts) {
            allowedNames = new Set();
            result.push(content);
            continue;
        }
        const hasFunctionResponse = content.parts.some((part) => part.functionResponse);
        if (!hasFunctionResponse) {
            allowedNames = new Set();
            result.push(content);
            continue;
        }
        const filtered = content.parts.filter((part) =>
            part.functionResponse ? allowedNames.has(part.functionResponse.name ?? '') : true,
        );
        // Drop the content if every part was an orphaned functionResponse.
        if (filtered.length === 0) continue;
        result.push(filtered.length === content.parts.length ? content : { ...content, parts: filtered });
    }
    return result;
}

const supportedFinishReasons: FinishReason[] = [
    FinishReason.MAX_TOKENS,
    FinishReason.STOP,
    FinishReason.FINISH_REASON_UNSPECIFIED,
];

// Finish reasons that indicate tool call issues but should be recovered gracefully
// instead of throwing an error. The tool_use is still extracted and returned
// so the workflow can generate a proper toolError response.
const recoverableToolCallReasons = [FinishReason.UNEXPECTED_TOOL_CALL];

function isRecoverableGeminiFinishReason(finishReason: FinishReason | undefined): boolean {
    return finishReason !== undefined && recoverableToolCallReasons.includes(finishReason);
}

function assertSupportedGeminiFinishReason(candidate: {
    finishReason?: FinishReason;
    finishMessage?: string;
    safetyRatings?: SafetyRating[];
}): boolean {
    const isRecoverableToolCall = isRecoverableGeminiFinishReason(candidate.finishReason);
    if (candidate.finishReason && !supportedFinishReasons.includes(candidate.finishReason) && !isRecoverableToolCall) {
        throw new GeminiFinishReasonError(candidate.finishReason, candidate.finishMessage, candidate.safetyRatings);
    }
    return isRecoverableToolCall;
}

function geminiThinkingLevelForEffort(effort: VertexAIGeminiOptions['effort']): ThinkingLevel | undefined {
    switch (effort) {
        case 'minimal':
            return ThinkingLevel.MINIMAL;
        case 'low':
            return ThinkingLevel.LOW;
        case 'medium':
            return ThinkingLevel.MEDIUM;
        case 'high':
            return ThinkingLevel.HIGH;
        default:
            return undefined;
    }
}

function geminiBudgetForEffort(model: string, effort: NonNullable<VertexAIGeminiOptions['effort']>): number {
    const isFlashLite = model.includes('flash-lite');
    const isFlash = model.includes('flash') && !isFlashLite;
    const isPro = model.includes('pro');

    if (effort === 'minimal') {
        if (isPro) return 128;
        if (isFlashLite) return 512;
        if (isFlash) return 1;
        return 1024;
    }
    if (effort === 'low') {
        if (isPro) return 128;
        if (isFlashLite) return 512;
        if (isFlash) return 1;
        return 1024;
    }
    if (effort === 'medium') {
        return 8192;
    }
    if (isPro) return 32768;
    if (isFlash || isFlashLite) return 24576;
    return 8192;
}

export function geminiThinkingConfig(option: StatelessExecutionOptions): ThinkingConfig | undefined {
    const model_options = option.model_options as VertexAIGeminiOptions | undefined;

    // If thinking options are explicitly set in model options, use them directly
    const include_thoughts = model_options?.include_thoughts !== false;
    if (model_options?.thinking_budget_tokens !== undefined || model_options?.thinking_level) {
        if (model_options.thinking_budget_tokens === 0 && !model_options.thinking_level) return undefined;
        return {
            includeThoughts: true,
            ...(model_options.thinking_budget_tokens !== undefined && {
                thinkingBudget: model_options.thinking_budget_tokens,
            }),
            ...(model_options.thinking_level && { thinkingLevel: model_options.thinking_level }),
        };
    }
    if (model_options?.effort) {
        if (isGeminiModelVersionGte(option.model, '3.0')) {
            return {
                includeThoughts: include_thoughts,
                thinkingLevel: geminiThinkingLevelForEffort(model_options.effort),
            };
        }
        return {
            includeThoughts: include_thoughts,
            thinkingBudget: geminiBudgetForEffort(option.model, model_options.effort),
        };
    }

    // When no thinking control is supplied, preserve the provider's model-specific default.
    if (model_options?.include_thoughts !== undefined) {
        return { includeThoughts: include_thoughts };
    }
}

function isFileAudioModel(model: string): boolean {
    return /(?:tts|transcribe)/.test(model) && !/(?:live|native-audio)/.test(model);
}

function normalizeGeminiFinishReason(finishReason: FinishReason | undefined): string | undefined {
    switch (finishReason) {
        case FinishReason.MAX_TOKENS:
            return 'length';
        case FinishReason.STOP:
            return 'stop';
        default:
            return finishReason;
    }
}

function geminiProvider(driver: VertexAIDriver): string {
    return typeof driver.provider === 'string' && driver.provider.length > 0 ? driver.provider : 'vertexai';
}

function geminiFileAudioRequest(
    prompt: GenerateContentPrompt,
    options: ExecutionOptions,
    modelName: string,
): {
    model_options: VertexAIGeminiOptions | undefined;
    payload: GenerateContentParameters;
    speech: boolean;
} {
    const modelOptions = options.model_options as VertexAIGeminiOptions | undefined;
    const speech = modelName.includes('tts');
    const config: GenerateContentConfig = speech
        ? {
              responseModalities: [Modality.AUDIO],
              speechConfig: {
                  languageCode: modelOptions?.speech_language,
                  voiceConfig: { prebuiltVoiceConfig: { voiceName: modelOptions?.speech_voice ?? 'Kore' } },
              },
          }
        : {
              systemInstruction: prompt.system,
              audioTranscriptionConfig: {
                  languageCodes: modelOptions?.transcription_language_codes,
                  diarization: modelOptions?.transcription_diarization,
                  wordTimestamp: modelOptions?.transcription_word_timestamps,
                  customVocabulary: modelOptions?.transcription_vocabulary,
              },
          };
    const contents = speech
        ? [
              {
                  role: 'user',
                  parts: [
                      ...(prompt.system?.parts ?? []),
                      ...prompt.contents.flatMap((content) => content.parts ?? []),
                  ],
              },
          ]
        : prompt.contents;
    return { model_options: modelOptions, payload: { model: modelName, contents, config }, speech };
}

function geminiFileAudioTransportRequest(
    payload: GenerateContentParameters,
    signal: AbortSignal | undefined,
): GenerateContentParameters {
    if (signal === undefined) return payload;
    signal.throwIfAborted();
    // AbortSignal is SDK transport state, not provider JSON. Keep it out of the exact request
    // fingerprint. Experimental receipts that included abortSignal:{} remain incompatible rather
    // than being silently relabeled as a signal-free request.
    return { ...payload, config: { ...payload.config, abortSignal: signal } };
}

export class GeminiModelDefinition implements ModelDefinition<GenerateContentPrompt> {
    readonly canonical_conversation_supported = true;
    model: AIModel;

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: isFileAudioModel(modelId) ? ModelType.Audio : ModelType.Text,
            can_stream: !isFileAudioModel(modelId),
        } satisfies AIModel;
    }

    async createPrompt(
        _driver: VertexAIDriver,
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<GenerateContentPrompt> {
        const splits = options.model.split('/');
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        if (isFileAudioModel(modelName)) {
            if (
                (options.conversation !== undefined && !isConversationDocumentFormat(options.conversation)) ||
                options.tools?.length ||
                options.result_schema ||
                options.format
            ) {
                throw new Error(
                    'File audio operations do not accept conversation, tools, result schemas, or custom formatting',
                );
            }
            if (segments.some((segment) => segment.role === PromptRole.tool || segment.role === PromptRole.assistant)) {
                throw new Error('File audio operations accept only user and system input');
            }
            const files = segments.flatMap((segment) => segment.files ?? []);
            if (modelName.includes('tts')) {
                const text = segments
                    .map((segment) => segment.content ?? '')
                    .join('\n')
                    .trim();
                if (files.length || !text || text.length > 4096) {
                    throw new Error('Speech synthesis requires 1–4096 characters and no files');
                }
                if (!options.store_audio) throw new Error('Speech synthesis requires a durable audio storage sink');
            } else if (files.length !== 1 || !files[0].mime_type.startsWith('audio/')) {
                throw new Error('Transcription requires exactly one audio file');
            }
        }

        const schema = options.result_schema;
        let contents: Content[] = [];
        let system: Content | undefined = { role: 'user', parts: [] }; // Single content block for system messages

        const safety: Content[] = [];

        for (const msg of segments) {
            // Role specific handling
            if (msg.role === PromptRole.system) {
                // Text only for system messages
                if (msg.files && msg.files.length > 0) {
                    throw new Error(
                        'Gemini does not support files/images etc. in system messages. Only text content is allowed.',
                    );
                }

                if (msg.content) {
                    system.parts?.push({
                        text: msg.content,
                    });
                }
            } else if (msg.role === PromptRole.tool) {
                if (!msg.tool_use_id) {
                    throw new Error('Tool response missing tool_use_id');
                }
                // A tool result can carry attachments - typically an image the tool rendered or
                // promoted into the conversation. Gemini takes those as `FunctionResponse.parts`;
                // sending the JSON response alone drops them and leaves the model unable to see
                // what it just asked for.
                const responseParts: FunctionResponsePart[] = [];
                for (const f of msg.files ?? []) {
                    responseParts.push(await fileToMediaPart(f));
                }
                // Build functionResponse part with optional thought_signature for Gemini thinking models
                const functionResponsePart: Part & {
                    _llumiverse_tool_result_status?: NonNullable<PromptSegment['tool_result_status']>;
                    _llumiverse_tool_result_text?: string;
                } = {
                    functionResponse: {
                        id: msg.tool_use_id,
                        response: formatGeminiFunctionResponse(msg.content ?? ''),
                        ...(responseParts.length > 0 && { parts: responseParts }),
                    },
                    // Include thought_signature if provided (required for Gemini 2.5+/3.0+ thinking models)
                    thoughtSignature: msg.thought_signature,
                    ...(msg.tool_result_status === undefined
                        ? {}
                        : { _llumiverse_tool_result_status: msg.tool_result_status }),
                    _llumiverse_tool_result_text: msg.content ?? '',
                };
                contents.push({
                    role: 'user',
                    parts: [functionResponsePart],
                });
            } else {
                // PromptRole.user, PromptRole.assistant, PromptRole.safety
                const parts: Part[] = [];
                // Text content handling
                if (msg.content) {
                    parts.push({
                        text: msg.content,
                    });
                }

                // File content handling
                if (msg.files) {
                    for (const f of msg.files) {
                        parts.push(await fileToMediaPart(f));
                    }
                }

                if (parts.length > 0) {
                    if (msg.role === PromptRole.safety) {
                        safety.push({
                            role: 'user',
                            parts,
                        });
                    } else {
                        contents.push({
                            role: msg.role === PromptRole.assistant ? 'model' : 'user',
                            parts,
                        });
                    }
                }
            }
        }

        // Adding JSON Schema to system message
        if (schema) {
            if (supportsStructuredOutput(options) && !options.tools) {
                // Gemini structured output is unnecessarily sparse. Adding encouragement to fill the fields.
                // Putting JSON in prompt is not recommended by Google, when using structured output.
                system.parts?.push({ text: 'Fill all appropriate fields in the JSON output.' });
            } else {
                // Fallback to putting the schema in the system instructions, if not using structured output.
                if (options.tools) {
                    system.parts?.push({
                        text: `When not calling tools, the output must be a JSON object using the following JSON Schema:\n${JSON.stringify(schema)}`,
                    });
                } else {
                    system.parts?.push({
                        text: `The output must be a JSON object using the following JSON Schema:\n${JSON.stringify(schema)}`,
                    });
                }
            }
        }

        // If no system messages, set system to undefined.
        if (!system.parts || system.parts.length === 0) {
            system = undefined;
        }

        // Add safety messages to the end of contents. They are in effect user messages that come at the end.
        if (safety.length > 0) {
            contents = contents.concat(safety);
        }

        // Preserve PromptSegment boundaries through the provider request. Besides retaining the
        // explicit-cache breakpoint, this avoids changing the caller's conversation turn shape.
        return { contents, system };
    }

    usageMetadataToTokenUsage(
        driver: VertexAIDriver,
        usageMetadata: GenerateContentResponseUsageMetadata | undefined,
    ): ExecutionTokenUsage {
        if (!usageMetadata?.totalTokenCount) {
            return {};
        }
        const tokenUsage: ExecutionTokenUsage = {
            total: usageMetadata.totalTokenCount,
            prompt: usageMetadata.promptTokenCount,
            prompt_cached: usageMetadata.cachedContentTokenCount ?? undefined,
            prompt_new: (usageMetadata.promptTokenCount ?? 0) - (usageMetadata.cachedContentTokenCount ?? 0),
        };

        //Output/Response side
        tokenUsage.result =
            (usageMetadata.candidatesTokenCount ?? 0) +
            (usageMetadata.thoughtsTokenCount ?? 0) +
            (usageMetadata.toolUsePromptTokenCount ?? 0);

        if ((tokenUsage.total ?? 0) !== (tokenUsage.prompt ?? 0) + tokenUsage.result) {
            // Token-accounting mismatch: warn-level diagnostic (the call still
            // returns the best-effort tokenUsage). Use the driver's structured
            // logger so we don't promote stderr writes to ERROR in serverless
            // log aggregators — see the recoverable-tool-call sites below.
            driver.logger.warn(
                { total: tokenUsage.total, prompt: tokenUsage.prompt, result: tokenUsage.result },
                '[VertexAI] Gemini token usage mismatch: total does not equal prompt + result',
            );
        }

        if (!tokenUsage.result) {
            tokenUsage.result = undefined; // If no result, mark as undefined
        }

        // Generated images are part of the candidate tokens, priced at their own rate.
        const imageTokens = usageMetadata.candidatesTokensDetails?.find(
            (detail) => detail.modality === MediaModality.IMAGE,
        )?.tokenCount;
        if (imageTokens) {
            tokenUsage.result_image = imageTokens;
        }

        return tokenUsage;
    }

    async requestCanonicalTextCompletion(
        driver: VertexAIDriver,
        prompt: GenerateContentPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionResponse> {
        const requestedOptions = options;
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) region = splits[1];
        const modelName = splits[splits.length - 1];
        const fileAudioRequest = isFileAudioModel(modelName)
            ? geminiFileAudioRequest(prompt, requestedOptions, modelName)
            : undefined;
        if (isFileAudioModel(modelName) && isConversationDocumentFormat(requestedOptions.conversation)) {
            const document = parseConversationDocument(requestedOptions.conversation);
            const runtime = resolveConversationRuntime(requestedOptions);
            if (
                requestedOptions.conversation_runtime?.conversation_id !== undefined &&
                runtime.conversation_id !== document.id
            ) {
                throw new Error('conversation_runtime.conversation_id does not match the canonical document');
            }
            const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
            if (accepted === undefined && document.revision !== 0) {
                throw new Error('Gemini file audio does not support conversation continuation');
            }
            if (accepted !== undefined) {
                if (
                    accepted.generation.request_id !== runtime.request_id ||
                    accepted.generation.provider !== geminiProvider(driver) ||
                    accepted.generation.protocol !== GEMINI_GENERATE_CONTENT_PROTOCOL ||
                    accepted.generation.requested_model !== requestedOptions.model ||
                    fileAudioRequest === undefined ||
                    accepted.generation.request_receipt.request_fingerprint !==
                        (await fingerprintJson(providerJsonValue(fileAudioRequest.payload)))
                ) {
                    throw new Error(
                        `Accepted response operation ${runtime.response_operation_id} has incompatible request identity`,
                    );
                }
                if (requestedOptions.include_original_response) {
                    throw new Error('An idempotently recovered Gemini response cannot reconstruct original_response');
                }
                return recoverCanonicalExecutionResponse(
                    { document, runtime, accepted_response: accepted },
                    requestedOptions,
                );
            }
        }
        const canonicalState = await prepareGeminiCanonicalState({
            conversation: requestedOptions.conversation,
            prompt,
            options: requestedOptions,
            provider: geminiProvider(driver),
            resolve_asset: hostCapabilities?.resolve_canonical_asset,
            signal,
        });
        if (isFileAudioModel(modelName)) {
            return this.requestPreparedCanonicalFileAudioCompletion(
                driver,
                canonicalState,
                requestedOptions,
                modelName,
                region,
                signal,
            );
        }
        return this.requestPreparedCanonicalTextCompletion(
            driver,
            canonicalState,
            requestedOptions,
            modelName,
            region,
            signal,
        );
    }

    async requestCanonicalContextCompletion(
        driver: VertexAIDriver,
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionResponse> {
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) region = splits[1];
        const modelName = splits.at(-1) ?? options.model;
        const canonicalState = await prepareGeminiCanonicalContext({
            options,
            provider: geminiProvider(driver),
            resolve_asset: hostCapabilities?.resolve_canonical_asset,
            signal,
        });
        if (isFileAudioModel(modelName)) {
            return this.requestPreparedCanonicalFileAudioCompletion(
                driver,
                canonicalState,
                options,
                modelName,
                region,
                signal,
                true,
            );
        }
        return this.requestPreparedCanonicalTextCompletion(
            driver,
            canonicalState,
            options,
            modelName,
            region,
            signal,
            true,
        );
    }

    private async requestPreparedCanonicalFileAudioCompletion(
        driver: VertexAIDriver,
        canonicalState: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
        requestedOptions: ExecutionOptions,
        modelName: string,
        region: string | undefined,
        signal?: AbortSignal,
        contextOnly = false,
    ): Promise<CanonicalExecutionResponse> {
        if (contextOnly && canonicalState.tool_definitions.length > 0) {
            throw new Error('Gemini file audio canonical context does not support tool definitions');
        }
        if (contextOnly && requestedOptions.result_schema !== undefined) {
            throw new Error('Gemini file audio canonical context does not support result schemas');
        }
        const transportOptions = { ...requestedOptions, model: modelName };
        const canonicalPrompt = prepareCanonicalGeminiProjection(canonicalState, requestedOptions, contextOnly);
        const {
            model_options: modelOptions,
            payload,
            speech,
        } = geminiFileAudioRequest(canonicalPrompt, requestedOptions, modelName);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            {
                provider: geminiProvider(driver),
                protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                model: requestedOptions.model,
            },
            providerJsonValue(payload),
        );
        if (canonicalState.accepted_response !== undefined) {
            if (requestedOptions.include_original_response) {
                throw new Error('An idempotently recovered Gemini response cannot reconstruct original_response');
            }
            return recoverCanonicalExecutionResponse(canonicalState, requestedOptions, {
                service_tier: canonicalGeminiServiceTier(canonicalState),
            });
        }
        const prepared = await finalizeGeminiPreparedRequest(
            { ...canonicalState, native_conversation: canonicalPrompt },
            payload,
        );
        await publishCanonicalPreparedRequest(prepared, requestedOptions);
        const client = driver.getGoogleGenAIClient(
            region,
            resolveVertexAIServiceTier(modelOptions),
            transportOptions.httpTimeout,
        );
        const response = await client.models.generateContent(geminiFileAudioTransportRequest(payload, signal));
        const candidate = response.candidates?.[0];
        if (candidate?.content === undefined) throw new Error('Audio model returned no candidate content');
        const persistedAudio: Array<{
            data: string;
            result: Extract<CompletionResult, { type: 'audio' }>;
            byte_length: number;
        }> = [];
        for (const part of candidate.content.parts ?? []) {
            if (!part.inlineData?.mimeType?.startsWith('audio/')) continue;
            if (!speech) throw new Error('Unexpected audio output from transcription model');
            const data = part.inlineData.data ?? '';
            if (data.length > Math.ceil(50_000_000 / 3) * 4) throw new Error('Audio exceeds the 50000000 byte limit');
            const bytes = Buffer.from(data, 'base64');
            const result = await storeAudioResult(
                new Blob([bytes]).stream(),
                {
                    mime_type: part.inlineData.mimeType,
                    container: 'raw',
                    codec: 'pcm',
                    sample_rate: 24000,
                    channels: 1,
                    sample_encoding: 'int16',
                    byte_order: 'little',
                },
                transportOptions,
                signal,
            );
            persistedAudio.push({ data, result, byte_length: bytes.byteLength });
        }
        if (speech && persistedAudio.length === 0) throw new Error('Audio model returned no usable audio result');
        const finishReason = normalizeGeminiFinishReason(candidate.finishReason);
        let decoded = await decodeGeminiCanonicalResponse({
            response,
            content: candidate.content,
            prepared,
            finish_reason: finishReason,
        });
        const remainingAudio = [...persistedAudio];
        decoded = {
            ...decoded,
            ...(speech
                ? {
                      turns: decoded.turns.map((turn) =>
                          turn.kind === 'agent' && 'generation_id' in turn
                              ? {
                                    ...turn,
                                    blocks: turn.blocks.filter((block) => block.type !== 'native_replay'),
                                }
                              : turn,
                      ),
                  }
                : {}),
            assets: (decoded.assets ?? []).map((asset) => {
                if (asset.kind !== 'audio' || asset.storage.type !== 'inline_base64') return asset;
                const inlineData = asset.storage.data;
                const matchIndex = remainingAudio.findIndex((entry) => entry.data === inlineData);
                if (matchIndex < 0) return asset;
                const [match] = remainingAudio.splice(matchIndex, 1);
                if (match === undefined) return asset;
                return {
                    ...asset,
                    storage: canonicalAudioAssetStorage(match.result.value),
                    byte_length: match.byte_length,
                    media: {
                        ...(match.result.container === undefined ? {} : { container: match.result.container }),
                        ...(match.result.codec === undefined ? {} : { codec: match.result.codec }),
                        ...(match.result.sample_rate === undefined ? {} : { sample_rate: match.result.sample_rate }),
                        ...(match.result.channels === undefined ? {} : { channels: match.result.channels }),
                        ...(match.result.sample_encoding === undefined
                            ? {}
                            : { sample_encoding: match.result.sample_encoding }),
                        ...(match.result.byte_order === undefined ? {} : { byte_order: match.result.byte_order }),
                    },
                    metadata: {
                        audio_result: {
                            value: match.result.value,
                            mime_type: match.result.mime_type,
                            ...(match.result.container === undefined ? {} : { container: match.result.container }),
                            ...(match.result.codec === undefined ? {} : { codec: match.result.codec }),
                            ...(match.result.sample_rate === undefined
                                ? {}
                                : { sample_rate: match.result.sample_rate }),
                            ...(match.result.channels === undefined ? {} : { channels: match.result.channels }),
                            ...(match.result.sample_encoding === undefined
                                ? {}
                                : { sample_encoding: match.result.sample_encoding }),
                            ...(match.result.byte_order === undefined ? {} : { byte_order: match.result.byte_order }),
                        },
                    },
                };
            }),
        };
        const document = await appendGeminiCanonicalResponseWithProcessing(prepared, decoded);
        return createCanonicalExecutionResponse(document, prepared.runtime.response_operation_id, {
            service_tier: normalizeVertexAIResolvedServiceTier(response.usageMetadata?.trafficType),
            ...(requestedOptions.include_original_response ? { original_response: response } : {}),
        });
    }

    private async requestPreparedCanonicalTextCompletion(
        driver: VertexAIDriver,
        canonicalState: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
        requestedOptions: ExecutionOptions,
        modelName: string,
        region: string | undefined,
        signal?: AbortSignal,
        contextOnly = false,
    ): Promise<CanonicalExecutionResponse> {
        const transportOptions = { ...requestedOptions, model: modelName };
        if (transportOptions.model.includes('gemini-2.5-flash-image')) region = 'global';
        const modelOptions = transportOptions.model_options as VertexAIGeminiOptions | undefined;
        const canonicalPrompt = prepareCanonicalGeminiProjection(canonicalState, requestedOptions, contextOnly);
        const client = driver.getGoogleGenAIClient(
            region,
            resolveVertexAIServiceTier(modelOptions),
            transportOptions.httpTimeout,
        );
        const payload = getGeminiPayload(transportOptions, canonicalPrompt, 'execute', canonicalState.tool_definitions);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            {
                provider: geminiProvider(driver),
                protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                model: requestedOptions.model,
            },
            providerJsonValue(payload),
        );
        if (canonicalState.accepted_response !== undefined) {
            if (requestedOptions.include_original_response) {
                throw new Error('An idempotently recovered Gemini response cannot reconstruct original_response');
            }
            return recoverCanonicalExecutionResponse(canonicalState, requestedOptions, {
                service_tier: canonicalGeminiServiceTier(canonicalState),
            });
        }
        const prepared = await finalizeGeminiPreparedRequest(
            { ...canonicalState, native_conversation: canonicalPrompt },
            payload,
        );
        await publishCanonicalPreparedRequest(prepared, requestedOptions);
        if (signal) payload.config = { ...payload.config, abortSignal: signal };
        const cacheExecution = await generateWithGeminiContextCache(
            driver,
            client,
            transportOptions,
            canonicalPrompt,
            payload,
            (request) => client.models.generateContent(request),
            region ?? driver.getVertexRegion?.() ?? 'global',
        );
        const response = cacheExecution.value;

        let finalContent: Content = { role: 'model', parts: [] };
        let finishReason: string | undefined;
        let toolUse: ToolUse[] | undefined;
        const candidates = response.candidates ?? [];
        if (candidates.length > 1) {
            throw new Error(
                `Gemini returned ${candidates.length} candidates; canonical ingestion requires one candidate`,
            );
        }
        const candidate = candidates[0];
        if (candidate !== undefined) {
            if (candidate.finishReason === undefined) {
                throw new Error('Gemini response candidate has no terminal finish reason');
            }
            finishReason = normalizeGeminiFinishReason(candidate.finishReason);
            const isRecoverableToolCall = assertSupportedGeminiFinishReason(candidate);
            if (candidate.content !== undefined) {
                toolUse = await geminiToolUsesFromContent(candidate.content, prepared.runtime.response_operation_id);
                if (isRecoverableToolCall && toolUse && toolUse.length > 0) {
                    driver.logger.warn(
                        `[Gemini] Recoverable tool call issue (${candidate.finishReason}): ` +
                            `Model tried to call undeclared tool(s): ${toolUse.map((tool) => tool.tool_name).join(', ')}`,
                    );
                }
                finalContent = candidate.content;
            }
        } else if (response.promptFeedback?.blockReason !== undefined) {
            finishReason = response.promptFeedback.blockReason;
            const blockMessage = response.promptFeedback.blockReasonMessage ?? '';
            finalContent = { role: 'model', parts: [{ text: blockMessage }] };
        } else {
            throw new Error('Gemini response has no candidate or prompt block reason');
        }
        if (toolUse?.length) finishReason = 'tool_use';

        const rawDecoded = await decodeGeminiCanonicalResponse({
            response,
            content: finalContent,
            prepared,
            finish_reason: finishReason,
        });
        const normalized =
            !toolUse?.length && requestedOptions.result_schema
                ? normalizeDecodedStructuredOutputForSchema(rawDecoded, requestedOptions.result_schema)
                : undefined;
        let decoded =
            normalized?.status === 'valid'
                ? await decodeGeminiCanonicalResponse({
                      response,
                      content: finalContent,
                      prepared,
                      finish_reason: finishReason,
                      structured_output: normalized.structured_output,
                  })
                : rawDecoded;
        if (normalized?.status === 'invalid') decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
        const document = await appendGeminiCanonicalResponseWithProcessing(prepared, decoded);
        return createCanonicalExecutionResponse(document, prepared.runtime.response_operation_id, {
            service_tier: normalizeVertexAIResolvedServiceTier(response.usageMetadata?.trafficType),
            prompt_cache_diagnostic: cacheExecution.diagnostic,
            ...(requestedOptions.include_original_response ? { original_response: response } : {}),
        });
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: GenerateContentPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const requestedOptions = options;
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) {
            region = splits[1];
        }
        const modelName = splits[splits.length - 1];
        const transportOptions = { ...options, model: modelName };

        if (isFileAudioModel(modelName)) {
            const modelOptions = transportOptions.model_options as VertexAIGeminiOptions | undefined;
            const client = driver.getGoogleGenAIClient(
                region,
                resolveVertexAIServiceTier(modelOptions),
                transportOptions.httpTimeout,
            );
            const speech = modelName.includes('tts');
            const config: GenerateContentConfig = speech
                ? {
                      responseModalities: [Modality.AUDIO],
                      speechConfig: {
                          languageCode: modelOptions?.speech_language,
                          voiceConfig: { prebuiltVoiceConfig: { voiceName: modelOptions?.speech_voice ?? 'Kore' } },
                      },
                  }
                : {
                      systemInstruction: prompt.system,
                      audioTranscriptionConfig: {
                          languageCodes: modelOptions?.transcription_language_codes,
                          diarization: modelOptions?.transcription_diarization,
                          wordTimestamp: modelOptions?.transcription_word_timestamps,
                          customVocabulary: modelOptions?.transcription_vocabulary,
                      },
                  };
            config.abortSignal = signal;
            const response = await client.models.generateContent({
                model: modelName,
                contents: speech
                    ? [
                          {
                              role: 'user',
                              parts: [
                                  ...(prompt.system?.parts ?? []),
                                  ...prompt.contents.flatMap((content) => content.parts ?? []),
                              ],
                          },
                      ]
                    : prompt.contents,
                config,
            });
            const parts = response.candidates?.[0]?.content?.parts ?? [];
            const hasText = parts.some((part) => part.text);
            const results: CompletionResult[] = [];
            for (const part of parts) {
                if (part.text) results.push({ type: 'text', value: part.text });
                if (part.audioTranscription) {
                    if (part.audioTranscription.text && !hasText)
                        results.push({ type: 'text', value: part.audioTranscription.text });
                    const transcription = part.audioTranscription;
                    results.push({
                        type: 'json',
                        value: {
                            text: transcription.text ?? '',
                            language_code: transcription.languageCode ?? null,
                            speaker_label: transcription.speakerLabel ?? null,
                            words:
                                transcription.words?.map((word) => ({
                                    word: word.word ?? '',
                                    start_offset: word.startOffset ?? null,
                                    end_offset: word.endOffset ?? null,
                                })) ?? [],
                        },
                    });
                }
                if (part.inlineData?.mimeType?.startsWith('audio/')) {
                    if (!speech) throw new Error('Unexpected audio output from transcription model');
                    const data = part.inlineData.data ?? '';
                    if (data.length > Math.ceil(50_000_000 / 3) * 4)
                        throw new Error('Audio exceeds the 50000000 byte limit');
                    const bytes = Buffer.from(data, 'base64');
                    results.push(
                        await storeAudioResult(
                            new Blob([bytes]).stream(),
                            {
                                mime_type: part.inlineData.mimeType,
                                container: 'raw',
                                codec: 'pcm',
                                sample_rate: 24000,
                                channels: 1,
                                sample_encoding: 'int16',
                                byte_order: 'little',
                            },
                            transportOptions,
                            signal,
                        ),
                    );
                }
            }
            if (!results.length || (speech && !results.some((result) => result.type === 'audio'))) {
                throw new Error('Audio model returned no usable result');
            }
            return {
                result: results,
                finish_reason: normalizeGeminiFinishReason(response.candidates?.[0]?.finishReason),
                token_usage: this.usageMetadataToTokenUsage(driver, response.usageMetadata),
                original_response: transportOptions.include_original_response ? response : undefined,
            };
        }

        const canonicalState = await prepareGeminiCanonicalState({
            conversation: requestedOptions.conversation,
            prompt,
            options: requestedOptions,
            provider: geminiProvider(driver),
        });

        // TODO: Remove hack, use global endpoint manually if needed.
        if (transportOptions.model.includes('gemini-2.5-flash-image')) {
            region = 'global'; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const model_options = transportOptions.model_options as VertexAIGeminiOptions | undefined;
        const includeThoughts = model_options?.include_thoughts !== false;
        const canonicalPrompt = prepareCanonicalGeminiProjection(canonicalState, requestedOptions);
        const client = driver.getGoogleGenAIClient(
            region,
            resolveVertexAIServiceTier(model_options),
            transportOptions.httpTimeout,
        );

        const payload = getGeminiPayload(transportOptions, canonicalPrompt, 'execute');
        await assertAcceptedCanonicalRequest(
            canonicalState,
            {
                provider: geminiProvider(driver),
                protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                model: requestedOptions.model,
            },
            providerJsonValue(payload),
        );
        if (canonicalState.accepted_response !== undefined) {
            return recoverGeminiCompletion(canonicalState, this, driver, requestedOptions, includeThoughts);
        }
        const prepared = await finalizeGeminiPreparedRequest(
            { ...canonicalState, native_conversation: canonicalPrompt },
            payload,
        );
        await publishCanonicalPreparedRequest(prepared, requestedOptions);
        if (signal) payload.config = { ...payload.config, abortSignal: signal };
        // Routes through an explicit Vertex context cache when this execution carries a
        // prompt_cache_key; sends `payload` untouched otherwise, and on any cache failure.
        const cacheExecution = await generateWithGeminiContextCache(
            driver,
            client,
            transportOptions,
            canonicalPrompt,
            payload,
            (request) => client.models.generateContent(request),
            region ?? driver.getVertexRegion?.() ?? 'global',
        );
        const response = cacheExecution.value;

        const token_usage: ExecutionTokenUsage = this.usageMetadataToTokenUsage(driver, response.usageMetadata);

        let tool_use: ToolUse[] | undefined;
        let finalContent: Content = { role: 'model', parts: [] };
        let finish_reason: string | undefined, result: CompletionResult[] | undefined;
        const candidates = response.candidates ?? [];
        if (candidates.length > 1) {
            throw new Error(
                `Gemini returned ${candidates.length} candidates; canonical ingestion requires one candidate`,
            );
        }
        const candidate = candidates[0];
        if (candidate) {
            if (candidate.finishReason === undefined) {
                throw new Error('Gemini response candidate has no terminal finish reason');
            }
            finish_reason = normalizeGeminiFinishReason(candidate.finishReason);
            const content = candidate.content;

            // Provider finish reasons are terminal responses, not transport failures. Classify them
            // explicitly while allowing recoverable tool-call issues to continue through the workflow.
            const isRecoverableToolCall = assertSupportedGeminiFinishReason(candidate);

            if (content) {
                tool_use = await geminiToolUsesFromContent(content, prepared.runtime.response_operation_id);

                // For recoverable tool call issues, log warning but continue processing
                // The workflow will handle the invalid tool call gracefully.
                // Route through the driver's structured logger instead of `console.warn`
                // so downstream runtimes (e.g. Cloud Run) don't promote stderr writes
                // to ERROR severity for what is, by definition, a recoverable event.
                if (isRecoverableToolCall && tool_use && tool_use.length > 0) {
                    driver.logger.warn(
                        `[Gemini] Recoverable tool call issue (${candidate.finishReason}): ` +
                            `Model tried to call undeclared tool(s): ${tool_use.map((t) => t.tool_name).join(', ')}`,
                    );
                }

                result = extractCompletionResults(content, includeThoughts);
                finalContent = content;
            }
        } else if (response.promptFeedback?.blockReason !== undefined) {
            finish_reason = response.promptFeedback.blockReason;
            const blockMessage = response.promptFeedback.blockReasonMessage ?? '';
            finalContent = { role: 'model', parts: [{ text: blockMessage }] };
            result = blockMessage.length === 0 ? [] : [{ type: 'text', value: blockMessage }];
        } else {
            throw new Error('Gemini response has no candidate or prompt block reason');
        }

        if (tool_use) {
            finish_reason = 'tool_use';
        }

        const completionResults = result && result.length > 0 ? result : [{ type: 'text' as const, value: '' }];
        const rawDecoded = await decodeGeminiCanonicalResponse({
            response,
            content: finalContent,
            prepared,
            finish_reason,
        });
        const normalized =
            !tool_use?.length && requestedOptions.result_schema
                ? normalizeDecodedStructuredOutputForSchema(rawDecoded, requestedOptions.result_schema)
                : undefined;
        const decoded =
            normalized?.status === 'valid'
                ? await decodeGeminiCanonicalResponse({
                      response,
                      content: finalContent,
                      prepared,
                      finish_reason,
                      structured_output: normalized.structured_output,
                  })
                : rawDecoded;
        const finalConversation = await appendGeminiCanonicalResponseWithProcessing(prepared, decoded);

        return {
            result: completionResults,
            token_usage: token_usage,
            service_tier: normalizeVertexAIResolvedServiceTier(response.usageMetadata?.trafficType),
            finish_reason: finish_reason,
            original_response: requestedOptions.include_original_response ? response : undefined,
            conversation: finalConversation,
            tool_use,
            prompt_cache_diagnostic: cacheExecution.diagnostic,
        } satisfies Completion;
    }

    async requestTextCompletionStream(
        driver: VertexAIDriver,
        prompt: GenerateContentPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        const requestedOptions = options;
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) {
            region = splits[1];
        }
        const modelName = splits[splits.length - 1];
        const transportOptions = { ...options, model: modelName };
        const canonicalState = await prepareGeminiCanonicalState({
            conversation: requestedOptions.conversation,
            prompt,
            options: requestedOptions,
            provider: geminiProvider(driver),
        });

        if (transportOptions.model.includes('gemini-2.5-flash-image')) {
            region = 'global'; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const model_options = transportOptions.model_options as VertexAIGeminiOptions | undefined;
        const includeThoughts = model_options?.include_thoughts !== false;
        const canonicalPrompt = prepareCanonicalGeminiProjection(canonicalState, requestedOptions);
        const client = driver.getGoogleGenAIClient(
            region,
            resolveVertexAIServiceTier(model_options),
            transportOptions.httpTimeout,
        );

        const payload = getGeminiPayload(transportOptions, canonicalPrompt, 'stream');
        await assertAcceptedCanonicalRequest(
            canonicalState,
            {
                provider: geminiProvider(driver),
                protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                model: requestedOptions.model,
            },
            providerJsonValue(payload),
        );
        if (canonicalState.accepted_response !== undefined) {
            return recoveredGeminiStream(
                await recoverGeminiCompletion(canonicalState, this, driver, requestedOptions, includeThoughts),
            );
        }
        const prepared = await finalizeGeminiPreparedRequest(
            { ...canonicalState, native_conversation: canonicalPrompt },
            payload,
        );
        await publishCanonicalPreparedRequest(prepared, requestedOptions);
        payload.config = { ...payload.config, abortSignal: signal };
        const cacheExecution = await generateWithGeminiContextCache(
            driver,
            client,
            transportOptions,
            canonicalPrompt,
            payload,
            (request) => client.models.generateContentStream(request),
            region ?? driver.getVertexRegion?.() ?? 'global',
        );
        const response = cacheExecution.value;

        const nativeParts: Part[] = [];
        let streamedToolCallCount = 0;
        let streamedToolUseFound = false;
        let terminalResponse: GenerateContentResponse | undefined;
        let terminalCandidate: NonNullable<GenerateContentResponse['candidates']>[number] | undefined;
        let terminalPromptFeedback: GenerateContentResponse['promptFeedback'] | undefined;
        let terminalFinishReason: string | undefined;
        let finalUsageMetadata: GenerateContentResponseUsageMetadata | undefined;
        const stream = asyncMap(response, async (item) => {
            if (item.usageMetadata !== undefined) finalUsageMetadata = item.usageMetadata;
            const token_usage: ExecutionTokenUsage = this.usageMetadataToTokenUsage(driver, item.usageMetadata);
            if (item.candidates && item.candidates.length > 0) {
                if (item.candidates.length > 1) {
                    throw new Error(
                        `Gemini stream returned ${item.candidates.length} candidates; canonical ingestion requires one candidate`,
                    );
                }
                const candidate = item.candidates[0];
                let tool_use: ToolUse[] | undefined;
                let finish_reason = normalizeGeminiFinishReason(candidate.finishReason);
                const isRecoverableToolCall = assertSupportedGeminiFinishReason(candidate);
                if (candidate.finishReason !== undefined) {
                    terminalResponse = item;
                    terminalCandidate = candidate;
                    terminalPromptFeedback = undefined;
                    terminalFinishReason = finish_reason;
                }
                if (candidate.content?.role === 'model') {
                    appendGeminiStreamParts(nativeParts, candidate.content.parts ?? []);
                    // Collect all parts in order (text and images)
                    const combinedResults = extractCompletionResults(candidate.content, includeThoughts);
                    tool_use = await geminiToolUsesFromContent(
                        candidate.content,
                        prepared.runtime.response_operation_id,
                        streamedToolCallCount,
                    );
                    streamedToolCallCount += tool_use?.length ?? 0;
                    if (tool_use) {
                        finish_reason = 'tool_use';
                        streamedToolUseFound = true;
                        // Log warning for recoverable tool call issues — see the
                        // matching site in `requestTextCompletion` above for why
                        // we route through the driver's logger instead of
                        // `console.warn`.
                        if (isRecoverableToolCall) {
                            driver.logger.warn(
                                `[Gemini] Recoverable tool call issue (${candidate.finishReason}): ` +
                                    `Model tried to call undeclared tool(s): ${tool_use.map((t) => t.tool_name).join(', ')}`,
                            );
                        }
                    }
                    return {
                        result: combinedResults.length > 0 ? combinedResults : [],
                        token_usage: token_usage,
                        service_tier: normalizeVertexAIResolvedServiceTier(item.usageMetadata?.trafficType),
                        finish_reason: finish_reason,
                        tool_use,
                    };
                }
            }
            //No normal output, returning block reason if it exists.
            if (item.promptFeedback?.blockReason !== undefined) {
                terminalResponse = item;
                terminalCandidate = undefined;
                terminalPromptFeedback = item.promptFeedback;
                terminalFinishReason = item.promptFeedback.blockReason;
            }
            return {
                result: item.promptFeedback?.blockReasonMessage
                    ? [{ type: 'text' as const, value: item.promptFeedback.blockReasonMessage }]
                    : [],
                finish_reason: item.promptFeedback?.blockReason ?? '',
                token_usage: token_usage,
                service_tier: normalizeVertexAIResolvedServiceTier(item.usageMetadata?.trafficType),
            };
        });

        async function computeDecodedFinalResponse() {
            if (
                terminalResponse === undefined ||
                (terminalCandidate === undefined && terminalPromptFeedback === undefined)
            ) {
                throw new Error('Gemini stream ended without a terminal finish reason');
            }
            const blockMessage = terminalPromptFeedback?.blockReasonMessage ?? '';
            const content: Content =
                terminalCandidate === undefined
                    ? { role: 'model', parts: [{ text: blockMessage }] }
                    : { role: 'model', parts: nativeParts };
            const finalResponse = {
                ...terminalResponse,
                ...(finalUsageMetadata === undefined ? {} : { usageMetadata: finalUsageMetadata }),
                ...(terminalCandidate === undefined ? {} : { candidates: [{ ...terminalCandidate, content }] }),
            } as GenerateContentResponse;
            const rawDecoded = await decodeGeminiCanonicalResponse({
                response: finalResponse,
                content,
                prepared,
                finish_reason: streamedToolUseFound ? 'tool_use' : terminalFinishReason,
            });
            const normalized =
                !streamedToolUseFound && requestedOptions.result_schema
                    ? normalizeDecodedStructuredOutputForSchema(rawDecoded, requestedOptions.result_schema)
                    : undefined;
            const decoded =
                normalized?.status === 'valid'
                    ? await decodeGeminiCanonicalResponse({
                          response: finalResponse,
                          content,
                          prepared,
                          finish_reason: streamedToolUseFound ? 'tool_use' : terminalFinishReason,
                          structured_output: normalized.structured_output,
                      })
                    : rawDecoded;
            return { decoded, finalResponse, normalized };
        }
        let decodedFinalResponse: ReturnType<typeof computeDecodedFinalResponse> | undefined;
        const decodeFinalResponse = () => {
            decodedFinalResponse ??= computeDecodedFinalResponse();
            return decodedFinalResponse;
        };
        return Object.assign(stream, {
            finalizePromptCacheDiagnostic: () => cacheExecution.diagnostic,
            finalizeConversation: async () => {
                const { decoded } = await decodeFinalResponse();
                return await appendGeminiCanonicalResponseWithProcessing(prepared, decoded);
            },
        });
    }

    async requestCanonicalTextCompletionEventStream(
        driver: VertexAIDriver,
        prompt: GenerateContentPrompt,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionEventStream> {
        const requestedOptions = options;
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) region = splits[1];
        const modelName = splits.at(-1) ?? options.model;

        if (isFileAudioModel(modelName)) {
            const runtime = resolveConversationRuntime(requestedOptions);
            const document = isConversationDocumentFormat(requestedOptions.conversation)
                ? parseConversationDocument(requestedOptions.conversation)
                : undefined;
            const accepted =
                document === undefined ? undefined : acceptedCanonicalResponse(document, runtime.response_operation_id);
            const state =
                accepted === undefined
                    ? await prepareGeminiCanonicalState({
                          conversation: requestedOptions.conversation,
                          prompt,
                          options: requestedOptions,
                          provider: geminiProvider(driver),
                          resolve_asset: hostCapabilities?.resolve_canonical_asset,
                          signal,
                      })
                    : undefined;
            const identity = {
                request_id: accepted?.generation.request_id ?? state?.runtime.request_id ?? runtime.request_id,
                attempt_id: accepted?.generation.attempt_id ?? state?.runtime.attempt_id ?? runtime.attempt_id,
                response_operation_id: runtime.response_operation_id,
                generation_id:
                    accepted?.generation.id ?? state?.generation_id ?? `${runtime.response_operation_id}:generation`,
                draft_turn_id: accepted?.turn.id ?? state?.response_turn_id ?? `${runtime.response_operation_id}:turn`,
            };
            return new FallbackCanonicalExecutionEventStream(
                identity,
                (fallbackSignal) =>
                    this.requestCanonicalTextCompletion(
                        driver,
                        prompt,
                        requestedOptions,
                        signal ? AbortSignal.any([signal, fallbackSignal]) : fallbackSignal,
                        hostCapabilities,
                    ),
                { ...open, origin: accepted === undefined ? 'live_transport' : 'accepted_recovery' },
            );
        }

        const canonicalState = await prepareGeminiCanonicalState({
            conversation: requestedOptions.conversation,
            prompt,
            options: requestedOptions,
            provider: geminiProvider(driver),
            resolve_asset: hostCapabilities?.resolve_canonical_asset,
            signal,
        });
        return this.requestPreparedCanonicalTextCompletionEventStream(
            driver,
            canonicalState,
            requestedOptions,
            modelName,
            region,
            signal,
            open,
        );
    }

    async requestCanonicalContextCompletionEventStream(
        driver: VertexAIDriver,
        options: CanonicalExecutionContextOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
        hostCapabilities?: CanonicalHostCapabilities,
    ): Promise<CanonicalExecutionEventStream> {
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) region = splits[1];
        const modelName = splits.at(-1) ?? options.model;
        if (isFileAudioModel(modelName)) {
            throw new Error(`Gemini file audio model ${options.model} does not support canonical context streaming`);
        }
        const canonicalState = await prepareGeminiCanonicalContext({
            options,
            provider: geminiProvider(driver),
            resolve_asset: hostCapabilities?.resolve_canonical_asset,
            signal,
        });
        return this.requestPreparedCanonicalTextCompletionEventStream(
            driver,
            canonicalState,
            options,
            modelName,
            region,
            signal,
            open,
            true,
        );
    }

    private async requestPreparedCanonicalTextCompletionEventStream(
        driver: VertexAIDriver,
        canonicalState: Omit<PreparedGeminiConversation, 'payload' | 'receipt' | 'diagnostics'>,
        requestedOptions: ExecutionOptions,
        modelName: string,
        region: string | undefined,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
        contextOnly = false,
    ): Promise<CanonicalExecutionEventStream> {
        const transportOptions = { ...requestedOptions, model: modelName };
        if (transportOptions.model.includes('gemini-2.5-flash-image')) region = 'global';
        const modelOptions = transportOptions.model_options as VertexAIGeminiOptions | undefined;
        const canonicalPrompt = prepareCanonicalGeminiProjection(canonicalState, requestedOptions, contextOnly);
        const payload = getGeminiPayload(transportOptions, canonicalPrompt, 'stream', canonicalState.tool_definitions);
        await assertAcceptedCanonicalRequest(
            canonicalState,
            {
                provider: geminiProvider(driver),
                protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                model: requestedOptions.model,
            },
            providerJsonValue(payload),
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
            if (requestedOptions.include_original_response) {
                throw new Error('An idempotently recovered Gemini response cannot reconstruct original_response');
            }
            return new FallbackCanonicalExecutionEventStream(
                identity,
                () =>
                    recoverCanonicalExecutionResponse(canonicalState, requestedOptions, {
                        service_tier: canonicalGeminiServiceTier(canonicalState),
                    }),
                { ...open, origin: 'accepted_recovery' },
            );
        }

        const prepared = await finalizeGeminiPreparedRequest(
            { ...canonicalState, native_conversation: canonicalPrompt },
            payload,
        );
        const abortController = new AbortController();
        const forwardAbort = () => abortController.abort(signal?.reason);
        payload.config = { ...payload.config, abortSignal: abortController.signal };
        const client = driver.getGoogleGenAIClient(
            region,
            resolveVertexAIServiceTier(modelOptions),
            transportOptions.httpTimeout,
        );
        const nativeParts: Part[] = [];
        const drafts = new Map<string, GeminiCanonicalDraft>();
        let streamedToolCallCount = 0;
        let terminalResponse: GenerateContentResponse | undefined;
        let terminalCandidate: NonNullable<GenerateContentResponse['candidates']>[number] | undefined;
        let terminalPromptFeedback: GenerateContentResponse['promptFeedback'] | undefined;
        let terminalFinishReason: string | undefined;
        let finalUsageMetadata: GenerateContentResponseUsageMetadata | undefined;
        let cacheExecution: GeminiContextCacheExecution<AsyncIterable<GenerateContentResponse>> | undefined;

        const eventStream = canonicalNativeExecutionEventStream({
            identity,
            open,
            classifyFailure: (error) => {
                const classified = LlumiverseError.isLlumiverseError(error)
                    ? error
                    : driver.formatLlumiverseError(error, {
                          provider: geminiProvider(driver),
                          model: requestedOptions.model,
                          operation: 'stream',
                      });
                return classified.retryable;
            },
            openSource: async () => {
                cacheExecution = await generateWithGeminiContextCache(
                    driver,
                    client,
                    transportOptions,
                    canonicalPrompt,
                    payload,
                    (request) => client.models.generateContentStream(request),
                    region ?? driver.getVertexRegion?.() ?? 'global',
                );
                return cacheExecution.value;
            },
            map: async (item, writer) => {
                if (item.usageMetadata !== undefined) finalUsageMetadata = item.usageMetadata;
                if ((item.candidates?.length ?? 0) > 1) {
                    throw new Error(
                        `Gemini stream returned ${item.candidates?.length ?? 0} candidates; canonical ingestion requires one candidate`,
                    );
                }
                const candidate = item.candidates?.[0];
                if (candidate === undefined) {
                    if (item.promptFeedback?.blockReason !== undefined) {
                        terminalResponse = item;
                        terminalCandidate = undefined;
                        terminalPromptFeedback = item.promptFeedback;
                        terminalFinishReason = item.promptFeedback.blockReason;
                    }
                    return;
                }
                assertSupportedGeminiFinishReason(candidate);
                if (candidate.finishReason !== undefined) {
                    terminalResponse = item;
                    terminalCandidate = candidate;
                    terminalPromptFeedback = undefined;
                    terminalFinishReason = normalizeGeminiFinishReason(candidate.finishReason);
                }
                if (candidate.content?.role !== 'model') return;
                for (const part of candidate.content.parts ?? []) {
                    if (part.audioTranscription !== undefined) {
                        throw new Error(
                            'Gemini streaming transcription output requires the finite file-audio canonical path',
                        );
                    }
                    const partIndex = appendGeminiStreamPart(nativeParts, part);
                    const position = geminiStreamPosition(partIndex, part.functionCall?.id);
                    const key = canonicalJsonContentString(position);
                    if (typeof part.text === 'string') {
                        if (part.text.length === 0) continue;
                        let draft = drafts.get(key);
                        if (draft === undefined) {
                            draft = {
                                draft_block_id: `${prepared.response_turn_id}:gemini:${partIndex}`,
                                native_position: position,
                                kind: part.thought ? 'reasoning' : 'text',
                                text: '',
                            };
                            drafts.set(key, draft);
                            await writer.startBlock({
                                draft_block_id: draft.draft_block_id,
                                native_position: position,
                                block: part.thought ? { type: 'reasoning', visibility: 'display' } : { type: 'text' },
                            });
                        }
                        draft.text += part.text;
                        if (draft.kind === 'reasoning') {
                            await writer.reasoning({
                                draft_block_id: draft.draft_block_id,
                                native_position: position,
                                text: part.text,
                            });
                        } else {
                            await writer.text({
                                draft_block_id: draft.draft_block_id,
                                native_position: position,
                                text: part.text,
                            });
                        }
                        continue;
                    }
                    if (part.functionCall !== undefined) {
                        const call = (
                            await geminiToolUsesFromContent(
                                { role: 'model', parts: [part] },
                                prepared.runtime.response_operation_id,
                                streamedToolCallCount,
                            )
                        )?.[0];
                        streamedToolCallCount += 1;
                        if (call === undefined)
                            throw new Error('Gemini stream function call has no canonical identity');
                        const toolArguments = providerJsonValue(part.functionCall.args ?? {}) as JsonValue;
                        const draft: GeminiCanonicalDraft = {
                            draft_block_id: `${prepared.response_turn_id}:gemini:${partIndex}`,
                            native_position: position,
                            kind: 'tool_call',
                            text: '',
                            tool_arguments: toolArguments,
                        };
                        drafts.set(key, draft);
                        await writer.startBlock({
                            draft_block_id: draft.draft_block_id,
                            native_position: position,
                            block: {
                                type: 'tool_call',
                                executor: 'application',
                                call_id: call.id,
                                tool_name: call.tool_name,
                            },
                        });
                        await writer.toolArgumentsSnapshot({
                            draft_block_id: draft.draft_block_id,
                            native_position: position,
                            value: toolArguments,
                        });
                        continue;
                    }
                    const media = part.inlineData ?? part.fileData;
                    if (media !== undefined) {
                        const mediaKind = geminiDraftMediaKind(media.mimeType);
                        const draft: GeminiCanonicalDraft = {
                            draft_block_id: `${prepared.response_turn_id}:gemini:${partIndex}`,
                            native_position: position,
                            kind: mediaKind,
                            text: '',
                        };
                        drafts.set(key, draft);
                        await writer.startBlock({
                            draft_block_id: draft.draft_block_id,
                            native_position: position,
                            block: { type: mediaKind, ...(media.mimeType ? { mime_type: media.mimeType } : {}) },
                        });
                    }
                }
            },
            finalize: async () => {
                if (
                    terminalResponse === undefined ||
                    (terminalCandidate === undefined && terminalPromptFeedback === undefined)
                ) {
                    throw new Error('Gemini stream ended without a terminal finish reason');
                }
                const blockedMessage = terminalPromptFeedback?.blockReasonMessage ?? '';
                const content: Content =
                    terminalCandidate === undefined
                        ? { role: 'model', parts: [{ text: blockedMessage }] }
                        : { role: 'model', parts: nativeParts };
                const finalResponse = {
                    ...terminalResponse,
                    ...(finalUsageMetadata === undefined ? {} : { usageMetadata: finalUsageMetadata }),
                    ...(terminalCandidate === undefined ? {} : { candidates: [{ ...terminalCandidate, content }] }),
                } as GenerateContentResponse;
                const toolUse = await geminiToolUsesFromContent(content, prepared.runtime.response_operation_id);
                const finishReason = toolUse?.length ? 'tool_use' : terminalFinishReason;
                const rawDecoded = await decodeGeminiCanonicalResponse({
                    response: finalResponse,
                    content,
                    prepared,
                    finish_reason: finishReason,
                });
                const normalized =
                    !toolUse?.length && requestedOptions.result_schema
                        ? normalizeDecodedStructuredOutputForSchema(rawDecoded, requestedOptions.result_schema)
                        : undefined;
                let decoded =
                    normalized?.status === 'valid'
                        ? await decodeGeminiCanonicalResponse({
                              response: finalResponse,
                              content,
                              prepared,
                              finish_reason: finishReason,
                              structured_output: normalized.structured_output,
                          })
                        : rawDecoded;
                if (normalized?.status === 'invalid') {
                    decoded = rejectDecodedStructuredOutput(decoded, normalized.error);
                }
                const document = await appendGeminiCanonicalResponseWithProcessing(prepared, decoded);
                const serviceTier = normalizeVertexAIResolvedServiceTier(finalUsageMetadata?.trafficType);
                const response = createCanonicalExecutionResponse(document, prepared.runtime.response_operation_id, {
                    service_tier: serviceTier,
                    ...(cacheExecution?.diagnostic === undefined
                        ? {}
                        : { prompt_cache_diagnostic: cacheExecution.diagnostic }),
                    ...(requestedOptions.include_original_response ? { original_response: finalResponse } : {}),
                });
                return {
                    decoded,
                    response,
                    prepare_reconciliation: async () => {
                        const positions =
                            terminalCandidate === undefined && blockedMessage.length > 0
                                ? [geminiPromptFeedbackPosition()]
                                : geminiSemanticPositions(content);
                        const rawBlocks = geminiSemanticBlocks(rawDecoded, prepared.response_turn_id);
                        if (rawBlocks.length !== positions.length) {
                            throw new Error('Gemini stream decode does not match terminal native content positions');
                        }
                        const completeDrafts = positions.map((position) =>
                            drafts.get(canonicalJsonContentString(position)),
                        );
                        const terminalOnly = terminalCandidate === undefined;
                        if (!terminalOnly && completeDrafts.some((draft) => draft === undefined)) {
                            throw new Error('Gemini terminal response has no matching native stream draft');
                        }
                        const orderedDrafts = completeDrafts as Array<GeminiCanonicalDraft | undefined>;
                        for (const [index, block] of rawBlocks.entries()) {
                            const draft = orderedDrafts[index];
                            if (draft === undefined && terminalOnly) continue;
                            if (draft === undefined) throw new Error('Gemini terminal response has no matching draft');
                            assertGeminiToolDraftArguments(draft, block);
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
                            const result = geminiSemanticBlocks(decoded, prepared.response_turn_id).find(
                                (block) => block.type === 'json',
                            );
                            if (sources.length === 0 || result?.type !== 'json') {
                                throw new Error('Gemini structured stream is missing source or result blocks');
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
                            if (draft === undefined && terminalOnly) continue;
                            if (draft === undefined) throw new Error('Gemini direct reconciliation has no draft');
                            reconciliations.push({
                                draft_block_ids: [draft.draft_block_id],
                                native_positions: [draft.native_position],
                                committed_block_ids: [block.id],
                                disposition: 'direct' as const,
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
                                for (const [index, block] of rawBlocks.entries()) {
                                    const draft = orderedDrafts[index];
                                    if (draft === undefined) continue;
                                    await writer.finishBlock({
                                        draft_block_id: draft.draft_block_id,
                                        native_position: draft.native_position,
                                        outcome:
                                            block.type === 'tool_call' && block.arguments.type === 'invalid'
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
                                    ...(serviceTier === undefined ? {} : { service_tier: serviceTier }),
                                });
                            },
                        };
                    },
                    ...(normalized?.status === 'valid' && requestedOptions.result_schema !== undefined
                        ? { result_schema: requestedOptions.result_schema }
                        : {}),
                };
            },
            abort: () => abortController.abort(),
            close: () => signal?.removeEventListener('abort', forwardAbort),
        });
        await publishCanonicalPreparedRequest(prepared, requestedOptions);
        if (signal?.aborted) forwardAbort();
        else signal?.addEventListener('abort', forwardAbort, { once: true });
        return eventStream;
    }

    /**
     * Format Google API errors into LlumiverseError with proper status codes and retryability.
     *
     * Google API errors follow AIP-193 standard:
     * - ApiError.status: HTTP status code
     * - ApiError.message: Error message
     *
     * Common error codes:
     * - 400 (INVALID_ARGUMENT): Invalid request parameters
     * - 401 (UNAUTHENTICATED): Authentication required
     * - 403 (PERMISSION_DENIED): Insufficient permissions
     * - 404 (NOT_FOUND): Resource not found
     * - 429 (RESOURCE_EXHAUSTED): Rate limit/quota exceeded
     * - 500 (INTERNAL): Internal server error
     * - 503 (UNAVAILABLE): Service temporarily unavailable
     * - 504 (DEADLINE_EXCEEDED): Request timeout
     *
     * @see https://google.aip.dev/193
     * @see https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/api-errors
     */
    formatLlumiverseError(_driver: VertexAIDriver, error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        if (error instanceof GeminiFinishReasonError) {
            return new LlumiverseError(
                `[${context.provider}] ${error.message}`,
                error.retryable,
                context,
                error,
                undefined,
                error.finishReason,
            );
        }

        // Check if it's a Google API error with status code
        const isApiError = this.isGoogleApiError(error);

        if (!isApiError) {
            // Not a Google API error, use default handling
            // This will be called by the driver's default formatLlumiverseError
            throw error;
        }

        const apiError = error;
        const httpStatusCode = apiError.status;

        // Extract error message
        const message = apiError.message;

        // Build user-facing message with status code
        let userMessage = message;

        // Include status code in message (for end-user visibility)
        if (httpStatusCode) {
            userMessage = `[${httpStatusCode}] ${userMessage}`;
        }

        // Determine retryability based on Google error codes
        const retryable = this.isGeminiErrorRetryable(httpStatusCode, message);

        // Extract error name/type from message if present
        const errorName = this.extractErrorName(message);

        return new LlumiverseError(
            `[${context.provider}] ${userMessage}`,
            retryable,
            context,
            error,
            httpStatusCode,
            errorName,
        );
    }

    /**
     * Type guard to check if error is a Google API error.
     */
    private isGoogleApiError(error: unknown): error is GoogleApiErrorLike {
        return (
            error !== null &&
            typeof error === 'object' &&
            'status' in error &&
            typeof (error as { status?: unknown }).status === 'number' &&
            'message' in error &&
            typeof (error as { message?: unknown }).message === 'string'
        );
    }

    /**
     * Determine if a Google API error is retryable based on HTTP status code.
     *
     * Retryable errors (per Google AIP-194):
     * - 408 (REQUEST_TIMEOUT): Request timeout
     * - 499 (CANCELLED / Client Closed Request): Transport cancellation
     * - 429 (RESOURCE_EXHAUSTED): Rate limit exceeded, quota exhausted
     * - 500 (INTERNAL): Internal server error
     * - 502 (BAD_GATEWAY): Bad gateway
     * - 503 (UNAVAILABLE): Service temporarily unavailable
     * - 504 (DEADLINE_EXCEEDED): Gateway timeout
     *
     * Non-retryable errors:
     * - 400 (INVALID_ARGUMENT): Invalid request parameters
     * - 401 (UNAUTHENTICATED): Authentication required
     * - 403 (PERMISSION_DENIED): Insufficient permissions
     * - 404 (NOT_FOUND): Resource not found
     * - 409 (CONFLICT): Resource conflict
     * - Other 4xx client errors
     *
     * Exception: certain 400s from Vertex AI's inline URL fetcher (used when
     * passing a file by URL to multimodal models) surface as INVALID_ARGUMENT
     * but are actually transient throttling/rate-limit signals on the
     * fetcher, not a bad request. Detect those by message substring and
     * treat them as retryable.
     *
     * @param httpStatusCode - The HTTP status code from the API error
     * @param message - The error message (used to detect transient 400 sub-cases)
     * @returns True if retryable, false if not retryable, undefined if unknown
     */
    private isGeminiErrorRetryable(httpStatusCode: number, message?: string): boolean | undefined {
        if (message && this.isNonRetryableAuthError(message)) return false;

        // Retryable status codes
        if (httpStatusCode === 408) return true; // Request timeout
        if (httpStatusCode === 429) return true; // Rate limit/quota
        if (httpStatusCode === 499) return true; // Client closed / operation cancelled
        if (httpStatusCode === 502) return true; // Bad gateway
        if (httpStatusCode === 503) return true; // Service unavailable
        if (httpStatusCode === 504) return true; // Gateway timeout
        if (httpStatusCode >= 500 && httpStatusCode < 600) return true; // Other 5xx server errors

        // Vertex AI URL fetcher transient throttling, surfaced as 400 INVALID_ARGUMENT
        // but really a Google-side rate limit on the inline-content fetcher. The fetcher
        // rejects with a family of transient throttle statuses that share the
        // THROTTLED / RATE_LIMITED / TOO_MANY_PENDING markers, e.g.
        //   URL_REJECTED-REJECTED_CLIENT_THROTTLED
        //   URL_REJECTED-REJECTED_PROXY_THROTTLED
        //   URL_REJECTED-REJECTED_RATE_LIMITED
        //   URL_REJECTED-REJECTED_FC_TOO_MANY_PENDING
        // Match on the marker rather than an exact status so new fetcher-throttle
        // variants are retried too; permanent URL rejections (robots-denied, unsafe,
        // unsupported content) lack these markers and fall through to non-retryable.
        if (httpStatusCode === 400 && message && message.includes('URL_REJECTED')) {
            if (
                message.includes('THROTTLED') ||
                message.includes('RATE_LIMITED') ||
                message.includes('TOO_MANY_PENDING')
            ) {
                return true;
            }
        }

        // A transport-level abort/cancel (request-timeout / dropped connection, sometimes reported
        // as 499 client-closed) or a deadline-exceeded is transient and should be retried,
        // even though it carries a 4xx status. Honor it before the 4xx -> non-retryable rule.
        if (message) {
            const lower = message.toLowerCase();
            if (lower.includes('aborted') || lower.includes('cancelled') || lower.includes('deadline')) return true;
        }

        // Non-retryable 4xx client errors
        if (httpStatusCode >= 400 && httpStatusCode < 500) return false;

        // Unknown status codes - let consumer decide retry strategy
        return undefined;
    }

    private isNonRetryableAuthError(message: string): boolean {
        const lowerMessage = message.toLowerCase();
        return lowerMessage.includes('invalid_grant') || lowerMessage.includes("credential's issuer");
    }

    /**
     * Extract error type name from error message.
     * Google errors often include the error type in the message.
     * Examples: "INVALID_ARGUMENT", "RESOURCE_EXHAUSTED", "PERMISSION_DENIED"
     */
    private extractErrorName(message: string): string | undefined {
        // Common Google error patterns
        const patterns = [
            /^Error code ([a-zA-Z0-9_-]+):/, // "Error code invalid_grant: message"
            /^([A-Z_]+):/, // "ERROR_NAME: message"
            /\[([A-Z_]+)\]/, // "[ERROR_NAME] message"
            /^(\w+Error):/, // "ErrorTypeError: message"
        ];

        for (const pattern of patterns) {
            const match = message.match(pattern);
            if (match) {
                return match[1];
            }
        }

        return undefined;
    }
}

/**
 * Converts functionCall and functionResponse parts to text parts in Gemini Content[].
 * Preserves tool call information while removing structured parts that require
 * tools/toolConfig to be defined in the API request.
 */
export function convertGeminiFunctionPartsToText(contents: Content[]): Content[] {
    return contents.map((content) => {
        if (!content.parts) return content;
        const hasFunctionParts = content.parts.some((p) => p.functionCall || p.functionResponse);
        if (!hasFunctionParts) return content;

        const newParts = content.parts.map((part) => {
            if (part.functionCall) {
                const argsStr = part.functionCall.args ? JSON.stringify(part.functionCall.args) : '';
                const truncated = argsStr.length > 500 ? `${argsStr.substring(0, 500)}...` : argsStr;
                return { text: `[Tool call: ${part.functionCall.name}(${truncated})]` };
            }
            if (part.functionResponse) {
                const respStr = part.functionResponse.response
                    ? JSON.stringify(part.functionResponse.response)
                    : 'No response';
                const truncated = respStr.length > 500 ? `${respStr.substring(0, 500)}...` : respStr;
                return { text: `[Tool result for ${part.functionResponse.name}: ${truncated}]` };
            }
            return part;
        });
        return { ...content, parts: newParts };
    });
}

function getToolDefinitions(tools: readonly GeminiToolDefinition[] | undefined | null): Tool | undefined {
    if (!tools || tools.length === 0) {
        return undefined;
    }
    // VertexAI Gemini only supports one tool at a time.
    // For multiple tools, we have multiple functions in one tool.
    return {
        functionDeclarations: tools.map(getToolFunction),
    };
}

function getToolFunction(tool: GeminiToolDefinition): FunctionDeclaration {
    return {
        name: tool.name,
        description: tool.description,
        // Pass the input_schema directly as a JSON Schema object.
        // parametersJsonSchema accepts standard JSON Schema and is mutually exclusive
        // with the legacy parameters field (which required a proprietary Gemini Schema type).
        parametersJsonSchema: tool.input_schema,
    };
}

/**
 * Media reference shape shared by `Part` and `FunctionResponsePart`, so the same attachment
 * mapping serves both a user turn and a tool result. Files already in Google Cloud Storage are
 * passed by URI; anything else is inlined as base64.
 */
type GeminiMediaPart =
    | { fileData: { fileUri: string; mimeType?: string } }
    | { inlineData: { data: string; mimeType?: string } };

async function fileToMediaPart(file: DataSource): Promise<GeminiMediaPart> {
    const fileUri = await file.getURI();
    if (fileUri.startsWith('gs://') || fileUri.startsWith('https://storage.googleapis.com/')) {
        return { fileData: { fileUri, mimeType: file.mime_type } };
    }
    const source = await file.getStream();
    const data = await readStreamAsBase64(
        file.mime_type.startsWith('audio/') ? boundedAudioStream(source, 25_000_000) : source,
    );
    return { inlineData: { data, mimeType: file.mime_type } };
}
