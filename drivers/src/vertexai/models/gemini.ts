import type { ApiError } from '@google/genai';
import {
    type Content,
    FinishReason,
    FunctionCallingConfigMode,
    type FunctionDeclaration,
    type FunctionResponsePart,
    type GenerateContentConfig,
    type GenerateContentParameters,
    type GenerateContentResponseUsageMetadata,
    HarmBlockThreshold,
    HarmCategory,
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
    type AIModel,
    type Completion,
    type CompletionResult,
    type DataSource,
    type DriverCompletionStream,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    getConversationMeta,
    incrementConversationTurn,
    isGeminiModelVersionGte,
    type JSONObject,
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
    unwrapConversationArray,
    type VertexAIGeminiOptions,
} from '@llumiverse/core';
import { asyncMap } from '@llumiverse/core/async';
import { truncateBinaryForDebug } from '../../shared/debug-prompt.js';
import type { GenerateContentPrompt, VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';
import { generateWithGeminiContextCache } from './gemini-context-cache.js';

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
            if (!part.inlineData?.data) {
                return part;
            }
            return {
                ...part,
                inlineData: {
                    ...part.inlineData,
                    data: truncateBinaryForDebug(part.inlineData.data),
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

export function getGeminiPayload(options: ExecutionOptions, prompt: GenerateContentPrompt): GenerateContentParameters {
    const model_options = options.model_options as VertexAIGeminiOptions | undefined;
    const tools = getToolDefinitions(options.tools);

    // When no tools are provided but conversation contains functionCall/functionResponse parts
    // (e.g. checkpoint summary calls), convert them to text to avoid API errors.
    // Use a local variable to avoid mutating the caller's conversation object.
    let payloadContents = mergeFunctionResponseContents(prompt.contents ?? []);
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
                      mode: FunctionCallingConfigMode.AUTO,
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

function finalizeGeminiConversation(
    conversation: Content[],
    assistantContent: Content | undefined,
    system: Content | undefined,
    options: ExecutionOptions,
): GenerateContentPrompt['contents'] {
    let completed = assistantContent ? updateConversation(conversation, [assistantContent]) : conversation;
    completed = incrementConversationTurn(completed) as Content[];
    const currentTurn = getConversationMeta(completed).turnNumber;
    const preserveSubtree = (value: unknown): boolean => {
        if (!value || typeof value !== 'object') return false;
        const thoughtSignature = (value as { thoughtSignature?: unknown }).thoughtSignature;
        return typeof thoughtSignature === 'string' && thoughtSignature.length > 0;
    };
    const stripOptions = {
        keepForTurns: options.stripImagesAfterTurns ?? Infinity,
        currentTurn,
        textMaxTokens: options.stripTextMaxTokens,
        preserveSubtree,
    };
    let processed = stripBase64ImagesFromConversation(completed, stripOptions);
    processed = truncateLargeTextInConversation(processed, stripOptions);
    processed = stripHeartbeatsFromConversation(processed, {
        keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
        currentTurn,
        preserveSubtree,
    });
    return storeSystemInConversation(processed, system) as Content[];
}

function appendGeminiStreamParts(target: Part[], incoming: Part[]): void {
    for (const part of incoming) {
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
        } else {
            target.push(structuredClone(part));
        }
    }
}

function collectToolUseParts(content: Content): ToolUse[] | undefined {
    const out: ToolUse[] = [];
    const parts = content.parts ?? [];
    for (const part of parts) {
        if (part.functionCall) {
            const toolUse: ToolUse = {
                id: part.functionCall.name ?? '',
                tool_name: part.functionCall.name ?? '',
                tool_input: part.functionCall.args as JSONObject,
            };
            // Capture thought_signature for Gemini thinking models (2.5+/3.0+)
            // This must be passed back with the function response
            if (part.thoughtSignature) {
                toolUse.thought_signature = part.thoughtSignature;
            }
            out.push(toolUse);
        }
    }
    return out.length > 0 ? out : undefined;
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

export class GeminiModelDefinition implements ModelDefinition<GenerateContentPrompt> {
    model: AIModel;

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
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
                const functionResponsePart: Part = {
                    functionResponse: {
                        name: msg.tool_use_id,
                        response: formatFunctionResponse(msg.content || ''),
                        ...(responseParts.length > 0 && { parts: responseParts }),
                    },
                    // Include thought_signature if provided (required for Gemini 2.5+/3.0+ thinking models)
                    thoughtSignature: msg.thought_signature,
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

        return tokenUsage;
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: GenerateContentPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) {
            region = splits[1];
        }
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        // Restore system instruction from stored conversation on resume.
        // The stored _llumiverse_system contains the complete system (interaction prompt + schema)
        // from the initial call. Always prefer it over the prompt's system, which on resume only
        // contains the schema instruction (no interaction system segments are present on resume).
        const existingSystem = extractSystemFromConversation(options.conversation);
        if (existingSystem) {
            prompt.system = existingSystem;
        }

        const conversation = updateConversation(options.conversation, prompt.contents);
        prompt.contents = conversation;

        // TODO: Remove hack, use global endpoint manually if needed.
        if (options.model.includes('gemini-2.5-flash-image')) {
            region = 'global'; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const model_options = options.model_options as VertexAIGeminiOptions | undefined;
        const includeThoughts = model_options?.include_thoughts !== false;
        const client = driver.getGoogleGenAIClient(
            region,
            resolveVertexAIServiceTier(model_options),
            options.httpTimeout,
        );

        const payload = getGeminiPayload(options, prompt);
        if (signal) payload.config = { ...payload.config, abortSignal: signal };
        // Routes through an explicit Vertex context cache when this execution carries a
        // prompt_cache_key; sends `payload` untouched otherwise, and on any cache failure.
        const cacheExecution = await generateWithGeminiContextCache(
            driver,
            client,
            options,
            prompt,
            payload,
            (request) => client.models.generateContent(request),
            region ?? driver.getVertexRegion?.() ?? 'global',
        );
        const response = cacheExecution.value;

        const token_usage: ExecutionTokenUsage = this.usageMetadataToTokenUsage(driver, response.usageMetadata);

        let tool_use: ToolUse[] | undefined;
        let finalContent: Content | undefined;
        let finish_reason: string | undefined, result: CompletionResult[] | undefined;
        const candidate = response.candidates?.[0];
        if (candidate) {
            switch (candidate.finishReason) {
                case FinishReason.MAX_TOKENS:
                    finish_reason = 'length';
                    break;
                case FinishReason.STOP:
                    finish_reason = 'stop';
                    break;
                default:
                    finish_reason = candidate.finishReason;
            }
            const content = candidate.content;

            // Provider finish reasons are terminal responses, not transport failures. Classify them
            // explicitly while allowing recoverable tool-call issues to continue through the workflow.
            const isRecoverableToolCall = assertSupportedGeminiFinishReason(candidate);

            if (content) {
                tool_use = collectToolUseParts(content);

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
        }

        if (tool_use) {
            finish_reason = 'tool_use';
        }

        const finalConversation = finalizeGeminiConversation(conversation, finalContent, prompt.system, options);

        return {
            result: result && result.length > 0 ? result : [{ type: 'text' as const, value: '' }],
            token_usage: token_usage,
            finish_reason: finish_reason,
            original_response: options.include_original_response ? response : undefined,
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
        const splits = options.model.split('/');
        let region: string | undefined;
        if (splits[0] === 'locations' && splits.length >= 2) {
            region = splits[1];
        }
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        // Restore system instruction from stored conversation on resume.
        // The stored _llumiverse_system contains the complete system (interaction prompt + schema)
        // from the initial call. Always prefer it over the prompt's system, which on resume only
        // contains the schema instruction (no interaction system segments are present on resume).
        const existingSystem = extractSystemFromConversation(options.conversation);
        if (existingSystem) {
            prompt.system = existingSystem;
        }

        // Include conversation history in prompt contents (same as non-streaming)
        const conversation = updateConversation(options.conversation, prompt.contents);
        prompt.contents = conversation;

        if (options.model.includes('gemini-2.5-flash-image')) {
            region = 'global'; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const model_options = options.model_options as VertexAIGeminiOptions | undefined;
        const includeThoughts = model_options?.include_thoughts !== false;
        const client = driver.getGoogleGenAIClient(
            region,
            resolveVertexAIServiceTier(model_options),
            options.httpTimeout,
        );

        const payload = getGeminiPayload(options, prompt);
        payload.config = { ...payload.config, abortSignal: signal };
        const cacheExecution = await generateWithGeminiContextCache(
            driver,
            client,
            options,
            prompt,
            payload,
            (request) => client.models.generateContentStream(request),
            region ?? driver.getVertexRegion?.() ?? 'global',
        );
        const response = cacheExecution.value;

        const nativeParts: Part[] = [];
        const stream = asyncMap(response, async (item) => {
            const token_usage: ExecutionTokenUsage = this.usageMetadataToTokenUsage(driver, item.usageMetadata);
            if (item.candidates && item.candidates.length > 0) {
                for (const candidate of item.candidates) {
                    let tool_use: ToolUse[] | undefined;
                    let finish_reason: string | undefined;
                    switch (candidate.finishReason) {
                        case FinishReason.MAX_TOKENS:
                            finish_reason = 'length';
                            break;
                        case FinishReason.STOP:
                            finish_reason = 'stop';
                            break;
                        default:
                            finish_reason = candidate.finishReason;
                    }
                    const isRecoverableToolCall = assertSupportedGeminiFinishReason(candidate);
                    if (candidate.content?.role === 'model') {
                        appendGeminiStreamParts(nativeParts, candidate.content.parts ?? []);
                        // Collect all parts in order (text and images)
                        const combinedResults = extractCompletionResults(candidate.content, includeThoughts);
                        tool_use = collectToolUseParts(candidate.content);
                        if (tool_use) {
                            finish_reason = 'tool_use';
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
                            finish_reason: finish_reason,
                            tool_use,
                        };
                    }
                }
            }
            //No normal output, returning block reason if it exists.
            return {
                result: item.promptFeedback?.blockReasonMessage
                    ? [{ type: 'text' as const, value: item.promptFeedback.blockReasonMessage }]
                    : [],
                finish_reason: item.promptFeedback?.blockReason ?? '',
                token_usage: token_usage,
            };
        });

        return Object.assign(stream, {
            finalizePromptCacheDiagnostic: () => cacheExecution.diagnostic,
            finalizeConversation: () =>
                finalizeGeminiConversation(
                    conversation,
                    nativeParts.length > 0 ? { role: 'model', parts: nativeParts } : undefined,
                    prompt.system,
                    options,
                ),
        });
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

function getToolDefinitions(tools: ToolDefinition[] | undefined | null): Tool | undefined {
    if (!tools || tools.length === 0) {
        return undefined;
    }
    // VertexAI Gemini only supports one tool at a time.
    // For multiple tools, we have multiple functions in one tool.
    return {
        functionDeclarations: tools.map(getToolFunction),
    };
}

function getToolFunction(tool: ToolDefinition): FunctionDeclaration {
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
 * Update the conversation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: unknown, prompt: Content[]): Content[] {
    // Unwrap array if wrapped, otherwise treat as array
    const unwrapped = unwrapConversationArray<Content>(conversation);
    const convArray = unwrapped ?? ((conversation as Content[]) || []);
    return convArray.concat(prompt);
}

const SYSTEM_KEY = '_llumiverse_system';

/**
 * Extract the stored system instruction from a Gemini conversation object.
 * Returns undefined if no system was stored.
 */
function extractSystemFromConversation(conversation: unknown): Content | undefined {
    if (typeof conversation === 'object' && conversation !== null) {
        const c = conversation as Record<string, unknown>;
        if (c[SYSTEM_KEY] && typeof c[SYSTEM_KEY] === 'object') {
            return c[SYSTEM_KEY] as Content;
        }
    }
    return undefined;
}

/**
 * Store the system instruction in the Gemini conversation wrapper object.
 * The conversation is already wrapped by incrementConversationTurn into
 * { _arrayConversation: Content[], _llumiverse_meta: {...} }.
 * We add _llumiverse_system alongside these fields.
 */
function storeSystemInConversation(conversation: unknown, system: Content | undefined): unknown {
    if (!system) return conversation;
    if (typeof conversation === 'object' && conversation !== null) {
        return { ...(conversation as object), [SYSTEM_KEY]: system };
    }
    return conversation;
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
    const data = await readStreamAsBase64(await file.getStream());
    return { inlineData: { data, mimeType: file.mime_type } };
}

/**
 *
 * Gemini supports JSON output in the response. so we test if the response is a valid JSON object. otherwise we treat the response as a string.
 *
 * This is an excerpt from googleapis.github.io/python-genai:
 *
 * The function response in JSON object format.
 * Use “output” key to specify function output and “error” key to specify error details (if any).
 * If “output” and “error” keys are not specified, then whole “response” is treated as function output.
 * @see https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse
 */
function formatFunctionResponse(response: string): JSONObject {
    response = response.trim();
    if (response.startsWith('{') && response.endsWith('}')) {
        try {
            return JSON.parse(response);
        } catch {
            return { output: response };
        }
    } else {
        return { output: response };
    }
}
