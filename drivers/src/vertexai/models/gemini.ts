import type { ApiError } from "@google/genai";
import {
    type Content, FinishReason, FunctionCallingConfigMode, type FunctionDeclaration, type GenerateContentConfig, type GenerateContentParameters,
    type GenerateContentResponseUsageMetadata,
    HarmBlockThreshold, HarmCategory, Modality, type Part,
    ProminentPeople,
    type SafetySetting, type ThinkingConfig,
    ThinkingLevel,
    type Tool
} from "@google/genai";
import {
    type AIModel, type Completion, type CompletionChunkObject, type CompletionResult, type ExecutionOptions,
    type ExecutionTokenUsage,
    getConversationMeta,
    getGeminiModelVersion,
    incrementConversationTurn,
    isGeminiModelVersionGte,
    type JSONObject, LlumiverseError, type LlumiverseErrorContext, ModelType, type PromptOptions, PromptRole,
    type PromptSegment, readStreamAsBase64, type StatelessExecutionOptions,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    type ToolDefinition, type ToolUse,
    truncateLargeTextInConversation,
    unwrapConversationArray,
    type VertexAIGeminiOptions
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import type { GenerateContentPrompt, VertexAIDriver } from "../index.js";
import type { ModelDefinition } from "../models.js";

function supportsStructuredOutput(options: PromptOptions): boolean {
    // Gemini 1.0 Ultra does not support JSON output, 1.0 Pro does.
    return !!options.result_schema && !options.model.includes("ultra");
}

const geminiSafetySettings: SafetySetting[] = [
    {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH
    },
    {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH
    },
    {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH
    },
    {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH
    },
    {
        category: HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH
    },
    {
        category: HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
];

// We do the mapping here rather than in common to avoid bringing the SDK into the common package.
function getProminentPeopleOption(prominentPeople?: "PROMINENT_PEOPLE_UNSPECIFIED" | "ALLOW_PROMINENT_PEOPLE" | "BLOCK_PROMINENT_PEOPLE") {
    switch (prominentPeople) {
        case "ALLOW_PROMINENT_PEOPLE":
            return ProminentPeople.ALLOW_PROMINENT_PEOPLE;
        case "BLOCK_PROMINENT_PEOPLE":
            return ProminentPeople.BLOCK_PROMINENT_PEOPLE;
        case "PROMINENT_PEOPLE_UNSPECIFIED":
            return ProminentPeople.PROMINENT_PEOPLE_UNSPECIFIED;
        default:
            return undefined;
    }
}

function getGeminiPayload(options: ExecutionOptions, prompt: GenerateContentPrompt): GenerateContentParameters {
    const model_options = options.model_options as VertexAIGeminiOptions | undefined;
    const tools = getToolDefinitions(options.tools);

    // When no tools are provided but conversation contains functionCall/functionResponse parts
    // (e.g. checkpoint summary calls), convert them to text to avoid API errors.
    // Use a local variable to avoid mutating the caller's conversation object.
    let payloadContents = prompt.contents;
    if (!tools && payloadContents) {
        const hasToolParts = payloadContents.some(c =>
            c.parts?.some(p => p.functionCall || p.functionResponse)
        );
        if (hasToolParts) {
            payloadContents = convertGeminiFunctionPartsToText(payloadContents);
        }
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
        }
    }

    const config: GenerateContentConfig = {
        systemInstruction: prompt.system,
        safetySettings: geminiSafetySettings,
        tools: tools ? [tools] : undefined,
        toolConfig: tools ? {
            functionCallingConfig: {
                mode: FunctionCallingConfigMode.AUTO,
            }
        } : undefined,
        candidateCount: 1,
        //JSON/Structured output
        responseMimeType: useStructuredOutput ? "application/json" : undefined,
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
    }

    return {
        model: options.model,
        contents: payloadContents,
        config: options.model.toLowerCase().includes("image") ? configNanoBanana : config,
    };
}

/**
 * Collect all parts (text and images) from content in order.
 * This preserves the original ordering of text and image parts.
 */
function extractCompletionResults(content: Content): CompletionResult[] {
    const results: CompletionResult[] = [];
    const parts = content.parts;
    if (parts) {
        for (const part of parts) {
            if (part.text) {
                results.push({
                    type: "text",
                    value: part.text
                });
            } else if (part.inlineData) {
                const base64ImageBytes: string = part.inlineData.data ?? "";
                const mimeType = part.inlineData.mimeType ?? "image/png";
                const imageUrl = `data:${mimeType};base64,${base64ImageBytes}`;
                results.push({
                    type: "image",
                    value: imageUrl
                });
            }
        }
    }
    return results;
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

export function mergeConsecutiveRole(contents: Content[] | undefined): Content[] {
    if (!contents || contents.length === 0) return [];

    const needsMerging = contents.some((content, i) =>
        i < contents.length - 1 && content.role === contents[i + 1].role
    );
    // If no merging needed, return original array
    if (!needsMerging) {
        return contents;
    }

    const result: Content[] = [];
    let currentContent = { ...contents[0], parts: [...(contents[0].parts || [])] };

    for (let i = 1; i < contents.length; i++) {
        if (currentContent.role === contents[i].role) {
            // Same role - concatenate parts (without merging individual parts)
            currentContent.parts = (currentContent.parts || []).concat(...(contents[i].parts || []));
        } else {
            // Different role - push current and start new
            result.push(currentContent);
            currentContent = { ...contents[i], parts: [...(contents[i].parts || [])] };
        }
    }

    result.push(currentContent);
    return result;
}

const supportedFinishReasons: FinishReason[] = [
    FinishReason.MAX_TOKENS,
    FinishReason.STOP,
    FinishReason.FINISH_REASON_UNSPECIFIED,
]

// Finish reasons that indicate tool call issues but should be recovered gracefully
// instead of throwing an error. The tool_use is still extracted and returned
// so the workflow can generate a proper toolError response.
const recoverableToolCallReasons = [
    'UNEXPECTED_TOOL_CALL', // Model called an undeclared tool
]


function geminiThinkingBudget(option: StatelessExecutionOptions) {
    const model_options = option.model_options as VertexAIGeminiOptions | undefined;
    // If thinking_budget_tokens is explicitly set in model options, use it directly
    if (model_options?.thinking_budget_tokens !== undefined) {
        return model_options.thinking_budget_tokens;
    }
    if (model_options?.effort) {
        return geminiBudgetForEffort(option.model, model_options.effort);
    }
    // Set minimum thinking level by default.
    // Docs: https://ai.google.dev/gemini-api/docs/thinking#set-budget
    if (getGeminiModelVersion(option.model) === '2.5') {
        if (option.model.includes("pro")) {
            return 128;
        }
        return 0;
    }
    return undefined;
}

function geminiThinkingLevelForEffort(model: string, effort: VertexAIGeminiOptions["effort"]): ThinkingLevel | undefined {
    if (model.includes("gemini-3-pro-image")) {
        return ThinkingLevel.HIGH;
    }
    if (model.includes("gemini-3.1-flash-image")) {
        return effort === "low" ? ThinkingLevel.MINIMAL : ThinkingLevel.HIGH;
    }
    switch (effort) {
        case "low":
            return ThinkingLevel.LOW;
        case "medium":
            return ThinkingLevel.MEDIUM;
        case "high":
            return ThinkingLevel.HIGH;
        default:
            return undefined;
    }
}

function geminiBudgetForEffort(model: string, effort: NonNullable<VertexAIGeminiOptions["effort"]>): number {
    const isFlashLite = model.includes("flash-lite");
    const isFlash = model.includes("flash") && !isFlashLite;
    const isPro = model.includes("pro");

    if (effort === "low") {
        if (isPro) return 128;
        if (isFlashLite) return 512;
        if (isFlash) return 1;
        return 1024;
    }
    if (effort === "medium") {
        return 8192;
    }
    if (isPro) return 32768;
    if (isFlash || isFlashLite) return 24576;
    return 8192;
}

function geminiThinkingConfig(option: StatelessExecutionOptions): ThinkingConfig | undefined {
    const model_options = option.model_options as VertexAIGeminiOptions | undefined;

    // If thinking options are explicitly set in model options, use them directly
    const include_thoughts = model_options?.include_thoughts ?? false;
    if (model_options?.thinking_budget_tokens !== undefined || model_options?.thinking_level) {
        return {
            includeThoughts: include_thoughts,
            thinkingBudget: model_options.thinking_budget_tokens,
            thinkingLevel: model_options.thinking_level,
        };
    }
    if (model_options?.effort) {
        if (isGeminiModelVersionGte(option.model, '3.0')) {
            return {
                includeThoughts: include_thoughts,
                thinkingLevel: geminiThinkingLevelForEffort(option.model, model_options.effort),
            };
        }
        return {
            includeThoughts: include_thoughts,
            thinkingBudget: geminiBudgetForEffort(option.model, model_options.effort),
        };
    }

    // Set a low thinking level by default.
    // Docs: https://ai.google.dev/gemini-api/docs/thinking#set-budget
    // https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking
    if (isGeminiModelVersionGte(option.model, '3.0')) {
        return {
            includeThoughts: include_thoughts,
            thinkingLevel: option.model.includes("gemini-3-pro-image") ? ThinkingLevel.HIGH : ThinkingLevel.LOW
        };
    }
    if (isGeminiModelVersionGte(option.model, '2.5')) {
        const thinking_budget_tokens = geminiThinkingBudget(option) ?? 0;
        return {
            includeThoughts: include_thoughts,
            thinkingBudget: thinking_budget_tokens
        };
    }
}

export class GeminiModelDefinition implements ModelDefinition<GenerateContentPrompt> {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true
        } satisfies AIModel;
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: ExecutionOptions): Promise<GenerateContentPrompt> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const schema = options.result_schema;
        let contents: Content[] = [];
        let system: Content | undefined = { role: "user", parts: [] }; // Single content block for system messages

        const safety: Content[] = [];

        for (const msg of segments) {
            // Role specific handling
            if (msg.role === PromptRole.system) {
                // Text only for system messages
                if (msg.files && msg.files.length > 0) {
                    throw new Error("Gemini does not support files/images etc. in system messages. Only text content is allowed.");
                }

                if (msg.content) {
                    system.parts?.push({
                        text: msg.content
                    });
                }
            } else if (msg.role === PromptRole.tool) {
                if (!msg.tool_use_id) {
                    throw new Error("Tool response missing tool_use_id");
                }
                // Build functionResponse part with optional thought_signature for Gemini thinking models
                const functionResponsePart: Part = {
                    functionResponse: {
                        name: msg.tool_use_id,
                        response: formatFunctionResponse(msg.content || ''),
                    },
                    // Include thought_signature if provided (required for Gemini 2.5+/3.0+ thinking models)
                    thoughtSignature: msg.thought_signature,
                };
                contents.push({
                    role: 'user',
                    parts: [functionResponsePart]
                });
            } else {    // PromptRole.user, PromptRole.assistant, PromptRole.safety
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
                        const fileUrl = await f.getURL();
                        const isGsUrl = fileUrl.startsWith('gs://') || fileUrl.startsWith('https://storage.googleapis.com/');

                        if (isGsUrl) {
                            parts.push({
                                fileData: {
                                    fileUri: fileUrl,
                                    mimeType: f.mime_type
                                }
                            });
                        } else {
                            // Inline data handling
                            const stream = await f.getStream();
                            const data = await readStreamAsBase64(stream);
                            parts.push({
                                inlineData: {
                                    data,
                                    mimeType: f.mime_type
                                }
                            });
                        }
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
                system.parts?.push({ text: "Fill all appropriate fields in the JSON output." });
            } else {
                // Fallback to putting the schema in the system instructions, if not using structured output.
                if (options.tools) {
                    system.parts?.push({
                        text: "When not calling tools, the output must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema)
                    });
                } else {
                    system.parts?.push({ text: "The output must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema) });
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

        // Merge consecutive messages with the same role. Note: this may not be necessary, works without it, keeping to match previous behavior.
        contents = mergeConsecutiveRole(contents);

        return { contents, system };
    }

    usageMetadataToTokenUsage(usageMetadata: GenerateContentResponseUsageMetadata | undefined): ExecutionTokenUsage {
        if (!usageMetadata || !usageMetadata.totalTokenCount) {
            return {};
        }
        const tokenUsage: ExecutionTokenUsage = {
            total: usageMetadata.totalTokenCount,
            prompt: usageMetadata.promptTokenCount,
            prompt_cached: usageMetadata.cachedContentTokenCount ?? undefined,
            prompt_new: (usageMetadata.promptTokenCount ?? 0) - (usageMetadata.cachedContentTokenCount ?? 0),
        };

        //Output/Response side
        tokenUsage.result = (usageMetadata.candidatesTokenCount ?? 0)
            + (usageMetadata.thoughtsTokenCount ?? 0)
            + (usageMetadata.toolUsePromptTokenCount ?? 0);

        if ((tokenUsage.total ?? 0) !== (tokenUsage.prompt ?? 0) + tokenUsage.result) {
            console.warn("[VertexAI] Gemini token usage mismatch: total does not equal prompt + result", {
                total: tokenUsage.total,
                prompt: tokenUsage.prompt,
                result: tokenUsage.result
            });
        }

        if (!tokenUsage.result) {
            tokenUsage.result = undefined; // If no result, mark as undefined
        }

        return tokenUsage;
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: GenerateContentPrompt, options: ExecutionOptions): Promise<Completion> {
        const splits = options.model.split("/");
        let region: string | undefined = undefined;
        if (splits[0] === "locations" && splits.length >= 2) {
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

        let conversation = updateConversation(options.conversation, prompt.contents);
        prompt.contents = conversation;

        // TODO: Remove hack, use global endpoint manually if needed.
        if (options.model.includes("gemini-2.5-flash-image")) {
            region = "global"; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const model_options = options.model_options as VertexAIGeminiOptions | undefined;
        const client = driver.getGoogleGenAIClient(region, model_options?.flex ?? false);

        const payload = getGeminiPayload(options, prompt);
        const response = await client.models.generateContent(payload);

        const token_usage: ExecutionTokenUsage = this.usageMetadataToTokenUsage(response.usageMetadata);

        let tool_use: ToolUse[] | undefined;
        let finish_reason: string | undefined, result: any;
        const candidate = response.candidates && response.candidates[0];
        if (candidate) {
            switch (candidate.finishReason) {
                case FinishReason.MAX_TOKENS: finish_reason = "length"; break;
                case FinishReason.STOP: finish_reason = "stop"; break;
                default: finish_reason = candidate.finishReason;
            }
            const content = candidate.content;

            // Check for unsupported finish reasons, but allow recoverable tool call issues
            const isRecoverableToolCall = recoverableToolCallReasons.includes(candidate.finishReason as string);
            if (candidate.finishReason && !supportedFinishReasons.includes(candidate.finishReason) && !isRecoverableToolCall) {
                throw new Error(`Unsupported finish reason: ${candidate.finishReason}, `
                    + `finish message: ${candidate.finishMessage}, `
                    + `content: ${JSON.stringify(content, null, 2)}, safety: ${JSON.stringify(candidate.safetyRatings, null, 2)}`);
            }

            if (content) {
                tool_use = collectToolUseParts(content);

                // For recoverable tool call issues, log warning but continue processing
                // The workflow will handle the invalid tool call gracefully
                if (isRecoverableToolCall && tool_use && tool_use.length > 0) {
                    console.warn(`[Gemini] Recoverable tool call issue (${candidate.finishReason}): ` +
                        `Model tried to call undeclared tool(s): ${tool_use.map(t => t.tool_name).join(', ')}`);
                }

                result = extractCompletionResults(content);
                conversation = updateConversation(conversation, [content]);
            }
        }



        if (tool_use) {
            finish_reason = "tool_use";
        }

        // Increment turn counter for deferred stripping
        conversation = incrementConversationTurn(conversation) as Content[];

        // Strip large base64 image data based on options.stripImagesAfterTurns
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens
        };
        let processedConversation = stripBase64ImagesFromConversation(conversation, stripOptions);

        // Truncate large text content if configured
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);

        // Strip old heartbeat status messages
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        // Preserve system instruction in conversation for multi-turn support
        const finalConversation = storeSystemInConversation(processedConversation, prompt.system);

        return {
            result: result && result.length > 0 ? result : [{ type: "text" as const, value: '' }],
            token_usage: token_usage,
            finish_reason: finish_reason,
            original_response: options.include_original_response ? response : undefined,
            conversation: finalConversation,
            tool_use
        } satisfies Completion;
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: GenerateContentPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const splits = options.model.split("/");
        let region: string | undefined = undefined;
        if (splits[0] === "locations" && splits.length >= 2) {
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

        if (options.model.includes("gemini-2.5-flash-image")) {
            region = "global"; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const model_options = options.model_options as VertexAIGeminiOptions | undefined;
        const client = driver.getGoogleGenAIClient(region, model_options?.flex ?? false);

        const payload = getGeminiPayload(options, prompt);
        const response = await client.models.generateContentStream(payload);

        const stream = asyncMap(response, async (item) => {
            const token_usage: ExecutionTokenUsage = this.usageMetadataToTokenUsage(item.usageMetadata);
            if (item.candidates && item.candidates.length > 0) {
                for (const candidate of item.candidates) {
                    let tool_use: ToolUse[] | undefined;
                    let finish_reason: string | undefined;
                    switch (candidate.finishReason) {
                        case FinishReason.MAX_TOKENS: finish_reason = "length"; break;
                        case FinishReason.STOP: finish_reason = "stop"; break;
                        default: finish_reason = candidate.finishReason;
                    }
                    // Check for unsupported finish reasons, but allow recoverable tool call issues
                    const isRecoverableToolCall = recoverableToolCallReasons.includes(candidate.finishReason as string);
                    if (candidate.finishReason && !supportedFinishReasons.includes(candidate.finishReason) && !isRecoverableToolCall) {
                        throw new Error(`Unsupported finish reason: ${candidate.finishReason}, `
                            + `finish message: ${candidate.finishMessage}, `
                            + `content: ${JSON.stringify(candidate.content, null, 2)}, safety: ${JSON.stringify(candidate.safetyRatings, null, 2)}`);
                    }
                    if (candidate.content?.role === 'model') {
                        // Collect all parts in order (text and images)
                        const combinedResults = extractCompletionResults(candidate.content);
                        tool_use = collectToolUseParts(candidate.content);
                        if (tool_use) {
                            finish_reason = "tool_use";
                            // Log warning for recoverable tool call issues
                            if (isRecoverableToolCall) {
                                console.warn(`[Gemini] Recoverable tool call issue (${candidate.finishReason}): ` +
                                    `Model tried to call undeclared tool(s): ${tool_use.map(t => t.tool_name).join(', ')}`);
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
                result: item.promptFeedback?.blockReasonMessage ? [{ type: "text" as const, value: item.promptFeedback.blockReasonMessage }] : [],
                finish_reason: item.promptFeedback?.blockReason ?? "",
                token_usage: token_usage,
            };
        });

        return stream;
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
    formatLlumiverseError(
        _driver: VertexAIDriver,
        error: unknown,
        context: LlumiverseErrorContext
    ): LlumiverseError {
        // Check if it's a Google API error with status code
        const isApiError = this.isGoogleApiError(error);

        if (!isApiError) {
            // Not a Google API error, use default handling
            // This will be called by the driver's default formatLlumiverseError
            throw error;
        }

        const apiError = error as ApiError;
        const httpStatusCode = apiError.status;

        // Extract error message
        const message = apiError.message || String(error);

        // Build user-facing message with status code
        let userMessage = message;

        // Include status code in message (for end-user visibility)
        if (httpStatusCode) {
            userMessage = `[${httpStatusCode}] ${userMessage}`;
        }

        // Determine retryability based on Google error codes
        const retryable = this.isGeminiErrorRetryable(httpStatusCode);

        // Extract error name/type from message if present
        const errorName = this.extractErrorName(message);

        return new LlumiverseError(
            `[${context.provider}] ${userMessage}`,
            retryable,
            context,
            error,
            httpStatusCode,
            errorName
        );
    }

    /**
     * Type guard to check if error is a Google API error.
     */
    private isGoogleApiError(error: unknown): error is ApiError {
        return (
            error !== null &&
            typeof error === 'object' &&
            'status' in error &&
            typeof (error as any).status === 'number' &&
            'message' in error
        );
    }

    /**
     * Determine if a Google API error is retryable based on HTTP status code.
     * 
     * Retryable errors (per Google AIP-194):
     * - 408 (REQUEST_TIMEOUT): Request timeout
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
     * @param httpStatusCode - The HTTP status code from the API error
     * @returns True if retryable, false if not retryable, undefined if unknown
     */
    private isGeminiErrorRetryable(httpStatusCode: number): boolean | undefined {
        // Retryable status codes
        if (httpStatusCode === 408) return true; // Request timeout
        if (httpStatusCode === 429) return true; // Rate limit/quota
        if (httpStatusCode === 502) return true; // Bad gateway
        if (httpStatusCode === 503) return true; // Service unavailable
        if (httpStatusCode === 504) return true; // Gateway timeout
        if (httpStatusCode >= 500 && httpStatusCode < 600) return true; // Other 5xx server errors

        // Non-retryable 4xx client errors
        if (httpStatusCode >= 400 && httpStatusCode < 500) return false;

        // Unknown status codes - let consumer decide retry strategy
        return undefined;
    }

    /**
     * Extract error type name from error message.
     * Google errors often include the error type in the message.
     * Examples: "INVALID_ARGUMENT", "RESOURCE_EXHAUSTED", "PERMISSION_DENIED"
     */
    private extractErrorName(message: string): string | undefined {
        // Common Google error patterns
        const patterns = [
            /^([A-Z_]+):/,  // "ERROR_NAME: message"
            /\[([A-Z_]+)\]/, // "[ERROR_NAME] message"
            /^(\w+Error):/,  // "ErrorTypeError: message"
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
    return contents.map(content => {
        if (!content.parts) return content;
        const hasFunctionParts = content.parts.some(p => p.functionCall || p.functionResponse);
        if (!hasFunctionParts) return content;

        const newParts = content.parts.map(part => {
            if (part.functionCall) {
                const argsStr = part.functionCall.args ? JSON.stringify(part.functionCall.args) : '';
                const truncated = argsStr.length > 500 ? argsStr.substring(0, 500) + '...' : argsStr;
                return { text: `[Tool call: ${part.functionCall.name}(${truncated})]` };
            }
            if (part.functionResponse) {
                const respStr = part.functionResponse.response
                    ? JSON.stringify(part.functionResponse.response) : 'No response';
                const truncated = respStr.length > 500 ? respStr.substring(0, 500) + '...' : respStr;
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
    }
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
    const convArray = unwrapped ?? (conversation as Content[] || []);
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
        return { ...conversation as object, [SYSTEM_KEY]: system };
    }
    return conversation;
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
    if (response.startsWith("{") && response.endsWith("}")) {
        try {
            return JSON.parse(response);
        } catch (e) {
            return { output: response };
        }
    } else {
        return { output: response };
    }
}
