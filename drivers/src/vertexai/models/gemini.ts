import type { ApiError } from "@google/genai";
import {
    Content, FinishReason, FunctionCallingConfigMode, FunctionDeclaration, GenerateContentConfig, GenerateContentParameters,
    GenerateContentResponseUsageMetadata,
    HarmBlockThreshold, HarmCategory, Modality, Part, SafetySetting, Schema, ThinkingConfig, Tool, Type
} from "@google/genai";
import {
    AIModel, Completion, CompletionChunkObject, CompletionResult, ExecutionOptions,
    ExecutionTokenUsage,
    getConversationMeta,
    getMaxTokensLimitVertexAi,
    incrementConversationTurn,
    JSONObject, JSONSchema, LlumiverseError, LlumiverseErrorContext, ModelType, PromptOptions, PromptRole,
    PromptSegment, readStreamAsBase64, StatelessExecutionOptions,
    stripBase64ImagesFromConversation,
    ToolDefinition, ToolUse,
    truncateLargeTextInConversation,
    unwrapConversationArray,
    VertexAIGeminiOptions
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { GenerateContentPrompt, VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";

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

function getGeminiPayload(options: ExecutionOptions, prompt: GenerateContentPrompt): GenerateContentParameters {
    const model_options = options.model_options as VertexAIGeminiOptions | undefined;
    const tools = getToolDefinitions(options.tools);

    const useStructuredOutput = supportsStructuredOutput(options) && !tools;

    const thinkingConfigNeeded = model_options?.include_thoughts
        || model_options?.thinking_budget_tokens
        || options.model.includes("gemini-2.5");

    const configNanoBanana: GenerateContentConfig = {
        systemInstruction: prompt.system,
        safetySettings: geminiSafetySettings,
        responseModalities: [Modality.TEXT, Modality.IMAGE], // This is an error if only Text, and Only Image just gets blank responses.
        candidateCount: 1,
        //Model options
        temperature: model_options?.temperature,
        topP: model_options?.top_p,
        maxOutputTokens: geminiMaxTokens(options),
        stopSequences: model_options?.stop_sequence,
        imageConfig: {
            aspectRatio: model_options?.image_aspect_ratio,
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
        responseSchema: useStructuredOutput ? parseJSONtoSchema(options.result_schema, true) : undefined,
        //Model options
        temperature: model_options?.temperature,
        topP: model_options?.top_p,
        topK: model_options?.top_k,
        maxOutputTokens: geminiMaxTokens(options),
        stopSequences: model_options?.stop_sequence,
        presencePenalty: model_options?.presence_penalty,
        frequencyPenalty: model_options?.frequency_penalty,
        seed: model_options?.seed,
        thinkingConfig: thinkingConfigNeeded ? geminiThinkingConfig(options) : undefined,
    }

    return {
        model: options.model,
        contents: prompt.contents,
        config: options.model.toLowerCase().includes("image") ? configNanoBanana : config,
    };
}

/**
 * Convert JSONSchema to Gemini Schema,
 * Make all properties required by default
 * Properties previously marked as optional will be marked as nullable.
 */
function parseJSONtoSchema(schema?: JSONSchema, requiredAll = false): Schema {
    if (!schema) {
        return {};
    }

    return convertSchema(schema, 0, requiredAll);
}

/**
 * Convert JSONSchema type to Gemini Schema Type
 */
function convertType(type?: string | string[]): Type | undefined {
    if (!type) return undefined;

    // Handle single type
    if (typeof type === 'string') {
        switch (type) {
            case 'string': return Type.STRING;
            case 'number': return Type.NUMBER;
            case 'integer': return Type.INTEGER;
            case 'boolean': return Type.BOOLEAN;
            case 'object': return Type.OBJECT;
            case 'array': return Type.ARRAY;
            default: return type as Type; // For unsupported types, return as is
        }
    }

    // For array of types, take the first valid one as the primary type
    // The full set of types will be handled with anyOf
    for (const t of type) {
        const converted = convertType(t);
        if (converted) return converted;
    }

    return undefined;
}

/**
 * Deep clone and convert the schema from JSONSchema to Gemini Schema
 * @throws {Error} If circular references are detected (max depth exceeded)
 */
function convertSchema(jsSchema?: JSONSchema, depth: number = 0, requiredAll = false): Schema {
    // Prevent circular references
    if (depth > 20) {
        throw new Error("Maximum schema depth (20) exceeded. Possible circular reference detected.");
    }

    if (!jsSchema) return {};

    // Create new schema object rather than mutating
    const result: Schema = {};

    // Handle types
    result.type = convertSchemaType(jsSchema);

    // Handle description
    if (jsSchema.description) {
        result.description = jsSchema.description;
    }

    // Handle properties and required fields
    if (jsSchema.properties) {
        const propertyResult = convertSchemaProperties(jsSchema, depth + 1, requiredAll);
        Object.assign(result, propertyResult);
    }

    // Handle items for arrays
    if (jsSchema.items) {
        result.items = convertSchema(jsSchema.items, depth + 1);
    }

    // Handle enum values
    if (jsSchema.enum) {
        result.enum = [...jsSchema.enum]; // Create a copy instead of reference
    }

    // Copy constraints
    Object.assign(result, extractConstraints(jsSchema));

    return result;
}

/**
 * Convert schema type information, handling anyOf for multiple types
 */
function convertSchemaType(jsSchema: JSONSchema): Type | undefined {
    // Handle multiple types using anyOf
    if (jsSchema.type && Array.isArray(jsSchema.type) && jsSchema.type.length > 1) {
        // Since anyOf is an advanced type, we'll return the first valid type
        // and handle the multi-type case separately in the schema
        return convertType(jsSchema.type[0]);
    }
    // Handle single type
    else if (jsSchema.type) {
        return convertType(jsSchema.type);
    }

    return undefined;
}

/**
 * Handle properties conversion and required fields 
 */
function convertSchemaProperties(jsSchema: JSONSchema, depth: number, requiredAll: boolean): Partial<Schema> {
    const result: Partial<Schema> = { properties: {} };
    if (jsSchema.required) {
        result.required = [...jsSchema.required]; // Create a copy
    }

    // Extract property ordering from the object keys
    const propertyNames = Object.keys(jsSchema.properties || {});

    // Set property ordering based on the existing order in the schema
    if (propertyNames.length > 0) {
        result.propertyOrdering = propertyNames;

        if (requiredAll) {
            // Mark all properties as required by default
            // This ensures the model fills all fields
            result.required = propertyNames;

            // Get the original required properties
            const originalRequired = jsSchema.required || [];

            // Make previously optional properties nullable since we're marking them as required
            for (const key of propertyNames) {
                const propSchema = jsSchema.properties?.[key];
                if (propSchema && !originalRequired.includes(key)) {
                    // Initialize the property if needed
                    if (!result.properties![key]) {
                        result.properties![key] = {};
                    }

                    // Mark as nullable
                    result.properties![key].nullable = true;
                }
            }
        }
    }

    // Convert each property schema
    for (const [key, value] of Object.entries(jsSchema.properties || {})) {
        if (!result.properties![key]) {
            result.properties![key] = {};
        }

        // Merge with converted schema
        result.properties![key] = {
            ...result.properties![key],
            ...convertSchema(value, depth)
        };
    }

    // Override with explicit propertyOrdering if present
    if (jsSchema.propertyOrdering) {
        result.propertyOrdering = [...jsSchema.propertyOrdering]; // Create a copy
    }

    return result;
}

/**
 * Extract schema constraints (min/max values, formats, etc.)
 */
function extractConstraints(jsSchema: JSONSchema): Partial<Schema> {
    const constraints: Partial<Schema> = {};

    if (jsSchema.minimum !== undefined) constraints.minimum = jsSchema.minimum;
    if (jsSchema.maximum !== undefined) constraints.maximum = jsSchema.maximum;
    if (jsSchema.minLength !== undefined) constraints.minLength = jsSchema.minLength;
    if (jsSchema.maxLength !== undefined) constraints.maxLength = jsSchema.maxLength;
    if (jsSchema.minItems !== undefined) constraints.minItems = jsSchema.minItems;
    if (jsSchema.maxItems !== undefined) constraints.maxItems = jsSchema.maxItems;
    if (jsSchema.nullable !== undefined) constraints.nullable = jsSchema.nullable;
    if (jsSchema.pattern) constraints.pattern = jsSchema.pattern;
    if (jsSchema.format) constraints.format = jsSchema.format;
    if (jsSchema.default !== undefined) constraints.default = jsSchema.default;
    if (jsSchema.example !== undefined) constraints.example = jsSchema.example;

    return constraints;
}

/**
 * Check if a value is empty (null, undefined, empty string, empty array, empty object)
 * @param value The value to check
 * @returns True if the value is considered empty
 */
function isEmpty(value: any): boolean {
    if (value === null || value === undefined) {
        return true;
    }

    if (typeof value === 'string' && value.trim() === '') {
        return true;
    }

    if (Array.isArray(value) && value.length === 0) {
        return true;
    }

    // Check for empty object (no own enumerable properties)
    if (typeof value === 'object' && Object.keys(value).length === 0) {
        return true;
    }

    // Check for array of empty objects
    if (Array.isArray(value) && value.every(item => isEmpty(item))) {
        return true;
    }

    return false;
}

// No array cleaning function needed as we're only working with JSONObjects

/**
 * Clean up the JSON result by removing empty values for optional fields
 * Uses immutable patterns to create a new Content object rather than modifying the original
 * @param content The original content from Gemini
 * @param result_schema The JSON schema to use for cleaning
 * @returns A new Content object with cleaned JSON text
 */
function cleanEmptyFieldsContent(content: Content, result_schema?: JSONSchema): Content {
    // If no schema provided, return original content
    if (!result_schema) {
        return content;
    }

    // Create a new content object (shallow copy)
    const cleanedContent: Content = { ...content };

    // Create a new parts array if it exists
    if (cleanedContent.parts) {
        cleanedContent.parts = cleanedContent.parts.map(part => {
            // Only process parts with text
            if (!part.text) {
                return part; // Return unchanged if no text
            }

            // Create a new part object
            const newPart = { ...part };

            try {
                // Parse JSON, clean it based on schema, then stringify
                const jsonText = JSON.parse(part.text);
                // Skip cleaning if not an object
                if (typeof jsonText === 'object' && jsonText !== null && !Array.isArray(jsonText)) {
                    const cleanedJson = removeEmptyFields(jsonText, result_schema);
                    newPart.text = JSON.stringify(cleanedJson);
                } else {
                    // Keep original if not an object (string, number, array, etc.)
                    newPart.text = part.text;
                }
            } catch (e) {
                // On error, keep the original text
                console.warn("Error parsing Gemini output to JSON in part:", e);
            }

            return newPart;
        });
    }

    return cleanedContent;
}

/**
 * Removes empty optional fields from the JSON result based on the provided schema
 * @param object The object to clean
 * @param schema The JSON schema to use for cleaning
 * @returns A new object with empty optional fields removed
 */
function removeEmptyFields(object: JSONObject | any[], schema: JSONSchema): JSONObject | any[] {
    if (!object) {
        return object
    }

    if (Array.isArray(object)) {
        return removeEmptyJSONArray(object, schema);
    }
    if (typeof object == 'object' || object === null) {
        return removeEmptyJSONObject(object, schema);
    }

    return object;
}

function removeEmptyJSONObject(object: JSONObject, schema: JSONSchema): JSONObject {
    // Get the original required properties from schema
    const requiredProps = schema.required || [];
    const cleanedResult: JSONObject = { ...object };

    // Process each property
    for (const [key, value] of Object.entries(object)) {
        const isRequired = requiredProps.includes(key);
        const propSchema = schema.properties?.[key];

        // Recursively clean nested objects based on their schema
        cleanedResult[key] = removeEmptyFields(value as JSONObject, propSchema ?? {});

        if (isEmpty(value)) {
            if (isRequired) {
                continue; // Keep required fields even if empty
            } else {
                delete cleanedResult[key]; // Remove empty optional fields
            }
        }
    }

    return cleanedResult;
}

function removeEmptyJSONArray(array: any[], schema: JSONSchema): any[] {
    const cleanedArray = array.map(item => {
        return removeEmptyFields(item, schema);
    });

    // Filter out empty objects from the array
    return cleanedArray.filter(item => !isEmpty(item));
}

function collectTextParts(content: Content): CompletionResult[] {
    const results: CompletionResult[] = [];
    const parts = content.parts;
    if (parts) {
        for (const part of parts) {
            if (part.text) {
                results.push({
                    type: "text",
                    value: part.text
                });
            }
        }
    }
    return results;
}

function collectInlineDataParts(content: Content): CompletionResult[] {
    const results: CompletionResult[] = [];
    const parts = content.parts;
    if (parts) {
        for (const part of parts) {
            if (part.inlineData) {
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

function geminiMaxTokens(option: StatelessExecutionOptions) {
    const model_options = option.model_options as VertexAIGeminiOptions | undefined;
    if (model_options?.max_tokens) {
        return model_options.max_tokens;
    }
    if (option.model.includes("gemini-2.5")) {
        const maxSupportedTokens = getMaxTokensLimitVertexAi(option.model);
        const thinkingBudget = geminiThinkingBudget(option) ?? 0;
        return Math.min(maxSupportedTokens, 16000 + thinkingBudget);
    }
    return undefined;
}

function geminiThinkingBudget(option: StatelessExecutionOptions) {
    const model_options = option.model_options as VertexAIGeminiOptions | undefined;
    if (model_options?.thinking_budget_tokens) {
        return model_options.thinking_budget_tokens;
    }
    // Set minimum thinking level by default.
    // Docs: https://ai.google.dev/gemini-api/docs/thinking#set-budget
    if (option.model.includes("gemini-2.5")) {
        if (option.model.includes("pro")) {
            return 128;
        }
        return 0;
    }
    return undefined;
}

function geminiThinkingConfig(option: StatelessExecutionOptions): ThinkingConfig | undefined {
    const model_options = option.model_options as VertexAIGeminiOptions | undefined;
    const include_thoughts = model_options?.include_thoughts ?? false;
    if (model_options?.thinking_budget_tokens) {
        return { includeThoughts: include_thoughts, thinkingBudget: model_options.thinking_budget_tokens };
    }

    // Set minimum thinking level by default.
    // Docs: https://ai.google.dev/gemini-api/docs/thinking#set-budget
    if (option.model.includes("gemini-2.5") || option.model.includes("gemini-3")) {
        const thinking_budget_tokens = geminiThinkingBudget(option) ?? 0;
        return { includeThoughts: include_thoughts, thinkingBudget: thinking_budget_tokens };
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

    preValidationProcessing(result: Completion, options: ExecutionOptions): { result: Completion, options: ExecutionOptions } {
        // Guard clause, if no result_schema, error, or tool use, skip processing
        if (!options.result_schema || !result.result || result.tool_use || result.error) {
            return { result, options };
        }
        try {
            // Extract text content for JSON processing - only process first text result
            const textResult = result.result.find(r => r.type === 'text')?.value;
            if (textResult) {
                const jsonResult = JSON.parse(textResult);
                const cleanedJson = JSON.stringify(removeEmptyFields(jsonResult, options.result_schema));
                // Replace the text result with cleaned version
                result.result = result.result.map(r =>
                    r.type === 'text' ? { ...r, value: cleanedJson } : r
                );
            }
            return { result, options };
        } catch (error) {
            // Log error during processing but don't fail the completion
            console.warn('Error during Gemini JSON pre-validation: ', error);
            // Return original result if cleanup fails
            return { result, options };
        }
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
                        let fileUrl = await f.getURL();
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
        const tokenUsage: ExecutionTokenUsage = { total: usageMetadata.totalTokenCount, prompt: usageMetadata.promptTokenCount };

        //Output/Response side
        tokenUsage.result = (usageMetadata.candidatesTokenCount ?? 0)
            + (usageMetadata.thoughtsTokenCount ?? 0)
            + (usageMetadata.toolUsePromptTokenCount ?? 0);

        if ((tokenUsage.total ?? 0) != (tokenUsage.prompt ?? 0) + tokenUsage.result) {
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

        let conversation = updateConversation(options.conversation, prompt.contents);
        prompt.contents = conversation;

        // TODO: Remove hack, use global endpoint manually if needed.
        if (options.model.includes("gemini-2.5-flash-image")) {
            region = "global"; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const client = driver.getGoogleGenAIClient(region);

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

                // We clean the content before validation, so we can update the conversation.
                const cleanedContent = cleanEmptyFieldsContent(content, options.result_schema);
                const textResults = collectTextParts(cleanedContent);
                const imageResults = collectInlineDataParts(cleanedContent);
                result = [...textResults, ...imageResults];
                conversation = updateConversation(conversation, [cleanedContent]);
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

        return {
            result: result && result.length > 0 ? result : [{ type: "text" as const, value: '' }],
            token_usage: token_usage,
            finish_reason: finish_reason,
            original_response: options.include_original_response ? response : undefined,
            conversation: processedConversation,
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

        // Include conversation history in prompt contents (same as non-streaming)
        const conversation = updateConversation(options.conversation, prompt.contents);
        prompt.contents = conversation;

        if (options.model.includes("gemini-2.5-flash-image")) {
            region = "global"; // Gemini Flash Image only available in global region, this is for nano-banana model
        }

        const client = driver.getGoogleGenAIClient(region);

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
                        const textResults = collectTextParts(candidate.content);
                        const imageResults = collectInlineDataParts(candidate.content);
                        const combinedResults = [...textResults, ...imageResults];
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
        let message = apiError.message || String(error);

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
    // If input_schema is a string, parse it; if it's already an object, use it directly
    let toolSchema: Schema | undefined;

    // Using a try-catch for safety, as the input_schema might not be a valid JSONSchema
    try {
        toolSchema = parseJSONtoSchema(tool.input_schema as JSONSchema, false);
    }
    catch (e) {
        toolSchema = { ...tool.input_schema, type: Type.OBJECT } as unknown as Schema;
    }

    return {
        name: tool.name,
        description: tool.description,
        parameters: toolSchema,
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