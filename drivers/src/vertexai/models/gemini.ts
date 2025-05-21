import {
    Content, FinishReason, FunctionDeclaration, GenerateContentParameters,
    HarmBlockThreshold, HarmCategory, Part, SafetySetting, Schema, Tool, Type
} from "@google/genai";
import {
    AIModel, Completion, CompletionChunkObject, ExecutionOptions,
    ExecutionTokenUsage, JSONObject, JSONSchema, ModelType, PromptOptions, PromptRole,
    PromptSegment, readStreamAsBase64, ToolDefinition, ToolUse
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIDriver, GenerateContentPrompt } from "../index.js";
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
    //1.0 Ultra does not support JSON output, 1.0 Pro does.
    const useStructuredOutput = supportsStructuredOutput(options);

    const model_options = options.model_options as any;
    const tools = getToolDefinitions(options.tools);
    prompt.tools = tools ? [tools] : undefined;

    const parsedSchema = parseJSONtoSchema(options.result_schema);
    console.log("Parsed Schema", JSON.stringify(parsedSchema, null, 2));
    return {
        model: options.model,
        contents: prompt.contents,
        config: {
            systemInstruction: prompt.system,
            safetySettings: geminiSafetySettings,
            tools: tools ? [tools] : undefined,
            candidateCount: 1,
            //JSON/Structured output
            responseMimeType: useStructuredOutput ? "application/json" : "text/plain",
            responseSchema: useStructuredOutput ? parseJSONtoSchema(options.result_schema) : undefined,
            //Model options
            temperature: model_options?.temperature,
            topP: model_options?.top_p,
            topK: model_options?.top_k,
            maxOutputTokens: model_options?.max_tokens,
            stopSequences: model_options?.stop_sequence,
            presencePenalty: model_options?.presence_penalty,
            frequencyPenalty: model_options?.frequency_penalty,
            seed: model_options?.seed,
            thinkingConfig: model_options?.include_thoughts || model_options?.thinking_budget ?
                {
                    includeThoughts: model_options?.include_thoughts,
                    thinkingBudget: model_options?.thinking_budget,
                } : undefined,
        }
    };
}

/**
 * Convert JSONSchema to Gemini Schema,
 * Make all properties required by default
 * Properties previously marked as optional will be marked as nullable.
 */
function parseJSONtoSchema(schema?: JSONSchema): Schema {
    if (!schema) {
        return {};
    }

    return convertSchema(schema);
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
            default: return undefined;
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
function convertSchema(jsSchema?: JSONSchema, depth: number = 0): Schema {
    // Prevent circular references
    if (depth > 50) {
        throw new Error("Maximum schema depth exceeded. Possible circular reference detected.");
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
        const propertyResult = convertSchemaProperties(jsSchema, depth + 1);
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
function convertSchemaProperties(jsSchema: JSONSchema, depth: number): Partial<Schema> {
    const result: Partial<Schema> = { properties: {} };
    if (jsSchema.required) {
        result.required = [...jsSchema.required]; // Create a copy
    }

    // Extract property ordering from the object keys
    const propertyNames = Object.keys(jsSchema.properties || {});

    // Set property ordering based on the existing order in the schema
    if (propertyNames.length > 0) {
        result.propertyOrdering = propertyNames;

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
                const cleanedJson = cleanResultBasedOnSchema(jsonText, result_schema);
                newPart.text = JSON.stringify(cleanedJson);
            } catch (e) {
                // On error, keep the original text
            }

            return newPart;
        });
    }

    return cleanedContent;
}

/**
 * Cleans the result based on the original schema to respect optionality
 * Removes empty values from fields that weren't explicitly required in the original schema
 * @param obj The object to clean
 * @param schema The JSON schema to use for cleaning
 * @returns A new object with empty optional fields removed
 */
function cleanResultBasedOnSchema(obj: unknown, schema: JSONSchema): unknown {
    // Handle non-object results
    if (obj === null || obj === undefined || typeof obj !== 'object') {
        return obj;
    }

    // Handle arrays
    if (Array.isArray(obj)) {
        return cleanArrayResult(obj, schema);
    }

    // For objects, check each property against the schema
    if (schema.properties && typeof obj === 'object' && !Array.isArray(obj)) {
        return cleanObjectResult(obj as Record<string, unknown>, schema);
    }

    return obj;
}

/**
 * Clean an array result based on schema
 */
function cleanArrayResult(array: unknown[], schema: JSONSchema): unknown[] {
    // If array is empty and field is optional, it should be removed entirely by parent
    if (array.length === 0) {
        return array;
    }

    // Process each item in the array with its item schema
    const processedItems = array
        .map(item => {
            if (schema.items) {
                return cleanResultBasedOnSchema(item, schema.items);
            }
            return item;
        })
        .filter(item => item !== null && item !== undefined);

    // Return the processed array
    return processedItems;
}

/**
 * Clean an object result based on schema
 */
function cleanObjectResult(obj: Record<string, unknown>, schema: JSONSchema): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    const requiredProps = schema.required || [];
    const schemaProperties = schema.properties || {};

    for (const [key, value] of Object.entries(obj)) {
        const propSchema = schemaProperties[key];

        // If the field has a schema definition
        if (propSchema) {
            // First recursively clean the nested value
            const cleanedValue = cleanResultBasedOnSchema(value, propSchema);

            // For required fields, always keep the value
            if (requiredProps.includes(key)) {
                result[key] = cleanedValue;
            }
            // For optional fields, only keep non-empty values
            else if (!isEffectivelyEmpty(cleanedValue, propSchema)) {
                result[key] = cleanedValue;
            }
        }
        // No schema for this property, keep it if non-empty
        else if (!isEffectivelyEmpty(value, null)) {
            result[key] = value;
        }
    }

    return result;
}

/**
 * Checks if a value should be considered effectively empty
 * Takes into account the expected schema type to make better decisions
 */
function isEffectivelyEmpty(value: any, schema: JSONSchema | null): boolean {
    // Basic empty values
    if (value === null || value === undefined) {
        return true;
    }

    // Empty strings
    if (typeof value === 'string' && value.trim() === '') {
        return true;
    }

    // Empty arrays - consider empty arrays as effectively empty
    if (Array.isArray(value) && value.length === 0) {
        return true;
    }

    // Empty objects
    if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0) {
        return true;
    }

    // Check for objects with only empty values
    if (schema && schema.type === 'object' && typeof value === 'object' && !Array.isArray(value)) {
        // If every property is effectively empty, consider the whole object empty
        const hasNonEmptyProperty = Object.entries(value).some(([key, propValue]) => {
            const propSchema = schema.properties?.[key];
            return !isEffectivelyEmpty(propValue, propSchema || null);
        });
        return !hasNonEmptyProperty;
    }

    return false;
}

function collectTextParts(content: Content) {
    const out = [];
    const parts = content.parts;
    if (parts) {
        for (const part of parts) {
            if (part.text) {
                out.push(part.text);
            }
        }
    }
    return out.join('\n');
}

function collectToolUseParts(content: Content): ToolUse[] | undefined {
    const out: ToolUse[] = [];
    const parts = content.parts ?? [];
    for (const part of parts) {
        if (part.functionCall) {
            out.push({
                id: part.functionCall.name ?? '',
                tool_name: part.functionCall.name ?? '',
                tool_input: part.functionCall.args as JSONObject,
            });
        }
    }
    return out.length > 0 ? out : undefined;
}

export class GeminiModelDefinition implements ModelDefinition<GenerateContentPrompt> {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } satisfies AIModel;
    }

    preValidationProcessing(result: Completion, options: ExecutionOptions): { result: Completion, options: ExecutionOptions } {
        try {
            // If there's no schema or result is not an object, no processing needed
            if (!options.result_schema || !result.result || typeof result.result !== 'object') {
                return { result, options };
            }

            // Create a new result object with a cleaned result property (immutable pattern)
            const newResult: Completion = {
                ...result,
                result: cleanResultBasedOnSchema(result.result, options.result_schema)
            };

            return { result: newResult, options };
        } catch (error) {
            // Log error during processing but don't fail the completion
            if (process.env.NODE_ENV === 'development') {
                console.warn('Error during preValidationProcessing:', error instanceof Error ? error.message : error);
            }
            // Return original result if cleanup fails
            return { result, options };
        }
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<GenerateContentPrompt> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const schema = options.result_schema;
        const contents: Content[] = [];
        const safety: string[] = [];
        const system: string[] = [];

        let lastUserContent: Content | undefined = undefined;
        const toolParts = [];

        for (const msg of segments) {

            if (msg.role === PromptRole.safety) {
                safety.push(msg.content);
            } else if (msg.role === PromptRole.system) {
                system.push(msg.content);
            } else {
                const fileParts: Part[] = [];
                if (msg.files) {
                    for (const f of msg.files) {
                        const stream = await f.getStream();
                        const data = await readStreamAsBase64(stream);
                        fileParts.push({
                            inlineData: {
                                data,
                                mimeType: f.mime_type!
                            }
                        });
                    }
                }

                if (msg.role === PromptRole.tool) {
                    toolParts.push({
                        functionResponse: {
                            name: msg.tool_use_id!,
                            response: formatFunctionResponse(msg.content || ''),
                        }
                    });
                    continue;
                }

                const role = msg.role === PromptRole.assistant ? "model" : "user";

                if (lastUserContent && lastUserContent.role === role) {
                    lastUserContent?.parts?.push({ text: msg.content });
                    fileParts?.forEach(p => lastUserContent?.parts?.push(p));
                } else {
                    const content: Content = {
                        role,
                        parts: [{ text: msg.content }],
                    }
                    fileParts?.forEach(p => content?.parts?.push(p));

                    if (role === 'user') {
                        lastUserContent = content;
                    }
                    contents.push(content);
                }
            }
        }

        if (schema && !supportsStructuredOutput(options)) {
            // Fallback to putting the schema in the prompt, if not using structured output.
            safety.push("The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema));
        } else if (schema) {
            // Gemini structured output is unnecessarily sparse.
            safety.push("The answer must be a JSON, fill fields using relevant data.");
        }

        if (safety.length > 0) {
            const content = safety.join('\n');
            if (lastUserContent) {
                lastUserContent?.parts?.push({ text: content });
            } else {
                contents.push({
                    role: 'user',
                    parts: [{ text: content }],
                })
            }
        }

        if (toolParts.length > 0) {
            contents.push({
                role: 'user',
                parts: toolParts,
            });
        }

        // put system messages first and safety last
        return { contents, system: system.join('\n'), tools: undefined };
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: GenerateContentPrompt, options: ExecutionOptions): Promise<Completion> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        let conversation = updateConversation(options.conversation as Content[], prompt.contents);
        prompt.contents = conversation;

        const client = driver.getGoogleGenAIClient();

        const payload = getGeminiPayload(options, prompt);
        const response = await client.models.generateContent(payload);

        const usage = response.usageMetadata;
        const token_usage: ExecutionTokenUsage = {
            prompt: usage?.promptTokenCount,
            result: usage?.candidatesTokenCount,
            total: usage?.totalTokenCount,
        }

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
            if (content) {
                tool_use = collectToolUseParts(content);

                // We clean the content before validation, so we can update the conversation.
                const cleanedContent = cleanEmptyFieldsContent(content, options.result_schema);
                result = collectTextParts(cleanedContent);
                conversation = updateConversation(conversation, [cleanedContent]);
            }
        }

        if (tool_use) {
            finish_reason = "tool_use";
        }

        return {
            result: result ?? '',
            token_usage: token_usage,
            finish_reason: finish_reason,
            original_response: options.include_original_response ? response : undefined,
            conversation,
            tool_use
        } satisfies Completion;
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: GenerateContentPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const client = driver.getGoogleGenAIClient();

        const payload = getGeminiPayload(options, prompt);
        const response = await client.models.generateContentStream(payload);

        const stream = asyncMap(response, async (item) => {
            const usage = item.usageMetadata;
            const token_usage: ExecutionTokenUsage = {
                prompt: usage?.promptTokenCount,
                result: usage?.candidatesTokenCount,
                total: usage?.totalTokenCount,
            }
            if (item.candidates && item.candidates.length > 0) {
                for (const candidate of item.candidates) {
                    let tool_use: ToolUse[] | undefined;
                    let finish_reason: string | undefined;
                    switch (candidate.finishReason) {
                        case FinishReason.MAX_TOKENS: finish_reason = "length"; break;
                        case FinishReason.STOP: finish_reason = "stop"; break;
                        default: finish_reason = candidate.finishReason;
                    }
                    if (candidate.content?.role === 'model') {
                        const text = collectTextParts(candidate.content);
                        tool_use = collectToolUseParts(candidate.content);
                        if (tool_use) {
                            finish_reason = "tool_use";
                        }
                        return {
                            result: text,
                            token_usage: token_usage,
                            finish_reason: finish_reason,
                            tool_use,
                        };
                    }
                }
            }
            //No normal output, returning block reason if it exists.
            return {
                result: item.promptFeedback?.blockReasonMessage ?? "",
                finish_reason: item.promptFeedback?.blockReason ?? "",
            };
        });

        return stream;
    }

}


function getToolDefinitions(tools: ToolDefinition[] | undefined | null): Tool | undefined {
    if (!tools || tools.length === 0) {
        return undefined;
    }
    // VertexAI Gemini only supports one tool at a time.
    // For multiple tools, we have multiple functions in one tool.
    const tool_array = tools.map(getToolDefinition);
    let mergedTool = tool_array[0];
    for (let i = 1; i < tool_array.length; i++) {
        mergedTool.functionDeclarations = mergedTool.functionDeclarations?.concat(tool_array[i].functionDeclarations ?? []);
    }
    return mergedTool;
}

function getToolDefinition(tool: ToolDefinition) {
    return {
        functionDeclarations: [
            {
                name: tool.name,
                description: tool.description,
                parameters: {
                    ...tool.input_schema,
                    type: Type.OBJECT,
                    properties: (tool.input_schema && typeof tool.input_schema.properties === 'object'
                        ? tool.input_schema.properties
                        : undefined) as Record<string, any> | undefined,
                },
            } satisfies FunctionDeclaration
        ]
    };
}


/**
 * Update the conversation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: Content[], prompt: Content[]): Content[] {
    return (conversation || [] satisfies Content[]).concat(prompt);
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