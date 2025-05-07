import { Content, FinishReason, FunctionDeclarationSchema, FunctionDeclarationsTool, FunctionResponsePart, GenerateContentRequest, HarmBlockThreshold, HarmCategory, InlineDataPart, ModelParams, TextPart, Tool } from "@google-cloud/vertexai";
import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ExecutionTokenUsage, JSONObject, ModelType, PromptOptions, PromptRole, PromptSegment, readStreamAsBase64, TextFallbackOptions, ToolDefinition, ToolUse } from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";

function getGenerativeModel(driver: VertexAIDriver, options: ExecutionOptions, modelParams?: ModelParams) {

    //1.0 Ultra does not support JSON output, 1.0 Pro does.
    const jsonMode = options.result_schema && !(options.model.includes("ultra"));

    if (options.model_options?._option_id !== "text-fallback") {
        driver.logger.warn("Invalid model options", { options: options.model_options });
    }
    options.model_options = options.model_options as TextFallbackOptions;

    const model_options = options.model_options;

    const client = driver.getVertexAIClient();
    const model = client.getGenerativeModel({
        model: options.model,
        safetySettings: modelParams?.safetySettings ?? [{
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
        }
        ],
        generationConfig: {
            responseMimeType: jsonMode ? "application/json" : "text/plain",
            candidateCount: modelParams?.generationConfig?.candidateCount ?? 1,
            temperature: model_options?.temperature,
            maxOutputTokens: model_options?.max_tokens,
            topP: model_options?.top_p,
            topK: model_options?.top_k,
            frequencyPenalty: model_options?.frequency_penalty,
            stopSequences: model_options?.stop_sequence,
        },
    });

    return model;
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
    const parts = content.parts;
    for (const part of parts) {
        if (part.functionCall) {
            out.push({
                id: part.functionCall.name,
                tool_name: part.functionCall.name,
                tool_input: part.functionCall.args as JSONObject,
            });
        }
    }
    return out.length > 0 ? out : undefined;
}

export class GeminiModelDefinition implements ModelDefinition<GenerateContentRequest> {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } as AIModel;
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<GenerateContentRequest> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const schema = options.result_schema;
        const contents: Content[] = [];
        const safety: string[] = [];

        let lastUserContent: Content | undefined = undefined;
        const toolParts: FunctionResponsePart[] = [];

        for (const msg of segments) {

            if (msg.role === PromptRole.safety) {
                safety.push(msg.content);
            } else {
                const fileParts: InlineDataPart[] = [];
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
                    lastUserContent.parts.push({ text: msg.content } as TextPart);
                    fileParts?.forEach(p => lastUserContent?.parts.push(p));
                } else {
                    const content: Content = {
                        role,
                        parts: [{ text: msg.content } as TextPart],
                    }
                    fileParts?.forEach(p => content.parts.push(p));

                    if (role === 'user') {
                        lastUserContent = content;
                    }
                    contents.push(content);
                }
            }
        }

        let tools: any = undefined;
        if (schema) {
            safety.push("The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema));
        }

        if (safety.length > 0) {
            const content = safety.join('\n');
            if (lastUserContent) {
                lastUserContent.parts.push({ text: content } as TextPart);
            } else {
                contents.push({
                    role: 'user',
                    parts: [{ text: content } as TextPart],
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
        return { contents, tools } as GenerateContentRequest;
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: GenerateContentRequest, options: ExecutionOptions): Promise<Completion> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        let conversation = updateConversation(options.conversation as Content[], prompt.contents);

        prompt.contents = conversation;
        const tools = getToolDefinitions(options.tools);
        prompt.tools = tools ? [tools] : undefined;

        const model = getGenerativeModel(driver, options);
        const r = await model.generateContent(prompt);
        const response = await r.response;
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
                result = collectTextParts(content);
                conversation = updateConversation(conversation, [content]);
                // if (options.result_schema) {
                //     result = candidate.;
                // } else {
                // }
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
        } as Completion;
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: GenerateContentRequest, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const tools = getToolDefinitions(options.tools);
        prompt.tools = tools ? [tools] : undefined;

        const model = getGenerativeModel(driver, options);
        const streamingResp = await model.generateContentStream(prompt);

        const stream = asyncMap(streamingResp.stream, async (item) => {
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
    let mergedTool: FunctionDeclarationsTool = tool_array[0];
    for (let i = 1; i < tool_array.length; i++) {
        mergedTool.functionDeclarations = mergedTool.functionDeclarations?.concat(tool_array[i].functionDeclarations ?? []);
    }
    return mergedTool;
}
function getToolDefinition(tool: ToolDefinition): FunctionDeclarationsTool {
    return {
        functionDeclarations: [
            {
                name: tool.name,
                description: tool.description,
                parameters: { ...tool.input_schema, type: "OBJECT" } as FunctionDeclarationSchema,
            }
        ]
    } satisfies FunctionDeclarationsTool;
}


/**
 * Update the conversation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: Content[], prompt: Content[]): Content[] {
    return (conversation || [] as Content[]).concat(prompt);
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