import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ExecutionTokenUsage, JSONObject, ModelType, PromptOptions, PromptRole, PromptSegment, readStreamAsBase64, ToolDefinition, ToolUse } from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";
import { Content, ContentListUnion, ContentUnion, FinishReason, FunctionDeclaration, GenerateContentConfig, Part, Tool } from "@google/genai";
import { VertexAIGeminiOptions } from "../../../../core/src/options/vertexai.js";

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
    if (!parts) {
        return undefined;
    }
    for (const part of parts) {
        if (part.functionCall) {
            out.push({
                id: part.functionCall.name ?? "",
                tool_name: part.functionCall.name ?? "",
                tool_input: part.functionCall.args as JSONObject,
            });
        }
    }
    return out.length > 0 ? out : undefined;
}

export class GeminiModelDefinition implements ModelDefinition<ContentListUnion> {

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

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<ContentListUnion> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const schema = options.result_schema;
        const contents: ContentUnion[] = [];
        const safety: string[] = [];

        let lastUserContent: Content | undefined = undefined;

        for (const msg of segments) {

            if (msg.role === PromptRole.safety) {
                safety.push(msg.content);
            } else {
                let fileParts: Part[] | undefined;
                if (msg.files) {
                    fileParts = [];
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
                    const content: Content = {
                        role: 'user',
                        parts: [{
                            functionResponse: {
                                name: msg.tool_use_id!,
                                response: formatFunctionResponse(msg.content || ''),
                            }
                        }]
                    }
                    contents.push(content);
                    continue;
                }

                const role = msg.role === PromptRole.assistant ? "model" : "user";

                if (lastUserContent && lastUserContent.role === role) {
                    lastUserContent.parts?.push({ text: msg.content } as Part);
                    fileParts?.forEach(p => lastUserContent?.parts?.push(p));
                } else {
                    const content: Content = {
                        role,
                        parts: [{ text: msg.content } as Part],
                    }
                    fileParts?.forEach(p => content?.parts?.push(p));

                    if (role === 'user') {
                        lastUserContent = content;
                    }
                    contents.push(content as Content);
                }
            }
        }

        if (schema) {
            safety.push("The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema));
        }

        if (safety.length > 0) {
            const content = safety.join('\n');
            if (lastUserContent) {
                lastUserContent?.parts?.push({ text: content } as Part);
            } else {
                contents.push({
                    role: 'user',
                    parts: [{ text: content } as Part],
                } as Content);
            }
        }

        return contents as ContentListUnion;
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: ContentListUnion, options: ExecutionOptions): Promise<Completion> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        if (options.model_options?._option_id !== "vertexai-gemini") {
            driver.logger.warn("Invalid model options", { options: options.model_options });
        }
        const model_options = options.model_options as VertexAIGeminiOptions;

        const conversation = updateConversation(options.conversation as Content[], prompt as Content[]);

        prompt = conversation;
        const tools = getToolDefinitions(options.tools);

        const client = driver.getGenAIClient();
        const response = await client.models.generateContent({
            model: options.model,
            contents: prompt satisfies ContentListUnion,
            config: {
                temperature: model_options.temperature,
                maxOutputTokens: model_options.max_tokens,
                topK: model_options.top_k,
                topP: model_options.top_p,
                stopSequences: model_options.stop_sequence,
                presencePenalty: model_options.presence_penalty,
                frequencyPenalty: model_options.frequency_penalty,
                tools: tools,
            } satisfies GenerateContentConfig,
        })

        const usage = response?.usageMetadata;
        const token_usage: ExecutionTokenUsage = {
            prompt: usage?.promptTokenCount,
            result: usage?.candidatesTokenCount,
            total: usage?.totalTokenCount,
        }

        let tool_use: ToolUse[] | undefined;
        let finish_reason: string | undefined, result: any;
        const candidate = response?.candidates && response.candidates[0];
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

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: ContentListUnion, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        if (options.model_options?._option_id !== "vertexai-gemini") {
            driver.logger.warn("Invalid model options", { options: options.model_options });
        }
        const model_options = options.model_options as VertexAIGeminiOptions;

        const conversation = updateConversation(options.conversation as Content[], prompt as Content[]);

        prompt = conversation;
        const tools = getToolDefinitions(options.tools);

        const client = driver.getGenAIClient();
        const streamingResp = await client.models.generateContentStream({
            model: options.model,
            contents: prompt satisfies ContentListUnion,
            config: {
                temperature: model_options.temperature,
                maxOutputTokens: model_options.max_tokens,
                topK: model_options.top_k,
                topP: model_options.top_p,
                stopSequences: model_options.stop_sequence,
                presencePenalty: model_options.presence_penalty,
                frequencyPenalty: model_options.frequency_penalty,
                tools: tools,
            } satisfies GenerateContentConfig
        })

        const stream = asyncMap(streamingResp, async (item) => {
            const usage = item.usageMetadata;
            const token_usage: ExecutionTokenUsage = {
                prompt: usage?.promptTokenCount,
                result: usage?.candidatesTokenCount,
                total: usage?.totalTokenCount,
            }
            if (item.candidates && item.candidates.length > 0) {
                for (const candidate of item.candidates) {
                    let finish_reason: string | undefined;
                    switch (candidate.finishReason) {
                        case FinishReason.MAX_TOKENS: finish_reason = "length"; break;
                        case FinishReason.STOP: finish_reason = "stop"; break;
                        default: finish_reason = candidate.finishReason;
                    }
                    if (candidate.content?.role === 'model') {
                        const text = collectTextParts(candidate.content);
                        return {
                            result: text,
                            token_usage: token_usage,
                            finish_reason: finish_reason,
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


function getToolDefinitions(tools: ToolDefinition[] | undefined | null) {
    return tools ? tools.map(getToolDefinition) : undefined;
}
function getToolDefinition(tool: ToolDefinition): Tool {
    return {
        functionDeclarations: [
            {
                name: tool.name,
                description: tool.description,
                parameters: { ...tool.input_schema, type: "OBJECT" } as FunctionDeclaration,
            }
        ]
    };
}


/**
 * Update the converatation messages
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