import * as AnthropicAPI from '@anthropic-ai/sdk';
import { ContentBlock, Message, TextBlockParam } from "@anthropic-ai/sdk/resources/index.js";
import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, JSONObject, ModelType, PromptOptions, PromptRole, PromptSegment, ToolUse } from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIClaudeOptions } from "../../../../core/src/options/vertexai.js";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";

type MessageParam = AnthropicAPI.Anthropic.MessageParam;

interface ClaudePrompt {
    messages: MessageParam[];
    system: TextBlockParam[];
}

function getFullModelName(model: string): string {
    if (model.includes("claude-3-5-sonnet-v2")) {
        return "claude-3-5-sonnet-v2@20241022"
    } else if (model.includes("claude-3-5-sonnet")) {
        return "claude-3-5-sonnet@20240620"
    } else if (model.includes("claude-3-5-haiku")) {
        return "claude-3-5-haiku@20241022"
    } else if (model.includes("claude-3-opus")) {
        return "claude-3-opus@20240229"
    } else if (model.includes("claude-3-sonnet")) {
        return "claude-3-sonnet@20240229"
    } else if (model.includes("claude-3-haike")) {
        return "claude-3-haiku@20240307"
    }
    return model;
}

function claudeFinishReason(reason: string | undefined) {
    if (!reason) return undefined;
    switch (reason) {
        case 'end_turn': return "stop";
        case 'max_tokens': return "length";
        default: return reason; //stop_sequence
    }
}

function collectTextParts(content: any) {
    const out = [];

    for (const block of content) {
        if (block?.text) {
            out.push(block.text);
        }
    }
    return out.join('\n');
}

function maxToken(max_tokens: number | undefined, model: string): number {
    const contains = (str: string, substr: string) => str.indexOf(substr) !== -1;
    if (max_tokens) {
        return max_tokens;
    } else if (contains(model, "claude-3-5")) {
        return 8192;
    } else {
        return 4096
    }
}

export class ClaudeModelDefinition implements ModelDefinition<ClaudePrompt> {

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

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<ClaudePrompt> {
        // Convert the prompt to the format expected by the Claude API
        const systemSegments: TextBlockParam[] = segments
            .filter(segment => segment.role === PromptRole.system)
            .map(segment => ({
                text: segment.content,
                type: 'text'
            }));

        const safetySegments: TextBlockParam[] = segments
            .filter(segment => segment.role === PromptRole.safety)
            .map(segment => ({
                text: segment.content,
                type: 'text'
            }));

        if (options.result_schema) {
            const schemaSegments: TextBlockParam = {
                text: "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema),
                type: 'text'
            }
            safetySegments.push(schemaSegments);
        }

        const messages: MessageParam[] = segments.filter(segment =>
            segment.role == PromptRole.user
            || segment.role == PromptRole.assistant
            || segment.role === PromptRole.tool)
            .map(segment => {
                if (segment.role === PromptRole.tool) {
                    if (!segment.tool_use_id) {
                        throw new Error("Tool prompt segment must have a tool_use_id");
                    }
                    return {
                        role: 'user',
                        content: [
                            {
                                type: 'tool_result',
                                tool_use_id: segment.tool_use_id,
                                content: segment.content || undefined
                            }
                        ]
                    }
                } else {
                    return {
                        role: segment.role !== PromptRole.user ? 'assistant' : 'user',
                        content: segment.content
                    }
                }
            });

        const system = systemSegments.concat(safetySegments);

        return {
            messages: messages,
            system: system
        }
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<Completion> {
        const client = driver.getAnthropicClient();
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };
        options.model_options = options.model_options as VertexAIClaudeOptions;

        if (options.model_options?._option_id !== "vertexai-claude") {
            driver.logger.warn("Invalid model options", { options: options.model_options });
        }

        let conversation = updateConversation(options.conversation as ClaudePrompt, prompt);

        const result = await client.messages.create({
            ...conversation, // messages, system,
            tools: options.tools, // we are using the same shape as claude for tools
            temperature: options.model_options?.temperature,
            model: modelName,
            max_tokens: maxToken(options.model_options?.max_tokens, modelName),
            top_p: options.model_options?.top_p,
            top_k: options.model_options?.top_k,
            stop_sequences: options.model_options?.stop_sequence,
            thinking: options.model_options?.thinking_mode ?
                {
                    budget_tokens: options.model_options?.thinking_budget_tokens ?? 1024,
                    type: "enabled"
                } : {
                    type: "disabled"
                }
        }) as Message;

        const text = collectTextParts(result.content);
        const tool_use = collectTools(result.content);

        conversation = updateConversation(options.conversation as ClaudePrompt, createPromptFromResponse(result));

        return {
            chat: [prompt, { role: result.role, content: result.content }],
            result: text ?? '',
            tool_use,
            token_usage: {
                prompt: result?.usage.input_tokens,
                result: result?.usage.output_tokens,
                total: result?.usage.input_tokens + result?.usage.output_tokens
            },
            // make sure we set finish_reason to the correct value (claude is normally setting this by itself)
            finish_reason: tool_use ? "tool_use" : claudeFinishReason(result?.stop_reason ?? ''),
            conversation
        } as Completion;
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const client = driver.getAnthropicClient();
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };
        options.model_options = options.model_options as VertexAIClaudeOptions;

        if (options.model_options?._option_id !== "vertexai-claude") {
            driver.logger.warn("Invalid model options", { options: options.model_options });
        }

        const response_stream = await client.messages.stream({
            ...prompt, // messages, system,
            tools: options.tools, // we are using the same shape as claude for tools
            temperature: options.model_options?.temperature,
            model: modelName,
            max_tokens: maxToken(options.model_options?.max_tokens, modelName),
            top_p: options.model_options?.top_p,
            top_k: options.model_options?.top_k,
            stop_sequences: options.model_options?.stop_sequence,
            thinking: options.model_options?.thinking_mode ?
                {
                    budget_tokens: options.model_options?.thinking_budget_tokens ?? 1024,
                    type: "enabled"
                } : {
                    type: "disabled"
                }
        });

        //Streaming does not give information on the input tokens,
        //So we use a seperate call to get the input tokens.
        //Non-critical and model name sensitive so we put it in a try catch block
        let count_tokens = { input_tokens: 0 };
        try {
            count_tokens = await client.messages.countTokens({
                ...prompt,  // messages, system
                model: getFullModelName(modelName),
            });
        } catch (e) {
            driver.logger.warn("Failed to get token count for model " + modelName);
        }

        const stream = asyncMap(response_stream, async (item: any) => {
            return {
                result: item?.delta?.text ?? '',
                token_usage: { prompt: count_tokens.input_tokens, result: item?.usage?.output_tokens },
                finish_reason: claudeFinishReason(item?.delta?.stop_reason ?? ''),
            }
        });

        return stream;
    }
}

export function collectTools(content: ContentBlock[]): ToolUse[] | undefined {
    const out: ToolUse[] = [];

    for (const block of content) {
        if (block?.type === "tool_use") {
            out.push({
                id: block.id,
                name: block.name,
                input: block.input as JSONObject,
            });
        }
    }

    return out.length > 0 ? out : undefined;
}

function createPromptFromResponse(response: Message): ClaudePrompt {
    return {
        messages: [{
            role: PromptRole.assistant,
            content: response.content,
        }],
        system: []
    }
}

/**
 * Update the converatation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: ClaudePrompt | undefined | null, prompt: ClaudePrompt): ClaudePrompt {
    const baseSystemMessages = conversation ? conversation.system : [];
    const baseMessages = conversation ? conversation.messages : []
    return {
        messages: baseMessages.concat(prompt.messages || []),
        system: baseSystemMessages.concat(prompt.system || [])
    };
}
