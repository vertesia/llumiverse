import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType, PromptOptions, PromptRole, PromptSegment, TextFallbackOptions } from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";
import * as AnthropicAPI from '@anthropic-ai/sdk';
type MessageParam = AnthropicAPI.Anthropic.MessageParam;
import { TextBlockParam } from "@anthropic-ai/sdk/resources/index.js";

interface ClaudePrompt {
    messages: MessageParam[];
    system: TextBlockParam[];
}

function getFullModelName(model: string) : string {
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

function maxToken(max_tokens: number | undefined, model: string) : number {
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

        const messages: MessageParam[] = segments
        .filter(segment => segment.role == PromptRole.user || segment.role == PromptRole.assistant)
        .map(segment => ({
            role: segment.role !== PromptRole.user ? 'assistant' : 'user',
            content: segment.content
        }));

        const system = systemSegments.concat(safetySegments);

        return {
            messages: messages,
            system: system,
        }
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<Completion> {
        const client = driver.getAnthropicClient();   
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        if (options.model_options?._option_id !== "text-fallback") {
            driver.logger.warn("Invalid model options", options.model_options);
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const result = await client.messages.create({
            ...prompt,  // messages, system
            temperature: options.model_options?.temperature,
            model: modelName,
            max_tokens: maxToken(options.model_options?.max_tokens, modelName),
            top_p: options.model_options?.top_p,
            top_k: options.model_options?.top_k,
            stop_sequences: options.model_options?.stop_sequence,
        });
        
        const text = collectTextParts(result.content);

        return {
            result: text ?? '',
            token_usage: {
                prompt: result?.usage.input_tokens,
                result: result?.usage.output_tokens,
                total: result?.usage.input_tokens + result?.usage.output_tokens
            },
            finish_reason: claudeFinishReason(result?.stop_reason ?? ''),
        } as Completion;
    }
    
    async requestTextCompletionStream(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise < AsyncIterable < CompletionChunkObject >> {
        const client = driver.getAnthropicClient();
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        if (options.model_options?._option_id !== "text-fallback") {
            driver.logger.warn("Invalid model options", options.model_options);
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const response_stream = await client.messages.stream({
            ...prompt,  // messages, system
            temperature: options.model_options?.temperature,
            model: modelName,
            max_tokens: maxToken(options.model_options?.max_tokens, modelName),
            top_p: options.model_options?.top_p,
            top_k: options.model_options?.top_k,
            stop_sequences: options.model_options?.stop_sequence,
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