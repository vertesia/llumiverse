import { AIModel, AbstractDriver, Completion, CompletionChunkObject, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, PromptSegment, TextFallbackOptions, ToolDefinition, ToolUse } from "@llumiverse/core";
import { transformAsyncIterator } from "@llumiverse/core/async";
import { formatOpenAILikeMultimodalPrompt } from "../openai/openai_format.js";

import Groq from "groq-sdk";
import type OpenAI from "openai";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "groq-sdk/resources/chat/completions";
import type { FunctionParameters } from "groq-sdk/resources/shared";

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type EasyInputMessage = OpenAI.Responses.EasyInputMessage;

interface GroqDriverOptions extends DriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export class GroqDriver extends AbstractDriver<GroqDriverOptions, ChatCompletionMessageParam[]> {
    static PROVIDER = "groq";
    provider = GroqDriver.PROVIDER;
    apiKey: string;
    client: Groq;
    endpointUrl?: string;

    constructor(options: GroqDriverOptions) {
        super(options);
        this.apiKey = options.apiKey;
        this.client = new Groq({
            apiKey: options.apiKey,
            baseURL: options.endpoint_url
        });
    }

    // protected canStream(options: ExecutionOptions): Promise<boolean> {
    //     if (options.result_schema) {
    //         // not yet streaming json responses
    //         return Promise.resolve(false);
    //     } else {
    //         return Promise.resolve(true);
    //     }
    // }

    getResponseFormat(_options: ExecutionOptions): undefined {
        //TODO: when forcing json_object type the streaming is not supported.
        // either implement canStream as above or comment the code below:
        // const responseFormatJson: Groq.Chat.Completions.CompletionCreateParams.ResponseFormat = {
        //     type: "json_object",
        // }

        // return _options.result_schema ? responseFormatJson : undefined;
        return undefined;
    }

    protected async formatPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<ChatCompletionMessageParam[]> {
        // Use OpenAI's multimodal formatter as base then convert to Groq types
        const responseItems = await formatOpenAILikeMultimodalPrompt(segments, {
            ...opts,
            multimodal: true,
        });

        // Convert ResponseInputItem[] to Groq ChatCompletionMessageParam[]
        return convertResponseItemsToGroqMessages(responseItems);
    }

    private getToolDefinitions(tools: ToolDefinition[] | undefined): ChatCompletionTool[] | undefined {
        if (!tools || tools.length === 0) {
            return undefined;
        }

        return tools.map(tool => ({
            type: 'function' as const,
            function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.input_schema satisfies FunctionParameters,
            }
        }));
    }

    private extractToolUse(message: any): ToolUse[] | undefined {
        if (!message.tool_calls || message.tool_calls.length === 0) {
            return undefined;
        }

        return message.tool_calls.map((toolCall: any) => ({
            id: toolCall.id,
            tool_name: toolCall.function.name,
            tool_input: JSON.parse(toolCall.function.arguments || '{}'),
        }));
    }

    private sanitizeMessagesForGroq(messages: ChatCompletionMessageParam[]): ChatCompletionMessageParam[] {
        return messages.map(message => {
            // Remove any reasoning field from message objects
            const { reasoning, ...sanitizedMessage } = message as any;

            // If message has content array, filter out reasoning content types
            if (Array.isArray(sanitizedMessage.content)) {
                sanitizedMessage.content = sanitizedMessage.content.filter((part: any) => {
                    // Filter out any reasoning-related content parts
                    return part.type !== 'reasoning' && !('reasoning' in part);
                });
            }

            return sanitizedMessage as ChatCompletionMessageParam;
        });
    }

    async requestTextCompletion(messages: ChatCompletionMessageParam[], options: ExecutionOptions): Promise<Completion> {
        if (options.model_options?._option_id !== "text-fallback" && options.model_options?._option_id !== "groq-deepseek-thinking") {
            this.logger.warn({ options: options.model_options }, "Invalid model options");
        }
        options.model_options = options.model_options as TextFallbackOptions;

        // Update conversation with current messages
        let conversation = updateConversation(options.conversation as ChatCompletionMessageParam[], messages);

        // Filter out any reasoning content that Groq doesn't support
        conversation = this.sanitizeMessagesForGroq(conversation);

        const tools = this.getToolDefinitions(options.tools);

        const res = await this.client.chat.completions.create({
            model: options.model,
            messages: conversation,
            max_completion_tokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            response_format: this.getResponseFormat(options),
            tools: tools,
        });

        const choice = res.choices[0];
        const result = choice.message.content;

        // Extract tool use from the response
        const tool_use = this.extractToolUse(choice.message);

        // Update conversation with the response
        conversation = updateConversation(conversation, [choice.message]);

        let finish_reason = choice.finish_reason;
        if (tool_use && tool_use.length > 0) {
            finish_reason = "tool_calls";
        }

        return {
            result: result ? [{ type: "text", value: result }] : [],
            token_usage: {
                prompt: res.usage?.prompt_tokens,
                result: res.usage?.completion_tokens,
                total: res.usage?.total_tokens,
            },
            finish_reason: finish_reason,
            original_response: options.include_original_response ? res : undefined,
            conversation,
            tool_use,
        };
    }

    async requestTextCompletionStream(messages: ChatCompletionMessageParam[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn({ options: options.model_options }, "Invalid model options");
        }
        options.model_options = options.model_options as TextFallbackOptions;

        // Update conversation with current messages
        let conversation = updateConversation(options.conversation as ChatCompletionMessageParam[], messages);

        // Filter out any reasoning content that Groq doesn't support
        conversation = this.sanitizeMessagesForGroq(conversation);

        const tools = this.getToolDefinitions(options.tools);

        const res = await this.client.chat.completions.create({
            model: options.model,
            messages: conversation,
            max_completion_tokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            stream: true,
            tools: tools,
        });

        return transformAsyncIterator(res, (chunk) => {
            const choice = chunk.choices[0];
            let finish_reason = choice.finish_reason;

            // Check for tool calls in the delta
            if (choice.delta.tool_calls && choice.delta.tool_calls.length > 0) {
                finish_reason = "tool_calls";
            }

            return {
                result: choice.delta.content ? [{ type: "text", value: choice.delta.content }] : [],
                finish_reason: finish_reason ?? undefined,
                token_usage: {
                    prompt: chunk.x_groq?.usage?.prompt_tokens,
                    result: chunk.x_groq?.usage?.completion_tokens,
                    total: chunk.x_groq?.usage?.total_tokens,
                },
            } satisfies CompletionChunkObject;
        });
    }

    async listModels(): Promise<AIModel<string>[]> {
        const models = await this.client.models.list();

        if (!models.data) {
            throw new Error("No models found");
        }

        const aiModels = models.data?.map(m => {
            if (!m.id) {
                throw new Error("Model id is missing");
            }
            return {
                id: m.id,
                name: m.id,
                description: undefined,
                provider: this.provider,
                owner: m.owned_by || '',
            }
        });

        return aiModels;
    }

    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }

    async generateEmbeddings({ }: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error("Method not implemented.");
    }

}

/**
 * Update the conversation messages by combining existing conversation with new messages
 * @param conversation Existing conversation messages
 * @param messages New messages to add
 * @returns Combined conversation
 */
function updateConversation(
    conversation: ChatCompletionMessageParam[] | undefined,
    messages: ChatCompletionMessageParam[]
): ChatCompletionMessageParam[] {
    return (conversation || []).concat(messages);
}

/**
 * Convert ResponseInputItem[] to Groq ChatCompletionMessageParam[]
 */
function convertResponseItemsToGroqMessages(items: ResponseInputItem[]): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [];

    for (const item of items) {
        // Handle EasyInputMessage (has role and content)
        if ('role' in item && 'content' in item) {
            const msg = item as EasyInputMessage;
            const role = msg.role;

            // Handle system/developer messages
            if (role === 'system' || role === 'developer') {
                let content: string;
                if (typeof msg.content === 'string') {
                    content = msg.content;
                } else if (Array.isArray(msg.content)) {
                    content = msg.content
                        .filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text')
                        .map(part => part.text)
                        .join('\n');
                } else {
                    content = '';
                }
                messages.push({ role: 'system', content });
                continue;
            }

            // Handle user messages
            if (role === 'user') {
                let content: string | Array<{ type: 'text', text: string } | { type: 'image_url', image_url: { url: string, detail?: 'auto' | 'low' | 'high' } }>;
                if (typeof msg.content === 'string') {
                    content = msg.content;
                } else if (Array.isArray(msg.content)) {
                    const parts: Array<{ type: 'text', text: string } | { type: 'image_url', image_url: { url: string, detail?: 'auto' | 'low' | 'high' } }> = [];
                    for (const part of msg.content) {
                        if (part.type === 'input_text') {
                            parts.push({ type: 'text', text: part.text });
                        } else if (part.type === 'input_image') {
                            const imgPart = part as OpenAI.Responses.ResponseInputImage;
                            if (imgPart.image_url) {
                                parts.push({
                                    type: 'image_url',
                                    image_url: {
                                        url: imgPart.image_url,
                                        ...(imgPart.detail && { detail: imgPart.detail })
                                    }
                                });
                            }
                        }
                    }
                    content = parts.length > 0 ? parts : '';
                } else {
                    content = '';
                }
                messages.push({ role: 'user', content });
                continue;
            }

            // Handle assistant messages
            if (role === 'assistant') {
                let content: string | null;
                if (typeof msg.content === 'string') {
                    content = msg.content;
                } else if (Array.isArray(msg.content)) {
                    content = msg.content
                        .filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text')
                        .map(part => part.text)
                        .join('\n') || null;
                } else {
                    content = null;
                }
                messages.push({ role: 'assistant', content });
                continue;
            }
        }

        // Handle function_call_output (tool response)
        if ('type' in item && item.type === 'function_call_output') {
            const output = item as OpenAI.Responses.ResponseInputItem.FunctionCallOutput;
            messages.push({
                role: 'tool',
                tool_call_id: output.call_id,
                content: typeof output.output === 'string' ? output.output : JSON.stringify(output.output),
            });
            continue;
        }

        // Handle function_call (assistant tool call)
        if ('type' in item && item.type === 'function_call') {
            const call = item as OpenAI.Responses.ResponseFunctionToolCall;
            // Groq expects tool_calls in assistant message, but we handle them separately
            // This is a simplification - in practice tool_calls come from model responses
            messages.push({
                role: 'assistant',
                content: null,
                tool_calls: [{
                    id: call.call_id,
                    type: 'function',
                    function: {
                        name: call.name,
                        arguments: call.arguments,
                    }
                }]
            });
            continue;
        }
    }

    return messages;
}