import type {
    AIModel,
    Completion,
    CompletionChunkObject,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    PromptSegment,
    TextFallbackOptions,
    ToolDefinition,
    ToolUse,
} from '@llumiverse/core';
import { transformAsyncIterator } from '@llumiverse/core/async';
import { AbstractDriver } from '@llumiverse/core/driver';
import Groq from 'groq-sdk';
import type { ChatCompletionMessageParam, ChatCompletionTool } from 'groq-sdk/resources/chat/completions';
import type { FunctionParameters } from 'groq-sdk/resources/shared';
import type OpenAI from 'openai';
import { convertResponseItemsToChatMessages, formatOpenAILikeMultimodalPrompt } from '../openai/openai_format.js';
import { truncateDataUrlForDebug } from '../shared/debug-prompt.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type GroqToolCallMessage = {
    tool_calls?: Array<{
        id: string;
        function: {
            name: string;
            arguments?: string;
        };
    }>;
};

interface GroqDriverOptions extends DriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

type GroqTextContentPart = { type: 'text'; text: string };
type GroqImageContentPart = { type: 'image_url'; image_url: { url: string; detail?: 'auto' | 'low' | 'high' } };
type GroqContentPart = GroqTextContentPart | GroqImageContentPart;
type GroqUserMessageWithArrayContent = Extract<ChatCompletionMessageParam, { role: 'user' }> & {
    content: GroqContentPart[];
};

function hasGroqArrayContent(message: ChatCompletionMessageParam): message is GroqUserMessageWithArrayContent {
    return message.role === 'user' && Array.isArray(message.content);
}

function formatGroqContentPartForDebug(part: GroqContentPart): GroqContentPart {
    if (part.type !== 'image_url') {
        return part;
    }
    return {
        ...part,
        image_url: {
            ...part.image_url,
            url: truncateDataUrlForDebug(part.image_url.url),
        },
    };
}

export class GroqDriver extends AbstractDriver<GroqDriverOptions, ChatCompletionMessageParam[]> {
    static PROVIDER = 'groq';
    provider = GroqDriver.PROVIDER;
    apiKey: string;
    client: Groq;
    endpointUrl?: string;

    constructor(options: GroqDriverOptions) {
        super(options);
        this.apiKey = options.apiKey;
        this.client = new Groq({
            apiKey: options.apiKey,
            baseURL: options.endpoint_url,
            fetch: this.getDriverFetch(),
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

    protected async formatPrompt(
        segments: PromptSegment[],
        opts: ExecutionOptions,
    ): Promise<ChatCompletionMessageParam[]> {
        // Use OpenAI's multimodal formatter as base then convert to Groq types
        const responseItems = await formatOpenAILikeMultimodalPrompt(segments, {
            ...opts,
            multimodal: true,
        });

        // Convert ResponseInputItem[] to Groq ChatCompletionMessageParam[]
        return convertResponseItemsToGroqMessages(responseItems);
    }

    public formatDebugPrompt(messages: ChatCompletionMessageParam[]): ChatCompletionMessageParam[] {
        return messages.map((message) => {
            if (!hasGroqArrayContent(message)) {
                return message;
            }
            return {
                ...message,
                content: message.content.map(formatGroqContentPartForDebug),
            };
        });
    }

    private getToolDefinitions(tools: ToolDefinition[] | undefined): ChatCompletionTool[] | undefined {
        if (!tools || tools.length === 0) {
            return undefined;
        }

        return tools.map((tool) => ({
            type: 'function' as const,
            function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.input_schema satisfies FunctionParameters,
            },
        }));
    }

    private extractToolUse(message: GroqToolCallMessage): ToolUse<unknown>[] | undefined {
        if (!message.tool_calls || message.tool_calls.length === 0) {
            return undefined;
        }

        return message.tool_calls.map((toolCall) => ({
            id: toolCall.id,
            tool_name: toolCall.function.name,
            tool_input: JSON.parse(toolCall.function.arguments || '{}'),
        }));
    }

    private sanitizeMessagesForGroq(messages: ChatCompletionMessageParam[]): ChatCompletionMessageParam[] {
        return messages.map((message) => {
            // Remove any reasoning field from message objects
            const sanitizedMessage = { ...(message as unknown as Record<string, unknown>) };
            delete sanitizedMessage.reasoning;

            // If message has content array, filter out reasoning content types
            const content = sanitizedMessage.content;
            if (Array.isArray(content)) {
                sanitizedMessage.content = content.filter((part) => {
                    // Filter out any reasoning-related content parts
                    const typedPart = part as { type?: string; reasoning?: unknown };
                    return typedPart.type !== 'reasoning' && !('reasoning' in typedPart);
                });
            }

            return sanitizedMessage as unknown as ChatCompletionMessageParam;
        });
    }

    async requestTextCompletion(
        messages: ChatCompletionMessageParam[],
        options: ExecutionOptions,
    ): Promise<Completion> {
        if (
            options.model_options?._option_id !== undefined &&
            options.model_options?._option_id !== 'text-fallback' &&
            options.model_options?._option_id !== 'groq-deepseek-thinking'
        ) {
            this.logger.debug({ options: options.model_options }, 'Unexpected option id');
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
            finish_reason = 'tool_calls';
        }

        return {
            result: result ? [{ type: 'text', value: result }] : [],
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

    async requestTextCompletionStream(
        messages: ChatCompletionMessageParam[],
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        if (options.model_options?._option_id !== undefined && options.model_options?._option_id !== 'text-fallback') {
            this.logger.debug({ options: options.model_options }, 'Unexpected option id');
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
                finish_reason = 'tool_calls';
            }

            return {
                result: choice.delta.content ? [{ type: 'text', value: choice.delta.content }] : [],
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
            throw new Error('No models found');
        }

        const aiModels = models.data?.map((m) => {
            if (!m.id) {
                throw new Error('Model id is missing');
            }
            return {
                id: m.id,
                name: m.id,
                description: undefined,
                provider: this.provider,
                owner: m.owned_by || '',
            };
        });

        return aiModels;
    }

    validateConnection(): Promise<boolean> {
        throw new Error('Method not implemented.');
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Method not implemented.');
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
    messages: ChatCompletionMessageParam[],
): ChatCompletionMessageParam[] {
    return (conversation || []).concat(messages);
}

/**
 * Convert ResponseInputItem[] to Groq ChatCompletionMessageParam[].
 *
 * Delegates to the shared Response->Chat-Completions converter in openai_format.ts.
 * The `as unknown as` cast bridges the structurally-identical `ChatCompletionMessageParam`
 * types from the `openai` and `groq-sdk` packages (same OpenAI wire format, distinct nominal
 * declarations).
 */
function convertResponseItemsToGroqMessages(items: ResponseInputItem[]): ChatCompletionMessageParam[] {
    return convertResponseItemsToChatMessages(items) as unknown as ChatCompletionMessageParam[];
}
