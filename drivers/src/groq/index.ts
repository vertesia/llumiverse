import { AIModel, AbstractDriver, Completion, CompletionChunkObject, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, PromptSegment, TextFallbackOptions } from "@llumiverse/core";
import { transformAsyncIterator } from "@llumiverse/core/async";
import { formatOpenAILikeMultimodalPrompt } from "../openai/openai_format.js";

import Groq from "groq-sdk";
import type { ChatCompletionMessageParam } from "groq-sdk/resources/chat/completions";

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
        const openaiMessages = await formatOpenAILikeMultimodalPrompt(segments, {
            ...opts,
            multimodal: true,
        });
        
        // Convert OpenAI ChatCompletionMessageParam[] to Groq ChatCompletionMessageParam[]
        // Handle differences between OpenAI and Groq SDK types
        const groqMessages: ChatCompletionMessageParam[] = openaiMessages.map(msg => {
            // Handle OpenAI developer messages - convert to system messages for Groq
            if (msg.role === 'developer' || msg.role === 'system') {
                const systemMsg: ChatCompletionMessageParam = {
                    role: 'system',
                    content: Array.isArray(msg.content) 
                        ? msg.content.map(part => part.text).join('\n')
                        : msg.content,
                    // Preserve name if present
                    ...(msg.name && { name: msg.name })
                };
                return systemMsg;
            }
            
            // Handle user messages - filter content parts to only supported types
            if (msg.role === 'user') {
                let content: string | Array<{type: 'text', text: string} | {type: 'image_url', image_url: {url: string, detail?: 'auto' | 'low' | 'high'}}> | undefined = undefined;
                
                if (typeof msg.content === 'string') {
                    content = msg.content;
                } else if (Array.isArray(msg.content)) {
                    // Filter to only text and image_url parts that Groq supports
                    const supportedParts = msg.content.filter(part => 
                        part.type === 'text' || part.type === 'image_url'
                    ).map(part => {
                        if (part.type === 'text') {
                            return { type: 'text' as const, text: part.text };
                        } else if (part.type === 'image_url') {
                            return { 
                                type: 'image_url' as const, 
                                image_url: {
                                    url: part.image_url.url,
                                    ...(part.image_url.detail && { detail: part.image_url.detail })
                                }
                            };
                        }
                        return null;
                    }).filter(Boolean) as Array<{type: 'text', text: string} | {type: 'image_url', image_url: {url: string, detail?: 'auto' | 'low' | 'high'}}>;
                    
                    content = supportedParts.length > 0 ? supportedParts : 'Content not supported';
                }
                
                const userMsg: ChatCompletionMessageParam = {
                    role: 'user',
                    content: content ?? "",
                    // Preserve name if present
                    ...(msg.name && { name: msg.name })
                };
                return userMsg;
            }
            
            // Handle assistant messages - handle content arrays if needed
            if (msg.role === 'assistant') {
                const assistantMsg: ChatCompletionMessageParam = {
                    role: 'assistant',
                    content: Array.isArray(msg.content) 
                        ? msg.content.map(part => 'text' in part ? part.text : '').filter(Boolean).join('\n') || null
                        : msg.content,
                    // Preserve other assistant message properties
                    ...(msg.function_call && { function_call: msg.function_call }),
                    ...(msg.tool_calls && { tool_calls: msg.tool_calls }),
                    ...(msg.name && { name: msg.name })
                };
                return assistantMsg;
            }
            
            // For tool and function messages, they should be compatible
            if (msg.role === 'tool') {
                const toolMsg: ChatCompletionMessageParam = {
                    role: 'tool',
                    tool_call_id: msg.tool_call_id,
                    content: Array.isArray(msg.content) 
                        ? msg.content.map(part => part.text).join('\n')
                        : msg.content
                };
                return toolMsg;
            }
            
            if (msg.role === 'function') {
                const functionMsg: ChatCompletionMessageParam = {
                    role: 'function',
                    name: msg.name,
                    content: msg.content
                };
                return functionMsg;
            }
            
            // Fallback - should not reach here but provides type safety
            throw new Error(`Unsupported message role: ${(msg as any).role}`);
        });

        return groqMessages;
    }

    async requestTextCompletion(messages: ChatCompletionMessageParam[], options: ExecutionOptions): Promise<Completion> {
        if (options.model_options?._option_id !== "text-fallback" && options.model_options?._option_id !== "groq-deepseek-thinking") {
            this.logger.warn("Invalid model options", {options: options.model_options });
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const res = await this.client.chat.completions.create({
            model: options.model,
            messages: messages,
            max_completion_tokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            response_format: this.getResponseFormat(options),
        });


        const choice = res.choices[0];
        const result = choice.message.content;

        return {
            result: result,
            token_usage: {
                prompt: res.usage?.prompt_tokens,
                result: res.usage?.completion_tokens,
                total: res.usage?.total_tokens,
            },
            finish_reason: choice.finish_reason,
            original_response: options.include_original_response ? res : undefined,
        };
    }

    async requestTextCompletionStream(messages: ChatCompletionMessageParam[], options: ExecutionOptions): Promise <AsyncIterable<CompletionChunkObject>> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn("Invalid model options", {options: options.model_options });
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const res = await this.client.chat.completions.create({
            model: options.model,
            messages: messages,
            max_completion_tokens: options.model_options?.max_tokens,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            stream: true,
        });

        return transformAsyncIterator(res, (res) => ({
            result: res.choices[0].delta.content ?? '',
            finish_reason: res.choices[0].finish_reason,
            token_usage: {
                prompt: res.x_groq?.usage?.prompt_tokens,
                result: res.x_groq?.usage?.completion_tokens,
                total: res.x_groq?.usage?.total_tokens,
            },
            } as CompletionChunkObject));
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