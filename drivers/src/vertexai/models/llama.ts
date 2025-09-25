import {
    AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType,
    PromptOptions, PromptRole, PromptSegment,
    TextFallbackOptions
} from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";
import { transformSSEStream } from "@llumiverse/core/async";

interface LLamaMessage {
    role: string;
    content: string;
}

interface LLamaPrompt {
    messages: LLamaMessage[];
}

interface LLamaResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: {
        index: number;
        message: {
            role: string;
            content: string;
            refusal?: string;
        };
        finish_reason: string;
    }[];
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

interface LLamaStreamResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: {
        index: number;
        delta: {
            role?: string;
            content?: string;
            refusal?: string;
        };
        finish_reason?: string;
    }[];
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

/**
 * Convert a stream to a string
 */
async function streamToString(stream: any): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks).toString('utf-8');
}

/**
 * Update the conversation messages
 * @param conversation The previous conversation context
 * @param prompt The new prompt to add to the conversation
 * @returns Updated conversation with combined messages
 */
function updateConversation(conversation: LLamaPrompt | undefined | null, prompt: LLamaPrompt): LLamaPrompt {
    const baseMessages = conversation ? conversation.messages : [];

    return {
        messages: [...baseMessages, ...(prompt.messages || [])],
    };
}

export class LLamaModelDefinition implements ModelDefinition<LLamaPrompt> {

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

    // Return the appropriate region based on the Llama model
    getLlamaModelRegion(modelName: string): string {
        // Llama 4 models are in us-east5, Llama 3.x models are in us-central1
        if (modelName.startsWith('llama-4')) {
            return 'us-east5';
        } else {
            return 'us-central1';
        }
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<LLamaPrompt> {
        const messages: LLamaMessage[] = [];

        // Process segments and convert them to the Llama MaaS format
        for (const segment of segments) {
            // Convert the prompt segments to messages
            const role = segment.role === PromptRole.assistant ? 'assistant' : 'user';

            // Combine files and text content if needed
            let messageContent = segment.content || '';

            if (segment.files && segment.files.length > 0) {
                for (const file of segment.files) {
                    if (file.mime_type?.startsWith("text/")) {
                        const fileStream = await file.getStream();
                        const fileContent = await streamToString(fileStream);
                        messageContent += `\n\nFile content:\n${fileContent}`;
                    }
                }
            }

            messages.push({
                role: role,
                content: messageContent
            });
        }

        if (options.result_schema) {
            messages.push({
                role: 'user',
                content: "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema)
            });
        }

        // Return the prompt in the format expected by Llama MaaS API
        return {
            messages: messages,
        };
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: LLamaPrompt, options: ExecutionOptions): Promise<Completion> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];

        let conversation = updateConversation(options.conversation as LLamaPrompt, prompt);

        const modelOptions = options.model_options as TextFallbackOptions;

        const payload: Record<string, any> = {
            model: `meta/${modelName}`,
            messages: conversation.messages,
            stream: false,
            max_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
            top_p: modelOptions?.top_p,
            top_k: modelOptions?.top_k,
            // Disable llama guard
            extra_body: {
                google: {
                    model_safety_settings: {
                        enabled: false,
                        llama_guard_settings: {}
                    }
                }
            }
        };

        // Make POST request to the Llama MaaS API
        const region = this.getLlamaModelRegion(modelName);
        const client = driver.getLLamaClient(region);
        const openaiEndpoint = `endpoints/openapi/chat/completions`;
        const result = await client.post(openaiEndpoint, {
            payload
        }) as LLamaResponse;

        // Extract response data
        const assistantMessage = result?.choices[0]?.message;
        const text = assistantMessage?.content;

        // Update conversation with the response
        conversation = updateConversation(conversation, {
            messages: [{
                role: assistantMessage?.role,
                content: text
            }],
        });

        return {
            result: [{ type: "text", value: text }],
            token_usage: {
                prompt: result.usage.prompt_tokens,
                result: result.usage.completion_tokens,
                total: result.usage.total_tokens
            },
            finish_reason: result.choices[0].finish_reason,
            conversation
        };
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: LLamaPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];

        const conversation = updateConversation(options.conversation as LLamaPrompt, prompt);

        const modelOptions = options.model_options as TextFallbackOptions;

        const payload: Record<string, any> = {
            model: `meta/${modelName}`,
            messages: conversation.messages,
            stream: true,
            max_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
            top_p: modelOptions?.top_p,
            top_k: modelOptions?.top_k,
            // Disable llama guard
            extra_body: {
                google: {
                    model_safety_settings: {
                        enabled: false,
                        llama_guard_settings: {}
                    }
                }
            }
        };

        // Make POST request to the Llama MaaS API
        //TODO: Fix error handling with the fetch client, errors will return a empty response
        //But not throw any error
        const region = this.getLlamaModelRegion(modelName);
        const client = driver.getLLamaClient(region);
        const openaiEndpoint = `endpoints/openapi/chat/completions`;
        const stream = await client.post(openaiEndpoint, {
            payload,
            reader: 'sse'
        });

        return transformSSEStream(stream, (data: string): CompletionChunkObject => {
            const json = JSON.parse(data) as LLamaStreamResponse;
            const choice = json.choices?.[0];
            const content = choice?.delta?.content ?? '';
            return {
                result: content ? [{ type: "text", value: content }] : [],
                finish_reason: choice?.finish_reason,
                token_usage: json.usage ? {
                    prompt: json.usage.prompt_tokens,
                    result: json.usage.completion_tokens,
                    total: json.usage.total_tokens,
                } : undefined
            };
        });
    }
}
