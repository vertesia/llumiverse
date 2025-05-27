import {
    AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType,
    PromptOptions, PromptRole, PromptSegment
} from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";

interface LLamaMessage {
    role: string;
    content: string;
}

interface LLamaPrompt {
    messages: LLamaMessage[];
    system?: string;
}

// // Define specific options for Llama MaaS
// interface LLamaModelOptions {
//     temperature?: number;
//     max_tokens?: number;
//     top_p?: number;
//     top_k?: number;
// }

// // Define specific options for Llama MaaS
// interface LLamaModelOptions {
//     temperature?: number;
//     max_tokens?: number;
//     top_p?: number;
//     top_k?: number;
// }

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
        };
        finish_reason: string | null;
    }[];
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
 * Get the max token limit for the model
 */
function maxToken(max_tokens: number | undefined, _model: string): number {
    return max_tokens || 8192;
}

/**
 * Update the conversation messages
 * @param conversation The previous conversation context
 * @param prompt The new prompt to add to the conversation
 * @returns Updated conversation with combined messages
 */
function updateConversation(conversation: LLamaPrompt | undefined | null, prompt: LLamaPrompt): LLamaPrompt {
    const baseMessages = conversation ? conversation.messages : [];
    const baseSystem = conversation?.system || undefined;

    return {
        messages: [...baseMessages, ...(prompt.messages || [])],
        system: prompt.system || baseSystem
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
            can_stream: false,
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

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], _options: PromptOptions): Promise<LLamaPrompt> {
        const messages: LLamaMessage[] = [];
        let systemContent = '';

        // Process segments and convert them to the Llama MaaS format
        for (const segment of segments) {
            if (segment.role === PromptRole.system) {
                // Collect system messages
                systemContent += (systemContent ? "\n" : "") + segment.content;
                continue;
            }

            if (segment.role === PromptRole.safety) {
                // Add safety instructions to system content
                systemContent += (systemContent ? "\n" : "") + segment.content;
                continue;
            }

            // Convert the prompt segments to messages
            const role = segment.role === PromptRole.assistant ? 'assistant' :
                segment.role === PromptRole.user ? 'user' : 'user';

            // Combine files and text content if needed
            let messageContent = segment.content || '';

            // Add any files as text attachments
            if (segment.files && segment.files.length > 0) {
                for (const file of segment.files) {
                    if (file.mime_type?.startsWith("text/")) {
                        const fileStream = await file.getStream();
                        const fileContent = await streamToString(fileStream);
                        messageContent += `\n\nFile content:\n${fileContent}`;
                    }
                    // Note: Images are not supported in this basic implementation
                }
            }

            messages.push({
                role: role,
                content: messageContent
            });
        }

        // Return the prompt in the format expected by Llama MaaS API
        return {
            messages: messages,
            system: systemContent || undefined
        };
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: LLamaPrompt, options: ExecutionOptions): Promise<Completion> {
        const splits = options.model.split("/");

        let conversation = updateConversation(options.conversation as LLamaPrompt, prompt);

        // Extract model name for API payload (without publisher path)
        const modelName = splits[splits.length - 1];

        // Determine the correct region based on model name
        const region = this.getLlamaModelRegion(modelName);

        // Build the request payload with the correct model format "meta/MODEL"
        const payload: Record<string, any> = {
            model: `meta/${modelName}`,
            messages: conversation.messages,
            stream: false,
            // Add a default max_tokens to avoid truncated responses
            max_tokens: 1024
        };

        // Construct the API endpoint for the OpenAI-compatible Llama MaaS API
        const openaiEndpoint = `endpoints/openapi/chat/completions`;

        // Make the API call using the FetchClient's post method
        const client = driver.getLLamaClient(region);
        const result = await client.post(openaiEndpoint, {
            payload
        }) as LLamaResponse;

        console.log("Llama response:", result);
        console.error("Llama response:", JSON.stringify(result, null, 2));

        // Extract response data
        const assistantMessage = result?.choices[0]?.message;
        const text = assistantMessage?.content;

        // Update conversation with the response
        conversation = updateConversation(conversation, {
            messages: [{
                role: assistantMessage?.role,
                content: text
            }],
            system: undefined
        });

        return {
            chat: [prompt, { role: assistantMessage?.role, content: text }],
            result: text,
            token_usage: {
                prompt: result.usage.prompt_tokens,
                result: result.usage.completion_tokens,
                total: result.usage.total_tokens
            },
            finish_reason: result.choices[0].finish_reason,
            conversation
        } as Completion;
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: LLamaPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];

        // Determine the correct region based on model name
        const region = this.getLlamaModelRegion(modelName);

        const client = driver.getLLamaClient(region);

        let conversation = updateConversation(options.conversation as LLamaPrompt, prompt);

        // Construct the API endpoint for the OpenAI-compatible Llama MaaS API
        const openaiEndpoint = `endpoints/openapi/chat/completions`;

        // Build the request payload with the correct model format "meta/MODEL"
        const payload: Record<string, any> = {
            model: `meta/${modelName}`,
            messages: conversation.messages,
            stream: true,
        };

        // Add optional parameters if they exist
        const modelOptions = options.model_options as any;
        if (modelOptions?.temperature !== undefined) {
            payload.temperature = modelOptions.temperature;
        }

        if (modelOptions?.max_tokens !== undefined) {
            payload.max_tokens = modelOptions.max_tokens;
        } else {
            payload.max_tokens = maxToken(undefined, modelName);
        }

        if (modelOptions?.top_p !== undefined) {
            payload.top_p = modelOptions.top_p;
        }

        if (conversation.system) {
            payload.system = conversation.system;
        }

        // Make the streaming API call
        const response = await client.fetch(openaiEndpoint, {
            method: 'POST',
            body: JSON.stringify(payload)
        });

        // Create an async generator to parse the SSE stream
        const parseStream = async function* () {
            const reader = response.body!.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        const data = line.slice(5).trim();

                        if (data === '[DONE]') {
                            continue;
                        }

                        try {
                            const parsedData = JSON.parse(data) as LLamaStreamResponse;
                            const choice = parsedData.choices[0];

                            yield {
                                result: choice.delta?.content || '',
                                token_usage: { prompt: 0, result: 0 }, // Not available in the stream
                                finish_reason: choice.finish_reason || undefined,
                            } as CompletionChunkObject;
                        } catch (error) {
                            console.error('Error parsing SSE data:', error);
                        }
                    }
                }
            }

            if (buffer.length > 0) {
                const lines = buffer.split('\n');
                for (const line of lines) {
                    if (line.startsWith('data:') && line.slice(5).trim() !== '[DONE]') {
                        try {
                            const parsedData = JSON.parse(line.slice(5).trim()) as LLamaStreamResponse;
                            const choice = parsedData.choices[0];

                            yield {
                                result: choice.delta?.content || '',
                                token_usage: { prompt: 0, result: 0 },
                                finish_reason: choice.finish_reason || undefined,
                            } as CompletionChunkObject;
                        } catch (error) {
                            console.error('Error parsing SSE data:', error);
                        }
                    }
                }
            }
        };

        return parseStream();
    }
}
