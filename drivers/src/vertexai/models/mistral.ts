import { MistralGoogleCloud } from "@mistralai/mistralai-gcp";
import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType, PromptOptions, PromptRole, PromptSegment } from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";
import { logger } from "@azure/identity";

interface VertexAIMistralMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    tool_call_id?: string | null;
    name?: string | null;
    prefix?: boolean;
    tool_calls?: string[];
}

//One of the benefits of using a client SDK, is we can use their types directly
//In this case if you get the type of "event" from the eventStream, 
//Right click, go to type definition. That will give you the format.
// interface VertexAIMistralResponse {
//     id: string;
//     object: string;
//     created: number;
//     model: string;
//     choices: Array<{
//         index: number;
//         message: VertexAIMistralMessage;
//         finish_reason: string;
//         delta?: {
//             content?: string;
//         };
//     }>;
//     usage: {
//         prompt_tokens: number;
//         completion_tokens: number;
//         total_tokens: number;
//     };
// }

interface VertexAIMistralRequest {
    model: string;
    messages: VertexAIMistralMessage[];
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    stream?: boolean;
    safe_mode?: boolean;
    random_seed?: number;
    stop?: string[];
    response_format?: { type: 'text' | 'json_object' };
}


const client = new MistralGoogleCloud({
    region: process.env.GOOGLE_REGION,
    projectId: process.env.GOOGLE_PROJECT_ID
});

function getMistralModel(driver: VertexAIDriver, options: ExecutionOptions): VertexAIMistralRequest {
    const splits = options.model.split("/");
    const modelName = splits[splits.length - 1];

    logger.info(`Using Mistral driver: ${driver}`);

    return {
        model: modelName,
        messages: [],
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        top_p: options.top_p,
        safe_mode: true,
        response_format: options.result_schema ? { type: 'json_object' } : { type: 'text' }
        // random_seed: options.seed,
        // stop: options.stop_sequence ? options.stop_sequence : undefined,
    };
}

export class MistralModelDefinition implements ModelDefinition<VertexAIMistralMessage[]> {
    model: AIModel;

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } as AIModel;
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<VertexAIMistralMessage[]> {
        const contents: VertexAIMistralMessage[] = [];
        const safety: string[] = [];
        let lastUserMessage: VertexAIMistralMessage | undefined = undefined;

        //Loop over segments, to process prompt
        for (const msg of segments) {
            if (msg.role === PromptRole.safety) {
                safety.push(msg.content as string);
            } else {
                const role = msg.role === PromptRole.assistant ? 'assistant' :
                    msg.role === PromptRole.system ? 'system' : 'user';

                if (lastUserMessage && lastUserMessage.role === role) {
                    lastUserMessage.content += '\n' + msg.content;
                } else {
                    const message: VertexAIMistralMessage = {
                        role,
                        content: msg.content as string
                    };
                    if (role === 'user') {
                        lastUserMessage = message;
                    }
                    contents.push(message);
                }
            }
        }

        //If a schema is provided, tell the model to follow it.
        if (options.result_schema) {
            safety.push(
                `The answer must be a JSON object using the following JSON Schema: ${JSON.stringify(options.result_schema)}`
            );
        }

        if (safety.length > 0) {
            safety.map((s) => contents.push({
                role: 'system',
                content: s
            }));
        }

        return contents;
    }

    async requestCompletion(driver: VertexAIDriver, prompt: VertexAIMistralMessage[], options: ExecutionOptions): Promise<Completion> {
        //Use the fetch client to access the model
        // const client = driver.fetchClient;
        const request = getMistralModel(driver, options);
        request.messages = prompt;
        request.stream = false;

        const response = await client.chat.complete(request);

        return {
            result: response,
            token_usage: {
                prompt: response.usage.promptTokens,
                result: response.usage.completionTokens,
                total: response.usage.totalTokens
            },
            finish_reason: response?.choices?.[0]?.finishReason ?? undefined,
        };
    }

    async requestCompletionStream(driver: VertexAIDriver, prompt: VertexAIMistralMessage[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const request = getMistralModel(driver, options);
        request.messages = prompt;

        const eventStream = await client.chat.stream(request);

        // Create an async generator to process the stream
        async function* generateChunks(): AsyncGenerator<CompletionChunkObject> {
            try {
                for await (const chunk of eventStream) {
                    // Extract the content from the chunk while handling potential undefined values
                    const content = chunk?.data?.choices?.[0]?.delta?.content ?? undefined;
                    if (content) {
                        yield {
                            result: content,
                            // Optional fields that may not be available in streaming mode
                            finish_reason: chunk?.data?.choices?.[0]?.finishReason ?? undefined,
                            token_usage: chunk?.data?.usage ? {
                                prompt: chunk?.data?.usage?.promptTokens ?? undefined,
                                result: chunk?.data?.usage?.completionTokens ?? undefined,
                                total: chunk?.data?.usage?.totalTokens ?? undefined
                            } : undefined
                        };
                    }
                }
            } catch (err) {
                // Log error but don't throw to maintain stream
                logger.error('Error processing Mistral stream chunk:', err);
            }
        }

        return generateChunks();
    }
}
