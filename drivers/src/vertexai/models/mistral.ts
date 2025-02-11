import { MistralGoogleCloud } from "@mistralai/mistralai-gcp";
import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType, PromptOptions, PromptRole, PromptSegment } from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";
import { asyncMap } from "@llumiverse/core/async";
import { logger } from "@azure/identity";
// import { FetchClient } from "api-fetch-client";

interface VertexAIMistralMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    tool_call_id?: string | null;
    name?: string | null;
    prefix?: boolean;
    tool_calls?: string[];
}

interface VertexAIMistralResponse {
    data: {
        id: string;
        object: string;
        created: number;
        model: string;
        choices: Array<{
            index: number;
            message: VertexAIMistralMessage;
            finishReason: string;
            delta?: {
                content?: string;
            };
        }>;
        usage: {
            promptTokens: number;
            completionTokens: number;
            totalTokens: number;
        };
    };
}

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
    region: "us-central1",
    projectId: process.env.GOOGLE_PROJECT_ID as string,
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

        console.log(request);
        
        const response = await client.chat.complete(request);
        if (response.choices && response.choices.length > 0) {
            console.log(response.choices[0]);
        }

        return {
            result: '',
            token_usage: { 
                prompt: response.usage?.promptTokens,
                result: response.usage?.completionTokens,
            },
            finish_reason: 'stop',
        } as Completion;
    }

    async requestCompletionStream(driver: VertexAIDriver, prompt: VertexAIMistralMessage[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const request = getMistralModel(driver, options);
        request.messages = prompt;
        console.log(request);

        const response = await client.chat.stream(request);

        const stream = asyncMap(response as any, async (chunk: VertexAIMistralResponse) => {
            const data = chunk.data;
            return {
                result: data.choices[0]?.delta?.content ?? '',
                token_usage: { 
                    prompt: data.usage?.promptTokens,
                    result: data.usage?.completionTokens,
                },
                finish_reason: data.choices[0]?.finishReason,
            } as CompletionChunkObject;
        });

        return stream;
    }
}
