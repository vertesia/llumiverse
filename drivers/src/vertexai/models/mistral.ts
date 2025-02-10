import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType, PromptOptions, PromptRole, PromptSegment } from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";
import { asyncMap } from "@llumiverse/core/async";
import { logger } from "@azure/identity";

interface VertexAIMistralMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    tool_call_id?: string | null;
    name?: string | null;
    prefix?: boolean;
    tool_calls?: string[];
}

interface VertexAIMistralResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: Array<{
        index: number;
        message: VertexAIMistralMessage;
        finish_reason: string;
        delta?: {
            content?: string;
        };
    }>;
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
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

function getMistralModel(driver: VertexAIDriver, options: ExecutionOptions): VertexAIMistralRequest {
    const splits = options.model.split("/");
    const modelName = splits[splits.length - 1];

    logger.info(`Using Mistral model: ${modelName}`);
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
            logger.info(`Processing message: ${JSON.stringify(msg)}`);
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
            safety.push("The answer must be a JSON object using the following JSON Schema:\n" +
                JSON.stringify(options.result_schema));
        }

        // put system mesages first and safety last
        if (safety.length > 0) {
            const safetyContent = safety.join('\n');
            if (lastUserMessage) {
                lastUserMessage.content += '\n' + safetyContent;
            } else {
                contents.push({
                    role: 'system',
                    content: safetyContent
                });
            }
        }

        return contents;
    }

    async requestCompletion(driver: VertexAIDriver, prompt: VertexAIMistralMessage[], options: ExecutionOptions): Promise<Completion> {
        //Use the fetch client to access the model
        const client = driver.fetchClient;
        const request = getMistralModel(driver, options);
        request.messages = prompt;
        request.stream = false;

        const response = await client.post('/v1/chat/completions', { payload: request });

        return {
            result: response.choices[0].message.content,
            token_usage: {
                prompt: response.usage.prompt_tokens,
                result: response.usage.completion_tokens,
                total: response.usage.total_tokens
            },
            finish_reason: response.choices[0].finish_reason,
            original_response: options.include_original_response ? response : undefined,
        } as Completion;
    }

    async requestCompletionStream(driver: VertexAIDriver, prompt: VertexAIMistralMessage[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        // return {} as any; //To allow it to build
        const client = driver.fetchClient;
        const request = getMistralModel(driver, options);
        request.messages = prompt;
        request.stream = true;

        const response = await client.post('/v1/chat/completions', { payload: request });

        const stream = asyncMap(response, async (chunk: VertexAIMistralResponse) => {
            return {
                result: chunk.choices[0]?.delta?.content ?? '',
                token_usage: {
                    prompt: chunk.usage?.prompt_tokens,
                    result: chunk.usage?.completion_tokens,
                    total: chunk.usage?.total_tokens
                },
                finish_reason: chunk.choices[0]?.finish_reason
            };
        });

        return stream;
    }
}
