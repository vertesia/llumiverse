import {
    AIModel,
    AbstractDriver,
    Completion,
    CompletionChunkObject,
    DataSource,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ExecutionTokenUsage,
    JSONSchema,
    ModelType,
    ToolDefinition,
    ToolUse,
    TrainingJob,
    TrainingJobStatus,
    TrainingOptions,
    TrainingPromptOptions,
    getInputModality,
    getOutputModality,
    modelModalitiesToArray,
    supportsToolUse,
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { formatOpenAILikeMultimodalPrompt, noStructuredOutputModels } from "@llumiverse/core/formatters";
import OpenAI, { AzureOpenAI } from "openai";
import { Stream } from "openai/streaming";

//For code readability
type OpenAIMessageBlock = OpenAI.Chat.Completions.ChatCompletionMessageParam;

//TODO: Do we need a list?, replace with if statements and modernise?
const supportFineTunning = new Set([
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "babbage-002",
    "davinci-002",
    "gpt-4-0613"
]);

export interface BaseOpenAIDriverOptions extends DriverOptions {
}

export abstract class BaseOpenAIDriver extends AbstractDriver<
    BaseOpenAIDriverOptions,
    OpenAIMessageBlock[]
> {
    abstract provider: "azure_openai" | "openai" | "xai";
    abstract service: OpenAI | AzureOpenAI;

    constructor(opts: BaseOpenAIDriverOptions) {
        super(opts);
        this.formatPrompt = formatOpenAILikeMultimodalPrompt as any
        //TODO: better type, we send back OpenAI.Chat.Completions.ChatCompletionMessageParam[] but just not compatible with Function call that we don't use here
    }

    extractDataFromResponse(
        _options: ExecutionOptions,
        result: OpenAI.Chat.Completions.ChatCompletion
    ): Completion {
        const tokenInfo: ExecutionTokenUsage = {
            prompt: result.usage?.prompt_tokens,
            result: result.usage?.completion_tokens,
            total: result.usage?.total_tokens,
        };

        const choice = result.choices[0];

        const tools = collectTools(choice.message.tool_calls);
        const data = choice.message.content ?? undefined;

        if (!data && !tools) {
            this.logger?.error("[OpenAI] Response is not valid", result);
            throw new Error("Response is not valid: no data");
        }

        return {
            result: data,
            token_usage: tokenInfo,
            finish_reason: openAiFinishReason(choice.finish_reason),
            tool_use: tools,
        };
    }

    async requestTextCompletionStream(prompt: OpenAIMessageBlock[], options: ExecutionOptions): Promise<any> {
        if (options.model_options?._option_id !== "openai-text" && options.model_options?._option_id !== "openai-thinking") {
            this.logger.warn("Invalid model options", { options: options.model_options });
        }
        
        const toolDefs = getToolDefinitions(options.tools);
        const useTools: boolean = toolDefs ? supportsTools(options.model) : false;

        const mapFn = (chunk: OpenAI.Chat.Completions.ChatCompletionChunk) => {
            let result = undefined
            if (useTools && this.provider !== "xai" && options.result_schema) {
                result = chunk.choices[0]?.delta?.tool_calls?.[0].function?.arguments ?? "";
            } else {
                result = chunk.choices[0]?.delta.content ?? "";
            }

            return {
                result: result,
                finish_reason: openAiFinishReason(chunk.choices[0]?.finish_reason ?? undefined),         //Uses expected "stop" , "length" format
                token_usage: {
                    prompt: chunk.usage?.prompt_tokens,
                    result: chunk.usage?.completion_tokens,
                    total: (chunk.usage?.prompt_tokens ?? 0) + (chunk.usage?.completion_tokens ?? 0),
                }
            } satisfies CompletionChunkObject;
        };

        convertRoles(prompt, options.model);

        const model_options = options.model_options as any;
        insert_image_detail(prompt, model_options?.image_detail ?? "auto");

        let parsedSchema: JSONSchema | undefined = undefined;
        let strictMode = false;
        if (options.result_schema && supportsSchema(options.model)) {
            try {
                parsedSchema = openAISchemaFormat(options.result_schema);
                strictMode = true;
            }
            catch (e) {
                parsedSchema = limitedSchemaFormat(options.result_schema);
                strictMode = false;
            }
        }

        const stream = await this.service.chat.completions.create({
            stream: true,
            stream_options: { include_usage: true },
            model: options.model,
            messages: prompt,
            reasoning_effort: model_options?.reasoning_effort,
            temperature: model_options?.temperature,
            top_p: model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: model_options?.presence_penalty,
            frequency_penalty: model_options?.frequency_penalty,
            n: 1,
            max_completion_tokens: model_options?.max_tokens, //TODO: use max_tokens for older models, currently relying on OpenAI to handle it
            tools: useTools ? toolDefs : undefined,
            stop: model_options?.stop_sequence,
            response_format: parsedSchema ? {
                type: "json_schema",
                json_schema: {
                    name: "format_output",
                    schema: parsedSchema,
                    strict: strictMode,
                }
            } : undefined,
        } satisfies OpenAI.Chat.ChatCompletionCreateParamsStreaming
        ) satisfies Stream<OpenAI.Chat.Completions.ChatCompletionChunk>;

        return asyncMap(stream, mapFn);
    }

    async requestTextCompletion(prompt: OpenAIMessageBlock[], options: ExecutionOptions): Promise<any> {
        if (options.model_options?._option_id !== "openai-text" && options.model_options?._option_id !== "openai-thinking") {
            this.logger.warn("Invalid model options", { options: options.model_options });
        }

        convertRoles(prompt, options.model);

        const model_options = options.model_options as any;
        insert_image_detail(prompt, model_options?.image_detail ?? "auto");

        const toolDefs = getToolDefinitions(options.tools);
        const useTools: boolean = toolDefs ? supportsTools(options.model) : false;

        let conversation = updateConversation(options.conversation as OpenAIMessageBlock[], prompt);

        let parsedSchema: JSONSchema | undefined = undefined;
        let strictMode = false;
        if (options.result_schema && supportsSchema(options.model)) {
            try {
                parsedSchema = openAISchemaFormat(options.result_schema);
                strictMode = true;
            }
            catch (e) {
                parsedSchema = limitedSchemaFormat(options.result_schema);
                strictMode = false;
            }
        }

        const res = await this.service.chat.completions.create({
            stream: false,
            model: options.model,
            messages: conversation,
            reasoning_effort: model_options?.reasoning_effort,
            temperature: model_options?.temperature,
            top_p: model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: model_options?.presence_penalty,
            frequency_penalty: model_options?.frequency_penalty,
            n: 1,
            max_completion_tokens: model_options?.max_tokens, //TODO: use max_tokens for older models, currently relying on OpenAI to handle it
            tools: useTools ? toolDefs : undefined,
            stop: model_options?.stop_sequence,
            response_format: parsedSchema ? {
                type: "json_schema",
                json_schema: {
                    name: "format_output",
                    schema: parsedSchema,
                    strict: strictMode,
                }
            } : undefined,
        });

        const completion = this.extractDataFromResponse(options, res);
        if (options.include_original_response) {
            completion.original_response = res;
        }

        conversation = updateConversation(conversation, createPromptFromResponse(res.choices[0].message));
        completion.conversation = conversation;

        return completion;
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        if (_options.model.includes("o1")
            && !(_options.model.includes("mini") || _options.model.includes("preview"))) {
            //o1 full does not support streaming
            //TODO: Update when OpenAI adds support for streaming, last check 16/02/2025
            return Promise.resolve(false);
        }
        return Promise.resolve(true);
    }

    createTrainingPrompt(options: TrainingPromptOptions): Promise<string> {
        if (options.model.includes("gpt")) {
            return super.createTrainingPrompt(options);
        } else {
            // babbage, davinci not yet implemented
            throw new Error("Unsupported model for training: " + options.model);
        }
    }

    async startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob> {
        const url = await dataset.getURL();
        const file = await this.service.files.create({
            file: await fetch(url),
            purpose: "fine-tune",
        });

        const job = await this.service.fineTuning.jobs.create({
            training_file: file.id,
            model: options.model,
            hyperparameters: options.params
        })

        return jobInfo(job);
    }

    async cancelTraining(jobId: string): Promise<TrainingJob> {
        const job = await this.service.fineTuning.jobs.cancel(jobId);
        return jobInfo(job);
    }

    async getTrainingJob(jobId: string): Promise<TrainingJob> {
        const job = await this.service.fineTuning.jobs.retrieve(jobId);
        return jobInfo(job);
    }

    // ========= management API =============

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.models.list();
            return true;
        } catch (error) {
            return false;
        }
    }

    listTrainableModels(): Promise<AIModel<string>[]> {
        return this._listModels((m) => supportFineTunning.has(m.id));
    }

    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(filter?: (m: OpenAI.Models.Model) => boolean): Promise<AIModel[]> {
        let result = (await this.service.models.list()).data;

        //Some of these use the completions API instead of the chat completions API.
        //Others are for non-text input modalities. Therefore common to both.
        const wordBlacklist = ["embed", "whisper", "transcribe", "audio", "moderation", "tts", "realtime", "dall-e", "babbage", "davinci"];
        
        if (this.provider === "azure_openai") {
            //Azure OpenAI has additional information about the models
            result = result.filter((m) => {
                return !(m as any)?.capabilities?.embeddings;
            });
        }

        //OpenAI has very little information, filtering based on name.
        result = result.filter((m) => {
            return !wordBlacklist.some((word) => m.id.includes(word));
        });

        const models = filter ? result.filter(filter) : result;
        return models.map((m) => ({
            id: m.id,
            name: m.id,
            provider: this.provider,
            owner: m.owned_by,
            type: m.object === "model" ? ModelType.Text : ModelType.Unknown,
            can_stream: true,
            is_multimodal: m.id.includes("gpt-4"),
            input_modalities: modelModalitiesToArray(getInputModality(m.id, "openai")),
            output_modalities: modelModalitiesToArray(getOutputModality(m.id, "openai")),
            tool_support: supportsToolUse(m.id, "openai", false),
        } satisfies AIModel<string>)).sort((a, b) => a.id.localeCompare(b.id));
    }


    async generateEmbeddings({ text, image, model = "text-embedding-3-small" }: EmbeddingsOptions): Promise<EmbeddingsResult> {

        if (image) {
            throw new Error("Image embeddings not supported by OpenAI");
        }

        if (!text) {
            throw new Error("No text provided");
        }

        const res = await this.service.embeddings.create({
            input: text,
            model: model,
        });

        const embeddings = res.data[0].embedding;

        if (!embeddings || embeddings.length === 0) {
            throw new Error("No embedding found");
        }

        return { values: embeddings, model } satisfies EmbeddingsResult;
    }

}


function jobInfo(job: OpenAI.FineTuning.Jobs.FineTuningJob): TrainingJob {
    //validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
    const jobStatus = job.status;
    let status = TrainingJobStatus.running;
    let details: string | undefined;
    if (jobStatus === 'succeeded') {
        status = TrainingJobStatus.succeeded;
    } else if (jobStatus === 'failed') {
        status = TrainingJobStatus.failed;
        details = job.error ? `${job.error.code} - ${job.error.message} ${job.error.param ? " [" + job.error.param + "]" : ""}` : "error";
    } else if (jobStatus === 'cancelled') {
        status = TrainingJobStatus.cancelled;
    } else {
        status = TrainingJobStatus.running;
        details = jobStatus;
    }
    return {
        id: job.id,
        model: job.fine_tuned_model || undefined,
        status,
        details
    }
}

function insert_image_detail(messages: OpenAIMessageBlock[], detail_level: string): OpenAIMessageBlock[] {
    if (detail_level == "auto" || detail_level == "low" || detail_level == "high") {
        for (const message of messages) {
            if (message.role !== 'assistant' && message.content) {
                for (const part of message.content) {
                    if (typeof part === "string") {
                        continue;
                    }
                    if (part.type === 'image_url') {
                        part.image_url = { ...part.image_url, detail: detail_level };
                    }
                }
            }
        }
    }
    return messages;
}

function convertRoles(messages: OpenAIMessageBlock[], model: string): OpenAIMessageBlock[] {
    //New openai models use developer role instead of system
    if (model.includes("o1") || model.includes("o3")) {
        if (model.includes("o1-mini") || model.includes("o1-preview")) {
            //o1-mini and o1-preview support neither system nor developer
            for (const message of messages) {
                if (message.role === 'system') {
                    (message.role as any) = 'user';
                }
            }
        } else {
            //Models newer than o1 use developer role
            for (const message of messages) {
                if (message.role === 'system') {
                    (message.role as any) = 'developer';
                }
            }
        }
    }
    return messages
}

function supportsTools(model: string): boolean {
    const list_check = !noStructuredOutputModels.some((m) => model.includes(m));
    if (!list_check && model.includes("gpt-4o") && !model.includes("gpt-4o-2024-05-13")) {
        return true;
    }
    return list_check
}

function supportsSchema(model: string): boolean {
    const list_check = !noStructuredOutputModels.some((m) => model.includes(m));
    if (!list_check && model.includes("gpt-4o") && !model.includes("gpt-4o-2024-05-13")) {
        return true; 
    }
    return list_check
}

function getToolDefinitions(tools: ToolDefinition[] | undefined | null): OpenAI.ChatCompletionTool[] | undefined {
    return tools ? tools.map(getToolDefinition) : undefined;
}
function getToolDefinition(toolDef: ToolDefinition): OpenAI.ChatCompletionTool {
    let parsedSchema: JSONSchema | undefined = undefined;
    let strictMode = false;
    if (toolDef.input_schema) {
        try {
            parsedSchema = openAISchemaFormat(toolDef.input_schema as JSONSchema);
            strictMode = true;
        }
        catch (e) {
            parsedSchema = limitedSchemaFormat(toolDef.input_schema as JSONSchema);
            strictMode = false;
        }
    }

    return {
        type: "function",
        function: {
            name: toolDef.name,
            description: toolDef.description,
            parameters: parsedSchema,
            strict: strictMode,
        },
    } satisfies OpenAI.ChatCompletionTool;
}

function openAiFinishReason(finish_reason?: string): string | undefined {
    if (finish_reason === "tool_calls") {
        return "tool_use";
    }
    return finish_reason;
}

function updateConversation(conversation: OpenAIMessageBlock[], message: OpenAIMessageBlock[]): OpenAIMessageBlock[] {
    if (!message) {
        return conversation;
    }
    if (!conversation) {
        return message;
    }
    return [...conversation, ...message];
}

export function collectTools(toolCalls?: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[]): ToolUse[] | undefined {
    if (!toolCalls) {
        return undefined;
    }

    const tools: ToolUse[] = [];
    for (const call of toolCalls) {
        tools.push({
            id: call.id,
            tool_name: call.function.name,
            tool_input: JSON.parse(call.function.arguments),
        });

    }
    return tools.length > 0 ? tools : undefined;
}

function createPromptFromResponse(response: OpenAI.Chat.Completions.ChatCompletionMessage) : OpenAIMessageBlock[] {
    const messages: OpenAIMessageBlock[] = [];
    if (response) {
        messages.push({
            role: response.role,
            content: response.content,
            tool_calls: response.tool_calls,
        });
    }
    return messages;
}

//For strict mode false
function limitedSchemaFormat(schema: JSONSchema): JSONSchema {
    const formattedSchema = { ...schema };

    // Defaults not supported
    delete formattedSchema.default;

    if (formattedSchema?.properties) {
        // Process each property recursively
        for (const propName of Object.keys(formattedSchema.properties)) {
            const property = formattedSchema.properties[propName];

            // Recursively process properties
            formattedSchema.properties[propName] = limitedSchemaFormat(property);

            // Process arrays with items of type object
            if (property?.type === 'array' && property.items && property.items?.type === 'object') {
                formattedSchema.properties[propName] = {
                    ...property,
                    items: limitedSchemaFormat(property.items),
                };
            }
        }
    }

    return formattedSchema;
}

//For strict mode true
function openAISchemaFormat(schema: JSONSchema, nesting: number = 0): JSONSchema {
    if (nesting > 5) {
        throw new Error("OpenAI schema nesting too deep");
    }

    const formattedSchema = { ...schema };

    // Defaults not supported
    delete formattedSchema.default 

    // Additional properties not supported, required to be set.
    if (formattedSchema?.type === "object") {
        formattedSchema.additionalProperties = false;
    }

    if (formattedSchema?.properties) {
        // Set all properties as required
        formattedSchema.required = Object.keys(formattedSchema.properties);
        
        // Process each property recursively
        for (const propName of Object.keys(formattedSchema.properties)) {
            const property = formattedSchema.properties[propName];
            
            // Recursively process properties
            formattedSchema.properties[propName] = openAISchemaFormat(property, nesting + 1);
            
            // Process arrays with items of type object
            if (property?.type === 'array' && property.items && property.items?.type === 'object') {
                formattedSchema.properties[propName] = {
                    ...property,
                    items: openAISchemaFormat(property.items, nesting + 1),
                };
            }
        }
    }
    if (formattedSchema?.type === 'object' && (!formattedSchema?.properties || Object.keys(formattedSchema?.properties ?? {}).length == 0)) {
        //If no properties are defined, then additionalProperties: true was set or the object would be empty.
        //OpenAI does not support this on structured output/ strict mode.
        throw new Error("OpenAI does not support empty objects or objects with additionalProperties set to true");
    }
    
    return formattedSchema
}