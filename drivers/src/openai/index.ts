import {
    AIModel,
    AbstractDriver,
    Completion,
    CompletionChunkObject,
    DataSource,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionTokenUsage,
    ModelType,
    ExecutionOptions,
    TrainingJob,
    TrainingJobStatus,
    TrainingOptions,
    TrainingPromptOptions,
    TextFallbackOptions,
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { formatOpenAILikeMultimodalPrompt } from "@llumiverse/core/formatters";
import OpenAI, { AzureOpenAI } from "openai";
import { Stream } from "openai/streaming";

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
    OpenAI.Chat.Completions.ChatCompletionMessageParam[]
> {
    abstract provider: "azure_openai" | "openai" | "xai";
    abstract service: OpenAI | AzureOpenAI;

    constructor(opts: BaseOpenAIDriverOptions) {
        super(opts);
        this.formatPrompt = formatOpenAILikeMultimodalPrompt as any //TODO: better type, we send back OpenAI.Chat.Completions.ChatCompletionMessageParam[] but just not compatbile with Function call that we don't use here

    }

    extractDataFromResponse(
        options: ExecutionOptions,
        result: OpenAI.Chat.Completions.ChatCompletion
    ): Completion {
        const tokenInfo: ExecutionTokenUsage = {
            prompt: result.usage?.prompt_tokens,
            result: result.usage?.completion_tokens,
            total: result.usage?.total_tokens,
        };

        const choice = result.choices[0];

        //if no schema, return content
        if (!options.result_schema) {
            return {
                result: choice.message.content ?? undefined,
                token_usage: tokenInfo,
                finish_reason: choice.finish_reason, //Uses expected "stop" , "length" format
            }
        }

        const useTools: boolean = !isNonStructureSupporting(options.model);
        let data = undefined;
        if (useTools) {
            //we have a schema: get the content and return after validation
            data = choice?.message.tool_calls?.[0].function.arguments ?? choice.message.content ?? undefined;
        } else {
            data = choice.message.content ?? undefined;
        }

        if (!data) {
            this.logger?.error("[OpenAI] Response is not valid", result);
            throw new Error("Response is not valid: no data");
        }

        return {
            result: data,
            token_usage: tokenInfo,
            finish_reason: choice.finish_reason,
        };
    }

    async requestTextCompletionStream(prompt: OpenAI.Chat.Completions.ChatCompletionMessageParam[], options: ExecutionOptions): Promise<any> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn("Invalid model options", options.model_options);
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const useTools: boolean = !isNonStructureSupporting(options.model);

        const mapFn = (chunk: OpenAI.Chat.Completions.ChatCompletionChunk) => {
            let result = undefined
            if (useTools && this.provider !== "xai" && options.result_schema) {
                result = chunk.choices[0]?.delta?.tool_calls?.[0].function?.arguments ?? "";
            } else {
                result = chunk.choices[0]?.delta.content ?? "";
            }

            return {
                result: result,
                finish_reason: chunk.choices[0]?.finish_reason,         //Uses expected "stop" , "length" format
                token_usage: {
                    prompt: chunk.usage?.prompt_tokens,
                    result: chunk.usage?.completion_tokens,
                    total: (chunk.usage?.prompt_tokens ?? 0) + (chunk.usage?.completion_tokens ?? 0),
                }
            } as CompletionChunkObject;
        };
        
        convertRoles(prompt, options.model);

        //TODO: OpenAI o1 support requires max_completions_tokens
        const stream = (await this.service.chat.completions.create({
            stream: true,
            stream_options: { include_usage: true },
            model: options.model,
            messages: prompt,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            n: 1,
            max_completion_tokens: options.model_options?.max_tokens, //TODO: use max_tokens for older models, currently relying on OpenAI to handle it
            tools: useTools ? options.result_schema && this.provider.includes("openai")
                ? [
                    {
                        function: {
                            name: "format_output",
                            parameters: options.result_schema as any,
                        },
                        type: "function"
                    } as OpenAI.Chat.ChatCompletionTool,
                ]
                : undefined : undefined,
            tool_choice: useTools ? options.result_schema
                ? {
                    type: 'function',
                    function: { name: "format_output" }
                } : undefined : undefined,
        })) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>;

        return asyncMap(stream, mapFn);
    }

    async requestTextCompletion(prompt: OpenAI.Chat.Completions.ChatCompletionMessageParam[], options: ExecutionOptions): Promise<any> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn("Invalid model options", options.model_options);
        }
        options.model_options = options.model_options as TextFallbackOptions;
        const functions = options.result_schema && this.provider.includes("openai")
            ? [
                {
                    function: {
                        name: "format_output",
                        parameters: options.result_schema as any,
                    },
                    type: 'function'
                } as OpenAI.Chat.ChatCompletionTool,
            ]
            : undefined;
        
        convertRoles(prompt, options.model);

        const useTools: boolean = !isNonStructureSupporting(options.model);

        //TODO: OpenAI o1 support requires max_completions_tokens
        const res = await this.service.chat.completions.create({
            stream: false,
            model: options.model,
            messages: prompt,
            temperature: options.model_options?.temperature,
            top_p: options.model_options?.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.model_options?.presence_penalty,
            frequency_penalty: options.model_options?.frequency_penalty,
            n: 1,
            max_completion_tokens: options.model_options.max_tokens, //TODO: use max_tokens for older models, currently relying on OpenAI to handle it
            tools: useTools ? functions : undefined,
            tool_choice: useTools ? options.result_schema && this.provider.includes("openai")
                ? {
                    type: 'function',
                    function: { name: "format_output" }
                } : undefined : undefined,
            // functions: functions,
            // function_call: options.result_schema
            //     ? { name: "format_output" }
            //     : undefined,
        });

        const completion = this.extractDataFromResponse(options, res);
        if (options.include_original_response) {
            completion.original_response = res;
        }
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
        let result = await this.service.models.list();
        const models = filter ? result.data.filter(filter) : result.data;
        return models.map((m) => ({
            id: m.id,
            name: m.id,
            provider: this.provider,
            owner: m.owned_by,
            type: m.object === "model" ? ModelType.Text : ModelType.Unknown,
            can_stream: true,
            is_multimodal: m.id.includes("gpt-4")
        }));
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

        return { values: embeddings, model } as EmbeddingsResult;
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

function convertRoles(messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[], model: string): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
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

function isNonStructureSupporting(model: string): boolean {
    return model.includes("o1-mini") || model.includes("o1-preview")
        || model.includes("chatgpt-4o");
}