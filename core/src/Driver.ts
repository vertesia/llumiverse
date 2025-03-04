/**
 * Classes to handle the execution of an interaction in an execution environment.
 * Base abstract class is then implemented by each environment
 * (eg: OpenAI, HuggingFace, etc.)
 */

import { DefaultCompletionStream, FallbackCompletionStream } from "./CompletionStream.js";
import { formatTextPrompt } from "./formatters/index.js";
import {
    AIModel,
    Completion,
    CompletionChunk,
    CompletionStream,
    DataSource,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ExecutionResponse,
    ImageGeneration,
    Logger,
    Modalities,
    ModelSearchPayload,
    PromptOptions,
    PromptSegment,
    TrainingJob,
    TrainingOptions,
    TrainingPromptOptions
} from "./types.js";
import { validateResult } from "./validation.js";

const ConsoleLogger: Logger = {
    debug: console.debug,
    info: console.info,
    warn: console.warn,
    error: console.error,
}

const noop = () => void 0;
const NoopLogger: Logger = {
    debug: noop,
    info: noop,
    warn: noop,
    error: noop,
}

export function createLogger(logger: Logger | "console" | undefined) {
    if (logger === "console") {
        return ConsoleLogger;
    } else if (logger) {
        return logger;
    } else {
        return NoopLogger;
    }
}

export interface Driver<PromptT = unknown> {

    /**
     *
     * @param segments
     * @param completion
     * @param model the model to train
     */
    createTrainingPrompt(options: TrainingPromptOptions): Promise<string>;

    createPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<PromptT>;

    execute(segments: PromptSegment[], options: ExecutionOptions): Promise<ExecutionResponse<PromptT>>;

    // by default no stream is supported. we block and we return all at once
    //stream(segments: PromptSegment[], options: ExecutionOptions): Promise<StreamingExecutionResponse<PromptT>>;
    stream(segments: PromptSegment[], options: ExecutionOptions): Promise<CompletionStream<PromptT>>;

    startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob>;

    cancelTraining(jobId: string): Promise<TrainingJob>;

    getTrainingJob(jobId: string): Promise<TrainingJob>;

    //list models available for this environement
    listModels(params?: ModelSearchPayload): Promise<AIModel[]>;

    //list models that can be trained
    listTrainableModels(): Promise<AIModel[]>;

    //check that it is possible to connect to the environment
    validateConnection(): Promise<boolean>;

    //generate embeddings for a given text or image
    generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult>;

}

/**
 * To be implemented by each driver
 */
export abstract class AbstractDriver<OptionsT extends DriverOptions = DriverOptions, PromptT = unknown> implements Driver<PromptT> {
    options: OptionsT;
    logger: Logger;

    abstract provider: string; // the provider name

    constructor(opts: OptionsT) {
        this.options = opts;
        this.logger = createLogger(opts.logger);
    }

    async createTrainingPrompt(options: TrainingPromptOptions): Promise<string> {
        const prompt = await this.createPrompt(options.segments, { result_schema: options.schema, model: options.model })
        return JSON.stringify({
            prompt,
            completion: typeof options.completion === 'string' ? options.completion : JSON.stringify(options.completion)
        });
    }

    startTraining(_dataset: DataSource, _options: TrainingOptions): Promise<TrainingJob> {
        throw new Error("Method not implemented.");
    }

    cancelTraining(_jobId: string): Promise<TrainingJob> {
        throw new Error("Method not implemented.");
    }

    getTrainingJob(_jobId: string): Promise<TrainingJob> {
        throw new Error("Method not implemented.");
    }

    validateResult(result: Completion, options: ExecutionOptions) {
        if (!result.error && options.result_schema) {
            try {
                result.result = validateResult(result.result, options.result_schema);
            } catch (error: any) {
                this.logger?.error({ err: error, data: result.result }, `[${this.provider}] [${options.model}] ${error.code ? '[' + error.code + '] ' : ''}Result validation error: ${error.message}`);
                result.error = {
                    code: error.code || error.name,
                    message: error.message,
                    data: result.result,
                }
            }
        }
    }

    async execute(segments: PromptSegment[], options: ExecutionOptions): Promise<ExecutionResponse<PromptT>> {
        const prompt = await this.createPrompt(segments, options);
        return this._execute(prompt, options);
    }

    async _execute(prompt: PromptT, options: ExecutionOptions): Promise<ExecutionResponse<PromptT>> {
        this.logger.debug(
            `[${this.provider}] Executing prompt on ${options.model}`);
        try {
            const start = Date.now();
            let result;

            switch (options.output_modality) {
                case Modalities.text:
                    result = await this.requestTextCompletion(prompt, options);
                    this.validateResult(result, options);
                    break;
                case Modalities.image:
                    result = await this.requestImageGeneration(prompt, options);
                    break;
                default:
                    throw new Error(`Unsupported modality: ${options['output_modality'] ?? "No modality specified"}`);
            }

            const execution_time = Date.now() - start;
            return { ...result, prompt, execution_time };
        } catch (error) {
            (error as any).prompt = prompt;
            throw error;
        }
    }

    // by default no stream is supported. we block and we return all at once
    async stream(segments: PromptSegment[], options: ExecutionOptions): Promise<CompletionStream<PromptT>> {
        const prompt = await this.createPrompt(segments, options);
        const canStream = await this.canStream(options);
        if (options.output_modality === Modalities.text && canStream) {
            return new DefaultCompletionStream(this, prompt, options);
        } else {
            return new FallbackCompletionStream(this, prompt, options);
        }
    }

    /**
     * Override this method to provide a custom prompt formatter
     * @param segments
     * @param options
     * @returns
     */
    protected async formatPrompt(segments: PromptSegment[], opts: PromptOptions): Promise<PromptT> {
        return formatTextPrompt(segments, opts.result_schema) as PromptT;
    }

    public async createPrompt(segments: PromptSegment[], opts: PromptOptions): Promise<PromptT> {
        return await (opts.format ? opts.format(segments, opts.result_schema) : this.formatPrompt(segments, opts));
    }

    /**
     * Must be overrided if the implementation cannot stream.
     * Some implementation may be able to stream for certain models but not for others.
     * You must overwrite and return false if the current model doesn't support streaming.
     * The default implementation returns true, so it is assumed that the streaming can be done.
     * If this method returns false then the streaming execution will fallback on a blocking execution streaming the entire response as a single event.
     * @param options the execution options containing the target model name.
     * @returns true if the exeuction can be streamed false otherwise.
     */
    protected canStream(_options: ExecutionOptions) {
        return Promise.resolve(true);
    }

    /**
     * Get a list of models that can be trained.
     * The default is to return an empty array
     * @returns
     */
    async listTrainableModels(): Promise<AIModel[]> {
        return [];
    }

    abstract requestTextCompletion(prompt: PromptT, options: ExecutionOptions): Promise<Completion>;

    abstract requestTextCompletionStream(prompt: PromptT, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunk>>;

    async requestImageGeneration(_prompt: PromptT, _options: ExecutionOptions): Promise<Completion<ImageGeneration>> {
        throw new Error("Image generation not implemented.");
        //Cannot be made abstract, as abstract methods are required in the derived class
    }

    //list models available for this environement
    abstract listModels(params?: ModelSearchPayload): Promise<AIModel[]>;

    //check that it is possible to connect to the environment
    abstract validateConnection(): Promise<boolean>;

    //generate embeddings for a given text
    abstract generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult>;

}
