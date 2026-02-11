/**
 * Classes to handle the execution of an interaction in an execution environment.
 * Base abstract class is then implemented by each environment
 * (eg: OpenAI, HuggingFace, etc.)
 */

import {
    AIModel,
    BatchDestination,
    BatchJob,
    BatchSource,
    Completion,
    CompletionChunkObject,
    CompletionStream,
    CreateBatchJobOptions,
    DataSource,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ExecutionResponse,
    ListBatchJobsOptions,
    ListBatchJobsResult,
    Logger,
    Modalities,
    ModelSearchPayload,
    PromptOptions,
    PromptSegment,
    TrainingJob,
    TrainingOptions,
    TrainingPromptOptions
} from "@llumiverse/common";
import { DefaultCompletionStream, FallbackCompletionStream } from "./CompletionStream.js";
import { formatTextPrompt } from "./formatters/index.js";
import { validateResult } from "./validation.js";

// Helper to create logger methods that support both message-only and object-first signatures
function createConsoleLoggerMethod(consoleMethod: (...args: unknown[]) => void): Logger['info'] {
    return ((objOrMsg: any, msgOrNever?: any, ...args: (string | number | boolean)[]) => {
        if (typeof objOrMsg === 'string') {
            // Message-only: logger.info("message", ...args)
            consoleMethod(objOrMsg, msgOrNever, ...args);
        } else if (msgOrNever !== undefined) {
            // Object-first: logger.info({ obj }, "message", ...args)
            consoleMethod(msgOrNever, objOrMsg, ...args);
        } else {
            // Object-only: logger.info({ obj })
            consoleMethod(objOrMsg, ...args);
        }
    }) as Logger['info'];
}

const ConsoleLogger: Logger = {
    debug: createConsoleLoggerMethod(console.debug.bind(console)),
    info: createConsoleLoggerMethod(console.info.bind(console)),
    warn: createConsoleLoggerMethod(console.warn.bind(console)),
    error: createConsoleLoggerMethod(console.error.bind(console)),
}

const noop = () => void 0;
const NoopLogger: Logger = {
    debug: noop as Logger['debug'],
    info: noop as Logger['info'],
    warn: noop as Logger['warn'],
    error: noop as Logger['error'],
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

export interface Driver<PromptT = unknown, BatchSourceT = unknown, BatchDestinationT = unknown> {

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

    //list models available for this environment
    listModels(params?: ModelSearchPayload): Promise<AIModel[]>;

    //list models that can be trained
    listTrainableModels(): Promise<AIModel[]>;

    //check that it is possible to connect to the environment
    validateConnection(): Promise<boolean>;

    //generate embeddings for a given text or image
    generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult>;

    // Batch operations for high-throughput, cost-effective processing
    createBatchJob(options: CreateBatchJobOptions<BatchSourceT, BatchDestinationT>): Promise<BatchJob<BatchSourceT, BatchDestinationT>>;
    getBatchJob(jobId: string): Promise<BatchJob<BatchSourceT, BatchDestinationT>>;
    listBatchJobs(options?: ListBatchJobsOptions): Promise<ListBatchJobsResult<BatchSourceT, BatchDestinationT>>;
    cancelBatchJob(jobId: string): Promise<BatchJob<BatchSourceT, BatchDestinationT>>;
    deleteBatchJob(jobId: string): Promise<void>;

}

/**
 * To be implemented by each driver
 */
export abstract class AbstractDriver<
    OptionsT extends DriverOptions = DriverOptions,
    PromptT = unknown,
    BatchSourceT extends BatchSource = BatchSource,
    BatchDestinationT extends BatchDestination = BatchDestination
> implements Driver<PromptT, BatchSourceT, BatchDestinationT> {
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

    // Default batch implementations - override in drivers that support batch
    createBatchJob(_options: CreateBatchJobOptions<BatchSourceT, BatchDestinationT>): Promise<BatchJob<BatchSourceT, BatchDestinationT>> {
        throw new Error("Batch operations not implemented for this driver.");
    }

    getBatchJob(_jobId: string): Promise<BatchJob<BatchSourceT, BatchDestinationT>> {
        throw new Error("Batch operations not implemented for this driver.");
    }

    listBatchJobs(_options?: ListBatchJobsOptions): Promise<ListBatchJobsResult<BatchSourceT, BatchDestinationT>> {
        throw new Error("Batch operations not implemented for this driver.");
    }

    cancelBatchJob(_jobId: string): Promise<BatchJob<BatchSourceT, BatchDestinationT>> {
        throw new Error("Batch operations not implemented for this driver.");
    }

    deleteBatchJob(_jobId: string): Promise<void> {
        throw new Error("Batch operations not implemented for this driver.");
    }

    validateResult(result: Completion, options: ExecutionOptions) {
        if (!result.tool_use && !result.error && options.result_schema) {
            try {
                result.result = validateResult(result.result, options.result_schema);
            } catch (error: any) {
                const errorMessage = `[${this.provider}] [${options.model}] ${error.code ? '[' + error.code + '] ' : ''}Result validation error: ${error.message}`;
                this.logger.error({ err: error, data: result.result }, errorMessage);
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
        return this._execute(prompt, options).catch((error: any) => {
            (error as any).prompt = prompt;
            throw error;
        });
    }

    async _execute(prompt: PromptT, options: ExecutionOptions): Promise<ExecutionResponse<PromptT>> {
        try {
            const start = Date.now();
            let result;

            if (this.isImageModel(options.model)) {
                this.logger.debug(
                    `[${this.provider}] Executing prompt on ${options.model}, image pathway.`);
                result = await this.requestImageGeneration(prompt, options);
            } else {
                this.logger.debug(
                    `[${this.provider}] Executing prompt on ${options.model}, text pathway.`);
                result = await this.requestTextCompletion(prompt, options);
                this.validateResult(result, options);
            }

            const execution_time = Date.now() - start;
            return { ...result, prompt, execution_time };
        } catch (error) {
            (error as any).prompt = prompt;
            throw error;
        }
    }

    protected isImageModel(_model: string): boolean {
        return false;
    }

    // by default no stream is supported. we block and we return all at once
    async stream(segments: PromptSegment[], options: ExecutionOptions): Promise<CompletionStream<PromptT>> {
        const prompt = await this.createPrompt(segments, options);
        const canStream = await this.canStream(options);
        if (options.output_modality === Modalities.text && canStream) {
            return new DefaultCompletionStream(this, prompt, options);
        } else if (this.isImageModel(options.model)) {
            return new FallbackCompletionStream(this, prompt, options);
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
     * Must be overridden if the implementation cannot stream.
     * Some implementation may be able to stream for certain models but not for others.
     * You must overwrite and return false if the current model doesn't support streaming.
     * The default implementation returns true, so it is assumed that the streaming can be done.
     * If this method returns false then the streaming execution will fallback on a blocking execution streaming the entire response as a single event.
     * @param options the execution options containing the target model name.
     * @returns true if the execution can be streamed false otherwise.
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

    /**
     * Build the conversation context after streaming completion.
     * Override this in driver implementations that support multi-turn conversations.
     *
     * @param prompt - The prompt that was sent (includes prior conversation context)
     * @param result - The completion results from the streamed response
     * @param toolUse - The tool calls from the streamed response (if any)
     * @param options - The execution options
     * @returns The updated conversation context, or undefined if not supported
     */
    buildStreamingConversation(
        _prompt: PromptT,
        _result: unknown[],
        _toolUse: unknown[] | undefined,
        _options: ExecutionOptions
    ): unknown | undefined {
        // Default implementation returns undefined - drivers can override
        return undefined;
    }

    abstract requestTextCompletion(prompt: PromptT, options: ExecutionOptions): Promise<Completion>;

    abstract requestTextCompletionStream(prompt: PromptT, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>>;

    async requestImageGeneration(_prompt: PromptT, _options: ExecutionOptions): Promise<Completion> {
        throw new Error("Image generation not implemented.");
        //Cannot be made abstract, as abstract methods are required in the derived class
    }

    //list models available for this environment
    abstract listModels(params?: ModelSearchPayload): Promise<AIModel[]>;

    //check that it is possible to connect to the environment
    abstract validateConnection(): Promise<boolean>;

    //generate embeddings for a given text
    abstract generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult>;

}
