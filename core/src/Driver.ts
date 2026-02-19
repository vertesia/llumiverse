/**
 * Classes to handle the execution of an interaction in an execution environment.
 * Base abstract class is then implemented by each environment
 * (eg: OpenAI, HuggingFace, etc.)
 */

import {
    AIModel,
    Completion,
    CompletionChunkObject,
    CompletionStream,
    DataSource,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ExecutionResponse,
    LlumiverseErrorContext,
    Logger,
    ModelSearchPayload,
    PromptOptions,
    PromptSegment,
    Providers,
    TrainingJob,
    TrainingOptions,
    TrainingPromptOptions
} from "@llumiverse/common";
import { DefaultCompletionStream, FallbackCompletionStream } from "./CompletionStream.js";
import { formatTextPrompt } from "./formatters/index.js";
import { LlumiverseError } from "./LlumiverseError.js";
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

    //list models available for this environment
    listModels(params?: ModelSearchPayload): Promise<AIModel[]>;

    //list models that can be trained
    listTrainableModels(): Promise<AIModel[]>;

    //check that it is possible to connect to the environment
    validateConnection(): Promise<boolean>;

    //generate embeddings for a given text or image
    generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult>;

    /**
     * Optional cleanup method called when the driver is evicted from the cache.
     * Override this in driver implementations that need to release resources.
     */
    destroy?(): void;
}

/**
 * To be implemented by each driver
 */
export abstract class AbstractDriver<OptionsT extends DriverOptions = DriverOptions, PromptT = unknown> implements Driver<PromptT> {
    options: OptionsT;
    logger: Logger;

    abstract provider: Providers | string; // the provider name

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
            // Don't wrap if already a LlumiverseError
            if (LlumiverseError.isLlumiverseError(error)) {
                throw error;
            }
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model: options.model,
                operation: 'execute',
            });
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
            // Don't wrap if already a LlumiverseError
            if (LlumiverseError.isLlumiverseError(error)) {
                throw error;
            }
            // Log the original error for debugging
            this.logger.error({ err: error, data: { provider: this.provider, model: options.model, operation: 'execute', prompt } }, `Error during execution in provider ${this.provider}:`);
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model: options.model,
                operation: 'execute',
            });
        }
    }

    protected isImageModel(_model: string): boolean {
        return false;
    }

    // by default no stream is supported. we block and we return all at once
    async stream(segments: PromptSegment[], options: ExecutionOptions): Promise<CompletionStream<PromptT>> {
        this.logger.info(options, `Executing prompt with provider ${this.provider} with options: ${JSON.stringify(options)}`);
        const prompt = await this.createPrompt(segments, options);
        const canStream = await this.canStream(options);
        if (canStream) {
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

    /**
     * Format an error into LlumiverseError. Override in driver implementations
     * to provide provider-specific error parsing.
     * 
     * The default implementation uses common patterns:
     * - Status 429, 408: retryable (rate limit, timeout)
     * - Status 529: retryable (overloaded)
     * - Status 5xx: retryable (server errors)
     * - Status 4xx (except above): not retryable (client errors)
     * - Error messages containing "rate limit", "timeout", etc.: retryable
     * 
     * @param error - The error to format
     * @param context - Context about where the error occurred
     * @returns A standardized LlumiverseError
     */
    public formatLlumiverseError(
        error: unknown,
        context: LlumiverseErrorContext
    ): LlumiverseError {
        // Extract status code from common locations (only if numeric)
        let code: number | undefined;
        const rawCode = (error as any)?.status
            || (error as any)?.statusCode
            || (error as any)?.code;

        if (typeof rawCode === 'number') {
            code = rawCode;
        }

        // Extract error name if available
        const errorName = (error as any)?.name;

        // Extract message
        const message = error instanceof Error
            ? error.message
            : String(error);

        // Determine retryability
        const retryable = this.isRetryableError(code, message);

        return new LlumiverseError(
            `[${this.provider}] ${message}`,
            retryable,
            context,
            error,
            code,
            errorName
        );
    }

    /**
     * Determine if an error is retryable based on status code and message.
     * Can be overridden by drivers for provider-specific logic.
     * 
     * @param statusCode - The HTTP status code (if available)
     * @param message - The error message
     * @returns True if retryable, false if not retryable, undefined if unknown
     */
    protected isRetryableError(statusCode: number | undefined, message: string): boolean | undefined {
        // Numeric status codes
        if (statusCode !== undefined) {
            if (statusCode === 429 || statusCode === 408) return true; // Rate limit, timeout
            if (statusCode === 529) return true; // Overloaded
            if (statusCode >= 500 && statusCode < 600) return true; // Server errors
            return false; // 4xx client errors not retryable
        }

        // Message-based detection for non-HTTP errors
        const lowerMessage = message.toLowerCase();

        // Rate limit variations
        if (lowerMessage.includes('rate') && lowerMessage.includes('limit')) return true;

        // Timeout variations (timeout, timed out, time out)
        if (lowerMessage.includes('timeout')) return true;
        if (lowerMessage.includes('timed') && lowerMessage.includes('out')) return true;
        if (lowerMessage.includes('time') && lowerMessage.includes('out')) return true;

        // Resource exhausted variations
        if (lowerMessage.includes('resource') && lowerMessage.includes('exhaust')) return true;

        // Other retryable patterns
        if (lowerMessage.includes('retry')) return true;
        if (lowerMessage.includes('overload')) return true;
        if (lowerMessage.includes('throttl')) return true;
        if (lowerMessage.includes('429')) return true;
        if (lowerMessage.includes('529')) return true;

        // Unknown errors - let consumer decide retry strategy
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

    /**
     * Cleanup method called when the driver is evicted from the cache.
     * Override this in driver implementations that need to release resources.
     */
    destroy(): void {
        // No-op by default
    }
}
