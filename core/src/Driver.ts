/**
 * Classes to handle the execution of an interaction in an execution environment.
 * Base abstract class is then implemented by each environment
 * (eg: OpenAI, HuggingFace, etc.)
 */

import {
    type AIModel,
    type Completion,
    type CompletionStream,
    type DataSource,
    type DriverCompletionStream,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionResponse,
    LlumiverseError,
    type LlumiverseErrorContext,
    type Logger,
    type ModelSearchPayload,
    type PromptOptions,
    type PromptSegment,
    type Providers,
    type TrainingJob,
    type TrainingOptions,
    type TrainingPromptOptions,
} from '@llumiverse/common';
import type { Agent } from 'undici';
import { DefaultCompletionStream, FallbackCompletionStream } from './CompletionStream.js';
import { formatTextPrompt } from './formatters/index.js';
import {
    createAgentBackedFetch,
    createDriverHttpAgent,
    createDriverHttpAgentScope,
    type DriverHttpAgentScope,
} from './http-agent.js';
import { createLogger } from './logger.js';
import { validateResult } from './validation.js';

export { createLogger } from './logger.js';

function getObjectProperty(value: unknown, key: string): unknown {
    if (value && typeof value === 'object' && key in value) {
        return (value as Record<string, unknown>)[key];
    }
    return undefined;
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

    /**
     * Generate embeddings for one or more inputs.
     * Inputs may be text, image, video, or audio depending on the model and
     * provider. Returns one result item per input, each with one or more
     * output vectors (single-vector for text/image, multi-vector for
     * segmented video/audio or joint-multimodal models).
     */
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
export abstract class AbstractDriver<OptionsT extends DriverOptions = DriverOptions, PromptT = unknown>
    implements Driver<PromptT>
{
    options: OptionsT;
    logger: Logger;

    abstract provider: Providers | string; // the provider name

    private _httpAgent?: Agent;
    private _driverFetch?: typeof fetch;

    constructor(opts: OptionsT) {
        this.options = opts;
        this.logger = createLogger(opts.logger);
    }

    /**
     * Lazily-created undici `Agent` driven by `options.httpTimeout`.
     * Pools sockets for the lifetime of the driver. Subclasses can
     * either pass this directly to an SDK that accepts a `dispatcher`
     * option (rare), or — much more commonly — use {@link getDriverFetch}
     * to get a fetch implementation backed by it.
     *
     * Released via {@link destroy}.
     */
    protected getHttpAgent(): Agent {
        if (!this._httpAgent) {
            this._httpAgent = createDriverHttpAgent(this.options.httpTimeout);
        }
        return this._httpAgent;
    }

    /**
     * Fetch-compatible function backed by the driver's HTTP agent.
     * Pass to any SDK that accepts a custom `fetch` option (OpenAI,
     * Anthropic, `@google/genai`, Bedrock via Smithy, …) or use as a
     * drop-in replacement for the global `fetch` in drivers that make
     * raw HTTP calls.
     */
    protected getDriverFetch(): typeof fetch {
        if (!this._driverFetch) {
            this._driverFetch = createAgentBackedFetch(this.getHttpAgent());
        }
        return this._driverFetch;
    }

    public createExecutionHttpAgentScope(options: Pick<ExecutionOptions, 'httpTimeout'>): DriverHttpAgentScope {
        return createDriverHttpAgentScope(this.options.httpTimeout, options.httpTimeout);
    }

    async createTrainingPrompt(options: TrainingPromptOptions): Promise<string> {
        const prompt = await this.createPrompt(options.segments, {
            result_schema: options.schema,
            model: options.model,
        });
        return JSON.stringify({
            prompt,
            completion:
                typeof options.completion === 'string' ? options.completion : JSON.stringify(options.completion),
        });
    }

    startTraining(_dataset: DataSource, _options: TrainingOptions): Promise<TrainingJob> {
        throw new Error('Method not implemented.');
    }

    cancelTraining(_jobId: string): Promise<TrainingJob> {
        throw new Error('Method not implemented.');
    }

    getTrainingJob(_jobId: string): Promise<TrainingJob> {
        throw new Error('Method not implemented.');
    }

    validateResult(result: Completion, options: ExecutionOptions) {
        if (!result.tool_use && !result.error && options.result_schema) {
            try {
                result.result = validateResult(result.result, options.result_schema);
            } catch (error: unknown) {
                const validationError = error instanceof Error ? error : new Error(String(error));
                const rawCode = getObjectProperty(error, 'code');
                const code = rawCode === 'json_error' || rawCode === 'validation_error' ? rawCode : undefined;
                const errorMessage = `[${this.provider}] [${options.model}] ${code ? `[${code}] ` : ''}Result validation error: ${validationError.message}`;
                this.logger.error({ err: error, data: result.result }, errorMessage);
                result.error = {
                    code: code || 'validation_error',
                    message: validationError.message,
                    data: result.result,
                };
            }
        }
    }

    async execute(segments: PromptSegment[], options: ExecutionOptions): Promise<ExecutionResponse<PromptT>> {
        const prompt = await this.createPrompt(segments, options);
        return this._execute(prompt, options).catch((error: unknown) => {
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
        const httpScope = this.createExecutionHttpAgentScope(options);
        try {
            return await httpScope.run(async () => {
                try {
                    const start = Date.now();
                    let result: Completion;

                    if (this.isImageModel(options.model)) {
                        this.logger.debug(`[${this.provider}] Executing prompt on ${options.model}, image pathway.`);
                        result = await this.requestImageGeneration(prompt, options);
                    } else {
                        this.logger.debug(`[${this.provider}] Executing prompt on ${options.model}, text pathway.`);
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
                    this.logger.error(
                        {
                            err: error,
                            data: { provider: this.provider, model: options.model, operation: 'execute', prompt },
                        },
                        `Error during execution in provider ${this.provider}:`,
                    );
                    throw this.formatLlumiverseError(error, {
                        provider: this.provider,
                        model: options.model,
                        operation: 'execute',
                    });
                }
            });
        } finally {
            await httpScope.close();
        }
    }

    public formatDebugPrompt(prompt: PromptT): PromptT {
        return prompt;
    }

    protected isImageModel(_model: string): boolean {
        return false;
    }

    // by default no stream is supported. we block and we return all at once
    async stream(segments: PromptSegment[], options: ExecutionOptions): Promise<CompletionStream<PromptT>> {
        this.logger.debug(
            options,
            `Executing prompt with provider ${this.provider} with options: ${JSON.stringify(options)}`,
        );
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
        return await (opts.format
            ? (opts.format(segments, opts.result_schema) as PromptT)
            : this.formatPrompt(segments, opts));
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
        _options: ExecutionOptions,
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
     * - High-confidence transient messages containing "rate limit", "timeout", etc.: retryable
     * - Status 4xx (except above and transient provider quirks): not retryable (client errors)
     *
     * @param error - The error to format
     * @param context - Context about where the error occurred
     * @returns A standardized LlumiverseError
     */
    public formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        // Extract status code from common locations (only if numeric)
        let code: number | undefined;
        const rawCode =
            getObjectProperty(error, 'status') ||
            getObjectProperty(error, 'statusCode') ||
            getObjectProperty(error, 'code');

        if (typeof rawCode === 'number') {
            code = rawCode;
        }

        // Extract error name if available
        const rawErrorName = getObjectProperty(error, 'name');
        const errorName = typeof rawErrorName === 'string' ? rawErrorName : undefined;

        // Extract message
        const message = error instanceof Error ? error.message : String(error);

        // Determine retryability
        const retryable = this.isRetryableError(code, message);

        return new LlumiverseError(`[${this.provider}] ${message}`, retryable, context, error, code, errorName);
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
        const lowerMessage = message.toLowerCase();

        // Provider APIs sometimes surface transient failures under misleading
        // client status codes, so high-confidence transient message signals
        // must be honored before the generic 4xx classification below.
        if (lowerMessage.includes('url_rejected-rejected_client_throttled')) return true;
        if (lowerMessage.includes('url_rejected-rejected_rate_limited')) return true;
        if (lowerMessage.includes('rate') && lowerMessage.includes('limit')) return true;
        if (lowerMessage.includes('timeout')) return true;
        if (lowerMessage.includes('timed') && lowerMessage.includes('out')) return true;
        if (lowerMessage.includes('time') && lowerMessage.includes('out')) return true;
        if (lowerMessage.includes('resource') && lowerMessage.includes('exhaust')) return true;
        if (lowerMessage.includes('overload')) return true;
        if (lowerMessage.includes('throttl')) return true;
        if (lowerMessage.includes('429')) return true;
        if (lowerMessage.includes('529')) return true;
        // A transport-level abort (request-timeout / dropped connection) or a
        // deadline-exceeded is transient and should be retried — even when the provider
        // surfaces it under a misleading 4xx status. A deliberate cancellation is raised
        // as a Temporal CancelledFailure (not an LLM error), so it never reaches here.
        if (lowerMessage.includes('aborted')) return true;
        if (lowerMessage.includes('deadline')) return true;

        // Explicit auth failures should never be retried even when they arrive without
        // a numeric status code (for example, google-auth-library invalid_grant errors).
        if (lowerMessage.includes('invalid_grant')) return false;
        if (lowerMessage.includes("credential's issuer")) return false;

        // Numeric status codes
        if (statusCode !== undefined) {
            if (statusCode === 429 || statusCode === 408) return true; // Rate limit, timeout
            if (statusCode === 529) return true; // Overloaded
            if (statusCode >= 500 && statusCode < 600) return true; // Server errors
            return false; // 4xx client errors not retryable
        }

        // Message-based detection for non-HTTP errors
        if (lowerMessage.includes('retry')) return true;

        // Unknown errors - let consumer decide retry strategy
        return undefined;
    }

    abstract requestTextCompletion(prompt: PromptT, options: ExecutionOptions): Promise<Completion>;

    abstract requestTextCompletionStream(prompt: PromptT, options: ExecutionOptions): Promise<DriverCompletionStream>;

    async requestImageGeneration(_prompt: PromptT, _options: ExecutionOptions): Promise<Completion> {
        throw new Error('Image generation not implemented.');
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
     * Releases the lazily-created HTTP agent socket pool. Override this
     * in driver implementations that need to release additional resources
     * — MUST call `super.destroy()` to avoid leaking sockets.
     */
    destroy(): void {
        this._httpAgent?.close().catch(() => {
            /* shutdown best-effort */
        });
        this._httpAgent = undefined;
        this._driverFetch = undefined;
    }
}
