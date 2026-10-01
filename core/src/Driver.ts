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
    type HttpTimeoutOptions,
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
import { deriveConversationId, isConversationDocumentFormat } from '@llumiverse/conversation';
import type { Agent } from 'undici';
import {
    CanonicalAcceptedOutputRecovered,
    type CanonicalExecutionContextInputOptions,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionInputOptions,
    type CanonicalExecutionOptions,
    type CanonicalExecutionResponse,
    type CanonicalExecutionStream,
    createCanonicalExecutionResponse,
    legacyCompletionFromCanonicalExecution,
    resolveCanonicalExecutionContextOptions,
    resolveCanonicalExecutionOptions,
} from './CanonicalExecution.js';
import {
    type CanonicalExecutionEventStream,
    type CanonicalStreamOpenOptions,
    FallbackCanonicalExecutionEventStream,
    LegacyCanonicalExecutionEventProjection,
} from './CanonicalStreaming.js';
import {
    DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS,
    DefaultCompletionStream,
    FallbackCompletionStream,
    leaseCanonicalExecutionEventStream,
    leaseCanonicalExecutionStream,
    leaseCompletionStream,
} from './CompletionStream.js';
import { stripAudioFromCompletion, stripAudioPayloads } from './conversation-utils.js';
import { formatTextPrompt } from './formatters/index.js';
import {
    createAgentBackedFetch,
    createDriverHttpAgent,
    createDriverHttpAgentScope,
    type DriverHttpAgentScope,
    resolveDriverRequestTimeoutMs,
} from './http-agent.js';
import { createLogger } from './logger.js';
import { normalizeCompletionResult } from './validation.js';

export { createLogger } from './logger.js';

function getObjectProperty(value: unknown, key: string): unknown {
    if (value && typeof value === 'object' && key in value) {
        return (value as Record<string, unknown>)[key];
    }
    return undefined;
}

// Nominal lifecycle contract: subclasses inherit this through AbstractDriver, while unrelated
// structural implementations are excluded so resource-owning drivers cannot bypass its guards.
const driverLifecycleBrand: unique symbol = Symbol('driverLifecycle');

export interface Driver<PromptT = unknown> {
    readonly [driverLifecycleBrand]: true;
    /**
     *
     * @param segments
     * @param completion
     * @param model the model to train
     */
    createTrainingPrompt(options: TrainingPromptOptions): Promise<string>;

    createPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<PromptT>;

    /** Supported legacy Completion/ExecutionTokenUsage projection boundary. */
    execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<PromptT>>;

    executeCanonical(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse>;

    /** Current canonical execution over an already materialized document; no prompt authoring occurs. */
    executeCanonicalContext(
        options: CanonicalExecutionContextInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse>;

    /** Supported legacy string/CompletionResult stream boundary. */
    stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<PromptT>>;

    streamCanonical(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionStream>;

    /** Legacy string preview projected from the current typed canonical context stream. */
    streamCanonicalContext(
        options: CanonicalExecutionContextInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionStream>;

    streamCanonicalEvents(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream>;

    streamCanonicalContextEvents(
        options: CanonicalExecutionContextInputOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream>;

    /** Report whether this concrete provider/model path can produce canonical execution authority. */
    supportsCanonicalExecution(options: ExecutionOptions): Promise<boolean>;

    /** Report whether this path compiles an already materialized canonical context without prompt authoring. */
    supportsCanonicalContextExecution(options: CanonicalExecutionContextInputOptions): Promise<boolean>;

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

    /** Request cleanup after all active operations and owned streams finish. */
    destroy(): void;
}

type AsyncDriverOperationName = {
    [Name in keyof Driver]-?: Driver[Name] extends (...args: never[]) => Promise<unknown> ? Name : never;
}[keyof Driver];

type LifecycleGuardedOperationName = AsyncDriverOperationName;

const lifecycleGuardedOperationNames = [
    'execute',
    'executeCanonical',
    'executeCanonicalContext',
    'stream',
    'streamCanonical',
    'streamCanonicalContext',
    'streamCanonicalEvents',
    'streamCanonicalContextEvents',
    'supportsCanonicalExecution',
    'supportsCanonicalContextExecution',
    'startTraining',
    'cancelTraining',
    'getTrainingJob',
    'listModels',
    'listTrainableModels',
    'validateConnection',
    'generateEmbeddings',
    'createPrompt',
    'createTrainingPrompt',
] as const satisfies readonly LifecycleGuardedOperationName[];

type MissingLifecycleGuard = Exclude<LifecycleGuardedOperationName, (typeof lifecycleGuardedOperationNames)[number]>;
const lifecycleGuardsAreExhaustive: MissingLifecycleGuard extends never ? true : never = true;

/**
 * To be implemented by each driver
 */
export abstract class AbstractDriver<OptionsT extends DriverOptions = DriverOptions, PromptT = unknown>
    implements Driver<PromptT>
{
    readonly [driverLifecycleBrand] = true;
    options: OptionsT;
    logger: Logger;

    abstract provider: Providers | string; // the provider name

    private _httpAgent?: Agent;
    private _driverFetch?: typeof fetch;
    private _activeOperations = 0;
    private _destroyRequested = false;
    private _resourcesDestroyed = false;

    constructor(opts: OptionsT) {
        if (
            opts.streamStartTimeoutMs !== undefined &&
            (!Number.isSafeInteger(opts.streamStartTimeoutMs) ||
                opts.streamStartTimeoutMs <= 0 ||
                opts.streamStartTimeoutMs > 2_147_483_647)
        ) {
            throw new RangeError('streamStartTimeoutMs must be a positive integer no greater than 2147483647');
        }
        this.options = opts;
        this.logger = createLogger(opts.logger);
        this.installOperationGuards();
    }

    /** Whether this concrete model path has adopted canonical conversation input. */
    protected supportsCanonicalConversation(_options: ExecutionOptions): boolean {
        return false;
    }

    /** Whether this model path can compile an already materialized canonical document directly. */
    protected supportsCanonicalContextConversation(_options: CanonicalExecutionContextOptions): boolean {
        return false;
    }

    /** Whether this concrete image model path can produce canonical execution authority directly. */
    protected supportsCanonicalImageGeneration(_options: ExecutionOptions): boolean {
        return false;
    }

    async supportsCanonicalExecution(options: ExecutionOptions): Promise<boolean> {
        return this.isImageModel(options.model)
            ? this.supportsCanonicalImageGeneration(options)
            : this.supportsCanonicalConversation(options);
    }

    async supportsCanonicalContextExecution(options: CanonicalExecutionContextInputOptions): Promise<boolean> {
        const canonicalOptions = options as CanonicalExecutionContextOptions;
        return (
            !this.isImageModel(canonicalOptions.model) && this.supportsCanonicalContextConversation(canonicalOptions)
        );
    }

    /** Validate raw media-generation input before prompt formatting can read or discard unsupported content. */
    protected validateCanonicalImageInput(_segments: PromptSegment[], _options: ExecutionOptions): void {}

    private assertConversationInputSupported(options: ExecutionOptions): void {
        if (isConversationDocumentFormat(options.conversation) && !this.supportsCanonicalConversation(options)) {
            throw new Error(
                `Provider ${this.provider} model ${options.model} does not support canonical conversation input`,
            );
        }
    }

    /**
     * Guard provider operations implemented by subclasses without changing the public
     * driver API. Capturing the prototype methods here keeps lifecycle accounting in
     * one place, so new cache users cannot accidentally call an unleased network path.
     */
    private installOperationGuards(): void {
        void lifecycleGuardsAreExhaustive;
        for (const name of lifecycleGuardedOperationNames) {
            const installedOperation = this[name] as (...args: unknown[]) => Promise<unknown>;
            // Canonical string streaming is an explicit compatibility projection of the typed event stream.
            // Ignore provider overrides left during migration so a current canonical call can never select the
            // CompletionChunkObject/CompletionResult accumulator as its operational authority.
            const operation =
                name === 'streamCanonical'
                    ? (AbstractDriver.prototype.streamCanonical as (...args: unknown[]) => Promise<unknown>)
                    : installedOperation;
            const invoke = (args: unknown[]) => {
                const segmentCanonicalOperation =
                    name === 'executeCanonical' || name === 'streamCanonical' || name === 'streamCanonicalEvents';
                const contextCanonicalOperation =
                    name === 'executeCanonicalContext' ||
                    name === 'streamCanonicalContext' ||
                    name === 'streamCanonicalContextEvents' ||
                    name === 'supportsCanonicalContextExecution';
                const operationArgs = segmentCanonicalOperation
                    ? [
                          args[0],
                          resolveCanonicalExecutionOptions(args[1] as CanonicalExecutionInputOptions),
                          ...args.slice(2),
                      ]
                    : contextCanonicalOperation
                      ? [
                            resolveCanonicalExecutionContextOptions(args[0] as CanonicalExecutionContextInputOptions),
                            ...args.slice(1),
                        ]
                      : args;
                return operation.apply(this, operationArgs);
            };
            Object.defineProperty(this, name, {
                configurable: false,
                value:
                    name === 'stream' ||
                    name === 'streamCanonical' ||
                    name === 'streamCanonicalContext' ||
                    name === 'streamCanonicalEvents' ||
                    name === 'streamCanonicalContextEvents'
                        ? (...args: unknown[]) =>
                              name === 'stream'
                                  ? this.runStreamOperation(
                                        () => invoke(args) as Promise<CompletionStream<PromptT>>,
                                        args[2] as AbortSignal | undefined,
                                    )
                                  : name === 'streamCanonical' || name === 'streamCanonicalContext'
                                    ? this.runCanonicalStreamOperation(
                                          () => invoke(args) as Promise<CanonicalExecutionStream>,
                                          args[name === 'streamCanonical' ? 2 : 1] as AbortSignal | undefined,
                                      )
                                    : this.runCanonicalEventStreamOperation(
                                          () => invoke(args) as Promise<CanonicalExecutionEventStream>,
                                          args[name === 'streamCanonicalEvents' ? 2 : 1] as AbortSignal | undefined,
                                      )
                        : (...args: unknown[]) => this.runOperation(() => invoke(args)),
                writable: false,
            });
        }
    }

    /** Provider SDKs without a fetch hook can scope their global HTTP requests here. */
    protected runInHttpContext<T>(operation: () => T): T {
        return operation();
    }

    private async runOperation<T>(operation: () => Promise<T>): Promise<T> {
        const release = this.acquireOperationLease();
        try {
            return await this.runInHttpContext(operation);
        } finally {
            release();
        }
    }

    private async runStreamOperation(operation: () => Promise<CompletionStream<PromptT>>, signal?: AbortSignal) {
        const release = this.acquireOperationLease();
        try {
            const stream = await this.runInHttpContext(operation);
            return leaseCompletionStream(
                stream,
                release,
                this.options.streamStartTimeoutMs ?? DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS,
                signal,
            );
        } catch (error: unknown) {
            release();
            throw error;
        }
    }

    private async runCanonicalStreamOperation(
        operation: () => Promise<CanonicalExecutionStream>,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionStream> {
        const release = this.acquireOperationLease();
        try {
            const stream = await this.runInHttpContext(operation);
            return leaseCanonicalExecutionStream(
                stream,
                release,
                this.options.streamStartTimeoutMs ?? DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS,
                signal,
            );
        } catch (error: unknown) {
            release();
            throw error;
        }
    }

    private async runCanonicalEventStreamOperation(
        operation: () => Promise<CanonicalExecutionEventStream>,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionEventStream> {
        const release = this.acquireOperationLease();
        try {
            const stream = await this.runInHttpContext(operation);
            return leaseCanonicalExecutionEventStream(
                stream,
                release,
                this.options.streamStartTimeoutMs ?? DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS,
                signal,
            );
        } catch (error: unknown) {
            release();
            throw error;
        }
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

    /**
     * Resolve the single request deadline expected by SDKs that do not expose
     * separate response-header and streaming-body inactivity timeouts.
     */
    protected getDriverRequestTimeoutMs(httpTimeout?: HttpTimeoutOptions): number {
        return resolveDriverRequestTimeoutMs(this.options.httpTimeout, httpTimeout);
    }

    protected getDriverRequestOptions(
        options: Pick<ExecutionOptions, 'httpTimeout'>,
        signal?: AbortSignal,
    ): { signal?: AbortSignal; timeout?: number } | undefined {
        const timeout = options.httpTimeout ? this.getDriverRequestTimeoutMs(options.httpTimeout) : undefined;
        if (signal && timeout !== undefined) return { signal, timeout };
        if (signal) return { signal };
        if (timeout !== undefined) return { timeout };
        return undefined;
    }

    public createExecutionHttpAgentScope(
        options: Pick<ExecutionOptions, 'httpTimeout'>,
        force = false,
    ): DriverHttpAgentScope {
        const scope = createDriverHttpAgentScope(this.options.httpTimeout, options.httpTimeout, force);
        return {
            ...scope,
            run: <T>(callback: () => T): T => this.runInHttpContext(() => scope.run(callback)),
        };
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
            const normalized = normalizeCompletionResult(result.result, options.result_schema);
            if (normalized.status === 'valid') {
                result.result = normalized.result;
            } else {
                const validationError = normalized.error;
                const rawCode = getObjectProperty(validationError, 'code');
                const code = rawCode === 'json_error' || rawCode === 'validation_error' ? rawCode : undefined;
                const errorMessage = `[${this.provider}] [${options.model}] ${code ? `[${code}] ` : ''}Result validation error: ${validationError.message}`;
                this.logger.error({ err: validationError, data: result.result }, errorMessage);
                result.error = {
                    code: code || 'validation_error',
                    message: validationError.message,
                    data: result.result,
                };
            }
        }
    }

    async execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<PromptT>> {
        this.assertConversationInputSupported(options);
        if (this.isImageModel(options.model) && this.supportsCanonicalImageGeneration(options)) {
            this.validateCanonicalImageInput(segments, options);
        }
        const prompt = await this.createPrompt(segments, options);
        return await this._execute(prompt, options, signal).catch((error: unknown) => {
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

    async executeCanonical(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const canonicalOptions = options as CanonicalExecutionOptions;
        this.assertConversationInputSupported(canonicalOptions);
        if (canonicalOptions.conversation_runtime.materialized_input !== undefined && segments.length > 0) {
            throw new Error('A materialized canonical input requires an empty new prompt');
        }
        if (this.isImageModel(canonicalOptions.model)) {
            if (!this.supportsCanonicalImageGeneration(canonicalOptions)) {
                throw new Error(
                    `Provider ${this.provider} model ${canonicalOptions.model} does not support canonical execution`,
                );
            }
            this.validateCanonicalImageInput(segments, canonicalOptions);
        }
        const prompt = await this.createPrompt(segments, canonicalOptions);
        return await this._executeCanonical(prompt, canonicalOptions, signal).catch((error: unknown) => {
            if (CanonicalAcceptedOutputRecovered.is(error)) throw error;
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model: canonicalOptions.model,
                operation: 'execute',
            });
        });
    }

    async executeCanonicalContext(
        options: CanonicalExecutionContextInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const canonicalOptions = options as CanonicalExecutionContextOptions;
        if (!this.supportsCanonicalContextConversation(canonicalOptions) || this.isImageModel(canonicalOptions.model)) {
            throw new Error(
                `Provider ${this.provider} model ${canonicalOptions.model} does not support canonical context execution`,
            );
        }
        return await this._executeCanonicalContext(canonicalOptions, signal).catch((error: unknown) => {
            if (CanonicalAcceptedOutputRecovered.is(error)) throw error;
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model: canonicalOptions.model,
                operation: 'execute',
            });
        });
    }

    async _executeCanonicalContext(
        options: CanonicalExecutionContextOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        const httpScope = this.createExecutionHttpAgentScope(options, signal !== undefined);
        const abort = () => void httpScope.abort();
        if (signal?.aborted) abort();
        else signal?.addEventListener('abort', abort, { once: true });
        const start = Date.now();
        try {
            return await httpScope.run(async () => {
                const response = await this.requestCanonicalContextCompletion(options, signal);
                return response.execution_time === undefined
                    ? { ...response, execution_time: Date.now() - start }
                    : response;
            });
        } finally {
            signal?.removeEventListener('abort', abort);
            await httpScope.close();
        }
    }

    async _executeCanonical(
        prompt: PromptT,
        options: CanonicalExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        this.assertConversationInputSupported(options);
        const httpScope = this.createExecutionHttpAgentScope(options, signal !== undefined);
        const abort = () => void httpScope.abort();
        if (signal?.aborted) abort();
        else signal?.addEventListener('abort', abort, { once: true });
        const start = Date.now();
        try {
            return await httpScope.run(async () => {
                const response = this.isImageModel(options.model)
                    ? await this.requestCanonicalImageGeneration(prompt, options, signal)
                    : await this.requestCanonicalTextCompletion(prompt, options, signal);
                return response.execution_time === undefined
                    ? { ...response, execution_time: Date.now() - start }
                    : response;
            });
        } finally {
            signal?.removeEventListener('abort', abort);
            await httpScope.close();
        }
    }

    async _execute(
        prompt: PromptT,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<PromptT>> {
        this.assertConversationInputSupported(options);
        const httpScope = this.createExecutionHttpAgentScope(options, signal !== undefined);
        const abort = () => void httpScope.abort();
        if (signal?.aborted) abort();
        else signal?.addEventListener('abort', abort, { once: true });
        try {
            return await httpScope.run(async () => {
                try {
                    const start = Date.now();
                    let result: Completion;

                    if (this.isImageModel(options.model)) {
                        this.logger.debug(`[${this.provider}] Executing prompt on ${options.model}, image pathway.`);
                        result = this.supportsCanonicalImageGeneration(options)
                            ? legacyCompletionFromCanonicalExecution(
                                  await this.requestCanonicalImageGeneration(prompt, options, signal),
                              )
                            : await this.requestImageGeneration(prompt, options, signal);
                    } else {
                        this.logger.debug(`[${this.provider}] Executing prompt on ${options.model}, text pathway.`);
                        result = await this.requestTextCompletion(prompt, options, signal);
                        this.validateResult(result, options);
                    }

                    const execution_time = Date.now() - start;
                    return stripAudioFromCompletion({ ...result, prompt, execution_time });
                } catch (error) {
                    // Don't wrap if already a LlumiverseError
                    if (LlumiverseError.isLlumiverseError(error)) {
                        throw error;
                    }
                    // Log the original error for debugging
                    this.logger.error(
                        {
                            err: error,
                            data: {
                                provider: this.provider,
                                model: options.model,
                                operation: 'execute',
                                prompt: stripAudioPayloads(prompt),
                            },
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
            signal?.removeEventListener('abort', abort);
            await httpScope.close();
        }
    }

    public formatDebugPrompt(prompt: PromptT): PromptT {
        return prompt;
    }

    protected isImageModel(_model: string): boolean {
        return false;
    }

    /** Supported legacy string/CompletionResult stream boundary. Canonical callers use streamCanonicalEvents. */
    async stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<PromptT>> {
        this.assertConversationInputSupported(options);
        signal?.throwIfAborted();
        if (this.isImageModel(options.model) && this.supportsCanonicalImageGeneration(options)) {
            this.validateCanonicalImageInput(segments, options);
        }
        this.logger.debug(
            options,
            `Executing prompt with provider ${this.provider} with options: ${JSON.stringify(options)}`,
        );
        const prompt = await this.createPrompt(segments, options);
        signal?.throwIfAborted();
        if (await this.canStream(options, signal)) {
            signal?.throwIfAborted();
            return new DefaultCompletionStream(this, prompt, options);
        }
        signal?.throwIfAborted();
        return new FallbackCompletionStream(this, prompt, options);
    }

    async streamCanonical(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionStream> {
        const canonicalOptions = options as CanonicalExecutionOptions;
        const source = await this.streamCanonicalEvents(segments, canonicalOptions, signal, {
            stream_id: canonicalOptions.conversation_runtime.response_operation_id,
        });
        return new LegacyCanonicalExecutionEventProjection(
            source,
            (canonicalOptions.model_options as { include_thoughts?: unknown } | undefined)?.include_thoughts === true,
        );
    }

    async streamCanonicalContext(
        options: CanonicalExecutionContextInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionStream> {
        const canonicalOptions = options as CanonicalExecutionContextOptions;
        const source = await this.streamCanonicalContextEvents(canonicalOptions, signal, {
            stream_id: canonicalOptions.conversation_runtime.response_operation_id,
        });
        return new LegacyCanonicalExecutionEventProjection(
            source,
            (canonicalOptions.model_options as { include_thoughts?: unknown } | undefined)?.include_thoughts === true,
        );
    }

    async streamCanonicalEvents(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const canonicalOptions = options as CanonicalExecutionOptions;
        this.assertConversationInputSupported(options);
        if (canonicalOptions.conversation_runtime.materialized_input !== undefined && segments.length > 0) {
            throw new Error('A materialized canonical input requires an empty new prompt');
        }
        const runtime = canonicalOptions.conversation_runtime;
        if (this.isImageModel(canonicalOptions.model)) {
            if (!this.supportsCanonicalImageGeneration(canonicalOptions)) {
                throw new Error(
                    `Provider ${this.provider} model ${canonicalOptions.model} does not support canonical typed streaming`,
                );
            }
            this.validateCanonicalImageInput(segments, canonicalOptions);
        }
        signal?.throwIfAborted();
        const prompt = await this.createPrompt(segments, canonicalOptions);
        signal?.throwIfAborted();
        if (this.isImageModel(canonicalOptions.model) || !(await this.canStream(canonicalOptions, signal))) {
            const retainedDocument = canonicalOptions.conversation;
            const retainedResponse =
                retainedDocument !== undefined &&
                Object.hasOwn(retainedDocument.operation_receipts, runtime.response_operation_id)
                    ? createCanonicalExecutionResponse(retainedDocument, runtime.response_operation_id)
                    : undefined;
            const [generationId, responseTurnId] =
                retainedResponse === undefined
                    ? await Promise.all([
                          deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
                          deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
                      ])
                    : [retainedResponse.accepted_output.generation.id, retainedResponse.accepted_output.turn.id];
            const identityGeneration = retainedResponse?.accepted_output.generation;
            return new FallbackCanonicalExecutionEventStream(
                {
                    request_id: identityGeneration?.request_id ?? runtime.request_id,
                    attempt_id: identityGeneration?.attempt_id ?? runtime.attempt_id,
                    response_operation_id: runtime.response_operation_id,
                    generation_id: generationId,
                    draft_turn_id: responseTurnId,
                },
                (fallbackSignal) =>
                    this._executeCanonical(
                        prompt,
                        canonicalOptions,
                        signal ? AbortSignal.any([signal, fallbackSignal]) : fallbackSignal,
                    ),
                { ...open, ...(retainedResponse === undefined ? {} : { origin: 'accepted_recovery' as const }) },
            );
        }
        return await this.requestCanonicalTextCompletionEventStream(prompt, canonicalOptions, signal, open);
    }

    async streamCanonicalContextEvents(
        options: CanonicalExecutionContextInputOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const canonicalOptions = options as CanonicalExecutionContextOptions;
        if (!this.supportsCanonicalContextConversation(canonicalOptions) || this.isImageModel(canonicalOptions.model)) {
            throw new Error(
                `Provider ${this.provider} model ${canonicalOptions.model} does not support canonical context execution`,
            );
        }
        const runtime = canonicalOptions.conversation_runtime;
        signal?.throwIfAborted();
        if (!(await this.canStream(canonicalOptions, signal))) {
            const retainedDocument = canonicalOptions.conversation;
            const retainedResponse = Object.hasOwn(retainedDocument.operation_receipts, runtime.response_operation_id)
                ? createCanonicalExecutionResponse(retainedDocument, runtime.response_operation_id)
                : undefined;
            const [generationId, responseTurnId] =
                retainedResponse === undefined
                    ? await Promise.all([
                          deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
                          deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
                      ])
                    : [retainedResponse.accepted_output.generation.id, retainedResponse.accepted_output.turn.id];
            const identityGeneration = retainedResponse?.accepted_output.generation;
            return new FallbackCanonicalExecutionEventStream(
                {
                    request_id: identityGeneration?.request_id ?? runtime.request_id,
                    attempt_id: identityGeneration?.attempt_id ?? runtime.attempt_id,
                    response_operation_id: runtime.response_operation_id,
                    generation_id: generationId,
                    draft_turn_id: responseTurnId,
                },
                (fallbackSignal) =>
                    this._executeCanonicalContext(
                        canonicalOptions,
                        signal ? AbortSignal.any([signal, fallbackSignal]) : fallbackSignal,
                    ),
                { ...open, ...(retainedResponse === undefined ? {} : { origin: 'accepted_recovery' as const }) },
            );
        }
        return await this.requestCanonicalContextCompletionEventStream(canonicalOptions, signal, open);
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
    protected canStream(_options: ExecutionOptions, _signal?: AbortSignal) {
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

    abstract requestTextCompletion(
        prompt: PromptT,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion>;

    abstract requestTextCompletionStream(
        prompt: PromptT,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream>;

    async requestCanonicalTextCompletion(
        _prompt: PromptT,
        options: CanonicalExecutionOptions,
        _signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        throw new Error(`Provider ${this.provider} model ${options.model} does not support canonical execution`);
    }

    async requestCanonicalTextCompletionEventStream(
        _prompt: PromptT,
        options: CanonicalExecutionOptions,
        _signal: AbortSignal | undefined,
        _open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        throw new Error(`Provider ${this.provider} model ${options.model} does not support canonical typed streaming`);
    }

    async requestCanonicalContextCompletion(
        options: CanonicalExecutionContextOptions,
        _signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        throw new Error(
            `Provider ${this.provider} model ${options.model} does not support canonical context execution`,
        );
    }

    async requestCanonicalContextCompletionEventStream(
        options: CanonicalExecutionContextOptions,
        _signal: AbortSignal | undefined,
        _open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        throw new Error(
            `Provider ${this.provider} model ${options.model} does not support canonical context typed streaming`,
        );
    }

    async requestCanonicalImageGeneration(
        _prompt: PromptT,
        options: ExecutionOptions,
        _signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        throw new Error(`Provider ${this.provider} model ${options.model} does not support canonical image generation`);
    }

    async requestImageGeneration(
        _prompt: PromptT,
        _options: ExecutionOptions,
        _signal?: AbortSignal,
    ): Promise<Completion> {
        throw new Error('Image generation not implemented.');
        //Cannot be made abstract, as abstract methods are required in the derived class
    }

    //list models available for this environment
    abstract listModels(params?: ModelSearchPayload): Promise<AIModel[]>;

    //check that it is possible to connect to the environment
    abstract validateConnection(): Promise<boolean>;

    //generate embeddings for a given text
    abstract generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult>;

    private acquireOperationLease(): () => void {
        if (this._destroyRequested || this._resourcesDestroyed) {
            throw new Error(`Cannot use destroyed ${this.provider} driver`);
        }

        this._activeOperations++;
        let released = false;
        return () => {
            if (released) return;
            released = true;
            this._activeOperations--;
            this.destroyWhenIdle();
        };
    }

    /** Provider-specific resource cleanup, called once after active operations finish. */
    protected destroyProviderResources(): void | Promise<void> {}

    /**
     * Request cleanup when the driver is evicted from a cache. Active executions
     * and streams keep their resources until they complete, fail, or are cancelled.
     */
    destroy(): void {
        this._destroyRequested = true;
        this.destroyWhenIdle();
    }

    private destroyWhenIdle(): void {
        if (!this._destroyRequested || this._resourcesDestroyed || this._activeOperations > 0) return;

        this._resourcesDestroyed = true;
        try {
            void Promise.resolve(this.destroyProviderResources()).catch((error: unknown) => {
                this.logger.warn({ error }, `Failed to destroy provider resources for ${this.provider}`);
            });
        } catch (error: unknown) {
            this.logger.warn({ error }, `Failed to destroy provider resources for ${this.provider}`);
        } finally {
            try {
                this._httpAgent?.close().catch((error: unknown) => {
                    this.logger.warn({ error }, `Failed to close HTTP resources for ${this.provider}`);
                });
            } catch (error: unknown) {
                this.logger.warn({ error }, `Failed to close HTTP resources for ${this.provider}`);
            }
            this._httpAgent = undefined;
            this._driverFetch = undefined;
        }
    }
}

export { FallbackCompletionStream } from './CompletionStream.js';
