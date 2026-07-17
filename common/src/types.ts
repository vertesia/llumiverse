import type { BedrockOptions } from './options/bedrock.js';
import type { BedrockMantleOptions } from './options/bedrock_mantle.js';
import type { TextFallbackOptions } from './options/fallback.js';
import type { GroqOptions } from './options/groq.js';
import type { OpenAiOptions } from './options/openai.js';
import type { VertexAIOptions } from './options/vertexai.js';

// ============== Provider details ===============

export enum Providers {
    openai = 'openai',
    openai_compatible = 'openai_compatible',
    azure_openai = 'azure_openai',
    azure_foundry = 'azure_foundry',
    huggingface_ie = 'huggingface_ie',
    replicate = 'replicate',
    bedrock = 'bedrock',
    bedrock_mantle = 'bedrock_mantle',
    vertexai = 'vertexai',
    togetherai = 'togetherai',
    mistralai = 'mistralai',
    groq = 'groq',
    watsonx = 'watsonx',
    xai = 'xai',
    anthropic = 'anthropic',
}

export interface ProviderParams {
    id: Providers;
    name: string;
    requiresApiKey: boolean;
    requiresEndpointUrl: boolean;
    endpointPlaceholder?: string;
    supportSearch?: boolean;
}

export const ProviderList: Record<Providers, ProviderParams> = {
    openai: {
        id: Providers.openai,
        name: 'OpenAI',
        requiresApiKey: true,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    azure_openai: {
        id: Providers.azure_openai,
        name: 'Azure OpenAI',
        requiresApiKey: false,
        requiresEndpointUrl: true,
        supportSearch: false,
    },
    azure_foundry: {
        id: Providers.azure_foundry,
        name: 'Microsoft Foundry',
        requiresApiKey: true,
        requiresEndpointUrl: true,
        supportSearch: false,
    },
    huggingface_ie: {
        id: Providers.huggingface_ie,
        name: 'HuggingFace Inference Endpoint',
        requiresApiKey: true,
        requiresEndpointUrl: true,
    },
    replicate: {
        id: Providers.replicate,
        name: 'Replicate',
        requiresApiKey: true,
        requiresEndpointUrl: false,
        supportSearch: true,
    },
    bedrock: {
        id: Providers.bedrock,
        name: 'Amazon Bedrock',
        requiresApiKey: false,
        requiresEndpointUrl: false,
        endpointPlaceholder: 'region name (eg. us-east-1)',
        supportSearch: false,
    },
    bedrock_mantle: {
        id: Providers.bedrock_mantle,
        name: 'Amazon Bedrock Mantle',
        requiresApiKey: false,
        requiresEndpointUrl: true,
        endpointPlaceholder: 'region name (eg. us-west-2)',
        supportSearch: false,
    },
    vertexai: {
        id: Providers.vertexai,
        name: 'Google Agent Platform (Vertex AI)',
        requiresApiKey: false,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    togetherai: {
        id: Providers.togetherai,
        name: 'Together AI',
        requiresApiKey: true,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    mistralai: {
        id: Providers.mistralai,
        name: 'Mistral AI',
        requiresApiKey: false,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    groq: {
        id: Providers.groq,
        name: 'Groq Cloud',
        requiresApiKey: false,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    watsonx: {
        id: Providers.watsonx,
        name: 'IBM WatsonX',
        requiresApiKey: true,
        requiresEndpointUrl: true,
        supportSearch: false,
    },
    xai: {
        id: Providers.xai,
        name: 'xAI (Grok)',
        requiresApiKey: true,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    openai_compatible: {
        id: Providers.openai_compatible,
        name: 'OpenAI Compatible',
        requiresApiKey: true,
        requiresEndpointUrl: true,
        endpointPlaceholder: 'https://api.example.com/v1',
        supportSearch: false,
    },
    anthropic: {
        id: Providers.anthropic,
        name: 'Anthropic',
        requiresApiKey: true,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
};

// ============== Embeddings ===============

/**
 * Semantic task type for embedding models. Drivers map these to provider-
 * specific values (e.g. "RETRIEVAL_QUERY" for Vertex, "search_query" for Cohere).
 * - "query"    — a search query to find relevant documents
 * - "document" — a document to be indexed and retrieved
 */
export type EmbeddingTaskType = 'query' | 'document';

/**
 * One input to an embedding model. Discriminated by `type`.
 * Drivers consume binary content via DataSource and may pass a provider-native
 * URL (gs://, s3://, https://) when one is available.
 */
export type EmbeddingInput = TextEmbeddingInput | ImageEmbeddingInput | VideoEmbeddingInput | AudioEmbeddingInput;

export interface TextEmbeddingInput {
    type: 'text';
    text: string;
    /** Overrides EmbeddingsOptions.task_type for this input. */
    task_type?: EmbeddingTaskType;
    /** Document title hint — used by Vertex AI for RETRIEVAL_DOCUMENT. */
    title?: string;
}

export interface ImageEmbeddingInput {
    type: 'image';
    source: DataSource;
}

export interface VideoEmbeddingInput {
    type: 'video';
    source: DataSource;
    start_sec?: number;
    length_sec?: number;
    /** Vertex multimodal segment length. */
    interval_sec?: number;
    /** TwelveLabs Marengo. */
    use_fixed_length_sec?: boolean;
    /** TwelveLabs Marengo. */
    min_clip_sec?: number;
    /** TwelveLabs Marengo: which views to extract per segment. */
    embedding_option?: ('visual-text' | 'visual-image' | 'audio')[];
}

export interface AudioEmbeddingInput {
    type: 'audio';
    source: DataSource;
    start_sec?: number;
    length_sec?: number;
}

export interface EmbeddingsOptions {
    inputs: EmbeddingInput[];
    /**
     * Provider model id. When omitted the driver picks a sensible default
     * for the modalities present in `inputs`.
     */
    model?: string;

    /** Default task type applied to text inputs that don't set their own. */
    task_type?: EmbeddingTaskType;
    /** Output dimension truncation (MRL). Vertex/OpenAI where supported. */
    dimensions?: number;
}

export interface EmbeddingsResult {
    /** One result item per input, in the same order as EmbeddingsOptions.inputs. */
    results: EmbeddingResultItem[];
    /** The provider model id that produced the result. */
    model: string;
    /** Aggregate token usage when reported by the provider. */
    usage?: EmbeddingsTokenUsage;
}

export interface EmbeddingResultItem {
    /**
     * One or more vectors produced for this input.
     * Single vector for text/image; multiple for segmented video/audio or
     * joint-multimodal models that return per-modality vectors.
     */
    outputs: EmbeddingOutput[];
    /** Token count attributed to this input, when reported by the provider. */
    input_tokens?: number;
}

export interface EmbeddingOutput {
    values: number[];
    /** Which modality this vector represents (useful for joint-multimodal results). */
    modality?: 'text' | 'image' | 'video' | 'audio';
    /** Segment start time for video/audio. */
    start_sec?: number;
    /** Segment end time for video/audio. */
    end_sec?: number;
    /** TwelveLabs Marengo: which view of the segment this vector represents. */
    embedding_option?: string;
}

export interface EmbeddingsTokenUsage {
    input_tokens?: number;
    input_text_tokens?: number;
    input_image_tokens?: number;
}

export interface ResultValidationError {
    code: 'validation_error' | 'json_error' | 'content_policy_violation';
    message: string;
    data?: CompletionResult[];
}

/**
 * Context information for LlumiverseError.
 * Provides details about where and how the error occurred.
 */
export interface LlumiverseErrorContext {
    /** The provider that generated the error (e.g., 'openai', 'bedrock', 'vertexai') */
    provider: string;
    /** The model that was being used when the error occurred */
    model: string;
    /** The operation that failed */
    operation: 'execute' | 'stream';
}

/**
 * Standardized error class for Llumiverse driver errors.
 *
 * Normalizes errors from different LLM providers (OpenAI, Anthropic, Bedrock, VertexAI, etc.)
 * into a consistent format. The primary value is the `retryable` flag, which enables upstream
 * consumers to implement smart retry logic.
 *
 * @example
 * ```typescript
 * try {
 *   const result = await driver.execute(segments, options);
 * } catch (error) {
 *   if (LlumiverseError.isLlumiverseError(error)) {
 *     console.log(`Provider: ${error.context.provider}`);
 *     console.log(`Model: ${error.context.model}`);
 *     console.log(`Retryable: ${error.retryable}`);
 *
 *     if (error.retryable) {
 *       // Implement retry logic with exponential backoff
 *       await retryWithBackoff(() => driver.execute(segments, options));
 *     } else {
 *       // Handle non-retryable error (e.g., invalid API key, malformed request)
 *       logError(error);
 *     }
 *   }
 *   throw error;
 * }
 * ```
 */
export class LlumiverseError extends Error {
    /**
     * HTTP status code (e.g., 429, 500) if available.
     * Undefined if the error doesn't have a numeric status code.
     */
    readonly code?: number;

    /**
     * Provider-specific error name/type (e.g., "ThrottlingException", "ValidationException").
     * Optional - used to preserve the semantic error type from the provider SDK.
     */
    readonly name: string;

    /**
     * Whether this error is retryable.
     * - True: Definitely retryable (rate limits, timeouts, server errors)
     * - False: Definitely not retryable (auth failures, invalid requests, malformed schemas)
     * - Undefined: Unknown retryability - allows consumers to decide default behavior
     *
     * When undefined, consumers can choose their retry strategy:
     * - Conservative: Don't retry unknown errors (avoid spam)
     * - Resilient: Retry unknown errors (prioritize success)
     */
    readonly retryable?: boolean;

    /**
     * Context about where and how the error occurred.
     * Includes provider, model, operation type.
     */
    readonly context: LlumiverseErrorContext;

    /**
     * The original error from the provider SDK.
     * Preserved for debugging and detailed error inspection.
     */
    readonly originalError: unknown;

    constructor(
        message: string,
        retryable: boolean | undefined,
        context: LlumiverseErrorContext,
        originalError: unknown,
        code?: number,
        name?: string,
    ) {
        super(message);
        this.name = name || 'LlumiverseError';
        this.code = code;
        this.retryable = retryable;
        this.context = context;
        this.originalError = originalError;

        // Preserve stack trace from original error if available
        if (originalError instanceof Error && originalError.stack) {
            this.stack = originalError.stack;
        }
    }

    /**
     * Serialize the error to JSON for logging or transmission.
     * Includes all error properties except the original error object itself.
     */
    toJSON(): Record<string, unknown> {
        return {
            name: this.name,
            message: this.message,
            code: this.code,
            retryable: this.retryable,
            context: this.context,
            stack: this.stack,
            // Include original error message if available
            originalErrorMessage:
                this.originalError instanceof Error ? this.originalError.message : String(this.originalError),
        };
    }

    /**
     * Type guard to check if an error is a LlumiverseError.
     * Useful for conditional error handling.
     *
     * @param error - The error to check
     * @returns True if the error is a LlumiverseError
     */
    static isLlumiverseError(error: unknown): error is LlumiverseError {
        return error instanceof LlumiverseError;
    }
}

// ============== Result Types ===============

export interface BaseResult {
    type: 'text' | 'json' | 'image';
    value: unknown;
}

export interface TextResult extends BaseResult {
    type: 'text';
    value: string;
}

export interface JsonResult extends BaseResult {
    type: 'json';
    value: JSONValue;
}

export interface ImageResult extends BaseResult {
    type: 'image';
    value: string; // base64 data url or real url
}

/**
 * @discriminator type
 */
export type CompletionResult = TextResult | JsonResult | ImageResult;

//Internal structure used in driver implementation.
export interface CompletionChunkObject {
    result: CompletionResult[];
    token_usage?: ExecutionTokenUsage;
    finish_reason?: 'stop' | 'length' | string;
    /**
     * Tool calls returned by the model during streaming.
     * Each chunk may contain partial tool call information that needs to be aggregated.
     * Typed with `unknown` params because streamed chunks carry partial (often string)
     * tool_input that is only parsed into a JSONObject once fully accumulated.
     */
    tool_use?: ToolUse<unknown>[];
}

/**
 * Internal provider stream contract.
 *
 * Native protocol adapters attach a request-local finalizer when conversation
 * replay requires information that cannot be reconstructed from generic results.
 */
export interface DriverCompletionStream extends AsyncIterable<CompletionChunkObject> {
    finalizeConversation?: () => unknown | Promise<unknown>;
}

/**
 * Tool definition for LLM tool use.
 * The input_schema uses a permissive type to support both:
 * - AJV's JSONSchemaType<T> for type-safe schema generation
 * - Plain object schemas for simpler cases
 */
export interface ToolDefinition {
    name: string;
    description?: string;
    input_schema: {
        type?: unknown;
        properties?: Record<string, unknown> | null | undefined;
        required?: readonly string[] | string[];
        additionalProperties?: unknown;
        [k: string]: unknown;
    };
}
/**
 * A tool use instance represents a call to a tool.
 * The id property is used to identify the tool call.
 */
export interface ToolUse<ParamsT = JSONObject> {
    id: string;
    tool_name: string;
    tool_input: ParamsT | null;
    /**
     * Gemini thinking models require thought_signature to be passed back with tool results.
     * This preserves the model's reasoning state during multi-turn tool use.
     */
    thought_signature?: string;
}

export interface Completion {
    // the driver impl must return the result and optionally the token_usage. the execution time is computed by the extended abstract driver
    result: CompletionResult[];
    token_usage?: ExecutionTokenUsage;
    /**
     * Contains the tools from which the model awaits information.
     */
    tool_use?: ToolUse<unknown>[];
    /**
     * The finish reason as reported by the model: stop | length or other model specific values
     */
    finish_reason?: 'stop' | 'length' | 'tool_use' | string;

    /**
     * Set only if a result validation error occurred, otherwise if the result is valid the error field is undefined
     * This can only be set if the result_schema is set and the result could not be parsed as a json or if the result does not match the schema
     */
    error?: ResultValidationError;

    /**
     * The original response. Only included if the option include_original_response is set to true and the request is made using execute. Not supported when streaming.
     */
    original_response?: unknown;

    /**
     * The conversation context. This is an opaque structure that can be passed to the next request to restore the context.
     */
    conversation?: unknown;
}

export interface ExecutionResponse<PromptT = unknown> extends Completion {
    prompt: PromptT;
    /**
     * The time it took to execute the request in seconds
     */
    execution_time?: number;
    /**
     * The number of chunks for streamed executions
     */
    chunks?: number;
}

export interface CompletionStream<PromptT = unknown> extends AsyncIterable<string> {
    completion: ExecutionResponse<PromptT> | undefined;
}

/**
 * Minimal logger interface for LLM drivers.
 * Follows pino v10 signature: supports message-only or object-first (never message-first).
 * - Message-only: logger.info("message")
 * - Object-first: logger.info({ data }, "message")
 * - PREVENTS: logger.info("message", { data }) - compile error (objects not allowed in ...args)
 *
 * Additional args must be primitives (string | number | boolean) for string interpolation.
 */
export interface Logger {
    debug(msg: string, ...args: (string | number | boolean)[]): void;
    debug<T>(obj: T, msg?: T extends string ? never : string, ...args: (string | number | boolean)[]): void;
    info(msg: string, ...args: (string | number | boolean)[]): void;
    info<T>(obj: T, msg?: T extends string ? never : string, ...args: (string | number | boolean)[]): void;
    warn(msg: string, ...args: (string | number | boolean)[]): void;
    warn<T>(obj: T, msg?: T extends string ? never : string, ...args: (string | number | boolean)[]): void;
    error(msg: string, ...args: (string | number | boolean)[]): void;
    error<T>(obj: T, msg?: T extends string ? never : string, ...args: (string | number | boolean)[]): void;
}

/**
 * HTTP timeouts applied to a driver's upstream LLM-provider calls.
 *
 * All values are in milliseconds. Drivers should map these onto whatever
 * HTTP client their SDK uses; the defaults applied in
 * `@llumiverse/core/createDriverHttpAgent` are:
 *   - headersTimeout:   60_000
 *   - bodyTimeout:      60_000
 *   - connectTimeout:   10_000
 *   - keepAliveTimeout: 30_000
 *
 * The defaults are deliberately tighter than Node's undici default
 * (5 minutes for headers/body) so a hung upstream surfaces quickly. Bump
 * `bodyTimeout` for streaming flows that have legitimate silent gaps
 * (e.g. tool-using agents).
 */
export interface HttpTimeoutOptions {
    /** Time (ms) to wait for the first response byte after the request is sent. */
    headersTimeout?: number;
    /** Time (ms) between body chunks once streaming has started. */
    bodyTimeout?: number;
    /** TCP/TLS connect timeout (ms). */
    connectTimeout?: number;
    /** Idle socket reuse timeout (ms). */
    keepAliveTimeout?: number;
}

export interface DriverOptions {
    logger?: Logger | 'console';
    /**
     * Optional HTTP timeouts applied to the driver's upstream LLM-provider
     * calls. Drivers that don't make HTTP calls (e.g. the in-process test
     * driver) ignore this. See {@link HttpTimeoutOptions} for defaults.
     */
    httpTimeout?: HttpTimeoutOptions;
}

export type JSONSchemaTypeName =
    | 'string' //
    | 'number'
    | 'integer'
    | 'boolean'
    | 'object'
    | 'array'
    | 'null'
    | 'any';

export type JSONSchemaType =
    | string //
    | number
    | boolean
    | JSONSchemaObject
    | JSONSchemaArray
    | null;

export interface JSONSchemaObject {
    [key: string]: JSONSchemaType;
}

export interface JSONSchemaArray extends Array<JSONSchemaType> {}

export interface JSONSchemaProperties {
    [key: string]: JSONSchema;
}

export interface JSONSchema {
    type?: JSONSchemaTypeName | JSONSchemaTypeName[];
    description?: string;
    properties?: JSONSchemaProperties;
    items?: JSONSchema;
    format?: string;
    editor?: unknown;
    default?: unknown;
    additionalProperties?: boolean | JSONSchema;
    required?: string[];
    [k: string]: unknown;
}

export type PromptFormatter<T = unknown> = (messages: PromptSegment[], schema?: JSONSchema) => T;

//Options are split into PromptOptions, ModelOptions and ExecutionOptions.
//ExecutionOptions are most often used within llumiverse as they are the most complete.
//The base types are useful for external code that needs to interact with llumiverse.
export interface PromptOptions {
    model: string;
    /**
     * A custom formatter to use for format the final model prompt from the input prompt segments.
     * If no one is specified the driver will choose a formatter compatible with the target model
     */
    format?: PromptFormatter;
    result_schema?: JSONSchema;
    /**
     * Provider-specific opt-in to put the result schema after the cached prompt
     * prefix instead of including it in native structured-output configuration.
     * The returned JSON is still validated against result_schema by Llumiverse.
     */
    prompt_cache_schema_suffix?: boolean;
}

export interface StatelessExecutionOptions extends PromptOptions {
    /**
     * If set to true the original response from the target LLM will be included in the response under the original_response field.
     * This is useful for debugging and for some advanced use cases.
     * It is ignored on streaming requests
     */
    include_original_response?: boolean;
    model_options?: ModelOptions;

    /**
     * Stable identity for prompt caching. Providers with cache routing keys receive the value directly;
     * providers with cache breakpoints use its presence to cache the stable prefix before the final dynamic block.
     * Providers with fully implicit caching still require an identical prompt prefix.
     */
    prompt_cache_key?: string;

    /**
     * Per-call HTTP timeouts for upstream LLM-provider calls. These override
     * the driver's default `DriverOptions.httpTimeout` for this execution only.
     */
    httpTimeout?: HttpTimeoutOptions;

    /**
     * @deprecated This is deprecated. Use CompletionResult.type information instead.
     */
    output_modality?: Modalities;
}

export interface ExecutionOptions extends StatelessExecutionOptions {
    /**
     * Available tools for the request
     */
    tools?: ToolDefinition[];
    /**
     * This is an opaque structure that provides a conversation context
     * Each driver implementation will return a conversation property in the execution response
     * that can be passed here to restore the context when a new prompt is sent to the model.
     */
    conversation?: unknown | null;
    /**
     * Labels for billing attribution and cost tracking.
     * Passed through to provider APIs that support request-level labels (e.g. Vertex AI).
     * Keys and values must be lowercase, max 64 characters, containing only letters, numbers, underscores, and dashes.
     */
    labels?: Record<string, string>;
    /**
     * Number of turns to keep images in conversation history before stripping them.
     * - 0 (default): Strip images immediately after each turn
     * - 1: Keep images for current turn only, strip in next turn
     * - N: Keep images for N turns before stripping
     * - undefined: Same as 0, strip immediately
     *
     * Images are stripped to prevent JSON.stringify corruption (Uint8Array) and reduce storage bloat (base64).
     */
    stripImagesAfterTurns?: number;

    /**
     * Maximum tokens to keep for text content in tool results.
     * Text exceeding this limit will be truncated with a "[Content truncated...]" marker.
     * - undefined/0: No text truncation (default)
     * - N > 0: Truncate text to approximately N tokens (using ~4 chars/token estimate)
     */
    stripTextMaxTokens?: number;

    /**
     * Number of turns to keep heartbeat messages before stripping.
     * - 0: Strip immediately
     * - 1 (default): Keep for 1 turn then strip
     * - undefined: Same as 1, strip after 1 turn
     */
    stripHeartbeatsAfterTurns?: number;
}

//Common names to share between different models
export enum SharedOptions {
    //Text
    max_tokens = 'max_tokens',
    temperature = 'temperature',
    top_p = 'top_p',
    top_k = 'top_k',
    presence_penalty = 'presence_penalty',
    frequency_penalty = 'frequency_penalty',
    stop_sequence = 'stop_sequence',
    effort = 'effort',

    //Image
    seed = 'seed',
    number_of_images = 'number_of_images',
}

export enum OptionType {
    numeric = 'numeric',
    enum = 'enum',
    boolean = 'boolean',
    string_list = 'string_list',
}

export type ReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

// ============== Model Options ===============

/**
 * @discriminator _option_id
 */
export type ModelOptions =
    | TextFallbackOptions
    | VertexAIOptions
    | BedrockOptions
    | BedrockMantleOptions
    | OpenAiOptions
    | GroqOptions;

// ============== Option Info ===============

export interface ModelOptionsInfo {
    options: ModelOptionInfoItem[];
    _option_id: string; //Should follow same ids as ModelOptions
}

export type ModelOptionInfoItem = NumericOptionInfo | EnumOptionInfo | BooleanOptionInfo | StringListOptionInfo;
interface OptionInfoPrototype {
    type: OptionType;
    name: string;
    description?: string;

    //If this is true, whether other options apply is dependent on this option
    //Therefore, if this option is changed, the set of available options should be refreshed.
    refresh?: boolean;
}

export interface NumericOptionInfo extends OptionInfoPrototype {
    type: OptionType.numeric;
    value?: number;
    min?: number;
    max?: number;
    step?: number;
    integer?: boolean;
    default?: number;
}

export interface EnumOptionInfo extends OptionInfoPrototype {
    type: OptionType.enum;
    value?: string;
    enum: Record<string, string>;
    default?: string;
}

export interface BooleanOptionInfo extends OptionInfoPrototype {
    type: OptionType.boolean;
    value?: boolean;
    default?: boolean;
}

export interface StringListOptionInfo extends OptionInfoPrototype {
    type: OptionType.string_list;
    value?: string[];
    default?: string[];
}

// ============== Prompts ===============
export enum PromptRole {
    safety = 'safety',
    system = 'system',
    user = 'user',
    assistant = 'assistant',
    negative = 'negative',
    mask = 'mask',
    /**
     * Used to send the response of a tool
     */
    tool = 'tool',
}

export interface PromptSegment {
    role: PromptRole;
    content: string;
    /**
     * The tool use id if the segment is a tool response
     */
    tool_use_id?: string;
    /**
     * Gemini thinking models require thought_signature to be passed back with tool results.
     * This should be copied from the ToolUse.thought_signature when sending tool responses.
     */
    thought_signature?: string;
    files?: DataSource[];
}

export interface ExecutionTokenUsage {
    prompt?: number;
    result?: number;
    total?: number;
    /**
     * Number of input tokens read from prompt cache (discounted rate).
     */
    prompt_cached?: number;
    /**
     * Number of input tokens written to prompt cache.
     */
    prompt_cache_write?: number;

    /* Number of new input tokens not from cache. This is useful for providers with prompt caching to understand how many tokens were actually newly processed in the prompt, separate from any cached tokens. Calculated as prompt - prompt_cached. */
    prompt_new?: number; // Number of new input tokens not from cache (calculated as prompt - prompt_cached)
}

/**
 * @deprecated This is deprecated. Use CompletionResult.type information instead.
 */
export enum Modalities {
    text = 'text',
    image = 'image',
}

/**
 * Represents the output and input modalities a model can support
 */
export interface ModelModalities {
    text?: boolean;
    image?: boolean;
    video?: boolean;
    audio?: boolean;
    embed?: boolean; //Only for output
}

export interface ModelCapabilities {
    input: ModelModalities;
    output: ModelModalities;
    tool_support?: boolean; //if the model supports tool use
    tool_support_streaming?: boolean; //if the model supports tool use with streaming
}

// ============== AI MODEL ==============

export interface AIModel<ProviderKeys = string> {
    id: string; //id of the model known by the provider
    name: string; //human readable name
    provider: ProviderKeys; //provider name
    description?: string;
    version?: string; //if any version is specified
    type?: ModelType; //type of the model
    tags?: string[]; //tags for searching
    owner?: string; //owner of the model
    status?: AIModelStatus; //status of the model
    can_stream?: boolean; //if the model's response can be streamed
    is_custom?: boolean; //if the model is a custom model (a trained model)
    is_multimodal?: boolean; //if the model support files and images
    input_modalities?: string[]; //Input modalities supported by the model (e.g. text, image, video, audio)
    output_modalities?: string[]; //Output modalities supported by the model (e.g. text, image, video, audio)
    tool_support?: boolean; //if the model supports tool use
    environment?: string; //the environment name
}

export enum AIModelStatus {
    Available = 'available',
    Pending = 'pending',
    Stopped = 'stopped',
    Unavailable = 'unavailable',
    Unknown = 'unknown',
    Legacy = 'legacy',
}

/**
 * payload to list available models for an environment
 * @param environmentId id of the environment
 * @param query text to search for in model name/description
 * @param type type of the model
 * @param tags tags for searching
 */
export interface ModelSearchPayload {
    text: string;
    type?: ModelType;
    tags?: string[];
    owner?: string;
}

export enum ModelType {
    Classifier = 'classifier',
    Regressor = 'regressor',
    Clustering = 'clustering',
    AnomalyDetection = 'anomaly-detection',
    TimeSeries = 'time-series',
    Text = 'text',
    Image = 'image',
    Audio = 'audio',
    Video = 'video',
    Embedding = 'embedding',
    Chat = 'chat',
    Code = 'code',
    NLP = 'nlp',
    MultiModal = 'multi-modal',
    Test = 'test',
    Other = 'other',
    Unknown = 'unknown',
}

// ============== training =====================

export interface DataSource {
    name: string;
    mime_type: string;
    getStream(): Promise<ReadableStream<Uint8Array | string>>;
    getURL(): Promise<string>;
    getURI(): Promise<string>;
}

export interface TrainingOptions {
    name: string; // the new model name
    model: string; // the model to train
    params?: JSONObject; // the training parameters
}

export interface TrainingPromptOptions {
    segments: PromptSegment[];
    completion: CompletionResult[];
    model: string; // the model to train
    schema?: JSONSchema; // the result schema f any
}

export enum TrainingJobStatus {
    running = 'running',
    succeeded = 'succeeded',
    failed = 'failed',
    cancelled = 'cancelled',
}

export interface TrainingJob {
    id: string; // id of the training job
    status: TrainingJobStatus; // status of the training job - depends on the implementation
    details?: string;
    model?: string; // the name of the fine tuned model which is created
}

export type JSONPrimitive = string | number | boolean | null;
export type JSONArray = JSONValue[];
export type JSONObject = { [key: string]: JSONValue };
export type JSONComposite = JSONArray | JSONObject;
export type JSONValue = JSONPrimitive | JSONComposite;
