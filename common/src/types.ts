import { BedrockOptions } from './options/bedrock.js';
import { TextFallbackOptions } from './options/fallback.js';
import { GroqOptions } from './options/groq.js';
import { OpenAiOptions } from './options/openai.js';
import { VertexAIOptions } from './options/vertexai.js';

// ============== Provider details ===============

export enum Providers {
    openai = 'openai',
    azure_openai = 'azure_openai',
    azure_foundry = 'azure_foundry',
    huggingface_ie = 'huggingface_ie',
    replicate = 'replicate',
    bedrock = 'bedrock',
    vertexai = 'vertexai',
    togetherai = 'togetherai',
    mistralai = 'mistralai',
    groq = 'groq',
    watsonx = 'watsonx'
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
    openai:
    {
        id: Providers.openai,
        name: "OpenAI",
        requiresApiKey: true,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    azure_openai:
    {
        id: Providers.azure_openai,
        name: "Azure OpenAI",
        requiresApiKey: false,
        requiresEndpointUrl: true,
        supportSearch: false,
    },
    azure_foundry:
    {
        id: Providers.azure_foundry,
        name: "Azure Foundry",
        requiresApiKey: true,
        requiresEndpointUrl: true,
        supportSearch: false,
    },
    huggingface_ie:
    {
        id: Providers.huggingface_ie,
        name: "HuggingFace Inference Endpoint",
        requiresApiKey: true,
        requiresEndpointUrl: true,
    },
    replicate:
    {
        id: Providers.replicate,
        name: "Replicate",
        requiresApiKey: true,
        requiresEndpointUrl: false,
        supportSearch: true,
    },
    bedrock:
    {
        id: Providers.bedrock,
        name: "AWS Bedrock",
        requiresApiKey: false,
        requiresEndpointUrl: false,
        endpointPlaceholder: "region name (eg. us-east-1)",
        supportSearch: false,
    },
    vertexai: {
        id: Providers.vertexai,
        name: "Google Vertex AI",
        requiresApiKey: false,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    togetherai: {
        id: Providers.togetherai,
        name: "Together AI",
        requiresApiKey: false,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    mistralai: {
        id: Providers.mistralai,
        name: "Mistral AI",
        requiresApiKey: false,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    groq: {
        id: Providers.groq,
        name: "Groq Cloud",
        requiresApiKey: false,
        requiresEndpointUrl: false,
        supportSearch: false,
    },
    watsonx: {
        id: Providers.watsonx,
        name: "IBM WatsonX",
        requiresApiKey: true,
        requiresEndpointUrl: true,
        supportSearch: false
    },
}

// ============== Embeddings ===============

export interface EmbeddingsOptions {
    /**
     * The text to generate the embeddings for. One of text or image is required.
     */
    text?: string;
    /**
     * The image to generate embeddings for
     */
    image?: string
    /**
     * The model to use to generate the embeddings. Optional.
     */
    model?: string;

}

export interface EmbeddingsResult {
    /**
     * The embedding vectors corresponding to the words in the input text.
     */
    values: number[];
    /**
     * The model used to generate the embeddings.
     */
    model: string;
    /**
     * Number of tokens of the input text.
     */
    token_count?: number;

}

export interface ResultValidationError {
    code: 'validation_error' | 'json_error' | 'content_policy_violation';
    message: string;
    data?: string;
}

// ============== Result Types ===============

export interface BaseResult {
    type: "text" | "json" | "image";
    value: any;
}

export interface TextResult extends BaseResult {
    type: "text";
    value: string;
}

export interface JsonResult extends BaseResult {
    type: "json";
    value: JSONValue;
}

export interface ImageResult extends BaseResult {
    type: "image";
    value: string; // base64 data url or real url
}

export type CompletionResult = TextResult | JsonResult | ImageResult;

//Internal structure used in driver implementation.
export interface CompletionChunkObject {
    result: CompletionResult[];
    token_usage?: ExecutionTokenUsage;
    finish_reason?: "stop" | "length" | string;
}

export interface ToolDefinition {
    name: string,
    description?: string,
    input_schema: {
        type: 'object';
        properties?: JSONSchema | null | undefined;
        [k: string]: unknown;
    },
}
/**
 * A tool use instance represents a call to a tool.
 * The id property is used to identify the tool call.
 */
export interface ToolUse<ParamsT = JSONObject> {
    id: string,
    tool_name: string,
    tool_input: ParamsT | null
}

export interface Completion {
    // the driver impl must return the result and optionally the token_usage. the execution time is computed by the extended abstract driver
    result: CompletionResult[];
    token_usage?: ExecutionTokenUsage;
    /**
     * Contains the tools from which the model awaits information.
     */
    tool_use?: ToolUse[];
    /**
     * The finish reason as reported by the model: stop | length or other model specific values
     */
    finish_reason?: "stop" | "length" | "tool_use" | string;

    /**
     * Set only if a result validation error occurred, otherwise if the result is valid the error field is undefined
     * This can only be set if the result_schema is set and the result could not be parsed as a json or if the result does not match the schema
     */
    error?: ResultValidationError;

    /**
     * The original response. Only included if the option include_original_response is set to true and the request is made using execute. Not supported when streaming.
     */
    original_response?: Record<string, any>;

    /**
     * The conversation context. This is an opaque structure that can be passed to the next request to restore the context.
     */
    conversation?: unknown;
}

export interface ImageGeneration {

    images?: string[];

}

export interface ExecutionResponse<PromptT = any> extends Completion {
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


export interface CompletionStream<PromptT = any> extends AsyncIterable<string> {
    completion: ExecutionResponse<PromptT> | undefined;
}

export interface Logger {
    debug: (...obj: any[]) => void;
    info: (...obj: any[]) => void;
    warn: (...obj: any[]) => void;
    error: (...obj: any[]) => void;
}

export interface DriverOptions {
    logger?: Logger | "console";
}

export type JSONSchemaTypeName =
    | "string" //
    | "number"
    | "integer"
    | "boolean"
    | "object"
    | "array"
    | "null"
    | "any";

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

export interface JSONSchemaArray extends Array<JSONSchemaType> { }

export interface JSONSchema {
    type?: JSONSchemaTypeName | JSONSchemaTypeName[];
    description?: string;
    properties?: Record<string, JSONSchema>;
    required?: string[];
    [k: string]: any;
}

export type PromptFormatter<T = any> = (messages: PromptSegment[], schema?: JSONSchema) => T;

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
}

export interface StatelessExecutionOptions extends PromptOptions {
    /**
     * If set to true the original response from the target LLM will be included in the response under the original_response field.
     * This is useful for debugging and for some advanced use cases.
     * It is ignored on streaming requests
     */
    include_original_response?: boolean;
    model_options?: ModelOptions;
    output_modality: Modalities;
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
}

//Common names to share between different models
export enum SharedOptions {
    //Text
    max_tokens = "max_tokens",
    temperature = "temperature",
    top_p = "top_p",
    top_k = "top_k",
    presence_penalty = "presence_penalty",
    frequency_penalty = "frequency_penalty",
    stop_sequence = "stop_sequence",

    //Image
    seed = "seed",
    number_of_images = "number_of_images",
}

export enum OptionType {
    numeric = "numeric",
    enum = "enum",
    boolean = "boolean",
    string_list = "string_list"
}

// ============== Model Options ===============

export type ModelOptions = TextFallbackOptions | VertexAIOptions | BedrockOptions | OpenAiOptions | GroqOptions;

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
    safety = "safety",
    system = "system",
    user = "user",
    assistant = "assistant",
    negative = "negative",
    mask = "mask",
    /**
     * Used to send the response of a tool
     */
    tool = "tool"
}

export interface PromptSegment {
    role: PromptRole;
    content: string;
    /**
     * The tool use id if the segment is a tool response
     */
    tool_use_id?: string;
    files?: DataSource[]
}

export interface ExecutionTokenUsage {
    prompt?: number;
    result?: number;
    total?: number;
}

export enum Modalities {
    text = "text",
    image = "image",
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
    is_multimodal?: boolean //if the model support files and images
    input_modalities?: string[]; //Input modalities supported by the model (e.g. text, image, video, audio)
    output_modalities?: string[]; //Output modalities supported by the model (e.g. text, image, video, audio)
    tool_support?: boolean; //if the model supports tool use
    environment?: string; //the environment name
}

export enum AIModelStatus {
    Available = "available",
    Pending = "pending",
    Stopped = "stopped",
    Unavailable = "unavailable",
    Unknown = "unknown",
    Legacy = "legacy",
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
    Classifier = "classifier",
    Regressor = "regressor",
    Clustering = "clustering",
    AnomalyDetection = "anomaly-detection",
    TimeSeries = "time-series",
    Text = "text",
    Image = "image",
    Audio = "audio",
    Video = "video",
    Embedding = "embedding",
    Chat = "chat",
    Code = "code",
    NLP = "nlp",
    MultiModal = "multi-modal",
    Test = "test",
    Other = "other",
    Unknown = "unknown"
}


// ============== training =====================



export interface DataSource {
    name: string;
    mime_type: string;
    getStream(): Promise<ReadableStream<Uint8Array | string>>;
    getURL(): Promise<string>;
}

export interface TrainingOptions {
    name: string; // the new model name
    model: string; // the model to train
    params?: JSONObject; // the training parameters
}

export interface TrainingPromptOptions {
    segments: PromptSegment[];
    completion: string | JSONObject;
    model: string; // the model to train
    schema?: JSONSchema; // the result schema f any
}

export enum TrainingJobStatus {
    running = "running",
    succeeded = "succeeded",
    failed = "failed",
    cancelled = "cancelled",
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
