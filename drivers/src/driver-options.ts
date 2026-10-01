/**
 * Constructor options of every driver, kept apart from the driver implementations.
 *
 * Nothing here imports a provider SDK, so code that only builds options, or creates drivers through
 * `@llumiverse/drivers/factory`, type-checks without loading the declarations of every provider SDK.
 */

import type { AwsCredentialIdentity, Provider } from '@aws-sdk/types';
import type { TokenCredential } from '@azure/core-auth';
import type { DriverOptions, ExecutionOptions } from '@llumiverse/core';
import type { GoogleAuthOptions } from 'google-auth-library';

export interface AnthropicDriverOptions extends DriverOptions {
    apiKey?: string;
    baseURL?: string;
}

export interface AzureFoundryDriverOptions extends DriverOptions {
    /** Source model for opaque image deployment names. */
    sourceModel?: string;
    /**
     * The credentials to use to access Azure AI Foundry
     */
    azureADTokenProvider?: TokenCredential;

    endpoint?: string;

    apiVersion?: string;
}

export interface BedrockDriverOptions extends DriverOptions {
    /**
     * The AWS region
     */
    region: string;
    /**
     * The bucket name to be used for training.
     * It will be created if does not already exist.
     */
    training_bucket?: string;

    /**
     * The role ARN to be used for training
     */
    training_role_arn?: string;

    /**
     * The credentials to use to access AWS (IAM access key + secret)
     */
    credentials?: AwsCredentialIdentity | Provider<AwsCredentialIdentity>;
}

export interface BedrockMantleDriverOptions extends DriverOptions {
    region: string;
    credentials?: AwsCredentialIdentity | Provider<AwsCredentialIdentity>;
}

export interface GroqDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export interface HuggingFaceIEDriverOptions extends DriverOptions {
    apiKey: string;
    endpoint_url: string;
}

export interface MistralAIDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint_url?: string;
}

export interface AzureOpenAIDriverOptions extends DriverOptions {
    /** Source model for opaque image deployment names. */
    sourceModel?: string;
    /**
     * The credentials to use to access Azure OpenAI
     */
    azureADTokenProvider?: (options?: unknown) => Promise<string>;

    apiKey?: string;

    endpoint?: string;

    apiVersion?: string;

    deployment?: string;
}

export interface OpenAIDriverOptions extends DriverOptions {
    /**
     * The OpenAI api key
     */
    apiKey?: string; //type with azure credentials
}

export interface OpenAIChatCompletionsProtocolOptions {
    /** The model identifier to send in the request body (for example, "zai-org/glm-5-maas"). */
    modelName?: string;
    /** Model API contract default used only when callers do not provide max_tokens. */
    defaultMaxTokens?: number;
    /** Extra OpenAI-compatible request body fields for model-family-specific options. */
    extraBody?: Record<string, unknown>;
    /**
     * How result_schema should be requested. Vertex MaaS supports response_format, while
     * TogetherAI stays prompt-instruction based because its OpenAI-compatible surface is
     * Chat Completions only and response_format support is not reliable across hosted models.
     */
    resultSchemaMode?: 'response_format' | 'prompt';
    /** Supplement native structured output with prompt alignment for providers with unreliable enforcement. */
    includeResultSchemaInPrompt?: boolean;
    /** Model-specific form of the prompt alignment guard for mixed-model providers. */
    includeResultSchemaInPromptForModel?: (model: string) => boolean;
    /**
     * OpenAI supports strict function schemas. Some OpenAI-compatible providers reject
     * or mis-handle those OpenAI-specific fields, so adapters can request a looser
     * JSON Schema payload for tools while preserving the shared Chat Completions path.
     */
    toolSchemaMode?: 'openai_strict' | 'compatible';
    /** Resolve SDK options from the same driver/per-execution policy as the HTTP transport. */
    resolveRequestOptions?: (
        options: Pick<ExecutionOptions, 'httpTimeout'>,
        signal?: AbortSignal,
    ) => { signal?: AbortSignal; timeout?: number } | undefined;
}

export interface OpenAIChatCompletionsDriverOptions extends DriverOptions {
    defaultMaxTokens?: number;
    extraBody?: Record<string, unknown>;
    resultSchemaMode?: OpenAIChatCompletionsProtocolOptions['resultSchemaMode'];
    includeResultSchemaInPrompt?: OpenAIChatCompletionsProtocolOptions['includeResultSchemaInPrompt'];
    includeResultSchemaInPromptForModel?: OpenAIChatCompletionsProtocolOptions['includeResultSchemaInPromptForModel'];
    toolSchemaMode?: OpenAIChatCompletionsProtocolOptions['toolSchemaMode'];
}

export interface OpenAIResponsesDriverOptions extends DriverOptions {
    /**
     * The API key for the OpenAI-compatible service
     */
    apiKey: string;

    /**
     * The base URL of the OpenAI-compatible API endpoint
     * Example: https://api.example.com/v1
     */
    endpoint: string;

    /**
     * Custom headers to include in every request.
     * Useful for Apigee proxies or custom auth schemes.
     */
    default_headers?: Record<string, string>;
}

export interface OpenRouterDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint?: string;
    httpReferer?: string;
    appTitle?: string;
    appCategories?: string;
}

export interface ReplicateDriverOptions extends DriverOptions {
    apiKey: string;
}

export interface TogetherAIDriverOptions extends OpenAIChatCompletionsDriverOptions {
    apiKey: string;
    endpoint?: string;
}

export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
    /**
     * Kill switch for explicit Gemini context caching (Vertex `cachedContents`). Caching is normally
     * decided per execution — see `ExecutionOptions.prompt_cache_mode`, which defaults to caching the
     * static prefix whenever `prompt_cache_key` is set. Setting this to `false` disables the whole
     * path for every execution this driver runs, whatever the execution options say.
     */
    geminiContextCache?: boolean;
    /**
     * Default lifetime, in seconds, of the `cachedContents` resources this driver creates.
     * Defaults to 1800 (30 minutes). `ExecutionOptions.prompt_cache_ttl_seconds` overrides it per call.
     */
    geminiContextCacheTtlSeconds?: number;
    /** Host-supplied fleet coordinator. Llumiverse itself has no Redis dependency. */
    geminiContextCacheCoordinator?: GeminiContextCacheCoordinator;
    /** Host isolation scope, normally the Studio environment ID. */
    geminiContextCacheScope?: string;
}

export interface GeminiContextCacheEntry {
    /** Server-generated resource name, e.g. `projects/p/locations/l/cachedContents/123`. */
    name: string;
    expiresAtMs: number;
}

export interface GeminiContextCacheCoordinationKey {
    /** Studio environment ID or another caller-defined isolation scope. */
    scope?: string;
    project: string;
    location: string;
    model: string;
    contentHash: string;
}

/**
 * Optional fleet coordinator supplied by the host application.
 *
 * Llumiverse deliberately owns no Redis dependency. Studio injects these functions when it creates
 * a Vertex driver; another host can implement the same semantics with its own coordination store.
 * A rejected operation means coordination is unavailable and causes a safe uncached fallback.
 */
export interface GeminiContextCacheCoordinator {
    getEntry(key: GeminiContextCacheCoordinationKey): Promise<GeminiContextCacheEntry | undefined>;
    acquireLease(key: GeminiContextCacheCoordinationKey, leaseMs: number): Promise<string | undefined>;
    waitForEntry(
        key: GeminiContextCacheCoordinationKey,
        timeoutMs: number,
    ): Promise<GeminiContextCacheEntry | undefined>;
    publishEntry(
        key: GeminiContextCacheCoordinationKey,
        leaseToken: string,
        entry: GeminiContextCacheEntry,
        ttlMs: number,
    ): Promise<boolean>;
    releaseLease(key: GeminiContextCacheCoordinationKey, leaseToken: string): Promise<void>;
    invalidateEntry(key: GeminiContextCacheCoordinationKey, expectedName: string): Promise<void>;
    getCooldownUntil(key: GeminiContextCacheCoordinationKey): Promise<number | undefined>;
    setCooldownUntil(key: GeminiContextCacheCoordinationKey, untilMs: number): Promise<void>;
    acquireCreatePermit(
        key: GeminiContextCacheCoordinationKey,
        limit: number,
        leaseMs: number,
        waitMs: number,
    ): Promise<string | undefined>;
    releaseCreatePermit(key: GeminiContextCacheCoordinationKey, permitToken: string): Promise<void>;
}

export interface WatsonxDriverOptions extends DriverOptions {
    apiKey: string;
    projectId: string;
    endpointUrl: string;
}

export interface xAiDriverOptions extends DriverOptions {
    apiKey: string;

    endpoint?: string;
}
