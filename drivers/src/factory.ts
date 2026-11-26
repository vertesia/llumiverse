/**
 * Creates a driver by provider name, loading only that provider's implementation and SDK.
 *
 * Exposed as `@llumiverse/drivers/factory`. Its declarations name only the option types in
 * `driver-options.ts` and the `Driver` interface, so a consumer that creates drivers through it does not
 * type-check the SDK declarations of every provider — which a `typeof import('@llumiverse/drivers/<x>')`
 * or `await import(...)` of each driver subpath would, even when the runtime load is lazy.
 */

import { type Driver, type DriverOptions, isOpenAIApi, type OpenAIApi, resolveOpenAIApi } from '@llumiverse/core';
import type {
    AnthropicDriverOptions,
    AzureFoundryDriverOptions,
    AzureOpenAIDriverOptions,
    BedrockDriverOptions,
    BedrockMantleDriverOptions,
    GroqDriverOptions,
    HuggingFaceIEDriverOptions,
    MistralAIDriverOptions,
    OpenAIApiOption,
    OpenAICompatibleDriverOptions,
    OpenAIDriverOptions,
    OpenRouterDriverOptions,
    ReplicateDriverOptions,
    TogetherAIDriverOptions,
    VertexAIDriverOptions,
    WatsonxDriverOptions,
    xAiDriverOptions,
} from './driver-options.js';

/** The constructor options of each driver {@link createDriver} can create, keyed by provider. */
export interface DriverFactoryOptions {
    anthropic: AnthropicDriverOptions;
    azure_foundry: AzureFoundryDriverOptions;
    azure_openai: AzureOpenAIDriverOptions;
    bedrock: BedrockDriverOptions;
    bedrock_mantle: BedrockMantleDriverOptions;
    groq: GroqDriverOptions & OpenAIApiOption;
    huggingface_ie: HuggingFaceIEDriverOptions;
    mistralai: MistralAIDriverOptions;
    openai: OpenAIDriverOptions;
    openai_compatible: OpenAICompatibleDriverOptions;
    openrouter: OpenRouterDriverOptions & OpenAIApiOption;
    replicate: ReplicateDriverOptions;
    /** The in-process test driver takes no options. */
    test: DriverOptions;
    togetherai: TogetherAIDriverOptions & OpenAIApiOption;
    vertexai: VertexAIDriverOptions;
    watsonx: WatsonxDriverOptions;
    xai: xAiDriverOptions & OpenAIApiOption;
}

export type DriverFactoryProvider = keyof DriverFactoryOptions;

/**
 * Creates the driver of the OpenAI API a provider is configured with: `openai_api`, or the provider's default in
 * `OPENAI_API_DEFAULTS`. Any other value throws rather than silently picking an API.
 */
function selectOpenAIApi(
    provider: DriverFactoryProvider,
    openaiApi: OpenAIApi | undefined,
    drivers: Record<OpenAIApi, () => Promise<Driver>>,
): Promise<Driver> {
    if (openaiApi !== undefined && !isOpenAIApi(openaiApi)) {
        throw new Error(`Unsupported openai_api for the ${provider} driver: ${String(openaiApi)}`);
    }
    const api = resolveOpenAIApi(provider, openaiApi);
    if (!api) throw new Error(`The ${provider} driver has no openai_api option`);
    return drivers[api]();
}

const loaders: { [P in DriverFactoryProvider]: (options: DriverFactoryOptions[P]) => Promise<Driver> } = {
    anthropic: async (options) => new (await import('./anthropic/index.js')).AnthropicDriver(options),
    azure_foundry: async (options) => new (await import('./azure/azure_foundry.js')).AzureFoundryDriver(options),
    azure_openai: async (options) => new (await import('./openai/drivers.js')).AzureOpenAIDriver(options),
    bedrock: async (options) => new (await import('./bedrock/index.js')).BedrockDriver(options),
    bedrock_mantle: async (options) => new (await import('./bedrock-mantle/index.js')).BedrockMantleDriver(options),
    groq: async ({ openai_api, ...options }) =>
        selectOpenAIApi('groq', openai_api, {
            chat_completions: async () => new (await import('./groq/index.js')).GroqDriver(options),
            responses: async () => new (await import('./groq/responses.js')).GroqResponsesDriver(options),
        }),
    huggingface_ie: async (options) => new (await import('./huggingface_ie.js')).HuggingFaceIEDriver(options),
    mistralai: async (options) => new (await import('./mistral/index.js')).MistralAIDriver(options),
    openai: async (options) => new (await import('./openai/drivers.js')).OpenAIDriver(options),
    openai_compatible: async ({ openai_api, ...options }) =>
        selectOpenAIApi('openai_compatible', openai_api, {
            chat_completions: async () =>
                new (await import('./openai/drivers.js')).OpenAIChatCompletionsDriver(options),
            responses: async () => new (await import('./openai/drivers.js')).OpenAIResponsesDriver(options),
        }),
    openrouter: async ({ openai_api, ...options }) =>
        selectOpenAIApi('openrouter', openai_api, {
            chat_completions: async () => new (await import('./openrouter/index.js')).OpenRouterDriver(options),
            responses: async () => new (await import('./openrouter/responses.js')).OpenRouterResponsesDriver(options),
        }),
    replicate: async (options) => new (await import('./replicate.js')).ReplicateDriver(options),
    test: async () => new (await import('./test-driver/index.js')).TestDriver(),
    togetherai: async ({ openai_api, ...options }) =>
        selectOpenAIApi('togetherai', openai_api, {
            chat_completions: async () => new (await import('./togetherai/index.js')).TogetherAIDriver(options),
            responses: async () => new (await import('./togetherai/responses.js')).TogetherAIResponsesDriver(options),
        }),
    vertexai: async (options) => new (await import('./vertexai/index.js')).VertexAIDriver(options),
    watsonx: async (options) => new (await import('./watsonx/index.js')).WatsonxDriver(options),
    xai: async ({ openai_api, ...options }) =>
        selectOpenAIApi('xai', openai_api, {
            chat_completions: async () =>
                new (await import('./xai/chat_completions.js')).xAIChatCompletionsDriver(options),
            responses: async () => new (await import('./xai/index.js')).xAIDriver(options),
        }),
};

/** Whether {@link createDriver} can create a driver for `provider`. */
export function isDriverFactoryProvider(provider: string): provider is DriverFactoryProvider {
    return Object.hasOwn(loaders, provider);
}

/**
 * Creates the driver for `provider`. Only that provider's driver module and SDK are loaded.
 *
 * @throws Error when `provider` is not a {@link DriverFactoryProvider} (reachable from untyped callers).
 */
export async function createDriver<P extends DriverFactoryProvider>(
    provider: P,
    options: DriverFactoryOptions[P],
): Promise<Driver> {
    if (!isDriverFactoryProvider(provider)) {
        throw new Error(`Unknown driver provider: ${provider}`);
    }
    return await loaders[provider](options);
}
