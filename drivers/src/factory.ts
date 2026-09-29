/**
 * Creates a driver by provider name, loading only that provider's implementation and SDK.
 *
 * Exposed as `@llumiverse/drivers/factory`. Its declarations name only the option types in
 * `driver-options.ts` and the `Driver` interface, so a consumer that creates drivers through it does not
 * type-check the SDK declarations of every provider — which a `typeof import('@llumiverse/drivers/<x>')`
 * or `await import(...)` of each driver subpath would, even when the runtime load is lazy.
 */

import type { Driver, DriverOptions } from '@llumiverse/core';
import type {
    AnthropicDriverOptions,
    AzureFoundryDriverOptions,
    AzureOpenAIDriverOptions,
    BedrockDriverOptions,
    BedrockMantleDriverOptions,
    GroqDriverOptions,
    HuggingFaceIEDriverOptions,
    MistralAIDriverOptions,
    OpenAIDriverOptions,
    OpenAIResponsesDriverOptions,
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
    groq: GroqDriverOptions;
    huggingface_ie: HuggingFaceIEDriverOptions;
    mistralai: MistralAIDriverOptions;
    openai: OpenAIDriverOptions;
    openai_compatible: OpenAIResponsesDriverOptions;
    openrouter: OpenRouterDriverOptions;
    replicate: ReplicateDriverOptions;
    /** The in-process test driver takes no options. */
    test: DriverOptions;
    togetherai: TogetherAIDriverOptions;
    vertexai: VertexAIDriverOptions;
    watsonx: WatsonxDriverOptions;
    xai: xAiDriverOptions;
}

export type DriverFactoryProvider = keyof DriverFactoryOptions;

const loaders: { [P in DriverFactoryProvider]: (options: DriverFactoryOptions[P]) => Promise<Driver> } = {
    anthropic: async (options) => new (await import('./anthropic/index.js')).AnthropicDriver(options),
    azure_foundry: async (options) => new (await import('./azure/azure_foundry.js')).AzureFoundryDriver(options),
    azure_openai: async (options) => new (await import('./openai/drivers.js')).AzureOpenAIDriver(options),
    bedrock: async (options) => new (await import('./bedrock/index.js')).BedrockDriver(options),
    bedrock_mantle: async (options) => new (await import('./bedrock-mantle/index.js')).BedrockMantleDriver(options),
    groq: async (options) => new (await import('./groq/index.js')).GroqDriver(options),
    huggingface_ie: async (options) => new (await import('./huggingface_ie.js')).HuggingFaceIEDriver(options),
    mistralai: async (options) => new (await import('./mistral/index.js')).MistralAIDriver(options),
    openai: async (options) => new (await import('./openai/drivers.js')).OpenAIDriver(options),
    openai_compatible: async (options) => new (await import('./openai/drivers.js')).OpenAIResponsesDriver(options),
    openrouter: async (options) => new (await import('./openrouter/index.js')).OpenRouterDriver(options),
    replicate: async (options) => new (await import('./replicate.js')).ReplicateDriver(options),
    test: async () => new (await import('./test-driver/index.js')).TestDriver(),
    togetherai: async (options) => new (await import('./togetherai/index.js')).TogetherAIDriver(options),
    vertexai: async (options) => new (await import('./vertexai/index.js')).VertexAIDriver(options),
    watsonx: async (options) => new (await import('./watsonx/index.js')).WatsonxDriver(options),
    xai: async (options) => new (await import('./xai/index.js')).xAIDriver(options),
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
