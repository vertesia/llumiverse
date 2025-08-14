import { getBedrockOptions } from "./options/bedrock.js";
import { getGroqOptions } from "./options/groq.js";
import { getOpenAiOptions } from "./options/openai.js";
import { getVertexAiOptions } from "./options/vertexai.js";
import { textOptionsFallback } from "./options/fallback.js";
import { ModelOptionsInfo, ModelOptions, Providers } from "./types.js";
import { getAzureFoundryOptions } from "./options/azure_foundry.js";

export function getOptions(model: string, provider?: string | Providers, options?: ModelOptions): ModelOptionsInfo {
    if(!provider) {
        return textOptionsFallback;
    }
    switch (provider.toLowerCase()) {
        case Providers.bedrock:
            return getBedrockOptions(model, options);
        case Providers.vertexai:
            return getVertexAiOptions(model, options);
        case Providers.openai:
            return getOpenAiOptions(model, options);
        case Providers.groq:
            return getGroqOptions(model, options);
        case Providers.azure_foundry:
            return getAzureFoundryOptions(model, options);
        default:
            return textOptionsFallback;
    }
}
