import { getBedrockOptions } from "./options/bedrock.js";
import { textOptionsFallback } from "./options/fallback.js";
import { getGroqOptions } from "./options/groq.js";
import { getOpenAiOptions } from "./options/openai.js";
import { getVertexAiOptions } from "./options/vertexai.js";
import { ModelOptions, ModelOptionsInfo } from "./types.js";

//Export types from providers
export * from "./options/bedrock.js";
export * from "./options/fallback.js";
export * from "./options/groq.js";
export * from "./options/openai.js";
export * from "./options/vertexai.js";


export function getOptions(model: string, provider?: string, options?: ModelOptions): ModelOptionsInfo {
    switch (provider?.toLowerCase()) {
        case "bedrock":
            return getBedrockOptions(model ?? "", options);
        case "vertexai":
            return getVertexAiOptions(model ?? "", options);
        case "openai":
            return getOpenAiOptions(model ?? "", options);
        case "groq":
            return getGroqOptions(model ?? "", options);
        default:
            return textOptionsFallback;
    }
}
