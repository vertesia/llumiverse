import { getBedrockOptions } from "./options/bedrock.js";
import { getGroqOptions } from "./options/groq.js";
import { getOpenAiOptions } from "./options/openai.js";
import { getVertexAiOptions } from "./options/vertexai.js";
import { textOptionsFallback } from "./options/fallback.js";
import { ModelOptionsInfo, ModelOptions } from "./types.js";

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
