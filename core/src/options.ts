import {
    getBedrockOptions,
    textOptionsFallback,
    getGroqOptions,
    getOpenAiOptions,
    getVertexAiOptions,
    ModelOptions,
    ModelOptionsInfo
} from "@llumiverse/common";

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
