import { supportsToolUseBedrock } from "./tools/bedrock.js";
import { supportsToolUseVertexAI } from "./tools/vertexai.js";
import { supportsToolUseOpenAI } from "./tools/openai.js";

/**
 * Determines if a specified provider and model combination supports tool use
 * @param provider The LLM provider (e.g., "bedrock", "vertexai", "openai")
 * @param model Optional model identifier. If not provided, will attempt to infer from model ID, not to be relied on.
 * @param streaming Optional flag indicating if streaming mode is being used
 * @returns True if tool use is supported, false if not supported, undefined if unknown
 */
export function supportsToolUse(provider: string, model?: string, streaming?: boolean): boolean | undefined {
    // If no model is provided, use an empty string as fallback
    const modelId = model ?? "";

    switch (provider.toLowerCase()) {
        case "bedrock":
            return supportsToolUseBedrock(modelId, streaming);
        case "vertexai":
            return supportsToolUseVertexAI(modelId, streaming);
        case "openai":
            return supportsToolUseOpenAI(modelId, streaming);
        default:
            // Try to infer provider from model ID if explicit provider is not recognized
            if (modelId.includes("arn:aws") || modelId.startsWith("anthropic.") || modelId.startsWith("amazon.")) {
                return supportsToolUseBedrock(modelId, streaming);
            } else if (modelId.includes("publishers/") && modelId.includes("/models/")) {
                return supportsToolUseVertexAI(modelId, streaming);
            } else if (modelId.startsWith("gpt-")) {
                return supportsToolUseOpenAI(modelId, streaming);
            }
            return undefined;
    }
}