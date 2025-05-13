import { supportsToolUseBedrock } from "./tools/bedrock.js";
import { supportsToolUseVertexAI } from "./tools/vertexai.js";
import { supportsToolUseOpenAI } from "./tools/openai.js";

/**
 * Determines if a specified provider and model combination supports tool use
 * @param model model identifier. 
 * @param provider The LLM provider (e.g., "bedrock", "vertexai", "openai"). If not provided, will attempt to infer from model ID, not to be relied on.
 * @param streaming Optional flag indicating if streaming mode is being used
 * @returns True if tool use is supported, false if not supported, undefined if unknown
 */
export function supportsToolUse(model: string, provider?: string, streaming?: boolean): boolean | undefined {
    switch (provider?.toLowerCase()) {
        case "bedrock":
            return supportsToolUseBedrock(model, streaming);
        case "vertexai":
            return supportsToolUseVertexAI(model, streaming);
        case "openai":
            return supportsToolUseOpenAI(model, streaming);
        default:
            // Try to infer provider from model ID if explicit provider is not recognized
            if (model.includes("arn:aws") || model.startsWith("anthropic.") || model.startsWith("amazon.")) {
                return supportsToolUseBedrock(model, streaming);
            } else if (model.includes("publishers/") && model.includes("/models/")) {
                return supportsToolUseVertexAI(model, streaming);
            } else if (model.startsWith("gpt-")) {
                return supportsToolUseOpenAI(model, streaming);
            }
            return undefined;
    }
}