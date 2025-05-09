import { ModelModalities } from "./types.js";
import {
    getInputModalityBedrock,
    getOutputModalityBedrock,
    supportsInputModalityBedrock,
    supportsOutputModalityBedrock
} from "./modalities/bedrock.js";
import {
    getInputModalityVertexAI,
    getOutputModalityVertexAI,
    supportsInputModalityVertexAI,
    supportsOutputModalityVertexAI
} from "./modalities/vertexai.js";
import {
    getInputModalityOpenAI,
    getOutputModalityOpenAI,
    supportsInputModalityOpenAI,
    supportsOutputModalityOpenAI
} from "./modalities/openai.js";

/**
 * Get the input modalities supported by a model across providers
 * 
 * @param provider The LLM provider (e.g., "bedrock", "vertexai", "openai")
 * @param model Model identifier
 * @returns Object containing the supported input modalities
 */
export function getInputModality(provider: string, model: string): ModelModalities {
    const providerLower = provider.toLowerCase();

    switch (providerLower) {
        case "bedrock":
            return getInputModalityBedrock(model);
        case "vertexai":
            return getInputModalityVertexAI(model);
        case "openai":
            return getInputModalityOpenAI(model);
        default:
            // Try to infer provider from model ID if explicit provider is not recognized
            if (model.includes("arn:aws") || model.startsWith("anthropic.") || model.startsWith("amazon.")) {
                return getInputModalityBedrock(model);
            } else if (model.includes("publishers/") && model.includes("/models/")) {
                return getInputModalityVertexAI(model);
            } else if (model.startsWith("gpt-") || model.startsWith("o1") || model.startsWith("o3-") || model.startsWith("o4-")) {
                return getInputModalityOpenAI(model);
            }
            // Default to text-only if provider can't be determined
            return { text: true };
    }
}

/**
 * Get the output modalities supported by a model across providers
 * 
 * @param provider The LLM provider (e.g., "bedrock", "vertexai", "openai") 
 * @param model Model identifier
 * @returns Object containing the supported output modalities
 */
export function getOutputModality(provider: string, model: string): ModelModalities {
    const providerLower = provider.toLowerCase();

    switch (providerLower) {
        case "bedrock":
            return getOutputModalityBedrock(model);
        case "vertexai":
            return getOutputModalityVertexAI(model);
        case "openai":
            return getOutputModalityOpenAI(model);
        default:
            // Try to infer provider from model ID if explicit provider is not recognized
            if (model.includes("arn:aws") || model.startsWith("anthropic.") || model.startsWith("amazon.")) {
                return getOutputModalityBedrock(model);
            } else if (model.includes("publishers/") && model.includes("/models/")) {
                return getOutputModalityVertexAI(model);
            } else if (model.startsWith("gpt-") || model.startsWith("o1") || model.startsWith("o3-") || model.startsWith("o4-")) {
                return getOutputModalityOpenAI(model);
            }
            // Default to text-only if provider can't be determined
            return { text: true };
    }
}

/**
 * Check if a model supports a specific input modality across providers
 * 
 * @param provider The LLM provider (e.g., "bedrock", "vertexai", "openai")
 * @param model Model identifier
 * @param modality The modality to check for
 * @returns Boolean indicating if the model supports the specified input modality
 */
export function supportsInputModality(
    provider: string,
    model: string,
    modality: keyof ModelModalities
): boolean {
    const providerLower = provider.toLowerCase();

    switch (providerLower) {
        case "bedrock":
            return supportsInputModalityBedrock(model, modality);
        case "vertexai":
            return supportsInputModalityVertexAI(model, modality);
        case "openai":
            return supportsInputModalityOpenAI(model, modality);
        default:
            // Try to infer provider from model ID if explicit provider is not recognized
            if (model.includes("arn:aws") || model.startsWith("anthropic.") || model.startsWith("amazon.")) {
                return supportsInputModalityBedrock(model, modality);
            } else if (model.includes("publishers/") && model.includes("/models/")) {
                return supportsInputModalityVertexAI(model, modality);
            } else if (model.startsWith("gpt-") || model.startsWith("o1") || model.startsWith("o3-") || model.startsWith("o4-")) {
                return supportsInputModalityOpenAI(model, modality);
            }
            // Default to supporting only text input if provider can't be determined
            return modality === 'text';
    }
}

/**
 * Check if a model supports a specific output modality across providers
 * 
 * @param provider The LLM provider (e.g., "bedrock", "vertexai", "openai")
 * @param model Model identifier
 * @param modality The modality to check for
 * @returns Boolean indicating if the model supports the specified output modality
 */
export function supportsOutputModality(
    provider: string,
    model: string,
    modality: keyof ModelModalities
): boolean {
    const providerLower = provider.toLowerCase();

    switch (providerLower) {
        case "bedrock":
            return supportsOutputModalityBedrock(model, modality);
        case "vertexai":
            return supportsOutputModalityVertexAI(model, modality);
        case "openai":
            return supportsOutputModalityOpenAI(model, modality);
        default:
            // Try to infer provider from model ID if explicit provider is not recognized
            if (model.includes("arn:aws") || model.startsWith("anthropic.") || model.startsWith("amazon.")) {
                return supportsOutputModalityBedrock(model, modality);
            } else if (model.includes("publishers/") && model.includes("/models/")) {
                return supportsOutputModalityVertexAI(model, modality);
            } else if (model.startsWith("gpt-") || model.startsWith("o1") || model.startsWith("o3-") || model.startsWith("o4-")) {
                return supportsOutputModalityOpenAI(model, modality);
            }
            // Default to supporting only text output if provider can't be determined
            return modality === 'text';
    }
}

export function modelModalitiesToArray(modalities: ModelModalities): string[] {
    return Object.entries(modalities)
        .filter(([_, isSupported]) => isSupported)
        .map(([modality]) => modality);
}