import { getModelCapabilitiesBedrock } from "./capability/bedrock.js";
import { getModelCapabilitiesOpenAI } from "./capability/openai.js";
import { getModelCapabilitiesVertexAI } from "./capability/vertexai.js";
import { ModelCapabilities, ModelModalities } from "./types.js";

export function getModelCapabilities(model: string, provider?: string): ModelCapabilities {
    switch (provider?.toLowerCase()) {
        case "vertexai":
            return getModelCapabilitiesVertexAI(model);
        case "openai":
            return getModelCapabilitiesOpenAI(model);
        case "bedrock":
            return getModelCapabilitiesBedrock(model);
        default:
            throw new Error(`Unsupported provider: ${provider}`);
    }
}

export function supportsToolUse(model: string, provider?: string, streaming?: boolean): boolean {
    const capabilities = getModelCapabilities(model, provider);
    return streaming ? !!capabilities.tool_support : !!capabilities.tool_support_streaming;
}

export function modelModalitiesToArray(modalities: ModelModalities): string[] {
    return Object.entries(modalities)
        .filter(([_, isSupported]) => isSupported)
        .map(([modality]) => modality);
}