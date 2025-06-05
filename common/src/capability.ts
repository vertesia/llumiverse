import { getModelCapabilitiesBedrock } from "./capability/bedrock.js";
import { getModelCapabilitiesOpenAI } from "./capability/openai.js";
import { getModelCapabilitiesVertexAI } from "./capability/vertexai.js";
import { ModelCapabilities, ModelModalities } from "./types.js";

export function getModelCapabilities(model: string, provider?: string): ModelCapabilities {
    const capabilities = _getModelCapabilities(model, provider);
    // Globally disable audio and video for all models, as we don't support them yet
    // We also do not support tool use while streaming
    // TODO: Remove this when we add support.
    capabilities.input.audio = false;
    capabilities.output.audio = false;
    capabilities.output.video = false;
    capabilities.tool_support_streaming = false;
    return capabilities;
}

function _getModelCapabilities(model: string, provider?: string): ModelCapabilities {
    switch (provider?.toLowerCase()) {
        case "vertexai":
            return getModelCapabilitiesVertexAI(model);
        case "openai":
            return getModelCapabilitiesOpenAI(model);
        case "bedrock":
            return getModelCapabilitiesBedrock(model);
        default:
            // Guess the provider based on the model name
            if (model.startsWith("gpt")) {
                return getModelCapabilitiesOpenAI(model);
            } else if (model.startsWith("publishers/")) {
                return getModelCapabilitiesVertexAI(model);
            } else if (model.startsWith("arn:aws")) {
                return getModelCapabilitiesBedrock(model);
            }
            // Fallback to a generic model with no capabilities
            return { input: {}, output: {} } satisfies ModelCapabilities;
    }
}

export function supportsToolUse(model: string, provider?: string, streaming: boolean = false): boolean {
    const capabilities = getModelCapabilities(model, provider);
    return streaming ? !!capabilities.tool_support_streaming : !!capabilities.tool_support;
}

export function modelModalitiesToArray(modalities: ModelModalities): string[] {
    return Object.entries(modalities)
        .filter(([_, isSupported]) => isSupported)
        .map(([modality]) => modality);
}