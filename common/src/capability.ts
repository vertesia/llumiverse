import { getModelCapabilitiesAzureFoundry } from "./capability/azure_foundry.js";
import { getModelCapabilitiesBedrock } from "./capability/bedrock.js";
import { getModelCapabilitiesOpenAI } from "./capability/openai.js";
import { getModelCapabilitiesVertexAI } from "./capability/vertexai.js";
import { ModelCapabilities, ModelModalities, Providers } from "./types.js";

export function getModelCapabilities(model: string, provider?: string | Providers): ModelCapabilities {
    //Check for locations/<location>/ prefix and remove it
    if (model.startsWith("locations/")) {
        const parts = model.split("/");
        if (parts.length >= 3) {
            model = parts.slice(2).join("/");
        }
    }
    const capabilities = _getModelCapabilities(model, provider);
    // Globally disable audio and video for all models, as we don't support them yet
    // TODO: Remove this when we add support.
    capabilities.input.audio = false;
    capabilities.output.audio = false;
    capabilities.output.video = false;
    // Preserve tool_support_streaming from provider-specific capabilities if set,
    // otherwise default to false for providers that haven't been verified
    return capabilities;
}

function _getModelCapabilities(model: string, provider?: string | Providers): ModelCapabilities {
    switch (provider?.toLowerCase()) {
        case Providers.vertexai:
            return getModelCapabilitiesVertexAI(model);
        case Providers.openai:
        case Providers.openai_compatible:
            return getModelCapabilitiesOpenAI(model);
        case Providers.bedrock:
            return getModelCapabilitiesBedrock(model);
        case Providers.azure_foundry:
            // Azure Foundry uses OpenAI capabilities
            return getModelCapabilitiesAzureFoundry(model);
        case Providers.xai:
            // xAI (Grok) models support tool use and are text-based
            return {
                input: { text: true, image: model.includes("vision") },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: false, // Conservative - may work but not tested
            };
        default:
            // Guess the provider based on the model name
            if (model.startsWith("gpt")) {
                return getModelCapabilitiesOpenAI(model);
            } else if (model.startsWith("grok")) {
                // xAI Grok models
                return {
                    input: { text: true, image: model.includes("vision") },
                    output: { text: true },
                    tool_support: true,
                    tool_support_streaming: false,
                };
            } else if (model.startsWith("publishers/")) {
                return getModelCapabilitiesVertexAI(model);
            } else if (model.startsWith("arn:aws")) {
                return getModelCapabilitiesBedrock(model);
            }
            // Fallback to a generic model with no capabilities
            return { input: {}, output: {} } satisfies ModelCapabilities;
    }
}

export function supportsToolUse(model: string, provider?: string | Providers, streaming: boolean = false): boolean {
    const capabilities = getModelCapabilities(model, provider);
    return streaming ? !!capabilities.tool_support_streaming : !!capabilities.tool_support;
}

export function modelModalitiesToArray(modalities: ModelModalities): string[] {
    return Object.entries(modalities)
        .filter(([_, isSupported]) => isSupported)
        .map(([modality]) => modality);
}