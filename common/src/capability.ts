import { getModelCapabilitiesAzureFoundry } from './capability/azure_foundry.js';
import { isModelDirectoryEmbedding, resolveModelProfile } from './model-directory.js';
import { type ModelCapabilities, type ModelModalities, Providers } from './types.js';

export function getModelCapabilities(model: string, provider?: string | Providers): ModelCapabilities {
    //Check for locations/<location>/ prefix and remove it
    if (model.startsWith('locations/')) {
        const parts = model.split('/');
        if (parts.length >= 3) {
            model = parts.slice(2).join('/');
        }
    }
    const capabilities = _getModelCapabilities(model, provider);
    // Globally disable audio and video for all models, as we don't support them yet
    // TODO: Remove this when we add support.
    capabilities.input.audio = false;
    capabilities.output.audio = false;
    capabilities.output.video = false;
    // tool_support_streaming is optional: when omitted, supportsToolUse falls back to tool_support.
    // Only set it explicitly to false for models that can tool-call but not while streaming.
    return capabilities;
}

function _getModelCapabilities(model: string, provider?: string | Providers): ModelCapabilities {
    switch (provider?.toLowerCase()) {
        case Providers.anthropic:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.vertexai:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.openai:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.azure_openai:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.openai_compatible:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.bedrock:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.bedrock_mantle:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.azure_foundry:
            // Azure Foundry uses OpenAI capabilities
            return getModelCapabilitiesAzureFoundry(model);
        case Providers.groq:
        case Providers.mistralai:
            // These providers host text models that generally support tool use
            return resolveModelProfile(model, provider).capabilities;
        case Providers.togetherai:
            return resolveModelProfile(model, provider).capabilities;
        case Providers.xai:
            // xAI (Grok) uses the OpenAI Responses API; tool use matches OpenAI-compatible defaults.
            // Do not set tool_support_streaming — leave it unset so it defaults to tool_support.
            return resolveModelProfile(model, provider).capabilities;
        default:
            return resolveModelProfile(model, provider).capabilities;
    }
}

export function supportsToolUse(model: string, provider?: string | Providers, streaming: boolean = false): boolean {
    const capabilities = getModelCapabilities(model, provider);
    if (!streaming) {
        return !!capabilities.tool_support;
    }
    // Unset tool_support_streaming means "same as tool_support" (OpenAI-compatible default).
    // Only an explicit false opts out of tools-while-streaming.
    return !!(capabilities.tool_support_streaming ?? capabilities.tool_support);
}

export function modelModalitiesToArray(modalities: ModelModalities): string[] {
    return Object.entries(modalities)
        .filter(([_, isSupported]) => isSupported)
        .map(([modality]) => modality);
}

export interface ModelListing {
    id: string;
    type?: string;
    input_modalities?: readonly string[];
    output_modalities?: readonly string[];
}

/** Embedding endpoints are not executable through the standard inference path. */
export function isEmbeddingModel(model: ModelListing, provider?: string | Providers): boolean {
    if (isModelDirectoryEmbedding(model.id, model)) return true;

    const modalities = [...(model.input_modalities ?? []), ...(model.output_modalities ?? [])].map((modality) =>
        modality.toLowerCase(),
    );
    if (modalities.some((modality) => modality === 'embed' || modality === 'embedding' || modality === 'vectors')) {
        return true;
    }

    return getModelCapabilities(model.id, provider).output.embed === true;
}
