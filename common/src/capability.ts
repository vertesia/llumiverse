import { isModelDirectoryEmbedding, resolveModelProfile } from './model-directory.js';
import type { ModelCapabilities, ModelModalities, Providers } from './types.js';

export function getModelCapabilities(model: string, provider: Providers): ModelCapabilities {
    //Check for locations/<location>/ prefix and remove it
    if (model.startsWith('locations/')) {
        const parts = model.split('/');
        if (parts.length >= 3) {
            model = parts.slice(2).join('/');
        }
    }
    const capabilities = resolveModelProfile(model, provider).capabilities;
    // The platform accepts audio/video inputs but cannot return those modalities yet. Keep source output metadata in
    // the directory so enabling output support later only requires removing this execution-path mask.
    return {
        ...capabilities,
        input: { ...capabilities.input },
        output: { ...capabilities.output, audio: false, video: false },
    };
}

export function supportsToolUse(model: string, provider: Providers, streaming: boolean = false): boolean {
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
export function isEmbeddingModel(model: ModelListing, provider: Providers): boolean {
    if (model.type?.toLowerCase() === 'embedding') return true;
    if (isModelDirectoryEmbedding(model.id)) return true;

    const modalities = [...(model.input_modalities ?? []), ...(model.output_modalities ?? [])].map((modality) =>
        modality.toLowerCase(),
    );
    if (modalities.some((modality) => modality === 'embed' || modality === 'embedding' || modality === 'vectors')) {
        return true;
    }

    return getModelCapabilities(model.id, provider).output.embed === true;
}
