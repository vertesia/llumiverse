import type { Providers } from '@llumiverse/core';
import { getModelCapabilities, modelModalitiesToArray, resolveModelProfile } from '@llumiverse/core';

export interface RuntimeModelListingMetadata {
    input_modalities?: readonly string[];
    output_modalities?: readonly string[];
}

export interface ResolvedModelListingMetadata {
    input_modalities: string[];
    output_modalities: string[];
    tool_support?: boolean;
}

function normalizeRuntimeModalities(modalities: readonly string[] | undefined): string[] | undefined {
    if (!modalities) return undefined;
    return modalities.map((modality) => {
        const normalized = modality.toLowerCase();
        if (normalized === 'embedding' || normalized === 'vectors') return 'embed';
        if (normalized === 'speech') return 'audio';
        return normalized;
    });
}

function hasCatalogInformation(profile: ReturnType<typeof resolveModelProfile>): boolean {
    return (
        profile.family !== 'generic' ||
        profile.context_window !== undefined ||
        profile.max_output_tokens !== undefined ||
        profile.reasoning_effort_levels !== undefined ||
        profile.capabilities.tool_support !== undefined ||
        profile.capabilities.tool_support_streaming !== undefined
    );
}

/**
 * Catalog metadata is curated and therefore authoritative as a complete set. Runtime listing metadata is used only
 * when no catalog rule matched; mixing the two lets incomplete or misleading provider data erase known capabilities.
 */
export function resolveModelListingMetadata(
    model: string,
    provider: Providers,
    runtime: RuntimeModelListingMetadata = {},
): ResolvedModelListingMetadata {
    const profile = resolveModelProfile(model, provider);
    if (hasCatalogInformation(profile)) {
        const capabilities = getModelCapabilities(model, provider);
        return {
            input_modalities: modelModalitiesToArray(capabilities.input),
            output_modalities: modelModalitiesToArray(capabilities.output),
            tool_support: capabilities.tool_support,
        };
    }

    // Unknown models stay usable. Prefer explicit provider metadata, then use conservative text inference defaults.
    return {
        input_modalities: normalizeRuntimeModalities(runtime.input_modalities) ?? ['text'],
        output_modalities: normalizeRuntimeModalities(runtime.output_modalities) ?? ['text'],
    };
}
