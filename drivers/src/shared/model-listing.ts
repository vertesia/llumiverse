import type { ModelLifecycleDates, Providers } from '@llumiverse/core';
import {
    canSendInputModality,
    getModelCapabilities,
    getModelLifecycleDates,
    modelModalitiesToArray,
    resolveModelProfile,
    toLifecycleDate,
} from '@llumiverse/core';

export interface RuntimeModelListingMetadata {
    input_modalities?: readonly string[];
    output_modalities?: readonly string[];
    /** Lifecycle dates reported by the provider's listing (`YYYY-MM-DD`, RFC 3339 or `Date`). */
    deprecation_date?: string | Date | null;
    retirement_date?: string | Date | null;
    /** Model version, for providers that list a model name and a version separately. */
    version?: string;
}

export interface ResolvedModelListingMetadata extends ModelLifecycleDates {
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

/**
 * Whether a catalog rule describes the model's capabilities. Effort levels alone do not: they say nothing about
 * modalities or tools, so a generic-family model with only curated effort keeps the provider's listing metadata.
 */
function hasCatalogInformation(profile: ReturnType<typeof resolveModelProfile>): boolean {
    return (
        profile.family !== 'generic' ||
        profile.context_window !== undefined ||
        profile.max_output_tokens !== undefined ||
        profile.capabilities.tool_support !== undefined ||
        profile.capabilities.tool_support_streaming !== undefined
    );
}

/**
 * Lifecycle dates of a listed model. Dates the provider's listing reports win; the curated table fills in for
 * providers whose listing has none. Each date is resolved on its own, so a provider that reports only a retirement
 * date keeps the table's deprecation date. Keys are omitted when no date is known.
 */
export function resolveModelLifecycleDates(
    model: string,
    provider: Providers,
    runtime: Pick<RuntimeModelListingMetadata, 'deprecation_date' | 'retirement_date' | 'version'> = {},
): ModelLifecycleDates {
    const curated = getModelLifecycleDates(model, provider, runtime.version);
    const deprecation_date = toLifecycleDate(runtime.deprecation_date) ?? curated?.deprecation_date;
    const retirement_date = toLifecycleDate(runtime.retirement_date) ?? curated?.retirement_date;
    return {
        ...(deprecation_date ? { deprecation_date } : {}),
        ...(retirement_date ? { retirement_date } : {}),
    };
}

/**
 * Catalog metadata is curated and therefore authoritative as a complete set. Runtime listing metadata is used only
 * when no catalog rule matched; mixing the two lets incomplete or misleading provider data erase known capabilities.
 * Lifecycle dates are the exception: they are dated facts, so a date reported by the provider always wins.
 */
export function resolveModelListingMetadata(
    model: string,
    provider: Providers,
    runtime: RuntimeModelListingMetadata = {},
): ResolvedModelListingMetadata {
    const lifecycle = resolveModelLifecycleDates(model, provider, runtime);
    const profile = resolveModelProfile(model, provider);
    if (hasCatalogInformation(profile)) {
        const capabilities = getModelCapabilities(model, provider);
        return {
            input_modalities: modelModalitiesToArray(capabilities.input),
            output_modalities: modelModalitiesToArray(capabilities.output),
            tool_support: capabilities.tool_support,
            ...lifecycle,
        };
    }

    // Unknown models stay usable. Prefer explicit provider metadata, then use conservative text inference defaults.
    // Provider metadata describes the model, so inputs the driver cannot send are dropped as for catalog models.
    return {
        input_modalities: (normalizeRuntimeModalities(runtime.input_modalities) ?? ['text']).filter((modality) =>
            canSendInputModality(provider, modality),
        ),
        output_modalities: normalizeRuntimeModalities(runtime.output_modalities) ?? ['text'],
        ...lifecycle,
    };
}
