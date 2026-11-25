import { Providers } from '../types.js';

/**
 * Announced deprecation and retirement dates of a model, as `YYYY-MM-DD` calendar dates.
 *
 * `deprecation_date` is when the provider deprecated the model or moved it to legacy; `retirement_date` is when
 * requests to it stop being served (the provider's shutdown, retirement or end-of-life date).
 */
export interface ModelLifecycleDates {
    deprecation_date?: string;
    retirement_date?: string;
}

export interface ModelLifecycleEntry extends ModelLifecycleDates {
    /** Model ID as the provider lists it, lowercased. */
    id: string;
    /** Model version, for providers that list a model name and a version separately (Microsoft Foundry). */
    version?: string;
    retirement_date: string;
}

// Curated dates for providers whose model-listing API returns no lifecycle data. Anthropic (`retires_at`),
// OpenAI (`shutdown_date`), Amazon Bedrock (`modelLifecycle`) and OpenRouter (`expiration_date`) report their dates
// when listing models; those drivers read them and use this table only as a fallback.
//
// Amazon Bedrock has no entries: its lifecycle dates differ by region, and the regional `ListFoundationModels`
// response is the only source that knows which region a listing is for. Bedrock Mantle lists no dates.
//
// Rules for entries:
// - Only dates the provider announced. Tentative "no sooner than" dates are left out: they are floors, not dates.
// - Exact model IDs, because a retirement date belongs to one model version; this table never decides whether a
//   model is supported, it only annotates listings.
// - Entries whose retirement date passed before 2026-10-01 are left out. Remove an entry a few months after its
//   retirement date has passed.
// Last checked against the sources below on 2026-11-16.

// https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
// https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deprecations/partner-models
const VERTEXAI_LIFECYCLE: readonly ModelLifecycleEntry[] = [
    { id: 'gemini-2.5-pro', retirement_date: '2026-10-20' },
    { id: 'gemini-2.5-flash', retirement_date: '2026-10-20' },
    { id: 'gemini-2.5-flash-lite', retirement_date: '2026-10-20' },
    { id: 'gemini-3.6-flash', retirement_date: '2026-11-19' },
    { id: 'gemini-live-2.5-flash-native-audio', retirement_date: '2026-12-13' },
    { id: 'gemini-3.7-flash', retirement_date: '2027-01-28' },
    { id: 'gemini-2.5-flash-image', retirement_date: '2027-03-15' },
    { id: 'text-embedding-004', retirement_date: '2027-04-01' },
    { id: 'text-embedding-005', retirement_date: '2027-04-01' },
    { id: 'text-multilingual-embedding-002', retirement_date: '2027-04-01' },
    { id: 'multimodalembedding@001', retirement_date: '2027-04-01' },
    { id: 'claude-sonnet-4-5', deprecation_date: '2026-09-29', retirement_date: '2026-11-30' },
];

// https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-retirement-schedule
// Azure OpenAI and Microsoft Foundry models. Fine-tuned model dates and rows the page lists twice with different
// dates (gpt-realtime-mini) are left out.
const AZURE_LIFECYCLE: readonly ModelLifecycleEntry[] = [
    { id: 'claude-fable-5', retirement_date: '2027-12-05' },
    { id: 'claude-fable-5-1', retirement_date: '2027-12-05' },
    { id: 'claude-haiku-4-5', retirement_date: '2027-02-22' },
    { id: 'claude-mythos-5-1', retirement_date: '2027-09-01' },
    { id: 'claude-mythos-preview', retirement_date: '2027-04-02' },
    { id: 'claude-opus-4-5', retirement_date: '2027-02-22' },
    { id: 'claude-opus-4-6', retirement_date: '2027-02-02' },
    { id: 'claude-opus-4-7', retirement_date: '2027-04-06' },
    { id: 'claude-opus-4-8', retirement_date: '2027-09-01' },
    { id: 'claude-opus-5', retirement_date: '2027-07-08' },
    { id: 'claude-opus-5-5-fast', retirement_date: '2027-01-15' },
    { id: 'claude-sonnet-4-5', retirement_date: '2026-11-30' },
    { id: 'claude-sonnet-4-6', retirement_date: '2027-02-10' },
    { id: 'claude-sonnet-5', retirement_date: '2027-06-30' },
    { id: 'codestral-2501', retirement_date: '2027-02-15' },
    { id: 'codex-mini', version: '2025-05-16', retirement_date: '2026-11-17' },
    { id: 'cohere-command-a', retirement_date: '2027-02-15' },
    { id: 'cohere-command-a-plus-05-2026', retirement_date: '2026-10-16' },
    { id: 'cohere-embed-v3-english', deprecation_date: '2026-11-15', retirement_date: '2027-02-15' },
    { id: 'cohere-embed-v3-multilingual', deprecation_date: '2026-11-15', retirement_date: '2027-02-15' },
    { id: 'cohere-parse-v5', retirement_date: '2026-12-15' },
    { id: 'cohere-rerank-v4.0-fast', retirement_date: '2027-09-13' },
    { id: 'cohere-rerank-v4.0-pro', retirement_date: '2027-09-13' },
    { id: 'deepseek-v3.2', retirement_date: '2027-11-01' },
    { id: 'deepseek-v3.2-speciale', retirement_date: '2027-11-01' },
    { id: 'deepseek-v4-flash', version: '2026-04-23', retirement_date: '2028-02-20' },
    { id: 'deepseek-v4-flash-0731', version: '2026-07-31', retirement_date: '2026-12-03' },
    { id: 'deepseek-v4-pro', version: '2026-04-23', retirement_date: '2028-02-20' },
    { id: 'embed-v-4-0', retirement_date: '2027-06-01' },
    { id: 'flux-1.1-pro', retirement_date: '2027-01-05' },
    { id: 'flux.1-kontext-pro', retirement_date: '2027-01-05' },
    { id: 'flux.2-flex', retirement_date: '2027-01-20' },
    { id: 'flux.2-pro', retirement_date: '2027-01-20' },
    { id: 'gpt-4.1', version: '2025-04-14', retirement_date: '2027-04-14' },
    { id: 'gpt-4.1-mini', version: '2025-04-14', retirement_date: '2027-04-14' },
    { id: 'gpt-4.1-nano', version: '2025-04-14', retirement_date: '2027-04-14' },
    { id: 'gpt-4o', version: '2024-05-13', retirement_date: '2027-01-14' },
    { id: 'gpt-4o', version: '2024-08-06', retirement_date: '2027-04-14' },
    { id: 'gpt-4o', version: '2024-11-20', retirement_date: '2027-04-14' },
    { id: 'gpt-4o-mini', version: '2024-07-18', retirement_date: '2027-04-14' },
    { id: 'gpt-4o-mini-transcribe', version: '2025-03-20', retirement_date: '2026-10-15' },
    { id: 'gpt-4o-mini-transcribe', version: '2025-12-15', retirement_date: '2027-06-15' },
    { id: 'gpt-4o-mini-tts', version: '2025-03-20', retirement_date: '2026-10-15' },
    { id: 'gpt-4o-mini-tts', version: '2025-12-15', retirement_date: '2027-06-15' },
    { id: 'gpt-4o-transcribe', version: '2025-03-20', retirement_date: '2026-10-15' },
    { id: 'gpt-4o-transcribe-diarize', version: '2025-10-15', retirement_date: '2027-04-15' },
    { id: 'gpt-5', version: '2025-08-07', retirement_date: '2027-02-09' },
    { id: 'gpt-5-codex', version: '2025-09-15', retirement_date: '2027-03-17' },
    { id: 'gpt-5-mini', version: '2025-08-07', retirement_date: '2027-07-13' },
    { id: 'gpt-5-nano', version: '2025-08-07', retirement_date: '2027-02-09' },
    { id: 'gpt-5-pro', version: '2025-10-06', retirement_date: '2027-04-07' },
    { id: 'gpt-5.1', version: '2025-11-13', retirement_date: '2027-07-13' },
    { id: 'gpt-5.1-codex', version: '2025-11-13', retirement_date: '2027-05-15' },
    { id: 'gpt-5.1-codex-max', version: '2025-12-04', retirement_date: '2027-05-18' },
    { id: 'gpt-5.1-codex-mini', version: '2025-11-13', retirement_date: '2027-05-15' },
    { id: 'gpt-5.2', version: '2025-12-11', retirement_date: '2027-06-08' },
    { id: 'gpt-5.2-codex', version: '2026-01-14', retirement_date: '2027-07-13' },
    { id: 'gpt-5.3-codex', version: '2026-02-24', retirement_date: '2027-08-24' },
    { id: 'gpt-5.4', version: '2026-03-05', retirement_date: '2027-09-02' },
    { id: 'gpt-5.4-mini', version: '2026-03-17', retirement_date: '2027-09-21' },
    { id: 'gpt-5.4-nano', version: '2026-03-17', retirement_date: '2027-09-21' },
    { id: 'gpt-5.4-pro', version: '2026-03-05', retirement_date: '2027-09-07' },
    { id: 'gpt-5.5', version: '2026-04-24', retirement_date: '2027-10-26' },
    { id: 'gpt-5.6-luna', version: '2026-07-09', retirement_date: '2028-01-11' },
    { id: 'gpt-5.6-sol', version: '2026-07-09', retirement_date: '2028-01-11' },
    { id: 'gpt-5.6-terra', version: '2026-07-09', retirement_date: '2028-01-11' },
    { id: 'gpt-audio', version: '2025-08-28', retirement_date: '2027-03-02' },
    { id: 'gpt-audio-1.5', version: '2026-02-23', retirement_date: '2027-08-24' },
    { id: 'gpt-audio-mini', version: '2025-10-06', retirement_date: '2027-04-06' },
    { id: 'gpt-audio-mini', version: '2025-12-15', retirement_date: '2027-06-15' },
    { id: 'gpt-chat-latest', version: '2026-05-05', retirement_date: '2026-11-17' },
    { id: 'gpt-chat-latest', version: '2026-05-28', retirement_date: '2026-12-08' },
    { id: 'gpt-chat-latest', version: '2026-06-24', retirement_date: '2026-11-17' },
    { id: 'gpt-chat-latest', version: '2026-08-06', retirement_date: '2027-01-12' },
    { id: 'gpt-chat-latest', version: '2026-10-07', retirement_date: '2027-02-02' },
    { id: 'gpt-image-1', version: '2025-04-15', retirement_date: '2026-10-23' },
    { id: 'gpt-image-1-mini', version: '2025-10-06', retirement_date: '2027-04-07' },
    { id: 'gpt-image-1.5', version: '2025-12-16', retirement_date: '2026-12-16' },
    { id: 'gpt-image-2', version: '2026-04-21', retirement_date: '2027-10-21' },
    { id: 'gpt-image-2.5-flare', version: '2026-09-09', retirement_date: '2027-09-09' },
    { id: 'gpt-image-2.5-sunburst', version: '2026-09-09', retirement_date: '2027-09-09' },
    { id: 'gpt-oss-120b', retirement_date: '2027-09-15' },
    { id: 'gpt-realtime', version: '2025-08-28', retirement_date: '2027-03-02' },
    { id: 'gpt-realtime-1.5', version: '2026-02-23', retirement_date: '2027-08-24' },
    { id: 'gpt-realtime-2', version: '2026-05-06', retirement_date: '2027-01-20' },
    { id: 'gpt-realtime-2.1', version: '2026-07-07', retirement_date: '2027-06-25' },
    { id: 'gpt-realtime-2.1-mini', version: '2026-07-07', retirement_date: '2027-06-25' },
    { id: 'grok-4', retirement_date: '2027-01-31' },
    { id: 'grok-4-1-fast-non-reasoning', retirement_date: '2027-05-19' },
    { id: 'grok-4-1-fast-reasoning', retirement_date: '2027-05-19' },
    { id: 'grok-4-20-non-reasoning', retirement_date: '2027-04-06' },
    { id: 'grok-4-20-reasoning', retirement_date: '2027-04-06' },
    { id: 'grok-code-fast-1', retirement_date: '2027-01-31' },
    { id: 'kimi-k2.5', retirement_date: '2027-01-26' },
    { id: 'kimi-k2.6', version: '2026-04-20', retirement_date: '2027-04-16' },
    { id: 'kimi-k2.7-code', version: '2026-06-12', retirement_date: '2027-04-16' },
    { id: 'kimi-k3.1-pro', version: '2026-11-04', retirement_date: '2027-11-04' },
    { id: 'llama-3.3-70b-instruct', retirement_date: '2027-01-31' },
    { id: 'llama-4-maverick-17b-128e-instruct-fp8', retirement_date: '2026-11-19' },
    { id: 'llama-4-scout-17b-16e-instruct', retirement_date: '2027-01-31' },
    { id: 'mai-image-2.5', version: '2026-06-02', retirement_date: '2026-11-24' },
    { id: 'mai-image-2.5-flash', version: '2026-06-02', retirement_date: '2026-11-24' },
    { id: 'mai-image-2.5-pro', version: '2026-06-19', retirement_date: '2026-11-24' },
    { id: 'ministral-3b', retirement_date: '2026-12-09' },
    { id: 'mistral-document-ai-2512', deprecation_date: '2026-12-15', retirement_date: '2027-02-15' },
    { id: 'mistral-medium-2505', retirement_date: '2026-12-09' },
    { id: 'mistral-small-2503', deprecation_date: '2026-10-09', retirement_date: '2027-02-15' },
    { id: 'model-router', version: '2025-05-19', retirement_date: '2027-01-27' },
    { id: 'model-router', version: '2025-08-07', retirement_date: '2027-01-27' },
    { id: 'model-router', version: '2025-11-18', retirement_date: '2027-05-20' },
    { id: 'o1', version: '2024-12-17', retirement_date: '2026-11-19' },
    { id: 'o1-pro', version: '2025-03-19', retirement_date: '2026-11-19' },
    { id: 'o3', version: '2025-04-16', retirement_date: '2026-11-19' },
    { id: 'o3-deep-research', version: '2025-06-26', retirement_date: '2026-11-19' },
    { id: 'o3-mini', version: '2025-01-31', retirement_date: '2026-11-19' },
    { id: 'o3-pro', version: '2025-06-10', retirement_date: '2026-11-19' },
    { id: 'o4-mini', version: '2025-04-16', retirement_date: '2026-11-19' },
    { id: 'phi-4', retirement_date: '2027-01-17' },
    { id: 'phi-4-mini-instruct', retirement_date: '2027-01-17' },
    { id: 'phi-4-mini-reasoning', retirement_date: '2027-01-17' },
    { id: 'phi-4-multimodal-instruct', retirement_date: '2027-01-17' },
    { id: 'phi-4-reasoning', retirement_date: '2027-01-17' },
    { id: 'sora-2', version: '2025-12-08', retirement_date: '2026-10-15' },
    { id: 'text-embedding-3-large', retirement_date: '2028-02-09' },
    { id: 'text-embedding-3-small', retirement_date: '2028-02-09' },
    { id: 'text-embedding-ada-002', retirement_date: '2028-02-09' },
    { id: 'tts', retirement_date: '2026-12-15' },
    { id: 'tts-hd', retirement_date: '2026-12-15' },
    { id: 'whisper', retirement_date: '2026-12-15' },
];

// https://docs.x.ai/docs/release-notes, https://docs.x.ai/developers/migration/imagine-image-quality-nov-2
const XAI_LIFECYCLE: readonly ModelLifecycleEntry[] = [
    { id: 'grok-build-0.1', retirement_date: '2026-11-19' },
    { id: 'grok-voice-transcribe-1.0', retirement_date: '2026-10-02' },
    { id: 'grok-imagine-image-quality', deprecation_date: '2026-09-02', retirement_date: '2026-11-02' },
];

/** The curated table, by provider. Exported for consistency checks; use `getModelLifecycleDates` to look up a model. */
export const CURATED_MODEL_LIFECYCLE: Partial<Record<Providers, readonly ModelLifecycleEntry[]>> = {
    [Providers.vertexai]: VERTEXAI_LIFECYCLE,
    [Providers.azure_openai]: AZURE_LIFECYCLE,
    [Providers.azure_foundry]: AZURE_LIFECYCLE,
    [Providers.xai]: XAI_LIFECYCLE,
};

function lifecycleLookupId(model: string): string {
    // Vertex AI resource names (`locations/global/publishers/google/models/gemini-3.7-flash`) end with the model ID.
    return (model.split('/').pop() ?? model).toLowerCase();
}

function toDates(entry: ModelLifecycleEntry): ModelLifecycleDates {
    return {
        ...(entry.deprecation_date ? { deprecation_date: entry.deprecation_date } : {}),
        retirement_date: entry.retirement_date,
    };
}

/**
 * Returns the announced deprecation and retirement dates of a model from the curated table, or `undefined` when no
 * date is known. `version` narrows the match for providers that list a model name and a version separately.
 */
export function getModelLifecycleDates(
    model: string,
    provider: Providers,
    version?: string,
): ModelLifecycleDates | undefined {
    const entries = CURATED_MODEL_LIFECYCLE[provider];
    if (!entries) return undefined;
    const id = lifecycleLookupId(model);
    // Vertex AI publisher model versions (`claude-sonnet-4-5@20250929`) share the dates of the model.
    const unversionedId = id.includes('@') ? id.slice(0, id.indexOf('@')) : id;

    const exact = entries.find((entry) => (entry.version ? `${entry.id}-${entry.version}` : entry.id) === id);
    if (exact) return toDates(exact);

    const candidates = entries.filter(
        (entry) => entry.id === id || (entry.id === unversionedId && !entry.id.includes('@')),
    );
    if (version) {
        const versioned = candidates.find((entry) => entry.version === version);
        if (versioned) return toDates(versioned);
        const unversioned = candidates.filter((entry) => !entry.version);
        return unversioned.length === 1 ? toDates(unversioned[0]) : undefined;
    }
    // Without a version, a model whose versions retire on different dates has no single date.
    const [first, ...others] = candidates;
    if (!first) return undefined;
    const sameDates = others.every(
        (entry) => entry.retirement_date === first.retirement_date && entry.deprecation_date === first.deprecation_date,
    );
    return sameDates ? toDates(first) : undefined;
}

/**
 * Normalizes a provider date (`YYYY-MM-DD`, an RFC 3339 timestamp or a `Date`) to a `YYYY-MM-DD` calendar date in
 * UTC. Returns `undefined` for a missing or unparseable value.
 */
export function toLifecycleDate(value: string | Date | null | undefined): string | undefined {
    if (value === null || value === undefined || value === '') return undefined;
    if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)) return value;
    const date = value instanceof Date ? value : new Date(value);
    return Number.isNaN(date.getTime()) ? undefined : date.toISOString().slice(0, 10);
}
