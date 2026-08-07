import { getModelCapabilitiesAnthropic } from './capability/anthropic.js';
import { getBedrockModelCapabilities, getBedrockModelKnowledge } from './capability/bedrock-models.js';
import { getModelCapabilitiesOpenAI } from './capability/openai.js';
import { getModelCapabilitiesVertexAI } from './capability/vertexai.js';
import { getContextWindowSize, getMaxOutputTokens } from './options/context-windows.js';
import { getOpenAIReasoningEffortLevels, isGeminiModelVersionGte } from './options/version-parsing.js';
import type { ModelCapabilities, ModelModalities } from './types.js';
import { Providers } from './types.js';

export interface ModelDirectoryMetadata {
    type?: string;
    input_modalities?: readonly string[];
    output_modalities?: readonly string[];
    tool_support?: boolean;
    tool_support_streaming?: boolean;
    context_window?: number;
    max_output_tokens?: number;
    supported_parameters?: readonly string[];
}

export interface ModelProfile {
    model_id: string;
    canonical_id: string;
    family: string;
    source_provider?: string;
    capabilities: ModelCapabilities;
    context_window?: number;
    max_output_tokens?: number;
    reasoning_effort_levels?: readonly string[];
}

type ModelProfileOverride = Partial<
    Pick<ModelProfile, 'family' | 'source_provider' | 'context_window' | 'max_output_tokens'>
> & {
    capabilities?: Partial<ModelCapabilities>;
};

const EXACT_MODEL_OVERRIDES: Record<string, ModelProfileOverride> = {
    // OpenAI image endpoints are not text inference models even though their IDs contain `gpt`.
    'gpt-image-1': {
        family: 'image',
        capabilities: { input: { text: true, image: true }, output: { image: true }, tool_support: false },
    },
};

const PROVIDER_MODEL_OVERRIDES: Partial<Record<Providers, Record<string, ModelProfileOverride>>> = {
    [Providers.bedrock_mantle]: {
        'openai.gpt-5.5': { context_window: 272_000, max_output_tokens: 128_000 },
        'openai.gpt-5.6': { context_window: 272_000, max_output_tokens: 128_000 },
    },
};

const GENERIC_CAPABILITIES: ModelCapabilities = {
    input: { text: true },
    output: { text: true },
    tool_support: true,
    tool_support_streaming: true,
};

function normalizeModelId(model: string): string {
    const normalized = model.trim().toLowerCase();
    const slash = normalized.lastIndexOf('/');
    return slash === -1 ? normalized : normalized.slice(slash + 1);
}

function getModelAliases(model: string): string[] {
    const normalized = model.trim().toLowerCase();
    const aliases = [normalized, normalizeModelId(normalized)];
    const inferenceProfile = normalized.lastIndexOf('inference-profile/');
    if (inferenceProfile !== -1) {
        aliases.push(normalizeModelId(normalized.slice(inferenceProfile + 'inference-profile/'.length)));
    }
    return [...new Set(aliases)];
}

function inferFamily(model: string): { family: string; source_provider?: string } {
    const normalized = normalizeModelId(model);
    if (normalized.includes('embedding') || normalized.includes('embed') || normalized.includes('vector')) {
        return { family: 'embedding' };
    }
    if (normalized.includes('gpt-image') || normalized.includes('dall-e') || normalized.includes('imagen-')) {
        return { family: 'image', source_provider: normalized.includes('imagen') ? 'google' : 'openai' };
    }
    if (normalized.includes('gemini')) return { family: 'gemini', source_provider: 'google' };
    if (normalized.includes('claude')) return { family: 'claude', source_provider: 'anthropic' };
    if (normalized.includes('gpt') || normalized.startsWith('o1') || normalized.startsWith('o3')) {
        return { family: 'gpt', source_provider: 'openai' };
    }
    if (normalized.includes('grok')) return { family: 'grok', source_provider: 'xai' };
    for (const family of ['llama', 'qwen', 'deepseek', 'gemma', 'mistral', 'kimi', 'minimax', 'glm']) {
        if (normalized.includes(family)) return { family };
    }
    return { family: 'generic' };
}

function getCanonicalCapabilities(model: string, family: string): ModelCapabilities {
    switch (family) {
        case 'gemini':
            return getModelCapabilitiesVertexAI(model);
        case 'claude':
            return getModelCapabilitiesAnthropic(model);
        case 'gpt': {
            const capabilities = getModelCapabilitiesOpenAI(model);
            return {
                ...capabilities,
                input: {
                    ...capabilities.input,
                    text: capabilities.input.text ?? true,
                    image: capabilities.input.image ?? true,
                },
                output: {
                    ...capabilities.output,
                    text: capabilities.output.text ?? true,
                },
                tool_support: capabilities.tool_support ?? true,
                tool_support_streaming: capabilities.tool_support_streaming ?? capabilities.tool_support ?? true,
            };
        }
        case 'embedding':
            return {
                input: { text: true },
                output: { embed: true },
                tool_support: false,
                tool_support_streaming: false,
            };
        case 'image':
            return { input: { text: true, image: true }, output: { image: true }, tool_support: false };
        case 'grok':
            return {
                input: { text: true, image: model.includes('vision') },
                output: { text: true },
                tool_support: true,
            };
        case 'llama':
            return {
                input: { text: true, image: model.includes('vision') || model.includes('llama-4') },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'gemma': {
            const normalized = model.toLowerCase();
            const image =
                (normalized.includes('gemma-3') &&
                    !normalized.includes('gemma-3-1b') &&
                    !normalized.includes('gemma-3-270m')) ||
                normalized.includes('gemma-4');
            return {
                input: { text: true, image },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
        case 'qwen':
            return {
                input: { text: true, image: /qwen(?:2\.5|3)[^/]*[-.]vl|qwen3\.5/.test(model.toLowerCase()) },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'deepseek':
        case 'mistral':
        case 'kimi':
        case 'minimax':
        case 'glm':
            return { input: { text: true }, output: { text: true }, tool_support: true, tool_support_streaming: true };
        default:
            return { ...GENERIC_CAPABILITIES, input: { ...GENERIC_CAPABILITIES.input } };
    }
}

function applyProviderOverlay(
    model: string,
    provider: string | Providers | undefined,
    family: string,
    capabilities: ModelCapabilities,
): { capabilities: ModelCapabilities; context_window?: number; max_output_tokens?: number } {
    const normalizedProvider = provider?.toLowerCase();
    if (normalizedProvider === 'bedrock') {
        const bedrock = getBedrockModelCapabilities(model, 'runtime');
        const knowledge = getBedrockModelKnowledge(model);
        return { capabilities: bedrock, ...knowledge };
    }
    if (normalizedProvider === 'bedrock_mantle') {
        const bedrock = getBedrockModelCapabilities(model, 'mantle');
        const knowledge = getBedrockModelKnowledge(model);
        return { capabilities: bedrock, ...knowledge };
    }

    // OpenRouter and other OpenAI-compatible transports retain the source model's semantic
    // capabilities, but cannot expose provider-native fields such as Vertex Flex or thinking_level.
    if (normalizedProvider === 'openai_compatible') {
        return {
            capabilities: {
                ...capabilities,
                tool_support_streaming: capabilities.tool_support_streaming ?? capabilities.tool_support,
            },
            context_window: getContextWindowSize(model),
            max_output_tokens: getMaxOutputTokens(model),
        };
    }

    if (family === 'gpt' && normalizedProvider === 'azure_openai') {
        return {
            capabilities,
            context_window: getContextWindowSize(model),
            max_output_tokens: getMaxOutputTokens(model),
        };
    }
    return { capabilities, context_window: getContextWindowSize(model), max_output_tokens: getMaxOutputTokens(model) };
}

function applyListingMetadata(profile: ModelProfile, metadata?: ModelDirectoryMetadata): ModelProfile {
    if (!metadata) return profile;
    const input = metadata.input_modalities
        ? Object.fromEntries(metadata.input_modalities.map((modality) => [modality, true]))
        : profile.capabilities.input;
    const output = metadata.output_modalities
        ? Object.fromEntries(metadata.output_modalities.map((modality) => [modality, true]))
        : profile.capabilities.output;
    return {
        ...profile,
        capabilities: {
            ...profile.capabilities,
            input: input as ModelModalities,
            output: output as ModelModalities,
            ...(metadata.tool_support !== undefined && { tool_support: metadata.tool_support }),
            ...(metadata.tool_support_streaming !== undefined && {
                tool_support_streaming: metadata.tool_support_streaming,
            }),
        },
        ...(metadata.context_window !== undefined && { context_window: metadata.context_window }),
        ...(metadata.max_output_tokens !== undefined && { max_output_tokens: metadata.max_output_tokens }),
    };
}

export function resolveModelProfile(
    model: string,
    provider?: string | Providers,
    metadata?: ModelDirectoryMetadata,
): ModelProfile {
    const canonical_id = normalizeModelId(model);
    const exactOverride = EXACT_MODEL_OVERRIDES[canonical_id];
    const inferred = inferFamily(model);
    const family = exactOverride?.family ?? inferred.family;
    const source_provider = exactOverride?.source_provider ?? inferred.source_provider;
    const baseCapabilities = {
        ...(provider?.toLowerCase() === Providers.vertexai
            ? getModelCapabilitiesVertexAI(model)
            : getCanonicalCapabilities(model, family)),
        ...exactOverride?.capabilities,
    };
    const overlay = applyProviderOverlay(model, provider, family, baseCapabilities);
    const reasoningEffortLevels =
        family === 'gpt'
            ? Object.values(getOpenAIReasoningEffortLevels(model) ?? {}).map((value) => value)
            : isGeminiModelVersionGte(model, '3.5')
              ? ['minimal', 'low', 'medium', 'high']
              : undefined;
    const providerOverride = provider
        ? PROVIDER_MODEL_OVERRIDES[provider.toLowerCase() as Providers]?.[canonical_id]
        : undefined;
    const profile = applyListingMetadata(
        {
            model_id: model,
            canonical_id,
            family,
            source_provider,
            capabilities: {
                ...overlay.capabilities,
                ...providerOverride?.capabilities,
                ...(providerOverride?.capabilities?.input && {
                    input: { ...overlay.capabilities.input, ...providerOverride.capabilities.input },
                }),
                ...(providerOverride?.capabilities?.output && {
                    output: { ...overlay.capabilities.output, ...providerOverride.capabilities.output },
                }),
            },
            context_window: providerOverride?.context_window ?? overlay.context_window,
            max_output_tokens: providerOverride?.max_output_tokens ?? overlay.max_output_tokens,
            reasoning_effort_levels: reasoningEffortLevels,
        },
        metadata,
    );
    return profile;
}

export function isModelDirectoryEmbedding(model: string, metadata?: ModelDirectoryMetadata): boolean {
    return isModelDirectoryNonInference(model, metadata, 'embedding');
}

export function isModelDirectoryNonInference(
    model: string,
    metadata?: ModelDirectoryMetadata,
    kind?: 'embedding',
): boolean {
    const type = metadata?.type?.toLowerCase();
    if (type === 'embedding' || type === 'audio' || type === 'image' || type === 'video' || type === 'moderation') {
        return true;
    }
    if (metadata?.output_modalities?.some((modality) => /embed|vector|audio|image|video/i.test(modality))) return true;

    const aliases = getModelAliases(model);
    return aliases.some((alias) =>
        kind === 'embedding'
            ? /(^|[-_.:])(?:embed|embedding|vector)(?:[-_.:]|$)/.test(alias)
            : /(?:embed|embedding|vector|whisper|speech|tts|audio|orpheus|prompt-guard|moderation|gpt-image|dall-e|imagen|nova-canvas|nova-reel|sora|veo|pegasus)/.test(
                  alias,
              ),
    );
}
