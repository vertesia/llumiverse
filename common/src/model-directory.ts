import { getModelCapabilitiesAnthropic } from './capability/anthropic.js';
import { getModelCapabilitiesAzureFoundry } from './capability/azure_foundry.js';
import { getBedrockModelCapabilities, getBedrockModelKnowledge } from './capability/bedrock-models.js';
import { getModelCapabilitiesOpenAI } from './capability/openai.js';
import { getModelCapabilitiesVertexAI } from './capability/vertexai.js';
import { getContextWindowSize, getMaxOutputTokens } from './options/context-windows.js';
import {
    getOpenAIReasoningEffortLevels,
    isGeminiModelVersionGte,
    isModelFamilyVersionGTE,
} from './options/version-parsing.js';
import type { ModelCapabilities } from './types.js';
import { Providers } from './types.js';

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

const GENERIC_CAPABILITIES: ModelCapabilities = {
    input: { text: true },
    output: { text: true },
};

function normalizeSourceModelId(model: string): string {
    const normalized = model.trim().toLowerCase().replace(/^~/, '');
    const slash = normalized.lastIndexOf('/');
    const leaf = slash === -1 ? normalized : normalized.slice(slash + 1);
    return leaf.replace(/^(?:global|us|eu|apac)\./, '').replace(/^(?:openai|anthropic|google|xai)\./, '');
}

function getModelAliases(model: string): string[] {
    const normalized = model.trim().toLowerCase();
    const aliases = [normalized, normalizeSourceModelId(normalized)];
    return [...new Set(aliases)];
}

function inferFamily(model: string): { family: string; source_provider?: string } {
    const normalized = normalizeSourceModelId(model);
    if (/(^|[-_.:])(?:embed|embedding|vector)(?:[-_.:]|$)/.test(normalized)) {
        return { family: 'embedding' };
    }
    if (normalized.includes('prompt-guard')) return { family: 'moderation' };
    if (normalized.includes('gpt-image') || normalized.includes('dall-e') || normalized.includes('imagen-')) {
        return { family: 'image', source_provider: normalized.includes('imagen') ? 'google' : 'openai' };
    }
    if (normalized.includes('gemini')) return { family: 'gemini', source_provider: 'google' };
    if (normalized.includes('claude')) return { family: 'claude', source_provider: 'anthropic' };
    if (normalized.includes('gpt') || /^o\d+(?:[-_.]|$)/.test(normalized)) {
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
        case 'moderation':
            return {
                input: { text: true },
                output: { text: true },
                tool_support: false,
                tool_support_streaming: false,
            };
        case 'image':
            return { input: { text: true, image: true }, output: { image: true }, tool_support: false };
        case 'grok':
            return {
                input: {
                    text: true,
                    image: model.includes('vision') || isModelFamilyVersionGTE(model, 'grok-', 4, 3),
                },
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
    sourceModel: string,
    provider: Providers,
    capabilities: ModelCapabilities,
): { capabilities: ModelCapabilities; context_window?: number; max_output_tokens?: number } {
    if (provider === Providers.bedrock) {
        const bedrock = getBedrockModelCapabilities(model, 'runtime');
        const knowledge = getBedrockModelKnowledge(model);
        return { capabilities: bedrock, ...knowledge };
    }
    if (provider === Providers.bedrock_mantle) {
        const bedrock = getBedrockModelCapabilities(model, 'mantle');
        const knowledge = getBedrockModelKnowledge(model);
        return { capabilities: bedrock, ...knowledge };
    }

    // OpenRouter and other OpenAI-compatible transports retain the source model's semantic
    // capabilities, but cannot expose provider-native fields such as Vertex Flex or thinking_level.
    if (provider === Providers.openai_compatible) {
        return {
            capabilities: {
                ...capabilities,
                tool_support_streaming: capabilities.tool_support_streaming ?? capabilities.tool_support,
            },
            context_window: getContextWindowSize(sourceModel),
            max_output_tokens: getMaxOutputTokens(sourceModel),
        };
    }

    if (provider === Providers.azure_foundry) {
        return {
            capabilities: getModelCapabilitiesAzureFoundry(model),
            context_window: getContextWindowSize(sourceModel),
            max_output_tokens: getMaxOutputTokens(sourceModel),
        };
    }

    return {
        capabilities,
        context_window: getContextWindowSize(sourceModel),
        max_output_tokens: getMaxOutputTokens(sourceModel),
    };
}

function getReasoningEffortLevels(model: string, family: string, provider: Providers): readonly string[] | undefined {
    if (family === 'gpt') {
        if (model.includes('gpt-oss')) {
            return provider === Providers.togetherai || provider === Providers.openai_compatible
                ? ['low', 'medium', 'high']
                : undefined;
        }
        if (
            provider === Providers.openai ||
            provider === Providers.azure_openai ||
            provider === Providers.openai_compatible
        ) {
            if (/^o\d+(?:[-_.]|$)/.test(model)) return ['low', 'medium', 'high'];
            return Object.values(getOpenAIReasoningEffortLevels(model) ?? {});
        }
        return undefined;
    }
    if (family === 'gemini' && provider === Providers.openai_compatible && isGeminiModelVersionGte(model, '3.5')) {
        return ['minimal', 'low', 'medium', 'high'];
    }
    if (provider === Providers.mistralai && /mistral-(?:small-latest|medium-3-5)/.test(model)) {
        return ['none', 'high'];
    }
    if (provider === Providers.xai && family === 'grok') {
        if (/grok-4\.20[^/]*multi-agent/.test(model)) return ['low', 'medium', 'high', 'xhigh'];
        if (isSingleDigitGrokVersionGte(model, 4, 5)) return ['low', 'medium', 'high'];
        if (isSingleDigitGrokVersionGte(model, 4, 3)) return ['none', 'low', 'medium', 'high'];
    }
    return undefined;
}

function isSingleDigitGrokVersionGte(model: string, targetMajor: number, targetMinor: number): boolean {
    const match = model.match(/grok-(\d+)(?:\.(\d))?(?:[-_.]|$)/);
    if (!match) return false;
    const major = Number(match[1]);
    const minor = Number(match[2] ?? 0);
    return major > targetMajor || (major === targetMajor && minor >= targetMinor);
}

export function resolveModelProfile(model: string, provider: Providers): ModelProfile {
    const canonical_id = normalizeSourceModelId(model);
    const exactOverride = EXACT_MODEL_OVERRIDES[canonical_id];
    const inferred = inferFamily(model);
    const family = exactOverride?.family ?? inferred.family;
    const source_provider = exactOverride?.source_provider ?? inferred.source_provider;
    const canonicalCapabilities = getCanonicalCapabilities(canonical_id, family);
    const vertexCapabilities = provider === Providers.vertexai ? getModelCapabilitiesVertexAI(model) : undefined;
    const hasVertexCapabilities =
        vertexCapabilities &&
        (Object.values(vertexCapabilities.input).some((value) => value === true) ||
            Object.values(vertexCapabilities.output).some((value) => value === true) ||
            vertexCapabilities.tool_support !== undefined);
    const baseCapabilities = {
        ...(hasVertexCapabilities ? vertexCapabilities : canonicalCapabilities),
        ...exactOverride?.capabilities,
    };
    const overlay = applyProviderOverlay(model, canonical_id, provider, baseCapabilities);
    const reasoningEffortLevels = getReasoningEffortLevels(canonical_id, family, provider);
    return {
        model_id: model,
        canonical_id,
        family,
        source_provider,
        capabilities: overlay.capabilities,
        context_window: overlay.context_window,
        max_output_tokens: overlay.max_output_tokens,
        ...(reasoningEffortLevels?.length && { reasoning_effort_levels: reasoningEffortLevels }),
    };
}

export function isModelDirectoryEmbedding(model: string): boolean {
    const aliases = getModelAliases(model);
    return aliases.some((alias) => /(^|[-_.:])(?:embed|embedding|vector)(?:[-_.:]|$)/.test(alias));
}
