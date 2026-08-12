import { getModelCapabilitiesAnthropic } from './capability/anthropic.js';
import { getModelCapabilitiesAzureFoundry } from './capability/azure_foundry.js';
import { getBedrockModelCapabilities, getBedrockModelKnowledge } from './capability/bedrock-models.js';
import { getMistralModelKnowledge } from './capability/mistral.js';
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
    'chat-latest': {
        family: 'gpt',
        source_provider: 'openai',
        context_window: 400_000,
        max_output_tokens: 128_000,
    },
    // OpenAI image endpoints are not text inference models even though their IDs contain `gpt`.
    'gpt-image-1': {
        family: 'image',
        capabilities: {
            input: { text: true, image: true, audio: false, video: false },
            output: { text: false, image: true, audio: false, video: false },
            tool_support: false,
            tool_support_streaming: false,
        },
    },
    'gpt-realtime-translate': {
        family: 'realtime',
        context_window: 16_000,
        max_output_tokens: 2_000,
        capabilities: {
            input: { text: false, image: false, audio: true, video: false },
            output: { text: true, image: false, audio: true, video: false },
            tool_support: false,
            tool_support_streaming: false,
        },
    },
    'gpt-5-chat-latest': {
        family: 'gpt',
        source_provider: 'openai',
        context_window: 128_000,
        max_output_tokens: 16_384,
    },
};

const GENERIC_CAPABILITIES: ModelCapabilities = {
    input: { text: true },
    output: { text: true },
};

function normalizeSourceModelId(model: string): string {
    const normalized = model.trim().toLowerCase().replace(/^~/, '');
    // Azure Foundry uses deployment::source-model. Resolve family semantics from the source half, not the
    // customer-chosen deployment name, before applying transport behavior.
    const sourceQualified = normalized.includes('::') ? (normalized.split('::').pop() ?? normalized) : normalized;
    const slash = sourceQualified.lastIndexOf('/');
    const leaf = slash === -1 ? sourceQualified : sourceQualified.slice(slash + 1);
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
    if (/(?:prompt-guard|moderation|safeguard)/.test(normalized)) return { family: 'moderation' };
    if (normalized.includes('gpt-image') || normalized.includes('dall-e') || normalized.includes('imagen-')) {
        return { family: 'image', source_provider: normalized.includes('imagen') ? 'google' : 'openai' };
    }
    if (/(?:whisper|transcribe)/.test(normalized)) return { family: 'transcription', source_provider: 'openai' };
    if (/(?:^|[-_.])tts(?:[-_.]|$)/.test(normalized)) return { family: 'speech', source_provider: 'openai' };
    if (normalized.includes('realtime')) return { family: 'realtime', source_provider: 'openai' };
    if (/(?:^|[-_.])audio(?:[-_.]|$)/.test(normalized)) return { family: 'audio', source_provider: 'openai' };
    if (normalized.includes('sora')) return { family: 'video', source_provider: 'openai' };
    if (normalized.includes('gemini')) return { family: 'gemini', source_provider: 'google' };
    if (normalized.includes('claude')) return { family: 'claude', source_provider: 'anthropic' };
    if (normalized.includes('gpt') || /^o\d+(?:[-_.]|$)/.test(normalized)) {
        return { family: 'gpt', source_provider: 'openai' };
    }
    if (normalized.includes('grok')) return { family: 'grok', source_provider: 'xai' };
    if (normalized.includes('nemotron')) return { family: 'nemotron', source_provider: 'nvidia' };
    if (
        /(?:mistral|mixtral|ministral|magistral|voxtral|codestral|devstral|leanstral|mathstral|pixtral)/.test(
            normalized,
        )
    ) {
        return { family: 'mistral', source_provider: 'mistralai' };
    }
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
        case 'transcription':
            return {
                input: { audio: true },
                output: { text: true },
                tool_support: false,
                tool_support_streaming: false,
            };
        case 'speech':
            return {
                input: { text: true },
                output: { audio: true },
                tool_support: false,
                tool_support_streaming: false,
            };
        case 'realtime':
            return {
                input: { text: true, image: true, audio: true },
                output: { text: true, audio: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'audio':
            return {
                input: { text: true, audio: true },
                output: { text: true, audio: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'video':
            return {
                input: { text: true, image: true, video: true },
                output: { video: true, audio: true },
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
                // Future Llama generations inherit the newest known family behavior until a provider documents
                // a narrower exception.
                input: { text: true, image: model.includes('vision') || isLlamaVersionGte(model, 4) },
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
        case 'qwen': {
            const normalized = model.toLowerCase();
            const qwenVision =
                normalized.includes('qwen3.5') ||
                ['qwen2.5', 'qwen3'].some((familyPrefix) => {
                    const familyIndex = normalized.indexOf(familyPrefix);
                    if (familyIndex === -1) return false;
                    const suffix = normalized.slice(familyIndex + familyPrefix.length);
                    return suffix.includes('-vl') || suffix.includes('.vl');
                });
            return {
                input: { text: true, image: qwenVision },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
        case 'deepseek':
        case 'kimi':
        case 'minimax':
        case 'glm':
            return { input: { text: true }, output: { text: true }, tool_support: true, tool_support_streaming: true };
        case 'nemotron':
            return {
                input: { text: true, image: model.includes('nemotron-nano-12b') },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'mistral':
            return getMistralModelKnowledge(model).capabilities;
        default:
            return { ...GENERIC_CAPABILITIES, input: { ...GENERIC_CAPABILITIES.input } };
    }
}

function isLlamaVersionGte(model: string, targetMajor: number): boolean {
    return (
        isModelFamilyVersionGTE(model, 'llama-', targetMajor, 0) ||
        isModelFamilyVersionGTE(model, 'llama', targetMajor, 0)
    );
}

function mergeCapabilities(base: ModelCapabilities, override?: Partial<ModelCapabilities>): ModelCapabilities {
    if (!override) return base;
    return {
        ...base,
        ...override,
        input: { ...base.input, ...override.input },
        output: { ...base.output, ...override.output },
    };
}

function getCanonicalLimits(
    sourceModel: string,
    family: string,
): Pick<ModelProfile, 'context_window' | 'max_output_tokens'> {
    // Profiles only expose limits backed by known family data so option UIs do not present guesses as authoritative.
    if (family === 'audio') return { context_window: 128_000, max_output_tokens: 16_384 };
    if (family === 'realtime' && isModelFamilyVersionGTE(sourceModel, 'gpt-realtime-', 2, 0)) {
        // Future Realtime generations inherit the latest documented family limits until a narrower rule is known.
        return { context_window: 128_000, max_output_tokens: 32_000 };
    }
    if (family === 'gemini' && sourceModel.includes('gemini-3.1-flash-image')) {
        return { context_window: 131_072, max_output_tokens: 32_768 };
    }
    if (family === 'nemotron') {
        if (sourceModel.includes('nemotron-super-')) return { context_window: 256_000, max_output_tokens: 32_768 };
        const generation = sourceModel.match(/nemotron-nano-(\d+)-/)?.[1];
        return {
            context_window: generation && Number(generation) >= 3 ? 256_000 : 128_000,
            max_output_tokens: 8_192,
        };
    }
    if (
        ['generic', 'embedding', 'moderation', 'image', 'transcription', 'speech', 'realtime', 'video'].includes(family)
    ) {
        return {};
    }
    if (family === 'mistral') {
        return { context_window: getMistralModelKnowledge(sourceModel).context_window };
    }
    return {
        context_window: getContextWindowSize(sourceModel),
        max_output_tokens: getMaxOutputTokens(sourceModel),
    };
}

function applyProviderOverlay(
    model: string,
    sourceModel: string,
    family: string,
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
            ...getCanonicalLimits(sourceModel, family),
        };
    }

    if (provider === Providers.azure_foundry) {
        // Deployment metadata describes the transport. Dedicated source endpoints keep their source modalities so a
        // deployment named like a GPT model cannot become text inference accidentally.
        const providerCapabilities = getModelCapabilitiesAzureFoundry(model);
        return {
            capabilities: [
                'embedding',
                'moderation',
                'image',
                'transcription',
                'speech',
                'realtime',
                'audio',
                'video',
            ].includes(family)
                ? capabilities
                : providerCapabilities,
            ...getCanonicalLimits(sourceModel, family),
        };
    }

    return {
        capabilities,
        ...getCanonicalLimits(sourceModel, family),
    };
}

function getReasoningEffortLevels(model: string, family: string, provider: Providers): readonly string[] | undefined {
    if (family === 'gpt') {
        if (model.includes('gpt-oss')) {
            return provider === Providers.togetherai ||
                provider === Providers.openai_compatible ||
                provider === Providers.vertexai ||
                provider === Providers.bedrock ||
                provider === Providers.bedrock_mantle ||
                provider === Providers.azure_foundry
                ? ['low', 'medium', 'high']
                : undefined;
        }
        if (
            provider === Providers.openai ||
            provider === Providers.azure_openai ||
            provider === Providers.openai_compatible ||
            provider === Providers.azure_foundry ||
            provider === Providers.bedrock_mantle
        ) {
            if (/^o\d+(?:[-_.]|$)/.test(model)) return ['low', 'medium', 'high'];
            return Object.values(getOpenAIReasoningEffortLevels(model) ?? {});
        }
        return undefined;
    }
    if (family === 'gemini' && provider === Providers.openai_compatible && isGeminiModelVersionGte(model, '3.5')) {
        return ['minimal', 'low', 'medium', 'high'];
    }
    if (provider === Providers.mistralai && family === 'mistral') {
        return getMistralModelKnowledge(model).reasoning_effort_levels;
    }
    if (provider === Providers.xai && family === 'grok') {
        const grok420Index = model.indexOf('grok-4.20');
        if (grok420Index !== -1 && model.indexOf('multi-agent', grok420Index + 'grok-4.20'.length) !== -1) {
            return ['low', 'medium', 'high', 'xhigh'];
        }
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
    // Family identity follows the normalized source model. Transport paths and customer deployment names must not
    // accidentally select a different family rule.
    const inferred = inferFamily(canonical_id);
    const family = exactOverride?.family ?? inferred.family;
    const source_provider = exactOverride?.source_provider ?? inferred.source_provider;
    const canonicalCapabilities = getCanonicalCapabilities(canonical_id, family);
    const vertexCapabilities = provider === Providers.vertexai ? getModelCapabilitiesVertexAI(model) : undefined;
    const hasVertexCapabilities =
        vertexCapabilities &&
        (Object.values(vertexCapabilities.input).some((value) => value === true) ||
            Object.values(vertexCapabilities.output).some((value) => value === true) ||
            vertexCapabilities.tool_support !== undefined);
    const baseCapabilities = hasVertexCapabilities ? vertexCapabilities : canonicalCapabilities;
    const overlay = applyProviderOverlay(model, canonical_id, family, provider, baseCapabilities);
    const reasoningEffortLevels = getReasoningEffortLevels(canonical_id, family, provider);
    return {
        model_id: model,
        canonical_id,
        family,
        source_provider,
        // Exact source-model semantics win after transport overlays. Provider inference must not turn a dedicated
        // image or other non-chat endpoint into a text model merely because its ID resembles a known family.
        capabilities: mergeCapabilities(overlay.capabilities, exactOverride?.capabilities),
        context_window: exactOverride?.context_window ?? overlay.context_window,
        max_output_tokens: exactOverride?.max_output_tokens ?? overlay.max_output_tokens,
        ...(reasoningEffortLevels?.length && { reasoning_effort_levels: reasoningEffortLevels }),
    };
}

export function isModelDirectoryEmbedding(model: string): boolean {
    const aliases = getModelAliases(model);
    return aliases.some((alias) => /(^|[-_.:])(?:embed|embedding|vector)(?:[-_.:]|$)/.test(alias));
}
