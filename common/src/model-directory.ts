import { getModelCapabilitiesAnthropic } from './capability/anthropic.js';
import { getAzureFoundryModelLimits, getModelCapabilitiesAzureFoundry } from './capability/azure_foundry.js';
import { getBedrockModelCapabilities, getBedrockModelKnowledge } from './capability/bedrock-models.js';
import { getMistralModelKnowledge } from './capability/mistral.js';
import { getModelCapabilitiesOpenAI } from './capability/openai.js';
import { getModelCapabilitiesVertexAI } from './capability/vertexai.js';
import { getVertexAIModelLimits } from './capability/vertexai-limits.js';
import { getContextWindowSize, getMaxOutputTokens } from './options/context-windows.js';
import { getGeminiModelLimits } from './options/gemini-limits.js';
import {
    getAvailableEffortLevels,
    getGrokReasoningEffortLevels,
    getOpenAIReasoningEffortLevels,
    hasGeminiSamplingRestriction,
    isClaudeVersionGTE,
    isDeepSeekVersionGte,
    isGeminiModelVersionGte,
    isGeminiOmniVideoModel,
    isInklingModel,
    isKimiVersionGte,
    isMetaMuseModel,
    isModelFamilyVersionGTE,
    isQwenVersionGte,
    isSingleDigitGrokVersionGte,
    supportsGeminiXHighThinking,
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
    return (
        leaf
            .replace(/^(?:global|us|eu|apac)\./, '')
            .replace(/^(?:openai|anthropic|google|xai)\./, '')
            // OpenRouter routing variants (`:batch`, `:free`, ...) select a serving mode, not a different model.
            .replace(/:(?:batch|free|nitro|floor|extended|exacto|online|thinking)$/, '')
    );
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
    if (normalized.includes('grok') && normalized.includes('image')) {
        return { family: 'image', source_provider: 'xai' };
    }
    if (normalized.includes('gpt-image') || normalized.includes('dall-e') || normalized.includes('imagen-')) {
        return { family: 'image', source_provider: normalized.includes('imagen') ? 'google' : 'openai' };
    }
    if (/(?:realtime|live|native-audio)/.test(normalized))
        return { family: 'realtime', source_provider: normalized.includes('gemini') ? 'google' : 'openai' };
    if (/(?:whisper|transcribe)/.test(normalized)) return { family: 'transcription', source_provider: 'openai' };
    if (/(?:^|[-_.])tts(?:[-_.]|$)/.test(normalized)) return { family: 'speech', source_provider: 'openai' };
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
    // Models offered through aggregators such as OpenRouter; there is no first-party driver for these makers.
    if (isMetaMuseModel(normalized)) return { family: 'muse', source_provider: 'meta' };
    if (/(?:^|[-_.])mimo-v\d/.test(normalized)) return { family: 'mimo', source_provider: 'xiaomi' };
    if (/(?:^|[-_.])step-\d/.test(normalized)) return { family: 'step', source_provider: 'stepfun' };
    if (isInklingModel(normalized)) return { family: 'inkling', source_provider: 'thinkingmachines' };
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
                tool_support: false,
                tool_support_streaming: false,
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
                    // https://docs.x.ai/developers/models/grok-build-0.1 lists "text, image → text".
                    image:
                        model.includes('vision') ||
                        model.includes('grok-build') ||
                        isModelFamilyVersionGTE(model, 'grok-', 4, 3),
                },
                output: { text: true },
                tool_support: true,
            };
        case 'llama':
            return {
                // Future Llama generations inherit the newest known family behavior until a provider documents
                // a narrower exception.
                // Llama Guard 4 takes mixed text-and-image prompts. https://huggingface.co/meta-llama/Llama-Guard-4-12B
                input: {
                    text: true,
                    image:
                        model.includes('vision') ||
                        isLlamaVersionGte(model, 4) ||
                        isModelFamilyVersionGTE(model, 'llama-guard-', 4, 0),
                },
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
            // Gemma 4 takes video; audio only on the E2B, E4B and 12B sizes. https://ai.google.dev/gemma/docs/core
            return {
                input: {
                    text: true,
                    image,
                    video: normalized.includes('gemma-4'),
                    audio: /gemma-4-(?:e2b|e4b|12b)(?:[-_.:]|$)/.test(normalized),
                },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
        case 'qwen': {
            const normalized = model.toLowerCase();
            // From Qwen 3.5 the hosted tiers and open-weight models take image and video
            // (https://www.alibabacloud.com/help/en/model-studio/qwen3-8-flash). Exceptions: the current Qwen3.7-Max is
            // text-only (https://www.alibabacloud.com/help/en/model-studio/qwen3-7-max), OpenRouter lists the 2.4T
            // open-weight model as text-only, and coder models stay text-only.
            const qwenMultimodal =
                isQwenVersionGte(normalized, 3, 5) &&
                !normalized.includes('coder') &&
                !/qwen3\.7-max(?![-_]2026-06-08)/.test(normalized) &&
                !/qwen\d+(?:\.\d+)?-\d+(?:\.\d+)?t(?:[-_.:]|$)/.test(normalized);
            const qwenVision =
                qwenMultimodal ||
                ['qwen2.5', 'qwen3'].some((familyPrefix) => {
                    const familyIndex = normalized.indexOf(familyPrefix);
                    if (familyIndex === -1) return false;
                    const suffix = normalized.slice(familyIndex + familyPrefix.length);
                    return suffix.includes('-vl') || suffix.includes('.vl');
                });
            return {
                input: {
                    text: true,
                    image: qwenVision,
                    video: qwenMultimodal,
                    // https://www.alibabacloud.com/help/en/model-studio/qwen3-8-omni-flash
                    audio: qwenMultimodal && normalized.includes('-omni'),
                },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
        case 'deepseek':
            return {
                // DeepSeek V4.1 Flash adds image input; earlier and non-Flash DeepSeek models remain text-only.
                input: {
                    text: true,
                    image:
                        (isModelFamilyVersionGTE(model, 'deepseek-v', 4, 1) && model.includes('flash')) ||
                        model.includes('vision'),
                },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'kimi':
            return {
                // Kimi K2.5 added image input and K3 video. https://platform.kimi.ai/docs/overview
                input: { text: true, image: isKimiVersionGte(model, 2, 5), video: isKimiVersionGte(model, 3, 0) },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'glm': {
            // The GLM vision models (`glm-4.6v`, `glm-5v-turbo`) and the GLM-5.3 Flash tier take image and video; the
            // other GLM models are text-only. https://docs.z.ai/guides/llm/glm-5.3-flash
            const vision =
                /glm-\d+(?:\.\d+)?v(?:[-_.:]|$)/.test(model) ||
                (isModelFamilyVersionGTE(model, 'glm-', 5, 3) && model.includes('-flash'));
            return {
                input: { text: true, image: vision, video: vision },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
        case 'minimax': {
            // MiniMax M3 adds image and video input. https://openrouter.ai/minimax/minimax-m3
            const isM3OrLater = isModelFamilyVersionGTE(model, 'minimax-m', 3, 0);
            return {
                input: { text: true, image: isM3OrLater, video: isM3OrLater },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
        // Source modalities as published on OpenRouter: Muse Wonder and MiMo also accept audio.
        case 'muse':
            return {
                input: {
                    text: true,
                    image: true,
                    video: !model.includes('glimmer'),
                    audio: model.includes('wonder'),
                },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'mimo': {
            // MiMo-V2.5-Pro is a text-only language model. https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
            const multimodal = !model.includes('mimo-v2.5-pro');
            return {
                input: { text: true, image: multimodal, video: multimodal, audio: multimodal },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
        case 'step':
            return {
                input: { text: true, image: true, video: true },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'inkling':
            // Thinking Machines Inkling. Tool calls are served through Chat Completions and Responses; no image input
            // is published for the Mini, so input stays text until a provider documents more.
            return {
                input: { text: true },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        case 'nemotron': {
            // https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b (text, image, video),
            // nemotron-3-nano-omni-30b-a3b-reasoning (video, audio, image, text) and
            // nemotron-3.5-content-safety (text, image).
            const omni = model.includes('-omni');
            const video = omni || /-vl(?:[-_.:]|$)/.test(model);
            const image =
                video ||
                model.includes('nemotron-nano-12b') ||
                (model.includes('content-safety') && isModelFamilyVersionGTE(model, 'nemotron-', 3, 5));
            return {
                input: { text: true, image, video, audio: omni },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: true,
            };
        }
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
    if (sourceModel.includes('gemini')) return getGeminiModelLimits(sourceModel);
    if (family === 'nemotron') {
        if (sourceModel.includes('nemotron-super-')) return { context_window: 256_000, max_output_tokens: 32_768 };
        // NVIDIA also names models generation-first (`nemotron-3-super-120b`, `nemotron-3.5-lightning`).
        const generation = (sourceModel.match(/nemotron-nano-(\d+)-/) ?? sourceModel.match(/nemotron-(\d+)[.-]/))?.[1];
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
    if (provider === Providers.xai && isSingleDigitGrokVersionGte(sourceModel, 4, 7)) {
        return { capabilities, context_window: 500_000 };
    }
    if (provider === Providers.groq) {
        // Groq publishes different completion limits from the canonical open-weight models.
        if (/gpt-oss-(?:20b|120b)(?:-|$)/.test(sourceModel)) {
            return { capabilities, context_window: 131_072, max_output_tokens: 65_536 };
        }
        if (isQwenVersionGte(sourceModel, 3, 8) && sourceModel.includes('-27b')) {
            return { capabilities, context_window: 131_072, max_output_tokens: 16_384 };
        }
    }
    if (provider === Providers.vertexai) {
        const vertexLimits = getVertexAIModelLimits(model);
        return {
            capabilities,
            // A documented hosted profile is authoritative, including an unknown output limit.
            ...(vertexLimits.context_window !== undefined ? vertexLimits : getCanonicalLimits(sourceModel, family)),
        };
    }

    // OpenRouter and other OpenAI-compatible transports retain the source model's semantic
    // capabilities, but cannot expose provider-native fields such as Vertex Flex or thinking_level.
    if (isOpenAICompatibleTransport(provider)) {
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
                'claude',
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
            ...getAzureFoundryModelLimits(sourceModel),
        };
    }

    return {
        capabilities,
        ...getCanonicalLimits(sourceModel, family),
    };
}

function getReasoningEffortLevels(model: string, family: string, provider: Providers): readonly string[] | undefined {
    if (provider === Providers.groq && !model.includes('coder') && isQwenVersionGte(model, 3, 8)) {
        return ['none', 'low', 'medium', 'high'];
    }
    // The 0813 release changed DeepSeek V4 Pro effort levels; earlier snapshots keep their own defaults.
    if (provider === Providers.togetherai && model.includes('deepseek-v4-pro-0813')) return ['high', 'max'];
    if (provider === Providers.groq && /qwen3(?:\.6-27b|-32b)(?:-|$)/.test(model)) return ['none'];
    // Together serves Qwen 3.5-3.7 as hybrid reasoning models: thinking is on by default and can only be switched off,
    // which the driver sends as `reasoning: { enabled: false }`. Qwen 3.8 takes low/medium/xhigh effort instead.
    // https://docs.together.ai/docs/inference/chat/reasoning, https://www.together.ai/models/qwen3-8-flash
    if (provider === Providers.togetherai && family === 'qwen' && !model.includes('coder')) {
        if (isQwenVersionGte(model, 3, 8)) return ['low', 'medium', 'xhigh'];
        if (isQwenVersionGte(model, 3, 5)) return ['none'];
    }
    if (family === 'gpt') {
        if (model.includes('gpt-oss')) {
            if (provider === Providers.groq && model.includes('safeguard')) return undefined;
            return provider === Providers.groq ||
                provider === Providers.togetherai ||
                isOpenAICompatibleTransport(provider) ||
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
            isOpenAICompatibleTransport(provider) ||
            provider === Providers.azure_foundry ||
            provider === Providers.bedrock_mantle ||
            provider === Providers.bedrock
        ) {
            if (/^o\d+(?:[-_.]|$)/.test(model)) return ['low', 'medium', 'high'];
            return Object.values(getOpenAIReasoningEffortLevels(model) ?? {});
        }
        return undefined;
    }
    if (family === 'gemini' && isOpenAICompatibleTransport(provider) && isGeminiModelVersionGte(model, '3.5')) {
        // Sampling-restricted generations (Flash 3.7+, Gemini 4+) reject `minimal`, matching native Vertex metadata.
        if (supportsGeminiXHighThinking(model)) return ['low', 'medium', 'high', 'xhigh'];
        if (hasGeminiSamplingRestriction(model)) return ['low', 'medium', 'high'];
        return ['minimal', 'low', 'medium', 'high'];
    }
    // GLM 5.3 replaces thinking on/off with low/high/max effort (default max). https://docs.z.ai/guides/llm/glm-5.3
    if (
        family === 'glm' &&
        isModelFamilyVersionGTE(model, 'glm-', 5, 3) &&
        (isOpenAICompatibleTransport(provider) || provider === Providers.togetherai || provider === Providers.mistralai)
    ) {
        return ['low', 'high', 'max'];
    }
    // Vertex accepts minimal, low, medium, high and max on DeepSeek V4.1 Flash, and rejects none and xhigh.
    // Reasoning is off until an effort is sent.
    if (provider === Providers.vertexai && family === 'deepseek' && isDeepSeekVersionGte(model, 4, 1)) {
        return ['minimal', 'low', 'medium', 'high', 'max'];
    }
    // Vertex rejects `max` on Wonder and Spark 1.3 and accepts it on Ember; no Muse model takes `none`.
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/meta
    if (provider === Providers.vertexai && family === 'muse') {
        return model.includes('muse-ember')
            ? ['minimal', 'low', 'medium', 'high', 'xhigh', 'max']
            : ['minimal', 'low', 'medium', 'high', 'xhigh'];
    }
    if (isOpenAICompatibleTransport(provider) && family === 'muse') {
        // Muse models always reason; Spark 1.3+, Wonder and Ember add max. https://openrouter.ai/meta
        if (model.includes('glimmer')) return ['low', 'medium', 'high', 'xhigh'];
        if (/muse-(?:wonder|ember)/.test(model) || isModelFamilyVersionGTE(model, 'muse-spark-', 1, 3)) {
            return ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
        }
        return ['minimal', 'low', 'medium', 'high', 'xhigh'];
    }
    if (isOpenAICompatibleTransport(provider) && family === 'step') return ['low', 'medium', 'high'];
    if (provider === Providers.mistralai && family === 'mistral') {
        return getMistralModelKnowledge(model).reasoning_effort_levels;
    }
    // Bedrock Runtime hosts Grok with the same reasoning controls as xAI.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-7.html
    if ((provider === Providers.xai || provider === Providers.bedrock) && family === 'grok') {
        return getGrokReasoningEffortLevels(model);
    }
    if (provider === Providers.openrouter) return getOpenRouterReasoningEffortLevels(model, family);
    return undefined;
}

/**
 * Effort levels of models that OpenRouter normalizes to its `reasoning.effort` field. Each rule matches the
 * `reasoning.supported_efforts` that OpenRouter's model listing publishes for the family
 * (https://openrouter.ai/api/v1/models). OpenRouter also lists `none` for models whose reasoning can be switched
 * off; the source-model rules (Claude, Grok, Mistral) keep their native levels, a subset of OpenRouter's list.
 */
function getOpenRouterReasoningEffortLevels(model: string, family: string): readonly string[] | undefined {
    switch (family) {
        case 'claude': {
            // OpenRouter publishes no `supported_efforts` for Claude 4.5 and older, so it offers no effort there.
            if (!isClaudeVersionGTE(model, 4, 6)) return undefined;
            const levels = getAvailableEffortLevels(model);
            return levels ? Object.values(levels) : undefined;
        }
        case 'grok':
            return getGrokReasoningEffortLevels(model);
        case 'mistral':
            return getMistralModelKnowledge(model).reasoning_effort_levels;
        case 'kimi':
            return isKimiVersionGte(model, 3, 0) ? ['none', 'low', 'high', 'max'] : undefined;
        case 'gemini':
            // Omni video models are chat models on OpenRouter, with two thinking levels.
            return isGeminiOmniVideoModel(model) ? ['low', 'high'] : undefined;
        case 'qwen': {
            // Qwen Flash, Omni and Coder models take no effort on OpenRouter.
            if (/(?:flash|omni|coder)/.test(model) || !isQwenVersionGte(model, 3, 8)) return undefined;
            if (!/-max(?:[-_.:]|$)/.test(model)) {
                // The 27B model reasons by default and can switch it off; the 2.4T model requires reasoning.
                return /-27b(?:[-_.:]|$)/.test(model) ? ['none', 'low', 'medium', 'xhigh'] : ['low', 'medium', 'xhigh'];
            }
            return isQwenVersionGte(model, 4, 0)
                ? ['none', 'low', 'high', 'xhigh']
                : ['minimal', 'low', 'medium', 'high', 'xhigh'];
        }
    }
    // Tencent Hunyuan Hy 3+ (`hy3`, `hy4-preview`); the Flash tier and the `hy-mt` translation models take no effort.
    const hunyuan = /(?:^|\/)hy(\d+)(?:-preview)?$/.exec(model);
    if (hunyuan && Number(hunyuan[1]) >= 3) return ['none', 'low', 'high'];
    return undefined;
}

function isOpenAICompatibleTransport(provider: Providers): boolean {
    return provider === Providers.openai_compatible || provider === Providers.openrouter;
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
