import { getModelCapabilitiesAnthropic } from './capability/anthropic.js';
import { getModelCapabilitiesAzureFoundry } from './capability/azure_foundry.js';
import { getModelCapabilitiesBedrock } from './capability/bedrock.js';
import { getModelCapabilitiesBedrockMantle } from './capability/bedrock_mantle.js';
import { getModelCapabilitiesOpenAI } from './capability/openai.js';
import { getModelCapabilitiesVertexAI } from './capability/vertexai.js';
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
    // Preserve tool_support_streaming from provider-specific capabilities if set,
    // otherwise default to false for providers that haven't been verified
    return capabilities;
}

function _getModelCapabilities(model: string, provider?: string | Providers): ModelCapabilities {
    switch (provider?.toLowerCase()) {
        case Providers.anthropic:
            return getModelCapabilitiesAnthropic(model);
        case Providers.vertexai:
            return getModelCapabilitiesVertexAI(model);
        case Providers.openai:
            return getModelCapabilitiesOpenAI(model);
        case Providers.openai_compatible:
            return getModelCapabilitiesOpenAICompatible(model);
        case Providers.bedrock:
            return getModelCapabilitiesBedrock(model);
        case Providers.bedrock_mantle:
            return getModelCapabilitiesBedrockMantle(model);
        case Providers.azure_foundry:
            // Azure Foundry uses OpenAI capabilities
            return getModelCapabilitiesAzureFoundry(model);
        case Providers.groq:
        case Providers.mistralai:
            // These providers host text models that generally support tool use
            return getModelCapabilitiesOpenAICompatible(model);
        case Providers.togetherai:
            // Same OpenAI-compatible tool-use default, but also flag the natively-multimodal
            // model families TogetherAI hosts so is_multimodal is reported correctly.
            return getModelCapabilitiesTogetherAI(model);
        case Providers.xai:
            // xAI (Grok) models support tool use and are text-based
            return {
                input: { text: true, image: model.includes('vision') },
                output: { text: true },
                tool_support: true,
                tool_support_streaming: false, // Conservative - may work but not tested
            };
        default:
            // Guess the provider based on the model name
            if (model.startsWith('gpt')) {
                return getModelCapabilitiesOpenAI(model);
            } else if (model.startsWith('claude')) {
                return getModelCapabilitiesAnthropic(model);
            } else if (model.startsWith('grok')) {
                // xAI Grok models
                return {
                    input: { text: true, image: model.includes('vision') },
                    output: { text: true },
                    tool_support: true,
                    tool_support_streaming: false,
                };
            } else if (model.startsWith('publishers/')) {
                return getModelCapabilitiesVertexAI(model);
            } else if (model.startsWith('arn:aws')) {
                return getModelCapabilitiesBedrock(model);
            }
            // Fallback to a generic model with no capabilities
            return { input: {}, output: {} } satisfies ModelCapabilities;
    }
}

// Patterns for models known NOT to support tool use on OpenAI-compatible endpoints
const NO_TOOL_SUPPORT_PATTERNS = ['image', 'embed', 'moderation', 'whisper', 'sora', 'dall-e', 'tts'];

/**
 * For OpenAI-compatible endpoints (e.g., OpenRouter), try OpenAI capability lookup first.
 * If no explicit match is found, default to tool_support: true since most models
 * on these platforms support tool use. Blacklist known non-tool-supporting patterns.
 */
function getModelCapabilitiesOpenAICompatible(model: string): ModelCapabilities {
    const caps = getModelCapabilitiesOpenAI(model);
    if (caps.tool_support !== undefined) {
        return caps;
    }
    const normalized = model.toLowerCase();
    const isNonToolModel = NO_TOOL_SUPPORT_PATTERNS.some((p) => normalized.includes(p));
    return {
        input: { text: true },
        output: { text: true },
        tool_support: !isNonToolModel,
        tool_support_streaming: !isNonToolModel,
    };
}

// TogetherAI vision-capable model families. Conservative on purpose: only families that are
// natively multimodal when served on Together are listed, so text-only models are not falsely
// flagged as multimodal. This list can be extended as Together adds vision models.
const TOGETHER_VISION_PATTERNS = [
    'vision', // meta-llama/Llama-3.2-*-Vision-Instruct
    'llama-4', // Llama 4 Scout / Maverick are natively multimodal
    'gemma-3', // Gemma 3 is multimodal (gemma-2 and earlier are text-only)
    'qwen2-vl',
    'qwen2.5-vl',
    '-vl-', // generic Qwen*-VL / other *-VL-* naming
];

/**
 * TogetherAI capability resolver. Starts from the OpenAI-compatible defaults (tool_support, etc.)
 * and additionally marks `input.image: true` for known natively-multimodal families, so the
 * driver's `is_multimodal` flag is accurate. TogetherAI vision requests are sent via the shared
 * OpenAI Chat Completions driver path.
 */
function getModelCapabilitiesTogetherAI(model: string): ModelCapabilities {
    const caps = getModelCapabilitiesOpenAICompatible(model);
    const normalized = model.toLowerCase();
    if (TOGETHER_VISION_PATTERNS.some((p) => normalized.includes(p))) {
        caps.input = { ...caps.input, image: true };
    }
    return caps;
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
