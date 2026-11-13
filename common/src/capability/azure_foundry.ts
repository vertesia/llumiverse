import { isKimiVersionGte, isModelFamilyVersionGTE, isSingleDigitGrokVersionGte } from '../options/version-parsing.js';
import type { ModelCapabilities, ModelModalities } from '../types.js';
import { isMistralLargeGte } from './mistral.js';

// Record of Azure Foundry model capabilities keyed by model ID (lowercased)
// Only include models with specific exceptions that differ from their family patterns
const RECORD_MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
    // OpenAI O-series exceptions - o1-mini doesn't have tool support like other o1 models
    'o1-mini': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    // OpenAI o3 is text-only unlike other o-series models
    o3: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },

    // Models with special properties not covered by family patterns
    'deepseek-r1-0528': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'deepseek-v3-0324': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'mistral-medium-2505': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'mistral-nemo': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'llama-4-scout-17b-16e-instruct': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
};

// Populate RECORD_FAMILY_CAPABILITIES as a const record (lowest common denominator for each family)
const RECORD_FAMILY_CAPABILITIES: Record<string, ModelCapabilities> = {
    // OpenAI GPT families
    'gpt-3.5-turbo': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'gpt-35-turbo': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'gpt-35': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'gpt-4': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    'gpt-4.1': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    'gpt-4.5': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    'gpt-4o': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    gpt: {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },

    // OpenAI O-series families
    o1: {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'o1-preview': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'o1-pro': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'o3-mini': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'o4-mini': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    o4: {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    o: {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },

    // Llama families
    'llama-3.1': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'llama-3.2': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'llama-3.3': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'llama-3': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'llama-4': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    llama: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },

    // Mistral families
    'mistral-large': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    'mistral-small': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    mistral: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },

    // Microsoft Phi families
    'phi-4': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    phi: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },

    // DeepSeek families
    'deepseek-r1': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },
    'deepseek-v3': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: false,
    },
    deepseek: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: false,
        tool_support_streaming: false,
    },

    // AI21 families
    'ai21-jamba': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    ai21: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    jamba: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },

    // Cohere families
    'cohere-command': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    cohere: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    command: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },

    // xAI families
    'grok-3': {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    grok: {
        input: { text: true, image: false, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
};

// Fallback pattern lists for inferring modalities and tool support
const IMAGE_INPUT_MODELS = ['image', 'vision'];
const VIDEO_INPUT_MODELS = ['video'];
const AUDIO_INPUT_MODELS = ['audio'];
const TEXT_INPUT_MODELS = ['text'];
const IMAGE_OUTPUT_MODELS = ['image'];
const VIDEO_OUTPUT_MODELS = ['video'];
const AUDIO_OUTPUT_MODELS = ['audio'];
const TEXT_OUTPUT_MODELS = ['text'];
const EMBEDDING_OUTPUT_MODELS = ['embed'];
const TOOL_SUPPORT_MODELS = [
    'tool',
    'gpt-4',
    'gpt-4o',
    'o1',
    'o3',
    'o4',
    'llama-3',
    'mistral-large',
    'mistral-small',
    'jamba',
    'cohere',
    'command',
    'grok',
];

function modelMatches(modelName: string, patterns: string[]): boolean {
    return patterns.some((pattern) => modelName.includes(pattern));
}

/**
 * Context and output limits that Microsoft Foundry documents differently from the source provider, by family.
 * Other models keep the source model's limits. Same sources as {@link FOUNDRY_PARTNER_RULES}.
 */
export function getAzureFoundryModelLimits(model: string): { context_window?: number; max_output_tokens?: number } {
    const normalized = parseAzureFoundryModelId(model).baseModel.toLowerCase();
    // Kimi K3 and later: 1M context, 262,144 output. K2.5 to K2.x: 262,144 for both.
    if (isKimiVersionGte(normalized, 3, 0)) return { context_window: 1_000_000, max_output_tokens: 262_144 };
    if (isKimiVersionGte(normalized, 2, 5)) return { context_window: 262_144, max_output_tokens: 262_144 };
    // Grok 4.7 and later share a 500,000-token window between input and output.
    if (isSingleDigitGrokVersionGte(normalized, 4, 7)) return { context_window: 500_000 };
    if (normalized.startsWith('mai-thinking')) return { context_window: 256_000, max_output_tokens: 64_000 };
    return {};
}

/**
 * Get the full ModelCapabilities for a Microsoft Foundry model.
 * Checks RECORD_MODEL_CAPABILITIES first, then falls back to family pattern matching.
 */
export function getModelCapabilitiesAzureFoundry(model: string): ModelCapabilities {
    // Extract base model from composite ID (deployment::baseModel)
    const { baseModel } = parseAzureFoundryModelId(model);
    const normalized = baseModel.toLowerCase();

    // 1. Exact match in record
    const record = RECORD_MODEL_CAPABILITIES[normalized];
    if (record) {
        return applyFoundryPartnerRules(record, normalized);
    }

    if (isModelFamilyVersionGTE(normalized, 'llama-', 4, 0) || isModelFamilyVersionGTE(normalized, 'llama', 4, 0)) {
        // Future Llama deployments inherit the latest known Foundry family behavior.
        return applyFoundryPartnerRules(RECORD_FAMILY_CAPABILITIES['llama-4'], normalized);
    }

    // 2. Fallback: find the longest matching family prefix in RECORD_FAMILY_CAPABILITIES
    let bestFamilyKey: string | undefined;
    let bestFamilyLength = 0;
    for (const key of Object.keys(RECORD_FAMILY_CAPABILITIES)) {
        if (normalized.startsWith(key) && key.length > bestFamilyLength) {
            bestFamilyKey = key;
            bestFamilyLength = key.length;
        }
    }
    if (bestFamilyKey) {
        return applyFoundryPartnerRules(RECORD_FAMILY_CAPABILITIES[bestFamilyKey], normalized);
    }

    // 3. Fallback: infer from normalized name using patterns
    const input: ModelModalities = {
        text: modelMatches(normalized, TEXT_INPUT_MODELS) || true, // Default to text input
        image: modelMatches(normalized, IMAGE_INPUT_MODELS) || undefined,
        video: modelMatches(normalized, VIDEO_INPUT_MODELS) || undefined,
        audio: modelMatches(normalized, AUDIO_INPUT_MODELS) || undefined,
        embed: false,
    };
    const output: ModelModalities = {
        text: modelMatches(normalized, TEXT_OUTPUT_MODELS) || true, // Default to text output
        image: modelMatches(normalized, IMAGE_OUTPUT_MODELS) || undefined,
        video: modelMatches(normalized, VIDEO_OUTPUT_MODELS) || undefined,
        audio: modelMatches(normalized, AUDIO_OUTPUT_MODELS) || undefined,
        embed: modelMatches(normalized, EMBEDDING_OUTPUT_MODELS) || undefined,
    };
    const tool_support = modelMatches(normalized, TOOL_SUPPORT_MODELS) || undefined;
    const tool_support_streaming = tool_support || undefined;

    const inferredCapabilities = { input, output, tool_support, tool_support_streaming };
    return applyFoundryPartnerRules(inferredCapabilities, normalized);
}

/**
 * What Microsoft Foundry documents for a model it serves through Chat Completions, by family and generation:
 * whether it calls tools, and the input modalities beyond text. A family that is not listed here keeps tools off,
 * because the catalog does not say that its Foundry deployment accepts `tools`.
 *
 * Sources (the "Tool calling" and "Input" lines of each model card):
 * https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
 * https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-from-partners
 */
interface FoundryPartnerRule {
    matches: (model: string) => boolean;
    tools: boolean;
    /** Image input; a function when only some generations of the family take images. */
    image?: boolean | ((model: string) => boolean);
    audio?: boolean;
}

const FOUNDRY_PARTNER_RULES: FoundryPartnerRule[] = [
    // DeepSeek V3.2 and later (V3.2, V3.2-Speciale, V4 Pro/Flash, V4.1 Flash) call tools; all are text-only on
    // Foundry. R1 and V3-0324 are not documented with tools.
    { matches: isDeepSeekV32OrLater, tools: true },
    // Moonshot Kimi K2.5 and later take text and images and call tools.
    { matches: (model) => isKimiVersionGte(model, 2, 5), tools: true, image: true },
    // Every Grok model Foundry sells calls tools. Grok 4.1 Fast and Grok 4.6 and later also take images; Grok 4,
    // 4.3, 4.20 and Code Fast are text-only.
    {
        matches: (model) => model.startsWith('grok-'),
        tools: true,
        image: (model) => model.startsWith('grok-4.1-fast') || isSingleDigitGrokVersionGte(model, 4, 6),
    },
    // Mistral Large calls tools; Large 3 and later also take images. Mistral Small and Ministral call tools.
    {
        matches: (model) => model.startsWith('mistral-large'),
        tools: true,
        image: (model) => isMistralLargeGte(model, 3),
    },
    { matches: (model) => model.startsWith('mistral-small') || model.startsWith('ministral'), tools: true },
    // Mistral Medium takes images but is documented without tool calling.
    { matches: (model) => model.startsWith('mistral-medium'), tools: false, image: true },
    // Cohere Command A (and Command A Plus) call tools.
    { matches: (model) => /^(?:cohere-)?command-a(?:[-_.]|$)/.test(model), tools: true },
    // MAI-Thinking is a Chat Completions reasoning model with tool calling.
    { matches: (model) => model.startsWith('mai-thinking'), tools: true },
    // Phi-4 multimodal takes images and audio; no Phi model is documented with tool calling.
    { matches: (model) => model.startsWith('phi-4-multimodal'), tools: false, image: true, audio: true },
];
/** DeepSeek writes minor versions after a dot (`deepseek-v3.2`); a dash starts a date snapshot (`deepseek-v3-0324`). */
function isDeepSeekV32OrLater(model: string): boolean {
    const match = model.match(/^deepseek-v(\d+)(?:\.(\d+))?(?:[-_]|$)/);
    if (!match) return false;
    const major = Number(match[1]);
    const minor = Number(match[2] ?? 0);
    return major > 3 || (major === 3 && minor >= 2);
}

function isOpenAIModel(model: string): boolean {
    return /^(?:gpt-|o1|o3|o4)/.test(model);
}

/** Applies the documented tool calling and input modalities of non-OpenAI Foundry models. */
function applyFoundryPartnerRules(capabilities: ModelCapabilities, model: string): ModelCapabilities {
    if (isOpenAIModel(model)) return capabilities;
    const rule = FOUNDRY_PARTNER_RULES.find((candidate) => candidate.matches(model));
    if (!rule) {
        return { ...capabilities, tool_support: false, tool_support_streaming: false };
    }
    const image = typeof rule.image === 'function' ? rule.image(model) : rule.image;
    return {
        ...capabilities,
        input: {
            ...capabilities.input,
            ...(image !== undefined ? { image } : {}),
            ...(rule.audio !== undefined ? { audio: rule.audio } : {}),
        },
        tool_support: rule.tools,
        tool_support_streaming: rule.tools,
    };
}

// Helper function to parse composite model IDs
function parseAzureFoundryModelId(compositeId: string): { deploymentName: string; baseModel: string } {
    const parts = compositeId.split('::');
    if (parts.length === 2) {
        return {
            deploymentName: parts[0],
            baseModel: parts[1],
        };
    }

    // Backwards compatibility: if no delimiter found, treat as deployment name
    return {
        deploymentName: compositeId,
        baseModel: compositeId,
    };
}
