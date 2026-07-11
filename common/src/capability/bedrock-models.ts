import type { ModelCapabilities, ModelModalities } from '../types.js';

export type BedrockEndpoint = 'runtime' | 'mantle';

export interface BedrockModelKnowledge {
    input: ModelModalities;
    output: ModelModalities;
    context_window?: number;
    max_output_tokens?: number;
}

function includesAny(model: string, patterns: string[]): boolean {
    return patterns.some((pattern) => model.includes(pattern));
}

function isFamilyVersionGte(model: string, family: string, targetMajor: number, targetMinor = 0): boolean {
    const familyIndex = model.indexOf(family);
    if (familyIndex === -1) return false;
    const match = model.slice(familyIndex + family.length).match(/^(\d+)(?:[.-](\d+))?/);
    if (!match) return false;
    const major = Number(match[1]);
    const minor = Number(match[2] ?? 0);
    return major > targetMajor || (major === targetMajor && minor >= targetMinor);
}

export function normalizeBedrockModelId(model: string): string {
    const normalized = model.toLowerCase();
    const slash = normalized.lastIndexOf('/');
    const modelId = slash === -1 ? normalized : normalized.slice(slash + 1);
    return normalized.includes('inference-profile/') ? modelId.replace(/^[^.]+\./, '') : modelId;
}

function getModalities(model: string): Pick<BedrockModelKnowledge, 'input' | 'output'> {
    if (includesAny(model, ['nova-2-multimodal-embeddings', 'titan-embed', 'cohere.embed', 'marengo-embed'])) {
        return {
            input: {
                text: true,
                image: includesAny(model, ['multimodal', 'embed-image', 'embed-v4', 'marengo']),
                video: includesAny(model, ['multimodal', 'marengo']),
                audio: model.includes('multimodal'),
                embed: false,
            },
            output: { text: false, image: false, video: false, audio: false, embed: true },
        };
    }
    if (includesAny(model, ['nova-canvas', 'titan-image-generator'])) {
        return {
            input: { text: true, image: true, video: false, audio: false, embed: false },
            output: { text: false, image: true, video: false, audio: false, embed: false },
        };
    }
    if (model.includes('nova-reel')) {
        return {
            input: { text: true, image: true, video: false, audio: false, embed: false },
            output: { text: false, image: false, video: true, audio: false, embed: false },
        };
    }

    const image =
        (model.includes('anthropic.claude') && !model.includes('claude-3-5-haiku')) ||
        model.includes('google.gemma') ||
        includesAny(model, ['meta.llama3-2-11b', 'meta.llama3-2-90b', 'meta.llama4']) ||
        includesAny(model, [
            'mistral.magistral-',
            'mistral.ministral-3-',
            'mistral.mistral-large-3-',
            'mistral.pixtral-',
        ]) ||
        isFamilyVersionGte(model, 'kimi-k', 2, 5) ||
        model.includes('nemotron-nano-12b') ||
        model.includes('qwen3-vl') ||
        model.includes('palmyra-vision') ||
        (model.startsWith('openai.gpt-') && !model.includes('gpt-oss')) ||
        model.startsWith('xai.grok-');
    const video =
        (model.includes('amazon.nova') && !model.includes('nova-micro')) ||
        isFamilyVersionGte(model, 'google.gemma-', 4);
    const audio = includesAny(model, ['nova-sonic', 'mistral.voxtral-']) || model.includes('google.gemma-4-e2b');

    return {
        input: { text: !model.includes('nova-sonic-v1'), image, video, audio, embed: false },
        output: { text: true, image: false, video: false, audio: model.includes('nova-sonic'), embed: false },
    };
}

function getLimits(model: string): Pick<BedrockModelKnowledge, 'context_window' | 'max_output_tokens'> {
    if (model.includes('ai21.jamba')) return { context_window: 256_000, max_output_tokens: 4_096 };
    if (model.includes('amazon.nova-2')) return { context_window: 1_000_000, max_output_tokens: 65_536 };
    if (model.includes('amazon.nova-premier')) return { context_window: 1_000_000, max_output_tokens: 25_600 };
    if (model.includes('amazon.nova-micro')) return { context_window: 128_000, max_output_tokens: 5_120 };
    if (includesAny(model, ['amazon.nova-lite', 'amazon.nova-pro'])) {
        return { context_window: 300_000, max_output_tokens: 5_120 };
    }
    if (model.includes('anthropic.claude')) {
        if (
            includesAny(model, [
                'claude-fable-5',
                'claude-mythos',
                'claude-sonnet-5',
                'claude-opus-4-6',
                'claude-opus-4-7',
                'claude-opus-4-8',
            ])
        ) {
            return { context_window: 1_000_000, max_output_tokens: 131_072 };
        }
        if (model.includes('claude-sonnet-4-6')) return { context_window: 1_000_000, max_output_tokens: 65_536 };
        // Bedrock documents 64K for Haiku 4.5 but rejects max_tokens=64000; its upper bound is exclusive.
        if (model.includes('claude-haiku-4-5')) return { context_window: 200_000, max_output_tokens: 63_999 };
        if (includesAny(model, ['claude-opus-4-5', 'claude-sonnet-4'])) {
            return { context_window: 200_000, max_output_tokens: 65_536 };
        }
        if (model.includes('claude-opus-4-1')) return { context_window: 200_000, max_output_tokens: 32_000 };
        if (model.includes('claude-3-5-haiku')) return { context_window: 200_000, max_output_tokens: 8_192 };
        if (model.includes('claude-3-haiku')) return { context_window: 200_000, max_output_tokens: 4_096 };
    }
    if (model.includes('cohere.command-r')) return { context_window: 128_000, max_output_tokens: 4_096 };
    if (isFamilyVersionGte(model, 'deepseek.v', 3, 2)) {
        return { context_window: 163_840, max_output_tokens: 8_192 };
    }
    if (includesAny(model, ['deepseek.v3', 'deepseek.r1']))
        return { context_window: 128_000, max_output_tokens: 8_192 };
    if (model.includes('google.gemma-4-31b') || model.includes('google.gemma-4-26b')) {
        return { context_window: 256_000 };
    }
    if (model.includes('google.gemma-4-e2b')) return { context_window: 128_000 };
    if (model.includes('google.gemma-3-')) return { context_window: 128_000, max_output_tokens: 8_192 };
    if (model.includes('meta.llama4-scout')) return { context_window: 10_000_000, max_output_tokens: 8_192 };
    if (model.includes('meta.llama4-maverick')) return { context_window: 1_000_000, max_output_tokens: 8_192 };
    if (
        model.includes('meta.llama3-') &&
        !model.includes('llama3-1') &&
        !model.includes('llama3-2') &&
        !model.includes('llama3-3')
    ) {
        return { context_window: 8_192, max_output_tokens: 8_192 };
    }
    if (includesAny(model, ['meta.llama3-1', 'meta.llama3-2', 'meta.llama3-3'])) {
        return { context_window: 128_000, max_output_tokens: 4_096 };
    }
    if (isFamilyVersionGte(model, 'minimax.minimax-m', 2, 1)) {
        return { context_window: 196_000, max_output_tokens: 8_192 };
    }
    if (model.includes('minimax.minimax-m2')) return { context_window: 1_000_000, max_output_tokens: 8_192 };
    if (includesAny(model, ['mistral.devstral-2', 'mistral.mistral-large-3'])) {
        return { context_window: 256_000, max_output_tokens: 32_768 };
    }
    if (model.includes('mistral.magistral-')) return { context_window: 128_000, max_output_tokens: 40_960 };
    if (model.includes('mistral.ministral-3-')) return { context_window: 128_000, max_output_tokens: 8_192 };
    if (model.includes('mistral.pixtral-')) return { context_window: 128_000, max_output_tokens: 16_384 };
    if (
        includesAny(model, [
            'mistral.mistral-7b',
            'mistral.mistral-large-2402',
            'mistral.mistral-small-2402',
            'mistral.mixtral-',
            'mistral.voxtral-',
        ])
    ) {
        return { context_window: 32_000, max_output_tokens: model.includes('voxtral') ? undefined : 4_096 };
    }
    if (includesAny(model, ['moonshot.kimi-k2', 'moonshotai.kimi-k2'])) {
        return { context_window: 256_000, max_output_tokens: 16_384 };
    }
    if (includesAny(model, ['nvidia.nemotron-nano-3', 'nvidia.nemotron-super-3'])) {
        return { context_window: 256_000, max_output_tokens: model.includes('super') ? 32_768 : 8_192 };
    }
    if (model.includes('nvidia.nemotron-nano-')) return { context_window: 128_000, max_output_tokens: 8_192 };
    if (model.startsWith('openai.gpt-5')) return { context_window: 272_000 };
    if (model.includes('openai.gpt-oss')) return { context_window: 128_000, max_output_tokens: 16_384 };
    if (model.includes('qwen3-235b')) return { context_window: 256_000, max_output_tokens: 8_192 };
    if (model.includes('qwen3-32b')) return { context_window: 32_000, max_output_tokens: 8_192 };
    if (model.includes('qwen3-coder-30b')) return { context_window: 256_000, max_output_tokens: 16_384 };
    if (model.includes('qwen3-coder-480b')) return { context_window: 128_000, max_output_tokens: 16_384 };
    if (model.includes('qwen3-coder-next')) return { context_window: 256_000, max_output_tokens: 16_384 };
    if (includesAny(model, ['qwen3-next-', 'qwen3-vl-'])) return { context_window: 256_000, max_output_tokens: 8_192 };
    if (model.includes('writer.palmyra-vision')) return { context_window: 4_096, max_output_tokens: 4_096 };
    if (includesAny(model, ['writer.palmyra-x4', 'writer.palmyra-x5'])) {
        return { context_window: 128_000, max_output_tokens: 8_192 };
    }
    if (model.startsWith('xai.grok-')) return { context_window: 1_000_000 };
    if (includesAny(model, ['zai.glm-4.7', 'zai.glm-4.6']))
        return { context_window: 203_000, max_output_tokens: 4_096 };
    if (model.includes('zai.glm-5')) return { context_window: 200_000, max_output_tokens: 131_072 };
    return {};
}

function supportsRuntimeTools(model: string): boolean {
    if (model.includes('anthropic.claude')) return true;
    if (model.includes('ai21.jamba')) return model.includes('large');
    if (model.includes('amazon.nova')) {
        return includesAny(model, ['nova-2-lite', 'nova-lite', 'nova-micro', 'nova-premier']);
    }
    if (model.includes('cohere.command-r')) return true;
    if (model.includes('deepseek.')) return model.includes('deepseek.v3-v1');
    if (model.includes('meta.llama')) {
        return includesAny(model, ['llama3-1', 'llama3-2-90b', 'llama4-maverick']);
    }
    if (model.includes('mistral.')) {
        return includesAny(model, ['mistral-large-2402', 'mistral-small-2402', 'pixtral-large-2502']);
    }
    if (model.includes('qwen.')) return includesAny(model, ['qwen3-235b', 'qwen3-coder-480b']);
    return includesAny(model, ['writer.palmyra-x4', 'writer.palmyra-x5']);
}

export function getBedrockModelKnowledge(model: string): BedrockModelKnowledge {
    const modelId = normalizeBedrockModelId(model);
    return { ...getModalities(modelId), ...getLimits(modelId) };
}

export function getBedrockModelCapabilities(model: string, endpoint: BedrockEndpoint): ModelCapabilities {
    const modelId = normalizeBedrockModelId(model);
    if (
        endpoint === 'runtime' &&
        ((modelId.startsWith('openai.gpt-') && !modelId.includes('gpt-oss')) ||
            modelId.startsWith('xai.grok-') ||
            isFamilyVersionGte(modelId, 'google.gemma-', 4))
    ) {
        return { input: {}, output: {} };
    }
    const knowledge = getBedrockModelKnowledge(modelId);
    const toolSupport = endpoint === 'mantle' ? true : supportsRuntimeTools(modelId);
    return {
        input: knowledge.input,
        output: knowledge.output,
        tool_support: toolSupport,
        tool_support_streaming: toolSupport,
    };
}
