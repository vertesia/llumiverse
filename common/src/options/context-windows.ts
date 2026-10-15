import { getMistralModelKnowledge } from '../capability/mistral.js';
import { getGeminiModelLimits } from './gemini-limits.js';
import {
    isDeepSeekVersionGte,
    isKimiVersionGte,
    isMetaMuseModel,
    isModelFamilyVersionGTE,
    isOpenAIGptProModel,
    isOpenAIGptVersionGTE,
    parseClaudeVersion,
    parseOpenAIGptVersion,
} from './version-parsing.js';

/** Claude limits by parsed generation, so dashed (`claude-opus-4-6`) and dotted (`claude-opus-4.6`) ids agree. */
function getClaudeLimits(model: string): { context_window: number; max_output_tokens: number } {
    const claude = parseClaudeVersion(model);
    const atLeast = (major: number, minor: number) =>
        !!claude && (claude.major > major || (claude.major === major && claude.minor >= minor));
    const isOpus = claude?.variant === 'opus';
    // Sonnet 4.6 supports 128K output on Anthropic and Vertex; Bedrock Runtime overrides it to 64K.
    const isLongContext = atLeast(4, 7) || ((isOpus || claude?.variant === 'sonnet') && atLeast(4, 6));
    let max_output_tokens = 4_096; // Claude 3 older
    if (isLongContext) max_output_tokens = 128_000;
    else if (isOpus && atLeast(4, 5)) max_output_tokens = 64_000;
    else if (isOpus && atLeast(4, 0))
        max_output_tokens = 32_000; // Opus 4.0, 4.1
    else if (atLeast(3, 7))
        max_output_tokens = 64_000; // Sonnet 4.x, Haiku 4.5; 3.7 defaults to 64K
    else if (atLeast(3, 5)) max_output_tokens = 8_192;
    else if (model.includes('opus-')) max_output_tokens = 32_000; // Preserve the legacy undated-Opus metadata.
    return { context_window: isLongContext ? 1_000_000 : 200_000, max_output_tokens };
}

/**
 * Returns the canonical max output tokens for a model, or undefined for dedicated APIs without published limits.
 * Provider-specific limits must be applied by the provider option/capability layer.
 */
export function getMaxOutputTokens(model: string): number | undefined {
    model = model.toLowerCase();
    if (model.includes('claude')) return getClaudeLimits(model).max_output_tokens;
    // Gemini models
    if (model.includes('gemini')) return getGeminiModelLimits(model).max_output_tokens;
    // OpenAI o-series
    if (model.includes('o1-mini')) return 65_536;
    if (model.includes('o1')) return 100_000;
    if (/(?:^|[~/.])o\d+(?:[-_.]|$)/.test(model.toLowerCase())) return 100_000;
    // GPT models
    const gptVersion = parseOpenAIGptVersion(model);
    // GPT-5.x chat snapshots (`gpt-5-chat-latest`, `gpt-5.2-chat`) keep the smaller chat-model limits.
    if (/gpt-5(?:\.\d+)?-chat/.test(model)) return 16_384;
    if (isOpenAIGptProModel(model) && gptVersion?.major === 5 && gptVersion.minor === 0) return 272_000;
    if (isOpenAIGptVersionGTE(model, 5, 0)) return 128_000;
    if (model.includes('gpt-oss')) return 16_384;
    if (model.includes('gpt-4o')) return 16_384;
    if (model.includes('gpt-4')) return 8_192;
    if (model.includes('gpt-3.5')) return 4_096;
    // Grok
    if (isModelFamilyVersionGTE(model, 'grok-', 4, 3)) return 131_072;
    // Amazon Nova
    if (model.includes('nova')) return 10_000;
    // Mistral
    if (model.includes('mistral')) return 8_192;
    // DeepSeek
    if (model.includes('deepseek-r1-0528')) return 32_768;
    // DeepSeek V4 moved to a 1M context with up to 384K output. https://api-docs.deepseek.com/quick_start/pricing
    if (isDeepSeekVersionGte(model, 4, 0)) return 393_216;
    if (isDeepSeekVersionGte(model, 3, 2)) return 65_536;
    if (model.includes('deepseek')) return 128_000;
    // Qwen
    if (model.includes('qwen3-next')) return 262_144;
    if (model.includes('qwen3-coder')) return 65_536;
    if (model.includes('qwen')) return 32_768;
    // Kimi
    if (model.includes('kimi-k2-thinking')) return 262_144;
    // MiniMax
    if (isModelFamilyVersionGTE(model, 'minimax-m', 3, 0)) return 512_000;
    if (model.includes('minimax-m2')) return 196_608;
    // ZAI GLM
    if (isModelFamilyVersionGTE(model, 'glm-', 5, 3)) return 131_072;
    if (model.includes('glm-')) return 32_768;
    // Xiaomi MiMo and StepFun Step, as published for their OpenRouter listings
    if (isModelFamilyVersionGTE(model, 'mimo-v', 2, 6)) return 131_072;
    if (isModelFamilyVersionGTE(model, 'step-', 5, 0)) return 64_000;
    // Meta publishes no Muse output limit, and hosting providers report different caps.
    if (isMetaMuseModel(model)) return undefined;
    // Gemma
    if (model.includes('gemma-4')) return 128_000;
    // Llama
    if (model.includes('llama')) return 8_192;
    // Cohere
    if (model.includes('command-a')) return 8_000;
    if (model.includes('command')) return 4_096;

    return 8_192; // conservative default
}

/**
 * Returns the max input tokens for a known model (context window minus max output), or undefined when the context
 * window is not known.
 */
export function getMaxInputTokens(model: string): number | undefined {
    const contextWindow = getContextWindowSize(model);
    const maxOutput = getMaxOutputTokens(model);
    return contextWindow === undefined || maxOutput === undefined ? undefined : contextWindow - maxOutput;
}

/**
 * Returns the known context window size (input + output), or undefined rather than guessing for an unknown model.
 */
export function getContextWindowSize(model: string): number | undefined {
    model = model.toLowerCase();
    if (model.includes('claude')) return getClaudeLimits(model).context_window;
    // Gemini models
    if (model.includes('gemini')) return getGeminiModelLimits(model).context_window;
    // OpenAI o-series (check before gpt-4 to avoid false matches)
    if (/(?:^|[~/.])o\d+(?:[-_.]|$)/.test(model.toLowerCase())) return 200_000;
    // GPT models — provider-specific limits are applied by the provider overlay.
    if (/gpt-5(?:\.\d+)?-chat/.test(model)) return 128_000;
    if (model.includes('gpt-5.4-mini') || model.includes('gpt-5.4-nano')) return 400_000;
    if (isOpenAIGptVersionGTE(model, 5, 4)) return 1_050_000;
    if (isOpenAIGptVersionGTE(model, 5, 0)) return 400_000;
    if (model.includes('gpt-oss')) return 131_072;
    if (model.includes('gpt-4.1') || model.includes('gpt-4-1')) return 1_000_000;
    if (model.includes('gpt-4-turbo') || model.includes('gpt-4o')) return 128_000;
    if (model.includes('gpt-4')) return 8_000;
    if (model.includes('gpt-3.5')) return 16_000;
    // Grok
    if (isModelFamilyVersionGTE(model, 'grok-', 4, 3)) return 1_000_000;
    if (model.includes('grok-4.1') || model.includes('grok-4-1')) return 256_000;
    if (model.includes('grok')) return 131_072;
    // Amazon Nova
    if (model.includes('nova')) return 300_000;
    // Mistral
    if (/(?:mistral|mixtral|ministral|magistral|voxtral|codestral|devstral|leanstral|mathstral|pixtral)/.test(model)) {
        return getMistralModelKnowledge(model).context_window;
    }
    // DeepSeek
    if (isDeepSeekVersionGte(model, 4, 0)) return 1_048_576;
    if (isDeepSeekVersionGte(model, 3, 2)) return 163_840;
    if (model.includes('deepseek-r1-0528')) return 163_840;
    if (model.includes('deepseek')) return 128_000;
    // Qwen
    if (model.includes('qwen3-next')) return 262_144;
    if (model.includes('qwen3-coder')) return 262_144;
    if (model.includes('qwen')) return 262_144;
    // Kimi
    if (isKimiVersionGte(model, 3, 0)) return 1_048_576;
    if (isKimiVersionGte(model, 2, 5) || /kimi-k2-(?:thinking|0905)/.test(model)) return 262_144;
    if (model.includes('kimi-k2')) return 131_072;
    // MiniMax
    if (isModelFamilyVersionGTE(model, 'minimax-m', 3, 0)) return 1_048_576;
    if (model.includes('minimax-m2')) return 196_608;
    // ZAI GLM
    if (isModelFamilyVersionGTE(model, 'glm-', 5, 3)) return 1_000_000;
    if (model.includes('glm-')) return 128_000;
    // Meta Muse (Glimmer is the small 128K model), Xiaomi MiMo and StepFun Step
    if (model.includes('muse-glimmer')) return 131_072;
    if (isMetaMuseModel(model)) return 1_048_576;
    if (isModelFamilyVersionGTE(model, 'mimo-v', 2, 6)) return 1_050_000;
    if (isModelFamilyVersionGTE(model, 'step-', 5, 0)) return 1_048_576;
    // Gemma
    if (model.includes('gemma-4')) return 256_000;
    // Gemma 3 1B is 32K; the larger Gemma 3 models are 128K.
    if (model.includes('gemma-3-1b')) return 32_768;
    if (model.includes('gemma-3-')) return 131_072;
    // Llama
    if (isModelFamilyVersionGTE(model, 'llama-', 4, 0) || isModelFamilyVersionGTE(model, 'llama', 4, 0)) {
        return model.includes('scout') ? 10_000_000 : 1_000_000;
    }
    if (model.includes('llama-3.1') || model.includes('llama-3.2') || model.includes('llama-3.3')) return 128_000;
    if (model.includes('llama')) return 8_000;
    // Cohere
    if (model.includes('command-a')) return 256_000;
    if (model.includes('command')) return 128_000;

    return undefined;
}
