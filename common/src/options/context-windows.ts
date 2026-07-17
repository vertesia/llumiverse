<<<<<<< HEAD
import { isClaudeVersionGTE, isModelFamilyVersionGTE } from './version-parsing.js';

function isDeepSeekV32OrLater(model: string): boolean {
    return isModelFamilyVersionGTE(model, 'deepseek.v', 3, 2) || isModelFamilyVersionGTE(model, 'deepseek-v', 3, 2);
}
=======
import {
    isClaudeVersionGTE,
    isOpenAIGptProModel,
    isOpenAIGptVersionGTE,
    parseOpenAIGptVersion,
} from './version-parsing.js';
>>>>>>> 49b427d (fix: improve Claude reasoning and truncation reliability (#539))

/**
 * Returns the max output tokens for a given model (provider-agnostic).
 * When a model's limits vary by provider, returns a conservative value
 * that works across all providers.
 */
export function getMaxOutputTokens(model: string): number {
    // Claude models
    if (model.includes('claude')) {
        if (isClaudeVersionGTE(model, 4, 7)) return 128_000;
        if (model.includes('opus-4-6')) return 128_000;
        if (model.includes('opus-4-5')) return 64_000;
        if (model.includes('opus-')) return 32_000; // Opus 4.0, 4.1
        if (model.includes('-4-')) return 64_000; // Sonnet 4.x, Haiku 4.5
        if (model.includes('-3-7')) return 64_000; // 128K with beta header, default 64K
        if (model.includes('-3-5')) return 8_192;
        return 4_096; // Claude 3 older
    }
    // Gemini models
    if (model.includes('gemini')) {
        if (model.includes('-1.0-')) return 2_048;
        if (model.includes('flash-image') || model.includes('pro-image')) return 32_768;
        return 65_535; // Gemini 1.5, 2.0, 2.5, 3 — API upper bound is exclusive
    }
    // OpenAI o-series
    if (model.includes('o1-mini')) return 65_536;
    if (model.includes('o1')) return 100_000;
    if (model.includes('o3') || model.includes('o4')) return 100_000;
    // GPT models
<<<<<<< HEAD
    if (isModelFamilyVersionGTE(model, 'openai.gpt-', 5, 4)) return 128_000;
    if (model.includes('gpt-5')) return 128_000;
    if (model.includes('gpt-oss')) return 16_384;
=======
    const gptVersion = parseOpenAIGptVersion(model);
    if (isOpenAIGptProModel(model) && gptVersion?.major === 5 && gptVersion.minor === 0) return 272_000;
    if (isOpenAIGptVersionGTE(model, 5, 0)) return 128_000;
>>>>>>> 49b427d (fix: improve Claude reasoning and truncation reliability (#539))
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
    if (isDeepSeekV32OrLater(model)) return 65_536;
    if (model.includes('deepseek')) return 128_000;
    // Qwen
    if (model.includes('qwen3-next')) return 262_144;
    if (model.includes('qwen3-coder')) return 65_536;
    if (model.includes('qwen')) return 32_768;
    // Kimi
    if (model.includes('kimi-k2-thinking')) return 262_144;
    // MiniMax
    if (model.includes('minimax-m2')) return 196_608;
    // ZAI GLM
    if (model.includes('glm-')) return 32_768;
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
 * Returns the max input tokens for a given model (context window minus max output).
 */
export function getMaxInputTokens(model: string): number {
    return getContextWindowSize(model) - getMaxOutputTokens(model);
}

/**
 * Returns the context window size (input + output) for a given model.
 */
export function getContextWindowSize(model: string): number {
    // Claude models
    if (model.includes('claude')) {
        if (isClaudeVersionGTE(model, 4, 7)) return 1_000_000;
        return 200_000;
    }
    // Gemini models
    if (model.includes('gemini')) {
        if (model.includes('-1.0-')) return 32_000;
        return 1_000_000; // Gemini 1.5, 2.0, 2.5, 3 all support 1M
    }
    // OpenAI o-series (check before gpt-4 to avoid false matches)
    if (model.includes('o1') || model.includes('o3') || model.includes('o4')) return 200_000;
    // GPT models — check specific variants before generic gpt-4
<<<<<<< HEAD
    if (isModelFamilyVersionGTE(model, 'openai.gpt-', 5, 4)) return 272_000;
    if (model.includes('gpt-5')) return 400_000;
    if (model.includes('gpt-oss')) return 131_072;
=======
    if (isOpenAIGptVersionGTE(model, 5, 4)) return 1_050_000;
    if (isOpenAIGptVersionGTE(model, 5, 0)) return 400_000;
>>>>>>> 49b427d (fix: improve Claude reasoning and truncation reliability (#539))
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
    if (model.includes('mistral-large')) return 128_000;
    if (model.includes('mistral')) return 32_000;
    // DeepSeek
    if (isDeepSeekV32OrLater(model)) return 163_840;
    if (model.includes('deepseek-r1-0528')) return 163_840;
    if (model.includes('deepseek')) return 128_000;
    // Qwen
    if (model.includes('qwen3-next')) return 262_144;
    if (model.includes('qwen3-coder')) return 262_144;
    if (model.includes('qwen')) return 262_144;
    // Kimi
    if (model.includes('kimi-k2-thinking')) return 262_144;
    // MiniMax
    if (model.includes('minimax-m2')) return 196_608;
    // ZAI GLM
    if (model.includes('glm-')) return 128_000;
    // Gemma
    if (model.includes('gemma-4')) return 256_000;
    // Llama
    if (model.includes('llama-4') || model.includes('llama4')) return 1_000_000;
    if (model.includes('llama-3.1') || model.includes('llama-3.2') || model.includes('llama-3.3')) return 128_000;
    if (model.includes('llama')) return 8_000;
    // Cohere
    if (model.includes('command-a')) return 256_000;
    if (model.includes('command')) return 128_000;

    return 128_000; // conservative default
}
