/**
 * Shared Claude option builders used by multiple providers (Bedrock, VertexAI, …).
 *
 * Centralising these helpers means a single edit propagates to all drivers, and
 * the per-provider files stay focused on provider-specific concerns (base options,
 * max-tokens limits for non-Claude models, etc.).
 */

import { type ModelOptionInfoItem, OptionType } from "../types.js";
import { getMaxOutputTokens } from "./context-windows.js";
import {
    getAvailableEffortLevels,
    isClaudeVersionGTE,
    supportsAdaptiveThinking,
} from "./version-parsing.js";

// ============================================================================
// Max tokens
// ============================================================================

/**
 * Canonical max output-tokens limit for Claude models, shared by all providers.
 *
 * - Claude 3.7:   128 K (all providers support extended output)
 * - Claude Opus 4.7+: 128 K
 * - All others:   delegated to {@link getMaxOutputTokens}
 */
export function getClaudeMaxTokensLimit(model: string): number {
    if (model.includes('-3-7')) return 128000;
    if (model.includes('opus-4-7')) return 128000;
    return getMaxOutputTokens(model);
}

// ============================================================================
// Cache options
// ============================================================================

/** cache_enabled toggle — identical across providers. */
export function buildClaudeCacheOptions(): ModelOptionInfoItem[] {
    return [
        {
            name: "cache_enabled",
            type: OptionType.boolean,
            default: false,
            description: "Enable prompt caching. Injects cache breakpoints at the system prompt, tools, and conversation pivot.",
        },
    ];
}

/**
 * cache_ttl selector — only shown when caching is enabled.
 * Pass `option?.cache_enabled` from the current saved options.
 */
export function buildClaudeCacheTtlOptions(cacheEnabled?: boolean): ModelOptionInfoItem[] {
    if (!cacheEnabled) return [];
    return [
        {
            name: "cache_ttl",
            type: OptionType.enum,
            enum: { "5 minutes (default)": "5m", "1 hour": "1h" },
            default: "5m",
            description: "TTL for cache breakpoints. '1h' requires extended caching to be enabled on your account.",
        },
    ];
}

// ============================================================================
// Effort option
// ============================================================================

/**
 * Effort selector — shown only for models that support it (Opus 4.5+, Sonnet 4.6+, all 4.7+).
 * Returns an empty array for unsupported models.
 */
export function buildClaudeEffortOptions(model: string): ModelOptionInfoItem[] {
    const effortLevels = getAvailableEffortLevels(model);
    if (!effortLevels) return [];
    return [
        {
            name: "effort",
            type: OptionType.enum,
            enum: effortLevels,
            description: "Controls how many tokens Claude uses when responding. Lower effort trades thoroughness for speed and cost savings.",
        },
    ];
}

// ============================================================================
// Thinking / reasoning options
// ============================================================================

/**
 * Thinking budget option — shown for non-adaptive Claude thinking models (3.7, 4.5).
 * Setting this enables extended thinking with the given token budget.
 *
 * Returns an empty array for models that don't support extended thinking or that
 * use adaptive thinking instead (where effort is the control knob).
 */
export function buildClaudeThinkingBudgetOption(model: string): ModelOptionInfoItem[] {
    // Adaptive-only models (Opus 4.7+) don't accept budget_tokens at all.
    // Adaptive models (Opus/Sonnet 4.6) still accept budget_tokens but it's deprecated;
    // those models should use effort instead. Show budget only for non-adaptive thinking models.
    if (!isClaudeVersionGTE(model, 3, 7) || supportsAdaptiveThinking(model)) return [];
    return [
        {
            name: "thinking_budget_tokens",
            type: OptionType.numeric,
            min: 1024,
            integer: true,
            step: 1024,
            description: "Token budget for extended thinking. Enables thinking when set.",
        },
    ];
}

/**
 * include_thoughts display toggle — shown for all Claude thinking-capable models.
 * Controls whether thinking content is returned in the response.
 * This does not enable thinking; set thinking_budget_tokens (extended) or effort (adaptive).
 *
 * Returns an empty array for models with no thinking support.
 */
export function buildClaudeIncludeThoughtsOption(model: string): ModelOptionInfoItem[] {
    if (!isClaudeVersionGTE(model, 3, 7)) return [];
    return [
        {
            name: "include_thoughts",
            type: OptionType.boolean,
            default: false,
            description: "Include the model's thinking content in the response.",
        },
    ];
}

/**
 * @deprecated Use buildClaudeThinkingBudgetOption and buildClaudeIncludeThoughtsOption separately.
 * Kept for backwards compatibility — delegates to the two new helpers.
 */
export function buildClaudeThinkingOptions(model: string): ModelOptionInfoItem[] {
    return [
        ...buildClaudeThinkingBudgetOption(model),
        ...buildClaudeIncludeThoughtsOption(model),
    ];
}
