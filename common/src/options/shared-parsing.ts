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
    requiresAdaptiveThinkingOnly,
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
 * include_thoughts toggle — shown for Claude 3.7 and models with adaptive thinking
 * (Opus 4.6+, Sonnet 4.6+, all 4.7+).
 *
 * Returns an empty array for models that have no thinking support, so callers can
 * always spread the result without an explicit `if` guard.
 */
export function buildClaudeThinkingOptions(model: string): ModelOptionInfoItem[] {
    const supportsAdaptive = supportsAdaptiveThinking(model);
    if (!model.includes("-3-7") && !supportsAdaptive) return [];

    const adaptiveOnly = requiresAdaptiveThinkingOnly(model);
    return [
        {
            name: "include_thoughts",
            type: OptionType.boolean,
            default: false,
            description: supportsAdaptive
                ? (adaptiveOnly
                    ? "Show the summarized thinking content in the response"
                    : "Show the summarized thinking content in the response (default on this model)")
                : "Include the model's reasoning process in the response",
        },
    ];
}
