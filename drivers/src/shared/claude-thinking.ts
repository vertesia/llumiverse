import {
    hasSamplingParameterRestriction,
    isClaudeVersionGTE,
    supportsAdaptiveThinking,
} from "@llumiverse/core";

/**
 * Common Claude model options relevant to thinking/effort configuration.
 * Works with both VertexAIClaudeOptions and BedrockClaudeOptions.
 */
export interface ClaudeThinkingInput {
    thinking_budget_tokens?: number;
    effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
    /** Controls whether thinking content is included in the response. Does not enable thinking. */
    include_thoughts?: boolean;
}

/**
 * Thinking configuration for Claude models.
 * The shape is identical for both the Anthropic SDK (`thinking` field)
 * and the Bedrock Converse API (`reasoning_config` field).
 */
export type ClaudeThinkingConfig =
    | { type: "adaptive"; display: "summarized" | "omitted" }
    | { type: "enabled"; budget_tokens: number }
    | { type: "disabled" }
    | undefined;

/** Valid effort level values for Claude models. */
export type EffortLevel = 'low' | 'medium' | 'high' | 'xhigh' | 'max';

/**
 * Output config for Claude effort parameter.
 */
export type ClaudeOutputConfig = { effort: EffortLevel } | undefined;

/**
 * Result of resolving Claude thinking and effort configuration.
 */
export interface ClaudeThinkingResult {
    /** Thinking/reasoning config to include in the API payload. */
    thinking: ClaudeThinkingConfig;
    /** Output config (effort) to include in the API payload, if applicable. */
    outputConfig: ClaudeOutputConfig;
    /** Whether sampling parameters (temperature, top_p, top_k) should be stripped. */
    hasSamplingRestriction: boolean;
    /** Whether the model supports thinking at all (>= Claude 3.7). */
    supportsThinking: boolean;
}

/**
 * Resolve thinking and effort configuration for a Claude model.
 *
 * - Extended thinking: enabled by setting `thinking_budget_tokens`.
 * - Adaptive thinking: enabled by setting `effort` on models that support it (Opus 4.6+, Sonnet 4.6+).
 * - `include_thoughts`: display-only; does not enable thinking.
 *
 * @param model - The model identifier string
 * @param options - User-provided Claude options (thinking_budget_tokens, effort, include_thoughts)
 */
export function resolveClaudeThinking(model: string, options?: ClaudeThinkingInput): ClaudeThinkingResult {
    const supportsAdaptive = supportsAdaptiveThinking(model);
    const samplingRestriction = hasSamplingParameterRestriction(model);
    const supportsThinking = isClaudeVersionGTE(model, 3, 7);
    const budgetTokens = options?.thinking_budget_tokens;
    // Adaptive thinking is active when the caller supplies an effort level on a
    // model that supports it. Extended thinking is active when a budget is set.
    const adaptiveEnabled = supportsAdaptive && options?.effort != null;
    const extendedEnabled = budgetTokens != null;

    let thinking: ClaudeThinkingConfig;

    if (!supportsThinking) {
        // Pre-3.7 models: no thinking support
        thinking = undefined;
    } else if (extendedEnabled) {
        // Explicit budget — use extended thinking regardless of adaptive support.
        // On adaptive models this uses the deprecated path, but user input takes priority.
        thinking = {
            type: "enabled" as const,
            budget_tokens: budgetTokens,
        };
    } else if (supportsAdaptive) {
        // Adaptive models: enable when effort is set, omit otherwise (thinking is OFF by default).
        // display controls whether thinking blocks are returned; defaults to omitted.
        thinking = adaptiveEnabled
            ? { type: "adaptive" as const, display: options?.include_thoughts ? "summarized" : "omitted" }
            : undefined;
    } else {
        // Older thinking models (3.7, 4.5): no adaptive support, thinking is always disabled
        // unless an explicit budget is provided (handled above).
        thinking = { type: "disabled" as const };
    }

    // Output config for effort parameter (Opus 4.5+, Sonnet 4.6+, all 4.7+)
    const outputConfig: ClaudeOutputConfig = options?.effort
        ? { effort: options.effort }
        : undefined;

    return {
        thinking,
        outputConfig,
        hasSamplingRestriction: samplingRestriction,
        supportsThinking,
    };
}
