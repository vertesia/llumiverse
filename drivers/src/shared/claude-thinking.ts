import {
    isClaudeVersionGTE,
    hasSamplingParameterRestriction,
    supportsAdaptiveThinking,
} from "@llumiverse/core";

/**
 * Common Claude model options relevant to thinking/effort configuration.
 * Works with both VertexAIClaudeOptions and BedrockClaudeOptions.
 */
export interface ClaudeThinkingInput {
    thinking_mode?: boolean;
    thinking_budget_tokens?: number;
    effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
}

/**
 * Thinking configuration for Claude models.
 * The shape is identical for both the Anthropic SDK (`thinking` field)
 * and the Bedrock Converse API (`reasoning_config` field).
 */
export type ClaudeThinkingConfig =
    | { type: "adaptive"; display: "summarized" }
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
 * Priority for thinking mode:
 * 1. If thinking_budget_tokens is explicitly set → use extended thinking with that budget
 * 2. If the model supports adaptive and thinking_mode is on → use adaptive thinking
 * 3. If the model does not support adaptive and thinking_mode is on → use extended thinking with default budget
 * 4. If thinking_mode is off on an adaptive model → omit thinking (off by default)
 * 5. If thinking_mode is off on a non-adaptive model → explicitly disable thinking
 *
 * @param model - The model identifier string
 * @param options - User-provided Claude options (thinking_mode, thinking_budget_tokens, effort)
 */
export function resolveClaudeThinking(model: string, options?: ClaudeThinkingInput): ClaudeThinkingResult {
    const supportsAdaptive = supportsAdaptiveThinking(model);
    const samplingRestriction = hasSamplingParameterRestriction(model);
    const supportsThinking = isClaudeVersionGTE(model, 3, 7);
    const thinkingEnabled = options?.thinking_mode ?? false;
    const budgetTokens = options?.thinking_budget_tokens;

    let thinking: ClaudeThinkingConfig;

    if (!supportsThinking) {
        // Pre-3.7 models: no thinking support
        thinking = undefined;
    } else if (thinkingEnabled && budgetTokens != null) {
        // User explicitly set a budget — always respect it, regardless of adaptive support.
        // On adaptive models this uses the deprecated extended thinking path,
        // but user input takes priority.
        thinking = {
            type: "enabled" as const,
            budget_tokens: budgetTokens,
        };
    } else if (supportsAdaptive) {
        // Adaptive models: respect the thinking_mode toggle
        thinking = thinkingEnabled
            ? { type: "adaptive" as const, display: "summarized" as const }
            : undefined; // Omit thinking entirely when disabled (thinking is OFF by default)
    } else if (thinkingEnabled) {
        // Older thinking models (3.7): use extended thinking with default budget
        thinking = {
            type: "enabled" as const,
            budget_tokens: 1024,
        };
    } else {
        // Older thinking models with thinking disabled
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
