/**
 * Version parsing utilities for Claude and Gemini models.
 *
 * Provides version detection helpers that are forward-compatible with future
 * model releases (e.g., Haiku 4.7, Sonnet 4.7, Opus 4.8, Opus 5).
 *
 * These helpers are used to:
 * - Control option visibility in the UI
 * - Construct appropriate API payloads for each model version
 *
 * Note: llumiverse does NOT validate options here. Errors from invalid
 * parameters propagate to the provider side.
 */

// ============================================================================
// Claude Version Parsing
// ============================================================================

/**
 * Parsed Claude model version information.
 */
export interface ClaudeVersion {
    /** Major version number (e.g., 3, 4, 5) */
    major: number;
    /** Minor version number (e.g., 5, 6, 7) */
    minor: number;
    /** Model variant: opus, sonnet, or haiku */
    variant: 'opus' | 'sonnet' | 'haiku';
}

/**
 * Parse Claude model version from a model string.
 *
 * Examples:
 * - "claude-opus-4-7" -> { major: 4, minor: 7, variant: 'opus' }
 * - "claude-sonnet-4-6" -> { major: 4, minor: 6, variant: 'sonnet' }
 * - "claude-3-7-sonnet-20250219" -> { major: 3, minor: 7, variant: 'sonnet' }
 * - "claude-opus-4-6" -> { major: 4, minor: 6, variant: 'opus' }
 *
 * @param modelString - The model identifier string
 * @returns Parsed version info, or null if not parseable
 */
export function parseClaudeVersion(modelString: string): ClaudeVersion | null {
    // Match pattern: claude-[variant-]-{major}-[optional minor]
    // The minor version is limited to 1-2 digits to avoid matching dates (YYYYMMDD format)
    const match = modelString.match(/claude-(opus|sonnet|haiku)-?(\d+)(?:-(\d{1,2}))?(?:-|\b)/i);
    if (match) {
        const variant = match[1].toLowerCase() as 'opus' | 'sonnet' | 'haiku';
        const major = parseInt(match[2], 10);
        const minor = match[3] ? parseInt(match[3], 10) : 0;
        return { major, minor, variant };
    }

    // Fallback for older format: claude-3-7-sonnet-20250219
    const fallbackMatch = modelString.match(/claude-(\d+)-(\d+)-(\w+)/i);
    if (fallbackMatch) {
        const major = parseInt(fallbackMatch[1], 10);
        const minor = parseInt(fallbackMatch[2], 10);
        const variant = fallbackMatch[3].toLowerCase() as 'opus' | 'sonnet' | 'haiku';
        return { major, minor, variant };
    }

    return null;
}

/**
 * Check if a Claude model version is greater than or equal to a target version.
 *
 * @param modelString - The model identifier string
 * @param targetMajor - Target major version
 * @param targetMinor - Target minor version
 * @returns true if the model version is >= target version, false otherwise
 */
export function isClaudeVersionGTE(modelString: string, targetMajor: number, targetMinor: number): boolean {
    const version = parseClaudeVersion(modelString);
    if (!version) {
        return false;
    }
    if (version.major > targetMajor) {
        return true;
    }
    if (version.major === targetMajor && version.minor >= targetMinor) {
        return true;
    }
    return false;
}

/**
 * Check if a Claude variant model version is greater than or equal to a target version.
 *
 * @param modelString - The model identifier string
 * @param variant - Model variant: "opus" or "sonnet"
 * @param targetMajor - Target major version
 * @param targetMinor - Target minor version
 * @returns true if the model matches the variant and version >= target
 */
function isClaudeVariantVersionGTE(
    modelString: string,
    variant: "opus" | "sonnet",
    targetMajor: number,
    targetMinor: number
): boolean {
    const version = parseClaudeVersion(modelString);
    if (!version) return false;
    if (version.variant.toLowerCase() !== variant) return false;
    return version.major > targetMajor || (version.major === targetMajor && version.minor >= targetMinor);
}

/**
 * Check if a model requires sampling parameter removal (behavior: sampling params removed on Opus 4.7+).
 *
 * This includes:
 * - claude-opus-4-7
 * - Future Opus 4.x with minor >= 7
 * - Future Opus 5.x+
 *
 * @param modelString - The model identifier string
 * @returns true if Opus 4.7+ or equivalent future model
 */
export function hasSamplingParameterRemoval(modelString: string): boolean {
    return isClaudeVariantVersionGTE(modelString, "opus", 4, 7);
}

/**
 * Check if a model requires adaptive thinking (behavior: adaptive thinking required on Opus 4.6+).
 *
 * This includes:
 * - claude-opus-4-6
 * - claude-opus-4-7
 * - Future Opus 4.x with minor >= 6
 * - Future Opus 5.x+
 *
 * @param modelString - The model identifier string
 * @returns true if Opus 4.6+ or equivalent future model
 */
export function requiresAdaptiveThinking(modelString: string): boolean {
    return isClaudeVariantVersionGTE(modelString, "opus", 4, 6);
}

/**
 * Check if a model supports adaptive thinking.
 *
 * Adaptive thinking was introduced in:
 * - Claude Opus 4.6
 * - Claude Sonnet 4.6
 *
 * @param modelString - The model identifier string
 * @returns true if the model supports adaptive thinking
 */
export function supportsAdaptiveThinking(modelString: string): boolean {
    return requiresAdaptiveThinking(modelString) || isClaudeVariantVersionGTE(modelString, "sonnet", 4, 6);
}

/**
 * Check if extended thinking is deprecated for this model.
 *
 * Extended thinking (thinking.type: "enabled" with budget_tokens) is deprecated
 * but still functional on:
 * - Claude Opus 4.6+
 * - Claude Sonnet 4.6+
 *
 * @param modelString - The model identifier string
 * @returns true if extended thinking is deprecated (adaptive thinking recommended)
 */
export function isExtendedThinkingDeprecated(modelString: string): boolean {
    return supportsAdaptiveThinking(modelString);
}

/**
 * Check if a model requires adaptive thinking ONLY (extended thinking removed).
 *
 * On Opus 4.7+, extended thinking returns a 400 error. Only adaptive thinking is supported.
 * Future models (Sonnet 4.7+, Haiku 4.7+, any 5.0+) follow the same pattern.
 *
 * @param modelString - The model identifier string
 * @returns true if extended thinking is removed (returns 400 error)
 */
export function requiresAdaptiveThinkingOnly(modelString: string): boolean {
    return hasSamplingParameterRestriction(modelString);
}

/**
 * Check if a model has sampling parameter restrictions.
 *
 * On Opus 4.7+, setting temperature, top_p, or top_k to any non-default value
 * returns a 400 error. Future models following the same pattern will also match:
 * - Opus 4.7+ (current restriction)
 * - Sonnet 4.7+, Haiku 4.7+ (future minor versions >= 7)
 * - Sonnet 5.0+, Haiku 5.0+, Opus 5.0+ (future major versions)
 *
 * @param modelString - The model identifier string
 * @returns true if sampling parameters are restricted
 */
export function hasSamplingParameterRestriction(modelString: string): boolean {
    const version = parseClaudeVersion(modelString);
    if (!version) {
        return false;
    }

    // Future major versions (5.0+) follow the same pattern as 4.7
    if (version.major > 4) {
        return true;
    }

    // Version 4.7+ (Opus 4.7, Sonnet 4.7, Haiku 4.7, etc.)
    if (version.major === 4 && version.minor >= 7) {
        return true;
    }

    return false;
}

// ============================================================================
// Claude Effort Parameter Support
// ============================================================================

/** Available effort levels for Claude models. */
export type ClaudeEffortLevel = 'low' | 'medium' | 'high' | 'xhigh' | 'max';

/**
 * Check if a model supports the effort parameter.
 *
 * Effort is supported on:
 * - Claude Opus 4.5+
 * - Claude Opus 4.6+
 * - Claude Sonnet 4.6+
 * - All variants at 4.7+ (Opus, Sonnet, Haiku)
 * - All variants at 5.0+
 *
 * @param modelString - The model identifier string
 * @returns true if the model supports the effort parameter
 */
export function supportsEffort(modelString: string): boolean {
    // All 4.7+ variants support effort (covers future Sonnet 4.7, Haiku 4.7, etc.)
    if (hasSamplingParameterRestriction(modelString)) {
        return true;
    }
    // Opus 4.5+ supports effort
    if (isClaudeVariantVersionGTE(modelString, "opus", 4, 5)) {
        return true;
    }
    // Sonnet 4.6+ supports effort
    if (isClaudeVariantVersionGTE(modelString, "sonnet", 4, 6)) {
        return true;
    }
    return false;
}

/**
 * Check if a model supports the xhigh effort level.
 *
 * xhigh is only available on Opus 4.7+.
 *
 * @param modelString - The model identifier string
 * @returns true if the model supports xhigh effort
 */
export function supportsXHighEffort(modelString: string): boolean {
    return isClaudeVariantVersionGTE(modelString, "opus", 4, 7);
}

/**
 * Get the available effort levels for a given Claude model.
 *
 * - Opus 4.7+: low, medium, high, xhigh, max
 * - Opus 4.5+, Opus 4.6+, Sonnet 4.6+: low, medium, high, max
 * - Other models: empty (effort not supported)
 *
 * @param modelString - The model identifier string
 * @returns Record of display label to effort level value, or null if not supported
 */
export function getAvailableEffortLevels(modelString: string): Record<string, ClaudeEffortLevel> | null {
    if (!supportsEffort(modelString)) {
        return null;
    }
    const levels: Record<string, ClaudeEffortLevel> = {
        "Low": "low",
        "Medium": "medium",
        "High (default)": "high",
        "Max": "max",
    };
    if (supportsXHighEffort(modelString)) {
        // Insert xhigh between high and max
        return {
            "Low": "low",
            "Medium": "medium",
            "High (default)": "high",
            "Extra High": "xhigh",
            "Max": "max",
        };
    }
    return levels;
}

// ============================================================================
// Gemini Version Parsing
// ============================================================================

/**
 * Extract Gemini version from a model ID.
 *
 * Examples:
 * - "locations/global/publishers/google/models/gemini-2.5-flash" -> "2.5"
 * - "publishers/google/models/gemini-3-pro-image-preview" -> "3"
 * - "gemini-3.1-flash-lite-preview" -> "3.1"
 *
 * @param modelId - The model identifier string
 * @returns Version string (e.g., "2.5", "3", "3.1"), or undefined if not parseable
 */
export function getGeminiModelVersion(modelId: string): string | undefined {
    const modelName = modelId.split('/').pop() ?? modelId;
    const match = modelName.match(/^gemini-(\d+(?:\.\d+)?)/i);
    return match?.[1];
}

/**
 * Parse a version string into major.minor components.
 *
 * @param version - Version string (e.g., "2.5", "3", "3.1")
 * @returns Parsed version, or undefined if not parseable
 */
export function parseGeminiVersion(version: string): { major: number; minor: number } | undefined {
    const match = version.match(/^(\d+)(?:\.(\d+))?$/);
    if (!match) {
        return undefined;
    }

    return {
        major: Number(match[1]),
        minor: Number(match[2] ?? '0'),
    };
}

/**
 * Check if a Gemini model version is greater than or equal to a minimum version.
 *
 * @param modelId - The model identifier string
 * @param minVersion - Minimum version string (e.g., "2.5", "3.0")
 * @returns true if model version >= min version
 */
export function isGeminiModelVersionGte(modelId: string, minVersion: string): boolean {
    const modelVersion = getGeminiModelVersion(modelId);
    if (!modelVersion) {
        return false;
    }

    const current = parseGeminiVersion(modelVersion);
    const target = parseGeminiVersion(minVersion);
    if (!current || !target) {
        return false;
    }

    if (current.major > target.major) {
        return true;
    }
    if (current.major < target.major) {
        return false;
    }

    return current.minor >= target.minor;
}
