/**
 * Returns the context window size (input + output) for a given model.
 * This is used to calculate appropriate checkpoint thresholds.
 */
export function getContextWindowSize(model: string): number {
    // Claude models
    if (model.includes('claude')) {
        if (model.includes('-5-')) return 1_000_000;
        if (model.includes('-4-5') || model.includes('-4-')) return 200_000;
        if (model.includes('-3-7')) return 200_000;
        if (model.includes('-3-5')) return 200_000;
        return 100_000; // older claude
    }
    // Gemini models
    if (model.includes('gemini')) {
        if (model.includes('-2.5-') || model.includes('-3-')) return 1_000_000;
        return 128_000;
    }
    // GPT models
    if (model.includes('gpt-4-turbo') || model.includes('gpt-4o')) return 128_000;
    if (model.includes('gpt-4')) return 8_000;
    if (model.includes('gpt-3.5')) return 16_000;

    // Nova, Mistral, etc.
    if (model.includes('nova')) return 300_000;
    if (model.includes('mistral-large')) return 128_000;

    return 128_000; // conservative default
}

/**
 * Returns the recommended checkpoint threshold for a given model.
 * Uses 75% of the context window to leave room for:
 * - The checkpoint summarization call
 * - Tool definitions and system prompts
 * - Buffer for continuation after checkpoint
 */
export function getDefaultCheckpointThreshold(model: string): number {
    const contextWindow = getContextWindowSize(model);
    return Math.floor(contextWindow * 0.75);
}
