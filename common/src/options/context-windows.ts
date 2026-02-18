/**
 * Returns the context window size (input + output) for a given model.
 */
export function getContextWindowSize(model: string): number {
    // Claude models — all Claude 3+ have 200K context windows
    if (model.includes('claude')) {
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
    if (model.includes('gpt-5')) return 400_000;
    if (model.includes('gpt-4.1') || model.includes('gpt-4-1')) return 1_000_000;
    if (model.includes('gpt-4-turbo') || model.includes('gpt-4o')) return 128_000;
    if (model.includes('gpt-4')) return 8_000;
    if (model.includes('gpt-3.5')) return 16_000;
    // Grok
    if (model.includes('grok-4.1') || model.includes('grok-4-1')) return 256_000;
    if (model.includes('grok')) return 131_072;
    // Amazon Nova
    if (model.includes('nova')) return 300_000;
    // Mistral
    if (model.includes('mistral-large')) return 128_000;
    if (model.includes('mistral')) return 32_000;
    // DeepSeek
    if (model.includes('deepseek')) return 128_000;
    // Llama
    if (model.includes('llama-4') || model.includes('llama4')) return 1_000_000;
    if (model.includes('llama-3.1') || model.includes('llama-3.2') || model.includes('llama-3.3')) return 128_000;
    if (model.includes('llama')) return 8_000;
    // Cohere
    if (model.includes('command-a')) return 256_000;
    if (model.includes('command')) return 128_000;

    return 128_000; // conservative default
}
