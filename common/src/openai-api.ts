/** The OpenAI HTTP API a driver sends requests through. */
export type OpenAIApi = 'responses' | 'chat_completions';

/** Every {@link OpenAIApi} value, for validating a stored setting. */
export const OPENAI_APIS: readonly OpenAIApi[] = ['responses', 'chat_completions'];

/**
 * The providers whose driver can call either OpenAI API (the `openai_api` driver option), and the API each uses when
 * the option is omitted. Each default is the API the provider used before the option existed.
 */
export const OPENAI_API_DEFAULTS: Readonly<Record<string, OpenAIApi>> = {
    openai_compatible: 'responses',
    openrouter: 'chat_completions',
    groq: 'chat_completions',
    togetherai: 'chat_completions',
    xai: 'responses',
};

/** Whether `value` is an {@link OpenAIApi}. */
export function isOpenAIApi(value: unknown): value is OpenAIApi {
    return OPENAI_APIS.includes(value as OpenAIApi);
}

/**
 * The API a provider's driver uses with the given `openai_api` option, or undefined for a provider without the
 * option.
 */
export function resolveOpenAIApi(provider: string, openaiApi?: OpenAIApi): OpenAIApi | undefined {
    if (!Object.hasOwn(OPENAI_API_DEFAULTS, provider)) return undefined;
    return openaiApi ?? OPENAI_API_DEFAULTS[provider];
}
