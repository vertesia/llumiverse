import { isGeminiModelVersionGte } from './version-parsing.js';

interface GeminiModelLimits {
    context_window?: number;
    max_output_tokens?: number;
}

/** Documented Google model limits; later generations inherit the latest known family behavior. */
export function getGeminiModelLimits(model: string): GeminiModelLimits {
    const name = model.toLowerCase().split('/').pop() ?? '';
    // Special-purpose APIs do not inherit chat limits. Transcribe publishes no token limits.
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe
    if (/(?:transcribe|live|robotics|tts|embedding)/.test(name)) return {};
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-1-1-flash
    if (name.includes('gemini-omni-')) return { context_window: 131_072, max_output_tokens: 57_920 };
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/nano-banana-2-1
    if (name.includes('gemini-nano-banana')) return { context_window: 131_072, max_output_tokens: 32_768 };
    // Dedicated image families have smaller windows than chat models; order the narrower variants first.
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite-image
    if (name.includes('-flash-lite-image') && isGeminiModelVersionGte(name, '3.1')) {
        return { context_window: 65_536, max_output_tokens: 4_096 };
    }
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-pro-image
    if (name.includes('-pro-image') && isGeminiModelVersionGte(name, '3.0')) {
        return { context_window: 65_536, max_output_tokens: 32_768 };
    }
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-image
    if (name.includes('-flash-image') && isGeminiModelVersionGte(name, '3.1')) {
        return { context_window: 131_072, max_output_tokens: 32_768 };
    }
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash-image
    if (name.includes('-flash-image') && isGeminiModelVersionGte(name, '2.5')) {
        return { context_window: 32_768, max_output_tokens: 32_768 };
    }
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-pro
    if (isGeminiModelVersionGte(name, '2.5')) return { context_window: 1_048_576, max_output_tokens: 65_536 };
    // Older releases retain their documented limits even if a provider still discovers a retired version.
    if (name.includes('thinking') && isGeminiModelVersionGte(name, '2.0')) {
        // Preserve legacy metadata for the retired experimental thinking variant.
        return { context_window: 1_048_576, max_output_tokens: 65_535 };
    }
    if (isGeminiModelVersionGte(name, '2.0')) return { context_window: 1_048_576, max_output_tokens: 8_192 };
    if (name.includes('-1.0-')) {
        return { context_window: 32_000, max_output_tokens: 2_048 };
    }
    // Preserve the legacy unversioned/1.x behavior outside the audited current families.
    return { context_window: 1_000_000, max_output_tokens: 65_535 };
}
