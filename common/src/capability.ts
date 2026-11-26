import { isModelDirectoryEmbedding, resolveModelProfile } from './model-directory.js';
import { isGeminiOmniVideoModel } from './options/version-parsing.js';
import type { ModelCapabilities, ModelModalities, Providers } from './types.js';

const CHAT_COMPLETIONS_PROVIDERS: readonly string[] = ['openai_compatible', 'openrouter', 'groq', 'togetherai'];

/**
 * Whether a provider's driver can send an input modality. The Chat Completions drivers send images and text files
 * but no video, and OpenRouter's request conversion rejects audio parts. Listings, whether from the catalog or from
 * the provider's own metadata, advertise only the inputs that pass this check.
 */
export function canSendInputModality(provider: Providers | string, modality: string): boolean {
    if (modality === 'video') return !CHAT_COMPLETIONS_PROVIDERS.includes(provider);
    if (modality === 'audio') return provider !== 'openrouter';
    return true;
}

export function getModelCapabilities(model: string, provider: Providers): ModelCapabilities {
    //Check for locations/<location>/ prefix and remove it
    if (model.startsWith('locations/')) {
        const parts = model.split('/');
        if (parts.length >= 3) {
            model = parts.slice(2).join('/');
        }
    }
    const modelName = model.split('/').pop()?.toLowerCase();
    if (provider === 'vertexai' && modelName && isGeminiOmniVideoModel(modelName)) {
        return {
            input: { text: true, image: true, video: true, audio: false, embed: false },
            output: { text: true, image: false, video: true, audio: false, embed: false },
            tool_support: false,
            tool_support_streaming: false,
        };
    }
    const profile = resolveModelProfile(model, provider);
    const capabilities = profile.capabilities;
    const sendsVideo = canSendInputModality(provider, 'video');
    const sendsAudio = canSendInputModality(provider, 'audio');
    // Advertise speech output only for implemented file synthesis transports.
    return {
        ...capabilities,
        input: {
            ...capabilities.input,
            ...(sendsVideo ? {} : { video: false }),
            ...(sendsAudio ? {} : { audio: false }),
        },
        output: {
            ...capabilities.output,
            audio:
                (['openai', 'azure_foundry', 'openai_compatible', 'vertexai'].includes(provider) &&
                    profile.family === 'speech') ||
                (['openai', 'openai_compatible'].includes(provider) && profile.family === 'audio'),
            video: false,
        },
    };
}

/**
 * Whether the driver should send tools for a model. Only an explicit `false` in the model directory opts out: most
 * models the directory does not know (new releases, local and gateway-served open-weight models) do take tools, and
 * a request without them leaves the model to write tool calls into its text. A server that rejects tools fails
 * visibly instead.
 *
 * `tool_support_streaming` falls back to `tool_support` when unset, so only an explicit `false` turns tools off
 * while streaming.
 */
export function supportsToolUse(model: string, provider: Providers, streaming: boolean = false): boolean {
    const capabilities = getModelCapabilities(model, provider);
    const support = streaming
        ? (capabilities.tool_support_streaming ?? capabilities.tool_support)
        : capabilities.tool_support;
    return support !== false;
}

/**
 * Whether a model is known to answer `Driver.decide()` on a provider. This is capability metadata, not a runtime
 * guard: `decide()` still sends the request for any model and lets the provider answer.
 *
 * OpenAI serves `POST /v1/decisions` with the `luna` tier from GPT-6 on (`gpt-6-luna`, dated snapshots included).
 * Later generations of the tier are assumed to keep it.
 */
export function supportsDecisions(model: string, provider: Providers | string): boolean {
    const name = model.split('/').pop()?.toLowerCase() ?? '';
    if (provider === 'openai') {
        const match = /^gpt-(\d+)(?:\.\d+)?-luna(?:-\d{4}-\d{2}-\d{2})?$/.exec(name);
        return !!match && Number(match[1]) >= 6;
    }
    return false;
}

export function modelModalitiesToArray(modalities: ModelModalities): string[] {
    return Object.entries(modalities)
        .filter(([_, isSupported]) => isSupported)
        .map(([modality]) => modality);
}

export interface ModelListing {
    id: string;
    type?: string;
    input_modalities?: readonly string[];
    output_modalities?: readonly string[];
}

/** Dedicated endpoint models are not executable through the standard inference path. */
export function isDedicatedInferenceModel(model: string, provider: Providers): boolean {
    return ['embedding', 'moderation', 'image', 'transcription', 'speech', 'realtime', 'audio', 'video'].includes(
        resolveModelProfile(model, provider).family,
    );
}

/** Embedding endpoints are not executable through the standard inference path. */
export function isEmbeddingModel(model: ModelListing, provider: Providers): boolean {
    if (model.type?.toLowerCase() === 'embedding') return true;
    if (isModelDirectoryEmbedding(model.id)) return true;

    const modalities = [...(model.input_modalities ?? []), ...(model.output_modalities ?? [])].map((modality) =>
        modality.toLowerCase(),
    );
    if (modalities.some((modality) => modality === 'embed' || modality === 'embedding' || modality === 'vectors')) {
        return true;
    }

    return getModelCapabilities(model.id, provider).output.embed === true;
}
