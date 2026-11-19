import { type AIModel, isDedicatedInferenceModel, isEmbeddingModel, ModelType, Providers } from '@llumiverse/core';
import OpenAI from 'openai';
import type { OpenAIResponsesDriverOptions } from '../driver-options.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';
import { OpenAIResponsesDriverBase, type OpenAIResponsesPolicy } from './index.js';
import { NON_OPENAI_ACCOUNT_OPTIONS } from './openai_compatible.js';

export type { OpenAIResponsesDriverOptions } from '../driver-options.js';

/**
 * Family rules for models served by OpenAI-compatible Responses endpoints such as OpenRouter. They apply to any such
 * endpoint serving the model ID, not to one host.
 */
const OPENAI_COMPATIBLE_RESPONSES_POLICY: OpenAIResponsesPolicy = {
    // OpenRouter's OpenAI-compatible Responses surface advertises native structured output for GLM 5.3, but the model
    // returns unconstrained prose and downstream validation fails. Keep this generation on the existing prompt-schema
    // fallback; later GLM generations remain eligible for native support unless runtime evidence says otherwise.
    supportsSchema: (model) => (/^z-ai\/glm-5\.3(?:$|[-/:])/.test(model) ? false : undefined),
    // OpenRouter routes `minimax/` models across many hosts, and most do not enforce a JSON schema: some downgrade it
    // to JSON mode and return `{}`, others ignore it. The prompt schema gives every host the shape, while hosts that
    // enforce the schema still receive it natively.
    supplementsSchemaInPrompt: (model) => /^minimax\//i.test(model),
};

/**
 * A generic driver for OpenAI-compatible APIs.
 * This can be used with any service that implements the OpenAI API spec,
 * such as xAI (Grok), LM Studio, Ollama, vLLM, LocalAI, etc.
 */
export class OpenAIResponsesDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.openai_compatible;
    override readonly responsesPolicy = OPENAI_COMPATIBLE_RESPONSES_POLICY;

    constructor(opts: OpenAIResponsesDriverOptions) {
        super(opts);

        if (!opts.apiKey) {
            throw new Error('apiKey is required');
        }

        if (!opts.endpoint) {
            throw new Error('endpoint is required for OpenAI-compatible driver');
        }

        this.service = new OpenAI({
            apiKey: opts.apiKey,
            ...NON_OPENAI_ACCOUNT_OPTIONS,
            baseURL: opts.endpoint,
            defaultHeaders: opts.default_headers,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
    }

    async listModels(): Promise<AIModel[]> {
        try {
            const result = (await this.service.models.list()).data;

            const models = result
                .filter(
                    (m) =>
                        !isEmbeddingModel({ id: m.id }, this.provider) &&
                        (!isDedicatedInferenceModel(m.id, this.provider) || this.isFileAudioModel(m.id)),
                )
                .map((m) => {
                    const modelMetadata = resolveModelListingMetadata(m.id, this.provider);
                    let owner = m.owned_by;
                    if (owner === 'system') {
                        owner = 'unknown';
                    }
                    return {
                        id: m.id,
                        name: m.id,
                        provider: this.provider,
                        owner: owner,
                        type: this.isFileAudioModel(m.id) ? ModelType.Audio : ModelType.Text,
                        can_stream: true,
                        is_multimodal: modelMetadata.input_modalities.includes('image'),
                        ...modelMetadata,
                    } satisfies AIModel;
                })
                .sort((a, b) => a.id.localeCompare(b.id));

            return models;
        } catch (error) {
            this.logger.warn({ error }, '[OpenAIResponses] Failed to list models, returning empty list');
            throw new Error(`Failed to list models: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
}
