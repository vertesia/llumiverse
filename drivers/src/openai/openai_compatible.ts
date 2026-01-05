import { AIModel, DriverOptions, ModelType, Providers, getModelCapabilities, modelModalitiesToArray } from "@llumiverse/core";
import OpenAI from "openai";
import { BaseOpenAIDriver } from "./index.js";

export interface OpenAICompatibleDriverOptions extends DriverOptions {
    /**
     * The API key for the OpenAI-compatible service
     */
    apiKey: string;

    /**
     * The base URL of the OpenAI-compatible API endpoint
     * Example: https://api.example.com/v1
     */
    endpoint: string;
}

/**
 * A generic driver for OpenAI-compatible APIs.
 * This can be used with any service that implements the OpenAI API spec,
 * such as xAI (Grok), LM Studio, Ollama, vLLM, LocalAI, etc.
 */
export class OpenAICompatibleDriver extends BaseOpenAIDriver {
    service: OpenAI;
    readonly provider = Providers.openai_compatible;

    constructor(opts: OpenAICompatibleDriverOptions) {
        super(opts);

        if (!opts.apiKey) {
            throw new Error("apiKey is required");
        }

        if (!opts.endpoint) {
            throw new Error("endpoint is required for OpenAI-compatible driver");
        }

        this.service = new OpenAI({
            apiKey: opts.apiKey,
            baseURL: opts.endpoint,
        });
    }

    async listModels(): Promise<AIModel[]> {
        try {
            const result = (await this.service.models.list()).data;

            const models = result.map((m) => {
                const modelCapability = getModelCapabilities(m.id, "openai");
                let owner = m.owned_by;
                if (owner === "system") {
                    owner = "unknown";
                }
                return {
                    id: m.id,
                    name: m.id,
                    provider: this.provider,
                    owner: owner,
                    type: ModelType.Text,
                    can_stream: true,
                    is_multimodal: false,
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                } satisfies AIModel<string>;
            }).sort((a, b) => a.id.localeCompare(b.id));

            return models;
        } catch (error) {
            this.logger.warn({ error }, "[OpenAICompatible] Failed to list models, returning empty list");
            return [];
        }
    }
}
