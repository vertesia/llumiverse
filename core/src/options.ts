import { ModelOptions, ModelOptionsInfo, OptionType } from "./types.js";
import { getBedrockOptions } from "./options/bedrock.js";
import { getVertexAiOptions } from "./options/vertexai.js";

export interface TextFallbackOptions {
    _option_id: "text-fallback";    //For specific models should be format as "provider-model"
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    stop_sequence?: string[];
}

export const textOptionsFallback: ModelOptionsInfo = {
    _options_id: "text-fallback",
    options: [
        {
            name: "max_tokens", type: OptionType.numeric, min: 1, max: 8192, default: 512,
            integer: true, description: "The maximum number of tokens to generate"
        },
        {
            name: "temperature", type: OptionType.numeric, min: 0.0, max: 1.0, default: 0.7,
            integer: false, step: 0.1, description: "The temperature of the generated image"
        },
        { name: "top_p", type: OptionType.numeric, min: 0, max: 1, default: 1, description: "The nucleus sampling probability of the generated image" },
        { name: "top_k", type: OptionType.numeric, min: 0, max: 1024, default: 50, description: "The top k sampling of the generated image" },
        {
            name: "presence_penalty", type: OptionType.numeric, min: -2.0, max: -2.0, default: 0,
            integer: false, step: 0.1, description: "The presence penalty of the generated image"
        },
        {
            name: "frequency_penalty", type: OptionType.numeric, min: -2.0, max: -2.0, default: 0,
            integer: false, step: 0.1, description: "The frequency penalty of the generated image"
        },
        { name: "stop_sequence", type: OptionType.string_list, value: [], description: "The stop sequence of the generated image" },
    ]
};

export function getOptions(provider?: string, model?: string, options?: ModelOptions): ModelOptionsInfo {
    switch (provider) {
        case "bedrock":
            return getBedrockOptions(model ?? "", options);
        case "vertexai":
            return getVertexAiOptions(model ?? "", options);
        default:
            return textOptionsFallback;
    }
}