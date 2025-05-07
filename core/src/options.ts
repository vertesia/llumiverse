import { ModelOptions, ModelOptionsInfo, OptionType, SharedOptions } from "./types.js";
import { getBedrockOptions } from "./options/bedrock.js";
import { getVertexAiOptions } from "./options/vertexai.js";
import { getOpenAiOptions } from "./options/openai.js";
import { getGroqOptions } from "./options/groq.js";

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
    _option_id: "text-fallback",
    options: [
        {
            name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1,
            integer: true, step: 200, description: "The maximum number of tokens to generate"
        },
        {
            name: SharedOptions.temperature, type: OptionType.numeric, min: 0.0, default: 0.7,
            integer: false, step: 0.1, description: "A higher temperature biases toward less likely tokens, making the model more creative"
        },
        {
            name: SharedOptions.top_p, type: OptionType.numeric, min: 0, max: 1,
            integer: false, step: 0.1, description: "Limits token sampling to the cumulative probability of the top p tokens"
        },
        {
            name: SharedOptions.top_k, type: OptionType.numeric, min: 1,
            integer: true, step: 1, description: "Limits token sampling to the top k tokens"
        },
        {
            name: SharedOptions.presence_penalty, type: OptionType.numeric, min: -2.0, max: 2.0,
            integer: false, step: 0.1, description: "Penalise tokens if they appear at least once in the text"
        },
        {
            name: SharedOptions.frequency_penalty, type: OptionType.numeric, min: -2.0, max: 2.0,
            integer: false, step: 0.1, description: "Penalise tokens based on their frequency in the text"
        },
        { name: SharedOptions.stop_sequence, type: OptionType.string_list, value: [], description: "The generation will halt if one of the stop sequences is output" },
    ]
};

export function getOptions(provider?: string, model?: string, options?: ModelOptions): ModelOptionsInfo {
    switch (provider) {
        case "bedrock":
            return getBedrockOptions(model ?? "", options);
        case "vertexai":
            return getVertexAiOptions(model ?? "", options);
        case "openai":
            return getOpenAiOptions(model ?? "", options);
        case "groq":
            return getGroqOptions(model ?? "", options);
        default:
            return textOptionsFallback;
    }
}

export type * from "./options/bedrock.js";
export type * from "./options/vertexai.js";
