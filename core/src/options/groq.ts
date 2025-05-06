import { ModelOptionsInfo, ModelOptionInfoItem, ModelOptions, OptionType, SharedOptions } from "../types.js";
import { textOptionsFallback } from "../options.js";

// Union type of all Bedrock options
export type GroqOptions = GroqDeepseekThinkingOptions;

export interface GroqDeepseekThinkingOptions {
    _option_id: "groq-deepseek-thinking",
    max_tokens?: number,
    temperature?: number,
    top_p?: number,
    stop_sequence?: string[],
    reasoning_format: 'parsed' | 'raw' | 'hidden',
}

export function getGroqOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    if (model.includes("deepseek") && model.includes("r1")) {
        const commonOptions: ModelOptionInfoItem[] = [
            {
                name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: 131072,
                integer: true, description: "The maximum number of tokens to generate",
            },
            {
                name: SharedOptions.temperature, type: OptionType.numeric, min: 0.0, default: 0.7, max: 2.0,
                integer: false, step: 0.1, description: "A higher temperature biases toward less likely tokens, making the model more creative. A lower temperature than other models is recommended for deepseek R1, 0.3-0.7 approximately.",
            },
            {
                name: SharedOptions.top_p, type: OptionType.numeric, min: 0, max: 1,
                integer: false, step: 0.1, description: "Limits token sampling to the cumulative probability of the top p tokens",
            },
            {
                name: SharedOptions.stop_sequence, type: OptionType.string_list, value: [],
                description: "The generation will halt if one of the stop sequences is output",
            },
            {
                name: "reasoning_format", type: OptionType.enum, enum: { "Parsed": "parsed", "Raw": "raw", "Hidden": "hidden" },
                default: "parsed", description: "Controls how the reasoning is returned.",
            },
        ];

        return {
            _option_id: "groq-deepseek-thinking",
            options: commonOptions,
        };
    }
    return textOptionsFallback;
}