import { ModelOptionsInfo, ModelOptionInfoItem, ModelOptions, OptionType, SharedOptions } from "../types.js";
import { textOptionsFallback } from "./fallback.js";

// Union type of all Bedrock options
export type OpenAiOptions = OpenAiThinkingOptions | OpenAiTextOptions;

export interface OpenAiThinkingOptions {
    _option_id: "openai-thinking",
    max_tokens?: number,
    stop_sequence?: string[],
    reasoning_effort?: "low" | "medium" | "high",
    image_detail?: "low" | "high" | "auto",
}

export interface OpenAiTextOptions {
    _option_id: "openai-text",
    max_tokens?: number,
    temperature?: number,
    top_p?: number,
    presence_penalty?: number,
    frequency_penalty?: number,
    stop_sequence?: string[],
    image_detail?: "low" | "high" | "auto",
}

export function getOpenAiOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    const visionOptions: ModelOptionInfoItem[] = isVisionModel(model) ? [
        {
            name: "image_detail", type: OptionType.enum, enum: { "Low": "low", "High": "high", "Auto": "auto" },
            default: "auto", description: "Controls how the model processes an input image."
        },
    ] : [];

    if (model.includes("o1") || model.includes("o3")) {
        //Is thinking text model
        let max_tokens_limit = 4096;
        if (model.includes("o1")) {
            if (model.includes("preview")) {
                max_tokens_limit = 32768;
            }
            else if (model.includes("mini")) {
                max_tokens_limit = 65536;
            }
            else {
                max_tokens_limit = 100000;
            }
        }
        else if (model.includes("o3")) {
            max_tokens_limit = 100000;
        }

        const commonOptions: ModelOptionInfoItem[] = [
            {
                name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
                integer: true, description: "The maximum number of tokens to generate",
            },
            {
                name: SharedOptions.stop_sequence, type: OptionType.string_list, value: [],
                description: "The stop sequence of the generated image",
            },
        ];

        const reasoningOptions: ModelOptionInfoItem[] = model.includes("o3") || isO1Full(model) ? [
            {
                name: "reasoning_effort", type: OptionType.enum, enum: { "Low": "low", "Medium": "medium", "High": "high" },
                default: "medium", description: "How much effort the model should put into reasoning, lower values result in faster responses and less tokens used."
            },
        ] : [];

        return {
            _option_id: "openai-thinking",
            options: [
                ...commonOptions,
                ...reasoningOptions,
                ...visionOptions,
            ],
        };
    } else {
        let max_tokens_limit = 4096;
        if (model.includes("gpt-4o")) {
            max_tokens_limit = 16384;
            if (model.includes("gpt-4o-2024-05-13") || model.includes("realtime")) {
                max_tokens_limit = 4096;
            }
        }
        else if (model.includes("gpt-4")) {
            if (model.includes("turbo")) {
                max_tokens_limit = 4096;
            } else {
                max_tokens_limit = 8192;
            }
        }
        else if (model.includes("gpt-3-5")) {
            max_tokens_limit = 4096;
        }

        //Is non-thinking text model
        const commonOptions: ModelOptionInfoItem[] = [
            {
                name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
                integer: true, step: 200, description: "The maximum number of tokens to generate",
            },
            {
                name: "temperature", type: OptionType.numeric, min: 0.0, max: 2.0, default: 0.7,
                integer: false, step: 0.1, description: "A higher temperature biases toward less likely tokens, making the model more creative"
            },
            {
                name: "top_p", type: OptionType.numeric, min: 0, max: 1,
                integer: false, step: 0.1, description: "Limits token sampling to the cumulative probability of the top p tokens"
            },
            {
                name: "presence_penalty", type: OptionType.numeric, min: -2.0, max: 2.0,
                integer: false, step: 0.1, description: "Penalise tokens if they appear at least once in the text"
            },
            {
                name: "frequency_penalty", type: OptionType.numeric, min: -2.0, max: 2.0,
                integer: false, step: 0.1, description: "Penalise tokens based on their frequency in the text"
            },
            {
                name: SharedOptions.stop_sequence, type: OptionType.string_list, value: [],
                description: "The generation will halt if one of the stop sequences is output",
            }
        ]

        return {
            _option_id: "openai-text",
            options: [
                ...commonOptions,
                ...visionOptions,
            ],
        }
    }
    return textOptionsFallback;
}

function isO1Full(model: string): boolean {
    if (model.includes("o1")) {
        if (model.includes("mini") || model.includes("preview")) {
            return false;
        }
        return true;
    }
    return false;
}

function isVisionModel(model: string): boolean {
    return model.includes("gpt-4o") || isO1Full(model) || model.includes("gpt-4-turbo");
}