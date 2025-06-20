import { ModelOptionsInfo, ModelOptions, OptionType, ModelOptionInfoItem } from "../types.js";
import { textOptionsFallback } from "./fallback.js";

// Union type of all Bedrock options
export type BedrockOptions = NovaCanvasOptions | BaseConverseOptions | BedrockClaudeOptions | BedrockPalmyraOptions;

export interface NovaCanvasOptions {
    _option_id: "bedrock-nova-canvas"
    taskType: "TEXT_IMAGE" | "TEXT_IMAGE_WITH_IMAGE_CONDITIONING" | "COLOR_GUIDED_GENERATION" | "IMAGE_VARIATION" | "INPAINTING" | "OUTPAINTING" | "BACKGROUND_REMOVAL";
    width?: number;
    height?: number;
    quality?: "standard" | "premium";
    cfgScale?: number;
    seed?: number;
    numberOfImages?: number;
    controlMode?: "CANNY_EDGE" | "SEGMENTATION";
    controlStrength?: number;
    colors?: string[];
    similarityStrength?: number;
    outPaintingMode?: "DEFAULT" | "PRECISE";
}

export interface BaseConverseOptions {
    _option_id: "bedrock-converse" | "bedrock-claude" | "bedrock-nova" | "bedrock-mistral" | "bedrock-ai21" | "bedrock-cohere-command" | "bedrock-palmyra";
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop_sequence?: string[];
}

export interface BedrockClaudeOptions extends BaseConverseOptions {
    _option_id: "bedrock-claude";
    top_k?: number;
    thinking_mode?: boolean;
    thinking_budget_tokens?: number;
    include_thoughts?: boolean;
}

export interface BedrockPalmyraOptions extends BaseConverseOptions {
    _option_id: "bedrock-palmyra";
    min_tokens?: number;
    seed?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
}

export function getMaxTokensLimitBedrock(model: string): number | undefined {
    // Claude models
    if (model.includes("claude")) {
        if (model.includes("-4-")) {
            if (model.includes("opus-")) {
                return 32768;
            }
            return 65536;
        }
        else if (model.includes("-3-7-")) {
            return 131072;
        }
        else if (model.includes("-3-5-")) {
            return 8192;
        }
        else {
            return 4096;
        }
    }
    // Amazon models
    else if (model.includes("amazon")) {
        if (model.includes("titan")) {
            if (model.includes("lite")) {
                return 4096;
            } else if (model.includes("express")) {
                return 8192;
            } else if (model.includes("premier")) {
                return 3072;
            }

        }
        else if (model.includes("nova")) {
            return 10000;
        }
    }
    // Mistral models
    else if (model.includes("mistral")) {
        if (model.includes("8x7b")) {
            return 4096;
        }
        if (model.includes("pixtral-large")) {
            return 131072;
        }
        return 8192;
    }
    // AI21 models
    else if (model.includes("ai21")) {
        if (model.includes("j2")) {
            if (model.includes("large") || model.includes("mid") || model.includes("ultra")) {
                return 8191;
            }
            return 2048;
        }
        if (model.includes("jamba")) {
            return 4096;
        }
    }
    // Cohere models
    else if (model.includes("cohere.command")) {
        if (model.includes("command-a")) {
            return 8192;
        }
        return 4096;
    }
    // Meta models
    else if (model.includes("llama")) {
        if (model.includes("3-70b") || model.includes("3-8b")) {
            return 2048;
        }
        return 8192;
    }
    //Writer models
    else if (model.includes("writer")) {
        if (model.includes("palmyra-x5")) {
            return 8192;
        }
        else if (model.includes("palmyra-x4")) {
            return 8192;
        }
    }

    // Default fallback
    return undefined;
}

export function getBedrockOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    if (model.includes("canvas")) {
        const taskTypeList: ModelOptionInfoItem = {
            name: "taskType",
            type: OptionType.enum,
            enum: {
                "Text-To-Image": "TEXT_IMAGE",
                "Text-To-Image-with-Image-Conditioning": "TEXT_IMAGE_WITH_IMAGE_CONDITIONING",
                "Color-Guided-Generation": "COLOR_GUIDED_GENERATION",
                "Image-Variation": "IMAGE_VARIATION",
                "Inpainting": "INPAINTING",
                "Outpainting": "OUTPAINTING",
                "Background-Removal": "BACKGROUND_REMOVAL",
            },
            default: "TEXT_IMAGE",
            description: "The type of task to perform",
            refresh: true,
        };

        let otherOptions: ModelOptionInfoItem[] = [
            { name: "width", type: OptionType.numeric, min: 320, max: 4096, default: 512, step: 16, integer: true, description: "The width of the generated image" },
            { name: "height", type: OptionType.numeric, min: 320, max: 4096, default: 512, step: 16, integer: true, description: "The height of the generated image" },
            {
                name: "quality",
                type: OptionType.enum,
                enum: { "standard": "standard", "premium": "premium" },
                default: "standard",
                description: "The quality of the generated image"
            },
            { name: "cfgScale", type: OptionType.numeric, min: 1.1, max: 10.0, default: 6.5, step: 0.1, integer: false, description: "The scale of the generated image" },
            { name: "seed", type: OptionType.numeric, min: 0, max: 858993459, default: 12, integer: true, description: "The seed of the generated image" },
            { name: "numberOfImages", type: OptionType.numeric, min: 1, max: 5, default: 1, integer: true, description: "The number of images to generate" },
        ];

        let dependentOptions: ModelOptionInfoItem[] = [];

        switch ((option as NovaCanvasOptions)?.taskType ?? "TEXT_IMAGE") {
            case "TEXT_IMAGE_WITH_IMAGE_CONDITIONING":
                dependentOptions.push(
                    {
                        name: "controlMode", type: OptionType.enum, enum: { "CANNY_EDGE": "CANNY_EDGE", "SEGMENTATION": "SEGMENTATION" },
                        default: "CANNY_EDGE", description: "The control mode of the generated image"
                    },
                    { name: "controlStrength", type: OptionType.numeric, min: 0, max: 1, default: 0.7, description: "The control strength of the generated image" },
                );
                break;
            case "COLOR_GUIDED_GENERATION":
                dependentOptions.push(
                    { name: "colors", type: OptionType.string_list, value: [], description: "Hexadecimal color values to guide generation" },
                )
                break;
            case "IMAGE_VARIATION":
                dependentOptions.push(
                    { name: "similarityStrength", type: OptionType.numeric, min: 0.2, max: 1, default: 0.7, description: "The similarity strength of the generated image" },
                )
                break;
            case "INPAINTING":
                //No changes
                break;
            case "OUTPAINTING":
                dependentOptions.push(
                    {
                        name: "outPaintingMode", type: OptionType.enum, enum: { "DEFAULT": "DEFAULT", "PRECISE": "PRECISE" },
                        default: "default", description: "The outpainting mode of the generated image"
                    },
                )
                break;
            case "BACKGROUND_REMOVAL":
                dependentOptions = [];
                otherOptions = [];
                break;
        }

        return {
            _option_id: "bedrock-nova-canvas",
            options: [
                taskTypeList,
                ...otherOptions,
                ...dependentOptions,
            ]
        };
    } else {
        const max_tokens_limit = getMaxTokensLimitBedrock(model);
        //Not canvas, i.e normal AWS bedrock converse
        const baseConverseOptions: ModelOptionInfoItem[] = [
            {
                name: "max_tokens",
                type: OptionType.numeric,
                min: 1,
                max: max_tokens_limit,
                integer: true,
                step: 200,
                description: "The maximum number of tokens to generate",
            },
            {
                name: "temperature",
                type: OptionType.numeric,
                min: 0.0,
                default: 0.7,
                step: 0.1,
                description: "A higher temperature biases toward less likely tokens, making the model more creative"
            },
            {
                name: "top_p",
                type: OptionType.numeric,
                min: 0,
                max: 1,
                step: 0.1,
                description: "Limits token sampling to the cumulative probability of the top p tokens"
            },
            {
                name: "stop_sequence",
                type: OptionType.string_list,
                value: [],
                description: "The generation will halt if one of the stop sequences is output"
            }];

        if (model.includes("claude")) {
            const claudeConverseOptions: ModelOptionInfoItem[] = [
                {
                    name: "top_k",
                    type: OptionType.numeric,
                    min: 1,
                    integer: true,
                    step: 1,
                    description: "Limits token sampling to the top k tokens"
                },
            ];
            if (model.includes("-3-7-") || model.includes("-4-")) {
                const claudeModeOptions: ModelOptionInfoItem[] = [
                    {
                        name: "thinking_mode",
                        type: OptionType.boolean,
                        default: false,
                        description: "If true, use the extended reasoning mode"
                    },
                ];
                const claudeThinkingOptions: ModelOptionInfoItem[] = (option as BedrockClaudeOptions)?.thinking_mode ? [
                    {
                        name: "thinking_budget_tokens",
                        type: OptionType.numeric,
                        min: 1024,
                        default: 4000,
                        integer: true,
                        step: 100,
                        description: "The target number of tokens to use for reasoning, not a hard limit."
                    },
                    {
                        name: "include_thoughts",
                        type: OptionType.boolean,
                        default: false,
                        description: "If true, include the reasoning in the response"
                    },
                ] : [];

                return {
                    _option_id: "bedrock-claude",
                    options: [
                        ...baseConverseOptions,
                        ...claudeConverseOptions,
                        ...claudeModeOptions,
                        ...claudeThinkingOptions]
                }
            }
            return {
                _option_id: "bedrock-claude",
                options: [...baseConverseOptions, ...claudeConverseOptions]
            }
        }
        else if (model.includes("amazon")) {
            //Titan models also exists but does not support any additional options
            if (model.includes("nova")) {
                const novaConverseOptions: ModelOptionInfoItem[] = [
                    {
                        name: "top_k",
                        type: OptionType.numeric,
                        min: 1,
                        integer: true,
                        step: 1,
                        description: "Limits token sampling to the top k tokens"
                    },
                ];
                return {
                    _option_id: "bedrock-nova",
                    options: [...baseConverseOptions, ...novaConverseOptions]
                }
            }
        }
        else if (model.includes("mistral")) {
            //7b and 8x7b instruct
            if (model.includes("7b")) {
                const mistralConverseOptions: ModelOptionInfoItem[] = [
                    {
                        name: "top_k",
                        type: OptionType.numeric,
                        min: 1,
                        integer: true,
                        step: 1,
                        description: "Limits token sampling to the top k tokens"
                    },
                ];
                return {
                    _option_id: "bedrock-mistral",
                    options: [...baseConverseOptions, ...mistralConverseOptions]
                }
            }
            //Other models such as Mistral Small, Large and Large 2
            //Support no additional options
        }
        else if (model.includes("ai21")) {
            const ai21ConverseOptions: ModelOptionInfoItem[] = [
                {
                    name: "presence_penalty",
                    type: OptionType.numeric,
                    min: -2,
                    max: 2,
                    default: 0,
                    step: 0.1,
                    description: "A higher presence penalty encourages the model to talk about new topics"
                },
                {
                    name: "frequency_penalty",
                    type: OptionType.numeric,
                    min: -2,
                    max: 2,
                    default: 0,
                    step: 0.1,
                    description: "A higher frequency penalty encourages the model to use less common words"
                },
            ];

            return {
                _option_id: "bedrock-ai21",
                options: [...baseConverseOptions, ...ai21ConverseOptions]
            }
        }
        else if (model.includes("cohere.command")) {
            const cohereCommandOptions: ModelOptionInfoItem[] = [
                {
                    name: "top_k",
                    type: OptionType.numeric,
                    min: 1,
                    integer: true,
                    step: 1,
                    description: "Limits token sampling to the top k tokens"
                },
            ];
            if (model.includes("command-r")) {
                const cohereCommandROptions: ModelOptionInfoItem[] = [
                    {
                        name: "frequency_penalty",
                        type: OptionType.numeric,
                        min: -2,
                        max: 2,
                        default: 0,
                        step: 0.1,
                        description: "A higher frequency penalty encourages the model to use less common words"
                    },
                    {
                        name: "presence_penalty",
                        type: OptionType.numeric,
                        min: -2,
                        max: 2,
                        default: 0,
                        step: 0.1,
                        description: "A higher presence penalty encourages the model to talk about new topics"
                    },
                ];
                return {
                    _option_id: "bedrock-cohere-command",
                    options: [...baseConverseOptions, ...cohereCommandOptions, ...cohereCommandROptions]
                }
            }
        } else if (model.includes("writer")) {
            const palmyraConverseOptions: ModelOptionInfoItem[] = [
                {
                    name: "min_tokens",
                    type: OptionType.numeric,
                    min: 1,
                    max: max_tokens_limit,
                    integer: false,
                    step: 100,
                },
                {
                    name: "seed",
                    type: OptionType.numeric,
                    integer: true,
                    description: "Random seed for generation"
                },
                {
                    name: "frequency_penalty",
                    type: OptionType.numeric,
                    min: -2,
                    max: 2,
                    default: 0,
                    step: 0.1,
                    description: "A higher frequency penalty encourages the model to use less common words"
                },
                {
                    name: "presence_penalty",
                    type: OptionType.numeric,
                    min: -2,
                    max: 2,
                    default: 0,
                    step: 0.1,
                    description: "A higher presence penalty encourages the model to talk about new topics"
                },
            ]
            return {
                _option_id: "bedrock-palmyra",
                options: [...baseConverseOptions, ...palmyraConverseOptions]
            }
        }

        //Fallback to converse standard.
        return {
            _option_id: "bedrock-converse",
            options: baseConverseOptions
        };
    }
    return textOptionsFallback;
}