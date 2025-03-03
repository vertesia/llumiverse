import { ModelOptionsInfo, ModelOptionInfoItem, ModelOptions, OptionType, SharedOptions } from "../types.js";
import { textOptionsFallback } from "../options.js";

// Union type of all Bedrock options
export type VertexAIOptions = ImagenOptions | VertexAIClaudeOptions;

export interface ImagenOptions {
    _option_id: "vertexai-imagen"
    number_of_images?: number;
    seed?: number;
    person_generation?: "dont_allow" | "random" | "allow_all";
    safety_setting?: "block_none" | "block_only_high" | "block_medium_and_above" | "block_low_and_above";
    image_file_type?: "image/jpeg" | "image/png";
    jpeg_compression_quality?: number;
    aspect_ratio?: "1:1" | "4:3" | "16:9";
    add_watermark?: boolean;
    edit_mode?: "EDIT_MODE_INPAINT_REMOVAL" | "EDIT_MODE_INPAINT_INSERTION" | "EDIT_MODE_BGSWAP" | "EDIT_MODE_OUTPAINT";
    guidance_scale?: number;
}

export interface VertexAIClaudeOptions {
    _option_id: "vertexai-claude"
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    stop_sequence?: string[];
    thinking_mode?: boolean;
    thinking_budget_tokens?: number;
}

export function getVertexAiOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    if (model.includes("imagen-3.0")) {
        const commonOptions: ModelOptionInfoItem[] = [
            {
                name: SharedOptions.number_of_images, type: OptionType.numeric, min: 1, max: 4, default: 1,
                integer: true, description: "Number of Images to generate",
            },
            {
                name: SharedOptions.seed, type: OptionType.numeric, min: 0, max: 4294967295, default: 12,
                integer: true, description: "The seed of the generated image"
            },
            {
                name: "person_generation", type: OptionType.enum, enum: { "Disallow the inclusion of people or faces in images": "dont_allow", "Allow generation of adults only": "random", "Allow generation of people of all ages": "allow_all" },
                default: "allow_adult", description: "The type of person to generate"
            },
            {
                name: "safety_setting", type: OptionType.enum, enum: { "Block very few problematic prompts and responses": "block_none", "Block only few problematic prompts and responses": "block_only_high", "Block some problematic prompts and responses": "block_medium_and_above", "Strictest filtering": "block_low_and_above" },
                default: "block_medium_and_above", description: "The safety setting for the generated image"
            },
        ];


        const outputOptions: ModelOptionInfoItem[] = [
            {
                name: "image_file_type", type: OptionType.enum, enum: { "JPEG": "image/jpeg", "PNG": "image/png" },
                default: "image/png", description: "The file type of the generated image",
                refresh: true,
            },
        ]

        const jpegQuality: ModelOptionInfoItem = {
            name: "jpeg_compression_quality", type: OptionType.numeric, min: 0, max: 100, default: 75,
            integer: true, description: "The compression quality of the JPEG image",
        }

        if ((option as ImagenOptions)?.image_file_type === "image/jpeg") {
            outputOptions.push(jpegQuality);
        }
        if (model.includes("generate")) {
            const modeOptions: ModelOptionInfoItem[] = [
                {
                    name: "aspect_ratio", type: OptionType.enum, enum: { "1:1": "1:1", "4:3": "4:3", "16:9": "16:9" },
                    default: "1:1", description: "The aspect ratio of the generated image"
                },
                {
                    name: "add_watermark", type: OptionType.boolean, default: true, description: "Add an invisible watermark to the generated image, useful for detection of AI images"
                },
                
            ];

            return {
                _option_id: "vertexai-imagen",
                options: [
                    ...commonOptions,
                    ...modeOptions,
                    ...outputOptions,
                ]
            };
        }
        if (model.includes("capability")) {
            let guidanceScaleDefault = 75;
            if ((option as ImagenOptions).edit_mode === "EDIT_MODE_INPAINT_INSERTION") {
                guidanceScaleDefault = 60;
            }
        
            const modeOptions: ModelOptionInfoItem[] = [
                {
                    name: "edit_mode", type: OptionType.enum,
                    enum: {
                        "Inpaint Removal": "EDIT_MODE_INPAINT_REMOVAL",
                        "Inpaint Insertion": "EDIT_MODE_INPAINT_INSERTION",
                        "Background Swap": "EDIT_MODE_BGSWAP",
                        "Outpaint": "EDIT_MODE_OUTPAINT",
                    },
                },
                {
                    name: "guidance_scale", type: OptionType.numeric, min: 0, max: 500, default: guidanceScaleDefault,
                    integer: true, description: "The scale of the guidance image"
                }
            ];

            return {
                _option_id: "vertexai-imagen",
                options: [
                    ...commonOptions,
                    ...modeOptions,
                    ...outputOptions,
                ]
            };
        }
    }
    else if (model.includes("gemini")) {
        const max_tokens_limit = getGeminiMaxTokensLimit(model);
        const excludeOptions = ["presence_penalty"];
        let commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));
        if (model.includes("1.5")) {
            commonOptions = commonOptions.filter((option) => option.name !== "frequency_penalty");
        }
        const max_tokens: ModelOptionInfoItem[] = [{
            name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
            integer: true, step: 200, description: "The maximum number of tokens to generate"
        }];
        return {
            _option_id: "vertexai-gemini",
            options: [
                ...commonOptions,
                ...max_tokens
            ]
        };
    }
    else if (model.includes("claude")) {
        const max_tokens_limit = getClaudeMaxTokensLimit(model, option as VertexAIClaudeOptions);
        const excludeOptions = ["presence_penalty", "frequency_penalty"];
        let commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));
        const max_tokens: ModelOptionInfoItem[] = [{
            name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
            integer: true, step: 200, description: "The maximum number of tokens to generate"
        }];
        if (model.includes("3-7")) {
            const claudeModeOptions: ModelOptionInfoItem[] = [
                {
                    name: "thinking_mode",
                    type: OptionType.boolean,
                    default: false,
                    description: "If true, use the extended reasoning mode"
                },
            ];
            const claudeThinkingOptions: ModelOptionInfoItem[] = (option as VertexAIClaudeOptions)?.thinking_mode ? [
                {
                    name: "thinking_budget_tokens",
                    type: OptionType.numeric,
                    min: 1024,
                    default: 4000,
                    integer: true,
                    step: 100,
                    description: "The target number of tokens to use for reasoning, not a hard limit."
                },
            ] : [];

            return {
                _option_id: "vertexai-claude",
                options: [
                    ...commonOptions,
                    ...max_tokens,
                    ...claudeModeOptions,
                    ...claudeThinkingOptions,
                ]
            };
        }
        return {
            _option_id: "vertexai-claude",
            options: [
                ...commonOptions,
                ...max_tokens,
            ]
        };
    }
    return textOptionsFallback;
}
function getGeminiMaxTokensLimit(model: string): number {
    if (model.includes("thinking")) {
        return 65536;
    }
    if (model.includes("ultra") || model.includes("vision")) {
        return 2048;
    }
    return 8192;
}
function getClaudeMaxTokensLimit(model: string, option?: VertexAIClaudeOptions): number {
    if (model.includes("3-7")) {
        if (option && option?.thinking_mode) {
            return 128000;
        } else {
            return 8192;
        }
    }
    else if (model.includes("3-5")) {
        return 8192;
    }
    else {
        return 4096;
    }
}

