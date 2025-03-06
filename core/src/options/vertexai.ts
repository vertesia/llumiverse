import { ModelOptionsInfo, ModelOptionInfoItem, OptionType, SharedOptions, ModelOptions } from "../types.js";
import { textOptionsFallback } from "../options.js";

// Union type of all Bedrock options
export type VertexAIOptions = ImagenOptions | VertexAIClaudeOptions;

export enum ImagenTaskType {
    TEXT_IMAGE = "TEXT_IMAGE",
    EDIT_MODE_INPAINT_REMOVAL = "EDIT_MODE_INPAINT_REMOVAL",
    EDIT_MODE_INPAINT_INSERTION = "EDIT_MODE_INPAINT_INSERTION",
    EDIT_MODE_BGSWAP = "EDIT_MODE_BGSWAP",
    EDIT_MODE_OUTPAINT = "EDIT_MODE_OUTPAINT",
    CUSTOMIZATION_GENERATE = "CUSTOMIZATION_GENERATE",
    CUSTOMIZATION_EDIT = "CUSTOMIZATION_EDIT",
}

export enum ImagenMaskMode {
    MASK_MODE_USER_PROVIDED = "MASK_MODE_USER_PROVIDED",
    MASK_MODE_BACKGROUND = "MASK_MODE_BACKGROUND",
    MASK_MODE_FOREGROUND = "MASK_MODE_FOREGROUND",
    MASK_MODE_SEMANTIC = "MASK_MODE_SEMANTIC",
}

export interface ImagenOptions {
    _option_id: "vertexai-imagen"

    //General and generate options
    number_of_images?: number;
    seed?: number;
    person_generation?: "dont_allow" | "allow_adults" | "allow_all";
    safety_setting?: "block_none" | "block_only_high" | "block_medium_and_above" | "block_low_and_above"; //The "off" option does not seem to work for Imagen 3, might be only for text models
    image_file_type?: "image/jpeg" | "image/png";
    jpeg_compression_quality?: number;
    aspect_ratio?: "1:1" | "4:3" | "3:4" | "16:9" | "9:16";
    add_watermark?: boolean;
    enhance_prompt?: boolean;

    //Capability options
    edit_mode?: ImagenTaskType
    guidance_scale?: number;
    edit_steps?: number;
    mask_mode?: ImagenMaskMode;
    mask_dilation?: number;
    mask_class?: number[];

    //Customization options
    controlType: "CONTROL_TYPE_FACE_MESH" | "CONTROL_TYPE_CANNY" | "CONTROL_TYPE_SCRIBBLE";
    controlImageComputation?: boolean;
    subjectType: "SUBJECT_TYPE_PERSON" | "SUBJECT_TYPE_ANIMAL" | "SUBJECT_TYPE_PRODUCT" | "SUBJECT_TYPE_DEFAULT";
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
                name: "person_generation", type: OptionType.enum, enum: { "Disallow the inclusion of people or faces in images": "dont_allow", "Allow generation of adults only": "allow_adult", "Allow generation of people of all ages": "allow_all" },
                default: "allow_adult", description: "The safety setting for allowing the generation of people in the image"
            },
            {
                name: "safety_setting", type: OptionType.enum, enum: { "Block very few problematic prompts and responses": "block_none", "Block only few problematic prompts and responses": "block_only_high", "Block some problematic prompts and responses": "block_medium_and_above", "Strictest filtering": "block_low_and_above" },
                default: "block_medium_and_above", description: "The overall safety setting"
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
            //Generate models
            const modeOptions: ModelOptionInfoItem[]
                = [
                    {
                        name: "aspect_ratio", type: OptionType.enum, enum: { "1:1": "1:1", "4:3": "4:3", "3:4": "3:4", "16:9": "16:9", "9:16": "9:16" },
                        default: "1:1", description: "The aspect ratio of the generated image"
                    },
                    {
                        name: "add_watermark", type: OptionType.boolean, default: false, description: "Add an invisible watermark to the generated image, useful for detection of AI images"
                    },

                ];

            const enhanceOptions: ModelOptionInfoItem[] = !model.includes("generate-001") ? [
                {
                    name: "enhance_prompt", type: OptionType.boolean, default: true, description: "VertexAI automatically rewrites the prompt to better reflect the prompt's intent."
                },
            ] : [];

            return {
                _option_id: "vertexai-imagen",
                options: [
                    ...commonOptions,
                    ...modeOptions,
                    ...outputOptions,
                    ...enhanceOptions,
                ]
            };
        }
        if (model.includes("capability")) {
            //Edit models
            let guidanceScaleDefault = 75;
            if ((option as ImagenOptions)?.edit_mode === "EDIT_MODE_INPAINT_INSERTION") {
                guidanceScaleDefault = 60;
            }

            const modeOptions: ModelOptionInfoItem[] = [
                {
                    name: "edit_mode", type: OptionType.enum,
                    enum: {
                        "EDIT_MODE_INPAINT_REMOVAL": "EDIT_MODE_INPAINT_REMOVAL",
                        "EDIT_MODE_INPAINT_INSERTION": "EDIT_MODE_INPAINT_INSERTION",
                        "EDIT_MODE_BGSWAP": "EDIT_MODE_BGSWAP",
                        "EDIT_MODE_OUTPAINT": "EDIT_MODE_OUTPAINT",
                        "CUSTOMIZATION_GENERATE": "CUSTOMIZATION_GENERATE",
                        "CUSTOMIZATION_EDIT": "CUSTOMIZATION_EDIT",
                    },
                    default: "EDIT_MODE_INPAINT_REMOVAL",
                    description: "The editing mode. CUSTOMIZATION options use few-shot learning to generate images based on a few examples."
                },
                {
                    name: "mask_mode", type: OptionType.enum,
                    enum: {
                        "MASK_MODE_USER_PROVIDED": "MASK_MODE_USER_PROVIDED",
                        "MASK_MODE_BACKGROUND": "MASK_MODE_BACKGROUND",
                        "MASK_MODE_FOREGROUND": "MASK_MODE_FOREGROUND",
                        "MASK_MODE_SEMANTIC": "MASK_MODE_SEMANTIC",
                    },
                    default: "MASK_MODE_USER_PROVIDED",
                    description: "How should the mask for the generation be provided"
                },
                {
                    name: "mask_dilation", type: OptionType.numeric, min: 0, max: 1, default: 0.01,
                    integer: true, description: "The mask dilation, grows the mask by a percetage of image width to compensate for imprecise masks."
                },
                {
                    name: "guidance_scale", type: OptionType.numeric, min: 0, max: 500, default: guidanceScaleDefault,
                    integer: true, description: "The scale of the guidance image"
                }
            ];

            const editOptions: ModelOptionInfoItem[] = (option as ImagenOptions)?.edit_mode?.includes("edit") ? [
                {
                    name: "edit_steps", type: OptionType.numeric, min: 1, max: 500, default: 35,
                    integer: true, description: "The number of steps for the base image generation, more steps means more time and better quality"
                },
            ] : [];

            const maskClassOptions: ModelOptionInfoItem[] = ((option as ImagenOptions)?.mask_mode === "MASK_MODE_SEMANTIC") ? [
                {
                    name: "mask_class", type: OptionType.string_list, default: [],
                    description: "Input Class IDs. Create a mask based on image class, based on https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api-customization#segment-ids"
                }
            ] : [];

            const customizationOptions: ModelOptionInfoItem[] = (option as ImagenOptions)?.edit_mode === "CUSTOMIZATION_GENERATE" ? [
                {
                    name: "controlType", type: OptionType.enum, enum: { "Face Mesh": "CONTROL_TYPE_FACE_MESH", "Canny": "CONTROL_TYPE_CANNY", "Scribble": "CONTROL_TYPE_SCRIBBLE" },
                    default: "CONTROL_TYPE_CANNY", description: "Method used to generate the control image"
                },
                {
                    name: "controlImageComputation", type: OptionType.boolean, default: true, description: "Should the control image be computed from the input image, or is it provided"
                }
            ] : [];

            return {
                _option_id: "vertexai-imagen",
                options: [
                    ...modeOptions,
                    ...commonOptions,
                    ...editOptions,
                    ...outputOptions,
                    ...maskClassOptions,
                    ...customizationOptions,
                ]
            };
        }
    }
    else if (model.includes("gemini")) {
        const max_tokens_limit = getGeminiMaxTokensLimit(model);
        const excludeOptions = ["max_tokens", "presence_penalty"];
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
                ...max_tokens,
                ...commonOptions,
            ]
        };
    }
    else if (model.includes("claude")) {
        const max_tokens_limit = getClaudeMaxTokensLimit(model, option as VertexAIClaudeOptions);
        const excludeOptions = ["max_tokens", "presence_penalty", "frequency_penalty"];
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
                    ...max_tokens,
                    ...commonOptions,
                    ...claudeModeOptions,
                    ...claudeThinkingOptions,
                ]
            };
        }
        return {
            _option_id: "vertexai-claude",
            options: [
                ...max_tokens,
                ...commonOptions,
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

