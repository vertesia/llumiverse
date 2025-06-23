import { ModelOptionsInfo, ModelOptionInfoItem, OptionType, SharedOptions, ModelOptions } from "../types.js";
import { textOptionsFallback } from "./fallback.js";

// Union type of all VertexAI options
export type VertexAIOptions = ImagenOptions | VertexAIClaudeOptions | VertexAIGeminiOptions;

export enum ImagenTaskType {
    TEXT_IMAGE = "TEXT_IMAGE",
    EDIT_MODE_INPAINT_REMOVAL = "EDIT_MODE_INPAINT_REMOVAL",
    EDIT_MODE_INPAINT_INSERTION = "EDIT_MODE_INPAINT_INSERTION",
    EDIT_MODE_BGSWAP = "EDIT_MODE_BGSWAP",
    EDIT_MODE_OUTPAINT = "EDIT_MODE_OUTPAINT",
    CUSTOMIZATION_SUBJECT = "CUSTOMIZATION_SUBJECT",
    CUSTOMIZATION_STYLE = "CUSTOMIZATION_STYLE",
    CUSTOMIZATION_CONTROLLED = "CUSTOMIZATION_CONTROLLED",
    CUSTOMIZATION_INSTRUCT = "CUSTOMIZATION_INSTRUCT",
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
    controlType?: "CONTROL_TYPE_FACE_MESH" | "CONTROL_TYPE_CANNY" | "CONTROL_TYPE_SCRIBBLE";
    controlImageComputation?: boolean;
    subjectType?: "SUBJECT_TYPE_PERSON" | "SUBJECT_TYPE_ANIMAL" | "SUBJECT_TYPE_PRODUCT" | "SUBJECT_TYPE_DEFAULT";
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
    include_thoughts?: boolean;
}

export interface VertexAIGeminiOptions {
    _option_id: "vertexai-gemini"
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    stop_sequence?: string[];
    presence_penalty?: number;
    frequency_penalty?: number;
    seed?: number;
    include_thoughts?: boolean;
    thinking_budget_tokens?: number;
}

export function getVertexAiOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    if (model.includes("imagen-")) {
        return getImagenOptions(model, option);
    } else if (model.includes("gemini")) {
        return getGeminiOptions(model, option);
    } else if (model.includes("claude")) {
        return getClaudeOptions(model, option);
    } else if (model.includes("llama")) {
        return getLlamaOptions(model);
    }
    return textOptionsFallback;
}

function getImagenOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
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
        // Generate models
        const modeOptions: ModelOptionInfoItem[] = [
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
        // Edit models
        let guidanceScaleDefault = 75;
        if ((option as ImagenOptions)?.edit_mode === ImagenTaskType.EDIT_MODE_INPAINT_INSERTION) {
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
                    "CUSTOMIZATION_SUBJECT": "CUSTOMIZATION_SUBJECT",
                    "CUSTOMIZATION_STYLE": "CUSTOMIZATION_STYLE",
                    "CUSTOMIZATION_CONTROLLED": "CUSTOMIZATION_CONTROLLED",
                    "CUSTOMIZATION_INSTRUCT": "CUSTOMIZATION_INSTRUCT",
                },
                description: "The editing mode. CUSTOMIZATION options use few-shot learning to generate images based on a few examples."
            },
            {
                name: "guidance_scale", type: OptionType.numeric, min: 0, max: 500, default: guidanceScaleDefault,
                integer: true, description: "How closely the generation follows the prompt"
            }
        ];

        const maskOptions: ModelOptionInfoItem[] = ((option as ImagenOptions)?.edit_mode?.includes("EDIT")) ? [
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
                name: "mask_dilation", type: OptionType.numeric, min: 0, max: 1,
                integer: true, description: "The mask dilation, grows the mask by a percentage of image width to compensate for imprecise masks."
            },
        ] : [];

        const maskClassOptions: ModelOptionInfoItem[] = ((option as ImagenOptions)?.mask_mode === ImagenMaskMode.MASK_MODE_SEMANTIC) ? [
            {
                name: "mask_class", type: OptionType.string_list, default: [],
                description: "Input Class IDs. Create a mask based on image class, based on https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api-customization#segment-ids"
            }
        ] : [];

        const editOptions: ModelOptionInfoItem[] = (option as ImagenOptions)?.edit_mode?.includes("EDIT") ? [
            {
                name: "edit_steps", type: OptionType.numeric, default: 75,
                integer: true, description: "The number of steps for the base image generation, more steps means more time and better quality"
            },
        ] : [];

        const customizationOptions: ModelOptionInfoItem[] = (option as ImagenOptions)?.edit_mode === ImagenTaskType.CUSTOMIZATION_CONTROLLED
            || (option as ImagenOptions)?.edit_mode === ImagenTaskType.CUSTOMIZATION_SUBJECT ? [
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
                ...maskOptions,
                ...maskClassOptions,
                ...editOptions,
                ...customizationOptions,
                ...outputOptions,
            ]
        };
    }

    return textOptionsFallback;
}

function getGeminiOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    const max_tokens_limit = getGeminiMaxTokensLimit(model);
    const excludeOptions = ["max_tokens"];
    let commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));

    const max_tokens: ModelOptionInfoItem[] = [{
        name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
        integer: true, step: 200, description: "The maximum number of tokens to generate"
    }];

    const seedOption: ModelOptionInfoItem = {
        name: SharedOptions.seed, type: OptionType.numeric, integer: true, description: "The seed for the generation, useful for reproducibility"
    };

    if (model.includes("-2.5-")) {
        // Gemini 2.5 thinking models
        
        // Set budget token ranges based on model variant
        let budgetMin = -1;
        let budgetMax = 24576;
        let budgetDescription = "";
        if (model.includes("flash-lite")) {
            budgetMin = -1;
            budgetMax = 24576;
            budgetDescription = "The target number of tokens to use for reasoning. " +
                "Flash Lite default: Model does not think. " +
                "Range: 512-24576 tokens. " +
                "Set to 0 to disable thinking, -1 for dynamic thinking.";
        } else if (model.includes("flash")) {
            budgetMin = -1;
            budgetMax = 24576;
            budgetDescription = "The target number of tokens to use for reasoning. " +
                "Flash default: Dynamic thinking (model decides when and how much to think). " +
                "Range: 0-24576 tokens. " +
                "Set to 0 to disable thinking, -1 for dynamic thinking.";
        } else if (model.includes("pro")) {
            budgetMin = -1;
            budgetMax = 32768;
            budgetDescription = "The target number of tokens to use for reasoning. " +
                "Pro default: Dynamic thinking (model decides when and how much to think). " +
                "Range: 128-32768 tokens. " +
                "Cannot disable thinking - minimum 128 tokens. Set to -1 for dynamic thinking.";
        }
        
        const geminiThinkingOptions: ModelOptionInfoItem[] = [
            {
                name: "include_thoughts",
                type: OptionType.boolean,
                default: false,
                description: "Include the model's reasoning process in the response"
            },
            {
                name: "thinking_budget_tokens",
                type: OptionType.numeric,
                min: budgetMin,
                max: budgetMax,
                default: undefined,
                integer: true,
                step: 100,
                description: budgetDescription,
            }
        ];

        return {
            _option_id: "vertexai-gemini",
            options: [
                ...max_tokens,
                ...commonOptions,
                seedOption,
                ...geminiThinkingOptions,
            ]
        };
    }

    return {
        _option_id: "vertexai-gemini",
        options: [
            ...max_tokens,
            ...commonOptions,
            seedOption,
        ]
    };
}

function getClaudeOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    const max_tokens_limit = getClaudeMaxTokensLimit(model);
    const excludeOptions = ["max_tokens", "presence_penalty", "frequency_penalty"];
    let commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));
    const max_tokens: ModelOptionInfoItem[] = [{
        name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
        integer: true, step: 200, description: "The maximum number of tokens to generate"
    }];

    if (model.includes("-3-7") || model.includes("-4")) {
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
                default: 1024,
                integer: true,
                step: 100,
                description: "The target number of tokens to use for reasoning, not a hard limit."
            },
            {
                name: "include_thoughts",
                type: OptionType.boolean,
                default: false,
                description: "Include the model's reasoning process in the response"
            }
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

function getLlamaOptions(model: string): ModelOptionsInfo {
    const max_tokens_limit = getLlamaMaxTokensLimit(model);
    const excludeOptions = ["max_tokens", "presence_penalty", "frequency_penalty", "stop_sequence"];
    let commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));
    const max_tokens: ModelOptionInfoItem[] = [{
        name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
        integer: true, step: 200, description: "The maximum number of tokens to generate"
    }];
    
    // Set max temperature to 1.0 for Llama models
    commonOptions = commonOptions.map((option) => {
        if (
            option.name === SharedOptions.temperature &&
            option.type === OptionType.numeric
        ) {
            return {
                ...option,
                max: 1.0,
            };
        }
        return option;
    });

    return {
        _option_id: "text-fallback",
        options: [
            ...max_tokens,
            ...commonOptions,
        ]
    };
}

function getGeminiMaxTokensLimit(model: string): number {
    if (model.includes("thinking") || model.includes("-2.5-")) {
        return 65536;
    }
    if (model.includes("ultra") || model.includes("vision")) {
        return 2048;
    }
    return 8192;
}

function getClaudeMaxTokensLimit(model: string): number {
    if (model.includes("-4-")) {
        if(model.includes("opus-")) {
            return 32768;
        }
        return 65536;
    }
    else if (model.includes("-3-7-")) {
        return 128000;
    }
    else if (model.includes("-3-5-")) {
        return 8192;
    }
    else {
        return 4096;
    }
}

function getLlamaMaxTokensLimit(_model: string): number {
    return 8192;
}

export function getMaxTokensLimitVertexAi(model: string): number {
    if (model.includes("imagen-")) {
        return 0; // Imagen models do not have a max tokens limit in the same way as text models
    } else if (model.includes("claude")) {
        return getClaudeMaxTokensLimit(model);
    } else if (model.includes("gemini")) {
        return getGeminiMaxTokensLimit(model);
    } else if (model.includes("llama")) {
        return getLlamaMaxTokensLimit(model);
    }
    return 8192; // Default fallback limit
}
