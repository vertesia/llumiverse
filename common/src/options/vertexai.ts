import { ModelOptionInfoItem, ModelOptions, ModelOptionsInfo, OptionType, SharedOptions } from "../types.js";
import { getMaxOutputTokens } from "./context-windows.js";
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

export enum ThinkingLevel {
    HIGH = "HIGH",
    MEDIUM = "MEDIUM",
    LOW = "LOW",
    MINIMAL = "MINIMAL",
    THINKING_LEVEL_UNSPECIFIED = "THINKING_LEVEL_UNSPECIFIED"
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
    cache_enabled?: boolean;
    cache_ttl?: '5m' | '1h';
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
    thinking_level?: ThinkingLevel;
    flex?: boolean;
    // ImageConfig properties
    image_aspect_ratio?: "1:1" | "2:3" | "3:2" | "3:4" | "4:3" | "9:16" | "16:9" | "21:9";
    image_size?: "1K" | "2K" | "4K";
    person_generation?: "ALLOW_ALL" | "ALLOW_ADULT" | "ALLOW_NONE";
    prominent_people?: "PROMINENT_PEOPLE_UNSPECIFIED" | "ALLOW_PROMINENT_PEOPLE" | "BLOCK_PROMINENT_PEOPLE";
    output_mime_type?: "image/png" | "image/jpeg";
    output_compression_quality?: number;
}

/** Models that support Flex processing (shared, cost-efficient tier). */
const FLEX_SUPPORTED_GEMINI_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-image-preview",
] as const;

export function isFlexSupportedGeminiModel(model: string): boolean {
    const modelName = model.split('/').pop() ?? model;
    return FLEX_SUPPORTED_GEMINI_MODELS.some(m => modelName.includes(m));
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

/**
 * Extract Gemini version from a model ID.
 *
 * Examples:
 * - locations/global/publishers/google/models/gemini-2.5-flash -> 2.5
 * - publishers/google/models/gemini-3-pro-image-preview -> 3
 */
export function getGeminiModelVersion(modelId: string): string | undefined {
    const modelName = modelId.split('/').pop() ?? modelId;
    const match = modelName.match(/^gemini-(\d+(?:\.\d+)?)/i);
    return match?.[1];
}

function parseVersion(version: string): { major: number; minor: number } | undefined {
    const match = version.match(/^(\d+)(?:\.(\d+))?$/);
    if (!match) {
        return undefined;
    }

    return {
        major: Number(match[1]),
        minor: Number(match[2] ?? '0'),
    };
}

export function isGeminiModelVersionGte(modelId: string, minVersion: string): boolean {
    const modelVersion = getGeminiModelVersion(modelId);
    if (!modelVersion) {
        return false;
    }

    const current = parseVersion(modelVersion);
    const target = parseVersion(minVersion);
    if (!current || !target) {
        return false;
    }

    if (current.major > target.major) {
        return true;
    }
    if (current.major < target.major) {
        return false;
    }

    return current.minor >= target.minor;
}

function getGeminiThinkingLevels(model: string): {
    levels: Record<string, ThinkingLevel>;
    defaultLevel: ThinkingLevel;
} {
    const normalized = model.toLowerCase();
    const isGemini3OrLater = isGeminiModelVersionGte(model, "3.0");
    const isGemini31OrLater = isGeminiModelVersionGte(model, "3.1");
    const isFlashLite = normalized.includes("flash-lite");
    const isFlash = normalized.includes("flash") && !isFlashLite;
    const isPro = normalized.includes("pro");

    // Gemini 3 / 3.1 thinking_level support summary:
    // - MINIMAL: Gemini 3 Flash and Gemini 3.1 Flash-Lite only.
    //   Gemini 3.1 Flash-Lite defaults to MINIMAL.
    // - LOW: Supported by Gemini 3+ models.
    // - MEDIUM: Gemini 3 Flash, Gemini 3.1 Pro, Gemini 3.1 Flash-Lite.
    // - HIGH: Supported by Gemini 3+ models.
    // - Thinking cannot be turned off for Gemini 3 Pro and Gemini 3.1 Pro.
    if (isFlashLite && isGemini31OrLater) {
        return {
            levels: {
                "Minimal": ThinkingLevel.MINIMAL,
                "Low": ThinkingLevel.LOW,
                "Medium": ThinkingLevel.MEDIUM,
                "High": ThinkingLevel.HIGH,
            },
            defaultLevel: ThinkingLevel.MINIMAL,
        };
    }

    // Gemini 3+ Flash supports MINIMAL and MEDIUM in addition to LOW/HIGH.
    if (isFlash) {
        return {
            levels: {
                "Minimal": ThinkingLevel.MINIMAL,
                "Low": ThinkingLevel.LOW,
                "Medium": ThinkingLevel.MEDIUM,
                "High": ThinkingLevel.HIGH,
            },
            defaultLevel: ThinkingLevel.MINIMAL,
        };
    }

    // Gemini 3.1 Pro adds MEDIUM, but does not support turning thinking off.
    if (isPro && isGemini31OrLater) {
        return {
            levels: {
                "Low": ThinkingLevel.LOW,
                "Medium": ThinkingLevel.MEDIUM,
                "High": ThinkingLevel.HIGH,
            },
            defaultLevel: ThinkingLevel.LOW,
        };
    }

    // Gemini 3 Pro supports LOW/HIGH. Thinking cannot be turned off.
    if (isPro) {
        return {
            levels: {
                "Low": ThinkingLevel.LOW,
                "High": ThinkingLevel.HIGH,
            },
            defaultLevel: ThinkingLevel.LOW,
        };
    }

    // Fallback for unknown Gemini 3+/4+ families:
    // prefer future-safe propagation by enabling the guaranteed baseline levels.
    if (isGemini3OrLater) {
        return {
            levels: {
                "Low": ThinkingLevel.LOW,
                "Medium": ThinkingLevel.MEDIUM,
                "High": ThinkingLevel.HIGH,
            },
            defaultLevel: ThinkingLevel.LOW,
        };
    }

    // Non-3.x models should not reach this helper in normal flow.
    return {
        levels: {
            "Low": ThinkingLevel.LOW,
            "High": ThinkingLevel.HIGH,
        },
        defaultLevel: ThinkingLevel.LOW,
    };
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

function getGeminiThinkingOptionItems(model: string): ModelOptionInfoItem[] {
    const thinkingLevelConfig = getGeminiThinkingLevels(model);
    return [
        {
            name: "include_thoughts",
            type: OptionType.boolean,
            default: false,
            description: "Include the model's reasoning process in the response"
        },
        {
            name: "thinking_level",
            type: OptionType.enum,
            enum: thinkingLevelConfig.levels,
            default: thinkingLevelConfig.defaultLevel,
            description: "Higher thinking levels may improve quality, but increase response times and token costs"
        }
    ];
}

function getGeminiOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    // Special handling for gemini image / nano banana models
    if (model.includes("image")) {
        const isGemini25OrLater = isGeminiModelVersionGte(model, "2.5");
        const isGemini3OrLater = isGeminiModelVersionGte(model, "3.0");

        const max_tokens_limit = getGeminiMaxTokensLimit(model);
        const excludeOptions = ["max_tokens", "presence_penalty", "frequency_penalty", "seed", "top_k"];
        let commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));

        // Set max temperature to 2.0
        commonOptions = commonOptions.map((option) => {
            if (
                option.name === SharedOptions.temperature &&
                option.type === OptionType.numeric
            ) {
                return {
                    ...option,
                    max: 2.0,
                };
            }
            return option;
        });

        const max_tokens: ModelOptionInfoItem[] = [{
            name: SharedOptions.max_tokens,
            type: OptionType.numeric,
            min: 1,
            max: max_tokens_limit,
            integer: true,
            step: 200,
            description: "Maximum output tokens"
        }];

        const imageOptions: ModelOptionInfoItem[] = [];

        // Aspect ratio, person generation, prominent people: 2.5+
        if (isGemini25OrLater) {
            imageOptions.push(
                {
                    name: "image_aspect_ratio",
                    type: OptionType.enum,
                    enum: {
                        "1:1": "1:1",
                        "2:3": "2:3",
                        "3:2": "3:2",
                        "3:4": "3:4",
                        "4:3": "4:3",
                        "9:16": "9:16",
                        "16:9": "16:9",
                        "21:9": "21:9"
                    },
                    default: "1:1",
                    description: "Aspect ratio of the generated images"
                },
                {
                    name: "person_generation",
                    type: OptionType.enum,
                    enum: {
                        "Allow all people": "ALLOW_ALL",
                        "Allow adults only": "ALLOW_ADULT",
                        "Do not generate people": "ALLOW_NONE"
                    },
                    default: "ALLOW_ALL",
                    description: "Controls the generation of people in images"
                },
                {
                    name: "prominent_people",
                    type: OptionType.enum,
                    enum: {
                        "Allow prominent people": "ALLOW_PROMINENT_PEOPLE",
                        "Block prominent people": "BLOCK_PROMINENT_PEOPLE"
                    },
                    description: "Controls whether prominent people (celebrities) can be generated"
                },
            );
        }

        // Resolution settings: 3.0+
        if (isGemini3OrLater) {
            imageOptions.push({
                name: "image_size",
                type: OptionType.enum,
                enum: {
                    "1K": "1K",
                    "2K": "2K",
                    "4K": "4K"
                },
                default: "1K",
                description: "Size of generated images"
            });
        }

        // Output format: all image models
        imageOptions.push({
            name: "output_mime_type",
            type: OptionType.enum,
            enum: {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
            },
            default: "image/png",
            description: "MIME type of the generated image",
            refresh: true,
        });

        if ((option as VertexAIGeminiOptions)?.output_mime_type === "image/jpeg") {
            imageOptions.push({
                name: "output_compression_quality",
                type: OptionType.numeric,
                min: 0,
                max: 100,
                default: 90,
                integer: true,
                description: "Compression quality for JPEG images (0-100)"
            });
        }

        // Thinking options: 3.0+ (same as non-image counterparts)
        const thinkingOptions = isGemini3OrLater ? getGeminiThinkingOptionItems(model) : [];

        return {
            _option_id: "vertexai-gemini",
            options: [
                ...max_tokens,
                ...commonOptions,
                ...imageOptions,
                ...thinkingOptions,
            ]
        };
    }
    const max_tokens_limit = getGeminiMaxTokensLimit(model);
    const excludeOptions = ["max_tokens"];
    const commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));

    const max_tokens: ModelOptionInfoItem[] = [{
        name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
        integer: true, step: 200, description: "The maximum number of tokens to generate"
    }];

    const seedOption: ModelOptionInfoItem = {
        name: SharedOptions.seed, type: OptionType.numeric, integer: true, description: "The seed for the generation, useful for reproducibility"
    };

    if (isGeminiModelVersionGte(model, "3.0")) {
        const flexOptions: ModelOptionInfoItem[] = isFlexSupportedGeminiModel(model) ? [{
            name: "flex",
            type: OptionType.boolean,
            default: false,
            description: "Use Flex processing tier for cost-efficient, batch-style execution with relaxed latency.",
        }] : [];
        return {
            _option_id: "vertexai-gemini",
            options: [
                ...max_tokens,
                ...commonOptions,
                seedOption,
                ...getGeminiThinkingOptionItems(model),
                ...flexOptions,
            ]
        };
    }

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
    const commonOptions = textOptionsFallback.options.filter((option) => !excludeOptions.includes(option.name));
    const max_tokens: ModelOptionInfoItem[] = [{
        name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
        integer: true, step: 200, description: "The maximum number of tokens to generate"
    }];

    const claudeCacheOptions: ModelOptionInfoItem[] = [
        {
            name: "cache_enabled",
            type: OptionType.boolean,
            default: false,
            description: "Enable prompt caching. Injects cache breakpoints at the system prompt, tools, and conversation pivot.",
        },
    ];
    const claudeCacheTtlOptions: ModelOptionInfoItem[] = (option as VertexAIClaudeOptions)?.cache_enabled ? [
        {
            name: "cache_ttl",
            type: OptionType.enum,
            enum: { "5 minutes (default)": "5m", "1 hour": "1h" },
            default: "5m",
            description: "TTL for cache breakpoints. '1h' requires extended caching to be enabled on your account.",
        }
    ] : [];

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
                ...claudeCacheOptions,
                ...claudeCacheTtlOptions,
            ]
        };
    }
    return {
        _option_id: "vertexai-claude",
        options: [
            ...max_tokens,
            ...commonOptions,
            ...claudeCacheOptions,
            ...claudeCacheTtlOptions,
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
    if (model.includes("image")) {
        return isGeminiModelVersionGte(model, "2.5") ? 32768 : 8192;
    }
    if (model.includes("thinking") || isGeminiModelVersionGte(model, "2.5")) {
        return 65535; // API upper bound is exclusive
    }
    if (model.includes("ultra") || model.includes("vision")) {
        return 2048;
    }
    return 8192;
}

// Delegate to provider-agnostic limits,
// override only where VertexAI supports extended output (128K for 3.7)
function getClaudeMaxTokensLimit(model: string): number {
    if (model.includes('-3-7')) return 128000;
    return getMaxOutputTokens(model);
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
