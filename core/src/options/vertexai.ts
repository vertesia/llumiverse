import { ModelOptionsInfo, ModelOptionInfoItem, ModelOptions, OptionType, SharedOptions } from "../types.js";
import { textOptionsFallback } from "../options.js";

// Union type of all Bedrock options
export type VertexAIOptions = ImagenOptions;

export interface ImagenOptions {
    _option_id: "vertexai-imagen"

    //General and generate options
    number_of_images?: number;
    seed?: number;
    person_generation?: "dont_allow" | "allow_adults" | "allow_all";
    safety_setting?: "block_none" | "block_only_high" | "block_medium_and_above" | "block_low_and_above"; //The "off" option does not seem to work for Imagen 3, might be only for text models
    image_file_type?: "image/jpeg" | "image/png";
    jpeg_compression_quality?: number;
    aspect_ratio?: "1:1" | "4:3" | "3:4" | "16:9" | "9:16" ;
    add_watermark?: boolean;
    enhance_prompt?: boolean;

    //Capability options
    edit_mode?: "EDIT_MODE_INPAINT_REMOVAL" | "EDIT_MODE_INPAINT_INSERTION" | "EDIT_MODE_BGSWAP" | "EDIT_MODE_OUTPAINT";
    guidance_scale?: number;
    base_steps?: number;
    mask_dilation?: number;
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
                name: "person_generation", type: OptionType.enum, enum: { "Disallow the inclusion of people or faces in images": "dont_allow", "Allow generation of adults only": "allow_adults", "Allow generation of people of all ages": "allow_all" },
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
                    name: "aspect_ratio", type: OptionType.enum, enum: { "1:1": "1:1", "4:3": "4:3", "3:4": "3:4", "16:9": "16:9" ,"9:16": "9:16" },
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
                    default: "EDIT_MODE_INPAINT_REMOVAL",
                    description: "The editing mode"
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
    return textOptionsFallback;
}