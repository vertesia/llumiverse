import { ModelOptionsInfo, ModelOptionInfoItem, ModelOptions, OptionType, SharedOptions } from "../types.js";
import { textOptionsFallback } from "../options.js";

// Union type of all Bedrock options
export type VertexAIOptions = ImagenOptions;

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
    return textOptionsFallback;
}