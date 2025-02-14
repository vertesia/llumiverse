import { ModelOptionsInfo, ModelOptions, OptionType, ModelOptionInfoItem } from "../types.js";
import { textOptionsFallback } from "../options.js";

// Union type of all Bedrock options
export type BedrockOptions = NovaCanvasOptions;

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

export function getBedrockOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    if (model.includes("canvas")) {
        const tasktypeList: ModelOptionInfoItem = {
            name: "taskType",
            type: OptionType.enum,
            enum: {
                "Text-To-Image": "TEXT_IMAGE",
                "Text-To-Image-with-Image-Conditioning": "TEXT_IMAGE_WITH_IMAGE_CONDITIONING",
                "Color-Guided-Generation": "COLOR_GUIDED_GENERATION",
                "Image-Variation": "IMAGE_VARIATION",
            //    "Inpainting": "INPAINTING",    Not implemented yet
            //    "Outpainting": "OUTPAINTING",
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

        switch ((option as BedrockOptions)?.taskType ?? "TEXT_IMAGE") {
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
                tasktypeList,
                ...otherOptions,
                ...dependentOptions,
            ]
        };
    }
    return textOptionsFallback;
}