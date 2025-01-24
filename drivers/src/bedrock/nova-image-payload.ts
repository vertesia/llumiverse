import { ExecutionOptions } from "@llumiverse/core";
import { NovaMessage, NovaMessagesPrompt } from "@llumiverse/core/formatters";

async function textToImagePayload(prompt: NovaMessagesPrompt, options: ExecutionOptions): Promise<NovaTextToImagePayload> {
    if (options.model_options?._option_id !== "bedrock-nova-canvas") {
        throw new Error("Invalid model options");
    }

    const textMessages = prompt.messages.map(m => m.content.map(c => c.text)).flat();
    let text = textMessages.join("\n\n");
    text += prompt.system ? "\n\n\nIMPORTANT: " + prompt.system?.map(m => m.text).join("\n\n") : '';

    const conditionImage = (conditionImage: boolean) => {
        const img = getFirstImageFromPrompt(prompt.messages);
        if (img && conditionImage) {
            return img
        }
        return undefined;
    }

    let payload: NovaTextToImagePayload = {
        taskType: NovaImageGenerationTaskType.TEXT_IMAGE,
        imageGenerationConfig: {
            quality: options.model_options?.quality,
            width: options.model_options?.width,
            height: options.model_options?.height,
            numberOfImages: options.model_options?.numberOfImages,
            seed: options.model_options?.seed,
            cfgScale: options.model_options?.cfgScale,
        },
        textToImageParams: {
            text: text,
            conditionImage: conditionImage(options.model_options.controlMode ? true : false)?.source.bytes,
            controlMode: options.model_options.controlMode,
            controlStrength: options.model_options.controlStrength,
        }
    }

    return payload;
}

function getFirstImageFromPrompt(prompt: NovaMessage[]) {

    const msgImage = prompt.find(m => m.content.find(c => c.image));
    if (!msgImage) {
        return undefined;
    }

    return msgImage.content.find(c => c.image)?.image;

}

//@ts-ignore
function getAllImagesFromPrompt(prompt: NovaMessage[]) {

    const contentMsg = prompt.filter(m => m.content).map(m => m.content).flat();
    const imgParts = contentMsg.filter(c => c.image);
    if (!imgParts?.length) {
        return undefined;
    }

    const images = imgParts.map(i => i.image).filter(i => i?.source?.bytes).map(i => i!.source.bytes);
    return images;
}

async function imageVariationPayload(prompt: NovaMessagesPrompt, options: ExecutionOptions): Promise<NovaImageVariationPayload> {
    if (options.model_options?._option_id !== "bedrock-nova-canvas") {
        throw new Error("Invalid model options");
    }

    const text = prompt.messages.map(m => m.content).join("\n\n");
    const images = getAllImagesFromPrompt(prompt.messages);

    let payload: NovaImageVariationPayload = {
        taskType: NovaImageGenerationTaskType.IMAGE_VARIATION,
        imageGenerationConfig: {
            quality: options.model_options?.quality,
            width: options.model_options?.width,
            height: options.model_options?.height,
            numberOfImages: options.model_options?.numberOfImages,
            seed: options.model_options?.seed,
            cfgScale: options.model_options?.cfgScale,
        },
        imageVariationParams: {
            images: images ?? [],
            text: text,
            similarityStrength: options.model_options?.similarityStrength,
        }
    }

    return payload;

}


export function formatNovaImageGenerationPayload(taskType: string, prompt: NovaMessagesPrompt, options: ExecutionOptions) {

    switch (taskType) {
        case NovaImageGenerationTaskType.TEXT_IMAGE:
            return textToImagePayload(prompt, options);
        case NovaImageGenerationTaskType.IMAGE_VARIATION:
            return imageVariationPayload(prompt, options);
        default:
            throw new Error("Task type not supported");
    }

}



export interface InvokeModelPayloadBase {

    taskType: NovaImageGenerationTaskType;
    imageGenerationConfig: {
        width?: number;
        height?: number;
        quality: "standard" | "premium";
        cfgScale?: number;
        seed?: number;
        numberOfImages?: number;
    };
}

export interface NovaTextToImagePayload extends InvokeModelPayloadBase {

    textToImageParams: {
        conditionImage?: string;
        controlMode?: "CANNY_EDGE" | "SEGMENTATION";
        controlStrength?: number;
        text: string;
        negativeText?: string;
    };

}


export interface NovaImageVariationPayload extends InvokeModelPayloadBase {
    imageVariationParams: {
        images: string[]  //(list of Base64 encoded images),
        similarityStrength?: number,
        text?: string,
        negativeText?: string
    }
}


export enum NovaImageGenerationTaskType {
    TEXT_IMAGE = "TEXT_IMAGE",
    COLOR_GUIDED_GENERATION = "COLOR_GUIDED_GENERATION",
    IMAGE_VARIATION = "IMAGE_VARIATION",
    INPAINTING = "INPAINTING",
    OUTPAINTING = "OUTPAINTING",
    BACKGROUND_REMOVAL = "BACKGROUND_REMOVAL",
}