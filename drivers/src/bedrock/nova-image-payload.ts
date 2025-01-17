import { ImageExecutionOptions } from "@llumiverse/core";
import { NovaMessage, NovaMessagesPrompt } from "@llumiverse/core/formatters";

async function textToImagePayload(prompt: NovaMessagesPrompt, options: ImageExecutionOptions): Promise<NovaTextToImagePayload> {

    const textMessages = prompt.messages.map(m => m.content.map(c => c.text)).flat();
    let text = textMessages.join("\n\n");
    text += prompt.system ? "\n\n\nIMPORTANT: " + prompt.system?.map(m => m.text).join("\n\n") : '';

    const conditionImage = () => {
        const img = getFirstImageFromPrompt(prompt.messages);
        if (img && options.model_options.input_image_use === "inspiration") {
            return getFirstImageFromPrompt(prompt.messages);
        }
        return undefined;
    }


    let payload: NovaTextToImagePayload = {
        taskType: NovaImageGenerationTaskType.TEXT_IMAGE,
        imageGenerationConfig: {
            quality: options.model_options.quality === "high" ? "premium" : "standard",
            width: options.model_options.width,
            height: options.model_options.height,
        },
        textToImageParams: {
            text: text,
            conditionImage: conditionImage()?.source.bytes
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

async function imageVariationPayload(prompt: NovaMessagesPrompt, options: ImageExecutionOptions): Promise<NovaImageVariationPayload> {

    const text = prompt.messages.map(m => m.content).join("\n\n");
    const images = getAllImagesFromPrompt(prompt.messages);

    let payload: NovaImageVariationPayload = {
        taskType: NovaImageGenerationTaskType.IMAGE_VARIATION,
        imageGenerationConfig: {
            quality: options.model_options.quality === "high" ? "premium" : "standard",
            width: options.model_options.width,
            height: options.model_options.height,
        },
        imageVariationParams: {
            images: images ?? [],
            text: text,
        }
    }

    return payload;

}


export function formatNovaImageGenerationPayload(taskType: NovaImageGenerationTaskType, prompt: NovaMessagesPrompt, options: ImageExecutionOptions) {

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