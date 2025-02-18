import { ExecutionOptions } from "@llumiverse/core";
import { NovaMessage, NovaMessagesPrompt } from "@llumiverse/core/formatters";

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

    const payload: NovaTextToImagePayload = {
        taskType: NovaImageGenerationTaskType.TEXT_IMAGE,   // Always TEXT_IMAGE, as TEXT_IMAGE_WITH_IMAGE_CONDITIONING is only an internal marker.
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

async function imageVariationPayload(prompt: NovaMessagesPrompt, options: ExecutionOptions): Promise<NovaImageVariationPayload> {
    if (options.model_options?._option_id !== "bedrock-nova-canvas") {
        throw new Error("Invalid model options");
    }

    const text = prompt.messages.map(m => m.content).join("\n\n");
    const images = getAllImagesFromPrompt(prompt.messages);

    const payload: NovaImageVariationPayload = {
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

async function colorGuidedGenerationPayload(prompt: NovaMessagesPrompt, options: ExecutionOptions): Promise<NovaColorGuidedGenerationPayload> {
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

    const payload: NovaColorGuidedGenerationPayload = {
        taskType: NovaImageGenerationTaskType.COLOR_GUIDED_GENERATION,
        imageGenerationConfig: {
            quality: options.model_options?.quality,
            width: options.model_options?.width,
            height: options.model_options?.height,
            numberOfImages: options.model_options?.numberOfImages,
            seed: options.model_options?.seed,
            cfgScale: options.model_options?.cfgScale,
        },
        colorGuidedGenerationParams: {
            colors: options.model_options.colors ?? [],
            text: text,
            referenceImage: conditionImage(options.model_options.controlMode ? true : false)?.source.bytes,
        }
    }

    return payload;
}

async function backgroundRemovalPayload(prompt: NovaMessagesPrompt, options: ExecutionOptions): Promise<NovaBackgroundRemovalPayload> {
    if (options.model_options?._option_id !== "bedrock-nova-canvas") {
        throw new Error("Invalid model options");
    }

    const image = getFirstImageFromPrompt(prompt.messages);
    if (!image?.source.bytes) {
        throw new Error("No image found in prompt");
    }

    const payload: NovaBackgroundRemovalPayload = {
        taskType: NovaImageGenerationTaskType.BACKGROUND_REMOVAL,
        backgroundRemovalParams: {
            image: image.source.bytes
        }
    }
    console.log(payload)

    return payload;
}

export function formatNovaImageGenerationPayload(taskType: string, prompt: NovaMessagesPrompt, options: ExecutionOptions) {

    switch (taskType) {
        case NovaImageGenerationTaskType.TEXT_IMAGE:
            return textToImagePayload(prompt, options);
        case NovaImageGenerationTaskType.TEXT_IMAGE_WITH_IMAGE_CONDITIONING:
            return textToImagePayload(prompt, options);
        case NovaImageGenerationTaskType.COLOR_GUIDED_GENERATION:
            return colorGuidedGenerationPayload(prompt, options);
        case NovaImageGenerationTaskType.IMAGE_VARIATION:
            return imageVariationPayload(prompt, options);
        case NovaImageGenerationTaskType.INPAINTING:
        //   return inpaintingPayload(prompt, options);    Needs mask prompt support
        case NovaImageGenerationTaskType.OUTPAINTING:
        //   return outpaintingPayload(prompt, options);
        case NovaImageGenerationTaskType.BACKGROUND_REMOVAL:
            return backgroundRemovalPayload(prompt, options);
        default:
            throw new Error("Task type not supported");
    }

}

export interface InvokeModelPayloadBase {
    taskType: NovaImageGenerationTaskType;
    imageGenerationConfig: {
        width?: number;
        height?: number;
        quality?: "standard" | "premium";
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

export interface NovaColorGuidedGenerationPayload extends InvokeModelPayloadBase {
    colorGuidedGenerationParams: {
        colors: string[]; //(list of hex color values),
        text: string;
        referenceImage?: string;
        negativeText?: string;
    }
}

export interface NovaInpaintingPayload extends InvokeModelPayloadBase {
    inPaintingParams: {
        image: string; //(Base64 encoded image),
        maskImage?: string; //(Base64 encoded image),
        maskPrompt?: string,
        negativeText?: string,
        text?: string,
    }
}

export interface NovaOutpaintingPayload extends InvokeModelPayloadBase {
    outPaintingParams: {
        image: string; //(Base64 encoded image),
        maskImage?: string; //(Base64 encoded image),
        maskPrompt?: string,
        negativeText?: string,
        text?: string,
        outPaintingMode: "DEFAULT" | "PRECISE";
    }
}

export interface NovaBackgroundRemovalPayload {
    taskType: NovaImageGenerationTaskType.BACKGROUND_REMOVAL;
    backgroundRemovalParams: {
        image: string //(Base64 encoded image),
    }
}

export enum NovaImageGenerationTaskType {
    TEXT_IMAGE = "TEXT_IMAGE",
    TEXT_IMAGE_WITH_IMAGE_CONDITIONING = "TEXT_IMAGE_WITH_IMAGE_CONDITIONING",
    COLOR_GUIDED_GENERATION = "COLOR_GUIDED_GENERATION",
    IMAGE_VARIATION = "IMAGE_VARIATION",
    INPAINTING = "INPAINTING",
    OUTPAINTING = "OUTPAINTING",
    BACKGROUND_REMOVAL = "BACKGROUND_REMOVAL",
}