import {
    AIModel, Completion, ExecutionOptions,
    ModelType, PromptRole, PromptSegment, readStreamAsBase64, ImagenOptions
} from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";

// Import the helper module for converting arbitrary protobuf.Value objects
import { protos, helpers } from '@google-cloud/aiplatform';

interface ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_RAW" | "REFERENCE_TYPE_MASK" | "REFERENCE_TYPE_SUBJECT" |
    "REFERENCE_TYPE_CONTROL" | "REFERENCE_TYPE_STYLE";
    referenceId: number;
    referenceImage: {
        bytesBase64Encoded: string; //10MB max
    }
}

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

interface ImagenReferenceRaw extends ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_RAW";
}

interface ImagenReferenceMask extends Omit<ImagenBaseReference, "referenceImage"> {
    referenceType: "REFERENCE_TYPE_MASK";
    maskImageConfig: {
        maskMode?: ImagenMaskMode;
        maskClasses?: number[]; //Used for MASK_MODE_SEMANTIC, based on https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api-customization#segment-ids
        dilation?: number; //Recommendation depends on mode: Inpaint: 0.01, BGSwap: 0.0, Outpaint: 0.01-0.03
    }
    referenceImage?: {  //Only used for MASK_MODE_USER_PROVIDED
        bytesBase64Encoded: string; //10MB max
    }
}

interface ImagenReferenceSubject extends ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_SUBJECT";
    subjectImageConfig: {
        subjectDescription: string;
        subjectType: "SUBJECT_TYPE_PERSON" | "SUBJECT_TYPE_ANIMAL" | "SUBJECT_TYPE_PRODUCT" | "SUBJECT_TYPE_DEFAULT";
    }
}

interface ImagenReferenceControl extends ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_CONTROL";
    controlImageConfig: {
        controlType: "CONTROL_TYPE_FACE_MESH" | "CONTROL_TYPE_CANNY" | "CONTROL_TYPE_SCRIBBLE";
        enableControlImageComputation?: boolean; //If true, the model will compute the control image
    }
}

interface ImagenReferenceStyle extends ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_STYLE";
    styleImageConfig: {
        styleDescription?: string;
    }
}

type ImagenMessage = ImagenReferenceRaw | ImagenReferenceMask | ImagenReferenceSubject | ImagenReferenceControl | ImagenReferenceStyle;

export interface ImagenPrompt {
    prompt: string;
    referenceImages?: ImagenMessage[];
    subjectDescription?: string; //Used for image customization to describe in the reference image
    negativePrompt?: string; //Used for negative prompts
}

function getImagenParameters(taskType: string, options: ImagenOptions) {
    const commonParameters = {
        sampleCount: options?.number_of_images,
        seed: options?.seed,
        safetySetting: options?.safety_setting,
        personGeneration: options?.person_generation,
        negativePrompt: taskType ? undefined : "", //Filled in later from the prompt
        //TODO: Add more safety and prompt rejection information
        //includeSafetyAttributes: true,
        //includeRaiReason: true,
    };
    switch (taskType) {
        case ImagenTaskType.EDIT_MODE_INPAINT_REMOVAL:
            return {
                ...commonParameters,
                editMode: "EDIT_MODE_INPAINT_REMOVAL",
                editConfig: {
                    baseSteps: options?.edit_steps,
                },
            }
        case ImagenTaskType.EDIT_MODE_INPAINT_INSERTION:
            return {
                ...commonParameters,
                editMode: "EDIT_MODE_INPAINT_INSERTION",
                editConfig: {
                    baseSteps: options?.edit_steps,
                },
            }
        case ImagenTaskType.EDIT_MODE_BGSWAP:
            return {
                ...commonParameters,
                editMode: "EDIT_MODE_BGSWAP",
                editConfig: {
                    baseSteps: options?.edit_steps,
                },
            }
        case ImagenTaskType.EDIT_MODE_OUTPAINT:
            return {
                ...commonParameters,
                editMode: "EDIT_MODE_OUTPAINT",
                editConfig: {
                    baseSteps: options?.edit_steps,
                },
            }
        case ImagenTaskType.TEXT_IMAGE:
            return {
                ...commonParameters,
                // You can't use a seed value and watermark at the same time.
                addWatermark: options?.add_watermark,
                aspectRatio: options?.aspect_ratio,
                enhancePrompt: options?.enhance_prompt,
            };
        case ImagenTaskType.CUSTOMIZATION_SUBJECT:
        case ImagenTaskType.CUSTOMIZATION_CONTROLLED:
        case ImagenTaskType.CUSTOMIZATION_INSTRUCT:
        case ImagenTaskType.CUSTOMIZATION_STYLE:
            return {
                ...commonParameters,
            }
        default:
            throw new Error("Task type not supported");
    }
}

export class ImagenModelDefinition {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Image,
            can_stream: false,
        };
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: ExecutionOptions): Promise<ImagenPrompt> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const prompt: ImagenPrompt = {
            prompt: "",
        }

        //Collect text prompts, Imagen does not support roles, so everything gets merged together 
        // however we still respect our typical pattern. System First, Safety Last.
        const system: string[] = [];
        const user: string[] = [];
        const safety: string[] = [];
        const negative: string[] = [];

        const mask_mode = (options.model_options as ImagenOptions)?.mask_mode;
        const imagenOptions = options.model_options as ImagenOptions;

        for (const msg of segments) {
            if (msg.role === PromptRole.safety) {
                safety.push(msg.content);
            } else if (msg.role === PromptRole.system) {
                system.push(msg.content);
            } else if (msg.role === PromptRole.negative) {
                negative.push(msg.content);
            } else {
                //Everything else is assumed to be user or user adjacent.
                user.push(msg.content);
            }
            if (msg.files) {
                //Get images from messages
                if (!prompt.referenceImages) {
                    prompt.referenceImages = [];
                }

                //Always required, but only used by customisation. 
                //Each ref ID refers to a single "reference", i.e. object. To provide multiple images of a single ref,
                //include multiple images in one prompt.
                const refId = prompt.referenceImages.length + 1;
                for (const img of msg.files) {
                    if (img.mime_type?.includes("image")) {
                        if (msg.role !== PromptRole.mask) {
                            //Editing based mode requires a reference image
                            if (imagenOptions?.edit_mode?.includes("EDIT_MODE")) {
                                prompt.referenceImages.push({
                                    referenceType: "REFERENCE_TYPE_RAW",
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    }
                                });
                                //If mask is auto-generated, add a mask reference
                                if (mask_mode !== ImagenMaskMode.MASK_MODE_USER_PROVIDED) {
                                    prompt.referenceImages.push({
                                        referenceType: "REFERENCE_TYPE_MASK",
                                        referenceId: refId,
                                        maskImageConfig: {
                                            maskMode: mask_mode,
                                            dilation: imagenOptions?.mask_dilation,
                                        }
                                    });
                                }
                            }
                            else if ((options.model_options as ImagenOptions)?.edit_mode === ImagenTaskType.CUSTOMIZATION_SUBJECT) {
                                //First image is always the control image
                                if (refId == 1) {
                                    //Customization subject mode requires a control image
                                    prompt.referenceImages.push({
                                        referenceType: "REFERENCE_TYPE_CONTROL",
                                        referenceId: refId,
                                        referenceImage: {
                                            bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                        },
                                        controlImageConfig: {
                                            controlType: imagenOptions?.controlType === "CONTROL_TYPE_FACE_MESH" ? "CONTROL_TYPE_FACE_MESH" : "CONTROL_TYPE_CANNY",
                                            enableControlImageComputation: imagenOptions?.controlImageComputation,
                                        }
                                    });
                                } else {
                                    // Subject images
                                    prompt.referenceImages.push({
                                        referenceType: "REFERENCE_TYPE_SUBJECT",
                                        referenceId: refId,
                                        referenceImage: {
                                            bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                        },
                                        subjectImageConfig: {
                                            subjectDescription: prompt.subjectDescription ?? msg.content,
                                            subjectType: imagenOptions?.subjectType ?? "SUBJECT_TYPE_DEFAULT",
                                        }
                                    });
                                }
                            } else if ((options.model_options as ImagenOptions)?.edit_mode === ImagenTaskType.CUSTOMIZATION_STYLE) {
                                // Style images
                                prompt.referenceImages.push({
                                    referenceType: "REFERENCE_TYPE_STYLE",
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    },
                                    styleImageConfig: {
                                        styleDescription: prompt.subjectDescription ?? msg.content,
                                    }
                                });
                            } else if ((options.model_options as ImagenOptions)?.edit_mode === ImagenTaskType.CUSTOMIZATION_CONTROLLED) {
                                // Control images
                                prompt.referenceImages.push({
                                    referenceType: "REFERENCE_TYPE_CONTROL",
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    },
                                    controlImageConfig: {
                                        controlType: imagenOptions?.controlType === "CONTROL_TYPE_FACE_MESH" ? "CONTROL_TYPE_FACE_MESH" : "CONTROL_TYPE_CANNY",
                                        enableControlImageComputation: imagenOptions?.controlImageComputation,
                                    }
                                });
                            } else if ((options.model_options as ImagenOptions)?.edit_mode === ImagenTaskType.CUSTOMIZATION_INSTRUCT) {
                                // Control images
                                prompt.referenceImages.push({
                                    referenceType: "REFERENCE_TYPE_RAW",
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    },
                                });
                            }
                        }
                        //If mask is user-provided, add a mask reference
                        if (msg.role === PromptRole.mask && mask_mode === ImagenMaskMode.MASK_MODE_USER_PROVIDED) {
                            prompt.referenceImages.push({
                                referenceType: "REFERENCE_TYPE_MASK",
                                referenceId: refId,
                                referenceImage: {
                                    bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                },
                                maskImageConfig: {
                                    maskMode: mask_mode,
                                    dilation: imagenOptions?.mask_dilation,
                                }
                            });
                        }
                    }
                }
            }
        }

        //Extract the text from the segments
        prompt.prompt += [system.join("\n\n"), user.join("\n\n"), safety.join("\n\n")].join("\n\n");

        //Negative prompt
        if (negative.length > 0) {
            prompt.negativePrompt = negative.join(", ");
        }

        console.log(prompt);

        return prompt
    }

    async requestImageGeneration(driver: VertexAIDriver, prompt: ImagenPrompt, options: ExecutionOptions): Promise<Completion> {
        if (options.model_options?._option_id !== "vertexai-imagen") {
            driver.logger.warn({ options: options.model_options }, "Invalid model options");
        }
        options.model_options = options.model_options as ImagenOptions | undefined;

        const taskType: string = options.model_options?.edit_mode ?? ImagenTaskType.TEXT_IMAGE;

        driver.logger.info("Task type: " + taskType);

        const modelName = options.model.split("/").pop() ?? '';

        // Configure the parent resource
        // TODO: make location configurable, fixed to us-central1 for now
        const endpoint = `projects/${driver.options.project}/locations/us-central1/publishers/google/models/${modelName}`;

        const instanceValue = helpers.toValue(prompt);
        if (!instanceValue) {
            throw new Error('No instance value found');
        }
        const instances = [instanceValue];

        let parameter: any = getImagenParameters(taskType, options.model_options ?? { _option_id: "vertexai-imagen" });
        parameter.negativePrompt = prompt.negativePrompt ?? undefined;

        const numberOfImages = options.model_options?.number_of_images ?? 1;

        // Remove all undefined values
        parameter = Object.fromEntries(
            Object.entries(parameter).filter(([_, v]) => v !== undefined)
        ) as any;

        const parameters = helpers.toValue(parameter);

        const request: protos.google.cloud.aiplatform.v1.IPredictRequest = {
            endpoint,
            instances,
            parameters,
        };

        const client = await driver.getImagenClient();

        // Predict request
        const [response] = await client.predict(request, { timeout: 120000 * numberOfImages }); //Extended timeout for image generation
        const predictions = response.predictions;

        if (!predictions) {
            throw new Error('No predictions found');
        }

        // Extract base64 encoded images from predictions
        const images: string[] = predictions.map(prediction =>
            prediction.structValue?.fields?.bytesBase64Encoded?.stringValue ?? ''
        );

        return {
            result: images.map(image => ({
                type: "image" as const,
                value: image
            })),
        };
    }
}