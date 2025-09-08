import {
    AIModel, Completion, ExecutionOptions, ImageGeneration, Modalities,
    ModelType, PromptRole, PromptSegment, readStreamAsBase64, ImagenOptions
} from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";

// Import the helper module for converting arbitrary protobuf.Value objects
import type { GenerateImagesParameters, EditImageParameters, UpscaleImageParameters, GenerateImagesResponse, EditImageResponse, UpscaleImageResponse } from '@google/genai';
// helpers/protos removed — PredictionServiceClient fallback removed; using GenAI client only

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

    async requestImageGeneration(driver: VertexAIDriver, prompt: ImagenPrompt, options: ExecutionOptions): Promise<Completion<ImageGeneration>> {
        if (options.model_options?._option_id !== "vertexai-imagen") {
            driver.logger.warn("Invalid model options", {options: options.model_options });
        }
        options.model_options = options.model_options as ImagenOptions | undefined;

        if (options.output_modality !== Modalities.image) {
            throw new Error(`Image generation requires image output_modality`);
        }

        const taskType: string = options.model_options?.edit_mode ?? ImagenTaskType.TEXT_IMAGE;

        driver.logger.info("Task type: " + taskType);

    const modelName = options.model.split("/").pop() ?? '';

    // Configure the parent resource (legacy PredictionServiceClient endpoint removed; using GenAI client instead)

    // Use the Google GenAI client image APIs.
    const genai = driver.getGoogleGenAIClient();

    // Map our prompt.referenceImages to genai ReferenceImage types
    const referenceImages: any[] = (prompt.referenceImages ?? []).map((ref) => {
                // Common base for image bytes reference
                if (ref.referenceType === 'REFERENCE_TYPE_RAW') {
                    return {
                        referenceId: ref.referenceId,
                        referenceImage: { imageBytes: (ref as any).referenceImage?.bytesBase64Encoded },
                        referenceType: 'RAW',
                    };
                }
                if (ref.referenceType === 'REFERENCE_TYPE_MASK') {
                    const r: any = {
                        referenceId: ref.referenceId,
                        referenceType: 'MASK',
                        maskImageConfig: (ref as any).maskImageConfig ? {
                            maskMode: (ref as any).maskImageConfig.maskMode,
                            maskClasses: (ref as any).maskImageConfig.maskClasses,
                            dilation: (ref as any).maskImageConfig.dilation,
                        } : undefined,
                    };
                    if ((ref as any).referenceImage?.bytesBase64Encoded) {
                        r.referenceImage = { imageBytes: (ref as any).referenceImage.bytesBase64Encoded };
                    }
                    return r;
                }
                if (ref.referenceType === 'REFERENCE_TYPE_SUBJECT') {
                    return {
                        referenceId: ref.referenceId,
                        referenceType: 'SUBJECT',
                        subjectImageConfig: {
                            subjectDescription: (ref as any).subjectImageConfig?.subjectDescription,
                            subjectType: (ref as any).subjectImageConfig?.subjectType,
                        },
                        referenceImage: { imageBytes: (ref as any).referenceImage?.bytesBase64Encoded },
                    };
                }
                if (ref.referenceType === 'REFERENCE_TYPE_CONTROL') {
                    return {
                        referenceId: ref.referenceId,
                        referenceType: 'CONTROL',
                        controlImageConfig: {
                            controlType: (ref as any).controlImageConfig?.controlType,
                            enableControlImageComputation: (ref as any).controlImageConfig?.enableControlImageComputation,
                        },
                        referenceImage: { imageBytes: (ref as any).referenceImage?.bytesBase64Encoded },
                    };
                }
                if (ref.referenceType === 'REFERENCE_TYPE_STYLE') {
                    return {
                        referenceId: ref.referenceId,
                        referenceType: 'STYLE',
                        styleImageConfig: { styleDescription: (ref as any).styleImageConfig?.styleDescription },
                        referenceImage: { imageBytes: (ref as any).referenceImage?.bytesBase64Encoded },
                    };
                }
                return ref;
            });

            // Build config from existing parameter mapping
            const config: any = getImagenParameters(taskType, options.model_options ?? { _option_id: 'vertexai-imagen' });
            config.numberOfImages = options.model_options?.number_of_images ?? 1;
            config.negativePrompt = prompt.negativePrompt ?? undefined;

            // Remove undefined keys
            Object.keys(config).forEach(k => config[k] === undefined && delete config[k]);

        // Select API method based on task type
        let response: GenerateImagesResponse | EditImageResponse | UpscaleImageResponse | undefined;
        if (taskType.startsWith('EDIT_MODE') || (options.model_options as any)?.edit_mode?.startsWith('EDIT_MODE')) {
            // Use editImage for edit modes
            const params: EditImageParameters = {
                model: modelName,
                prompt: prompt.prompt,
                referenceImages: referenceImages as any,
                config: config as any,
            };
            response = await genai.models.editImage(params);
        } else if ((options.model_options as any)?.upscale) {
            // Upscale expects a single image input; pick first reference image imageBytes
            const first = referenceImages.find((r: any) => r.referenceImage?.imageBytes);
            if (!first) {
                throw new Error('Upscale requires a reference image');
            }
            const params: UpscaleImageParameters = {
                model: modelName,
                image: first.referenceImage,
                upscaleFactor: (options.model_options as any).upscaleFactor ?? 'x2',
                config: config as any,
            };
            response = await genai.models.upscaleImage(params);
        } else {
            // Default to generateImages
            const params: GenerateImagesParameters = {
                model: modelName,
                prompt: prompt.prompt,
                config: config as any,
            };
            response = await genai.models.generateImages(params);
        }

        const generated = (response as any)?.generatedImages ?? (response as any)?.images ?? [];
        const images: string[] = generated.map((g: any) => g.image?.imageBytes ?? '').filter(Boolean);

        if (images.length === 0) {
            throw new Error('No images returned from GenAI');
        }

        return {
            result: { images },
        };
    }
}