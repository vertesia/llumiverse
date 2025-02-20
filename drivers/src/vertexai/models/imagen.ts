import { AIModel, Completion, ImageGeneration, Modalities, ModelType, PromptOptions, PromptRole, PromptSegment, ExecutionOptions } from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";

const projectId = process.env.GOOGLE_PROJECT_ID;
const location = 'us-central1';

import aiplatform, { protos } from '@google-cloud/aiplatform';

// Imports the Google Cloud Prediction Service Client library
const { PredictionServiceClient } = aiplatform.v1;

// Import the helper module for converting arbitrary protobuf.Value objects
import { helpers } from '@google-cloud/aiplatform';
import { ImagenOptions } from "../../../../core/src/options/vertexai.js";

interface ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_RAW" | "REFERENCE_TYPE_MASK" | "REFERENCE_TYPE_SUBJECT" |
        "REFERENCE_TYPE_CONTROL" | "REFERENCE_TYPE_STYLE";
    referenceId: number;
    referenceImage: {
        bytesBase64Encoded: string; //10MB max
    }
}

interface ImagenReferenceRaw extends ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_RAW";
}

interface ImagenReferenceMask extends ImagenBaseReference {
    referenceType: "REFERENCE_TYPE_MASK";
    maskImageConfig: {
        maskMode?: "MASK_MODE_USER_PROVIDED" | "MASK_MODE_BACKGROUND" | "MASK_MODE_FOREGROUND" | "MASK_MODE_SEMANTIC";
        maskClasses?: number[]; //Used for MASK_MODE_SEMANTIC, based on https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api-customization#segment-ids
        dilation?: number; //Recommendation depends on mode: Inpaint: 0.01, BGSwap: 0.0, Outpaint: 0.01-0.03
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
}

// Specifies the location of the api endpoint
const clientOptions = {
  apiEndpoint: `${location}-aiplatform.googleapis.com`,
};

// Instantiates a client
const predictionServiceClient = new PredictionServiceClient(clientOptions);

//TODO: Add more task types
export enum ImagenImageGenerationTaskType {
    TEXT_IMAGE = "TEXT_IMAGE",
    EDIT_MODE_INPAINT_REMOVAL = "EDIT_MODE_INPAINT_REMOVAL",
    EDIT_MODE_INPAINT_INSERTION = "EDIT_MODE_INPAINT_INSERTION",
    EDIT_MODE_BGSWAP = "EDIT_MODE_BGSWAP",
    EDIT_MODE_OUTPAINT = "EDIT_MODE_OUTPAINT",
}

function getImagenParameters(taskType: string, options: ImagenOptions) {
    switch (taskType) {
        case ImagenImageGenerationTaskType.EDIT_MODE_INPAINT_REMOVAL:
        case ImagenImageGenerationTaskType.EDIT_MODE_INPAINT_INSERTION:
        case ImagenImageGenerationTaskType.EDIT_MODE_BGSWAP:
        case ImagenImageGenerationTaskType.EDIT_MODE_OUTPAINT:
        case ImagenImageGenerationTaskType.TEXT_IMAGE:
            return {
                sampleCount: options?.number_of_images,
                // You can't use a seed value and watermark at the same time.
                seed: options?.seed,
                addWatermark: options?.add_watermark,
                aspectRatio: options?.aspect_ratio,
                //negativePrompt: options.model_options.negative_prompt ?? '',
                safetySetting: options?.safety_setting,
                personGeneration: options?.person_generation,
                enhancePrompt: options?.enhance_prompt,
                //TODO: Add more safety and prompt rejection information
                //includeSafetyAttributes: true,
                //includeRaiReason: true,
            };
        default:
            throw new Error("Task type not supported");
    }
}

export class ImagenModelDefinition  {

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

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<ImagenPrompt> {
        const splits = options.model.split("/");
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const prompt: ImagenPrompt = {
            prompt: "",
        }

        //Collect text prompts, Imagen does not support roles, so everything gets merged together 
        // however we still respect our typical pattern. System First, Safety Last.
        const system: PromptSegment[] = [];
        const user: PromptSegment[] = [];
        const safety: PromptSegment[] = [];

        for (const msg of segments) {
            if (msg.role === PromptRole.safety) {
                safety.push(msg);
            } else if (msg.role === PromptRole.system) {
                system.push(msg);
            } else {
                //Everything else is assumed to be user or user adjacent.
                user.push(msg);
            }
        }

        //Extract the text from the segments
        prompt.prompt += system.map(msg => msg.content).join("\n\n") + "\n\n";
        prompt.prompt += user.map(msg => msg.content).join("\n\n") + "\n\n";
        prompt.prompt += safety.map(msg => msg.content).join("\n\n");

        return prompt
    }
    
    async requestImageGeneration(driver: VertexAIDriver, prompt: ImagenPrompt, options: ExecutionOptions): Promise<Completion<ImageGeneration>> {
        if (options.model_options?._option_id !== "vertexai-imagen") {
            driver.logger.warn("Invalid model options", options.model_options);
        }
        options.model_options = options.model_options as ImagenOptions;
        
        if (options.output_modality !== Modalities.image) {
            throw new Error(`Image generation requires image output_modality`);
        }

        const taskType : string = options.model_options.edit_mode ?? ImagenImageGenerationTaskType.TEXT_IMAGE;
        
        driver.logger.info("Task type: " + taskType);

        const modelName = options.model.split("/").pop() ?? '';

        // Configure the parent resource
        const endpoint = `projects/${projectId}/locations/${location}/publishers/google/models/${modelName}`;

        const instanceValue = helpers.toValue(prompt) ?? {};
        const instances = [instanceValue];

        const parameter = getImagenParameters(taskType, options.model_options);
        const parameters = helpers.toValue(parameter);

        const request: protos.google.cloud.aiplatform.v1.IPredictRequest = {
            endpoint,
            instances,
            parameters,
        };

        // Predict request
        const [response] = await predictionServiceClient.predict(request);
        const predictions = response.predictions;

        if (!predictions) {
            throw new Error('No predictions found');
        }

        // Extract base64 encoded images from predictions
        const images : string[] = predictions.map(prediction =>
            prediction.structValue?.fields?.bytesBase64Encoded?.stringValue ?? ''
        );

        return {
            result: {
                images
            },
        };
    }
}