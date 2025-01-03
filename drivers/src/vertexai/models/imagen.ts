import { AIModel, Completion, ImageGeneration, ImageGenExecutionOptions, Modalities, ModelType } from "@llumiverse/core";
import { VertexAIDriver } from "../index.js";

const projectId = process.env.GOOGLE_PROJECT_ID;
const location = 'us-central1';

import aiplatform from '@google-cloud/aiplatform';

// Imports the Google Cloud Prediction Service Client library
const { PredictionServiceClient } = aiplatform.v1;

// Import the helper module for converting arbitrary protobuf.Value objects
import { helpers } from '@google-cloud/aiplatform';
import { GenerateContentRequest } from "@google-cloud/vertexai";

interface IPredictRequest {
    endpoint: string;
    instances: any[];
    parameters: any;
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
}

async function textToImagePayload(prompt: GenerateContentRequest): Promise<string> {
    //At runtime, prompt is sometimes not an array.
    const textMessages = prompt.contents.map((content) => {content.parts.map((part) => part.text)});
    
    const text = textMessages.join("\n\n");

    return text;
}

export function formatImagenImageGenerationPayload(taskType: ImagenImageGenerationTaskType, prompt: GenerateContentRequest, _options: ImageGenExecutionOptions) {

    switch (taskType) {
        case ImagenImageGenerationTaskType.TEXT_IMAGE:
            return textToImagePayload(prompt);
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
        } as AIModel;
    }
    
    async requestImageGeneration(driver: VertexAIDriver, prompt: GenerateContentRequest, options: ImageGenExecutionOptions): Promise<Completion<ImageGeneration>> {
    
        if (options.output_modality !== Modalities.image) {
            throw new Error(`Image generation requires image output_modality`);
        }

        if (!options.model.includes('imagen-3.0-generate-001')) {
            throw new Error(`Model ${options.model} not supported, use imagen-3.0-generate-001`);
        }

        const taskType = () => {
            switch (options.generation_type) {
                case "text-to-image":
                    return ImagenImageGenerationTaskType.TEXT_IMAGE
                default:
                    return ImagenImageGenerationTaskType.TEXT_IMAGE
            }
        }
        driver.logger.info("Task type: " + taskType());

        if (typeof prompt === "string") {
            throw new Error("Bad prompt format");
        }

        console.log('Prompt:', JSON.stringify(prompt));

        // Configure the parent resource
        const endpoint = `projects/${projectId}/locations/${location}/publishers/google/models/imagen-3.0-generate-001`;

        const promptText = {
            prompt: await formatImagenImageGenerationPayload(taskType(), prompt, options)
        };

        console.log('Prompt Text:', JSON.stringify(promptText));

        const instanceValue = helpers.toValue(promptText);
        const instances = [instanceValue];

        const parameter = {
            sampleCount: 1,
            // You can't use a seed value and watermark at the same time.
            // seed: 100,
            // addWatermark: false,
            aspectRatio: '1:1',
            //safetyFilterLevel: 'block_some',
            //personGeneration: 'allow_adult',
        };
        const parameters = helpers.toValue(parameter);

        const request = {
            endpoint,
            instances,
            parameters,
        } as IPredictRequest;

        // Predict request
        const [response] = await predictionServiceClient.predict(request);
        const predictions = response.predictions;

        if (!predictions) {
            throw new Error('No predictions found');
        }

        console.log('Response:', JSON.stringify(response));

        // Extract base64 encoded images from predictions
        const images : string[] = predictions.map(prediction =>
            prediction.structValue?.fields?.bytesBase64Encoded.stringValue ?? ''
        );

        return {
            result: {
                images
            }
        };
    }
}