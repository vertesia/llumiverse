import { AbstractDriver, ImageExecutionOptions, Modalities } from "@llumiverse/core";
import "dotenv/config";
import fs from "fs";
import { describe, expect, test } from "vitest";
import { BedrockDriver } from "../src";
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from "../src/bedrock/nova-image-payload";
import { testPrompt_imageVariations, testPrompt_textToImage, testPrompt_textToImageGuidance } from "./samples";

const TIMEOUT = 120 * 1000;


interface TestDriver {
    driver: AbstractDriver;
    models: string[];
    name: string;
}

const drivers: TestDriver[] = [];

if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: "bedrock",
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        models: [
            "amazon.nova-canvas-v1:0",
        ]
    }
    )
} else {
    console.warn("Bedrock tests are skipped: BEDROCK_REGION environment variable is not set");
}

//TODO: Add vertexai imagen tests

describe.concurrent.each(drivers)("Driver $name", ({ name, driver, models }) => {

    
    test("generate a prompt Nova canvas", async () => {
        const options: ImageExecutionOptions = {
            model: "amazon.nova-canvas-v1:0",
            output_modality: Modalities.image,
            model_options: {
                generation_type: "text-to-image",
                input_image_use: "none",
            }
        }
        const res = await formatNovaImageGenerationPayload(NovaImageGenerationTaskType.TEXT_IMAGE, testPrompt_textToImage, options)
        expect(res).toBeDefined();
        expect(res).toHaveProperty("textToImageParams");
        expect(res).toHaveProperty("taskType");
        console.log(res);
    });

    test.each(models)(`${name}: text to image generation`, {timeout: 300*1000}, async (model) => {

        console.log(`Testing model ${model}`);

        const options: ImageExecutionOptions = {
            model: model,
            output_modality: Modalities.image,
            model_options: {
                generation_type: "text-to-image",
                input_image_use: "none",
            }
        }

        const res = await driver.requestImageGeneration(testPrompt_textToImage, options);
        expect(res).toBeDefined();
        expect(res.result).toHaveProperty("images");
        expect(res.result.images).toHaveLength(1);
        saveImagesToOutput(res.result.images, `text-to-image-${model}`);
    });

    test.each(models)(`${name}: text to image guidance`, {timeout: 300*1000}, async (model) => {

        console.log(`Testing model ${model}`);

        const options: ImageExecutionOptions = {
            model: model,
            output_modality: Modalities.image,
            model_options: {
                generation_type: "text-to-image",
                input_image_use: "variation",
            }
        }

        const res = await driver.requestImageGeneration(testPrompt_textToImageGuidance, options);
        expect(res).toBeDefined();
        expect(res.result).toHaveProperty("images");
        expect(res.result.images).toHaveLength(1);
        saveImagesToOutput(res.result.images, `text-to-image-guidance-${model}`);
    });


    test.each(models)(`${name}: text to image variation`, {timeout: 300*1000}, async (model) => {

        const options: ImageExecutionOptions = {
            model: model,
            output_modality: Modalities.image,
            model_options: {
                generation_type: "text-to-image",
                input_image_use: "inspiration",
                quality: "standard",
            }
        }

        const res = await driver.requestImageGeneration(testPrompt_imageVariations, options);
        expect(res).toBeDefined();
        expect(res.result).toHaveProperty("images");
        expect(res.result.images).toHaveLength(1);
        saveImagesToOutput(res.result.images, `text-to-image-variation-${model}`);

    });

});

function saveImagesToOutput(images: string[] = [], name: string) {
    if (!fs.existsSync("output")) {
        fs.mkdirSync("output");
    }
    if (images.length === 0) {
        return
    }
    images.forEach((image, i) => {
        const filename = `output/${name}-${i}.png`;
        console.log(`Saving image to ${filename}`);
        fs.writeFileSync(filename, image, 'base64');
    });
}