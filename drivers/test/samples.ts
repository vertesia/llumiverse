import { DataSource, PromptRole, PromptSegment, readStreamAsBase64 } from "@llumiverse/core";
import { NovaMessagesPrompt } from "@llumiverse/core/formatters";
import { createReadStream } from "fs";
import { JSONSchema } from "@llumiverse/core";
import { createReadableStreamFromReadable } from "node-web-stream-adapters";
import { basename, dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Local test fixture images (512x512 JPEGs, within Bedrock's [320, 4096] range)
const TEST_IMAGE_1 = resolve(__dirname, "test_image_1.jpg");
const TEST_IMAGE_2 = resolve(__dirname, "test_image_2.jpg");

export const testPrompt_color: PromptSegment[] = [
    {
        role: PromptRole.user,
        content: "What color is the sky?"
    }
]

//json schema with 2 properties object and color
export const testSchema_color: JSONSchema = {
    type: "object",
    properties: {
        color: {
            type: "string"
        }
    }
}

class ImageSource implements DataSource {
    file: string;
    mime_type: "image/jpeg" = "image/jpeg";
    constructor(file: string) {
        this.file = resolve(file);
    }
    get name() {
        return basename(this.file);
    }
    async getURL(): Promise<string> {
        throw new Error("Method not implemented.");
    }
    async getStream(): Promise<ReadableStream<string | Uint8Array>> {
        const stream = createReadStream(this.file);
        return createReadableStreamFromReadable(stream);
    }
}

export const testPrompt_describeImage: PromptSegment[] = [
    {
        content: "You are a lab assistant analysing images of animals, then tag the images with accurate description of the animal shown in the picture.",
        role: PromptRole.user,
        files: [new ImageSource(TEST_IMAGE_1)]
    }
]

export const testSchema_animalDescription: JSONSchema =
{
    type: "object",
    properties: {
        name: {
            type: "string"
        },
        type: {
            type: "string"
        },
        species: {
            type: "string"
        },
        characteristics: {
            type: "array",
            items: {
                type: "string"
            }
        }
    }
}

export const testPrompt_textToImage: NovaMessagesPrompt =
{
    messages: [{
        role: PromptRole.user,
        content: [{
            text: "A blue sky with a purple unicorn flying"
        }]
    }]
}

export const testPrompt_textToImageGuidance: NovaMessagesPrompt =
{
    messages: [{
        role: PromptRole.user,
        content: [{
            text: "A blue sky with a purple unicorn flying"
        },
        {
            image: {
                format: "jpeg",
                source: { bytes: await getImageAsBase64(new ImageSource(TEST_IMAGE_1)) }
            }
        }
        ]
    }]
}

export const testPrompt_imageVariations: NovaMessagesPrompt =
{
    messages: [{
        role: PromptRole.user,
        content: [{
            text: "A purple cat in from of a cathedral"
        },
        {
            image: {
                format: "jpeg",
                source: { bytes: await getImageAsBase64(new ImageSource(TEST_IMAGE_1)), }
            }
        },
        {
            image: {
                format: "jpeg",
                source: { bytes: await getImageAsBase64(new ImageSource(TEST_IMAGE_2)), }
            }
        }
        ]
    }]
}

async function getImageAsBase64(source: DataSource) {
    const stream = await source.getStream();
    return readStreamAsBase64(stream);
}
