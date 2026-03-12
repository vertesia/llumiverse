import { DataSource, PromptRole, PromptSegment, readStreamAsBase64 } from "@llumiverse/core";
import { NovaMessagesPrompt } from "@llumiverse/core/formatters";
import { createReadStream } from "fs";
import { JSONSchema } from "@llumiverse/core";
import { createReadableStreamFromReadable } from "node-web-stream-adapters";
import { basename, dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// 512x512 JPEG fallback — within Bedrock's [320, 4096] range
const LOCAL_FALLBACK_IMAGE = resolve(__dirname, "test_image_1.jpg");

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
        return `file://${this.file}`;
    }
    async getStream(): Promise<ReadableStream<string | Uint8Array>> {
        const stream = createReadStream(this.file);
        return createReadableStreamFromReadable(stream);
    }
}

class ImageUrlSource implements DataSource {
    constructor(public url: string, public mime_type: string = "image/jpeg") {
    }
    get name() {
        return basename(this.url);
    }
    async getURL(): Promise<string> {
        return this.url;
    }
    async getStream(): Promise<ReadableStream<string | Uint8Array>> {
        return fetchWithFallback(this.url);
    }
}

/** Fetch with timeout and fallback to a local test image */
async function fetchWithFallback(url: string): Promise<ReadableStream<Uint8Array>> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);
    try {
        const res = await fetch(url, { signal: controller.signal });
        clearTimeout(timeout);
        if (!res.ok || !res.body) {
            throw new Error(`HTTP ${res.status}`);
        }
        return res.body;
    } catch (err) {
        clearTimeout(timeout);
        console.warn(`Failed to fetch ${url} (${err}), falling back to local test image`);
        return createReadableStreamFromReadable(createReadStream(LOCAL_FALLBACK_IMAGE));
    }
}

export const testPrompt_describeImage: PromptSegment[] = [
    {
        content: "You are a lab assistant analysing images of animals, then tag the images with accurate description of the animal shown in the picture.",
        role: PromptRole.user,
        files: [new ImageUrlSource("https://upload.wikimedia.org/wikipedia/commons/b/b2/WhiteCat.jpg")]
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
                source: { bytes: await getImageAsBase64(new ImageUrlSource("https://upload.wikimedia.org/wikipedia/commons/b/b2/WhiteCat.jpg")) }
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
                source: { bytes: await getImageAsBase64(new ImageUrlSource("https://upload.wikimedia.org/wikipedia/commons/b/b2/WhiteCat.jpg")), }
            }
        },
        {
            image: {
                format: "jpeg",
                source: { bytes: await getImageAsBase64(new ImageUrlSource("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg")), }
            }
        }
        ]
    }]
}

async function getImageAsBase64(source: DataSource) {
    const stream = await source.getStream();
    return readStreamAsBase64(stream);
}
