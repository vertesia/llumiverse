import { Base64DataSource, Driver } from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from "vitest";
import { BedrockDriver, MistralAIDriver, OpenAIDriver, VertexAIDriver, WatsonxDriver } from "../src/index.js";

const TIMEOUT = 15000;
const TEXT = "Hello";

const IMAGE_SRC = import.meta.dirname + "/hello_world.jpg";

const fs = require('fs').promises;

async function convertImageToBase64(path: string) {
    const data = await fs.readFile(path);
    return Buffer.from(data, 'binary').toString('base64');
}

const IMAGE = await convertImageToBase64(IMAGE_SRC);

function imageDataSource(): Base64DataSource {
    return new Base64DataSource("hello_world.jpg", "image/jpeg", IMAGE);
}

let vertex: VertexAIDriver | undefined;
if (process.env.GOOGLE_REGION) {
    vertex = new VertexAIDriver({
        project: process.env.GOOGLE_PROJECT_ID as string,
        region: process.env.GOOGLE_REGION as string,
    });
}

let bedrock: BedrockDriver | undefined;
if (process.env.BEDROCK_REGION) {
    bedrock = new BedrockDriver({
        region: process.env.BEDROCK_REGION as string,
    });
}

let openai: OpenAIDriver | undefined;
if (process.env.OPENAI_API_KEY) {
    openai = new OpenAIDriver({
        apiKey: process.env.OPENAI_API_KEY as string,
    });
}

let mistral: MistralAIDriver | undefined;
if (process.env.MISTRAL_API_KEY) {
    mistral = new MistralAIDriver({
        apiKey: process.env.MISTRAL_API_KEY as string,
    });
}

let watsonx: WatsonxDriver | undefined;
if (process.env.WATSONX_API_KEY) {
    watsonx = new WatsonxDriver({
        apiKey: process.env.WATSONX_API_KEY as string,
        projectId: process.env.WATSONX_PROJECT_ID as string,
        endpointUrl: process.env.WATSONX_ENDPOINT_URL as string,
    });
}

function firstValues(d: Driver) {
    return async (...inputs: Parameters<Driver["generateEmbeddings"]>[0]["inputs"]) => {
        const r = await d.generateEmbeddings({ inputs });
        return r;
    };
}

if (vertex) {
    describe('VertexAI: embeddings generation', () => {
        test('embeddings for text', async () => {
            const r = await firstValues(vertex!)({ type: "text", text: TEXT });
            expect(r.results).toHaveLength(1);
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
            expect(r.model).toBe("gemini-embedding-2");
        }, TIMEOUT);

        test('embeddings for image (multimodal)', async () => {
            const r = await firstValues(vertex!)({ type: "image", source: imageDataSource() });
            expect(r.results).toHaveLength(1);
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
        }, TIMEOUT);

        test('embeddings for batched text inputs', async () => {
            const r = await vertex!.generateEmbeddings({
                inputs: [
                    { type: "text", text: "first" },
                    { type: "text", text: "second" },
                ],
            });
            expect(r.results).toHaveLength(2);
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
            expect(r.results[1].outputs[0].values.length).toBeGreaterThan(0);
        }, TIMEOUT);
    });
}

if (bedrock) {
    describe('Bedrock: embeddings generation', () => {
        test('embeddings for text (Nova default)', async () => {
            const r = await firstValues(bedrock!)({ type: "text", text: TEXT });
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
        }, TIMEOUT);

        test('embeddings for image (Nova multimodal)', async () => {
            const r = await firstValues(bedrock!)({ type: "image", source: imageDataSource() });
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
        }, TIMEOUT);
    });
}

if (openai) {
    describe('OpenAI: embeddings generation', () => {
        test('embeddings for text', async () => {
            const r = await firstValues(openai!)({ type: "text", text: TEXT });
            expect(r.results).toHaveLength(1);
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
            expect(r.model).toBe("text-embedding-3-small");
            expect(r.usage?.input_tokens).toBeGreaterThan(0);
        }, TIMEOUT);

        test('embeddings for batched text inputs', async () => {
            const r = await openai!.generateEmbeddings({
                inputs: [
                    { type: "text", text: "alpha" },
                    { type: "text", text: "beta" },
                ],
            });
            expect(r.results).toHaveLength(2);
        }, TIMEOUT);
    });
}

if (mistral) {
    describe('MistralAI: embeddings generation', () => {
        test('embeddings for text', async () => {
            const r = await firstValues(mistral!)({ type: "text", text: TEXT });
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
            expect(r.model).toBe("mistral-embed");
        }, TIMEOUT);
    });
}

if (watsonx) {
    describe('Watsonx: embeddings generation', () => {
        test('embeddings for text', async () => {
            const r = await firstValues(watsonx!)({ type: "text", text: TEXT });
            expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
            expect(r.model).toBe("ibm/slate-125m-english-rtrvr");
        }, TIMEOUT);
    });
}
