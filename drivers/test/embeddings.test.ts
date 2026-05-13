/**
 * Cross-provider embedding integration tests.
 *
 * Default (USE_REAL_API = false): all provider describe blocks are skipped —
 * no credentials required.
 * Live mode  (USE_REAL_API = true): each provider block runs only when the
 * corresponding env vars are present. Set them in a .env file in the drivers/
 * package directory or export them before running vitest.
 *
 *   GOOGLE_REGION, GOOGLE_PROJECT_ID   — Vertex AI
 *   BEDROCK_REGION                     — AWS Bedrock (credential chain used)
 *   OPENAI_API_KEY                     — OpenAI
 *   MISTRAL_API_KEY                    — Mistral AI
 *   WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_ENDPOINT_URL — IBM Watsonx
 */
import { Base64DataSource, type Driver } from '@llumiverse/core';
import 'dotenv/config';
import { readFile } from 'fs/promises';
import { describe, expect, test } from "vitest";
import { BedrockDriver, MistralAIDriver, OpenAIDriver, VertexAIDriver, WatsonxDriver } from "../src/index.js";

// ── Live-mode toggle ────────────────────────────────────────────────────────
/** Set to true to run tests against real provider APIs. */
const USE_REAL_API = false;

const TIMEOUT = 15000;
const TEXT = "Hello";

const IMAGE_SRC = import.meta.dirname + "/hello_world.jpg";

async function convertImageToBase64(path: string) {
    const data = await readFile(path);
    return Buffer.from(data).toString('base64');
}

// Only load the image file when live testing to avoid FS errors in mock mode.
const IMAGE = USE_REAL_API ? await convertImageToBase64(IMAGE_SRC) : "";

function imageDataSource(): Base64DataSource {
    return new Base64DataSource("hello_world.jpg", "image/jpeg", IMAGE);
}

let vertex: VertexAIDriver | undefined;
if (USE_REAL_API && process.env.GOOGLE_REGION) {
    vertex = new VertexAIDriver({
        project: process.env.GOOGLE_PROJECT_ID as string,
        region: process.env.GOOGLE_REGION as string,
    });
}

let bedrock: BedrockDriver | undefined;
if (USE_REAL_API && process.env.BEDROCK_REGION) {
    bedrock = new BedrockDriver({
        region: process.env.BEDROCK_REGION as string,
    });
}

let openai: OpenAIDriver | undefined;
if (USE_REAL_API && process.env.OPENAI_API_KEY) {
    openai = new OpenAIDriver({
        apiKey: process.env.OPENAI_API_KEY as string,
    });
}

let mistral: MistralAIDriver | undefined;
if (USE_REAL_API && process.env.MISTRAL_API_KEY) {
    mistral = new MistralAIDriver({
        apiKey: process.env.MISTRAL_API_KEY as string,
    });
}

let watsonx: WatsonxDriver | undefined;
if (USE_REAL_API && process.env.WATSONX_API_KEY) {
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

describe.skipIf(!USE_REAL_API || !vertex)('VertexAI: embeddings generation', () => {
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

describe.skipIf(!USE_REAL_API || !bedrock)('Bedrock: embeddings generation', () => {
    test('embeddings for text (Nova default)', async () => {
        const r = await firstValues(bedrock!)({ type: "text", text: TEXT });
        expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
    }, TIMEOUT);

    test('embeddings for image (Nova multimodal)', async () => {
        const r = await firstValues(bedrock!)({ type: "image", source: imageDataSource() });
        expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
    }, TIMEOUT);
});

describe.skipIf(!USE_REAL_API || !openai)('OpenAI: embeddings generation', () => {
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

describe.skipIf(!USE_REAL_API || !mistral)('MistralAI: embeddings generation', () => {
    test('embeddings for text', async () => {
        const r = await firstValues(mistral!)({ type: "text", text: TEXT });
        expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
        expect(r.model).toBe("mistral-embed");
    }, TIMEOUT);
});

describe.skipIf(!USE_REAL_API || !watsonx)('Watsonx: embeddings generation', () => {
    test('embeddings for text', async () => {
        const r = await firstValues(watsonx!)({ type: "text", text: TEXT });
        expect(r.results[0].outputs[0].values.length).toBeGreaterThan(0);
        expect(r.model).toBe("ibm/slate-125m-english-rtrvr");
    }, TIMEOUT);
});
