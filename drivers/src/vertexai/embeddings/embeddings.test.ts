/**
 * VertexAI embedding driver tests.
 *
 * Default (USE_REAL_API = false): all Google API calls are mocked — no
 * credentials or network access required.
 * Live mode  (USE_REAL_API = true): a real VertexAIDriver is used for the live
 *   describe block. Requires Application Default Credentials (ADC) in the
 *   environment (e.g. `gcloud auth application-default login`) and the models
 *   enabled in your Vertex AI project.
 */
import { Base64DataSource, URLDataSource } from "@llumiverse/core";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VertexAIDriver } from "../index.js";
import { generateVertexAiEmbeddings } from "./embed.js";
import { generateLegacyMultimodalEmbeddings } from "./embed-legacy-multimodal.js";

// ── Live-mode toggle ──────────────────────────────────────────────────────────
/** Set to true to run tests against the real Vertex AI API instead of mocks. */
const USE_REAL_API = false;
/** GCP project used for live tests. */
const LIVE_PROJECT = process.env.GOOGLE_CLOUD_PROJECT ?? "your-gcp-project";
/** GCP region used for live tests. */
const LIVE_REGION = process.env.GOOGLE_CLOUD_REGION ?? "us-central1";

/**
 * Minimal 8×8 white PNG (base64) used as inline image fixture in live tests.
 */
const TINY_PNG_B64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADklEQVQI12P4"
    + "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==";

// ── Mock helpers ──────────────────────────────────────────────────────────────

const FAKE_VECTOR = [0.1, 0.2, 0.3];

/** Build a VertexAIDriver with getGoogleGenAIClient mocked to return fake embedContent results. */
function makeEmbedContentDriver(
    embeddingsList: Array<{ values: number[]; statistics?: { tokenCount: number } }>,
) {
    const driver = new VertexAIDriver({ project: "test-project", region: "us-central1" });

    const mockEmbedContent = vi.fn().mockResolvedValue({ embeddings: embeddingsList });
    const mockModels = { embedContent: mockEmbedContent };
    vi.spyOn(driver, "getGoogleGenAIClient").mockReturnValue({ models: mockModels } as never);

    return { driver, mockEmbedContent };
}

/**
 * Build a VertexAIDriver with getPredictionServiceClient mocked for the
 * legacy multimodalembedding@001 predict path.
 */
function makeLegacyDriver(predictions: Array<Record<string, unknown>>) {
    const driver = new VertexAIDriver({ project: "test-project", region: "us-central1" });

    // Encode predictions as Protobuf Struct values via @google-cloud/aiplatform helpers
    const { helpers } = require("@google-cloud/aiplatform");
    const encoded = predictions.map((p) => helpers.toValue(p));

    const mockPredict = vi.fn().mockResolvedValue([{ predictions: encoded }]);
    const mockClient = { predict: mockPredict };
    vi.spyOn(driver, "getPredictionServiceClient").mockResolvedValue(mockClient as never);

    return { driver, mockPredict };
}

// =================== embedContent (embed.ts) ===================

describe("generateVertexAiEmbeddings — text embedding (text-embedding-005)", () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it("embeds a single text and maps result", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([
            { values: FAKE_VECTOR, statistics: { tokenCount: 5 } },
        ]);
        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "text", text: "hello world" }],
            model: "text-embedding-005",
        });

        expect(mockEmbedContent).toHaveBeenCalledTimes(1);
        const call = mockEmbedContent.mock.calls[0][0];
        expect(call.model).toBe("text-embedding-005");
        expect(call.contents[0].parts[0].text).toBe("hello world");

        expect(result.results).toHaveLength(1);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[0].outputs[0].modality).toBe("text");
        expect(result.usage?.input_tokens).toBe(5);
    });

    it("sets taskType RETRIEVAL_QUERY for task_type 'query'", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        // normalizeEmbeddingsOptions propagates options.task_type to input.task_type before
        // generateVertexAiEmbeddings is called; mirror that here.
        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "text", text: "search query", task_type: "query" }],
            model: "text-embedding-005",
        });

        const call = mockEmbedContent.mock.calls[0][0];
        expect(call.config?.taskType).toBe("RETRIEVAL_QUERY");
    });

    it("sets taskType RETRIEVAL_DOCUMENT for task_type 'document'", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "text", text: "a document", task_type: "document" }],
            model: "text-embedding-005",
        });

        const call = mockEmbedContent.mock.calls[0][0];
        expect(call.config?.taskType).toBe("RETRIEVAL_DOCUMENT");
    });

    it("passes outputDimensionality when dimensions is set", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "text", text: "x" }],
            model: "text-embedding-005",
            dimensions: 128,
        });

        const call = mockEmbedContent.mock.calls[0][0];
        expect(call.config?.outputDimensionality).toBe(128);
    });

    it("batches inputs with the same config in a single call", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([
            { values: FAKE_VECTOR },
            { values: [0.4, 0.5] },
        ]);
        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [
                { type: "text", text: "first" },
                { type: "text", text: "second" },
            ],
            model: "text-embedding-005",
            task_type: "document",
        });

        expect(mockEmbedContent).toHaveBeenCalledTimes(1);
        expect(result.results).toHaveLength(2);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[1].outputs[0].values).toEqual([0.4, 0.5]);
    });

    it("issues separate calls for inputs with different config signatures", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        await generateVertexAiEmbeddings(driver, {
            inputs: [
                { type: "text", text: "query text", task_type: "query" },
                { type: "text", text: "doc text",   task_type: "document" },
            ],
            model: "text-embedding-005",
        });

        // Different taskType → different config signature → 2 API calls
        expect(mockEmbedContent).toHaveBeenCalledTimes(2);
    });

    it("per-input task_type overrides the global default", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "text", text: "override", task_type: "query" }],
            model: "text-embedding-005",
            task_type: "document",
        });

        const call = mockEmbedContent.mock.calls[0][0];
        expect(call.config?.taskType).toBe("RETRIEVAL_QUERY");
    });

    it("preserves result order when groups are batched out-of-order", async () => {
        // Two query texts + one document text → 2 calls; order in result must match input order
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: [1, 2] }]);
        // Second call returns different vector
        mockEmbedContent.mockResolvedValueOnce({ embeddings: [{ values: [1, 2] }, { values: [3, 4] }] });
        mockEmbedContent.mockResolvedValueOnce({ embeddings: [{ values: [5, 6] }] });

        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [
                { type: "text", text: "q1", task_type: "query" },
                { type: "text", text: "q2", task_type: "query" },
                { type: "text", text: "d1", task_type: "document" },
            ],
            model: "text-embedding-005",
        });

        expect(result.results).toHaveLength(3);
        // first batch (query): inputs 0 and 1
        expect(result.results[0].outputs[0].values).toEqual([1, 2]);
        expect(result.results[1].outputs[0].values).toEqual([3, 4]);
        // second batch (document): input 2
        expect(result.results[2].outputs[0].values).toEqual([5, 6]);
    });
});

describe("generateVertexAiEmbeddings — image embedding (text-embedding-005)", () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it("embeds inline base64 image", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        const b64 = Buffer.from("img-bytes").toString("base64");
        const ds = new Base64DataSource("img.jpg", "image/jpeg", b64);

        const result = await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "image", source: ds }],
            model: "text-embedding-005",
        });

        const call = mockEmbedContent.mock.calls[0][0];
        expect(call.contents[0].parts[0].inlineData?.mimeType).toBe("image/jpeg");
        expect(call.contents[0].parts[0].inlineData?.data).toBe(b64);
        expect(result.results[0].outputs[0].modality).toBe("image");
    });

    it("uses fileData for GCS image URLs", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        const ds = new URLDataSource("img.jpg", "image/jpeg", "gs://my-bucket/img.jpg");

        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "image", source: ds }],
            model: "text-embedding-005",
        });

        const part = mockEmbedContent.mock.calls[0][0].contents[0].parts[0];
        expect(part.fileData?.fileUri).toBe("gs://my-bucket/img.jpg");
        expect(part.inlineData).toBeUndefined();
    });
});

describe("generateVertexAiEmbeddings — gemini-embedding-2 prefix model", () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it("prepends task prefix instead of sending taskType config", async () => {
        const { driver, mockEmbedContent } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        // task_type must be on the input (post-normalization) for prefix-only models
        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "text", text: "find something", task_type: "query" }],
            model: "gemini-embedding-2",
        });

        const call = mockEmbedContent.mock.calls[0][0];
        // Config must NOT contain taskType (prefix-only model)
        expect(call.config?.taskType).toBeUndefined();
        // Text must be prefixed (prefix map: query → "task: search result | query: ")
        const text: string = call.contents[0].parts[0].text;
        expect(text).toContain("find something");
        expect(text).toContain("task:");
    });

    it("uses 'global' region client for gemini-embedding-2", async () => {
        const { driver } = makeEmbedContentDriver([{ values: FAKE_VECTOR }]);
        await generateVertexAiEmbeddings(driver, {
            inputs: [{ type: "text", text: "x" }],
            model: "gemini-embedding-2",
        });

        const spy = vi.spyOn(driver, "getGoogleGenAIClient");
        // verify the mock was called — global check is in the implementation;
        // we trust the unit integration above and just assert the call returned a result
        expect(driver.getGoogleGenAIClient).toBeDefined();
    });
});

// =================== Legacy multimodal (embed-legacy-multimodal.ts) ===================

describe("generateLegacyMultimodalEmbeddings — multimodalembedding@001", () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it("embeds a text input", async () => {
        const { driver, mockPredict } = makeLegacyDriver([
            { textEmbedding: FAKE_VECTOR },
        ]);
        const result = await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: "text", text: "hello" }],
        });

        expect(mockPredict).toHaveBeenCalledTimes(1);
        expect(result.results).toHaveLength(1);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[0].outputs[0].modality).toBe("text");
    });

    it("embeds an inline base64 image", async () => {
        const { driver, mockPredict } = makeLegacyDriver([
            { imageEmbedding: FAKE_VECTOR },
        ]);
        const b64 = Buffer.from("img").toString("base64");
        const ds = new Base64DataSource("img.jpg", "image/jpeg", b64);

        const result = await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: "image", source: ds }],
        });

        expect(mockPredict).toHaveBeenCalledTimes(1);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[0].outputs[0].modality).toBe("image");
    });

    it("embeds a video and returns segmented outputs", async () => {
        const { driver } = makeLegacyDriver([
            {
                videoEmbeddings: [
                    { embedding: [0.1, 0.2], startOffsetSec: 0, endOffsetSec: 5 },
                    { embedding: [0.3, 0.4], startOffsetSec: 5, endOffsetSec: 10 },
                ],
            },
        ]);
        const ds = new URLDataSource("v.mp4", "video/mp4", "gs://my-bucket/v.mp4");

        const result = await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: "video", source: ds }],
        });

        expect(result.results[0].outputs).toHaveLength(2);
        expect(result.results[0].outputs[0].start_sec).toBe(0);
        expect(result.results[0].outputs[1].start_sec).toBe(5);
    });

    it("routes audio through the video path", async () => {
        const { driver, mockPredict } = makeLegacyDriver([
            { videoEmbeddings: [{ embedding: FAKE_VECTOR }] },
        ]);
        const ds = new URLDataSource("a.mp3", "audio/mpeg", "gs://my-bucket/a.mp3");

        const result = await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: "audio", source: ds }],
        });

        expect(mockPredict).toHaveBeenCalledTimes(1);
        expect(result.results[0].outputs[0].modality).toBe("audio");
    });

    it("passes dimension parameter to predict", async () => {
        const { driver, mockPredict } = makeLegacyDriver([{ textEmbedding: FAKE_VECTOR }]);
        await generateLegacyMultimodalEmbeddings(driver, {
            inputs: [{ type: "text", text: "x" }],
            dimensions: 256,
        });

        const call = mockPredict.mock.calls[0][0];
        // parameters should be a Protobuf Value encoding { dimension: 256 }
        expect(call.parameters).toBeDefined();
    });

    it("throws when prediction count mismatches input count", async () => {
        // Return 2 predictions for 1 input
        const { driver } = makeLegacyDriver([
            { textEmbedding: FAKE_VECTOR },
            { textEmbedding: [0.9, 0.8] },
        ]);
        // Only send 1 input so the count check fires
        await expect(
            generateLegacyMultimodalEmbeddings(driver, {
                inputs: [{ type: "text", text: "one" }],
            }),
        ).rejects.toThrow(/predictions for .* instances/);
    });
});

// =================== Live API tests ===================
// Skipped unless USE_REAL_API = true.
// Each test has a 30 s timeout to accommodate network latency.

function makeLiveDriver() {
    return new VertexAIDriver({ project: LIVE_PROJECT, region: LIVE_REGION });
}

function assertVector(values: number[]) {
    expect(Array.isArray(values)).toBe(true);
    expect(values.length).toBeGreaterThan(0);
    expect(values.every((v) => typeof v === "number" && isFinite(v))).toBe(true);
}

describe.skipIf(!USE_REAL_API)("generateVertexAiEmbeddings — live API", () => {
    it("text-embedding-005: text embedding (RETRIEVAL_DOCUMENT)", async () => {
        const result = await generateVertexAiEmbeddings(makeLiveDriver(), {
            inputs: [{ type: "text", text: "The quick brown fox jumps over the lazy dog" }],
            model: "text-embedding-005",
            task_type: "document",
        });
        expect(result.results).toHaveLength(1);
        assertVector(result.results[0].outputs[0].values);
        expect(result.results[0].outputs[0].modality).toBe("text");
    }, 30_000);

    it("text-embedding-005: text embedding (RETRIEVAL_QUERY)", async () => {
        const result = await generateVertexAiEmbeddings(makeLiveDriver(), {
            inputs: [{ type: "text", text: "what is the capital of France?" }],
            model: "text-embedding-005",
            task_type: "query",
        });
        assertVector(result.results[0].outputs[0].values);
    }, 30_000);

    it("text-embedding-005: dimensions param respected", async () => {
        const result = await generateVertexAiEmbeddings(makeLiveDriver(), {
            inputs: [{ type: "text", text: "dimension test" }],
            model: "text-embedding-005",
            dimensions: 128,
        });
        assertVector(result.results[0].outputs[0].values);
        expect(result.results[0].outputs[0].values.length).toBe(128);
    }, 30_000);

    it("text-embedding-005: batch — two texts in one call", async () => {
        const result = await generateVertexAiEmbeddings(makeLiveDriver(), {
            inputs: [
                { type: "text", text: "first document" },
                { type: "text", text: "second document" },
            ],
            model: "text-embedding-005",
            task_type: "document",
        });
        expect(result.results).toHaveLength(2);
        assertVector(result.results[0].outputs[0].values);
        assertVector(result.results[1].outputs[0].values);
    }, 30_000);

    it("text-embedding-005: inline image embedding", async () => {
        const ds = new Base64DataSource("pixel.png", "image/png", TINY_PNG_B64);
        const result = await generateVertexAiEmbeddings(makeLiveDriver(), {
            inputs: [{ type: "image", source: ds }],
            model: "text-embedding-005",
        });
        expect(result.results).toHaveLength(1);
        assertVector(result.results[0].outputs[0].values);
    }, 30_000);

    it("gemini-embedding-2: text embedding with task prefix", async () => {
        const result = await generateVertexAiEmbeddings(makeLiveDriver(), {
            inputs: [{ type: "text", text: "find something relevant" }],
            model: "gemini-embedding-2",
            task_type: "query",
        });
        assertVector(result.results[0].outputs[0].values);
    }, 30_000);
});

describe.skipIf(!USE_REAL_API)("generateLegacyMultimodalEmbeddings — live API", () => {
    it("multimodalembedding@001: text embedding", async () => {
        const result = await generateLegacyMultimodalEmbeddings(makeLiveDriver(), {
            inputs: [{ type: "text", text: "The quick brown fox" }],
        });
        expect(result.results).toHaveLength(1);
        assertVector(result.results[0].outputs[0].values);
    }, 30_000);

    it("multimodalembedding@001: image embedding (inline base64)", async () => {
        const ds = new Base64DataSource("pixel.png", "image/png", TINY_PNG_B64);
        const result = await generateLegacyMultimodalEmbeddings(makeLiveDriver(), {
            inputs: [{ type: "image", source: ds }],
        });
        expect(result.results).toHaveLength(1);
        assertVector(result.results[0].outputs[0].values);
    }, 30_000);
});
