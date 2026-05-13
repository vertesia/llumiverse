/**
 * Unit tests for Bedrock embedding sub-driver paths.
 * All AWS API calls are mocked — no credentials required.
 */
import { Base64DataSource, URLDataSource } from "@llumiverse/core";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { generateBedrockEmbeddings } from "./embeddings.js";
import { BedrockDriver } from "./index.js";

// Minimal fake BedrockRuntime executor
function makeExecutor(responseBody: unknown) {
    return {
        invokeModel: vi.fn().mockResolvedValue({
            body: Buffer.from(JSON.stringify(responseBody)),
        }),
    };
}

function makeDriver(responseBody: unknown) {
    const driver = new BedrockDriver({ region: "us-east-1" });
    const executor = makeExecutor(responseBody);
    vi.spyOn(driver, "getExecutor").mockReturnValue(executor as never);
    return { driver, executor };
}

const FAKE_VECTOR = [0.1, 0.2, 0.3];

// =================== Routing ===================

describe("generateBedrockEmbeddings — routing", () => {
    it("routes to Nova for default (no model)", async () => {
        const { driver, executor } = makeDriver({ embeddings: [{ embeddingType: "TEXT", embedding: FAKE_VECTOR }] });
        await generateBedrockEmbeddings(driver, { inputs: [{ type: "text", text: "hi" }] });
        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(executor.invokeModel.mock.calls[0][0].modelId).toBe("amazon.nova-2-multimodal-embeddings-v1:0");
        expect(body.taskType).toBe("SINGLE_EMBEDDING");
    });

    it("routes to Titan for titan-embed-text model", async () => {
        const { driver, executor } = makeDriver({ embedding: FAKE_VECTOR });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "hi" }],
            model: "amazon.titan-embed-text-v2:0",
        });
        expect(executor.invokeModel.mock.calls[0][0].modelId).toBe("amazon.titan-embed-text-v2:0");
        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.inputText).toBe("hi");
    });

    it("routes to Cohere for cohere.embed model", async () => {
        const { driver, executor } = makeDriver({ embeddings: [FAKE_VECTOR] });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "hi" }],
            model: "cohere.embed-english-v3",
        });
        expect(executor.invokeModel.mock.calls[0][0].modelId).toBe("cohere.embed-english-v3");
        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.texts).toEqual(["hi"]);
    });
});

// =================== Nova ===================

describe("generateBedrockEmbeddings — Nova", () => {
    it("builds correct text request and maps result", async () => {
        const { driver, executor } = makeDriver({ embeddings: [{ embeddingType: "TEXT", embedding: FAKE_VECTOR }] });
        const result = await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "hello" }],
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.taskType).toBe("SINGLE_EMBEDDING");
        expect(body.singleEmbeddingParams.text.value).toBe("hello");
        expect(body.singleEmbeddingParams.text.truncationMode).toBe("END");
        expect(body.singleEmbeddingParams.embeddingPurpose).toBe("GENERIC_INDEX");

        expect(result.results).toHaveLength(1);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[0].outputs[0].modality).toBe("text");
    });

    it("sets embeddingPurpose GENERIC_RETRIEVAL for task_type query", async () => {
        const { driver, executor } = makeDriver({ embeddings: [{ embeddingType: "TEXT", embedding: FAKE_VECTOR }] });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "find me" }],
            task_type: "query",
        });
        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.singleEmbeddingParams.embeddingPurpose).toBe("GENERIC_RETRIEVAL");
    });

    it("builds correct image request (base64 bytes)", async () => {
        const { driver, executor } = makeDriver({ embeddings: [{ embeddingType: "IMAGE", embedding: FAKE_VECTOR }] });
        const b64 = Buffer.from("img-bytes").toString("base64");
        const ds = new Base64DataSource("img.jpg", "image/jpeg", b64);

        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "image", source: ds }],
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.taskType).toBe("SINGLE_EMBEDDING");
        expect(body.singleEmbeddingParams.image.source.bytes).toBe(b64);
        expect(body.singleEmbeddingParams.image.format).toBe("jpeg");
    });

    it("passes embeddingDimension when dimensions is set", async () => {
        const { driver, executor } = makeDriver({ embeddings: [{ embeddingType: "TEXT", embedding: FAKE_VECTOR }] });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "x" }],
            dimensions: 256,
        });
        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.singleEmbeddingParams.embeddingDimension).toBe(256);
    });

    it("throws when response has no embeddings array", async () => {
        const { driver } = makeDriver({ someOtherField: [] });
        await expect(
            generateBedrockEmbeddings(driver, { inputs: [{ type: "text", text: "x" }] }),
        ).rejects.toThrow("Nova embeddings response missing 'embeddings[0].embedding'");
    });
});

// =================== Nova S3 URL passthrough ===================

describe("generateBedrockEmbeddings — Nova S3 URL passthrough", () => {
    it("passes s3Location for s3:// URLs (video)", async () => {
        const { driver, executor } = makeDriver({ embeddings: [{ embeddingType: "VIDEO", embedding: FAKE_VECTOR }] });
        const ds = new URLDataSource("video.mp4", "video/mp4", "s3://my-bucket/video.mp4");

        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "video", source: ds }],
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.taskType).toBe("SINGLE_EMBEDDING");
        expect(body.singleEmbeddingParams.video.source.s3Location.uri).toBe("s3://my-bucket/video.mp4");
        expect(body.singleEmbeddingParams.video.source.bytes).toBeUndefined();
        expect(body.singleEmbeddingParams.video.embeddingMode).toBe("AUDIO_VIDEO_COMBINED");
    });

    it("passes s3Location for amazonaws.com URLs (audio)", async () => {
        const { driver, executor } = makeDriver({ embeddings: [{ embeddingType: "AUDIO", embedding: FAKE_VECTOR }] });
        const s3HttpsUrl = "https://s3.us-east-1.amazonaws.com/my-bucket/audio.mp3";
        const ds = new URLDataSource("audio.mp3", "audio/mpeg", s3HttpsUrl);

        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "audio", source: ds }],
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.taskType).toBe("SINGLE_EMBEDDING");
        expect(body.singleEmbeddingParams.audio.source.s3Location).toBeDefined();
        expect(body.singleEmbeddingParams.audio.source.bytes).toBeUndefined();
    });
});

// =================== Titan ===================

describe("generateBedrockEmbeddings — Titan", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("sends text to titan text model", async () => {
        const { driver, executor } = makeDriver({ embedding: FAKE_VECTOR, inputTextTokenCount: 1 });
        const result = await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "test" }],
            model: "amazon.titan-embed-text-v2:0",
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.inputText).toBe("test");
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
    });

    it("sends image base64 to titan image model", async () => {
        const { driver, executor } = makeDriver({ embedding: FAKE_VECTOR });
        const b64 = Buffer.from("img").toString("base64");
        const ds = new Base64DataSource("img.jpg", "image/jpeg", b64);

        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "image", source: ds }],
            model: "amazon.titan-embed-image-v1",
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.inputImage).toBe(b64);
        expect(body.inputText).toBeUndefined();
    });

    it("throws for video/audio input on titan", async () => {
        const { driver } = makeDriver({});
        const ds = new URLDataSource("v.mp4", "video/mp4", "s3://b/v.mp4");
        await expect(
            generateBedrockEmbeddings(driver, {
                inputs: [{ type: "video", source: ds }],
                model: "amazon.titan-embed-text-v2:0",
            }),
        ).rejects.toThrow("do not support 'video' input");
    });
});

// =================== Cohere ===================

describe("generateBedrockEmbeddings — Cohere task_type mapping", () => {
    it("maps task_type 'query' to input_type 'search_query'", async () => {
        const { driver, executor } = makeDriver({ embeddings: [FAKE_VECTOR] });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "find me" }],
            model: "cohere.embed-english-v3",
            task_type: "query",
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.input_type).toBe("search_query");
    });

    it("maps task_type 'document' to input_type 'search_document'", async () => {
        const { driver, executor } = makeDriver({ embeddings: [FAKE_VECTOR] });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "content" }],
            model: "cohere.embed-english-v3",
            task_type: "document",
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.input_type).toBe("search_document");
    });

    it("omits input_type when task_type is undefined", async () => {
        const { driver, executor } = makeDriver({ embeddings: [FAKE_VECTOR] });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "content" }],
            model: "cohere.embed-english-v3",
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.input_type).toBeUndefined();
    });

    it("batches multiple text inputs in a single call", async () => {
        const { driver, executor } = makeDriver({ embeddings: [FAKE_VECTOR, [0.4, 0.5]] });
        const result = await generateBedrockEmbeddings(driver, {
            inputs: [
                { type: "text", text: "first" },
                { type: "text", text: "second" },
            ],
            model: "cohere.embed-english-v3",
        });

        expect(executor.invokeModel).toHaveBeenCalledTimes(1);
        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.texts).toEqual(["first", "second"]);
        expect(result.results).toHaveLength(2);
        expect(result.results[0].outputs[0].values).toEqual(FAKE_VECTOR);
        expect(result.results[1].outputs[0].values).toEqual([0.4, 0.5]);
    });

    it("does not send embedding_types field", async () => {
        const { driver, executor } = makeDriver({ embeddings: [FAKE_VECTOR] });
        await generateBedrockEmbeddings(driver, {
            inputs: [{ type: "text", text: "hi" }],
            model: "cohere.embed-english-v3",
        });

        const body = JSON.parse(executor.invokeModel.mock.calls[0][0].body);
        expect(body.embedding_types).toBeUndefined();
        expect(body.truncate).toBeUndefined();
    });
});
