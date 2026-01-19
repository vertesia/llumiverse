/**
 * Gemini File API operations using the @google/genai SDK.
 *
 * Provides functions to upload, get, list, and delete files via the Gemini File API.
 * Files uploaded through this API can be used in batch operations like batch embeddings.
 *
 * API Documentation: https://ai.google.dev/api/files
 */

import type { FetchClient } from "@vertesia/api-fetch-client";
import type { VertexAIDriver } from "../index.js";
import {
    GeminiFileResource,
    GeminiFileState,
} from "./types.js";

/**
 * Maps SDK file state to our type.
 */
function mapFileState(state: string | undefined): GeminiFileState {
    switch (state) {
        case "PROCESSING":
            return "PROCESSING";
        case "ACTIVE":
            return "ACTIVE";
        case "FAILED":
            return "FAILED";
        default:
            return "STATE_UNSPECIFIED";
    }
}

/**
 * Maps SDK file response to our GeminiFileResource type.
 */
function mapFileResource(file: any): GeminiFileResource {
    return {
        name: file.name || "",
        displayName: file.displayName,
        mimeType: file.mimeType || "",
        downloadUri: file.downloadUri,
        sizeBytes: String(file.sizeBytes || "0"),
        createTime: file.createTime || "",
        updateTime: file.updateTime || "",
        expirationTime: file.expirationTime,
        sha256Hash: file.sha256Hash,
        uri: file.uri || "",
        state: mapFileState(file.state),
        source: file.source,
        error: file.error ? {
            code: file.error.code || 0,
            message: file.error.message || "Unknown error",
        } : undefined,
    };
}

export interface RegisterGeminiFilesOptions {
    /** Cloud Storage URIs (gs://...) that should be registered with Gemini. */
    uris: string[];
    /** Optional quota project override for billing headers. */
    quotaProject?: string;
    /** Advanced override for testing; defaults to driver's Gemini fetch client. */
    fetchClient?: FetchClient;
}

export interface GeminiRegisterFilesResponse {
    /** Canonical Gemini file resources corresponding to the registered URIs. */
    files: GeminiFileResource[];
    /** Raw JSON payload from the API in case callers need extra fields. */
    raw: unknown;
    /** Underlying HTTP response for header/status inspection. */
    httpResponse?: Response;
}

/**
 * Registers existing GCS objects with the Gemini File service via HTTP.
 * This fills the gap in the JS SDK until register_files ships upstream.
 */
export async function registerGeminiFiles(
    driver: VertexAIDriver,
    options: RegisterGeminiFilesOptions
): Promise<GeminiRegisterFilesResponse> {
    const { uris, quotaProject, fetchClient } = options;

    if (!Array.isArray(uris) || uris.length === 0) {
        throw new Error("registerGeminiFiles requires at least one gs:// URI");
    }

    const client = fetchClient ?? driver.getGeminiApiFetchClient();
    const headers = quotaProject ? { "x-goog-user-project": quotaProject } : undefined;

    // Log auth configuration for debugging
    driver.logger.debug({
        quotaProject,
        hasCustomHeaders: !!headers,
        clientBaseHeaders: client.headers,
        uriCount: uris.length,
    }, "registerGeminiFiles: Auth configuration");

    // Add request logging callback to see what's actually being sent
    const originalOnRequest = client.onRequest;
    client.onRequest = (req) => {
        const headers: Record<string, string> = {};
        req.headers.forEach((value, key) => {
            headers[key] = value;
        });
        driver.logger.info({
            url: req.url,
            method: req.method,
            headers,
        }, "registerGeminiFiles: Outgoing HTTP request");
        if (originalOnRequest) originalOnRequest(req);
    };

    try {
        const payload = await client.post("/files:register", {
            payload: { uris },
            headers,
        });

        const files = Array.isArray(payload?.files)
            ? payload.files.map(mapFileResource)
            : [];

        return {
            files,
            raw: payload,
            httpResponse: client.response,
        };
    } catch (error) {
        driver.logger.error({ error, uriCount: uris.length }, "Gemini files: register failed");
        throw error;
    }
}

/**
 * Uploads a file to the Gemini File API using the SDK.
 *
 * @param driver - The VertexAI driver instance
 * @param content - The file content (string, Buffer, or Blob)
 * @param mimeType - MIME type of the file (e.g., "application/jsonl")
 * @param displayName - Optional display name for the file
 * @returns The uploaded file resource
 */
export async function uploadFileToGemini(
    driver: VertexAIDriver,
    content: string | Blob,
    mimeType: string,
    displayName?: string
): Promise<GeminiFileResource> {
    // Get the Gemini API client (not VertexAI)
    const client = await driver.getGoogleGenAIClient(undefined, "GEMINI");

    // Convert content to Blob for SDK
    let blob: Blob;
    if (typeof content === "string") {
        blob = new Blob([content]);
    } else {
        blob = content;
    }

    driver.logger.debug({ displayName, mimeType, size: blob.size }, "Uploading file to Gemini File API");
    driver.logger.info({ blob }, "File content blob");
    driver.logger.info({ blob: JSON.stringify(blob) }, "File content blob stringified");

    // Use SDK's files.upload method
    try {
        const result = await client.files.upload({
            file: blob,
            config: {
                mimeType,
                //displayName: displayName || `upload-${Date.now()}`,
            },
        });

        driver.logger.debug({ name: result.name, state: result.state }, "File uploaded to Gemini File API");

        return mapFileResource(result);
    } catch (error: any) {
        driver.logger.error({ error }, "Error uploading file to Gemini File API in llumiverse");
        throw error;
    }
}

/**
 * Gets a file resource from the Gemini File API.
 *
 * @param driver - The VertexAI driver instance
 * @param fileId - The file ID (format: "files/{fileId}" or just "{fileId}")
 * @returns The file resource
 */
export async function getGeminiFile(
    driver: VertexAIDriver,
    fileId: string
): Promise<GeminiFileResource> {
    const client = await driver.getGoogleGenAIClient(undefined, "GEMINI");

    // Ensure the fileId has the correct format
    const normalizedId = fileId.startsWith("files/") ? fileId : `files/${fileId}`;

    const result = await client.files.get({ name: normalizedId });

    return mapFileResource(result);
}

/**
 * Lists files from the Gemini File API.
 *
 * @param driver - The VertexAI driver instance
 * @param pageSize - Optional number of files to return per page
 * @param pageToken - Optional token for pagination
 * @returns List of file resources and optional next page token
 */
export async function listGeminiFiles(
    driver: VertexAIDriver,
    pageSize?: number,
    pageToken?: string
): Promise<{ files: GeminiFileResource[]; nextPageToken?: string }> {
    const client = await driver.getGoogleGenAIClient(undefined, "GEMINI");

    const config: any = {};
    if (pageSize) config.pageSize = pageSize;
    if (pageToken) config.pageToken = pageToken;

    const result = await client.files.list(config);

    // The SDK returns an async iterable, but we'll convert to array
    const files: GeminiFileResource[] = [];
    if (result && Symbol.asyncIterator in result) {
        for await (const file of result) {
            files.push(mapFileResource(file));
        }
    }

    return {
        files,
        nextPageToken: undefined, // SDK handles pagination via async iterator
    };
}

/**
 * Deletes a file from the Gemini File API.
 *
 * @param driver - The VertexAI driver instance
 * @param fileId - The file ID (format: "files/{fileId}" or just "{fileId}")
 */
export async function deleteGeminiFile(
    driver: VertexAIDriver,
    fileId: string
): Promise<void> {
    const client = await driver.getGoogleGenAIClient(undefined, "GEMINI");

    // Ensure the fileId has the correct format
    const normalizedId = fileId.startsWith("files/") ? fileId : `files/${fileId}`;

    await client.files.delete({ name: normalizedId });
}

/**
 * Waits for a file to reach ACTIVE state.
 *
 * Files may be in PROCESSING state immediately after upload.
 * This function polls until the file is ACTIVE or FAILED.
 *
 * @param driver - The VertexAI driver instance
 * @param fileId - The file ID
 * @param maxWaitMs - Maximum time to wait in milliseconds (default: 60000)
 * @param pollIntervalMs - Polling interval in milliseconds (default: 1000)
 * @returns The file resource in ACTIVE state
 * @throws Error if file fails processing or timeout is reached
 */
export async function waitForFileActive(
    driver: VertexAIDriver,
    fileId: string,
    maxWaitMs: number = 60000,
    pollIntervalMs: number = 1000
): Promise<GeminiFileResource> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
        const file = await getGeminiFile(driver, fileId);

        if (file.state === "ACTIVE") {
            return file;
        }

        if (file.state === "FAILED") {
            throw new Error(
                `File processing failed: ${file.error?.message || "Unknown error"}`
            );
        }

        // Still processing, wait and try again
        await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }

    throw new Error(`Timeout waiting for file ${fileId} to become ACTIVE`);
}
