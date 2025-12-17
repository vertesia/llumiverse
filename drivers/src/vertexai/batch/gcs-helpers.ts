/**
 * GCS (Google Cloud Storage) helper functions for batch operations.
 * Handles upload/download of batch input/output files.
 */

import { Storage } from "@google-cloud/storage";
import type { VertexAIDriver } from "../index.js";

/**
 * Parses a GCS URI into bucket and path components.
 * @param uri - GCS URI in format gs://bucket/path
 * @returns Object with bucket and path
 */
export function parseGcsUri(uri: string): { bucket: string; path: string } {
    const match = uri.match(/^gs:\/\/([^/]+)\/(.+)$/);
    if (!match) {
        throw new Error(`Invalid GCS URI: ${uri}`);
    }
    return { bucket: match[1], path: match[2] };
}

/**
 * Creates a GCS Storage client using the driver's auth.
 */
async function getStorageClient(driver: VertexAIDriver): Promise<Storage> {
    const authClient = await driver.googleAuth.getClient();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return new Storage({ authClient: authClient as any });
}

/**
 * Uploads batch input data (JSONL lines) to GCS.
 *
 * @param driver - VertexAI driver instance (for auth)
 * @param lines - Array of JSONL strings to upload
 * @param bucket - GCS bucket name
 * @param path - Path within the bucket
 * @returns GCS URI of the uploaded file
 */
export async function uploadBatchInput(
    driver: VertexAIDriver,
    lines: string[],
    bucket: string,
    path: string
): Promise<string> {
    const storage = await getStorageClient(driver);
    const file = storage.bucket(bucket).file(path);
    const content = lines.join("\n");

    await file.save(content, {
        contentType: "application/jsonl",
        metadata: {
            contentType: "application/jsonl",
        },
    });

    return `gs://${bucket}/${path}`;
}

/**
 * Downloads batch output data from GCS.
 *
 * @param driver - VertexAI driver instance (for auth)
 * @param gcsUri - GCS URI of the file to download
 * @returns Array of JSONL strings (one per line)
 */
export async function downloadBatchOutput(
    driver: VertexAIDriver,
    gcsUri: string
): Promise<string[]> {
    const { bucket, path } = parseGcsUri(gcsUri);
    const storage = await getStorageClient(driver);
    const [content] = await storage.bucket(bucket).file(path).download();
    return content.toString().split("\n").filter(line => line.trim());
}

/**
 * Lists files in a GCS directory/prefix.
 *
 * @param driver - VertexAI driver instance (for auth)
 * @param gcsUri - GCS URI prefix (e.g., gs://bucket/output/)
 * @returns Array of GCS URIs for files in the directory
 */
export async function listGcsFiles(
    driver: VertexAIDriver,
    gcsUri: string
): Promise<string[]> {
    const { bucket, path } = parseGcsUri(gcsUri);
    const storage = await getStorageClient(driver);
    const [files] = await storage.bucket(bucket).getFiles({ prefix: path });
    return files.map(f => `gs://${bucket}/${f.name}`);
}
