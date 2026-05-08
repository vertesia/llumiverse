import {
    DataSource,
    EmbeddingInput,
    EmbeddingResultItem,
    EmbeddingsOptions,
    EmbeddingsResult,
    EmbeddingTaskType,
    TextEmbeddingInput,
} from "@llumiverse/common";
import { readStreamAsBase64, readStreamAsUint8Array } from "./stream.js";

/**
 * DataSource wrapping an in-memory base64 string. Useful for callers that
 * receive base64 over the wire (e.g. an HTTP API) and need to feed it into
 * a driver that expects a DataSource.
 */
export class Base64DataSource implements DataSource {
    constructor(
        public readonly name: string,
        public readonly mime_type: string,
        private readonly base64: string,
    ) {}

    async getStream(): Promise<ReadableStream<Uint8Array>> {
        const bytes = Buffer.from(this.base64, "base64");
        return new ReadableStream<Uint8Array>({
            start(controller) {
                controller.enqueue(new Uint8Array(bytes));
                controller.close();
            },
        });
    }

    async getURL(): Promise<string> {
        return `data:${this.mime_type};base64,${this.base64}`;
    }

    /** Convenience accessor that avoids re-encoding via getStream. */
    getBase64(): string {
        return this.base64;
    }
}

/**
 * DataSource wrapping a remote URL. The URL is exposed via getURL() so
 * drivers can pass it directly to providers that support URL inputs
 * (gs://, s3://, https://). getStream() fetches the URL on demand.
 */
export class URLDataSource implements DataSource {
    constructor(
        public readonly name: string,
        public readonly mime_type: string,
        private readonly url: string,
    ) {}

    async getStream(): Promise<ReadableStream<Uint8Array>> {
        const response = await fetch(this.url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${this.url}: ${response.status} ${response.statusText}`);
        }
        if (!response.body) {
            throw new Error(`Response body is empty for ${this.url}`);
        }
        return response.body as ReadableStream<Uint8Array>;
    }

    async getURL(): Promise<string> {
        return this.url;
    }
}

/**
 * Read a DataSource as a base64-encoded string. Equivalent to streaming the
 * source and base64-encoding the bytes.
 */
export async function dataSourceToBase64(ds: DataSource): Promise<string> {
    if (ds instanceof Base64DataSource) {
        return ds.getBase64();
    }
    return readStreamAsBase64(await ds.getStream());
}

/**
 * Read a DataSource fully into a Uint8Array.
 */
export async function dataSourceToBytes(ds: DataSource): Promise<Uint8Array> {
    return readStreamAsUint8Array(await ds.getStream());
}

/**
 * Convenience accessor for the common case where the caller embedded a
 * single text or image input and wants the first vector back.
 */
export function firstVector(result: EmbeddingsResult): number[] {
    const first = result.results[0]?.outputs[0]?.values;
    if (!first) {
        throw new Error("EmbeddingsResult contains no vectors");
    }
    return first;
}

/**
 * Validate an EmbeddingsOptions object and return a normalized copy where:
 * - request-level task_type and truncate are propagated to text inputs
 *   that don't define their own
 * - inputs is guaranteed non-empty
 *
 * Drivers should call this before doing any provider-specific work.
 */
export function normalizeEmbeddingsOptions(options: EmbeddingsOptions): EmbeddingsOptions {
    if (!options.inputs || options.inputs.length === 0) {
        throw new Error("EmbeddingsOptions.inputs must contain at least one input");
    }

    const inputs: EmbeddingInput[] = options.inputs.map((input) => {
        if (input.type === "text") {
            const text = input as TextEmbeddingInput;
            return {
                ...text,
                task_type: text.task_type ?? options.task_type,
                truncate: text.truncate ?? options.truncate,
            } satisfies TextEmbeddingInput;
        }
        return input;
    });

    return {
        ...options,
        inputs,
    };
}

/**
 * Apply a task-type prompt prefix to a text input. Used by drivers that need
 * to pass task type through the prompt when the model does not accept it as
 * an API parameter (e.g. Vertex AI gemini-embedding-2).
 */
export function applyTaskTypePrefix(
    text: string,
    taskType: EmbeddingTaskType | undefined,
    mapping: Partial<Record<EmbeddingTaskType, string>>,
): string {
    if (!taskType) return text;
    const prefix = mapping[taskType];
    return prefix ? prefix + text : text;
}

/**
 * Group result items so each item lines up with the corresponding input.
 * Helper for drivers that fan out multiple provider calls.
 */
export function buildEmbeddingsResult(
    model: string,
    results: EmbeddingResultItem[],
    usage?: EmbeddingsResult["usage"],
): EmbeddingsResult {
    return { model, results, usage };
}
