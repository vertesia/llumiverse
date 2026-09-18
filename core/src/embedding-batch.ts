import type { EmbeddingInput } from '@llumiverse/common';
import type { Driver } from './Driver.js';

export type EmbeddingBatchModality = 'text' | 'image';
export type EmbeddingBatchState = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'paused';

export interface EmbeddingBatchCapability {
    model: string;
    inputFormat: string;
    maxRows: number;
    location?: string;
}

/** Durable application artifacts, not provider file IDs or signed URLs. Implementations enforce tenant scope. */
export interface EmbeddingBatchArtifactStore {
    list(prefix: string): Promise<string[]>;
    read(uri: string): Promise<ReadableStream<Uint8Array>>;
    write(uri: string, content: ReadableStream<Uint8Array>): Promise<void>;
}

export interface EmbeddingBatchJob {
    /** Opaque provider job identity. */
    name: string;
    displayName?: string;
    state: EmbeddingBatchState;
    model?: string;
    /** Original application artifact locations, retained for ownership checks even with uploaded-file providers. */
    inputUri?: string;
    outputUri?: string;
    error?: string;
}

export interface EmbeddingBatchJobOptions {
    model: string;
    modality: EmbeddingBatchModality;
    name: string;
}

export interface EmbeddingBatchCreateOptions {
    model: string;
    modality: EmbeddingBatchModality;
    dimensions?: number;
    displayName: string;
    inputUri: string;
    outputUri: string;
}

export interface ParsedEmbeddingBatchResult {
    key?: string;
    vector?: number[];
    providerError: boolean;
    failureCategory?: string;
}

/** Optional provider support. Native payloads and file handles stay behind this interface.
 * Lifecycle methods must validate the requested model and preserve original artifact URIs for ownership checks.
 * Create retries must recover a matching displayName/model/input/output submission rather than create duplicates.
 * Result parsers return sanitized failure categories, never input content or native error payloads.
 */
export interface EmbeddingBatchAdapter {
    capability(
        model: string,
        modality: EmbeddingBatchModality,
        location?: string,
    ): EmbeddingBatchCapability | undefined;
    formatRow(options: {
        key: string;
        model: string;
        input: EmbeddingInput;
        dimensions: number;
        inputFormat: string;
    }): Promise<{ row: Record<string, unknown>; resultKey: string }>;
    parseResult(record: Record<string, unknown>): ParsedEmbeddingBatchResult;
    create(
        driver: Driver,
        options: EmbeddingBatchCreateOptions,
        artifacts: EmbeddingBatchArtifactStore,
    ): Promise<EmbeddingBatchJob>;
    get(driver: Driver, options: EmbeddingBatchJobOptions): Promise<EmbeddingBatchJob>;
    cancel(driver: Driver, options: EmbeddingBatchJobOptions): Promise<EmbeddingBatchJob>;
    delete(driver: Driver, options: EmbeddingBatchJobOptions): Promise<EmbeddingBatchJob>;
    /** Called only after job ownership validation. Include partial successes and errors on every terminal state.
     * Return existing object-store artifacts, or materialize provider files idempotently under job.outputUri.
     */
    outputArtifacts(driver: Driver, job: EmbeddingBatchJob, artifacts: EmbeddingBatchArtifactStore): Promise<string[]>;
}
