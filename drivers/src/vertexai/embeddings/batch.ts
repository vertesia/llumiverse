import { createHash } from 'node:crypto';
import type { BatchJob, JobState } from '@google/genai';
import type { EmbeddingInput, EmbeddingTaskType, TextEmbeddingInput } from '@llumiverse/core';
import type { VertexAIDriver } from '../index.js';
import { buildVertexEmbeddingText, toGoogleTaskType, vertexEmbeddingInputToContent } from './format.js';

export interface ParsedVertexEmbeddingBatchResult {
    key?: string;
    vector?: number[];
    providerError: boolean;
    failureCategory?: string;
}

/** Legacy batches echo only model input fields, dropping custom keys (including instanceConfig.keyField). */
export function vertexEmbeddingBatchCorrelationKey(row: Record<string, unknown>): string | undefined {
    if (typeof row.content === 'string') {
        if (row.task_type != null && typeof row.task_type !== 'string') return undefined;
        if (row.title != null && typeof row.title !== 'string') return undefined;
        return `input:${createHash('sha256')
            .update(JSON.stringify([row.content, row.task_type ?? null, row.title ?? null]))
            .digest('hex')}`;
    }
    return typeof row.key === 'string' ? row.key : undefined;
}

/** Normalize Gemini and legacy output rows without retaining input content or provider error messages. */
export function parseVertexEmbeddingBatchResult(record: Record<string, unknown>): ParsedVertexEmbeddingBatchResult {
    // Gemini JSONL encodes google.rpc.Status as a JSON string; legacy rows may use an object.
    let status = record.status;
    if (typeof status === 'string' && status.trim()) {
        try {
            status = JSON.parse(status);
        } catch {
            status = { code: -1 };
        }
    }
    const code = status && typeof status === 'object' && 'code' in status ? status.code : undefined;
    const statusError = code !== undefined && code !== 0;
    const response = record.response as Record<string, unknown> | undefined;
    const predictions = record.predictions as Array<Record<string, unknown>> | undefined;
    const embeddings = (response?.embedding ?? response?.embeddings ?? predictions?.[0]?.embeddings) as
        | Array<Record<string, unknown>>
        | Record<string, unknown>
        | undefined;
    const candidate = Array.isArray(embeddings) ? embeddings[0]?.values : embeddings?.values;
    const instance = record.instance as Record<string, unknown> | undefined;
    const key = record.key ?? (instance ? vertexEmbeddingBatchCorrelationKey(instance) : undefined);
    return {
        key: typeof key === 'string' ? key : undefined,
        vector: Array.isArray(candidate) ? (candidate as number[]) : undefined,
        providerError: statusError || !!record.error || !!response?.error,
        ...(statusError ? { failureCategory: code === 7 ? 'provider_permission_denied' : 'provider_row_error' } : {}),
    };
}

export type VertexEmbeddingBatchSchema = 'gemini' | 'legacy';
export type VertexEmbeddingBatchModality = 'text' | 'image';

export interface VertexEmbeddingBatchCapability {
    model: string;
    schema: VertexEmbeddingBatchSchema;
    modalities: readonly VertexEmbeddingBatchModality[];
    maxRows: number;
    location: 'environment' | 'global';
    taskEncoding: 'config' | 'prefix';
    maxSynchronousInputs?: number;
}

const CAPABILITIES = new Map<string, VertexEmbeddingBatchCapability>([
    [
        'gemini-embedding-2',
        {
            model: 'gemini-embedding-2',
            schema: 'gemini',
            modalities: ['text', 'image'],
            maxRows: 1_000_000,
            location: 'global',
            taskEncoding: 'prefix',
            maxSynchronousInputs: 1,
        },
    ],
    [
        'gemini-embedding-001',
        {
            model: 'gemini-embedding-001',
            schema: 'gemini',
            modalities: ['text'],
            maxRows: 1_000_000,
            location: 'environment',
            taskEncoding: 'config',
            maxSynchronousInputs: 1,
        },
    ],
    [
        'text-embedding-004',
        {
            model: 'text-embedding-004',
            schema: 'legacy',
            modalities: ['text'],
            maxRows: 30_000,
            location: 'environment',
            taskEncoding: 'config',
        },
    ],
    [
        'text-embedding-005',
        {
            model: 'text-embedding-005',
            schema: 'legacy',
            modalities: ['text'],
            maxRows: 30_000,
            location: 'environment',
            taskEncoding: 'config',
        },
    ],
]);

export function normalizeVertexEmbeddingModelId(model: string): string {
    return model.split('/').at(-1) ?? model;
}

export function getVertexEmbeddingBatchCapability(
    model: string,
    modality: VertexEmbeddingBatchModality,
): VertexEmbeddingBatchCapability | undefined {
    const capability = CAPABILITIES.get(normalizeVertexEmbeddingModelId(model));
    return capability?.modalities.includes(modality) ? capability : undefined;
}

export interface FormatVertexEmbeddingBatchRowOptions {
    key: string;
    model: string;
    input: EmbeddingInput;
    dimensions: number;
    schema?: VertexEmbeddingBatchSchema;
}

type GeminiEmbeddingBatchRow = {
    key: string;
    request: {
        content: { parts: NonNullable<Awaited<ReturnType<typeof vertexEmbeddingInputToContent>>['parts']> };
        embed_content_config: {
            output_dimensionality: number;
            task_type?: string;
            title?: string;
        };
    };
};

type LegacyEmbeddingBatchRow = {
    key: string;
    content: string;
    task_type?: string;
    title?: string;
};

export async function formatVertexEmbeddingBatchRow(
    options: FormatVertexEmbeddingBatchRowOptions,
): Promise<GeminiEmbeddingBatchRow | LegacyEmbeddingBatchRow> {
    const modality: VertexEmbeddingBatchModality = options.input.type === 'text' ? 'text' : 'image';
    const capability = getVertexEmbeddingBatchCapability(options.model, modality);
    if (!capability) {
        throw new Error(`Vertex embedding model ${options.model} does not support batch ${modality} inputs`);
    }
    if (options.schema && options.schema !== capability.schema) {
        throw new Error(
            `Vertex embedding model ${options.model} uses ${capability.schema} batch rows, not ${options.schema}`,
        );
    }

    if (capability.schema === 'legacy') {
        if (options.input.type !== 'text') {
            throw new Error(`Legacy Vertex embedding batches do not support ${options.input.type} inputs`);
        }
        const taskType = toGoogleTaskType(options.input.task_type);
        return {
            key: options.key,
            content: options.input.text,
            ...(taskType ? { task_type: taskType } : {}),
            ...(options.input.title ? { title: options.input.title } : {}),
        };
    }

    const viaPrefix = capability.taskEncoding === 'prefix';
    const content = await vertexEmbeddingInputToContent(options.input, viaPrefix);
    const config: GeminiEmbeddingBatchRow['request']['embed_content_config'] = {
        output_dimensionality: options.dimensions,
    };
    if (options.input.type === 'text' && !viaPrefix) {
        const taskType = toGoogleTaskType(options.input.task_type);
        if (taskType) config.task_type = taskType;
        if (options.input.title) config.title = options.input.title;
    }
    return {
        key: options.key,
        request: { content: { parts: content.parts ?? [] }, embed_content_config: config },
    };
}

export interface CreateVertexEmbeddingBatchOptions {
    model: string;
    modality: VertexEmbeddingBatchModality;
    /** Legacy models accept dimensionality as a job parameter, rather than per row. */
    dimensions?: number;
    inputUri: string;
    outputUri: string;
    displayName: string;
}

export type VertexEmbeddingBatchState = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'paused';

export interface VertexEmbeddingBatchJob {
    name: string;
    displayName?: string;
    state: VertexEmbeddingBatchState;
    model?: string;
    inputUri?: string;
    outputUri?: string;
    error?: string;
}

function normalizedJobState(state: JobState | undefined): VertexEmbeddingBatchState {
    switch (state as string | undefined) {
        case 'JOB_STATE_SUCCEEDED':
            return 'succeeded';
        case 'JOB_STATE_FAILED':
        case 'JOB_STATE_EXPIRED':
        // Terminal partial results must reach application, including successful rows.
        case 'JOB_STATE_PARTIALLY_SUCCEEDED':
            return 'failed';
        case 'JOB_STATE_CANCELLED':
            return 'cancelled';
        case 'JOB_STATE_PAUSED':
            return 'paused';
        case 'JOB_STATE_RUNNING':
        case 'JOB_STATE_UPDATING':
        case 'JOB_STATE_CANCELLING':
            return 'running';
        default:
            return 'pending';
    }
}

function firstGcsUri(value: unknown): string | undefined {
    if (typeof value === 'string') return value;
    if (Array.isArray(value)) return value.find((item): item is string => typeof item === 'string');
    if (value && typeof value === 'object' && 'gcsUri' in value) {
        return firstGcsUri((value as { gcsUri?: unknown }).gcsUri);
    }
    return undefined;
}

function toBatchJob(job: BatchJob): VertexEmbeddingBatchJob {
    if (!job.name) throw new Error('Vertex AI batch job response did not contain a resource name');
    return {
        name: job.name,
        displayName: job.displayName,
        state: normalizedJobState(job.state),
        model: job.model,
        inputUri: firstGcsUri(job.src),
        outputUri: firstGcsUri(job.dest),
        error: job.error?.message,
    };
}

function assertExpectedModel(job: VertexEmbeddingBatchJob, expectedModel: string): VertexEmbeddingBatchJob {
    if (job.model && normalizeVertexEmbeddingModelId(job.model) !== normalizeVertexEmbeddingModelId(expectedModel)) {
        throw new Error(`Vertex AI batch job ${job.name} belongs to model ${job.model}, not ${expectedModel}`);
    }
    return job;
}

function batchClient(driver: VertexAIDriver, capability: VertexEmbeddingBatchCapability) {
    return driver.getGoogleGenAIClient(capability.location === 'global' ? 'global' : undefined);
}

export async function createVertexEmbeddingBatch(
    driver: VertexAIDriver,
    options: CreateVertexEmbeddingBatchOptions,
): Promise<VertexEmbeddingBatchJob> {
    const capability = getVertexEmbeddingBatchCapability(options.model, options.modality);
    if (!capability) throw new Error(`Vertex embedding model ${options.model} is not batch capable`);
    const client = batchClient(driver, capability);

    const existing = await client.batches.list({
        config: {
            filter: `displayName=${JSON.stringify(options.displayName)}`,
            pageSize: 10,
            httpOptions: { apiVersion: 'v1' },
        },
    });
    for await (const candidate of existing) {
        const job = toBatchJob(candidate);
        if (
            job.displayName === options.displayName &&
            normalizeVertexEmbeddingModelId(job.model ?? '') === normalizeVertexEmbeddingModelId(options.model) &&
            job.inputUri === options.inputUri &&
            job.outputUri === options.outputUri
        ) {
            return job;
        }
    }

    const created = await client.batches.create({
        model: options.model,
        src: { gcsUri: [options.inputUri], format: 'jsonl' },
        config: {
            displayName: options.displayName,
            dest: { gcsUri: options.outputUri, format: 'jsonl' },
            httpOptions: {
                apiVersion: 'v1',
                ...(capability.schema === 'legacy' && options.dimensions !== undefined
                    ? // The legacy batch backend reads numeric JSON parameters as FLOAT64, not INT64.
                      { extraBody: { modelParameters: { outputDimensionality: String(options.dimensions) } } }
                    : {}),
            },
        },
    });
    return assertExpectedModel(toBatchJob(created), options.model);
}

export async function getVertexEmbeddingBatch(
    driver: VertexAIDriver,
    model: string,
    modality: VertexEmbeddingBatchModality,
    name: string,
): Promise<VertexEmbeddingBatchJob> {
    const capability = getVertexEmbeddingBatchCapability(model, modality);
    if (!capability) throw new Error(`Vertex embedding model ${model} is not batch capable`);
    return assertExpectedModel(
        toBatchJob(
            await batchClient(driver, capability).batches.get({ name, config: { httpOptions: { apiVersion: 'v1' } } }),
        ),
        model,
    );
}

export async function cancelVertexEmbeddingBatch(
    driver: VertexAIDriver,
    model: string,
    modality: VertexEmbeddingBatchModality,
    name: string,
): Promise<VertexEmbeddingBatchJob> {
    const capability = getVertexEmbeddingBatchCapability(model, modality);
    if (!capability) throw new Error(`Vertex embedding model ${model} is not batch capable`);
    await batchClient(driver, capability).batches.cancel({ name, config: { httpOptions: { apiVersion: 'v1' } } });
    return getVertexEmbeddingBatch(driver, model, modality, name);
}

export async function deleteVertexEmbeddingBatch(
    driver: VertexAIDriver,
    model: string,
    modality: VertexEmbeddingBatchModality,
    name: string,
): Promise<VertexEmbeddingBatchJob> {
    const capability = getVertexEmbeddingBatchCapability(model, modality);
    if (!capability) throw new Error(`Vertex embedding model ${model} is not batch capable`);
    await batchClient(driver, capability).batches.delete({ name, config: { httpOptions: { apiVersion: 'v1' } } });
    return { name, state: 'cancelled' };
}

export function vertexBatchTextForParity(input: TextEmbeddingInput, model: string): string {
    const capability = getVertexEmbeddingBatchCapability(model, 'text');
    return buildVertexEmbeddingText(input, capability?.taskEncoding === 'prefix');
}

export function vertexBatchTaskTypeForParity(taskType: EmbeddingTaskType | undefined): string | undefined {
    return toGoogleTaskType(taskType);
}
