
import type { EmbeddingsResult } from '@llumiverse/core';
import type { VertexAIDriver } from '../index.js';

export interface TextEmbeddingsOptions {
    model?: string;
    task_type?: "RETRIEVAL_QUERY" | "RETRIEVAL_DOCUMENT" | "SEMANTIC_SIMILARITY" | "CLASSIFICATION" | "CLUSTERING",
    title?: string, // the title for the embedding
    content: string // the text to generate embeddings for
}

interface EmbeddingsForTextPrompt {
    instances: TextEmbeddingsOptions[]
}

interface TextEmbeddingsResult {
    predictions: [
        {
            embeddings: TextEmbeddings
        }
    ]
}

interface TextEmbeddings {
    statistics: {
        truncated: boolean,
        token_count: number
    },
    values: [number]
}

export async function getEmbeddingsForText(driver: VertexAIDriver, options: TextEmbeddingsOptions): Promise<EmbeddingsResult> {
    const prompt = {
        instances: [{
            task_type: options.task_type,
            title: options.title,
            content: options.content
        }]
    } satisfies EmbeddingsForTextPrompt;

    const model = options.model || "gemini-embedding-2";

    if (model.includes("gemini-embedding-2")) {
        return genAIClientGetEmbeddingsForText(driver, { ...options, model });
    }

    const client = driver.getFetchClient();

    const result = await client.post(`/publishers/google/models/${model}:predict`, {
        payload: prompt
    }).catch((e: unknown) => {
        const detail = e instanceof Error ? e.message : String(e);
        const responseBody = e != null && typeof e === 'object' && 'payload' in e
            ? (e as { payload?: { text?: string } }).payload?.text
            : undefined;
        const suffix = responseBody ? `\nResponse body:\n${responseBody}` : '';
        throw new Error(`Failed to generate text embeddings with model '${model}': ${detail}${suffix}`);
    }) as TextEmbeddingsResult;

    return {
        ...result.predictions[0].embeddings,
        model,
        token_count: result.predictions[0].embeddings.statistics?.token_count
    };
}

async function genAIClientGetEmbeddingsForText(driver: VertexAIDriver, options: TextEmbeddingsOptions): Promise<EmbeddingsResult> {
    const model = options.model || 'gemini-embedding-2';
    const client = driver.getGoogleGenAIClient('global');

    const response = await client.models.embedContent({
        model,
        contents: 'hello world',
        config: {
            taskType: options.task_type,
            title: options.title,
        },
    });

    const embedding = response.embeddings?.[0];
    if (!embedding?.values) {
        throw new Error(`No embedding values returned from model '${model}'`);
    }

    return {
        values: embedding.values,
        model,
        token_count: embedding.statistics?.tokenCount,
    };
}
