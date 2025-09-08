
import type { EmbedContentParameters, EmbedContentResponse } from '@google/genai';
import { EmbeddingsResult } from '@llumiverse/core';
import { VertexAIDriver } from '../index.js';

export interface TextEmbeddingsOptions {
    model?: string;
    title?: string; // the title for the embedding
    content: string; // the text to generate embeddings for
}

export async function getEmbeddingsForText(driver: VertexAIDriver, options: TextEmbeddingsOptions): Promise<EmbeddingsResult> {
    const model = options.model || 'gemini-embedding-001';

    const genai = driver.getGoogleGenAIClient();
    const params: EmbedContentParameters = {
        model,
        contents: [options.content],
    };

    const resp = await genai.models.embedContent(params) as EmbedContentResponse;
    const emb = resp?.embeddings?.[0];
    if (!emb) {
        throw new Error('Empty embedding response');
    }

    return {
        values: emb.values ?? [],
        model,
        token_count: (emb.statistics as any)?.tokenCount ?? undefined,
    } as EmbeddingsResult;
}
