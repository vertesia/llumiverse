import type { Content, Part } from '@google/genai';
import type { DataSource, EmbeddingInput, EmbeddingTaskType, TextEmbeddingInput } from '@llumiverse/core';
import { dataSourceToVertexSourceData } from './source-utils.js';

export function buildVertexEmbeddingText(input: TextEmbeddingInput, viaPrefix: boolean): string {
    if (!viaPrefix) return input.text;
    if (!input.task_type) return input.text;
    if (input.task_type === 'query') return `task: search result | query: ${input.text}`;
    return `title: ${input.title ?? 'none'} | text: ${input.text}`;
}

export function toGoogleTaskType(
    taskType: EmbeddingTaskType | undefined,
): 'RETRIEVAL_QUERY' | 'RETRIEVAL_DOCUMENT' | undefined {
    if (taskType === 'query') return 'RETRIEVAL_QUERY';
    if (taskType === 'document') return 'RETRIEVAL_DOCUMENT';
    return undefined;
}

async function dataSourceToPart(ds: DataSource): Promise<Part> {
    const source = await dataSourceToVertexSourceData(ds);
    if (source.gcsUri) return { fileData: { fileUri: source.gcsUri, mimeType: ds.mime_type } };
    if (!source.bytesBase64Encoded) throw new Error('Data source conversion produced neither GCS URI nor inline bytes');
    return { inlineData: { data: source.bytesBase64Encoded, mimeType: ds.mime_type } };
}

export async function vertexEmbeddingInputToContent(input: EmbeddingInput, viaPrefix: boolean): Promise<Content> {
    if (input.type === 'text') return { role: 'user', parts: [{ text: buildVertexEmbeddingText(input, viaPrefix) }] };
    return { role: 'user', parts: [await dataSourceToPart(input.source)] };
}
