import {
    type AudioEmbeddingInput,
    buildEmbeddingsResult,
    type DataSource,
    type EmbeddingInput,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    LlumiverseError,
    type VideoEmbeddingInput,
} from '@llumiverse/core';
import type { VertexAIDriver } from '../index.js';
import { dataSourceToVertexSourceData } from './source-utils.js';

const DEFAULT_LEGACY_MULTIMODAL_MODEL = 'multimodalembedding@001';

async function dataSourceToImagePart(
    ds: DataSource,
): Promise<{ gcsUri?: string; mimeType?: string; bytesBase64Encoded?: string }> {
    const source = await dataSourceToVertexSourceData(ds);
    if (source.gcsUri) {
        return { gcsUri: source.gcsUri, mimeType: ds.mime_type };
    }

    if (!source.bytesBase64Encoded) {
        throw new Error('Data source conversion produced neither GCS URI nor inline bytes');
    }

    return { bytesBase64Encoded: source.bytesBase64Encoded, mimeType: ds.mime_type };
}

interface VideoSegmentConfig {
    startOffsetSec?: number;
    endOffsetSec?: number;
    intervalSec?: number;
}

async function dataSourceToVideoPart(
    ds: DataSource,
    input: VideoEmbeddingInput | AudioEmbeddingInput,
): Promise<{ gcsUri?: string; bytesBase64Encoded?: string; videoSegmentConfig?: VideoSegmentConfig }> {
    const source = await dataSourceToVertexSourceData(ds);
    const segmentConfig: VideoSegmentConfig = {};
    if (input.start_sec !== undefined) segmentConfig.startOffsetSec = input.start_sec;
    if (input.start_sec !== undefined && input.length_sec !== undefined) {
        segmentConfig.endOffsetSec = input.start_sec + input.length_sec;
    }
    if ('interval_sec' in input && input.interval_sec !== undefined) {
        segmentConfig.intervalSec = input.interval_sec;
    }
    const config = Object.keys(segmentConfig).length > 0 ? segmentConfig : undefined;
    if (source.gcsUri) {
        return { gcsUri: source.gcsUri, videoSegmentConfig: config };
    }

    if (!source.bytesBase64Encoded) {
        throw new Error('Data source conversion produced neither GCS URI nor inline bytes');
    }

    return { bytesBase64Encoded: source.bytesBase64Encoded, videoSegmentConfig: config };
}

type Modality = 'text' | 'image' | 'video' | 'audio';

type LegacyInstance =
    | { text: string }
    | { image: Awaited<ReturnType<typeof dataSourceToImagePart>> }
    | { video: Awaited<ReturnType<typeof dataSourceToVideoPart>> };

async function buildLegacyInstance(input: EmbeddingInput): Promise<{ instance: LegacyInstance; modality: Modality }> {
    switch (input.type) {
        case 'text':
            return { instance: { text: input.text }, modality: 'text' };
        case 'image':
            return { instance: { image: await dataSourceToImagePart(input.source) }, modality: 'image' };
        case 'video':
            return { instance: { video: await dataSourceToVideoPart(input.source, input) }, modality: 'video' };
        case 'audio':
            // multimodalembedding@001 has no dedicated audio modality; the API treats audio
            // like video (uses videoEmbeddings in the response). Route through the video path
            // and preserve the "audio" modality label so callers can identify the outputs.
            return { instance: { video: await dataSourceToVideoPart(input.source, input) }, modality: 'audio' };
    }
}

interface VideoSegment {
    embedding: number[];
    startOffsetSec?: number;
    endOffsetSec?: number;
}

interface MultimodalPrediction {
    textEmbedding?: number[];
    imageEmbedding?: number[];
    videoEmbeddings?: VideoSegment[];
}

function hasPredictionEmbeddings(prediction: MultimodalPrediction): boolean {
    return Boolean(prediction.textEmbedding || prediction.imageEmbedding || prediction.videoEmbeddings?.length);
}

function decodePredictionValue(value: unknown, index: number, model: string): MultimodalPrediction {
    if (!value || typeof value !== 'object') {
        throw new Error(`Vertex predict returned an empty prediction for input ${index} (model ${model})`);
    }

    const prediction = value as MultimodalPrediction;
    if (!hasPredictionEmbeddings(prediction)) {
        throw new Error(`Vertex predict returned no embeddings for input ${index} (model ${model})`);
    }

    return prediction;
}

function predictionToOutputs(prediction: MultimodalPrediction, modality: Modality): EmbeddingResultItem['outputs'] {
    if (modality === 'text' && prediction.textEmbedding) {
        return [{ values: prediction.textEmbedding, modality: 'text' }];
    }
    if (modality === 'image' && prediction.imageEmbedding) {
        return [{ values: prediction.imageEmbedding, modality: 'image' }];
    }
    if ((modality === 'video' || modality === 'audio') && prediction.videoEmbeddings?.length) {
        return prediction.videoEmbeddings.map((segment) => ({
            values: segment.embedding,
            modality,
            start_sec: segment.startOffsetSec,
            end_sec: segment.endOffsetSec,
        }));
    }
    // Fallback: return whatever is present
    if (prediction.textEmbedding) return [{ values: prediction.textEmbedding, modality: 'text' }];
    if (prediction.imageEmbedding) return [{ values: prediction.imageEmbedding, modality: 'image' }];
    if (prediction.videoEmbeddings?.length) {
        return prediction.videoEmbeddings.map((segment) => ({
            values: segment.embedding,
            modality: 'video' as const,
            start_sec: segment.startOffsetSec,
            end_sec: segment.endOffsetSec,
        }));
    }
    throw new Error('Vertex multimodal prediction returned no embedding values');
}

/**
 * Embed via the legacy multimodalembedding@001 predict API. Returns one result
 * item per input; video/audio segments expand into multiple outputs.
 */
export async function generateLegacyMultimodalEmbeddings(
    driver: VertexAIDriver,
    options: EmbeddingsOptions,
): Promise<EmbeddingsResult> {
    const model = options.model ?? DEFAULT_LEGACY_MULTIMODAL_MODEL;
    const built = await Promise.all(options.inputs.map(buildLegacyInstance));
    const modalities = built.map((b) => b.modality);

    const instances = built.map((b) => b.instance);
    const parameters = options.dimensions !== undefined ? { dimension: options.dimensions } : undefined;

    try {
        const response = await driver.postVertexModel<{ predictions?: unknown[] }>(model, 'predict', {
            instances,
            parameters,
        });
        const predictions = response.predictions ?? [];
        if (predictions.length !== built.length) {
            throw new Error(
                `Vertex predict returned ${predictions.length} predictions for ${built.length} instances (model ${model})`,
            );
        }

        const items: EmbeddingResultItem[] = predictions.map((value, i) => {
            const decoded = decodePredictionValue(value, i, model);
            return { outputs: predictionToOutputs(decoded, modalities[i]) };
        });

        return buildEmbeddingsResult(model, items);
    } catch (error) {
        if (LlumiverseError.isLlumiverseError(error)) throw error;
        if (error instanceof Error && typeof (error as { status?: unknown }).status !== 'number') throw error;
        throw driver.formatLlumiverseError(error, {
            provider: 'vertexai',
            model,
            operation: 'execute',
        });
    }
}
