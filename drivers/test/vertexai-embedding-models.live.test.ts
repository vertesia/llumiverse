import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { VertexAIDriver } from '../src/index.js';

const TIMEOUT = 30000;
const TEXT = 'Hello world';

// One model per Vertex request path; gemini-embedding-2 (the default) is covered by embeddings.live.test.ts.
const VERTEX_TEXT_MODELS = [
    'gemini-embedding-001', // embedContent, regional, one input per call
    'text-embedding-005', // embedContent, regional, grouped inputs
    'multimodalembedding@001', // legacy multimodal predict
] as const;

let vertex: VertexAIDriver | undefined;
if (process.env.GOOGLE_REGION && process.env.GOOGLE_PROJECT_ID) {
    vertex = new VertexAIDriver({
        project: process.env.GOOGLE_PROJECT_ID,
        region: process.env.GOOGLE_REGION,
    });
}

if (!vertex) {
    describe.skip('VertexAI text embedding model coverage (set GOOGLE_REGION / GOOGLE_PROJECT_ID to enable)', () => {
        test('placeholder', () => undefined);
    });
} else {
    describe('VertexAI text embedding model coverage via @google/genai', () => {
        for (const model of VERTEX_TEXT_MODELS) {
            test(
                `${model} — text-only single input`,
                async () => {
                    const result = await vertex?.generateEmbeddings({
                        inputs: [{ type: 'text', text: TEXT }],
                        model,
                    });

                    expect(result.results).toHaveLength(1);
                    const values = result.results[0]?.outputs[0]?.values;
                    expect(Array.isArray(values)).toBe(true);
                    expect(values.length).toBeGreaterThan(0);
                    expect(values.every((v) => typeof v === 'number' && Number.isFinite(v))).toBe(true);
                    expect(result.model).toBe(model);
                },
                TIMEOUT,
            );
        }
    });
}
