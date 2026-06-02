import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { VertexAIDriver } from '../src/index.js';

const TIMEOUT = 30000;
const TEXT = 'Hello world';

const VERTEX_TEXT_MODELS = [
    'gemini-embedding-001',
    'gemini-embedding-2',
    'text-embedding-005',
    'text-embedding-004',
    //"text-embedding-003",  Confirmed deprecated
    //"embedding-001", Confirmed deprecated
    'multimodalembedding@001',
    'text-multilingual-embedding-002',
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
    describe('VertexAI text embedding model coverage via REST', () => {
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
