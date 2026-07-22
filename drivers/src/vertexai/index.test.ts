import type { GoogleGenAI, Model } from '@google/genai';
import { describe, expect, it } from 'vitest';
import { VertexAIDriver } from './index.js';

type AIPlatformClient = Awaited<ReturnType<VertexAIDriver['getAIPlatformClient']>>;
type ModelGardenClient = Awaited<ReturnType<VertexAIDriver['getModelGardenClient']>>;

class TestVertexAIDriver extends VertexAIDriver {
    constructor() {
        super({ project: 'test-project', region: 'us-central1' });
    }

    override async getAIPlatformClient(): Promise<AIPlatformClient> {
        return {
            listModels: async () => [[]],
        } as unknown as AIPlatformClient;
    }

    override async getModelGardenClient(): Promise<ModelGardenClient> {
        return {
            listPublisherModels: async ({ parent }: { parent: string }) => {
                if (parent === 'publishers/xai') {
                    return [[{ name: 'publishers/xai/models/grok-4.1' }]];
                }
                return [[]];
            },
        } as unknown as ModelGardenClient;
    }

    override getGoogleGenAIClient(): GoogleGenAI {
        return {} as GoogleGenAI;
    }

    override async getGenAIModelsArray(_client: GoogleGenAI): Promise<Model[]> {
        return [];
    }
}

describe('VertexAIDriver listModels', () => {
    it('lists xAI publisher models with global location ids', async () => {
        const models = await new TestVertexAIDriver().listModels();
        const modelIds = models.map((model) => model.id);

        expect(modelIds).toContain('locations/global/publishers/xai/models/grok-4.1');
        expect(modelIds).not.toContain('publishers/xai/models/grok-4.1');
    });
});
