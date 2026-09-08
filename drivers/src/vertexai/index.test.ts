import type { GoogleGenAI, Model } from '@google/genai';
import { describe, expect, it } from 'vitest';
import { VertexAIDriver } from './index.js';

type AIPlatformClient = Awaited<ReturnType<VertexAIDriver['getAIPlatformClient']>>;
type ModelGardenClient = Awaited<ReturnType<VertexAIDriver['getModelGardenClient']>>;

class TestVertexAIDriver extends VertexAIDriver {
    constructor(private readonly googleModels: Model[] = []) {
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
                if (parent === 'publishers/google') {
                    return [
                        [
                            { name: 'publishers/google/models/gemini-4-future' },
                            { name: 'publishers/google/models/gemini-omni-flash-preview' },
                            { name: 'publishers/google/models/gemini-omni-1.1-flash-preview' },
                            { name: 'publishers/google/models/gemini-live-future' },
                            { name: 'publishers/google/models/gemini-4-tts' },
                        ],
                    ];
                }
                return [[]];
            },
        } as unknown as ModelGardenClient;
    }

    override getGoogleGenAIClient(): GoogleGenAI {
        return {} as GoogleGenAI;
    }

    override async getGenAIModelsArray(_client: GoogleGenAI): Promise<Model[]> {
        return this.googleModels;
    }
}

describe('VertexAIDriver listModels', () => {
    it('lists xAI publisher models with global location ids', async () => {
        const models = await new TestVertexAIDriver().listModels();
        const modelIds = models.map((model) => model.id);

        expect(modelIds).toContain('locations/global/publishers/xai/models/grok-4.1');
        expect(modelIds).not.toContain('publishers/xai/models/grok-4.1');
    });

    it('lists Gemini Omni only with its global location id', async () => {
        const models = await new TestVertexAIDriver().listModels();
        const omniModels = models.filter((model) => model.id.includes('gemini-omni'));

        expect(omniModels).toEqual([
            expect.objectContaining({
                id: 'locations/global/publishers/google/models/gemini-omni-1.1-flash-preview',
                name: 'Global gemini-omni-1.1-flash-preview',
            }),
            expect.objectContaining({
                id: 'locations/global/publishers/google/models/gemini-omni-flash-preview',
                name: 'Global gemini-omni-flash-preview',
            }),
        ]);
    });

    it('uses supported actions to keep only models executable by the implemented Google paths', async () => {
        const driver = new TestVertexAIDriver([
            { name: 'models/gemini-4-future', supportedActions: ['generateContent'] },
            { name: 'models/gemini-4-unannounced' },
            { name: 'models/gemini-live-preview', supportedActions: ['bidiGenerateContent'] },
            { name: 'models/gemini-tts-preview', supportedActions: ['predict'] },
            { name: 'models/text-embedding-future', supportedActions: ['embedContent'] },
            { name: 'models/imagen-5', supportedActions: ['generateImages'] },
            { name: 'models/veo-4', supportedActions: ['generateVideos'] },
        ]);

        const modelIds = (await driver.listModels()).map((model) => model.id);

        expect(modelIds).toContain('locations/global/models/gemini-4-future');
        expect(modelIds).toContain('locations/global/models/gemini-4-unannounced');
        expect(modelIds).toContain('locations/global/models/imagen-5');
        expect(modelIds).not.toContain('locations/global/models/gemini-live-preview');
        expect(modelIds).not.toContain('locations/global/models/gemini-tts-preview');
        expect(modelIds).not.toContain('locations/global/models/text-embedding-future');
        expect(modelIds).not.toContain('locations/global/models/veo-4');
        expect(modelIds).not.toContain('publishers/google/models/gemini-live-future');
        expect(modelIds).not.toContain('publishers/google/models/gemini-4-tts');
    });
});
