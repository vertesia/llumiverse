import type { GoogleGenAI, Model } from '@google/genai';
import { createConversationDocument } from '@llumiverse/conversation';
import { type CanonicalExecutionInputOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from './index.js';

type AIPlatformClient = Awaited<ReturnType<VertexAIDriver['getAIPlatformClient']>>;
type ModelGardenClient = Awaited<ReturnType<VertexAIDriver['getModelGardenClient']>>;
type AnthropicClient = Awaited<ReturnType<VertexAIDriver['getAnthropicClient']>>;

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
                            { name: 'publishers/google/models/gemini-robotics-er-2-preview-info' },
                            { name: 'publishers/google/models/gemini-3.5-live-translate-preview' },
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

class HostBarrierVertexAIDriver extends VertexAIDriver {
    readonly nativeRequest = vi.fn(() => {
        throw new Error('Native Vertex Claude transport must not open before host acceptance');
    });

    constructor() {
        super({ project: 'test-project', region: 'us-central1' });
    }

    override async getAnthropicClient(): Promise<AnthropicClient> {
        return { messages: { stream: this.nativeRequest } } as unknown as AnthropicClient;
    }
}

function canonicalClaudeOptions(attempt: string): CanonicalExecutionInputOptions {
    const conversation = createConversationDocument({
        id: `conversation:${attempt}`,
        created_at: '2026-10-02T00:00:00.000Z',
    });
    return {
        model: 'locations/global/publishers/anthropic/models/claude-sonnet-4-5',
        conversation,
        conversation_runtime: {
            conversation_id: conversation.id,
            request_id: `request:${attempt}`,
            attempt_id: `attempt:${attempt}`,
            input_operation_id: `input:${attempt}`,
            response_operation_id: `response:${attempt}`,
            recorded_at: '2026-10-02T00:00:00.000Z',
            purpose: 'conversation',
        },
    };
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
        expect(modelIds).not.toContain('publishers/google/models/gemini-robotics-er-2-preview-info');
        expect(modelIds).not.toContain('publishers/google/models/gemini-3.5-live-translate-preview');
        expect(modelIds).toContain('publishers/google/models/gemini-4-tts');
    });
});

describe('VertexAIDriver storage permission errors', () => {
    it.each(['gemini-2.5-flash', 'gemini-omni-flash-preview'])(
        'keeps %s storage denials non-retryable and explains who can repair access',
        (model) => {
            const cause = Object.assign(new Error('Permission denied: storage.objects.get on gs://test-bucket/input'), {
                status: 403,
            });
            const error = new VertexAIDriver({ project: 'test-project', region: 'us-central1' }).formatLlumiverseError(
                cause,
                { provider: 'vertexai', model, operation: 'execute' },
            );
            expect(error.code).toBe(403);
            expect(error.retryable).toBe(false);
            expect(error.message).toContain('Ask your project administrator');
            expect(error.message).toContain('configured storage principal');
        },
    );

    it('does not suggest changing bucket IAM for unrelated Vertex authorization failures', () => {
        const error = new VertexAIDriver({ project: 'test-project', region: 'us-central1' }).formatLlumiverseError(
            { status: 403, message: 'Permission denied: aiplatform.endpoints.predict' },
            { provider: 'vertexai', model: 'gemini-2.5-flash', operation: 'execute' },
        );
        expect(error.message).not.toContain('bucket permissions');
    });
});

describe('VertexAIDriver canonical host callback failures', () => {
    it.each([403, 409, 413, 422, 500])(
        'preserves exact host status %i before Vertex Claude transport',
        async (status) => {
            const driver = new HostBarrierVertexAIDriver();
            const hostError = Object.freeze(Object.assign(new Error('host barrier rejected'), { status }));
            const options = {
                ...canonicalClaudeOptions(`barrier-${status}`),
                on_canonical_request_prepared: async () => {
                    throw hostError;
                },
            };

            await expect(
                driver.executeCanonical([{ role: PromptRole.user, content: 'Answer.' }], options),
            ).rejects.toBe(hostError);
            expect(driver.nativeRequest).not.toHaveBeenCalled();
        },
    );

    it('preserves exact recovery lookup failure through typed Vertex Claude stream creation', async () => {
        const driver = new HostBarrierVertexAIDriver();
        const hostError = Object.freeze(Object.assign(new Error('accepted recovery lookup rejected'), { status: 409 }));
        const options = {
            ...canonicalClaudeOptions('typed-recovery-barrier'),
            on_canonical_request_prepared: async () => undefined,
            load_recovered_canonical_output: async () => {
                throw hostError;
            },
        };

        await expect(
            driver.streamCanonicalEvents([{ role: PromptRole.user, content: 'Answer.' }], options, undefined, {
                stream_id: 'stream:typed-recovery-barrier',
            }),
        ).rejects.toBe(hostError);
        expect(driver.nativeRequest).not.toHaveBeenCalled();
    });
});
