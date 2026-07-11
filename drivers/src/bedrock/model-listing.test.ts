import type { Bedrock } from '@aws-sdk/client-bedrock';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';

const DISCOVERY_CASES = [
    { id: 'ai21.jamba-2-large-v1:0', discoverable: true },
    { id: 'amazon.nova-3-lite-v1:0', discoverable: true },
    { id: 'anthropic.claude-sonnet-6', discoverable: true },
    { id: 'cohere.command-r-next-v1:0', discoverable: true },
    { id: 'deepseek.v4', discoverable: true },
    { id: 'google.gemma-3-30b-it', discoverable: true },
    { id: 'meta.llama5-scout-instruct-v1:0', discoverable: true },
    { id: 'minimax.minimax-m3', discoverable: true },
    { id: 'mistral.mistral-large-4-instruct', discoverable: true },
    { id: 'moonshot.kimi-k3-thinking', discoverable: true },
    { id: 'moonshotai.kimi-k3', discoverable: true },
    { id: 'nvidia.nemotron-super-4-180b', discoverable: true },
    { id: 'openai.gpt-oss-200b-1:0', discoverable: true },
    { id: 'qwen.qwen4-coder', discoverable: true },
    { id: 'writer.palmyra-x6-v1:0', discoverable: true },
    { id: 'zai.glm-6', discoverable: true },
    { id: 'amazon.titan-image-generator-v3', discoverable: false },
    { id: 'unverified.model-1', discoverable: false },
] as const;

describe('Bedrock Converse model discovery', () => {
    it('inherits discovery support for future models from known publishers', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const service = {
            listFoundationModels: vi.fn(async () => ({
                modelSummaries: DISCOVERY_CASES.map(({ id: modelId }) => ({
                    modelId,
                    modelName: modelId,
                    providerName: modelId.split('.')[0],
                    inferenceTypesSupported: ['ON_DEMAND'],
                    inputModalities: ['TEXT'],
                    outputModalities: ['TEXT'],
                    responseStreamingSupported: true,
                })),
            })),
            listCustomModels: vi.fn(async () => ({ modelSummaries: [] })),
            listInferenceProfiles: vi.fn(async () => ({ inferenceProfileSummaries: [] })),
        };
        vi.spyOn(driver, 'getService').mockReturnValue(service as unknown as Bedrock);

        const discoveredIds = (await driver.listModels()).map((model) => model.id);

        for (const { id, discoverable } of DISCOVERY_CASES) {
            if (discoverable) {
                expect(discoveredIds).toContain(id);
            } else {
                expect(discoveredIds).not.toContain(id);
            }
        }
    });

    it('uses the AWS model name without repeating the publisher', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const service = {
            listFoundationModels: vi.fn(async () => ({
                modelSummaries: [
                    {
                        modelId: 'deepseek.v3.2',
                        modelName: 'DeepSeek V3.2',
                        providerName: 'DeepSeek',
                        inferenceTypesSupported: ['ON_DEMAND'],
                        inputModalities: ['TEXT'],
                        outputModalities: ['TEXT'],
                    },
                ],
            })),
            listCustomModels: vi.fn(async () => ({ modelSummaries: [] })),
            listInferenceProfiles: vi.fn(async () => ({ inferenceProfileSummaries: [] })),
        };
        vi.spyOn(driver, 'getService').mockReturnValue(service as unknown as Bedrock);

        expect(await driver.listModels()).toEqual([
            expect.objectContaining({ name: 'DeepSeek V3.2', owner: 'DeepSeek' }),
        ]);
    });
});
