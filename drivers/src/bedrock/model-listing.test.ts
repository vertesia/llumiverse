import {
    Bedrock,
    type FoundationModelSummary,
    type ListCustomModelsCommandInput,
    type ListCustomModelsCommandOutput,
    type ListFoundationModelsCommandInput,
    type ListFoundationModelsCommandOutput,
    type ListInferenceProfilesCommandInput,
    type ListInferenceProfilesCommandOutput,
} from '@aws-sdk/client-bedrock';
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
    // Titan Image uses a different request schema; the current image execution path is Nova Canvas only.
    { id: 'amazon.titan-image-generator-v3', discoverable: false },
    { id: 'amazon.nova-canvas-v2:0', discoverable: true },
    { id: 'unverified.model-1', discoverable: false },
] as const;

function foundationModel(modelId: string, modelName = modelId): FoundationModelSummary {
    return {
        modelArn: undefined,
        modelId,
        modelName,
        providerName: modelId.split('.')[0],
        inferenceTypesSupported: ['ON_DEMAND'],
        inputModalities: ['TEXT'],
        outputModalities: ['TEXT'],
        responseStreamingSupported: true,
    };
}

const emptyCustomModels = {
    $metadata: {},
    modelSummaries: [],
} satisfies ListCustomModelsCommandOutput;

const emptyInferenceProfiles = {
    $metadata: {},
    inferenceProfileSummaries: [],
} satisfies ListInferenceProfilesCommandOutput;

function createMockService(): Bedrock {
    return new Bedrock({
        region: 'us-east-2',
        credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
    });
}

function mockModelListingMethods(
    service: Bedrock,
    foundationModels: ListFoundationModelsCommandOutput,
    customModels: ListCustomModelsCommandOutput = emptyCustomModels,
): void {
    const listFoundationModels = vi.fn(
        async (_input: ListFoundationModelsCommandInput): Promise<ListFoundationModelsCommandOutput> =>
            foundationModels,
    );
    const listCustomModels = vi.fn(
        async (_input: ListCustomModelsCommandInput): Promise<ListCustomModelsCommandOutput> => customModels,
    );
    const listInferenceProfiles = vi.fn(
        async (_input: ListInferenceProfilesCommandInput): Promise<ListInferenceProfilesCommandOutput> =>
            emptyInferenceProfiles,
    );
    Reflect.set(service, 'listFoundationModels', listFoundationModels);
    Reflect.set(service, 'listCustomModels', listCustomModels);
    Reflect.set(service, 'listInferenceProfiles', listInferenceProfiles);
}

describe('Bedrock Converse model discovery', () => {
    it('inherits discovery support for future models from known publishers', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const service = createMockService();
        const foundationModels = {
            $metadata: {},
            modelSummaries: DISCOVERY_CASES.map(({ id }) => foundationModel(id)),
        } satisfies ListFoundationModelsCommandOutput;
        mockModelListingMethods(service, foundationModels);
        vi.spyOn(driver, 'getService').mockReturnValue(service);

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
        const service = createMockService();
        const foundationModels = {
            $metadata: {},
            modelSummaries: [
                {
                    ...foundationModel('deepseek.v3.2', 'DeepSeek V3.2'),
                    providerName: 'DeepSeek',
                },
            ],
        } satisfies ListFoundationModelsCommandOutput;
        mockModelListingMethods(service, foundationModels);
        vi.spyOn(driver, 'getService').mockReturnValue(service);

        expect(await driver.listModels()).toEqual([
            expect.objectContaining({ name: 'DeepSeek V3.2', owner: 'DeepSeek' }),
        ]);
    });

    it('uses AWS modalities only when the catalog has no model information', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const service = createMockService();
        const catalogedNemotron = {
            ...foundationModel('nvidia.nemotron-nano-12b-v2-vl-bf16'),
            inputModalities: ['TEXT'],
            outputModalities: ['TEXT', 'IMAGE'],
        } satisfies FoundationModelSummary;
        const futureModel = {
            ...foundationModel('nvidia.future-foundation-model'),
            inputModalities: ['TEXT', 'IMAGE'],
            outputModalities: ['TEXT'],
        } satisfies FoundationModelSummary;
        mockModelListingMethods(service, {
            $metadata: {},
            modelSummaries: [catalogedNemotron, futureModel],
        });
        vi.spyOn(driver, 'getService').mockReturnValue(service);

        expect(await driver.listModels()).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: catalogedNemotron.modelId,
                    input_modalities: ['text', 'image'],
                    output_modalities: ['text'],
                }),
                expect.objectContaining({
                    id: futureModel.modelId,
                    input_modalities: ['text', 'image'],
                    output_modalities: ['text'],
                }),
            ]),
        );
    });

    it('excludes foundation embeddings using explicit AWS output modality metadata', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const service = createMockService();
        const embedding = {
            ...foundationModel('amazon.titan-embed-text-v2:0'),
            outputModalities: ['EMBEDDING'],
        } satisfies FoundationModelSummary;
        const foundationModels = {
            $metadata: {},
            modelSummaries: [foundationModel('amazon.nova-pro-v1:0'), embedding],
        } satisfies ListFoundationModelsCommandOutput;
        mockModelListingMethods(service, foundationModels);
        vi.spyOn(driver, 'getService').mockReturnValue(service);

        expect((await driver.listModels()).map((model) => model.id)).toEqual(['amazon.nova-pro-v1:0']);
    });

    it('derives custom-model capabilities from the reported base model identity', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const service = createMockService();
        const customModels = {
            $metadata: {},
            modelSummaries: [
                {
                    modelArn: 'arn:aws:bedrock:us-east-2:123456789012:custom-model/opaque-name',
                    modelName: 'opaque-name',
                    creationTime: new Date(0),
                    baseModelArn: 'arn:aws:bedrock:us-east-2::foundation-model/openai.gpt-5.6-sol',
                    baseModelName: 'openai.gpt-5.6-sol',
                },
            ],
        } satisfies ListCustomModelsCommandOutput;
        mockModelListingMethods(service, { $metadata: {}, modelSummaries: [] }, customModels);
        vi.spyOn(driver, 'getService').mockReturnValue(service);

        expect(await driver.listModels()).toEqual([
            expect.objectContaining({
                id: 'arn:aws:bedrock:us-east-2:123456789012:custom-model/opaque-name',
                input_modalities: ['text', 'image'],
                output_modalities: ['text'],
            }),
        ]);
    });
});
