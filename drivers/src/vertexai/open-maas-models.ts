import { type AIModel, getModelCapabilities, modelModalitiesToArray } from '@llumiverse/core';

export interface VertexOpenMaaSModel {
    publisher: string;
    model: string;
    requestPublisher: string;
    requestModel?: string;
    regions: readonly string[];
    apiVersion?: string;
    extraBody?: Record<string, unknown>;
}

const GLOBAL_REGIONS = ['global'] as const;
const US_CENTRAL1_REGIONS = ['us-central1'] as const;
const US_EAST5_REGIONS = ['us-east5'] as const;
const US_SOUTH1_AND_GLOBAL_REGIONS = ['global', 'us-south1'] as const;
const US_CENTRAL1_AND_GLOBAL_REGIONS = ['global', 'us-central1'] as const;

const LLAMA_SAFETY_EXTRA_BODY = {
    google: {
        model_safety_settings: {
            enabled: false,
            llama_guard_settings: {},
        },
    },
} satisfies Record<string, unknown>;

export const VERTEX_OPEN_MAAS_MODELS = [
    {
        publisher: 'meta',
        model: 'llama-4-maverick-17b-128e',
        requestPublisher: 'meta',
        requestModel: 'llama-4-maverick-17b-128e-instruct-maas',
        regions: US_EAST5_REGIONS,
        apiVersion: 'v1beta1',
        extraBody: LLAMA_SAFETY_EXTRA_BODY,
    },
    {
        publisher: 'meta',
        model: 'llama-4-scout-17b-16e',
        requestPublisher: 'meta',
        requestModel: 'llama-4-scout-17b-16e-instruct-maas',
        regions: US_EAST5_REGIONS,
        apiVersion: 'v1beta1',
        extraBody: LLAMA_SAFETY_EXTRA_BODY,
    },
    {
        publisher: 'meta',
        model: 'llama-3.3-70b',
        requestPublisher: 'meta',
        requestModel: 'llama-3.3-70b-instruct-maas',
        regions: US_CENTRAL1_REGIONS,
        apiVersion: 'v1beta1',
        extraBody: LLAMA_SAFETY_EXTRA_BODY,
    },
    {
        publisher: 'deepseek-ai',
        model: 'deepseek-ocr-maas',
        requestPublisher: 'deepseek-ai',
        regions: US_CENTRAL1_REGIONS,
    },
    {
        publisher: 'deepseek-ai',
        model: 'deepseek-v3.2-maas',
        requestPublisher: 'deepseek-ai',
        regions: GLOBAL_REGIONS,
    },
    {
        publisher: 'deepseek-ai',
        model: 'deepseek-v3.1-maas',
        requestPublisher: 'deepseek-ai',
        regions: US_CENTRAL1_REGIONS,
    },
    {
        publisher: 'deepseek-ai',
        model: 'deepseek-r1-0528-maas',
        requestPublisher: 'deepseek-ai',
        regions: US_CENTRAL1_REGIONS,
    },
    {
        publisher: 'qwen',
        model: 'qwen3-next-80b-a3b-instruct-maas',
        requestPublisher: 'qwen',
        regions: GLOBAL_REGIONS,
    },
    {
        publisher: 'qwen',
        model: 'qwen3-next-80b-a3b-thinking-maas',
        requestPublisher: 'qwen',
        regions: GLOBAL_REGIONS,
    },
    {
        publisher: 'qwen',
        model: 'qwen3-coder-480b-a35b-instruct-maas',
        requestPublisher: 'qwen',
        regions: US_SOUTH1_AND_GLOBAL_REGIONS,
    },
    {
        publisher: 'qwen',
        model: 'qwen3-235b-a22b-instruct-2507-maas',
        requestPublisher: 'qwen',
        regions: US_SOUTH1_AND_GLOBAL_REGIONS,
    },
    {
        publisher: 'zai-org',
        model: 'glm-5-maas',
        requestPublisher: 'zai-org',
        regions: GLOBAL_REGIONS,
    },
    {
        publisher: 'zai-org',
        model: 'glm-4.7-maas',
        requestPublisher: 'zai-org',
        regions: GLOBAL_REGIONS,
    },
    {
        publisher: 'moonshotai',
        model: 'kimi-k2-thinking-maas',
        requestPublisher: 'moonshotai',
        regions: GLOBAL_REGIONS,
    },
    {
        publisher: 'minimaxai',
        model: 'minimax-m2-maas',
        requestPublisher: 'minimaxai',
        regions: GLOBAL_REGIONS,
    },
    {
        publisher: 'openai',
        model: 'gpt-oss-120b-maas',
        requestPublisher: 'openai',
        regions: US_CENTRAL1_AND_GLOBAL_REGIONS,
    },
    {
        publisher: 'openai',
        model: 'gpt-oss-20b-maas',
        requestPublisher: 'openai',
        regions: US_CENTRAL1_REGIONS,
    },
    {
        publisher: 'google',
        model: 'gemma-4-26b-a4b-it-maas',
        requestPublisher: 'google',
        regions: GLOBAL_REGIONS,
    },
] as const satisfies readonly VertexOpenMaaSModel[];

export function getVertexOpenMaaSModel(publisher: string | undefined, model: string): VertexOpenMaaSModel | undefined {
    if (!publisher) return undefined;
    const normalizedPublisher = publisher === 'zaiorg' ? 'zai-org' : publisher;
    return VERTEX_OPEN_MAAS_MODELS.find((entry) => entry.publisher === normalizedPublisher && entry.model === model);
}

export function getVertexOpenMaaSRequestModel(
    publisher: string | undefined,
    model: string,
): { modelName: string; apiVersion?: string; extraBody?: Record<string, unknown> } | undefined {
    const entry = getVertexOpenMaaSModel(publisher, model);
    if (entry) {
        return {
            modelName: `${entry.requestPublisher}/${entry.requestModel ?? entry.model}`,
            apiVersion: entry.apiVersion,
            extraBody: entry.extraBody,
        };
    }

    if (publisher === 'xai') {
        return { modelName: `xai/${model}` };
    }

    return undefined;
}

export function vertexOpenMaaSModelToAIModel(entry: VertexOpenMaaSModel, region: string): AIModel<string> {
    const id = `locations/${region}/publishers/${entry.publisher}/models/${entry.model}`;
    const modelCapability = getModelCapabilities(entry.model, 'vertexai');
    return {
        id,
        name: region === 'global' ? `Global ${entry.model}` : entry.model,
        provider: 'vertexai',
        owner: entry.publisher,
        input_modalities: modelModalitiesToArray(modelCapability.input),
        output_modalities: modelModalitiesToArray(modelCapability.output),
        tool_support: modelCapability.tool_support,
    } satisfies AIModel<string>;
}

export function getListedVertexOpenMaaSModels(region: string): AIModel<string>[] {
    return (VERTEX_OPEN_MAAS_MODELS as readonly VertexOpenMaaSModel[]).flatMap((entry) => {
        const listingRegions = entry.regions.includes('global') ? ['global'] : [];
        if (region !== 'global' && entry.regions.includes(region)) {
            listingRegions.push(region);
        }
        return listingRegions.map((listingRegion) => vertexOpenMaaSModelToAIModel(entry, listingRegion));
    });
}
