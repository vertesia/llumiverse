import { getTokenProvider } from '@aws/bedrock-token-generator';
import type { AwsCredentialIdentity, Provider } from '@aws-sdk/types';
import {
    type AIModel,
    type DriverOptions,
    type ExecutionOptions,
    getBedrockMantleModelFamily,
    getModelCapabilities,
    ModelType,
    modelModalitiesToArray,
    Providers,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { BedrockOpenAI } from 'openai';
import { OpenAIResponsesDriverBase } from '../openai/index.js';

export type BedrockMantlePrompt = OpenAI.Responses.ResponseInputItem[];

interface BedrockMantleModelMetadata {
    name: string;
    owner: string;
}

export interface BedrockMantleDriverOptions extends DriverOptions {
    region: string;
    credentials?: AwsCredentialIdentity | Provider<AwsCredentialIdentity>;
}

function getBedrockMantleModelMetadata(model: string): BedrockMantleModelMetadata | undefined {
    const family = getBedrockMantleModelFamily(model);
    if (family === 'openai') {
        const modelName = model.slice('openai.'.length).replace(/^gpt-/i, 'GPT-');
        return { name: `OpenAI ${modelName}`, owner: 'OpenAI' };
    }
    if (family === 'grok') {
        const modelName = model.slice('xai.'.length).replace(/^grok-/i, 'Grok ');
        return { name: `xAI ${modelName}`, owner: 'xAI' };
    }
    return undefined;
}

export function isBedrockMantleModel(model: string): boolean {
    return getBedrockMantleModelMetadata(model) !== undefined;
}

export class BedrockMantleDriver extends OpenAIResponsesDriverBase {
    service: BedrockOpenAI;
    private modelsService: BedrockOpenAI;
    readonly provider = Providers.bedrock_mantle;

    constructor(opts: BedrockMantleDriverOptions) {
        super(opts);

        const bedrockTokenProvider = opts.credentials
            ? getTokenProvider({ region: opts.region, credentials: opts.credentials })
            : getTokenProvider({ region: opts.region });

        this.service = new BedrockOpenAI({
            awsRegion: opts.region,
            bedrockTokenProvider,
            fetch: this.getDriverFetch(),
        });
        this.modelsService = new BedrockOpenAI({
            baseURL: `https://bedrock-mantle.${opts.region}.api.aws/v1`,
            bedrockTokenProvider,
            fetch: this.getDriverFetch(),
        });
    }

    supportsStreaming(options: ExecutionOptions): Promise<boolean> {
        return this.canStream(options);
    }

    async listModels(): Promise<AIModel[]> {
        const models = (await this.modelsService.models.list()).data;
        return models
            .filter((model) => isBedrockMantleModel(model.id))
            .map((model) => {
                const metadata = getBedrockMantleModelMetadata(model.id);
                const modelCapability = getModelCapabilities(model.id, this.provider);
                return {
                    id: model.id,
                    name: metadata?.name ?? model.id,
                    provider: this.provider,
                    owner: metadata?.owner ?? model.owned_by,
                    type: ModelType.Text,
                    can_stream: true,
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                } satisfies AIModel<string>;
            })
            .sort((a, b) => a.id.localeCompare(b.id));
    }
}
