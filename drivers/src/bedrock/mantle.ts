import { getTokenProvider } from '@aws/bedrock-token-generator';
import type { AwsCredentialIdentity, Provider } from '@aws-sdk/types';
import {
    type AIModel,
    type DriverOptions,
    type ExecutionOptions,
    getModelCapabilities,
    ModelType,
    modelModalitiesToArray,
    Providers,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { BedrockOpenAI } from 'openai';
import { OpenAIResponsesDriverBase } from '../openai/index.js';

export type BedrockMantlePrompt = OpenAI.Responses.ResponseInputItem[];

const BEDROCK_MANTLE_MODEL_NAMES: Record<string, string> = {
    'openai.gpt-5.5': 'OpenAI GPT-5.5',
    'openai.gpt-5.4': 'OpenAI GPT-5.4',
};

export interface BedrockMantleDriverOptions extends DriverOptions {
    region: string;
    credentials?: AwsCredentialIdentity | Provider<AwsCredentialIdentity>;
}

export function isBedrockMantleModel(model: string): boolean {
    const normalized = model.toLowerCase();
    return normalized === 'openai.gpt-5.5' || normalized === 'openai.gpt-5.4';
}

export class BedrockMantleDriver extends OpenAIResponsesDriverBase {
    service: BedrockOpenAI;
    private modelsService: BedrockOpenAI;
    readonly provider = Providers.bedrock;

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
                const modelCapability = getModelCapabilities(model.id, this.provider);
                return {
                    id: model.id,
                    name: BEDROCK_MANTLE_MODEL_NAMES[model.id.toLowerCase()] ?? model.id,
                    provider: this.provider,
                    owner: 'OpenAI',
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
