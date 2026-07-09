import { getTokenProvider } from '@aws/bedrock-token-generator';
import type { AwsCredentialIdentity, Provider } from '@aws-sdk/types';
import { type DriverOptions, type ExecutionOptions, Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import { BedrockOpenAI } from 'openai';
import { OpenAIResponsesDriverBase } from '../openai/index.js';

export type BedrockMantlePrompt = OpenAI.Responses.ResponseInputItem[];

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
    }

    supportsStreaming(options: ExecutionOptions): Promise<boolean> {
        return this.canStream(options);
    }
}
