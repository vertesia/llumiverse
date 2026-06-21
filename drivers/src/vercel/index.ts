import { type AIModel, type DriverOptions, Providers } from '@llumiverse/core';
import { OpenAICompatibleDriver } from '../openai/openai_compatible.js';

export interface VercelAIGatewayDriverOptions extends DriverOptions {
    /** Vercel AI Gateway API key. */
    apiKey: string;
    /** Override the Vercel AI Gateway OpenAI-compatible endpoint. */
    endpoint?: string;
    default_headers?: Record<string, string>;
}

export class VercelAIGatewayDriver extends OpenAICompatibleDriver {
    static readonly PROVIDER = Providers.vercel_ai_gateway;
    readonly provider = Providers.vercel_ai_gateway;

    constructor(options: VercelAIGatewayDriverOptions) {
        super({
            ...options,
            endpoint: options.endpoint ?? 'https://ai-gateway.vercel.sh/v1',
        });
    }

    async listModels(): Promise<AIModel[]> {
        const models = await super.listModels();
        return models.map((model) => ({ ...model, provider: this.provider }));
    }
}
