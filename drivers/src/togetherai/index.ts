import { type DriverOptions, Providers } from '@llumiverse/core';
import OpenAI from 'openai';
import { BaseOpenAIDriver } from '../openai/index.js';

export interface TogetherAIDriverOptions extends DriverOptions {
    apiKey: string;
    endpoint?: string;
}

export class TogetherAIDriver extends BaseOpenAIDriver {
    static readonly PROVIDER = Providers.togetherai;
    readonly provider = Providers.togetherai;
    service: OpenAI;

    constructor(opts: TogetherAIDriverOptions) {
        super(opts);

        if (!opts.apiKey) {
            throw new Error('apiKey is required');
        }

        this.service = new OpenAI({
            apiKey: opts.apiKey,
            baseURL: opts.endpoint ?? 'https://api.together.ai/v1',
            fetch: this.getDriverFetch(),
        });
    }
}
