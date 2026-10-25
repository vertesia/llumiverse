import { Providers } from '@llumiverse/core';
import OpenAI from 'openai';
import type { OpenAIDriverOptions } from '../driver-options.js';
import { OpenAIResponsesDriverBase } from './index.js';

export type { OpenAIDriverOptions } from '../driver-options.js';

export class OpenAIDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.openai;

    constructor(opts: OpenAIDriverOptions) {
        super(opts);
        this.service = new OpenAI({
            apiKey: opts.apiKey,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
    }
}
