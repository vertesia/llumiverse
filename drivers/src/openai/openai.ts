import { Providers } from '@llumiverse/core';
import OpenAI from 'openai';
import type { OpenAIDriverOptions } from '../driver-options.js';
import { OpenAIResponsesDriverBase } from './index.js';

export type { OpenAIDataResidency, OpenAIDriverOptions } from '../driver-options.js';

/** The option accepts future regions as strings; the SDK validates the value. */
type OpenAIDataResidencyOption = NonNullable<ConstructorParameters<typeof OpenAI>[0]>['dataResidency'];

export class OpenAIDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.openai;

    constructor(opts: OpenAIDriverOptions) {
        super(opts);
        this.service = new OpenAI({
            apiKey: opts.apiKey,
            // Only set keys are passed: the SDK rejects a `baseURL` key, even undefined, next to `dataResidency`,
            // and a supplied pair is forwarded for the SDK to reject.
            ...(opts.endpoint !== undefined ? { baseURL: opts.endpoint } : {}),
            ...(opts.dataResidency !== undefined
                ? { dataResidency: opts.dataResidency as OpenAIDataResidencyOption }
                : {}),
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
    }
}
