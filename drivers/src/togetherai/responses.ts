import { Providers } from '@llumiverse/core';
import type { TogetherAIDriverOptions } from '../driver-options.js';
import type { OpenAIResponsesPolicy } from '../openai/index.js';
import { ProviderResponsesDriver } from '../openai/provider_responses.js';
import { TogetherAIDriver } from './index.js';

const TOGETHER_RESPONSES_POLICY: OpenAIResponsesPolicy = {
    // The result schema goes in the prompt, as on Together's Chat Completions driver.
    supportsSchema: () => false,
};

/**
 * Together AI on its Responses API. Together documents it for a few models only, and rejects file inputs on every
 * model (https://docs.together.ai/docs/inference/responses-api), so Chat Completions stays the default.
 */
export class TogetherAIResponsesDriver extends ProviderResponsesDriver {
    readonly provider = Providers.togetherai;
    override readonly responsesPolicy = TOGETHER_RESPONSES_POLICY;

    constructor(options: TogetherAIDriverOptions) {
        const together = new TogetherAIDriver(options);
        super(options, together, {
            apiKey: options.apiKey,
            // The Together SDK's base URL (an `endpoint` override or its default) and the path its own requests use.
            baseURL: `${together.service.baseURL.replace(/\/+$/, '')}/v1`,
        });
    }
}
