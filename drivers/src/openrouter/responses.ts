import { Providers } from '@llumiverse/core';
import { serverURLFromOptions } from '@openrouter/sdk/lib/config';
import { ProviderPreferences$outboundSchema } from '@openrouter/sdk/models';
import type { OpenRouterDriverOptions } from '../driver-options.js';
import type { OpenAIResponsesPolicy } from '../openai/index.js';
import { OPENAI_COMPATIBLE_RESPONSES_POLICY } from '../openai/openai_responses.js';
import { ProviderResponsesDriver } from '../openai/provider_responses.js';
import { OpenRouterDriver, toOpenRouterProviderPreferences } from './index.js';

const OPENROUTER_RESPONSES_POLICY: OpenAIResponsesPolicy = {
    ...OPENAI_COMPATIBLE_RESPONSES_POLICY,
    // OpenRouter takes the same `provider` routing object on Responses as on Chat Completions. The SDK's own schema
    // writes the SDK's preferences in the wire format.
    requestBody: (options) => {
        const preferences = toOpenRouterProviderPreferences(options.model_options);
        return preferences ? { provider: ProviderPreferences$outboundSchema.parse(preferences) } : undefined;
    },
};

/**
 * OpenRouter on its Responses API. The API is stateless, as the Responses driver uses it: every request carries the
 * whole conversation.
 */
export class OpenRouterResponsesDriver extends ProviderResponsesDriver {
    readonly provider = Providers.openrouter;
    override readonly responsesPolicy = OPENROUTER_RESPONSES_POLICY;

    constructor(options: OpenRouterDriverOptions) {
        // The server the SDK would call, so an `endpoint` override applies to both APIs.
        const serverURL = serverURLFromOptions({ serverURL: options.endpoint });
        if (!serverURL) throw new Error('OpenRouter server URL could not be resolved');
        super(options, new OpenRouterDriver(options), {
            apiKey: options.apiKey,
            baseURL: serverURL.href,
            // The attribution headers the SDK sends with every chat request.
            defaultHeaders: {
                ...(options.httpReferer ? { 'HTTP-Referer': options.httpReferer } : {}),
                ...(options.appTitle ? { 'X-OpenRouter-Title': options.appTitle } : {}),
                ...(options.appCategories ? { 'X-OpenRouter-Categories': options.appCategories } : {}),
            },
        });
    }
}
