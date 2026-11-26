import { type Completion, type DriverCompletionStream, type ExecutionOptions, Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import type { GroqDriverOptions } from '../driver-options.js';
import type { OpenAIResponsesPolicy } from '../openai/index.js';
import { ProviderResponsesDriver } from '../openai/provider_responses.js';
import { logModelOptionException } from '../shared/model-option-exceptions.js';
import { GroqDriver } from './index.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

const GROQ_RESPONSES_POLICY: OpenAIResponsesPolicy = {
    // The result schema goes in the prompt, as on Groq's Chat Completions driver: Groq enforces a native schema on a
    // few models only, and rejects one together with tools.
    supportsSchema: () => false,
    // Compatibility exception: Groq lists `include` and `prompt_cache_key` as unsupported on its Responses API
    // (https://console.groq.com/docs/responses-api), and the Responses driver sets both itself: `include` on every
    // reasoning request, `prompt_cache_key` whenever the caller caches prompts. The other fields Groq lists are left
    // for Groq to reject, since only a caller's `extra_body` sends them.
    omittedRequestFields: ['include', 'prompt_cache_key'],
};

/** Groq on its Responses API, which Groq serves at the OpenAI-compatible path of its API. */
export class GroqResponsesDriver extends ProviderResponsesDriver {
    readonly provider = Providers.groq;
    override readonly responsesPolicy = GROQ_RESPONSES_POLICY;

    constructor(options: GroqDriverOptions) {
        const groq = new GroqDriver(options);
        super(options, groq, {
            apiKey: options.apiKey,
            // The Groq SDK's base URL (an `endpoint_url` override or its default) and the path its own requests use.
            baseURL: `${groq.client.baseURL.replace(/\/+$/, '')}/openai/v1`,
        });
    }

    override requestTextCompletion(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        this.logOptionsWithoutResponsesField(options);
        return super.requestTextCompletion(prompt, options, signal);
    }

    override requestTextCompletionStream(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        this.logOptionsWithoutResponsesField(options);
        return super.requestTextCompletionStream(prompt, options, signal);
    }

    /**
     * Compatibility exception: the Responses API has no field for `reasoning_format`, which the Chat Completions
     * driver sends for Groq's DeepSeek thinking models, so it is not sent here.
     */
    private logOptionsWithoutResponsesField(options: ExecutionOptions): void {
        logModelOptionException(
            this.logger,
            options.model,
            options.model_options,
            ['reasoning_format'],
            'groq_reasoning_format_without_responses_field',
        );
    }
}
