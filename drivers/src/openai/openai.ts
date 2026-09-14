import {
    type CompletionStream,
    type DriverOptions,
    type ExecutionOptions,
    type ExecutionResponse,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import { FallbackCompletionStream } from '@llumiverse/core/driver';
import OpenAI from 'openai';
import { executeOpenAIAudio, openAIAudioTask } from './audio.js';
import { OpenAIResponsesDriverBase } from './index.js';

export interface OpenAIDriverOptions extends DriverOptions {
    /**
     * The OpenAI api key
     */
    apiKey?: string; //type with azure credentials
}

export class OpenAIDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.openai;

    override async execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<OpenAI.Responses.ResponseInputItem[]>> {
        if (!openAIAudioTask(options.model)) return super.execute(segments, options, signal);
        return this.executeFileAudio(segments, options, signal);
    }

    private async executeFileAudio(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<OpenAI.Responses.ResponseInputItem[]>> {
        const start = Date.now();
        const scope = this.createExecutionHttpAgentScope(options, signal !== undefined);
        const abort = () => void scope.abort();
        if (signal?.aborted) abort();
        else signal?.addEventListener('abort', abort, { once: true });
        try {
            const completion = await scope.run(() =>
                executeOpenAIAudio(this.service, segments, options, this.getDriverRequestOptions(options, signal)),
            );
            // Multipart files and provider binary responses never become a debug prompt or conversation.
            return { ...completion, prompt: [], execution_time: Date.now() - start };
        } catch (error) {
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model: options.model,
                operation: 'execute',
            });
        } finally {
            signal?.removeEventListener('abort', abort);
            await scope.close();
        }
    }

    override async stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<OpenAI.Responses.ResponseInputItem[]>> {
        if (!openAIAudioTask(options.model)) return super.stream(segments, options, signal);
        signal?.throwIfAborted();
        return new FallbackCompletionStream(this, [], options, (streamSignal) =>
            this.executeFileAudio(segments, options, signal ? AbortSignal.any([signal, streamSignal]) : streamSignal),
        );
    }

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
