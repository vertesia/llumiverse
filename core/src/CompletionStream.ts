import { AbstractDriver } from "./Driver.js";
import { CompletionStream, DriverOptions, ExecutionOptions, ExecutionResponse, ExecutionTokenUsage } from "@llumiverse/common";

export class DefaultCompletionStream<PromptT = any> implements CompletionStream<PromptT> {

    chunks: string[];
    completion: ExecutionResponse<PromptT> | undefined;

    constructor(public driver: AbstractDriver<DriverOptions, PromptT>,
        public prompt: PromptT,
        public options: ExecutionOptions) {
        this.chunks = [];
    }

    async *[Symbol.asyncIterator]() {
        // reset state
        this.completion = undefined;
        if (this.chunks.length > 0) {
            this.chunks = [];
        }
        const chunks = this.chunks;

        this.driver.logger.debug(
            `[${this.driver.provider}] Streaming Execution of ${this.options.model} with prompt`, this.prompt,
        );

        const start = Date.now();
        const stream = await this.driver.requestTextCompletionStream(this.prompt, this.options);

        let finish_reason: string | undefined = undefined;
        let promptTokens: number = 0;
        let resultTokens: number | undefined = undefined;
        for await (const chunk of stream) {
            if (chunk) {
                if (typeof chunk === 'string') {
                    chunks.push(chunk);
                    yield chunk;
                } else {
                    if (chunk.finish_reason) {                           //Do not replace non-null values with null values
                        finish_reason = chunk.finish_reason;             //Used to skip empty finish_reason chunks coming after "stop" or "length"
                    }
                    if (chunk.token_usage) {
                        //Tokens returned include prior parts of stream,
                        //so overwrite rather than accumulate
                        //Math.max used as some models report final token count at beginning of stream
                        promptTokens = Math.max(promptTokens, chunk.token_usage.prompt ?? 0);
                        resultTokens = Math.max(resultTokens ?? 0, chunk.token_usage.result ?? 0);
                    }
                    if (chunk.result) {
                        chunks.push(chunk.result);
                        yield chunk.result;
                    }
                }
            }
        }

        const content = chunks.join('');

        // Return undefined for the ExecutionTokenUsage object if there is nothing to fill it with.
        // Allows for checking for truthyness on token_usage, rather than it's internals. For testing and downstream usage.
        let tokens: ExecutionTokenUsage | undefined = resultTokens ?
            { prompt: promptTokens, result: resultTokens, total: resultTokens + promptTokens, } : undefined

        this.completion = {
            result: content,
            prompt: this.prompt,
            execution_time: Date.now() - start,
            token_usage: tokens,
            finish_reason: finish_reason,
            chunks: chunks.length,
        }

        this.driver.validateResult(this.completion, this.options);
    }

}

export class FallbackCompletionStream<PromptT = any> implements CompletionStream<PromptT> {

    completion: ExecutionResponse<PromptT> | undefined;

    constructor(public driver: AbstractDriver<DriverOptions, PromptT>,
        public prompt: PromptT,
        public options: ExecutionOptions) {
    }

    async *[Symbol.asyncIterator]() {
        // reset state
        this.completion = undefined;
        this.driver.logger.debug(
            `[${this.driver.provider}] Streaming is not supported, falling back to blocking execution`
        );
        const completion = await this.driver._execute(this.prompt, this.options);
        const content = typeof completion.result === 'string' ? completion.result : JSON.stringify(completion.result);
        yield content;

        this.completion = completion;
    }
}
