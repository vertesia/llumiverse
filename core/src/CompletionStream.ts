import { CompletionStream, DriverOptions, ExecutionOptions, ExecutionResponse, ExecutionTokenUsage, ToolUse } from "@llumiverse/common";
import { AbstractDriver } from "./Driver.js";

export class DefaultCompletionStream<PromptT = any> implements CompletionStream<PromptT> {

    chunks: number; // Counter for number of chunks instead of storing strings
    completion: ExecutionResponse<PromptT> | undefined;

    constructor(public driver: AbstractDriver<DriverOptions, PromptT>,
        public prompt: PromptT,
        public options: ExecutionOptions) {
        this.chunks = 0;
    }

    async *[Symbol.asyncIterator]() {
        // reset state
        this.completion = undefined;
        this.chunks = 0;
        const accumulatedResults: any[] = []; // Accumulate CompletionResult[] from chunks
        const accumulatedToolUse: Map<string, ToolUse> = new Map(); // Accumulate tool_use by id

        this.driver.logger.debug(
            `[${this.driver.provider}] Streaming Execution of ${this.options.model} with prompt`,
        );

        const start = Date.now();
        let finish_reason: string | undefined = undefined;
        let promptTokens: number = 0;
        let resultTokens: number | undefined = undefined;

        try {
            const stream = await this.driver.requestTextCompletionStream(this.prompt, this.options);
            for await (const chunk of stream) {
                if (chunk) {
                    if (typeof chunk === 'string') {
                        this.chunks++;
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
                        // Accumulate tool_use from chunks
                        // Note: During streaming, tool_input comes as string chunks that need concatenation
                        if (chunk.tool_use && chunk.tool_use.length > 0) {
                            for (const tool of chunk.tool_use) {
                                const existing = accumulatedToolUse.get(tool.id);
                                if (existing) {
                                    // Merge tool input (for streaming where arguments come as string pieces)
                                    if (tool.tool_input !== null && tool.tool_input !== undefined) {
                                        const existingInput = existing.tool_input as unknown;
                                        const newInput = tool.tool_input as unknown;
                                        if (typeof existingInput === 'string' && typeof newInput === 'string') {
                                            // Concatenate string arguments
                                            (existing as any).tool_input = existingInput + newInput;
                                        } else if (existingInput && typeof existingInput === 'object' && newInput && typeof newInput === 'object') {
                                            // Merge objects
                                            existing.tool_input = { ...(existingInput as object), ...(newInput as object) } as any;
                                        } else {
                                            existing.tool_input = tool.tool_input;
                                        }
                                    }
                                    // Update tool name if provided (might come in later chunk)
                                    if (tool.tool_name) {
                                        existing.tool_name = tool.tool_name;
                                    }
                                    // Update actual ID if provided (OpenAI sends id only in first chunk)
                                    if ((tool as any)._actual_id) {
                                        (existing as any)._actual_id = (tool as any)._actual_id;
                                    }
                                } else {
                                    // New tool call
                                    accumulatedToolUse.set(tool.id, { ...tool });
                                }
                            }
                        }
                        if (Array.isArray(chunk.result) && chunk.result.length > 0) {
                            // Process each result in the chunk, combining consecutive text/JSON
                            for (const result of chunk.result) {
                                // Check if we can combine with the last accumulated result
                                const lastResult = accumulatedResults[accumulatedResults.length - 1];

                                if (lastResult &&
                                    ((lastResult.type === 'text' && result.type === 'text') ||
                                        (lastResult.type === 'json' && result.type === 'json'))) {
                                    // Combine consecutive text or JSON results
                                    if (result.type === 'text') {
                                        lastResult.value += result.value;
                                    } else if (result.type === 'json') {
                                        // For JSON, combine the parsed objects directly
                                        try {
                                            const lastParsed = lastResult.value;
                                            const currentParsed = result.value;
                                            if (lastParsed !== null && typeof lastParsed === 'object' &&
                                                currentParsed !== null && typeof currentParsed === 'object') {
                                                const combined = { ...lastParsed, ...currentParsed };
                                                lastResult.value = combined;
                                            } else {
                                                // If not objects, convert to string and concatenate
                                                const lastStr = typeof lastParsed === 'string' ? lastParsed : JSON.stringify(lastParsed);
                                                const currentStr = typeof currentParsed === 'string' ? currentParsed : JSON.stringify(currentParsed);
                                                lastResult.value = lastStr + currentStr;
                                            }
                                        } catch {
                                            // If anything fails, just concatenate string representations
                                            lastResult.value = String(lastResult.value) + String(result.value);
                                        }
                                    }
                                } else {
                                    // Add as new result
                                    accumulatedResults.push(result);
                                }
                            }

                            // Convert CompletionResult[] to string for streaming
                            // Only yield if we have results to show
                            const resultText = chunk.result.map(r => {
                                switch (r.type) {
                                    case 'text':
                                        return r.value;
                                    case 'json':
                                        return JSON.stringify(r.value);
                                    case 'image':
                                        // Show truncated image placeholder for streaming
                                        const truncatedValue = typeof r.value === 'string' ? r.value.slice(0, 10) : String(r.value).slice(0, 10);
                                        return `\n[Image: ${truncatedValue}...]\n`;
                                    default:
                                        return String((r as any).value || '');
                                }
                            }).join('');

                            if (resultText) {
                                this.chunks++;
                                yield resultText;
                            }
                        }
                    }
                }
            }
        } catch (error: any) {
            error.prompt = this.prompt;
            throw error;
        }

        // Return undefined for the ExecutionTokenUsage object if there is nothing to fill it with.
        // Allows for checking for truthy-ness on token_usage, rather than it's internals. For testing and downstream usage.
        const tokens: ExecutionTokenUsage | undefined = resultTokens ?
            { prompt: promptTokens, result: resultTokens, total: resultTokens + promptTokens, } : undefined

        // Convert accumulated tool_use Map to array
        const toolUseArray = accumulatedToolUse.size > 0 ? Array.from(accumulatedToolUse.values()) : undefined;

        // Finalize tool calls: restore actual IDs and parse JSON arguments
        if (toolUseArray) {
            for (const tool of toolUseArray) {
                // Restore actual ID from OpenAI (was stored in _actual_id during streaming)
                if ((tool as any)._actual_id) {
                    tool.id = (tool as any)._actual_id;
                    delete (tool as any)._actual_id;
                }
                // Parse tool_input strings as JSON if needed (streaming sends arguments as string chunks)
                if (typeof tool.tool_input === 'string') {
                    try {
                        tool.tool_input = JSON.parse(tool.tool_input);
                    } catch {
                        // Keep as string if not valid JSON
                    }
                }
            }
        }

        this.completion = {
            result: accumulatedResults, // Return the accumulated CompletionResult[] instead of text
            prompt: this.prompt,
            execution_time: Date.now() - start,
            token_usage: tokens,
            finish_reason: finish_reason,
            chunks: this.chunks,
            tool_use: toolUseArray,
        }

        // Build conversation context for multi-turn support
        const conversation = this.driver.buildStreamingConversation(
            this.prompt,
            accumulatedResults,
            toolUseArray,
            this.options
        );
        if (conversation !== undefined) {
            this.completion.conversation = conversation;
        }

        try {
            if (this.completion) {
                this.driver.validateResult(this.completion, this.options);
            }
        } catch (error: any) {
            error.prompt = this.prompt;
            throw error;
        }
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
        try {
            const completion = await this.driver._execute(this.prompt, this.options);
            // For fallback streaming, yield the text content but keep the original completion
            const content = completion.result.map(r => {
                switch (r.type) {
                    case 'text':
                        return r.value;
                    case 'json':
                        return JSON.stringify(r.value);
                    case 'image':
                        // Show truncated image placeholder for streaming
                        const truncatedValue = typeof r.value === 'string' ? r.value.slice(0, 10) : String(r.value).slice(0, 10);
                        return `[Image: ${truncatedValue}...]`;
                    default:
                        return String((r as any).value || '');
                }
            }).join('');
            yield content;
            this.completion = completion; // Return the original completion with untouched CompletionResult[]
        } catch (error: any) {
            error.prompt = this.prompt;
            throw error;
        }
    }
}
