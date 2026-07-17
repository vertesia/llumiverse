import {
    type CompletionChunkObject,
    type CompletionResult,
    type CompletionStream,
    type DriverCompletionStream,
    type DriverOptions,
    type ExecutionOptions,
    type ExecutionResponse,
    type ExecutionTokenUsage,
    LlumiverseError,
    type ToolUse,
} from '@llumiverse/common';
import type { AbstractDriver } from './Driver.js';

type StreamingToolUse = ToolUse<unknown> & { _actual_id?: string };

/**
 * Merge a single streamed `tool_use` fragment into the accumulator map keyed by tool id.
 *
 * Providers stream a tool call across several chunks. Different fields arrive in
 * different chunks and must be reassembled:
 * - `tool_input` arrives as string pieces (concatenated) or partial objects (merged).
 * - `tool_name` and the provider's real id (`_actual_id`) may only appear in a later chunk.
 * - `thought_signature` (Gemini 2.5+/3.x thinking models) is an opaque base64 byte string
 *   that must be passed back verbatim with the assistant turn on the next request.
 *
 * Before this helper existed, the inline accumulator merged `tool_input`/`tool_name`/`_actual_id`
 * but silently dropped `thought_signature` for any chunk after the first for a given tool id.
 * When a provider delivers the signature in pieces (large high-effort signatures are split
 * across chunks) or only on a later chunk, the accumulated signature was left truncated or
 * absent — producing an invalid base64 byte value that the provider rejects with a
 * "Base64 decoding failed" (TYPE_BYTES) error when the assistant turn is threaded back.
 *
 * The signature is therefore reassembled by concatenating fragments in arrival order, exactly
 * like `tool_input` string pieces. When the whole signature arrives in a single chunk there is
 * only one fragment, so concatenation is a no-op and the value round-trips byte-identically.
 *
 * Exported for unit testing the round-trip in isolation (no network / no full driver).
 */
export function accumulateToolUseChunk(
    accumulatedToolUse: Map<string, StreamingToolUse>,
    tool: StreamingToolUse,
): void {
    const existing = accumulatedToolUse.get(tool.id);
    if (!existing) {
        // New tool call
        accumulatedToolUse.set(tool.id, { ...tool });
        return;
    }
    // Merge tool input (for streaming where arguments come as string pieces)
    if (tool.tool_input !== null && tool.tool_input !== undefined) {
        const existingInput = existing.tool_input as unknown;
        const newInput = tool.tool_input as unknown;
        if (typeof existingInput === 'string' && typeof newInput === 'string') {
            // Concatenate string arguments
            existing.tool_input = (existingInput + newInput) as typeof existing.tool_input;
        } else if (existingInput && typeof existingInput === 'object' && newInput && typeof newInput === 'object') {
            // Merge objects
            existing.tool_input = {
                ...(existingInput as Record<string, unknown>),
                ...(newInput as Record<string, unknown>),
            } as typeof existing.tool_input;
        } else {
            existing.tool_input = tool.tool_input;
        }
    }
    // Update tool name if provided (might come in later chunk)
    if (tool.tool_name) {
        existing.tool_name = tool.tool_name;
    }
    // Reassemble the thought signature: concatenate fragments in arrival order so a
    // signature split across chunks is restored byte-for-byte instead of being truncated.
    if (tool.thought_signature) {
        existing.thought_signature = (existing.thought_signature ?? '') + tool.thought_signature;
    }
    // Update actual ID if provided (OpenAI sends id only in first chunk)
    if (tool._actual_id) {
        existing._actual_id = tool._actual_id;
    }
}

export class DefaultCompletionStream<PromptT = unknown> implements CompletionStream<PromptT> {
    chunks: number; // Counter for number of chunks instead of storing strings
    completion: ExecutionResponse<PromptT> | undefined;

    constructor(
        public driver: AbstractDriver<DriverOptions, PromptT>,
        public prompt: PromptT,
        public options: ExecutionOptions,
    ) {
        this.chunks = 0;
    }

    async *[Symbol.asyncIterator]() {
        // reset state
        this.completion = undefined;
        this.chunks = 0;
        const accumulatedResults: CompletionResult[] = []; // Accumulate CompletionResult[] from chunks
        const accumulatedToolUse: Map<string, StreamingToolUse> = new Map(); // Accumulate tool_use by id

        this.driver.logger.debug(`[${this.driver.provider}] Streaming Execution of ${this.options.model} with prompt`);

        const start = Date.now();
        let finish_reason: string | undefined;
        let promptTokens: number = 0;
        let resultTokens: number | undefined;
        let promptCachedTokens: number | undefined;
        let promptCacheWriteTokens: number | undefined;
        let promptNewTokens: number | undefined;
        const httpScope = this.driver.createExecutionHttpAgentScope(this.options);
        let sourceIterator: AsyncIterator<CompletionChunkObject> | undefined;
        let stream: DriverCompletionStream | undefined;
        let streamCompleted = false;

        try {
            stream = await httpScope.run(() => this.driver.requestTextCompletionStream(this.prompt, this.options));
            const iterator = stream[Symbol.asyncIterator]();
            sourceIterator = iterator;
            while (true) {
                const next = await httpScope.run(() => iterator.next());
                if (next.done) {
                    streamCompleted = true;
                    break;
                }
                const chunk = next.value;
                if (chunk) {
                    if (typeof chunk === 'string') {
                        this.chunks++;
                        yield chunk;
                    } else {
                        if (chunk.finish_reason) {
                            //Do not replace non-null values with null values
                            finish_reason = chunk.finish_reason; //Used to skip empty finish_reason chunks coming after "stop" or "length"
                        }
                        if (chunk.token_usage) {
                            //Tokens returned include prior parts of stream,
                            //so overwrite rather than accumulate
                            //Math.max used as some models report final token count at beginning of stream
                            promptTokens = Math.max(promptTokens, chunk.token_usage.prompt ?? 0);
                            resultTokens = Math.max(resultTokens ?? 0, chunk.token_usage.result ?? 0);
                            if (chunk.token_usage.prompt_cached != null)
                                promptCachedTokens = chunk.token_usage.prompt_cached;
                            if (chunk.token_usage.prompt_cache_write != null)
                                promptCacheWriteTokens = chunk.token_usage.prompt_cache_write;
                            if (chunk.token_usage.prompt_new != null) promptNewTokens = chunk.token_usage.prompt_new;
                        }
                        // Accumulate tool_use from chunks
                        // Note: During streaming, tool_input comes as string chunks that need concatenation
                        if (chunk.tool_use && chunk.tool_use.length > 0) {
                            for (const tool of chunk.tool_use) {
                                accumulateToolUseChunk(accumulatedToolUse, tool as StreamingToolUse);
                            }
                        }
                        if (Array.isArray(chunk.result) && chunk.result.length > 0) {
                            // Process each result in the chunk, combining consecutive text/JSON
                            for (const result of chunk.result) {
                                // Check if we can combine with the last accumulated result
                                const lastResult = accumulatedResults[accumulatedResults.length - 1];

                                if (
                                    lastResult &&
                                    ((lastResult.type === 'text' && result.type === 'text') ||
                                        (lastResult.type === 'json' && result.type === 'json'))
                                ) {
                                    // Combine consecutive text or JSON results
                                    if (result.type === 'text') {
                                        lastResult.value += result.value;
                                    } else if (result.type === 'json') {
                                        // For JSON, combine the parsed objects directly
                                        try {
                                            const lastParsed = lastResult.value;
                                            const currentParsed = result.value;
                                            if (
                                                lastParsed !== null &&
                                                typeof lastParsed === 'object' &&
                                                currentParsed !== null &&
                                                typeof currentParsed === 'object'
                                            ) {
                                                const combined = { ...lastParsed, ...currentParsed };
                                                lastResult.value = combined;
                                            } else {
                                                // If not objects, convert to string and concatenate
                                                const lastStr =
                                                    typeof lastParsed === 'string'
                                                        ? lastParsed
                                                        : JSON.stringify(lastParsed);
                                                const currentStr =
                                                    typeof currentParsed === 'string'
                                                        ? currentParsed
                                                        : JSON.stringify(currentParsed);
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
                            const resultText = chunk.result
                                .map((r) => {
                                    switch (r.type) {
                                        case 'text':
                                            return r.value;
                                        case 'json':
                                            return JSON.stringify(r.value);
                                        case 'image': {
                                            const truncatedValue =
                                                typeof r.value === 'string'
                                                    ? r.value.slice(0, 10)
                                                    : String(r.value).slice(0, 10);
                                            return `\n[Image: ${truncatedValue}...]\n`;
                                        }
                                        default: {
                                            const _exhaustive: never = r;
                                            return String(_exhaustive);
                                        }
                                    }
                                })
                                .join('');

                            if (resultText) {
                                this.chunks++;
                                yield resultText;
                            }
                        }
                    }
                }
            }
        } catch (error: unknown) {
            // Don't wrap if already a LlumiverseError
            if (LlumiverseError.isLlumiverseError(error)) {
                throw error;
            }
            throw this.driver.formatLlumiverseError(error, {
                provider: this.driver.provider,
                model: this.options.model,
                operation: 'stream',
            });
        } finally {
            if (!streamCompleted && sourceIterator?.return) {
                const returnIterator = sourceIterator.return.bind(sourceIterator);
                try {
                    await httpScope.run(() => returnIterator());
                } catch {
                    /* stream cleanup best-effort */
                }
            }
            await httpScope.close();
        }

        // Return undefined only if we never received any token data from the provider.
        // Use !== undefined (not truthiness) because resultTokens === 0 is valid (e.g. empty output with stop).
        const tokens: ExecutionTokenUsage | undefined =
            resultTokens !== undefined
                ? {
                      prompt: promptTokens,
                      result: resultTokens,
                      total: resultTokens + promptTokens,
                      ...(promptCachedTokens != null && { prompt_cached: promptCachedTokens }),
                      ...(promptCacheWriteTokens != null && { prompt_cache_write: promptCacheWriteTokens }),
                      ...(promptNewTokens != null && { prompt_new: promptNewTokens }),
                  }
                : undefined;

        // Convert accumulated tool_use Map to array
        let toolUseArray = accumulatedToolUse.size > 0 ? Array.from(accumulatedToolUse.values()) : undefined;

        // Finalize tool calls: restore actual IDs and parse JSON arguments
        if (toolUseArray) {
            const truncatedToolIds = new Set<string>();
            for (const tool of toolUseArray) {
                // Restore actual ID from OpenAI (was stored in _actual_id during streaming)
                if (tool._actual_id) {
                    tool.id = tool._actual_id;
                    delete tool._actual_id;
                }
                // Parse tool_input strings as JSON if needed (streaming sends arguments as string chunks)
                if (typeof tool.tool_input === 'string') {
                    try {
                        tool.tool_input = JSON.parse(tool.tool_input);
                    } catch {
                        // JSON parse failed — tool_input was likely truncated by max_tokens.
                        // Set to empty object to prevent string tool_input from corrupting the conversation.
                        tool.tool_input = {};
                        truncatedToolIds.add(tool.id);
                    }
                }
            }

            // If finish_reason is "length" (max_tokens hit), drop truncated tool calls entirely —
            // they were cut off mid-generation and would produce invalid results.
            if (finish_reason === 'length' && truncatedToolIds.size > 0) {
                toolUseArray = toolUseArray.filter((t) => !truncatedToolIds.has(t.id));
                if (toolUseArray.length === 0) {
                    toolUseArray = undefined;
                }
            }
        }

        this.completion = {
            result: accumulatedResults, // Return the accumulated CompletionResult[] instead of text
            prompt: this.driver.formatDebugPrompt(this.prompt),
            execution_time: Date.now() - start,
            token_usage: tokens,
            finish_reason: finish_reason,
            chunks: this.chunks,
            tool_use: toolUseArray,
        };

        // Build conversation context for multi-turn support
        const conversation = stream?.finalizeConversation
            ? await stream.finalizeConversation()
            : this.driver.buildStreamingConversation(this.prompt, accumulatedResults, toolUseArray, this.options);
        if (conversation !== undefined) {
            this.completion.conversation = conversation;
        }

        try {
            if (this.completion) {
                this.driver.validateResult(this.completion, this.options);
            }
        } catch (error: unknown) {
            // Don't wrap if already a LlumiverseError
            if (LlumiverseError.isLlumiverseError(error)) {
                throw error;
            }
            throw this.driver.formatLlumiverseError(error, {
                provider: this.driver.provider,
                model: this.options.model,
                operation: 'stream',
            });
        }
    }
}

export class FallbackCompletionStream<PromptT = unknown> implements CompletionStream<PromptT> {
    completion: ExecutionResponse<PromptT> | undefined;

    constructor(
        public driver: AbstractDriver<DriverOptions, PromptT>,
        public prompt: PromptT,
        public options: ExecutionOptions,
    ) {}

    async *[Symbol.asyncIterator]() {
        // reset state
        this.completion = undefined;
        this.driver.logger.debug(
            `[${this.driver.provider}] Streaming is not supported, falling back to blocking execution`,
        );
        try {
            const completion = await this.driver._execute(this.prompt, this.options);
            // For fallback streaming, yield the text content but keep the original completion
            const content = completion.result
                .map((r) => {
                    switch (r.type) {
                        case 'text':
                            return r.value;
                        case 'json':
                            return JSON.stringify(r.value);
                        case 'image': {
                            const truncatedValue =
                                typeof r.value === 'string' ? r.value.slice(0, 10) : String(r.value).slice(0, 10);
                            return `[Image: ${truncatedValue}...]`;
                        }
                        default: {
                            const _exhaustive: never = r;
                            return String(_exhaustive);
                        }
                    }
                })
                .join('');
            yield content;
            this.completion = completion; // Return the original completion with untouched CompletionResult[]
        } catch (error: unknown) {
            // Don't wrap if already a LlumiverseError
            if (LlumiverseError.isLlumiverseError(error)) {
                throw error;
            }
            throw this.driver.formatLlumiverseError(error, {
                provider: this.driver.provider,
                model: this.options.model,
                operation: 'stream',
            });
        }
    }
}
