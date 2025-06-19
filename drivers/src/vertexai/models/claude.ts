import { ContentBlock, ContentBlockParam, DocumentBlockParam, ImageBlockParam, Message, MessageParam, TextBlockParam, ToolResultBlockParam } from "@anthropic-ai/sdk/resources/index.js";
import {
    AIModel, Completion, CompletionChunkObject, ExecutionOptions, getMaxTokensLimitVertexAi, JSONObject, ModelType,
    PromptRole, PromptSegment, readStreamAsBase64, StatelessExecutionOptions, ToolUse, VertexAIClaudeOptions
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";
import { MessageCreateParamsBase, MessageCreateParamsNonStreaming, RawMessageStreamEvent } from "@anthropic-ai/sdk/resources/messages.js";
import { MessageStreamParams } from "@anthropic-ai/sdk/resources/index.mjs";

interface ClaudePrompt {
    messages: MessageParam[];
    system: TextBlockParam[];
}

function claudeFinishReason(reason: string | undefined) {
    if (!reason) return undefined;
    switch (reason) {
        case 'end_turn': return "stop";
        case 'max_tokens': return "length";
        default: return reason; //stop_sequence
    }
}

function collectTextParts(content: any) {
    const out = [];

    for (const block of content) {
        if (block?.text) {
            out.push(block.text);
        }
    }
    return out.join('\n');
}

function maxToken(option: StatelessExecutionOptions): number {
    const modelOptions = option.model_options as VertexAIClaudeOptions | undefined;
    if (modelOptions && typeof modelOptions.max_tokens === "number") {
        return modelOptions.max_tokens;
    } else {
        // Fallback to the default max tokens limit for the model
        if (option.model.includes('claude-3-7-sonnet')) {
            return 64000; // Claude 3.7 can go up to 128k with a beta header, but when no max tokens is specified, we default to 64k.
        }
        return getMaxTokensLimitVertexAi(option.model);
    }
}

async function collectFileBlocks(segment: PromptSegment, contentBlocks: ContentBlockParam[], restrictedTypes: boolean = false): Promise<void> {
    // When restrictedTypes is true, we only allow text files and images.
    // i.e. output is Array<TextBlockParam | ImageBlockParam>
    
    for (const file of segment.files || []) {
        if (file.mime_type?.startsWith("image/")) {
            const allowedTypes = ["image/png", "image/jpeg", "image/gif", "image/webp"];
            if (!allowedTypes.includes(file.mime_type)) {
                throw new Error(`Unsupported image type: ${file.mime_type}`);
            }
            const mimeType = String(file.mime_type) as "image/png" | "image/jpeg" | "image/gif" | "image/webp";

            contentBlocks.push({
                type: 'image',
                source: {
                    type: 'base64',
                    data: await readStreamAsBase64(await file.getStream()),
                    media_type: mimeType
                }
            } satisfies ImageBlockParam);
        } else if (!restrictedTypes) {
            if (file.mime_type?.startsWith("text/")) {
                contentBlocks.push({
                    source: {
                        type: 'text',
                        data: await readStreamAsBase64(await file.getStream()),
                        media_type: 'text/plain'
                    },
                    type: 'document'
                } satisfies DocumentBlockParam);
            }
        }
    }
}

export class ClaudeModelDefinition implements ModelDefinition<ClaudePrompt> {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } satisfies AIModel;
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: ExecutionOptions): Promise<ClaudePrompt> {
        // Convert the prompt to the format expected by the Claude API
        const system: TextBlockParam[] = segments
            .filter(segment => segment.role === PromptRole.system)
            .map(segment => ({
                text: segment.content,
                type: 'text'
            }));

        if (options.result_schema) {
            let schemaText: string = '';
            if (options.tools && options.tools.length > 0) {
                schemaText = "When not calling tools, the answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema);
            } else {
                schemaText = "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema);
            }

            const schemaSegments: TextBlockParam = {
                text: schemaText,
                type: 'text'
            }
            system.push(schemaSegments);
        }

        let messages: MessageParam[] = [];
        const safetyMessages: MessageParam[] = [];
        for (const segment of segments) {
            if (segment.role === PromptRole.system) {
                continue;
            }

            if (segment.role === PromptRole.tool) {
                if (!segment.tool_use_id) {
                    throw new Error("Tool prompt segment must have a tool use ID");
                }

                const contentBlocks: Array<TextBlockParam | ImageBlockParam> = [];

                if (segment.content) {
                    contentBlocks.push({
                        type: 'text',
                        text: segment.content
                    } satisfies TextBlockParam);
                }
                
                await collectFileBlocks(segment, contentBlocks, true);

                messages.push({
                    role: 'user',
                    content: [{
                        type: 'tool_result',
                        tool_use_id: segment.tool_use_id,
                        content: contentBlocks,
                    } satisfies ToolResultBlockParam]
                });

            } else {
                const contentBlocks: ContentBlockParam[] = [];
                
                if (segment.content) {
                    contentBlocks.push({
                        type: 'text',
                        text: segment.content
                    } satisfies TextBlockParam);
                }

                await collectFileBlocks(segment, contentBlocks);

                if (contentBlocks.length === 0) {
                    continue; // skip empty segments
                }

                if (segment.role === PromptRole.safety) {
                    safetyMessages.push({
                        role: 'user',
                        content: contentBlocks
                    });
                } else {
                    messages.push({
                        role: segment.role === PromptRole.assistant ? 'assistant' : 'user',
                        content: contentBlocks
                    });
                }
                
            }
        }

        messages = messages.concat(safetyMessages);

        return {
            messages: messages,
            system: system
        }
    }

    async requestTextCompletion(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<Completion> {
        const client = driver.getAnthropicClient();
        options.model_options = options.model_options as VertexAIClaudeOptions;

        if (options.model_options?._option_id !== "vertexai-claude") {
            driver.logger.warn("Invalid model options", { options: options.model_options });
        }

        let conversation = updateConversation(options.conversation as ClaudePrompt, prompt);

        const { payload, requestOptions } = getClaudePayload(options, conversation);
        // disable streaming, the create function is overloaded so payload type matters.
        const nonStreamingPayload: MessageCreateParamsNonStreaming = { ...payload, stream: false };

        const result = await client.messages.create(nonStreamingPayload, requestOptions) satisfies Message;

        const text = collectTextParts(result.content);
        const tool_use = collectTools(result.content);

        conversation = updateConversation(conversation, createPromptFromResponse(result));

        return {
            result: text ?? '',
            tool_use,
            token_usage: {
                prompt: result?.usage.input_tokens,
                result: result?.usage.output_tokens,
                total: result?.usage.input_tokens + result?.usage.output_tokens
            },
            // make sure we set finish_reason to the correct value (claude is normally setting this by itself)
            finish_reason: tool_use ? "tool_use" : claudeFinishReason(result?.stop_reason ?? ''),
            conversation
        } satisfies Completion;
    }

    async requestTextCompletionStream(driver: VertexAIDriver, prompt: ClaudePrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const client = driver.getAnthropicClient();
        const model_options = options.model_options as VertexAIClaudeOptions | undefined;

        if (model_options?._option_id !== "vertexai-claude") {
            driver.logger.warn("Invalid model options", { options: options.model_options });
        }

        const { payload, requestOptions } = getClaudePayload(options, prompt);
        const streamingPayload: MessageStreamParams = { ...payload, stream: true };

        const response_stream = await client.messages.stream(streamingPayload, requestOptions);

        const stream = asyncMap(response_stream, async (streamEvent: RawMessageStreamEvent) => {
            console.log(JSON.stringify(streamEvent, null, 2));
            switch (streamEvent.type) {
                case "message_start":
                    return {
                        result: '',
                        token_usage: {
                            prompt: streamEvent.message.usage.input_tokens,
                            result: streamEvent.message.usage.output_tokens
                        }
                    } satisfies CompletionChunkObject;

                case "content_block_delta":
                    // Handle different delta types
                    switch (streamEvent.delta.type) {
                        case "text_delta":
                            return {
                                result: streamEvent.delta.text ?? ''
                            } satisfies CompletionChunkObject;
                        case "thinking_delta":
                            if (model_options?.include_thoughts) {
                                return {
                                    result: streamEvent.delta.thinking ?? '',
                                } satisfies CompletionChunkObject;
                            }
                            break;
                        case "signature_delta":
                            // Signature deltas, signify the end of the thoughts.
                            if (model_options?.include_thoughts) {
                                return {
                                    result: '\n\n', // Double newline for more spacing
                                } satisfies CompletionChunkObject;
                            }
                            break;
                    }
                    break;
                case "message_delta":
                    return {
                        result: '',
                        token_usage: {
                            result: streamEvent.usage.output_tokens
                        },
                        finish_reason: claudeFinishReason(streamEvent.delta.stop_reason ?? undefined),
                    } satisfies CompletionChunkObject;
            }

            // Default case for all other event types
            return {
                result: ''
            } satisfies CompletionChunkObject;
        });

        return stream;
    }
}

export function collectTools(content: ContentBlock[]): ToolUse[] | undefined {
    const out: ToolUse[] = [];

    for (const block of content) {
        if (block?.type === "tool_use") {
            out.push({
                id: block.id,
                tool_name: block.name,
                tool_input: block.input as JSONObject,
            });
        }
    }

    return out.length > 0 ? out : undefined;
}

function createPromptFromResponse(response: Message): ClaudePrompt {
    return {
        messages: [{
            role: PromptRole.assistant,
            content: response.content,
        }],
        system: []
    }
}

/**
 * Update the conversation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: ClaudePrompt | undefined | null, prompt: ClaudePrompt): ClaudePrompt {
    const baseSystemMessages = conversation ? conversation.system : [];
    const baseMessages = conversation ? conversation.messages : []
    return {
        messages: baseMessages.concat(prompt.messages || []),
        system: baseSystemMessages.concat(prompt.system || [])
    };
}
interface RequestOptions {
    headers?: Record<string, string>;
}

function getClaudePayload(options: ExecutionOptions, prompt: ClaudePrompt): { payload: MessageCreateParamsBase, requestOptions: RequestOptions | undefined} {
    const splits = options.model.split("/");
    const modelName = splits[splits.length - 1];
    const model_options = options.model_options as VertexAIClaudeOptions;

    // Add beta header for Claude 3.7 models to enable 128k output tokens
    let requestOptions: RequestOptions | undefined = undefined;
    if (modelName.includes('claude-3-7-sonnet') && (model_options?.max_tokens ?? 0) > 64000) {
        requestOptions = {
            headers: {
                'anthropic-beta': 'output-128k-2025-02-19'
            }
        };
    }

    const payload = {
        messages: prompt.messages,
        system: prompt.system,
        tools: options.tools, // we are using the same shape as claude for tools
        temperature: model_options?.temperature,
        model: modelName,
        max_tokens: maxToken(options),
        top_p: model_options?.top_p,
        top_k: model_options?.top_k,
        stop_sequences: model_options?.stop_sequence,
        thinking: model_options?.thinking_mode ?
            {
                budget_tokens: model_options?.thinking_budget_tokens ?? 1024,
                type: "enabled" as const
            } : {
                type: "disabled" as const
            }
    };

    return { payload, requestOptions };
}
