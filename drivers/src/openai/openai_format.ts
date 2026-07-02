// This file is used by multiple drivers
// to format prompts in a way that is compatible with OpenAI's API.

import { type PromptOptions, PromptRole, type PromptSegment } from '@llumiverse/common';
import { readStreamAsBase64 } from '@llumiverse/core';
import type OpenAI from 'openai';
import { truncateDataUrlForDebug } from '../shared/debug-prompt.js';

// Types for Response API
type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type ResponseInputContent = OpenAI.Responses.ResponseInputContent;
type EasyInputMessage = OpenAI.Responses.EasyInputMessage;

function isResponseInputContent(value: unknown): value is ResponseInputContent {
    return (
        typeof value === 'object' &&
        value !== null &&
        'type' in value &&
        (value.type === 'input_text' || value.type === 'input_image' || value.type === 'input_file')
    );
}

function isEasyInputMessageWithInputContent(
    item: ResponseInputItem,
): item is EasyInputMessage & { content: ResponseInputContent[] } {
    return 'content' in item && Array.isArray(item.content) && item.content.every(isResponseInputContent);
}

export function formatOpenAIDebugPrompt(prompt: ResponseInputItem[]): ResponseInputItem[] {
    return prompt.map((item) => {
        if (!isEasyInputMessageWithInputContent(item)) {
            return item;
        }
        const content = item.content.map((content) => {
            if (content.type === 'input_image' && 'image_url' in content && typeof content.image_url === 'string') {
                return {
                    ...content,
                    image_url: truncateDataUrlForDebug(content.image_url),
                } satisfies ResponseInputContent;
            }
            if (content.type === 'input_file' && 'file_data' in content && typeof content.file_data === 'string') {
                return {
                    ...content,
                    file_data: truncateDataUrlForDebug(content.file_data),
                } satisfies ResponseInputContent;
            }
            return content;
        });
        return {
            ...item,
            content,
        } satisfies EasyInputMessage;
    });
}

export interface OpenAITextMessage {
    content: string;
    role: 'system' | 'user' | 'assistant' | 'developer';
}

/**
 * OpenAI text only prompts
 * @param segments
 * @returns
 */
export function formatOpenAILikeTextPrompt(segments: PromptSegment[]): OpenAITextMessage[] {
    const system: OpenAITextMessage[] = [];
    const safety: OpenAITextMessage[] = [];
    const user: OpenAITextMessage[] = [];

    for (const msg of segments) {
        if (msg.role === PromptRole.system) {
            system.push({ content: msg.content, role: 'system' });
        } else if (msg.role === PromptRole.safety) {
            safety.push({ content: `IMPORTANT: ${msg.content}`, role: 'system' });
        } else if (msg.role !== PromptRole.negative && msg.role !== PromptRole.mask && msg.role !== PromptRole.tool) {
            user.push({
                content: msg.content,
                role: msg.role || 'user',
            });
        }
    }

    // put system messages first and safety last
    return system.concat(user).concat(safety);
}

export async function formatOpenAILikeMultimodalPrompt(
    segments: PromptSegment[],
    opts: PromptOptions & OpenAIPromptFormatterOptions,
): Promise<ResponseInputItem[]> {
    const system: ResponseInputItem[] = [];
    const safety: ResponseInputItem[] = [];
    const others: ResponseInputItem[] = [];

    for (const msg of segments) {
        const parts: ResponseInputContent[] = [];

        //generate the parts based on PromptSegment
        if (msg.files) {
            for (const file of msg.files) {
                const stream = await file.getStream();
                const data = await readStreamAsBase64(stream);
                parts.push({
                    type: 'input_image',
                    image_url: `data:${file.mime_type || 'image/jpeg'};base64,${data}`,
                    detail: 'auto',
                });
            }
        }

        if (msg.content) {
            parts.push({
                type: 'input_text',
                text: msg.content,
            });
        }

        if (msg.role === PromptRole.system) {
            // For system messages, filter to only text parts
            const textParts = parts.filter(
                (part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text',
            );
            const textContent = textParts.length === 1 && !msg.files ? textParts[0].text : textParts;
            const systemMsg: EasyInputMessage = {
                type: 'message',
                role: 'system',
                content: textContent,
            };
            system.push(systemMsg);

            if (opts.useToolForFormatting && opts.schema) {
                system.forEach((s) => {
                    if ((s as EasyInputMessage).role === 'system') {
                        const sysMsg = s as EasyInputMessage;
                        if (typeof sysMsg.content === 'string') {
                            sysMsg.content = `TOOL: ${sysMsg.content}`;
                        } else if (Array.isArray(sysMsg.content)) {
                            sysMsg.content.forEach((c) => {
                                if (c.type === 'input_text') c.text = `TOOL: ${c.text}`;
                            });
                        }
                    }
                });
            }
        } else if (msg.role === PromptRole.safety) {
            const textParts = parts.filter(
                (part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text',
            );
            const safetyMsg: EasyInputMessage = {
                type: 'message',
                role: 'system',
                content: textParts,
            };

            if (Array.isArray(safetyMsg.content)) {
                safetyMsg.content.forEach((c) => {
                    if (c.type === 'input_text') c.text = `DO NOT IGNORE - IMPORTANT: ${c.text}`;
                });
            }

            system.push(safetyMsg);
        } else if (msg.role === PromptRole.tool) {
            if (!msg.tool_use_id) {
                throw new Error('Tool use id is required for tool messages');
            }
            const toolOutputMsg: OpenAI.Responses.ResponseInputItem.FunctionCallOutput = {
                type: 'function_call_output',
                call_id: msg.tool_use_id,
                output: msg.content || '',
            };
            others.push(toolOutputMsg);
        } else if (msg.role !== PromptRole.negative && msg.role !== PromptRole.mask) {
            const inputMsg: EasyInputMessage = {
                type: 'message',
                role: msg.role === 'assistant' ? 'assistant' : 'user',
                content: parts,
            };
            others.push(inputMsg);
        }
    }

    if (opts.result_schema && !opts.useToolForFormatting) {
        const schemaMsg: EasyInputMessage = {
            type: 'message',
            role: 'system',
            content: [
                {
                    type: 'input_text',
                    text: `IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. <response_schema>${JSON.stringify(opts.result_schema)}</response_schema>`,
                },
            ],
        };
        system.push(schemaMsg);
    }

    // put system messages first and safety last
    return ([] as ResponseInputItem[]).concat(system).concat(others).concat(safety);
}

export interface OpenAIPromptFormatterOptions {
    multimodal?: boolean;
    useToolForFormatting?: boolean;
    schema?: object;
}

// Chat Completions API types
type ChatCompletionMessageParam = OpenAI.Chat.Completions.ChatCompletionMessageParam;
type ChatCompletionContentPart = OpenAI.Chat.Completions.ChatCompletionContentPart;

/**
 * Convert Response API items (`ResponseInputItem[]`) into Chat Completions messages
 * (`ChatCompletionMessageParam[]`).
 *
 * Some OpenAI-compatible providers expose only the *Chat Completions* surface and do NOT
 * implement the newer *Responses* API (e.g. TogetherAI, Groq). Those drivers still build
 * their prompts/conversations as `ResponseInputItem[]` — so all the shared conversation,
 * stripping and tool-handling logic keeps working — and convert to Chat Completions messages
 * right before the request. Crucially, images become `{ type: 'image_url', image_url: { url } }`
 * (the Chat Completions shape) instead of the Responses `input_image` shape, which those
 * providers silently ignore.
 */
export function convertResponseItemsToChatMessages(items: ResponseInputItem[]): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [];

    for (const item of items) {
        // Handle EasyInputMessage (has role and content)
        if ('role' in item && 'content' in item) {
            const msg = item as EasyInputMessage;
            const role = msg.role;

            // Handle system/developer messages
            if (role === 'system' || role === 'developer') {
                let content: string;
                if (typeof msg.content === 'string') {
                    content = msg.content;
                } else if (Array.isArray(msg.content)) {
                    content = msg.content
                        .filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text')
                        .map((part) => part.text)
                        .join('\n');
                } else {
                    content = '';
                }
                messages.push({ role: 'system', content });
                continue;
            }

            // Handle user messages (text + images)
            if (role === 'user') {
                let content: string | ChatCompletionContentPart[];
                if (typeof msg.content === 'string') {
                    content = msg.content;
                } else if (Array.isArray(msg.content)) {
                    const parts: ChatCompletionContentPart[] = [];
                    for (const part of msg.content) {
                        if (part.type === 'input_text') {
                            parts.push({ type: 'text', text: part.text });
                        } else if (part.type === 'input_image') {
                            const imgPart = part as OpenAI.Responses.ResponseInputImage;
                            if (imgPart.image_url) {
                                const image_url: { url: string; detail?: 'auto' | 'low' | 'high' } = {
                                    url: imgPart.image_url,
                                };
                                // Chat Completions only accepts auto|low|high; the Responses-only
                                // 'original' detail is dropped (TogetherAI ignores detail anyway).
                                if (imgPart.detail && imgPart.detail !== 'original') {
                                    image_url.detail = imgPart.detail;
                                }
                                parts.push({ type: 'image_url', image_url });
                            }
                        }
                    }
                    content = parts.length > 0 ? parts : '';
                } else {
                    content = '';
                }
                messages.push({ role: 'user', content });
                continue;
            }

            // Handle assistant messages
            if (role === 'assistant') {
                let content: string | null;
                if (typeof msg.content === 'string') {
                    content = msg.content;
                } else if (Array.isArray(msg.content)) {
                    content =
                        msg.content
                            .filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text')
                            .map((part) => part.text)
                            .join('\n') || null;
                } else {
                    content = null;
                }
                messages.push({ role: 'assistant', content });
                continue;
            }
        }

        // Handle function_call_output (tool response)
        if ('type' in item && item.type === 'function_call_output') {
            const output = item as OpenAI.Responses.ResponseInputItem.FunctionCallOutput;
            messages.push({
                role: 'tool',
                tool_call_id: output.call_id,
                content: typeof output.output === 'string' ? output.output : JSON.stringify(output.output),
            });
            continue;
        }

        // Handle function_call (assistant tool call)
        if ('type' in item && item.type === 'function_call') {
            const call = item as OpenAI.Responses.ResponseFunctionToolCall;
            messages.push({
                role: 'assistant',
                content: null,
                tool_calls: [
                    {
                        id: call.call_id,
                        type: 'function',
                        function: {
                            name: call.name,
                            arguments: call.arguments,
                        },
                    },
                ],
            });
        }
    }

    return messages;
}
