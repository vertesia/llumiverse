// This file is used by multiple drivers
// to format prompts in a way that is compatible with OpenAI's API.

import { PromptRole, PromptOptions } from "@llumiverse/common";
import { readStreamAsBase64 } from "../../../core/src/stream.js";
import { PromptSegment } from "@llumiverse/common";

import type {
    ChatCompletionMessageParam,
    ChatCompletionContentPart,
    ChatCompletionContentPartText,
    ChatCompletionContentPartImage,
    ChatCompletionContentPartRefusal,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam
} from 'openai/resources/chat/completions';

export interface OpenAITextMessage {
    content: string;
    role: 'system' | 'user' | 'assistant' | 'developer';
}
export type OpenAIInputMessage =
    OpenAIUserMessage |
    OpenAISystemMessage |
    OpenAIDeveloperMessage |
    OpenAIToolMessage |
    OpenAIAssistantMessage;
interface OpenAIBaseMessage {
    content: string | OpenAIContentPart[];
    role: 'system' | 'user' | 'assistant' | 'developer' | 'tool';
    name?: string;
}

export interface OpenAIUserMessage extends OpenAIBaseMessage {
    role: 'user';
}

export interface OpenAISystemMessage extends OpenAIBaseMessage {
    role: 'system';
    content: string | OpenAIContentPartText[];
}

export interface OpenAIDeveloperMessage extends OpenAIBaseMessage {
    role: 'developer';
    content: string | OpenAIContentPartText[];
}

export interface OpenAIToolMessage {
    role: 'tool';
    tool_call_id: string;
    content: string | OpenAIContentPartText[];
}

interface OpenAIRefusal {
    type: 'refusal';
    refusal: string;
}

export interface OpenAIAssistantMessage extends Omit<OpenAIBaseMessage, 'content'> {
    content: string | (OpenAIContentPart | OpenAIRefusal)[];
    role: 'assistant';
    tool_calls?: {
        id: string;
        function: {
            arguments: string;
            name: string;
        }
        type: 'function';
    }[];
}

export type OpenAIContentPart = OpenAIContentPartText | OpenAIContentPartImage | { type: string;[key: string]: any };
export interface OpenAIContentPartText {
    type: 'text';
    text: string
}

export interface OpenAIContentPartImage {
    type: 'image_url';
    image_url: {
        detail?: 'auto' | 'low' | 'high'
        url: string
    }
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
            system.push({ content: msg.content, role: "system" });
        } else if (msg.role === PromptRole.safety) {
            safety.push({ content: "IMPORTANT: " + msg.content, role: "system" });
        } else if (msg.role !== PromptRole.negative && msg.role !== PromptRole.mask && msg.role !== PromptRole.tool) {
            user.push({
                content: msg.content,
                role: msg.role || 'user',
            })
        }
    }

    // put system messages first and safety last
    return system.concat(user).concat(safety);
}


export async function formatOpenAILikeMultimodalPrompt(segments: PromptSegment[], opts: PromptOptions & OpenAIPromptFormatterOptions): Promise<OpenAIInputMessage[]> {
    const system: OpenAIInputMessage[] = [];
    const safety: OpenAIInputMessage[] = [];
    const others: OpenAIInputMessage[] = [];

    for (const msg of segments) {

        const parts: (OpenAIContentPartImage | OpenAIContentPartText)[] = [];

        //generate the parts based on PromptSegment
        if (msg.files) {
            for (const file of msg.files) {
                const stream = await file.getStream();
                const data = await readStreamAsBase64(stream);
                parts.push({
                    type: "image_url",
                    image_url: {
                        url: `data:${file.mime_type || "image/jpeg"};base64,${data}`,
                        //detail: "auto"  //This is modified just before execution to "low" | "high" | "auto"
                    },
                })
            }
        }

        if (msg.content) {
            parts.push({
                text: msg.content,
                type: "text"
            })
        }


        if (msg.role === PromptRole.system) {
            // For system messages, filter to only text parts
            const textParts = parts.filter((part): part is OpenAIContentPartText => part.type === 'text');
            system.push({
                role: "system",
                content: textParts.length === 1 && !msg.files ? textParts[0].text : textParts
            });

            if (opts.useToolForFormatting && opts.schema) {
                system.forEach(s => {
                    if (typeof s.content === 'string') {
                        s.content = "TOOL: " + s.content;
                    } else if (Array.isArray(s.content)) {
                        s.content.forEach(c => {
                            if (c.type === "text") c.text = "TOOL: " + c.text;
                        });
                    }
                });
            }

        } else if (msg.role === PromptRole.safety) {
            const textParts = parts.filter((part): part is OpenAIContentPartText => part.type === 'text');
            const safetyMsg: OpenAISystemMessage = {
                role: "system",
                content: textParts
            };

            if (Array.isArray(safetyMsg.content)) {
                safetyMsg.content.forEach(c => {
                    if (c.type === "text") c.text = "DO NOT IGNORE - IMPORTANT: " + c.text;
                });
            }

            system.push(safetyMsg);
        } else if (msg.role === PromptRole.tool) {
            if (!msg.tool_use_id) {
                throw new Error("Tool use id is required for tool messages")
            }
            others.push({
                role: "tool",
                tool_call_id: msg.tool_use_id,
                content: msg.content
            })
        } else if (msg.role !== PromptRole.negative && msg.role !== PromptRole.mask) {
            others.push({
                role: msg.role ?? 'user',
                content: parts
            })
        }

    }

    if (opts.result_schema && !opts.useToolForFormatting) {
        system.push({
            role: "system",
            content: [{
                type: "text",
                text: "IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. " + `<response_schema>${JSON.stringify(opts.result_schema)}</response_schema>`
            }]
        })
    }

    // put system messages first and safety last
    return ([] as OpenAIInputMessage[]).concat(system).concat(others).concat(safety);

}

export interface OpenAIPromptFormatterOptions {
    multimodal?: boolean
    useToolForFormatting?: boolean
    schema?: Object
}
