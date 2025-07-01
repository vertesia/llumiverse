// This file is used by multiple drivers
// to format prompts in a way that is compatible with OpenAI's API.

import { PromptRole, PromptOptions, PromptSegment } from "@llumiverse/common";
import { readStreamAsBase64 } from "@llumiverse/core";

import type {
    ChatCompletionMessageParam,
    ChatCompletionContentPartText,
    ChatCompletionContentPartImage,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam
} from 'openai/resources/chat/completions';

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


export async function formatOpenAILikeMultimodalPrompt(segments: PromptSegment[], opts: PromptOptions & OpenAIPromptFormatterOptions): Promise<ChatCompletionMessageParam[]> {
    const system: ChatCompletionMessageParam[] = [];
    const safety: ChatCompletionMessageParam[] = [];
    const others: ChatCompletionMessageParam[] = [];

    for (const msg of segments) {

        const parts: (ChatCompletionContentPartImage | ChatCompletionContentPartText)[] = [];

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
            const textParts = parts.filter((part): part is ChatCompletionContentPartText => part.type === 'text');
            const systemMsg: ChatCompletionSystemMessageParam = {
                role: "system",
                content: textParts.length === 1 && !msg.files ? textParts[0].text : textParts
            };
            system.push(systemMsg);

            if (opts.useToolForFormatting && opts.schema) {
                system.forEach(s => {
                    if (typeof s.content === 'string') {
                        s.content = "TOOL: " + s.content;
                    } else if (Array.isArray(s.content)) {
                        s.content.forEach((c: any) => {
                            if (c.type === "text") c.text = "TOOL: " + c.text;
                        });
                    }
                });
            }

        } else if (msg.role === PromptRole.safety) {
            const textParts = parts.filter((part): part is ChatCompletionContentPartText => part.type === 'text');
            const safetyMsg: ChatCompletionSystemMessageParam = {
                role: "system",
                content: textParts
            };

            if (Array.isArray(safetyMsg.content)) {
                safetyMsg.content.forEach((c: any) => {
                    if (c.type === "text") c.text = "DO NOT IGNORE - IMPORTANT: " + c.text;
                });
            }

            system.push(safetyMsg);
        } else if (msg.role === PromptRole.tool) {
            if (!msg.tool_use_id) {
                throw new Error("Tool use id is required for tool messages")
            }
            const toolMsg: ChatCompletionToolMessageParam = {
                role: "tool",
                tool_call_id: msg.tool_use_id,
                content: msg.content || ""
            };
            others.push(toolMsg);
        } else if (msg.role !== PromptRole.negative && msg.role !== PromptRole.mask) {
            if (msg.role === 'assistant') {
                const assistantMsg: ChatCompletionAssistantMessageParam = {
                    role: 'assistant',
                    content: parts as (ChatCompletionContentPartText)[]
                };
                others.push(assistantMsg);
            } else {
                const userMsg: ChatCompletionUserMessageParam = {
                    role: 'user',
                    content: parts
                };
                others.push(userMsg);
            }
        }

    }

    if (opts.result_schema && !opts.useToolForFormatting) {
        const schemaMsg: ChatCompletionSystemMessageParam = {
            role: "system",
            content: [{
                type: "text",
                text: "IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. " + `<response_schema>${JSON.stringify(opts.result_schema)}</response_schema>`
            }]
        };
        system.push(schemaMsg);
    }

    // put system messages first and safety last
    return ([] as ChatCompletionMessageParam[]).concat(system).concat(others).concat(safety);

}

export interface OpenAIPromptFormatterOptions {
    multimodal?: boolean
    useToolForFormatting?: boolean
    schema?: Object
}
