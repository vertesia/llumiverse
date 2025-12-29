// This file is used by multiple drivers
// to format prompts in a way that is compatible with OpenAI's API.

import { PromptRole, PromptOptions, PromptSegment } from "@llumiverse/common";
import { readStreamAsBase64 } from "@llumiverse/core";
import type OpenAI from "openai";

// Types for Response API
type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type ResponseInputContent = OpenAI.Responses.ResponseInputContent;
type EasyInputMessage = OpenAI.Responses.EasyInputMessage;

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


export async function formatOpenAILikeMultimodalPrompt(segments: PromptSegment[], opts: PromptOptions & OpenAIPromptFormatterOptions): Promise<ResponseInputItem[]> {
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
                    type: "input_image",
                    image_url: `data:${file.mime_type || "image/jpeg"};base64,${data}`,
                    detail: "auto",
                })
            }
        }

        if (msg.content) {
            parts.push({
                type: "input_text",
                text: msg.content,
            })
        }


        if (msg.role === PromptRole.system) {
            // For system messages, filter to only text parts
            const textParts = parts.filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text');
            const textContent = textParts.length === 1 && !msg.files ? textParts[0].text : textParts;
            const systemMsg: EasyInputMessage = {
                role: "system",
                content: textContent,
            };
            system.push(systemMsg);

            if (opts.useToolForFormatting && opts.schema) {
                system.forEach(s => {
                    if ((s as EasyInputMessage).role === 'system') {
                        const sysMsg = s as EasyInputMessage;
                        if (typeof sysMsg.content === 'string') {
                            sysMsg.content = "TOOL: " + sysMsg.content;
                        } else if (Array.isArray(sysMsg.content)) {
                            sysMsg.content.forEach((c: any) => {
                                if (c.type === "input_text") c.text = "TOOL: " + c.text;
                            });
                        }
                    }
                });
            }

        } else if (msg.role === PromptRole.safety) {
            const textParts = parts.filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text');
            const safetyMsg: EasyInputMessage = {
                role: "system",
                content: textParts,
            };

            if (Array.isArray(safetyMsg.content)) {
                safetyMsg.content.forEach((c: any) => {
                    if (c.type === "input_text") c.text = "DO NOT IGNORE - IMPORTANT: " + c.text;
                });
            }

            system.push(safetyMsg);
        } else if (msg.role === PromptRole.tool) {
            if (!msg.tool_use_id) {
                throw new Error("Tool use id is required for tool messages")
            }
            const toolOutputMsg: OpenAI.Responses.ResponseInputItem.FunctionCallOutput = {
                type: "function_call_output",
                call_id: msg.tool_use_id,
                output: msg.content || ""
            };
            others.push(toolOutputMsg);
        } else if (msg.role !== PromptRole.negative && msg.role !== PromptRole.mask) {
            const inputMsg: EasyInputMessage = {
                role: msg.role === 'assistant' ? 'assistant' : 'user',
                content: parts,
            };
            others.push(inputMsg);
        }

    }

    if (opts.result_schema && !opts.useToolForFormatting) {
        const schemaMsg: EasyInputMessage = {
            role: "system",
            content: [{
                type: "input_text",
                text: "IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. " + `<response_schema>${JSON.stringify(opts.result_schema)}</response_schema>`
            }]
        };
        system.push(schemaMsg);
    }

    // put system messages first and safety last
    return ([] as ResponseInputItem[]).concat(system).concat(others).concat(safety);

}

export interface OpenAIPromptFormatterOptions {
    multimodal?: boolean
    useToolForFormatting?: boolean
    schema?: Object
}
