import { PromptRole, PromptOptions } from "../index.js";
import { readStreamAsBase64 } from "../stream.js";
import { PromptSegment } from "../types.js";


export interface OpenAITextMessage {
    content: string;
    role: "system" | "user" | "assistant";
}

export interface OpenAIMessage {
    content: (OpenAIContentPartText | OpenAIContentPartImage)[]
    role: "system" | "user" | "assistant";
    name?: string;
}

export interface OpenAIContentPartText {
    type: "text";
    text: string
}

export interface OpenAIContentPartImage {
    type: "image_url";
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
        } else if (msg.role !== PromptRole.negative && msg.role !== PromptRole.mask) {
            user.push({
                content: msg.content,
                role: msg.role || 'user',
            })
        }
    }

    // put system mesages first and safety last
    return system.concat(user).concat(safety);
}


export async function formatOpenAILikeMultimodalPrompt(segments: PromptSegment[], opts: PromptOptions & OpenAIPromptFormatterOptions): Promise<OpenAIMessage[]> {
    const system: OpenAIMessage[] = [];
    const safety: OpenAIMessage[] = [];
    const others: OpenAIMessage[] = [];

    for (const msg of segments) {

        const parts: (OpenAIContentPartImage | OpenAIContentPartText)[] = [];

        //generate the parts based on promptsegment
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
            system.push({
                role: "system",
                content: parts
            })


            if (opts.useToolForFormatting && opts.schema) {
                system.forEach(s => {
                    s.content.forEach(c => {
                        if (c.type === "text") c.text = "TOOL: " + c.text;
                    })
                })
            }

        } else if (msg.role === PromptRole.safety) {
            const safetyMsg: OpenAIMessage = {
                role: "system",
                content: parts
            }

            safetyMsg.content.forEach(c => {
                if (c.type === "text") c.text = "DO NOT IGNORE - IMPORTANT: " + c.text;
            })

            system.push(safetyMsg)

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

    // put system mesages first and safety last
    return system.concat(others).concat(safety);

}

export interface OpenAIPromptFormatterOptions {
    multimodal?: boolean
    useToolForFormatting?: boolean
    schema?: Object
}