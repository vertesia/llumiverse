import { PromptRole } from "../index.js";
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
        } else {
            user.push({
                content: msg.content,
                role: msg.role || 'user',
            })
        }
    }

    // put system mesages first and safety last
    return system.concat(user).concat(safety);
}

export function formatOpenAILikeMultimodalPrompt(segments: PromptSegment[]): OpenAIMessage[] {
    const system: OpenAIMessage[] = [];
    const safety: OpenAIMessage[] = [];
    const others: OpenAIMessage[] = [];

    for (const msg of segments) {

        const parts: (OpenAIContentPartImage | OpenAIContentPartText)[] = [];

        //generate the parts based on promptsegment
        if (msg.files) {
            for (const file of msg.files) {
                parts.push({
                    image_url: {
                        url: file.url
                    },
                    type: "image_url"
                }
                )
            }
        } else {
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

            } else if (msg.role === PromptRole.safety) {
                const safetyMsg: OpenAIMessage = {
                    role: "system",
                    content: parts
                }

                safetyMsg.content.forEach( c => {
                    if (c.type === "text") c.text = "DO NOT IGNORE - IMPORTANT: " + c.text;
                })

                system.push(safetyMsg)
 
            } else {
                others.push({
                    role: msg.role ?? 'user',
                    content: parts
                })
            }


       
    }

    // put system mesages first and safety last
    return system.concat(others).concat(safety);

}