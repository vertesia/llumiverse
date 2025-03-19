import { JSONSchema4 } from "json-schema";
import { PromptRole, PromptSegment, readStreamAsBase64 } from "../index.js";
//import { readStreamAsBase64 } from "../stream.js";
import { getJSONSafetyNotice } from "./commons.js";

export interface NovaMessage {
    role: 'user' | 'assistant',
    content: NovaMessagePart[]
}

export interface NovaSystemMessage {
    text: string
}

interface NovaMessagePart {
    text?: string // only set for text messages
    image?: {
        format: "jpeg" | "png" | "gif" | "webp",
        source: {
            bytes: string //"base64",
        }
    }
    video?: {
        format: "mkv" | "mov" | "mp4" | "webm" | "three_gp" | "flv" | "mpeg" | "mpg" | "wmv",
        source: {
            //Option 1: sending a s3 location
            s3Location?: {
                uri: string, // example: s3://my-bucket/object-key
                bucketOwner: string // optional. example: "123456789012"
            }
            //Option 2: sending a base64 encoded video
            bytes?: string //"base64",
        }
    }
}

export interface NovaMessagesPrompt {
    system?: NovaSystemMessage[];
    messages: NovaMessage[];
    negative?: string;
    mask?: string;
}

/**
 * A formatter used by Bedrock to format prompts for nova related models
 */

export async function formatNovaPrompt(segments: PromptSegment[], schema?: JSONSchema4): Promise<NovaMessagesPrompt> {
    const system: string[] = [];
    const safety: string[] = [];
    const messages: NovaMessage[] = [];
    let negative: string = "";
    let mask: string = "";

    for (const segment of segments) {

        const parts: NovaMessagePart[] = [];
        if (segment.files) for (const f of segment.files) {
            //TODO add video support
            if (!f.mime_type?.startsWith('image')) {
                continue;
            }

            const source = await f.getStream();
            const data = await readStreamAsBase64(source);
            const format = f.mime_type?.split('/')[1] || 'png';

            parts.push({
                image: {
                    format: format as "jpeg" | "png" | "gif" | "webp",
                    source: {
                        bytes: data
                    }
                }
            })
        }

        if (segment.content) {
            parts.push({
                text: segment.content
            })
        }

        if (segment.role === PromptRole.system) {
            system.push(segment.content);
        } else if (segment.role === PromptRole.safety) {
            safety.push(segment.content);
        } else if (messages.length > 0 && messages[messages.length - 1].role === segment.role) {
            //Maybe can remove for nova?
            //concatenate messages of the same role (Claude requires alternative user and assistant roles)
            messages[messages.length - 1].content.push(...parts);
        } else if (segment.role === PromptRole.negative) {
            negative = negative.concat(segment.content, ', ');
        } else if (segment.role === PromptRole.mask) {
            mask = mask.concat(segment.content, ' ');
        } else if (segment.role !== PromptRole.tool) {
            messages.push({
                role: segment.role,
                content: parts
            });
        }
    }

    if (schema) {
        safety.push("IMPORTANT: " + getJSONSafetyNotice(schema));
    }

    // messages must contains at least 1 item. If the prompt doesn;t contains a user message (but only system messages)
    // we need to put the system messages in the messages array

    let systemMessage = system.join('\n').trim();
    if (messages.length === 0) {
        if (!systemMessage) {
            throw new Error('Prompt must contain at least one message');
        }
        messages.push({ content: [{ text: systemMessage }], role: 'user' });
        systemMessage = safety.join('\n');
    } else if (safety.length > 0) {
        systemMessage = systemMessage + '\n\nIMPORTANT: ' + safety.join('\n');
    }

    /*start Nova's message to amke sure it answers properly in JSON
   if enabled, this requires to add the { to Nova's response*/

    if (schema) {
        messages.push({
            role: "assistant",
            content: [{
                text: "{"
            }]
        });
    }

    // put system mesages first and safety last
    return {
        system: systemMessage ? [{ text: systemMessage }] : [{ text: "" }],
        messages: messages,
        negative: negative,
        mask: mask,
    }
}
