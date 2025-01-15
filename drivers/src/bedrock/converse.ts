import { JSONSchema4 } from "json-schema";
import { PromptSegment, PromptRole } from "@llumiverse/core";
import { ConversationRole, ConverseRequest, Message, SystemContentBlock, ContentBlock } from "@aws-sdk/client-bedrock-runtime";

function getJSONSafetyNotice(schema: JSONSchema4) {
    return "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema, undefined, 2);
}

function roleConversion(role: PromptRole): ConversationRole {
    return role === PromptRole.assistant ? ConversationRole.ASSISTANT : ConversationRole.USER;
}

function mimeToImageType(mime: string): "png" | "jpeg" | "gif" | "webp" {
    if (mime.startsWith('image/')) {
        return mime.split('/')[1] as "png" | "jpeg" | "gif" | "webp";
    }
    return 'png';
}

function mimeToDocType(mime: string): "pdf" | "csv" | "doc" | "docx" | "xls" | "xlsx" | "html" | "txt" | "md"{
    if (mime.startsWith('application/') || mime.startsWith('text/')) {
        return mime.split('/')[1] as "pdf" | "csv" | "doc" | "docx" | "xls" | "xlsx" | "html" | "txt" | "md";
    }
    return 'txt';
}
function mimeToVideoType(mime: string): "mov" | "mkv" | "mp4" | "webm" | "flv" | "mpeg" | "mpg" | "wmv" | "three_gp"{
    if (mime.startsWith('video/')) {
        return mime.split('/')[1] as "mov" | "mkv" | "mp4" | "webm" | "flv" | "mpeg" | "mpg" | "wmv" | "three_gp";
    }
    return 'mp4';
}

async function readStreamAsString(stream: ReadableStream) : Promise<string> {
    const out: Buffer[] = [];
    for await (const chunk of stream as any) {
        out.push(Buffer.from(chunk));
    }
    return Buffer.concat(out).toString();
}

async function readStreamAsUint8Array(stream: ReadableStream) : Promise<Uint8Array> {
    const out: Buffer[] = [];
    for await (const chunk of stream as any) {
        out.push(Buffer.from(chunk));
    }
    return Buffer.concat(out);
}

export function converseConcatMessages(messages: Message[] | undefined): Message[] {
    if (!messages) return [];
    //Concatenate messages of the same role. Required to have alternative user and assistant roles
    for (let i = 0; i < messages.length - 1; i++) {
        if (messages[i].role === messages[i + 1].role) {
            messages[i].content = messages[i].content?.concat(...messages[i+1].content || []);
            messages.splice(i + 1, 1);
            i--;
        }
    }
    return messages;
}

export function converseSystemToMessages(system: SystemContentBlock[]): Message {
    return ({
        content: [{text: system.map(system => system.text).join('\n').trim()}],
        role: ConversationRole.USER
    });
   
}

export async function fortmatConversePrompt(segments: PromptSegment[], schema?: JSONSchema4): Promise<ConverseRequest> {
    //Non-const for concat
    let system: SystemContentBlock[] = [];
    const safety: SystemContentBlock[] = [];
    let messages: Message[] = [];

    for (const segment of segments) {
        const parts: Message[] = [];

        //File segments
        if (segment.files) for (const f of segment.files) {
            const source = await f.getStream();
            let content: ContentBlock[];

            //Image file - "png" | "jpeg" | "gif" | "webp"
            if (f.mime_type && f.mime_type.startsWith('image')) {
                content = [{
                    image: {
                        format: mimeToImageType(f.mime_type),
                        source: { bytes: await readStreamAsUint8Array(source) },
                    }
                }];

            //Document file - "pdf | csv | doc | docx | xls | xlsx | html | txt | md"
            } else if (f.mime_type && (f.mime_type.startsWith('text') || f.mime_type?.startsWith('application'))) {
                content = [
                    { text: f.name },
                    {
                        document: {
                            format: mimeToDocType(f.mime_type),
                            name: f.name,
                            source: { bytes: await readStreamAsUint8Array(source) }
                        }
                    }
                ];

            //Video file - "mov | mkv | mp4 | webm | flv | mpeg | mpg | wmv | three_gp"
            } else if (f.mime_type && f.mime_type.startsWith('video')) {
                content = [{
                    video: {
                        format: mimeToVideoType(f.mime_type),
                        source: { bytes: await readStreamAsUint8Array(source) },
                    }
                }];
                
            //Fallback, send string
            } else {
                content = [{ text: await readStreamAsString(source) }];
            }

            parts.push({
                content: content,
                role: roleConversion(segment.role),
            })
        }

        //Text segments
        if (segment.content) {
            parts.push({
                content: [{text: segment.content}],
                role: roleConversion(segment.role),
            })
        }

        if (segment.role === PromptRole.system) {
            system.push({text: segment.content});
        } else if (segment.role === PromptRole.safety) {
            safety.push({ text: segment.content });
        } else { //User or Assistant
            messages = messages.concat(parts);
        }
    }

    //Conversations must start with a user message
    //Use the system messages if none are provided
    if (messages.length === 0) {
        const systemMessage = converseSystemToMessages(system);
        if (systemMessage?.content && systemMessage.content[0].text) {
            messages.push(systemMessage);
        } else {
            throw new Error('Prompt must contain at least one message');
        }
        system = [];
    }

    if (schema) {
        safety.push({text: "IMPORTANT: " + getJSONSafetyNotice(schema)});
        system = system.concat(safety);

        //prefill the json
        messages.push({
            content: [{text:"```json"}],
            role: ConversationRole.ASSISTANT
        });
    }

    messages = converseConcatMessages(messages);

    return {
        modelId: undefined,     //required property, but allowed to be undefined
        messages: messages,
        system: system,
    }
}
