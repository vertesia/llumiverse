import { DataSource, JSONSchema } from "@llumiverse/core";
import { PromptSegment, PromptRole } from "@llumiverse/core";
import {
    ConversationRole,
    ConverseRequest,
    Message,
    SystemContentBlock,
    ContentBlock,
    ToolResultContentBlock,
} from "@aws-sdk/client-bedrock-runtime";
import { parseS3UrlToUri } from "./s3.js";

function getJSONSafetyNotice(schema: JSONSchema) {
    return "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema, undefined, 2);
}

function roleConversion(role: PromptRole): ConversationRole {
    return role === PromptRole.assistant ? ConversationRole.ASSISTANT : ConversationRole.USER;
}

function mimeToImageType(mime: string): "png" | "jpeg" | "gif" | "webp" {
    if (mime.startsWith("image/")) {
        return mime.split("/")[1] as "png" | "jpeg" | "gif" | "webp";
    }
    return "png";
}

function mimeToDocType(mime: string): "pdf" | "csv" | "doc" | "docx" | "xls" | "xlsx" | "html" | "txt" | "md" {
    if (mime.startsWith("application/") || mime.startsWith("text/")) {
        return mime.split("/")[1] as "pdf" | "csv" | "doc" | "docx" | "xls" | "xlsx" | "html" | "txt" | "md";
    }
    return "txt";
}
function mimeToVideoType(mime: string): "mov" | "mkv" | "mp4" | "webm" | "flv" | "mpeg" | "mpg" | "wmv" | "three_gp" {
    if (mime.startsWith("video/")) {
        return mime.split("/")[1] as "mov" | "mkv" | "mp4" | "webm" | "flv" | "mpeg" | "mpg" | "wmv" | "three_gp";
    }
    return "mp4";
}

async function readStreamAsString(stream: ReadableStream): Promise<string> {
    const out: Buffer[] = [];
    for await (const chunk of stream as any) {
        out.push(Buffer.from(chunk));
    }
    return Buffer.concat(out).toString();
}

async function readStreamAsUint8Array(stream: ReadableStream): Promise<Uint8Array> {
    const out: Buffer[] = [];
    for await (const chunk of stream as any) {
        out.push(Buffer.from(chunk));
    }
    return Buffer.concat(out);
}

type FileProcessingMode = 'content' | 'tool';

async function processFile<T extends FileProcessingMode>(
    f: DataSource,
    mode: T
): Promise<T extends 'content' ? ContentBlock : ToolResultContentBlock> {
    const source = await f.getStream();

    //Image file - "png" | "jpeg" | "gif" | "webp"
    if (f.mime_type && f.mime_type.startsWith("image")) {
        const imageBlock = {
            image: {
                format: mimeToImageType(f.mime_type),
                source: { bytes: await readStreamAsUint8Array(source) },
            }
        };

        return mode === 'content'
            ? (imageBlock satisfies ContentBlock.ImageMember)
            : (imageBlock satisfies ToolResultContentBlock.ImageMember);
    }
    //Document file - "pdf | csv | doc | docx | xls | xlsx | html | txt | md"
    else if (f.mime_type && (f.mime_type.startsWith("text") || f.mime_type?.startsWith("application"))) {
        // Handle JSON files specially
        if (f.mime_type === "application/json" || (f.name && f.name.endsWith('.json'))) {
            const jsonContent = await readStreamAsString(source);
            try {
                const parsedJson = JSON.parse(jsonContent);
                if (mode === 'tool') {
                    return { json: parsedJson } satisfies ToolResultContentBlock.JsonMember as any;
                } else {
                    // ContentBlock doesn't support JSON, so treat as text
                    return { text: jsonContent } satisfies ContentBlock.TextMember;
                }
            } catch (error) {
                const textBlock = { text: jsonContent };
                return mode === 'content'
                    ? (textBlock satisfies ContentBlock.TextMember)
                    : (textBlock satisfies ToolResultContentBlock.TextMember);
            }
        } else {
            const documentBlock = {
                document: {
                    format: mimeToDocType(f.mime_type),
                    name: f.name,
                    source: { bytes: await readStreamAsUint8Array(source) },
                },
            };

            return mode === 'content'
                ? (documentBlock satisfies ContentBlock.DocumentMember)
                : (documentBlock satisfies ToolResultContentBlock.DocumentMember);
        }
    }
    //Video file - "mov | mkv | mp4 | webm | flv | mpeg | mpg | wmv | three_gp"
    else if (f.mime_type && f.mime_type.startsWith("video")) {
        let url_string = (await f.getURL()).toLowerCase();
        let url_format = new URL(url_string);
        if (url_format.hostname.endsWith("amazonaws.com") &&
            (url_format.hostname.startsWith("s3.") || url_format.hostname.includes(".s3."))) {
            //Convert to s3:// format
            const parsedUrl = parseS3UrlToUri(new URL(url_string));
            url_string = parsedUrl;
            url_format = new URL(parsedUrl);
        }

        const videoBlock = url_format.protocol === "s3:" ? {
            video: {
                format: mimeToVideoType(f.mime_type),
                source: {
                    s3Location: {
                        uri: url_string, //S3 URL
                        //bucketOwner:  We don't have this additional information.
                    }
                },
            },
        } : {
            video: {
                format: mimeToVideoType(f.mime_type),
                source: { bytes: await readStreamAsUint8Array(source) },
            },
        };

        return mode === 'content'
            ? (videoBlock satisfies ContentBlock.VideoMember)
            : (videoBlock satisfies ToolResultContentBlock.VideoMember);
    }
    //Fallback, send as text
    else {
        const textBlock = { text: await readStreamAsString(source) };
        return mode === 'content'
            ? (textBlock satisfies ContentBlock.TextMember)
            : (textBlock satisfies ToolResultContentBlock.TextMember);
    }
}

async function processFileToContentBlock(f: DataSource): Promise<ContentBlock> {
    return processFile(f, 'content');
}

async function processFileToToolContentBlock(f: DataSource): Promise<ToolResultContentBlock> {
    return processFile(f, 'tool');
}

export function converseConcatMessages(messages: Message[] | undefined): Message[] {
    if (!messages) return [];
    //Concatenate messages of the same role. Required to have alternative user and assistant roles
    for (let i = 0; i < messages.length - 1; i++) {
        if (messages[i].role === messages[i + 1].role) {
            messages[i].content = messages[i].content?.concat(...(messages[i + 1].content || []));
            messages.splice(i + 1, 1);
            i--;
        }
    }
    return messages;
}

export function converseSystemToMessages(system: SystemContentBlock[]): Message {
    return {
        content: [{ text: system.map(system => system.text).join('\n').trim() }],
        role: ConversationRole.USER
    };
}

export function converseRemoveJSONprefill(messages: Message[] | undefined): Message[] {
    //Remove the "```json" stop message
    if (messages && messages.length > 0) {
        if (messages[messages.length - 1].content?.[0].text === "```json") {
            messages.pop();
        }
    }
    return messages ?? [];
}

export function converseJSONprefill(messages: Message[] | undefined): Message[] {
    if (!messages) {
        messages = [];
    }

    //prefill the json
    messages.push({
        content: [{ text: "```json" }],
        role: ConversationRole.ASSISTANT,
    });
    return messages;
}

// Used to ignore unsupported roles. Typically these are things like image specific roles.
const unsupportedRoles = [
    PromptRole.negative,
    PromptRole.mask,
];

export async function formatConversePrompt(segments: PromptSegment[], schema?: JSONSchema): Promise<ConverseRequest> {
    //Non-const for concat
    let system: SystemContentBlock[] = [];
    const safety: SystemContentBlock[] = [];
    let messages: Message[] = [];

    for (const segment of segments) {
        // Role dependent processing
        if (segment.role === PromptRole.system) {
            system.push({ text: segment.content });
        } else if (segment.role === PromptRole.safety) {
            safety.push({ text: segment.content });
        } else if (segment.role === PromptRole.tool) {
            //Tool use results (i.e. the model has requested a tool and this it the answer to that request)
            const toolContentBlocks: ToolResultContentBlock[] = [];
            //Text segments
            if (segment.content) {
                toolContentBlocks.push({ text: segment.content });
            }
            //Handle attached files concurrently
            if (segment.files) {
                toolContentBlocks.push(
                    ...await Promise.all(segment.files.map(processFileToToolContentBlock))
                );
            }
            messages.push({
                content: [{
                    toolResult: {
                        toolUseId: segment.tool_use_id,
                        content: toolContentBlocks,
                    }
                }],
                role: ConversationRole.USER
            });
        } else if (!unsupportedRoles.includes(segment.role)) {
            //User or Assistant
            const contentBlocks: ContentBlock[] = [];
            //Text segments
            if (segment.content) {
                contentBlocks.push({ text: segment.content });
            }
            //Handle attached files concurrently
            if (segment.files) {
                contentBlocks.push(
                    ...await Promise.all(segment.files.map(processFileToContentBlock))
                );
            }
            messages.push({
                content: contentBlocks,
                role: roleConversion(segment.role),
            });
        }
    }

    //Conversations must start with a user message
    //Use the system messages if none are provided
    if (messages.length === 0) {
        const systemMessage = converseSystemToMessages(system);
        if (systemMessage?.content && systemMessage.content[0].text) {
            messages.push(systemMessage);
        } else {
            throw new Error("Prompt must contain at least one message");
        }
        system = [];
    }

    if (schema) {
        safety.push({ text: "IMPORTANT: " + getJSONSafetyNotice(schema) });
    }

    if (safety.length > 0) {
        system = system.concat(safety);
    }

    messages = converseConcatMessages(messages);

    return {
        modelId: undefined, //required property, but allowed to be undefined
        messages: messages,
        system: system,
    };
}
