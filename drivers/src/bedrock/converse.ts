import { DataSource, ExecutionOptions, readStreamAsString, readStreamAsUint8Array } from "@llumiverse/core";
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
                    return { json: parsedJson } satisfies ToolResultContentBlock.JsonMember as T extends 'content' ? ContentBlock : ToolResultContentBlock;
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
    try {
        return processFile(f, 'content');
    } catch (error) {
        throw new Error(`Failed to process file ${f.name} for prompt: ${error instanceof Error ? error.message : String(error)}`);
    }
}

async function processFileToToolContentBlock(f: DataSource): Promise<ToolResultContentBlock> {
    try {
        return processFile(f, 'tool');
    }
    catch (error) {
        throw new Error(`Failed to process file ${f.name} for tool response: ${error instanceof Error ? error.message : String(error)}`);
    }
}

export function converseConcatMessages(messages: Message[] | undefined): Message[] {
    if (!messages || messages.length === 0) return [];

    const needsMerging = messages.some((message, i) =>
        i < messages.length - 1 && message.role === messages[i + 1].role
    );
    // If no merging needed, return original array
    if (!needsMerging) {
        return messages;
    }

    const result: Message[] = [];
    let currentMessage = { ...messages[0] };
    for (let i = 1; i < messages.length; i++) {
        if (currentMessage.role === messages[i].role) {
            // Same role - concatenate content
            currentMessage.content = (currentMessage.content || []).concat(...(messages[i].content || []));
        } else {
            // Different role - push current and start new
            result.push(currentMessage);
            currentMessage = { ...messages[i] };
        }
    }

    result.push(currentMessage);
    return result;
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

export async function formatConversePrompt(segments: PromptSegment[], options: ExecutionOptions): Promise<ConverseRequest> {
    //Non-const for concat
    let system: SystemContentBlock.TextMember[] | undefined = [];
    const safety: Message[] = [];
    let messages: Message[] = [];

    for (const segment of segments) {
        // Role dependent processing
        if (segment.role === PromptRole.system) {
            system.push({ text: segment.content });
        } else if (segment.role === PromptRole.tool) {
            if (!segment.tool_use_id) {
                throw new Error("Tool use ID is required for tool segments");
            }
            //Tool use results (i.e. the model has requested a tool and this it the answer to that request)
            const toolContentBlocks: ToolResultContentBlock[] = [];
            //Text segments
            if (segment.content) {
                toolContentBlocks.push({ text: segment.content });
            }
            //Handle attached files
            for (const file of segment.files ?? []) {
                toolContentBlocks.push(await processFileToToolContentBlock(file));
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
            //User, Assistant or safety roles
            const contentBlocks: ContentBlock[] = [];
            //Text segments
            if (segment.content) {
                contentBlocks.push({ text: segment.content });
            }
            //Handle attached files
            for (const file of segment.files ?? []) {
                contentBlocks.push(await processFileToContentBlock(file));
            }
            //If there are no content blocks, skip this message
            if (contentBlocks.length !== 0) {
                const message = { content: contentBlocks, role: roleConversion(segment.role) };
                if (segment.role === PromptRole.safety) {
                    safety.push(message)
                } else {
                    messages.push(message);
                }
            }
        }
    }

    if (options.result_schema) {
        let schemaText: string;
        if (options.tools && options.tools.length > 0) {
            schemaText = "When not calling tools, the answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema, undefined, 2);
        } else {
            schemaText = "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(options.result_schema, undefined, 2);
        }
        system.push({ text: "IMPORTANT: " + schemaText });
    }

    // Safety messages are user messages that should be included at the end.
    if (safety.length > 0) {
        messages = messages.concat(safety);
    }

    //Conversations must start with a user message
    //Use the system messages if none are provided
    if (messages.length === 0) {
        const systemMessage = converseSystemToMessages(system);
        if (systemMessage?.content?.[0]?.text?.trim()) {
            messages.push(systemMessage);
        } else {
            throw new Error("Prompt must contain at least one message");
        }
        system = undefined;
    }

    if (system && system.length === 0) {
        system = undefined; // If no system messages, set to undefined
    }

    messages = converseConcatMessages(messages);

    return {
        modelId: undefined, //required property, but allowed to be undefined
        messages: messages,
        system: system,
    };
}
