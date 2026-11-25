// This file is used by multiple drivers
// to format prompts in a way that is compatible with OpenAI's API.

import {
    type DataSource,
    formatOmittedAttachmentNote,
    isToolResultAttachmentSupported,
    normalizeAttachmentMimeType,
    type PromptOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/common';
import { readStreamAsBase64, readStreamAsString } from '@llumiverse/core';
import type OpenAI from 'openai';
import { truncateDataUrlForDebug } from '../shared/debug-prompt.js';
import { buildResultSchemaInstruction } from '../shared/result-schema-instruction.js';
import { splitToolResultFiles, withAttachmentNotes } from '../shared/tool-result-attachments.js';

// Types for Response API
type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type ResponseInputContent = OpenAI.Responses.ResponseInputContent;
type EasyInputMessage = OpenAI.Responses.EasyInputMessage;

const imageMasks = new WeakMap<ResponseInputItem[], OpenAI.Responses.ResponseInputImage[]>();
export function getImageMasks(prompt: ResponseInputItem[]): OpenAI.Responses.ResponseInputImage[] {
    return imageMasks.get(prompt) ?? [];
}

function isResponseInputContent(value: unknown): value is ResponseInputContent {
    return (
        typeof value === 'object' &&
        value !== null &&
        'type' in value &&
        (value.type === 'input_text' || value.type === 'input_image' || value.type === 'input_file')
    );
}

/**
 * Map an attachment to the Responses content part that carries it. Types the API has no
 * representation for (audio, video) are skipped rather than mislabelled as an image, which
 * the API rejects outright. A missing mime type keeps the historical image assumption.
 */
async function fileToInputContent(file: DataSource): Promise<ResponseInputContent | undefined> {
    // The same normalised type the tool-result filter judged, so a kept `IMAGE/PNG` is not dropped here.
    const mimeType = normalizeAttachmentMimeType(file.mime_type);
    if (!mimeType || mimeType.startsWith('image/')) {
        return {
            type: 'input_image',
            image_url: `data:${mimeType || 'image/jpeg'};base64,${await readStreamAsBase64(await file.getStream())}`,
            detail: 'auto',
        };
    }
    if (mimeType === 'application/pdf') {
        return {
            type: 'input_file',
            filename: file.name,
            file_data: `data:${mimeType};base64,${await readStreamAsBase64(await file.getStream())}`,
        };
    }
    if (mimeType.startsWith('text/')) {
        return { type: 'input_text', text: await readStreamAsString(await file.getStream()) };
    }
    return undefined;
}

function inputTextParts(parts: ResponseInputContent[]): OpenAI.Responses.ResponseInputText[] {
    return parts.filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text');
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
    const masks: OpenAI.Responses.ResponseInputImage[] = [];

    for (const msg of segments) {
        const fileParts: ResponseInputContent[] = [];

        // A stored tool result is replayed on every later turn: an attachment the API cannot take (audio, bmp, …)
        // becomes a note in the tool output instead of failing or vanishing.
        const { files, notes } =
            msg.role === PromptRole.tool
                ? splitToolResultFiles(
                      msg.files,
                      opts.toolResultApi ?? 'openai_responses',
                      opts.attachmentModel ?? opts.model,
                      {
                          logger: opts.logger,
                          provider: opts.provider,
                      },
                  )
                : { files: msg.files, notes: [] };

        //generate the parts based on PromptSegment
        if (files) {
            for (const file of files) {
                const part = await fileToInputContent(file);
                if (part) fileParts.push(part);
            }
        }

        if (msg.role === PromptRole.mask) {
            for (const part of fileParts) if (part.type === 'input_image') masks.push(part);
            continue;
        }
        const parts: ResponseInputContent[] = [...fileParts];

        if (msg.content) {
            parts.push({
                type: 'input_text',
                text: msg.content,
            });
        }

        if (opts.imageGeneration) {
            if (msg.role !== PromptRole.negative) others.push({ role: 'user', content: parts });
            continue;
        }
        if (msg.role === PromptRole.system) {
            // For system messages, filter to only text parts
            const textParts = inputTextParts(parts);
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
            const safetyMsg: EasyInputMessage = {
                type: 'message',
                role: 'system',
                content: inputTextParts(parts),
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
            // A tool result can carry attachments - an image a tool rendered or promoted into the
            // conversation, a document it fetched. The Responses API accepts those as a content
            // list on `function_call_output`; emitting only the text would silently drop them and
            // leave the model insisting it cannot see the image it just asked for.
            const text = withAttachmentNotes(msg.content, notes);
            const toolOutputMsg: OpenAI.Responses.ResponseInputItem.FunctionCallOutput = {
                type: 'function_call_output',
                call_id: msg.tool_use_id,
                // The tool's own output reads first, then its attachments. With no attachments,
                // keep the plain-string form the API has always accepted.
                output:
                    fileParts.length > 0
                        ? [...(text ? [{ type: 'input_text' as const, text }] : []), ...fileParts]
                        : text,
            };
            others.push(toolOutputMsg);
        } else if (msg.role !== PromptRole.negative) {
            const inputMsg: EasyInputMessage = {
                type: 'message',
                role: msg.role === 'assistant' ? 'assistant' : 'user',
                content: parts,
            };
            others.push(inputMsg);
        }
    }

    if (opts.result_schema && !opts.useToolForFormatting) {
        const schemaPosition = opts.resultSchemaPosition ?? 'system';
        const schemaMsg: EasyInputMessage = {
            type: 'message',
            role: schemaPosition === 'suffix' ? 'user' : 'system',
            content: [
                {
                    type: 'input_text',
                    text: buildResultSchemaInstruction(opts.result_schema),
                },
            ],
        };
        if (schemaPosition === 'suffix') {
            others.push(schemaMsg);
        } else {
            system.push(schemaMsg);
        }
    }

    // put system messages first and safety last
    const result = ([] as ResponseInputItem[]).concat(system).concat(others).concat(safety);
    imageMasks.set(result, masks);
    return result;
}

const DATA_URL_MIME = /^data:([^;,]*)/i;

/** The MIME type a `data:` URL names, or undefined for any other URL or file reference. */
function dataUrlMimeType(url: string | null | undefined): string | undefined {
    return url ? DATA_URL_MIME.exec(url)?.[1] : undefined;
}

/**
 * A copy of `items` in which each tool output keeps only the attachments `api` can read for `model`, with a note in
 * place of each other one. New tool results are filtered when the prompt is built; this runs on the request copy of
 * the stored conversation, whose tool outputs were filtered for the model of the turn that added them, or by an
 * earlier release. The stored conversation is not changed.
 *
 * Exception to "preserve supplied input", as for new tool results: a replayed attachment the model cannot read fails
 * every later turn. The omission is logged with the file name and type only.
 */
export function omitUnreadableToolOutputParts(
    items: ResponseInputItem[],
    api: 'openai_responses' | 'openai_chat',
    model: string,
    options: { provider?: string; logger?: OpenAIPromptFormatterOptions['logger'] } = {},
): ResponseInputItem[] {
    const omitted: Array<{ file_name: string; mime_type: string }> = [];
    const result = items.map((item) => {
        if (!('type' in item) || item.type !== 'function_call_output' || !Array.isArray(item.output)) return item;
        const kept: typeof item.output = [];
        const notes: string[] = [];
        for (const part of item.output) {
            const attachment =
                part.type === 'input_image'
                    ? { name: 'image', mime_type: dataUrlMimeType(part.image_url) }
                    : part.type === 'input_file'
                      ? { name: part.filename || 'file', mime_type: dataUrlMimeType(part.file_data) }
                      : undefined;
            // A part given by URL or file ID names no type: Responses fetches it itself; Chat Completions cannot.
            const readable =
                !attachment ||
                (attachment.mime_type === undefined
                    ? api === 'openai_responses'
                    : isToolResultAttachmentSupported(api, attachment.mime_type, model, options.provider));
            if (readable) {
                kept.push(part);
            } else if (attachment) {
                notes.push(formatOmittedAttachmentNote(attachment, model));
                omitted.push({
                    file_name: attachment.name,
                    mime_type: normalizeAttachmentMimeType(attachment.mime_type),
                });
            }
        }
        if (notes.length === 0) return item;
        return { ...item, output: [...kept, { type: 'input_text' as const, text: notes.join('\n') }] };
    });
    if (omitted.length > 0) {
        options.logger?.warn(
            { model, api, provider: options.provider, omitted_attachments: omitted },
            '[llumiverse] Replayed tool result attachments the model cannot read were replaced with a note',
        );
    }
    return result;
}

export interface OpenAIPromptFormatterOptions {
    /** Receives a warning when a tool result attachment is replaced with a note. */
    logger?: { warn: (data: Record<string, unknown>, message: string) => void };
    /** The driver's provider, so tool-result images are not sent to a model known to take none. */
    provider?: string;
    /**
     * The API whose tool-result attachment rules apply: Chat Completions when the prompt is converted to it
     * (`convertResponseItemsToChatMessages`). Responses by default.
     */
    toolResultApi?: 'openai_responses' | 'openai_chat';
    /** The model the attachment rules judge when it differs from `model`, such as the source model of a deployment. */
    attachmentModel?: string;
    imageGeneration?: boolean;
    multimodal?: boolean;
    useToolForFormatting?: boolean;
    schema?: object;
    resultSchemaPosition?: 'system' | 'suffix';
}

// Chat Completions API types
type ChatCompletionMessageParam = OpenAI.Chat.Completions.ChatCompletionMessageParam;
type ChatCompletionContentPart = OpenAI.Chat.Completions.ChatCompletionContentPart;

/**
 * Convert Response API items (`ResponseInputItem[]`) into Chat Completions messages
 * (`ChatCompletionMessageParam[]`).
 *
 * This is useful for adapters that start from Response API conversation items but must call an
 * OpenAI Chat Completions-compatible endpoint. Crucially, images become `{ type: 'image_url', image_url: { url } }`
 * (the Chat Completions shape) instead of the Responses `input_image` shape, which those
 * providers silently ignore.
 */
export function convertResponseItemsToChatMessages(items: ResponseInputItem[]): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [];
    // A Chat Completions tool message carries text only: the images of consecutive tool outputs follow them in one
    // user message, so every tool call is still answered right after the assistant message that made it.
    let toolImages: ChatCompletionContentPart[] = [];
    const flushToolImages = () => {
        if (toolImages.length === 0) return;
        messages.push({ role: 'user', content: toolImages });
        toolImages = [];
    };

    for (const item of items) {
        if (!('type' in item && item.type === 'function_call_output')) flushToolImages();
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
            if (!output.call_id) {
                throw new Error('Cannot convert a function_call_output without call_id to Chat Completions.');
            }
            messages.push({
                role: 'tool',
                tool_call_id: output.call_id ?? output.id ?? '',
                content:
                    typeof output.output === 'string'
                        ? output.output
                        : toolOutputText(output.output, output.call_id, toolImages),
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

    flushToolImages();
    return messages;
}

/**
 * The text of a Responses tool output given as content parts, with each image replaced by a reference and moved to
 * `images`. Any other part (a file, which Chat Completions cannot carry in a tool message) becomes a note: serializing
 * it as JSON would put its base64 bytes in the tool message. Callers filter with
 * {@link omitUnreadableToolOutputParts} first, so this is a fallback. The text is never empty, which several
 * endpoints reject.
 */
function toolOutputText(
    parts: OpenAI.Responses.ResponseInputItem.FunctionCallOutput['output'],
    callId: string,
    images: ChatCompletionContentPart[],
): string {
    if (!Array.isArray(parts)) return JSON.stringify(parts);
    const text: string[] = [];
    let imageIndex = 0;
    for (const part of parts) {
        if (part.type === 'input_text') {
            text.push(part.text);
        } else if (part.type === 'input_image' && part.image_url) {
            imageIndex++;
            images.push(
                { type: 'text', text: `Image ${imageIndex} from tool result ${callId}:` },
                { type: 'image_url', image_url: { url: part.image_url } },
            );
            text.push(`[Image ${imageIndex} attached below]`);
        } else if (part.type === 'input_file') {
            const mimeType = normalizeAttachmentMimeType(dataUrlMimeType(part.file_data)) || 'unknown type';
            text.push(
                `[attachment omitted: ${part.filename || 'file'} (${mimeType}) — a Chat Completions tool result cannot carry a file]`,
            );
        }
    }
    return text.join('\n') || 'No output';
}
