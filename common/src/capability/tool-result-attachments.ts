import { getModelCapabilities } from '../capability.js';
import { type OpenAIApi, resolveOpenAIApi } from '../openai-api.js';
import { getBedrockMantleProtocol } from '../options/bedrock_mantle.js';
import { getGeminiModelVersion, parseGeminiVersion } from '../options/version-parsing.js';
import type { Providers } from '../types.js';

/**
 * Which attachments a tool result may carry to the model, per provider API.
 *
 * A tool result stays in the conversation history, so an attachment the provider rejects fails every later turn,
 * not just the one that added it. The drivers keep the attachments listed here and replace every other one with a
 * text note in the tool result ({@link formatOmittedAttachmentNote}). A new user upload is not filtered: the drivers
 * may still reject it, because the user can act on that error.
 *
 * The lists follow a live probe of each API (2026-11-25), not the documentation alone:
 * - Claude (Vertex, claude-haiku-5-5): `image` blocks take png, jpeg, gif and webp only, and base64 `document` blocks
 *   take application/pdf only; any other type is a 400. Text formats are sent as plain-text documents.
 * - Gemini 3.x on Vertex (gemini-3-flash-preview, gemini-3.5-flash): `functionResponse.parts` takes images (bmp, tiff
 *   and svg included), video, PDF and text (csv, JSON and XML included). Audio is accepted but not heard: the model
 *   described a sine tone as ducks, a recipe and Python code. A zip archive was a 400 on gemini-3-flash-preview.
 * - The Gemini Developer API was not probed (no key at hand); it is assumed to match Vertex.
 * - Gemini 2.5 on Vertex ignores `functionResponse.parts` ("I cannot see the file") and rejects video and svg there
 *   with a 400, so no attachment reaches it.
 * - OpenAI was not probed: its lists follow the formats the Responses and Chat Completions APIs document for image
 *   and file inputs, and what the drivers can encode.
 *
 * Keep the lists in this one place, next to the model capability tables, and update them from a probe.
 */
export type ToolResultAttachmentApi = 'claude' | 'gemini' | 'openai_responses' | 'openai_chat';

const COMMON_IMAGE_TYPES: ReadonlySet<string> = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp']);

const TEXT_APPLICATION_TYPES: ReadonlySet<string> = new Set([
    'application/json',
    'application/xml',
    'application/yaml',
    'application/x-yaml',
]);

/** Lower-cased, trimmed and without parameters, with the common `image/jpg` alias mapped to `image/jpeg`. */
export function normalizeAttachmentMimeType(mimeType: string | undefined): string {
    const type = (mimeType ?? '').split(';')[0].trim().toLowerCase();
    return type === 'image/jpg' ? 'image/jpeg' : type;
}

/**
 * Text formats the Claude driver sends as plain-text documents: text/*, JSON, XML and YAML. Media types such as
 * image/svg+xml are not text here: the drivers handle them as media.
 */
export function isTextAttachmentMimeType(mimeType: string | undefined): boolean {
    const type = normalizeAttachmentMimeType(mimeType);
    if (/^(image|audio|video)\//.test(type)) return false;
    return (
        type.startsWith('text/') || TEXT_APPLICATION_TYPES.has(type) || type.endsWith('+json') || type.endsWith('+xml')
    );
}

/**
 * Whether a Gemini model reads attachments in `functionResponse.parts`. Multimodal function responses arrived with
 * Gemini 3; a model whose version cannot be parsed is treated as the newest generation.
 */
function geminiReadsFunctionResponseParts(model: string): boolean {
    const version = getGeminiModelVersion(model);
    const parsed = version ? parseGeminiVersion(version) : undefined;
    return !parsed || parsed.major >= 3;
}

/**
 * Whether a model is known to take no image input on `provider`. A model the capability tables do not know is assumed
 * to take images, so a new model is not cut off from them.
 */
function isKnownTextOnlyModel(model: string, provider: string | undefined): boolean {
    if (!provider) return false;
    try {
        return getModelCapabilities(model, provider as Providers).input.image === false;
    } catch {
        return false;
    }
}

/**
 * Whether `api` can read an attachment of this MIME type in a tool result sent to `model`. With `provider`, an image
 * is also refused on the OpenAI APIs when the capability tables know the model takes no image input (gpt-oss, for
 * example); the Claude and Gemini families all take images.
 */
export function isToolResultAttachmentSupported(
    api: ToolResultAttachmentApi,
    mimeType: string | undefined,
    model: string,
    provider?: string,
): boolean {
    const type = normalizeAttachmentMimeType(mimeType);
    if (!type) return false;
    switch (api) {
        case 'claude':
            return COMMON_IMAGE_TYPES.has(type) || type === 'application/pdf' || isTextAttachmentMimeType(type);
        case 'gemini':
            return (
                geminiReadsFunctionResponseParts(model) &&
                (type.startsWith('image/') ||
                    type.startsWith('video/') ||
                    type === 'application/pdf' ||
                    isTextAttachmentMimeType(type))
            );
        case 'openai_responses':
            if (COMMON_IMAGE_TYPES.has(type)) return !isKnownTextOnlyModel(model, provider);
            return type === 'application/pdf' || type.startsWith('text/');
        case 'openai_chat':
            // A tool message carries text only: the driver moves each image into the user message that follows it.
            if (COMMON_IMAGE_TYPES.has(type)) return !isKnownTextOnlyModel(model, provider);
            return type.startsWith('text/');
    }
}

/** The wire API a driver is configured with (its `openai_api` option). */
export type ToolResultOpenAICompatibleApi = OpenAIApi;

/**
 * The API whose tool-result rules apply to `model` on `provider`, or undefined when no rule is known (the drivers
 * then send tool-result attachments as they always have). Lets an application tell ahead of a call which attachments
 * a model will not get. It follows the protocol the driver picks: `openaiApi` for a provider in
 * `OPENAI_API_DEFAULTS` (that provider's default when omitted, as in the driver factory), and the model's
 * protocol on Bedrock Mantle.
 */
export function toolResultAttachmentApi(
    provider: string,
    model: string,
    options: { openaiApi?: ToolResultOpenAICompatibleApi } = {},
): ToolResultAttachmentApi | undefined {
    const name = model.toLowerCase();
    const isClaude = name.includes('claude');
    switch (provider) {
        case 'anthropic':
            return 'claude';
        case 'vertexai':
            if (isClaude) return 'claude';
            return name.includes('gemini') ? 'gemini' : undefined;
        case 'bedrock':
        case 'azure_foundry':
            return isClaude ? 'claude' : undefined;
        case 'bedrock_mantle':
            switch (getBedrockMantleProtocol(model)) {
                case 'messages':
                    return 'claude';
                case 'responses':
                    return 'openai_responses';
                case 'chat_completions':
                    return 'openai_chat';
                default:
                    return undefined;
            }
        case 'openai':
        case 'azure_openai':
            return 'openai_responses';
        default:
            switch (resolveOpenAIApi(provider, options.openaiApi)) {
                case 'responses':
                    return 'openai_responses';
                case 'chat_completions':
                    return 'openai_chat';
                default:
                    return undefined;
            }
    }
}

/** The text that stands in for an attachment the model cannot read, appended to the tool result. */
export function formatOmittedAttachmentNote(attachment: { name?: string; mime_type?: string }, model: string): string {
    const modelName = model.split('/').pop() || model;
    const mimeType = normalizeAttachmentMimeType(attachment.mime_type) || 'unknown type';
    return `[attachment omitted: ${attachment.name || 'unnamed'} (${mimeType}) — ${modelName} cannot read this type in a tool result]`;
}
