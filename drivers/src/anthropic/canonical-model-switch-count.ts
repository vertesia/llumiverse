import type Anthropic from '@anthropic-ai/sdk';
import type { ModelTarget } from '@llumiverse/conversation';
import { preflightJsonInput } from '@llumiverse/conversation';
import { JsonObjectSchema } from '@llumiverse/conversation/schemas';
import type { CanonicalModelSwitchCountResult } from '@llumiverse/core';
import { boundedCanonicalProjectionOperation } from '../openai/canonical-request-measurement.js';
import {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
} from '../shared/claude-messages-conversation-adapter.js';

const COUNT_PROFILE = 'anthropic.messages.count_tokens:v1';
const COUNTABLE_BODY_FIELDS = new Set([
    'model',
    'messages',
    'system',
    'max_tokens',
    'stream',
    'temperature',
    'top_p',
    'top_k',
    'stop_sequences',
    'metadata',
    'service_tier',
]);

type TextPart = { type: 'text'; text: string };
type TextMessage = { role: 'user' | 'assistant'; content: string | TextPart[] };

function textParts(value: unknown): TextPart[] | undefined {
    if (!Array.isArray(value)) return undefined;
    const parts: TextPart[] = [];
    for (const part of value) {
        if (
            typeof part !== 'object' ||
            part === null ||
            Array.isArray(part) ||
            Object.keys(part).some((key) => key !== 'type' && key !== 'text') ||
            !('type' in part) ||
            part.type !== 'text' ||
            !('text' in part) ||
            typeof part.text !== 'string'
        )
            return undefined;
        parts.push({ type: 'text', text: part.text });
    }
    return parts;
}

function countableBody(
    nativeInput: unknown,
    target: ModelTarget,
): Anthropic.Messages.MessageCountTokensParams | undefined {
    if (!preflightJsonInput(nativeInput, { max_bytes: 768 * 1024 }).success) return undefined;
    const native = JsonObjectSchema.safeParse(structuredClone(nativeInput));
    if (!native.success) return undefined;
    const body = native.data;
    if (
        target.provider !== 'anthropic' ||
        target.protocol !== CLAUDE_MESSAGES_PROTOCOL ||
        target.adapter_version !== CLAUDE_MESSAGES_ADAPTER_VERSION ||
        body.model !== target.model ||
        Object.keys(body).some((key) => !COUNTABLE_BODY_FIELDS.has(key)) ||
        !Array.isArray(body.messages)
    )
        return undefined;
    const messages: TextMessage[] = [];
    for (const message of body.messages) {
        if (typeof message !== 'object' || message === null || Array.isArray(message)) return undefined;
        if (Object.keys(message).some((key) => key !== 'role' && key !== 'content')) return undefined;
        if (!('role' in message) || !('content' in message)) return undefined;
        const role = message.role;
        if (role !== 'user' && role !== 'assistant') return undefined;
        const content = typeof message.content === 'string' ? message.content : textParts(message.content);
        if (content === undefined) return undefined;
        messages.push({ role, content });
    }
    const system = body.system;
    const parsedSystem = system === undefined || typeof system === 'string' ? system : textParts(system);
    if (system !== undefined && parsedSystem === undefined) return undefined;
    return {
        model: target.model,
        messages,
        ...(parsedSystem === undefined ? {} : { system: parsedSystem }),
    };
}

/** Identified provider-estimated text profile; unsupported native fields fail closed. This is not generation. */
export async function countAnthropicModelSwitchRequest(
    client: Pick<Anthropic, 'messages'>,
    nativeInput: unknown,
    target: ModelTarget,
    signal?: AbortSignal,
): Promise<CanonicalModelSwitchCountResult> {
    const body = countableBody(nativeInput, target);
    if (body === undefined)
        return { status: 'unavailable', reason: 'Anthropic count profile supports only plain text Messages bodies' };
    const counted = await boundedCanonicalProjectionOperation(
        async (ownedSignal) => client.messages.countTokens(body, { signal: ownedSignal, timeout: 30000 }),
        signal,
        30000,
    );
    if (!Number.isSafeInteger(counted.input_tokens) || counted.input_tokens < 0)
        throw new Error('Anthropic token counter returned an invalid input count');
    return { status: 'counted', input_tokens: counted.input_tokens, profile: COUNT_PROFILE };
}
