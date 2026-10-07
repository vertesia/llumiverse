import type Anthropic from '@anthropic-ai/sdk';
import type { MessageCountTokensParams } from '@anthropic-ai/sdk/resources/messages.js';
import { fingerprintJson, type ModelTarget, preflightJsonInput } from '@llumiverse/conversation';
import { JsonObjectSchema, ModelTargetSchema } from '@llumiverse/conversation/schemas';
import { boundedCanonicalProjectionOperation } from '../openai/canonical-request-measurement.js';
import {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
} from '../shared/claude-messages-conversation-adapter.js';

export const ANTHROPIC_INDEXED_COUNT_PROFILE = 'anthropic.messages.count_tokens:v1';
// Identifies this registered count-body mapping, not an undisclosed provider tokenizer algorithm.
export const ANTHROPIC_INDEXED_COUNT_VERSION = 'anthropic.messages.count_tokens:projection-2026-10-05.v1';
const COUNT_FIELDS = [
    'model',
    'messages',
    'system',
    'tools',
    'tool_choice',
    'thinking',
    'output_config',
    'cache_control',
] as const;
// These generation controls do not alter the input sequence. Output reserve is separately enforced by the host.
const GENERATION_FIELDS = new Set([
    ...COUNT_FIELDS,
    'max_tokens',
    'stream',
    'temperature',
    'top_p',
    'top_k',
    'stop_sequences',
    'metadata',
    'service_tier',
]);

/** Deterministic projection from the actual generation body to the provider's count API body. */
export function anthropicIndexedCountBody(nativeInput: unknown, targetInput: ModelTarget): MessageCountTokensParams {
    if (!preflightJsonInput(nativeInput, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new RangeError('Anthropic indexed count body exceeds its bounded JSON profile');
    const body = JsonObjectSchema.parse(structuredClone(nativeInput));
    const target = ModelTargetSchema.parse(structuredClone(targetInput));
    if (
        target.provider !== 'anthropic' ||
        target.protocol !== CLAUDE_MESSAGES_PROTOCOL ||
        target.adapter_version !== CLAUDE_MESSAGES_ADAPTER_VERSION ||
        body.model !== target.model ||
        !Array.isArray(body.messages) ||
        Object.keys(body).some((key) => !GENERATION_FIELDS.has(key))
    )
        throw new TypeError('Anthropic indexed count has an unsupported target/body');
    if (
        body.tools !== undefined &&
        (!Array.isArray(body.tools) ||
            body.tools.some(
                (tool) =>
                    typeof tool !== 'object' ||
                    tool === null ||
                    Array.isArray(tool) ||
                    typeof tool.name !== 'string' ||
                    typeof tool.input_schema !== 'object' ||
                    (tool.type !== undefined && tool.type !== 'custom'),
            ))
    )
        throw new TypeError('Anthropic indexed count supports application tool definitions only');
    const countBody = Object.fromEntries(
        COUNT_FIELDS.flatMap((key) => (body[key] === undefined ? [] : [[key, body[key]]])),
    );
    // The registered compiler builds SDK-native fields. The real endpoint validates them before any coverage is published.
    return countBody as unknown as MessageCountTokensParams;
}

/** Provider preflight count, never billed usage or an eventual context-fit guarantee. */
export async function countAnthropicIndexedRequest(
    client: Pick<Anthropic, 'messages'>,
    nativeRequest: unknown,
    target: ModelTarget,
    signal?: AbortSignal,
) {
    if (!preflightJsonInput(nativeRequest, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new RangeError('Anthropic indexed count body exceeds its bounded JSON profile');
    const ownedRequest = JsonObjectSchema.parse(structuredClone(nativeRequest));
    const body = anthropicIndexedCountBody(ownedRequest, target);
    const requestFingerprint = await fingerprintJson(ownedRequest);
    const counted = await boundedCanonicalProjectionOperation(
        (ownedSignal) => client.messages.countTokens(body, { signal: ownedSignal, timeout: 30000 }),
        signal,
        30000,
    );
    if (!Number.isSafeInteger(counted.input_tokens) || counted.input_tokens < 0)
        throw new Error('Anthropic count endpoint returned an invalid input count');
    return {
        input_tokens: counted.input_tokens,
        profile: ANTHROPIC_INDEXED_COUNT_PROFILE as typeof ANTHROPIC_INDEXED_COUNT_PROFILE,
        tokenizer_version: ANTHROPIC_INDEXED_COUNT_VERSION,
        request_fingerprint: requestFingerprint,
    };
}
