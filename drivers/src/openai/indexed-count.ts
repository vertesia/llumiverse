import { fingerprintJson, type ModelTarget, preflightJsonInput } from '@llumiverse/conversation';
import { JsonObjectSchema, ModelTargetSchema, NonnegativeSafeIntegerSchema } from '@llumiverse/conversation/schemas';
import type OpenAI from 'openai';
import type { InputTokenCountParams } from 'openai/resources/responses/input-tokens.js';
import { boundedCanonicalProjectionOperation } from './canonical-request-measurement.js';
import {
    OPENAI_RESPONSES_ADAPTER_VERSION,
    OPENAI_RESPONSES_PROTOCOL,
} from './openai-responses-conversation-adapter.js';

export const OPENAI_INDEXED_COUNT_PROFILE = 'openai.responses.input_tokens:v1';
// Identifies the count-body projection, not an undisclosed provider tokenizer algorithm.
export const OPENAI_INDEXED_COUNT_VERSION = 'openai.responses.input_tokens:projection-2026-10-07.v1';
const COUNT_FIELDS = [
    'model',
    'input',
    'instructions',
    'tools',
    'tool_choice',
    'parallel_tool_calls',
    'reasoning',
    'text',
    'personality',
    'truncation',
] as const;
// These controls affect generation, delivery or caching, not the input sequence.
// Output allowance remains independently reserved by the host from the unchanged native body.
const GENERATION_FIELDS = new Set([
    ...COUNT_FIELDS,
    'max_output_tokens',
    'stream',
    'temperature',
    'top_p',
    'metadata',
    'service_tier',
    'store',
    'include',
    'prompt_cache_key',
    'prompt_cache_retention',
    'background',
    'safety_identifier',
    'user',
]);
function parseCountResponse(value: unknown) {
    const response = JsonObjectSchema.parse(value);
    if (
        response.object !== 'response.input_tokens' ||
        Object.keys(response).length !== 2 ||
        !Object.hasOwn(response, 'object') ||
        !Object.hasOwn(response, 'input_tokens')
    )
        throw new TypeError('OpenAI count endpoint returned an invalid exact response');
    return { input_tokens: NonnegativeSafeIntegerSchema.parse(response.input_tokens) };
}

/** The exact registered generation-to-count mapping. Hidden server-side history is unsupported. */
export function openAIIndexedCountBody(nativeInput: unknown, targetInput: ModelTarget): InputTokenCountParams {
    if (!preflightJsonInput(nativeInput, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new RangeError('OpenAI indexed count body exceeds its bounded JSON profile');
    const body = JsonObjectSchema.parse(structuredClone(nativeInput));
    const target = ModelTargetSchema.parse(structuredClone(targetInput));
    if (
        target.provider !== 'openai' ||
        target.protocol !== OPENAI_RESPONSES_PROTOCOL ||
        target.adapter_version !== OPENAI_RESPONSES_ADAPTER_VERSION ||
        body.model !== target.model ||
        !Array.isArray(body.input) ||
        Object.keys(body).some((key) => !GENERATION_FIELDS.has(key))
    )
        throw new TypeError('OpenAI indexed count has an unsupported target/body');
    if (
        body.max_output_tokens !== undefined &&
        (typeof body.max_output_tokens !== 'number' ||
            !Number.isSafeInteger(body.max_output_tokens) ||
            body.max_output_tokens < 0)
    )
        throw new TypeError('OpenAI indexed count has an invalid output allowance');
    if (body.truncation !== undefined && body.truncation !== 'disabled')
        throw new TypeError('OpenAI indexed count cannot truncate its selected input');
    if (
        body.tools !== undefined &&
        (!Array.isArray(body.tools) ||
            body.tools.some(
                (tool) =>
                    typeof tool !== 'object' ||
                    tool === null ||
                    Array.isArray(tool) ||
                    tool.type !== 'function' ||
                    typeof tool.name !== 'string' ||
                    typeof tool.parameters !== 'object' ||
                    tool.parameters === null,
            ))
    )
        throw new TypeError('OpenAI indexed count supports application tool definitions only');
    const countBody = Object.fromEntries(
        COUNT_FIELDS.flatMap((key) => (body[key] === undefined ? [] : [[key, body[key]]])),
    );
    // The owned registered compiler produces SDK-native input items; the real count endpoint
    // validates those fields before the host can publish coverage or prepared admission.
    return countBody as unknown as InputTokenCountParams;
}

/** Read the successful SDK response before JSON parsing; never allocate an unbounded count payload. */
async function readCountResponse(response: Response, signal: AbortSignal) {
    if (!response.body) throw new Error('OpenAI count endpoint returned no body');
    const reader = response.body.getReader();
    const bytes = new Uint8Array(4096);
    let size = 0;
    let chunks = 0;
    let completed = false;
    const abort = () => {
        void reader.cancel(signal.reason).catch(() => undefined);
    };
    signal.addEventListener('abort', abort, { once: true });
    try {
        while (true) {
            signal.throwIfAborted();
            const next = await reader.read();
            signal.throwIfAborted();
            if (next.done) {
                completed = true;
                break;
            }
            if (++chunks > 128 || size + next.value.byteLength > bytes.byteLength)
                throw new RangeError('OpenAI count response exceeds its bounded JSON profile');
            bytes.set(next.value, size);
            size += next.value.byteLength;
        }
        const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, size)));
        return parseCountResponse(value);
    } finally {
        signal.removeEventListener('abort', abort);
        if (!completed) await reader.cancel().catch(() => undefined);
        reader.releaseLock();
    }
}

/** Provider preflight count, never billed usage or a guarantee about eventual provider dispatch. */
export async function countOpenAIIndexedRequest(
    client: Pick<OpenAI, 'responses'>,
    nativeRequest: unknown,
    target: ModelTarget,
    signal?: AbortSignal,
) {
    if (!preflightJsonInput(nativeRequest, { max_bytes: 32 * 1024 * 1024 }).success)
        throw new RangeError('OpenAI indexed count body exceeds its bounded JSON profile');
    const ownedRequest = JsonObjectSchema.parse(structuredClone(nativeRequest));
    const body = openAIIndexedCountBody(ownedRequest, target);
    const requestFingerprint = await fingerprintJson(ownedRequest);
    const counted = await boundedCanonicalProjectionOperation(
        async (ownedSignal) =>
            readCountResponse(
                await client.responses.inputTokens.count(body, { signal: ownedSignal, timeout: 30000 }).asResponse(),
                ownedSignal,
            ),
        signal,
        30000,
    );
    const accepted = counted;
    return {
        input_tokens: accepted.input_tokens,
        profile: OPENAI_INDEXED_COUNT_PROFILE,
        tokenizer_version: OPENAI_INDEXED_COUNT_VERSION,
        request_fingerprint: requestFingerprint,
    };
}
