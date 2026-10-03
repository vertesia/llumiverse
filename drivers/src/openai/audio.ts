import { boundedAudioStream, canonicalAudioAssetStorage, storeAudioResult } from '../shared/audio.js';
import type { ChatCompletionsUsage } from './usage.js';

export { boundedAudioStream } from '../shared/audio.js';

import {
    OpenAiAudioOptionsSchema,
    OpenAiSpeechOptionsSchema,
    OpenAiTranscriptionOptionsSchema,
} from '@llumiverse/common/schemas';
import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponseWithProcessing,
    type ConversationDocument,
    deriveConversationId,
    fingerprintJson,
    type GenerationUsage,
    hashContentBytes,
    inlineAssetContentIntegrity,
    isConversationDocumentFormat,
    type JsonValue,
    type NativeItemMapping,
    parseConversationDocument,
    type RequestReceipt,
    type ResolvedConversationRuntimeContext,
    type UserContentBlock,
} from '@llumiverse/conversation';
import {
    type AudioResult,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    type Completion,
    type CompletionResult,
    createCanonicalExecutionResponse,
    type DataSource,
    type ExecutionOptions,
    type ExecutionResponse,
    type ExecutionTokenUsage,
    FallbackCanonicalExecutionEventStream,
    type PromptSegment,
    Providers,
    readStreamAsBase64,
    resolveModelProfile,
    stripAudioFromCompletion,
} from '@llumiverse/core';
import type { AbstractDriver } from '@llumiverse/core/driver';
import type OpenAI from 'openai';
import { toStreamingFile } from 'openai';
import {
    acceptedCanonicalResponse,
    appendCanonicalPrompt,
    assertAcceptedCanonicalRequest,
    assertCanonicalContextProjection,
    canonicalResponseIdentities,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    prepareCanonicalContext,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
    resolveConversationRuntime,
    selectedCanonicalTurns,
} from '../conversation/canonical-runtime.js';

const OPENAI_AUDIO_ADAPTER_VERSION = '2026-09-30.canonical.1';

interface OpenAIAudioNativeExecution {
    blocks: AgentContentBlock[];
    assets: Asset[];
    completed_at: string;
    finish_reason?: string | null;
    usage?: GenerationUsage;
    original_response?: unknown;
    response_fingerprint_payload: JsonValue;
}

interface OpenAIAudioOutputIdentity {
    response_operation_id: string;
    generation_id: string;
    response_turn_id: string;
    completed_at?: string;
    asset_storage: (value: string) => Asset['storage'];
}

interface OpenAIAudioInputFile {
    name: string;
    mime_type: string;
    getStream(): Promise<ReadableStream<Uint8Array | string>>;
}

type OpenAIAudioExecutionOptions = ExecutionOptions | CanonicalExecutionContextOptions;

type DecodedOpenAIAudioItem =
    | { type: 'text'; text: string }
    | { type: 'json'; value: JsonValue }
    | {
          type: 'audio';
          audio: AudioResult;
          byte_length: number;
          content_hash: string;
      };

const OPENAI_AUDIO_CHAT_PROTOCOL = 'openai.chat.completions.audio';
const OPENAI_AUDIO_TOKEN_BASIS = 'openai_audio_tokens';
const OPENAI_TRANSCRIPTION_TOKEN_BASIS = 'openai_transcription_tokens';
const OPENAI_TRANSCRIPTION_DURATION_BASIS = 'openai_transcription_duration';

function decimalAmount(value: number): string {
    if (!Number.isFinite(value) || value < 0) throw new Error('OpenAI audio response contains invalid usage cost');
    const source = String(value);
    if (!/[eE]/.test(source)) return source;
    const [coefficient, exponentText] = source.toLowerCase().split('e');
    const exponent = Number(exponentText);
    const negative = coefficient.startsWith('-');
    const unsigned = negative ? coefficient.slice(1) : coefficient;
    const [integer, fraction = ''] = unsigned.split('.');
    const digits = `${integer}${fraction}`;
    const decimalIndex = integer.length + exponent;
    const expanded =
        decimalIndex <= 0
            ? `0.${'0'.repeat(-decimalIndex)}${digits}`
            : decimalIndex >= digits.length
              ? `${digits}${'0'.repeat(decimalIndex - digits.length)}`
              : `${digits.slice(0, decimalIndex)}.${digits.slice(decimalIndex)}`;
    return negative ? `-${expanded}` : expanded;
}

function openAIChatAudioUsage(usage: ChatCompletionsUsage | null | undefined): GenerationUsage | undefined {
    if (usage === undefined || usage === null) return undefined;
    const reportedCacheReadTokens = usage.prompt_tokens_details?.cached_tokens;
    const reportedCacheWriteTokens = usage.prompt_tokens_details?.cache_write_tokens;
    const cacheReadTokens = reportedCacheReadTokens ?? 0;
    const cacheWriteTokens = reportedCacheWriteTokens ?? 0;
    const inputNewTokens = Math.max(0, usage.prompt_tokens - cacheReadTokens - cacheWriteTokens);
    const providerCost = usage.is_byok !== true && typeof usage.cost === 'number' ? usage.cost : undefined;
    return {
        input_tokens: usage.prompt_tokens,
        input_new_tokens: inputNewTokens,
        cache_read_tokens: cacheReadTokens,
        cache_write_tokens: cacheWriteTokens,
        output_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        accounting_provenance: {
            input_tokens: { method: 'reported', accounting_basis: OPENAI_AUDIO_TOKEN_BASIS },
            input_new_tokens: { method: 'derived', accounting_basis: OPENAI_AUDIO_TOKEN_BASIS },
            cache_read_tokens: {
                method: reportedCacheReadTokens == null ? 'derived' : 'reported',
                accounting_basis: OPENAI_AUDIO_TOKEN_BASIS,
            },
            cache_write_tokens: {
                method: reportedCacheWriteTokens == null ? 'derived' : 'reported',
                accounting_basis: OPENAI_AUDIO_TOKEN_BASIS,
            },
            output_tokens: { method: 'reported', accounting_basis: OPENAI_AUDIO_TOKEN_BASIS },
            total_tokens: { method: 'reported', accounting_basis: OPENAI_AUDIO_TOKEN_BASIS },
        },
        input_partition: { type: 'complete_disjoint', cache_write_bucket: 'included' },
        reported_usage: [
            {
                source: 'provider',
                protocol: OPENAI_AUDIO_CHAT_PROTOCOL,
                accounting_basis: OPENAI_AUDIO_TOKEN_BASIS,
                payload: providerJsonValue(usage),
            },
        ],
        ...(providerCost === undefined
            ? {}
            : { cost: { amount: decimalAmount(providerCost), currency: 'USD', provenance: 'reported' as const } }),
    };
}

function openAITranscriptionUsage(usage: OpenAI.Audio.Transcription['usage']): GenerationUsage | undefined {
    if (usage === undefined) return undefined;
    const accountingBasis =
        usage.type === 'tokens' ? OPENAI_TRANSCRIPTION_TOKEN_BASIS : OPENAI_TRANSCRIPTION_DURATION_BASIS;
    const reported_usage: NonNullable<GenerationUsage['reported_usage']> = [
        {
            source: 'provider',
            protocol: 'openai.audio.transcription',
            accounting_basis: accountingBasis,
            payload: providerJsonValue(usage),
        },
    ];
    if (usage.type !== 'tokens') return { reported_usage };
    return {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        accounting_provenance: {
            input_tokens: { method: 'reported', accounting_basis: accountingBasis },
            output_tokens: { method: 'reported', accounting_basis: accountingBasis },
            total_tokens: { method: 'reported', accounting_basis: accountingBasis },
        },
        reported_usage,
    };
}

async function canonicalOpenAIAudioOutput(
    items: DecodedOpenAIAudioItem[],
    identity: OpenAIAudioOutputIdentity,
): Promise<Pick<OpenAIAudioNativeExecution, 'blocks' | 'assets' | 'completed_at'>> {
    const completedAt = identity.completed_at ?? new Date().toISOString();
    const blocks: AgentContentBlock[] = [];
    const assets: Asset[] = [];
    for (let index = 0; index < items.length; index += 1) {
        const item = items[index];
        const blockId = await deriveConversationId('block', identity.response_operation_id, String(index));
        if (item.type === 'text') {
            blocks.push({ id: blockId, type: 'text', text: item.text, format: 'plain' });
            continue;
        }
        if (item.type === 'json') {
            blocks.push({ id: blockId, type: 'json', value: item.value });
            continue;
        }
        const assetId = await deriveConversationId('asset', identity.response_operation_id, String(index));
        assets.push({
            id: assetId,
            kind: 'audio',
            mime_type: item.audio.mime_type,
            storage: identity.asset_storage(item.audio.value),
            provenance: {
                type: 'generated',
                generation_id: identity.generation_id,
                source_turn_id: identity.response_turn_id,
            },
            created_at: completedAt,
            byte_length: item.byte_length,
            content_hash: item.content_hash,
            media: {
                ...(item.audio.container === undefined ? {} : { container: item.audio.container }),
                ...(item.audio.codec === undefined ? {} : { codec: item.audio.codec }),
                ...(item.audio.sample_rate === undefined ? {} : { sample_rate: item.audio.sample_rate }),
                ...(item.audio.channels === undefined ? {} : { channels: item.audio.channels }),
                ...(item.audio.sample_encoding === undefined ? {} : { sample_encoding: item.audio.sample_encoding }),
                ...(item.audio.byte_order === undefined ? {} : { byte_order: item.audio.byte_order }),
            },
        });
        blocks.push({ id: blockId, type: 'audio', asset_id: assetId });
    }
    return { blocks, assets, completed_at: completedAt };
}

export async function openAIInputAudioPart(
    file: Pick<DataSource, 'mime_type' | 'getStream'>,
    signal?: AbortSignal,
    unsupportedFormatMessage = 'Chat audio input requires MP3 or WAV',
): Promise<OpenAI.Chat.ChatCompletionContentPartInputAudio> {
    const format =
        file.mime_type === 'audio/wav' || file.mime_type === 'audio/x-wav'
            ? 'wav'
            : file.mime_type === 'audio/mpeg' || file.mime_type === 'audio/mp3'
              ? 'mp3'
              : undefined;
    if (!format) throw new Error(unsupportedFormatMessage);
    const data = await readStreamAsBase64(boundedAudioStream(await file.getStream(), 25_000_000, signal));
    return { type: 'input_audio', input_audio: { data, format } };
}

function openAIAudioMetadata(
    responseFormat: NonNullable<OpenAI.Audio.SpeechCreateParams['response_format']> | 'pcm16',
): Omit<AudioResult, 'type' | 'value'> {
    const format = responseFormat === 'pcm16' ? 'pcm' : responseFormat;
    return {
        mime_type: {
            mp3: 'audio/mpeg',
            wav: 'audio/wav',
            opus: 'audio/ogg',
            aac: 'audio/aac',
            flac: 'audio/flac',
            pcm: 'audio/pcm',
        }[format],
        container: format === 'opus' ? 'ogg' : format === 'pcm' ? 'raw' : format,
        ...(format === 'mp3' ? { codec: 'mp3' } : {}),
        ...(format === 'pcm'
            ? { codec: 'pcm', sample_rate: 24000, channels: 1, sample_encoding: 'int16', byte_order: 'little' }
            : {}),
    };
}

export function openAIAudioTask(model: string): 'transcription' | 'speech' | 'understanding' | undefined {
    const { family } = resolveModelProfile(model.split('::').pop() ?? model, Providers.openai);
    if (/(?:realtime|live)/i.test(model)) return undefined;
    return family === 'audio'
        ? 'understanding'
        : family === 'transcription' || family === 'speech'
          ? family
          : undefined;
}

const OPENAI_TRANSCRIPTION_MIME_TYPES = new Set([
    'audio/flac',
    'audio/m4a',
    'audio/mp3',
    'audio/mp4',
    'audio/mpeg',
    'audio/mpga',
    'audio/ogg',
    'audio/wav',
    'audio/webm',
    'audio/x-wav',
    'video/mp4',
    'video/webm',
]);

interface ValidatedOpenAIAudioInput {
    task: NonNullable<ReturnType<typeof openAIAudioTask>>;
    files: OpenAIAudioInputFile[];
    text: string;
}

function validateOpenAIAudioValues(input: ValidatedOpenAIAudioInput): ValidatedOpenAIAudioInput {
    const { task, files, text } = input;
    if (task === 'understanding') {
        if (files.length > 1) throw new Error('Audio understanding accepts at most one audio file');
        if (!files.length && !text) throw new Error('Audio understanding requires text or an audio file');
        if (
            files[0] !== undefined &&
            !['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3'].includes(files[0].mime_type)
        ) {
            throw new Error('OpenAI audio chat requires MP3 or WAV input');
        }
    } else if (task === 'transcription') {
        if (files.length !== 1) throw new Error('Transcription requires exactly one audio file');
        if (!OPENAI_TRANSCRIPTION_MIME_TYPES.has(files[0].mime_type)) {
            throw new Error(`Transcription does not support ${files[0].mime_type} input`);
        }
    } else {
        if (files.length) throw new Error('Speech synthesis accepts text only');
        if (!text || text.length > 4096) throw new Error('Speech synthesis requires 1–4096 characters');
    }
    return input;
}

function validateOpenAIAudioInput(segments: PromptSegment[], model: string): ValidatedOpenAIAudioInput {
    const task = openAIAudioTask(model);
    if (task === undefined) throw new Error(`Model ${model} is not an OpenAI file audio model`);
    if (segments.some((segment) => segment.role === 'tool' || segment.role === 'assistant')) {
        throw new Error('File audio operations accept only user and system input');
    }
    const files = segments.flatMap((segment) => segment.files ?? []);
    const text = segments
        .map((segment) => segment.content ?? '')
        .join('\n')
        .trim();
    return validateOpenAIAudioValues({ task, files, text });
}

function validateOpenAIAudioCanonicalInput(segments: PromptSegment[], options: ExecutionOptions) {
    const runtime = resolveConversationRuntime(options);
    const retainedDocument = isConversationDocumentFormat(options.conversation)
        ? parseConversationDocument(options.conversation)
        : undefined;
    if (options.conversation !== undefined && retainedDocument === undefined) {
        throw new Error('File audio operations do not accept legacy conversation input');
    }
    if (
        retainedDocument !== undefined &&
        options.conversation_runtime?.conversation_id !== undefined &&
        retainedDocument.id !== runtime.conversation_id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    if (options.tools?.length || options.result_schema || options.format) {
        throw new Error('File audio operations do not accept tools, result schemas, or custom formatting');
    }
    const validated = validateOpenAIAudioInput(segments, options.model);
    return { runtime, retainedDocument, validated };
}

/** Finite typed-event projection for canonical OpenAI file-audio execution. */
export async function streamOpenAIAudioCanonicalEvents(input: {
    segments: PromptSegment[];
    options: ExecutionOptions;
    signal?: AbortSignal;
    open: CanonicalStreamOpenOptions;
    execute: (signal: AbortSignal) => Promise<CanonicalExecutionResponse>;
}): Promise<CanonicalExecutionEventStream> {
    if (input.options.conversation_runtime === undefined) {
        throw new Error('Canonical typed streaming requires conversation_runtime');
    }
    if (input.options.conversation_runtime.materialized_input !== undefined && input.segments.length > 0) {
        throw new Error('A materialized canonical input requires an empty new prompt');
    }
    input.signal?.throwIfAborted();
    const { runtime, retainedDocument } = validateOpenAIAudioCanonicalInput(input.segments, input.options);
    const accepted =
        retainedDocument === undefined
            ? undefined
            : acceptedCanonicalResponse(retainedDocument, runtime.response_operation_id);
    const identities = await canonicalResponseIdentities(runtime);
    const identity = {
        request_id: accepted?.generation.request_id ?? runtime.request_id,
        attempt_id: accepted?.generation.attempt_id ?? runtime.attempt_id,
        response_operation_id: runtime.response_operation_id,
        generation_id: accepted?.generation.id ?? identities.generation_id,
        draft_turn_id: accepted?.turn.id ?? identities.response_turn_id,
    };
    return new FallbackCanonicalExecutionEventStream(
        identity,
        (fallbackSignal) =>
            input.execute(input.signal ? AbortSignal.any([input.signal, fallbackSignal]) : fallbackSignal),
        { ...input.open, ...(accepted === undefined ? {} : { origin: 'accepted_recovery' as const }) },
    );
}

export async function executeOpenAIAudioNative(
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    requestOptions: { signal?: AbortSignal; timeout?: number } | undefined,
    outputIdentity: OpenAIAudioOutputIdentity,
    requestModel = options.model,
): Promise<OpenAIAudioNativeExecution> {
    const signal = requestOptions?.signal;
    signal?.throwIfAborted();
    if (options.conversation || options.tools?.length || options.result_schema || options.format) {
        throw new Error(
            'File audio operations do not accept conversation, tools, result schemas, or custom formatting',
        );
    }
    return executeValidatedOpenAIAudioNative(
        service,
        validateOpenAIAudioInput(segments, options.model),
        options,
        requestOptions,
        outputIdentity,
        requestModel,
    );
}

async function executeValidatedOpenAIAudioNative(
    service: OpenAI,
    input: ValidatedOpenAIAudioInput,
    options: OpenAIAudioExecutionOptions,
    requestOptions: { signal?: AbortSignal; timeout?: number } | undefined,
    outputIdentity: OpenAIAudioOutputIdentity,
    requestModel = options.model,
): Promise<OpenAIAudioNativeExecution> {
    const signal = requestOptions?.signal;
    signal?.throwIfAborted();
    const { task, files, text } = input;
    if (task === 'understanding') {
        const content: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [{ type: 'text', text }];
        if (files[0]) {
            content.push(await openAIInputAudioPart(files[0], signal, 'OpenAI audio chat requires MP3 or WAV input'));
        }
        const params = OpenAiAudioOptionsSchema.parse(options.model_options ?? { _option_id: 'openai-audio' });
        if (!options.store_audio) throw new Error('Audio generation requires a durable audio storage sink');
        const format = params.response_format ?? 'wav';
        const result = await service.chat.completions.create(
            {
                model: requestModel,
                messages: [{ role: 'user', content }],
                modalities: ['text', 'audio'],
                audio: { voice: params.voice ?? 'alloy', format },
            },
            requestOptions,
        );
        signal?.throwIfAborted();
        const message = result.choices[0]?.message;
        if (!message?.audio?.data) throw new Error('OpenAI audio chat returned no audio data');
        const audioBytes = Buffer.from(message.audio.data, 'base64');
        const audio = await storeAudioResult(
            new Blob([audioBytes]).stream(),
            openAIAudioMetadata(format),
            options,
            signal,
        );
        const output = await canonicalOpenAIAudioOutput(
            [
                ...(message.content || message.audio.transcript
                    ? [{ type: 'text' as const, text: message.content ?? message.audio.transcript }]
                    : []),
                {
                    type: 'audio' as const,
                    audio,
                    byte_length: audioBytes.byteLength,
                    content_hash: (await hashContentBytes(audioBytes)).content_hash,
                },
            ],
            outputIdentity,
        );
        return {
            ...output,
            finish_reason: result.choices[0]?.finish_reason,
            usage: openAIChatAudioUsage(result.usage),
            original_response: options.include_original_response ? result : undefined,
            response_fingerprint_payload: providerJsonValue(result),
        };
    }
    if (task === 'transcription') {
        const params = OpenAiTranscriptionOptionsSchema.parse(
            options.model_options ?? {
                _option_id: 'openai-transcription',
            },
        );
        // The endpoint owns codec validation; MP4 and WebM recordings may carry a video MIME type.
        const file = files[0];
        const stream = boundedAudioStream(await file.getStream(), 25_000_000, signal);
        try {
            if (options.model.includes('diarize')) {
                if (text) throw new Error('Diarized transcription does not accept a prompt');
                const result = (await service.audio.transcriptions.create(
                    {
                        model: requestModel,
                        file: toStreamingFile(stream, file.name, { type: file.mime_type }),
                        response_format: 'diarized_json',
                        chunking_strategy: 'auto',
                        language: params.language,
                    },
                    requestOptions,
                )) as OpenAI.Audio.TranscriptionDiarized; // SDK overload omits its exported diarized response type.
                signal?.throwIfAborted();
                const output = await canonicalOpenAIAudioOutput(
                    [
                        { type: 'text', text: result.text },
                        {
                            type: 'json',
                            value: {
                                segments: result.segments.map((segment) => ({
                                    id: segment.id,
                                    speaker: segment.speaker,
                                    start: segment.start,
                                    end: segment.end,
                                    text: segment.text,
                                })),
                            },
                        },
                    ],
                    outputIdentity,
                );
                return {
                    ...output,
                    finish_reason: 'stop',
                    usage: openAITranscriptionUsage(result.usage),
                    original_response: options.include_original_response ? result : undefined,
                    response_fingerprint_payload: providerJsonValue(result),
                };
            }
            const result = await service.audio.transcriptions.create(
                {
                    model: requestModel,
                    file: toStreamingFile(stream, file.name, { type: file.mime_type }),
                    response_format: 'json',
                    language: params.language,
                    ...(text && { prompt: text }),
                },
                requestOptions,
            );
            signal?.throwIfAborted();
            const output = await canonicalOpenAIAudioOutput([{ type: 'text', text: result.text }], outputIdentity);
            return {
                ...output,
                finish_reason: 'stop',
                usage: openAITranscriptionUsage(result.usage),
                original_response: options.include_original_response ? result : undefined,
                response_fingerprint_payload: providerJsonValue(result),
            };
        } finally {
            if (!stream.locked) await stream.cancel().catch(() => undefined);
        }
    }
    const params = OpenAiSpeechOptionsSchema.parse(options.model_options ?? { _option_id: 'openai-speech' });
    if (!options.store_audio) throw new Error('Speech synthesis requires a durable audio storage sink');
    if (params.instructions && /^tts-1(?:-|$)/.test(options.model)) {
        throw new Error('tts-1 models do not support speech instructions');
    }
    const format = params.response_format ?? 'mp3';
    const response = await service.audio.speech.create(
        {
            model: requestModel,
            input: text,
            voice: params.voice ?? 'alloy',
            response_format: format,
            speed: params.speed,
            instructions: params.instructions,
        } satisfies OpenAI.Audio.SpeechCreateParams,
        requestOptions,
    );
    if (!response.body) throw new Error('Speech endpoint returned no audio body');
    const audioBytes = new Uint8Array(
        await new Response(boundedAudioStream(response.body, 50_000_000, signal)).arrayBuffer(),
    );
    const audio = await storeAudioResult(new Blob([audioBytes]).stream(), openAIAudioMetadata(format), options, signal);
    const output = await canonicalOpenAIAudioOutput(
        [
            {
                type: 'audio',
                audio,
                byte_length: audioBytes.byteLength,
                content_hash: (await hashContentBytes(audioBytes)).content_hash,
            },
        ],
        outputIdentity,
    );
    return {
        ...output,
        finish_reason: 'stop',
        original_response: options.include_original_response ? response : undefined,
        response_fingerprint_payload: {
            audio: {
                mime_type: audio.mime_type,
                data: Buffer.from(audioBytes).toString('base64'),
            },
        },
    };
}

export async function executeOpenAIAudio(
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    requestOptions: { signal?: AbortSignal; timeout?: number } | undefined,
    requestModel = options.model,
): Promise<Completion> {
    const native = await executeOpenAIAudioNative(
        service,
        segments,
        options,
        requestOptions,
        {
            response_operation_id: 'legacy-openai-audio-response',
            generation_id: 'legacy-openai-audio-generation',
            response_turn_id: 'legacy-openai-audio-response-turn',
            asset_storage: (value) => ({
                type: 'external',
                resolver: 'legacy_audio_result',
                locator: { uri: value },
            }),
        },
        requestModel,
    );
    const tokenUsage = legacyOpenAIAudioUsage(native.usage);
    return {
        result: legacyOpenAIAudioResults(native),
        ...(tokenUsage === undefined ? {} : { token_usage: tokenUsage }),
        ...(native.finish_reason == null ? { finish_reason: undefined } : { finish_reason: native.finish_reason }),
        ...(native.original_response === undefined ? {} : { original_response: native.original_response }),
    };
}

function legacyOpenAIAudioUsage(usage: GenerationUsage | undefined): ExecutionTokenUsage | undefined {
    if (usage === undefined) return undefined;
    const providerCostUsd = usage.cost?.currency === 'USD' ? Number(usage.cost.amount) : undefined;
    const hasScalarUsage =
        usage.input_tokens !== undefined ||
        usage.input_new_tokens !== undefined ||
        usage.cache_read_tokens !== undefined ||
        usage.cache_write_tokens !== undefined ||
        usage.output_tokens !== undefined ||
        usage.total_tokens !== undefined;
    if (!hasScalarUsage && (providerCostUsd === undefined || !Number.isFinite(providerCostUsd))) return undefined;
    const isChatAudio = usage.reported_usage?.some((reported) => reported.protocol === OPENAI_AUDIO_CHAT_PROTOCOL);
    return {
        prompt: usage.input_tokens,
        ...(isChatAudio
            ? {
                  prompt_new: usage.input_new_tokens,
                  prompt_cached: usage.cache_read_tokens || undefined,
                  prompt_cache_write: usage.cache_write_tokens || undefined,
              }
            : {}),
        result: usage.output_tokens,
        total: usage.total_tokens,
        ...(providerCostUsd === undefined || !Number.isFinite(providerCostUsd)
            ? {}
            : { provider_cost_usd: providerCostUsd }),
    };
}

function legacyOpenAIAudioAssetValue(asset: Asset): string {
    if (asset.storage.type !== 'external') throw new Error(`Canonical audio asset ${asset.id} is not external`);
    const uri = asset.storage.locator.uri;
    if (typeof uri === 'string' && uri.length > 0) return uri;
    const url = asset.storage.locator.url;
    if (typeof url === 'string' && url.length > 0) return url;
    throw new Error(`Canonical audio asset ${asset.id} has no legacy value`);
}

/** Explicit compatibility projection used only by legacy file-audio execute and stream APIs. */
function legacyOpenAIAudioResults(native: OpenAIAudioNativeExecution): CompletionResult[] {
    return native.blocks.map((block): CompletionResult => {
        if (block.type === 'text') return { type: 'text', value: block.text };
        if (block.type === 'json') return { type: 'json', value: block.value };
        if (block.type === 'audio') {
            const asset = native.assets.find((candidate) => candidate.id === block.asset_id);
            if (asset === undefined) throw new Error(`Canonical audio output is missing asset ${block.asset_id}`);
            return {
                type: 'audio',
                value: legacyOpenAIAudioAssetValue(asset),
                mime_type: asset.mime_type,
                ...(asset.media?.container === undefined ? {} : { container: asset.media.container }),
                ...(asset.media?.codec === undefined ? {} : { codec: asset.media.codec }),
                ...(asset.media?.sample_rate === undefined ? {} : { sample_rate: asset.media.sample_rate }),
                ...(asset.media?.channels === undefined ? {} : { channels: asset.media.channels }),
                ...(asset.media?.sample_encoding === undefined ? {} : { sample_encoding: asset.media.sample_encoding }),
                ...(asset.media?.byte_order === undefined ? {} : { byte_order: asset.media.byte_order }),
            };
        }
        throw new Error(`OpenAI audio produced unsupported canonical ${block.type} output`);
    });
}

async function readAudioSource(file: DataSource, signal?: AbortSignal): Promise<{ bytes: Uint8Array; data: string }> {
    const stream = boundedAudioStream(await file.getStream(), 25_000_000, signal);
    try {
        const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
        return { bytes, data: Buffer.from(bytes).toString('base64') };
    } finally {
        if (!stream.locked) await stream.cancel().catch(() => undefined);
    }
}

interface CanonicalOpenAIAudioInput {
    validated: ValidatedOpenAIAudioInput;
    item_mappings: NativeItemMapping[];
    segments: Array<{
        role: 'user';
        content: string;
        files: Array<{ asset_id: string; name: string; mime_type: string; content_hash: string }>;
    }>;
}

async function canonicalOpenAIAudioInput(
    document: ConversationDocument,
    model: string,
): Promise<CanonicalOpenAIAudioInput> {
    const task = openAIAudioTask(model);
    if (task === undefined) throw new Error(`Model ${model} is not an OpenAI file audio model`);
    const files: OpenAIAudioInputFile[] = [];
    const itemMappings: NativeItemMapping[] = [];
    const segments: CanonicalOpenAIAudioInput['segments'] = [];
    const selected = selectedCanonicalTurns(document);
    assertCanonicalContextProjection(document, selected, {
        label: 'OpenAI file audio',
        program_authorities: ['ordinary'],
    });
    for (let turnIndex = 0; turnIndex < selected.length; turnIndex += 1) {
        const turn = selected[turnIndex];
        if (turn.kind !== 'program' && turn.kind !== 'user') {
            throw new Error('File audio canonical context accepts only program and user turns');
        }
        const text: string[] = [];
        const segmentFiles: CanonicalOpenAIAudioInput['segments'][number]['files'] = [];
        itemMappings.push({ canonical_id: turn.id, native_id: `segments/${turnIndex}`, kind: 'turn' });
        for (let blockIndex = 0; blockIndex < turn.blocks.length; blockIndex += 1) {
            const block = turn.blocks[blockIndex];
            const nativeId = `segments/${turnIndex}/blocks/${blockIndex}`;
            itemMappings.push({ canonical_id: block.id, native_id: nativeId, kind: 'block' });
            if (block.type === 'text') {
                text.push(block.text);
                continue;
            }
            if (block.type !== 'audio' && block.type !== 'video') {
                throw new Error(`File audio canonical context does not support ${block.type} input`);
            }
            const asset = Object.hasOwn(document.assets, block.asset_id) ? document.assets[block.asset_id] : undefined;
            if (asset === undefined) throw new Error(`File audio canonical context is missing asset ${block.asset_id}`);
            if (asset.kind !== block.type) {
                throw new Error(
                    `File audio canonical context asset ${asset.id} does not match its ${block.type} block`,
                );
            }
            if (asset.storage.type !== 'inline_base64') {
                throw new Error(`File audio canonical context asset ${asset.id} must use inline_base64 storage`);
            }
            const integrity = await inlineAssetContentIntegrity(asset.storage);
            if (integrity === undefined) throw new Error(`File audio canonical context asset ${asset.id} has no bytes`);
            if (asset.content_hash !== undefined && asset.content_hash !== integrity.content_hash) {
                throw new Error(`File audio canonical context asset ${asset.id} content hash does not match its bytes`);
            }
            if (asset.byte_length !== undefined && asset.byte_length !== integrity.byte_length) {
                throw new Error(`File audio canonical context asset ${asset.id} byte length does not match its bytes`);
            }
            const bytes = new Uint8Array(Buffer.from(asset.storage.data, 'base64'));
            const file: OpenAIAudioInputFile = {
                name: asset.id,
                mime_type: asset.mime_type,
                getStream: async () => new Blob([bytes]).stream(),
            };
            files.push(file);
            segmentFiles.push({
                asset_id: asset.id,
                name: file.name,
                mime_type: file.mime_type,
                content_hash: integrity.content_hash,
            });
        }
        segments.push({
            // The native audio endpoint receives one flattened prompt and cannot preserve a system role.
            // Context preflight therefore accepts only ordinary program authority and records the truthful
            // native projection here as user input.
            role: 'user',
            content: text.join(''),
            files: segmentFiles,
        });
    }
    const text = segments
        .map((segment) => segment.content)
        .join('\n')
        .trim();
    return {
        validated: validateOpenAIAudioValues({ task, files, text }),
        item_mappings: itemMappings,
        segments,
    };
}

interface PreparedOpenAIAudioCanonicalResponse {
    document: ConversationDocument;
    runtime: ResolvedConversationRuntimeContext;
    receipt: RequestReceipt;
    generation_id: string;
    response_turn_id: string;
    provider: string;
    protocol: string;
    requested_model: string;
    resolved_model: string;
    payload: JsonValue;
}

async function finalizeOpenAIAudioCanonicalResponse(
    prepared: PreparedOpenAIAudioCanonicalResponse,
    native: OpenAIAudioNativeExecution,
    options: OpenAIAudioExecutionOptions,
): Promise<CanonicalExecutionResponse> {
    const generation = await createExecutedGeneration({
        id: prepared.generation_id,
        runtime: prepared.runtime,
        receipt: prepared.receipt,
        provider: prepared.provider,
        protocol: prepared.protocol,
        adapter_version: OPENAI_AUDIO_ADAPTER_VERSION,
        requested_model: prepared.requested_model,
        resolved_model: prepared.resolved_model,
        finish_reason: native.finish_reason ?? undefined,
        usage: native.usage,
    });
    const responseTurn = {
        id: prepared.response_turn_id,
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks: native.blocks,
        status: 'completed' as const,
        timestamps: { recorded_at: native.completed_at, completed_at: native.completed_at },
        model_visibility: 'include' as const,
        provenance: { type: 'generated' as const },
        generation_id: generation.id,
    };
    const finalDocument = (
        await appendDecodedConversationResponseWithProcessing(
            {
                document: prepared.document,
                generation_id: prepared.generation_id,
                response_turn_id: prepared.response_turn_id,
                receipt: prepared.receipt,
                payload: prepared.payload,
                diagnostics: [],
            },
            {
                turns: [responseTurn],
                assets: native.assets,
                generation,
                diagnostics: [],
                payload_fingerprint: await fingerprintJson(native.response_fingerprint_payload),
            },
            {
                operation_id: prepared.runtime.response_operation_id,
                recorded_at: native.completed_at,
            },
        )
    ).document;
    return createCanonicalExecutionResponse(finalDocument, prepared.runtime.response_operation_id, {
        ...(options.include_original_response && native.original_response !== undefined
            ? { original_response: native.original_response }
            : {}),
    });
}

/** Direct canonical execution for finite OpenAI audio endpoints. */
export async function executeOpenAIAudioCanonical(input: {
    service: OpenAI;
    segments: PromptSegment[];
    options: ExecutionOptions;
    provider: string;
    request_model?: string;
    request_options?: { signal?: AbortSignal; timeout?: number };
}): Promise<CanonicalExecutionResponse> {
    const { runtime, retainedDocument, validated } = validateOpenAIAudioCanonicalInput(input.segments, input.options);
    const { task } = validated;
    const loadedFiles = await Promise.all(
        input.segments
            .flatMap((segment) => segment.files ?? [])
            .map(async (source) => ({
                source,
                ...(await readAudioSource(source, input.request_options?.signal)),
            })),
    );
    let loadedIndex = 0;
    const transportSegments = input.segments.map((segment) => ({
        ...segment,
        ...(segment.files === undefined
            ? {}
            : {
                  files: segment.files.map((file) => {
                      const loaded = loadedFiles[loadedIndex++];
                      if (loaded === undefined) throw new Error('Audio input preload mismatch');
                      return {
                          ...file,
                          getStream: async () => new Blob([Buffer.from(loaded.bytes)]).stream(),
                      } satisfies DataSource;
                  }),
              }),
    }));
    const document = retainedDocument ?? newCanonicalConversation(runtime);
    const turns = [];
    const assets: Asset[] = [];
    const contextEntries = [];
    const itemMappings = [];
    loadedIndex = 0;
    for (let index = 0; index < input.segments.length; index += 1) {
        const segment = input.segments[index];
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, String(index));
        const blocks: UserContentBlock[] = [];
        if (segment.content !== undefined && segment.content.length > 0) {
            blocks.push({
                id: await deriveConversationId('block', runtime.input_operation_id, String(index), 'text'),
                type: 'text',
                text: segment.content,
                format: 'plain',
            });
        }
        for (let fileIndex = 0; fileIndex < (segment.files?.length ?? 0); fileIndex += 1) {
            const loaded = loadedFiles[loadedIndex++];
            if (loaded === undefined) throw new Error('Audio input preload mismatch');
            const assetId = await deriveConversationId(
                'asset',
                runtime.input_operation_id,
                String(index),
                String(fileIndex),
            );
            const blockId = await deriveConversationId(
                'block',
                runtime.input_operation_id,
                String(index),
                String(fileIndex),
            );
            const storage = { type: 'inline_base64' as const, data: loaded.data };
            assets.push({
                id: assetId,
                kind: loaded.source.mime_type.startsWith('audio/') ? 'audio' : 'video',
                mime_type: loaded.source.mime_type,
                storage,
                provenance: { type: 'received', source_turn_id: turnId },
                byte_length: loaded.bytes.byteLength,
                content_hash: (await hashContentBytes(loaded.bytes)).content_hash,
                created_at: runtime.recorded_at,
            });
            blocks.push({
                id: blockId,
                type: loaded.source.mime_type.startsWith('audio/') ? 'audio' : 'video',
                asset_id: assetId,
            });
        }
        const turn = {
            id: turnId,
            kind: segment.role === 'system' ? ('program' as const) : ('user' as const),
            authority: segment.role === 'system' ? ('system' as const) : ('ordinary' as const),
            blocks,
            status: 'completed' as const,
            timestamps: { recorded_at: runtime.recorded_at },
            model_visibility: 'include' as const,
            provenance: { type: 'received' as const },
        };
        turns.push(turn);
        const contextId = await deriveConversationId('context', runtime.input_operation_id, String(index));
        contextEntries.push({ id: contextId, type: 'source_turn' as const, turn_id: turnId });
        itemMappings.push(
            { canonical_id: turnId, native_id: `segments/${index}`, kind: 'turn' as const },
            ...blocks.map((block, blockIndex) => ({
                canonical_id: block.id,
                native_id: `segments/${index}/blocks/${blockIndex}`,
                kind: 'block' as const,
            })),
        );
    }
    const appended = await appendCanonicalPrompt(
        document,
        { turns, assets, context_entries: contextEntries, item_mappings: itemMappings },
        runtime,
        undefined,
        providerJsonValue({
            task,
            segments: input.segments.map((segment) => ({
                role: segment.role,
                content: segment.content ?? '',
                files: segment.files?.map((file) => ({ name: file.name, mime_type: file.mime_type })) ?? [],
            })),
        }),
    );
    const protocol = task === 'understanding' ? 'openai.chat.completions.audio' : `openai.audio.${task}`;
    const requestPayload = providerJsonValue({
        model: input.request_model ?? input.options.model,
        task,
        model_options: input.options.model_options ?? {},
        segments: input.segments.map((segment) => ({
            role: segment.role,
            content: segment.content ?? '',
            files: (segment.files ?? []).map((file) => ({ name: file.name, mime_type: file.mime_type })),
        })),
        input_assets: assets.map((asset) => ({ id: asset.id, content_hash: asset.content_hash })),
    });
    const accepted = acceptedCanonicalResponse(appended.document, runtime.response_operation_id);
    if (accepted !== undefined) {
        if (
            accepted.generation.request_id !== runtime.request_id ||
            accepted.generation.provider !== input.provider ||
            accepted.generation.protocol !== protocol ||
            accepted.generation.requested_model !== input.options.model ||
            accepted.generation.request_receipt.request_fingerprint !== (await fingerprintJson(requestPayload))
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible request identity`,
            );
        }
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered audio response cannot reconstruct original_response');
        }
        return recoverCanonicalExecutionResponse(
            { document: appended.document, runtime, accepted_response: accepted },
            input.options,
        );
    }
    if (retainedDocument !== undefined && retainedDocument.revision !== 0)
        throw new Error('File audio operations do not support conversation continuation');
    const receipt = await createRequestReceipt(
        appended.document,
        runtime,
        {
            provider: input.provider,
            protocol,
            model: input.options.model,
            adapter_version: OPENAI_AUDIO_ADAPTER_VERSION,
        },
        requestPayload,
        itemMappings,
        appended.tool_definitions,
    );
    const identities = await canonicalResponseIdentities(runtime);
    await publishCanonicalPreparedRequest(
        {
            document: appended.document,
            receipt,
            runtime,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            tool_definitions: appended.tool_definitions,
        },
        input.options,
    );
    const native = await executeOpenAIAudioNative(
        input.service,
        transportSegments,
        { ...input.options, conversation: undefined, include_original_response: true },
        input.request_options,
        {
            response_operation_id: runtime.response_operation_id,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            ...(runtime.completed_at === undefined ? {} : { completed_at: runtime.completed_at }),
            asset_storage: canonicalAudioAssetStorage,
        },
        input.request_model,
    );
    return finalizeOpenAIAudioCanonicalResponse(
        {
            document: appended.document,
            runtime,
            receipt,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            provider: input.provider,
            protocol,
            requested_model: input.options.model,
            resolved_model: input.request_model ?? input.options.model,
            payload: requestPayload,
        },
        native,
        input.options,
    );
}

/** Execute an already materialized canonical file-audio context without rebuilding PromptSegment input. */
export async function executeOpenAIAudioCanonicalContext(input: {
    service: OpenAI;
    options: CanonicalExecutionContextOptions;
    provider: string;
    request_model?: string;
    request_options?: { signal?: AbortSignal; timeout?: number };
}): Promise<CanonicalExecutionResponse> {
    input.request_options?.signal?.throwIfAborted();
    if (input.options.result_schema !== undefined) {
        throw new Error('File audio operations do not accept result schemas');
    }
    const task = openAIAudioTask(input.options.model);
    if (task === undefined) throw new Error(`Model ${input.options.model} is not an OpenAI file audio model`);
    const protocol = task === 'understanding' ? OPENAI_AUDIO_CHAT_PROTOCOL : `openai.audio.${task}`;
    const prepared = await prepareCanonicalContext({
        options: input.options,
        provider: input.provider,
        protocol,
        adapter_version: OPENAI_AUDIO_ADAPTER_VERSION,
    });
    if (prepared.tool_definitions.length > 0) {
        throw new Error('File audio canonical context does not support tool definitions');
    }
    const canonicalInput = await canonicalOpenAIAudioInput(prepared.request_document, input.options.model);
    const resolvedModel = input.request_model ?? input.options.model;
    const requestPayload = providerJsonValue({
        model: resolvedModel,
        task,
        model_options: input.options.model_options ?? {},
        segments: canonicalInput.segments,
        input_assets: canonicalInput.segments.flatMap((segment) =>
            segment.files.map((file) => ({ id: file.asset_id, content_hash: file.content_hash })),
        ),
    });
    await assertAcceptedCanonicalRequest(
        prepared,
        { provider: input.provider, protocol, model: input.options.model },
        requestPayload,
    );
    const receipt =
        prepared.accepted_response?.generation.request_receipt ??
        (await createRequestReceipt(
            prepared.document,
            prepared.runtime,
            {
                provider: input.provider,
                protocol,
                model: input.options.model,
                adapter_version: OPENAI_AUDIO_ADAPTER_VERSION,
            },
            requestPayload,
            canonicalInput.item_mappings,
            prepared.tool_definitions,
        ));
    const canonicalState = { ...prepared, receipt };
    if (prepared.accepted_response !== undefined) {
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered audio response cannot reconstruct original_response');
        }
        return recoverCanonicalExecutionResponse(canonicalState, input.options);
    }
    await publishCanonicalPreparedRequest(canonicalState, input.options);
    const native = await executeValidatedOpenAIAudioNative(
        input.service,
        canonicalInput.validated,
        input.options,
        input.request_options,
        {
            response_operation_id: prepared.runtime.response_operation_id,
            generation_id: prepared.generation_id,
            response_turn_id: prepared.response_turn_id,
            ...(prepared.runtime.completed_at === undefined ? {} : { completed_at: prepared.runtime.completed_at }),
            asset_storage: canonicalAudioAssetStorage,
        },
        input.request_model,
    );
    return finalizeOpenAIAudioCanonicalResponse(
        {
            document: prepared.document,
            runtime: prepared.runtime,
            receipt,
            generation_id: prepared.generation_id,
            response_turn_id: prepared.response_turn_id,
            provider: input.provider,
            protocol,
            requested_model: input.options.model,
            resolved_model: resolvedModel,
            payload: requestPayload,
        },
        native,
        input.options,
    );
}

/** Execute a finite SDK audio request without retaining multipart input in a prompt or conversation. */
export async function executeOpenAIAudioRequest<PromptT>(
    driver: Pick<AbstractDriver, 'createExecutionHttpAgentScope' | 'formatLlumiverseError' | 'provider'>,
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    emptyPrompt: PromptT,
    signal?: AbortSignal,
    requestModel = options.model,
    requestOptions?: { signal?: AbortSignal; timeout?: number },
): Promise<ExecutionResponse<PromptT>> {
    const start = Date.now();
    const scope = driver.createExecutionHttpAgentScope(options, signal !== undefined);
    const abort = () => void scope.abort();
    if (signal?.aborted) abort();
    else signal?.addEventListener('abort', abort, { once: true });
    try {
        const completion = await scope.run(() =>
            executeOpenAIAudio(service, segments, options, requestOptions ?? { signal }, requestModel),
        );
        return stripAudioFromCompletion({ ...completion, prompt: emptyPrompt, execution_time: Date.now() - start });
    } catch (error) {
        throw driver.formatLlumiverseError(error, {
            provider: driver.provider,
            model: options.model,
            operation: 'execute',
        });
    } finally {
        signal?.removeEventListener('abort', abort);
        await scope.close();
    }
}
