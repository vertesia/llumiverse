import { boundedAudioStream, canonicalAudioAssetStorage, storeAudioResult } from '../shared/audio.js';
import { mapOpenAIChatCompletionsUsage, mapOpenAITranscriptionUsage } from './usage.js';

export { boundedAudioStream } from '../shared/audio.js';

import {
    OpenAiAudioOptionsSchema,
    OpenAiSpeechOptionsSchema,
    OpenAiTranscriptionOptionsSchema,
} from '@llumiverse/common/schemas';
import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponse,
    deriveConversationId,
    fingerprintJson,
    hashContentBytes,
    isConversationDocumentFormat,
    type JsonValue,
    parseConversationDocument,
    type UserContentBlock,
} from '@llumiverse/conversation';
import {
    type AudioResult,
    type CanonicalExecutionResponse,
    type Completion,
    type CompletionResult,
    createCanonicalExecutionResponse,
    type DataSource,
    type ExecutionOptions,
    type ExecutionResponse,
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
    canonicalRecoveredOutputFragment,
    canonicalResponseIdentities,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    resolveConversationRuntime,
} from '../conversation/canonical-runtime.js';

const OPENAI_AUDIO_ADAPTER_VERSION = '2026-09-30.canonical.1';

interface OpenAIAudioNativeExecution {
    result: CompletionResult[];
    finish_reason?: string | null;
    token_usage?: Completion['token_usage'];
    original_response?: unknown;
    response_fingerprint_payload: JsonValue;
    audio_evidence?: Array<{
        value: string;
        byte_length: number;
        content_hash: string;
    }>;
}

export async function openAIInputAudioPart(
    file: DataSource,
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
    files: DataSource[];
    text: string;
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
    return { task, files, text };
}

export async function executeOpenAIAudioNative(
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    requestOptions: { signal?: AbortSignal; timeout?: number } | undefined,
    requestModel = options.model,
): Promise<OpenAIAudioNativeExecution> {
    const signal = requestOptions?.signal;
    signal?.throwIfAborted();
    if (options.conversation || options.tools?.length || options.result_schema || options.format) {
        throw new Error(
            'File audio operations do not accept conversation, tools, result schemas, or custom formatting',
        );
    }
    const { task, files, text } = validateOpenAIAudioInput(segments, options.model);
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
        return {
            result: [
                ...(message.content || message.audio.transcript
                    ? [{ type: 'text' as const, value: message.content ?? message.audio.transcript }]
                    : []),
                audio,
            ],
            finish_reason: result.choices[0]?.finish_reason,
            token_usage: mapOpenAIChatCompletionsUsage(result.usage),
            original_response: options.include_original_response ? result : undefined,
            response_fingerprint_payload: providerJsonValue(result),
            audio_evidence: [
                {
                    value: audio.value,
                    byte_length: audioBytes.byteLength,
                    content_hash: (await hashContentBytes(audioBytes)).content_hash,
                },
            ],
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
                return {
                    result: [
                        { type: 'text', value: result.text },
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
                    finish_reason: 'stop',
                    token_usage: mapOpenAITranscriptionUsage(result.usage),
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
            return {
                result: [{ type: 'text', value: result.text }],
                finish_reason: 'stop',
                token_usage: mapOpenAITranscriptionUsage(result.usage),
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
    return {
        result: [audio],
        finish_reason: 'stop',
        original_response: options.include_original_response ? response : undefined,
        response_fingerprint_payload: {
            audio: {
                mime_type: audio.mime_type,
                data: Buffer.from(audioBytes).toString('base64'),
            },
        },
        audio_evidence: [
            {
                value: audio.value,
                byte_length: audioBytes.byteLength,
                content_hash: (await hashContentBytes(audioBytes)).content_hash,
            },
        ],
    };
}

export async function executeOpenAIAudio(
    service: OpenAI,
    segments: PromptSegment[],
    options: ExecutionOptions,
    requestOptions: { signal?: AbortSignal; timeout?: number } | undefined,
    requestModel = options.model,
): Promise<Completion> {
    const {
        response_fingerprint_payload: _responseFingerprintPayload,
        audio_evidence: _audioEvidence,
        ...result
    } = await executeOpenAIAudioNative(service, segments, options, requestOptions, requestModel);
    return {
        ...result,
        ...(result.finish_reason == null ? { finish_reason: undefined } : { finish_reason: result.finish_reason }),
    };
}

function generationUsage(
    usage: Completion['token_usage'],
): import('@llumiverse/conversation').GenerationUsage | undefined {
    if (usage === undefined) return undefined;
    const basis = 'openai_audio_tokens';
    return {
        ...(usage.prompt === undefined ? {} : { input_tokens: usage.prompt }),
        ...(usage.prompt_new === undefined ? {} : { input_new_tokens: usage.prompt_new }),
        ...(usage.prompt_cached === undefined ? {} : { cache_read_tokens: usage.prompt_cached }),
        ...(usage.prompt_cache_write === undefined ? {} : { cache_write_tokens: usage.prompt_cache_write }),
        ...(usage.result === undefined ? {} : { output_tokens: usage.result }),
        ...(usage.total === undefined ? {} : { total_tokens: usage.total }),
        accounting_provenance: Object.fromEntries(
            [
                ['input_tokens', usage.prompt],
                ['input_new_tokens', usage.prompt_new],
                ['cache_read_tokens', usage.prompt_cached],
                ['cache_write_tokens', usage.prompt_cache_write],
                ['output_tokens', usage.result],
                ['total_tokens', usage.total],
            ].flatMap(([key, value]) =>
                value === undefined ? [] : [[key, { method: 'reported' as const, accounting_basis: basis }]],
            ),
        ),
    };
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

/** Direct canonical execution for finite OpenAI audio endpoints. */
export async function executeOpenAIAudioCanonical(input: {
    service: OpenAI;
    segments: PromptSegment[];
    options: ExecutionOptions;
    provider: string;
    request_model?: string;
    request_options?: { signal?: AbortSignal; timeout?: number };
}): Promise<CanonicalExecutionResponse> {
    const runtime = resolveConversationRuntime(input.options);
    const retainedDocument = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : undefined;
    if (input.options.conversation !== undefined && retainedDocument === undefined) {
        throw new Error('File audio operations do not accept legacy conversation input');
    }
    if (
        retainedDocument !== undefined &&
        input.options.conversation_runtime?.conversation_id !== undefined &&
        retainedDocument.id !== runtime.conversation_id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    if (input.options.tools?.length || input.options.result_schema || input.options.format) {
        throw new Error('File audio operations do not accept tools, result schemas, or custom formatting');
    }
    const { task } = validateOpenAIAudioInput(input.segments, input.options.model);
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
        return createCanonicalExecutionResponse(
            appended.document,
            runtime.response_operation_id,
            {},
            canonicalRecoveredOutputFragment(
                await input.options.load_recovered_canonical_output?.({
                    conversation_id: appended.document.id,
                    response_operation_id: runtime.response_operation_id,
                }),
            ),
        );
    }
    if (retainedDocument !== undefined)
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
            native_conversation: transportSegments,
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
        input.request_model,
    );
    const completedAt = runtime.completed_at ?? new Date().toISOString();
    const responseAssets: Asset[] = [];
    const responseBlocks: AgentContentBlock[] = [];
    const remainingAudioEvidence = [...(native.audio_evidence ?? [])];
    for (let index = 0; index < native.result.length; index += 1) {
        const result = native.result[index];
        const blockId = await deriveConversationId('block', runtime.response_operation_id, String(index));
        if (result.type === 'text')
            responseBlocks.push({ id: blockId, type: 'text', text: result.value, format: 'plain' });
        else if (result.type === 'json')
            responseBlocks.push({ id: blockId, type: 'json', value: result.value as JsonValue });
        else if (result.type === 'audio') {
            const assetId = await deriveConversationId('asset', runtime.response_operation_id, String(index));
            const evidenceIndex = remainingAudioEvidence.findIndex((candidate) => candidate.value === result.value);
            const evidence = evidenceIndex < 0 ? undefined : remainingAudioEvidence.splice(evidenceIndex, 1)[0];
            responseAssets.push({
                id: assetId,
                kind: 'audio',
                mime_type: result.mime_type,
                storage: canonicalAudioAssetStorage(result.value),
                provenance: {
                    type: 'generated',
                    generation_id: identities.generation_id,
                    source_turn_id: identities.response_turn_id,
                },
                created_at: completedAt,
                ...(evidence === undefined
                    ? {}
                    : { byte_length: evidence.byte_length, content_hash: evidence.content_hash }),
                media: {
                    ...(result.container === undefined ? {} : { container: result.container }),
                    ...(result.codec === undefined ? {} : { codec: result.codec }),
                    ...(result.sample_rate === undefined ? {} : { sample_rate: result.sample_rate }),
                    ...(result.channels === undefined ? {} : { channels: result.channels }),
                    ...(result.sample_encoding === undefined ? {} : { sample_encoding: result.sample_encoding }),
                    ...(result.byte_order === undefined ? {} : { byte_order: result.byte_order }),
                },
                metadata: {
                    audio_result: {
                        value: result.value,
                        mime_type: result.mime_type,
                        ...(result.container === undefined ? {} : { container: result.container }),
                        ...(result.codec === undefined ? {} : { codec: result.codec }),
                        ...(result.sample_rate === undefined ? {} : { sample_rate: result.sample_rate }),
                        ...(result.channels === undefined ? {} : { channels: result.channels }),
                        ...(result.sample_encoding === undefined ? {} : { sample_encoding: result.sample_encoding }),
                        ...(result.byte_order === undefined ? {} : { byte_order: result.byte_order }),
                    },
                },
            });
            responseBlocks.push({ id: blockId, type: 'audio', asset_id: assetId });
        }
    }
    const generation = await createExecutedGeneration({
        id: identities.generation_id,
        runtime,
        receipt,
        provider: input.provider,
        protocol,
        adapter_version: OPENAI_AUDIO_ADAPTER_VERSION,
        requested_model: input.options.model,
        resolved_model: input.request_model ?? input.options.model,
        finish_reason: native.finish_reason ?? undefined,
        usage: generationUsage(native.token_usage),
    });
    const responseTurn = {
        id: identities.response_turn_id,
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks: responseBlocks,
        status: 'completed' as const,
        timestamps: { recorded_at: completedAt, completed_at: completedAt },
        model_visibility: 'include' as const,
        provenance: { type: 'generated' as const },
        generation_id: generation.id,
    };
    const finalDocument = appendDecodedConversationResponse(
        {
            document: appended.document,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            receipt,
            payload: requestPayload,
            diagnostics: [],
        },
        {
            turns: [responseTurn],
            assets: responseAssets,
            generation,
            diagnostics: [],
            payload_fingerprint: await fingerprintJson(native.response_fingerprint_payload),
        },
        {
            operation_id: runtime.response_operation_id,
            recorded_at: completedAt,
        },
    ).document;
    return createCanonicalExecutionResponse(finalDocument, runtime.response_operation_id, {
        ...(input.options.include_original_response && native.original_response !== undefined
            ? { original_response: native.original_response }
            : {}),
    });
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
