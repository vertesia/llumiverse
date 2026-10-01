import type { Interactions } from '@google/genai';
import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponse,
    buildConversationTurn,
    type ConversationTurn,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    deriveConversationId,
    fingerprintJson,
    type GenerationUsage,
    type ImageBlock,
    isConversationDocumentFormat,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    parseConversationDocument,
    type TextBlock,
    type VideoBlock,
} from '@llumiverse/conversation';
import {
    type AIModel,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalExecutionStream,
    type CanonicalStreamOpenOptions,
    type Completion,
    type CompletionResult,
    createCanonicalExecutionResponse,
    type DriverCompletionStream,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    PromptRole,
    type PromptSegment,
    type VertexAIGeminiOmniVideoOptions,
} from '@llumiverse/core';
import {
    acceptedCanonicalResponse,
    appendCanonicalPrompt,
    assertAcceptedCanonicalRequest,
    canonicalResponseIdentities,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
    resolveConversationRuntime,
} from '../../conversation/canonical-runtime.js';
import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

export const GEMINI_OMNI_VIDEO_MODEL = 'gemini-omni-flash-preview';
export const GEMINI_OMNI_1_1_VIDEO_MODEL = 'gemini-omni-1.1-flash-preview';
export const GEMINI_OMNI_VIDEO_MODELS = [GEMINI_OMNI_VIDEO_MODEL, GEMINI_OMNI_1_1_VIDEO_MODEL] as const;

export type GeminiOmniVideoModel = (typeof GEMINI_OMNI_VIDEO_MODELS)[number];

export function isGeminiOmniVideoModel(model: string): model is GeminiOmniVideoModel {
    return GEMINI_OMNI_VIDEO_MODELS.some((candidate) => candidate === model);
}

const SUPPORTED_IMAGE_MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif']);
const SUPPORTED_VIDEO_MIME_TYPES = new Set([
    'video/3gpp',
    'video/mp4',
    'video/mpeg',
    'video/mpegs',
    'video/mpg',
    'video/quicktime',
    'video/webm',
    'video/wmv',
    'video/x-flv',
    'video/x-ms-wmv',
]);

const OMNI_VIDEO_PROTOCOL = 'google.vertex.interactions.video';
const OMNI_VIDEO_ADAPTER_VERSION = '2026-09-30.canonical.1';
const OMNI_VIDEO_REGION = 'global';
const OMNI_VIDEO_API_VERSION = 'v1beta1';

type OmniVideoMediaInput = Extract<Interactions.Content, { type: 'image' | 'video' }>;

export interface OmniVideoPrompt {
    text: string;
    media: OmniVideoMediaInput[];
    source: Array<{
        role: PromptRole;
        content: string;
        media: OmniVideoMediaInput[];
    }>;
}

class OmniVideoTerminalError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'OmniVideoTerminalError';
    }
}

class OmniVideoInteractionError extends Error {
    constructor(
        message: string,
        readonly retryable: boolean | undefined,
    ) {
        super(message);
        this.name = 'OmniVideoInteractionError';
    }
}

function isGcsUri(value: string, requireObject = false): boolean {
    const match = /^gs:\/\/([^/]+)\/(.*)$/.exec(value);
    return !!match && (!requireObject || match[2].length > 0);
}

function normalizeOutputPrefix(value: string | undefined): string {
    if (!value || !isGcsUri(value)) {
        throw new OmniVideoTerminalError('Gemini Omni video generation requires a valid GCS output prefix');
    }
    return value.endsWith('/') ? value : `${value}/`;
}

function resolveTask(
    model: GeminiOmniVideoModel,
    options: VertexAIGeminiOmniVideoOptions | undefined,
    media: OmniVideoMediaInput[],
) {
    const imageCount = media.filter((item) => item.type === 'image').length;
    const videoCount = media.filter((item) => item.type === 'video').length;
    const task = options?.task ?? (media.length === 0 ? 'text_to_video' : undefined);
    if (!task) {
        throw new OmniVideoTerminalError('Gemini Omni video generation with media requires an explicit task');
    }
    if (task === 'extend' && model !== GEMINI_OMNI_1_1_VIDEO_MODEL) {
        throw new OmniVideoTerminalError(`${model} does not support extend`);
    }
    if (options?.resolution && model === GEMINI_OMNI_VIDEO_MODEL && options.resolution !== '720p') {
        throw new OmniVideoTerminalError(`${model} only supports 720p output`);
    }

    switch (task) {
        case 'text_to_video':
            if (media.length !== 0) throw new OmniVideoTerminalError('text_to_video does not accept media inputs');
            break;
        case 'image_to_video': {
            const maximumImages = model === GEMINI_OMNI_1_1_VIDEO_MODEL ? 2 : 1;
            if (videoCount !== 0 || imageCount < 1 || imageCount > maximumImages) {
                const expected = maximumImages === 1 ? 'exactly one image input' : 'one or two image inputs';
                throw new OmniVideoTerminalError(`image_to_video requires ${expected}`);
            }
            break;
        }
        case 'reference_to_video':
            if (media.length === 0 || imageCount > 10 || videoCount > 3) {
                throw new OmniVideoTerminalError(
                    'reference_to_video requires media with at most ten images and three videos',
                );
            }
            break;
        case 'edit':
            if (imageCount !== 0 || videoCount !== 1) {
                throw new OmniVideoTerminalError('edit requires exactly one video input');
            }
            break;
        case 'extend':
            if (imageCount !== 0 || videoCount !== 1) {
                throw new OmniVideoTerminalError('extend requires exactly one video input');
            }
            break;
    }
    return task;
}

function interactionFailureRetryable(errors: Interactions.Error[] | undefined): boolean | undefined {
    const details = JSON.stringify(errors ?? []).toLowerCase();
    if (
        details.includes('invalid_argument') ||
        details.includes('unauthenticated') ||
        details.includes('permission_denied') ||
        details.includes('not_found') ||
        details.includes('failed_precondition') ||
        details.includes('out_of_range') ||
        details.includes('unimplemented')
    ) {
        return false;
    }
    if (
        details.includes('resource_exhausted') ||
        details.includes('rate_limit') ||
        details.includes('throttl') ||
        details.includes('aborted') ||
        details.includes('internal') ||
        details.includes('unavailable') ||
        details.includes('deadline_exceeded') ||
        details.includes('timeout')
    ) {
        return true;
    }
    return undefined;
}

type OmniVideoOutput =
    | { type: 'text'; text: string; native_id: string }
    | { type: 'video'; uri: string; mime_type: string; native_id: string };

interface PreparedOmniVideoRequest {
    payload: Interactions.CreateModelInteractionParamsNonStreaming;
    payload_json: JsonValue;
    output_prefix: string;
    target_options: JsonObject;
}

function decodeOmniResults(response: Interactions.Interaction, outputPrefix: string): OmniVideoOutput[] {
    if (response.status !== 'completed') {
        const details = response.errors?.length ? `: ${JSON.stringify(response.errors)}` : '';
        const permanent = new Set(['requires_action', 'cancelled', 'budget_exceeded']);
        const transient = new Set(['queued', 'in_progress', 'incomplete']);
        throw new OmniVideoInteractionError(
            `Gemini Omni video interaction did not complete (status: ${response.status})${details}`,
            permanent.has(response.status)
                ? false
                : transient.has(response.status)
                  ? true
                  : interactionFailureRetryable(response.errors),
        );
    }

    const results: OmniVideoOutput[] = [];
    let hasVideoOutput = false;
    for (let stepIndex = 0; stepIndex < (response.steps?.length ?? 0); stepIndex += 1) {
        const step = response.steps?.[stepIndex];
        if (step?.type !== 'model_output') continue;
        for (let contentIndex = 0; contentIndex < (step.content?.length ?? 0); contentIndex += 1) {
            const part = step.content?.[contentIndex];
            if (part?.type === 'text') {
                if (part.text) {
                    results.push({
                        type: 'text',
                        text: part.text,
                        native_id: `steps/${stepIndex}/content/${contentIndex}`,
                    });
                }
                continue;
            }
            if (part?.type === 'video') {
                hasVideoOutput = true;
                if (part.data !== undefined) {
                    throw new OmniVideoTerminalError('Gemini Omni returned inline video data instead of URI delivery');
                }
                if (!part.uri || !isGcsUri(part.uri, true)) {
                    throw new OmniVideoTerminalError('Gemini Omni returned a missing or invalid GCS video URI');
                }
                if (!part.uri.startsWith(outputPrefix)) {
                    throw new OmniVideoTerminalError(
                        'Gemini Omni returned a video URI outside the requested output prefix',
                    );
                }
                if (part.mime_type && !part.mime_type.startsWith('video/')) {
                    throw new OmniVideoTerminalError(
                        `Gemini Omni returned an invalid video MIME type: ${part.mime_type}`,
                    );
                }
                results.push({
                    type: 'video',
                    uri: part.uri,
                    mime_type: part.mime_type ?? 'video/mp4',
                    native_id: `steps/${stepIndex}/content/${contentIndex}`,
                });
            }
        }
    }
    if (!hasVideoOutput) {
        throw new OmniVideoTerminalError('Gemini Omni completed without a video output URI');
    }
    return results;
}

function legacyOmniResults(results: readonly OmniVideoOutput[]): CompletionResult[] {
    return results.map((result) =>
        result.type === 'text' ? { type: 'text', value: result.text } : { type: 'video', value: result.uri },
    );
}

function prepareOmniVideoRequest(
    modelId: GeminiOmniVideoModel,
    prompt: OmniVideoPrompt,
    options: ExecutionOptions,
): PreparedOmniVideoRequest {
    const modelOptions = options.model_options as VertexAIGeminiOmniVideoOptions | undefined;
    const task = resolveTask(modelId, modelOptions, prompt.media);
    const outputPrefix = normalizeOutputPrefix(options.output_storage_uri);
    const responseFormat = {
        type: 'video' as const,
        delivery: 'uri' as const,
        gcs_uri: outputPrefix,
        duration: `${modelOptions?.duration_seconds ?? 5}s`,
        ...(modelOptions?.aspect_ratio ? { aspect_ratio: modelOptions.aspect_ratio } : {}),
        ...(modelOptions?.resolution ? { resolution: modelOptions.resolution } : {}),
    } satisfies Interactions.VideoResponseFormat;
    const payload = {
        model: modelId,
        input: [{ type: 'text' as const, text: prompt.text }, ...prompt.media],
        response_format: [responseFormat],
        generation_config: { video_config: { task } },
    } satisfies Interactions.CreateModelInteractionParamsNonStreaming;
    return {
        payload,
        payload_json: providerJsonValue(payload),
        output_prefix: outputPrefix,
        target_options: providerJsonValue({
            region: OMNI_VIDEO_REGION,
            api_version: OMNI_VIDEO_API_VERSION,
            task,
            response_format: responseFormat,
            input_media: prompt.media.map((item, index) => ({
                index,
                type: item.type,
                mime_type: item.mime_type ?? (item.type === 'image' ? 'image/png' : 'video/mp4'),
            })),
        }) as JsonObject,
    };
}

function safeOmniUsageNumber(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function canonicalOmniUsage(usage: Interactions.Usage | undefined): GenerationUsage | undefined {
    if (usage === undefined) return undefined;
    const basis = 'google_interactions_tokens';
    const input = safeOmniUsageNumber(usage.total_input_tokens);
    const output = safeOmniUsageNumber(usage.total_output_tokens);
    const reportedTotal = safeOmniUsageNumber(usage.total_tokens);
    const total =
        input !== undefined &&
        output !== undefined &&
        Number.isSafeInteger(input + output) &&
        input + output === reportedTotal
            ? reportedTotal
            : undefined;
    const cacheCandidate = safeOmniUsageNumber(usage.total_cached_tokens);
    const cacheRead =
        cacheCandidate !== undefined && (input === undefined || cacheCandidate <= input) ? cacheCandidate : undefined;
    const inputNew = input !== undefined && cacheRead !== undefined ? input - cacheRead : undefined;
    const reasoningCandidate = safeOmniUsageNumber(usage.total_thought_tokens);
    const reasoning =
        reasoningCandidate !== undefined && (output === undefined || reasoningCandidate <= output)
            ? reasoningCandidate
            : undefined;
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(inputNew === undefined ? {} : { input_new_tokens: inputNew }),
        ...(cacheRead === undefined ? {} : { cache_read_tokens: cacheRead }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(reasoning === undefined ? {} : { reasoning_tokens: reasoning }),
        ...(total === undefined ? {} : { total_tokens: total }),
        accounting_provenance: {
            ...(input === undefined ? {} : { input_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(inputNew === undefined
                ? {}
                : { input_new_tokens: { method: 'derived' as const, accounting_basis: basis } }),
            ...(cacheRead === undefined
                ? {}
                : { cache_read_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(output === undefined
                ? {}
                : { output_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(reasoning === undefined
                ? {}
                : { reasoning_tokens: { method: 'reported' as const, accounting_basis: basis } }),
            ...(total === undefined ? {} : { total_tokens: { method: 'reported' as const, accounting_basis: basis } }),
        },
        ...(inputNew === undefined
            ? {}
            : { input_partition: { type: 'complete_disjoint' as const, cache_write_bucket: 'inapplicable' as const } }),
        reported_usage: [
            {
                source: 'provider',
                protocol: OMNI_VIDEO_PROTOCOL,
                accounting_basis: basis,
                payload: providerJsonValue(usage),
            },
        ],
    };
}

async function omniInputRecords(
    prompt: OmniVideoPrompt,
    runtime: ReturnType<typeof resolveConversationRuntime>,
): Promise<{ turns: ConversationTurn[]; assets: Asset[]; mappings: NativeItemMapping[] }> {
    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    let mediaIndex = 0;
    for (let segmentIndex = 0; segmentIndex < prompt.source.length; segmentIndex += 1) {
        const source = prompt.source[segmentIndex];
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, String(segmentIndex));
        const blocks: Array<TextBlock | ImageBlock | VideoBlock> = [];
        if (source.content.length > 0) {
            const textBlock = createTextBlock({
                id: await deriveConversationId('block', runtime.input_operation_id, String(segmentIndex), 'text'),
                text: source.content,
                format: 'plain',
            });
            blocks.push(textBlock);
            mappings.push({
                canonical_id: textBlock.id,
                native_id: `input/0/text/segments/${segmentIndex}`,
                kind: 'block',
            });
        }
        for (let segmentMediaIndex = 0; segmentMediaIndex < source.media.length; segmentMediaIndex += 1) {
            const media = source.media[segmentMediaIndex];
            if (media.uri === undefined) throw new Error('Gemini Omni canonical media input is missing its URI');
            const assetId = await deriveConversationId(
                'asset',
                runtime.input_operation_id,
                String(segmentIndex),
                String(segmentMediaIndex),
            );
            const blockId = await deriveConversationId(
                'block',
                runtime.input_operation_id,
                String(segmentIndex),
                String(segmentMediaIndex),
            );
            blocks.push({ id: blockId, type: media.type, asset_id: assetId });
            const mimeType = media.mime_type ?? (media.type === 'image' ? 'image/png' : 'video/mp4');
            assets.push({
                id: assetId,
                kind: media.type,
                mime_type: mimeType,
                storage: { type: 'external', resolver: 'google_uri', locator: { uri: media.uri } },
                provenance: { type: 'received', source_turn_id: turnId },
                created_at: runtime.recorded_at,
            });
            mappings.push({ canonical_id: blockId, native_id: `input/${mediaIndex + 1}`, kind: 'block' });
            mediaIndex += 1;
        }
        if (blocks.length === 0) continue;
        const common = {
            id: turnId,
            authority: 'ordinary' as const,
            model_visibility: 'include' as const,
            status: 'completed' as const,
            timestamps: { recorded_at: runtime.recorded_at },
            provenance: { type: 'received' as const },
            blocks,
        };
        const turn =
            source.role === PromptRole.system || source.role === PromptRole.safety
                ? createProgramTurn({ ...common, authority: 'system' })
                : source.role === PromptRole.assistant
                  ? buildConversationTurn({ ...common, kind: 'agent' })
                  : createUserTurn(common);
        turns.push(turn);
        mappings.push({ canonical_id: turn.id, native_id: `source/segments/${segmentIndex}`, kind: 'turn' });
    }
    return { turns, assets, mappings };
}

async function executeOmniVideoCanonical(input: {
    driver: VertexAIDriver;
    model_id: GeminiOmniVideoModel;
    prompt: OmniVideoPrompt;
    options: ExecutionOptions;
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const preparedRequest = prepareOmniVideoRequest(input.model_id, input.prompt, input.options);
    const runtime = resolveConversationRuntime(input.options);
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation !== undefined &&
        input.options.conversation !== null &&
        !isConversationDocumentFormat(input.options.conversation)
    ) {
        throw new TypeError('Gemini Omni video does not support legacy conversation input');
    }
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    const inputRecords = await omniInputRecords(input.prompt, runtime);
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        (Object.keys(document.generations).length > 0 ||
            document.turns.some((turn) => !inputRecords.turns.some((record) => record.id === turn.id)))
    ) {
        throw new Error('Gemini Omni video does not support conversation continuation');
    }
    const contextEntries = await Promise.all(
        inputRecords.turns.map(async (turn, index) => ({
            id: await deriveConversationId('context', runtime.input_operation_id, String(index)),
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
    );
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: inputRecords.turns,
            assets: inputRecords.assets,
            context_entries: contextEntries,
            item_mappings: inputRecords.mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        providerJsonValue(input.prompt.source),
    );
    document = appended.document;
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (accepted !== undefined) {
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.driver.provider, protocol: OMNI_VIDEO_PROTOCOL, model: input.options.model },
            preparedRequest.payload_json,
        );
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered Gemini Omni response cannot reconstruct original_response');
        }
        return recoverCanonicalExecutionResponse({ document, runtime, accepted_response: accepted }, input.options);
    }
    const receipt = await createRequestReceipt(
        document,
        { ...runtime, conversation_id: document.id },
        {
            provider: input.driver.provider,
            protocol: OMNI_VIDEO_PROTOCOL,
            model: input.options.model,
            adapter_version: OMNI_VIDEO_ADAPTER_VERSION,
            options: preparedRequest.target_options,
        },
        preparedRequest.payload_json,
        inputRecords.mappings,
        appended.tool_definitions,
    );
    const identities = await canonicalResponseIdentities(runtime);
    await publishCanonicalPreparedRequest(
        {
            document,
            receipt,
            runtime: { ...runtime, conversation_id: document.id },
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            tool_definitions: appended.tool_definitions,
        },
        input.options,
    );
    input.signal?.throwIfAborted();
    const response = await input.driver
        .getFetchClientForRegion(OMNI_VIDEO_REGION, OMNI_VIDEO_API_VERSION)
        .post<Interactions.Interaction>('interactions', {
            payload: preparedRequest.payload,
            signal: input.signal,
            timeoutMs: input.driver.getRequestTimeoutMs(input.options.httpTimeout),
        });
    input.signal?.throwIfAborted();
    const outputs = decodeOmniResults(response, preparedRequest.output_prefix);
    const completedAt = runtime.completed_at ?? runtime.recorded_at;
    const completedRuntime = { ...runtime, completed_at: completedAt };
    const assets: Asset[] = [];
    const blocks: AgentContentBlock[] = [];
    for (let index = 0; index < outputs.length; index += 1) {
        const output = outputs[index];
        if (output.type === 'text') {
            blocks.push(
                createTextBlock({
                    id: await deriveConversationId('block', runtime.response_operation_id, String(index)),
                    text: output.text,
                    format: 'plain',
                }),
            );
            continue;
        }
        const assetId = await deriveConversationId('asset', runtime.response_operation_id, String(index));
        assets.push({
            id: assetId,
            kind: 'video',
            mime_type: output.mime_type,
            storage: { type: 'external', resolver: 'google_uri', locator: { uri: output.uri } },
            provenance: {
                type: 'generated',
                generation_id: identities.generation_id,
                source_turn_id: identities.response_turn_id,
            },
            ...(output.mime_type === 'video/mp4' ? { media: { container: 'mp4' } } : {}),
            created_at: completedAt,
        });
        blocks.push({
            id: await deriveConversationId('block', runtime.response_operation_id, String(index)),
            type: 'video',
            asset_id: assetId,
        });
    }
    const generation = await createExecutedGeneration({
        id: identities.generation_id,
        runtime: completedRuntime,
        receipt,
        provider: input.driver.provider,
        protocol: OMNI_VIDEO_PROTOCOL,
        adapter_version: OMNI_VIDEO_ADAPTER_VERSION,
        requested_model: input.options.model,
        resolved_model: input.model_id,
        provider_response_id: response.id,
        finish_reason: 'stop',
        usage: canonicalOmniUsage(response.usage),
    });
    const responseTurn = {
        id: identities.response_turn_id,
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks,
        status: 'completed' as const,
        timestamps: { recorded_at: completedAt, completed_at: completedAt },
        model_visibility: 'include' as const,
        provenance: { type: 'generated' as const },
        generation_id: generation.id,
    };
    const responseEvidence = providerJsonValue({
        id: response.id,
        status: response.status,
        outputs,
        usage: response.usage,
    });
    const finalDocument = appendDecodedConversationResponse(
        {
            document,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            receipt,
            payload: preparedRequest.payload_json,
            diagnostics: [],
        },
        {
            turns: [responseTurn],
            assets,
            generation,
            diagnostics: [],
            payload_fingerprint: await fingerprintJson(responseEvidence),
        },
        { operation_id: runtime.response_operation_id, recorded_at: completedAt },
    ).document;
    return createCanonicalExecutionResponse(finalDocument, runtime.response_operation_id, {
        ...(input.options.include_original_response ? { original_response: response } : {}),
    });
}

export class GeminiOmniVideoModelDefinition implements ModelDefinition<OmniVideoPrompt> {
    readonly model: AIModel;
    readonly canonical_conversation_supported = true;

    constructor(private readonly modelId: GeminiOmniVideoModel = GEMINI_OMNI_VIDEO_MODEL) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Video,
            can_stream: false,
        } satisfies AIModel;
    }

    async createPrompt(
        _driver: VertexAIDriver,
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<OmniVideoPrompt> {
        if (options.conversation && !isConversationDocumentFormat(options.conversation))
            throw new OmniVideoTerminalError('Gemini Omni video does not support conversation resume');
        if (options.tools?.length) throw new OmniVideoTerminalError('Gemini Omni video does not support tools');
        if (isConversationDocumentFormat(options.conversation)) {
            const document = parseConversationDocument(options.conversation);
            if (document.context.active_tool_definition_ids.length > 0) {
                throw new OmniVideoTerminalError('Gemini Omni video does not support active canonical tools');
            }
        }
        if (options.result_schema)
            throw new OmniVideoTerminalError('Gemini Omni video does not support result schemas');
        if (options.conversation_runtime?.materialized_input !== undefined) {
            throw new OmniVideoTerminalError('Gemini Omni video does not support materialized conversation input');
        }
        normalizeOutputPrefix(options.output_storage_uri);

        const supportedRoles = new Set([PromptRole.system, PromptRole.safety, PromptRole.user, PromptRole.assistant]);
        const mediaKinds: OmniVideoMediaInput[] = [];
        for (const segment of segments) {
            if (!supportedRoles.has(segment.role)) {
                throw new OmniVideoTerminalError(`Gemini Omni video does not support ${segment.role} input`);
            }
            for (const file of segment.files ?? []) {
                const type = SUPPORTED_IMAGE_MIME_TYPES.has(file.mime_type)
                    ? 'image'
                    : SUPPORTED_VIDEO_MIME_TYPES.has(file.mime_type)
                      ? 'video'
                      : undefined;
                if (!type) {
                    throw new OmniVideoTerminalError(
                        `Gemini Omni video does not support input MIME type ${file.mime_type}`,
                    );
                }
                mediaKinds.push({ type, uri: 'gs://preflight/input', mime_type: file.mime_type });
            }
        }
        resolveTask(this.modelId, options.model_options as VertexAIGeminiOmniVideoOptions | undefined, mediaKinds);

        const text = segments
            .map((segment) => segment.content.trim())
            .filter(Boolean)
            .join('\n');
        if (!text) throw new OmniVideoTerminalError('Gemini Omni video requires a non-empty text prompt');

        const media: OmniVideoMediaInput[] = [];
        const source: OmniVideoPrompt['source'] = [];
        for (const segment of segments) {
            const sourceMedia: OmniVideoMediaInput[] = [];
            for (const file of segment.files ?? []) {
                const type = SUPPORTED_IMAGE_MIME_TYPES.has(file.mime_type)
                    ? 'image'
                    : SUPPORTED_VIDEO_MIME_TYPES.has(file.mime_type)
                      ? 'video'
                      : undefined;
                if (!type) {
                    throw new OmniVideoTerminalError(
                        `Gemini Omni video does not support input MIME type ${file.mime_type}`,
                    );
                }
                const uri = await file.getURI();
                if (!isGcsUri(uri, true)) {
                    throw new OmniVideoTerminalError('Gemini Omni video media inputs must expose a GCS object URI');
                }
                const item = { type, uri, mime_type: file.mime_type } as OmniVideoMediaInput;
                media.push(item);
                sourceMedia.push(item);
            }
            source.push({ role: segment.role, content: segment.content, media: sourceMedia });
        }

        resolveTask(this.modelId, options.model_options as VertexAIGeminiOmniVideoOptions | undefined, media);
        return { text, media, source };
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: OmniVideoPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        if (isConversationDocumentFormat(options.conversation)) {
            throw new OmniVideoTerminalError('Legacy Gemini Omni execution cannot consume a canonical conversation');
        }
        const prepared = prepareOmniVideoRequest(this.modelId, prompt, options);
        const response = await driver
            .getFetchClientForRegion(OMNI_VIDEO_REGION, OMNI_VIDEO_API_VERSION)
            .post<Interactions.Interaction>('interactions', {
                payload: prepared.payload,
                signal,
                timeoutMs: driver.getRequestTimeoutMs(options.httpTimeout),
            });

        const tokenUsage: ExecutionTokenUsage = {
            total: response.usage?.total_tokens,
            prompt: response.usage?.total_input_tokens,
            result: response.usage?.total_output_tokens,
        };
        return {
            result: legacyOmniResults(decodeOmniResults(response, prepared.output_prefix)),
            token_usage: tokenUsage,
            finish_reason: 'stop',
            ...(options.include_original_response ? { original_response: response } : {}),
        };
    }

    requestCanonicalTextCompletion(
        driver: VertexAIDriver,
        prompt: OmniVideoPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        return executeOmniVideoCanonical({ driver, model_id: this.modelId, prompt, options, signal });
    }

    requestCanonicalTextCompletionStream(
        _driver: VertexAIDriver,
        _prompt: OmniVideoPrompt,
        _options: ExecutionOptions,
        _signal?: AbortSignal,
    ): Promise<CanonicalExecutionStream> {
        return Promise.reject(new OmniVideoTerminalError('Gemini Omni video uses finite canonical streaming'));
    }

    requestCanonicalTextCompletionEventStream(
        _driver: VertexAIDriver,
        _prompt: OmniVideoPrompt,
        _options: ExecutionOptions,
        _signal: AbortSignal | undefined,
        _open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        return Promise.reject(new OmniVideoTerminalError('Gemini Omni video uses finite canonical typed streaming'));
    }

    requestTextCompletionStream(): Promise<DriverCompletionStream> {
        return Promise.reject(new OmniVideoTerminalError('Gemini Omni video does not support streaming'));
    }

    formatLlumiverseError(_driver: VertexAIDriver, error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        if (!(error instanceof OmniVideoTerminalError) && !(error instanceof OmniVideoInteractionError)) {
            if (!(error instanceof Error) || error.name !== 'AbortError') {
                // Let VertexAIDriver fall back to the shared HTTP/network classifier. In particular,
                // timeouts and all 5xx responses are retryable while ordinary 4xx responses are not.
                throw error;
            }
        }
        const retryable = error instanceof OmniVideoInteractionError ? error.retryable : false;
        const name = error instanceof Error ? error.name : 'GeminiOmniVideoError';
        return new LlumiverseError(
            error instanceof Error ? error.message : String(error),
            retryable,
            context,
            error,
            undefined,
            name,
        );
    }
}
