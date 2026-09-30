import type { InvokeModelCommandOutput } from '@aws-sdk/client-bedrock-runtime';
import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponse,
    createTextBlock,
    createUserTurn,
    deriveConversationId,
    fingerprintJson,
    inlineAssetContentIntegrity,
    isConversationDocumentFormat,
    type JsonObject,
    type NativeItemMapping,
    parseConversationDocument,
    type UserContentBlock,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionResponse,
    createCanonicalExecutionResponse,
    type ExecutionOptions,
    type NovaCanvasOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/core';
import type { NovaMessagesPrompt } from '@llumiverse/core/formatters';
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
    resolveConversationRuntime,
} from '../conversation/canonical-runtime.js';
import {
    generatedImageStorage,
    maximumGeneratedImageOutputBytes,
    type VerifiedGeneratedImage,
    verifiedBase64GeneratedImage,
} from '../shared/generated-image.js';
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from './nova-image-payload.js';

const NOVA_CANVAS_PROTOCOL = 'aws.bedrock.invoke_model.nova_canvas';
const NOVA_CANVAS_ADAPTER_VERSION = '2026-09-30.canonical.1';
const MAX_NOVA_RESPONSE_OVERHEAD_BYTES = 1024 * 1024;
const SUPPORTED_INPUT_MIME_TYPES = new Set(['image/jpeg', 'image/png']);
const SUPPORTED_TASKS = new Set<string>(Object.values(NovaImageGenerationTaskType));

export type NovaCanvasPayload = Awaited<ReturnType<typeof formatNovaImageGenerationPayload>>;
type NovaCanvasTask = NonNullable<NovaCanvasOptions['taskType']>;

interface DecodedNovaCanvasResponse {
    images: VerifiedGeneratedImage[];
    evidence: JsonObject;
    original_response: JsonObject;
}

function novaTask(options: ExecutionOptions): NovaCanvasTask {
    const task =
        (options.model_options as NovaCanvasOptions | undefined)?.taskType ?? NovaImageGenerationTaskType.TEXT_IMAGE;
    if (!SUPPORTED_TASKS.has(task)) throw new Error(`Nova Canvas does not support task ${String(task)}`);
    return task;
}

function inputImageCount(segments: PromptSegment[]): number {
    return segments.reduce((count, segment) => count + (segment.files?.length ?? 0), 0);
}

function assertInputImageCount(task: NovaCanvasTask, count: number, controlMode?: string): void {
    switch (task) {
        case NovaImageGenerationTaskType.TEXT_IMAGE:
            if ((controlMode === undefined && count !== 0) || (controlMode !== undefined && count !== 1)) {
                throw new Error(
                    controlMode === undefined
                        ? 'Nova Canvas text-to-image does not accept input images without controlMode'
                        : 'Nova Canvas conditioned text-to-image requires exactly one input image',
                );
            }
            return;
        case NovaImageGenerationTaskType.TEXT_IMAGE_WITH_IMAGE_CONDITIONING:
            if (controlMode === undefined || count !== 1) {
                throw new Error('Nova Canvas image-conditioned text-to-image requires controlMode and one image');
            }
            return;
        case NovaImageGenerationTaskType.COLOR_GUIDED_GENERATION:
            if (count > 1 || (count === 1 && controlMode === undefined)) {
                throw new Error('Nova Canvas color-guided generation accepts at most one control image');
            }
            return;
        case NovaImageGenerationTaskType.IMAGE_VARIATION:
            if (count < 1 || count > 5) {
                throw new Error('Nova Canvas image variation requires from one through five input images');
            }
            return;
        case NovaImageGenerationTaskType.INPAINTING:
        case NovaImageGenerationTaskType.OUTPAINTING:
            if (count !== 2) throw new Error(`Nova Canvas ${task.toLowerCase()} requires exactly two input images`);
            return;
        case NovaImageGenerationTaskType.BACKGROUND_REMOVAL:
            if (count !== 1) throw new Error('Nova Canvas background removal requires exactly one input image');
            return;
    }
}

export function validateNovaCanvasCanonicalInput(segments: PromptSegment[], options: ExecutionOptions): void {
    if (options.tools && options.tools.length > 0) throw new Error('Nova Canvas does not support tools');
    if (isConversationDocumentFormat(options.conversation)) {
        const document = parseConversationDocument(options.conversation);
        if (document.context.active_tool_definition_ids.length > 0) {
            throw new Error('Nova Canvas does not support active canonical tool definitions');
        }
    }
    if (options.result_schema !== undefined) throw new Error('Nova Canvas does not support structured output');
    if (options.conversation_runtime?.materialized_input !== undefined) {
        throw new Error('Nova Canvas does not support materialized conversation input');
    }
    if (segments.length === 0) throw new Error('Nova Canvas requires prompt or image input');
    const supportedRoles = new Set([PromptRole.user, PromptRole.system, PromptRole.safety, PromptRole.negative]);
    for (const segment of segments) {
        if (!supportedRoles.has(segment.role)) throw new Error(`Nova Canvas does not support ${segment.role} input`);
        for (const file of segment.files ?? []) {
            if (segment.role !== PromptRole.user) {
                throw new Error(`Nova Canvas does not support image files on ${segment.role} input`);
            }
            if (!SUPPORTED_INPUT_MIME_TYPES.has(file.mime_type)) {
                throw new Error(`Nova Canvas does not support ${file.mime_type || 'untyped'} input files`);
            }
        }
    }
    const modelOptions = options.model_options as NovaCanvasOptions | undefined;
    if (modelOptions?._option_id !== undefined && modelOptions._option_id !== 'bedrock-nova-canvas') {
        throw new Error(`Nova Canvas does not support model options ${modelOptions._option_id}`);
    }
    const numberOfImages = modelOptions?.numberOfImages;
    if (
        numberOfImages !== undefined &&
        (!Number.isSafeInteger(numberOfImages) || numberOfImages < 1 || numberOfImages > 5)
    ) {
        throw new Error('Nova Canvas numberOfImages must be an integer from 1 through 5');
    }
    const task = novaTask(options);
    if (
        task === NovaImageGenerationTaskType.BACKGROUND_REMOVAL &&
        segments.some((segment) => segment.content.trim().length > 0)
    ) {
        throw new Error('Nova Canvas background removal does not accept text, system, safety, or negative prompts');
    }
    assertInputImageCount(task, inputImageCount(segments), modelOptions?.controlMode);
}

function novaTargetOptions(region: string, options: ExecutionOptions, prompt: NovaMessagesPrompt): JsonObject {
    const modelOptions = options.model_options as NovaCanvasOptions | undefined;
    const { _option_id: _optionId, taskType: _taskType, ...parameters } = modelOptions ?? {};
    const inputImages = prompt.messages.flatMap((message, messageIndex) =>
        message.content.flatMap((part, contentIndex) =>
            part.image === undefined
                ? []
                : [
                      {
                          path: `messages/${messageIndex}/content/${contentIndex}/image`,
                          role: message.role,
                          format: part.image.format,
                      },
                  ],
        ),
    );
    return providerJsonValue({
        region,
        task_type: novaTask(options),
        parameters,
        input_images: inputImages,
    }) as JsonObject;
}

async function novaInputRecords(input: {
    prompt: NovaMessagesPrompt;
    payload: NovaCanvasPayload;
    runtime: ReturnType<typeof resolveConversationRuntime>;
}) {
    const turnId = await deriveConversationId('turn', input.runtime.input_operation_id, 'nova-canvas-prompt');
    const blocks: UserContentBlock[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    const text = input.prompt.messages
        .flatMap((message) => message.content.flatMap((part) => (part.text === undefined ? [] : [part.text])))
        .filter(Boolean)
        .join('\n\n');
    if (text.length > 0) {
        const block = createTextBlock({
            id: await deriveConversationId('block', input.runtime.input_operation_id, 'nova-canvas-text'),
            text,
            format: 'plain',
        });
        blocks.push(block);
        mappings.push({ canonical_id: block.id, native_id: 'body/prompt', kind: 'block' });
    }
    if (input.prompt.system?.length) {
        const block = {
            id: await deriveConversationId('block', input.runtime.input_operation_id, 'nova-canvas-system'),
            type: 'extension' as const,
            namespace: 'aws.bedrock.nova_canvas.system',
            version: '1',
            payload: { text: input.prompt.system.map((item) => item.text).join('\n') },
            model_projection: 'excluded' as const,
        };
        blocks.push(block);
        mappings.push({ canonical_id: block.id, native_id: 'body/system', kind: 'block' });
    }
    if (input.prompt.negative) {
        const block = {
            id: await deriveConversationId('block', input.runtime.input_operation_id, 'nova-canvas-negative'),
            type: 'extension' as const,
            namespace: 'aws.bedrock.nova_canvas.negative_prompt',
            version: '1',
            payload: { text: input.prompt.negative },
            model_projection: 'excluded' as const,
        };
        blocks.push(block);
        mappings.push({ canonical_id: block.id, native_id: 'body/negative', kind: 'block' });
    }
    const configuration = {
        id: await deriveConversationId('block', input.runtime.input_operation_id, 'nova-canvas-configuration'),
        type: 'extension' as const,
        namespace: 'aws.bedrock.nova_canvas.configuration',
        version: '1',
        payload: providerJsonValue({ task_type: input.payload.taskType }),
        model_projection: 'excluded' as const,
    };
    blocks.push(configuration);
    mappings.push({ canonical_id: configuration.id, native_id: 'body/taskType', kind: 'block' });

    for (let messageIndex = 0; messageIndex < input.prompt.messages.length; messageIndex += 1) {
        const message = input.prompt.messages[messageIndex];
        for (let contentIndex = 0; contentIndex < message.content.length; contentIndex += 1) {
            const image = message.content[contentIndex]?.image;
            if (image === undefined) continue;
            const storage = { type: 'inline_base64' as const, data: image.source.bytes };
            const integrity = await inlineAssetContentIntegrity(storage);
            if (integrity === undefined) throw new Error('Nova Canvas input image integrity is unavailable');
            const assetId = await deriveConversationId(
                'asset',
                input.runtime.input_operation_id,
                String(messageIndex),
                String(contentIndex),
            );
            const block = {
                id: await deriveConversationId(
                    'block',
                    input.runtime.input_operation_id,
                    'nova-canvas-image',
                    String(messageIndex),
                    String(contentIndex),
                ),
                type: 'image' as const,
                asset_id: assetId,
            };
            blocks.push(block);
            assets.push({
                id: assetId,
                kind: 'image',
                mime_type: `image/${image.format}`,
                storage,
                provenance: { type: 'received', source_turn_id: turnId },
                byte_length: integrity.byte_length,
                content_hash: integrity.content_hash,
                created_at: input.runtime.recorded_at,
                metadata: { nova_canvas: { role: message.role, format: image.format } },
            });
            mappings.push({
                canonical_id: block.id,
                native_id: `body/messages/${messageIndex}/content/${contentIndex}/image`,
                kind: 'block',
            });
        }
    }
    const turn = createUserTurn({
        id: turnId,
        authority: 'ordinary',
        model_visibility: 'include',
        status: 'completed',
        timestamps: { recorded_at: input.runtime.recorded_at },
        provenance: { type: 'received' },
        blocks,
    });
    mappings.unshift({ canonical_id: turn.id, native_id: 'body', kind: 'turn' });
    return { turn, assets, mappings };
}

function parseNovaCanvasBody(response: InvokeModelCommandOutput, maximumBytes: number): JsonObject {
    if (response.body === undefined) throw new Error('Nova Canvas response has no body');
    const body = new Uint8Array(response.body);
    const maximumEncodedBytes = Math.ceil(maximumBytes / 3) * 4 + MAX_NOVA_RESPONSE_OVERHEAD_BYTES;
    if (body.byteLength === 0 || body.byteLength > maximumEncodedBytes) {
        throw new Error(`Nova Canvas response exceeds the ${maximumBytes} byte generated-image limit`);
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(body));
    } catch {
        throw new Error('Nova Canvas response contains malformed JSON');
    }
    const value = providerJsonValue(parsed);
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new Error('Nova Canvas response must be a JSON object');
    }
    return value;
}

async function decodeNovaCanvasResponse(
    response: InvokeModelCommandOutput,
    maximumBytes: number,
): Promise<DecodedNovaCanvasResponse> {
    const body = parseNovaCanvasBody(response, maximumBytes);
    if (Object.hasOwn(body, 'error') && body.error !== undefined && body.error !== null && body.error !== '') {
        const detail = typeof body.error === 'string' ? body.error.slice(0, 512) : 'provider returned an error';
        throw new Error(`Nova Canvas rejected the request: ${detail}`);
    }
    if (!Array.isArray(body.images) || body.images.length === 0) {
        throw new Error('Nova Canvas response contains no images');
    }
    const images: VerifiedGeneratedImage[] = [];
    let totalBytes = 0;
    for (const value of body.images) {
        if (typeof value !== 'string') throw new Error('Nova Canvas response contains a malformed image');
        if (totalBytes >= maximumBytes) {
            throw new Error(`Nova Canvas generated images exceed the ${maximumBytes} byte total limit`);
        }
        const image = await verifiedBase64GeneratedImage(value, maximumBytes - totalBytes, 'Nova Canvas');
        totalBytes += image.bytes.byteLength;
        images.push(image);
    }
    return {
        images,
        evidence: {
            ...((response.$metadata?.requestId ?? '') === ''
                ? {}
                : { provider_response_id: response.$metadata?.requestId }),
            images: images.map((image) => ({
                content_hash: image.content_hash,
                mime_type: image.mime_type,
                byte_length: image.bytes.byteLength,
            })),
        },
        original_response: body,
    };
}

export async function executeNovaCanvasCanonical(input: {
    provider: string;
    region: string;
    prompt: NovaMessagesPrompt;
    options: ExecutionOptions;
    signal?: AbortSignal;
    invoke(
        payload: NovaCanvasPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<InvokeModelCommandOutput>;
}): Promise<CanonicalExecutionResponse> {
    const taskType = novaTask(input.options);
    const payload = await formatNovaImageGenerationPayload(taskType, input.prompt, input.options);
    const payloadJson = providerJsonValue(payload);
    const runtime = resolveConversationRuntime(input.options);
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation !== undefined &&
        input.options.conversation !== null &&
        !isConversationDocumentFormat(input.options.conversation)
    ) {
        throw new TypeError('Nova Canvas does not support legacy conversation input');
    }
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    const inputRecords = await novaInputRecords({ prompt: input.prompt, payload, runtime });
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        (Object.keys(document.generations).length > 0 ||
            document.turns.some((turn) => turn.id !== inputRecords.turn.id))
    ) {
        throw new Error('Nova Canvas does not support conversation continuation');
    }
    const contextEntryId = await deriveConversationId('context', runtime.input_operation_id, 'nova-canvas-prompt');
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: [inputRecords.turn],
            assets: inputRecords.assets,
            context_entries: [{ id: contextEntryId, type: 'source_turn', turn_id: inputRecords.turn.id }],
            item_mappings: inputRecords.mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        { prompt: providerJsonValue(input.prompt) },
    );
    document = appended.document;
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (accepted !== undefined) {
        const targetOptions = novaTargetOptions(input.region, input.options, input.prompt);
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.provider, protocol: NOVA_CANVAS_PROTOCOL, model: input.options.model },
            payloadJson,
        );
        if (
            accepted.generation.request_receipt.target.options === undefined ||
            (await fingerprintJson(accepted.generation.request_receipt.target.options)) !==
                (await fingerprintJson(targetOptions))
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible Nova Canvas target options`,
            );
        }
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered Nova Canvas response cannot reconstruct original_response');
        }
        return createCanonicalExecutionResponse(
            document,
            runtime.response_operation_id,
            {},
            await input.options.load_recovered_canonical_output?.({
                conversation_id: document.id,
                response_operation_id: runtime.response_operation_id,
            }),
        );
    }
    const receipt = await createRequestReceipt(
        document,
        { ...runtime, conversation_id: document.id },
        {
            provider: input.provider,
            protocol: NOVA_CANVAS_PROTOCOL,
            model: input.options.model,
            adapter_version: NOVA_CANVAS_ADAPTER_VERSION,
            options: novaTargetOptions(input.region, input.options, input.prompt),
        },
        payloadJson,
        inputRecords.mappings,
        appended.tool_definitions,
    );
    const identities = await canonicalResponseIdentities(runtime);
    const maximumBytes = maximumGeneratedImageOutputBytes(document, input.options);
    await publishCanonicalPreparedRequest(
        {
            document,
            native_conversation: input.prompt,
            receipt,
            runtime: { ...runtime, conversation_id: document.id },
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            tool_definitions: appended.tool_definitions,
        },
        input.options,
    );
    input.signal?.throwIfAborted();
    const nativeResponse = await input.invoke(payload, input.options, input.signal);
    const decoded = await decodeNovaCanvasResponse(nativeResponse, maximumBytes);
    const completedAt = runtime.completed_at ?? runtime.recorded_at;
    const completedRuntime = { ...runtime, completed_at: completedAt };
    const assets: Asset[] = [];
    const blocks: AgentContentBlock[] = [];
    for (let index = 0; index < decoded.images.length; index += 1) {
        const image = decoded.images[index];
        const assetId = await deriveConversationId('asset', runtime.response_operation_id, String(index));
        assets.push({
            id: assetId,
            kind: 'image',
            mime_type: image.mime_type,
            storage: await generatedImageStorage(image, input.options, 'Nova Canvas', input.signal),
            provenance: {
                type: 'generated',
                generation_id: identities.generation_id,
                source_turn_id: identities.response_turn_id,
            },
            byte_length: image.bytes.byteLength,
            content_hash: image.content_hash,
            created_at: completedAt,
        });
        blocks.push({
            id: await deriveConversationId('block', runtime.response_operation_id, String(index)),
            type: 'image',
            asset_id: assetId,
        });
    }
    const generation = await createExecutedGeneration({
        id: identities.generation_id,
        runtime: completedRuntime,
        receipt,
        provider: input.provider,
        protocol: NOVA_CANVAS_PROTOCOL,
        adapter_version: NOVA_CANVAS_ADAPTER_VERSION,
        requested_model: input.options.model,
        resolved_model: input.options.model,
        provider_response_id: nativeResponse.$metadata?.requestId,
        finish_reason: 'stop',
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
    const finalDocument = appendDecodedConversationResponse(
        {
            document,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            receipt,
            payload: payloadJson,
            diagnostics: [],
        },
        {
            turns: [responseTurn],
            assets,
            generation,
            diagnostics: [],
            payload_fingerprint: await fingerprintJson(decoded.evidence),
        },
        { operation_id: runtime.response_operation_id, recorded_at: completedAt },
    ).document;
    return createCanonicalExecutionResponse(finalDocument, runtime.response_operation_id, {
        ...(input.options.include_original_response ? { original_response: decoded.original_response } : {}),
    });
}
