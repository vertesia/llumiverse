// Import the helper module for converting arbitrary protobuf.Value objects

import type { MaskReferenceConfig } from '@google/genai';
import { helpers, type protos } from '@google-cloud/aiplatform';
import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponse,
    createTextBlock,
    createUserTurn,
    deriveConversationId,
    fingerprintJson,
    isConversationDocumentFormat,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    parseConversationDocument,
    type UserContentBlock,
} from '@llumiverse/conversation';
import {
    type AIModel,
    type CanonicalExecutionResponse,
    type Completion,
    createCanonicalExecutionResponse,
    type ExecutionOptions,
    type ImagenOptions,
    ModelType,
    PromptRole,
    type PromptSegment,
    readStreamAsBase64,
} from '@llumiverse/core';
import { resolveDriverRequestTimeoutMs } from '@llumiverse/core/http-agent';
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
} from '../../conversation/canonical-runtime.js';
import { truncateBinaryForDebug } from '../../shared/debug-prompt.js';
import {
    generatedImageStorage,
    maximumGeneratedImageOutputBytes,
    type VerifiedGeneratedImage,
    verifiedBase64GeneratedImage,
} from '../../shared/generated-image.js';
import type { VertexAIDriver } from '../index.js';

const IMAGEN_PROTOCOL = 'google.vertex.imagen.predict';
const IMAGEN_ADAPTER_VERSION = '2026-09-30.canonical.1';
const IMAGEN_LOCATION = 'us-central1';
const MAX_IMAGEN_REFERENCE_BYTES = 10_000_000;

interface ImagenBaseReference {
    referenceType:
        | 'REFERENCE_TYPE_RAW'
        | 'REFERENCE_TYPE_MASK'
        | 'REFERENCE_TYPE_SUBJECT'
        | 'REFERENCE_TYPE_CONTROL'
        | 'REFERENCE_TYPE_STYLE';
    referenceId: number;
    referenceImage: {
        bytesBase64Encoded: string; //10MB max
    };
}

export enum ImagenTaskType {
    TEXT_IMAGE = 'TEXT_IMAGE',
    EDIT_MODE_INPAINT_REMOVAL = 'EDIT_MODE_INPAINT_REMOVAL',
    EDIT_MODE_INPAINT_INSERTION = 'EDIT_MODE_INPAINT_INSERTION',
    EDIT_MODE_BGSWAP = 'EDIT_MODE_BGSWAP',
    EDIT_MODE_OUTPAINT = 'EDIT_MODE_OUTPAINT',
    CUSTOMIZATION_SUBJECT = 'CUSTOMIZATION_SUBJECT',
    CUSTOMIZATION_STYLE = 'CUSTOMIZATION_STYLE',
    CUSTOMIZATION_CONTROLLED = 'CUSTOMIZATION_CONTROLLED',
    CUSTOMIZATION_INSTRUCT = 'CUSTOMIZATION_INSTRUCT',
}

export enum ImagenMaskMode {
    MASK_MODE_USER_PROVIDED = 'MASK_MODE_USER_PROVIDED',
    MASK_MODE_BACKGROUND = 'MASK_MODE_BACKGROUND',
    MASK_MODE_FOREGROUND = 'MASK_MODE_FOREGROUND',
    MASK_MODE_SEMANTIC = 'MASK_MODE_SEMANTIC',
}

interface ImagenReferenceRaw extends ImagenBaseReference {
    referenceType: 'REFERENCE_TYPE_RAW';
}

interface ImagenReferenceMask extends Omit<ImagenBaseReference, 'referenceImage'> {
    referenceType: 'REFERENCE_TYPE_MASK';
    maskImageConfig: {
        maskMode?: ImagenMaskMode;
        maskClasses?: MaskReferenceConfig['segmentationClasses'];
        dilation?: number; //Recommendation depends on mode: Inpaint: 0.01, BGSwap: 0.0, Outpaint: 0.01-0.03
    };
    referenceImage?: {
        //Only used for MASK_MODE_USER_PROVIDED
        bytesBase64Encoded: string; //10MB max
    };
}

interface ImagenReferenceSubject extends ImagenBaseReference {
    referenceType: 'REFERENCE_TYPE_SUBJECT';
    subjectImageConfig: {
        subjectDescription: string;
        subjectType: 'SUBJECT_TYPE_PERSON' | 'SUBJECT_TYPE_ANIMAL' | 'SUBJECT_TYPE_PRODUCT' | 'SUBJECT_TYPE_DEFAULT';
    };
}

interface ImagenReferenceControl extends ImagenBaseReference {
    referenceType: 'REFERENCE_TYPE_CONTROL';
    controlImageConfig: {
        controlType: 'CONTROL_TYPE_FACE_MESH' | 'CONTROL_TYPE_CANNY' | 'CONTROL_TYPE_SCRIBBLE';
        enableControlImageComputation?: boolean; //If true, the model will compute the control image
    };
}

interface ImagenReferenceStyle extends ImagenBaseReference {
    referenceType: 'REFERENCE_TYPE_STYLE';
    styleImageConfig: {
        styleDescription?: string;
    };
}

type ImagenMessage =
    | ImagenReferenceRaw
    | ImagenReferenceMask
    | ImagenReferenceSubject
    | ImagenReferenceControl
    | ImagenReferenceStyle;

export interface ImagenPrompt {
    prompt: string;
    referenceImages?: ImagenMessage[];
    subjectDescription?: string; //Used for image customization to describe in the reference image
    negativePrompt?: string; //Used for negative prompts
}

export function formatImagenDebugPrompt(prompt: ImagenPrompt): ImagenPrompt {
    return {
        ...prompt,
        referenceImages: prompt.referenceImages?.map((reference) => {
            if (!reference.referenceImage?.bytesBase64Encoded) {
                return reference;
            }
            return {
                ...reference,
                referenceImage: {
                    ...reference.referenceImage,
                    bytesBase64Encoded: truncateBinaryForDebug(reference.referenceImage.bytesBase64Encoded),
                },
            };
        }),
    };
}

function getImagenParameters(taskType: string, options: ImagenOptions) {
    const commonParameters = {
        sampleCount: options?.number_of_images,
        seed: options?.seed,
        safetySetting: options?.safety_setting,
        personGeneration: options?.person_generation === 'allow_adults' ? 'allow_adult' : options?.person_generation,
        guidanceScale: options?.guidance_scale,
        outputOptions:
            options?.image_file_type !== undefined || options?.jpeg_compression_quality !== undefined
                ? {
                      ...(options.image_file_type !== undefined ? { mimeType: options.image_file_type } : {}),
                      ...(options.jpeg_compression_quality !== undefined
                          ? { compressionQuality: options.jpeg_compression_quality }
                          : {}),
                  }
                : undefined,
        negativePrompt: taskType ? undefined : '', //Filled in later from the prompt
        //TODO: Add more safety and prompt rejection information
        //includeSafetyAttributes: true,
        //includeRaiReason: true,
    };
    switch (taskType) {
        case ImagenTaskType.EDIT_MODE_INPAINT_REMOVAL:
        case ImagenTaskType.EDIT_MODE_INPAINT_INSERTION:
        case ImagenTaskType.EDIT_MODE_BGSWAP:
        case ImagenTaskType.EDIT_MODE_OUTPAINT:
            return {
                ...commonParameters,
                editMode: taskType,
                editConfig: options?.edit_steps !== undefined ? { baseSteps: options.edit_steps } : undefined,
            };
        case ImagenTaskType.TEXT_IMAGE:
            return {
                ...commonParameters,
                // You can't use a seed value and watermark at the same time.
                addWatermark: options?.add_watermark,
                aspectRatio: options?.aspect_ratio,
                enhancePrompt: options?.enhance_prompt,
            };
        case ImagenTaskType.CUSTOMIZATION_SUBJECT:
        case ImagenTaskType.CUSTOMIZATION_CONTROLLED:
        case ImagenTaskType.CUSTOMIZATION_INSTRUCT:
        case ImagenTaskType.CUSTOMIZATION_STYLE:
            return {
                ...commonParameters,
            };
        default:
            throw new Error('Task type not supported');
    }
}

interface PreparedImagenRequest {
    request: protos.google.cloud.aiplatform.v1.IPredictRequest;
    request_json: JsonValue;
    target_options: JsonObject;
    model_name: string;
    task_type: string;
}

function prepareImagenRequest(
    driver: VertexAIDriver,
    prompt: ImagenPrompt,
    options: ExecutionOptions,
): PreparedImagenRequest {
    const imagenOptions = options.model_options as ImagenOptions | undefined;
    const taskType = imagenOptions?.edit_mode ?? ImagenTaskType.TEXT_IMAGE;
    const modelName = options.model.split('/').pop() ?? '';
    if (modelName.length === 0) throw new Error('Imagen model name is empty');
    const endpoint = `projects/${driver.options.project}/locations/${IMAGEN_LOCATION}/publishers/google/models/${modelName}`;
    const instanceValue = helpers.toValue(prompt);
    if (!instanceValue) throw new Error('No Imagen instance value found');
    let parameter = getImagenParameters(taskType, imagenOptions ?? { _option_id: 'vertexai-imagen' }) as ReturnType<
        typeof getImagenParameters
    > & { negativePrompt?: string };
    parameter.negativePrompt = prompt.negativePrompt ?? undefined;
    parameter = Object.fromEntries(
        Object.entries(parameter).filter(([, value]) => value !== undefined),
    ) as typeof parameter;
    const parameters = helpers.toValue(parameter);
    const request: protos.google.cloud.aiplatform.v1.IPredictRequest = {
        endpoint,
        instances: [instanceValue],
        parameters,
    };
    const { negativePrompt: _negativePrompt, ...effectiveParameters } = parameter;
    const referenceConfiguration = (prompt.referenceImages ?? []).map((reference) => {
        const { referenceImage: _referenceImage, ...configuration } = reference;
        return configuration;
    });
    return {
        request,
        request_json: providerJsonValue({ endpoint, instances: [prompt], parameters: parameter }),
        target_options: providerJsonValue({
            endpoint,
            location: IMAGEN_LOCATION,
            task_type: taskType,
            parameters: effectiveParameters,
            references: referenceConfiguration,
        }) as JsonObject,
        model_name: modelName,
        task_type: taskType,
    };
}

async function predictImagen(
    driver: VertexAIDriver,
    request: protos.google.cloud.aiplatform.v1.IPredictRequest,
    options: ExecutionOptions,
    signal?: AbortSignal,
): Promise<protos.google.cloud.aiplatform.v1.IPredictResponse> {
    const client = await driver.getImagenClient();
    signal?.throwIfAborted();
    const timeout = resolveDriverRequestTimeoutMs(driver.options.httpTimeout, options.httpTimeout);
    const prediction = client.predict(request, { timeout }) as Promise<
        [protos.google.cloud.aiplatform.v1.IPredictResponse, unknown, unknown]
    > & { cancel(): void };
    const cancel = () => {
        try {
            prediction.cancel();
        } catch {
            // The provider promise still owns transport cleanup. A synchronous cancel error must not escape EventTarget.
        }
    };
    if (signal?.aborted) cancel();
    else signal?.addEventListener('abort', cancel, { once: true });
    try {
        const [response] = await prediction;
        signal?.throwIfAborted();
        return response;
    } finally {
        signal?.removeEventListener('abort', cancel);
    }
}

function imagenPredictionValue(prediction: protos.google.protobuf.IValue): JsonObject {
    const value = providerJsonValue(helpers.fromValue(prediction as Parameters<typeof helpers.fromValue>[0]));
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new Error('Imagen response prediction is malformed');
    }
    return value;
}

function optionalImagenString(value: JsonObject, key: string): string | undefined {
    if (!Object.hasOwn(value, key)) return undefined;
    const item = value[key];
    if (typeof item !== 'string' || item.length === 0) {
        throw new Error(`Imagen response ${key} is malformed`);
    }
    return item;
}

async function decodeImagenPredictions(
    response: protos.google.cloud.aiplatform.v1.IPredictResponse,
    maximumBytes: number,
): Promise<Array<VerifiedGeneratedImage & { enhanced_prompt?: string }>> {
    if (!Array.isArray(response.predictions) || response.predictions.length === 0) {
        throw new Error('Imagen response contains no predictions');
    }
    const images: Array<VerifiedGeneratedImage & { enhanced_prompt?: string }> = [];
    let totalBytes = 0;
    for (const prediction of response.predictions) {
        const value = imagenPredictionValue(prediction);
        const filteredReason = optionalImagenString(value, 'raiFilteredReason');
        const encoded = optionalImagenString(value, 'bytesBase64Encoded');
        if (encoded === undefined) {
            if (filteredReason !== undefined) throw new Error(`Imagen response was filtered: ${filteredReason}`);
            throw new Error('Imagen response prediction contains no image bytes');
        }
        if (filteredReason !== undefined) {
            throw new Error(`Imagen response contains contradictory image bytes and filter reason: ${filteredReason}`);
        }
        if (totalBytes >= maximumBytes) {
            throw new Error(`Imagen generated images exceed the ${maximumBytes} byte total limit`);
        }
        const declaredMimeType = optionalImagenString(value, 'mimeType');
        const image = await verifiedBase64GeneratedImage(
            encoded,
            maximumBytes - totalBytes,
            'Imagen',
            declaredMimeType,
        );
        totalBytes += image.bytes.byteLength;
        const enhancedPrompt = optionalImagenString(value, 'prompt');
        images.push({ ...image, ...(enhancedPrompt === undefined ? {} : { enhanced_prompt: enhancedPrompt }) });
    }
    return images;
}

export function validateImagenCanonicalInput(segments: PromptSegment[], options: ExecutionOptions): void {
    if (options.tools && options.tools.length > 0) throw new Error('Imagen does not support tools');
    if (options.result_schema !== undefined) throw new Error('Imagen does not support structured output');
    if (options.conversation_runtime?.materialized_input !== undefined) {
        throw new Error('Imagen does not support materialized conversation input');
    }
    if (segments.length === 0) throw new Error('Imagen requires prompt or reference-image input');
    const supportedRoles = new Set([
        PromptRole.user,
        PromptRole.system,
        PromptRole.safety,
        PromptRole.negative,
        PromptRole.mask,
    ]);
    for (const segment of segments) {
        if (!supportedRoles.has(segment.role)) throw new Error(`Imagen does not support ${segment.role} input`);
        for (const file of segment.files ?? []) {
            if (!file.mime_type?.startsWith('image/')) {
                throw new Error(`Imagen does not support ${file.mime_type ?? 'untyped'} input files`);
            }
        }
    }
}

async function imagenInputRecords(prompt: ImagenPrompt, runtime: ReturnType<typeof resolveConversationRuntime>) {
    const turnId = await deriveConversationId('turn', runtime.input_operation_id, 'imagen-prompt');
    const blocks: UserContentBlock[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    if (prompt.prompt.length > 0) {
        const block = createTextBlock({
            id: await deriveConversationId('block', runtime.input_operation_id, 'imagen-prompt'),
            text: prompt.prompt,
            format: 'plain',
        });
        blocks.push(block);
        mappings.push({ canonical_id: block.id, native_id: 'instances/0/prompt', kind: 'block' });
    }
    if (prompt.negativePrompt !== undefined) {
        const block = {
            id: await deriveConversationId('block', runtime.input_operation_id, 'imagen-negative-prompt'),
            type: 'extension' as const,
            namespace: 'vertexai.imagen.negative_prompt',
            version: '1',
            payload: { text: prompt.negativePrompt },
            model_projection: 'excluded' as const,
        };
        blocks.push(block);
        mappings.push({ canonical_id: block.id, native_id: 'parameters/negativePrompt', kind: 'block' });
    }
    for (let index = 0; index < (prompt.referenceImages?.length ?? 0); index += 1) {
        const reference = prompt.referenceImages?.[index];
        if (reference === undefined) continue;
        const { referenceImage: _referenceImage, ...configuration } = reference;
        const configurationBlock = {
            id: await deriveConversationId('block', runtime.input_operation_id, 'imagen-reference', String(index)),
            type: 'extension' as const,
            namespace: 'vertexai.imagen.reference',
            version: '1',
            payload: providerJsonValue(configuration),
            model_projection: 'excluded' as const,
        };
        blocks.push(configurationBlock);
        mappings.push({
            canonical_id: configurationBlock.id,
            native_id: `instances/0/referenceImages/${index}`,
            kind: 'block',
        });
        if (reference.referenceImage === undefined) continue;
        const verified = await verifiedBase64GeneratedImage(
            reference.referenceImage.bytesBase64Encoded,
            MAX_IMAGEN_REFERENCE_BYTES,
            'Imagen reference',
        );
        const assetId = await deriveConversationId(
            'asset',
            runtime.input_operation_id,
            'imagen-reference',
            String(index),
        );
        const imageBlock = {
            id: await deriveConversationId(
                'block',
                runtime.input_operation_id,
                'imagen-reference-image',
                String(index),
            ),
            type: 'image' as const,
            asset_id: assetId,
        };
        blocks.push(imageBlock);
        assets.push({
            id: assetId,
            kind: 'image',
            mime_type: verified.mime_type,
            storage: { type: 'inline_base64', data: reference.referenceImage.bytesBase64Encoded },
            provenance: { type: 'received', source_turn_id: turnId },
            byte_length: verified.bytes.byteLength,
            content_hash: verified.content_hash,
            created_at: runtime.recorded_at,
            metadata: {
                vertexai_imagen: {
                    reference_type: reference.referenceType,
                    reference_id: reference.referenceId,
                },
            },
        });
        mappings.push({
            canonical_id: imageBlock.id,
            native_id: `instances/0/referenceImages/${index}/referenceImage`,
            kind: 'block',
        });
    }
    if (blocks.length === 0) throw new Error('Imagen requires prompt or reference-image input');
    const turn = createUserTurn({
        id: turnId,
        authority: 'ordinary',
        model_visibility: 'include',
        status: 'completed',
        timestamps: { recorded_at: runtime.recorded_at },
        provenance: { type: 'received' },
        blocks,
    });
    mappings.unshift({ canonical_id: turn.id, native_id: 'instances/0', kind: 'turn' });
    return { turn, assets, mappings };
}

export async function executeImagenCanonical(input: {
    driver: VertexAIDriver;
    prompt: ImagenPrompt;
    options: ExecutionOptions;
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const preparedRequest = prepareImagenRequest(input.driver, input.prompt, input.options);
    const runtime = resolveConversationRuntime(input.options);
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation !== undefined &&
        input.options.conversation !== null &&
        !isConversationDocumentFormat(input.options.conversation)
    ) {
        throw new TypeError('Imagen does not support legacy conversation input');
    }
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    const inputRecords = await imagenInputRecords(input.prompt, runtime);
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        (Object.keys(document.generations).length > 0 ||
            document.turns.some((turn) => turn.id !== inputRecords.turn.id))
    ) {
        throw new Error('Imagen does not support conversation continuation');
    }
    const contextEntryId = await deriveConversationId('context', runtime.input_operation_id, 'imagen-prompt');
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
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.driver.provider, protocol: IMAGEN_PROTOCOL, model: input.options.model },
            preparedRequest.request_json,
        );
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered Imagen response cannot reconstruct original_response');
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
            provider: input.driver.provider,
            protocol: IMAGEN_PROTOCOL,
            model: input.options.model,
            adapter_version: IMAGEN_ADAPTER_VERSION,
            options: preparedRequest.target_options,
        },
        preparedRequest.request_json,
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
    const nativeResponse = await predictImagen(input.driver, preparedRequest.request, input.options, input.signal);
    const images = await decodeImagenPredictions(nativeResponse, maximumBytes);
    const completedAt = runtime.completed_at ?? runtime.recorded_at;
    const completedRuntime = { ...runtime, completed_at: completedAt };
    const assets: Asset[] = [];
    const blocks: AgentContentBlock[] = [];
    for (let index = 0; index < images.length; index += 1) {
        const image = images[index];
        const assetId = await deriveConversationId('asset', runtime.response_operation_id, String(index));
        assets.push({
            id: assetId,
            kind: 'image',
            mime_type: image.mime_type,
            storage: await generatedImageStorage(image, input.options, 'Imagen', input.signal),
            provenance: {
                type: 'generated',
                generation_id: identities.generation_id,
                source_turn_id: identities.response_turn_id,
            },
            byte_length: image.bytes.byteLength,
            content_hash: image.content_hash,
            created_at: completedAt,
            ...(image.enhanced_prompt === undefined
                ? {}
                : { metadata: { vertexai_imagen: { enhanced_prompt: image.enhanced_prompt } } }),
        });
        blocks.push({
            id: await deriveConversationId('block', runtime.response_operation_id, String(index)),
            type: 'image',
            asset_id: assetId,
            ...(image.enhanced_prompt === undefined ? {} : { caption: image.enhanced_prompt }),
        });
    }
    const generation = await createExecutedGeneration({
        id: identities.generation_id,
        runtime: completedRuntime,
        receipt,
        provider: input.driver.provider,
        protocol: IMAGEN_PROTOCOL,
        adapter_version: IMAGEN_ADAPTER_VERSION,
        requested_model: input.options.model,
        resolved_model: preparedRequest.model_name,
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
    const responseEvidence: JsonValue = {
        ...(nativeResponse.deployedModelId === undefined ? {} : { deployed_model_id: nativeResponse.deployedModelId }),
        ...(nativeResponse.model === undefined ? {} : { model: nativeResponse.model }),
        ...(nativeResponse.modelVersionId === undefined ? {} : { model_version_id: nativeResponse.modelVersionId }),
        predictions: images.map((image) => ({
            content_hash: image.content_hash,
            mime_type: image.mime_type,
            byte_length: image.bytes.byteLength,
            ...(image.enhanced_prompt === undefined ? {} : { enhanced_prompt: image.enhanced_prompt }),
        })),
    };
    const finalDocument = appendDecodedConversationResponse(
        {
            document,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            receipt,
            payload: preparedRequest.request_json,
            diagnostics: [],
        },
        {
            turns: [responseTurn],
            assets,
            generation,
            diagnostics: [],
            payload_fingerprint: await fingerprintJson(providerJsonValue(responseEvidence)),
        },
        { operation_id: runtime.response_operation_id, recorded_at: completedAt },
    ).document;
    return createCanonicalExecutionResponse(finalDocument, runtime.response_operation_id, {
        ...(input.options.include_original_response ? { original_response: nativeResponse } : {}),
    });
}

export class ImagenModelDefinition {
    model: AIModel;

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Image,
            can_stream: false,
        };
    }

    async createPrompt(
        _driver: VertexAIDriver,
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<ImagenPrompt> {
        const splits = options.model.split('/');
        const modelName = splits[splits.length - 1];
        options = { ...options, model: modelName };

        const prompt: ImagenPrompt = {
            prompt: '',
        };

        //Collect text prompts, Imagen does not support roles, so everything gets merged together
        // however we still respect our typical pattern. System First, Safety Last.
        const system: string[] = [];
        const user: string[] = [];
        const safety: string[] = [];
        const negative: string[] = [];

        const imagenOptions = options.model_options as ImagenOptions;
        const mask_mode = imagenOptions?.mask_mode ?? ImagenMaskMode.MASK_MODE_USER_PROVIDED;
        const maskImageConfig: ImagenReferenceMask['maskImageConfig'] = {
            maskMode: mask_mode,
            ...(mask_mode === ImagenMaskMode.MASK_MODE_SEMANTIC && imagenOptions?.mask_class !== undefined
                ? { maskClasses: imagenOptions.mask_class }
                : {}),
            ...(imagenOptions?.mask_dilation !== undefined ? { dilation: imagenOptions.mask_dilation } : {}),
        };

        for (const msg of segments) {
            if (msg.role === PromptRole.safety) {
                safety.push(msg.content);
            } else if (msg.role === PromptRole.system) {
                system.push(msg.content);
            } else if (msg.role === PromptRole.negative) {
                negative.push(msg.content);
            } else {
                //Everything else is assumed to be user or user adjacent.
                user.push(msg.content);
            }
            if (msg.files) {
                //Get images from messages
                if (!prompt.referenceImages) {
                    prompt.referenceImages = [];
                }

                //Always required, but only used by customisation.
                //Each ref ID refers to a single "reference", i.e. object. To provide multiple images of a single ref,
                //include multiple images in one prompt.
                const refId = prompt.referenceImages.length + 1;
                for (const img of msg.files) {
                    if (img.mime_type?.includes('image')) {
                        if (msg.role !== PromptRole.mask) {
                            //Editing based mode requires a reference image
                            if (imagenOptions?.edit_mode?.includes('EDIT_MODE')) {
                                prompt.referenceImages.push({
                                    referenceType: 'REFERENCE_TYPE_RAW',
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    },
                                });
                                //If mask is auto-generated, add a mask reference
                                if (mask_mode !== ImagenMaskMode.MASK_MODE_USER_PROVIDED) {
                                    prompt.referenceImages.push({
                                        referenceType: 'REFERENCE_TYPE_MASK',
                                        referenceId: refId,
                                        maskImageConfig,
                                    });
                                }
                            } else if (
                                (options.model_options as ImagenOptions)?.edit_mode ===
                                ImagenTaskType.CUSTOMIZATION_SUBJECT
                            ) {
                                //First image is always the control image
                                if (refId === 1) {
                                    //Customization subject mode requires a control image
                                    prompt.referenceImages.push({
                                        referenceType: 'REFERENCE_TYPE_CONTROL',
                                        referenceId: refId,
                                        referenceImage: {
                                            bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                        },
                                        controlImageConfig: {
                                            controlType: imagenOptions?.controlType ?? 'CONTROL_TYPE_CANNY',
                                            enableControlImageComputation: imagenOptions?.controlImageComputation,
                                        },
                                    });
                                } else {
                                    // Subject images
                                    prompt.referenceImages.push({
                                        referenceType: 'REFERENCE_TYPE_SUBJECT',
                                        referenceId: refId,
                                        referenceImage: {
                                            bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                        },
                                        subjectImageConfig: {
                                            subjectDescription: prompt.subjectDescription ?? msg.content,
                                            subjectType: imagenOptions?.subjectType ?? 'SUBJECT_TYPE_DEFAULT',
                                        },
                                    });
                                }
                            } else if (
                                (options.model_options as ImagenOptions)?.edit_mode ===
                                ImagenTaskType.CUSTOMIZATION_STYLE
                            ) {
                                // Style images
                                prompt.referenceImages.push({
                                    referenceType: 'REFERENCE_TYPE_STYLE',
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    },
                                    styleImageConfig: {
                                        styleDescription: prompt.subjectDescription ?? msg.content,
                                    },
                                });
                            } else if (
                                (options.model_options as ImagenOptions)?.edit_mode ===
                                ImagenTaskType.CUSTOMIZATION_CONTROLLED
                            ) {
                                // Control images
                                prompt.referenceImages.push({
                                    referenceType: 'REFERENCE_TYPE_CONTROL',
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    },
                                    controlImageConfig: {
                                        controlType: imagenOptions?.controlType ?? 'CONTROL_TYPE_CANNY',
                                        enableControlImageComputation: imagenOptions?.controlImageComputation,
                                    },
                                });
                            } else if (
                                (options.model_options as ImagenOptions)?.edit_mode ===
                                ImagenTaskType.CUSTOMIZATION_INSTRUCT
                            ) {
                                // Control images
                                prompt.referenceImages.push({
                                    referenceType: 'REFERENCE_TYPE_RAW',
                                    referenceId: refId,
                                    referenceImage: {
                                        bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                    },
                                });
                            }
                        }
                        //If mask is user-provided, add a mask reference
                        if (msg.role === PromptRole.mask && mask_mode === ImagenMaskMode.MASK_MODE_USER_PROVIDED) {
                            prompt.referenceImages.push({
                                referenceType: 'REFERENCE_TYPE_MASK',
                                referenceId: refId,
                                referenceImage: {
                                    bytesBase64Encoded: await readStreamAsBase64(await img.getStream()),
                                },
                                maskImageConfig,
                            });
                        }
                    }
                }
            }
        }

        //Extract the text from the segments
        prompt.prompt += [system.join('\n\n'), user.join('\n\n'), safety.join('\n\n')].join('\n\n');

        //Negative prompt
        if (negative.length > 0) {
            prompt.negativePrompt = negative.join(', ');
        }

        return prompt;
    }

    async requestImageGeneration(
        driver: VertexAIDriver,
        prompt: ImagenPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        if (
            options.model_options?._option_id !== undefined &&
            options.model_options?._option_id !== 'vertexai-imagen'
        ) {
            driver.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        const prepared = prepareImagenRequest(driver, prompt, options);
        driver.logger.info(`Task type: ${prepared.task_type}`);
        const response = await predictImagen(driver, prepared.request, options, signal);
        const predictions = response.predictions;

        if (!predictions) {
            throw new Error('No predictions found');
        }

        // Extract base64 encoded images from predictions
        const images: string[] = predictions.map(
            (prediction) => prediction.structValue?.fields?.bytesBase64Encoded?.stringValue ?? '',
        );

        return {
            result: images.map((image) => ({
                type: 'image' as const,
                value: image,
            })),
        };
    }
}
