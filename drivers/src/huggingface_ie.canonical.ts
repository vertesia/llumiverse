import type { InferenceClient, TextGenerationOutput, TextGenerationStreamOutput } from '@huggingface/inference';
import {
    appendDecodedConversationResponseWithProcessing,
    createGeneratedAgentTurn,
    createStructuredOutputTransformationProof,
    createTextBlock,
    createUserTurn,
    type DecodedConversationResponse,
    deriveConversationId,
    fingerprintJson,
    isConversationDocumentFormat,
    type JsonObject,
    type NativeItemMapping,
    type NativeStreamPosition,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    createCanonicalExecutionResponse,
    type ExecutionOptions,
    FallbackCanonicalExecutionEventStream,
    normalizeCanonicalStructuredOutput,
    type TextFallbackOptions,
} from '@llumiverse/core';
import { canonicalNativeExecutionEventStream } from './conversation/canonical-execution-event-stream.js';
import {
    acceptedCanonicalResponse,
    appendCanonicalPrompt,
    assertAcceptedCanonicalRequest,
    type CanonicalPreparedState,
    canonicalResponseIdentities,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
    resolveConversationRuntime,
} from './conversation/canonical-runtime.js';
import { rejectDecodedStructuredOutput } from './conversation/structured-output.js';
import type { HuggingFaceIEDriver } from './huggingface_ie.js';

const HUGGING_FACE_TEXT_PROTOCOL = 'huggingface.text-generation';
const HUGGING_FACE_TEXT_ADAPTER_VERSION = '2026-09-30.canonical.1';

interface HuggingFaceTextGenerationRequest {
    inputs: string;
    parameters: {
        temperature?: number;
        max_new_tokens?: number;
        details: true;
        return_full_text: false;
    };
    [property: string]: unknown;
}

interface HuggingFaceCanonicalNativeResponse {
    generated_text: string;
    finish_reason?: string;
    generated_tokens?: number;
    input_tokens?: number;
    provider_response: unknown;
}

function payload(prompt: string, options: ExecutionOptions): HuggingFaceTextGenerationRequest {
    const modelOptions = options.model_options as TextFallbackOptions | undefined;
    if (modelOptions?._option_id !== undefined && modelOptions._option_id !== 'text-fallback') {
        throw new TypeError(`Hugging Face canonical execution does not support option set ${modelOptions._option_id}`);
    }
    if (options.tools?.length) throw new TypeError('Hugging Face text generation does not support tools');
    return {
        inputs: prompt,
        parameters: {
            temperature: modelOptions?.temperature,
            max_new_tokens: modelOptions?.max_tokens,
            details: true,
            return_full_text: false,
        },
    };
}

interface HuggingFaceCanonicalOutcome {
    finish_reason?: string;
    generation_status: 'completed' | 'cancelled' | 'failed';
    turn_status: 'completed' | 'interrupted' | 'failed';
    draft_outcome: 'completed' | 'interrupted' | 'failed';
    block_outcome: 'native_complete' | 'interrupted' | 'failed';
}

function canonicalOutcome(reason: string | undefined): HuggingFaceCanonicalOutcome {
    switch (reason) {
        case undefined:
        case '':
            return {
                generation_status: 'completed',
                turn_status: 'completed',
                draft_outcome: 'completed',
                block_outcome: 'native_complete',
            };
        case 'eos_token':
            return {
                finish_reason: 'stop',
                generation_status: 'completed',
                turn_status: 'completed',
                draft_outcome: 'completed',
                block_outcome: 'native_complete',
            };
        case 'stop_sequence':
            return {
                finish_reason: reason,
                generation_status: 'completed',
                turn_status: 'completed',
                draft_outcome: 'completed',
                block_outcome: 'native_complete',
            };
        case 'length':
            return {
                finish_reason: reason,
                generation_status: 'cancelled',
                turn_status: 'interrupted',
                draft_outcome: 'interrupted',
                block_outcome: 'interrupted',
            };
        default:
            throw new Error(`Hugging Face response has unsupported finish reason ${reason.slice(0, 80)}`);
    }
}

function usage(response: HuggingFaceCanonicalNativeResponse) {
    const input =
        Number.isSafeInteger(response.input_tokens) && (response.input_tokens ?? -1) >= 0
            ? response.input_tokens
            : undefined;
    const output =
        Number.isSafeInteger(response.generated_tokens) && (response.generated_tokens ?? -1) >= 0
            ? response.generated_tokens
            : undefined;
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(input === undefined || output === undefined ? {} : { total_tokens: input + output }),
        accounting_provenance: {
            ...(input === undefined
                ? {}
                : { input_tokens: { method: 'reported' as const, accounting_basis: 'huggingface_tokens' } }),
            ...(output === undefined
                ? {}
                : { output_tokens: { method: 'reported' as const, accounting_basis: 'huggingface_tokens' } }),
            ...(input === undefined || output === undefined
                ? {}
                : { total_tokens: { method: 'derived' as const, accounting_basis: 'huggingface_tokens' } }),
        },
        reported_usage: [
            {
                source: 'provider' as const,
                protocol: HUGGING_FACE_TEXT_PROTOCOL,
                accounting_basis: 'huggingface_tokens',
                payload: providerJsonValue({
                    input_tokens: response.input_tokens,
                    generated_tokens: response.generated_tokens,
                }),
            },
        ],
    };
}

interface PreparedHuggingFaceCanonical extends CanonicalPreparedState<string> {
    request: HuggingFaceTextGenerationRequest;
    request_json: ReturnType<typeof providerJsonValue>;
    executor?: InferenceClient;
}

interface FinalizedHuggingFaceCanonical {
    raw_decoded: DecodedConversationResponse;
    decoded: DecodedConversationResponse;
    response: CanonicalExecutionResponse;
    normalized: ReturnType<typeof normalizeCanonicalStructuredOutput> | undefined;
}

async function prepareHuggingFaceCanonical(input: {
    driver: HuggingFaceIEDriver;
    prompt: string;
    options: ExecutionOptions;
}): Promise<PreparedHuggingFaceCanonical> {
    const runtime = resolveConversationRuntime(input.options);
    if (runtime.materialized_input !== undefined) {
        throw new TypeError('Hugging Face text generation does not support materialized canonical input');
    }
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation !== undefined &&
        input.options.conversation !== null &&
        !isConversationDocumentFormat(input.options.conversation)
    ) {
        throw new TypeError('Hugging Face canonical execution does not support legacy conversation input');
    }
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (document.turns.length > 0 && acceptedBefore === undefined) {
        throw new Error('Hugging Face text generation does not support conversation continuation');
    }
    const promptTurnId = await deriveConversationId('turn', runtime.input_operation_id, 'prompt', '0');
    const promptBlockId = await deriveConversationId('block', runtime.input_operation_id, 'prompt', '0');
    const promptTurn = createUserTurn({
        id: promptTurnId,
        authority: 'ordinary',
        blocks: [createTextBlock({ id: promptBlockId, text: input.prompt, format: 'plain' })],
        status: 'completed',
        timestamps: { recorded_at: runtime.recorded_at },
        model_visibility: 'include',
        provenance: { type: 'received' },
    });
    const contextId = await deriveConversationId('context', runtime.input_operation_id, '0');
    const mappings: NativeItemMapping[] = [
        { canonical_id: promptTurnId, native_id: 'input', kind: 'turn' },
        { canonical_id: promptBlockId, native_id: 'input/text', kind: 'block' },
    ];
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: [promptTurn],
            assets: [],
            context_entries: [{ id: contextId, type: 'source_turn', turn_id: promptTurnId }],
            item_mappings: mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        input.prompt,
    );
    document = appended.document;
    if (appended.tool_definitions.length > 0) {
        throw new TypeError('Hugging Face text generation does not support active tool definitions');
    }
    const request = payload(input.prompt, input.options);
    const requestJson = providerJsonValue(request);
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    let executor: InferenceClient | undefined;
    let targetOptions: JsonObject;
    if (accepted !== undefined) {
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.driver.provider, protocol: HUGGING_FACE_TEXT_PROTOCOL, model: input.options.model },
            requestJson,
        );
        const retainedOptions = accepted.generation.request_receipt.target.options;
        const retainedEndpoint = retainedOptions?.inference_endpoint;
        targetOptions = providerJsonValue({
            management_endpoint: input.driver.options.endpoint_url,
            inference_endpoint: retainedEndpoint,
            parameters: request.parameters,
        }) as JsonObject;
        if (
            accepted.generation.adapter_version !== HUGGING_FACE_TEXT_ADAPTER_VERSION ||
            accepted.generation.request_receipt.target.adapter_version !== HUGGING_FACE_TEXT_ADAPTER_VERSION ||
            typeof retainedEndpoint !== 'string' ||
            (await fingerprintJson(retainedOptions)) !== (await fingerprintJson(targetOptions))
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible Hugging Face target options`,
            );
        }
    } else {
        const target = await input.driver.getExecutorTarget(input.options.model);
        executor = target.executor;
        targetOptions = providerJsonValue({
            management_endpoint: input.driver.options.endpoint_url,
            inference_endpoint: target.url,
            parameters: request.parameters,
        }) as JsonObject;
    }
    const identities =
        accepted === undefined
            ? await canonicalResponseIdentities(runtime)
            : { generation_id: accepted.generation.id, response_turn_id: accepted.turn.id };
    const receipt =
        accepted?.generation.request_receipt ??
        (await createRequestReceipt(
            document,
            { ...runtime, conversation_id: document.id },
            {
                provider: input.driver.provider,
                protocol: HUGGING_FACE_TEXT_PROTOCOL,
                model: input.options.model,
                adapter_version: HUGGING_FACE_TEXT_ADAPTER_VERSION,
                options: targetOptions,
            },
            requestJson,
            mappings,
            appended.tool_definitions,
        ));
    return {
        document,
        native_conversation: input.prompt,
        receipt,
        runtime: { ...runtime, conversation_id: document.id },
        generation_id: identities.generation_id,
        response_turn_id: identities.response_turn_id,
        tool_definitions: appended.tool_definitions,
        request,
        request_json: requestJson,
        ...(executor === undefined ? {} : { executor }),
        ...(accepted === undefined ? {} : { accepted_response: accepted }),
    };
}

async function finalizeHuggingFaceCanonical(
    prepared: PreparedHuggingFaceCanonical,
    response: HuggingFaceCanonicalNativeResponse,
    options: ExecutionOptions,
): Promise<FinalizedHuggingFaceCanonical> {
    if (typeof response.generated_text !== 'string') {
        throw new Error('Hugging Face text generation response has no generated text');
    }
    const outcome = canonicalOutcome(response.finish_reason);
    const completedAt = prepared.runtime.completed_at ?? prepared.runtime.recorded_at;
    const generation = await createExecutedGeneration({
        id: prepared.generation_id,
        runtime: { ...prepared.runtime, completed_at: completedAt },
        receipt: prepared.receipt,
        provider: prepared.receipt.target.provider,
        protocol: HUGGING_FACE_TEXT_PROTOCOL,
        adapter_version: HUGGING_FACE_TEXT_ADAPTER_VERSION,
        requested_model: options.model,
        resolved_model: options.model,
        finish_reason: outcome.finish_reason,
        usage: usage(response),
    });
    generation.status = outcome.generation_status;
    const responseBlockId = await deriveConversationId('block', prepared.runtime.response_operation_id, '0');
    const normalized =
        options.result_schema === undefined
            ? undefined
            : normalizeCanonicalStructuredOutput(
                  { type: 'text', source_texts: [response.generated_text] },
                  options.result_schema,
              );
    const rawTurn = createGeneratedAgentTurn({
        id: prepared.response_turn_id,
        authority: 'ordinary',
        blocks: [createTextBlock({ id: responseBlockId, text: response.generated_text, format: 'plain' })],
        status: outcome.turn_status,
        timestamps: { recorded_at: completedAt, completed_at: completedAt },
        model_visibility: 'include',
        provenance: { type: 'generated' },
        generation_id: generation.id,
    });
    const rawDecoded: DecodedConversationResponse = {
        turns: [rawTurn],
        assets: [],
        generation,
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(providerJsonValue(response.provider_response)),
    };
    let decoded = rawDecoded;
    if (normalized?.status === 'valid') {
        const structuredBlockId = await deriveConversationId(
            'block',
            prepared.runtime.response_operation_id,
            'structured',
        );
        decoded = {
            ...rawDecoded,
            turns: [
                createGeneratedAgentTurn({
                    ...rawTurn,
                    blocks: [{ id: structuredBlockId, type: 'json', value: normalized.structured_output.value }],
                }),
            ],
        };
    } else if (normalized?.status === 'invalid') {
        decoded = rejectDecodedStructuredOutput(rawDecoded, normalized.error);
    }
    const final = (
        await appendDecodedConversationResponseWithProcessing(
            {
                document: prepared.document,
                generation_id: prepared.generation_id,
                response_turn_id: prepared.response_turn_id,
                receipt: prepared.receipt,
                payload: prepared.request_json,
                diagnostics: [],
            },
            decoded,
            { operation_id: prepared.runtime.response_operation_id, recorded_at: completedAt },
        )
    ).document;
    return {
        raw_decoded: rawDecoded,
        decoded,
        response: createCanonicalExecutionResponse(final, prepared.runtime.response_operation_id, {
            ...(options.include_original_response ? { original_response: response.provider_response } : {}),
        }),
        normalized,
    };
}

function assertRecoverableHuggingFaceResponse(prepared: PreparedHuggingFaceCanonical, options: ExecutionOptions): void {
    if (prepared.accepted_response !== undefined && options.include_original_response) {
        throw new Error('An idempotently recovered Hugging Face response cannot reconstruct original_response');
    }
}

function preparedExecutor(prepared: PreparedHuggingFaceCanonical): InferenceClient {
    if (prepared.executor === undefined) throw new Error('Fresh Hugging Face execution has no resolved endpoint');
    return prepared.executor;
}

export async function executeHuggingFaceCanonical(input: {
    driver: HuggingFaceIEDriver;
    prompt: string;
    options: ExecutionOptions;
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const prepared = await prepareHuggingFaceCanonical(input);
    assertRecoverableHuggingFaceResponse(prepared, input.options);
    if (prepared.accepted_response !== undefined) return recoverCanonicalExecutionResponse(prepared, input.options);
    await publishCanonicalPreparedRequest(prepared, input.options);
    input.signal?.throwIfAborted();
    const executor = preparedExecutor(prepared);
    const response = input.signal
        ? await executor.textGeneration(prepared.request, { signal: input.signal })
        : await executor.textGeneration(prepared.request);
    input.signal?.throwIfAborted();
    return (await finalizeHuggingFaceCanonical(prepared, canonicalSyncResponse(response), input.options)).response;
}

class HuggingFaceNativeStreamAccumulator {
    private terminal: HuggingFaceCanonicalNativeResponse | undefined;

    accept(event: TextGenerationStreamOutput): string {
        if (this.terminal !== undefined) throw new Error('Hugging Face stream emitted content after terminal details');
        if (
            typeof event.token !== 'object' ||
            event.token === null ||
            typeof event.token.text !== 'string' ||
            typeof event.token.special !== 'boolean'
        ) {
            throw new Error('Hugging Face stream event has an invalid token');
        }
        if (event.generated_text != null && event.details == null) {
            throw new Error('Hugging Face stream emitted full generated text without terminal details');
        }
        if (event.details != null) {
            if (typeof event.generated_text !== 'string') {
                throw new Error('Hugging Face terminal stream event has no full generated text');
            }
            canonicalOutcome(event.details.finish_reason);
            if (!Number.isSafeInteger(event.details.generated_tokens) || event.details.generated_tokens < 0) {
                throw new Error('Hugging Face stream has invalid generated token accounting');
            }
            const inputTokens = streamInputTokens(event.details);
            this.terminal = {
                generated_text: event.generated_text,
                finish_reason: event.details.finish_reason,
                generated_tokens: event.details.generated_tokens,
                ...(inputTokens === undefined ? {} : { input_tokens: inputTokens }),
                provider_response: structuredClone(event),
            };
        }
        return event.token.special ? '' : event.token.text;
    }

    response(): HuggingFaceCanonicalNativeResponse {
        if (this.terminal === undefined) throw new Error('Hugging Face stream ended without terminal details');
        return this.terminal;
    }
}

function streamInputTokens(details: object): number | undefined {
    // TGI's StreamDetails wire shape reports input_length, while the installed SDK declaration still lists prefill.
    const inputLength = Reflect.get(details, 'input_length');
    if (inputLength === undefined) return undefined;
    if (!Number.isSafeInteger(inputLength) || inputLength < 0) {
        throw new Error('Hugging Face stream has invalid prompt token accounting');
    }
    return inputLength;
}

function canonicalSyncResponse(response: TextGenerationOutput): HuggingFaceCanonicalNativeResponse {
    const prefill = response.details?.prefill;
    return {
        generated_text: response.generated_text,
        finish_reason: response.details?.finish_reason,
        generated_tokens: response.details?.generated_tokens,
        ...(prefill !== undefined && prefill.length > 0 ? { input_tokens: prefill.length } : {}),
        provider_response: response,
    };
}

function huggingFaceStreamPosition(): NativeStreamPosition {
    return { protocol: HUGGING_FACE_TEXT_PROTOCOL, path: ['token'] };
}

async function openHuggingFaceNativeStream(input: {
    executor: InferenceClient;
    request: HuggingFaceTextGenerationRequest;
    signal: AbortSignal;
}): Promise<AsyncIterable<TextGenerationStreamOutput>> {
    return input.executor.textGenerationStream(input.request, { signal: input.signal });
}

export async function streamHuggingFaceCanonicalEvents(input: {
    driver: HuggingFaceIEDriver;
    prompt: string;
    options: ExecutionOptions;
    signal: AbortSignal | undefined;
    open: CanonicalStreamOpenOptions;
}): Promise<CanonicalExecutionEventStream> {
    const prepared = await prepareHuggingFaceCanonical(input);
    assertRecoverableHuggingFaceResponse(prepared, input.options);
    const accepted = prepared.accepted_response;
    const identity = {
        request_id: accepted?.generation.request_id ?? prepared.runtime.request_id,
        attempt_id: accepted?.generation.attempt_id ?? prepared.runtime.attempt_id,
        response_operation_id: prepared.runtime.response_operation_id,
        generation_id: prepared.generation_id,
        draft_turn_id: prepared.response_turn_id,
    };
    if (accepted !== undefined) {
        return new FallbackCanonicalExecutionEventStream(
            identity,
            () => recoverCanonicalExecutionResponse(prepared, input.options),
            { ...input.open, origin: 'accepted_recovery' },
        );
    }

    const abortController = new AbortController();
    const forwardAbort = () => abortController.abort(input.signal?.reason);
    const accumulator = new HuggingFaceNativeStreamAccumulator();
    const position = huggingFaceStreamPosition();
    const draftBlockId = `${prepared.response_turn_id}:huggingface:text`;
    let draftStarted = false;
    const eventStream = canonicalNativeExecutionEventStream({
        identity,
        open: input.open,
        openSource: () =>
            openHuggingFaceNativeStream({
                executor: preparedExecutor(prepared),
                request: prepared.request,
                signal: abortController.signal,
            }),
        map: async (response, writer) => {
            const fragment = accumulator.accept(response);
            if (fragment.length === 0) return;
            if (!draftStarted) {
                draftStarted = true;
                await writer.startBlock({
                    draft_block_id: draftBlockId,
                    native_position: position,
                    block: { type: 'text' },
                });
            }
            await writer.text({ draft_block_id: draftBlockId, native_position: position, text: fragment });
        },
        finalize: async () => {
            const nativeResponse = accumulator.response();
            const finalized = await finalizeHuggingFaceCanonical(prepared, nativeResponse, input.options);
            return {
                decoded: finalized.decoded,
                response: finalized.response,
                prepare_reconciliation: async () => {
                    const rawBlock = finalized.raw_decoded.turns[0]?.blocks[0];
                    const committedBlock = finalized.decoded.turns[0]?.blocks[0];
                    if (rawBlock?.type !== 'text' || committedBlock === undefined) {
                        throw new Error('Hugging Face stream finalization has no semantic response block');
                    }
                    const transformations = [];
                    const reconciliations = [];
                    if (draftStarted && finalized.normalized?.status === 'valid') {
                        if (committedBlock.type !== 'json') {
                            throw new Error('Hugging Face structured stream finalization has no JSON result');
                        }
                        const proof = await createStructuredOutputTransformationProof({
                            id: `${prepared.generation_id}:structured-output`,
                            source_blocks: [rawBlock],
                            result_block: committedBlock,
                        });
                        transformations.push(proof);
                        reconciliations.push({
                            draft_block_ids: [draftBlockId],
                            native_positions: [position],
                            committed_block_ids: [committedBlock.id],
                            disposition: 'structured_output' as const,
                            transformation_id: proof.id,
                        });
                    } else if (draftStarted) {
                        reconciliations.push({
                            draft_block_ids: [draftBlockId],
                            native_positions: [position],
                            committed_block_ids: [committedBlock.id],
                            disposition: 'direct' as const,
                        });
                    }
                    const decoded: DecodedConversationResponse = {
                        ...finalized.decoded,
                        stream_evidence: {
                            item_mappings: draftStarted
                                ? [{ canonical_id: rawBlock.id, native_position: position, kind: 'block' as const }]
                                : [],
                            transformations,
                        },
                    };
                    const outcome = canonicalOutcome(nativeResponse.finish_reason);
                    return {
                        decoded,
                        reconciliations,
                        deliver_final_events: async (writer) => {
                            if (decoded.generation.usage !== undefined) await writer.usage(decoded.generation.usage);
                            if (draftStarted) {
                                await writer.finishBlock({
                                    draft_block_id: draftBlockId,
                                    native_position: position,
                                    outcome: outcome.block_outcome,
                                });
                            }
                            await writer.finish({
                                outcome: outcome.draft_outcome,
                                finish_reason: outcome.finish_reason,
                            });
                        },
                    };
                },
                ...(finalized.normalized?.status === 'valid' && input.options.result_schema !== undefined
                    ? { result_schema: input.options.result_schema }
                    : {}),
            };
        },
        abort: () => abortController.abort(),
        close: () => input.signal?.removeEventListener('abort', forwardAbort),
    });
    await publishCanonicalPreparedRequest(prepared, input.options);
    if (input.signal?.aborted) forwardAbort();
    else input.signal?.addEventListener('abort', forwardAbort, { once: true });
    return eventStream;
}
