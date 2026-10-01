import {
    appendDecodedConversationResponse,
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
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import { canonicalNativeExecutionEventStream } from '../conversation/canonical-execution-event-stream.js';
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
} from '../conversation/canonical-runtime.js';
import { rejectDecodedStructuredOutput } from '../conversation/structured-output.js';
import type { WatsonxDriver } from './index.js';
import type { WatsonxTextGenerationPayload, WatsonxTextGenerationResponse } from './interfaces.js';

const WATSONX_TEXT_PROTOCOL = 'watsonx.text-generation';
const WATSONX_TEXT_ADAPTER_VERSION = '2026-09-30.canonical.1';
const API_VERSION = '2024-03-14';

function payload(driver: WatsonxDriver, prompt: string, options: ExecutionOptions): WatsonxTextGenerationPayload {
    const modelOptions = options.model_options as TextFallbackOptions | undefined;
    if (modelOptions?._option_id !== undefined && modelOptions._option_id !== 'text-fallback') {
        throw new TypeError(`Watsonx canonical execution does not support option set ${modelOptions._option_id}`);
    }
    if (options.tools?.length) throw new TypeError('Watsonx text generation does not support tools');
    return {
        model_id: options.model,
        input: `${prompt}\n`,
        parameters: {
            max_new_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
            top_k: modelOptions?.top_k,
            top_p: modelOptions?.top_p,
            stop_sequences: modelOptions?.stop_sequence,
        },
        project_id: driver.projectId,
    };
}

interface WatsonxCanonicalOutcome {
    finish_reason: string;
    generation_status: 'completed' | 'cancelled' | 'failed';
    turn_status: 'completed' | 'interrupted' | 'failed';
    draft_outcome: 'completed' | 'interrupted' | 'failed';
    block_outcome: 'native_complete' | 'interrupted' | 'failed';
}

function canonicalOutcome(reason: string | undefined): WatsonxCanonicalOutcome {
    switch (reason) {
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
        case 'max_tokens':
        case 'token_limit':
        case 'time_limit':
        case 'canceled':
        case 'cancelled':
            return {
                finish_reason: reason === 'max_tokens' ? 'length' : reason,
                generation_status: 'cancelled',
                turn_status: 'interrupted',
                draft_outcome: 'interrupted',
                block_outcome: 'interrupted',
            };
        case 'error':
            return {
                finish_reason: reason,
                generation_status: 'failed',
                turn_status: 'failed',
                draft_outcome: 'failed',
                block_outcome: 'failed',
            };
        case 'not_finished':
            throw new Error('Watsonx response ended with a nonterminal stop reason');
        case undefined:
        case '':
            throw new Error('Watsonx response has no terminal stop reason');
        default:
            throw new Error(`Watsonx response has unsupported stop reason ${reason.slice(0, 80)}`);
    }
}

function usage(result: WatsonxTextGenerationResponse['results'][number]) {
    const input =
        Number.isSafeInteger(result.input_token_count) && result.input_token_count >= 0
            ? result.input_token_count
            : undefined;
    const output =
        Number.isSafeInteger(result.generated_token_count) && result.generated_token_count >= 0
            ? result.generated_token_count
            : undefined;
    return {
        ...(input === undefined ? {} : { input_tokens: input }),
        ...(output === undefined ? {} : { output_tokens: output }),
        ...(input === undefined || output === undefined ? {} : { total_tokens: input + output }),
        accounting_provenance: {
            ...(input === undefined
                ? {}
                : { input_tokens: { method: 'reported' as const, accounting_basis: 'watsonx_tokens' } }),
            ...(output === undefined
                ? {}
                : { output_tokens: { method: 'reported' as const, accounting_basis: 'watsonx_tokens' } }),
            ...(input === undefined || output === undefined
                ? {}
                : { total_tokens: { method: 'derived' as const, accounting_basis: 'watsonx_tokens' } }),
        },
        reported_usage: [
            {
                source: 'provider' as const,
                protocol: WATSONX_TEXT_PROTOCOL,
                accounting_basis: 'watsonx_tokens',
                payload: providerJsonValue({
                    input_token_count: result.input_token_count,
                    generated_token_count: result.generated_token_count,
                }),
            },
        ],
    };
}

interface PreparedWatsonxCanonical extends CanonicalPreparedState<string> {
    request: WatsonxTextGenerationPayload;
    request_json: ReturnType<typeof providerJsonValue>;
}

interface FinalizedWatsonxCanonical {
    raw_decoded: DecodedConversationResponse;
    decoded: DecodedConversationResponse;
    response: CanonicalExecutionResponse;
    normalized: ReturnType<typeof normalizeCanonicalStructuredOutput> | undefined;
}

async function prepareWatsonxCanonical(input: {
    driver: WatsonxDriver;
    prompt: string;
    options: ExecutionOptions;
}): Promise<PreparedWatsonxCanonical> {
    const runtime = resolveConversationRuntime(input.options);
    if (runtime.materialized_input !== undefined) {
        throw new TypeError('Watsonx text generation does not support materialized canonical input');
    }
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation !== undefined &&
        input.options.conversation !== null &&
        !isConversationDocumentFormat(input.options.conversation)
    ) {
        throw new TypeError('Watsonx canonical execution does not support legacy conversation input');
    }
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (document.turns.length > 0 && acceptedBefore === undefined) {
        throw new Error('Watsonx text generation does not support conversation continuation');
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
    const request = payload(input.driver, input.prompt, input.options);
    const requestJson = providerJsonValue(request);
    const targetOptions = providerJsonValue({
        endpoint: input.driver.endpoint_url,
        parameters: request.parameters,
    }) as JsonObject;
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (accepted !== undefined) {
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.driver.provider, protocol: WATSONX_TEXT_PROTOCOL, model: input.options.model },
            requestJson,
        );
        if (
            accepted.generation.adapter_version !== WATSONX_TEXT_ADAPTER_VERSION ||
            accepted.generation.request_receipt.target.adapter_version !== WATSONX_TEXT_ADAPTER_VERSION ||
            accepted.generation.request_receipt.target.options === undefined ||
            (await fingerprintJson(accepted.generation.request_receipt.target.options)) !==
                (await fingerprintJson(targetOptions))
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible Watsonx target options`,
            );
        }
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
                protocol: WATSONX_TEXT_PROTOCOL,
                model: input.options.model,
                adapter_version: WATSONX_TEXT_ADAPTER_VERSION,
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
        ...(accepted === undefined ? {} : { accepted_response: accepted }),
    };
}

async function finalizeWatsonxCanonical(
    prepared: PreparedWatsonxCanonical,
    response: WatsonxTextGenerationResponse,
    options: ExecutionOptions,
): Promise<FinalizedWatsonxCanonical> {
    const result = response.results[0];
    if (result === undefined || typeof result.generated_text !== 'string') {
        throw new Error('Watsonx text generation response has no first result');
    }
    const outcome = canonicalOutcome(result.stop_reason);
    const completedAt = prepared.runtime.completed_at ?? prepared.runtime.recorded_at;
    const generation = await createExecutedGeneration({
        id: prepared.generation_id,
        runtime: { ...prepared.runtime, completed_at: completedAt },
        receipt: prepared.receipt,
        provider: prepared.receipt.target.provider,
        protocol: WATSONX_TEXT_PROTOCOL,
        adapter_version: WATSONX_TEXT_ADAPTER_VERSION,
        requested_model: options.model,
        resolved_model: response.model_id,
        finish_reason: outcome.finish_reason,
        usage: usage(result),
    });
    generation.status = outcome.generation_status;
    const responseBlockId = await deriveConversationId('block', prepared.runtime.response_operation_id, '0');
    const normalized =
        options.result_schema === undefined
            ? undefined
            : normalizeCanonicalStructuredOutput(
                  { type: 'text', source_texts: [result.generated_text] },
                  options.result_schema,
              );
    const rawTurn = createGeneratedAgentTurn({
        id: prepared.response_turn_id,
        authority: 'ordinary',
        blocks: [createTextBlock({ id: responseBlockId, text: result.generated_text, format: 'plain' })],
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
        payload_fingerprint: await fingerprintJson(providerJsonValue(response)),
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
    const final = appendDecodedConversationResponse(
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
    ).document;
    return {
        raw_decoded: rawDecoded,
        decoded,
        response: createCanonicalExecutionResponse(final, prepared.runtime.response_operation_id, {
            ...(options.include_original_response ? { original_response: response } : {}),
        }),
        normalized,
    };
}

function assertRecoverableWatsonxResponse(prepared: PreparedWatsonxCanonical, options: ExecutionOptions): void {
    if (prepared.accepted_response !== undefined && options.include_original_response) {
        throw new Error('An idempotently recovered Watsonx response cannot reconstruct original_response');
    }
}

export async function executeWatsonxCanonical(input: {
    driver: WatsonxDriver;
    prompt: string;
    options: ExecutionOptions;
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const prepared = await prepareWatsonxCanonical(input);
    assertRecoverableWatsonxResponse(prepared, input.options);
    if (prepared.accepted_response !== undefined) return recoverCanonicalExecutionResponse(prepared, input.options);
    await publishCanonicalPreparedRequest(prepared, input.options);
    input.signal?.throwIfAborted();
    const response = (await input.driver.fetchClient.post(`/ml/v1/text/generation?version=${API_VERSION}`, {
        payload: prepared.request,
        signal: input.signal,
    })) as WatsonxTextGenerationResponse;
    input.signal?.throwIfAborted();
    return (await finalizeWatsonxCanonical(prepared, response, input.options)).response;
}

class WatsonxNativeStreamAccumulator {
    private modelId: string | undefined;
    private latestCreatedAt: string | undefined;
    private generatedText = '';
    private inputTokenCount: number | undefined;
    private generatedTokenCount: number | undefined;
    private terminalResult: WatsonxTextGenerationResponse['results'][number] | undefined;

    accept(response: WatsonxTextGenerationResponse): string {
        if (this.terminalResult !== undefined) {
            throw new Error('Watsonx stream emitted content after its terminal result');
        }
        if (typeof response.model_id !== 'string' || response.model_id.length === 0) {
            throw new Error('Watsonx stream event has no model identity');
        }
        if (this.modelId !== undefined && this.modelId !== response.model_id) {
            throw new Error('Watsonx stream changed model identity');
        }
        this.modelId = response.model_id;
        if (typeof response.created_at !== 'string' || response.created_at.length === 0) {
            throw new Error('Watsonx stream event has no creation timestamp');
        }
        this.latestCreatedAt = response.created_at;
        const result = response.results[0];
        if (result === undefined || typeof result.generated_text !== 'string') {
            throw new Error('Watsonx stream event has no first result');
        }
        if (!Number.isSafeInteger(result.input_token_count) || result.input_token_count < 0) {
            throw new Error('Watsonx stream event has invalid input token accounting');
        }
        if (!Number.isSafeInteger(result.generated_token_count) || result.generated_token_count < 0) {
            throw new Error('Watsonx stream event has invalid generated token accounting');
        }
        if (this.generatedTokenCount !== undefined && result.generated_token_count < this.generatedTokenCount) {
            throw new Error('Watsonx stream generated token count is not cumulative');
        }
        this.inputTokenCount = Math.max(this.inputTokenCount ?? 0, result.input_token_count);
        this.generatedTokenCount = result.generated_token_count;
        this.generatedText += result.generated_text;
        if (result.stop_reason !== undefined && result.stop_reason !== '' && result.stop_reason !== 'not_finished') {
            this.terminalResult = structuredClone(result);
        }
        return result.generated_text;
    }

    response(): WatsonxTextGenerationResponse {
        const terminal = this.terminalResult;
        if (terminal === undefined) throw new Error('Watsonx stream ended without a terminal stop reason');
        if (
            this.modelId === undefined ||
            this.latestCreatedAt === undefined ||
            this.inputTokenCount === undefined ||
            this.generatedTokenCount === undefined
        ) {
            throw new Error('Watsonx stream ended without complete response identity');
        }
        return {
            model_id: this.modelId,
            // Watsonx timestamps each event independently. The synthesized response represents the terminal observation.
            created_at: this.latestCreatedAt,
            results: [
                {
                    ...terminal,
                    generated_text: this.generatedText,
                    input_token_count: this.inputTokenCount,
                    generated_token_count: this.generatedTokenCount,
                },
            ],
        };
    }
}

async function* watsonxNativeSSE(
    stream: ReadableStream<ServerSentEvent>,
): AsyncIterable<WatsonxTextGenerationResponse> {
    for await (const event of stream as ReadableStream<ServerSentEvent> & AsyncIterable<ServerSentEvent>) {
        if (event.type === 'reconnect-interval') continue;
        if (event.event !== undefined && event.event !== '' && event.event !== 'message') {
            const eventName = event.event.slice(0, 80);
            throw new Error(
                eventName === 'error'
                    ? 'Watsonx stream reported a provider error event'
                    : `Watsonx stream reported unsupported SSE event ${eventName}`,
            );
        }
        if (event.data === '' || event.data === '[DONE]') continue;
        const parsed = JSON.parse(event.data) as unknown;
        if (
            typeof parsed === 'object' &&
            parsed !== null &&
            (Object.hasOwn(parsed, 'error') || Object.hasOwn(parsed, 'errors'))
        ) {
            throw new Error('Watsonx stream reported a provider error payload');
        }
        yield parsed as WatsonxTextGenerationResponse;
    }
}

function watsonxStreamPosition(): NativeStreamPosition {
    return { protocol: WATSONX_TEXT_PROTOCOL, path: ['results', 0, 'generated_text'] };
}

async function openWatsonxNativeStream(input: {
    driver: WatsonxDriver;
    request: WatsonxTextGenerationPayload;
    signal: AbortSignal;
}): Promise<AsyncIterable<WatsonxTextGenerationResponse>> {
    const stream = (await input.driver.fetchClient.post(`/ml/v1/text/generation_stream?version=${API_VERSION}`, {
        payload: input.request,
        reader: 'sse',
        signal: input.signal,
    })) as ReadableStream<ServerSentEvent>;
    return watsonxNativeSSE(stream);
}

export async function streamWatsonxCanonicalEvents(input: {
    driver: WatsonxDriver;
    prompt: string;
    options: ExecutionOptions;
    signal: AbortSignal | undefined;
    open: CanonicalStreamOpenOptions;
}): Promise<CanonicalExecutionEventStream> {
    const prepared = await prepareWatsonxCanonical(input);
    assertRecoverableWatsonxResponse(prepared, input.options);
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
    const accumulator = new WatsonxNativeStreamAccumulator();
    const position = watsonxStreamPosition();
    const draftBlockId = `${prepared.response_turn_id}:watsonx:text`;
    let draftStarted = false;
    const eventStream = canonicalNativeExecutionEventStream({
        identity,
        open: input.open,
        openSource: () =>
            openWatsonxNativeStream({
                driver: input.driver,
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
            const finalized = await finalizeWatsonxCanonical(prepared, nativeResponse, input.options);
            return {
                decoded: finalized.decoded,
                response: finalized.response,
                prepare_reconciliation: async () => {
                    const rawBlock = finalized.raw_decoded.turns[0]?.blocks[0];
                    const committedBlock = finalized.decoded.turns[0]?.blocks[0];
                    if (rawBlock?.type !== 'text' || committedBlock === undefined) {
                        throw new Error('Watsonx stream finalization has no semantic response block');
                    }
                    const transformations = [];
                    const reconciliations = [];
                    if (draftStarted && finalized.normalized?.status === 'valid') {
                        if (committedBlock.type !== 'json') {
                            throw new Error('Watsonx structured stream finalization has no JSON result');
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
                    const result = nativeResponse.results[0];
                    if (result === undefined) throw new Error('Watsonx stream has no terminal result');
                    const outcome = canonicalOutcome(result.stop_reason);
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
