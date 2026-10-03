import { LlumiverseError } from '@llumiverse/common';
import {
    appendConversationRecords,
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    ConversationStreamAccumulator,
    type ConversationStreamEvent,
    createConversationDocument,
    createStructuredOutputTransformationProof,
    type DecodedConversationResponse,
    externalizeToolCallArguments,
    prepareToolArgumentExternalization,
    validateConversationDocument,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    CanonicalAcceptedOutputRecovered,
    type CanonicalExecutionResponse,
    canonicalExecutionAccounting,
    canonicalHostCallbackFailure,
    createCanonicalExecutionResponse,
    FallbackCanonicalExecutionStream,
    isCanonicalAcceptedRecovery,
    legacyCompletionFromAcceptedOutput,
    legacyCompletionFromCanonicalExecution,
    markCanonicalAcceptedRecovery,
    markCanonicalHostCallbackFailure,
    rethrowCanonicalHostCallbackFailure,
} from './CanonicalExecution.js';
import {
    CANONICAL_FORBIDDEN_TOOL_CALL,
    CANONICAL_REQUIRED_TOOL_CALL_MISSING,
    CanonicalToolSelectionViolationError,
} from './CanonicalSelection.js';
import {
    type CanonicalExecutionEventStream,
    CanonicalStreamEventChannel,
    type CanonicalStreamTerminalEvent,
    FallbackCanonicalExecutionEventStream,
    finalizeCanonicalExecutionStreamResponse,
    LegacyCanonicalExecutionEventProjection,
} from './CanonicalStreaming.js';
import { MalformedStreamingToolArgumentsError } from './CompletionStream.js';

const RECORDED_AT = '2026-09-30T00:00:00Z';

describe('canonical host callback provenance', () => {
    it('recovers the exact frozen host error through provider wrappers', () => {
        const hostError = Object.freeze(new Error('durability barrier rejected'));
        const marked = markCanonicalHostCallbackFailure(hostError);
        const providerWrapper = new LlumiverseError(
            '[provider] callback failed',
            false,
            { provider: 'provider', model: 'model', operation: 'execute' },
            marked,
            502,
        );

        expect(canonicalHostCallbackFailure(providerWrapper)?.failure).toBe(hostError);
        expect(() => rethrowCanonicalHostCallbackFailure(providerWrapper)).toThrow(hostError);
    });

    it('does not classify an arbitrary provider error as a host callback failure', () => {
        const providerError = new LlumiverseError(
            '[provider] unavailable',
            true,
            { provider: 'provider', model: 'model', operation: 'execute' },
            new Error('provider unavailable'),
            503,
        );

        expect(canonicalHostCallbackFailure(providerError)).toBeUndefined();
        expect(() => rethrowCanonicalHostCallbackFailure(providerError)).not.toThrow();
    });

    it('retains non-Error callback values without stringifying them', () => {
        const marked = markCanonicalHostCallbackFailure(false);
        expect(canonicalHostCallbackFailure(marked)?.failure).toBe(false);
        try {
            rethrowCanonicalHostCallbackFailure(marked);
            throw new Error('Expected callback failure');
        } catch (error: unknown) {
            expect(error).toBe(false);
        }
    });
});

function streamEvent(
    sequence: number,
    body:
        | { type: 'draft_started'; origin: 'live_transport' }
        | { type: 'usage_snapshot'; usage: { input_tokens: number } }
        | { type: 'stream_terminated'; outcome: 'cancelled' | 'failed' },
): ConversationStreamEvent {
    return {
        format: CONVERSATION_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        stream_id: 'stream-channel',
        event_id: `stream-channel#${sequence}`,
        sequence,
        request_id: 'request',
        attempt_id: 'attempt',
        response_operation_id: 'response-operation',
        generation_id: 'generation',
        draft_turn_id: 'agent-turn',
        ...body,
    } as ConversationStreamEvent;
}

function acceptedDocument(options: { media?: boolean } = {}) {
    const initial = createConversationDocument({ id: 'conversation', created_at: RECORDED_AT });
    const requestReceipt = {
        id: 'request-receipt',
        request_id: 'request',
        attempt_id: 'attempt',
        source: { conversation_id: initial.id, revision: initial.revision },
        context_fingerprint: 'sha256:context',
        tool_set_fingerprint: 'sha256:tools',
        request_fingerprint: 'sha256:request',
        target: {
            provider: 'provider',
            protocol: 'provider.protocol',
            model: 'model',
            adapter_version: 'adapter',
        },
        tool_definition_ids: [],
        asset_versions: [],
        item_mappings: [],
        recorded_at: RECORDED_AT,
    };
    const generation = {
        id: 'generation',
        record_source: 'executed' as const,
        request_id: 'request',
        attempt_id: 'attempt',
        purpose: 'interaction',
        requested_model: 'model',
        provider: 'provider',
        protocol: 'provider.protocol',
        adapter_version: 'adapter',
        status: 'completed' as const,
        finish_reason: 'tool_use',
        timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
        source: { conversation_id: initial.id, revision: initial.revision },
        usage: {
            input_tokens: 5,
            output_tokens: 3,
            total_tokens: 8,
            accounting_provenance: {
                input_tokens: { method: 'reported' as const, accounting_basis: 'provider' as const },
                output_tokens: { method: 'reported' as const, accounting_basis: 'provider' as const },
                total_tokens: { method: 'derived' as const, accounting_basis: 'provider' as const },
            },
        },
        request_receipt: requestReceipt,
    };
    const turn = {
        id: 'agent-turn',
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks: [
            { id: 'text', type: 'text' as const, text: 'answer', format: 'plain' as const },
            { id: 'reasoning', type: 'reasoning' as const, text: 'why', representation: 'summary' as const },
            { id: 'json', type: 'json' as const, value: { ok: true } },
            {
                id: 'call-block',
                type: 'tool_call' as const,
                call_id: 'call-1',
                tool_name: 'lookup',
                executor: 'application' as const,
                arguments: { type: 'json' as const, value: { query: 'answer' } },
            },
            ...(options.media ? [{ id: 'image', type: 'image' as const, asset_id: 'image-asset' }] : []),
        ],
        status: 'completed' as const,
        timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
        provenance: { type: 'generated' as const },
        model_visibility: 'include' as const,
        generation_id: generation.id,
    };
    return appendConversationRecords(
        initial,
        {
            turns: [turn],
            generations: [generation],
            ...(options.media
                ? {
                      assets: [
                          {
                              id: 'image-asset',
                              kind: 'image' as const,
                              mime_type: 'image/png',
                              storage: {
                                  type: 'external' as const,
                                  resolver: 'url' as const,
                                  locator: { url: 'gs://bucket/image.png' },
                              },
                              provenance: {
                                  type: 'generated' as const,
                                  generation_id: generation.id,
                                  source_turn_id: turn.id,
                              },
                              created_at: RECORDED_AT,
                          },
                      ],
                  }
                : {}),
        },
        {
            expected_revision: 0,
            operation_id: 'response-operation',
            payload_fingerprint: 'sha256:response',
            recorded_at: RECORDED_AT,
        },
    ).document;
}

function projectionSource(
    response: CanonicalExecutionResponse,
    drafts: Array<{
        type: 'text' | 'reasoning' | 'image';
        text?: string;
        committed_block_ids: string[];
        disposition?: 'direct' | 'structured_output';
    }>,
    failure?: unknown,
): CanonicalExecutionEventStream {
    const output = response.accepted_output;
    const streamIdentity = {
        stream_id: 'stream-projection',
        request_id: output.generation.request_id,
        attempt_id: output.generation.attempt_id,
        response_operation_id: output.receipt.id,
        generation_id: output.generation.id,
        draft_turn_id: output.turn.id,
    } as const;
    const accumulator = new ConversationStreamAccumulator(streamIdentity);
    let sequence = 0;
    const envelope = () => ({
        format: CONVERSATION_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        ...streamIdentity,
        event_id: `stream-projection#${sequence}`,
        sequence: sequence++,
    });
    if (drafts.length > 0) {
        accumulator.append({ ...envelope(), type: 'draft_started', origin: 'live_transport' });
        for (const [index, draft] of drafts.entries()) {
            const draftBlockId = `draft-${index}`;
            const nativePosition = { protocol: 'provider.protocol', path: ['output', index] };
            const block =
                draft.type === 'reasoning'
                    ? ({ type: 'reasoning', visibility: 'display' } as const)
                    : ({ type: draft.type } as const);
            accumulator.append({
                ...envelope(),
                type: 'draft_block_started',
                draft_block_id: draftBlockId,
                native_position: nativePosition,
                block,
            });
            if (draft.text !== undefined) {
                if (draft.type === 'image') throw new Error('Projection fixture media cannot emit text');
                accumulator.append({
                    ...envelope(),
                    type: draft.type === 'text' ? 'draft_text_delta' : 'draft_reasoning_delta',
                    draft_block_id: draftBlockId,
                    native_position: nativePosition,
                    text: draft.text,
                });
            }
            accumulator.append({
                ...envelope(),
                type: 'draft_block_finished',
                draft_block_id: draftBlockId,
                native_position: nativePosition,
                outcome: 'native_complete',
            });
        }
        accumulator.append({ ...envelope(), type: 'draft_finished', outcome: 'completed' });
    }
    const reconciliations = drafts.map((draft, index) => ({
        draft_block_ids: [`draft-${index}`],
        native_positions: [{ protocol: 'provider.protocol', path: ['output', index] }],
        committed_block_ids: draft.committed_block_ids,
        disposition: draft.disposition ?? ('direct' as const),
        ...(draft.disposition === 'structured_output' ? { transformation_id: `transform-${index}` } : {}),
    }));
    const terminal = {
        ...envelope(),
        ...(failure === undefined
            ? {
                  type: 'response_accepted' as const,
                  origin: 'live_transport' as const,
                  conversation: {
                      conversation_id: response.conversation.id,
                      revision: response.conversation.revision,
                  },
                  operation_receipt_id: output.receipt.id,
                  committed_turn_id: output.turn.id,
                  turn_status: output.turn.status,
                  generation_status: output.generation.status,
                  committed_block_ids: output.turn.blocks.map((block) => block.id),
                  accepted_asset_ids: Object.keys(output.assets),
                  reconciliations,
              }
            : { type: 'stream_terminated' as const, outcome: 'failed' as const }),
    } satisfies CanonicalStreamTerminalEvent;
    accumulator.append(terminal);
    return {
        completion: failure === undefined ? response : undefined,
        terminal_event: terminal,
        execution_started: true,
        closed: Promise.resolve(),
        ...(failure === undefined ? {} : { failure }),
        async cancel() {
            return terminal;
        },
        async *[Symbol.asyncIterator]() {
            yield* accumulator.retained_events;
        },
    };
}

describe('canonical execution response', () => {
    it('preserves accepted-recovery provenance through compatibility projection without serializing it', () => {
        const response = markCanonicalAcceptedRecovery(
            createCanonicalExecutionResponse(acceptedDocument(), 'response-operation'),
        );
        const projected = legacyCompletionFromCanonicalExecution(response);

        expect(isCanonicalAcceptedRecovery(response)).toBe(true);
        expect(isCanonicalAcceptedRecovery(projected)).toBe(true);
        expect(JSON.stringify(response)).not.toContain('canonical-accepted-recovery');
        expect(JSON.stringify(projected)).not.toContain('canonical-accepted-recovery');
    });

    it('keeps the complete document authoritative and projects legacy completion only at its boundary', () => {
        const document = acceptedDocument();
        const response = createCanonicalExecutionResponse(document, 'response-operation', { execution_time: 12 });

        expect(response.conversation).toEqual(document);
        expect(response.accepted_output.source).toEqual({ conversation_id: document.id, revision: document.revision });
        const legacy = legacyCompletionFromCanonicalExecution(response, { include_reasoning: true });
        expect(legacy.result).toEqual([
            { type: 'text', value: 'answer' },
            { type: 'thoughts', value: 'why' },
            { type: 'json', value: { ok: true } },
        ]);
        expect(legacy.tool_use).toEqual([{ id: 'call-1', tool_name: 'lookup', tool_input: { query: 'answer' } }]);
        expect(legacy.token_usage).toMatchObject({ prompt: 5, result: 3 });
        expect(legacy.finish_reason).toBe('tool_use');
        expect(legacy.conversation).toEqual(document);
        expect(response).not.toHaveProperty('prompt');
    });

    it('recovers the exact accepted output from a verified fragment after later argument externalization', async () => {
        const accepted = acceptedDocument();
        const original = createCanonicalExecutionResponse(accepted, 'response-operation');
        const prepared = await prepareToolArgumentExternalization(accepted, 'call-1', ['query']);
        const externalized = await externalizeToolCallArguments(accepted, {
            operation_id: 'externalize-operation',
            expected_revision: accepted.revision,
            recorded_at: RECORDED_AT,
            call_id: 'call-1',
            input_path: ['query'],
            model_value: { query: '[stored externally]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: {
                id: 'externalized-argument',
                kind: 'text',
                mime_type: 'text/plain',
                storage: {
                    type: 'external',
                    resolver: 'test.artifact',
                    locator: { artifact_path: 'tool-inputs/call-1.txt' },
                },
                provenance: { type: 'imported', source: 'test' },
                byte_length: prepared.byte_length,
                content_hash: prepared.content_hash,
                created_at: RECORDED_AT,
            },
        });

        expect(() => createCanonicalExecutionResponse(externalized.document, 'response-operation')).toThrow();
        const recovered = createCanonicalExecutionResponse(
            externalized.document,
            'response-operation',
            {},
            original.accepted_output,
        );

        expect(recovered.conversation.revision).toBe(2);
        expect(recovered.accepted_output).toEqual(original.accepted_output);
        expect(legacyCompletionFromCanonicalExecution(recovered).tool_use).toEqual([
            { id: 'call-1', tool_name: 'lookup', tool_input: { query: 'answer' } },
        ]);
    });

    it('fails closed instead of exposing invalid or compact model-only arguments for execution', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const call = response.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Expected tool call');

        call.arguments = { type: 'invalid', raw: '{' };
        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(MalformedStreamingToolArgumentsError);

        call.arguments = {
            type: 'externalized_json',
            value: {},
            model_value: { content: '[stored externally]' },
            exact_arguments_hash: 'sha256:exact',
            hydration: [
                {
                    type: 'text_asset',
                    input_path: ['content'],
                    asset_id: 'asset:tool-input',
                    content_hash: 'sha256:content',
                },
            ],
        };
        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(/lossless argument hydration/);
    });

    it('does not expose provider-executed calls as application tool use', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const call = response.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Expected tool call');
        call.executor = 'provider';

        expect(legacyCompletionFromCanonicalExecution(response).tool_use).toBeUndefined();
    });

    it('projects an accepted canonical structured-output failure onto the legacy error boundary', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined) throw new Error('Expected generation');
        generation.status = 'failed';
        generation.metadata = {
            structured_output: {
                status: 'invalid',
                code: 'json_error',
                message: 'The response is not valid JSON',
            },
        };
        const turn = document.turns.find((candidate) => candidate.id === 'agent-turn');
        if (turn === undefined) throw new Error('Expected generated turn');
        turn.status = 'failed';

        expect(
            legacyCompletionFromCanonicalExecution(createCanonicalExecutionResponse(document, 'response-operation'))
                .error,
        ).toEqual({ code: 'json_error', message: 'The response is not valid JSON' });
    });

    it('preserves terminal cutoff reasons when a response also contains a complete tool call', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined) throw new Error('Expected generation');
        generation.finish_reason = 'max_tokens';

        expect(
            legacyCompletionFromCanonicalExecution(createCanonicalExecutionResponse(document, 'response-operation'))
                .finish_reason,
        ).toBe('length');
    });

    it('drops only malformed calls at a length cutoff and preserves complete parallel calls', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined) throw new Error('Expected generation');
        generation.finish_reason = 'max_tokens';
        const response = createCanonicalExecutionResponse(document, 'response-operation');
        response.accepted_output.turn.blocks.push({
            id: 'partial-call-block',
            type: 'tool_call',
            call_id: 'call-partial',
            tool_name: 'lookup',
            executor: 'application',
            arguments: { type: 'invalid', raw: '{"query":' },
        });

        const legacy = legacyCompletionFromCanonicalExecution(response);
        expect(legacy.finish_reason).toBe('length');
        expect(legacy.tool_use).toEqual([{ id: 'call-1', tool_name: 'lookup', tool_input: { query: 'answer' } }]);
        expect(response.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({
                type: 'tool_call',
                call_id: 'call-partial',
                arguments: { type: 'invalid', raw: '{"query":' },
            }),
        );
    });

    it('preserves malformed streamed tool error identity without a cutoff', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const call = response.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Expected tool call');
        call.arguments = { type: 'invalid', raw: '{"query":' };

        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(MalformedStreamingToolArgumentsError);
    });

    it('preserves supported legacy usage dimensions from retained provider evidence', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined || generation.usage === undefined) throw new Error('Expected generation usage');
        generation.protocol = 'aws.bedrock.converse';
        generation.usage.reported_usage = [
            {
                source: 'provider',
                protocol: 'aws.bedrock.converse',
                payload: { cacheDetails: [{ ttl: '1h', inputTokens: 2 }] },
            },
        ];
        generation.usage.cost = { amount: '0.125', currency: 'USD', provenance: 'reported' };
        const response = createCanonicalExecutionResponse(document, 'response-operation');

        expect(legacyCompletionFromCanonicalExecution(response).token_usage).toMatchObject({
            prompt_cache_write_1h: 2,
            provider_cost_usd: 0.125,
        });
    });

    it('reads scalar accounting without materializing malformed tool arguments or media', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const call = response.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Expected application call');
        call.arguments = { type: 'invalid', raw: '{' };
        response.accepted_output.turn.blocks.push({ id: 'missing', type: 'image', asset_id: 'missing' });
        expect(canonicalExecutionAccounting(response)).toMatchObject({
            token_usage: { prompt: 5, result: 3 },
            finish_reason: 'tool_use',
        });
        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow();
        expect(canonicalExecutionAccounting(response)).not.toHaveProperty('result');
    });

    it.each(['aws.bedrock.converse', 'openai.images.generate'])(
        'preserves %s accounting across accepted recovery',
        (protocol) => {
            const document = acceptedDocument({ media: true });
            const generation = document.generations.generation;
            if (!generation?.usage) throw new Error('Expected generation usage');
            generation.protocol = protocol;
            generation.usage.input_tokens = 20;
            generation.usage.total_tokens = 23;
            generation.usage.cache_read_tokens = 4;
            generation.usage.cache_write_tokens = 7;
            generation.usage.accounting_provenance ??= {};
            generation.usage.accounting_provenance.cache_read_tokens = {
                method: 'reported',
                accounting_basis: 'provider',
            };
            generation.usage.accounting_provenance.cache_write_tokens = {
                method: 'reported',
                accounting_basis: 'provider',
            };
            generation.usage.reported_usage = [
                {
                    source: 'provider',
                    protocol,
                    payload: {
                        cacheDetails: [
                            { ttl: '5m', inputTokens: 5 },
                            { ttl: '1h', inputTokens: 2 },
                        ],
                        output_tokens_details: { image_tokens: 3 },
                    },
                },
            ];
            generation.usage.cost = { amount: '0.125', currency: 'USD', provenance: 'reported' };
            const validation = validateConversationDocument(document);
            expect(validation.diagnostics).toEqual([]);
            const response = createCanonicalExecutionResponse(document, 'response-operation');
            const expected = legacyCompletionFromCanonicalExecution(response);
            const scalar = canonicalExecutionAccounting(response);
            expect(scalar).toEqual({ token_usage: expected.token_usage, finish_reason: expected.finish_reason });
            expect(canonicalExecutionAccounting(markCanonicalAcceptedRecovery(response))).toEqual(scalar);
            expect(scalar.token_usage).toMatchObject(
                protocol === 'aws.bedrock.converse'
                    ? { prompt_cached: 4, prompt_cache_write: 7, prompt_cache_write_1h: 2, provider_cost_usd: 0.125 }
                    : { result_image: 3, provider_cost_usd: 0.125 },
            );
        },
    );

    it('does not resolve inherited asset record keys', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.turn.blocks.push({ id: 'image', type: 'image', asset_id: 'toString' });

        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(/missing image asset toString/);
    });

    it('projects external URL audio with typed media metadata at the legacy boundary', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.assets.audio = {
            id: 'audio',
            kind: 'audio',
            mime_type: 'audio/pcm',
            storage: { type: 'external', resolver: 'url', locator: { url: 'gs://bucket/speech.pcm' } },
            provenance: { type: 'generated', generation_id: 'generation', source_turn_id: 'agent-turn' },
            media: {
                container: 'raw',
                codec: 'pcm',
                sample_rate: 24000,
                channels: 1,
                sample_encoding: 'int16',
                byte_order: 'little',
            },
            created_at: RECORDED_AT,
        };
        response.accepted_output.turn.blocks.push({ id: 'audio-block', type: 'audio', asset_id: 'audio' });

        expect(legacyCompletionFromCanonicalExecution(response).result).toContainEqual({
            type: 'audio',
            value: 'gs://bucket/speech.pcm',
            mime_type: 'audio/pcm',
            container: 'raw',
            codec: 'pcm',
            sample_rate: 24000,
            channels: 1,
            sample_encoding: 'int16',
            byte_order: 'little',
        });
    });

    it('keeps reasoning authoritative while hiding it from fallback previews unless explicitly requested', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const hidden = new FallbackCanonicalExecutionStream(async () => response);
        let hiddenPreview = '';
        for await (const chunk of hidden) hiddenPreview += chunk;
        expect(hiddenPreview).toBe('answer{"ok":true}');
        expect(hidden.completion?.accepted_output.turn.blocks.some((block) => block.type === 'reasoning')).toBe(true);

        const visible = new FallbackCanonicalExecutionStream(async () => response, true);
        let visiblePreview = '';
        for await (const chunk of visible) visiblePreview += chunk;
        expect(visiblePreview).toBe('answerwhy{"ok":true}');
    });

    it('projects output-only recovery without inventing conversation history or native replay', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');

        expect(legacyCompletionFromAcceptedOutput(response.accepted_output, { include_reasoning: true })).toEqual(
            expect.objectContaining({
                result: expect.arrayContaining([
                    expect.objectContaining({ type: 'text', value: 'answer' }),
                    expect.objectContaining({ type: 'thoughts', value: 'why' }),
                ]),
                token_usage: expect.objectContaining({ prompt: 5, result: 3 }),
            }),
        );
        expect(legacyCompletionFromAcceptedOutput(response.accepted_output)).not.toHaveProperty('conversation');
    });

    it('retains accepted recovery as finite-stream success control flow without a failed completion', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const recovery = new CanonicalAcceptedOutputRecovered({ accepted_output: response.accepted_output });
        const stream = new FallbackCanonicalExecutionStream(async () => {
            throw recovery;
        });

        const chunks: string[] = [];
        for await (const chunk of stream) chunks.push(chunk);

        expect(chunks).toEqual([]);
        expect(stream.completion).toBeUndefined();
        expect(stream.accepted_recovery).toBe(recovery);
    });
});

describe('canonical typed execution stream', () => {
    const identity = {
        request_id: 'request',
        attempt_id: 'attempt',
        response_operation_id: 'response-operation',
        generation_id: 'generation',
        draft_turn_id: 'agent-turn',
    };
    const selectionError = (code: typeof CANONICAL_REQUIRED_TOOL_CALL_MISSING | typeof CANONICAL_FORBIDDEN_TOOL_CALL) =>
        new CanonicalToolSelectionViolationError(
            {
                generation: { provider: 'provider', requested_model: 'model' },
            } as DecodedConversationResponse,
            code,
        );

    it('delivers one finite response_accepted event without fabricating native drafts', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const stream = new FallbackCanonicalExecutionEventStream(identity, async () => response, {
            stream_id: 'stream-finite',
        });

        const events: ConversationStreamEvent[] = [];
        for await (const event of stream) events.push(event);

        expect(events).toEqual([
            expect.objectContaining({
                type: 'response_accepted',
                origin: 'live_transport',
                stream_id: 'stream-finite',
                sequence: 0,
                event_id: 'stream-finite#0',
                operation_receipt_id: 'response-operation',
                committed_turn_id: 'agent-turn',
                committed_block_ids: ['text', 'reasoning', 'json', 'call-block'],
            }),
        ]);
        expect(stream.completion).toBe(response);
        expect(stream.terminal_event).toEqual(events[0]);
        expect(stream.execution_started).toBe(true);
    });

    it('propagates accepted recovery without synthesizing a failed terminal', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const recovery = new CanonicalAcceptedOutputRecovered({ accepted_output: response.accepted_output });
        const stream = new FallbackCanonicalExecutionEventStream(
            identity,
            async () => {
                throw recovery;
            },
            { stream_id: 'stream-recovery' },
        );

        const events: ConversationStreamEvent[] = [];
        await expect(
            (async () => {
                for await (const event of stream) events.push(event);
            })(),
        ).rejects.toBe(recovery);
        await expect(stream.closed).resolves.toBeUndefined();
        expect(events).toEqual([]);
        expect(stream.terminal_event).toBeUndefined();
        expect(stream.execution_started).toBe(true);
    });

    it('uses a distinct delivery stream for accepted-response recovery', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const live = new FallbackCanonicalExecutionEventStream(identity, async () => response, {
            stream_id: 'stream-live-delivery',
        });
        for await (const _event of live) {
            // Drain the original delivery.
        }
        const recovered = new FallbackCanonicalExecutionEventStream(identity, async () => response, {
            stream_id: 'stream-recovered-delivery',
            origin: 'accepted_recovery',
        });
        const events: ConversationStreamEvent[] = [];
        for await (const event of recovered) events.push(event);

        expect(live.terminal_event?.stream_id).toBe('stream-live-delivery');
        expect(live.execution_started).toBe(true);
        expect(recovered.execution_started).toBe(false);
        expect(events).toEqual([
            expect.objectContaining({
                type: 'response_accepted',
                origin: 'accepted_recovery',
                stream_id: 'stream-recovered-delivery',
                request_id: 'request',
                attempt_id: 'attempt',
            }),
        ]);
    });

    it('projects finite canonical output only at the explicit legacy string boundary', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const typed = new FallbackCanonicalExecutionEventStream(identity, async () => response, {
            stream_id: 'stream-projection',
        });
        const projected = new LegacyCanonicalExecutionEventProjection(typed);
        let preview = '';
        for await (const chunk of projected) preview += chunk;

        expect(preview).toBe('answer{"ok":true}');
        expect(projected.completion).toBe(response);
    });

    it('emits accepted text, JSON and media once after a reasoning-only draft', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument({ media: true }), 'response-operation');
        response.accepted_output.turn.blocks = response.accepted_output.turn.blocks.filter(
            (block) => block.type !== 'tool_call',
        );
        const projected = new LegacyCanonicalExecutionEventProjection(
            projectionSource(response, [
                { type: 'text', committed_block_ids: ['text'] },
                { type: 'reasoning', text: 'why', committed_block_ids: ['reasoning'] },
                { type: 'text', committed_block_ids: ['json'], disposition: 'structured_output' },
                { type: 'image', committed_block_ids: ['image'] },
            ]),
            true,
        );
        const chunks: string[] = [];
        for await (const chunk of projected) chunks.push(chunk);

        expect(chunks).toEqual(['why', 'answer{"ok":true}[Image]']);
        expect(projected.completion).toBe(response);
    });

    it('does not duplicate drafted text while retaining accepted nontext output', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument({ media: true }), 'response-operation');
        response.accepted_output.turn.blocks = response.accepted_output.turn.blocks.filter(
            (block) => block.type !== 'tool_call',
        );
        const projected = new LegacyCanonicalExecutionEventProjection(
            projectionSource(response, [
                { type: 'text', text: 'answer', committed_block_ids: ['text'] },
                { type: 'reasoning', committed_block_ids: ['reasoning'] },
                { type: 'text', committed_block_ids: ['json'], disposition: 'structured_output' },
                { type: 'image', committed_block_ids: ['image'] },
            ]),
        );
        const chunks: string[] = [];
        for await (const chunk of projected) chunks.push(chunk);

        expect(chunks).toEqual(['answer', '{"ok":true}[Image]']);
    });

    it('suppresses only the committed text block represented by a draft delta', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.turn.blocks = [
            { id: 'text', type: 'text', text: 'answer', format: 'plain' },
            { id: 'text-terminal', type: 'text', text: ' terminal', format: 'plain' },
        ];
        const projected = new LegacyCanonicalExecutionEventProjection(
            projectionSource(response, [
                { type: 'text', text: 'answer', committed_block_ids: ['text'] },
                { type: 'text', committed_block_ids: ['text-terminal'] },
            ]),
        );
        const chunks: string[] = [];
        for await (const chunk of projected) chunks.push(chunk);

        expect(chunks).toEqual(['answer', ' terminal']);
    });

    it('does not duplicate JSON committed from drafted structured-output text', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.turn.blocks = [{ id: 'json', type: 'json', value: { ok: true } }];
        const projected = new LegacyCanonicalExecutionEventProjection(
            projectionSource(response, [
                {
                    type: 'text',
                    text: '{"ok":true}',
                    committed_block_ids: ['json'],
                    disposition: 'structured_output',
                },
            ]),
        );
        const chunks: string[] = [];
        for await (const chunk of projected) chunks.push(chunk);

        expect(chunks).toEqual(['{"ok":true}']);
    });

    it('rethrows the retained in-process failure when a typed stream terminates', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const failure = new Error('durable publication failed');
        const projected = new LegacyCanonicalExecutionEventProjection(projectionSource(response, [], failure));

        await expect(
            (async () => {
                for await (const _chunk of projected) {
                    // The terminal is the only event.
                }
            })(),
        ).rejects.toBe(failure);
        expect(projected.completion).toBeUndefined();
    });

    it('cancels a pending fallback once and exposes its terminal even after iterator return', async () => {
        let calls = 0;
        const stream = new FallbackCanonicalExecutionEventStream(
            identity,
            async (signal) => {
                calls += 1;
                await new Promise<void>((_resolve, reject) => {
                    signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
                });
                throw new Error('unreachable');
            },
            { stream_id: 'stream-cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();
        await iterator.return?.();
        const terminal = await stream.cancel();

        expect(await pending).toEqual({ value: expect.objectContaining({ type: 'stream_terminated' }), done: false });
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled', sequence: 0 });
        expect(stream.terminal_event).toEqual(terminal);
        expect(calls).toBe(1);
    });

    it('settles cancellation without waiting for a transport that ignores abort', async () => {
        const stream = new FallbackCanonicalExecutionEventStream(
            identity,
            async () => new Promise<never>(() => undefined),
            { stream_id: 'stream-ignores-abort' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();

        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toEqual({ value: terminal, done: false });
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
    });

    it('retains a finite response that resolves after cancellation without emitting acceptance', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        let resolveResponse!: (value: CanonicalExecutionResponse) => void;
        let executionStarted!: () => void;
        const started = new Promise<void>((resolve) => {
            executionStarted = resolve;
        });
        const stream = new FallbackCanonicalExecutionEventStream(
            identity,
            async () => {
                executionStarted();
                return new Promise<CanonicalExecutionResponse>((resolve) => {
                    resolveResponse = resolve;
                });
            },
            { stream_id: 'stream-late-finite-response' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();
        await started;
        expect(stream.execution_started).toBe(true);

        const terminal = await stream.cancel();
        await expect(pending).resolves.toEqual({ value: terminal, done: false });
        expect(stream.completion).toBeUndefined();

        resolveResponse(response);
        await stream.closed;

        expect(stream.completion).toBe(response);
        expect(stream.terminal_event).toEqual(terminal);
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
    });

    it('rejects impossible terminal budgets before execution and fails bounded oversized acceptance', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        expect(
            () =>
                new FallbackCanonicalExecutionEventStream(identity, async () => response, {
                    stream_id: 'stream-tiny-event',
                    max_event_bytes: 1,
                }),
        ).toThrow('cannot hold a canonical stream terminal event');
        expect(
            () =>
                new FallbackCanonicalExecutionEventStream(identity, async () => response, {
                    stream_id: 'stream-tiny-total',
                    max_total_bytes: 1,
                }),
        ).toThrow('cannot hold a canonical stream terminal event');

        response.accepted_output.turn.blocks[0].id = 'oversized-'.repeat(1_000);
        const bounded = new FallbackCanonicalExecutionEventStream(identity, async () => response, {
            stream_id: 'stream-bounded-failure',
            max_event_bytes: 1_024,
            max_total_bytes: 1_024,
        });
        const events: ConversationStreamEvent[] = [];
        for await (const event of bounded) events.push(event);
        expect(events).toEqual([
            expect.objectContaining({
                type: 'stream_terminated',
                outcome: 'failed',
                sequence: 0,
                diagnostic: {
                    code: 'CANONICAL_EVENT_DELIVERY_FAILED',
                    message: 'Canonical event delivery failed',
                },
            }),
        ]);
        expect(bounded.completion).toBe(response);
    });

    it('preserves required-tool recovery classification at the exact finite terminal budget', async () => {
        const collect = async (
            code: typeof CANONICAL_REQUIRED_TOOL_CALL_MISSING | typeof CANONICAL_FORBIDDEN_TOOL_CALL,
            limits: { max_event_bytes?: number; max_total_bytes?: number } = {},
        ) => {
            const stream = new FallbackCanonicalExecutionEventStream(
                identity,
                async () => {
                    throw selectionError(code);
                },
                { stream_id: `stream-${code.toLowerCase()}`, max_events: 1, ...limits },
            );
            const events: ConversationStreamEvent[] = [];
            for await (const event of stream) events.push(event);
            return events;
        };
        const required = await collect(CANONICAL_REQUIRED_TOOL_CALL_MISSING);
        const forbidden = await collect(CANONICAL_FORBIDDEN_TOOL_CALL);
        expect(required.at(-1)).toMatchObject({
            type: 'stream_terminated',
            diagnostic: { code: CANONICAL_REQUIRED_TOOL_CALL_MISSING, retryable: false },
        });
        expect(forbidden.at(-1)).toMatchObject({
            type: 'stream_terminated',
            diagnostic: { code: CANONICAL_FORBIDDEN_TOOL_CALL, retryable: false },
        });

        const eventBytes = (event: ConversationStreamEvent) =>
            new TextEncoder().encode(JSON.stringify(event)).byteLength;
        const longer = [required.at(-1), forbidden.at(-1)]
            .filter((event): event is ConversationStreamEvent => event !== undefined)
            .sort((left, right) => eventBytes(right) - eventBytes(left))[0];
        if (longer?.type !== 'stream_terminated' || longer.diagnostic === undefined) {
            throw new Error('Expected a classified finite terminal');
        }
        const exactBytes = eventBytes(longer);
        const exact = await collect(longer.diagnostic.code as typeof CANONICAL_REQUIRED_TOOL_CALL_MISSING, {
            max_event_bytes: exactBytes,
            max_total_bytes: exactBytes,
        });
        expect(exact).toEqual([longer]);
    });

    it('preserves an already-blocked event before a terminal without waiting for buffer capacity', async () => {
        const channel = new CanonicalStreamEventChannel(1);
        const first = streamEvent(0, { type: 'draft_started', origin: 'live_transport' });
        const blockedEvent = streamEvent(1, { type: 'draft_started', origin: 'live_transport' });
        const terminal = streamEvent(2, { type: 'stream_terminated', outcome: 'cancelled' });
        await channel.emit(first);
        const blocked = channel.emit(blockedEvent);

        await channel.terminate(terminal as Extract<ConversationStreamEvent, { type: 'stream_terminated' }>);
        await expect(blocked).resolves.toBeUndefined();
        const received: ConversationStreamEvent[] = [];
        for await (const event of channel) received.push(event);

        expect(received).toEqual([first, blockedEvent, terminal]);
        await expect(channel.emit(blockedEvent)).rejects.toThrow('terminated');
    });

    it('abandons a cancelled retained replay without emitting its terminal after a skipped prefix', async () => {
        const retained = [
            streamEvent(0, { type: 'draft_started', origin: 'live_transport' }),
            streamEvent(1, { type: 'usage_snapshot', usage: { input_tokens: 1 } }),
            streamEvent(2, { type: 'usage_snapshot', usage: { input_tokens: 2 } }),
            streamEvent(3, { type: 'stream_terminated', outcome: 'cancelled' }),
        ];
        const stream = new FallbackCanonicalExecutionEventStream(
            identity,
            async () => {
                throw new Error('Retained replay must not execute transport');
            },
            {
                stream_id: 'stream-channel',
                retained_events: retained,
                max_buffered_events: 1,
            },
        );
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toEqual({ value: retained[0], done: false });
        await iterator.return?.();

        expect(stream.terminal_event).toEqual(retained[3]);
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
    });

    it('rejects a response that does not match the prepared request identity', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const stream = new FallbackCanonicalExecutionEventStream(
            { ...identity, request_id: 'different-request' },
            async () => response,
            { stream_id: 'stream-mismatch' },
        );
        const events: ConversationStreamEvent[] = [];
        for await (const event of stream) events.push(event);

        expect(events).toEqual([
            expect.objectContaining({
                type: 'stream_terminated',
                outcome: 'failed',
                diagnostic: {
                    code: 'CANONICAL_EXECUTION_FAILED',
                    message: 'Canonical execution failed',
                },
            }),
        ]);
        expect(stream.completion).toBeUndefined();
    });

    it('finalizes structured reconciliation from canonical decode blocks without a CompletionResult projection', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.turn.blocks = [{ id: 'json', type: 'json', value: { emoji: '😀', ok: true } }];
        const sourceBlocks = [
            { id: 'source-1', type: 'text' as const, text: '{"emoji":"😀","ok":', format: 'plain' as const },
            { id: 'source-2', type: 'text' as const, text: 'true}', format: 'plain' as const },
        ];
        const resultBlock = { id: 'json', type: 'json' as const, value: { emoji: '😀', ok: true } };
        const proof = await createStructuredOutputTransformationProof({
            id: 'transform-1',
            source_blocks: sourceBlocks,
            result_block: resultBlock,
        });
        const decoded = {
            turns: [{ id: 'agent-turn', status: 'completed', blocks: [resultBlock] }],
            generation: { id: 'generation', request_id: 'request', attempt_id: 'attempt', status: 'completed' },
            stream_evidence: {
                item_mappings: sourceBlocks.map((block, index) => ({
                    canonical_id: block.id,
                    native_position: { protocol: 'test.protocol', path: ['output', index] },
                    kind: 'block' as const,
                })),
                transformations: [proof],
            },
        } as unknown as Parameters<typeof finalizeCanonicalExecutionStreamResponse>[0]['decoded'];
        const streamIdentity = { ...identity, stream_id: 'stream-structured' };
        const envelope = (sequence: number) => ({
            format: CONVERSATION_FORMAT,
            schema_version: CONVERSATION_SCHEMA_VERSION,
            experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
            ...streamIdentity,
            sequence,
            event_id: `stream-structured#${sequence}`,
        });
        const buildAccumulator = (options: { wrong_content?: boolean; wrong_kind?: boolean } = {}) => {
            const candidate = new ConversationStreamAccumulator(streamIdentity);
            let sequence = 0;
            candidate.append({ ...envelope(sequence++), type: 'draft_started', origin: 'live_transport' });
            for (const [index, block] of sourceBlocks.entries()) {
                const nativePosition = { protocol: 'test.protocol', path: ['output', index] };
                const wrongKind = options.wrong_kind === true && index === 0;
                candidate.append({
                    ...envelope(sequence++),
                    type: 'draft_block_started',
                    draft_block_id: `draft-${index}`,
                    native_position: nativePosition,
                    block: wrongKind ? { type: 'tool_call', executor: 'application' } : { type: 'text' },
                });
                if (!wrongKind) {
                    const fragments =
                        index === 0
                            ? options.wrong_content
                                ? ['{"emoji":"wrong","ok":']
                                : ['{"emoji":"\ud83d', '\ude00","ok":']
                            : [block.text];
                    for (const text of fragments) {
                        candidate.append({
                            ...envelope(sequence++),
                            type: 'draft_text_delta',
                            draft_block_id: `draft-${index}`,
                            native_position: nativePosition,
                            text,
                        });
                    }
                }
                candidate.append({
                    ...envelope(sequence++),
                    type: 'draft_block_finished',
                    draft_block_id: `draft-${index}`,
                    native_position: nativePosition,
                    outcome: 'native_complete',
                });
            }
            candidate.append({ ...envelope(sequence), type: 'draft_finished', outcome: 'completed' });
            return candidate;
        };
        const accumulator = buildAccumulator();
        const reconciliations = [
            {
                draft_block_ids: ['draft-0', 'draft-1'],
                native_positions: sourceBlocks.map((_block, index) => ({
                    protocol: 'test.protocol',
                    path: ['output', index],
                })),
                committed_block_ids: ['json'],
                disposition: 'structured_output' as const,
                transformation_id: 'transform-1',
            },
        ];

        const accepted = await finalizeCanonicalExecutionStreamResponse({
            accumulator,
            decoded,
            response,
            result_schema: { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] },
            reconciliations,
        });
        expect(accepted).toMatchObject({ type: 'response_accepted', committed_block_ids: ['json'] });
        const structuredReconciliation = accepted.reconciliations[0];
        if (structuredReconciliation === undefined) throw new Error('Expected structured reconciliation');
        expect(accumulator.draft_snapshot()[0]).toMatchObject({ text: '{"emoji":"😀","ok":' });

        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator: buildAccumulator({ wrong_content: true }),
                decoded,
                response,
                result_schema: { type: 'object' },
                reconciliations,
            }),
        ).rejects.toThrow('source text differs');
        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator: buildAccumulator({ wrong_kind: true }),
                decoded,
                response,
                result_schema: { type: 'object' },
                reconciliations,
            }),
        ).rejects.toThrow('not bound to a text draft');

        const wrongResultAccumulator = new ConversationStreamAccumulator(streamIdentity);
        for (const event of accumulator.retained_events.slice(0, -1)) wrongResultAccumulator.append(event);
        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator: wrongResultAccumulator,
                decoded,
                response,
                result_schema: { type: 'object' },
                reconciliations: [
                    {
                        ...structuredReconciliation,
                        committed_block_ids: ['other-json'],
                    },
                ],
            }),
        ).rejects.toThrow('proof result does not match');

        const changedResponse = structuredClone(response);
        const acceptedJson = changedResponse.accepted_output.turn.blocks[0];
        if (acceptedJson?.type !== 'json') throw new Error('Expected accepted JSON block');
        acceptedJson.value = { ok: false };
        const mismatchedAccumulator = new ConversationStreamAccumulator(streamIdentity);
        for (const event of accumulator.retained_events.slice(0, -1)) mismatchedAccumulator.append(event);
        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator: mismatchedAccumulator,
                decoded,
                response: changedResponse,
                result_schema: { type: 'object' },
                reconciliations: accepted.reconciliations,
            }),
        ).rejects.toThrow('differs from decoded block');

        const changed = structuredClone(decoded);
        const transformation = changed.stream_evidence?.transformations[0];
        if (transformation === undefined) throw new Error('Expected transformation');
        transformation.source_texts[0] = '{"ok":false';
        const rejectedAccumulator = new ConversationStreamAccumulator(streamIdentity);
        for (const event of accumulator.retained_events.slice(0, -1)) rejectedAccumulator.append(event);
        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator: rejectedAccumulator,
                decoded: changed,
                response,
                result_schema: { type: 'object' },
                reconciliations: accepted.reconciliations,
            }),
        ).rejects.toThrow();
    });

    it('accepts mapped terminal-only blocks only when the live stream emitted no drafts', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const terminalBlock = {
            id: 'terminal-only',
            type: 'text' as const,
            text: 'Prompt blocked.',
            format: 'plain' as const,
        };
        response.accepted_output.turn.blocks = [terminalBlock];
        const streamIdentity = { ...identity, stream_id: 'stream-terminal-only' };
        const envelope = (sequence: number) => ({
            format: CONVERSATION_FORMAT,
            schema_version: CONVERSATION_SCHEMA_VERSION,
            experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
            ...streamIdentity,
            sequence,
            event_id: `stream-terminal-only#${sequence}`,
        });
        const buildAccumulator = () => {
            const accumulator = new ConversationStreamAccumulator(streamIdentity);
            accumulator.append({ ...envelope(0), type: 'draft_started', origin: 'live_transport' });
            accumulator.append({ ...envelope(1), type: 'draft_finished', outcome: 'completed' });
            return accumulator;
        };
        const nativePosition = { protocol: 'test.protocol', path: ['promptFeedback', 'blockReasonMessage'] };
        const decoded = {
            turns: [{ id: 'agent-turn', status: 'completed', blocks: [terminalBlock] }],
            generation: { id: 'generation', request_id: 'request', attempt_id: 'attempt', status: 'completed' },
            stream_evidence: {
                item_mappings: [{ canonical_id: terminalBlock.id, native_position: nativePosition, kind: 'block' }],
                transformations: [],
            },
        } as unknown as Parameters<typeof finalizeCanonicalExecutionStreamResponse>[0]['decoded'];

        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator: buildAccumulator(),
                decoded,
                response,
                reconciliations: [],
            }),
        ).resolves.toMatchObject({ type: 'response_accepted', committed_block_ids: [terminalBlock.id] });

        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator: buildAccumulator(),
                decoded: { ...decoded, stream_evidence: { item_mappings: [], transformations: [] } },
                response,
                reconciliations: [],
            }),
        ).rejects.toThrow('has no native decode mapping');
    });

    it('rejects final tool identity or executor changes from the cumulative native draft', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const acceptedCall = response.accepted_output.turn.blocks.find((block) => block.id === 'call-block');
        if (acceptedCall?.type !== 'tool_call') throw new Error('Expected accepted tool call');
        response.accepted_output.turn.blocks = [acceptedCall];
        const streamIdentity = { ...identity, stream_id: 'stream-tool-binding' };
        const accumulator = new ConversationStreamAccumulator(streamIdentity);
        const envelope = (sequence: number) => ({
            format: CONVERSATION_FORMAT,
            schema_version: CONVERSATION_SCHEMA_VERSION,
            experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
            ...streamIdentity,
            sequence,
            event_id: `stream-tool-binding#${sequence}`,
        });
        const nativePosition = { protocol: 'test.protocol', path: ['output', 0] };
        accumulator.append({ ...envelope(0), type: 'draft_started', origin: 'live_transport' });
        accumulator.append({
            ...envelope(1),
            type: 'draft_block_started',
            draft_block_id: 'tool-draft',
            native_position: nativePosition,
            block: { type: 'tool_call', executor: 'provider', call_id: 'call-1', tool_name: 'lookup' },
        });
        accumulator.append({
            ...envelope(2),
            type: 'draft_block_finished',
            draft_block_id: 'tool-draft',
            native_position: nativePosition,
            outcome: 'native_complete',
        });
        accumulator.append({ ...envelope(3), type: 'draft_finished', outcome: 'completed' });
        const decoded = {
            turns: [
                {
                    id: 'agent-turn',
                    status: 'completed',
                    blocks: [structuredClone(acceptedCall)],
                },
            ],
            generation: { id: 'generation', request_id: 'request', attempt_id: 'attempt', status: 'completed' },
            stream_evidence: {
                item_mappings: [
                    {
                        canonical_id: 'call-block',
                        native_position: nativePosition,
                        kind: 'block',
                    },
                ],
                transformations: [],
            },
        } as unknown as Parameters<typeof finalizeCanonicalExecutionStreamResponse>[0]['decoded'];

        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator,
                decoded,
                response,
                reconciliations: [
                    {
                        draft_block_ids: ['tool-draft'],
                        native_positions: [nativePosition],
                        committed_block_ids: ['call-block'],
                        disposition: 'direct',
                    },
                ],
            }),
        ).rejects.toThrow('changes identity or executor');
    });

    it('rejects swapped native positions across same-type direct text blocks', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.turn.blocks = [
            { id: 'text-1', type: 'text', text: 'first', format: 'plain' },
            { id: 'text-2', type: 'text', text: 'second', format: 'plain' },
        ];
        const streamIdentity = { ...identity, stream_id: 'stream-swapped-text' };
        const accumulator = new ConversationStreamAccumulator(streamIdentity);
        const envelope = (sequence: number) => ({
            format: CONVERSATION_FORMAT,
            schema_version: CONVERSATION_SCHEMA_VERSION,
            experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
            ...streamIdentity,
            sequence,
            event_id: `stream-swapped-text#${sequence}`,
        });
        const positions = [
            { protocol: 'test.protocol', path: ['output', 0] },
            { protocol: 'test.protocol', path: ['output', 1] },
        ];
        let sequence = 0;
        accumulator.append({ ...envelope(sequence++), type: 'draft_started', origin: 'live_transport' });
        for (const [index, position] of positions.entries()) {
            accumulator.append({
                ...envelope(sequence++),
                type: 'draft_block_started',
                draft_block_id: `draft-${index}`,
                native_position: position,
                block: { type: 'text' },
            });
            accumulator.append({
                ...envelope(sequence++),
                type: 'draft_text_delta',
                draft_block_id: `draft-${index}`,
                native_position: position,
                text: index === 0 ? 'first' : 'second',
            });
            accumulator.append({
                ...envelope(sequence++),
                type: 'draft_block_finished',
                draft_block_id: `draft-${index}`,
                native_position: position,
                outcome: 'native_complete',
            });
        }
        accumulator.append({ ...envelope(sequence), type: 'draft_finished', outcome: 'completed' });
        const decoded = {
            turns: [
                {
                    id: 'agent-turn',
                    status: 'completed',
                    blocks: structuredClone(response.accepted_output.turn.blocks),
                },
            ],
            generation: { id: 'generation', request_id: 'request', attempt_id: 'attempt', status: 'completed' },
            stream_evidence: {
                item_mappings: [
                    { canonical_id: 'text-1', native_position: positions[1], kind: 'block' },
                    { canonical_id: 'text-2', native_position: positions[0], kind: 'block' },
                ],
                transformations: [],
            },
        } as unknown as Parameters<typeof finalizeCanonicalExecutionStreamResponse>[0]['decoded'];

        await expect(
            finalizeCanonicalExecutionStreamResponse({
                accumulator,
                decoded,
                response,
                reconciliations: positions.map((position, index) => ({
                    draft_block_ids: [`draft-${index}`],
                    native_positions: [position],
                    committed_block_ids: [`text-${index + 1}`],
                    disposition: 'direct',
                })),
            }),
        ).rejects.toThrow('changes native position');
    });
});
