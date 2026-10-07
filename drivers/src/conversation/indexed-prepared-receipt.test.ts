import Anthropic from '@anthropic-ai/sdk';
import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import {
    appendConversationRecords,
    ConversationValidationError,
    createAcceptedOutputFragment,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    decodedResponseBatchFromAcceptedRecord,
    deriveConversationId,
    fingerprintJson,
    INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
    INDEXED_MEASURED_OUTPUT_PREPARED_VALIDATOR_PROFILE,
    INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE,
    type IndexedConversationRecordStore,
    IndexedMeasuredOutputReceiptSchema,
    indexedProcessingContextFingerprint,
    loadIndexedSelectedTextContext,
    parseConversationPreparedRequestRecord,
    setProcessingPolicy,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingCoverage,
} from '@llumiverse/conversation';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from '../anthropic/index.js';
import { OpenAIDriver } from '../openai/openai.js';
import { OpenAIChatCompletionsDriver } from '../openai/openai_chat_completions.js';
import { indexedPreparedReceiptMatches } from './indexed-prepared-receipt.js';

// These test adapter integrity constraints only. Actual epoch/phase authority and measured count
// are exercised by the installed handoff HTTP/Mongo/Temporal fixture, never conferred by this data.
async function fixture(protocol: 'chat' | 'responses' | 'responses_counted' | 'anthropic') {
    const at = '2026-10-05T00:00:00.000Z';
    let document = createConversationDocument({ id: 'conversation:measured-indexed', created_at: at });
    for (let i = 0; i < 4; i++) {
        const turn = createUserTurn({
            id: `turn:${i}`,
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: `block:${i}`, text: `text ${i}`, format: 'plain' })],
        });
        document = appendConversationRecords(
            document,
            {
                turns: [turn],
                context_entries: [{ id: `entry:${i}`, type: 'source_turn', turn_id: turn.id }],
            },
            {
                expected_revision: document.revision,
                operation_id: `operation:${i}`,
                payload_fingerprint: `sha256:input${i}`,
                recorded_at: at,
            },
        ).document;
    }
    const bytes = new Map<string, Uint8Array>();
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const value = bytes.get(ref.content_hash);
            if (!value) throw new Error('Missing page');
            return value;
        },
        async write(value, ref) {
            bytes.set(ref.content_hash, Uint8Array.from(value));
        },
        async readRecord(ref) {
            const value = bytes.get(ref.content_hash);
            if (!value) throw new Error('Missing record');
            return value;
        },
        async writeRecord(ref, value) {
            bytes.set(ref.content_hash, Uint8Array.from(value));
        },
    };
    const staged = await stageIndexedConversationSnapshot(document, undefined, store);
    const selection = await loadIndexedSelectedTextContext(store, staged.root, staged.locator);
    const runtime = {
        conversation_id: document.id,
        request_id: 'request:measured',
        attempt_id: 'attempt:measured',
        input_operation_id: 'operation:0',
        response_operation_id: 'operation:answer',
        recorded_at: at,
        purpose: 'interaction',
        materialized_input: { operation_id: 'operation:0', result_revision: 1 },
    };
    const driver =
        protocol === 'anthropic'
            ? new AnthropicDriver({ apiKey: 'test' })
            : protocol === 'chat'
              ? new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://indexed.test/v1' })
              : new OpenAIDriver({ apiKey: 'test' });
    const prepared = await driver.prepareIndexedTextRequest({
        selection,
        runtime,
        options: { model: protocol === 'anthropic' ? 'claude-sonnet-4-6' : 'gpt-4o-mini-2024-07-18' },
        stream: false,
    });
    const measurement = {
        input_tokens: 20,
        method:
            protocol === 'anthropic' || protocol === 'responses_counted'
                ? ('provider_counted' as const)
                : ('estimated' as const),
        tokenizer:
            protocol === 'anthropic'
                ? 'anthropic.messages.count_tokens:v1'
                : protocol === 'responses_counted'
                  ? 'openai.responses.input_tokens:v1'
                  : 'fixture:bpe',
        tokenizer_version:
            protocol === 'anthropic'
                ? 'anthropic.messages.count_tokens:projection-2026-10-05.v1'
                : protocol === 'responses_counted'
                  ? 'openai.responses.input_tokens:projection-2026-10-07.v1'
                  : '1',
        adapter: prepared.receipt.target.protocol,
        adapter_version: prepared.receipt.target.adapter_version,
        source_fingerprint: await indexedProcessingContextFingerprint({
            ...selection,
            completeness: 'active_processing_dependencies_verified',
        }),
        target_model: prepared.receipt.target.model,
        measured_at: at,
    };
    const record = parseConversationPreparedRequestRecord({
        source: selection.source,
        runtime,
        request_receipt: { ...prepared.receipt, measurement },
        generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
        response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        indexed_source: {
            version: 1,
            validator_profile: INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE,
            root: staged.locator,
            context_revision: selection.context.revision,
            processing_input: {
                version: 1,
                activation_operation_id: runtime.input_operation_id,
                original_source: { conversation_id: document.id, revision: 0 },
                original_root: staged.locator,
                accepted_input_revision: 1,
                accepted_input_receipt_fingerprint: 'sha256:accepted',
                settled_source: { conversation_id: document.id, revision: 3 },
                settled_root: staged.locator,
                coverage: {
                    operation_id: 'operation:coverage',
                    expected_revision: 3,
                    target_fingerprint: await fingerprintJson(prepared.receipt.target),
                    measured_input_tokens: 20,
                    tokenizer_id: `${measurement.tokenizer}:${measurement.tokenizer_version}`,
                    measurement_fingerprint: await fingerprintJson(measurement),
                    recorded_at: at,
                },
            },
        },
    });
    return { record, selection, receipt: prepared.receipt, driver, document, store, at };
}

async function rejectUnregisteredOpenAICount(
    record: Parameters<typeof indexedPreparedReceiptMatches>[0],
    selection: Parameters<typeof indexedPreparedReceiptMatches>[1],
    compiled: Parameters<typeof indexedPreparedReceiptMatches>[2],
) {
    for (const field of ['provider', 'protocol', 'version'] as const) {
        const changed = structuredClone(record);
        const receipt = structuredClone(compiled);
        const measurement = changed.request_receipt.measurement;
        const witness = changed.indexed_source?.processing_input ?? changed.indexed_source?.native_measurement;
        if (!measurement || !witness) throw new Error('Expected measured provider count witness');
        if (field === 'provider') receipt.target.provider = 'openai_compatible';
        if (field === 'protocol') receipt.target.protocol = 'openai.chat.completions';
        if (field === 'version') measurement.tokenizer_version = 'unregistered:counter-version';
        // Rebind every dependent hash so refusal proves the finite counter registration,
        // rather than merely detecting an unrelated receipt or stale coverage fingerprint.
        changed.request_receipt.target = structuredClone(receipt.target);
        measurement.adapter = receipt.target.protocol;
        witness.coverage.target_fingerprint = await fingerprintJson(receipt.target);
        witness.coverage.tokenizer_id = `${measurement.tokenizer}:${measurement.tokenizer_version}`;
        witness.coverage.measurement_fingerprint = await fingerprintJson(measurement);
        expect(await indexedPreparedReceiptMatches(changed, selection, receipt), field).toBe(false);
    }
}

describe('indexed measured native receipt integrity', () => {
    it.each(['chat', 'responses', 'responses_counted'] as const)(
        'retains exact %s native receipt with its separate measured coverage',
        async (protocol) => {
            const { record, selection, receipt } = await fixture(protocol);
            expect(await indexedPreparedReceiptMatches(record, selection, receipt)).toBe(true);
            if (protocol === 'responses_counted') await rejectUnregisteredOpenAICount(record, selection, receipt);
            const oldProfile = parseConversationPreparedRequestRecord({
                ...record,
                indexed_source: {
                    version: 1,
                    validator_profile: 'llumiverse.conversation/indexed-selected-text/2026-10-03.v1',
                    root: record.indexed_source?.root,
                    context_revision: selection.context.revision,
                },
            });
            expect(await indexedPreparedReceiptMatches(oldProfile, selection, receipt)).toBe(false);
            for (const field of ['count', 'target', 'source', 'native', 'coverage'] as const) {
                const changed = structuredClone(record);
                const measured = changed.request_receipt.measurement;
                const witness = changed.indexed_source?.processing_input;
                if (!measured || !witness) throw new Error('Expected measured witness');
                if (field === 'count') measured.input_tokens++;
                if (field === 'target') measured.target_model = 'foreign-model';
                if (field === 'source') measured.source_fingerprint = 'sha256:foreign';
                if (field === 'native') changed.request_receipt.request_fingerprint = 'sha256:foreign-native';
                if (field === 'coverage') witness.coverage.target_fingerprint = 'sha256:foreign-target';
                expect(await indexedPreparedReceiptMatches(changed, selection, receipt), field).toBe(false);
            }
            const changedSelection = structuredClone(selection);
            const turn = changedSelection.turns[0];
            const block = turn?.selected_blocks[0];
            if (block?.type !== 'text') throw new Error('Expected selected text');
            block.text = 'foreign selected bytes';
            expect(await indexedPreparedReceiptMatches(record, changedSelection, receipt)).toBe(false);
        },
    );
});

describe('generic measured native prepared profile', () => {
    it.each(['chat', 'responses', 'responses_counted', 'anthropic'] as const)(
        'binds %s count to the accepted input without borrowing an epoch',
        async (protocol) => {
            const { record: ordinary, selection, receipt } = await fixture(protocol);
            const epoch = ordinary.indexed_source?.processing_input;
            if (!epoch) throw new Error('Expected independent measured receipt fixture');
            const record = parseConversationPreparedRequestRecord({
                ...ordinary,
                indexed_source: {
                    version: 1,
                    validator_profile: INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
                    root: selection.root,
                    context_revision: selection.context.revision,
                    native_measurement: {
                        version: 1,
                        accepted_source: { conversation_id: selection.source.conversation_id, revision: 1 },
                        accepted_root: selection.root,
                        accepted_operation_id: ordinary.runtime.materialized_input?.operation_id,
                        runtime_input_operation_id: ordinary.runtime.input_operation_id,
                        accepted_receipt_fingerprint: epoch.accepted_input_receipt_fingerprint,
                        settled_source: epoch.settled_source,
                        settled_root: epoch.settled_root,
                        coverage: epoch.coverage,
                    },
                },
            });
            expect(await indexedPreparedReceiptMatches(record, selection, receipt)).toBe(true);
            if (protocol === 'responses_counted') await rejectUnregisteredOpenAICount(record, selection, receipt);
            if (protocol === 'anthropic') {
                expect(await indexedPreparedReceiptMatches(ordinary, selection, receipt)).toBe(true);
                const unsupported = structuredClone(record);
                const measurement = unsupported.request_receipt.measurement;
                const witness = unsupported.indexed_source?.native_measurement;
                if (!measurement || !witness) throw new Error('Expected provider-counted native witness');
                measurement.tokenizer_version = 'unknown-provider-counter';
                witness.coverage.tokenizer_id = `${measurement.tokenizer}:${measurement.tokenizer_version}`;
                witness.coverage.measurement_fingerprint = await fingerprintJson(measurement);
                expect(await indexedPreparedReceiptMatches(unsupported, selection, receipt)).toBe(false);
            }
            for (const mutation of ['operation', 'accepted_revision', 'coverage', 'count', 'native'] as const) {
                const changed = structuredClone(record);
                const witness = changed.indexed_source?.native_measurement;
                const measurement = changed.request_receipt.measurement;
                if (!witness || !measurement) throw new Error('Expected measured native witness');
                if (mutation === 'operation') witness.accepted_operation_id = 'operation:unrelated';
                if (mutation === 'accepted_revision') witness.accepted_source.revision++;
                if (mutation === 'coverage') witness.coverage.expected_revision++;
                if (mutation === 'count') measurement.input_tokens++;
                if (mutation === 'native') changed.request_receipt.request_fingerprint = 'sha256:another-native-body';
                expect(await indexedPreparedReceiptMatches(changed, selection, receipt), mutation).toBe(false);
            }
            expect(() =>
                parseConversationPreparedRequestRecord({
                    ...record,
                    indexed_source: { ...record.indexed_source, processing_input: epoch },
                }),
            ).toThrow();
        },
    );
});

describe('measured accepted-output native profile', () => {
    it.each(['chat', 'responses', 'anthropic'] as const)(
        'executes %s from a genuine prior output through a newer policy and coverage root',
        async (protocol) => {
            const prior = await fixture(protocol);
            let transports = 0;
            if (prior.driver instanceof AnthropicDriver) {
                prior.driver.client = new Anthropic({
                    apiKey: 'owned-fixture',
                    baseURL: 'https://indexed-anthropic.test',
                    fetch: async (url) => {
                        if (new URL(String(url)).pathname === '/v1/messages/count_tokens')
                            return new Response(JSON.stringify({ input_tokens: 20 }), {
                                headers: { 'content-type': 'application/json' },
                            });
                        transports++;
                        const text = transports === 1 ? 'Actual prior accepted output.' : 'Continued exactly once.';
                        const message: Message = {
                            id: `message:${transports}`,
                            type: 'message',
                            role: 'assistant',
                            model: 'claude-sonnet-4-6',
                            content: [],
                            stop_reason: null,
                            stop_sequence: null,
                            container: null,
                            diagnostics: null,
                            stop_details: null,
                            usage: {
                                input_tokens: 11,
                                output_tokens: 0,
                                cache_creation: null,
                                cache_creation_input_tokens: null,
                                cache_read_input_tokens: null,
                                inference_geo: null,
                                output_tokens_details: null,
                                server_tool_use: null,
                                service_tier: null,
                            },
                        };
                        const frames = [
                            { type: 'message_start', message },
                            {
                                type: 'content_block_start',
                                index: 0,
                                content_block: { type: 'text', text: '', citations: [] },
                            },
                            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } },
                            { type: 'content_block_stop', index: 0 },
                            {
                                type: 'message_delta',
                                delta: {
                                    stop_reason: 'end_turn',
                                    stop_sequence: null,
                                    container: null,
                                    stop_details: null,
                                },
                                usage: {
                                    output_tokens: 7,
                                    input_tokens: null,
                                    cache_creation_input_tokens: null,
                                    cache_read_input_tokens: null,
                                    output_tokens_details: null,
                                    server_tool_use: null,
                                },
                            },
                            { type: 'message_stop' },
                        ] satisfies RawMessageStreamEvent[];
                        return new Response(
                            frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''),
                            { headers: { 'content-type': 'text/event-stream' } },
                        );
                    },
                });
            } else {
                prior.driver.service = prior.driver.service.withOptions({
                    fetch: async (url) => {
                        if (
                            prior.driver instanceof OpenAIDriver &&
                            new URL(String(url)).pathname === '/v1/responses/input_tokens'
                        )
                            return Response.json({ object: 'response.input_tokens', input_tokens: 20 });
                        transports++;
                        const text = transports === 1 ? 'Actual prior accepted output.' : 'Continued exactly once.';
                        const body =
                            protocol === 'responses'
                                ? {
                                      id: `response:${transports}`,
                                      object: 'response',
                                      created_at: 1750000000,
                                      model: 'gpt-4o-mini-2024-07-18',
                                      status: 'completed',
                                      output: [
                                          {
                                              id: `message:${transports}`,
                                              type: 'message',
                                              role: 'assistant',
                                              status: 'completed',
                                              content: [{ type: 'output_text', text, annotations: [], logprobs: [] }],
                                          },
                                      ],
                                      output_text: text,
                                      error: null,
                                      incomplete_details: null,
                                      usage: {
                                          input_tokens: 11,
                                          output_tokens: 7,
                                          total_tokens: 18,
                                          input_tokens_details: { cached_tokens: 0 },
                                          output_tokens_details: { reasoning_tokens: 0 },
                                      },
                                  }
                                : {
                                      id: `completion:${transports}`,
                                      object: 'chat.completion',
                                      created: 1750000000,
                                      model: 'gpt-4o-mini-2024-07-18',
                                      choices: [
                                          {
                                              index: 0,
                                              message: { role: 'assistant', content: text },
                                              finish_reason: 'stop',
                                              logprobs: null,
                                          },
                                      ],
                                      usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
                                  };
                        return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
                    },
                });
            }
            const epoch = prior.record.indexed_source?.processing_input;
            if (!epoch) throw new Error('Expected independent input measurement fixture');
            const initialRecord =
                prior.driver instanceof AnthropicDriver
                    ? parseConversationPreparedRequestRecord({
                          ...prior.record,
                          indexed_source: {
                              version: 1,
                              validator_profile: INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
                              root: prior.selection.root,
                              context_revision: prior.selection.context.revision,
                              native_measurement: {
                                  version: 1,
                                  accepted_source: {
                                      conversation_id: prior.selection.source.conversation_id,
                                      revision: 1,
                                  },
                                  accepted_root: prior.selection.root,
                                  accepted_operation_id: prior.record.runtime.materialized_input?.operation_id,
                                  runtime_input_operation_id: prior.record.runtime.input_operation_id,
                                  accepted_receipt_fingerprint: epoch.accepted_input_receipt_fingerprint,
                                  settled_source: epoch.settled_source,
                                  settled_root: epoch.settled_root,
                                  coverage: epoch.coverage,
                              },
                          },
                      })
                    : prior.record;
            // The actual adapter supplies executed generation/request evidence; the canonical
            // append validator supplies the original output receipt. No generated turn or
            // execution authority is invented by this integrity-only transport fixture.
            const decoded = await prior.driver.executeCommittedIndexedTextRequest({
                selection: prior.selection,
                record: initialRecord,
                options: { model: prior.record.request_receipt.target.model },
                assert_committed: async () => undefined,
            });
            const decodedTurn = decoded.turns[0];
            if (!decodedTurn) throw new Error('Actual decoded output turn missing');
            // The real adapter timestamps output at completion, after the original input request.
            // Acceptance cannot predate that turn or public fragment chronology is inconsistent.
            const response = decodedResponseBatchFromAcceptedRecord(prior.record, decoded, {
                operation_id: prior.record.runtime.response_operation_id,
                recorded_at: decodedTurn.timestamps.recorded_at,
            });
            const accepted = appendConversationRecords(prior.document, response.batch, response.options);
            const acceptedReceipt = accepted.document.operation_receipts[response.options.operation_id];
            if (!acceptedReceipt) throw new Error('Actual accepted output append receipt missing');
            const output = createAcceptedOutputFragment(accepted.document, acceptedReceipt.id).receipt;
            expect(output.recorded_at).toBe(decodedTurn.timestamps.recorded_at);
            const backdated = structuredClone(accepted.document);
            const backdatedReceipt = backdated.operation_receipts[acceptedReceipt.id];
            if (!backdatedReceipt) throw new Error('Copied actual output receipt missing');
            backdatedReceipt.recorded_at = prior.at;
            expect(() => createAcceptedOutputFragment(backdated, acceptedReceipt.id)).toThrow(
                'inconsistent canonical references',
            );
            expect(decoded.generation.record_source).toBe('executed');
            expect(output.accepted_generation_ids).toEqual([decoded.generation.id]);
            expect(await fingerprintJson(acceptedReceipt)).not.toBe(await fingerprintJson(output));
            const policy = await setProcessingPolicy(accepted.document, {
                operation_id: 'policy:after-output',
                expected_revision: accepted.document.revision,
                recorded_at: output.recorded_at,
                enabled: false,
                processors: [],
            });
            const pinned = await stageIndexedConversationSnapshot(policy.document, undefined, prior.store);
            const measuredSelection = await loadIndexedSelectedTextContext(prior.store, pinned.root, pinned.locator);
            const { materialized_input: _input, ...originalRuntime } = prior.record.runtime;
            const runtime = {
                ...originalRuntime,
                request_id: 'request:output-continuation',
                attempt_id: 'attempt:output-continuation',
                input_operation_id: 'input:output-continuation',
                response_operation_id: 'response:output-continuation',
                recorded_at: output.recorded_at,
            };
            const preparedBeforeCoverage = await prior.driver.prepareIndexedTextRequest({
                selection: measuredSelection,
                runtime,
                options: { model: prior.record.request_receipt.target.model },
                stream: false,
            });
            const providerCount =
                prior.driver instanceof AnthropicDriver || prior.driver instanceof OpenAIDriver
                    ? await prior.driver.countIndexedNativeRequest(
                          preparedBeforeCoverage.native_request,
                          preparedBeforeCoverage.receipt.target,
                      )
                    : undefined;
            const measurement = {
                ...prior.record.request_receipt.measurement,
                input_tokens: providerCount?.input_tokens ?? 20,
                method: providerCount ? ('provider_counted' as const) : ('estimated' as const),
                tokenizer: providerCount?.profile ?? 'fixture:bpe',
                tokenizer_version: providerCount?.tokenizer_version ?? '1',
                adapter: preparedBeforeCoverage.receipt.target.protocol,
                adapter_version: preparedBeforeCoverage.receipt.target.adapter_version,
                target_model: preparedBeforeCoverage.receipt.target.model,
                measured_at: runtime.recorded_at,
                source_fingerprint: await indexedProcessingContextFingerprint({
                    ...measuredSelection,
                    completeness: 'active_processing_dependencies_verified',
                }),
            };
            const coverage = {
                operation_id: 'coverage:accepted-output',
                expected_revision: pinned.root.source.revision,
                target_fingerprint: await fingerprintJson(preparedBeforeCoverage.receipt.target),
                measured_input_tokens: measurement.input_tokens,
                tokenizer_id: `${measurement.tokenizer}:${measurement.tokenizer_version}`,
                measurement_fingerprint: await fingerprintJson(measurement),
                recorded_at: runtime.recorded_at,
            };
            const covered = await stageIndexedProcessingCoverage(prior.store, pinned.root, pinned.locator, coverage);
            expect(covered.coverage.status).toBe('ready');
            const selection = await loadIndexedSelectedTextContext(prior.store, covered.root, covered.locator);
            const prepared = await prior.driver.prepareIndexedTextRequest({
                selection,
                runtime,
                options: { model: prior.record.request_receipt.target.model },
                stream: false,
            });
            const record = parseConversationPreparedRequestRecord({
                source: selection.source,
                runtime,
                request_receipt: { ...prepared.receipt, measurement },
                generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
                response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
                indexed_source: {
                    version: 1,
                    validator_profile: INDEXED_MEASURED_OUTPUT_PREPARED_VALIDATOR_PROFILE,
                    root: selection.root,
                    context_revision: selection.context.revision,
                    native_measurement: {
                        version: 1,
                        accepted_source: pinned.root.source,
                        accepted_root: pinned.locator,
                        accepted_operation_id: output.id,
                        runtime_input_operation_id: runtime.input_operation_id,
                        accepted_receipt_fingerprint: await fingerprintJson(acceptedReceipt),
                        retained_output: IndexedMeasuredOutputReceiptSchema.parse(acceptedReceipt),
                        settled_source: pinned.root.source,
                        settled_root: pinned.locator,
                        coverage,
                    },
                },
            });
            expect(output.result_revision).toBeLessThan(pinned.root.source.revision);
            expect(record.runtime).not.toHaveProperty('materialized_input');
            expect(await indexedPreparedReceiptMatches(record, selection, prepared.receipt)).toBe(true);
            if (prior.driver instanceof OpenAIDriver) {
                expect(measurement.method).toBe('provider_counted');
                await rejectUnregisteredOpenAICount(record, selection, prepared.receipt);
            }
            const assertCommitted = vi.fn(async () => undefined);
            const baseTransports = transports;
            for (const mutation of [
                'receipt',
                'payload',
                'operation',
                'accepted-source',
                'source',
                'native',
                'input',
            ] as const) {
                const changed = structuredClone(record);
                const witness = changed.indexed_source?.native_measurement;
                if (!witness?.retained_output) throw new Error('Authentic retained output witness missing');
                if (mutation === 'receipt') witness.retained_output.accepted_generation_ids[0] = 'generation:foreign';
                if (mutation === 'payload')
                    witness.retained_output.payload_fingerprint = await fingerprintJson({ wrong: 'original payload' });
                if (mutation === 'operation') witness.accepted_operation_id = 'operation:foreign';
                if (mutation === 'accepted-source') witness.accepted_source.revision = output.base_revision;
                if (mutation === 'source') changed.source.revision++;
                if (mutation === 'native')
                    changed.request_receipt.request_fingerprint = await fingerprintJson({ changed: 'body' });
                if (mutation === 'input')
                    changed.runtime.materialized_input = {
                        operation_id: output.id,
                        result_revision: output.result_revision,
                    };
                await expect(
                    prior.driver.executeCommittedIndexedTextRequest({
                        selection,
                        record: changed,
                        options: { model: record.request_receipt.target.model },
                        assert_committed: assertCommitted,
                    }),
                ).rejects.toThrow();
                expect(transports, mutation).toBe(baseTransports);
                expect(assertCommitted, mutation).not.toHaveBeenCalled();
            }
            let wrongProfileError: unknown;
            try {
                parseConversationPreparedRequestRecord({
                    ...record,
                    indexed_source: {
                        ...record.indexed_source,
                        validator_profile: INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
                    },
                });
            } catch (error: unknown) {
                wrongProfileError = error;
            }
            expect(wrongProfileError).toBeInstanceOf(ConversationValidationError);
            if (!(wrongProfileError instanceof ConversationValidationError))
                throw new Error('Input-only profile did not reject the distinct retained output witness');
            expect(wrongProfileError.message).toBe('Prepared request record failed schema validation');
            expect(wrongProfileError.diagnostics).toEqual(
                expect.arrayContaining([
                    expect.objectContaining({
                        code: 'SCHEMA_INVALID',
                        path: '/indexed_source/native_measurement/retained_output',
                    }),
                ]),
            );
            const continued = await prior.driver.executeCommittedIndexedTextRequest({
                selection,
                record,
                options: { model: record.request_receipt.target.model },
                assert_committed: assertCommitted,
            });
            expect(continued.generation.record_source).toBe('executed');
            expect(transports).toBe(baseTransports + 1);
            expect(assertCommitted).toHaveBeenCalledTimes(1);
        },
    );
});
